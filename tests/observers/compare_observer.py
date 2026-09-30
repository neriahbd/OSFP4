"""Compare observer snapshots in separate CPU processes with unchanged dependencies.

Run explicitly (not collected by pytest)::

    .venv/bin/python tests/observers/compare_observer.py \
        --reference /tmp/before/src --candidate src --output /tmp/observer-parity

Records complete cold/warm LUTs, interpolation gradients, initialization,
coefficients, every Adam loss/gradient/update, canonicalized scales and selected
qparams for FP32/BF16/FP16 inputs. Also exercises selection transitions and ties
with forced tiling. Use compare_base.py for the 48-case Llama/checkpoint matrix.
"""

import argparse
import json
import os
import runpy
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

# Reuse the existing tensor metadata/raw-byte recorder and mismatch reporting.
_shared = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "modifiers/compare_base.py")
)
Trace = _shared["Trace"]
compare = _shared["compare"]


def record_tables(trace, dtype):
    from llmcompressor_osfp4.observers import lut, scale_selection

    lut._PHI_LUT_CACHE.clear()
    lut._LUT_INTERPOLATION_CACHE.clear()
    for cache in ("cold", "warm"):
        trace.add(cache + "/table", lut._get_phi_lut("cpu", dtype))
        starts, slopes = lut._get_lut_interpolation_data("cpu", dtype)
        trace.add(cache + "/interpolation_data", (starts, slopes))
        values = torch.tensor(
            [
                0.0,
                -0.0,
                2.0**-5,
                2.0**-4,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                128.0,
                256.0,
                512.0,
            ],
            dtype=dtype,
        )
        if dtype != torch.float32:
            values = values[values.abs() <= 128.0]
            try:
                lut.phi_lut(torch.tensor([256.0], dtype=dtype))
            except IndexError as error:
                trace.add(cache + "/upper_bound_failure", type(error).__name__)
            else:
                raise AssertionError(
                    "Expected pre-existing low-precision boundary failure"
                )
        neighbors = torch.cat(
            (
                values,
                torch.nextafter(values, torch.full_like(values, float("inf"))),
                torch.nextafter(values, torch.full_like(values, -float("inf"))),
            )
        )
        values = torch.stack((neighbors, -neighbors)).t().requires_grad_()
        output = lut.phi_lut(values)
        trace.add(cache + "/phi", output)
        trace.add(cache + "/phi_gradient", torch.autograd.grad(output.sum(), values))
        # Positions stay representable even in FP16; phi above tests the domain ends.
        pos = torch.tensor(
            [-1.0, -0.0, 0.0, 0.5, 1.0, 1.5, 127.5], dtype=dtype
        ).requires_grad_()
        output = lut._interpolate_lut_at_position(pos, starts, slopes)
        if isinstance(output, tuple):  # Pre-refactor helper returned two unused values.
            output = output[0]
        trace.add(cache + "/interpolated", output)
        trace.add(cache + "/position_gradient", torch.autograd.grad(output.sum(), pos))
    trace.add("grid", scale_selection._get_e4m3_scale_grid("cpu", dtype))


def record_optimization(trace, dtype, noncontiguous, joint, steps):
    from compressed_tensors.quantization import preset_name_to_scheme

    from llmcompressor_osfp4.observers import OSFP4Observer, observer
    from llmcompressor_osfp4.observers import scale_optimization as optimization

    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(2, 3, 16, generator=generator).to(dtype)
    activations = (
        torch.randn(2, 16, 5, generator=generator).to(dtype) if joint else None
    )
    metric = (torch.rand(2, 16, generator=generator) + 0.1).to(dtype)
    if noncontiguous:
        weight = weight.transpose(1, 2).contiguous().transpose(1, 2)
        if activations is not None:
            activations = activations.transpose(1, 2).contiguous().transpose(1, 2)
    trace.add("inputs_before", (weight, activations, metric))
    # Also compare initialization before normal calibration's FP32 conversion.
    trace.add(
        "native_initialization",
        optimization.initialize_scale_values_from_absmax(weight, activations),
    )
    args = preset_name_to_scheme(
        "NVFP4" if joint else "NVFP4A16", []
    ).weights.model_copy(
        update={
            "observer": "osfp4",
            "observer_kwargs": {"num_iters": steps, "lr": 0.01},
        },
        deep=True,
    )
    obs = OSFP4Observer(base_name="weight", args=args)
    obs(weight.reshape(-1, 16))
    trace.add("statistics_before", (obs.min_vals, obs.max_vals))

    def record_function(name, function):
        def wrapped(*args, **kwargs):
            result = function(*args, **kwargs)
            trace.add(name, result)
            return result

        return wrapped

    original_step = torch.optim.Adam.step

    def step(adam, *args, **kwargs):
        parameters = [p for group in adam.param_groups for p in group["params"]]
        trace.add("gradients", [p.grad for p in parameters])
        result = original_step(adam, *args, **kwargs)
        trace.add("updated_parameters", parameters)
        trace.add("adam_state", [list(adam.state[p].values()) for p in parameters])
        return result

    original_canonicalize = observer.canonicalize_log_scale_parameters

    def canonicalize(parameters):
        result = original_canonicalize(parameters)
        trace.add("canonicalized", parameters)
        return result

    with ExitStack() as stack:
        for name in (
            "initialize_scale_values_from_absmax",
            "create_log_scale_parameters",
            "build_joint_loss_coefficients",
            "build_weight_loss_coefficients",
            "compute_joint_loss",
            "compute_weight_loss",
            "materialize_scale_values",
        ):
            stack.enter_context(
                patch.object(
                    observer, name, record_function(name, getattr(observer, name))
                )
            )
        stack.enter_context(patch.object(torch.optim.Adam, "step", step))
        stack.enter_context(
            patch.object(observer, "canonicalize_log_scale_parameters", canonicalize)
        )
        optimized = obs.optimize_quantization_group_scales(
            weight,
            activation_quantization_groups=activations,
            weight_metric=metric,
            alpha_dtype=dtype,
        )
    trace.add("optimized", optimized)
    trace.add(
        "qparams", obs.select_weight_qparams(weight, optimized, weight_metric=metric)
    )
    trace.add("statistics_after", (obs.min_vals, obs.max_vals))
    trace.add("inputs_after", (weight, activations, metric))


def record_selection(trace, dtype):
    from llmcompressor_osfp4.observers import scale_selection as selection

    grid = selection._get_e4m3_scale_grid("cpu", torch.float32)
    bounds = torch.cat((grid * 0.3, grid * 1.2))
    gamma = (
        torch.cat(
            (
                bounds,
                torch.nextafter(bounds, torch.full_like(bounds, float("inf"))),
                torch.nextafter(bounds, torch.zeros_like(bounds)),
                torch.tensor([0.0, 1e20]),
            )
        )
        .to(dtype)
        .view(1, -1, 1)
    )
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(1, gamma.numel(), 16, generator=generator).to(dtype)
    alpha = torch.ones(1, 16, dtype=dtype)
    metric = torch.ones_like(alpha)
    for tiled in (False, True):
        budget = 19 * 16 * 4 * 3 if tiled else 64 * 1024 * 1024
        with patch.object(selection, "_E4M3_SEARCH_PRIMARY_TENSOR_BYTES", budget):
            trace.add(
                f"selection/{tiled}",
                selection._search_e4m3_gamma_star(weight, alpha, gamma, metric),
            )
            trace.add(
                f"ties/{tiled}",
                selection._search_e4m3_gamma_star(weight * 0, alpha, gamma, metric * 0),
            )


def worker(output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        directory = output / str(dtype)
        directory.mkdir(parents=True)
        trace = Trace()
        record_tables(trace, dtype)
        record_selection(trace, dtype)
        for noncontiguous in (False, True):
            for joint in (False, True):
                for steps in (0, 1, 80):
                    trace.add("case", (str(dtype), noncontiguous, joint, steps))
                    record_optimization(trace, dtype, noncontiguous, joint, steps)
        trace.save(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.output)
        return
    for label in ("reference", "candidate"):
        source = getattr(args, label)
        if source is None or not (source / "llmcompressor").is_dir():
            parser.error(f"--{label} must contain llmcompressor")
    args.output.mkdir(parents=True, exist_ok=False)
    for label in ("reference", "candidate"):
        source = getattr(args, label).resolve()
        with (args.output / f"{label}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--output",
                    str(args.output.resolve() / label),
                ],
                env=dict(os.environ, PYTHONPATH=str(source), PYTHONHASHSEED="0"),
                cwd=source.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    report = {
        path.name: compare(path, args.output / "candidate" / path.name)
        for path in sorted((args.output / "reference").iterdir())
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Matched {sum(report.values())} observer tensor snapshots across three dtypes."
    )


if __name__ == "__main__":
    main()
