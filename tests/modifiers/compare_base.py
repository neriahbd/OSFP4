"""Compare two source snapshots with unchanged dependencies and CPU execution.

Run explicitly (not collected by pytest)::

    .venv/bin/python tests/modifiers/compare_base.py \
        --reference /tmp/before/src --candidate ./src --output /tmp/osfp4-parity

The default matrix covers both pipelines, both modes, both schemes, steps 0/1/80,
with sampled/full/disabled sampling for NVFP4 (48 cases per source). Use --steps 0
for a smoke run. Outputs contain raw byte traces, compressed saves, worker logs,
and a report. No models, tokenizers, or datasets are downloaded. Source paths must
contain llmcompressor; the worker imports exclusively from the selected snapshot.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch


class Trace:
    """Store ordered observations and raw bytes, including signed zeros."""

    def __init__(self):
        self.events = []
        self.tensors = {}

    def add(self, name, value):
        self.events.append((name, self._encode(value, f"{len(self.events)}/{name}")))

    def _encode(self, value, path):
        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().contiguous().reshape(-1)
            self.tensors[path] = flat.view(torch.uint8).clone()
            return {
                "tensor": path,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        if isinstance(value, dict):
            return [
                (key, self._encode(item, f"{path}/{key}"))
                for key, item in value.items()
            ]
        if isinstance(value, (tuple, list)):
            return [
                self._encode(item, f"{path}/{index}")
                for index, item in enumerate(value)
            ]
        if isinstance(value, torch.dtype):
            return str(value)
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"Unrecorded type at {path}: {type(value)}")

    def save(self, directory):
        (directory / "trace.json").write_text(json.dumps(self.events, indent=2) + "\n")
        torch.save(self.tensors, directory / "bytes.pt")


def _mapping_attr(mapping, current, reference):
    """Read mapping terminology from either side of the rename boundary."""
    return (
        getattr(mapping, current)
        if hasattr(mapping, current)
        else getattr(mapping, reference)
    )


def instrument(stack, trace, model):
    from llmcompressor_osfp4.modifiers import base, osfp4_quantize
    from llmcompressor_osfp4.observers import observer

    names = {module: name for name, module in model.named_modules()}
    original_quantize = base.quantize_mapping
    original_optimizer = observer.run_optimizer
    original_step = torch.optim.Adam.step
    original_initialize = base.OSFP4Modifier.on_initialize
    original_start = base.OSFP4Modifier.on_calibration_start
    original_end = base.OSFP4Modifier.on_calibration_end
    original_finalize = base.OSFP4Modifier.on_finalize

    def observers(mapping):
        result = {}
        balance_layers = _mapping_attr(mapping, "balance_layers", "target_layers")
        for layer in balance_layers:
            for kind in ("weight", "input"):
                obs = getattr(layer, f"{kind}_observer", None)
                if obs is not None:
                    result[f"{names[layer]}/{kind}"] = {
                        "args": obs.args.model_dump(mode="json"),
                        "min": getattr(obs, "min_vals", None),
                        "max": getattr(obs, "max_vals", None),
                        "fusions": [names[ref()] for ref in obs._fusions.values()],
                    }
        return result

    def initialize(modifier, state, **kwargs):
        result = original_initialize(modifier, state, **kwargs)
        trace.add(
            "mappings",
            [
                (
                    _mapping_attr(mapping, "mapping_name", "name"),
                    _mapping_attr(
                        mapping,
                        "requires_runtime_smoothing",
                        "uses_runtime_smooth_quant_scale",
                    ),
                    [
                        names[layer]
                        for layer in _mapping_attr(
                            mapping, "balance_layers", "target_layers"
                        )
                    ],
                )
                for mapping in modifier._resolved_mappings
            ],
        )
        return result

    def start(modifier, state, event, **kwargs):
        result = original_start(modifier, state, event, **kwargs)
        for mapping in modifier._resolved_mappings:
            trace.add("attached_observers", observers(mapping))
        return result

    def end(modifier, state, event, **kwargs):
        result = original_end(modifier, state, event, **kwargs)
        trace.add(
            "calibration_end",
            {
                "runtime_hooks": [
                    names[layer] for layer in modifier._runtime_smoothing_hooks
                ],
                "status": [
                    (names[layer], str(layer.quantization_status))
                    for layer in model.modules()
                    if hasattr(layer, "quantization_status")
                ],
                "remaining_observers": [
                    name
                    for name, _ in model.named_modules()
                    if name.endswith("_observer")
                ],
            },
        )
        return result

    def finalize(modifier, state, **kwargs):
        result = original_finalize(modifier, state, **kwargs)
        assert not modifier._runtime_smoothing_hooks
        assert not modifier._calibration.sample_count
        trace.add("sampling", modifier.activation_subsampling_records)
        return result

    def quantize(mapping, cache, **kwargs):
        # Trace CPU buffers only after their producer has finished. Both source
        # snapshots receive the same instrumentation; stress CUDA ordering separately.
        mapping_name = _mapping_attr(mapping, "mapping_name", "name")
        cache.wait(mapping_name, torch.device("cpu"))
        trace.add("mapping", mapping_name)
        trace.add(
            "statistics",
            {
                "count": cache.sample_count[mapping_name],
                "inputs": cache.inputs.get(mapping_name),
                "hessian": cache.hessian.get(mapping_name),
                "energy": cache.sigma_x_squared.get(mapping_name),
            },
        )
        trace.add("observers_before", observers(mapping))
        result = original_quantize(mapping, cache, **kwargs)
        trace.add("observers_after", observers(mapping))
        trace.add("deployment", [(names[layer], params) for layer, params in result])
        return result

    def optimizer(parameters, evaluate, **kwargs):
        trace.add("initial_log_scales", parameters)

        def loss():
            value = evaluate()
            trace.add("loss", value)
            return value

        def step(adam, *args, **kw):
            params = [p for group in adam.param_groups for p in group["params"]]
            trace.add("gradients", [p.grad for p in params])
            result = original_step(adam, *args, **kw)
            trace.add("updated_log_scales", params)
            return result

        with patch.object(torch.optim.Adam, "step", step):
            return original_optimizer(parameters, loss, **kwargs)

    def numerical_wrapper(original):
        def wrapped(*args, **kwargs):
            obs = kwargs["observer"]
            owners = [
                names[layer]
                for layer in model.modules()
                if getattr(layer, "weight_observer", None) is obs
            ]
            assert owners
            trace.add("optimizer_observer", owners)
            trace.add(
                "optimizer_inputs",
                (
                    args,
                    {key: value for key, value in kwargs.items() if key != "observer"},
                ),
            )
            result = original(*args, **kwargs)
            trace.add("optimized_scales_and_weights", result)
            return result

        return wrapped

    for method, replacement in (
        ("on_initialize", initialize),
        ("on_calibration_start", start),
        ("on_calibration_end", end),
        ("on_finalize", finalize),
    ):
        stack.enter_context(patch.object(base.OSFP4Modifier, method, replacement))
    stack.enter_context(patch.object(base, "quantize_mapping", quantize))
    stack.enter_context(patch.object(observer, "run_optimizer", optimizer))
    for method in ("optimize_rtn", "optimize_sic"):
        original = getattr(osfp4_quantize, method)
        stack.enter_context(
            patch.object(osfp4_quantize, method, numerical_wrapper(original))
        )

    # Native hooks stay active during the pipeline's propagation pass. Restrict
    # capture to tensors so symbolic tracing itself creates no trace events.
    def second_layer(layer, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if isinstance(hidden, torch.Tensor):
            trace.add("second_layer_inputs", hidden)

    handle = model.model.layers[1].register_forward_pre_hook(
        second_layer, with_kwargs=True
    )
    stack.callback(handle.remove)


def run_case(
    directory,
    pipeline,
    scheme,
    mode,
    sampling,
    steps,
    dtype="float32",
    offload=False,
    device="cpu",
    selection="sequential",
):
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM

    from llmcompressor.args import DatasetArguments
    from llmcompressor.core import Event, State, create_session
    from llmcompressor_osfp4.modifiers import OSFP4Modifier
    from llmcompressor.pipelines import CalibrationPipeline
    from llmcompressor.pipelines.sequential import pipeline as sequential
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        use_cache=False,
    )
    config.architectures = ["LlamaForCausalLM"]
    config._attn_implementation = "eager"
    model = (
        LlamaForCausalLM(config).to(device=device, dtype=getattr(torch, dtype)).eval()
    )
    modify_save_pretrained(model)
    modifier = OSFP4Modifier(
        scheme=scheme,
        ignore=["lm_head"],
        optimization_mode=mode,
        steps=steps,
        offload_hessians=offload,
        activation_subsample_size={"sampled": 17, "full": 65536, "none": None}[
            sampling
        ],
    )
    tokens = torch.arange(24).reshape(2, 12)
    batches = [{"input_ids": tokens % 64}, {"input_ids": (tokens * 7 + 3) % 64}]
    trace = Trace()
    with ExitStack() as stack:
        instrument(stack, trace, model)
        if pipeline == "direct":
            state = State(model=model)
            modifier.on_initialize(state)
            modifier.on_calibration_start(state, Event())
            with torch.no_grad():
                for batch in batches:
                    model(**{key: value.to(device) for key, value in batch.items()})
            modifier.on_calibration_end(state, Event())
            modifier.on_finalize(state)
        else:
            # Select the requested device explicitly on both snapshots.
            stack.enter_context(
                patch.object(
                    sequential, "get_main_device", lambda: torch.device(device)
                )
            )
            with create_session() as session:
                loader = torch.utils.data.DataLoader(batches, batch_size=None)
                session.initialize(
                    model=model,
                    recipe=[modifier],
                    start=-1,
                    calib_data=loader,
                    sequential_targets=["LlamaDecoderLayer"],
                )
                selected = CalibrationPipeline.from_modifiers(
                    [modifier], user=None if selection == "infer" else selection
                )
                selected(
                    model,
                    loader,
                    DatasetArguments(
                        sequential_targets=["LlamaDecoderLayer"],
                        sequential_offload_device="cpu",
                        propagate_error=True,
                    ),
                )
                session.finalize()
        assert any(name == "second_layer_inputs" for name, _ in trace.events)
        trace.add("deployed", model.state_dict())
        directory.mkdir(parents=True)
        model.save_pretrained(directory, save_compressed=True)
        packed = {}
        for path in sorted(directory.glob("*.safetensors")):
            packed.update(load_file(path))
        assert any(name.endswith("smooth_quant_scale") for name in packed)
        trace.add("packed", dict(sorted(packed.items())))
        trace.add("saved_config", json.loads((directory / "config.json").read_text()))
        trace.save(directory)


def matrix(steps, dtypes=("float32",), offloads=(False,)):
    for pipeline in ("direct", "sequential"):
        for mode in ("rtn", "sic"):
            for scheme in ("NVFP4", "NVFP4A16"):
                for sampling in (
                    ("sampled", "full", "none") if scheme == "NVFP4" else ("none",)
                ):
                    for count in steps:
                        for dtype in dtypes:
                            for offload in offloads:
                                yield (
                                    pipeline,
                                    scheme,
                                    mode,
                                    sampling,
                                    count,
                                    dtype,
                                    offload,
                                )


def compare(reference, candidate):
    if (reference / "config.json").exists():
        assert (reference / "config.json").read_bytes() == (
            candidate / "config.json"
        ).read_bytes(), f"{reference.name}: saved config bytes differ"
    before = json.loads((reference / "trace.json").read_text())
    after = json.loads((candidate / "trace.json").read_text())
    if len(before) != len(after):
        raise AssertionError(
            f"{reference.name}: event count {len(before)} != {len(after)}"
        )
    for index, (left, right) in enumerate(zip(before, after)):
        if left != right:
            raise AssertionError(
                f"{reference.name}: metadata at event {index}: {left} != {right}"
            )
    before_bytes = torch.load(reference / "bytes.pt", weights_only=True)
    after_bytes = torch.load(candidate / "bytes.pt", weights_only=True)
    assert before_bytes.keys() == after_bytes.keys(), f"{reference.name}: tensor names"
    for name, value in before_bytes.items():
        if not torch.equal(value, after_bytes[name]):
            raise AssertionError(f"{reference.name}: raw bytes differ at {name}")
    return len(before_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1, 80])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float32"],
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument(
        "--offload-hessians", choices=["false", "true", "both"], default="false"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--repeat-reference", action="store_true")
    parser.add_argument(
        "--candidate-selection",
        choices=["sequential", "infer", "independent"],
        default="sequential",
    )
    parser.add_argument("--selection", default="sequential", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA validation requires a CUDA host")
    offloads = {"false": (False,), "true": (True,), "both": (False, True)}[
        args.offload_hessians
    ]
    cases = list(matrix(args.steps, args.dtypes, offloads))
    if args.worker:
        for case in cases:
            name = "-".join(map(str, case))
            run_case(
                args.output / name, *case, device=args.device, selection=args.selection
            )
            print(f"Saved {name}", flush=True)
        return
    for label in ("reference", "candidate"):
        source = getattr(args, label)
        if source is None or not (source / "llmcompressor").is_dir():
            parser.error(f"--{label} must contain llmcompressor")
    args.output.mkdir(parents=True, exist_ok=False)
    labels = (
        ("reference", "repeat", "candidate")
        if args.repeat_reference
        else ("reference", "candidate")
    )
    for label in labels:
        source = getattr(args, "reference" if label == "repeat" else label).resolve()
        env = dict(
            os.environ,
            PYTHONPATH=str(source),
            PYTHONHASHSEED="0",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        with (args.output / f"{label}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--output",
                    str(args.output.resolve() / label),
                    "--steps",
                    *map(str, args.steps),
                    "--dtypes",
                    *args.dtypes,
                    "--offload-hessians",
                    args.offload_hessians,
                    "--device",
                    args.device,
                    "--selection",
                    args.candidate_selection if label == "candidate" else "sequential",
                ],
                env=env,
                cwd=source.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    report = {}
    file_report = {}
    for case in cases:
        name = "-".join(map(str, case))
        reference = args.output / "reference" / name
        candidate = args.output / "candidate" / name
        if args.repeat_reference:
            repeated = args.output / "repeat" / name
            compare(reference, repeated)
            files = sorted(reference.glob("*.safetensors"))
            assert [path.name for path in files] == sorted(
                path.name for path in candidate.glob("*.safetensors")
            )
            for path in files:

                def digest(value):
                    return hashlib.sha256(value.read_bytes()).hexdigest()

                stable = digest(path) == digest(repeated / path.name)
                if stable:
                    assert digest(path) == digest(
                        candidate / path.name
                    ), f"{name}: checkpoint file bytes differ"
                file_report[f"{name}/{path.name}"] = {
                    "baseline_file_byte_stable": stable,
                    "tensor_payloads_must_match": True,
                }
        report[name] = compare(reference, candidate)
    (args.output / "checkpoint_files.json").write_text(
        json.dumps(file_report, indent=2) + "\n"
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Matched {len(report)} cases and {sum(report.values())} tensor snapshots.")


if __name__ == "__main__":
    main()
