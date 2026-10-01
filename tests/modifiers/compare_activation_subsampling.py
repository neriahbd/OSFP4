"""Compare configurable fixed sampling against a baseline in isolated CPU processes.

Run with the plugin's stock dependency environment::

    .venv/bin/python tests/modifiers/compare_activation_subsampling.py \
        --reference build/activation-parity/baseline/src --candidate src \
        --output build/activation-parity/results

To expand the fixed-cap matrix, add::

    --caps 1 3 16 32 128 256 8192 16384 65536 \
    --rows 256 16512 --dtypes float32 float16 bfloat16

The worker also compares candidate auto sampling against an explicit cap of twice
the input width. No models or datasets are downloaded. Raw tensor bytes, source identities,
dependency versions, checkpoint configs, and provenance are retained for review.
"""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from compare_base import Trace, compare


def run_case(directory, mode, streaming, rows, cap, dtype, *, single_width=False):
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM

    from llmcompressor.args import DatasetArguments
    from llmcompressor.core import create_session
    from llmcompressor.pipelines import CalibrationPipeline
    from llmcompressor.pipelines.sequential import pipeline as sequential
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )
    from llmcompressor_osfp4.modifiers import OSFP4Modifier

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(dtype=getattr(torch, dtype)).eval()
    modify_save_pretrained(model)
    modifier = OSFP4Modifier(
        scheme="NVFP4",
        targets=["re:.*self_attn.*_proj$"] if single_width else ["Linear"],
        ignore=["lm_head"],
        optimization_mode=mode,
        steps=2,
        activation_subsample_size=cap,
    )
    generator = torch.Generator().manual_seed(17)
    batches = [
        {
            "input_ids": torch.randint(
                0, 64, (1, min(128, rows - start)), generator=generator
            )
        }
        for start in range(0, rows, 128)
    ]
    loader = torch.utils.data.DataLoader(batches, batch_size=None)
    trace = Trace()
    original_sample = OSFP4Modifier._sample_optimization_inputs

    def sample(self, mapping):
        assert (mapping.mapping_name in self._calibration.sample_indices) == streaming
        selected = original_sample(self, mapping)
        actual = torch.cat(selected or self._calibration.inputs[mapping.mapping_name])
        record = self.activation_subsampling_records[mapping.mapping_name]
        n = mapping.balance_layers[0].weight.shape[1]
        assert record["k1"] == rows
        assert record["k"] == min(rows, 2 * n if cap == "auto" else cap)
        if cap == "auto":
            assert record["policy"] == "auto" and record["n"] == n
        else:
            assert record["policy"] == "fixed" and "n" not in record
        trace.add(f"sample/{mapping.mapping_name}", actual)
        if mode == "sic":
            trace.add(
                f"hessian/{mapping.mapping_name}",
                self._calibration.hessian[mapping.mapping_name],
            )
        else:
            trace.add(
                f"energy/{mapping.mapping_name}",
                self._calibration.sigma_x_squared[mapping.mapping_name],
            )
        return selected

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(OSFP4Modifier, "_sample_optimization_inputs", sample)
        )
        if not streaming:
            stack.enter_context(
                patch.object(
                    OSFP4Modifier, "_calibration_token_count", return_value=None
                )
            )
        stack.enter_context(
            patch.object(sequential, "get_main_device", lambda: torch.device("cpu"))
        )
        session = stack.enter_context(create_session())
        session.initialize(
            model=model,
            recipe=[modifier],
            start=-1,
            calib_data=loader,
            sequential_targets=["LlamaDecoderLayer"],
        )
        CalibrationPipeline.from_modifiers([modifier])(
            model,
            loader,
            DatasetArguments(
                sequential_targets=["LlamaDecoderLayer"],
                sequential_offload_device="cpu",
                propagate_error=True,
            ),
        )
        session.finalize()
    records = modifier.activation_subsampling_records
    assert records
    directory.mkdir(parents=True)
    (directory / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    comparable_records = {name: record.copy() for name, record in records.items()}
    if single_width and cap == "auto":
        for record in comparable_records.values():
            del record["n"]
            record["policy"] = "fixed"
    trace.add("provenance", comparable_records)
    trace.add("model_state", dict(sorted(model.state_dict().items())))
    model.save_pretrained(directory, save_compressed=True)
    packed = {}
    for file in sorted(directory.glob("*.safetensors")):
        packed.update(load_file(file))
    assert any(name.endswith("smooth_quant_scale") for name in packed)
    trace.add("checkpoint", dict(sorted(packed.items())))
    saved_config = json.loads((directory / "config.json").read_text())
    assert saved_config["osfp4_metadata"] == model.config.osfp4_metadata
    trace.add("saved_config", saved_config)
    trace.save(directory)


def worker(source, output, candidate, caps, row_counts, dtypes):
    sys.path.insert(0, str(source.resolve()))
    import llmcompressor
    import llmcompressor_osfp4

    assert Path(llmcompressor_osfp4.__file__).resolve().is_relative_to(source.resolve())
    versions = {
        name: importlib.metadata.version(name)
        for name in ("llmcompressor", "compressed-tensors", "torch", "transformers")
    }
    assert versions["llmcompressor"] == "0.14.0"
    assert versions["compressed-tensors"] == "0.19.0"
    assert not Path(llmcompressor.__file__).resolve().is_relative_to(source.resolve())
    output.mkdir(parents=True)
    (output / "environment.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "dependencies": versions,
                "llmcompressor": llmcompressor.__file__,
                "matrix": {"caps": caps, "rows": row_counts, "dtypes": dtypes},
            },
            indent=2,
        )
        + "\n"
    )
    for dtype in dtypes:
        for mode in ("rtn", "sic"):
            for streaming in (False, True):
                for rows in row_counts:
                    for cap in caps:
                        name = f"fixed-{dtype}-{mode}-{streaming}-{rows}-{cap}"
                        run_case(output / name, mode, streaming, rows, cap, dtype)
                        print(f"Completed {name}", flush=True)
                if candidate:
                    for cap in ("auto", 64):
                        name = f"width-{dtype}-{mode}-{streaming}-{cap}"
                        run_case(
                            output / name,
                            mode,
                            streaming,
                            256,
                            cap,
                            dtype,
                            single_width=True,
                        )
                        print(f"Completed {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[16384])
    parser.add_argument("--rows", type=int, nargs="+", default=[256, 16512])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float16", "bfloat16"],
        default=["bfloat16"],
    )
    parser.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--candidate-worker", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    for name in ("caps", "rows", "dtypes"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name} must not contain duplicates")
    if any(value <= 0 for value in args.caps + args.rows):
        parser.error("--caps and --rows must be positive integers")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.worker_source is not None:
        worker(
            args.worker_source,
            args.output,
            args.candidate_worker,
            args.caps,
            args.rows,
            args.dtypes,
        )
        return
    for label in ("reference", "candidate"):
        source = getattr(args, label)
        if source is None or not (source / "llmcompressor_osfp4").is_dir():
            parser.error(f"--{label} must contain llmcompressor_osfp4")
    args.output.mkdir(parents=True, exist_ok=False)
    for label in ("reference", "candidate"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-source",
            str(getattr(args, label).resolve()),
            "--output",
            str((args.output / label).resolve()),
            "--caps",
            *map(str, args.caps),
            "--rows",
            *map(str, args.rows),
            "--dtypes",
            *args.dtypes,
        ]
        if label == "candidate":
            command.append("--candidate-worker")
        with (args.output / f"{label}.log").open("w") as log:
            subprocess.run(
                command,
                env=dict(os.environ, PYTHONHASHSEED="0"),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    environments = [
        json.loads((args.output / label / "environment.json").read_text())
        for label in ("reference", "candidate")
    ]
    assert environments[0]["dependencies"] == environments[1]["dependencies"]
    assert environments[0]["matrix"] == environments[1]["matrix"]
    report = {}
    for reference in sorted((args.output / "reference").glob("fixed-*")):
        candidate = args.output / "candidate" / reference.name
        assert (reference / "provenance.json").read_bytes() == (
            candidate / "provenance.json"
        ).read_bytes(), f"{reference.name}: provenance bytes differ"
        report[reference.name] = compare(reference, candidate)
    assert len(report) == len(args.dtypes) * 4 * len(args.rows) * len(args.caps)
    for dtype in args.dtypes:
        for mode in ("rtn", "sic"):
            for streaming in (False, True):
                report[f"auto-{dtype}-{mode}-{streaming}"] = compare(
                    args.output / "candidate" / f"width-{dtype}-{mode}-{streaming}-64",
                    args.output
                    / "candidate"
                    / f"width-{dtype}-{mode}-{streaming}-auto",
                )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Matched {len(report)} comparisons and {sum(report.values())} tensor snapshots."
    )


if __name__ == "__main__":
    main()
