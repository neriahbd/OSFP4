#!/usr/bin/env python3
"""Shared FP-Quant evaluation, validation, resumption, and summarization.

The task and prompt settings match FP-Quant commit d2e3092f.  Each task is run
separately because Open LLM v1 assigns different few-shot and chat-template
settings to the four tasks. The ``qwen-table7`` profile additionally disables
Qwen3 thinking and produces a method summary for later Table 7 aggregation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from examples.fpquant.common import protocol

# Single source of truth for task settings lives in protocol.py (see its
# module docstring for the FP-Quant reference and our documented
# deviations). TASKS keeps this file's original 5-tuple shape -- consumed
# positionally throughout this module (row[4], `for task, *_ in TASKS`,
# full unpack in main()) -- so the rest of this file needs no further edits.
PAPER_SCORES = protocol.PAPER_SCORES
TASKS = tuple(
    (
        spec.name,
        spec.num_fewshot,
        spec.apply_chat_template,
        spec.fewshot_as_multiturn,
        spec.metric_candidates,
    )
    for spec in protocol.TASK_SPECS
)

COMMON_LOCAL_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

COMPRESSED_MODEL_FILES = (
    "model.safetensors",
    "recipe.yaml",
    "calibration-manifest.json",
)

SHARDED_MODEL_INDEX = "model.safetensors.index.json"
OSFP4_OPTIMIZATION_MODES = frozenset(("rtn", "sic"))
OSFP4_RUNTIME_METADATA = {
    "version": 1,
}


def parse_args(profile: protocol.BenchmarkProfile) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=profile.methods,
    )
    parser.add_argument(
        "--model",
        help="Checkpoint path or Hub id; defaults to the selected profile's model.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "vLLM/HF dtype string. Defaults to the profile's native precision "
            "(bfloat16 for both bundled profiles). This is the concrete value "
            "to which FP-Quant's public dtype=auto resolves for these models."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "hf"),
        default="vllm",
        help="lm-eval model backend; use hf if the local vLLM engine cannot start.",
    )
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[task for task, *_ in TASKS],
        help=(
            "Run only these tasks. Existing raw task files are reused when making "
            "a full summary."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--linear-backend",
        default="detect",
        help=(
            "vLLM quantized linear backend. Default 'detect' picks 'auto' "
            "(vLLM's own default, needed for the weight-only Marlin kernel) "
            "for a weight-only NVFP4A16 checkpoint and 'cutlass' for a full "
            "W4A4 NVFP4 checkpoint or the BF16 baseline -- see "
            "protocol.detect_linear_backend. Pass an explicit value "
            "to override."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "Not auto-derived from CUDA_VISIBLE_DEVICES: an explicit "
            "value keeps runs reproducible."
        ),
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help=(
            "Embed every per-document sample in each {task}.json "
            "(large; off by default)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate at most this many documents per task (smoke tests only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse existing task JSON files only when they contain a valid expected "
            "metric."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate local dependencies, task registration, and model access only.",
    )
    args = parser.parse_args()
    if args.model is None:
        args.model = (
            profile.default_model
            if args.method == "bf16"
            else str(Path(profile.run_root) / args.method / "seed-42" / "model")
        )
    if args.output_dir is None:
        args.output_dir = Path(profile.run_root) / "eval" / args.method
    if args.dtype is None:
        args.dtype = profile.evaluation_dtype
    return args


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


# Re-export the shared metric helpers for callers and tests.
task_result = protocol.task_result
select_metric = protocol.select_metric


def build_summary(
    raw_results: dict[str, dict[str, Any]],
    *,
    paper_scores: dict[str, float] | None = PAPER_SCORES,
    paper_baseline_label: str = "FP16",
) -> dict[str, Any]:
    scores: dict[str, float] = {}
    metrics: dict[str, str] = {}
    for task, _, _, _, candidates in TASKS:
        metrics[task], scores[task] = select_metric(
            task_result(raw_results[task], task), candidates
        )

    average = sum(scores.values()) / len(scores)
    summary = {
        "scores": scores,
        "metrics": metrics,
        "average": average,
    }
    if paper_scores is None:
        return summary
    paper_average = sum(paper_scores.values()) / len(paper_scores)
    return summary | {
        "recovery_percent": 100.0 * average / paper_average,
        "paper_reference": {
            "baseline_label": paper_baseline_label,
            "scores": paper_scores,
            "average": paper_average,
        },
        "delta_vs_paper": {
            task: scores[task] - paper_scores[task] for task in paper_scores
        }
        | {"average": average - paper_average},
    }


def environment_manifest(
    args: argparse.Namespace,
    profile: protocol.BenchmarkProfile,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "benchmark_profile": profile.name,
        "table_number": profile.table_number,
        "model": args.model,
        "architecture": profile.architecture,
        "native_dtype": profile.evaluation_dtype,
        "tokenizer": profile.default_model,
        "method": args.method,
        "optimization_mode": (
            args.method.removeprefix("osfp4-")
            if args.method.startswith("osfp4-")
            else None
        ),
        "dataset_contract": {
            "name": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "num_samples": 1024,
            "max_sequence_length": 2048,
            "seed": 42,
            "shuffle_buffer_size": 1000,
        },
        "arguments": vars(args),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name, name in {
                "lm_eval": "lm-eval",
                "transformers": "transformers",
                "torch": "torch",
                "vllm": "vllm",
                "datasets": "datasets",
            }.items()
        },
    }
    model_path = Path(args.model)
    if model_path.is_dir():
        manifest["local_checkpoint"] = local_checkpoint_metadata(model_path)
    checkpoint_manifest = model_path / "calibration-manifest.json"
    if checkpoint_manifest.is_file():
        try:
            manifest["checkpoint_manifest"] = json.loads(
                checkpoint_manifest.read_text()
            )
        except json.JSONDecodeError as exc:
            manifest["checkpoint_manifest_error"] = str(exc)
    try:
        import torch

        manifest["cuda"] = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        manifest["cuda"] = {"available": False}
    return manifest


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def local_checkpoint_metadata(model_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "resolved_path": str(model_path.resolve()),
    }
    if model_path.parent.name == "snapshots":
        metadata["snapshot_revision"] = model_path.name

    try:
        config = read_json(model_path / "config.json")
        metadata["checkpoint_type"] = (
            "compressed" if config.get("quantization_config") else "unquantized"
        )
        metadata["architectures"] = config.get("architectures")
        metadata["stored_dtype"] = config.get("torch_dtype")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        metadata["config_error"] = f"{type(exc).__name__}: {exc}"

    weight_files: list[str] = []
    index_path = model_path / SHARDED_MODEL_INDEX
    if index_path.is_file():
        try:
            index = read_json(index_path)
            weight_map = index.get("weight_map", {})
            if isinstance(weight_map, dict):
                weight_files = sorted(set(weight_map.values()))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    elif (model_path / "model.safetensors").is_file():
        weight_files = ["model.safetensors"]

    metadata["weight_files"] = [
        {
            "name": name,
            "size_bytes": (model_path / name).stat().st_size,
        }
        for name in weight_files
        if isinstance(name, str) and (model_path / name).is_file()
    ]
    return metadata


def validate_unquantized_model(model_path: Path) -> list[str]:
    errors: list[str] = []
    index_path = model_path / SHARDED_MODEL_INDEX
    if not index_path.is_file():
        return [f"local checkpoint is missing {SHARDED_MODEL_INDEX}: {index_path}"]

    try:
        index = read_json(index_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"cannot read local checkpoint index: {type(exc).__name__}: {exc}"]

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return [f"local checkpoint index has no weight_map: {index_path}"]

    shards = set(weight_map.values())
    invalid_shards = sorted(
        str(name)
        for name in shards
        if not isinstance(name, str) or Path(name).name != name
    )
    if invalid_shards:
        errors.append(
            f"local checkpoint index has invalid shard names: {invalid_shards}"
        )
        return errors

    missing_shards = sorted(
        name for name in shards if not (model_path / name).is_file()
    )
    if missing_shards:
        errors.append(f"local checkpoint is missing weight shards: {missing_shards}")
    return errors


def validate_local_model(
    model_path: Path,
    profile: protocol.BenchmarkProfile,
    method: str,
) -> list[str]:
    errors = [
        f"local checkpoint is missing {name}: {model_path / name}"
        for name in COMMON_LOCAL_MODEL_FILES
        if not (model_path / name).is_file()
    ]
    if errors:
        return errors

    try:
        config = read_json(model_path / "config.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"cannot read local checkpoint config: {type(exc).__name__}: {exc}"]

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or profile.architecture not in architectures:
        errors.append(
            "local checkpoint does not declare the "
            f"{profile.architecture} architecture"
        )

    quantization = config.get("quantization_config")
    if not quantization:
        if method != "bf16":
            errors.append(f"{method} requires a compressed checkpoint")
        errors.extend(validate_unquantized_model(model_path))
        return errors
    if not isinstance(quantization, dict):
        errors.append("local checkpoint has an invalid quantization_config")
        return errors
    if method == "bf16":
        errors.append("bf16 requires an unquantized checkpoint")
        return errors

    errors.extend(
        f"local checkpoint is missing {name}: {model_path / name}"
        for name in COMPRESSED_MODEL_FILES
        if not (model_path / name).is_file()
    )
    if errors:
        return errors

    try:
        manifest = read_json(model_path / "calibration-manifest.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"cannot read calibration manifest: {type(exc).__name__}: {exc}")
        return errors
    manifest_expected = {
        "benchmark_profile": profile.name,
        "model": profile.default_model,
        "architecture": profile.architecture,
        "native_dtype": profile.calibration_dtype,
        "torch_dtype": profile.calibration_dtype,
        "method": method,
        "calibration_dataset": "HuggingFaceFW/fineweb-edu",
        "calibration_config": "sample-10BT",
        "calibration_split": "train",
        "calibration_seed": 42,
        "num_calibration_samples": 1024,
        "max_sequence_length": 2048,
        "shuffle_buffer_size": 1000,
        "tokenizer": profile.default_model,
    }
    manifest_mismatches = {
        key: (manifest.get(key), value)
        for key, value in manifest_expected.items()
        if manifest.get(key) != value
    }
    if manifest_mismatches:
        errors.append(f"calibration manifest mismatch: {manifest_mismatches}")
        return errors

    if not method.startswith("osfp4-"):
        return errors

    if quantization.get("quant_method") != "osfp4":
        errors.append("local checkpoint does not declare osfp4")
    if quantization.get("format") != "nvfp4-pack-quantized":
        errors.append("local checkpoint does not declare nvfp4-pack-quantized")

    osfp4_metadata = config.get("osfp4_metadata")
    if not isinstance(osfp4_metadata, dict):
        errors.append("local checkpoint has no OSFP4 runtime metadata")
        return errors
    runtime_mismatches = {
        key: (osfp4_metadata.get(key), value)
        for key, value in OSFP4_RUNTIME_METADATA.items()
        if osfp4_metadata.get(key) != value
    }
    if runtime_mismatches:
        errors.append(f"OSFP4 runtime metadata mismatch: {runtime_mismatches}")
    runtime_smooth_quant_scale_targets = osfp4_metadata.get(
        "smooth_quant_scale_targets"
    )
    if not isinstance(runtime_smooth_quant_scale_targets, list):
        errors.append("local checkpoint has invalid OSFP4 smooth_quant_scale_targets")
        return errors

    recipe_expected = {
        "osfp4_architecture": profile.architecture,
        "smooth_quant_scale_targets": runtime_smooth_quant_scale_targets,
        "scheme": "NVFP4A16" if method.endswith("-a16") else "NVFP4",
        "steps": 80,
        "lr": 0.12,
    }
    mismatches: dict[str, tuple[Any, Any]] = {}
    recipe = manifest.get("recipe", {})
    optimization_mode = recipe.get("optimization_mode")
    if optimization_mode not in OSFP4_OPTIMIZATION_MODES:
        errors.append(
            "calibration manifest has invalid recipe.optimization_mode="
            f"{optimization_mode!r}; expected one of "
            f"{sorted(OSFP4_OPTIMIZATION_MODES)}"
        )
        return errors
    mismatches.update(
        {
            f"recipe.{key}": (recipe.get(key), value)
            for key, value in recipe_expected.items()
            if recipe.get(key) != value
        }
    )
    activation_subsample_size = recipe.get("activation_subsample_size")
    activation_subsampling = recipe.get("activation_subsampling")
    if activation_subsample_size is not None:
        if not (
            activation_subsample_size == "auto"
            or (
                isinstance(activation_subsample_size, int)
                and not isinstance(activation_subsample_size, bool)
                and activation_subsample_size > 0
            )
        ):
            errors.append(
                "calibration manifest has invalid "
                "recipe.activation_subsample_size="
                f"{activation_subsample_size!r}"
            )
        elif not isinstance(activation_subsampling, dict) or not activation_subsampling:
            errors.append(
                "calibration manifest is missing activation-subsampling provenance"
            )
        else:
            expected_policy = "auto" if activation_subsample_size == "auto" else "fixed"
            for mapping_name, record in activation_subsampling.items():
                valid_record = (
                    isinstance(mapping_name, str)
                    and isinstance(record, dict)
                    and record.get("policy") == expected_policy
                    and record.get("seed") == 42
                    and isinstance(record.get("m"), int)
                    and record.get("m", 0) > 0
                    and isinstance(record.get("k"), int)
                    and isinstance(record.get("k1"), int)
                    and 0 < record.get("k", 0) <= record.get("k1", 0)
                    and isinstance(record.get("index_sha256"), str)
                    and len(record.get("index_sha256", "")) == 64
                )
                if not valid_record:
                    errors.append(
                        "calibration manifest has invalid activation-subsampling "
                        f"provenance for {mapping_name!r}"
                    )
                    break
    if mismatches:
        errors.append(f"calibration manifest mismatch: {mismatches}")
    return errors


def _declares_osfp4(model_path: Path) -> bool:
    try:
        config = read_json(model_path / "config.json")
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    quantization = config.get("quantization_config")
    return (
        isinstance(quantization, dict) and quantization.get("quant_method") == "osfp4"
    )


def validate(
    args: argparse.Namespace,
    profile: protocol.BenchmarkProfile,
    task_manager: Any | None = None,
) -> list[str]:
    errors: list[str] = []
    if importlib.util.find_spec("lm_eval") is None:
        errors.append("lm_eval is not installed")
    if args.backend == "vllm" and importlib.util.find_spec("vllm") is None:
        errors.append("vllm is not installed (required for the FP-Quant protocol)")
    model_path = Path(args.model)
    if (
        args.backend == "vllm"
        and model_path.is_dir()
        and _declares_osfp4(model_path)
        and importlib.util.find_spec("vllm_osfp4") is None
    ):
        # Confirmed gap: the repo's own .venv has vllm but not the vllm-osfp4
        # plugin (it lives only in the dedicated serving venv, pinned to a
        # matching vllm version). Without this check, vLLM raises "Unknown
        # quantization method: osfp4" mid-engine-start regardless of
        # VLLM_PLUGINS or an explicit quantization= -- this turns that into a
        # legible --check-only failure instead.
        errors.append(
            "checkpoint declares quant_method=osfp4 but vllm_osfp4 is not "
            f"installed for {sys.executable}; run with the serving venv, "
            "e.g. PYTHON_BIN=/workspace/venvs/serve/bin/python"
        )
    if not errors:
        if task_manager is None:
            from lm_eval.tasks import TaskManager

            task_manager = TaskManager()
        available = set(task_manager.all_tasks)
        missing = [task for task, *_ in TASKS if task not in available]
        if missing:
            errors.append(f"lm-eval does not register required tasks: {missing}")
    if model_path.is_dir():
        errors.extend(validate_local_model(model_path, profile, args.method))
    else:
        try:
            from huggingface_hub import HfApi

            HfApi().model_info(args.model)
        except Exception as exc:  # Auth may be needed for gated remote models.
            errors.append(
                f"cannot access model {args.model!r}: {type(exc).__name__}: {exc}"
            )
    return errors


def main(profile: protocol.BenchmarkProfile) -> int:
    args = parse_args(profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "environment.json",
        environment_manifest(args, profile),
    )

    task_manager = None
    if importlib.util.find_spec("lm_eval") is not None:
        from lm_eval.tasks import TaskManager

        task_manager = TaskManager()
    errors = validate(args, profile, task_manager)
    write_json(args.output_dir / "validation.json", {"errors": errors})
    if errors:
        print("Validation failed:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 2
    if args.check_only:
        print("Validation passed")
        return 0

    if args.backend == "vllm":
        linear_backend = (
            protocol.detect_linear_backend(args.model)
            if args.linear_backend == "detect"
            else args.linear_backend
        )
        model_args = protocol.build_model_args(
            model=args.model,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            linear_backend=linear_backend,
            enable_thinking=profile.enable_thinking,
        )
    else:
        model_args = f"pretrained={args.model},dtype={args.dtype},device=cuda"
        if profile.enable_thinking is not None:
            model_args += f",enable_thinking={profile.enable_thinking}"
    raw_results: dict[str, dict[str, Any]] = {}
    for task, *_ in TASKS:
        raw_path = args.output_dir / f"{task}.json"
        if args.resume and raw_path.is_file():
            try:
                existing = json.loads(raw_path.read_text())
                candidates = next(row[4] for row in TASKS if row[0] == task)
                select_metric(task_result(existing, task), candidates)
                raw_results[task] = existing
                print(f"Reusing valid existing result: {raw_path}", flush=True)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(
                    f"Ignoring invalid existing raw result {raw_path}: {exc}",
                    file=sys.stderr,
                )

    selected = set(args.tasks) if args.tasks else {task for task, *_ in TASKS}
    for task, *_ in TASKS:
        if task not in selected:
            continue
        if task in raw_results:
            continue
        spec = protocol.TASK_SPECS_BY_NAME[task]
        result = protocol.run_task(
            spec,
            model_args,
            backend=args.backend,
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=args.log_samples,
            task_manager=task_manager,
        )
        if result is None:
            raise RuntimeError(f"lm-eval returned no result for {task}")
        raw_results[task] = result
        write_json(args.output_dir / f"{task}.json", result)

    missing = [task for task, *_ in TASKS if task not in raw_results]
    if missing:
        print(f"Raw results saved; summary awaits: {', '.join(missing)}")
        return 0

    summary = build_summary(
        raw_results,
        paper_scores=profile.paper_scores,
        paper_baseline_label=profile.paper_baseline_label or "FP16",
    )
    summary.update(
        {
            "benchmark_profile": profile.name,
            "method": args.method,
            "model": args.model,
            "dtype": args.dtype,
            "enable_thinking": profile.enable_thinking,
            "paper_baseline_label": profile.paper_baseline_label,
        }
    )
    write_json(args.output_dir / profile.summary_filename, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
