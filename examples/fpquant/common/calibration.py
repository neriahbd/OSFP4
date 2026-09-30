"""Shared calibration machinery for the FP-Quant table reproductions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.fpquant.common.protocol import BenchmarkProfile

DEFAULT_NUM_SAMPLES = 1_024
DEFAULT_MAX_SEQUENCE_LENGTH = 2_048


@dataclass(frozen=True)
class CalibrationSource:
    """Immutable dataset settings for a calibration cache."""

    dataset_name: str
    dataset_config: str
    dataset_split: str
    shuffle_buffer_size: int


FINEWEB_EDU_SOURCE = CalibrationSource(
    dataset_name="HuggingFaceFW/fineweb-edu",
    dataset_config="sample-10BT",
    dataset_split="train",
    shuffle_buffer_size=1_000,
)


@dataclass(frozen=True)
class CalibrationConfig:
    """Resolved settings for one preparation or calibration job."""

    profile: BenchmarkProfile
    model_id: str
    seed: int
    num_samples: int
    max_sequence_length: int
    calibration_data_root: Path
    save_dir: Path | None = None
    method: str | None = None
    source: CalibrationSource = FINEWEB_EDU_SOURCE

    @property
    def prepared_dataset_dir(self) -> Path:
        return self.calibration_data_root / f"seed-{self.seed}"

    def data_manifest(self) -> dict[str, Any]:
        return {
            "benchmark_profile": self.profile.name,
            "model": self.model_id,
            "architecture": self.profile.architecture,
            "native_dtype": self.profile.calibration_dtype,
            "calibration_dataset": self.source.dataset_name,
            "calibration_config": self.source.dataset_config,
            "calibration_split": self.source.dataset_split,
            "calibration_seed": self.seed,
            "num_calibration_samples": self.num_samples,
            "max_sequence_length": self.max_sequence_length,
            "shuffle_buffer_size": self.source.shuffle_buffer_size,
            "tokenizer": self.model_id,
        }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def validate_prepared_dataset(
    config: CalibrationConfig,
    dataset_dir: Path | None = None,
) -> Dataset:
    """Load a prepared cache only when its complete table contract matches."""
    dataset_dir = dataset_dir or config.prepared_dataset_dir
    manifest_path = dataset_dir / "calibration-data-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing calibration manifest: {manifest_path}")
    manifest = _read_object(manifest_path)
    expected = config.data_manifest()
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"calibration manifest mismatch in {manifest_path}: {mismatches}"
        )

    dataset = load_from_disk(str(dataset_dir))
    if not isinstance(dataset, Dataset):
        raise RuntimeError(f"expected Dataset at {dataset_dir}")
    if len(dataset) != config.num_samples:
        raise RuntimeError(
            f"expected {config.num_samples} samples at {dataset_dir}, got "
            f"{len(dataset)}"
        )
    if dataset.column_names != ["input_ids"]:
        raise RuntimeError(f"unexpected calibration columns: {dataset.column_names}")
    bad_index = next(
        (
            index
            for index, input_ids in enumerate(dataset["input_ids"])
            if len(input_ids) != config.max_sequence_length
        ),
        None,
    )
    if bad_index is not None:
        raise RuntimeError(
            f"invalid sequence length at {dataset_dir}, sample {bad_index}"
        )
    return dataset


def collect_dataset(tokenizer: Any, config: CalibrationConfig) -> Dataset:
    """Collect deterministic random token windows from a streaming source.

    A shuffle buffer size of 0 (used by callers that want the source's own
    row order preserved) is skipped rather than passed to `.shuffle()`: a
    zero-size reservoir buffer is a no-op by definition, but `datasets`'
    streaming shuffle rejects it outright (`rng.integers(0, 0, ...)` raises
    `ValueError: high <= 0`).
    """
    rng = random.Random(config.seed)
    source = load_dataset(
        config.source.dataset_name,
        config.source.dataset_config,
        split=config.source.dataset_split,
        streaming=True,
    )
    if config.source.shuffle_buffer_size > 0:
        source = source.shuffle(
            seed=config.seed,
            buffer_size=config.source.shuffle_buffer_size,
        )
    samples: list[dict[str, list[int]]] = []
    for record in source:
        input_ids = tokenizer(record["text"])["input_ids"]
        if len(input_ids) < config.max_sequence_length:
            continue
        start = rng.randint(0, len(input_ids) - config.max_sequence_length)
        samples.append(
            {"input_ids": input_ids[start : start + config.max_sequence_length]}
        )
        if len(samples) == config.num_samples:
            break
    if len(samples) != config.num_samples:
        raise RuntimeError(
            f"expected {config.num_samples} calibration samples, got {len(samples)}"
        )
    return Dataset.from_list(samples)


def _load_prepared_prefix(
    source_root: Path,
    config: CalibrationConfig,
) -> Dataset:
    source_config = CalibrationConfig(
        profile=config.profile,
        model_id=config.model_id,
        seed=config.seed,
        num_samples=config.num_samples,
        max_sequence_length=config.max_sequence_length,
        calibration_data_root=source_root,
        source=config.source,
    )
    manifest_path = (
        source_config.prepared_dataset_dir / "calibration-data-manifest.json"
    )
    if not manifest_path.is_file():
        raise RuntimeError(f"missing source calibration manifest: {manifest_path}")
    source_manifest = _read_object(manifest_path)
    source_count = source_manifest.get("num_calibration_samples")
    if not isinstance(source_count, int) or source_count < config.num_samples:
        raise RuntimeError(
            f"source cache contains {source_count!r} samples; "
            f"need at least {config.num_samples}"
        )
    source_config = CalibrationConfig(
        profile=config.profile,
        model_id=config.model_id,
        seed=config.seed,
        num_samples=source_count,
        max_sequence_length=config.max_sequence_length,
        calibration_data_root=source_root,
        source=config.source,
    )
    return validate_prepared_dataset(source_config).select(range(config.num_samples))


def prepare_one(
    tokenizer: Any,
    config: CalibrationConfig,
    *,
    source_root: Path | None = None,
    collector: Callable[[Any, CalibrationConfig], Dataset] = collect_dataset,
) -> None:
    """Atomically create or validate one tokenized calibration cache."""
    if config.prepared_dataset_dir.exists():
        validate_prepared_dataset(config)
        print(f"Validated existing cache: {config.prepared_dataset_dir}", flush=True)
        return
    config.calibration_data_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".seed-{config.seed}-preparing-",
            dir=config.calibration_data_root,
        )
    )
    try:
        dataset = (
            _load_prepared_prefix(source_root, config)
            if source_root is not None
            else collector(tokenizer, config)
        )
        dataset.save_to_disk(str(temporary_dir))
        (temporary_dir / "calibration-data-manifest.json").write_text(
            json.dumps(config.data_manifest(), indent=2, sort_keys=True) + "\n"
        )
        validate_prepared_dataset(config, temporary_dir)
        temporary_dir.rename(config.prepared_dataset_dir)
        validate_prepared_dataset(config)
        print(f"Saved calibration cache: {config.prepared_dataset_dir}", flush=True)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def prepare_main(
    profile: BenchmarkProfile,
    *,
    source: CalibrationSource = FINEWEB_EDU_SOURCE,
    collector: Callable[[Any, CalibrationConfig], Dataset] = collect_dataset,
) -> int:
    description = (
        f"Prepare {source.dataset_name} calibration windows for "
        f"Table {profile.table_number}."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=profile.default_model)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(profile.run_root) / "calibration-data",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for seed, count in ((0, 2), (42, args.num_samples)):
        config = CalibrationConfig(
            profile=profile,
            model_id=args.model,
            seed=seed,
            num_samples=count,
            max_sequence_length=args.max_sequence_length,
            calibration_data_root=args.output_root,
            source=source,
        )
        prepare_one(
            tokenizer,
            config,
            source_root=args.source_root,
            collector=collector,
        )
    return 0


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]
    except KeyError as exc:
        raise ValueError(f"unsupported model dtype: {name}") from exc


def _build_recipe(args: argparse.Namespace) -> tuple[list[Any], Any]:
    from llmcompressor_osfp4.modifiers import OSFP4Modifier

    if args.method not in {
        "osfp4-rtn",
        "osfp4-sic",
        "osfp4-rtn-a16",
        "osfp4-sic-a16",
    }:
        raise ValueError(f"unsupported calibration method: {args.method}")
    is_a16 = args.method.endswith("-a16")
    scheme = "NVFP4A16" if is_a16 else "NVFP4"
    mode = args.method.removeprefix("osfp4-").removesuffix("-a16")
    modifier = OSFP4Modifier(
        scheme=scheme,
        targets=["Linear"],
        ignore=["lm_head"],
        optimization_mode=mode,
        steps=args.steps,
        lr=args.lr,
        activation_subsample_size=(None if is_a16 else args.activation_subsample_size),
    )

    def manifest(model: Any) -> dict[str, Any]:
        metadata = model.config.osfp4_metadata
        return {
            "method": args.method,
            "optimization_mode": modifier.optimization_mode,
            "steps": args.steps,
            "lr": args.lr,
            "osfp4_architecture": model.config.architectures[0],
            "smooth_quant_scale_targets": metadata["smooth_quant_scale_targets"],
            "scheme": scheme,
            "ignore": ["lm_head"],
            "activation_subsample_size": modifier.activation_subsample_size,
            "activation_subsampling": modifier.activation_subsampling_records,
        }

    return [modifier], manifest


def _checkpoint_manifest(
    config: CalibrationConfig,
    model: Any,
    recipe_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        **config.data_manifest(),
        "method": config.method,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "torch_dtype": config.profile.calibration_dtype,
        "recipe": recipe_manifest,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "datasets", "llmcompressor")
        },
    }


def calibrate_main(
    profile: BenchmarkProfile,
    *,
    source: CalibrationSource = FINEWEB_EDU_SOURCE,
) -> int:
    from llmcompressor import oneshot

    methods = tuple(method for method in profile.methods if method != "bf16")
    parser = argparse.ArgumentParser(
        description=f"Calibrate one Table {profile.table_number} method."
    )
    parser.add_argument("--method", required=True, choices=methods)
    parser.add_argument("--model", default=profile.default_model)
    parser.add_argument(
        "--calibration-data-root",
        type=Path,
        default=Path(profile.run_root) / "calibration-data",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--activation-subsample-size", type=int, default=16_384)
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = (
            Path(profile.run_root) / args.method / f"seed-{args.seed}" / "model"
        )

    config = CalibrationConfig(
        profile=profile,
        model_id=args.model,
        seed=args.seed,
        num_samples=args.num_samples,
        max_sequence_length=args.max_sequence_length,
        calibration_data_root=args.calibration_data_root,
        source=source,
        save_dir=args.save_dir,
        method=args.method,
    )
    torch.manual_seed(args.seed)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_torch_dtype(profile.calibration_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = validate_prepared_dataset(config)
    recipe, recipe_manifest_builder = _build_recipe(args)
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        pipeline="sequential",
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_samples,
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save_dir, save_compressed=True)
    tokenizer.save_pretrained(args.save_dir)
    recipe_manifest = recipe_manifest_builder(model)
    (args.save_dir / "calibration-manifest.json").write_text(
        json.dumps(
            _checkpoint_manifest(config, model, recipe_manifest),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Saved {args.method} checkpoint to {args.save_dir}", flush=True)
    return 0
