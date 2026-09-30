#!/usr/bin/env python3
"""Calibrate a plain (non-OSFP4) NVFP4 baseline on the WikiText-2 train cache.

Companion to `calibrate.py`: same calibration cache, same checkpoint/manifest
layout, but quantizes with a stock llmcompressor modifier (`QuantizationModifier`
RTN or `GPTQModifier`) instead of `OSFP4Modifier`, so the two are directly
comparable via `evaluate.py --kind plain`.
"""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from examples.fpquant.common.calibration import (  # noqa: E402
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_NUM_SAMPLES,
    CalibrationConfig,
    _torch_dtype,
    validate_prepared_dataset,
)
from examples.wikitext_ppl.config import (  # noqa: E402
    CALIBRATION_SOURCE,
    PROFILE,
)

METHODS = ("rtn-nvfp4", "gptq-nvfp4", "rtn-nvfp4a16", "gptq-nvfp4a16")


def build_recipe(method: str):
    from compressed_tensors.quantization import (
        FP8_E4M3_DATA,
        DynamicType,
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    if method == "rtn-nvfp4":
        from llmcompressor.modifiers.quantization import QuantizationModifier

        modifier = QuantizationModifier(
            targets="Linear", scheme="NVFP4", ignore=["lm_head"]
        )
        manifest = {"method": method, "scheme": "NVFP4", "ignore": ["lm_head"]}
    elif method == "gptq-nvfp4":
        from llmcompressor.modifiers.gptq import GPTQModifier

        nvfp4 = dict(
            weights=QuantizationArgs(
                num_bits=4,
                actorder="static",
                type=QuantizationType.FLOAT,
                strategy=QuantizationStrategy.TENSOR_GROUP,
                symmetric=True,
                dynamic=False,
                group_size=16,
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=FP8_E4M3_DATA.dtype,
                observer="memoryless_minmax",
            ),
            input_activations=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.FLOAT,
                strategy=QuantizationStrategy.TENSOR_GROUP,
                symmetric=True,
                dynamic=DynamicType.LOCAL,
                group_size=16,
                observer="static_minmax",
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=FP8_E4M3_DATA.dtype,
            ),
            targets=["Linear"],
        )
        modifier = GPTQModifier(config_groups={"group_0": nvfp4}, ignore=["lm_head"])
        manifest = {"method": method, "scheme": "NVFP4", "ignore": ["lm_head"]}
    elif method == "rtn-nvfp4a16":
        from llmcompressor.modifiers.quantization import QuantizationModifier

        modifier = QuantizationModifier(
            targets="Linear", scheme="NVFP4A16", ignore=["lm_head"]
        )
        manifest = {"method": method, "scheme": "NVFP4A16", "ignore": ["lm_head"]}
    elif method == "gptq-nvfp4a16":
        from llmcompressor.modifiers.gptq import GPTQModifier

        modifier = GPTQModifier(scheme="NVFP4A16", targets="Linear", ignore=["lm_head"])
        manifest = {"method": method, "scheme": "NVFP4A16", "ignore": ["lm_head"]}
    else:
        raise ValueError(f"unsupported method: {method}")
    return [modifier], manifest


def main() -> int:
    from llmcompressor import oneshot

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--model", default=PROFILE.default_model)
    parser.add_argument("--calibration-data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument("--save-dir", type=Path, required=True)
    args = parser.parse_args()

    config = CalibrationConfig(
        profile=PROFILE,
        model_id=args.model,
        seed=args.seed,
        num_samples=args.num_samples,
        max_sequence_length=args.max_sequence_length,
        calibration_data_root=args.calibration_data_root,
        source=CALIBRATION_SOURCE,
        save_dir=args.save_dir,
        method=args.method,
    )
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=_torch_dtype(PROFILE.calibration_dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = validate_prepared_dataset(config)
    recipe, recipe_manifest = build_recipe(args.method)

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
    (args.save_dir / "calibration-manifest.json").write_text(
        json.dumps(
            {
                **config.data_manifest(),
                "method": args.method,
                "model_revision": getattr(model.config, "_commit_hash", None),
                "torch_dtype": PROFILE.calibration_dtype,
                "recipe": recipe_manifest,
                "platform": platform.platform(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "datasets", "llmcompressor")
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Saved {args.method} checkpoint to {args.save_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
