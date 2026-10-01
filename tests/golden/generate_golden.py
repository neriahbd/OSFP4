"""Capture OSFP4 calibration outputs on a tiny seeded Llama.

Run once in an environment with the original fork installed to produce the
reference fixtures, then ``tests/test_golden.py`` replays the same runs against
this plugin on stock llm-compressor and compares state dicts byte for byte.

    python tests/golden/generate_golden.py --output tests/golden/fixtures
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.pipelines.sequential import pipeline as sequential

try:  # plugin on stock llm-compressor
    from llmcompressor_osfp4 import OSFP4Modifier
except ImportError:  # original fork
    from llmcompressor.modifiers.osfp4 import OSFP4Modifier

CASES = [
    (scheme, mode) for scheme in ("NVFP4", "NVFP4A16") for mode in ("rtn", "sic")
]


def run_case(scheme: str, mode: str) -> tuple[dict, dict[str, torch.Tensor]]:
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    generator = torch.Generator().manual_seed(0)
    samples = [
        {"input_ids": torch.randint(0, 64, (1, 16), generator=generator)}
        for _ in range(4)
    ]
    loader = torch.utils.data.DataLoader(samples, batch_size=None)
    modifier = OSFP4Modifier(
        scheme=scheme,
        optimization_mode=mode,
        ignore=["lm_head"],
        steps=2,
        activation_subsample_size=16384,
    )
    with ExitStack() as stack:
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
        pipeline = CalibrationPipeline.from_modifiers(
            session.lifecycle.recipe.modifiers
        )
        pipeline(
            model,
            loader,
            DatasetArguments(
                sequential_targets=["LlamaDecoderLayer"],
                sequential_offload_device="cpu",
                propagate_error=True,
            ),
        )
        session.finalize()
    state = {
        name: value.detach().contiguous().clone()
        for name, value in model.state_dict().items()
    }
    return dict(model.config.osfp4_metadata), state


def main() -> None:
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for scheme, mode in CASES:
        metadata, state = run_case(scheme, mode)
        stem = f"{scheme.lower()}-{mode}"
        save_file(state, args.output / f"{stem}.safetensors")
        (args.output / f"{stem}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {stem}: {len(state)} tensors")


if __name__ == "__main__":
    main()
