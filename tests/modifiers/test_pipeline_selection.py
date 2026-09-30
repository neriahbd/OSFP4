"""Default independent inference must execute the same sequential calibration."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor_osfp4.modifiers import OSFP4Modifier
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.pipelines.sequential import pipeline as sequential


@pytest.mark.parametrize("mode", ["rtn", "sic"])
@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
def test_automatic_and_default_independent_match_explicit_sequential(mode, scheme):
    snapshots = []
    for selection in ("sequential", None, "independent"):
        torch.manual_seed(42)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).eval()
        modifier = OSFP4Modifier(
            scheme=scheme, optimization_mode=mode, ignore=["lm_head"], steps=1
        )
        loader = torch.utils.data.DataLoader(
            [{"input_ids": torch.arange(8).reshape(1, 8)}], batch_size=None
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
                session.lifecycle.recipe.modifiers, user=selection
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
        snapshots.append(
            (
                model.config.osfp4_metadata,
                {
                    name: (
                        value.dtype,
                        value.shape,
                        value.detach()
                        .contiguous()
                        .reshape(-1)
                        .view(torch.uint8)
                        .clone(),
                    )
                    for name, value in model.state_dict().items()
                },
            )
        )
    metadata, expected = snapshots[0]
    for actual_metadata, actual in snapshots[1:]:
        assert actual_metadata == metadata
        assert list(actual) == list(expected)
        for name, (dtype, shape, values) in expected.items():
            assert actual[name][:2] == (dtype, shape)
            assert torch.equal(actual[name][2], values), name
