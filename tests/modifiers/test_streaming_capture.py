"""Bounded capture must preserve the legacy sample and checkpoint tensor bytes."""

import random
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args import DatasetArguments
from llmcompressor.core import Event, State, create_session
from llmcompressor_osfp4.modifiers import OSFP4Modifier
from llmcompressor_osfp4.modifiers.activation_subsampling import subsample_activations
from llmcompressor_osfp4.modifiers.calibration_cache import OSFP4CalibrationCache
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.pipelines.sequential import pipeline as sequential
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)

from ._testing import LlamaForCausalLM as TinyModel

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
]


def assert_bytes(actual, expected, name="tensor"):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, name
    assert torch.equal(
        actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
    ), name


@pytest.mark.parametrize("embeddings", [False, True])
def test_automatic_count_uses_collated_rows_and_preserves_rng(embeddings):
    def collate(rows):
        random.random()
        np.random.random()
        torch.rand(1)
        padded = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
        return (
            {"inputs_embeds": padded[..., None].expand(-1, -1, 16)}
            if embeddings
            else {"input_ids": padded}
        )

    generator = torch.Generator().manual_seed(7)
    loader = torch.utils.data.DataLoader(
        [torch.arange(n) for n in (2, 7, 3, 5, 4)],
        batch_size=2,
        shuffle=True,
        drop_last=True,
        generator=generator,
        collate_fn=collate,
    )
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng, loader_rng = torch.get_rng_state(), generator.get_state()
    count = OSFP4Modifier._calibration_token_count(loader)
    assert random.getstate() == python_rng
    assert np.random.get_state()[0] == numpy_rng[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:]
    assert_bytes(torch.get_rng_state(), torch_rng)
    assert_bytes(generator.get_state(), loader_rng)
    assert count == sum(
        next(iter(batch.values())).shape[0] * next(iter(batch.values())).shape[1]
        for batch in loader
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cap", [3, 11, 20])
def test_capture_matches_legacy_bytes_and_statistics(device, cap):
    batches = [
        torch.randn(n, 16, generator=torch.Generator().manual_seed(n)) for n in (6, 5)
    ]
    batches[0][:, 0] = -0.0
    expected = subsample_activations(batches, cap, output_rows=4)
    layer = torch.nn.Linear(16, 4).to(device)
    cache = OSFP4CalibrationCache()
    hook = cache.make_capture_hook(
        "x",
        capture_hessian=True,
        capture_sigma_x_squared=True,
        expected_tokens=11,
        subsample_size=cap,
    )
    hessian, energy = torch.zeros(16, 16, device=device), torch.zeros(16, device=device)
    rng = torch.get_rng_state().clone()
    for batch in batches:
        value = batch.to(device)
        hook(layer, (value,))
        hessian.addmm_(value.T, value)
        energy.add_(value.square().sum(0))
        assert sum(x.shape[0] for x in cache.inputs.get("x", ())) <= cap
    cache.wait("x", torch.device("cpu"))
    actual = cache.sampled_inputs("x", output_rows=4)
    assert actual.provenance == expected.provenance
    assert_bytes(
        torch.cat(actual.batches or cache.inputs["x"]),
        torch.cat(expected.batches or batches),
    )
    assert_bytes(cache.hessian["x"], hessian)
    assert_bytes(cache.sigma_x_squared["x"], energy)
    assert_bytes(cache.input_absmax["x"], torch.cat(batches).abs().amax(0))
    assert torch.equal(rng, torch.get_rng_state())
    cache.clear_mapping("x")
    assert (
        not cache.sample_indices
        and not cache.expected_tokens
        and not cache.input_absmax
    )


def test_count_does_not_consume_one_shot_data():
    batches = iter([{"input_ids": torch.arange(5)}])
    assert OSFP4Modifier._calibration_token_count(batches) is None
    assert_bytes(next(batches)["input_ids"], torch.arange(5))

    class OneShotDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            raise AssertionError("must not consume iterable data to count it")

    loader = torch.utils.data.DataLoader(OneShotDataset())
    assert OSFP4Modifier._calibration_token_count(loader) is None


@pytest.mark.parametrize("expected", [10, 12])
def test_incorrect_count_fails_before_optimization(expected):
    model = TinyModel()
    modifier = OSFP4Modifier(
        scheme="NVFP4",
        ignore=["lm_head"],
        activation_subsample_size=3,
    )
    state, event = State(model=model), Event()
    state.data.calib = [{"input_ids": torch.zeros(1, expected, dtype=torch.long)}]
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, event)
    try:
        with patch.object(
            OSFP4Modifier,
            "_deploy_mapping",
            side_effect=AssertionError("must not deploy"),
        ):
            with pytest.raises(ValueError, match="calibration tokens"):
                model(torch.randn(1, 11, 16))
                modifier.on_sequential_epoch_end(state, event, list(model.modules()))
    finally:
        modifier.remove_hooks()


@pytest.mark.parametrize("fallback", ["unknown", "uncapped", "custom", "weight_only"])
def test_existing_capture_fallback(fallback):
    model = TinyModel()
    modifier = OSFP4Modifier(
        scheme="NVFP4A16" if fallback == "weight_only" else "NVFP4",
        ignore=["lm_head"],
        activation_subsample_size=None if fallback == "uncapped" else 3,
    )
    state = State(model=model)
    if fallback != "unknown":
        state.data.calib = [{"input_ids": torch.zeros(1, 11, dtype=torch.long)}]
    modifier.on_initialize(state)
    if fallback == "custom":
        # Use a real custom configuration before observers are attached.
        for layer in model.modules():
            scheme = getattr(layer, "quantization_scheme", None)
            if scheme is not None:
                scheme.input_activations.observer = "minmax"
    modifier.on_calibration_start(state, Event())
    try:
        model(torch.randn(1, 11, 16))
        assert not modifier._calibration.sample_indices
        for batches in modifier._calibration.inputs.values():
            assert sum(batch.shape[0] for batch in batches) == 11
        assert bool(modifier._calibration.inputs) == (fallback != "weight_only")
    finally:
        modifier.remove_hooks()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_pipeline_and_saved_checkpoint_byte_parity(tmp_path, device, dtype, mode):
    def collate(rows):
        ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
        return {"input_ids": ids, "attention_mask": ids.ne(0).long()}

    snapshots = []
    for streaming in (False, True):
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
        model = LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
        modify_save_pretrained(model)
        targets = [
            name
            for name, layer in model.named_modules()
            if isinstance(layer, torch.nn.Linear) and name != "lm_head"
        ]
        modifier = OSFP4Modifier(
            scheme="NVFP4",
            ignore=["lm_head"],
            optimization_mode=mode,
            steps=1,
            activation_subsample_size=3,
        )
        loader = torch.utils.data.DataLoader(
            [torch.arange(1, n + 1) for n in (7, 5, 3)],
            batch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(17),
            collate_fn=collate,
        )
        with ExitStack() as stack:
            if not streaming:
                stack.enter_context(
                    patch.object(
                        OSFP4Modifier, "_calibration_token_count", return_value=None
                    )
                )
            capture = stack.enter_context(
                patch.object(
                    modifier._calibration,
                    "make_capture_hook",
                    wraps=modifier._calibration.make_capture_hook,
                )
            )
            stack.enter_context(
                patch.object(
                    sequential, "get_main_device", lambda: torch.device(device)
                )
            )
            session = stack.enter_context(create_session())
            session.initialize(
                model=model, recipe=[modifier], start=-1, calib_data=loader
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
        assert capture.call_args_list
        total = next(iter(modifier.activation_subsampling_records.values()))["k1"]
        assert all(
            call.kwargs["expected_tokens"] == (total if streaming else None)
            for call in capture.call_args_list
        )
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        folder = tmp_path / str(streaming)
        model.save_pretrained(folder, save_compressed=True)
        packed = {
            name: value
            for path in folder.glob("*.safetensors")
            for name, value in load_file(path).items()
        }
        for tensors in (state, packed):
            for kind in ("input", "weight"):
                assert {
                    name for name in tensors if name.endswith(f"{kind}_global_scale")
                } == {f"{name}.{kind}_global_scale" for name in targets}
        snapshots.append((state, packed, modifier.activation_subsampling_records))
    assert snapshots[0][2] == snapshots[1][2]
    for before, after in zip(snapshots[0][:2], snapshots[1][:2]):
        assert before.keys() == after.keys()
        for name in before:
            assert_bytes(after[name], before[name], name)
