"""Repeated finalization must preserve metadata and every deployed tensor byte."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from llmcompressor.core import Event, State
from llmcompressor_osfp4.modifiers import base
from llmcompressor_osfp4.modifiers.calibration_cache import OSFP4CalibrationCache

from ._testing import LlamaForCausalLM, make_osfp4_modifier, run_modifier
from .compare_base import Trace, compare


def _snapshot(model):
    return [
        (
            name,
            value.shape,
            value.dtype,
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).clone(),
        )
        for name, value in model.state_dict().items()
    ]


def _assert_unchanged(model, expected):
    actual = _snapshot(model)
    assert len(actual) == len(expected)
    for (name, shape, dtype, data), (old_name, old_shape, old_dtype, old_data) in zip(
        actual, expected
    ):
        assert (name, shape, dtype) == (old_name, old_shape, old_dtype)
        assert torch.equal(data, old_data), name


@pytest.mark.parametrize("mode", ["rtn", "sic"])
@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("selection", ["runtime", "norm", "empty"])
def test_repeated_finalize_is_a_byte_exact_noop(
    monkeypatch, mode, scheme, dtype, selection
):
    torch.manual_seed(42)
    model = LlamaForCausalLM().to(dtype)
    targets = {
        "runtime": ["Linear"],
        "norm": ["re:.*[qkv]_proj"],
        "empty": [],
    }[selection]
    modifier = make_osfp4_modifier(
        optimization_mode=mode, scheme=scheme, steps=1, targets=targets
    )
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16, dtype=dtype))
    modifier.on_sequential_epoch_end(state, Event(), list(model.modules()))
    modifier.on_calibration_end(state, Event())
    before = _snapshot(model)
    expected_targets = sorted(
        mapping.mapping_name
        for mapping in modifier._resolved_mappings
        if mapping.requires_runtime_smoothing
    )
    assert bool(expected_targets) == (selection == "runtime")
    assert not modifier._finalization_complete
    assert modifier.on_finalize(state) is True
    assert modifier._finalization_complete
    _assert_unchanged(model, before)
    metadata = deepcopy(model.config.osfp4_metadata)
    assert metadata["smooth_quant_scale_targets"] == expected_targets
    save = model.save_pretrained
    assert not modifier._resolved_mappings
    assert not modifier._runtime_smoothing_hooks
    assert not modifier._calibration.sample_count

    def unexpected(*args, **kwargs):
        raise AssertionError("Repeated finalization performed work")

    monkeypatch.setattr(base, "attach_osfp4_runtime_contract", unexpected)
    monkeypatch.setattr(base, "update_offload_parameter", unexpected)
    monkeypatch.setattr(base.OSFP4Modifier, "_require_complete_calibration", unexpected)
    monkeypatch.setattr(
        base.OSFP4Modifier, "_remove_runtime_smoothing_hooks", unexpected
    )
    monkeypatch.setattr(OSFP4CalibrationCache, "clear_all", unexpected)
    assert modifier.on_finalize(state) is True
    assert model.config.osfp4_metadata == metadata
    assert model.save_pretrained is save
    _assert_unchanged(model, before)


@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
def test_missing_calibration_does_not_mark_finalization_complete(scheme):
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(scheme=scheme)
    state = State(model=model)
    modifier.on_initialize(state)
    save = model.save_pretrained
    for _ in range(2):
        with pytest.raises(RuntimeError, match="missing"):
            modifier.on_finalize(state)
        assert not modifier._finalization_complete
        assert modifier._resolved_mappings
        assert model.save_pretrained == save
        assert not hasattr(model.config, "osfp4_metadata")


def test_unfinished_finalize_rejects_deployment_failure():
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier()
    state = State(model=model)
    modifier.on_initialize(state)
    modifier._deployment_failure = ("test_mapping", "smoothing")
    save = model.save_pretrained

    with pytest.raises(RuntimeError, match="cannot be retried"):
        modifier.on_finalize(state)

    assert not modifier._finalization_complete
    assert model.save_pretrained == save
    assert not hasattr(model.config, "osfp4_metadata")


def test_failed_attachment_propagates_and_allows_retry():
    model, modifier = run_modifier(optimization_mode="rtn")
    state = State(model=model)
    mappings = modifier._resolved_mappings.copy()
    hooks = modifier._runtime_smoothing_hooks.copy()
    error = OSError("attachment failed")
    with patch.object(base, "attach_osfp4_runtime_contract", side_effect=error):
        with pytest.raises(OSError) as raised:
            modifier.on_finalize(state)
    assert raised.value is error
    assert not modifier._finalization_complete
    assert modifier._resolved_mappings == mappings
    assert modifier._runtime_smoothing_hooks == hooks
    assert modifier.on_finalize(state) is True
    assert modifier._finalization_complete


def test_normal_finalize_still_rejects_duplicate_calls():
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(steps=0)
    state = State(model=model)
    modifier.initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))
    modifier.on_sequential_epoch_end(state, Event(), list(model.modules()))
    modifier.on_calibration_end(state, Event())
    modifier.finalize(state)
    assert modifier.finalized_
    assert modifier._finalization_complete
    with pytest.raises(RuntimeError, match="cannot finalize a modifier twice"):
        modifier.finalize(state)


def test_successful_direct_initialize_resets_completion():
    model, modifier = run_modifier(optimization_mode="rtn")
    modifier.on_finalize(State(model=model))
    state = State(model=LlamaForCausalLM())
    assert modifier.on_initialize(state)
    assert not modifier._finalization_complete
    with pytest.raises(RuntimeError, match="missing"):
        modifier.on_finalize(state)


def test_failed_initialization_does_not_reset_completion():
    model, modifier = run_modifier(optimization_mode="rtn")
    modifier.on_finalize(State(model=model))
    with patch.object(
        base.QuantizationMixin,
        "initialize_quantization",
        side_effect=RuntimeError("initialization failed"),
    ):
        with pytest.raises(RuntimeError, match="initialization failed"):
            modifier.on_initialize(State(model=LlamaForCausalLM()))
    assert modifier._finalization_complete


@pytest.mark.parametrize("mode", ["rtn", "sic"])
@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
def test_compressed_saves_match_after_one_and_two_finalizations(tmp_path, mode, scheme):
    import json

    from safetensors.torch import load_file
    from transformers import LlamaConfig
    from transformers import LlamaForCausalLM as TransformersLlama

    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )

    # Saving packs weights in place; compare independently calibrated models
    # so each is saved once, after one versus two finalization calls.
    for index in (1, 2):
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
        model = TransformersLlama(config).cpu().eval()
        modify_save_pretrained(model)
        modifier = make_osfp4_modifier(optimization_mode=mode, scheme=scheme, steps=1)
        state = State(model=model)
        modifier.on_initialize(state)
        modifier.on_calibration_start(state, Event())
        with torch.no_grad():
            model(input_ids=torch.arange(24).reshape(2, 12))
        modifier.on_sequential_epoch_end(state, Event(), list(model.modules()))
        modifier.on_calibration_end(state, Event())
        modifier.on_finalize(state)
        metadata = deepcopy(model.config.osfp4_metadata)
        save = model.save_pretrained
        assert len(metadata["smooth_quant_scale_targets"]) == 4

        if index == 2:
            before = _snapshot(model)
            assert modifier.on_finalize(state) is True
            _assert_unchanged(model, before)
            assert model.save_pretrained is save
        directory = tmp_path / str(index)
        model.save_pretrained(directory, save_compressed=True)
        packed = {}
        for path in sorted(directory.glob("*.safetensors")):
            packed.update(load_file(path))
        saved_config = json.loads((directory / "config.json").read_text())
        assert saved_config["osfp4_metadata"] == metadata
        assert saved_config["quantization_config"]["quant_method"] == "osfp4"
        assert any(name.endswith("weight_packed") for name in packed)
        assert any(name.endswith("smooth_quant_scale") for name in packed)
        trace = Trace()
        trace.add("packed", dict(sorted(packed.items())))
        trace.add("config", saved_config)
        trace.save(directory)
    assert compare(tmp_path / "1", tmp_path / "2") > 0
