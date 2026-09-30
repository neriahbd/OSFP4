"""Real deployment failures must never silently apply smoothing twice."""

import pytest
import torch

from llmcompressor.core import Event, State
from llmcompressor_osfp4.modifiers import base, osfp4_quantize, smoothing

from ._testing import LlamaForCausalLM, make_osfp4_modifier


@pytest.mark.parametrize("mode", ["rtn", "sic"])
@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("stage", ["smoothing", "replay", "installation", "hook"])
def test_partial_deployment_requires_fresh_model_and_modifier(
    monkeypatch, mode, scheme, runtime, stage
):
    if stage == "replay" and scheme == "NVFP4A16":
        return  # Weight-only mappings have no input observer or replay.
    torch.manual_seed(21)
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(
        scheme=scheme,
        optimization_mode=mode,
        steps=1,
        targets=["re:.*o_proj" if runtime else "re:.*[qkv]_proj"],
    )
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))
    mapping = modifier._resolved_mappings[0]
    first_weight = mapping.balance_layers[0].weight.detach().clone()
    error = RuntimeError("injected failure after mutation")

    def after_write(original):
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise error

        return fail

    with monkeypatch.context() as patch:
        if stage == "smoothing":
            patch.setattr(
                smoothing,
                "update_offload_parameter",
                after_write(smoothing.update_offload_parameter),
            )
        elif stage == "replay":
            patch.setattr(
                osfp4_quantize,
                "_update_smoothed_input_global_scale",
                after_write(osfp4_quantize._update_smoothed_input_global_scale),
            )
        elif stage == "installation":
            patch.setattr(
                base,
                "update_offload_parameter",
                after_write(base.update_offload_parameter),
            )
        else:
            patch.setattr(
                base.OSFP4Modifier,
                "_register_runtime_smoothing_hook",
                after_write(base.OSFP4Modifier._register_runtime_smoothing_hook),
            )
        with pytest.raises(RuntimeError, match="Reload a fresh model") as raised:
            modifier._optimize_mapping(mapping)
    assert raised.value.__cause__ is error
    assert not torch.equal(first_weight, mapping.balance_layers[0].weight)
    assert mapping.mapping_name not in modifier._optimized_mapping_names
    assert modifier._calibration.has_observation(mapping.mapping_name)
    assert "_deployment_failure" not in modifier.model_dump()
    assert not hasattr(model.config, "osfp4_metadata")

    snapshot = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for action in (
        lambda: modifier._optimize_mapping(mapping),
        modifier._optimize_available_mappings,
        lambda: modifier.on_calibration_start(state, Event()),
        lambda: modifier.on_sequential_epoch_end(state, Event(), []),
        lambda: modifier.on_calibration_end(state, Event()),
        lambda: modifier.on_finalize(state),
        lambda: modifier.on_initialize(State(model=LlamaForCausalLM())),
    ):
        with pytest.raises(RuntimeError, match="cannot be retried"):
            action()
    assert snapshot.keys() == model.state_dict().keys()
    for name, expected in snapshot.items():
        assert torch.equal(
            expected.view(torch.uint8), model.state_dict()[name].view(torch.uint8)
        )
    modifier.remove_hooks()
    modifier._remove_runtime_smoothing_hooks()


def test_optimizer_failure_before_deployment_is_retryable(monkeypatch):
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(
        optimization_mode="rtn", steps=1, targets=["re:.*o_proj"]
    )
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))
    mapping = modifier._resolved_mappings[0]
    before = mapping.balance_layers[0].weight.detach().clone()
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("optimizer failed")

        patch.setattr(osfp4_quantize, "optimize_rtn", fail)
        with pytest.raises(RuntimeError, match="optimizer failed"):
            modifier._optimize_mapping(mapping)
    assert modifier._deployment_failure is None
    assert torch.equal(before, mapping.balance_layers[0].weight)
    modifier._optimize_mapping(mapping)
    modifier.on_calibration_end(state, Event())
    modifier.on_finalize(state)
    assert modifier._finalization_complete
