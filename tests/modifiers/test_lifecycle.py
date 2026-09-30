import pytest
import torch
from compressed_tensors.offload import OffloadCache, offload_module

from llmcompressor.core import Event, State
from llmcompressor_osfp4.modifiers import OSFP4Modifier as _OSFP4Modifier
from llmcompressor_osfp4.modifiers import osfp4_quantize
from llmcompressor_osfp4.modifiers.base import OSFP4Mapping
from llmcompressor_osfp4.observers import observer as osfp4_observer

from ._testing import LlamaForCausalLM, make_osfp4_modifier, run_modifier


@pytest.mark.parametrize(
    ("mode", "captures_hessian"),
    [("rtn", False), ("sic", True)],
)
def test_only_sic_captures_mapping_hessians(mode, captures_hessian):
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(steps=0, optimization_mode=mode)
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))

    assert modifier._calibration.inputs
    for mapping in modifier._resolved_mappings:
        assert (
            mapping.mapping_name in modifier._calibration.hessian
        ) is captures_hessian
        assert (mapping.mapping_name in modifier._calibration.sigma_x_squared) is (
            mode == "rtn"
        )

    modifier.remove_hooks()


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_nvfp4a16_captures_only_required_statistics(mode):
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(
        scheme="NVFP4A16",
        steps=0,
        optimization_mode=mode,
    )
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))

    assert not modifier._calibration.inputs
    assert set(modifier._calibration.sample_count) == {
        mapping.mapping_name for mapping in modifier._resolved_mappings
    }
    if mode == "sic":
        assert not modifier._calibration.sigma_x_squared
        assert set(modifier._calibration.hessian) == set(
            modifier._calibration.sample_count
        )
    else:
        assert not modifier._calibration.hessian
        assert set(modifier._calibration.sigma_x_squared) == set(
            modifier._calibration.sample_count
        )
    modifier.remove_hooks()


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_nvfp4a16_lifecycle_creates_no_activation_quantization_state(mode):
    model, modifier = run_modifier(
        scheme="NVFP4A16",
        optimization_mode=mode,
    )
    targets = [
        layer for layer in model.modules() if hasattr(layer, "quantization_scheme")
    ]

    assert targets
    for layer in targets:
        assert layer.quantization_scheme.input_activations is None
        assert not hasattr(layer, "input_observer")
        assert not hasattr(layer, "input_global_scale")
        assert not hasattr(layer, "input_scale")
        assert not hasattr(layer, "input_zero_point")
    runtime_smooth_quant_scale_mappings = [
        mapping
        for mapping in modifier._resolved_mappings
        if mapping.requires_runtime_smoothing
    ]
    assert runtime_smooth_quant_scale_mappings
    assert all(
        mapping.balance_layers[0].smooth_quant_scale.dtype == torch.bfloat16
        for mapping in runtime_smooth_quant_scale_mappings
    )


def test_calibration_starts_with_nvfp4_disabled_and_mixin_observers():
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(steps=0)
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())

    targets = [
        layer for layer in model.modules() if hasattr(layer, "quantization_scheme")
    ]
    assert targets
    for layer in targets:
        assert layer.quantization_status.name == "CALIBRATION"
        assert not layer.quantization_enabled
        assert hasattr(layer, "input_observer")
        assert layer.weight_observer.args.observer_kwargs["mode"] == "sic"
        assert not hasattr(layer, "output_observer")

    assert not hasattr(model.lm_head, "quantization_scheme")
    modifier.remove_hooks()


def test_mapping_optimization_reuses_first_balance_layer_observer(monkeypatch):
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(steps=0)
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))
    mapping = modifier._resolved_mappings[0]

    expected_observer = mapping.balance_layers[0].weight_observer
    expected_observer(mapping.balance_layers[0].weight)
    min_vals = expected_observer.min_vals.detach().clone()
    max_vals = expected_observer.max_vals.detach().clone()
    fusions = tuple(expected_observer.fusion_handler._group)
    assert fusions
    original = osfp4_quantize.optimize_sic
    seen = []

    def record_observer(*args, **kwargs):
        seen.append(kwargs["observer"])
        return original(*args, **kwargs)

    monkeypatch.setattr(osfp4_quantize, "optimize_sic", record_observer)
    modifier._optimize_mapping(mapping)

    assert seen == [expected_observer]
    assert torch.equal(expected_observer.min_vals, min_vals)
    assert torch.equal(expected_observer.max_vals, max_vals)
    assert tuple(expected_observer.fusion_handler._group) == fusions
    assert mapping.mapping_name in modifier._optimized_mapping_names
    modifier.remove_hooks()


def test_distributed_calibration_is_rejected(monkeypatch):
    from llmcompressor_osfp4.modifiers import base

    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(steps=0)
    state = State(model=model)
    modifier.on_initialize(state)
    monkeypatch.setattr(base, "is_distributed", lambda: True)

    with pytest.raises(NotImplementedError, match="single-process"):
        modifier.on_calibration_start(state, Event())


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_finalize_removes_only_temporary_runtime_smoothing_hooks(mode):
    model, modifier = run_modifier(optimization_mode=mode)
    runtime_smooth_quant_scale_layers = [
        mapping.balance_layers[0]
        for mapping in modifier._resolved_mappings
        if mapping.requires_runtime_smoothing
    ]
    smooth_layer_mapping_layers = [
        layer
        for mapping in modifier._resolved_mappings
        if not mapping.requires_runtime_smoothing
        for layer in mapping.balance_layers
    ]
    assert runtime_smooth_quant_scale_layers
    hook_ids = {
        layer: modifier._runtime_smoothing_hooks[layer].id
        for layer in runtime_smooth_quant_scale_layers
    }
    for mapping in modifier._resolved_mappings:
        modifier._register_runtime_smoothing_hook(mapping)
    assert hook_ids == {
        layer: modifier._runtime_smoothing_hooks[layer].id
        for layer in runtime_smooth_quant_scale_layers
    }
    saved_runtime_smooth_quant_scales = {
        layer: layer.smooth_quant_scale.detach().clone()
        for layer in runtime_smooth_quant_scale_layers
    }
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    assert all(
        layer not in modifier._runtime_smoothing_hooks
        for layer in smooth_layer_mapping_layers
    )

    modifier.on_finalize(State(model=model))

    for layer in runtime_smooth_quant_scale_layers:
        assert layer not in modifier._runtime_smoothing_hooks
        assert hook_ids[layer] not in layer._forward_pre_hooks
        assert "smooth_quant_scale" in layer._parameters
        assert torch.equal(
            layer.smooth_quant_scale,
            saved_runtime_smooth_quant_scales[layer],
        )
    assert tuple(model.state_dict()) == tuple(state_before)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name])


def test_finalize_does_not_reinstall_qparams(monkeypatch):
    from llmcompressor_osfp4.modifiers import base

    model, modifier = run_modifier(optimization_mode="rtn")

    def fail(*_args, **_kwargs):
        raise AssertionError("finalization reinstalled qparams")

    monkeypatch.setattr(base, "update_offload_parameter", fail)
    modifier.on_finalize(State(model=model))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("offload_hessians", [False, True])
def test_cuda_capture_uses_pinned_buffers_events_and_device_statistics(
    offload_hessians,
):
    model = LlamaForCausalLM().cuda()
    modifier = make_osfp4_modifier(
        steps=0,
        optimization_mode="sic",
        offload_hessians=offload_hessians,
    )
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16, device="cuda"))

    for mapping in modifier._resolved_mappings:
        cached = modifier._calibration.inputs[mapping.mapping_name]
        events = modifier._calibration.events[mapping.mapping_name]
        assert cached and len(events) == 1
        assert all(value.device.type == "cpu" and value.is_pinned() for value in cached)
        assert all(event is not None for event in events.values())
        assert not hasattr(modifier._calibration, "channel_absmax")
        expected_hessian_device = "cpu" if offload_hessians else "cuda"
        assert (
            modifier._calibration.hessian[mapping.mapping_name].device.type
            == expected_hessian_device
        )
        full_weight = torch.cat(
            [layer.weight.detach().float() for layer in mapping.balance_layers],
            dim=0,
        )
        modifier._calibration.wait(mapping.mapping_name, full_weight.device)

    torch.cuda.synchronize()
    modifier.remove_hooks()


def test_mapping_cache_is_retained_when_optimization_fails(monkeypatch):
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(optimization_mode="rtn", steps=0)
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))
    mapping = next(
        mapping
        for mapping in modifier._resolved_mappings
        if mapping.requires_runtime_smoothing
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("optimization failed")

    monkeypatch.setattr(osfp4_observer, "_optimize_joint_scales", fail)
    with pytest.raises(RuntimeError, match="optimization failed"):
        modifier._optimize_mapping(mapping)

    assert mapping.mapping_name in modifier._calibration.inputs
    assert mapping.mapping_name not in modifier._optimized_mapping_names
    assert mapping.balance_layers[0] not in modifier._runtime_smoothing_hooks


def test_nvfp4a16_statistics_are_retained_when_optimization_fails(monkeypatch):
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(
        scheme="NVFP4A16",
        optimization_mode="rtn",
        steps=0,
    )
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))
    mapping = modifier._resolved_mappings[0]

    def fail(*_args, **_kwargs):
        raise RuntimeError("optimization failed")

    monkeypatch.setattr(osfp4_observer, "_optimize_weight_scales", fail)
    with pytest.raises(RuntimeError, match="optimization failed"):
        modifier._optimize_mapping(mapping)

    assert mapping.mapping_name in modifier._calibration.sigma_x_squared
    assert mapping.mapping_name in modifier._calibration.sample_count
    assert mapping.mapping_name not in modifier._optimized_mapping_names
    modifier.remove_hooks()


def test_observed_mapping_is_optimized_only_once(monkeypatch):
    from llmcompressor_osfp4.modifiers import base

    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(steps=0)
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))

    optimized_names = []

    def record(mapping, *_args, **_kwargs):
        optimized_names.append(mapping.mapping_name)
        return []

    monkeypatch.setattr(base, "quantize_mapping", record)
    modifier._optimize_available_mappings()
    modifier._optimize_available_mappings()

    assert optimized_names == [
        mapping.mapping_name for mapping in modifier._resolved_mappings
    ]
    assert modifier._optimized_mapping_names == set(optimized_names)
    assert not modifier._calibration.inputs
    modifier.remove_hooks()


def test_disabled_activation_subsampling_never_calls_sampler(monkeypatch):
    from llmcompressor_osfp4.modifiers import base

    def fail(*_args, **_kwargs):
        raise AssertionError("disabled activation subsampling called the sampler")

    monkeypatch.setattr(base, "subsample_activations", fail)

    _model, modifier = run_modifier(
        steps=1,
        activation_subsample_size=None,
    )

    assert modifier.activation_subsampling_records == {}


def _assert_state_dict_bit_exact(reference, candidate):
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    assert tuple(reference_state) == tuple(candidate_state)
    for name, expected in reference_state.items():
        actual = candidate_state[name]
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected.dtype, name
        expected_bytes = (
            expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        )
        actual_bytes = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(actual_bytes, expected_bytes), name


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_default_none_and_full_cap_are_bit_exact_when_cap_reaches_k1(mode):
    default_model, default_modifier = run_modifier(
        steps=2,
        optimization_mode=mode,
    )
    none_model, none_modifier = run_modifier(
        steps=2,
        optimization_mode=mode,
        activation_subsample_size=None,
    )
    full_cap_model, full_cap_modifier = run_modifier(
        steps=2,
        optimization_mode=mode,
        activation_subsample_size=6,
    )

    _assert_state_dict_bit_exact(default_model, none_model)
    _assert_state_dict_bit_exact(default_model, full_cap_model)
    state_names = tuple(default_model.state_dict())
    for required_suffix in (
        "weight",
        "weight_scale",
        "weight_zero_point",
        "weight_global_scale",
        "smooth_quant_scale",
        "input_global_scale",
    ):
        assert any(name.endswith(required_suffix) for name in state_names)
    assert default_modifier.activation_subsampling_records
    assert all(
        record["policy"] == "fixed" and record["k"] == record["k1"] == 6
        for record in default_modifier.activation_subsampling_records.values()
    )
    assert none_modifier.activation_subsampling_records == {}
    assert full_cap_modifier.activation_subsampling_records
    assert all(
        record["k"] == record["k1"] == 6
        for record in full_cap_modifier.activation_subsampling_records.values()
    )


def test_optimize_mapping_installs_returned_offloaded_parameters(monkeypatch):
    from llmcompressor_osfp4.modifiers import base

    layer = torch.nn.Linear(16, 3)
    model = torch.nn.Module()
    model.layer = layer
    modifier = _OSFP4Modifier(
        scheme="NVFP4",
        targets=["layer"],
        steps=0,
        activation_subsample_size=None,
    )
    modifier.initialize_quantization(model)
    layer.register_parameter(
        "smooth_quant_scale",
        torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16), requires_grad=False),
    )
    offload_module(layer, onload_device="cpu", offload_device="cpu")
    mapping = OSFP4Mapping(
        mapping_name="layer",
        smooth_layer=None,
        balance_layers=(layer,),
    )
    q_param_dict = {
        "weight": torch.zeros_like(layer.weight),
        "weight_scale": torch.full_like(layer.weight_scale, 0.5),
        "weight_zero_point": torch.zeros_like(layer.weight_zero_point),
        "weight_global_scale": torch.ones_like(layer.weight_global_scale),
    }
    installed = []
    original_update = base.update_offload_parameter

    def record_update(module, name, value):
        installed.append((module, name))
        return original_update(module, name, value)

    monkeypatch.setattr(
        base,
        "quantize_mapping",
        lambda *_args, **_kwargs: [(layer, q_param_dict)],
    )
    monkeypatch.setattr(base, "update_offload_parameter", record_update)

    modifier._optimize_mapping(mapping)

    assert isinstance(layer._parameters, OffloadCache)
    assert installed == [(layer, name) for name in q_param_dict]
    for name, value in q_param_dict.items():
        assert torch.equal(getattr(layer, name), value)
    assert layer in modifier._runtime_smoothing_hooks
    assert mapping.mapping_name in modifier._optimized_mapping_names
    modifier._resolved_mappings = [mapping]
    modifier._remove_runtime_smoothing_hooks()


def test_qparam_install_failure_preserves_mapping_lifecycle_state(monkeypatch):
    from llmcompressor_osfp4.modifiers import base

    layer = torch.nn.Linear(16, 2)
    layer.register_parameter(
        "smooth_quant_scale",
        torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16), requires_grad=False),
    )
    mapping = OSFP4Mapping(
        mapping_name="mapping",
        smooth_layer=None,
        balance_layers=(layer,),
    )
    modifier = make_osfp4_modifier(steps=0)
    modifier._calibration.inputs[mapping.mapping_name] = [torch.ones(1, 16)]
    monkeypatch.setattr(
        base,
        "quantize_mapping",
        lambda *_args, **_kwargs: [(layer, {"weight": torch.zeros_like(layer.weight)})],
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("installation failed")

    monkeypatch.setattr(base, "update_offload_parameter", fail)

    with pytest.raises(RuntimeError, match="Reload a fresh model") as raised:
        modifier._optimize_mapping(mapping)

    assert str(raised.value.__cause__) == "installation failed"
    assert mapping.mapping_name in modifier._calibration.inputs
    assert mapping.mapping_name not in modifier._optimized_mapping_names
    assert layer not in modifier._runtime_smoothing_hooks


@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
def test_missing_calibration_fails_with_sequential_pipeline_guidance(scheme):
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(scheme=scheme, steps=0)
    state = State(model=model)
    event = Event()
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, event)
    with pytest.raises(RuntimeError, match="pipeline='sequential'"):
        modifier.on_calibration_end(state, event)


def test_unobserved_runtime_smooth_quant_scale_mapping_fails_complete_calibration():
    model = LlamaForCausalLM()
    model.model.layers[0].conditional = torch.nn.Linear(16, 16)
    modifier = make_osfp4_modifier(steps=0)
    state = State(model=model)
    event = Event()
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, event)
    model(torch.randn(2, 3, 16))

    with pytest.raises(RuntimeError, match="model.layers.0.conditional"):
        modifier.on_calibration_end(state, event)


@pytest.mark.parametrize("cap", [None, 2, 6])
def test_sampling_wait_precedes_cpu_gather_and_preserves_none_sentinel(
    monkeypatch, cap
):
    from llmcompressor_osfp4.modifiers import base
    from llmcompressor_osfp4.modifiers.calibration_cache import OSFP4CalibrationCache

    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(steps=0, activation_subsample_size=cap)
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))
    mapping = modifier._resolved_mappings[0]
    calls = []
    original_sample = base.subsample_activations
    original_clear = OSFP4CalibrationCache.clear_mapping

    def wait(cache, name, device):
        assert name == mapping.mapping_name
        assert device == torch.device("cpu")
        calls.append("wait")

    def sample(*args, **kwargs):
        calls.append("sample")
        return original_sample(*args, **kwargs)

    def quantize(current, cache, *, optimization_input_batches, **kwargs):
        calls.append("quantize")
        assert current is mapping
        assert cache is modifier._calibration
        assert cache.inputs[mapping.mapping_name][0].shape == (6, 16)
        if cap == 2:
            assert optimization_input_batches[0].shape == (2, 16)
        else:
            assert optimization_input_batches is None
        return []

    def clear(cache, name):
        assert name in modifier._optimized_mapping_names
        calls.append("clear")
        original_clear(cache, name)

    monkeypatch.setattr(OSFP4CalibrationCache, "wait", wait)
    monkeypatch.setattr(OSFP4CalibrationCache, "clear_mapping", clear)
    monkeypatch.setattr(base, "subsample_activations", sample)
    monkeypatch.setattr(base, "quantize_mapping", quantize)
    modifier._optimize_mapping(mapping)
    assert calls == ([] if cap is None else ["wait", "sample"]) + ["quantize", "clear"]
    if cap is None:
        assert modifier.activation_subsampling_records == {}
    else:
        records = modifier.activation_subsampling_records
        assert records[mapping.mapping_name]["k"] == min(cap, 6)
        records[mapping.mapping_name]["k"] = -1
        assert modifier.activation_subsampling_records[mapping.mapping_name][
            "k"
        ] == min(cap, 6)
    modifier.remove_hooks()


def test_runtime_smoothing_survives_disabled_calibration_hooks():
    from llmcompressor.modifiers.utils.hooks import HooksMixin

    layer = torch.nn.Linear(16, 2, bias=False)
    scale = torch.linspace(0.1, 1.9, 16).to(torch.bfloat16)
    layer.register_parameter("smooth_quant_scale", torch.nn.Parameter(scale))
    mapping = OSFP4Mapping("layer", None, (layer,))
    modifier = make_osfp4_modifier()
    modifier._register_runtime_smoothing_hook(mapping)
    values = torch.linspace(-2, 2, 32).reshape(2, 16)
    expected = torch.nn.functional.linear(values * scale.float(), layer.weight)
    with HooksMixin.disable_hooks():
        actual = layer(values)
    assert torch.equal(
        actual.detach().view(torch.uint8), expected.detach().view(torch.uint8)
    )
    modifier._remove_runtime_smoothing_hooks()
    assert not layer._forward_pre_hooks
