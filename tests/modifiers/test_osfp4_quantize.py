# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

import inspect
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload import OffloadCache, offload_module

from llmcompressor.core import Event, State
from llmcompressor_osfp4.modifiers import OSFP4Modifier, base, smoothing
from llmcompressor_osfp4.modifiers.base import OSFP4Mapping
from llmcompressor_osfp4.modifiers.calibration_cache import OSFP4CalibrationCache
from llmcompressor_osfp4.modifiers.optimization import (
    RTNQuantizationResult,
    SICQuantizationResult,
)
from llmcompressor_osfp4.modifiers.osfp4_quantize import quantize_mapping

from ._quantize_phases import run_phases
from ._testing import LlamaForCausalLM, run_modifier


def test_quantize_mapping_signature_has_no_optimizer_or_batch_controls():
    parameters = inspect.signature(quantize_mapping).parameters

    assert "steps" not in parameters
    assert "lr" not in parameters
    assert "quantization_groups_per_batch" not in parameters
    assert "diagnostics_recorder" not in parameters
    assert parameters["weight_only"].default is inspect.Parameter.empty


def _prepared_layer(optimization_mode, dtype=torch.float32, weight_only=False):
    layer = torch.nn.Linear(16, 2, dtype=dtype)
    model = torch.nn.Module()
    model.layer = layer
    modifier = OSFP4Modifier(
        scheme="NVFP4A16" if weight_only else "NVFP4",
        targets=["layer"],
        optimization_mode=optimization_mode,
        steps=0,
        lr=0.01,
    )
    modifier.initialize_quantization(model)
    modifier.start_calibration(model)
    return model, layer


@pytest.mark.parametrize("optimization_mode", ["rtn", "sic"])
@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("activation_policy", ["full", "sampled", "weight_only"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offloaded", [False, True])
def test_mapping_phases_reuse_attached_observer(
    monkeypatch, optimization_mode, runtime, activation_policy, dtype, offloaded
):
    from llmcompressor_osfp4.modifiers import osfp4_quantize

    weight_only = activation_policy == "weight_only"
    model, layer = _prepared_layer(optimization_mode, dtype, weight_only)
    if offloaded:
        offload_module(layer, onload_device="cpu", offload_device="cpu")
    original_weight = layer.weight.detach().float().clone()
    attached_observer = layer.weight_observer
    mapping = OSFP4Mapping(
        mapping_name="mapping",
        smooth_layer=None if runtime else torch.nn.LayerNorm(16),
        balance_layers=(layer,),
    )
    result = SICQuantizationResult(
        alpha=torch.ones(16),
        gamma_w_final=torch.ones(2, 1),
        weight_zero_point=torch.zeros(2, 1),
        quantized_weight=torch.zeros(2, 16),
    )
    if optimization_mode == "rtn":
        result = RTNQuantizationResult(*result[:3])
    seen = {}
    events = []

    def optimize(*args, **kwargs):
        events.append("optimize")
        seen.update(args=args, kwargs=kwargs)
        return result

    monkeypatch.setattr(
        osfp4_quantize,
        "optimize_sic" if optimization_mode == "sic" else "optimize_rtn",
        optimize,
    )
    cache = OSFP4CalibrationCache(
        cached_activations={} if weight_only else {"mapping": [torch.ones(4, 16)]},
        hessian={"mapping": torch.eye(16)},
        sigma_x_squared_sum={"mapping": torch.arange(1, 17, dtype=torch.float32)},
        token_count={"mapping": 4},
    )
    original_cat = torch.cat

    def cat(*args, **kwargs):
        events.append("cat")
        return original_cat(*args, **kwargs)

    def wait_for_capture(name, device):
        assert name == mapping.mapping_name
        assert device == layer.weight.device
        events.append("wait")

    monkeypatch.setattr(torch, "cat", cat)
    monkeypatch.setattr(cache, "wait_for_capture", wait_for_capture)
    sampled = (torch.full((2, 16), 2.0),) if activation_policy == "sampled" else None

    deployment = run_phases(
        mapping,
        cache,
        optimization_mode=optimization_mode,
        dampening_frac=0.01,
        weight_only=weight_only,
        sampled_activations=sampled,
    )

    assert events[:3] == ["cat", "wait", "optimize"]
    assert _tensor_bytes(seen["args"][0]) == _tensor_bytes(original_weight)
    expected_batches = (
        sampled if sampled is not None else cache.cached_activations.get("mapping")
    )
    assert seen["args"][1] is expected_batches
    assert seen["kwargs"]["observer"] is attached_observer
    assert seen["kwargs"]["alpha_dtype"] is (torch.bfloat16 if runtime else dtype)
    assert "quantization_groups_per_batch" not in seen["kwargs"]
    if optimization_mode == "sic":
        assert seen["kwargs"]["weight_dtype"] is layer.weight.dtype
        assert torch.equal(seen["args"][2], torch.eye(16) / 4)
    else:
        assert torch.equal(
            seen["kwargs"]["sigma_x_squared"],
            torch.arange(1, 17, dtype=torch.float32) / 4,
        )
    deployed_layer, qparams = deployment[0]
    assert deployed_layer is layer
    assert list(qparams) == (
        (["weight"] if optimization_mode == "sic" else [])
        + ["weight_scale", "weight_zero_point", "weight_global_scale"]
    )
    if offloaded:
        assert isinstance(layer._parameters, OffloadCache)
    assert torch.equal(
        qparams["weight_scale"], result.gamma_w_final.to(layer.weight_scale)
    )
    assert torch.equal(
        qparams["weight_zero_point"],
        result.weight_zero_point.to(layer.weight_zero_point),
    )
    assert torch.equal(
        qparams["weight_global_scale"], torch.ones_like(layer.weight_global_scale)
    )
    if optimization_mode == "sic":
        assert torch.equal(qparams["weight"], result.quantized_weight)
    else:
        assert "weight" not in qparams


def _tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize("optimization_mode", ["rtn", "sic"])
def test_optimization_only_uses_sampled_activations(monkeypatch, optimization_mode):
    from llmcompressor_osfp4.observers import observer as osfp4_observer

    seen_shapes = []
    original = osfp4_observer._optimize_joint_scales

    def record(weight, activations, **kwargs):
        seen_shapes.append((tuple(weight.shape), tuple(activations.shape)))
        return original(weight, activations, **kwargs)

    monkeypatch.setattr(osfp4_observer, "_optimize_joint_scales", record)
    model, modifier = run_modifier(
        steps=1,
        optimization_mode=optimization_mode,
        activation_subsample_size=3,
    )

    assert seen_shapes == [
        ((1, 48, 16), (1, 16, 3)),
        ((1, 64, 16), (1, 16, 3)),
        ((1, 16, 16), (1, 16, 3)),
        ((2, 16, 16), (2, 16, 3)),
    ]
    assert all(
        record["k"] == 3 and record["k1"] == 6
        for record in modifier.activation_subsampling_records.values()
    )
    assert model.state_dict()


@pytest.mark.parametrize("optimization_mode", ["rtn", "sic"])
def test_mapping_phases_return_fused_row_slices(monkeypatch, optimization_mode):
    from llmcompressor_osfp4.modifiers import osfp4_quantize

    layers = (torch.nn.Linear(16, 3), torch.nn.Linear(16, 2))
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(layers)
    modifier = OSFP4Modifier(
        scheme="NVFP4",
        targets=["Linear"],
        optimization_mode=optimization_mode,
        steps=0,
    )
    modifier.initialize_quantization(model)
    modifier.start_calibration(model)
    mapping = OSFP4Mapping("mapping", torch.nn.LayerNorm(16), layers)
    weight_scale = torch.arange(5, dtype=torch.float32).reshape(5, 1) + 0.5
    weight_zero_point = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    quantized_weight = torch.arange(80, dtype=torch.float32).reshape(5, 16)
    result = SICQuantizationResult(
        alpha=torch.ones(16),
        gamma_w_final=weight_scale,
        weight_zero_point=weight_zero_point,
        quantized_weight=quantized_weight,
    )
    if optimization_mode == "rtn":
        result = RTNQuantizationResult(*result[:3])
    monkeypatch.setattr(
        osfp4_quantize,
        "optimize_sic" if optimization_mode == "sic" else "optimize_rtn",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        base,
        "deploy_mapping_smooth_quant_scale",
        lambda _mapping, scale: scale,
    )
    monkeypatch.setattr(
        base,
        "update_smoothed_input_global_scale",
        lambda *_args: None,
    )
    cache = OSFP4CalibrationCache(
        cached_activations={"mapping": [torch.ones(1, 16)]},
        hessian={"mapping": torch.eye(16)},
        sigma_x_squared_sum={"mapping": torch.ones(16)},
        token_count={"mapping": 1},
    )

    deployment = run_phases(
        mapping,
        cache,
        optimization_mode=optimization_mode,
        dampening_frac=0.01,
        weight_only=False,
    )

    for (layer, qparams), row_slice in zip(
        deployment,
        (slice(0, 3), slice(3, 5)),
    ):
        assert list(qparams) == (
            (["weight"] if optimization_mode == "sic" else [])
            + ["weight_scale", "weight_zero_point", "weight_global_scale"]
        )
        assert torch.equal(
            qparams["weight_scale"],
            weight_scale[row_slice].to(layer.weight_scale),
        )
        assert torch.equal(
            qparams["weight_zero_point"],
            weight_zero_point[row_slice].to(layer.weight_zero_point),
        )
        if optimization_mode == "sic":
            assert torch.equal(qparams["weight"], quantized_weight[row_slice])
        else:
            assert "weight" not in qparams


class RecordingInputObserver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.min_vals = torch.tensor([-1000.0])
        self.max_vals = torch.tensor([1000.0])
        self.observed = []

    def forward(self, observed):
        assert not hasattr(self, "min_vals")
        assert not hasattr(self, "max_vals")
        self.observed.append(observed.detach().clone())


@pytest.mark.parametrize(
    "missing_statistics", [(), ("min_vals",), ("max_vals",), ("min_vals", "max_vals")]
)
def test_smoothed_input_replay_updates_every_target(monkeypatch, missing_statistics):
    layers = (torch.nn.Linear(4, 3), torch.nn.Linear(4, 2))
    observers = (RecordingInputObserver(), RecordingInputObserver())
    for layer, observer in zip(layers, observers):
        layer.input_observer = observer
        for statistic in missing_statistics:
            delattr(observer, statistic)
    batches = (
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
    )
    scale = torch.tensor([0.5, 1.0, 1.5, 2.0])
    updates = []
    monkeypatch.setattr(
        smoothing,
        "update_qparams",
        lambda modules, base_name: updates.append((modules, base_name)),
    )

    order = []
    handles = [
        observer.register_forward_pre_hook(
            lambda module, args, index=index: order.append(index)
        )
        for index, observer in enumerate(observers)
    ]
    mapping = OSFP4Mapping("mapping", None, layers)
    cache = OSFP4CalibrationCache(cached_activations={"mapping": list(batches)})
    smoothing.update_smoothed_input_global_scale(mapping, cache, scale)
    for handle in handles:
        handle.remove()

    assert order == [0, 1, 0, 1]
    for observer in observers:
        assert len(observer.observed) == len(batches)
        for observed, batch in zip(observer.observed, batches):
            assert torch.equal(observed, batch * scale)
    assert updates == [(layers, "input")]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", ["mixed", "zeros", "small"])
def test_extrema_recalibration_matches_full_and_two_row_global_scale(dtype, case):
    _, layer = _prepared_layer("rtn", dtype)
    mapping = OSFP4Mapping("mapping", None, (layer,))
    rows = torch.arange(64, dtype=torch.float32).reshape(4, 16) - 32
    rows[0].zero_()
    rows[-1].mul_(-4)
    if case == "zeros":
        rows.zero_()
    elif case == "small":
        rows.mul_(1e-6)
    rows = rows.to(dtype)
    scale = torch.linspace(0.125, 8, 16).to(torch.bfloat16).float()
    maximum = rows.float().abs().amax(0)

    cache = OSFP4CalibrationCache(cached_activations={"mapping": list(rows.split(2))})
    smoothing.update_smoothed_input_global_scale(mapping, cache, scale)
    full_scale = layer.input_global_scale.clone()
    cache = OSFP4CalibrationCache(
        cached_activations={"mapping": [torch.stack((-maximum, maximum))]}
    )
    smoothing.update_smoothed_input_global_scale(mapping, cache, scale)
    two_row_scale = layer.input_global_scale.clone()

    # The retained sample misses the extrema; recalibration must use all-token maxima.
    cache = OSFP4CalibrationCache(
        cached_activations={"mapping": [rows[:1]]},
        input_channel_absmax={"mapping": maximum},
    )
    smoothing.update_smoothed_input_global_scale(mapping, cache, scale)
    actual = layer.input_global_scale
    for expected in (full_scale, two_row_scale):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_invalid_optimized_scales_fail_before_weight_smoothing(monkeypatch):
    from llmcompressor_osfp4.observers import observer as osfp4_observer

    model = LlamaForCausalLM()
    modifier = OSFP4Modifier(
        scheme="NVFP4",
        ignore=["lm_head"],
        optimization_mode="rtn",
        steps=0,
    )
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(2, 3, 16))
    mapping = modifier._resolved_mappings[0]
    smooth_layer_weight = mapping.smooth_layer.weight.detach().clone()
    balance_weights = [
        layer.weight.detach().clone() for layer in mapping.balance_layers
    ]

    def invalid_result(weight, activations, **_kwargs):
        groups, rows, width = weight.shape
        samples = activations.shape[-1]
        return SimpleNamespace(
            alpha=torch.full((groups, width), float("inf")),
            gamma_w=torch.ones(groups, rows, 1),
            gamma_x=torch.ones(groups, 1, samples),
        )

    monkeypatch.setattr(osfp4_observer, "_optimize_joint_scales", invalid_result)
    with pytest.raises(ValueError, match="invalid scales"):
        modifier._optimize_mapping(mapping)

    assert torch.equal(mapping.smooth_layer.weight, smooth_layer_weight)
    for layer, expected in zip(mapping.balance_layers, balance_weights):
        assert torch.equal(layer.weight, expected)
    assert mapping.mapping_name in modifier._calibration_cache.cached_activations


@pytest.mark.parametrize("optimization_mode", ["rtn", "sic"])
def test_weight_only_uses_statistics_without_activation_replay(
    monkeypatch, optimization_mode
):
    model, modifier = run_modifier(
        scheme="NVFP4A16",
        optimization_mode=optimization_mode,
        steps=1,
    )

    assert modifier.activation_subsample_size is None
    assert not modifier._calibration_cache.cached_activations
    for layer in model.modules():
        if hasattr(layer, "weight_scale"):
            assert not hasattr(layer, "input_global_scale")


@pytest.mark.parametrize("smooth_in_features", [8, 6])
def test_linear_smooth_layer_folding_preserves_output(smooth_in_features):
    from llmcompressor_osfp4.modifiers.smoothing import (
        deploy_mapping_smooth_quant_scale,
    )

    torch.manual_seed(0)
    smooth = torch.nn.Linear(smooth_in_features, 8, bias=True)
    balance = torch.nn.Linear(8, 4, bias=False)
    x = torch.randn(3, smooth_in_features)
    expected = balance(smooth(x))
    mapping = OSFP4Mapping(
        mapping_name="smooth",
        smooth_layer=smooth,
        balance_layers=(balance,),
    )

    deploy_mapping_smooth_quant_scale(mapping, torch.rand(8) + 0.5)

    torch.testing.assert_close(balance(smooth(x)), expected)
