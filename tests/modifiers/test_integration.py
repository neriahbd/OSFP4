import pytest
import torch
from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from compressed_tensors.quantization.lifecycle.forward import (
    fake_quantize,
    quantize,
)

from llmcompressor.core import Event, State

from ._testing import LlamaForCausalLM, make_osfp4_modifier, run_modifier


def test_rtn_keeps_balanced_source_weights():
    torch.manual_seed(0)
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(steps=0, optimization_mode="rtn")
    state = State(model=model)
    event = Event()
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, event)
    model(torch.randn(2, 3, 16))
    modifier.on_sequential_epoch_end(state, event, list(model.modules()))

    assert modifier._optimized_mapping_names
    any_non_qdq_weight = False
    for layer in model.modules():
        if not hasattr(layer, "weight_scale"):
            continue
        assert not layer.quantization_enabled
        fake_quantized = fake_quantize(
            layer.weight,
            layer.weight_scale,
            layer.weight_zero_point,
            layer.quantization_scheme.weights,
            global_scale=layer.weight_global_scale,
        )
        any_non_qdq_weight |= not torch.equal(layer.weight, fake_quantized)

    assert any_non_qdq_weight
    modifier.remove_hooks()


@pytest.mark.parametrize(
    "weight_dtype",
    [torch.float32, torch.bfloat16, torch.float16],
)
def test_sic_quantized_weights_are_already_canonical(weight_dtype):
    torch.manual_seed(0)
    model = LlamaForCausalLM().to(dtype=weight_dtype)
    modifier = make_osfp4_modifier(steps=1, optimization_mode="sic")
    state = State(model=model)
    event = Event()
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, event)
    model(torch.randn(2, 3, 16, dtype=weight_dtype))
    modifier.on_sequential_epoch_end(state, event, list(model.modules()))

    target_count = 0
    for layer in model.modules():
        if not hasattr(layer, "weight_scale"):
            continue
        target_count += 1
        canonical_weight = fake_quantize(
            layer.weight,
            layer.weight_scale,
            layer.weight_zero_point,
            layer.quantization_scheme.weights,
            global_scale=layer.weight_global_scale,
        )
        assert torch.equal(layer.weight, canonical_weight)

    assert target_count
    modifier.remove_hooks()


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_all_modes_finalize(mode):
    model, modifier = run_modifier(optimization_mode=mode)
    assert len(modifier._optimized_mapping_names) == 4
    block = model.model.layers[0]
    layers = (
        block.self_attn.q_proj,
        block.self_attn.k_proj,
        block.self_attn.v_proj,
        block.self_attn.o_proj,
        block.mlp.gate_proj,
        block.mlp.up_proj,
        block.mlp.down_proj,
    )
    for layer in layers:
        assert torch.equal(layer.weight_global_scale, torch.ones(1))
        assert not hasattr(layer, "osfp4_checkpoint_scales")
    assert block.self_attn.o_proj.smooth_quant_scale.dtype == torch.bfloat16
    assert block.mlp.down_proj.smooth_quant_scale.dtype == torch.bfloat16
    assert not hasattr(model.lm_head, "quantization_scheme")
    assert not hasattr(modifier, "_energy_cache")


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_nvfp4a16_all_modes_optimize_and_finalize(mode):
    model, modifier = run_modifier(
        scheme="NVFP4A16",
        optimization_mode=mode,
        steps=1,
    )
    for layer in model.modules():
        if not hasattr(layer, "weight_scale"):
            continue
        assert torch.isfinite(layer.weight_scale).all()
        assert torch.all(layer.weight_scale > 0)
        assert torch.equal(layer.weight_global_scale, torch.ones(1))
        assert not hasattr(layer, "input_global_scale")
        assert layer.quantization_scheme.input_activations is None

    modifier.on_finalize(State(model=model))
    metadata = model.config.osfp4_metadata
    assert metadata["version"] == 1
    assert "activation_quantization" not in metadata
    assert "optimization_mode" not in metadata


def test_sic_mode_is_used_end_to_end(monkeypatch):
    from llmcompressor_osfp4.modifiers import osfp4_quantize

    percdamps = []
    weights = []
    original = osfp4_quantize.optimize_sic

    def record(*args, **kwargs):
        assert "mode" not in kwargs
        percdamps.append(kwargs["percdamp"])
        assert "real_global_scale" not in kwargs
        weights.append(args[0].detach().clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(osfp4_quantize, "optimize_sic", record)
    model, modifier = run_modifier(
        optimization_mode="sic",
        dampening_frac=0.025,
    )

    assert percdamps == [0.025] * 4
    assert [tuple(W.shape) for W in weights] == [
        (48, 16),
        (64, 16),
        (16, 16),
        (16, 32),
    ]
    assert modifier.optimization_mode == "sic"
    modifier.on_finalize(State(model=model))
    assert model.config.osfp4_metadata["version"] == 1


def test_sic_installs_scale_times_fp4_code_before_qparams():
    model, _ = run_modifier(optimization_mode="sic")
    for layer in model.model.layers[0].modules():
        if not hasattr(layer, "weight_scale"):
            continue
        rows, columns = layer.weight.shape
        weight_quantization_groups = (
            layer.weight.detach()
            .float()
            .view(rows, columns // 16, 16)
            .permute(1, 0, 2)
            .contiguous()
        )
        scales = layer.weight_scale.T.unsqueeze(-1).float()
        codes = weight_quantization_groups / scales
        expected_codes = torch.tensor(
            [
                -6.0,
                -4.0,
                -3.0,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
            ]
        )
        assert torch.isin(codes, expected_codes).all()
        assert torch.equal(weight_quantization_groups, codes * scales)


def test_sic_finalizes_with_minimal_runtime_metadata():
    model, modifier = run_modifier(optimization_mode="sic")
    modifier.on_finalize(State(model=model))

    assert set(model.config.osfp4_metadata) == {
        "version",
        "smooth_quant_scale_targets",
    }
    state_names = tuple(model.state_dict())
    assert not any(
        token in name
        for name in state_names
        for token in (
            "gamma_w",
            "gamma_x",
            "T_eff",
            ".U",
            ".Y",
            "weight_optimal_alpha_diag",
        )
    )


def test_global_scales_use_exact_post_alpha_calibration_absmax():
    torch.manual_seed(0)
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(optimization_mode="rtn", steps=0)
    state = State(model=model)
    event = Event()
    modifier.on_initialize(state)
    original_smooth_layer_weights = {
        mapping.mapping_name: mapping.smooth_layer.weight.detach().float().cpu().clone()
        for mapping in modifier._resolved_mappings
        if mapping.smooth_layer is not None
    }
    modifier.on_calibration_start(state, event)

    calibration_batches = [
        torch.randn(2, 3, 16),
        torch.randn(1, 4, 16) * 2,
    ]
    for batch in calibration_batches:
        model(batch)
    cached = {
        name: [value.clone() for value in values]
        for name, values in modifier._calibration.inputs.items()
    }
    modifier.on_calibration_end(state, event)

    expected_by_layer = {}
    for mapping in modifier._resolved_mappings:
        if mapping.requires_runtime_smoothing:
            deployed_smooth_quant_scale = (
                mapping.balance_layers[0].smooth_quant_scale.float().cpu()
            )
        else:
            deployed_smooth_quant_scale = (
                mapping.smooth_layer.weight.detach().float().cpu()
                / original_smooth_layer_weights[mapping.mapping_name]
            )
        absmax = (
            torch.cat(cached[mapping.mapping_name])
            .float()
            .mul(deployed_smooth_quant_scale)
            .abs()
            .max()
        )
        numerator = torch.tensor(2688.0, dtype=torch.float32)
        expected = (
            torch.tensor([1.0]) if absmax == 0 else (numerator / absmax).reshape(1)
        )
        for layer in mapping.balance_layers:
            expected_by_layer[layer] = expected

    modifier.on_finalize(state)
    for layer, expected in expected_by_layer.items():
        assert layer.input_global_scale.dtype == torch.float32
        assert not layer.input_global_scale.requires_grad
        assert torch.equal(layer.input_global_scale.cpu(), expected)

    block = model.model.layers[0]
    assert torch.equal(
        block.self_attn.q_proj.input_global_scale,
        block.self_attn.k_proj.input_global_scale,
    )
    assert torch.equal(
        block.self_attn.q_proj.input_global_scale,
        block.self_attn.v_proj.input_global_scale,
    )


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_integrated_quantization_registers_exact_qparams_and_removes_observers(mode):
    model, _ = run_modifier(optimization_mode=mode)

    for layer in model.modules():
        if not hasattr(layer, "weight_scale"):
            continue
        assert layer.quantization_status.name == "FROZEN"
        assert layer.quantization_enabled
        assert layer.quantization_scheme.weights.observer == "osfp4"
        assert layer.quantization_scheme.weights.observer_kwargs["mode"] == mode
        assert not hasattr(layer, "osfp4_checkpoint_scales")
        assert torch.isfinite(layer.weight_scale).all()
        assert torch.all(layer.weight_scale > 0)
        assert torch.equal(layer.weight_global_scale, torch.ones(1))
        assert torch.equal(
            layer.weight_zero_point,
            torch.zeros_like(layer.weight_zero_point),
        )
        assert tuple(layer.input_global_scale.shape) == (1,)
        assert not hasattr(layer, "input_scale")
        assert not hasattr(layer, "input_zero_point")
        for observer_name in (
            "input_observer",
            "weight_observer",
            "output_observer",
        ):
            assert not hasattr(layer, observer_name)

        if mode == "sic":
            fake_quantized = fake_quantize(
                layer.weight,
                layer.weight_scale,
                layer.weight_zero_point,
                layer.quantization_scheme.weights,
                global_scale=layer.weight_global_scale,
            )
            assert torch.equal(fake_quantized, layer.weight)

        codes = quantize(
            layer.weight,
            layer.weight_scale,
            layer.weight_zero_point,
            layer.quantization_scheme.weights,
            global_scale=layer.weight_global_scale,
        )
        unpacked_codes = unpack_fp4_from_uint8(
            pack_fp4_to_uint8(codes),
            *codes.shape,
            dtype=codes.dtype,
        )
        assert torch.equal(unpacked_codes, codes)

    assert not hasattr(model.lm_head, "quantization_scheme")


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_integrated_quantization_state_dict_round_trip_is_exact(mode, tmp_path):
    model, modifier = run_modifier(optimization_mode=mode)
    modifier.on_finalize(State(model=model))
    expected = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    checkpoint_path = tmp_path / f"osfp4-{mode}.pt"
    torch.save(expected, checkpoint_path)
    restored = torch.load(checkpoint_path, weights_only=True)

    assert tuple(restored) == tuple(expected)
    for name, value in restored.items():
        assert value.dtype == expected[name].dtype
        assert value.shape == expected[name].shape
        assert torch.equal(value, expected[name])
