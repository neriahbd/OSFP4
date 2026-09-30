import pytest
import torch
from compressed_tensors.offload import OffloadCache, offload_module

from llmcompressor_osfp4.modifiers import smoothing
from llmcompressor_osfp4.modifiers.base import OSFP4Mapping
from llmcompressor_osfp4.modifiers.smoothing import (
    deploy_mapping_smooth_quant_scale,
)


def _tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _make_mapping(dtype, bias, runtime, smooth_layer_dtype=torch.float32):
    torch.manual_seed(7)
    balance_layers = tuple(
        torch.nn.Linear(16, width, bias=bias, dtype=dtype)
        for width in ((9,) if runtime else (9, 4))
    )
    # A wider smooth layer exercises prefix slicing; mixed dtypes expose rounding.
    smooth_layer = (
        None if runtime else torch.nn.LayerNorm(24, bias=bias, dtype=smooth_layer_dtype)
    )
    modules = balance_layers if runtime else (*balance_layers, smooth_layer)
    with torch.no_grad():
        for module in modules:
            for parameter in module.parameters():
                parameter.copy_(torch.randn_like(parameter))
                parameter.view(-1)[:2] = torch.tensor(
                    [0.0, -0.0], dtype=parameter.dtype
                )
    return OSFP4Mapping("mapping", smooth_layer, balance_layers)


@pytest.mark.parametrize(
    "dtype,smooth_layer_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("offloaded", [False, True])
def test_mapping_deployment_bytes_and_installation_order(
    monkeypatch, dtype, smooth_layer_dtype, bias, runtime, offloaded
):
    mapping = _make_mapping(dtype, bias, runtime, smooth_layer_dtype)
    balance_layers = mapping.balance_layers
    smooth_layer = mapping.smooth_layer
    modules = balance_layers if runtime else (*balance_layers, smooth_layer)
    if offloaded:
        for module in modules:
            offload_module(module, onload_device="cpu", offload_device="cpu")

    # Noncontiguous, nonflat FP64 input also guards the explicit FP32 -> BF16 path.
    scale = torch.linspace(0.51013, 1.49017, 16, dtype=torch.float64).view(4, 4).T
    original_scale = _tensor_bytes(scale)
    expected_scale = scale.reshape(-1)
    if runtime:
        expected_scale = expected_scale.float().bfloat16().float()
    else:
        expected_scale = expected_scale.to(dtype).float()

    expected = {
        (module, name): value.detach().clone()
        for module in modules
        for name, value in module.named_parameters()
    }
    for layer in balance_layers:
        expected[layer, "weight"] /= expected_scale.to(dtype).view(1, -1)
    if runtime:
        expected[balance_layers[0], "smooth_quant_scale"] = expected_scale.bfloat16()
    else:
        expected[smooth_layer, "weight"][:16] *= expected_scale.to(smooth_layer.weight)
        if bias:
            expected[smooth_layer, "bias"][:16] *= expected_scale.to(
                smooth_layer.weight
            )

    installed = []
    original_update = smoothing.update_offload_parameter

    def record_update(module, name, value):
        assert value.data_ptr() != getattr(module, name).data_ptr()
        assert not value.requires_grad
        installed.append((module, name))
        original_update(module, name, value)

    monkeypatch.setattr(smoothing, "update_offload_parameter", record_update)
    if runtime:
        original_register = balance_layers[0].register_parameter

        def record_register(name, value):
            assert isinstance(value, torch.nn.Parameter)
            assert not value.requires_grad
            installed.append((balance_layers[0], name))
            original_register(name, value)

        monkeypatch.setattr(balance_layers[0], "register_parameter", record_register)

    deployed_scale = deploy_mapping_smooth_quant_scale(mapping, scale)

    assert deployed_scale.dtype == torch.float32
    assert _tensor_bytes(deployed_scale) == _tensor_bytes(expected_scale)
    assert _tensor_bytes(scale) == original_scale
    expected_order = [(layer, "weight") for layer in balance_layers]
    if runtime:
        expected_order.append((balance_layers[0], "smooth_quant_scale"))
    else:
        expected_order.append((smooth_layer, "weight"))
        if bias:
            expected_order.append((smooth_layer, "bias"))
    assert installed == expected_order
    for (module, name), value in expected.items():
        actual = getattr(module, name)
        assert actual.dtype == value.dtype
        assert _tensor_bytes(actual) == _tensor_bytes(value)
        if offloaded:
            assert isinstance(module._parameters, OffloadCache)
            assert _tensor_bytes(module._parameters.offloaded_values[name]) == (
                _tensor_bytes(value)
            )


def test_deploy_smooth_layer_scale_preserves_outputs():
    torch.manual_seed(0)
    smooth_layer = torch.nn.LayerNorm(16)
    linear_1 = torch.nn.Linear(16, 8)
    linear_2 = torch.nn.Linear(16, 4)
    mapping = OSFP4Mapping(
        mapping_name="norm",
        smooth_layer=smooth_layer,
        balance_layers=(linear_1, linear_2),
    )
    x = torch.randn(2, 3, 16)

    with torch.no_grad():
        before_1 = linear_1(smooth_layer(x))
        before_2 = linear_2(smooth_layer(x))

    smooth_quant_scale = torch.linspace(0.5, 1.5, 16)
    deployed_smooth_quant_scale = deploy_mapping_smooth_quant_scale(
        mapping, smooth_quant_scale
    )

    with torch.no_grad():
        after_1 = linear_1(smooth_layer(x))
        after_2 = linear_2(smooth_layer(x))

    assert torch.allclose(after_1, before_1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(after_2, before_2, atol=1e-5, rtol=1e-5)
    assert torch.equal(deployed_smooth_quant_scale, smooth_quant_scale)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_deploy_runtime_smooth_quant_scale_is_canonical_and_preserves_output(
    dtype, bias
):
    torch.manual_seed(1)
    layer = torch.nn.Linear(16, 9, bias=bias, dtype=dtype)
    mapping = OSFP4Mapping("projection", None, (layer,))
    original_weight = layer.weight.detach().clone()
    x = torch.randn(2, 3, 16, dtype=dtype)
    smooth_quant_scale = torch.linspace(0.51, 1.49, 16, dtype=torch.float32)

    with torch.no_grad():
        before = layer(x)
        deployed_smooth_quant_scale = deploy_mapping_smooth_quant_scale(
            mapping, smooth_quant_scale
        )
        after = layer(x * deployed_smooth_quant_scale.to(dtype=dtype))

    expected_smooth_quant_scale = smooth_quant_scale.to(torch.bfloat16)
    expected_weight = original_weight / expected_smooth_quant_scale.to(dtype).view(
        1, -1
    )

    assert deployed_smooth_quant_scale.dtype == torch.float32
    assert layer.smooth_quant_scale.dtype == torch.bfloat16
    assert not layer.smooth_quant_scale.requires_grad
    assert torch.equal(deployed_smooth_quant_scale, expected_smooth_quant_scale.float())
    assert torch.equal(layer.weight, expected_weight)
    tolerance = 2e-2 if dtype == torch.float16 else 4e-2
    assert torch.allclose(after.float(), before.float(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("offloaded", [False, True])
def test_deploy_runtime_smooth_quant_scale_rejects_second_application(
    monkeypatch, offloaded
):
    layer = torch.nn.Linear(16, 8)
    if offloaded:
        offload_module(layer, onload_device="cpu", offload_device="cpu")
    mapping = OSFP4Mapping("projection", None, (layer,))
    scale = torch.linspace(0.51, 1.49, 16)
    deploy_mapping_smooth_quant_scale(mapping, scale)
    before = {name: _tensor_bytes(value) for name, value in layer.named_parameters()}

    def unexpected_update(*args):
        pytest.fail("Duplicate deployment must fail before parameter installation")

    monkeypatch.setattr(smoothing, "update_offload_parameter", unexpected_update)
    monkeypatch.setattr(layer, "register_parameter", unexpected_update)

    with pytest.raises(RuntimeError, match="already registered"):
        deploy_mapping_smooth_quant_scale(mapping, scale * 2)

    assert before == {
        name: _tensor_bytes(value) for name, value in layer.named_parameters()
    }


def test_deploy_mapping_smooth_quant_scale_installs_runtime_scale_without_hook():
    torch.manual_seed(2)
    layer = torch.nn.Linear(16, 8)
    mapping = OSFP4Mapping(
        mapping_name="projection",
        smooth_layer=None,
        balance_layers=(layer,),
    )
    inputs = torch.randn(2, 3, 16)
    smooth_quant_scale = torch.linspace(0.51, 1.49, 16)

    with torch.no_grad():
        expected = layer(inputs)
        deployed_smooth_quant_scale = deploy_mapping_smooth_quant_scale(
            mapping, smooth_quant_scale
        )
        actual = layer(inputs * deployed_smooth_quant_scale)

    assert torch.equal(deployed_smooth_quant_scale, layer.smooth_quant_scale.float())
    assert not hasattr(layer, "_osfp4_runtime_scaling_hook")
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert "smooth_quant_scale" in layer._parameters


def test_deploy_mapping_smooth_quant_scale_dispatches_smooth_layer_mapping():
    smooth_layer = torch.nn.LayerNorm(16)
    layer = torch.nn.Linear(16, 8)
    mapping = OSFP4Mapping(
        mapping_name="norm",
        smooth_layer=smooth_layer,
        balance_layers=(layer,),
    )
    smooth_quant_scale = torch.linspace(0.5, 1.5, 16)

    deployed_smooth_quant_scale = deploy_mapping_smooth_quant_scale(
        mapping, smooth_quant_scale
    )

    assert torch.equal(deployed_smooth_quant_scale, smooth_quant_scale)
    assert "smooth_quant_scale" not in layer._parameters
