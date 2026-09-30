"""Allocation and CUDA producer/consumer contracts for the OSFP4 cache."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from llmcompressor_osfp4.modifiers.calibration_cache import OSFP4CalibrationCache
from llmcompressor_osfp4.modifiers.optimization import sic
from llmcompressor_osfp4.modifiers.optimization._activation_groups import (
    assemble_activation_quantization_groups,
)
from llmcompressor_osfp4.modifiers.optimization.rtn import optimize_rtn
from llmcompressor_osfp4.observers import scale_selection


def assert_bytes(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(
        left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
    )


def test_activation_assembly_releases_previous_conversion(monkeypatch):
    original = torch.Tensor.to
    converted = []

    def convert(tensor, **kwargs):
        assert all(value() is None for value in converted)
        result = original(tensor, **kwargs)
        converted.append(weakref.ref(result))
        return result

    batches = [torch.randn(n, 32).bfloat16() for n in (3, 0, 7)]
    expected = (
        torch.cat(batches).float().reshape(-1, 2, 16).permute(1, 2, 0).contiguous()
    )
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", convert)
        result = assemble_activation_quantization_groups(
            batches, 2, torch.device("cpu")
        )
    assert_bytes(result, expected)
    assert result.is_contiguous()


def test_rtn_releases_activation_groups_before_scale_selection():
    activation_reference = []

    class Observer:
        def optimize_quantization_group_scales(
            self, weight, *, activation_quantization_groups, **kwargs
        ):
            activation_reference.append(weakref.ref(activation_quantization_groups))
            return SimpleNamespace(alpha_star=torch.ones(weight.shape[0], 16))

        def select_weight_qparams(self, weight, optimized, *, weight_metric):
            gc.collect()
            assert activation_reference[0]() is None
            shape = (*weight.shape[:2], 1)
            return torch.ones(shape), torch.zeros(shape)

    optimize_rtn(
        torch.randn(3, 32),
        [torch.randn(4, 32)],
        observer=Observer(),
        alpha_dtype=torch.float32,
        sigma_x_squared=torch.ones(32),
    )


def test_sic_releases_hessian_clone_and_weight_groups_before_scale_selection(
    monkeypatch,
):
    hessian_reference = []
    weight_reference = []
    original_cholesky = torch.linalg.cholesky

    def capture_hessian(value, **kwargs):
        hessian_reference.append(weakref.ref(value))
        return original_cholesky(value, **kwargs)

    class Observer:
        def optimize_quantization_group_scales(
            self, weight, *, activation_quantization_groups, **kwargs
        ):
            gc.collect()
            assert hessian_reference[0]() is None
            weight_reference.append(weakref.ref(weight))
            return SimpleNamespace(
                alpha_star=torch.ones(weight.shape[0], 16),
                gamma_w=torch.ones(weight.shape[0], weight.shape[1], 1),
            )

        def select_weight_qparams(self, weight, optimized, *, weight_metric):
            gc.collect()
            assert weight_reference[0]() is None
            shape = (*weight.shape[:2], 1)
            return torch.ones(shape), torch.zeros(shape)

    monkeypatch.setattr(torch.linalg, "cholesky", capture_hessian)
    sic.optimize_sic(
        torch.randn(3, 32),
        [torch.randn(4, 32)],
        torch.eye(32),
        observer=Observer(),
        alpha_dtype=torch.float32,
        weight_dtype=torch.float32,
    )


def test_scale_search_does_not_allocate_full_repeated_alpha_or_metric(monkeypatch):
    groups, rows, width = 3, 7, 16
    # Two rows per tile forces tiles both within and across group boundaries.
    monkeypatch.setattr(
        scale_selection, "_E4M3_SEARCH_PRIMARY_TENSOR_BYTES", 2 * 19 * width * 4
    )

    class Allocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.clone.default:
                assert tuple(result.shape) != (groups, rows, width)
            if func == torch.ops.aten.index.Tensor and args[0].shape == (groups, width):
                assert result.shape[0] <= 2
            return result

    weight = torch.randn(groups, rows, width)
    alpha = torch.ones(groups, width)
    gamma = torch.ones(groups, rows, 1)
    metric = torch.ones(groups, width)
    with Allocations():
        actual, invalid, fallback = scale_selection._search_e4m3_gamma_star(
            weight, alpha, gamma, metric
        )
    # One group at a time supplies an independently partitioned reference.
    expected = torch.cat(
        [
            scale_selection._search_e4m3_gamma_star(
                weight[g : g + 1], alpha[g : g + 1], gamma[g : g + 1], metric[g : g + 1]
            )[0]
            for g in range(groups)
        ]
    )
    assert not invalid.any()
    assert_bytes(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("mode", ["rtn", "sic"])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("cache_inputs", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_multiple_cuda_captures_match_ordered_reference(
    mode, offload, cache_inputs, dtype, noncontiguous
):
    torch.manual_seed(42)
    layer = torch.nn.Linear(16, 7, device="cuda", dtype=dtype)
    cache = OSFP4CalibrationCache(offload_hessians=offload)
    capture = cache.make_capture_hook(
        "mapping",
        capture_hessian=mode == "sic",
        capture_sigma_x_squared=mode == "rtn",
        cache_inputs=cache_inputs,
    )
    batches = [
        torch.randn(n, 3, 32 if noncontiguous else 16, device="cuda", dtype=dtype)
        for n in (2, 1, 3)
    ]
    if noncontiguous:
        batches = [value[..., ::2] for value in batches]
    expected_hessian = torch.zeros(16, 16, device="cuda")
    expected_energy = torch.zeros(16, device="cuda")
    for batch in batches:
        flat = batch.float().reshape(-1, 16)
        expected_hessian.addmm_(flat.T, flat)
        expected_energy.add_(flat.square().sum(0))
        capture(layer, (batch,))
        assert len(cache.events["mapping"]) == 1
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        cache.wait("mapping", torch.device("cuda"))
        actual = (
            cache.hessian["mapping"].to("cuda")
            if mode == "sic"
            else cache.sigma_x_squared["mapping"]
        ).clone()
    consumer.synchronize()
    assert_bytes(actual, expected_hessian if mode == "sic" else expected_energy)
    cache.wait("mapping", torch.device("cpu"))
    if cache_inputs:
        for cached, batch in zip(cache.inputs["mapping"], batches):
            assert cached.is_pinned()
            assert_bytes(cached, batch.reshape(-1, 16))
    else:
        assert not cache.inputs
    if mode == "sic":
        assert cache.hessian["mapping"].device.type == ("cpu" if offload else "cuda")
    cache.clear_mapping("mapping")
    assert not cache.events
