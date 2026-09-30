"""Exact layouts and arithmetic at the optimization package boundaries."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from llmcompressor_osfp4.modifiers.optimization import sic
from llmcompressor_osfp4.modifiers.optimization._activation_groups import (
    assemble_activation_quantization_groups,
)
from llmcompressor_osfp4.observers import OptimizedQuantizationGroupScales

DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("lengths", [(3, 0, 7), (0, 0)])
def test_activation_assembly_and_hessian_bytes(dtype, noncontiguous, lengths):
    generator = torch.Generator().manual_seed(17)
    batches = []
    for rows in lengths:
        values = torch.randn(rows, 64, generator=generator).to(dtype)
        batch = values[:, ::2] if noncontiguous else values[:, :32].contiguous()
        if rows:
            batch[0, 0] = -0.0
        batches.append(batch)
    before = [batch.clone() for batch in batches]
    grouped = assemble_activation_quantization_groups(batches, 2, torch.device("cpu"))
    expected_groups = torch.cat(
        [
            batch.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1, 2, 16)
            .permute(1, 2, 0)
            for batch in batches
        ],
        dim=2,
    ).contiguous()
    assert_bytes(grouped, expected_groups)
    assert grouped.is_contiguous()
    layer = torch.nn.Linear(32, 3, bias=False).to(dtype)
    hessian = sic.make_empty_hessian(layer, device="cpu")
    expected = torch.zeros(32, 32, dtype=torch.float32)
    for batch in batches:
        assert sic.accumulate_hessian(batch, layer, hessian) is hessian
        flat = batch.to(device="cpu", dtype=torch.float32).reshape(-1, 32)
        expected.addmm_(flat.transpose(0, 1), flat)
        assert_bytes(hessian, expected)
    for batch, original in zip(batches, before):
        assert_bytes(batch, original)


class ResidualTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.updates = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        is_update = func == torch.ops.aten.sub_.Tensor and args[0].ndim == 2
        if is_update:
            before, operand = args[0].clone(), args[1].clone()
        result = func(*args, **(kwargs or {}))
        if is_update:
            self.updates.append((before, operand, result.clone()))
        return result


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("groups", [1, 3])
def test_sic_quantizer_inputs_and_each_residual_update_match_bytes(
    monkeypatch, dtype, groups
):
    generator = torch.Generator().manual_seed(41)
    columns = groups * 16
    weight = torch.randn(3, columns * 2, generator=generator).to(dtype)[:, ::2]
    inputs = torch.randn(67, columns, generator=generator)
    hessian = inputs.T @ inputs / 67
    original_weight, original_hessian = weight.clone(), hessian.clone()
    alpha = torch.linspace(0.7, 1.5, columns).reshape(groups, 16).to(dtype).float()
    scales = torch.arange(1, groups * 3 + 1).reshape(groups, 3, 1).float() / 8
    # Distinct input dtype verifies SIC's explicit zero-point conversion.
    zero_points = torch.zeros_like(scales, dtype=torch.float64)
    W = weight.detach().float()
    H = hessian.detach().float().clone()
    H[torch.arange(columns), torch.arange(columns)] += 0.01 * torch.mean(torch.diag(H))
    U = torch.linalg.cholesky(H, upper=True)
    metric = torch.diag(U).square().reshape(groups, 16)
    expected_groups = W.reshape(3, groups, 16).permute(1, 0, 2).contiguous()
    calls = []

    class Observer:
        def optimize_quantization_group_scales(
            self, value, *, weight_metric, activation_quantization_groups, alpha_dtype
        ):
            calls.append("optimize")
            assert_bytes(value, expected_groups)
            assert_bytes(weight_metric, metric)
            assert activation_quantization_groups is None
            assert alpha_dtype == dtype
            return OptimizedQuantizationGroupScales(alpha, scales)

        def select_weight_qparams(self, value, optimized, *, weight_metric):
            group = groups - 1 - calls.count("select")
            assert len(ratios) == (groups - 1 - group) * 16
            calls.append("select")
            selection_targets.append(value.clone())
            assert_bytes(weight_metric, metric[group : group + 1])
            assert_bytes(optimized.alpha_star, alpha[group : group + 1])
            assert_bytes(optimized.gamma_w, scales[group : group + 1])
            return scales[group : group + 1], zero_points[group : group + 1]

    ratios = []
    selection_targets = []
    original_quantize = sic.quantize_e2m1

    def quantize(value):
        ratios.append(value.clone())
        return original_quantize(value)

    monkeypatch.setattr(sic, "quantize_e2m1", quantize)
    with ResidualTrace() as trace:
        result = sic.optimize_sic(
            weight,
            None,
            hessian,
            observer=Observer(),
            alpha_dtype=dtype,
            weight_dtype=dtype,
        )
    assert calls == ["optimize"] + ["select"] * groups
    assert len(ratios) == len(trace.updates) == columns
    Y = U @ W.transpose(0, 1)
    W_hat = torch.zeros_like(W)
    index = 0
    for group in range(groups - 1, -1, -1):
        start = group * 16
        scale = scales[group].view(-1)
        block_u = U[start : start + 16, start : start + 16]
        proxy = (
            Y[start : start + 16] - torch.triu(block_u, 1) @ W[:, start : start + 16].T
        )
        target = (proxy / block_u.diag()[:, None]).T.unsqueeze(0)
        assert_bytes(selection_targets[groups - 1 - group], target)
        for j in range(start + 15, start - 1, -1):
            alpha_j = alpha[group, j - start]
            ratio = (Y[j, :] / U[j, j]) / (alpha_j * scale)
            assert_bytes(ratios[index], ratio)
            W_hat[:, j] = (original_quantize(ratio) * scale).to(dtype).float()
            operand = U[:j, j].unsqueeze(1) * (alpha_j * W_hat[:, j]).unsqueeze(0)
            before, recorded_operand, after = trace.updates[index]
            assert_bytes(before, Y[:j, :])
            assert_bytes(recorded_operand, operand)
            Y[:j, :] -= operand
            assert_bytes(after, Y[:j, :])
            index += 1
    assert_bytes(result.quantized_weight, W_hat)
    assert_bytes(result.alpha_star, alpha.reshape(columns))
    assert_bytes(result.gamma_w_star, scales.squeeze(-1).transpose(0, 1).contiguous())
    assert_bytes(
        result.weight_zero_point,
        zero_points.squeeze(-1).transpose(0, 1).float().contiguous(),
    )
    assert_bytes(weight, original_weight)
    assert_bytes(hessian, original_hessian)


def test_sic_preserves_native_cholesky_failure_and_input_bytes():
    weight = torch.ones(3, 16)
    hessian = -torch.eye(16)
    before = hessian.clone()
    with pytest.raises(torch.linalg.LinAlgError):
        sic.optimize_sic(
            weight,
            None,
            hessian,
            observer=None,
            alpha_dtype=torch.float32,
            weight_dtype=torch.float32,
        )
    assert_bytes(hessian, before)


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_activation_transfer_only_uses_nonblocking_for_pinned_cuda(
    monkeypatch, pinned, device
):
    calls = []
    original_to = torch.Tensor.to
    original_empty = torch.empty

    def record_to(tensor, *, device, dtype, non_blocking):
        calls.append((device, dtype, non_blocking))
        return original_to(tensor, device="cpu", dtype=dtype)

    monkeypatch.setattr(torch.Tensor, "to", record_to)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: pinned)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: original_empty(*args, **dict(kwargs, device="cpu")),
    )
    batch = torch.arange(32).reshape(2, 16)
    result = assemble_activation_quantization_groups([batch], 1, torch.device(device))
    assert calls == [(torch.device(device), torch.float32, pinned and device == "cuda")]
    assert_bytes(result, batch.float().reshape(2, 1, 16).permute(1, 2, 0).contiguous())
