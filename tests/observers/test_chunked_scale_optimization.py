# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Block-chunked scale optimization must match the whole-tensor optimizer."""

import pytest
import torch

from llmcompressor_osfp4.observers import observer as osfp4_observer
from llmcompressor_osfp4.observers.loss import (
    JointLossCoefficients,
    build_joint_loss_coefficients,
    build_weight_loss_coefficients,
    compute_joint_loss,
    compute_weight_loss,
)
from llmcompressor_osfp4.observers.lut import phi_lut
from llmcompressor_osfp4.observers.scale_optimization import (
    create_log_scale_parameters,
    initialize_scale_values_from_absmax,
    run_chunked_optimizer,
    run_optimizer,
    select_blocks_per_chunk,
    uses_exact_alpha_reduction,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def make_blocks(L=7, m=5, k=11, device="cpu"):
    generator = torch.Generator().manual_seed(1234)
    W = torch.randn(L, m, 16, generator=generator) * 0.02
    X = torch.randn(L, 16, k, generator=generator)
    # Exact zeros and values outside the LUT domain on both sides.
    X[0, 0, :3] = 0.0
    X[1, 2, :2] = 1e-6
    X[2, 3, :2] = 1e4
    W[3, 0, :4] = 0.0
    W[4, 1, 0] = 5.0
    metric = torch.rand(L, 16, generator=generator) + 0.1
    return W.to(device), X.to(device), metric.to(device)


def optimize(W, X, metric, *, steps, blocks_per_chunk, exact=False):
    """Run Adam with the whole-tensor runner (None) or the chunked runner."""
    L = W.shape[0]
    initial = initialize_scale_values_from_absmax(
        W, X, blocks_per_chunk=blocks_per_chunk
    )
    log_scales = create_log_scale_parameters(initial)
    if X is None:
        coefficients = build_weight_loss_coefficients(W, weight_metric=metric)
    else:
        coefficients = build_joint_loss_coefficients(W, X, weight_metric=metric)

    def loss(blocks, scales):
        if X is None:
            return compute_weight_loss(
                W[blocks],
                scales.log_alpha.exp(),
                scales.log_gamma_w.exp(),
                coefficients[blocks],
                exact_alpha_reduction=exact,
            ).sum()
        return compute_joint_loss(
            W[blocks],
            X[blocks],
            scales.log_alpha.exp(),
            scales.log_gamma_w.exp(),
            scales.log_gamma_x.exp(),
            JointLossCoefficients(
                coefficients.coeff_w[blocks], coefficients.coeff_x[blocks]
            ),
            exact_alpha_reduction=exact,
        ).sum()

    if blocks_per_chunk is None:
        run_optimizer(
            log_scales, lambda: loss(slice(None), log_scales), steps=steps, lr=0.12
        )
    else:
        run_chunked_optimizer(
            log_scales,
            loss,
            steps=steps,
            lr=0.12,
            num_blocks=L,
            blocks_per_chunk=blocks_per_chunk,
        )
    return [p.detach().clone() for p in log_scales if p is not None]


@pytest.mark.parametrize("weight_only", [False, True])
@pytest.mark.parametrize("blocks_per_chunk", [1, 2, 3, 7, 100])
def test_chunked_adam_matches_whole_tensor_bytes_on_cpu(weight_only, blocks_per_chunk):
    W, X, metric = make_blocks()
    X = None if weight_only else X

    expected = optimize(W, X, metric, steps=4, blocks_per_chunk=None)
    actual = optimize(W, X, metric, steps=4, blocks_per_chunk=blocks_per_chunk)

    for a, e in zip(actual, expected):
        assert_bytes(a, e)


@pytest.mark.parametrize("weight_only", [False, True])
@pytest.mark.parametrize("blocks_per_chunk", [1, 3, 7])
def test_chunked_initialization_matches_bytes(weight_only, blocks_per_chunk):
    W, X, _ = make_blocks()
    X = None if weight_only else X

    expected = initialize_scale_values_from_absmax(W, X)
    actual = initialize_scale_values_from_absmax(
        W, X, blocks_per_chunk=blocks_per_chunk
    )

    for a, e in zip(actual, expected):
        if e is None:
            assert a is None
        else:
            assert_bytes(a, e)


@pytest.mark.parametrize("weight_only", [False, True])
def test_observer_optimization_does_not_depend_on_chunking_on_cpu(
    monkeypatch, weight_only
):
    W, X, metric = make_blocks()

    def run():
        if weight_only:
            return osfp4_observer._optimize_weight_scales(W, metric, steps=3, lr=0.12)
        return osfp4_observer._optimize_joint_scales(
            W, X, weight_metric=metric, steps=3, lr=0.12
        )

    expected = run()  # small problem: one chunk
    monkeypatch.setattr(
        osfp4_observer, "select_blocks_per_chunk", lambda m, k, num_blocks: 2
    )
    actual = run()

    for a, e in zip(actual, expected):
        if e is None:
            assert a is None
        else:
            assert_bytes(a, e)


def test_chunked_optimizer_clears_gradients_and_rejects_empty_chunks():
    W, X, metric = make_blocks()
    log_scales = create_log_scale_parameters(initialize_scale_values_from_absmax(W, X))
    coefficients = build_joint_loss_coefficients(W, X, weight_metric=metric)

    def loss(blocks, scales):
        return compute_joint_loss(
            W[blocks],
            X[blocks],
            scales.log_alpha.exp(),
            scales.log_gamma_w.exp(),
            scales.log_gamma_x.exp(),
            JointLossCoefficients(
                coefficients.coeff_w[blocks], coefficients.coeff_x[blocks]
            ),
        ).sum()

    run_chunked_optimizer(
        log_scales, loss, steps=1, lr=0.1, num_blocks=W.shape[0], blocks_per_chunk=3
    )
    assert all(p.grad is None for p in log_scales)
    with pytest.raises(ValueError, match="positive"):
        run_chunked_optimizer(
            log_scales, loss, steps=1, lr=0.1, num_blocks=W.shape[0], blocks_per_chunk=0
        )


def test_blocks_per_chunk_depends_only_on_shape():
    # 16 graph copies of 64 blocks of width 16, each FP32.
    assert not uses_exact_alpha_reduction(4095, 0, 64)
    assert not uses_exact_alpha_reduction(4096, 0, 64)
    assert uses_exact_alpha_reduction(4097, 0, 64)
    # Small problems keep a single chunk, matching the whole-tensor path.
    assert select_blocks_per_chunk(m=5, k=11, num_blocks=7) == 7
    # Llama-3.1-8B shapes: qkv, gate_up, and down_proj with auto subsampling.
    assert select_blocks_per_chunk(m=6144, k=8192, num_blocks=256) == 16
    assert select_blocks_per_chunk(m=28672, k=8192, num_blocks=256) == 16
    assert select_blocks_per_chunk(m=4096, k=28672, num_blocks=896) == 56
    # Weight-only gate_up is still large enough to split.
    assert select_blocks_per_chunk(m=28672, k=0, num_blocks=256) == 16
    # Never fewer than one block per chunk.
    assert select_blocks_per_chunk(m=10**6, k=10**6, num_blocks=3) == 1


@CUDA
@pytest.mark.parametrize("weight_only", [False, True])
@pytest.mark.parametrize("blocks_per_chunk", [3, 4, 31])
def test_chunked_adam_matches_whole_tensor_bytes_on_cuda(weight_only, blocks_per_chunk):
    # m = 1536 makes CUDA choose a different FP32 reduction order for 4 and 32
    # blocks, so this shape fails if alpha's gradient is summed in FP32.
    W, X, metric = make_blocks(L=32, m=1536, k=512, device="cuda")
    X = None if weight_only else X

    def run(chunk):
        return optimize(W, X, metric, steps=3, blocks_per_chunk=chunk, exact=True)

    expected = run(None)
    first = run(blocks_per_chunk)
    second = run(blocks_per_chunk)

    for a, b, e in zip(first, second, expected):
        assert_bytes(a, b)
        assert_bytes(a, e)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=CUDA)],
)
def test_alpha_gradient_is_correctly_rounded_and_chunk_invariant(device):
    W, X, metric = make_blocks(L=8, m=1536, k=512, device=device)
    coefficients = build_joint_loss_coefficients(W, X, weight_metric=metric)
    initial = initialize_scale_values_from_absmax(W, X)
    rows, samples = W.shape[1], X.shape[2]

    def alpha_gradient(blocks):
        alpha = initial.alpha[blocks].clone().requires_grad_(True)
        total = compute_joint_loss(
            W[blocks],
            X[blocks],
            alpha,
            initial.gamma_w[blocks],
            initial.gamma_x[blocks],
            JointLossCoefficients(
                coefficients.coeff_w[blocks], coefficients.coeff_x[blocks]
            ),
            exact_alpha_reduction=True,
        ).sum()
        return torch.autograd.grad(total, alpha)[0]

    def reference_alpha_gradient():
        # Independent formula: expanding alpha in FP64 makes autograd sum the same
        # FP32 terms in FP64 and round each of the two sums once.
        alpha = initial.alpha.clone().requires_grad_(True)
        alpha_w = alpha.double().unsqueeze(1).expand(-1, rows, -1).float()
        alpha_x = alpha.double().unsqueeze(2).expand(-1, -1, samples).float()
        phi_w = phi_lut(W / (alpha_w * initial.gamma_w))
        loss_w = (coefficients.coeff_w * phi_w).sum(dim=(1, 2)) / rows
        phi_x = phi_lut(alpha_x * X / initial.gamma_x)
        loss_x = (coefficients.coeff_x * phi_x).sum(dim=(1, 2)) / (rows * samples)
        return torch.autograd.grad((loss_w + loss_x).sum(), alpha)[0]

    whole = alpha_gradient(slice(None))
    chunks = [alpha_gradient(slice(start, start + 3)) for start in (0, 3, 6)]
    assert_bytes(torch.cat(chunks), whole)
    assert_bytes(whole, reference_alpha_gradient())


@CUDA
def test_joint_scale_optimization_peak_memory_is_about_three_copies():
    device = torch.device("cuda")
    L, m, k = 256, 4096, 16384  # n = 4096; whole-tensor peak is about 15 copies
    generator = torch.Generator().manual_seed(0)
    W = (torch.randn(L, m, 16, generator=generator) * 0.02).to(device)
    X = torch.randn(L, 16, k, generator=generator).to(device)
    metric = torch.rand(L, 16, generator=generator).to(device)
    block_bytes = (W.numel() + X.numel()) * 4

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    osfp4_observer._optimize_joint_scales(W, X, weight_metric=metric, steps=2, lr=0.12)
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - baseline

    # W_blocks and X_blocks already exist (copy 1); coefficients add copy 2 and the
    # chunk graph about one more.
    assert peak_extra <= 2.5 * block_bytes, peak_extra / block_bytes


def test_exact_alpha_reduction_covers_every_problem_that_may_be_split():
    # Every problem select_blocks_per_chunk may split uses the FP64 reduction, so
    # chunking never changes results; smaller problems keep the FP32 reference.
    shapes = [
        (m, k, L)
        for m in (5, 512, 1536, 4096, 6144, 28672)
        for k in (0, 11, 512, 8192, 28672)
        for L in (1, 7, 32, 256, 896)
    ]
    for m, k, L in shapes:
        if select_blocks_per_chunk(m, k, L) < L:
            assert uses_exact_alpha_reduction(m, k, L), (m, k, L)
    # Golden-fixture and unit-test shapes stay on the original arithmetic.
    assert not uses_exact_alpha_reduction(m=5, k=11, num_blocks=7)
    assert not uses_exact_alpha_reduction(m=128, k=64, num_blocks=4)
    # All Llama-3.1-8B mappings, both modes, take the exact path.
    for m, n, k in [
        (6144, 4096, 8192),
        (4096, 4096, 8192),
        (28672, 4096, 8192),
        (4096, 14336, 28672),
    ]:
        assert uses_exact_alpha_reduction(m, k, n // 16)
        assert uses_exact_alpha_reduction(m, 0, n // 16)


@pytest.mark.parametrize("weight_only", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_observer_passes_shape_gate_to_loss(monkeypatch, weight_only, exact):
    W, X, metric = make_blocks()
    seen = []
    name = "compute_weight_loss" if weight_only else "compute_joint_loss"
    original = getattr(osfp4_observer, name)

    def recording(*args, exact_alpha_reduction, **kwargs):
        seen.append(exact_alpha_reduction)
        return original(*args, exact_alpha_reduction=exact_alpha_reduction, **kwargs)

    monkeypatch.setattr(osfp4_observer, name, recording)
    monkeypatch.setattr(
        osfp4_observer, "uses_exact_alpha_reduction", lambda m, k, num_blocks: exact
    )
    if weight_only:
        osfp4_observer._optimize_weight_scales(W, metric, steps=2, lr=0.12)
    else:
        osfp4_observer._optimize_joint_scales(
            W, X, weight_metric=metric, steps=2, lr=0.12
        )
    assert seen and all(flag is exact for flag in seen)
