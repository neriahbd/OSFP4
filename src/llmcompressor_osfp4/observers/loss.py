# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

from typing import NamedTuple

import torch

from .lut import phi_lut


class _BroadcastAlpha(torch.autograd.Function):
    """Expand ``alpha [L, 16]`` over one dimension; reduce its gradient in FP64.

    Autograd would otherwise sum alpha's gradient over the ``m`` weight rows (or
    the ``k`` activation rows) in FP32, in an order CUDA picks from the number of
    blocks evaluated together. Accumulating in FP64 and rounding once yields the
    correctly rounded FP32 sum unless that sum lies within FP64 error of an FP32
    rounding boundary, so the gradient is more accurate and in practice does not
    depend on how many blocks share a chunk. The forward values are unchanged.

    Used only when ``exact_alpha_reduction`` is set, which the observer does for
    problems large enough to be chunked (``uses_exact_alpha_reduction``). Smaller
    problems keep the original FP32 broadcast and its exact numerics.
    """

    @staticmethod
    def forward(ctx, alpha: torch.Tensor, dim: int, size: int) -> torch.Tensor:
        ctx.dim = dim
        L, block_width = alpha.shape
        if dim == 1:
            return alpha.view(L, 1, block_width).expand(L, size, block_width)
        return alpha.view(L, block_width, 1).expand(L, block_width, size)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        reduced = torch.sum(grad, dim=ctx.dim, dtype=torch.float64)
        return reduced.to(grad.dtype), None, None


def _broadcast_alpha(alpha: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Return alpha ``[L, 16]`` expanded along ``dim`` of an ``[L, *, *]`` block."""
    return _BroadcastAlpha.apply(alpha, dim, size)


class JointLossCoefficients(NamedTuple):
    """Invariant W4 ``[L, m, 16]`` and A4 ``[L, 16, k]`` loss coefficients."""

    # L = n / 16 blocks, m output channels, k sampled activation rows.
    coeff_w: torch.Tensor
    coeff_x: torch.Tensor


def build_weight_loss_coefficients(
    W_blocks: torch.Tensor,
    *,
    weight_metric: torch.Tensor,
) -> torch.Tensor:
    """Build fixed ``Lw`` coefficients for ``W_blocks [L, m, 16]``."""
    L, _, block_width = W_blocks.shape
    with torch.no_grad():
        coefficients = W_blocks.square()
        coefficients.mul_(weight_metric.view(L, 1, block_width))
    return coefficients


def build_activation_loss_coefficients(
    X_blocks: torch.Tensor,
    *,
    W_blocks: torch.Tensor,
) -> torch.Tensor:
    """Build fixed ``Lx`` coefficients for ``X_blocks [L, 16, k]``."""
    L, block_width, _ = X_blocks.shape
    with torch.no_grad():
        coefficients = X_blocks.square()
        weight_column_energy = W_blocks.square().sum(dim=1)
        coefficients.mul_(weight_column_energy.view(L, block_width, 1))
    return coefficients


def build_joint_loss_coefficients(
    W_blocks: torch.Tensor,
    X_blocks: torch.Tensor,
    *,
    weight_metric: torch.Tensor,
) -> JointLossCoefficients:
    """Build fixed weight and activation coefficients for the joint loss."""
    return JointLossCoefficients(
        coeff_w=build_weight_loss_coefficients(
            W_blocks,
            weight_metric=weight_metric,
        ),
        coeff_x=build_activation_loss_coefficients(
            X_blocks,
            W_blocks=W_blocks,
        ),
    )


def compute_weight_loss(
    W_target_blocks: torch.Tensor,
    alpha: torch.Tensor,
    gamma_w: torch.Tensor,
    coeff_w: torch.Tensor,
    *,
    exact_alpha_reduction: bool = False,
) -> torch.Tensor:
    """Return ``Lw`` per block from normalized metric coefficients."""
    L, m, _ = W_target_blocks.shape
    if exact_alpha_reduction:
        alpha_rows = _broadcast_alpha(alpha, 1, m)
    else:
        alpha_rows = alpha.view(L, 1, -1)
    phi_w = phi_lut(W_target_blocks / (alpha_rows * gamma_w))
    weighted_w = coeff_w * phi_w
    return torch.sum(weighted_w, dim=(1, 2)) / m


def compute_activation_loss(
    X_blocks: torch.Tensor,
    alpha: torch.Tensor,
    gamma_x: torch.Tensor,
    coeff_x: torch.Tensor,
    *,
    m: int,
    exact_alpha_reduction: bool = False,
) -> torch.Tensor:
    """Return normalized ``Lx`` per block without updating scale tensors."""
    L = X_blocks.shape[0]
    k = X_blocks.shape[2]
    if exact_alpha_reduction:
        alpha_columns = _broadcast_alpha(alpha, 2, k)
    else:
        alpha_columns = alpha.view(L, -1, 1)
    phi_x = phi_lut(alpha_columns * X_blocks / gamma_x)
    return torch.sum(coeff_x * phi_x, dim=(1, 2)) / (m * k)


def compute_joint_loss(
    W_target_blocks: torch.Tensor,
    X_blocks: torch.Tensor,
    alpha: torch.Tensor,
    gamma_w: torch.Tensor,
    gamma_x: torch.Tensor,
    coefficients: JointLossCoefficients,
    *,
    exact_alpha_reduction: bool = False,
) -> torch.Tensor:
    """Return one differentiable joint W4/A4 loss per block."""
    loss_w = compute_weight_loss(
        W_target_blocks,
        alpha,
        gamma_w,
        coefficients.coeff_w,
        exact_alpha_reduction=exact_alpha_reduction,
    )
    loss_x = compute_activation_loss(
        X_blocks,
        alpha,
        gamma_x,
        coefficients.coeff_x,
        m=W_target_blocks.shape[1],
        exact_alpha_reduction=exact_alpha_reduction,
    )
    return loss_w + loss_x
