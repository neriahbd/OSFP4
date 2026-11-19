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
    """Invariant W4 ``[G, R, 16]`` and A4 ``[G, 16, S]`` loss coefficients."""

    coeff_w: torch.Tensor
    coeff_x: torch.Tensor


def build_weight_loss_coefficients(
    weight_target: torch.Tensor,
    *,
    weight_metric: torch.Tensor,
) -> torch.Tensor:
    """Build immutable ``Lw`` coefficients for ``[groups, rows, 16]`` weights."""
    groups, _, width = weight_target.shape
    with torch.no_grad():
        coefficients = weight_target.square()
        coefficients.mul_(
            weight_metric.view(
                groups,
                1,
                width,
            )
        )
    return coefficients


def build_activation_loss_coefficients(
    activation_quantization_groups: torch.Tensor,
    *,
    activation_reference: torch.Tensor,
) -> torch.Tensor:
    """Build immutable ``Lx`` coefficients for ``[groups, 16, samples]`` inputs."""
    groups, width, _ = activation_quantization_groups.shape
    with torch.no_grad():
        coefficients = activation_quantization_groups.square()
        column_energy = activation_reference.square().sum(dim=1)
        coefficients.mul_(
            column_energy.view(
                groups,
                width,
                1,
            )
        )
    return coefficients


def build_joint_loss_coefficients(
    target: torch.Tensor,
    activations: torch.Tensor,
    *,
    weight_metric: torch.Tensor,
    activation_reference: torch.Tensor | None = None,
) -> JointLossCoefficients:
    """Build immutable weight and activation coefficients for the joint loss."""
    return JointLossCoefficients(
        coeff_w=build_weight_loss_coefficients(
            target,
            weight_metric=weight_metric,
        ),
        coeff_x=build_activation_loss_coefficients(
            activations,
            activation_reference=target
            if activation_reference is None
            else activation_reference,
        ),
    )


def compute_weight_loss(
    weight_target: torch.Tensor,
    alpha: torch.Tensor,
    gamma_w: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    exact_alpha_reduction: bool = False,
) -> torch.Tensor:
    """Return ``Lw`` per block from normalized metric coefficients."""
    L, m, _ = weight_target.shape
    if exact_alpha_reduction:
        alpha_rows = _broadcast_alpha(alpha, 1, m)
    else:
        alpha_rows = alpha.view(L, 1, -1)
    phi_w = phi_lut(weight_target / (alpha_rows * gamma_w))
    weighted_w = coefficients * phi_w
    return torch.sum(weighted_w, dim=(1, 2)) / m


def compute_activation_loss(
    activation_quantization_groups: torch.Tensor,
    alpha: torch.Tensor,
    gamma_x: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    row_count: int,
    exact_alpha_reduction: bool = False,
) -> torch.Tensor:
    """Return normalized ``Lx`` per block without updating scale tensors."""
    L = activation_quantization_groups.shape[0]
    k = activation_quantization_groups.shape[2]
    if exact_alpha_reduction:
        alpha_columns = _broadcast_alpha(alpha, 2, k)
    else:
        alpha_columns = alpha.view(L, -1, 1)
    phi_x = phi_lut(alpha_columns * activation_quantization_groups / gamma_x)
    return torch.sum(coefficients * phi_x, dim=(1, 2)) / (row_count * k)


def compute_joint_loss(
    target: torch.Tensor,
    activations: torch.Tensor,
    alpha: torch.Tensor,
    gamma_w: torch.Tensor,
    gamma_x: torch.Tensor,
    coefficients: JointLossCoefficients,
    *,
    exact_alpha_reduction: bool = False,
) -> torch.Tensor:
    """Return one differentiable joint W4/A4 loss per block."""
    loss_w = compute_weight_loss(
        target,
        alpha,
        gamma_w,
        coefficients.coeff_w,
        exact_alpha_reduction=exact_alpha_reduction,
    )
    loss_x = compute_activation_loss(
        activations,
        alpha,
        gamma_x,
        coefficients.coeff_x,
        row_count=target.shape[1],
        exact_alpha_reduction=exact_alpha_reduction,
    )
    return loss_w + loss_x
