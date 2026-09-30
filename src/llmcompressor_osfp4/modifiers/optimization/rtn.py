"""RTN optimization for OSFP4."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from ._activation_groups import assemble_activation_quantization_groups

if TYPE_CHECKING:
    from llmcompressor_osfp4.observers import OSFP4Observer

__all__ = [
    "RTNQuantizationResult",
    "optimize_rtn",
]

_QUANTIZATION_GROUP_SIZE = 16


class RTNQuantizationResult(NamedTuple):
    """Final RTN alpha and checkpoint-layout weight qparams for one mapping."""

    alpha_star: torch.Tensor
    gamma_w_star: torch.Tensor
    weight_zero_point: torch.Tensor


def optimize_rtn(
    W: torch.Tensor,
    cached_activation_batches: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    *,
    observer: OSFP4Observer,
    alpha_dtype: torch.dtype,
    sigma_x_squared: torch.Tensor,
) -> RTNQuantizationResult:
    """Optimize RTN qparams for one validated ``W[rows, columns]`` mapping."""
    # Prepare weight groups and the normalized activation-energy metric.
    rows, columns = W.shape
    groups = columns // _QUANTIZATION_GROUP_SIZE
    weight_groups = (
        W.detach()
        .reshape(rows, groups, _QUANTIZATION_GROUP_SIZE)
        .permute(1, 0, 2)
        .contiguous()
    )
    metric = (
        sigma_x_squared.detach()
        .to(device=W.device, dtype=torch.float32)
        .reshape(groups, _QUANTIZATION_GROUP_SIZE)
    )
    activation_groups = None
    if cached_activation_batches is not None:
        activation_groups = assemble_activation_quantization_groups(
            cached_activation_batches,
            groups,
            W.device,
        )
    optimized = observer.optimize_quantization_group_scales(
        weight_groups,
        activation_quantization_groups=activation_groups,
        weight_metric=metric,
        alpha_dtype=alpha_dtype,
    )
    del activation_groups
    scales, zero_points = observer.select_weight_qparams(
        weight_groups,
        optimized,
        weight_metric=metric,
    )

    # Convert selected qparams to checkpoint layout.
    alpha_star = optimized.alpha_star.reshape(columns)
    gamma_w_star = scales.squeeze(-1).transpose(0, 1).contiguous()
    weight_zero_point = zero_points.squeeze(-1).transpose(0, 1).contiguous()

    return RTNQuantizationResult(alpha_star, gamma_w_star, weight_zero_point)
