# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""RTN optimization for OSFP4."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from ._activation_groups import assemble_activation_quantization_groups

if TYPE_CHECKING:
    from llmcompressor.observers.osfp4 import OSFP4Observer

__all__ = [
    "RTNQuantizationResult",
    "optimize_rtn",
]

_QUANTIZATION_GROUP_SIZE = 16


class RTNQuantizationResult(NamedTuple):
    """Final RTN alpha and checkpoint-layout weight qparams for one mapping."""

    alpha: torch.Tensor
    gamma_w_final: torch.Tensor
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
    # Each block spans 16 input channels: W_blocks[L, m, 16].
    m, n = W.shape
    L = n // _QUANTIZATION_GROUP_SIZE
    W_blocks = (
        W.detach().reshape(m, L, _QUANTIZATION_GROUP_SIZE).permute(1, 0, 2).contiguous()
    )
    sigma_x_squared_blocks = (
        sigma_x_squared.detach()
        .to(device=W.device, dtype=torch.float32)
        .reshape(L, _QUANTIZATION_GROUP_SIZE)
    )
    X_blocks = None
    if cached_activation_batches is not None:
        X_blocks = assemble_activation_quantization_groups(
            cached_activation_batches,
            L,
            W.device,
        )
    optimized = observer.optimize_quantization_group_scales(
        W_blocks,
        X_blocks=X_blocks,
        weight_metric=sigma_x_squared_blocks,
        alpha_dtype=alpha_dtype,
    )
    del X_blocks
    scales, zero_points = observer.select_weight_qparams(
        W_blocks,
        optimized,
        weight_metric=sigma_x_squared_blocks,
    )

    # Convert selected qparams to checkpoint layout.
    alpha = optimized.alpha.reshape(n)
    gamma_w_final = scales.squeeze(-1).transpose(0, 1).contiguous()
    weight_zero_point = zero_points.squeeze(-1).transpose(0, 1).contiguous()

    return RTNQuantizationResult(alpha, gamma_w_final, weight_zero_point)
