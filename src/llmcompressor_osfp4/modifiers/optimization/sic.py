# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Successive interference cancellation for OSFP4."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA
from torch.nn import Linear

from llmcompressor.observers.osfp4 import OptimizedQuantizationGroupScales

from ._activation_groups import assemble_activation_quantization_groups

if TYPE_CHECKING:
    from llmcompressor.observers.osfp4 import OSFP4Observer

__all__ = [
    "SICQuantizationResult",
    "accumulate_hessian",
    "make_empty_hessian",
    "optimize_sic",
]

_QUANTIZATION_GROUP_SIZE = 16


def make_empty_hessian(
    module: Linear,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Allocate an FP32 Hessian matrix for an OSFP4 Linear module."""
    device = module.weight.device if device is None else device
    n = module.weight.shape[1]
    return torch.zeros(
        (n, n),
        device=device,
        dtype=torch.float32,
    )


def accumulate_hessian(
    inp: torch.Tensor,
    module: Linear,
    hessian: torch.Tensor,
) -> torch.Tensor:
    """Accumulate the FP32 Hessian matrix using ``X.T @ X``."""
    X_rows = inp.to(device=hessian.device, dtype=torch.float32).reshape(
        -1, inp.shape[-1]
    )
    hessian.addmm_(X_rows.transpose(0, 1), X_rows)
    return hessian


class SICQuantizationResult(NamedTuple):
    """Final SIC qparams and post-alpha quantized weights for one mapping."""

    alpha: torch.Tensor
    gamma_w_final: torch.Tensor
    weight_zero_point: torch.Tensor
    quantized_weight: torch.Tensor


def optimize_sic(
    W: torch.Tensor,
    cached_activation_batches: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    sigma_x: torch.Tensor,
    *,
    observer: OSFP4Observer,
    alpha_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    percdamp: float = 0.01,
) -> SICQuantizationResult:
    """Optimize qparams and run right-to-left SIC for one validated mapping."""
    m, n = W.shape
    W = W.detach().to(dtype=torch.float32)
    sigma_x_damped = sigma_x.detach().to(device=W.device, dtype=torch.float32).clone()
    damp = percdamp * torch.mean(torch.diag(sigma_x_damped))
    diag = torch.arange(sigma_x_damped.shape[0], device=sigma_x_damped.device)
    sigma_x_damped[diag, diag] += damp
    U = torch.linalg.cholesky(sigma_x_damped, upper=True)
    U_diag = torch.diag(U)
    del sigma_x_damped

    L = n // _QUANTIZATION_GROUP_SIZE
    W_blocks = W.reshape(m, L, _QUANTIZATION_GROUP_SIZE).permute(1, 0, 2).contiguous()
    u_diag_squared_blocks = U_diag.square().reshape(L, _QUANTIZATION_GROUP_SIZE)
    X_blocks = None
    if cached_activation_batches is not None:
        X_blocks = assemble_activation_quantization_groups(
            cached_activation_batches, L, W.device
        )
    optimized = observer.optimize_quantization_group_scales(
        W_blocks,
        X_blocks=X_blocks,
        weight_metric=u_diag_squared_blocks,
        alpha_dtype=alpha_dtype,
    )
    del W_blocks, X_blocks

    alpha = optimized.alpha.reshape(n)
    gamma_w_final = W.new_empty((m, L))
    weight_zero_point = torch.empty_like(gamma_w_final)

    Y = U @ W.transpose(0, 1)
    W_hat = torch.zeros_like(W)
    for block_idx in range(L - 1, -1, -1):
        block_start = block_idx * _QUANTIZATION_GROUP_SIZE
        block_end = block_start + _QUANTIZATION_GROUP_SIZE

        U_block = U[block_start:block_end, block_start:block_end]
        U_block_diag = U_diag[block_start:block_end]
        U_block_causal = torch.triu(U_block, diagonal=1)
        W_l = W[:, block_start:block_end]
        Y_block = Y[block_start:block_end, :]
        Y_block_approx = Y_block - U_block_causal @ W_l.transpose(0, 1)
        W_block_target = (
            (Y_block_approx / U_block_diag[:, None]).transpose(0, 1).unsqueeze(0)
        )
        block_optimized = OptimizedQuantizationGroupScales(
            optimized.alpha[block_idx : block_idx + 1],
            optimized.gamma_w[block_idx : block_idx + 1],
        )
        scales, zero_points = observer.select_weight_qparams(
            W_block_target,
            block_optimized,
            weight_metric=u_diag_squared_blocks[block_idx : block_idx + 1],
        )
        gamma_w_final_l = scales.view(-1)
        gamma_w_final[:, block_idx] = gamma_w_final_l
        weight_zero_point[:, block_idx] = zero_points.view(-1)
        alpha_l = optimized.alpha[block_idx]

        for t in range(block_end - 1, block_start - 1, -1):
            alpha_r = alpha_l[t - block_start]
            W_SIC_t = FP4_E2M1_DATA.cast_to_fp4(
                (Y[t, :] / U[t, t]) / (alpha_r * gamma_w_final_l)
            )
            W_hat[:, t] = (W_SIC_t * gamma_w_final_l).to(dtype=weight_dtype).float()
            Y[:t, :] -= U[:t, t].unsqueeze(1) * (alpha_r * W_hat[:, t]).unsqueeze(0)

    return SICQuantizationResult(
        alpha=alpha,
        gamma_w_final=gamma_w_final,
        weight_zero_point=weight_zero_point,
        quantized_weight=W_hat,
    )
