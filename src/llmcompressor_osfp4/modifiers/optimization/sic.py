"""Sequential importance compensation for OSFP4."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch.nn import Linear

from llmcompressor_osfp4.observers import OptimizedQuantizationGroupScales
from llmcompressor_osfp4.observers.fp4 import quantize_e2m1

from ._activation_groups import assemble_activation_quantization_groups

if TYPE_CHECKING:
    from llmcompressor_osfp4.observers import OSFP4Observer

__all__ = [
    "SICQuantizationResult",
    "accumulate_hessian",
    "make_empty_hessian",
    "optimize_sic",
]

SIC_PRECISION = torch.float32
_QUANTIZATION_GROUP_SIZE = 16


def make_empty_hessian(
    module: Linear,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Allocate an FP32 raw input Gram matrix for an OSFP4 Linear module."""
    device = module.weight.device if device is None else device
    columns = module.weight.shape[1]
    return torch.zeros(
        (columns, columns),
        device=device,
        dtype=SIC_PRECISION,
    )


def accumulate_hessian(
    inp: torch.Tensor,
    module: Linear,
    hessian: torch.Tensor,
) -> torch.Tensor:
    """Accumulate the raw FP32 activation Gram matrix ``X.T @ X``."""
    flat = inp.to(device=hessian.device, dtype=SIC_PRECISION).reshape(-1, inp.shape[-1])
    hessian.addmm_(flat.transpose(0, 1), flat)
    return hessian


class SICQuantizationResult(NamedTuple):
    """Final SIC qparams and post-alpha quantized weights for one mapping."""

    alpha_star: torch.Tensor
    gamma_w_star: torch.Tensor
    weight_zero_point: torch.Tensor
    quantized_weight: torch.Tensor


def optimize_sic(
    W: torch.Tensor,
    cached_activation_batches: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    hessian: torch.Tensor,
    *,
    observer: OSFP4Observer,
    alpha_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    percdamp: float = 0.01,
) -> SICQuantizationResult:
    """Optimize qparams and run right-to-left SIC for one validated mapping."""
    # Factor the damped second moment without mutating the cached Hessian.
    num_rows, num_columns = W.shape
    W = W.detach().to(dtype=torch.float32)
    H = hessian.detach().to(device=W.device, dtype=torch.float32).clone()
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp
    U = torch.linalg.cholesky(H, upper=True)
    diag_U = torch.diag(U)
    del H

    # Optimize original-weight groups together, defer E4M3 selection to the sweep.
    groups = num_columns // _QUANTIZATION_GROUP_SIZE
    weight_groups = (
        W.reshape(num_rows, groups, _QUANTIZATION_GROUP_SIZE)
        .permute(1, 0, 2)
        .contiguous()
    )
    metric = diag_U.square().reshape(
        groups,
        _QUANTIZATION_GROUP_SIZE,
    )
    activation_groups = None
    if cached_activation_batches is not None:
        activation_groups = assemble_activation_quantization_groups(
            cached_activation_batches, groups, W.device
        )
    optimized = observer.optimize_quantization_group_scales(
        weight_groups,
        activation_quantization_groups=activation_groups,
        weight_metric=metric,
        alpha_dtype=alpha_dtype,
    )
    del weight_groups, activation_groups

    # Store each block's selected qparams in checkpoint layout.
    alpha_values = optimized.alpha_star.reshape(num_columns)
    scale_values = W.new_empty((num_rows, groups))
    zero_point_values = torch.empty_like(scale_values)

    # Quantize right-to-left and propagate the alpha-restored residual.
    Y = U @ W.transpose(0, 1)
    Q = torch.zeros_like(W)
    for block_idx in range(groups - 1, -1, -1):
        block_start = block_idx * _QUANTIZATION_GROUP_SIZE
        block_end = block_start + _QUANTIZATION_GROUP_SIZE

        U_local = U[block_start:block_end, block_start:block_end]
        U_diag_block = diag_U[block_start:block_end]
        U_upper_strict = torch.triu(U_local, diagonal=1)
        W_local = W[:, block_start:block_end]
        unwanted_cross_terms = U_upper_strict @ W_local.transpose(0, 1)
        Y_local = Y[block_start:block_end, :]
        T_eff = (
            ((Y_local - unwanted_cross_terms) / U_diag_block[:, None])
            .transpose(0, 1)
            .unsqueeze(0)
        )
        block_optimized = OptimizedQuantizationGroupScales(
            optimized.alpha_star[block_idx : block_idx + 1],
            optimized.gamma_w[block_idx : block_idx + 1],
        )
        scales, zero_points = observer.select_weight_qparams(
            T_eff,
            block_optimized,
            weight_metric=metric[block_idx : block_idx + 1],
        )
        scale_1d = scales.view(-1)
        scale_values[:, block_idx] = scale_1d
        zero_point_values[:, block_idx] = zero_points.view(-1)
        opt_alpha_block = optimized.alpha_star[block_idx]

        for ii in range(block_end - 1, block_start - 1, -1):
            alpha_i = opt_alpha_block[ii - block_start]
            q_i = quantize_e2m1((Y[ii, :] / U[ii, ii]) / (alpha_i * scale_1d))
            Q[:, ii] = (q_i * scale_1d).to(dtype=weight_dtype).float()
            Y[:ii, :] -= U[:ii, ii].unsqueeze(1) * (alpha_i * Q[:, ii]).unsqueeze(0)

    return SICQuantizationResult(
        alpha_star=alpha_values,
        gamma_w_star=scale_values,
        weight_zero_point=zero_point_values,
        quantized_weight=Q,
    )
