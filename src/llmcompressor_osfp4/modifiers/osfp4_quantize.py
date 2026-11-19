# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Compute OSFP4 quantization results for one mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from .calibration_cache import OSFP4CalibrationCache
from .optimization import (
    RTNQuantizationResult,
    SICQuantizationResult,
    optimize_rtn,
    optimize_sic,
)

if TYPE_CHECKING:
    from .base import OSFP4Mapping

__all__ = ["quantize_mapping"]


def quantize_mapping(
    mapping: OSFP4Mapping,
    cache: OSFP4CalibrationCache,
    *,
    optimization_mode: Literal["rtn", "sic"],
    dampening_frac: float,
    weight_only: bool,
    sampled_activations: tuple[torch.Tensor, ...] | None = None,
) -> RTNQuantizationResult | SICQuantizationResult:
    """Optimize one mapping without installing weights or smoothing."""
    balance_layers = mapping.balance_layers
    weight_dtype = balance_layers[0].weight.dtype
    device = balance_layers[0].weight.device
    weight = torch.cat(
        [
            layer.weight.detach().to(device=device, dtype=torch.float32)
            for layer in balance_layers
        ],
        dim=0,
    )
    cache.wait_for_capture(mapping.mapping_name, weight.device)
    alpha_dtype = torch.bfloat16 if mapping.requires_runtime_smoothing else weight_dtype
    observer = balance_layers[0].weight_observer
    cached_activations = (
        None if weight_only else cache.cached_activations[mapping.mapping_name]
    )
    optimization_activations = (
        cached_activations if sampled_activations is None else sampled_activations
    )
    token_count = cache.token_count[mapping.mapping_name]
    if optimization_mode == "sic":
        result = optimize_sic(
            weight,
            optimization_activations,
            cache.hessian[mapping.mapping_name].to(weight.device) / token_count,
            observer=observer,
            alpha_dtype=alpha_dtype,
            weight_dtype=weight_dtype,
            percdamp=dampening_frac,
        )
    else:
        sigma_x_squared = (
            cache.sigma_x_squared_sum[mapping.mapping_name].to(weight.device) / token_count
        )
        result = optimize_rtn(
            weight,
            optimization_activations,
            observer=observer,
            alpha_dtype=alpha_dtype,
            sigma_x_squared=sigma_x_squared,
        )

    return result
