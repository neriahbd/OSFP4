"""Prepare plugin checkpoint parameters from a diagonal calibration result."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import Module

from .calibration_cache import OSFP4CalibrationCache
from .osfp4_quantize import quantize_mapping
from .smoothing import (
    deploy_mapping_smooth_quant_scale,
    update_smoothed_input_global_scale,
)

if TYPE_CHECKING:
    from .base import OSFP4Mapping


def prepare_mapping_deployment(
    mapping: OSFP4Mapping,
    cache: OSFP4CalibrationCache,
    *,
    mode: Literal["rtn", "sic"],
    dampening_frac: float,
    weight_only: bool,
    optimization_input_batches: tuple[torch.Tensor, ...] | None = None,
    _deployment_stage: Callable[[str], None] | None = None,
) -> list[tuple[Module, dict[str, torch.Tensor]]]:
    """Normalize cached statistics, optimize one mapping, and return deployment."""
    layers = mapping.balance_layers
    result = quantize_mapping(
        mapping,
        cache,
        optimization_mode=mode,
        dampening_frac=dampening_frac,
        weight_only=weight_only,
        sampled_activations=optimization_input_batches,
    )
    quantized_weight = result.quantized_weight if mode == "sic" else None
    if _deployment_stage is not None:
        _deployment_stage("smoothing")
    scale = deploy_mapping_smooth_quant_scale(mapping, result.alpha)

    if not weight_only:
        if _deployment_stage is not None:
            _deployment_stage("input-scale replay")
        update_smoothed_input_global_scale(mapping, cache, scale)
    if _deployment_stage is not None:
        _deployment_stage("parameter preparation")
    deployment: list[tuple[Module, dict[str, torch.Tensor]]] = []
    row_start = 0
    for layer in layers:
        row_end = row_start + layer.weight.shape[0]
        qparams: dict[str, torch.Tensor] = {}
        if quantized_weight is not None:
            qparams["weight"] = quantized_weight[row_start:row_end].to(layer.weight)
        qparams["weight_scale"] = result.gamma_w_final[row_start:row_end].to(
            layer.weight_scale
        )
        qparams["weight_zero_point"] = result.weight_zero_point[row_start:row_end].to(
            layer.weight_zero_point
        )
        qparams["weight_global_scale"] = torch.ones_like(layer.weight_global_scale)
        deployment.append((layer, qparams))
        row_start = row_end
    return deployment
