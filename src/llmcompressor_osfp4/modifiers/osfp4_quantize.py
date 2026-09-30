"""OSFP4 mapping quantization orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import Module

from llmcompressor.modifiers.quantization.calibration import update_qparams

from .calibration_cache import OSFP4CalibrationCache
from .optimization import optimize_rtn, optimize_sic
from .smoothing import deploy_mapping_smooth_quant_scale

if TYPE_CHECKING:
    from .base import OSFP4Mapping

__all__ = ["quantize_mapping"]


def _update_smoothed_input_global_scale(
    layers: tuple[Module, ...],
    batches: list[torch.Tensor] | tuple[torch.Tensor, ...],
    scale: torch.Tensor,
) -> None:
    """Recompute the saved input global scale from smoothed calibration inputs."""
    observers = [layer.input_observer for layer in layers]
    scale = scale.detach().float().reshape(1, -1)
    # Discard pre-SmoothQuant statistics before replaying smoothed inputs.
    for observer in observers:
        for statistic in ("min_vals", "max_vals"):
            if hasattr(observer, statistic):
                delattr(observer, statistic)

    for inputs in batches:
        smoothed_inputs = inputs.detach().float() * scale.to(device=inputs.device)
        for observer in observers:
            observer(smoothed_inputs)

    update_qparams(layers, "input")


def quantize_mapping(
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
    weight_dtype = layers[0].weight.dtype
    device = layers[0].weight.device
    weight = torch.cat(
        [
            layer.weight.detach().to(device=device, dtype=torch.float32)
            for layer in layers
        ],
        dim=0,
    )
    cache.wait(mapping.mapping_name, weight.device)
    alpha_dtype = torch.bfloat16 if mapping.requires_runtime_smoothing else weight_dtype
    observer = layers[0].weight_observer
    input_batches = None if weight_only else cache.inputs[mapping.mapping_name]
    optimizer_batches = (
        input_batches
        if optimization_input_batches is None
        else optimization_input_batches
    )
    sample_count = cache.sample_count[mapping.mapping_name]
    if mode == "sic":
        result = optimize_sic(
            weight,
            optimizer_batches,
            cache.hessian[mapping.mapping_name].to(weight.device) / sample_count,
            observer=observer,
            alpha_dtype=alpha_dtype,
            weight_dtype=weight_dtype,
            percdamp=dampening_frac,
        )
        quantized_weight = result.quantized_weight
    else:
        sigma_x_squared = (
            cache.sigma_x_squared[mapping.mapping_name].to(weight.device) / sample_count
        )
        result = optimize_rtn(
            weight,
            optimizer_batches,
            observer=observer,
            alpha_dtype=alpha_dtype,
            sigma_x_squared=sigma_x_squared,
        )
        quantized_weight = None

    if _deployment_stage is not None:
        _deployment_stage("smoothing")
    scale = deploy_mapping_smooth_quant_scale(mapping, result.alpha_star)

    if not weight_only:
        if _deployment_stage is not None:
            _deployment_stage("input-scale replay")
        replay_inputs = cache.inputs[mapping.mapping_name]
        if mapping.mapping_name in cache.input_absmax:
            # max(abs(s * x)) = max(abs(s) * max_tokens(abs(x))). Replay
            # symmetric channel extrema using the original FP32 scale arithmetic.
            maximum = cache.input_absmax[mapping.mapping_name].cpu()
            replay_inputs = (torch.stack((-maximum, maximum)),)
        _update_smoothed_input_global_scale(
            layers,
            replay_inputs,
            scale,
        )
    if _deployment_stage is not None:
        _deployment_stage("parameter preparation")
    deployment: list[tuple[Module, dict[str, torch.Tensor]]] = []
    row_start = 0
    for layer in layers:
        row_end = row_start + layer.weight.shape[0]
        qparams: dict[str, torch.Tensor] = {}
        if quantized_weight is not None:
            qparams["weight"] = quantized_weight[row_start:row_end].to(layer.weight)
        qparams["weight_scale"] = result.gamma_w_star[row_start:row_end].to(
            layer.weight_scale
        )
        qparams["weight_zero_point"] = result.weight_zero_point[row_start:row_end].to(
            layer.weight_zero_point
        )
        qparams["weight_global_scale"] = torch.ones_like(layer.weight_global_scale)
        deployment.append((layer, qparams))
        row_start = row_end
    return deployment
