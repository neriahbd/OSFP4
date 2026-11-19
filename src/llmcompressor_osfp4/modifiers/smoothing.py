# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from compressed_tensors.offload import update_offload_parameter
from torch.nn import Linear, Module, Parameter

from llmcompressor.modifiers.quantization.calibration import update_qparams

from .calibration_cache import OSFP4CalibrationCache

if TYPE_CHECKING:
    from .base import OSFP4Mapping

__all__ = ["deploy_mapping_smooth_quant_scale", "update_smoothed_input_global_scale"]


def deploy_mapping_smooth_quant_scale(
    mapping: OSFP4Mapping,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Deploy smooth-layer or runtime smoothing and return the FP32 scale."""
    balance_layers = mapping.balance_layers
    scale = scale.reshape(-1).to(device=balance_layers[0].weight.device)

    if mapping.requires_runtime_smoothing:
        return _deploy_runtime_smoothing(balance_layers[0], scale)

    return _deploy_smooth_layer_smoothing(mapping.smooth_layer, balance_layers, scale)


def _deploy_smooth_layer_smoothing(
    smooth_layer: Module,
    balance_layers: tuple[Linear, ...],
    scale: torch.Tensor,
) -> torch.Tensor:
    """Balance weights and fold the scale into the preceding smooth layer."""
    input_width = balance_layers[0].weight.shape[1]
    deployed_scale = scale.to(balance_layers[0].weight).float()
    # Divide each balance layer's weight columns by the scale.
    for layer in balance_layers:
        scale = deployed_scale.to(layer.weight)
        weight = layer.weight.detach().clone()
        weight[:, :input_width] /= scale.view(1, -1)
        update_offload_parameter(layer, "weight", weight)

    # Multiply norm weights and bias (or Linear output rows and bias) by the scale.
    scale = deployed_scale.to(smooth_layer.weight)
    weight = smooth_layer.weight.detach().clone()
    weight[:input_width] *= scale if weight.ndim == 1 else scale.view(-1, 1)
    update_offload_parameter(smooth_layer, "weight", weight)

    if getattr(smooth_layer, "bias", None) is not None:
        bias = smooth_layer.bias.detach().clone()
        bias[:input_width] *= scale
        update_offload_parameter(smooth_layer, "bias", bias)

    return deployed_scale


def _deploy_runtime_smoothing(layer: Linear, scale: torch.Tensor) -> torch.Tensor:
    """Store the BF16 scale in smooth_quant_scale and return its FP32 values."""
    if "smooth_quant_scale" in layer._parameters:
        raise RuntimeError("smooth_quant_scale is already registered on balance layer")

    deployed_scale = scale.detach().float().to(torch.bfloat16)
    weight = layer.weight.detach().clone()
    weight /= deployed_scale.to(weight).view(1, -1)
    update_offload_parameter(layer, "weight", weight)

    deployed_scale = Parameter(
        deployed_scale.to(device=layer.weight.device),
        requires_grad=False,
    )
    layer.register_parameter("smooth_quant_scale", deployed_scale)
    return deployed_scale.detach().float()


def update_smoothed_input_global_scale(
    mapping: OSFP4Mapping,
    cache: OSFP4CalibrationCache,
    scale: torch.Tensor,
) -> None:
    """Update input global scales from smoothed calibration statistics."""
    # Replay reads pinned CPU buffers on the host; finish their D2H copies.
    cache.wait_for_capture(mapping.mapping_name, torch.device("cpu"))
    calibration_activations = cache.cached_activations[mapping.mapping_name]
    if mapping.mapping_name in cache.input_channel_absmax:
        # Recalibrate from one row of per-channel maxima collected over all tokens.
        input_channel_absmax = cache.input_channel_absmax[mapping.mapping_name].cpu()
        calibration_activations = (input_channel_absmax.unsqueeze(0),)

    balance_layers = mapping.balance_layers
    observers = [layer.input_observer for layer in balance_layers]
    scale = scale.detach().float().reshape(1, -1)

    # Clear statistics collected before applying the smoothing scale.
    for observer in observers:
        for statistic in ("min_vals", "max_vals"):
            if hasattr(observer, statistic):
                delattr(observer, statistic)

    for activations in calibration_activations:
        smoothed_activations = activations.detach().float() * scale.to(
            device=activations.device
        )
        for observer in observers:
            observer(smoothed_activations)

    update_qparams(balance_layers, "input")
