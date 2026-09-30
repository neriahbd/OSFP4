from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from compressed_tensors.offload import update_offload_parameter
from torch.nn import Linear, Module, Parameter

if TYPE_CHECKING:
    from .base import OSFP4Mapping

__all__ = ["deploy_mapping_smooth_quant_scale"]


def deploy_mapping_smooth_quant_scale(
    mapping: OSFP4Mapping,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Deploy smooth-layer or runtime smoothing and return the FP32 scale."""
    balance_layers = mapping.balance_layers
    scale = scale.reshape(-1).to(device=balance_layers[0].weight.device)
    if mapping.requires_runtime_smoothing:
        return _deploy_runtime_smoothing(balance_layers[0], scale).detach().float()
    return _deploy_smooth_layer_smoothing(mapping.smooth_layer, balance_layers, scale)


def _deploy_smooth_layer_smoothing(
    smooth_layer: Module,
    balance_layers: tuple[Linear, ...],
    scale: torch.Tensor,
) -> torch.Tensor:
    """Balance weights and fold the scale into the preceding smooth layer."""
    input_width = balance_layers[0].weight.shape[1]
    # Round through the first balance layer's dtype before updating any layer.
    deployed_scale = scale.to(balance_layers[0].weight).float()
    for layer in balance_layers:
        scale = deployed_scale.to(layer.weight)
        weight = layer.weight.detach().clone()
        weight[:, :input_width] /= scale.view(1, -1)
        update_offload_parameter(layer, "weight", weight)

    scale = deployed_scale.to(smooth_layer.weight)
    weight = smooth_layer.weight.detach().clone()
    weight[:input_width] *= scale
    update_offload_parameter(smooth_layer, "weight", weight)

    if getattr(smooth_layer, "bias", None) is not None:
        bias = smooth_layer.bias.detach().clone()
        bias[:input_width] *= scale
        update_offload_parameter(smooth_layer, "bias", bias)
    return deployed_scale


def _deploy_runtime_smoothing(layer: Linear, scale: torch.Tensor) -> Parameter:
    """Deploy and return a canonical BF16 runtime smoothing scale."""
    if "smooth_quant_scale" in layer._parameters:
        raise RuntimeError("smooth_quant_scale is already registered on balance layer")

    deployed_scale = scale.detach().float().to(torch.bfloat16)
    weight = layer.weight.detach().clone()
    weight /= deployed_scale.float().to(weight).view(1, -1)
    update_offload_parameter(layer, "weight", weight)

    deployed_scale = Parameter(
        deployed_scale.to(device=layer.weight.device),
        requires_grad=False,
    )
    layer.register_parameter("smooth_quant_scale", deployed_scale)
    return deployed_scale
