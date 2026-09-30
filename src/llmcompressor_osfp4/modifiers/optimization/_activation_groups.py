"""Shared activation quantization-group assembly for OSFP4."""

from __future__ import annotations

import torch

__all__ = ["assemble_activation_quantization_groups"]

_QUANTIZATION_GROUP_SIZE = 16


def assemble_activation_quantization_groups(
    cached_activation_batches: list[torch.Tensor] | tuple[torch.Tensor, ...],
    quantization_group_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return contiguous FP32 activation groups ``[G, 16, samples]``."""
    sample_count = sum(
        batch.numel() // (quantization_group_count * _QUANTIZATION_GROUP_SIZE)
        for batch in cached_activation_batches
    )
    groups = torch.empty(
        quantization_group_count,
        _QUANTIZATION_GROUP_SIZE,
        sample_count,
        device=device,
        dtype=torch.float32,
    )
    start = 0
    for activations in cached_activation_batches:
        converted = (
            activations.detach()
            .to(
                device=device,
                dtype=torch.float32,
                non_blocking=(
                    activations.device.type == "cpu"
                    and device.type == "cuda"
                    and activations.is_pinned()
                ),
            )
            .reshape(-1, quantization_group_count, _QUANTIZATION_GROUP_SIZE)
            .permute(1, 2, 0)
        )
        end = start + converted.shape[-1]
        groups[:, :, start:end].copy_(converted)
        start = end
        # Release this transfer before allocating the next batch's FP32 buffer.
        del converted
    return groups
