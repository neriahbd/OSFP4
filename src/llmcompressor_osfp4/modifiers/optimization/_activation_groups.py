# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Shared activation block assembly for OSFP4."""

from __future__ import annotations

import torch

__all__ = ["assemble_activation_quantization_groups"]

_QUANTIZATION_GROUP_SIZE = 16


def assemble_activation_quantization_groups(
    cached_activation_batches: list[torch.Tensor] | tuple[torch.Tensor, ...],
    L: int,
    device: torch.device,
) -> torch.Tensor:
    """Assemble contiguous FP32 ``X_blocks [L, 16, k]``, preserving token order.
    L = n / 16 blocks; k is the number of supplied activation rows."""
    k = sum(
        batch.numel() // (L * _QUANTIZATION_GROUP_SIZE)
        for batch in cached_activation_batches
    )

    X_blocks = torch.empty(
        L,
        _QUANTIZATION_GROUP_SIZE,
        k,
        device=device,
        dtype=torch.float32,
    )

    token_start = 0
    for batch in cached_activation_batches:
        # Transfer in the cached dtype; copy_ widens to FP32 while permuting, so no
        # full-size FP32 temporary is created next to X_blocks.
        X_batch = batch.detach().to(
            device=device,
            dtype=batch.dtype,
            non_blocking=(
                batch.device.type == "cpu"
                and device.type == "cuda"
                and batch.is_pinned()
            ),
        )
        X_batch = X_batch.reshape(-1, L, _QUANTIZATION_GROUP_SIZE).permute(1, 2, 0)
        token_end = token_start + X_batch.shape[-1]
        X_blocks[:, :, token_start:token_end].copy_(X_batch)
        token_start = token_end
        del X_batch

    return X_blocks
