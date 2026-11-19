# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Token counting and deterministic activation subsampling for OSFP4 optimization."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

__all__ = [
    "ActivationSubsample",
    "count_calibration_tokens",
    "subsample_activations",
]

_ACTIVATION_SUBSAMPLE_SEED = 42


@dataclass(frozen=True, slots=True)
class ActivationSubsample:
    """Optimization-only activation batches and their provenance."""

    batches: tuple[torch.Tensor, ...] | None
    provenance: dict[str, int | str]


def count_calibration_tokens(dataloader) -> int | None:
    """Count collated token rows without model forwards or advancing RNGs."""
    if (
        dataloader is None
        or isinstance(dataloader, Iterator)
        or isinstance(getattr(dataloader, "dataset", None), IterableDataset)
    ):
        return None
    # Persistent worker RNGs cannot be restored from the main process.
    if getattr(dataloader, "persistent_workers", False):
        return None
    generators = {
        generator
        for owner in (dataloader, getattr(dataloader, "sampler", None))
        if isinstance(
            generator := getattr(owner, "generator", None), torch.Generator
        )
    }
    generator_states = {g: g.get_state() for g in generators}
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = (
        list(range(torch.get_device_module("cuda").device_count()))
        if torch.get_device_module("cuda").is_initialized()
        else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            total_num_tokens = 0
            for batch in dataloader:
                if not isinstance(batch, Mapping):
                    return None
                inputs = batch.get("input_ids")
                if isinstance(inputs, torch.Tensor):
                    total_num_tokens += inputs.numel()
                else:
                    inputs = batch.get("inputs_embeds")
                    if not isinstance(inputs, torch.Tensor):
                        return None
                    total_num_tokens += inputs.numel() // inputs.shape[-1]
            return total_num_tokens or None
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for generator, rng in generator_states.items():
            generator.set_state(rng)


def _select_activation_indices(
    total_num_tokens: int, activation_subsample_size: int
) -> torch.Tensor:
    """Select sorted activation indices without changing global RNG state."""
    num_sampled_tokens = min(total_num_tokens, activation_subsample_size)
    if num_sampled_tokens == total_num_tokens:
        return torch.arange(total_num_tokens, device="cpu", dtype=torch.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_ACTIVATION_SUBSAMPLE_SEED)
    sampled_indices = torch.randperm(
        total_num_tokens, generator=generator, device="cpu", dtype=torch.int64
    )[:num_sampled_tokens]
    return sampled_indices.sort().values


def _build_subsample_provenance(
    sampled_indices: torch.Tensor, total_num_tokens: int, output_rows: int
) -> dict[str, int | str]:
    """Describe the selected activation rows and their source."""
    return {
        "policy": "fixed",
        "seed": _ACTIVATION_SUBSAMPLE_SEED,
        "m": output_rows,
        "k": sampled_indices.numel(),
        "k1": total_num_tokens,
        "index_sha256": hashlib.sha256(sampled_indices.numpy().tobytes()).hexdigest(),
    }


def _gather_rows(
    batches: Sequence[torch.Tensor],
    sampled_indices: torch.Tensor,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    """Collect the selected activation rows from multiple batches into one tensor."""
    first_batch = batches[0]
    sampled_activations = torch.empty(
        (sampled_indices.numel(), first_batch.shape[1]),
        dtype=first_batch.dtype,
        device="cpu",
        pin_memory=pin_memory,
    )
    output_start = 0
    batch_token_offset = 0
    for batch in batches:
        batch_end = batch_token_offset + batch.shape[0]
        index_start = int(
            torch.searchsorted(sampled_indices, batch_token_offset, side="left").item()
        )
        index_end = int(
            torch.searchsorted(sampled_indices, batch_end, side="left").item()
        )
        if index_start != index_end:
            batch_sampled_indices = sampled_indices[index_start:index_end] - batch_token_offset
            output_end = output_start + batch_sampled_indices.numel()
            sampled_activations[output_start:output_end].copy_(
                batch.index_select(0, batch_sampled_indices)
            )
            output_start = output_end
        batch_token_offset = batch_end
    return sampled_activations


def subsample_activations(
    cached_activation_batches: Sequence[torch.Tensor],
    activation_subsample_size: int,
    *,
    output_rows: int,
) -> ActivationSubsample:
    """Select complete activation rows without changing global RNG state."""
    batches = cached_activation_batches
    dtype = batches[0].dtype
    total_num_tokens = 0
    all_pinned = True

    for batch in batches:
        if batch.dtype != dtype:
            raise ValueError("cached activation batches must share one dtype")
        total_num_tokens += batch.shape[0]
        all_pinned = all_pinned and batch.is_pinned()

    sampled_indices = _select_activation_indices(
        total_num_tokens, activation_subsample_size
    )
    if sampled_indices.numel() == total_num_tokens:
        sampled_batches = None  # Reuse all original batches without copying.
    else:
        sampled_batches = (
            _gather_rows(batches, sampled_indices, pin_memory=all_pinned),
        )

    return ActivationSubsample(
        batches=sampled_batches,
        provenance=_build_subsample_provenance(
            sampled_indices, total_num_tokens, output_rows
        ),
    )
