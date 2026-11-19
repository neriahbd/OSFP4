# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

from collections.abc import Callable
from typing import NamedTuple

import torch
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA

__all__ = [
    "QuantizationGroupScaleValues",
    "LogScaleParameters",
    "create_log_scale_parameters",
    "initialize_scale_values_from_absmax",
    "materialize_scale_values",
    "canonicalize_log_scale_parameters",
    "run_optimizer",
    "run_chunked_optimizer",
    "select_blocks_per_chunk",
    "uses_exact_alpha_reduction",
]

_FP8_MAX = 448.0
_FP8_SAFETY_FACTOR = 2.0
_QUANTIZATION_GROUP_SIZE = 16
# Measured peak of one joint-loss forward/backward: about 15.3 FP32 copies of the
# activation blocks and 14 of the weight blocks; 16 leaves a margin.
_GRAPH_COPIES = 16
_MIN_SPLIT_BYTES = 256 * 1024 * 1024


# Scale representations


class QuantizationGroupScaleValues(NamedTuple):
    """Positive scales: alpha ``[L, 16]``, gamma_w ``[L, m, 1]``,
    and optional gamma_x ``[L, 1, k]``."""

    alpha: torch.Tensor
    gamma_w: torch.Tensor
    gamma_x: torch.Tensor | None


class LogScaleParameters(NamedTuple):
    """Adam parameters for ``QuantizationGroupScaleValues`` in log space."""

    log_alpha: torch.Tensor
    log_gamma_w: torch.Tensor
    log_gamma_x: torch.Tensor | None


# Initialization


def initialize_scale_values_from_absmax(
    W_blocks: torch.Tensor,
    X_blocks: torch.Tensor | None,
    *,
    blocks_per_chunk: int | None = None,
) -> QuantizationGroupScaleValues:
    """Initialize alpha, gamma_w, and optional gamma_x from block absolute maxima.

    ``blocks_per_chunk`` bounds temporary memory; results do not depend on it.
    """

    def normalize(
        values: torch.Tensor,
        target_absmax: float,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize each block and return its divisor."""
        divisors = torch.clamp(
            values.abs().amax(dim=(1, 2), keepdim=True) / target_absmax,
            min=eps,
        )
        return values / divisors, divisors

    def absmax_divisor(
        values: torch.Tensor,
        reduction_dim: int,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Return FP4 absolute-maximum divisors along one dimension."""
        maxima = torch.clamp(
            values.abs().amax(dim=reduction_dim, keepdim=True),
            min=eps,
        )
        return maxima / FP4_E2M1_DATA.max

    L, _, block_width = W_blocks.shape
    alpha = torch.ones(
        L,
        block_width,
        device=W_blocks.device,
        dtype=W_blocks.dtype,
    )
    target_absmax = FP4_E2M1_DATA.max * _FP8_MAX / _FP8_SAFETY_FACTOR
    # Every statistic is per block, so chunking over blocks bounds the normalized
    # temporaries without changing any value.
    gamma_w_chunks, gamma_x_chunks = [], []
    for chunk in _block_chunks(L, L if blocks_per_chunk is None else blocks_per_chunk):
        num_chunk_blocks = chunk.stop - chunk.start
        W_blocks_normalized, weight_divisor = normalize(
            W_blocks[chunk],
            target_absmax,
        )
        gamma_w_chunks.append(
            weight_divisor
            * absmax_divisor(
                W_blocks_normalized / alpha[chunk].view(num_chunk_blocks, 1, -1),
                reduction_dim=2,
            )
        )
        del W_blocks_normalized
        if X_blocks is not None:
            X_blocks_normalized, activation_divisor = normalize(
                X_blocks[chunk],
                target_absmax,
            )
            gamma_x_chunks.append(
                activation_divisor
                * absmax_divisor(
                    X_blocks_normalized * alpha[chunk].view(num_chunk_blocks, -1, 1),
                    reduction_dim=1,
                )
            )
            del X_blocks_normalized
    return QuantizationGroupScaleValues(
        alpha=alpha,
        gamma_w=_concat_block_chunks(gamma_w_chunks),
        gamma_x=None if X_blocks is None else _concat_block_chunks(gamma_x_chunks),
    )


def _concat_block_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    """Join per-chunk results along the block dimension."""
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


def create_log_scale_parameters(
    scales: QuantizationGroupScaleValues,
) -> LogScaleParameters:
    """Create trainable log tensors from initialized scales."""
    return LogScaleParameters(
        scales.alpha.log().detach().clone().requires_grad_(True),
        scales.gamma_w.log().detach().clone().requires_grad_(True),
        None
        if scales.gamma_x is None
        else scales.gamma_x.log().detach().clone().requires_grad_(True),
    )


# Adam


def run_optimizer(
    log_scales: LogScaleParameters,
    loss_evaluator: Callable[[], torch.Tensor],
    *,
    steps: int,
    lr: float,
) -> None:
    """Optimize log scales."""
    optimizer = torch.optim.Adam(
        [parameter for parameter in log_scales if parameter is not None],
        lr=lr,
    )
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            loss = loss_evaluator()
            loss.backward()
            optimizer.step()


# Block-chunked Adam


def _block_chunks(num_blocks: int, blocks_per_chunk: int) -> list[slice]:
    """Split ``range(num_blocks)`` into consecutive slices."""
    if blocks_per_chunk < 1:
        raise ValueError("blocks_per_chunk must be positive")
    return [
        slice(start, min(start + blocks_per_chunk, num_blocks))
        for start in range(0, num_blocks, blocks_per_chunk)
    ]


def select_blocks_per_chunk(m: int, k: int, num_blocks: int) -> int:
    """Return the number of quantization groups per chunk for loss evaluation.

    One chunk's autograd graph holds about ``_GRAPH_COPIES`` FP32 copies of its
    ``W_blocks [c, m, 16]`` and ``X_blocks [c, 16, k]`` slices (``k = 0`` for
    weight-only). Chunking by ``num_blocks // _GRAPH_COPIES`` keeps that workspace
    near one full copy, so scale optimization holds about three copies in total:
    the blocks, their loss coefficients, and the workspace. Problems whose whole
    graph fits within ``_MIN_SPLIT_BYTES`` are not split.

    The result depends only on the shape, never on free device memory. Every
    problem that may be split uses ``uses_exact_alpha_reduction``, so optimized
    scales do not depend on the chunk size.
    """
    if not _exceeds_split_threshold(m, k, num_blocks):
        return num_blocks
    return max(1, num_blocks // _GRAPH_COPIES)


def uses_exact_alpha_reduction(m: int, k: int, num_blocks: int) -> bool:
    """Return whether alpha's gradient must be reduced in FP64 for this shape.

    True for exactly the problems ``select_blocks_per_chunk`` may split. Their
    FP32 reduction order would depend on the chunk size; FP64 accumulation makes
    chunked and whole-tensor optimization agree bit for bit. Smaller problems are
    never chunked and keep the original FP32 arithmetic (and reference fixtures).
    """
    return _exceeds_split_threshold(m, k, num_blocks)


def _exceeds_split_threshold(m: int, k: int, num_blocks: int) -> bool:
    """Whether one whole-problem autograd graph exceeds ``_MIN_SPLIT_BYTES``."""
    full_graph_bytes = (
        _GRAPH_COPIES
        * num_blocks
        * _QUANTIZATION_GROUP_SIZE
        * (m + k)
        * torch.finfo(torch.float32).bits
        // 8
    )
    return full_graph_bytes > _MIN_SPLIT_BYTES


def run_chunked_optimizer(
    log_scales: LogScaleParameters,
    chunk_loss: Callable[[slice, LogScaleParameters], torch.Tensor],
    *,
    steps: int,
    lr: float,
    num_blocks: int,
    blocks_per_chunk: int,
) -> None:
    """Optimize log scales, evaluating and differentiating one block chunk at a time.

    Each block's loss depends only on that block's parameters and Adam is
    elementwise, so accumulating gradients chunk by chunk reproduces the
    whole-tensor step while keeping only one chunk's autograd graph alive.
    ``chunk_loss(blocks, chunk_log_scales)`` must return the summed loss of the
    selected blocks, computed from ``chunk_log_scales`` (leaf slices of
    ``log_scales``).
    """
    parameters = [parameter for parameter in log_scales if parameter is not None]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    chunks = _block_chunks(num_blocks, blocks_per_chunk)
    with torch.enable_grad():
        for _ in range(steps):
            for parameter in parameters:
                # Every element is written by exactly one chunk below.
                parameter.grad = torch.empty_like(parameter)
            for chunk in chunks:
                chunk_log_scales = LogScaleParameters(
                    *(
                        None
                        if parameter is None
                        else parameter.detach()[chunk].requires_grad_(True)
                        for parameter in log_scales
                    )
                )
                leaves = [leaf for leaf in chunk_log_scales if leaf is not None]
                # Copy instead of accumulating so signed zeros match a full backward.
                gradients = torch.autograd.grad(
                    chunk_loss(chunk, chunk_log_scales), leaves
                )
                for parameter, gradient in zip(parameters, gradients):
                    parameter.grad[chunk].copy_(gradient)
                del chunk_log_scales, leaves, gradients
            optimizer.step()
    for parameter in parameters:
        parameter.grad = None


# Canonicalization


def canonicalize_log_scale_parameters(log_scales: LogScaleParameters) -> None:
    """Center log_gamma_w and fold its mean into log_alpha in place."""
    with torch.no_grad():
        mean_log_gamma_w = log_scales.log_gamma_w.mean(dim=(1, 2), keepdim=True)
        log_scales.log_alpha.add_(mean_log_gamma_w.squeeze(-1))
        log_scales.log_gamma_w.sub_(mean_log_gamma_w)
        if log_scales.log_gamma_x is not None:
            log_scales.log_gamma_x.add_(mean_log_gamma_w)


# Materialization


def materialize_scale_values(
    log_scales: LogScaleParameters,
) -> QuantizationGroupScaleValues:
    """Return detached positive scales without mutating the optimized parameters."""
    return QuantizationGroupScaleValues(
        alpha=log_scales.log_alpha.detach().exp(),
        gamma_w=log_scales.log_gamma_w.detach().exp(),
        gamma_x=(
            None
            if log_scales.log_gamma_x is None
            else log_scales.log_gamma_x.detach().exp()
        ),
    )
