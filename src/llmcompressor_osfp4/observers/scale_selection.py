# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

import torch
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA

_QUANTIZATION_GROUP_SIZE = 16
_BETA_MIN = 0.3
_BETA_MAX = 1.2
# At most 17 scales in the beta interval, plus two boundary neighbors.
_E4M3_MAX_CANDIDATES = 19
_E4M3_SEARCH_PRIMARY_TENSOR_BYTES = 64 * 1024 * 1024
_E4M3_GRID_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_e4m3_scale_grid(
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return C_scale, the positive finite E4M3 values in ascending order."""
    device = torch.device(device)
    key = (device, dtype)
    cached = _E4M3_GRID_CACHE.get(key)
    if cached is None:
        bits = torch.arange(1, 127, dtype=torch.uint8)
        cached = bits.view(torch.float8_e4m3fn).to(device=device, dtype=dtype)
        _E4M3_GRID_CACHE[key] = cached
    return cached


def _search_e4m3_gamma_star(
    W_target_blocks: torch.Tensor,
    alpha: torch.Tensor,
    gamma_w: torch.Tensor,
    weight_metric: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select E4M3 weight scales that minimize weighted reconstruction error."""
    # Prepare weights, scales, and error weights for scoring in float32.
    L, m, _ = W_target_blocks.shape
    W_target_blocks = W_target_blocks.detach().to(dtype=torch.float32)
    alpha = alpha.detach().to(device=W_target_blocks.device, dtype=torch.float32)
    gamma_w = gamma_w.detach().to(device=W_target_blocks.device, dtype=torch.float32)
    weight_metric = weight_metric.detach().to(
        device=W_target_blocks.device,
        dtype=torch.float32,
    )

    C_scale = _get_e4m3_scale_grid(W_target_blocks.device, W_target_blocks.dtype)

    # Determine each row's allowed scale interval.
    total_rows = L * m
    gamma_w_flat = gamma_w.reshape(total_rows)
    gamma_lower = gamma_w_flat / _BETA_MAX
    gamma_upper = gamma_w_flat / _BETA_MIN

    # Include neighboring scales for boundary tolerance and fallback.
    candidate_starts = (
        torch.searchsorted(C_scale, gamma_lower, right=False)
        .sub(1)
        .clamp(0, C_scale.numel())
    )
    candidate_ends = (
        torch.searchsorted(C_scale, gamma_upper, right=True)
        .add(1)
        .clamp(0, C_scale.numel())
    )

    # Limit the number of rows scored together.
    bytes_per_row = (
        _E4M3_MAX_CANDIDATES
        * _QUANTIZATION_GROUP_SIZE
        * torch.tensor([], dtype=W_target_blocks.dtype).element_size()
    )
    rows_per_tile = max(
        1,
        _E4M3_SEARCH_PRIMARY_TENSOR_BYTES // bytes_per_row,
    )
    offsets = torch.arange(
        _E4M3_MAX_CANDIDATES,
        device=W_target_blocks.device,
    )
    W_target_flat = W_target_blocks.reshape(
        total_rows,
        _QUANTIZATION_GROUP_SIZE,
    )
    gamma_w_final_flat = torch.empty(
        total_rows,
        device=W_target_blocks.device,
        dtype=W_target_blocks.dtype,
    )
    invalid_rows = torch.empty(
        total_rows,
        device=W_target_blocks.device,
        dtype=torch.bool,
    )
    beta_min = torch.tensor(_BETA_MIN, device=W_target_blocks.device)
    beta_max = torch.tensor(_BETA_MAX, device=W_target_blocks.device)
    fallback_used = torch.zeros((), device=W_target_blocks.device, dtype=torch.bool)

    for row_start in range(0, total_rows, rows_per_tile):
        row_end = min(row_start + rows_per_tile, total_rows)

        # Collect candidate scales for these rows.
        tile_starts = candidate_starts[row_start:row_end]
        tile_ends = candidate_ends[row_start:row_end]
        indices = tile_starts[:, None] + offsets[None, :]
        within_slice = indices < tile_ends[:, None]
        indices = indices.clamp(max=C_scale.numel() - 1)
        gamma_candidates = C_scale[indices]
        # Accept candidates within the beta bounds, including boundary tolerance.
        beta = gamma_w_flat[row_start:row_end, None] / gamma_candidates
        valid = (
            within_slice
            & ((beta >= beta_min) | torch.isclose(beta, beta_min))
            & ((beta <= beta_max) | torch.isclose(beta, beta_max))
        )

        # Use the nearest E4M3 boundary scale for out-of-range intervals only when
        # gamma_w is positive and finite.
        no_valid_candidates = ~valid.any(dim=-1)
        gamma_w_tile = gamma_w_flat[row_start:row_end]
        eligible = (
            no_valid_candidates & torch.isfinite(gamma_w_tile) & (gamma_w_tile > 0)
        )
        below = eligible & (gamma_upper[row_start:row_end] < C_scale[0])
        above = eligible & (gamma_lower[row_start:row_end] > C_scale[-1])
        valid |= below[:, None] & (gamma_candidates == C_scale[0])
        valid |= above[:, None] & (gamma_candidates == C_scale[-1])
        fallback_used |= (below | above).any()

        # Quantize and reconstruct the target weights with each candidate scale.
        block_indices = (
            torch.arange(row_start, row_end, device=W_target_blocks.device) // m
        )
        tile_alpha = alpha[block_indices]
        tile_metric = weight_metric[block_indices]
        denominator = tile_alpha[:, None, :] * gamma_candidates[:, :, None]
        W_target_tile = W_target_flat[row_start:row_end, None, :]
        quantized = FP4_E2M1_DATA.cast_to_fp4(W_target_tile / denominator)
        reconstructed = denominator * quantized

        # Measure reconstruction error using the weight metric.
        candidate_errors = (
            (W_target_tile - reconstructed).square() * tile_metric[:, None, :]
        ).sum(dim=-1)
        candidate_errors.masked_fill_(~valid, torch.inf)

        # Choose the lowest-error valid scale; ties favor the smaller scale.
        winning_errors, winning_offsets = candidate_errors.min(dim=-1)
        gamma_w_final_tile = gamma_candidates.gather(
            1, winning_offsets[:, None]
        ).squeeze(1)
        gamma_w_final_flat[row_start:row_end] = gamma_w_final_tile
        invalid_rows[row_start:row_end] = ~torch.isfinite(winning_errors)

    return (
        gamma_w_final_flat.view(L, m, 1),
        invalid_rows,
        fallback_used,
    )
