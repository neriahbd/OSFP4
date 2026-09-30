import torch

from .fp4 import quantize_e2m1

_QUANTIZATION_GROUP_SIZE = 16
_BETA_MIN = 0.3
_BETA_MAX = 1.2
_E4M3_MAX_CANDIDATES = 19
_E4M3_SEARCH_PRIMARY_TENSOR_BYTES = 64 * 1024 * 1024
_E4M3_GRID_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_e4m3_scale_grid(
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return every positive finite E4M3 value in ascending order."""
    device = torch.device(device)
    key = (device, dtype)
    cached = _E4M3_GRID_CACHE.get(key)
    if cached is None:
        bits = torch.arange(1, 127, dtype=torch.uint8)
        cached = bits.view(torch.float8_e4m3fn).to(device=device, dtype=dtype)
        _E4M3_GRID_CACHE[key] = cached
    return cached


def _search_e4m3_gamma_star(
    target: torch.Tensor,
    alpha: torch.Tensor,
    gamma_w: torch.Tensor,
    error_metric: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select E4M3 scales, clipping empty beta intervals to a finite boundary."""
    # Prepare full-mapping tensors.
    groups, rows, _ = target.shape
    target = target.detach().to(dtype=torch.float32)
    alpha = alpha.detach().to(device=target.device, dtype=torch.float32)
    gamma_w = gamma_w.detach().to(device=target.device, dtype=torch.float32)
    error_metric = error_metric.detach().to(
        device=target.device,
        dtype=torch.float32,
    )
    grid = _get_e4m3_scale_grid(target.device, target.dtype)

    # Bound candidates and tile the scoring tensors.
    total_rows = groups * rows
    gamma_flat = gamma_w.reshape(total_rows)
    lower = gamma_flat / _BETA_MAX
    upper = gamma_flat / _BETA_MIN
    starts = torch.searchsorted(grid, lower, right=False).sub(1).clamp(0, grid.numel())
    ends = torch.searchsorted(grid, upper, right=True).add(1).clamp(0, grid.numel())
    bytes_per_row = (
        _E4M3_MAX_CANDIDATES
        * _QUANTIZATION_GROUP_SIZE
        * torch.tensor([], dtype=target.dtype).element_size()
    )
    rows_per_tile = max(
        1,
        _E4M3_SEARCH_PRIMARY_TENSOR_BYTES // bytes_per_row,
    )
    offsets = torch.arange(
        _E4M3_MAX_CANDIDATES,
        device=target.device,
    )
    target_flat = target.reshape(
        total_rows,
        _QUANTIZATION_GROUP_SIZE,
    )
    best_scales = torch.empty(
        total_rows,
        device=target.device,
        dtype=target.dtype,
    )
    invalid_rows = torch.empty(
        total_rows,
        device=target.device,
        dtype=torch.bool,
    )
    beta_min = torch.tensor(_BETA_MIN, device=target.device)
    beta_max = torch.tensor(_BETA_MAX, device=target.device)
    fallback_used = torch.zeros((), device=target.device, dtype=torch.bool)

    for start in range(
        0,
        total_rows,
        rows_per_tile,
    ):
        end = min(
            start + rows_per_tile,
            total_rows,
        )
        tile_starts = starts[start:end]
        tile_ends = ends[start:end]
        indices = tile_starts[:, None] + offsets[None, :]
        within_slice = indices < tile_ends[:, None]
        indices = indices.clamp(max=grid.numel() - 1)
        candidates = grid[indices]
        beta = gamma_flat[start:end, None] / candidates
        valid = (
            within_slice
            & ((beta >= beta_min) | torch.isclose(beta, beta_min))
            & ((beta <= beta_max) | torch.isclose(beta, beta_max))
        )
        # Relax the beta constraint only for empty intervals outside the grid.
        # Invalid learned scales and nonfinite reconstruction errors still fail.
        empty = ~valid.any(dim=-1)
        tile_gamma = gamma_flat[start:end]
        eligible = empty & torch.isfinite(tile_gamma) & (tile_gamma > 0)
        below = eligible & (upper[start:end] < grid[0])
        above = eligible & (lower[start:end] > grid[-1])
        valid |= below[:, None] & (candidates == grid[0])
        valid |= above[:, None] & (candidates == grid[-1])
        fallback_used |= (below | above).any()

        # Score in candidate order; min keeps the first tie.
        # Flattened rows are group-major. Gather only this tile's repeated values.
        group_indices = torch.arange(start, end, device=target.device) // rows
        tile_alpha = alpha[group_indices]
        tile_metric = error_metric[group_indices]
        denominator = tile_alpha[:, None, :] * candidates[:, :, None]
        original = target_flat[start:end, None, :]
        quantized = quantize_e2m1(original / denominator)
        reconstructed = denominator * quantized
        errors = ((original - reconstructed).square() * tile_metric[:, None, :]).sum(
            dim=-1
        )
        errors.masked_fill_(~valid, torch.inf)
        winning_errors, winning_offsets = errors.min(dim=-1)
        tile_scales = candidates.gather(1, winning_offsets[:, None]).squeeze(1)
        best_scales[start:end] = tile_scales
        invalid_rows[start:end] = ~torch.isfinite(winning_errors)

    # Restore the group/row layout.
    return (
        best_scales.view(groups, rows, 1),
        invalid_rows,
        fallback_used,
    )
