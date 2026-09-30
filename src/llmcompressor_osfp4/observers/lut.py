import numpy as np
import torch

__all__ = ["phi_lut"]


# Derive the positive FP4 levels and their decision boundaries.
_FP4_POS_NP = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=np.float32,
)
_FP4_BOUNDARIES_NP = 0.5 * (_FP4_POS_NP[:-1] + _FP4_POS_NP[1:])
_LN2 = float(np.log(2.0))
_LOG2_FP4_BOUNDARIES_NP = np.log2(_FP4_BOUNDARIES_NP)

# Table domain and caches

# Define the uniform log2 LUT grid.
_LUT_LOG2_MIN = -4.0
_LUT_LOG2_MAX = 8.0
_LUT_LOG2_STEP = 2.0**-12

# Cache the phi LUT and interpolation data by device and dtype.
_PHI_LUT_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_LUT_INTERPOLATION_CACHE: dict[
    tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
] = {}


# Table construction


def _g_antiderivative(t: np.ndarray, quantized_value: float) -> np.ndarray:
    """Evaluate the antiderivative within one constant FP4 region."""
    p1 = np.exp2(-t)
    return (
        t
        + (2.0 * quantized_value / _LN2) * p1
        - (quantized_value * quantized_value / (2.0 * _LN2)) * (p1 * p1)
    )


def _integrate_g_over_interval(start: float, end: float) -> float:
    """Integrate g across FP4 regions in log2 space."""
    result = 0.0
    edges = np.concatenate(([-np.inf], _LOG2_FP4_BOUNDARIES_NP, [np.inf]))
    for region, quantized_value in enumerate(_FP4_POS_NP):
        region_low = float(edges[region])
        region_high = float(edges[region + 1])
        low = max(start, region_low)
        high = min(end, region_high)

        if high <= low:
            continue

        if quantized_value == 0.0:
            result += high - low
        else:
            result += _g_antiderivative(high, quantized_value) - _g_antiderivative(
                low,
                quantized_value,
            )
    return result


def _phi(log2_x: np.ndarray) -> np.ndarray:
    """Compute phi as a sliding one-octave integral of g."""
    L = np.asarray(log2_x, dtype=np.float64)
    flat = L.ravel()
    out = np.array(
        [_integrate_g_over_interval(li - 1.0, li) for li in flat],
        dtype=np.float64,
    )
    return out.reshape(L.shape)


def _get_phi_lut(
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build and cache the phi table by device and dtype."""
    device = torch.device(device)
    key = (device, dtype)
    cached = _PHI_LUT_CACHE.get(key)
    if cached is not None:
        return cached

    n = int(round((_LUT_LOG2_MAX - _LUT_LOG2_MIN) / _LUT_LOG2_STEP)) + 1
    t_np = _LUT_LOG2_MIN + _LUT_LOG2_STEP * np.arange(n, dtype=np.float64)
    phi_np = _phi(t_np)

    table = torch.from_numpy(phi_np).to(dtype=dtype).to(device=device)
    _PHI_LUT_CACHE[key] = table
    return table


def _get_lut_interpolation_data(
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache each interpolation cell’s start value and slope."""
    device = torch.device(device)
    key = (device, dtype)
    cached = _LUT_INTERPOLATION_CACHE.get(key)
    if cached is not None:
        return cached

    table = _get_phi_lut(device, dtype)
    cell_start_values = table[:-1].contiguous()
    cell_slopes = (table[1:] - table[:-1]).contiguous()

    _LUT_INTERPOLATION_CACHE[key] = (cell_start_values, cell_slopes)
    return cell_start_values, cell_slopes


# Interpolation


def _interpolate_lut_at_position(
    pos: torch.Tensor,
    cell_start_values: torch.Tensor,
    cell_slopes: torch.Tensor,
) -> torch.Tensor:
    """Interpolate clamped positions with detached integer indices."""
    cells = cell_start_values.shape[0]
    pos = pos.clamp(0.0, float(cells - 1))
    idx = pos.detach().to(torch.int32)  # pos >= 0, so truncation == floor
    frac = pos - idx.to(pos.dtype)
    flat = idx.reshape(-1)
    cell_start = torch.index_select(cell_start_values, 0, flat).view_as(frac)
    cell_slope = torch.index_select(cell_slopes, 0, flat).view_as(frac)
    return cell_start + frac * cell_slope


def phi_lut(value: torch.Tensor) -> torch.Tensor:
    """Look up the dither-integrated FP4 relative error."""
    starts, slopes = _get_lut_interpolation_data(value.device, value.dtype)
    entries = starts.shape[0] + 1
    magnitudes = value.abs().clamp(min=2.0**_LUT_LOG2_MIN)
    pos = (torch.log2(magnitudes) - _LUT_LOG2_MIN) / _LUT_LOG2_STEP
    pos = pos.clamp(min=0.0, max=float(entries - 2))
    return _interpolate_lut_at_position(pos, starts, slopes)
