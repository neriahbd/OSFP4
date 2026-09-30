import inspect

import pytest
import torch

from llmcompressor_osfp4.modifiers import optimization
from llmcompressor_osfp4.observers import scale_selection
from llmcompressor_osfp4.observers.fp4 import (
    quantize_e2m1,
)


def test_public_optimization_api_contains_only_canonical_entrypoints():
    assert set(optimization.__all__) == {
        "RTNQuantizationResult",
        "SICQuantizationResult",
        "optimize_rtn",
        "optimize_sic",
    }


def exhaustive_fp8_scale_search(weight, alpha, gamma_w, sigma_x_squared):
    grid = scale_selection._get_e4m3_scale_grid(weight.device, torch.float32)
    denominator = alpha[:, None, None, :] * grid[None, None, :, None]
    original = weight[:, :, None, :]
    quantized = quantize_e2m1(original / denominator)
    reconstructed = denominator * quantized
    candidate_errors = (
        (original - reconstructed).square() * sigma_x_squared[:, None, None, :]
    ).sum(dim=-1)
    beta = gamma_w / grid.view(1, 1, -1)
    valid = (
        (beta >= scale_selection._BETA_MIN)
        | torch.isclose(
            beta,
            torch.tensor(scale_selection._BETA_MIN),
        )
    ) & (
        (beta <= scale_selection._BETA_MAX)
        | torch.isclose(
            beta,
            torch.tensor(scale_selection._BETA_MAX),
        )
    )
    candidate_errors.masked_fill_(~valid, torch.inf)
    _, index = candidate_errors.min(dim=-1)
    return grid[index].unsqueeze(-1)


def fp8_scale_search_problem():
    torch.manual_seed(4)
    weight = torch.randn(3, 5, 16)
    alpha = torch.rand(3, 16) + 0.5
    gamma_w = torch.rand(3, 5, 1) * 2 + 0.2
    sigma_x_squared = torch.rand(3, 16) + 0.1
    return weight, alpha, gamma_w, sigma_x_squared


def test_fp8_scale_search_matches_full_126_value_grid():
    problem = fp8_scale_search_problem()
    selected, invalid, fallback = scale_selection._search_e4m3_gamma_star(*problem)
    expected = exhaustive_fp8_scale_search(*problem)
    assert not invalid.any()
    assert torch.equal(selected, expected)
    assert not fallback


def test_fixed_candidate_window_covers_every_beta_interval_transition():
    grid = scale_selection._get_e4m3_scale_grid("cpu", torch.float32)
    transitions = torch.cat(
        (
            torch.zeros(1),
            grid * scale_selection._BETA_MIN,
            grid * scale_selection._BETA_MAX,
        )
    ).unique(sorted=True)
    midpoints = (transitions[:-1] + transitions[1:]) / 2
    gamma = torch.cat((transitions, midpoints, grid[-1:] * 2))
    starts = (
        torch.searchsorted(
            grid,
            gamma / scale_selection._BETA_MAX,
            right=False,
        )
        .sub(1)
        .clamp(0, grid.numel())
    )
    ends = (
        torch.searchsorted(
            grid,
            gamma / scale_selection._BETA_MIN,
            right=True,
        )
        .add(1)
        .clamp(0, grid.numel())
    )
    assert int((ends - starts).max()) <= scale_selection._E4M3_MAX_CANDIDATES


def test_fp8_scale_search_internal_tiling_matches(monkeypatch):
    problem = fp8_scale_search_problem()
    expected, expected_invalid, fallback = scale_selection._search_e4m3_gamma_star(
        *problem
    )
    monkeypatch.setattr(
        scale_selection,
        "_E4M3_SEARCH_PRIMARY_TENSOR_BYTES",
        16 * 4,
    )
    tiled, tiled_invalid, fallback = scale_selection._search_e4m3_gamma_star(*problem)
    assert torch.equal(tiled_invalid, expected_invalid)
    assert torch.equal(tiled, expected)


def test_full_mapping_scale_search_matches_repeated_batch_searches():
    problem = fp8_scale_search_problem()
    full_scales, full_invalid, fallback = scale_selection._search_e4m3_gamma_star(
        *problem
    )
    batch_results = [
        scale_selection._search_e4m3_gamma_star(
            *(
                value[quantization_group_start:quantization_group_end]
                for value in problem
            )
        )
        for quantization_group_start, quantization_group_end in ((0, 1), (1, 3))
    ]

    assert torch.equal(
        full_scales,
        torch.cat([scales for scales, _, _ in batch_results]),
    )
    assert torch.equal(
        full_invalid,
        torch.cat([invalid for _, invalid, _ in batch_results]),
    )


def test_fp8_scale_search_tie_chooses_smallest_valid_scale():
    weight = torch.zeros(1, 1, 16)
    alpha = torch.ones(1, 16)
    gamma_w = torch.ones(1, 1, 1)
    sigma_x_squared = torch.zeros(1, 16)
    selected, invalid, fallback = scale_selection._search_e4m3_gamma_star(
        weight, alpha, gamma_w, sigma_x_squared
    )
    expected = exhaustive_fp8_scale_search(weight, alpha, gamma_w, sigma_x_squared)
    assert not invalid.any()
    assert torch.equal(selected, expected)
    assert not fallback


@pytest.mark.parametrize("gamma,expected", [(1e-20, 2**-9), (1e20, 448.0)])
def test_fp8_scale_search_clips_empty_candidate_sets(gamma, expected):
    weight = torch.ones(1, 1, 16)
    alpha = torch.ones(1, 16)
    gamma_w = torch.full((1, 1, 1), gamma)
    sigma_x_squared = torch.ones(1, 16)
    selected, invalid, fallback = scale_selection._search_e4m3_gamma_star(
        weight,
        alpha,
        gamma_w,
        sigma_x_squared,
    )

    assert not invalid.any()
    assert selected.item() == expected
    assert fallback


@pytest.mark.parametrize("gamma", [0.0, -1.0, float("nan"), float("inf")])
def test_fp8_scale_search_does_not_hide_invalid_scales(gamma):
    _, invalid, fallback = scale_selection._search_e4m3_gamma_star(
        torch.ones(1, 1, 16),
        torch.ones(1, 16),
        torch.full((1, 1, 1), gamma),
        torch.ones(1, 16),
    )
    assert invalid.all()


def test_fp8_scale_search_fallback_does_not_hide_nonfinite_errors():
    _, invalid, fallback = scale_selection._search_e4m3_gamma_star(
        torch.full((1, 1, 16), float("nan")),
        torch.ones(1, 16),
        torch.full((1, 1, 1), 1e20),
        torch.ones(1, 16),
    )
    assert invalid.all()


@pytest.mark.parametrize(
    "beta_bound,outside_beta,seed",
    [
        (
            scale_selection._BETA_MIN,
            scale_selection._BETA_MIN - 1e-3,
            4,
        ),
        (
            scale_selection._BETA_MAX,
            scale_selection._BETA_MAX + 1e-3,
            10,
        ),
    ],
)
def test_fp8_scale_search_beta_bound_is_inclusive(
    beta_bound,
    outside_beta,
    seed,
):
    grid = scale_selection._get_e4m3_scale_grid("cpu", torch.float32)
    target = grid[10]
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn(16, generator=generator) * target * 2.3).view(1, 1, 16)
    alpha = torch.ones(1, 16)
    sigma_x_squared = torch.ones(1, 16)

    boundary, boundary_invalid, fallback = scale_selection._search_e4m3_gamma_star(
        weight,
        alpha,
        (target * beta_bound).view(1, 1, 1),
        sigma_x_squared,
    )
    outside, outside_invalid, fallback = scale_selection._search_e4m3_gamma_star(
        weight,
        alpha,
        (target * outside_beta).view(1, 1, 1),
        sigma_x_squared,
    )

    assert not boundary_invalid.any()
    assert not outside_invalid.any()
    assert torch.equal(boundary, target.view(1, 1, 1))
    assert not torch.equal(outside, target.view(1, 1, 1))


def test_beta_bounds_are_not_public_parameters():
    assert all(
        "beta" not in name
        for name in inspect.signature(
            scale_selection._search_e4m3_gamma_star
        ).parameters
    )
