"""Independent pre-refactor formulas; equality includes signed zeros and NaNs."""

import pytest
import torch

from llmcompressor_osfp4.observers import loss, lut, scale_selection


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def reference_interpolation(pos, starts, slopes):
    # Retain the pre-refactor clamp/index/fraction formula independently.
    cells = starts.shape[0]
    pos = pos.clamp(0.0, float(cells - 1))
    idx = pos.detach().to(torch.int32)
    frac = pos - idx.to(pos.dtype)
    flat = idx.reshape(-1)
    start = torch.index_select(starts, 0, flat).view_as(frac)
    slope = torch.index_select(slopes, 0, flat).view_as(frac)
    return start + frac * slope


def reference_phi(value):
    starts, slopes = lut._get_lut_interpolation_data(value.device, value.dtype)
    entries = starts.shape[0] + 1
    magnitudes = value.abs().clamp(min=2.0**lut._LUT_LOG2_MIN)
    pos = (torch.log2(magnitudes) - lut._LUT_LOG2_MIN) / lut._LUT_LOG2_STEP
    pos = pos.clamp(min=0.0, max=float(entries - 2))
    return reference_interpolation(pos, starts, slopes)


def reference_joint(target, activations, alpha, gamma_w, gamma_x, coefficients):
    # This is the old inline joint expression, not a call to the split losses.
    groups, rows, _ = target.shape
    phi_w = reference_phi(target / (alpha.view(groups, 1, -1) * gamma_w))
    weighted_w = coefficients.coeff_w * phi_w
    loss_w = torch.sum(weighted_w, dim=(1, 2)) / rows
    phi_x = reference_phi(alpha.view(groups, -1, 1) * activations / gamma_x)
    samples = activations.shape[2]
    weighted_x = coefficients.coeff_x * phi_x
    loss_x = torch.sum(weighted_x, dim=(1, 2)) / (rows * samples)
    return loss_w + loss_x


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_joint_coefficients_loss_and_gradients_match_pre_refactor(dtype, noncontiguous):
    generator = torch.Generator().manual_seed(32)
    weight = torch.randn(2, 3, 16, generator=generator).to(dtype)
    activations = torch.randn(2, 16, 5, generator=generator).to(dtype)
    if noncontiguous:
        weight = weight.transpose(1, 2).contiguous().transpose(1, 2)
        activations = activations.transpose(1, 2).contiguous().transpose(1, 2)
    metric = (torch.rand(2, 16, generator=generator) + 0.1).to(dtype)
    snapshots = [value.clone() for value in (weight, activations, metric)]
    coefficients = loss.build_joint_loss_coefficients(
        weight, activations, weight_metric=metric, activation_reference=weight
    )
    coeff_w = weight.square()
    coeff_w.mul_(metric.view(2, 1, 16))
    coeff_x = activations.square()
    energy = weight.square().sum(dim=1)
    coeff_x.mul_(energy.view(2, 16, 1))
    assert_bytes(coefficients.coeff_w, coeff_w)
    assert_bytes(coefficients.coeff_x, coeff_x)
    parameters = tuple(
        (torch.rand(shape, generator=generator) + 0.5).to(dtype).requires_grad_()
        for shape in [(2, 16), (2, 3, 1), (2, 1, 5)]
    )
    expected_parameters = tuple(p.detach().clone().requires_grad_() for p in parameters)
    actual = loss.compute_joint_loss(weight, activations, *parameters, coefficients)
    expected = reference_joint(weight, activations, *expected_parameters, coefficients)
    assert_bytes(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), parameters),
        torch.autograd.grad(expected.sum(), expected_parameters),
    ):
        assert_bytes(actual_grad, expected_grad)
    for value, snapshot in zip((weight, activations, metric), snapshots):
        assert_bytes(value, snapshot)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_phi_boundaries_values_gradients_and_cache_reuse(dtype):
    lut._PHI_LUT_CACHE.clear()
    lut._LUT_INTERPOLATION_CACHE.clear()
    bounds = torch.tensor(
        [
            0.0,
            2.0**-4,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            256.0 if dtype == torch.float32 else 128.0,
        ],
        dtype=dtype,
    )
    values = torch.cat(
        (
            bounds,
            torch.nextafter(bounds, torch.full_like(bounds, float("inf"))),
            torch.nextafter(bounds, torch.full_like(bounds, -float("inf"))),
        )
    )
    values = torch.stack((values, -values)).t().requires_grad_()
    expected_values = values.detach().clone().requires_grad_()
    actual = lut.phi_lut(values)
    expected = reference_phi(expected_values)
    assert_bytes(actual, expected)
    assert_bytes(
        torch.autograd.grad(actual.sum(), values)[0],
        torch.autograd.grad(expected.sum(), expected_values)[0],
    )
    table = lut._get_phi_lut("cpu", dtype)
    interpolation = lut._get_lut_interpolation_data("cpu", dtype)
    assert lut._get_phi_lut(torch.device("cpu"), dtype) is table
    assert lut._get_lut_interpolation_data("cpu", dtype)[0] is interpolation[0]
    assert_bytes(lut.phi_lut(values), actual)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_interpolation_clamps_and_fraction_gradients(dtype):
    starts = torch.tensor([0.1, 0.3, -0.2, 0.8], dtype=dtype)
    slopes = torch.tensor([0.2, -0.5, 1.0, 0.4], dtype=dtype)
    positions = (
        torch.tensor([[-1.0, -0.0, 0.0, 0.25], [1.0, 1.5, 3.0, 4.0]], dtype=dtype)
        .t()
        .requires_grad_()
    )
    reference_positions = positions.detach().clone().requires_grad_()
    actual = lut._interpolate_lut_at_position(positions, starts, slopes)
    expected = reference_interpolation(reference_positions, starts, slopes)
    assert_bytes(actual, expected)
    assert_bytes(
        torch.autograd.grad(actual.sum(), positions)[0],
        torch.autograd.grad(expected.sum(), reference_positions)[0],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_every_cached_e4m3_grid_entry_is_finite_positive_and_ordered(dtype):
    grid = scale_selection._get_e4m3_scale_grid("cpu", dtype)
    assert grid.numel() == 126
    assert torch.isfinite(grid).all()
    assert (grid > 0).all()
    assert (grid[1:] > grid[:-1]).all()
    assert scale_selection._get_e4m3_scale_grid(torch.device("cpu"), dtype) is grid


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_preexisting_low_precision_lut_upper_bound_failure_is_unchanged(dtype):
    # Normal calibration calls the LUT in FP32. Direct low-precision indexing
    # rounds the last cell position past the table; this refactor does not fix it.
    value = torch.tensor([256.0], dtype=dtype)
    for evaluate in (reference_phi, lut.phi_lut):
        with pytest.raises(IndexError):
            evaluate(value)
