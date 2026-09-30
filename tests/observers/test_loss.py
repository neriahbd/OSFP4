import inspect

import torch

from llmcompressor_osfp4.observers import loss
from llmcompressor_osfp4.observers.lut import phi_lut


def test_weight_loss_interface_has_no_sample_count():
    parameters = inspect.signature(loss.compute_weight_loss).parameters
    assert "sample_count" not in parameters


def test_loss_forward_and_gradients_match_direct_formula():
    generator = torch.Generator().manual_seed(31)
    weight = torch.randn(2, 3, 16, generator=generator)
    activations = torch.randn(2, 16, 5, generator=generator)
    sigma_x_squared = torch.rand(2, 16, generator=generator) + 0.1
    coefficients = loss.build_joint_loss_coefficients(
        weight,
        activations,
        weight_metric=sigma_x_squared,
        activation_reference=weight,
    )
    alpha = (torch.rand(2, 16, generator=generator) + 0.5).requires_grad_(True)
    gamma_w = (torch.rand(2, 3, 1, generator=generator) + 0.5).requires_grad_(True)
    gamma_x = (torch.rand(2, 1, 5, generator=generator) + 0.5).requires_grad_(True)

    total = loss.compute_joint_loss(
        weight,
        activations,
        alpha,
        gamma_w,
        gamma_x,
        coefficients,
    )

    expected_w = (
        torch.sum(
            coefficients.coeff_w * phi_lut(weight / (alpha.view(2, 1, 16) * gamma_w)),
            dim=(1, 2),
        )
        / 3
    )
    expected_x = (
        torch.sum(
            coefficients.coeff_x
            * phi_lut(alpha.view(2, 16, 1) * activations / gamma_x),
            dim=(1, 2),
        )
        / 15
    )
    assert isinstance(coefficients, loss.JointLossCoefficients)
    torch.testing.assert_close(total, expected_w + expected_x)

    gradients = torch.autograd.grad(total.sum(), (alpha, gamma_w, gamma_x))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


def test_separated_losses_are_bit_identical_to_joint_loss_and_gradients():
    generator = torch.Generator().manual_seed(32)
    weight = torch.randn(2, 3, 16, generator=generator)
    activations = torch.randn(2, 16, 5, generator=generator)
    metric = torch.rand(2, 16, generator=generator) + 0.1
    joint_coefficients = loss.build_joint_loss_coefficients(
        weight,
        activations,
        weight_metric=metric,
        activation_reference=weight,
    )
    weight_coefficients = loss.build_weight_loss_coefficients(
        weight,
        weight_metric=metric,
    )
    activation_coefficients = loss.build_activation_loss_coefficients(
        activations,
        activation_reference=weight,
    )
    assert torch.equal(joint_coefficients.coeff_w, weight_coefficients)
    assert torch.equal(joint_coefficients.coeff_x, activation_coefficients)

    joint_parameters = tuple(
        value.requires_grad_(True)
        for value in (
            torch.rand(2, 16, generator=generator) + 0.5,
            torch.rand(2, 3, 1, generator=generator) + 0.5,
            torch.rand(2, 1, 5, generator=generator) + 0.5,
        )
    )
    split_parameters = tuple(
        value.detach().clone().requires_grad_(True) for value in joint_parameters
    )
    joint = loss.compute_joint_loss(
        weight,
        activations,
        *joint_parameters,
        joint_coefficients,
    )
    split = loss.compute_weight_loss(
        weight,
        split_parameters[0],
        split_parameters[1],
        weight_coefficients,
    ) + loss.compute_activation_loss(
        activations,
        split_parameters[0],
        split_parameters[2],
        activation_coefficients,
        row_count=weight.shape[1],
    )
    assert torch.equal(joint, split)
    joint_gradients = torch.autograd.grad(joint.sum(), joint_parameters)
    split_gradients = torch.autograd.grad(split.sum(), split_parameters)
    for expected, actual in zip(joint_gradients, split_gradients):
        assert torch.equal(expected, actual)


def test_rtn_weight_and_activation_losses_use_the_same_normalization():
    generator = torch.Generator().manual_seed(34)
    weight = torch.randn(2, 7, 16, generator=generator)
    activations = weight.transpose(1, 2).contiguous()
    sigma_x_squared = activations.square().mean(dim=2)
    coefficients = loss.build_joint_loss_coefficients(
        weight,
        activations,
        weight_metric=sigma_x_squared,
        activation_reference=weight,
    )
    alpha = torch.ones(2, 16)
    gamma_w = torch.ones(2, 7, 1)
    gamma_x = torch.ones(2, 1, 7)

    loss_w = loss.compute_weight_loss(
        weight,
        alpha,
        gamma_w,
        coefficients.coeff_w,
    )
    loss_x = loss.compute_activation_loss(
        activations,
        alpha,
        gamma_x,
        coefficients.coeff_x,
        row_count=weight.shape[1],
    )

    torch.testing.assert_close(loss_w, loss_x)


def test_weight_loss_synthetic_values_equal_phi4_comparator_on_lut_plateau():
    values = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
    weight_target = torch.zeros(1, 1, 16)
    weight_target[0, 0, : values.numel()] = values
    weight_metric = torch.zeros(1, 16)
    weight_metric[0, : values.numel()] = 1.0
    coefficients = loss.build_weight_loss_coefficients(
        weight_target,
        weight_metric=weight_metric,
    )

    actual = loss.compute_weight_loss(
        weight_target,
        alpha=torch.ones(1, 16),
        gamma_w=torch.ones(1, 1, 1),
        coefficients=coefficients,
    )
    phi4 = phi_lut(torch.tensor(4.0))
    phi4_lower_bound = values.square().sum() * phi4

    torch.testing.assert_close(phi_lut(values), torch.full_like(values, phi4))
    torch.testing.assert_close(actual, phi4_lower_bound.reshape_as(actual))
    torch.testing.assert_close(actual, torch.tensor([0.9651118]))


def test_normalized_weight_loss_matches_raw_metric_formula_and_gradients():
    generator = torch.Generator().manual_seed(33)
    weight = torch.randn(2, 3, 16, generator=generator)
    raw_metric = torch.rand(2, 16, generator=generator) + 0.1
    sample_count = 11
    normalized_metric = raw_metric / sample_count
    normalized_coefficients = loss.build_weight_loss_coefficients(
        weight,
        weight_metric=normalized_metric,
    )
    raw_coefficients = loss.build_weight_loss_coefficients(
        weight,
        weight_metric=raw_metric,
    )
    normalized_alpha = (torch.rand(2, 16, generator=generator) + 0.5).requires_grad_(
        True
    )
    normalized_gamma = (torch.rand(2, 3, 1, generator=generator) + 0.5).requires_grad_(
        True
    )
    raw_alpha = normalized_alpha.detach().clone().requires_grad_(True)
    raw_gamma = normalized_gamma.detach().clone().requires_grad_(True)

    normalized = loss.compute_weight_loss(
        weight,
        normalized_alpha,
        normalized_gamma,
        normalized_coefficients,
    )
    raw_phi = phi_lut(weight / (raw_alpha.view(2, 1, 16) * raw_gamma))
    raw = torch.sum(raw_coefficients * raw_phi, dim=(1, 2)) / (
        weight.shape[1] * sample_count
    )

    torch.testing.assert_close(normalized, raw, atol=1e-6, rtol=1e-6)
    normalized_gradients = torch.autograd.grad(
        normalized.sum(),
        (normalized_alpha, normalized_gamma),
    )
    raw_gradients = torch.autograd.grad(raw.sum(), (raw_alpha, raw_gamma))
    for actual, expected in zip(normalized_gradients, raw_gradients):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
