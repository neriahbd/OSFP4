from typing import NamedTuple

import torch

from .lut import phi_lut


class JointLossCoefficients(NamedTuple):
    """Invariant W4 ``[G, R, 16]`` and A4 ``[G, 16, S]`` loss coefficients."""

    coeff_w: torch.Tensor
    coeff_x: torch.Tensor


def build_weight_loss_coefficients(
    weight_target: torch.Tensor,
    *,
    weight_metric: torch.Tensor,
) -> torch.Tensor:
    """Build immutable ``Lw`` coefficients for ``[groups, rows, 16]`` weights."""
    groups, _, width = weight_target.shape
    with torch.no_grad():
        coefficients = weight_target.square()
        coefficients.mul_(
            weight_metric.view(
                groups,
                1,
                width,
            )
        )
    return coefficients


def build_activation_loss_coefficients(
    activation_quantization_groups: torch.Tensor,
    *,
    activation_reference: torch.Tensor,
) -> torch.Tensor:
    """Build immutable ``Lx`` coefficients for ``[groups, 16, samples]`` inputs."""
    groups, width, _ = activation_quantization_groups.shape
    with torch.no_grad():
        coefficients = activation_quantization_groups.square()
        column_energy = activation_reference.square().sum(dim=1)
        coefficients.mul_(
            column_energy.view(
                groups,
                width,
                1,
            )
        )
    return coefficients


def build_joint_loss_coefficients(
    target: torch.Tensor,
    activations: torch.Tensor,
    *,
    weight_metric: torch.Tensor,
    activation_reference: torch.Tensor,
) -> JointLossCoefficients:
    """Build immutable weight and activation coefficients for the joint loss."""
    return JointLossCoefficients(
        coeff_w=build_weight_loss_coefficients(
            target,
            weight_metric=weight_metric,
        ),
        coeff_x=build_activation_loss_coefficients(
            activations,
            activation_reference=activation_reference,
        ),
    )


def compute_weight_loss(
    weight_target: torch.Tensor,
    alpha: torch.Tensor,
    gamma_w: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Return ``Lw`` per group from normalized metric coefficients."""
    groups, row_count, _ = weight_target.shape
    phi_w = phi_lut(weight_target / (alpha.view(groups, 1, -1) * gamma_w))
    weighted_w = coefficients * phi_w
    return torch.sum(weighted_w, dim=(1, 2)) / row_count


def compute_activation_loss(
    activation_quantization_groups: torch.Tensor,
    alpha: torch.Tensor,
    gamma_x: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    row_count: int,
) -> torch.Tensor:
    """Return normalized ``Lx`` per group without updating scale tensors."""
    groups = activation_quantization_groups.shape[0]
    samples = activation_quantization_groups.shape[2]
    phi_x = phi_lut(
        alpha.view(groups, -1, 1) * activation_quantization_groups / gamma_x
    )
    return torch.sum(coefficients * phi_x, dim=(1, 2)) / (row_count * samples)


def compute_joint_loss(
    target: torch.Tensor,
    activations: torch.Tensor,
    alpha: torch.Tensor,
    gamma_w: torch.Tensor,
    gamma_x: torch.Tensor,
    coefficients: JointLossCoefficients,
) -> torch.Tensor:
    """Return one differentiable joint W4/A4 loss per quantization group."""
    loss_w = compute_weight_loss(target, alpha, gamma_w, coefficients.coeff_w)
    loss_x = compute_activation_loss(
        activations, alpha, gamma_x, coefficients.coeff_x, row_count=target.shape[1]
    )
    return loss_w + loss_x
