from collections.abc import Callable
from typing import NamedTuple

import torch

from .fp4 import FP4_ABSMAX

__all__ = [
    "QuantizationGroupScaleValues",
    "LogScaleParameters",
    "create_log_scale_parameters",
    "initialize_scale_values_from_absmax",
    "materialize_scale_values",
    "canonicalize_log_scale_parameters",
    "run_optimizer",
]

_FP8_MAX = 448.0
_FP8_SAFETY_FACTOR = 2.0


# Scale representations


class QuantizationGroupScaleValues(NamedTuple):
    """Positive scales with alpha ``[groups, 16]`` and per-group gammas."""

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
    target: torch.Tensor,
    activations: torch.Tensor | None,
) -> QuantizationGroupScaleValues:
    """Initialize alpha and gamma scales from quantization-group absolute maxima."""

    def normalize(
        values: torch.Tensor,
        target_absmax: float,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize each quantization group and return its divisor."""
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
        return maxima / FP4_ABSMAX

    groups, _, width = target.shape
    alpha = torch.ones(
        groups,
        width,
        device=target.device,
        dtype=target.dtype,
    )
    target_absmax = FP4_ABSMAX * _FP8_MAX / _FP8_SAFETY_FACTOR
    normalized_target, weight_divisor = normalize(
        target,
        target_absmax,
    )
    gamma_w = weight_divisor * absmax_divisor(
        normalized_target / alpha.view(groups, 1, -1),
        reduction_dim=2,
    )
    gamma_x = None
    if activations is not None:
        normalized_activations, activation_divisor = normalize(
            activations,
            target_absmax,
        )
        gamma_x = activation_divisor * absmax_divisor(
            normalized_activations * alpha.view(groups, -1, 1),
            reduction_dim=1,
        )
    return QuantizationGroupScaleValues(
        alpha=alpha,
        gamma_w=gamma_w,
        gamma_x=gamma_x,
    )


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
    parameters: LogScaleParameters,
    loss_evaluator: Callable[[], torch.Tensor],
    *,
    steps: int,
    lr: float,
) -> None:
    """Optimize log scales."""
    optimizer = torch.optim.Adam(
        [parameter for parameter in parameters if parameter is not None],
        lr=lr,
    )
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            loss = loss_evaluator()
            loss.backward()
            optimizer.step()


# Canonicalization


def canonicalize_log_scale_parameters(parameters: LogScaleParameters) -> None:
    """Center log weight gamma and fold its mean into alpha in place."""
    with torch.no_grad():
        shift = parameters.log_gamma_w.mean(dim=(1, 2), keepdim=True)
        parameters.log_alpha.add_(shift.squeeze(-1))
        parameters.log_gamma_w.sub_(shift)
        if parameters.log_gamma_x is not None:
            parameters.log_gamma_x.add_(shift)


# Materialization


def materialize_scale_values(
    parameters: LogScaleParameters,
) -> QuantizationGroupScaleValues:
    """Return detached positive scales without mutating the optimized parameters."""
    return QuantizationGroupScaleValues(
        alpha=parameters.log_alpha.detach().exp(),
        gamma_w=parameters.log_gamma_w.detach().exp(),
        gamma_x=(
            None
            if parameters.log_gamma_x is None
            else parameters.log_gamma_x.detach().exp()
        ),
    )
