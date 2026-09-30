from typing import NamedTuple

import torch
from loguru import logger

from llmcompressor.observers.base import Observer
from .loss import (
    build_joint_loss_coefficients,
    build_weight_loss_coefficients,
    compute_joint_loss,
    compute_weight_loss,
)
from .scale_optimization import (
    QuantizationGroupScaleValues,
    canonicalize_log_scale_parameters,
    create_log_scale_parameters,
    initialize_scale_values_from_absmax,
    materialize_scale_values,
    run_optimizer,
)
from .scale_selection import (
    _search_e4m3_gamma_star,
)

# Scale optimization


class OptimizedQuantizationGroupScales(NamedTuple):
    """Deployment-rounded alpha and continuous weight gamma before E4M3 selection."""

    alpha_star: torch.Tensor
    gamma_w: torch.Tensor


def _optimize_joint_scales(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    weight_metric: torch.Tensor,
    steps: int,
    lr: float,
) -> QuantizationGroupScaleValues:
    """Learn joint weight and activation scales from an explicit weight metric."""
    weight = weight.detach().to(dtype=torch.float32)
    activations = activations.detach().to(
        device=weight.device,
        dtype=torch.float32,
    )
    weight_metric = weight_metric.detach().to(
        device=weight.device,
        dtype=torch.float32,
    )
    initial = initialize_scale_values_from_absmax(weight, activations)
    coefficients = build_joint_loss_coefficients(
        weight,
        activations,
        weight_metric=weight_metric,
        activation_reference=weight,
    )
    parameters = create_log_scale_parameters(initial)

    def loss() -> torch.Tensor:
        """Evaluate the summed joint weight and activation loss."""
        return compute_joint_loss(
            weight,
            activations,
            parameters.log_alpha.exp(),
            parameters.log_gamma_w.exp(),
            parameters.log_gamma_x.exp(),  # type: ignore[union-attr]
            coefficients,
        ).sum()

    run_optimizer(parameters, loss, steps=steps, lr=lr)
    canonicalize_log_scale_parameters(parameters)
    return materialize_scale_values(parameters)


def _optimize_weight_scales(
    weight_target: torch.Tensor,
    weight_metric: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> QuantizationGroupScaleValues:
    """Learn A16 alpha and weight gamma from normalized activation statistics."""
    weight_target = weight_target.detach().to(dtype=torch.float32)
    weight_metric = weight_metric.detach().to(
        device=weight_target.device,
        dtype=torch.float32,
    )
    initial = initialize_scale_values_from_absmax(weight_target, None)
    coefficients = build_weight_loss_coefficients(
        weight_target,
        weight_metric=weight_metric,
    )
    parameters = create_log_scale_parameters(initial)

    def loss() -> torch.Tensor:
        """Evaluate the summed weight-only loss."""
        return compute_weight_loss(
            weight_target,
            parameters.log_alpha.exp(),
            parameters.log_gamma_w.exp(),
            coefficients,
        ).sum()

    run_optimizer(parameters, loss, steps=steps, lr=lr)
    canonicalize_log_scale_parameters(parameters)
    return materialize_scale_values(parameters)


# Observer


@Observer.register("osfp4")
class OSFP4Observer(Observer):
    """Optimize alpha and return NVFP4 quantization-group qparams."""

    # Calibration statistics

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        """Update per-observation minimum and maximum statistics."""
        self.min_vals = torch.amin(observed, dim=(0, -1))
        self.max_vals = torch.amax(observed, dim=(0, -1))

    # Optimization and deployment rounding

    def optimize_quantization_group_scales(
        self,
        weight: torch.Tensor,
        *,
        activation_quantization_groups: torch.Tensor | None = None,
        weight_metric: torch.Tensor,
        alpha_dtype: torch.dtype,
    ) -> OptimizedQuantizationGroupScales:
        """Optimize continuous scales for independent OSFP4 quantization groups."""
        weight = weight.detach().to(dtype=torch.float32)
        weight_metric = weight_metric.detach().to(
            device=weight.device,
            dtype=torch.float32,
        )
        config = self.args.observer_kwargs
        if activation_quantization_groups is None:
            learned = _optimize_weight_scales(
                weight,
                weight_metric,
                steps=config["num_iters"],
                lr=config["lr"],
            )
        else:
            learned = _optimize_joint_scales(
                weight,
                activation_quantization_groups,
                weight_metric=weight_metric,
                steps=config["num_iters"],
                lr=config["lr"],
            )

        alpha_star = learned.alpha.to(dtype=alpha_dtype).float()
        valid = (
            torch.isfinite(alpha_star).all()
            & torch.all(alpha_star > 0)
            & torch.isfinite(learned.gamma_w).all()
            & torch.all(learned.gamma_w > 0)
        )
        if learned.gamma_x is not None:
            valid = (
                valid
                & torch.isfinite(learned.gamma_x).all()
                & torch.all(learned.gamma_x > 0)
            )
        if not valid:
            raise ValueError("OSFP4 observer produced invalid scales")

        return OptimizedQuantizationGroupScales(alpha_star, learned.gamma_w)

    # Checkpoint scale selection

    def select_weight_qparams(
        self,
        weight: torch.Tensor,
        optimized: OptimizedQuantizationGroupScales,
        *,
        weight_metric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select independent E4M3 qparams using the full-mapping metric."""
        scales, invalid_rows, fallback_used = _search_e4m3_gamma_star(
            weight,
            optimized.alpha_star,
            optimized.gamma_w,
            weight_metric,
        )

        # Read both device flags together at the existing validation boundary.
        invalid, used_fallback = torch.stack(
            (invalid_rows.any(), fallback_used)
        ).tolist()
        if used_fallback:
            logger.warning(
                "OSFP4: empty scale candidate set; applied boundary fallback."
            )
        if invalid:
            raise ValueError("OSFP4 observer produced invalid checkpoint scales")

        zero_points = torch.zeros_like(scales, dtype=self.args.zp_dtype)
        return scales, zero_points
