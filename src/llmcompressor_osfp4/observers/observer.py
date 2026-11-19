from typing import NamedTuple

import torch
from loguru import logger

from llmcompressor.observers.base import Observer
from .loss import (
    JointLossCoefficients,
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
    LogScaleParameters,
    run_chunked_optimizer,
    select_blocks_per_chunk,
    uses_exact_alpha_reduction,
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
    W_blocks: torch.Tensor,
    X_blocks: torch.Tensor,
    *,
    weight_metric: torch.Tensor,
    steps: int,
    lr: float,
) -> QuantizationGroupScaleValues:
    """Learn alpha, gamma_w, and gamma_x from W_blocks, X_blocks, and weight_metric."""
    W_blocks = W_blocks.detach().to(dtype=torch.float32)
    X_blocks = X_blocks.detach().to(
        device=W_blocks.device,
        dtype=torch.float32,
    )
    weight_metric = weight_metric.detach().to(
        device=W_blocks.device,
        dtype=torch.float32,
    )
    L, m, _ = W_blocks.shape
    blocks_per_chunk = select_blocks_per_chunk(m, X_blocks.shape[2], L)
    exact_alpha_reduction = uses_exact_alpha_reduction(m, X_blocks.shape[2], L)
    initial_scales = initialize_scale_values_from_absmax(
        W_blocks, X_blocks, blocks_per_chunk=blocks_per_chunk
    )
    coefficients = build_joint_loss_coefficients(
        W_blocks,
        X_blocks,
        weight_metric=weight_metric,
    )
    log_scales = create_log_scale_parameters(initial_scales)
    del initial_scales

    def chunk_loss(blocks: slice, chunk_log_scales: LogScaleParameters) -> torch.Tensor:
        """Evaluate the summed joint weight and activation loss of selected blocks."""
        return compute_joint_loss(
            W_blocks[blocks],
            X_blocks[blocks],
            chunk_log_scales.log_alpha.exp(),
            chunk_log_scales.log_gamma_w.exp(),
            chunk_log_scales.log_gamma_x.exp(),  # type: ignore[union-attr]
            JointLossCoefficients(
                coefficients.coeff_w[blocks], coefficients.coeff_x[blocks]
            ),
            exact_alpha_reduction=exact_alpha_reduction,
        ).sum()

    run_chunked_optimizer(
        log_scales,
        chunk_loss,
        steps=steps,
        lr=lr,
        num_blocks=L,
        blocks_per_chunk=blocks_per_chunk,
    )
    canonicalize_log_scale_parameters(log_scales)
    return materialize_scale_values(log_scales)


def _optimize_weight_scales(
    W_blocks: torch.Tensor,
    weight_metric: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> QuantizationGroupScaleValues:
    """Learn A16 alpha and gamma_w from W_blocks and weight_metric."""
    W_blocks = W_blocks.detach().to(dtype=torch.float32)
    weight_metric = weight_metric.detach().to(
        device=W_blocks.device,
        dtype=torch.float32,
    )
    L, m, _ = W_blocks.shape
    blocks_per_chunk = select_blocks_per_chunk(m, 0, L)
    exact_alpha_reduction = uses_exact_alpha_reduction(m, 0, L)
    initial_scales = initialize_scale_values_from_absmax(
        W_blocks, None, blocks_per_chunk=blocks_per_chunk
    )
    coefficients = build_weight_loss_coefficients(
        W_blocks,
        weight_metric=weight_metric,
    )
    log_scales = create_log_scale_parameters(initial_scales)
    del initial_scales

    def chunk_loss(blocks: slice, chunk_log_scales: LogScaleParameters) -> torch.Tensor:
        """Evaluate the summed weight-only loss of selected blocks."""
        return compute_weight_loss(
            W_blocks[blocks],
            chunk_log_scales.log_alpha.exp(),
            chunk_log_scales.log_gamma_w.exp(),
            coefficients[blocks],
            exact_alpha_reduction=exact_alpha_reduction,
        ).sum()

    run_chunked_optimizer(
        log_scales,
        chunk_loss,
        steps=steps,
        lr=lr,
        num_blocks=L,
        blocks_per_chunk=blocks_per_chunk,
    )
    canonicalize_log_scale_parameters(log_scales)
    return materialize_scale_values(log_scales)


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
