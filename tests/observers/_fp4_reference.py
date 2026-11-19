import torch

__all__ = ["FP4_ABSMAX", "quantize_e2m1"]

FP4_ABSMAX = 6.0


def quantize_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Round to the canonical NVFP4 E2M1 values without mutating the input."""
    magnitude = values.abs()
    quantized = torch.full_like(magnitude, FP4_ABSMAX)
    quantized = torch.where(magnitude <= 5.0, 4.0, quantized)
    quantized = torch.where(magnitude < 3.5, 3.0, quantized)
    quantized = torch.where(magnitude <= 2.5, 2.0, quantized)
    quantized = torch.where(magnitude < 1.75, 1.5, quantized)
    quantized = torch.where(magnitude <= 1.25, 1.0, quantized)
    quantized = torch.where(magnitude < 0.75, 0.5, quantized)
    quantized = torch.where(magnitude <= 0.25, 0.0, quantized)
    return quantized * values.sign()
