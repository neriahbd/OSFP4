"""Internal OSFP4 optimization entry points and their result contracts."""

from .rtn import RTNQuantizationResult, optimize_rtn
from .sic import SICQuantizationResult, optimize_sic

__all__ = [
    "RTNQuantizationResult",
    "SICQuantizationResult",
    "optimize_rtn",
    "optimize_sic",
]
