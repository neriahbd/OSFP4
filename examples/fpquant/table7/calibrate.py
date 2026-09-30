#!/usr/bin/env python3
"""Calibrate one FP-Quant Table 7 method."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from examples.fpquant.common.calibration import calibrate_main  # noqa: E402
from examples.fpquant.table7.config import PROFILE  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(calibrate_main(PROFILE))
