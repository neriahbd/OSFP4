#!/usr/bin/env python3
"""Prepare the tokenized FineWeb-Edu cache for FP-Quant Table 1."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from examples.fpquant.common.calibration import prepare_main  # noqa: E402
from examples.fpquant.table1.config import PROFILE  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(prepare_main(PROFILE))
