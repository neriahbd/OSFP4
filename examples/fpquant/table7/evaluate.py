#!/usr/bin/env python3
"""Evaluate one FP-Quant Table 7 method with Qwen thinking disabled."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from examples.fpquant.common.evaluation import main  # noqa: E402
from examples.fpquant.table7.config import PROFILE  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(PROFILE))
