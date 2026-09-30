#!/usr/bin/env python3
"""Calibrate one OSFP4 method on the WikiText-2 train cache.

Run `prepare_data.py` first to build the cache this reads. See that script's
docstring for why the dataset differs from the FP-Quant table reproductions.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from examples.fpquant.common import calibration  # noqa: E402
from examples.wikitext_ppl.config import (  # noqa: E402
    CALIBRATION_SOURCE,
    PROFILE,
)

if __name__ == "__main__":
    raise SystemExit(calibration.calibrate_main(PROFILE, source=CALIBRATION_SOURCE))
