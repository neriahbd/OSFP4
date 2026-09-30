#!/usr/bin/env python3
"""Prepare the tokenized WikiText-2 calibration cache.

Reuses the FP-Quant common calibration pipeline (`fpquant/common/calibration.py`)
but points it at the released NestQuant code's WikiText-2 train source instead
of the default streaming FineWeb-Edu, so the resulting checkpoints are
comparable to `evaluate.py`'s NestQuant WikiText-2 perplexity protocol.

The shared pipeline's `collect_dataset()` skips any record shorter than
`max_sequence_length` and windows within a single record -- correct for
FineWeb-Edu's long documents, but WikiText-2's raw `train` split is
line-based (mean ~300 characters, well under one 2048-token window), so
every record gets skipped and it collects zero samples. This module
overrides `collect_dataset` with a WikiText-2-appropriate implementation
that joins the whole split into one token stream first, then samples
random contiguous windows *with replacement* -- matching the
`"random_contiguous_windows_with_replacement"` method recorded in the
prior (pre-reorg) WikiText-2 calibration cache's manifest.
"""

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.fpquant.common import calibration  # noqa: E402
from examples.wikitext_ppl.config import (  # noqa: E402
    CALIBRATION_SOURCE,
    PROFILE,
)


def collect_wikitext_dataset(tokenizer: Any, config: Any) -> Any:
    """Sample random contiguous windows from the whole joined WikiText-2 split."""
    import random

    from datasets import Dataset, load_dataset

    rows = load_dataset(
        config.source.dataset_name,
        config.source.dataset_config,
        split=config.source.dataset_split,
    )
    joined_text = "\n\n".join(rows["text"])
    input_ids = tokenizer(joined_text, add_special_tokens=False)["input_ids"]
    if len(input_ids) < config.max_sequence_length:
        raise RuntimeError(
            f"joined WikiText-2 {config.source.dataset_split} split has only "
            f"{len(input_ids)} tokens, need at least {config.max_sequence_length}"
        )
    rng = random.Random(config.seed)
    span = len(input_ids) - config.max_sequence_length
    samples = [
        {"input_ids": input_ids[start : start + config.max_sequence_length]}
        for start in (rng.randint(0, span) for _ in range(config.num_samples))
    ]
    return Dataset.from_list(samples)


if __name__ == "__main__":
    raise SystemExit(
        calibration.prepare_main(
            PROFILE,
            source=CALIBRATION_SOURCE,
            collector=collect_wikitext_dataset,
        )
    )
