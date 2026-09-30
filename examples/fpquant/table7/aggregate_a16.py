#!/usr/bin/env python3
"""Aggregate BF16, OSFP4A16-RTN, and OSFP4A16-SIC into a Table 7 report.

Sibling of aggregate.py for the weight-only NVFP4A16 scheme. Reuses the
already-computed BF16 row from the NVFP4 (W4A4) run, since the BF16 baseline
is scheme-independent -- no separate BF16 evaluation is required for this
report.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from examples.fpquant.table7.config import PROFILE  # noqa: E402

MODEL_ID = PROFILE.default_model
TASK_COLUMNS = (
    ("MMLU", "mmlu_cot_llama"),
    ("GSM8K", "gsm8k_llama"),
    ("HellaSwag", "hellaswag"),
    ("WinoGrande", "winogrande"),
)
METHODS = (
    ("BF16", "bf16", None),
    ("OSFP4A16-RTN", "osfp4-rtn-a16", "rtn"),
    ("OSFP4A16-SIC", "osfp4-sic-a16", "sic"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(PROFILE.run_root))
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing required result: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def load_method(run_root: Path, label: str, slug: str) -> dict[str, Any]:
    path = run_root / "eval" / slug / "method-summary.json"
    summary = read_object(path)
    if summary.get("benchmark_profile") != "qwen-table7":
        raise RuntimeError(f"{path} is not a qwen-table7 result")
    expected_summary = {
        "method": slug,
        "model": MODEL_ID
        if slug == "bf16"
        else str(run_root / slug / "seed-42" / "model"),
        "dtype": "bfloat16",
    }
    summary_mismatches = {
        key: (summary.get(key), value)
        for key, value in expected_summary.items()
        if summary.get(key) != value
    }
    if summary_mismatches:
        raise RuntimeError(f"method summary mismatch in {path}: {summary_mismatches}")
    if summary.get("enable_thinking") is not False:
        raise RuntimeError(f"{path} did not disable Qwen thinking")
    scores = summary.get("scores")
    if not isinstance(scores, dict):
        raise RuntimeError(f"{path} has no scores object")

    ordered_scores: dict[str, float] = {}
    for display_name, task_name in TASK_COLUMNS:
        value = scores.get(task_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError(f"{path} has no numeric score for {task_name}")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise RuntimeError(f"{path} has invalid score for {task_name}: {value}")
        ordered_scores[display_name] = value

    average = sum(ordered_scores.values()) / len(ordered_scores)
    recorded_average = summary.get("average")
    if not isinstance(recorded_average, (int, float)) or not math.isclose(
        float(recorded_average), average, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"{path} average does not match its unrounded task scores: "
            f"{recorded_average!r} != {average}"
        )
    return {
        "format": "-" if slug == "bf16" else "OSFP4A16",
        "method": label,
        "scores": ordered_scores,
        "average": average,
        "source": str(path),
    }


def validate_checkpoint_mode(run_root: Path, slug: str, expected_mode: str) -> None:
    path = run_root / slug / "seed-42" / "model" / "calibration-manifest.json"
    manifest = read_object(path)
    recipe = manifest.get("recipe")
    actual_mode = recipe.get("optimization_mode") if isinstance(recipe, dict) else None
    actual_scheme = recipe.get("scheme") if isinstance(recipe, dict) else None
    expected = {
        "benchmark_profile": "qwen-table7",
        "model": MODEL_ID,
        "architecture": "Qwen3ForCausalLM",
        "native_dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "method": slug,
        "calibration_dataset": "HuggingFaceFW/fineweb-edu",
        "calibration_config": "sample-10BT",
        "calibration_seed": 42,
        "num_calibration_samples": 1024,
        "max_sequence_length": 2048,
        "shuffle_buffer_size": 1000,
        "tokenizer": MODEL_ID,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if actual_mode != expected_mode:
        mismatches["recipe.optimization_mode"] = (actual_mode, expected_mode)
    if actual_scheme != "NVFP4A16":
        mismatches["recipe.scheme"] = (actual_scheme, "NVFP4A16")
    if mismatches:
        raise RuntimeError(f"checkpoint manifest mismatch in {path}: {mismatches}")


def build_report(run_root: Path) -> dict[str, Any]:
    rows = [load_method(run_root, label, slug) for label, slug, _ in METHODS]
    baseline_average = rows[0]["average"]
    if baseline_average <= 0.0:
        raise RuntimeError("measured BF16 average must be positive")
    for row in rows:
        row["recovery_percent"] = 100.0 * row["average"] / baseline_average

    for _label, slug, mode in METHODS:
        if mode is not None:
            validate_checkpoint_mode(run_root, slug, mode)

    return {
        "benchmark": "Qwen3-8B Table 7 OSFP4A16 (weight-only) comparison",
        "model": MODEL_ID,
        "measured_baseline": "native BF16 (reused from the NVFP4 W4A4 run)",
        "recovery_definition": "100 * method average / measured BF16 average",
        "measured_rows": rows,
    }


def tabular_rows(report: dict[str, Any]):
    for row in report["measured_rows"]:
        yield {
            "section": "measured",
            "format": row["format"],
            "method": row["method"],
            **row["scores"],
            "Avg.": row["average"],
            "Recovery%": row["recovery_percent"],
        }


def csv_text(report: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    columns = (
        "section",
        "format",
        "method",
        "MMLU",
        "GSM8K",
        "HellaSwag",
        "WinoGrande",
        "Avg.",
        "Recovery%",
    )
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in tabular_rows(report):
        writer.writerow(
            {key: "" if value is None else value for key, value in row.items()}
        )
    return stream.getvalue()


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Format | Method | MMLU | GSM8K | HellaSwag | WinoGrande | "
        "Avg. | Recovery% |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        scores = row["scores"]
        recovery = row["recovery_percent"]
        recovery_text = "–" if recovery is None else f"{recovery:.2f}"
        lines.append(
            f"| {row['format']} | {row['method']} | {scores['MMLU']:.2f} | "
            f"{scores['GSM8K']:.2f} | {scores['HellaSwag']:.2f} | "
            f"{scores['WinoGrande']:.2f} | {row['average']:.2f} | "
            f"{recovery_text} |"
        )
    return "\n".join(lines)


def markdown_text(report: dict[str, Any]) -> str:
    return (
        "# Qwen3-8B OSFP4A16 Table 7\n\n"
        "## Measured locally\n\n"
        f"{markdown_table(report['measured_rows'])}\n\n"
        "Recovery is computed against the unrounded, measured BF16 average. "
        "The BF16 row is reused from the NVFP4 (W4A4) run -- the baseline is "
        "scheme-independent.\n"
    )


def write_report(run_root: Path, report: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "table7-osfp4-a16.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (run_root / "table7-osfp4-a16.csv").write_text(csv_text(report))
    (run_root / "table7-osfp4-a16.md").write_text(markdown_text(report))


def main() -> int:
    args = parse_args()
    report = build_report(args.run_root)
    write_report(args.run_root, report)
    print(f"Wrote Table 7 (NVFP4A16) reports under {args.run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
