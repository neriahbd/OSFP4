import csv
import json

import pytest

from examples.fpquant.table7 import aggregate as table7


def _write_method(run_root, slug, scores):
    output_dir = run_root / "eval" / slug
    output_dir.mkdir(parents=True)
    average = sum(scores.values()) / len(scores)
    (output_dir / "method-summary.json").write_text(
        json.dumps(
            {
                "benchmark_profile": "qwen-table7",
                "method": slug,
                "model": (
                    "Qwen/Qwen3-8B"
                    if slug == "bf16"
                    else str(run_root / slug / "seed-42" / "model")
                ),
                "dtype": "bfloat16",
                "enable_thinking": False,
                "scores": scores,
                "average": average,
            }
        )
    )


def _write_checkpoint_manifest(run_root, slug, mode):
    model_dir = run_root / slug / "seed-42" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "calibration-manifest.json").write_text(
        json.dumps(
            {
                "benchmark_profile": "qwen-table7",
                "model": "Qwen/Qwen3-8B",
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
                "tokenizer": "Qwen/Qwen3-8B",
                "recipe": {"optimization_mode": mode},
            }
        )
    )


def test_table7_report_uses_unrounded_local_bf16_recovery_and_separate_reference(
    tmp_path,
):
    task_names = [task_name for _display, task_name in table7.TASK_COLUMNS]
    bf16 = dict(zip(task_names, (72.981, 90.909, 75.527, 70.568)))
    rtn = dict(zip(task_names, (70.781, 90.307, 74.639, 70.729)))
    sic = dict(zip(task_names, (71.111, 90.117, 74.889, 70.409)))
    _write_method(tmp_path, "bf16", bf16)
    _write_method(tmp_path, "osfp4-rtn", rtn)
    _write_method(tmp_path, "osfp4-sic", sic)
    _write_checkpoint_manifest(tmp_path, "osfp4-rtn", "rtn")
    _write_checkpoint_manifest(tmp_path, "osfp4-sic", "sic")

    report = table7.build_report(tmp_path)
    table7.write_report(tmp_path, report)

    baseline_average = sum(bf16.values()) / 4
    rtn_average = sum(rtn.values()) / 4
    assert report["measured_rows"][0]["recovery_percent"] == 100.0
    assert report["measured_rows"][1]["recovery_percent"] == pytest.approx(
        100.0 * rtn_average / baseline_average
    )
    assert report["paper_reference"]["rows"][0]["method"] == "FP16"
    assert report["paper_reference"]["rows"][1]["method"] == "RTN"
    assert len(report["paper_reference"]["rows"]) == 9

    assert (tmp_path / "table7-osfp4.json").is_file()
    assert "## Measured locally" in (tmp_path / "table7-osfp4.md").read_text()
    assert "## Paper reference" in (tmp_path / "table7-osfp4.md").read_text()
    with (tmp_path / "table7-osfp4.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["section"] for row in rows[:3]] == ["measured"] * 3
    assert all(row["section"] == "paper_reference" for row in rows[3:])


def test_table7_report_rejects_wrong_checkpoint_mode(tmp_path):
    scores = {
        "mmlu_cot_llama": 70.0,
        "gsm8k_llama": 80.0,
        "hellaswag": 75.0,
        "winogrande": 72.0,
    }
    for slug in ("bf16", "osfp4-rtn", "osfp4-sic"):
        _write_method(tmp_path, slug, scores)
    _write_checkpoint_manifest(tmp_path, "osfp4-rtn", "sic")
    _write_checkpoint_manifest(tmp_path, "osfp4-sic", "sic")

    with pytest.raises(RuntimeError, match="recipe.optimization_mode"):
        table7.build_report(tmp_path)
