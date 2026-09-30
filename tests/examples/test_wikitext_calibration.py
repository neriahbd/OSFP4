import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import datasets
import pytest

from examples.fpquant.common.calibration import CalibrationConfig
from examples.wikitext_ppl.config import CALIBRATION_SOURCE, PROFILE
from examples.wikitext_ppl.prepare_data import collect_wikitext_dataset

REPO_ROOT = Path(__file__).parents[2]


def _config(tmp_path):
    return CalibrationConfig(
        profile=PROFILE,
        model_id=PROFILE.default_model,
        seed=17,
        num_samples=3,
        max_sequence_length=4,
        calibration_data_root=tmp_path,
        source=CALIBRATION_SOURCE,
    )


def test_wikitext_source_is_immutable():
    with pytest.raises(FrozenInstanceError):
        CALIBRATION_SOURCE.dataset_name = "replacement"


def test_wikitext_manifest_bytes_match_existing_contract(tmp_path):
    manifest = (
        json.dumps(_config(tmp_path).data_manifest(), indent=2, sort_keys=True) + "\n"
    )

    assert manifest == (
        "{\n"
        '  "architecture": "LlamaForCausalLM",\n'
        '  "benchmark_profile": "wikitext-ppl",\n'
        '  "calibration_config": "wikitext-2-raw-v1",\n'
        '  "calibration_dataset": "Salesforce/wikitext",\n'
        '  "calibration_seed": 17,\n'
        '  "calibration_split": "train",\n'
        '  "max_sequence_length": 4,\n'
        '  "model": "meta-llama/Meta-Llama-3-8B",\n'
        '  "native_dtype": "bfloat16",\n'
        '  "num_calibration_samples": 3,\n'
        '  "shuffle_buffer_size": 0,\n'
        '  "tokenizer": "meta-llama/Meta-Llama-3-8B"\n'
        "}\n"
    )


def test_wikitext_sampling_matches_existing_sequence(tmp_path, monkeypatch):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return {"text": ["abcde", "fghij"]}

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    result = collect_wikitext_dataset(
        lambda text, **kwargs: {"input_ids": list(range(len(text)))},
        _config(tmp_path),
    )

    assert result["input_ids"] == [[8, 9, 10, 11], [6, 7, 8, 9], [4, 5, 6, 7]]
    assert calls == [
        (
            ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            {"split": "train"},
        )
    ]


@pytest.mark.parametrize("order", ["common-first", "wikitext-first"])
def test_import_order_does_not_change_fineweb_defaults(order):
    script = f"""
import json
from pathlib import Path
if {order!r} == "common-first":
    from examples.fpquant.common import calibration
    import examples.wikitext_ppl.prepare_data
else:
    import examples.wikitext_ppl.prepare_data
    from examples.fpquant.common import calibration
from examples.fpquant.table1.config import PROFILE
config = calibration.CalibrationConfig(
    profile=PROFILE,
    model_id=PROFILE.default_model,
    seed=17,
    num_samples=3,
    max_sequence_length=4,
    calibration_data_root=Path("/tmp/unused"),
)
print(json.dumps({{
    "collector": calibration.collect_dataset.__module__,
    "manifest": config.data_manifest(),
}}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(result.stdout)

    assert state["collector"] == "examples.fpquant.common.calibration"
    assert state["manifest"]["calibration_dataset"] == ("HuggingFaceFW/fineweb-edu")
    assert state["manifest"]["calibration_config"] == "sample-10BT"
    assert state["manifest"]["shuffle_buffer_size"] == 1_000
