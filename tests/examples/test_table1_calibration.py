import json
from types import SimpleNamespace

import pytest
from datasets import Dataset

from examples.fpquant.common import calibration
from examples.fpquant.common.calibration import CalibrationConfig
from examples.fpquant.table1.config import PROFILE as TABLE1
from examples.fpquant.table7.config import PROFILE as TABLE7


@pytest.mark.parametrize(
    "method", ["osfp4-rtn", "osfp4-sic", "osfp4-rtn-a16", "osfp4-sic-a16"]
)
def test_recipe_manifest_reads_architecture_from_model_config(method):
    args = SimpleNamespace(
        method=method, steps=80, lr=0.12, activation_subsample_size=65536
    )
    _, manifest = calibration._build_recipe(args)
    targets = ["model.layers.0.mlp.down_proj"]
    model = SimpleNamespace(
        config=SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            osfp4_metadata={"version": 1, "smooth_quant_scale_targets": targets},
        )
    )

    result = manifest(model)

    assert result["osfp4_architecture"] == "LlamaForCausalLM"
    assert result["smooth_quant_scale_targets"] == targets
    assert result["scheme"] == ("NVFP4A16" if method.endswith("a16") else "NVFP4")


def _config(tmp_path, profile=TABLE1, *, samples=2, length=4):
    return CalibrationConfig(
        profile=profile,
        model_id=profile.default_model,
        seed=17,
        num_samples=samples,
        max_sequence_length=length,
        calibration_data_root=tmp_path,
    )


def _write_manifest(config):
    config.prepared_dataset_dir.mkdir(parents=True)
    (config.prepared_dataset_dir / "calibration-data-manifest.json").write_text(
        json.dumps(config.data_manifest())
    )


def test_prepared_dataset_requires_complete_manifest_contract(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_manifest(config)
    manifest_path = config.prepared_dataset_dir / "calibration-data-manifest.json"
    manifest = config.data_manifest()
    del manifest["tokenizer"]
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        calibration,
        "load_from_disk",
        lambda _path: Dataset.from_dict({"input_ids": [[1, 2, 3, 4]] * 2}),
    )

    with pytest.raises(RuntimeError, match="tokenizer"):
        calibration.validate_prepared_dataset(config)


@pytest.mark.parametrize(
    "dataset",
    [
        Dataset.from_dict({"input_ids": [[1, 2, 3, 4]]}),
        Dataset.from_dict({"input_ids": [[1, 2, 3], [1, 2, 3, 4]]}),
        Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3, 4]] * 2,
                "attention_mask": [[1, 1, 1, 1]] * 2,
            }
        ),
    ],
)
def test_prepared_dataset_validates_shape(tmp_path, monkeypatch, dataset):
    config = _config(tmp_path)
    _write_manifest(config)
    monkeypatch.setattr(calibration, "load_from_disk", lambda _path: dataset)

    with pytest.raises(RuntimeError):
        calibration.validate_prepared_dataset(config)


def test_streaming_crops_are_deterministic(tmp_path, monkeypatch):
    config = _config(tmp_path, samples=3, length=4)

    class Source:
        def __init__(self):
            self.shuffle_calls = []

        def shuffle(self, **kwargs):
            self.shuffle_calls.append(kwargs)
            return self

        def __iter__(self):
            return iter([{"text": "3"}, {"text": "9"}, {"text": "10"}, {"text": "8"}])

    sources = []

    def fake_load_dataset(*_args, **_kwargs):
        source = Source()
        sources.append(source)
        return source

    monkeypatch.setattr(calibration, "load_dataset", fake_load_dataset)

    def tokenizer(text, **_kwargs):
        return {"input_ids": list(range(int(text)))}

    first = calibration.collect_dataset(tokenizer, config)
    second = calibration.collect_dataset(tokenizer, config)

    assert first["input_ids"] == second["input_ids"]
    assert all(
        source.shuffle_calls == [{"seed": 17, "buffer_size": 1_000}]
        for source in sources
    )
    assert all(len(ids) == 4 for ids in first["input_ids"])


def test_table_profiles_are_native_bf16_and_have_separate_caches():
    assert TABLE1.calibration_dtype == TABLE1.evaluation_dtype == "bfloat16"
    assert TABLE7.calibration_dtype == TABLE7.evaluation_dtype == "bfloat16"
    assert TABLE1.paper_baseline_label == TABLE7.paper_baseline_label == "FP16"
    assert TABLE1.run_root != TABLE7.run_root
    assert TABLE1.default_model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert TABLE7.default_model == "Qwen/Qwen3-8B"
    assert TABLE1.methods == (
        "bf16",
        "osfp4-rtn",
        "osfp4-sic",
        "osfp4-rtn-a16",
        "osfp4-sic-a16",
    )
    assert TABLE7.methods == TABLE1.methods


def test_cache_from_other_table_is_rejected(tmp_path, monkeypatch):
    llama = _config(tmp_path, TABLE1)
    _write_manifest(llama)
    monkeypatch.setattr(
        calibration,
        "load_from_disk",
        lambda _path: Dataset.from_dict({"input_ids": [[1, 2, 3, 4]] * 2}),
    )
    qwen = _config(tmp_path, TABLE7)

    with pytest.raises(RuntimeError, match="benchmark_profile"):
        calibration.validate_prepared_dataset(qwen)


def test_dataset_contract_matches_fpquant_recipe(tmp_path):
    config = CalibrationConfig(
        profile=TABLE7,
        model_id=TABLE7.default_model,
        seed=42,
        num_samples=1024,
        max_sequence_length=2048,
        calibration_data_root=tmp_path,
    )
    manifest = config.data_manifest()
    assert manifest["calibration_dataset"] == "HuggingFaceFW/fineweb-edu"
    assert manifest["calibration_config"] == "sample-10BT"
    assert manifest["shuffle_buffer_size"] == 1000
    assert manifest["tokenizer"] == "Qwen/Qwen3-8B"


@pytest.mark.parametrize(
    "value,expected", [("auto", "auto"), ("16384", 16384), ("128", 128)]
)
def test_activation_subsample_cli_parser(value, expected):
    assert calibration._activation_subsample_size(value) == expected


def test_activation_subsample_cli_rejects_unknown_policy():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError, match="integer or 'auto'"):
        calibration._activation_subsample_size("adaptive")


@pytest.mark.parametrize(
    "option,expected", [(None, "auto"), ("auto", "auto"), ("16384", 16384)]
)
def test_calibration_cli_passes_sampling_policy_to_recipe(
    tmp_path, monkeypatch, option, expected
):
    import sys

    args = ["calibrate.py", "--method", "osfp4-rtn", "--save-dir", str(tmp_path)]
    if option is not None:
        args.extend(["--activation-subsample-size", option])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        calibration.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: None
    )
    monkeypatch.setattr(
        calibration.AutoTokenizer, "from_pretrained", lambda *a, **k: None
    )
    monkeypatch.setattr(calibration, "validate_prepared_dataset", lambda *a: [])

    class RecipeReached(Exception):
        pass

    def inspect_recipe(parsed):
        assert parsed.activation_subsample_size == expected
        recipe, _ = build_recipe(parsed)
        assert recipe[0].activation_subsample_size == expected
        raise RecipeReached

    build_recipe = calibration._build_recipe
    monkeypatch.setattr(calibration, "_build_recipe", inspect_recipe)
    with pytest.raises(RecipeReached):
        calibration.calibrate_main(TABLE1)
