import json

import pytest

from examples.fpquant.common import evaluation
from examples.fpquant.table1.config import PROFILE as TABLE1
from examples.fpquant.table7.config import PROFILE as TABLE7


def test_build_summary_accepts_current_lm_eval_metrics():
    raw = {
        "winogrande": {"results": {"winogrande": {"acc,none": 0.70}}},
        "hellaswag": {"results": {"hellaswag": {"acc_norm,none": 0.71}}},
        "gsm8k_llama": {
            "results": {"gsm8k_llama": {"exact_match,flexible_extract": 0.72}}
        },
        "mmlu_cot_llama": {
            "results": {"mmlu_cot_llama": {"exact_match,strict_match": 0.73}}
        },
    }
    summary = evaluation.build_summary(raw)
    assert summary["average"] == pytest.approx(71.5)
    assert summary["paper_reference"]["baseline_label"] == "FP16"
    assert summary["paper_reference"]["scores"]["winogrande"] == 77.90


def _write_osfp4_checkpoint(path, profile=TABLE1, mode="rtn"):
    path.mkdir()
    for name in (
        *evaluation.COMMON_LOCAL_MODEL_FILES,
        *evaluation.COMPRESSED_MODEL_FILES,
    ):
        (path / name).write_text("")
    targets = ["model.layers.0.mlp.down_proj"]
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": [profile.architecture],
                "quantization_config": {
                    "quant_method": "osfp4",
                    "format": "nvfp4-pack-quantized",
                },
                "osfp4_metadata": {
                    **evaluation.OSFP4_RUNTIME_METADATA,
                    "smooth_quant_scale_targets": targets,
                },
            }
        )
    )
    (path / "calibration-manifest.json").write_text(
        json.dumps(
            {
                "benchmark_profile": profile.name,
                "model": profile.default_model,
                "architecture": profile.architecture,
                "native_dtype": "bfloat16",
                "torch_dtype": "bfloat16",
                "method": f"osfp4-{mode}",
                "calibration_dataset": "HuggingFaceFW/fineweb-edu",
                "calibration_config": "sample-10BT",
                "calibration_split": "train",
                "calibration_seed": 42,
                "num_calibration_samples": 1024,
                "max_sequence_length": 2048,
                "shuffle_buffer_size": 1000,
                "tokenizer": profile.default_model,
                "recipe": {
                    "optimization_mode": mode,
                    "osfp4_architecture": profile.architecture,
                    "smooth_quant_scale_targets": targets,
                    "scheme": "NVFP4",
                    "steps": 80,
                    "lr": 0.12,
                    "activation_subsample_size": 65536,
                    "activation_subsampling": {
                        "mapping": {
                            "policy": "fixed",
                            "seed": 42,
                            "m": 4,
                            "k": 4,
                            "k1": 4,
                            "index_sha256": "a" * 64,
                        }
                    },
                },
            }
        )
    )


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_validate_local_model_accepts_table_profile(tmp_path, mode):
    checkpoint = tmp_path / "model"
    _write_osfp4_checkpoint(checkpoint, TABLE1, mode)
    assert evaluation.validate_local_model(checkpoint, TABLE1, f"osfp4-{mode}") == []


def test_validate_local_model_ignores_extra_runtime_metadata(tmp_path):
    checkpoint = tmp_path / "model"
    _write_osfp4_checkpoint(checkpoint)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["osfp4_metadata"].update(
        architecture="ignored",
        scale_parameter="ignored",
        scale_dtype="ignored",
        scale_semantics="ignored",
    )
    config_path.write_text(json.dumps(config))
    assert evaluation.validate_local_model(checkpoint, TABLE1, "osfp4-rtn") == []


@pytest.mark.parametrize("version", [None, 2])
def test_validate_local_model_rejects_unsupported_contract(tmp_path, version):
    checkpoint = tmp_path / "model"
    _write_osfp4_checkpoint(checkpoint)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["osfp4_metadata"]["version"] = version
    config_path.write_text(json.dumps(config))
    errors = evaluation.validate_local_model(checkpoint, TABLE1, "osfp4-rtn")
    assert any("OSFP4 runtime metadata mismatch" in error for error in errors)


def test_checkpoint_from_other_table_is_rejected(tmp_path):
    checkpoint = tmp_path / "model"
    _write_osfp4_checkpoint(checkpoint, TABLE1, "sic")
    errors = evaluation.validate_local_model(checkpoint, TABLE7, "osfp4-sic")
    assert any("Qwen3ForCausalLM" in error for error in errors)


def test_unknown_optimization_mode_is_rejected(tmp_path):
    checkpoint = tmp_path / "model"
    _write_osfp4_checkpoint(checkpoint, TABLE1, "sic")
    manifest_path = checkpoint / "calibration-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["recipe"]["optimization_mode"] = "optimized"
    manifest_path.write_text(json.dumps(manifest))
    errors = evaluation.validate_local_model(checkpoint, TABLE1, "osfp4-sic")
    assert any("invalid recipe.optimization_mode" in error for error in errors)


def test_select_metric_rejects_invalid_score():
    with pytest.raises(ValueError):
        evaluation.select_metric({"acc,none": float("nan")}, ("acc,none",))


def test_validate_local_model_accepts_auto_sampling_provenance(tmp_path):
    checkpoint = tmp_path / "model"
    _write_osfp4_checkpoint(checkpoint)
    path = checkpoint / "calibration-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["recipe"]["activation_subsample_size"] = "auto"
    record = manifest["recipe"]["activation_subsampling"]["mapping"]
    record.update(policy="auto", n=4)
    path.write_text(json.dumps(manifest))
    assert evaluation.validate_local_model(checkpoint, TABLE1, "osfp4-rtn") == []
