import json
import stat
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from llmcompressor_osfp4.modifiers import runtime_contract
from llmcompressor_osfp4.modifiers.runtime_contract import (
    attach_osfp4_runtime_contract,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = LlamaConfig(num_hidden_layers=2)
        self.config.architectures = ["Model"]

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        return "saved"


TARGETS = [
    "model.layers.1.mlp.down_proj",
    "model.layers.0.self_attn.o_proj",
]
EXPECTED_METADATA = {
    "version": 1,
    "smooth_quant_scale_targets": sorted(TARGETS),
}


def test_attaches_minimal_deterministic_runtime_metadata():
    model = Model()

    assert attach_osfp4_runtime_contract(model, TARGETS) is None

    assert model.config.osfp4_metadata == EXPECTED_METADATA


@pytest.mark.parametrize("targets", [[], (), TARGETS, tuple(reversed(TARGETS))])
def test_attaches_without_architecture(targets):
    model = Model()
    model.config = SimpleNamespace()

    attach_osfp4_runtime_contract(model, targets)

    assert model.config.osfp4_metadata == {
        "version": 1,
        "smooth_quant_scale_targets": sorted(targets),
    }


def test_save_marks_osfp4_and_preserves_compressed_tensors_fields(tmp_path):
    model = Model()
    model.config.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
    }
    attach_osfp4_runtime_contract(model, TARGETS)

    assert model.save_pretrained(tmp_path) == "saved"

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["osfp4_metadata"] == EXPECTED_METADATA
    assert config["quantization_config"] == {
        "quant_method": "osfp4",
        "format": "float-quantized",
    }


@pytest.mark.parametrize(
    "quantization_config",
    [
        None,
        [],
        {},
        {"quant_method": "other"},
    ],
)
def test_invalid_saved_config_is_not_replaced(
    tmp_path,
    quantization_config,
):
    model = Model()
    saved_config = json.dumps({"quantization_config": quantization_config})

    def save(directory):
        (directory / "config.json").write_text(saved_config, encoding="utf-8")

    model.save_pretrained = save
    attach_osfp4_runtime_contract(model, TARGETS)

    with pytest.raises(ValueError, match="Expected .* quantization_config"):
        model.save_pretrained(tmp_path)

    assert (tmp_path / "config.json").read_bytes() == saved_config.encode("utf-8")


def test_save_forwards_arguments_and_supports_repeated_saves(tmp_path):
    model = Model()
    model.config.quantization_config = {"quant_method": "compressed-tensors"}
    calls = []
    result = object()

    def save(directory, *args, **kwargs):
        calls.append((directory, args, kwargs))
        model.config.save_pretrained(directory)
        return result

    model.save_pretrained = save
    attach_osfp4_runtime_contract(model, TARGETS)
    for quant_method in ("compressed-tensors", "osfp4"):
        model.config.quantization_config["quant_method"] = quant_method
        assert (
            model.save_pretrained(tmp_path, "argument", save_compressed=True) is result
        )
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["quantization_config"]["quant_method"] == "osfp4"
        assert config["osfp4_metadata"] == EXPECTED_METADATA
    assert calls == [(tmp_path, ("argument",), {"save_compressed": True})] * 2


def test_save_failure_does_not_mark_config(tmp_path, monkeypatch):
    model = Model()
    failure = OSError("save failed")

    def fail_save(*args, **kwargs):
        raise failure

    def unexpected_mark(*args):
        pytest.fail("Failed saves must not mark the config")

    model.save_pretrained = fail_save
    monkeypatch.setattr(runtime_contract, "_mark_saved_config", unexpected_mark)
    attach_osfp4_runtime_contract(model, TARGETS)
    with pytest.raises(OSError) as raised:
        model.save_pretrained(tmp_path)
    assert raised.value is failure


def test_config_write_failure_propagates(tmp_path, monkeypatch):
    model = Model()
    model.config.quantization_config = {"quant_method": "compressed-tensors"}
    model.config.save_pretrained(tmp_path)
    # Keep the wrapped save successful without touching the existing config.
    model.save_pretrained = lambda *args, **kwargs: None
    attach_osfp4_runtime_contract(model, TARGETS)
    failure = OSError("write failed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(runtime_contract.Path, "write_text", fail)

    with pytest.raises(OSError) as raised:
        model.save_pretrained(tmp_path)
    assert raised.value is failure
    assert (
        json.loads((tmp_path / "config.json").read_text())["quantization_config"][
            "quant_method"
        ]
        == "compressed-tensors"
    )
    assert not list(tmp_path.glob(".config.json.*.tmp"))


def test_config_replace_failure_preserves_original_and_cleans_temporary_file(
    tmp_path, monkeypatch
):
    model = Model()
    model.config.quantization_config = {"quant_method": "compressed-tensors"}
    model.config.save_pretrained(tmp_path)
    original = (tmp_path / "config.json").read_bytes()
    model.save_pretrained = lambda *args, **kwargs: None
    attach_osfp4_runtime_contract(model, TARGETS)
    failure = OSError("replace failed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(runtime_contract.Path, "replace", fail)

    with pytest.raises(OSError) as raised:
        model.save_pretrained(tmp_path)
    assert raised.value is failure
    assert (tmp_path / "config.json").read_bytes() == original
    assert not list(tmp_path.glob(".config.json.*.tmp"))


def test_atomic_config_replace_preserves_permissions(tmp_path):
    model = Model()
    model.config.quantization_config = {"quant_method": "compressed-tensors"}
    attach_osfp4_runtime_contract(model, TARGETS)
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.chmod(0o640)

    model.save_pretrained(tmp_path)

    assert stat.S_IMODE(config_path.stat().st_mode) == 0o640


def test_push_to_hub_is_rejected_before_saving(tmp_path):
    model = Model()
    calls = []

    def save(*args, **kwargs):
        calls.append((args, kwargs))

    model.save_pretrained = save
    attach_osfp4_runtime_contract(model, TARGETS)

    with pytest.raises(ValueError, match="Save the completed checkpoint locally"):
        model.save_pretrained(tmp_path, push_to_hub=True)

    assert calls == []
    assert not any(tmp_path.iterdir())


def test_saved_json_bytes_match_original_format(tmp_path):
    model = Model()
    config_path = tmp_path / "config.json"

    def save(directory):
        (directory / "config.json").write_text(
            '{"quantization_config": {"quant_method": "compressed-tensors"}, '
            '"label": "caf\u00e9", "values": [-0.0, true, null]}',
            encoding="utf-8",
        )

    model.save_pretrained = save
    attach_osfp4_runtime_contract(model, TARGETS)
    model.save_pretrained(tmp_path)

    assert config_path.read_bytes() == (
        b'{\n  "label": "caf\\u00e9",\n'
        b'  "quantization_config": {\n    "quant_method": "osfp4"\n  },\n'
        b'  "values": [\n    -0.0,\n    true,\n    null\n  ]\n}\n'
    )
