import json
import os
import stat
import tempfile
from functools import wraps
from pathlib import Path

from torch.nn import Module

__all__ = ["attach_osfp4_runtime_contract"]


def _mark_saved_config(save_directory: str | os.PathLike) -> None:
    """Atomically mark the saved config for the OSFP4 runtime."""
    config_path = Path(save_directory) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict) or quantization_config.get(
        "quant_method"
    ) not in ("compressed-tensors", "osfp4"):
        raise ValueError("Expected a compressed-tensors or osfp4 quantization_config")
    quantization_config["quant_method"] = "osfp4"

    mode = stat.S_IMODE(config_path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.",
        suffix=".tmp",
        dir=config_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.close(descriptor)
        temporary_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.chmod(mode)
        temporary_path.replace(config_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def attach_osfp4_runtime_contract(
    model: Module,
    runtime_smooth_quant_scale_targets: list[str] | tuple[str, ...],
) -> None:
    """Attach version 1 metadata for BF16 SmoothQuant activation multipliers."""
    model.config.osfp4_metadata = {
        "version": 1,
        "smooth_quant_scale_targets": sorted(runtime_smooth_quant_scale_targets),
    }

    save_pretrained = model.save_pretrained

    @wraps(save_pretrained)
    def save_pretrained_osfp4(save_directory, *args, **kwargs):
        """Save the model and mark its configuration for the OSFP4 runtime."""
        if kwargs.get("push_to_hub"):
            raise ValueError(
                "OSFP4 does not support save_pretrained(push_to_hub=True). "
                "Save the completed checkpoint locally, then upload that folder."
            )
        result = save_pretrained(save_directory, *args, **kwargs)
        _mark_saved_config(save_directory)
        return result

    model.save_pretrained = save_pretrained_osfp4
