"""Capture and compare independent OSFP4 calibration runs; see README.md."""

import argparse
import hashlib
import importlib
import json
from importlib.metadata import version
import subprocess
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from check_source import REFERENCE, check as check_source

CASES = [(s, m) for s in ("NVFP4", "NVFP4A16") for m in ("rtn", "sic")]


def check_reference(module, revision):
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(module.__file__).parent,
            text=True,
        ).strip()
    )
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        == revision
    )
    subprocess.run(
        ["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "src"], check=True
    )
    return root


def chunked_fixture(obs, device, folder, forced_chunk=None, projection=None):
    """Trigger the production split policy for both objectives on manageable tensors."""
    shapes = {
        "qkv": (256, 6144, 8192),
        "o": (256, 4096, 8192),
        "gate-up": (256, 24576, 8192),
        "down": (768, 4096, 24576),
    }
    blocks, rows, samples = shapes[projection] if projection else (65, 4096, 17)
    generator = torch.Generator().manual_seed(47)
    weight = torch.randn(blocks, rows, 16, generator=generator).to(device)
    activations = torch.randn(blocks, 16, samples, generator=generator).to(device)
    metric = torch.rand(blocks, 16, generator=generator).to(device) + 0.25
    values = {}
    peaks = {}
    for objective, k in [("joint", samples), ("weight", 0)]:
        assert obs.uses_exact_alpha_reduction(rows, k, blocks)
        assert obs.select_blocks_per_chunk(rows, k, blocks) < blocks
        with ExitStack() as stack:
            if forced_chunk is not None:
                stack.enter_context(
                    patch.object(
                        obs, "select_blocks_per_chunk", return_value=forced_chunk
                    )
                )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if objective == "joint":
                scales = obs._optimize_joint_scales(
                    weight, activations, weight_metric=metric, steps=3, lr=0.12
                )
            else:
                scales = obs._optimize_weight_scales(weight, metric, steps=3, lr=0.12)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peaks[objective] = torch.cuda.max_memory_allocated(device)
        for name, value in zip(scales._fields, scales):
            if value is not None:
                values[f"{objective}.{name}"] = value.detach().cpu().contiguous()
    save_file(values, folder / "chunked.safetensors")
    if peaks:
        (folder / "peak_allocated_bytes.json").write_text(json.dumps(peaks))


def capture(args):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from llmcompressor.args import DatasetArguments
    from llmcompressor.core import create_session
    from llmcompressor.pipelines import CalibrationPipeline
    from llmcompressor.pipelines.sequential import pipeline
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )

    reference = args.impl == "diagonal"
    mod = importlib.import_module(
        "llmcompressor.modifiers.osfp4"
        if reference
        else "llmcompressor_osfp4.modifiers"
    )
    obs = importlib.import_module(
        "llmcompressor.observers.osfp4.observer"
        if reference
        else "llmcompressor_osfp4.observers.observer"
    )
    if reference:
        check_source(check_reference(mod, REFERENCE))
        check_reference(
            importlib.import_module("compressed_tensors"),
            "30d017eec16e3d365be66017cbc1a57782ff7bcb",
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(119)
    width = 256 if args.serving_fixture else 64
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=width,
        intermediate_size=2 * width,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=width // 4,
        use_cache=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    original = args.output / "input"
    Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(original)
    ids = torch.randint(0, 128, (12, 32), generator=torch.Generator().manual_seed(42))
    samples = [{"input_ids": sample.unsqueeze(0)} for sample in ids]
    device = torch.device(args.device)
    chunked_fixture(obs, device, args.output, args.force_chunk, args.projection)
    for scheme, mode in CASES:
        model = Qwen3ForCausalLM.from_pretrained(
            original, dtype=torch.bfloat16, attn_implementation="eager"
        ).eval()
        modify_save_pretrained(model)
        loader = torch.utils.data.DataLoader(samples, batch_size=None)
        modifier = mod.OSFP4Modifier(
            scheme=scheme,
            optimization_mode=mode,
            ignore=["lm_head"],
            steps=4,
            activation_subsample_size="auto",
        )
        folder = args.output / f"{scheme}-{mode}"
        folder.mkdir()
        tensors = {"input.token_ids": ids}
        mapping_name = None

        def record(name, value):
            if value is not None:
                key = f"{mapping_name}.{name}"
                if key in tensors:
                    raise ValueError(f"Duplicate capture {key}")
                tensors[key] = value.detach().cpu().contiguous().clone()

        original_mapping = mod.OSFP4Modifier._optimize_mapping

        def mapping(instance, mapping):
            nonlocal mapping_name
            mapping_name = mapping.mapping_name
            cache = instance._calibration_cache
            cache.wait_for_capture(mapping_name, torch.device("cpu"))
            for name, attr in [
                (
                    "sampled_indices",
                    "sampled_indices",
                ),
                ("hessian", "hessian"),
                (
                    "second_moment",
                    "sigma_x_squared_sum",
                ),
                (
                    "input_absmax",
                    "input_channel_absmax",
                ),
            ]:
                value = getattr(cache, attr).get(mapping_name)
                record(name, value)
            return original_mapping(instance, mapping)

        def watch(func, prefix):
            def wrapped(*args, **kwargs):
                if prefix == "initial":
                    record("weight_blocks", args[0])
                    record("activation_blocks", args[1])
                result = func(*args, **kwargs)
                for field, value in zip(result._fields, result):
                    record(f"{prefix}.{field}", value)
                return result

            return wrapped

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(pipeline, "get_main_device", lambda: device)
            )
            stack.enter_context(
                patch.object(mod.OSFP4Modifier, "_optimize_mapping", mapping)
            )
            for attr, stage in [
                ("initialize_scale_values_from_absmax", "initial"),
                ("materialize_scale_values", "continuous"),
            ]:
                stack.enter_context(
                    patch.object(obs, attr, watch(getattr(obs, attr), stage))
                )
            quantize = importlib.import_module(mod.__name__ + ".osfp4_quantize")
            opt = "optimize_" + mode
            original_opt = getattr(quantize, opt)

            def capture_result(*inputs, **kwargs):
                result = original_opt(*inputs, **kwargs)
                record("rounded_alpha", result.alpha)
                record(
                    "selected_gamma",
                    result.gamma_w_final,
                )
                record("zero_points", result.weight_zero_point)
                if mode == "sic":
                    record("sic_weight", result.quantized_weight)
                return result

            stack.enter_context(patch.object(quantize, opt, capture_result))
            session = stack.enter_context(create_session())
            session.initialize(
                model=model,
                recipe=[modifier],
                start=-1,
                calib_data=loader,
                sequential_targets=["Qwen3DecoderLayer"],
            )
            CalibrationPipeline.from_modifiers(session.lifecycle.recipe.modifiers)(
                model,
                loader,
                DatasetArguments(
                    sequential_targets=["Qwen3DecoderLayer"],
                    sequential_offload_device="cpu",
                    propagate_error=True,
                ),
            )
            # Public runtime hooks still apply alpha here. Finalization removes them.
            with torch.inference_mode():
                tensors["probe.logits"] = (
                    model(input_ids=ids[:1].to(model.device))
                    .logits.detach()
                    .cpu()
                    .contiguous()
                )
            session.finalize()
        save_file(tensors, folder / "optimizer.safetensors")
        save_file(
            {
                k: v.detach().cpu().contiguous().clone()
                for k, v in model.state_dict().items()
            },
            folder / "deployed.safetensors",
        )
        model.save_pretrained(folder / "export", save_compressed=True)
        exported = read_tensors((folder / "export").glob("*.safetensors"))
        for name, alpha in exported.items():
            if name.endswith(".smooth_quant_scale"):
                observed = tensors[
                    name.removesuffix(".smooth_quant_scale") + ".rounded_alpha"
                ]
                assert (
                    alpha.dtype == torch.bfloat16
                    and torch.isfinite(alpha).all()
                    and (alpha > 0).all()
                )
                assert torch.equal(
                    alpha.view(torch.uint8),
                    observed.flatten().bfloat16().view(torch.uint8),
                )

        if not reference:
            saved = json.loads((folder / "export/config.json").read_text())
            assert saved["quantization_config"]["quant_method"] == "osfp4"
            targets = sorted(
                k.removesuffix(".smooth_quant_scale")
                for k in model.state_dict()
                if k.endswith(".smooth_quant_scale")
            )
            assert targets and saved["osfp4_metadata"] == {
                "version": 1,
                "smooth_quant_scale_targets": targets,
            }
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                implementation=args.impl,
                reference=REFERENCE,
                torch=torch.__version__,
                device=str(device),
                cuda=torch.version.cuda,
                hardware=torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu",
                transformers=version("transformers"),
                forced_chunk=args.force_chunk,
            ),
            indent=2,
        )
    )


def read_tensors(paths):
    result = {}
    for path in paths:
        for key, tensor in load_file(path).items():
            if key.endswith(".smooth_quant_input.weight"):
                key = (
                    key.removesuffix(".smooth_quant_input.weight")
                    + ".smooth_quant_scale"
                )
            if key in result:
                raise ValueError(f"Duplicate tensor {key}")
            result[key] = tensor
    if not result:
        raise ValueError(f"Empty tensor inventory: {paths}")
    return result


def compare(args):
    ref, candidate = (
        json.loads((path / "run.json").read_text())
        for path in (args.reference, args.candidate)
    )
    if (
        ref["implementation"] != "diagonal"
        or ref["reference"] != REFERENCE
        or ref["forced_chunk"] is not None
    ):
        raise ValueError(
            "Expected a pinned diagonal reference with default chunk selection"
        )
    for field in ("reference", "torch", "device", "cuda", "hardware", "transformers"):
        if ref[field] != candidate[field]:
            raise ValueError(f"Incompatible runs: {field} differs")
    count = 0
    inventory = {}
    for pattern in ["input/*.safetensors", "chunked.safetensors"] + [
        f"{s}-{m}/{part}"
        for s, m in CASES
        for part in (
            ["optimizer.safetensors", "deployed.safetensors", "export/*.safetensors"]
            + (["serving.safetensors"] if args.serving else [])
        )
    ]:
        lhs, rhs = (
            read_tensors(sorted(root.glob(pattern)))
            for root in (args.reference, args.candidate)
        )
        if lhs.keys() != rhs.keys():
            raise ValueError(
                f"{pattern}: differing inventory {sorted(lhs.keys() ^ rhs.keys())}"
            )
        for key, left in lhs.items():
            right = rhs[key]
            if left.dtype != right.dtype or left.shape != right.shape:
                raise ValueError(
                    f"{pattern}/{key}: {left.dtype} {left.shape} vs {right.dtype} {right.shape}"
                )
            a, b = (t.contiguous().reshape(-1).view(torch.uint8) for t in (left, right))
            if not torch.equal(a, b):
                first = (a != b).nonzero()[0].item()
                hashes = [
                    hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in (a, b)
                ]
                raise ValueError(f"{pattern}/{key}: byte {first}, SHA256 {hashes}")
            if args.receipt:
                inventory[f"{pattern}/{key}"] = {
                    "dtype": str(left.dtype),
                    "shape": list(left.shape),
                    "sha256": hashlib.sha256(a.numpy().tobytes()).hexdigest(),
                }
            count += 1
        print(pattern, len(lhs), "byte-identical tensors")
    if args.receipt:
        args.receipt.write_text(
            json.dumps(
                {
                    "reference": ref,
                    "candidate": candidate,
                    "matched_tensors": inventory,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    print(count, "tensor comparisons passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    cap = subs.add_parser("capture")
    cap.add_argument("impl", choices=["diagonal", "public"])
    cap.add_argument("output", type=Path)
    cap.add_argument(
        "--projection",
        choices=["qkv", "o", "gate-up", "down"],
        help="Exercise a full Qwen3-8B optimizer shape",
    )
    cap.add_argument(
        "--serving-fixture",
        action="store_true",
        help="Use kernel-aligned Qwen dimensions for vLLM",
    )
    cap.add_argument(
        "--force-chunk",
        type=int,
        help="Override chunk size for the large optimizer fixture only",
    )
    cap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    cmp = subs.add_parser("compare")
    cmp.add_argument(
        "--serving",
        action="store_true",
        help="Also compare saved token IDs and log probabilities",
    )
    cmp.add_argument(
        "--receipt",
        type=Path,
        help="Save dtype, shape and SHA256 for each matched tensor",
    )
    cmp.add_argument("reference", type=Path)
    cmp.add_argument("candidate", type=Path)
    args = parser.parse_args()
    (capture if args.command == "capture" else compare)(args)
