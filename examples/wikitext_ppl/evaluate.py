#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MODEL_REVISION = "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_SPLIT = "test"

SEQUENCE_LENGTH = 2048
EXPECTED_ROWS = 4358
EXPECTED_JOINED_TOKENS = 289076
EXPECTED_TOKENS = 289078
EXPECTED_CHUNKS = 141
EXPECTED_RETAINED_TOKENS = 288768
EXPECTED_TAIL = 310
EXPECTED_SCORED_TOKENS = 288627
EXPECTED_HF_PPL = 6.139

REPO = Path(__file__).resolve().parents[2]
PLUGIN_REPO = Path("/workspace/osfp4/vllm-osfp4")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(repo: Path, *args: str) -> str | None:
    if not (repo / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def nvidia_smi_value(field: str) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={field}",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(values) == 1, values
    return values[0]


def environment_manifest() -> dict[str, Any]:
    import torch

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "datasets",
                "vllm",
                "compressed-tensors",
                "vllm-osfp4",
            )
        },
        "cuda_runtime": torch.version.cuda,
        "driver_version": nvidia_smi_value("driver_version"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": nvidia_smi_value("uuid"),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory": props.total_memory,
        "process_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_PLUGINS",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "OMP_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
        "llmcompressor_revision": git_value(REPO, "rev-parse", "HEAD"),
        "llmcompressor_status": git_value(REPO, "status", "--short"),
        "vllm_osfp4_revision": git_value(PLUGIN_REPO, "rev-parse", "HEAD"),
        "vllm_osfp4_status": git_value(PLUGIN_REPO, "status", "--short"),
    }


def detect_linear_backend(model_ref: str, kind: str) -> str:
    """Pick the vLLM NVFP4 kernel backend the checkpoint actually needs.

    vLLM 0.24.0 only auto-selects the weight-only Marlin kernel when
    linear_backend="auto"; forcing "cutlass" globally breaks weight-only
    (NVFP4A16) checkpoints (CutlassNvFp4LinearKernel unconditionally reads
    layer.input_global_scale_inv, which a weight-only scheme never creates).
    Full W4A4 (NVFP4) checkpoints must keep "cutlass": on compute capability
    12.0 (RTX 5090 / RTX PRO 6000 Blackwell), linear_backend="auto" prefers
    FlashInferCutlassNvFp4LinearKernel, which raises "No supported CUDA
    architectures found for major versions [12]" on this platform. Detected
    directly from the checkpoint's own config.json so the eval script needs
    no extra CLI flag.
    """
    if kind == "fp":
        return "cutlass"
    quant_config = json.loads(
        (Path(model_ref) / "config.json").read_text()
    )["quantization_config"]
    is_weight_only = all(
        group.get("input_activations") is None
        for group in quant_config["config_groups"].values()
    )
    return "auto" if is_weight_only else "cutlass"


def protocol_manifest(linear_backend: str) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
        "sequence_length": SEQUENCE_LENGTH,
        "rows": EXPECTED_ROWS,
        "tokens": EXPECTED_TOKENS,
        "chunks": EXPECTED_CHUNKS,
        "retained_tokens": EXPECTED_RETAINED_TOKENS,
        "discarded_tail": EXPECTED_TAIL,
        "scored_tokens": EXPECTED_SCORED_TOKENS,
        "keep_empty_rows": True,
        "add_special_tokens": False,
        "prepend_bos": True,
        "append_eos": True,
        "row_separator": "\\n\\n",
        "chunk_context_reset": True,
        "batch_size": 1,
        "dtype": "bfloat16",
        "position_averaging": (
            "exp(sum_negative_log_likelihood / total_scored_labels)"
        ),
        "vllm_settings": {
            "quantization_for_fp": None,
            "quantization_for_osfp4": "osfp4",
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.80,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "linear_backend": linear_backend,
            "seed": 0,
            "temperature": 0.0,
            "max_generated_tokens": 1,
            "prompt_logprobs": 0,
        },
        "evaluator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def build_chunks():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    assert len(dataset) == EXPECTED_ROWS, len(dataset)
    text = "\n\n".join(dataset["text"])

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    # Match the released NestQuant helper: join all rows, tokenize the text,
    # then add exactly one BOS and one EOS around the complete token stream.
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(token_ids) == EXPECTED_JOINED_TOKENS, len(token_ids)
    assert tokenizer.bos_token_id is not None
    assert tokenizer.eos_token_id is not None
    token_ids = [tokenizer.bos_token_id, *token_ids, tokenizer.eos_token_id]
    assert len(token_ids) == EXPECTED_TOKENS, len(token_ids)

    complete = len(token_ids) // SEQUENCE_LENGTH
    retained = complete * SEQUENCE_LENGTH
    tail = len(token_ids) - retained
    chunks = [
        token_ids[start : start + SEQUENCE_LENGTH]
        for start in range(0, retained, SEQUENCE_LENGTH)
    ]
    assert len(chunks) == EXPECTED_CHUNKS, len(chunks)
    assert retained == EXPECTED_RETAINED_TOKENS, retained
    assert tail == EXPECTED_TAIL, tail
    assert sum(len(chunk) - 1 for chunk in chunks) == EXPECTED_SCORED_TOKENS
    assert all(len(chunk) == SEQUENCE_LENGTH for chunk in chunks)
    return chunks


def model_revision_kwargs(model: str) -> dict[str, str]:
    return {} if Path(model).exists() else {"revision": MODEL_REVISION}


def score_hf(model_ref: str, chunks: list[list[int]]) -> tuple[float, float]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        dtype=torch.bfloat16,
        **model_revision_kwargs(model_ref),
    )
    model.to("cuda")
    model.eval()

    nll_sum = 0.0
    with torch.inference_mode():
        for index, chunk in enumerate(chunks):
            input_ids = torch.tensor(
                [chunk],
                dtype=torch.long,
                device="cuda",
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=input_ids.clone())
            value = float(outputs.loss.detach().float().item()) * (len(chunk) - 1)
            assert math.isfinite(value) and value > 0.0
            nll_sum += value
            if (index + 1) % 10 == 0 or index + 1 == len(chunks):
                print(
                    f"hf chunks={index + 1}/{len(chunks)} "
                    f"running_ppl={math.exp(nll_sum / ((index + 1) * 2047)):.6f}",
                    flush=True,
                )

    ppl = math.exp(nll_sum / EXPECTED_SCORED_TOKENS)
    return ppl, nll_sum


def score_vllm(
    model_ref: str,
    kind: str,
    chunks: list[list[int]],
    linear_backend: str,
) -> tuple[float, float]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kwargs: dict[str, Any] = {}
    if not Path(model_ref).exists():
        kwargs["revision"] = MODEL_REVISION

    llm = LLM(
        model=model_ref,
        tokenizer=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        dtype="bfloat16",
        quantization="osfp4" if kind == "osfp4" else None,
        tensor_parallel_size=1,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        enable_prefix_caching=False,
        linear_backend=linear_backend,
        seed=0,
        **kwargs,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
        detokenize=False,
        ignore_eos=True,
    )

    nll_sum = 0.0
    for index, chunk in enumerate(chunks):
        output = llm.generate(
            [TokensPrompt(prompt_token_ids=chunk)],
            sampling_params=sampling,
            use_tqdm=False,
        )[0]
        assert output.prompt_token_ids == chunk
        prompt_logprobs = output.prompt_logprobs
        assert prompt_logprobs is not None
        assert len(prompt_logprobs) == len(chunk)
        assert prompt_logprobs[0] is None

        chunk_nll = 0.0
        for token_id, position in zip(chunk[1:], prompt_logprobs[1:]):
            assert position is not None
            assert token_id in position, (token_id, position.keys())
            logprob = float(position[token_id].logprob)
            assert math.isfinite(logprob) and logprob <= 0.0
            chunk_nll -= logprob
        nll_sum += chunk_nll
        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(
                f"vllm kind={kind} chunks={index + 1}/{len(chunks)} "
                f"running_ppl={math.exp(nll_sum / ((index + 1) * 2047)):.6f}",
                flush=True,
            )

    ppl = math.exp(nll_sum / EXPECTED_SCORED_TOKENS)
    return ppl, nll_sum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("hf", "vllm"))
    parser.add_argument("--kind", required=True, choices=("fp", "osfp4", "plain"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "hf" and args.kind != "fp":
        raise ValueError("OSFP4 must be evaluated through vLLM")

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    if args.backend == "vllm":
        assert os.environ.get("VLLM_PLUGINS") == "osfp4"

    chunks = build_chunks()
    linear_backend = detect_linear_backend(args.model, args.kind)
    if args.backend == "hf":
        ppl, nll_sum = score_hf(args.model, chunks)
    else:
        ppl, nll_sum = score_vllm(args.model, args.kind, chunks, linear_backend)

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": args.backend,
        "kind": args.kind,
        "model": args.model,
        "perplexity": ppl,
        "negative_log_likelihood_sum": nll_sum,
        "mean_negative_log_likelihood": nll_sum / EXPECTED_SCORED_TOKENS,
        "released_code_bf16_target": EXPECTED_HF_PPL,
        "released_code_bf16_absolute_delta": (
            abs(ppl - EXPECTED_HF_PPL)
            if args.backend == "hf" and args.kind == "fp"
            else None
        ),
        "protocol": protocol_manifest(linear_backend),
        "environment": environment_manifest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "backend": args.backend,
        "kind": args.kind,
        "perplexity": ppl,
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
