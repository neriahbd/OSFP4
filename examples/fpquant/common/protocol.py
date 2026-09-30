"""Single source of truth for the shared FP-Quant evaluation protocol.

Reference: /workspace/FP-Quant README.md "Evaluation" (commit d2e3092f), the
RECOMMENDED vLLM path -- four separate `lm_eval --model vllm` invocations:

    export OMP_NUM_THREADS=8
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    MODEL_ARGS="pretrained=$MODEL,max_model_len=4096,tensor_parallel_size=$NUM_GPUS,\
dtype=auto,gpu_memory_utilization=0.8,enforce_eager=True"

    lm_eval --model vllm --model_args $MODEL_ARGS --batch_size auto \
        --tasks winogrande --num_fewshot=5
    lm_eval --model vllm --model_args $MODEL_ARGS --batch_size auto \
        --tasks hellaswag --num_fewshot=10
    lm_eval --model vllm --model_args $MODEL_ARGS --batch_size auto \
        --tasks gsm8k_llama --apply_chat_template --fewshot_as_multiturn
    lm_eval --model vllm --model_args $MODEL_ARGS --batch_size auto \
        --tasks mmlu_cot_llama --apply_chat_template --fewshot_as_multiturn

Only these four tasks -- `arc_challenge_llama` and `truthfulqa` appear in
FP-Quant's own default task list (model_quant.py) but have no `if` branch in
its eval code, so FP-Quant itself never runs them.

Deviations from FP-Quant are deliberate:
  - dtype is pinned to bfloat16 rather than written as "auto". This is the
    native dtype to which `auto` resolves for both configured checkpoints and
    makes the concrete precision explicit in every manifest.
  - linear_backend is appended, chosen per checkpoint via
    detect_linear_backend() rather than one fixed value. Root-caused
    empirically (RTX PRO 6000, compute capability 12.0): forcing
    linear_backend="cutlass" for a weight-only NVFP4A16 checkpoint crashes
    with AttributeError: '...ParallelLinear' object has no attribute
    'input_global_scale_inv' (CutlassNvFp4LinearKernel unconditionally
    expects an activation scale a weight-only scheme never creates).
    Leaving it unset (vLLM's own "auto") fixes that -- vLLM force-selects
    the weight-only Marlin kernel whenever use_a16=True, regardless of
    platform -- but forcing "auto" for a full W4A4 (NVFP4) checkpoint
    instead crashes with RuntimeError: No supported CUDA architectures
    found for major versions [12] (auto-selection prefers
    FlashInferCutlassNvFp4LinearKernel, unsupported on sm_120). Only
    "auto"/unset for weight-only and "cutlass" for W4A4 both work; neither
    single fixed value serves both kernel requirements on this platform.

HARD RULE: standard library only at module scope. This module is imported
by table-specific evaluators under the vLLM serving environment, which may not
have llmcompressor installed. Import lm_eval lazily, inside functions, never at
module scope.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# lm-eval's own CLI/simple_evaluate defaults. FP-Quant passes none of these
# explicitly, so these four values ARE FP-Quant's seeds by default -- pinning
# them here keeps that true even if lm-eval's own defaults ever change.
SEEDS: dict[str, int] = {
    "random_seed": 0,
    "numpy_random_seed": 1234,
    "torch_random_seed": 1234,
    "fewshot_random_seed": 1234,
}


@dataclass(frozen=True)
class BenchmarkProfile:
    """Table-specific settings layered on the shared FP-Quant protocol."""

    name: str
    table_number: int
    default_model: str
    architecture: str
    run_root: str
    calibration_dtype: str
    evaluation_dtype: str
    enable_thinking: bool | None
    summary_filename: str
    methods: tuple[str, ...]
    paper_scores: dict[str, float] | None
    paper_baseline_label: str | None


@dataclass(frozen=True)
class TaskSpec:
    name: str
    num_fewshot: int | None  # None => use the task YAML's own default
    apply_chat_template: bool
    fewshot_as_multiturn: bool
    metric_candidates: tuple[str, ...]
    paper_score: float  # FP-Quant Table 1, percent


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec(
        name="winogrande",
        num_fewshot=5,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        metric_candidates=("acc,none",),
        paper_score=77.90,
    ),
    TaskSpec(
        name="hellaswag",
        num_fewshot=10,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        metric_candidates=("acc_norm,none",),
        paper_score=80.01,
    ),
    TaskSpec(
        name="gsm8k_llama",
        num_fewshot=None,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        metric_candidates=(
            "exact_match,flexible_extract",
            "exact_match,flexible-extract",
            "exact_match,strict_match",
            "exact_match,strict-match",
        ),
        paper_score=85.06,
    ),
    TaskSpec(
        name="mmlu_cot_llama",
        num_fewshot=None,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        metric_candidates=(
            "exact_match,strict_match",
            "exact_match,strict-match",
            "acc,none",
        ),
        paper_score=72.76,
    ),
)
TASK_SPECS_BY_NAME: dict[str, TaskSpec] = {spec.name: spec for spec in TASK_SPECS}
TASK_NAMES: tuple[str, ...] = tuple(spec.name for spec in TASK_SPECS)
PAPER_SCORES: dict[str, float] = {spec.name: spec.paper_score for spec in TASK_SPECS}


def detect_linear_backend(model: str) -> str | None:
    """Pick the vLLM NVFP4 kernel backend an OSFP4 checkpoint actually needs.

    Returns None (omit the field, vLLM's true "auto" default) for a
    weight-only NVFP4A16 checkpoint, "cutlass" for a full W4A4 NVFP4
    checkpoint, and "cutlass" (the previously safe default, unused since no
    NVFP4 kernel gets invoked) for anything that isn't a local OSFP4
    checkpoint -- the unquantized BF16 baseline, or a remote model id. See
    the "Deviations from FP-Quant" note above for why a single fixed value
    cannot serve both kernel requirements on this platform.
    """
    path = Path(model)
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.is_file():
        return "cutlass"
    quantization = json.loads(config_path.read_text()).get("quantization_config")
    if (
        not isinstance(quantization, dict)
        or quantization.get("quant_method") != "osfp4"
    ):
        return "cutlass"
    config_groups = quantization.get("config_groups", {})
    is_weight_only = all(
        group.get("input_activations") is None for group in config_groups.values()
    )
    return None if is_weight_only else "cutlass"


def build_model_args(
    *,
    model: str,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    enforce_eager: bool = True,
    linear_backend: str | None = "cutlass",
    quantization: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """Build a vLLM lm-eval model_args string.

    Emits FP-Quant's six fields in FP-Quant's order first, so the output is a
    strict prefix match against FP-Quant's own MODEL_ARGS (modulo the dtype
    deviation documented at module level) for anyone diffing the two. Our
    additions (linear_backend, optional quantization) are appended after.

    `quantization` is omitted by default: vLLM reads quant_method out of the
    checkpoint's config.json on its own (see vllm/config/model.py), so passing
    it is redundant for a quantized checkpoint and actively wrong for an
    unquantized BF16 baseline (there is no quantization method to name).
    """
    parts = [
        f"pretrained={model}",
        f"max_model_len={max_model_len}",
        f"tensor_parallel_size={tensor_parallel_size}",
        f"dtype={dtype}",
        f"gpu_memory_utilization={gpu_memory_utilization}",
        f"enforce_eager={enforce_eager}",
    ]
    if linear_backend:
        parts.append(f"linear_backend={linear_backend}")
    if quantization:
        parts.append(f"quantization={quantization}")
    if enable_thinking is not None:
        parts.append(f"enable_thinking={enable_thinking}")
    return ",".join(parts)


def task_result(raw_result: dict[str, Any], task: str) -> dict[str, Any]:
    """Extract one task's result dict from a `simple_evaluate` return value.

    Kept here so both table wrappers use identical extraction behavior.
    """
    results = raw_result["results"]
    if task in results:
        return results[task]
    if len(results) == 1:
        return next(iter(results.values()))
    raise KeyError(f"could not find {task!r} in result tasks {list(results)}")


def select_metric(
    result: dict[str, Any], candidates: tuple[str, ...]
) -> tuple[str, float]:
    """Pick the first present metric from `candidates` and scale it to 0-100.

    Kept here so both table wrappers use identical metric validation.
    """
    for metric in candidates:
        if metric in result:
            score = float(result[metric]) * 100.0
            if not math.isfinite(score) or not 0.0 <= score <= 100.0:
                raise ValueError(f"invalid score for {metric}: {score}")
            return metric, score
    raise KeyError(
        f"none of expected metrics {candidates} found; available={sorted(result)}"
    )


def resolved_fewshot(raw_result: dict[str, Any], task: str) -> int | None:
    """The fewshot count lm-eval actually used, not the one requested.

    `simple_evaluate` records this in `result["n-shot"][task]` regardless of
    whether `num_fewshot` was passed explicitly or left as the task YAML's
    own default (our gsm8k_llama/mmlu_cot_llama case). Prefer this over a
    hardcoded literal, which can silently go stale if the task YAML changes.
    """
    return raw_result.get("n-shot", {}).get(task)


def run_task(
    spec: TaskSpec,
    model_args: str,
    *,
    backend: str = "vllm",
    batch_size: str = "auto",
    limit: int | None = None,
    log_samples: bool = False,
    task_manager: Any | None = None,
) -> dict[str, Any]:
    """Run one task via lm_eval.simple_evaluate with FP-Quant-matching kwargs.

    log_samples defaults to False here, unlike simple_evaluate's own default
    of True -- that default is why historical reports (e.g. osfp4-gsm8k.json)
    ballooned to tens of MB with every per-document sample embedded. Scores
    are unaffected either way; pass log_samples=True to opt back in.
    """
    import lm_eval

    return lm_eval.simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=[spec.name],
        num_fewshot=spec.num_fewshot,
        apply_chat_template=spec.apply_chat_template,
        fewshot_as_multiturn=spec.fewshot_as_multiturn,
        batch_size=batch_size,
        limit=limit,
        log_samples=log_samples,
        task_manager=task_manager,
        **SEEDS,
    )
