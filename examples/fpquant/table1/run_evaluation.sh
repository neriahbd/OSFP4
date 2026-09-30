#!/usr/bin/env bash
# Evaluate one Table 1 method in one lm-eval process per task.
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=${PYTHON_BIN:-/workspace/venvs/serve/bin/python}
run_root=${RUN_ROOT:-/workspace/other/runs/table1-runs}
model_id=${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}
method=${1:-osfp4-sic}
requested_task=${2:-}
evaluator="$repo_root/examples/fpquant/table1/evaluate.py"
tasks=(winogrande hellaswag gsm8k_llama mmlu_cot_llama)

[[ -x "$python_bin" ]] || { echo "Python not found: $python_bin" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

if [[ -n "$requested_task" ]]; then
  tasks=("$requested_task")
fi
if [[ "$method" == "bf16" ]]; then
  model=$model_id
else
  model="$run_root/$method/seed-42/model"
fi
for task in "${tasks[@]}"; do
  "$python_bin" "$evaluator" --method "$method" --model "$model" \
    --output-dir "$run_root/eval/$method" --tasks "$task" --resume
done
