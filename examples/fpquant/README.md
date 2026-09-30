# FP-Quant reproduction pipelines

`table1/` and `table7/` are parallel wrappers around `common/`. Both use
deterministic random 2,048-token windows from FineWeb-Edu `sample-10BT`, a
1,000-record streaming shuffle buffer, seed 42, and native BF16 model loading.
Their tokenizer caches are separate and tagged with the table profile, model,
architecture, dtype, and full dataset contract.

OSFP4 checkpoints use `quant_method="osfp4"` and top-level `osfp4_metadata`
containing only `version: 1` and sorted `smooth_quant_scale_targets`. Version 1
defines each target's `smooth_quant_scale` as a BF16 activation multiplier.
Architecture remains in the model config and calibration manifest. Use these
checkpoints with the matching `vllm-osfp4` plugin.

- [`table1/`](table1/) reproduces the Llama 3.1 8B comparison.
- [`table7/`](table7/) produces the Qwen3-8B OSFP4 comparison.

The paper reports its dense baselines as FP16. Local dense runs deliberately
use each model's native BF16, matching FP-Quant's public `dtype=auto` behavior;
paper values remain separately labelled reference data.

The evaluator runs these four tasks with FP-Quant's public few-shot settings:
`winogrande`, `hellaswag`, `gsm8k_llama`, and `mmlu_cot_llama`. Every invocation
may select one or more tasks with `--tasks` and safely reuse valid raw task
results with `--resume`.
