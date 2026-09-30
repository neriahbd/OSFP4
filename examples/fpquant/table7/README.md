# FP-Quant Table 7: Qwen3-8B

This profile uses `Qwen/Qwen3-8B`, `Qwen3ForCausalLM`, and native BF16.
Qwen thinking is disabled through lm-eval's `enable_thinking=False` model
argument. Measured methods are `bf16`, `osfp4-rtn`, and `osfp4-sic`.

```bash
python examples/fpquant/table7/prepare_data.py
examples/fpquant/table7/run_calibration.sh osfp4-rtn
examples/fpquant/table7/run_calibration.sh osfp4-sic
examples/fpquant/table7/run_evaluation.sh bf16
examples/fpquant/table7/run_evaluation.sh osfp4-rtn
examples/fpquant/table7/run_evaluation.sh osfp4-sic
python examples/fpquant/table7/aggregate.py
```

Outputs default to `/workspace/other/runs/table7-runs/qwen3-8b`. Aggregation requires all
three methods and all four tasks, calculates recovery from unrounded measured
BF16 scores, and writes `table7-osfp4.json`, `table7-osfp4.csv`, and
`table7-osfp4.md`. The paper's FP16 and NVFP rows are a separate reference
block, not mixed with the local measurements.
