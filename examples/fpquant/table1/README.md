# FP-Quant Table 1: Llama 3.1 8B

This profile uses `meta-llama/Meta-Llama-3.1-8B-Instruct`,
`LlamaForCausalLM`, and native BF16. The measured baseline is `bf16`; the only
supported calibration methods are `osfp4-rtn` and `osfp4-sic`.

Prepare the Llama-tokenized cache once:

```bash
python examples/fpquant/table1/prepare_data.py
```

Calibrate a method directly, or use the shell launcher to run a two-sample
smoke job before the full 1,024-sample job:

```bash
python examples/fpquant/table1/calibrate.py --method osfp4-sic
examples/fpquant/table1/run_calibration.sh osfp4-sic
```

Evaluate one method. The shell launcher uses a fresh process for each task:

```bash
python examples/fpquant/table1/evaluate.py --method bf16 --resume
examples/fpquant/table1/run_evaluation.sh osfp4-sic
```

Outputs default to `/workspace/other/runs/table1-runs`. `table1-summary.json` contains
the local scores and a separately labelled FP16 paper reference.
