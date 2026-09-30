"""FP-Quant Table 1 configuration."""

from examples.fpquant.common.protocol import BenchmarkProfile

PROFILE = BenchmarkProfile(
    name="llama-table1",
    table_number=1,
    default_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    architecture="LlamaForCausalLM",
    run_root="/workspace/other/runs/table1-runs",
    calibration_dtype="bfloat16",
    evaluation_dtype="bfloat16",
    enable_thinking=None,
    summary_filename="table1-summary.json",
    methods=(
        "bf16",
        "osfp4-rtn",
        "osfp4-sic",
        "osfp4-rtn-a16",
        "osfp4-sic-a16",
    ),
    paper_scores={
        "winogrande": 77.90,
        "hellaswag": 80.01,
        "gsm8k_llama": 85.06,
        "mmlu_cot_llama": 72.76,
    },
    paper_baseline_label="FP16",
)
