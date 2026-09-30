"""FP-Quant Table 7 configuration."""

from examples.fpquant.common.protocol import BenchmarkProfile

PROFILE = BenchmarkProfile(
    name="qwen-table7",
    table_number=7,
    default_model="Qwen/Qwen3-8B",
    architecture="Qwen3ForCausalLM",
    run_root="/workspace/other/runs/table7-runs/qwen3-8b",
    calibration_dtype="bfloat16",
    evaluation_dtype="bfloat16",
    enable_thinking=False,
    summary_filename="method-summary.json",
    methods=(
        "bf16",
        "osfp4-rtn",
        "osfp4-sic",
        "osfp4-rtn-a16",
        "osfp4-sic-a16",
    ),
    paper_scores=None,
    paper_baseline_label="FP16",
)
