"""NestQuant-comparable WikiText-2 perplexity reproduction: calibration profile."""

from examples.fpquant.common.calibration import CalibrationSource
from examples.fpquant.common.protocol import BenchmarkProfile

CALIBRATION_SOURCE = CalibrationSource(
    dataset_name="Salesforce/wikitext",
    dataset_config="wikitext-2-raw-v1",
    dataset_split="train",
    shuffle_buffer_size=0,
)

PROFILE = BenchmarkProfile(
    name="wikitext-ppl",
    table_number=0,
    default_model="meta-llama/Meta-Llama-3-8B",
    architecture="LlamaForCausalLM",
    run_root="/workspace/other/runs/wikitext-ppl",
    calibration_dtype="bfloat16",
    evaluation_dtype="bfloat16",
    enable_thinking=None,
    summary_filename="wikitext-ppl-summary.json",
    methods=("bf16", "osfp4-rtn", "osfp4-sic", "osfp4-rtn-a16", "osfp4-sic-a16"),
    paper_scores=None,
    paper_baseline_label=None,
)
