from examples.fpquant.common import evaluation, protocol
from examples.fpquant.table1.config import PROFILE as TABLE1
from examples.fpquant.table7.config import PROFILE as TABLE7


def test_task_names_and_order():
    assert protocol.TASK_NAMES == (
        "winogrande",
        "hellaswag",
        "gsm8k_llama",
        "mmlu_cot_llama",
    )


def test_task_specs_match_fpquant_readme_flags():
    expected = {
        "winogrande": (5, False, False),
        "hellaswag": (10, False, False),
        "gsm8k_llama": (None, True, True),
        "mmlu_cot_llama": (None, True, True),
    }
    for name, values in expected.items():
        spec = protocol.TASK_SPECS_BY_NAME[name]
        assert (
            spec.num_fewshot,
            spec.apply_chat_template,
            spec.fewshot_as_multiturn,
        ) == values


def test_seeds_match_lm_eval_defaults():
    assert protocol.SEEDS == {
        "random_seed": 0,
        "numpy_random_seed": 1234,
        "torch_random_seed": 1234,
        "fewshot_random_seed": 1234,
    }


def test_model_args_match_fpquant_and_disable_qwen_thinking():
    llama_args = protocol.build_model_args(model=TABLE1.default_model)
    assert "dtype=bfloat16" in llama_args
    assert "enable_thinking" not in llama_args
    qwen_args = protocol.build_model_args(
        model=TABLE7.default_model,
        dtype=TABLE7.evaluation_dtype,
        enable_thinking=TABLE7.enable_thinking,
    )
    assert qwen_args.endswith("enable_thinking=False")


def test_table_wrappers_share_protocol_functions():
    assert evaluation.select_metric is protocol.select_metric
    assert evaluation.task_result is protocol.task_result


def test_resolved_fewshot_reads_n_shot():
    raw = {"n-shot": {"gsm8k_llama": 8}}
    assert protocol.resolved_fewshot(raw, "gsm8k_llama") == 8
    assert protocol.resolved_fewshot(raw, "missing") is None
