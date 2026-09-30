"""Minimal OSFP4 calibration example."""

import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "Qwen3-0.6B-OSFP4"
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 512


def main() -> None:
    from llmcompressor import oneshot
    from llmcompressor_osfp4.modifiers import OSFP4Modifier

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
    ).shuffle(seed=42)
    dataset = dataset.select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(sample):
        return {
            "text": tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        }

    dataset = dataset.map(preprocess)
    recipe = OSFP4Modifier(
        scheme="NVFP4",
        optimization_mode="sic",
        targets=["Linear"],
        ignore=["lm_head"],
    )
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        recipe=recipe,
        pipeline="sequential",
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )
    model.save_pretrained(OUTPUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
