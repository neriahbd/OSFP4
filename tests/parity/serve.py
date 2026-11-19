"""Run one exported fixture through its installed vLLM runtime on a GPU."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("result", type=Path)
    args = p.parse_args()
    model = LLM(
        model=str(args.checkpoint),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_logprobs=128,
        seed=42,
    )
    params = SamplingParams(max_tokens=1, temperature=0, logprobs=128, detokenize=False)
    result = model.generate([{"prompt_token_ids": list(range(16))}], params)[0].outputs[
        0
    ]
    dist = result.logprobs[0]
    keys = sorted(dist)
    if keys != list(range(128)):
        raise ValueError(
            "Expected a next-token log probability for each of the 128 fixture tokens"
        )
    save_file(
        {
            "token": torch.tensor(list(result.token_ids)),
            "next_token_logprobs": torch.tensor(
                [dist[key].logprob for key in keys], dtype=torch.float32
            ),
        },
        args.result,
    )


if __name__ == "__main__":
    main()
