# OSFP4Modifier

`OSFP4Modifier` produces optimized NVFP4 checkpoints. It supports A4W4
(`NVFP4`) and weight-only W4A16 (`NVFP4A16`) with RTN or SIC optimization.

## Usage

```python
from llmcompressor import oneshot
from llmcompressor_osfp4.modifiers import OSFP4Modifier

recipe = OSFP4Modifier(
    scheme="NVFP4",
    targets=["Linear"],
    ignore=["lm_head"],
)

oneshot(
    model=model,
    processor=tokenizer,
    dataset=calibration_dataset,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=1024,
)

model.save_pretrained(save_dir, save_compressed=True)
```

Save the completed checkpoint locally before uploading it. OSFP4 rejects
`save_pretrained(push_to_hub=True)` because the runtime configuration is
finalized atomically after the local model save returns.

OSFP4 must be the only modifier in the one-shot recipe. Sequential calibration
is selected automatically; an explicit `pipeline="sequential"` is also valid.
A runnable example is [available here](../../../examples/calibrate.py).

## Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `optimization_mode` | `"sic"` | `"sic"` or `"rtn"`. |
| `steps` | `80` | Adam optimization steps for the complete mapping. |
| `lr` | `0.12` | Adam learning rate. |
| `dampening_frac` | `0.01` | SIC Hessian damping fraction. |
| `offload_hessians` | `False` | Move the full Hessian to the GPU for each capture and back to CPU afterward. |
| `activation_subsample_size` | `"auto"` | NVFP4 activation-loss row cap; auto uses twice the mapping's input width, an integer sets a fixed cap, and `None` uses every row. |

Auto defines `n = weight.shape[1]` for the mapping's balance layers and retains
`k = min(total_rows, 2 * n)` complete activation vectors. Shared projections must
have the same input width; their output widths do not affect the cap. The sample
is selected once using seed 42 and reused throughout optimization. Sampling
records include `policy="auto"` and `n`. Explicit integer caps retain the
existing fixed-policy records and behavior; use `16384` for the previous default.

With activation subsampling enabled, OSFP4 automatically counts token rows in the
collated calibration batches, including padding. This adds one dataloader iteration
with RNG states restored, but no model forward pass. No manual token count is needed.
Capture retains the same seed-42 sampled rows and derives input global scales from
full-data channel maxima. Hessian/RTN statistics still use every row. Count mismatches
fail before optimization (for example, if a mapping changes the token count).
Without a sample cap, with custom observers, or when the loader cannot be safely
counted (iterable datasets, one-shot iterators, persistent workers, or missing token
inputs), full caching and replay remain unchanged. `NVFP4A16` disables activation caching.
The shared sequential cache and temporary `randperm(N)` indices still grow with N.

OSFP4 captures each mapping's inputs and RTN/SIC statistics, optimizes all
16-column groups together, selects E4M3 weight scales, deploys channel
smoothing, replays the full inputs for NVFP4 activation scales, and installs the
checkpoint parameters. Related projections share the first target's observer.
RTN selects all final scales together. SIC selects final scales one block at a
time, right to left, quantizing and propagating each block before the next scale
search. Its scale-search proxy includes completed blocks' quantization errors.
If a positive finite learned scale yields an empty E4M3 candidate interval below
or above the representable range, scale selection falls back to `2^-9` or `448`,
respectively. This relaxes the beta constraint and logs a simple fallback warning
only when clipping occurs. Invalid scales and nonfinite scoring errors
still fail.
Unmapped Linears store a BF16 `smooth_quant_scale` for the `vllm-osfp4` plugin.

A failure before deployment can be retried. A failure after smoothing starts
may leave modified weights and requires a fresh model and modifier.
