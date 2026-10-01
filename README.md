# OSFP4

OSFP4 is an optimized-scale NVFP4 quantization plugin for
[LLM Compressor](https://github.com/vllm-project/llm-compressor). It adds
`OSFP4Modifier`, which calibrates a model and learns better FP4 scales than
plain round-to-nearest NVFP4. The result is a standard compressed-tensors
checkpoint.

Supported schemes:

| Scheme | Weights | Activations |
| --- | --- | --- |
| `NVFP4` | FP4 | FP4 (A4W4) |
| `NVFP4A16` | FP4 | 16-bit (weight-only) |

Scale optimization modes:

- **`sic`** (default): Hessian-aware. Quantizes block by block and propagates
  the error. More accurate, slower.
- **`rtn`**: faster. Uses a diagonal activation statistic.

## Requirements

| Component | Version |
| --- | --- |
| Python | 3.10 or newer |
| `llmcompressor` | 0.14.0 (installed automatically) |
| `compressed-tensors` | 0.19.0 (installed automatically) |
| `torch` | 2.10–2.14 |
| `transformers` | 5.15–5.17 |

A CUDA GPU is recommended for real models. CPU works for small models and
tests.

## Installation

From GitHub:

```bash
pip install "llmcompressor-osfp4 @ git+https://github.com/neriahbd/OSFP4.git"
```

For development, install from a checkout:

```bash
git clone https://github.com/neriahbd/OSFP4.git
cd OSFP4
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Check that it works:

```bash
python -c "from llmcompressor_osfp4 import OSFP4Modifier; print('ok')"
```

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor_osfp4 import OSFP4Modifier

model_id = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_id)

recipe = OSFP4Modifier(
    scheme="NVFP4",            # or "NVFP4A16"
    optimization_mode="sic",   # or "rtn"
    targets=["Linear"],
    ignore=["lm_head"],
)

oneshot(
    model=model,
    processor=tokenizer,
    dataset=calibration_dataset,   # any dataset supported by oneshot
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

model.save_pretrained("Qwen3-0.6B-OSFP4", save_compressed=True)
tokenizer.save_pretrained("Qwen3-0.6B-OSFP4")
```

A complete, runnable version is [`examples/calibrate.py`](examples/calibrate.py).

Notes:

- `OSFP4Modifier` must be the only modifier in the recipe.
- OSFP4 automatically uses llm-compressor's sequential calibration pipeline.
- Save the checkpoint locally first. `save_pretrained(push_to_hub=True)` is
  not supported; upload the saved folder afterwards.
- To load YAML or string recipes that refer to `OSFP4Modifier` by name, run
  `import llmcompressor_osfp4` first.

## Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `scheme` | — | `"NVFP4"` or `"NVFP4A16"`. |
| `optimization_mode` | `"sic"` | `"sic"` or `"rtn"`. |
| `targets` | `["Linear"]` | Modules to quantize. |
| `ignore` | `[]` | Modules to skip (usually `["lm_head"]`). |
| `steps` | `80` | Adam steps for scale optimization. |
| `lr` | `0.12` | Adam learning rate. |
| `dampening_frac` | `0.01` | Hessian damping for SIC. |
| `offload_hessians` | `False` | Keep Hessians on the CPU between uses to save GPU memory. |
| `activation_subsample_size` | `"auto"` | Use up to the mapping's input width in activation rows; an integer sets a fixed cap and `None` uses all rows. |

When `activation_subsample_size` is omitted, it defaults to `"auto"`. Each
mapping retains `k = min(total_activation_rows, n)` complete activation vectors,
where `n = weight.shape[1]` is the shared input width of its balance layers.
MLP down projections use their intermediate input width. Mappings whose balance
layers have different input widths cannot use auto subsampling.

An integer sets a fixed row cap; pass `16384` to retain the previous sampling
policy. `None` uses all rows. `NVFP4A16` disables activation subsampling
regardless of this option. Subsampling limits optimization inputs; Hessians and
activation statistics still use all calibration rows.

The FP-Quant calibration CLI accepts `--activation-subsample-size auto` or an
integer and defaults to auto.

The [modifier README](src/llmcompressor_osfp4/modifiers/README.md) explains the
algorithm and failure behavior. The
[optimization README](src/llmcompressor_osfp4/modifiers/optimization/README.md)
explains the RTN and SIC math.

## Serving

Checkpoints are saved with `quant_method="osfp4"`. Layers that need runtime
activation smoothing store a BF16 `smooth_quant_scale`, and the checkpoint
config lists them in `osfp4_metadata`. To serve these checkpoints in vLLM, use
the matching `vllm-osfp4` plugin.

## Repository contents

```
src/llmcompressor_osfp4/
  modifiers/        OSFP4Modifier: calibration, smoothing, deployment
    optimization/   RTN and SIC scale optimization
  observers/        "osfp4" observer: FP4 math, losses, E4M3 scale selection
examples/
  calibrate.py      minimal end-to-end calibration
  wikitext_ppl/     WikiText-2 perplexity calibration and evaluation
  fpquant/          FP-Quant Table 1 (Llama 3.1 8B) and Table 7 (Qwen3-8B)
tests/              unit, integration, and golden-output tests
```

Run the examples from the repository root. Each example folder has its own
README with the exact commands.

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest -ra tests
```

CUDA tests are skipped automatically when no GPU is available.
`tests/test_golden.py` checks that calibration outputs match stored reference
tensors byte for byte.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
