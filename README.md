# OSFP4

[![Paper](https://img.shields.io/badge/arXiv-2610.08231-b31b1b.svg)](https://arxiv.org/abs/2610.08231)


OSFP4 provides an NVFP4 quantization modifier for
[LLM Compressor](https://github.com/vllm-project/llm-compressor).

Paper: [*WSFP4: Joint Optimization of Diagonal Smoothing and Block Scales for NVFP4 Quantization*](https://arxiv.org/abs/2610.08231)


## Install

```bash
pip install "llmcompressor-osfp4 @ git+https://github.com/neriahbd/OSFP4.git"
```

## Quick start

With your prepared `calibration_dataset`:

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
    dataset=calibration_dataset,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

model.save_pretrained("Qwen3-0.6B-OSFP4", save_compressed=True)
tokenizer.save_pretrained("Qwen3-0.6B-OSFP4")
```

OSFP4 must be the only modifier in the recipe; sequential calibration is
selected automatically.

The [complete calibration example](examples/calibrate.py) loads a model and
calibration dataset, runs quantization, and saves the model and tokenizer.

## Requirements

| Component | Version |
| --- | --- |
| Python | 3.10 or newer |
| `llmcompressor` | 0.14.0 (installed automatically) |
| `compressed-tensors` | 0.19.0 (installed automatically) |
| `torch` | 2.10–2.14 |
| `transformers` | 5.15–5.17 |

A CUDA GPU is recommended for real models. CPU works for small models and tests.

## Options

| Scheme | Weights | Activations |
| --- | --- | --- |
| `NVFP4` | FP4 | FP4 |
| `NVFP4A16` | FP4 | 16-bit |

Scale optimization modes:

- **`sic`** (default): Hessian-aware. Quantizes block by block and propagates
  the error. More accurate, slower.
- **`rtn`**: faster. Uses a diagonal activation statistic.

| Parameter | Default | Description |
| --- | --- | --- |
| `scheme` | `None` | Set `"NVFP4"` or `"NVFP4A16"`. |
| `optimization_mode` | `"sic"` | `"sic"` or `"rtn"`. |
| `steps` | `80` | Adam optimization steps per mapping. |
| `lr` | `0.12` | Adam learning rate. |
| `dampening_frac` | `0.01` | SIC Hessian damping fraction. |
| `offload_hessians` | `False` | Keep SIC Hessians on the CPU between uses. |
| `activation_subsample_size` | `"auto"` | Optimization row cap: twice the mapping input width, an integer, or `None` for all rows. |

`activation_subsample_size` defaults to `"auto"`, retaining
`k = min(total_rows, 2 * mapping_input_width)` complete activation vectors per
mapping. The width is the layers' shared `weight.shape[1]`; MLP down projections
use their intermediate input width.

An integer sets a fixed row cap; `None` uses all rows. `NVFP4A16` disables
activation subsampling. Sampling limits optimization inputs; Hessians and
activation statistics still use all calibration rows.

<details>
<summary>Inherited LLM Compressor configuration</summary>

| Parameter | Default | Description |
| --- | --- | --- |
| `config_groups` | `None` | Explicit quantization scheme groups and targets, instead of `scheme`. |
| `kv_cache_scheme` | `None` | KV-cache quantization configuration. |
| `weight_observer` | `None` | OSFP4 sets the weight observer to `"osfp4"` during configuration. |
| `input_observer` | `None` | Override the input activation observer. |
| `output_observer` | `None` | Override the output activation observer. |
| `observer` | `None` | Observer dictionary with `weights`, `input`, and `output` keys; alternative to individual observer fields. OSFP4 sets the weight observer. |
| `bypass_divisibility_checks` | `False` | Skip LLM Compressor's group-size checks; OSFP4 still requires widths divisible by 16. |
| `requires_calibration_data` | `True` | Calibration requirement flag; keep `True` for OSFP4. |
| `index` | `None` | Modifier ordering metadata. |
| `group` | `None` | Modifier group name. |
| `start` | `None` | Inherited start step; use the default for calibration. |
| `end` | `None` | Inherited end step; use the default for calibration. |
| `update` | `None` | Inherited update step; use the default for calibration. |

Defaults are shown before initialization. Lifecycle state flags are managed by
LLM Compressor.

</details>

## Serving with vLLM

Stock vLLM cannot load OSFP4 checkpoints. Serve them with the
[vllm-osfp4](https://github.com/neriahbd/vllm-osfp4) plugin, which needs
vLLM 0.24–0.28 and a CUDA GPU with NVFP4 support.

See the [vllm-osfp4 README](https://github.com/neriahbd/vllm-osfp4#readme)
for installation, checkpoint requirements, and serving instructions.

Some layers, such as `o_proj` and `down_proj`, have no preceding layer to
absorb their smoothing scale. OSFP4 stores that scale in the checkpoint as
`smooth_quant_scale`, and the plugin multiplies it into the layer's input before
running vLLM's stock NVFP4 kernels.

Install the plugin in the vLLM environment, then serve:

```bash
pip install "git+https://github.com/neriahbd/vllm-osfp4.git"
VLLM_PLUGINS=osfp4 vllm serve ./Qwen3-0.6B-OSFP4
```

Saved checkpoints declare `quant_method="osfp4"`; vLLM selects it automatically.
Save locally with `save_pretrained(..., save_compressed=True)` before uploading.

## Performance

Mean accuracy (%) of Llama-3.1-8B-Instruct, averaged over WinoGrande,
HellaSwag, GSM8K and MMLU-CoT. **Bold** marks the best quantized result in each
group. The BF16 (unquantized) baseline scores **79.22**.

**W4A4, absmax activation scaling**

| Method | Avg. |
| --- | ---: |
| RTN (FP-Quant) | 75.65 |
| GPTQ (FP-Quant) | 76.34 |
| MR-GPTQ (FP-Quant) | 76.18 |
| SOAR | 76.60 |
| H-Scale (W4A4 extension) | 76.67 |
| NVIDIA released checkpoint | 76.11 |
| OSFP4 W-RTN / X-RTN | 76.36 |
| OSFP4 W-SIC / X-RTN | **77.03** |

**W4A16, weights only**

| Method | Avg. |
| --- | ---: |
| RTN | 77.67 |
| H-Scale | 78.00 |
| OSFP4 W-RTN | 78.15 |
| OSFP4 W-SIC | **78.42** |

The full experimental setup and additional experiments are in the paper.

## Documentation and license

The [modifier README](src/llmcompressor_osfp4/modifiers/README.md) explains the
algorithm and failure behavior. The
[optimization README](src/llmcompressor_osfp4/modifiers/optimization/README.md)
explains the RTN and SIC math.

See [examples](examples/README.md) for calibration and evaluation workflows.

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
