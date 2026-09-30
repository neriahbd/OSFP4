# llmcompressor-osfp4

OSFP4 (optimized-scale NVFP4) quantization for
[LLM Compressor](https://github.com/vllm-project/llm-compressor), packaged as a
plugin that runs on the **unmodified upstream release**. It provides
`OSFP4Modifier` (A4W4 `NVFP4` and weight-only `NVFP4A16`, with RTN or SIC scale
optimization) and the `"osfp4"` observer.

## Supported versions

| Component | Version |
| --- | --- |
| `llmcompressor` | **0.14.0** (exact pin) |
| `compressed-tensors` | 0.19.0 (pinned by llmcompressor 0.14.0) |
| `transformers` | >=5.15.0, <=5.17.0 |
| `torch` | >=2.10.0, <=2.14.0 |
| Python | >=3.10 |

The plugin subclasses llm-compressor internals (`Modifier`,
`QuantizationMixin`, `Observer`), so it is pinned to one release. Other
versions are not supported until the test suite passes against them.

## Install

```bash
pip install "llmcompressor-osfp4 @ git+<this-repo-url>"
# or, from a checkout
pip install -e ".[dev]"
```

## Usage

```python
from llmcompressor import oneshot
from llmcompressor_osfp4 import OSFP4Modifier

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
    num_calibration_samples=1024,
)
model.save_pretrained(save_dir, save_compressed=True)
```

The sequential calibration pipeline is selected automatically.
`OSFP4Modifier` must be the only modifier in the recipe.

**YAML or string recipes:** run `import llmcompressor_osfp4` before loading
the recipe. The import registers `OSFP4Modifier` with llm-compressor's
`ModifierFactory`, so recipes can refer to it by name.

For all options and the algorithm, see the
[modifier README](src/llmcompressor_osfp4/modifiers/README.md) and the
[optimization README](src/llmcompressor_osfp4/modifiers/optimization/README.md).
Checkpoints are served with the matching `vllm-osfp4` plugin.

## Examples

See [`examples/`](examples/README.md):

- a minimal calibration script
- the NestQuant WikiText-2 perplexity protocol
- FP-Quant Table 1 and Table 7 reproductions

Run them from the repository root, for example
`python examples/calibrate.py`.

## Tests

```bash
python -m pytest -ra tests
```

CUDA tests skip when CUDA is unavailable. `tests/test_golden.py` compares the
plugin's outputs byte for byte with fixtures captured from the original fork.
The committed fixtures cover `NVFP4` and `NVFP4A16`, each with RTN and SIC.
They were generated with the fork at `fc1a72f` on compressed-tensors
`0.18.1a20260910`, torch 2.12.0 and transformers 5.12.1 (CPU, Python 3.12).
To regenerate them, run this in an environment with the fork installed:

```bash
python tests/golden/generate_golden.py --output tests/golden/fixtures
```

## Provenance

Ported from the `neriahbd/llmcompressor-osfp4` fork. That fork was based on
upstream llm-compressor `main` shortly after the 0.12.0 release
(compressed-tensors 0.17.1). The OSFP4 algorithms are unchanged. The port made
these changes:

- **Packaging:** the modules moved from `llmcompressor.modifiers.osfp4` and
  `llmcompressor.observers.osfp4` to `llmcompressor_osfp4.modifiers` and
  `llmcompressor_osfp4.observers`.
- **Pipeline selection:** the fork patched upstream's `pipelines/registry.py`.
  The plugin instead sets `OSFP4Modifier.requires_calibration_data = True`,
  which llm-compressor 0.14.0 uses to infer the pipeline.
- **Recipes:** `OSFP4Modifier` is registered with `ModifierFactory` so recipes
  can refer to it by name.

## License

Apache-2.0, like LLM Compressor. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
