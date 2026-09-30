# OSFP4 examples

This directory contains three supported workflows.

## Minimal calibration

[`calibrate.py`](calibrate.py) is the smallest end-to-end OSFP4 example. It
loads `Qwen/Qwen3-0.6B` in native BF16, calibrates SIC NVFP4 on 20 UltraChat
samples of at most 512 tokens, and saves `Qwen3-0.6B-OSFP4`.

```bash
python examples/calibrate.py
```

## NestQuant WikiText-2 perplexity

Calibration and evaluation scripts reproducing the NestQuant WikiText-2
perplexity protocol live under [`wikitext_ppl/`](wikitext_ppl/), for direct
comparability with released NestQuant/FP-Quant numbers. See that folder's
README for the full calibrate -> evaluate workflow.

## FP-Quant reproductions

The parallel Table 1 and Table 7 pipelines live under [`fpquant/`](fpquant/).
They share dataset, validation, evaluation, and protocol code while retaining
clear table-specific entrypoints and configuration.

Research scripts and paper-writing assets are maintained outside this checkout.

## Compatibility checks

With this plugin installed (`pip install -e ".[dev]"`), run the OSFP4 tests
from the repository root:

```bash
python -m pytest -ra tests
```

The command uses your current Python interpreter and installed dependencies.
CUDA tests run when CUDA is available and otherwise skip.
