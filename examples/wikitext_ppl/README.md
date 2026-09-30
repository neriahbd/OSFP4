# NestQuant WikiText-2 perplexity reproduction

Calibrates and evaluates OSFP4 checkpoints for direct comparability with
released NestQuant/FP-Quant WikiText-2 perplexity numbers: fixed 2048-token,
non-overlapping chunks over `meta-llama/Meta-Llama-3-8B`'s tokenizer, a
single BOS/EOS around the whole joined corpus, calibrated on the WikiText-2
*train* split rather than FineWeb-Edu.

Not the same protocol as [`fpquant/`](../fpquant/)'s Table 1/7 reproductions
(different dataset, different model, different metric) -- kept in its own
folder for that reason.

```bash
# 1. Build the WikiText-2 train calibration cache
python examples/wikitext_ppl/prepare_data.py \
  --output-root /workspace/other/runs/wikitext-ppl/calibration-data

# 2. Calibrate one method
python examples/wikitext_ppl/calibrate.py \
  --method osfp4-rtn \
  --calibration-data-root /workspace/other/runs/wikitext-ppl/calibration-data \
  --save-dir /workspace/other/runs/wikitext-ppl/a4-rtn/seed-42/model

# 3. Evaluate perplexity (HF or vLLM backend)
python examples/wikitext_ppl/evaluate.py \
  --backend vllm --kind osfp4 \
  --model /workspace/other/runs/wikitext-ppl/a4-rtn/seed-42/model \
  --output /workspace/other/runs/wikitext-ppl/a4-rtn/seed-42/eval/osfp4-a4-rtn-vllm.json
```

`prepare_data.py` and `calibrate.py` are thin wrappers around
[`fpquant/common/calibration.py`](../fpquant/common/calibration.py) that
retarget its dataset constants at WikiText-2 train before delegating --
everything else (recipe construction, checkpoint/manifest layout, CLI flags)
is identical to the FP-Quant table reproductions. `evaluate.py`'s protocol is
independent of that shared code (see its own docstring for why: it is not
comparable to `fpquant/`'s lm-eval-based accuracy protocol either).
