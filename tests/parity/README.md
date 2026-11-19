# Byte parity against the LLM Compressor diagonal branch

Run from the public OSFP4 checkout. Use the same hardware, PyTorch and Transformers
versions in both calibration environments. Set `REF_PY` to the Python executable
with LLM Compressor installed from a clean checkout of `osfp4-diagonal-transform`
at `6a650621ac86842a3d8c08bb76b9350ebb5112d5` and compressed-tensors at
`30d017eec16e3d365be66017cbc1a57782ff7bcb`. Set `PUBLIC_PY` to the
Python executable with this checkout installed via `pip install -e '.[dev]'`.
Use a new output directory for each capture. Model weights and tokens are
generated locally from fixed seeds. No Hugging Face downloads are needed.

```bash
"$REF_PY" tests/parity/diagonal.py capture diagonal output/reference
"$REF_PY" tests/parity/diagonal.py capture diagonal output/repeat
"$PUBLIC_PY" tests/parity/diagonal.py compare output/reference output/repeat
"$PUBLIC_PY" tests/parity/diagonal.py capture public output/public
"$PUBLIC_PY" tests/parity/diagonal.py compare output/reference output/public
```

To check every production Python file in the public calibration package against the pinned Git tree, run

```bash
python3 tests/parity/check_source.py /path/to/llm-compressor-for-osfp4
```

The checker compares shared modules byte for byte after translating package imports. In the adapted modifier and smoothing modules it checks shared field names, method signatures, and the bodies of shared calibration methods. It also checks for unexpected public files.

The tensor comparison covers a two-layer Qwen3 model with grouped attention in
NVFP4 and NVFP4A16, each with RTN and SIC. It checks the input state, sampled
indices, calibration statistics, optimizer inputs, initial and learned scales,
rounded alpha, selected scales, zero points, SIC weights, a fixed-input forward
pass before finalization, the deployed state dict, and the packed safetensors
export. The public runtime alpha is matched to the diagonal transform weight
by module name. Every shape, dtype and tensor byte must agree. The two formats
retain their own configuration and tensor naming.

A separate joint and weight-only fixture in each capture exceeds the actual
256 MiB graph threshold. Repeat the public capture with `--force-chunk 7`
and `--force-chunk 65`, comparing each to the default reference above.
These settings give an uneven final chunk and one whole-tensor chunk.

## GPU verification

With the four Python executables set as described above and below, run the
CUDA optimizer tests, four Qwen3-8B projection fixtures, checkpoint byte
comparisons, and vLLM serving probes with one command:

```bash
bash tests/parity/run_gpu.sh output/gpu-parity
```

Use a fresh output directory. The runner stops on the first failure and
writes tensor hashes for each successful comparison. The individual commands
below can be used to isolate a failing stage.

Repeat the capture commands with `--device cuda --serving-fixture` in both
environments and fresh directories. This selects a wider model with aligned
projections for the serving kernels. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`
in both environments before starting Python. To exercise production Qwen3-8B
projection shapes, add the same `--projection qkv`, `o`, `gate-up`, or
`down` to each paired capture. CUDA captures record peak allocated bytes for
each objective in `peak_allocated_bytes.json`. Use a GPU with sufficient
memory for a whole-tensor comparison of the chosen projection.

Run each exported checkpoint through its corresponding serving environment.
Use a native diagonal vLLM environment for the reference and stock vLLM with
the public plugin at `4f681ffd7fbbef580760267f6172fdc01058433c` for the
public candidate. Run each command in a separate process to release GPU memory.
The probes use explicit token IDs and save the greedy token and next-token
log probabilities in vocabulary order.

```bash
for case in NVFP4-rtn NVFP4-sic NVFP4A16-rtn NVFP4A16-sic; do
  "$NATIVE_SERVE_PY" tests/parity/serve.py "output/reference-gpu/$case/export" "output/reference-gpu/$case/serving.safetensors" || exit 1
  "$PUBLIC_SERVE_PY" tests/parity/serve.py "output/public-gpu/$case/export" "output/public-gpu/$case/serving.safetensors" || exit 1
done
"$PUBLIC_PY" tests/parity/diagonal.py compare output/reference-gpu output/public-gpu --serving
```

Also run the serving plugin's own `tests/test_plugin.py` in its installed
vLLM environment; its delegation test checks that alpha is multiplied once.
