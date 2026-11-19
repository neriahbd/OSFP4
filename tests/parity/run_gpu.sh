#!/usr/bin/env bash
# Run from any directory. Each capture creates a fresh directory.
set -euo pipefail

: "${REF_PY:?Set REF_PY to the diagonal calibration Python executable}"
: "${PUBLIC_PY:?Set PUBLIC_PY to the public OSFP4 calibration Python executable}"
: "${NATIVE_SERVE_PY:?Set NATIVE_SERVE_PY to the native diagonal vLLM Python executable}"
: "${PUBLIC_SERVE_PY:?Set PUBLIC_SERVE_PY to the public plugin vLLM Python executable}"

suite=$(cd "$(dirname "$0")" && pwd)
out=${1:?Usage: run_gpu.sh NEW_OUTPUT_DIRECTORY}
mkdir -p "$(dirname "$out")"
mkdir "$out"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PUBLIC_PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA required in public calibration environment"'
"$REF_PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA required in reference calibration environment"'
"$PUBLIC_PY" -m pytest -q "$suite/../observers/test_chunked_scale_optimization.py" \
    "$suite/../modifiers/test_memory_and_cuda.py"

capture() {
    local py=$1 implementation=$2 name=$3
    shift 3
    "$py" "$suite/diagonal.py" capture "$implementation" "$out/$name" --device cuda --serving-fixture "$@"
}
compare() {
    "$PUBLIC_PY" "$suite/diagonal.py" compare "$out/$1" "$out/$2" --receipt "$out/$2-receipt.json"
}

capture "$REF_PY" diagonal reference
capture "$REF_PY" diagonal repeat
compare reference repeat
capture "$PUBLIC_PY" public public
compare reference public
for chunk in 7 65; do
    capture "$PUBLIC_PY" public "public-chunk-$chunk" --force-chunk "$chunk"
    compare reference "public-chunk-$chunk"
done

for projection in qkv o gate-up down; do
    capture "$REF_PY" diagonal "reference-$projection" --projection "$projection"
    capture "$PUBLIC_PY" public "public-$projection" --projection "$projection"
    compare "reference-$projection" "public-$projection"
done

for variant in NVFP4-rtn NVFP4-sic NVFP4A16-rtn NVFP4A16-sic; do
    for runtime in reference public; do
        py=$NATIVE_SERVE_PY
        if [[ $runtime == public ]]; then py=$PUBLIC_SERVE_PY; fi
        folder="$out/$runtime/$variant"
        "$py" "$suite/serve.py" "$folder/export" "$folder/serving.safetensors"
    done
done
"$PUBLIC_PY" "$suite/diagonal.py" compare "$out/reference" "$out/public" --serving --receipt "$out/serving-receipt.json"
