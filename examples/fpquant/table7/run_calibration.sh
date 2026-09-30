#!/usr/bin/env bash
# Run the reduced smoke job and full calibration for one Table 7 method.
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
run_root=${RUN_ROOT:-/workspace/other/runs/table7-runs/qwen3-8b}
method=${1:-osfp4-sic}
data_root=${CALIBRATION_DATA_ROOT:-"$run_root/calibration-data"}
calibrator="$repo_root/examples/fpquant/table7/calibrate.py"

case "$method" in
  osfp4-rtn|osfp4-sic) ;;
  *) echo "Unsupported Table 7 calibration method: $method" >&2; exit 2 ;;
esac
[[ -x "$python_bin" ]] || { echo "Python not found: $python_bin" >&2; exit 1; }
for seed in 0 42; do
  manifest="$data_root/seed-$seed/calibration-data-manifest.json"
  [[ -f "$manifest" ]] || { echo "Prepared cache not found: $manifest" >&2; exit 1; }
done

common=(--method "$method" --calibration-data-root "$data_root")
smoke_dir="$run_root/$method/smoke/model"
if [[ ! -f "$smoke_dir/calibration-manifest.json" ]]; then
  "$python_bin" -u "$calibrator" "${common[@]}" \
    --seed 0 --num-samples 2 --steps 1 --save-dir "$smoke_dir"
fi

checkpoint="$run_root/$method/seed-42/model"
if [[ -f "$checkpoint/calibration-manifest.json" ]]; then
  echo "Completed checkpoint already exists: $checkpoint"
  exit 0
fi
"$python_bin" -u "$calibrator" "${common[@]}" --save-dir "$checkpoint"
