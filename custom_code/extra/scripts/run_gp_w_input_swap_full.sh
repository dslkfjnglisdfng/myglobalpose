#!/usr/bin/env bash
set -euo pipefail

PY=/home/lingfeng/.conda/envs/globalpose-gpu/bin/python
ROOT=data/experiments/gp_w_input_swap_lag2_ema03_20260712
EVAL=custom_code/extra/evaluate_gp_w_input_swap.py
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}

if [[ ${1:-} == dip ]]; then
  cache=data/dataset_work/L4Cache/prephysics_pose_velocity_diptest_official_neural_only_offset_r/baseline_cache_manifest.json
  groups=(
    "s09_01_a s09_01_b s09_01_c s09_02_a s09_02_b"
    "s09_03_a s09_03_b s09_04 s09_05 s10_01_a"
    "s10_01_b s10_01_c s10_02 s10_03_a s10_03_b"
    "s10_04_a s10_04_b s10_04_c s10_05"
  )
elif [[ ${1:-} == totalcapture ]]; then
  cache=data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_test_official_neural_only/baseline_cache_manifest.json
  groups=("s5_freestyle1" "s5_freestyle3" "s5_rom3" "s5_walking2")
else
  echo "usage: $0 dip|totalcapture" >&2
  exit 2
fi

dataset=$1
pids=()
parts=()
for i in "${!groups[@]}"; do
  part="$ROOT/${dataset}_parts/part_$i"
  log="$ROOT/logs/${dataset}_part_$i.log"
  args=()
  for sequence in ${groups[$i]}; do args+=(--sequence "$sequence"); done
  "$PY" "$EVAL" --dataset "$dataset" --cache "$cache" --output-dir "$part" "${args[@]}" >"$log" 2>&1 &
  pids+=("$!")
  parts+=("$part")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" custom_code/extra/merge_gp_w_input_swap_parts.py --dataset "$dataset" --output-dir "$ROOT/${dataset}_test" "${parts[@]}"
