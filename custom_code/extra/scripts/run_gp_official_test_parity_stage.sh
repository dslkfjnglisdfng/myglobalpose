#!/usr/bin/env bash
set -euo pipefail

stage=${1:?usage: $0 baseline_g0|g2}
root=$(cd "$(dirname "$0")/../../.." && pwd)
baseline=/home/lingfeng/projects/GlobalposeMy/gp_official_parity_baseline_90523d6
out="$root/data/experiments/gp_w_input_swap_official_test_parity_20260712"
py=/home/lingfeng/.conda/envs/globalpose-gpu/bin/python
runner="$root/custom_code/extra/run_official_test_parity.py"

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}"

run_one() {
  local variant=$1 repo=$2 dataset=$3
  local variant_dir="$out/$variant"
  local run_dir="$variant_dir/${dataset}_run"
  mkdir -p "$variant_dir"
  if [[ -e "$run_dir" || -e "$variant_dir/$dataset.log" ]]; then
    echo "refusing to overwrite $run_dir or $variant_dir/$dataset.log" >&2
    exit 2
  fi
  "$py" -u "$runner" \
    --repo-root "$repo" \
    --variant "$variant" \
    --dataset "$dataset" \
    --output-dir "$run_dir" \
    >"$variant_dir/$dataset.log" 2>&1
}

case "$stage" in
  baseline_g0)
    run_one baseline_original "$baseline" dip
    run_one baseline_original "$baseline" totalcapture
    run_one current_g0 "$root" dip
    run_one current_g0 "$root" totalcapture
    ;;
  g2)
    run_one g2_vr_swap "$root" dip
    run_one g2_vr_swap "$root" totalcapture
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
