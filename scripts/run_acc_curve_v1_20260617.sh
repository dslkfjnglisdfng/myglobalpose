#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="data/experiments/acc_curve_v1_20260617"
CACHE_ROOT="code/outputs/smooth_acc_cache_amass_dip_20260617"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

python scripts/build_acc_curve_cache.py \
  --all-defaults \
  --output-root "${CACHE_ROOT}" \
  --device cuda \
  --shard-size 32 \
  --progress-every 25 \
  --overwrite

python acc_curve_train.py \
  --amass-cache "${CACHE_ROOT}/amass_train/acc_curve_cache_manifest.json" \
  --dip-train-cache "${CACHE_ROOT}/dip_train/acc_curve_cache_manifest.json" \
  --dip-val-cache "${CACHE_ROOT}/dip_val/acc_curve_cache_manifest.json" \
  --dip-test-cache "${CACHE_ROOT}/dip_test/acc_curve_cache_manifest.json" \
  --output-dir "${OUT_ROOT}" \
  --epochs 30 \
  --dip-epochs 20 \
  --window 240 \
  --stride 120 \
  --batch-size 64 \
  --num-workers 8 \
  --hidden-size 512 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --control-prior-weight 1e-5 \
  --resume
