#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_DIR="${ENV_DIR:-/home/lingfeng/.conda/envs/globalpose-gpu}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"

VERSION="acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617"
OUT_ROOT="${OUT_ROOT:-data/experiments/${VERSION}}"
CACHE_ROOT="${CACHE_ROOT:-code/outputs/${VERSION}}"
SMOKE_OUT_ROOT="${SMOKE_OUT_ROOT:-data/experiments/${VERSION}_smoke}"
SMOKE_CACHE_ROOT="${SMOKE_CACHE_ROOT:-code/outputs/${VERSION}_smoke}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL_CACHE="${RUN_FULL_CACHE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
USE_LONGRUN="${USE_LONGRUN:-1}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  for preset in amass_train dip_train dip_val dip_test; do
    "${PYTHON}" scripts/build_acc_curve_gtfk_cache.py \
      --preset "${preset}" \
      --output-root "${SMOKE_CACHE_ROOT}" \
      --max-sequences 2 \
      --max-frames 180 \
      --shard-size 2 \
      --progress-every 1 \
      --overwrite
  done

  "${PYTHON}" acc_curve_train.py \
    --amass-cache "${SMOKE_CACHE_ROOT}/amass_train/acc_curve_gtfk_cache_manifest.json" \
    --dip-train-cache "${SMOKE_CACHE_ROOT}/dip_train/acc_curve_gtfk_cache_manifest.json" \
    --dip-val-cache "${SMOKE_CACHE_ROOT}/dip_val/acc_curve_gtfk_cache_manifest.json" \
    --dip-test-cache "${SMOKE_CACHE_ROOT}/dip_test/acc_curve_gtfk_cache_manifest.json" \
    --output-dir "${SMOKE_OUT_ROOT}" \
    --target-key aFK_gtfk_smooth \
    --epochs 1 \
    --dip-epochs 1 \
    --window 120 \
    --stride 60 \
    --batch-size 2 \
    --num-workers 0 \
    --hidden-size 64 \
    --overwrite
fi

if [[ "${RUN_FULL_CACHE}" == "1" ]]; then
  "${PYTHON}" scripts/build_acc_curve_gtfk_cache.py \
    --all-defaults \
    --output-root "${CACHE_ROOT}" \
    --shard-size 32 \
    --progress-every 25 \
    --overwrite
fi

TRAIN_CMD=(
  "${PYTHON}" acc_curve_train.py
  --amass-cache "${CACHE_ROOT}/amass_train/acc_curve_gtfk_cache_manifest.json"
  --dip-train-cache "${CACHE_ROOT}/dip_train/acc_curve_gtfk_cache_manifest.json"
  --dip-val-cache "${CACHE_ROOT}/dip_val/acc_curve_gtfk_cache_manifest.json"
  --dip-test-cache "${CACHE_ROOT}/dip_test/acc_curve_gtfk_cache_manifest.json"
  --output-dir "${OUT_ROOT}"
  --target-key aFK_gtfk_smooth
  --epochs 30
  --dip-epochs 20
  --window 240
  --stride 120
  --batch-size 64
  --num-workers 8
  --hidden-size 512
  --lr 1e-4
  --weight-decay 1e-4
  --control-prior-weight 1e-5
  --resume
)

printf '%q ' "${TRAIN_CMD[@]}" > "${LOG_DIR}/train_command.sh"
printf '\n' >> "${LOG_DIR}/train_command.sh"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  if [[ "${USE_LONGRUN}" == "1" && -x /home/lingfeng/bin/longrun ]]; then
    /home/lingfeng/bin/longrun -- "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_DIR}/run.log"
  else
    "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_DIR}/run.log"
  fi
fi
