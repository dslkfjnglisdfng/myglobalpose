#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=${PY:-/home/lingfeng/.conda/envs/globalpose-gpu/bin/python}
export LD_LIBRARY_PATH="/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}"

OUT_ROOT=${OUT_ROOT:-data/experiments/footlock_transpose_rjs_smoothacc_20260609}
TRANSPOSE_ROOT=${TRANSPOSE_ROOT:-/home/lingfeng/projects/TransPose}
TRANSPOSE_WEIGHTS=${TRANSPOSE_WEIGHTS:-data/weights.pt}

DIP_TRAIN=${DIP_TRAIN:-data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json}
DIP_VAL=${DIP_VAL:-data/dataset_work/L4Cache/prephysics_pose_velocity_dip_val_globalpose_neural_only/baseline_cache_manifest.json}
DIP_TEST=${DIP_TEST:-data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json}

TC_TRAIN=${TC_TRAIN:-data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json}
TC_VAL=${TC_VAL:-data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json}
TC_TEST=${TC_TEST:-data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_test_official_neural_only_offset_r/baseline_cache_manifest.json}

COMMON_ARGS=(
  --method footlock_transpose_v1
  --device "${DEVICE:-cpu}"
  --transpose-root "$TRANSPOSE_ROOT"
  --transpose-weights "$TRANSPOSE_WEIGHTS"
  --contact-threshold "${CONTACT_THRESHOLD:-0.85}"
  --contact-margin "${CONTACT_MARGIN:-0.15}"
  --contact-selection-mode "${CONTACT_SELECTION_MODE:-transpose_winner}"
  --contact-height-margin "${CONTACT_HEIGHT_MARGIN:-0.08}"
  --transpose-prob-low "${TRANSPOSE_PROB_LOW:-0.5}"
  --transpose-prob-high "${TRANSPOSE_PROB_HIGH:-0.9}"
  --min-contact-frames "${MIN_CONTACT_FRAMES:-24}"
  --max-contact-frames "${MAX_CONTACT_FRAMES:-180}"
  --min-fit-frames "${MIN_FIT_FRAMES:-48}"
  --min-fit-improvement "${MIN_FIT_IMPROVEMENT:-0.05}"
  --max-condition-number "${MAX_CONDITION_NUMBER:-1e5}"
  --max-offset-norm "${MAX_OFFSET_NORM:-0.5}"
  --smooth-window "${SMOOTH_WINDOW:-9}"
  --derivative-mode "${DERIVATIVE_MODE:-centered}"
)

run_split() {
  local dataset="$1"
  local split="$2"
  local input="$3"
  "$PY" scripts/build_imu_position_offsets.py \
    --input "$input" \
    --output "$OUT_ROOT/${split}_footlock_transpose_rjs.pt" \
    --summary-json "$OUT_ROOT/${split}_summary.json" \
    --dataset "$dataset" \
    "${COMMON_ARGS[@]}"
}

run_split dip dip_train "$DIP_TRAIN"
run_split dip dip_val "$DIP_VAL"
run_split dip dip_test "$DIP_TEST"

if [[ "${RUN_TOTALCAPTURE:-1}" == "1" ]]; then
  run_split totalcapture totalcapture_train "$TC_TRAIN"
  run_split totalcapture totalcapture_val "$TC_VAL"
  run_split totalcapture totalcapture_test "$TC_TEST"
fi
