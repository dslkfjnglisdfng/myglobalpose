#!/usr/bin/env bash
set -euo pipefail

ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
PY="$ENV/bin/python"

SMOOTH_WINDOW=${SMOOTH_WINDOW:-9}
SMOOTH_MODE=${SMOOTH_MODE:-centered_moving_average}
DEFAULT_ROOT=data/experiments/newpl_v5_smoothacc_20260612
if [[ "${SMOKE:-0}" == "1" && -z "${ROOT:-}" ]]; then
  DEFAULT_ROOT="${DEFAULT_ROOT}_smoke"
fi
ROOT=${ROOT:-$DEFAULT_ROOT}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/caches}
EVAL_ROOT="$ROOT/eval"
LOG_ROOT="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

if [[ "${NO_TEE:-0}" != "1" ]]; then
  exec > >(tee -a "$LOG_ROOT/run.log") 2>&1
fi

echo "ROOT=$ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "SMOOTH_WINDOW=$SMOOTH_WINDOW"

AMASS_RAW=data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
DIP_TRAIN_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json
DIP_VAL_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json
DIP_TEST_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
TC_TEST_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json

AMASS_CONTROL=data/dataset_work/GTControlCache/amass_train/gt_control_cache_manifest.json
DIP_TRAIN_CONTROL=data/dataset_work/GTControlCache/dip_train/gt_control_cache_manifest.json
DIP_VAL_CONTROL=data/dataset_work/GTControlCache/dip_val/gt_control_cache_manifest.json

NEWPL_V4=data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
RAW_V5_AMASS=data/experiments/newpl_v5_official_protocol_20260607_tuned/amass_pretrain/best_loss.pt
RAW_V5_DIP=data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt

for required in "$AMASS_RAW" "$DIP_TRAIN_RAW" "$DIP_VAL_RAW" "$DIP_TEST_RAW" "$TC_TEST_RAW" "$AMASS_CONTROL" "$DIP_TRAIN_CONTROL" "$DIP_VAL_CONTROL" "$NEWPL_V4" "$RAW_V5_AMASS" "$RAW_V5_DIP"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done

if [[ "${SMOKE:-0}" == "1" ]]; then
  CACHE_MAX=${CACHE_MAX:-8}
  AMASS_EPOCHS=${AMASS_EPOCHS:-1}
  DIP_EPOCHS=${DIP_EPOCHS:-1}
  AMASS_BATCH=${AMASS_BATCH:-16}
  DIP_BATCH=${DIP_BATCH:-16}
  AMASS_MAX_TRAIN=${AMASS_MAX_TRAIN:-8}
  AMASS_MAX_VAL=${AMASS_MAX_VAL:-2}
  AMASS_VAL_WINDOW=${AMASS_VAL_WINDOW:-61}
  DIP_MAX_TRAIN=${DIP_MAX_TRAIN:-8}
  DIP_MAX_VAL=${DIP_MAX_VAL:-2}
  DIP_VAL_WINDOW=${DIP_VAL_WINDOW:-61}
  EVAL_MAX=${EVAL_MAX:-4}
  AMASS_EVAL_MAX=${AMASS_EVAL_MAX:-4}
else
  CACHE_MAX=${CACHE_MAX:-0}
  AMASS_EPOCHS=${AMASS_EPOCHS:-80}
  DIP_EPOCHS=${DIP_EPOCHS:-40}
  AMASS_BATCH=${AMASS_BATCH:-256}
  DIP_BATCH=${DIP_BATCH:-24}
  AMASS_MAX_TRAIN=${AMASS_MAX_TRAIN:-0}
  AMASS_MAX_VAL=${AMASS_MAX_VAL:-20}
  AMASS_VAL_WINDOW=${AMASS_VAL_WINDOW:-61}
  DIP_MAX_TRAIN=${DIP_MAX_TRAIN:-0}
  DIP_MAX_VAL=${DIP_MAX_VAL:-0}
  DIP_VAL_WINDOW=${DIP_VAL_WINDOW:-61}
  EVAL_MAX=${EVAL_MAX:-0}
  AMASS_EVAL_MAX=${AMASS_EVAL_MAX:-20}
fi

ensure_smooth_cache() {
  local input_manifest="$1"
  local output_dir="$2"
  if [[ ! -f "$output_dir/baseline_cache_manifest.json" ]]; then
    local max_args=()
    if [[ "$CACHE_MAX" != "0" ]]; then
      max_args+=(--max-sequences "$CACHE_MAX")
    fi
    "$PY" scripts/build_smooth_acc_cache.py \
      --input-cache "$input_manifest" \
      --output-dir "$output_dir" \
      --window "$SMOOTH_WINDOW" \
      --mode "$SMOOTH_MODE" \
      --notes "NewPL v5 smooth-acc experiment; replace aM only, keep wM/RMB/targets unchanged." \
      "${max_args[@]}"
  fi
}

ensure_pl_cache() {
  local input_manifest="$1"
  local output_dir="$2"
  if [[ ! -f "$output_dir/pl_curve_cache_manifest.json" ]]; then
    "$PY" pl_curve_cache.py \
      --input-cache "$input_manifest" \
      --output-dir "$output_dir" \
      --imu-input-mode official \
      --feature-mode legacy \
      --shard-size 100
  fi
}

ensure_smooth_cache "$AMASS_RAW" "$CACHE_ROOT/raw_amass_smooth_w${SMOOTH_WINDOW}"
ensure_smooth_cache "$DIP_TRAIN_RAW" "$CACHE_ROOT/raw_dip_train_smooth_w${SMOOTH_WINDOW}"
ensure_smooth_cache "$DIP_VAL_RAW" "$CACHE_ROOT/raw_dip_val_smooth_w${SMOOTH_WINDOW}"
ensure_smooth_cache "$DIP_TEST_RAW" "$CACHE_ROOT/raw_dip_test_smooth_w${SMOOTH_WINDOW}"
ensure_smooth_cache "$TC_TEST_RAW" "$CACHE_ROOT/raw_tc_test_smooth_w${SMOOTH_WINDOW}"

ensure_pl_cache "$CACHE_ROOT/raw_amass_smooth_w${SMOOTH_WINDOW}/baseline_cache_manifest.json" "$CACHE_ROOT/pl_amass_smoothacc_init36"
ensure_pl_cache "$CACHE_ROOT/raw_dip_train_smooth_w${SMOOTH_WINDOW}/baseline_cache_manifest.json" "$CACHE_ROOT/pl_dip_train_smoothacc_init36"
ensure_pl_cache "$CACHE_ROOT/raw_dip_val_smooth_w${SMOOTH_WINDOW}/baseline_cache_manifest.json" "$CACHE_ROOT/pl_dip_val_smoothacc_init36"
ensure_pl_cache "$CACHE_ROOT/raw_dip_test_smooth_w${SMOOTH_WINDOW}/baseline_cache_manifest.json" "$CACHE_ROOT/pl_dip_test_smoothacc_init36"
ensure_pl_cache "$CACHE_ROOT/raw_tc_test_smooth_w${SMOOTH_WINDOW}/baseline_cache_manifest.json" "$CACHE_ROOT/pl_tc_test_smoothacc_init36"

COMMON_TRAIN_ARGS=(
  --init-size 36
  --window 61
  --hidden-size 512
  --tail-length 4
  --residual-scale 0.005
  --dropout 0.4
  --grad-clip 1.0
  --disable-ik-distill
  --baseline-pRB-weight 0.0
  --baseline-gR1-weight 0.0
  --gt-control-pRB-weight 0.3
  --gt-control-gR1-weight 0.1
  --pRB-ddot-smooth-weight 0.000001
  --gR1-dot-weight 0.03
  --gR1-ddot-weight 0.001
  --selection-metric control_physical
)

echo "Training batches: AMASS_BATCH=$AMASS_BATCH DIP_BATCH=$DIP_BATCH"

RUN_SMOKE=${RUN_SMOKE:-1}
if [[ "$RUN_SMOKE" == "1" && ! -f "$ROOT/smoke/best_loss.pt" ]]; then
  "$PY" pl_curve_train.py \
    --train-cache "$CACHE_ROOT/pl_dip_train_smoothacc_init36/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/pl_dip_val_smoothacc_init36/pl_curve_cache_manifest.json" \
    --train-gt-control-cache "$DIP_TRAIN_CONTROL" \
    --val-gt-control-cache "$DIP_VAL_CONTROL" \
    --output-dir "$ROOT/smoke" \
    --experiment-name newpl_v5_smoothacc_smoke \
    --epochs 1 \
    --lr 1e-5 \
    --batch-size 16 \
    --max-train-sequences 8 \
    --max-val-sequences 2 \
    "${COMMON_TRAIN_ARGS[@]}"
fi

if [[ ! -f "$ROOT/amass_pretrain/best_loss.pt" ]]; then
  "$PY" pl_curve_train.py \
    --train-cache "$CACHE_ROOT/pl_amass_smoothacc_init36/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/pl_amass_smoothacc_init36/pl_curve_cache_manifest.json" \
    --train-gt-control-cache "$AMASS_CONTROL" \
    --val-gt-control-cache "$AMASS_CONTROL" \
    --output-dir "$ROOT/amass_pretrain" \
    --experiment-name newpl_v5_smoothacc_amass_pretrain \
    --epochs "$AMASS_EPOCHS" \
    --lr 1e-4 \
    --batch-size "$AMASS_BATCH" \
    --max-train-sequences "$AMASS_MAX_TRAIN" \
    --max-val-sequences "$AMASS_MAX_VAL" \
    --val-window-length "$AMASS_VAL_WINDOW" \
    --early-stop-min-delta 0.00000005 \
    --early-stop-patience 12 \
    "${COMMON_TRAIN_ARGS[@]}"
fi

eval_max_args=()
if [[ "$EVAL_MAX" != "0" ]]; then
  eval_max_args+=(--max-eval-sequences "$EVAL_MAX")
fi
amass_eval_max_args=()
if [[ "$AMASS_EVAL_MAX" != "0" ]]; then
  amass_eval_max_args+=(--max-eval-sequences "$AMASS_EVAL_MAX")
fi

"$PY" newpl_root_eval.py \
  --cache "$CACHE_ROOT/raw_amass_smooth_w${SMOOTH_WINDOW}/baseline_cache_manifest.json" \
  --output-json "$EVAL_ROOT/amass_after_amass_pretrain_smoothinput.json" \
  --dataset amass \
  --dataset-label AMASS-smoothacc-proxy-val \
  --imu-input-mode official \
  "${amass_eval_max_args[@]}" \
  --version official_PL_smoothacc=official \
  --version newpl_v4_init36_smoothacc="$NEWPL_V4" \
  --version newpl_v5_raw_amass_smoothinput="$RAW_V5_AMASS" \
  --version newpl_v5_smoothacc_amass_best="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_v5_smoothacc_amass_last="$ROOT/amass_pretrain/last.pt"

"$PY" newpl_root_eval.py \
  --cache "$CACHE_ROOT/raw_dip_test_smooth_w${SMOOTH_WINDOW}/baseline_cache_manifest.json" \
  --output-json "$EVAL_ROOT/dip_test_after_amass_pretrain_smoothinput.json" \
  --dataset dip \
  --dataset-label DIP-IMU-test-smoothacc-input \
  --imu-input-mode official \
  "${eval_max_args[@]}" \
  --version official_PL_smoothacc=official \
  --version newpl_v4_init36_smoothacc="$NEWPL_V4" \
  --version newpl_v5_raw_amass_smoothinput="$RAW_V5_AMASS" \
  --version newpl_v5_smoothacc_amass_best="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_v5_smoothacc_amass_last="$ROOT/amass_pretrain/last.pt"

"$PY" newpl_root_eval.py \
  --cache "$CACHE_ROOT/raw_tc_test_smooth_w${SMOOTH_WINDOW}/baseline_cache_manifest.json" \
  --output-json "$EVAL_ROOT/tc_test_after_amass_pretrain_smoothinput.json" \
  --dataset totalcapture \
  --dataset-label TotalCapture-test-smoothacc-input \
  --imu-input-mode official \
  "${eval_max_args[@]}" \
  --version official_PL_smoothacc=official \
  --version newpl_v4_init36_smoothacc="$NEWPL_V4" \
  --version newpl_v5_raw_amass_smoothinput="$RAW_V5_AMASS" \
  --version newpl_v5_smoothacc_amass_best="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_v5_smoothacc_amass_last="$ROOT/amass_pretrain/last.pt"

if [[ ! -f "$ROOT/dip_finetune/best_loss.pt" ]]; then
  "$PY" pl_curve_train.py \
    --train-cache "$CACHE_ROOT/pl_dip_train_smoothacc_init36/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/pl_dip_val_smoothacc_init36/pl_curve_cache_manifest.json" \
    --train-gt-control-cache "$DIP_TRAIN_CONTROL" \
    --val-gt-control-cache "$DIP_VAL_CONTROL" \
    --output-dir "$ROOT/dip_finetune" \
    --experiment-name newpl_v5_smoothacc_amass_to_dip_finetune \
    --epochs "$DIP_EPOCHS" \
    --lr 5e-6 \
    --batch-size "$DIP_BATCH" \
    --max-train-sequences "$DIP_MAX_TRAIN" \
    --max-val-sequences "$DIP_MAX_VAL" \
    --val-window-length "$DIP_VAL_WINDOW" \
    --init-checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
    --early-stop-min-delta 0.00000005 \
    --early-stop-patience 10 \
    "${COMMON_TRAIN_ARGS[@]}"
fi

"$PY" newpl_root_eval.py \
  --cache "$CACHE_ROOT/raw_dip_test_smooth_w${SMOOTH_WINDOW}/baseline_cache_manifest.json" \
  --output-json "$EVAL_ROOT/dip_test_after_dip_finetune_smoothinput.json" \
  --dataset dip \
  --dataset-label DIP-IMU-test-smoothacc-input \
  --imu-input-mode official \
  "${eval_max_args[@]}" \
  --version official_PL_smoothacc=official \
  --version newpl_v4_init36_smoothacc="$NEWPL_V4" \
  --version newpl_v5_raw_dip_smoothinput="$RAW_V5_DIP" \
  --version newpl_v5_smoothacc_amass_best="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_v5_smoothacc_dip_best="$ROOT/dip_finetune/best_loss.pt" \
  --version newpl_v5_smoothacc_dip_last="$ROOT/dip_finetune/last.pt"

"$PY" newpl_root_eval.py \
  --cache "$CACHE_ROOT/raw_tc_test_smooth_w${SMOOTH_WINDOW}/baseline_cache_manifest.json" \
  --output-json "$EVAL_ROOT/tc_test_after_dip_finetune_smoothinput.json" \
  --dataset totalcapture \
  --dataset-label TotalCapture-test-smoothacc-input \
  --imu-input-mode official \
  "${eval_max_args[@]}" \
  --version official_PL_smoothacc=official \
  --version newpl_v4_init36_smoothacc="$NEWPL_V4" \
  --version newpl_v5_raw_dip_smoothinput="$RAW_V5_DIP" \
  --version newpl_v5_smoothacc_amass_best="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_v5_smoothacc_dip_best="$ROOT/dip_finetune/best_loss.pt" \
  --version newpl_v5_smoothacc_dip_last="$ROOT/dip_finetune/last.pt"

"$PY" scripts/summarize_newpl_v5_smoothacc.py --root "$ROOT"

echo "done: $ROOT"
