#!/usr/bin/env bash
set -euo pipefail

ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
PY="$ENV/bin/python"

FILTER_MODE=${FILTER_MODE:-causal_iir}
CUTOFF_HZ=${CUTOFF_HZ:-20}
FILTER_FS=${FILTER_FS:-60}
FILTER_ORDER=${FILTER_ORDER:-2}

DEFAULT_ROOT=data/experiments/newpl_v5_realtime_residual_20260612
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
echo "FILTER_MODE=$FILTER_MODE CUTOFF_HZ=$CUTOFF_HZ FILTER_FS=$FILTER_FS FILTER_ORDER=$FILTER_ORDER"

AMASS_RAW=data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
DIP_TRAIN_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json
DIP_VAL_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json
DIP_TEST_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
TC_TEST_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json

AMASS_CONTROL=data/dataset_work/GTControlCache/amass_train/gt_control_cache_manifest.json
DIP_TRAIN_CONTROL=data/dataset_work/GTControlCache/dip_train/gt_control_cache_manifest.json
DIP_VAL_CONTROL=data/dataset_work/GTControlCache/dip_val/gt_control_cache_manifest.json

RAW_PL_AMASS=data/experiments/newpl_v5_official_protocol_20260607/caches/pl_amass_official_init36/pl_curve_cache_manifest.json
RAW_PL_DIP_TEST=data/experiments/newpl_v5_official_protocol_20260607/caches/pl_dip_test_official_init36/pl_curve_cache_manifest.json
RAW_PL_TC_TEST=data/experiments/newpl_v5_official_protocol_20260607/caches/pl_tc_test_official_init36/pl_curve_cache_manifest.json

RAW_V5_AMASS=data/experiments/newpl_v5_official_protocol_20260607_tuned/amass_pretrain/best_loss.pt
RAW_V5_DIP=data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt

for required in \
  "$AMASS_RAW" "$DIP_TRAIN_RAW" "$DIP_VAL_RAW" "$DIP_TEST_RAW" "$TC_TEST_RAW" \
  "$AMASS_CONTROL" "$DIP_TRAIN_CONTROL" "$DIP_VAL_CONTROL" \
  "$RAW_PL_AMASS" "$RAW_PL_DIP_TEST" "$RAW_PL_TC_TEST" "$RAW_V5_AMASS" "$RAW_V5_DIP"; do
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
  AMASS_BATCH=${AMASS_BATCH:-512}
  DIP_BATCH=${DIP_BATCH:-64}
  AMASS_MAX_TRAIN=${AMASS_MAX_TRAIN:-0}
  AMASS_MAX_VAL=${AMASS_MAX_VAL:-20}
  AMASS_VAL_WINDOW=${AMASS_VAL_WINDOW:-61}
  DIP_MAX_TRAIN=${DIP_MAX_TRAIN:-0}
  DIP_MAX_VAL=${DIP_MAX_VAL:-0}
  DIP_VAL_WINDOW=${DIP_VAL_WINDOW:-61}
  EVAL_MAX=${EVAL_MAX:-0}
  AMASS_EVAL_MAX=${AMASS_EVAL_MAX:-20}
fi

ensure_residual_pl_cache() {
  local input_manifest="$1"
  local output_dir="$2"
  if [[ ! -f "$output_dir/pl_curve_cache_manifest.json" ]]; then
    local max_args=()
    if [[ "$CACHE_MAX" != "0" ]]; then
      max_args+=(--max-sequences "$CACHE_MAX")
    fi
    "$PY" pl_curve_cache.py \
      --input-cache "$input_manifest" \
      --output-dir "$output_dir" \
      --imu-input-mode official \
      --feature-mode smooth_residual \
      --acc-filter-mode "$FILTER_MODE" \
      --cutoff-hz "$CUTOFF_HZ" \
      --filter-fs "$FILTER_FS" \
      --filter-order "$FILTER_ORDER" \
      --shard-size 100 \
      "${max_args[@]}"
  fi
}

run_eval() {
  local pl_cache="$1"
  local checkpoint="$2"
  local output_json="$3"
  local max_sequences="$4"
  local max_args=()
  if [[ "$max_sequences" != "0" ]]; then
    max_args+=(--max-sequences "$max_sequences")
  fi
  "$PY" pl_curve_pl_accuracy_eval.py \
    --pl-cache "$pl_cache" \
    --checkpoint "$checkpoint" \
    --output-json "$output_json" \
    "${max_args[@]}"
}

ensure_residual_pl_cache "$AMASS_RAW" "$CACHE_ROOT/pl_amass_realtime_residual_init36"
ensure_residual_pl_cache "$DIP_TRAIN_RAW" "$CACHE_ROOT/pl_dip_train_realtime_residual_init36"
ensure_residual_pl_cache "$DIP_VAL_RAW" "$CACHE_ROOT/pl_dip_val_realtime_residual_init36"
ensure_residual_pl_cache "$DIP_TEST_RAW" "$CACHE_ROOT/pl_dip_test_realtime_residual_init36"
ensure_residual_pl_cache "$TC_TEST_RAW" "$CACHE_ROOT/pl_tc_test_realtime_residual_init36"

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
    --train-cache "$CACHE_ROOT/pl_dip_train_realtime_residual_init36/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/pl_dip_val_realtime_residual_init36/pl_curve_cache_manifest.json" \
    --train-gt-control-cache "$DIP_TRAIN_CONTROL" \
    --val-gt-control-cache "$DIP_VAL_CONTROL" \
    --output-dir "$ROOT/smoke" \
    --experiment-name newpl_v5_realtime_residual_smoke \
    --epochs 1 \
    --lr 1e-5 \
    --batch-size 16 \
    --max-train-sequences 8 \
    --max-val-sequences 2 \
    "${COMMON_TRAIN_ARGS[@]}"
fi

if [[ ! -f "$ROOT/amass_pretrain/best_loss.pt" ]]; then
  "$PY" pl_curve_train.py \
    --train-cache "$CACHE_ROOT/pl_amass_realtime_residual_init36/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/pl_amass_realtime_residual_init36/pl_curve_cache_manifest.json" \
    --train-gt-control-cache "$AMASS_CONTROL" \
    --val-gt-control-cache "$AMASS_CONTROL" \
    --output-dir "$ROOT/amass_pretrain" \
    --experiment-name newpl_v5_realtime_residual_amass_pretrain \
    --epochs "$AMASS_EPOCHS" \
    --lr 1e-4 \
    --batch-size "$AMASS_BATCH" \
    --max-train-sequences "$AMASS_MAX_TRAIN" \
    --max-val-sequences "$AMASS_MAX_VAL" \
    --val-window-length "$AMASS_VAL_WINDOW" \
    --init-checkpoint "$RAW_V5_AMASS" \
    --early-stop-min-delta 0.00000005 \
    --early-stop-patience 12 \
    "${COMMON_TRAIN_ARGS[@]}"
fi

run_eval "$CACHE_ROOT/pl_amass_realtime_residual_init36/pl_curve_cache_manifest.json" \
  "$ROOT/amass_pretrain/best_loss.pt" "$EVAL_ROOT/amass_after_amass_pretrain_realtime_residual.json" "$AMASS_EVAL_MAX"
run_eval "$CACHE_ROOT/pl_dip_test_realtime_residual_init36/pl_curve_cache_manifest.json" \
  "$ROOT/amass_pretrain/best_loss.pt" "$EVAL_ROOT/dip_test_after_amass_pretrain_realtime_residual.json" "$EVAL_MAX"
run_eval "$CACHE_ROOT/pl_tc_test_realtime_residual_init36/pl_curve_cache_manifest.json" \
  "$ROOT/amass_pretrain/best_loss.pt" "$EVAL_ROOT/tc_test_after_amass_pretrain_realtime_residual.json" "$EVAL_MAX"

if [[ ! -f "$ROOT/dip_finetune/best_loss.pt" ]]; then
  "$PY" pl_curve_train.py \
    --train-cache "$CACHE_ROOT/pl_dip_train_realtime_residual_init36/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/pl_dip_val_realtime_residual_init36/pl_curve_cache_manifest.json" \
    --train-gt-control-cache "$DIP_TRAIN_CONTROL" \
    --val-gt-control-cache "$DIP_VAL_CONTROL" \
    --output-dir "$ROOT/dip_finetune" \
    --experiment-name newpl_v5_realtime_residual_amass_to_dip_finetune \
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

run_eval "$CACHE_ROOT/pl_dip_test_realtime_residual_init36/pl_curve_cache_manifest.json" \
  "$ROOT/dip_finetune/best_loss.pt" "$EVAL_ROOT/dip_test_after_dip_finetune_realtime_residual.json" "$EVAL_MAX"
run_eval "$CACHE_ROOT/pl_tc_test_realtime_residual_init36/pl_curve_cache_manifest.json" \
  "$ROOT/dip_finetune/best_loss.pt" "$EVAL_ROOT/tc_test_after_dip_finetune_realtime_residual.json" "$EVAL_MAX"

"$PY" scripts/summarize_newpl_v5_realtime_residual.py \
  --root "$ROOT" \
  --filter-mode "$FILTER_MODE" \
  --cutoff-hz "$CUTOFF_HZ"

echo "done: $ROOT"
