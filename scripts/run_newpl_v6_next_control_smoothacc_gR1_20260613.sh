#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
PY="${PY:-$ENV/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

EXP="${EXP:-data/experiments/newpl_v6_next_control_smoothacc_gR1_20260613}"
CANDIDATE_PREFIX="${CANDIDATE_PREFIX:-newpl_v6_smoothacc}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-newpl_v6_next_control_smoothacc_gR1}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-9}"
SMOOTH_MODE="${SMOOTH_MODE:-centered_moving_average}"
CACHE_ROOT="${CACHE_ROOT:-$EXP/caches}"
RUN_ROOT="$EXP/$MODE"
if [[ -n "${RUN_SUFFIX:-}" ]]; then
  RUN_ROOT="${EXP}/${MODE}_${RUN_SUFFIX}"
fi
NEXT_CACHE_ROOT="${NEXT_CACHE_ROOT:-$EXP/caches/$MODE}"
LOG_ROOT="$EXP/logs"
mkdir -p "$CACHE_ROOT" "$NEXT_CACHE_ROOT" "$RUN_ROOT" "$LOG_ROOT"

if [[ "${NO_TEE:-0}" != "1" ]]; then
  exec > >(tee -a "$LOG_ROOT/run_${MODE}.log") 2>&1
fi

echo "EXP=$EXP"
echo "MODE=$MODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "SMOOTH_WINDOW=$SMOOTH_WINDOW"
echo "SMOOTH_MODE=$SMOOTH_MODE"
echo "EXPERIMENT_LABEL=$EXPERIMENT_LABEL"
echo "CANDIDATE_PREFIX=$CANDIDATE_PREFIX"

AMASS_RAW=data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
DIP_TRAIN_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json
DIP_VAL_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json
DIP_TEST_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
TC_TEST_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json

GT_AMASS=data/dataset_work/GTControlCache/amass_train/gt_control_cache_manifest.json
GT_DIP_TRAIN=data/dataset_work/GTControlCache/dip_train/gt_control_cache_manifest.json
GT_DIP_VAL=data/dataset_work/GTControlCache/dip_val/gt_control_cache_manifest.json
GT_DIP_TEST=data/dataset_work/GTControlCache/dip_test/gt_control_cache_manifest.json
GT_TC_TEST=data/dataset_work/GTControlCache/totalcapture_test/gt_control_cache_manifest.json

NEWPL_V4="${NEWPL_V4:-data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt}"
RAW_V5_AMASS="${RAW_V5_AMASS:-data/experiments/newpl_v5_official_protocol_20260607_tuned/amass_pretrain/best_loss.pt}"
RAW_V5_DIP="${RAW_V5_DIP:-data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt}"
RAW_V6_AMASS="${RAW_V6_AMASS:-data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/amass_pretrain/best_next_module_metric.pt}"
RAW_V6_DIP="${RAW_V6_DIP:-data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/dip_finetune/best_next_module_metric.pt}"

for required in "$AMASS_RAW" "$DIP_TRAIN_RAW" "$DIP_VAL_RAW" "$DIP_TEST_RAW" "$TC_TEST_RAW" \
  "$GT_AMASS" "$GT_DIP_TRAIN" "$GT_DIP_VAL" "$GT_DIP_TEST" "$GT_TC_TEST" \
  "$NEWPL_V4" "$RAW_V5_AMASS" "$RAW_V5_DIP"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 3
  fi
done

if [[ "$MODE" == "smoke" ]]; then
  CACHE_MAX="${CACHE_MAX:-8}"
  EPOCHS_AMASS="${EPOCHS_AMASS:-1}"
  EPOCHS_DIP="${EPOCHS_DIP:-1}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  WINDOW="${WINDOW:-61}"
  MAX_TRAIN_SEQS="${MAX_TRAIN_SEQS:-8}"
  MAX_VAL_SEQS="${MAX_VAL_SEQS:-4}"
  VAL_WINDOW="${VAL_WINDOW:-61}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
  MAX_EVAL_SEQS="${MAX_EVAL_SEQS:-4}"
  AMASS_MAX_EVAL_SEQS="${AMASS_MAX_EVAL_SEQS:-$MAX_EVAL_SEQS}"
  MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-512}"
else
  CACHE_MAX="${CACHE_MAX:-0}"
  EPOCHS_AMASS="${EPOCHS_AMASS:-80}"
  EPOCHS_DIP="${EPOCHS_DIP:-40}"
  BATCH_SIZE="${BATCH_SIZE:-512}"
  WINDOW="${WINDOW:-81}"
  MAX_TRAIN_SEQS="${MAX_TRAIN_SEQS:-0}"
  MAX_VAL_SEQS="${MAX_VAL_SEQS:-128}"
  VAL_WINDOW="${VAL_WINDOW:-512}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
  MAX_EVAL_SEQS="${MAX_EVAL_SEQS:-0}"
  AMASS_MAX_EVAL_SEQS="${AMASS_MAX_EVAL_SEQS:-20}"
  MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-0}"
fi

ensure_smooth_cache() {
  local input_manifest="$1"
  local output_dir="$2"
  if [[ -f "$output_dir/baseline_cache_manifest.json" ]]; then
    return
  fi
  local max_args=()
  if [[ "$CACHE_MAX" != "0" ]]; then
    max_args+=(--max-sequences "$CACHE_MAX")
  fi
  "$PY" scripts/build_smooth_acc_cache.py \
    --input-cache "$input_manifest" \
    --output-dir "$output_dir" \
    --window "$SMOOTH_WINDOW" \
    --mode "$SMOOTH_MODE" \
    --notes "NewPL v6 next-control smoothacc gR1 experiment; replace aM only, keep wM/RMB/targets unchanged." \
    "${max_args[@]}"
}

ensure_pl_cache() {
  local input_manifest="$1"
  local output_dir="$2"
  if [[ -f "$output_dir/pl_curve_cache_manifest.json" ]]; then
    return
  fi
  "$PY" pl_curve_cache.py \
    --input-cache "$input_manifest" \
    --output-dir "$output_dir" \
    --imu-input-mode official \
    --feature-mode legacy \
    --shard-size 100
}

ensure_next_cache() {
  local pl_cache="$1"
  local gt_cache="$2"
  local output_dir="$3"
  local manifest="$output_dir/pl_next_control_cache_manifest.json"
  if [[ -f "$manifest" ]]; then
    local cache_type
    cache_type="$("$PY" - "$manifest" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f).get("type", ""))
PY
)"
    if [[ "$cache_type" == "pl_next_control_cache_v2" ]]; then
      return
    fi
    echo "Existing next-control cache is type=$cache_type at $manifest; use a fresh EXP." >&2
    exit 4
  fi
  "$PY" pl_next_control_cache.py \
    --pl-cache "$pl_cache" \
    --gt-control-cache "$gt_cache" \
    --output-dir "$output_dir" \
    --shard-size 100 \
    --max-sequences "$CACHE_MAX"
}

smooth_amass="$CACHE_ROOT/raw_amass_smooth_w${SMOOTH_WINDOW}"
smooth_dip_train="$CACHE_ROOT/raw_dip_train_smooth_w${SMOOTH_WINDOW}"
smooth_dip_val="$CACHE_ROOT/raw_dip_val_smooth_w${SMOOTH_WINDOW}"
smooth_dip_test="$CACHE_ROOT/raw_dip_test_smooth_w${SMOOTH_WINDOW}"
smooth_tc_test="$CACHE_ROOT/raw_tc_test_smooth_w${SMOOTH_WINDOW}"

ensure_smooth_cache "$AMASS_RAW" "$smooth_amass"
ensure_smooth_cache "$DIP_TRAIN_RAW" "$smooth_dip_train"
ensure_smooth_cache "$DIP_VAL_RAW" "$smooth_dip_val"
ensure_smooth_cache "$DIP_TEST_RAW" "$smooth_dip_test"
ensure_smooth_cache "$TC_TEST_RAW" "$smooth_tc_test"

pl_amass="$CACHE_ROOT/pl_amass_smoothacc_init36"
pl_dip_train="$CACHE_ROOT/pl_dip_train_smoothacc_init36"
pl_dip_val="$CACHE_ROOT/pl_dip_val_smoothacc_init36"
pl_dip_test="$CACHE_ROOT/pl_dip_test_smoothacc_init36"
pl_tc_test="$CACHE_ROOT/pl_tc_test_smoothacc_init36"

ensure_pl_cache "$smooth_amass/baseline_cache_manifest.json" "$pl_amass"
ensure_pl_cache "$smooth_dip_train/baseline_cache_manifest.json" "$pl_dip_train"
ensure_pl_cache "$smooth_dip_val/baseline_cache_manifest.json" "$pl_dip_val"
ensure_pl_cache "$smooth_dip_test/baseline_cache_manifest.json" "$pl_dip_test"
ensure_pl_cache "$smooth_tc_test/baseline_cache_manifest.json" "$pl_tc_test"

next_amass="$NEXT_CACHE_ROOT/next_amass_train"
next_dip_train="$NEXT_CACHE_ROOT/next_dip_train"
next_dip_val="$NEXT_CACHE_ROOT/next_dip_val"
next_dip_test="$NEXT_CACHE_ROOT/next_dip_test"
next_tc_test="$NEXT_CACHE_ROOT/next_tc_test"

ensure_next_cache "$pl_amass/pl_curve_cache_manifest.json" "$GT_AMASS" "$next_amass"
ensure_next_cache "$pl_dip_train/pl_curve_cache_manifest.json" "$GT_DIP_TRAIN" "$next_dip_train"
ensure_next_cache "$pl_dip_val/pl_curve_cache_manifest.json" "$GT_DIP_VAL" "$next_dip_val"
ensure_next_cache "$pl_dip_test/pl_curve_cache_manifest.json" "$GT_DIP_TEST" "$next_dip_test"
ensure_next_cache "$pl_tc_test/pl_curve_cache_manifest.json" "$GT_TC_TEST" "$next_tc_test"

NEXT_AMASS="$next_amass/pl_next_control_cache_manifest.json"
NEXT_DIP_TRAIN="$next_dip_train/pl_next_control_cache_manifest.json"
NEXT_DIP_VAL="$next_dip_val/pl_next_control_cache_manifest.json"
NEXT_DIP_TEST="$next_dip_test/pl_next_control_cache_manifest.json"
NEXT_TC_TEST="$next_tc_test/pl_next_control_cache_manifest.json"

COMMON_TRAIN_ARGS=(
  --model-variant newpl_v6_next_control
  --init-size 36
  --hidden-size 512
  --tail-length 4
  --residual-scale 0.005
  --next-residual-scale 0.005
  --dropout 0.4
  --grad-clip 1.0
  --pRB-weight "${PRB_WEIGHT:-0.7}"
  --gR1-weight "${GR1_WEIGHT:-3.0}"
  --gt-control-pRB-weight "${GT_CONTROL_PRB_WEIGHT:-0.2}"
  --gt-control-gR1-weight "${GT_CONTROL_GR1_WEIGHT:-0.5}"
  --pRB-dot-weight "${PRB_DOT_WEIGHT:-0.02}"
  --pRB-ddot-smooth-weight "${PRB_DDOT_SMOOTH_WEIGHT:-0.000001}"
  --gR1-dot-weight "${GR1_DOT_WEIGHT:-0.05}"
  --gR1-ddot-weight "${GR1_DDOT_WEIGHT:-0.002}"
  --next-pRB-weight "${NEXT_PRB_WEIGHT:-0.5}"
  --next-gR1-weight "${NEXT_GR1_WEIGHT:-3.0}"
  --next-gt-control-pRB-weight "${NEXT_GT_CONTROL_PRB_WEIGHT:-0.15}"
  --next-gt-control-gR1-weight "${NEXT_GT_CONTROL_GR1_WEIGHT:-0.5}"
  --next-pRB-vel-weight "${NEXT_PRB_VEL_WEIGHT:-0.02}"
  --next-pRB-acc-weight "${NEXT_PRB_ACC_WEIGHT:-0.0002}"
  --next-gR1-vel-weight "${NEXT_GR1_VEL_WEIGHT:-0.05}"
  --next-gR1-acc-weight "${NEXT_GR1_ACC_WEIGHT:-0.002}"
  --next-control-delta-prior-weight "${NEXT_CONTROL_DELTA_PRIOR_WEIGHT:-0.01}"
  --last-control-pRB-weight "${LAST_CONTROL_PRB_WEIGHT:-0.15}"
  --last-control-gR1-weight "${LAST_CONTROL_GR1_WEIGHT:-0.5}"
  --next-tail4-control-pRB-weight "${NEXT_TAIL4_CONTROL_PRB_WEIGHT:-0.1}"
  --next-tail4-control-gR1-weight "${NEXT_TAIL4_CONTROL_GR1_WEIGHT:-0.35}"
)

echo "Training: BATCH_SIZE=$BATCH_SIZE WINDOW=$WINDOW VAL_BATCH_SIZE=$VAL_BATCH_SIZE"

if [[ ! -f "$RUN_ROOT/amass_pretrain/best_current_gR1.pt" ]]; then
  "$PY" pl_next_control_train.py \
    --train-cache "$NEXT_AMASS" \
    --val-cache "$NEXT_AMASS" \
    --output-dir "$RUN_ROOT/amass_pretrain" \
    --experiment-name "${EXPERIMENT_LABEL}_amass_${MODE}" \
    --epochs "$EPOCHS_AMASS" \
    --window "$WINDOW" \
    --batch-size "$BATCH_SIZE" \
    --val-window-length "$VAL_WINDOW" \
    --max-train-sequences "$MAX_TRAIN_SEQS" \
    --max-val-sequences "$MAX_VAL_SEQS" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --early-stop-min-delta "${AMASS_EARLY_STOP_MIN_DELTA:-0.00000005}" \
    --early-stop-patience "${AMASS_EARLY_STOP_PATIENCE:-12}" \
    --lr "${AMASS_LR:-1e-4}" \
    "${COMMON_TRAIN_ARGS[@]}"
fi

eval_max_args=()
if [[ "$MAX_EVAL_SEQS" != "0" ]]; then
  eval_max_args+=(--max-eval-sequences "$MAX_EVAL_SEQS")
fi
amass_eval_max_args=()
if [[ "$AMASS_MAX_EVAL_SEQS" != "0" ]]; then
  amass_eval_max_args+=(--max-eval-sequences "$AMASS_MAX_EVAL_SEQS")
fi
eval_frame_args=()
if [[ "$MAX_EVAL_FRAMES" != "0" ]]; then
  eval_frame_args+=(--max-frames-per-sequence "$MAX_EVAL_FRAMES")
fi

version_args_amass=(
  --version "official_PL_smoothacc=official"
  --version "newpl_v4_init36_smoothacc=$NEWPL_V4"
  --version "newpl_v5_raw_amass_on_smoothinput=$RAW_V5_AMASS"
  --version "${CANDIDATE_PREFIX}_amass_current_gR1=$RUN_ROOT/amass_pretrain/best_current_gR1.pt"
  --version "${CANDIDATE_PREFIX}_amass_next_gR1=$RUN_ROOT/amass_pretrain/best_next_gR1.pt"
  --version "${CANDIDATE_PREFIX}_amass_gravity_control=$RUN_ROOT/amass_pretrain/best_gravity_control.pt"
  --version "${CANDIDATE_PREFIX}_amass_balanced=$RUN_ROOT/amass_pretrain/best_current_module_metric.pt"
)
if [[ -f "$RAW_V6_AMASS" ]]; then
  version_args_amass+=(--version "newpl_v6_raw_amass_on_smoothinput=$RAW_V6_AMASS")
fi

if [[ ! -f "$RUN_ROOT/eval_amass_after_pretrain.json" ]]; then
  "$PY" pl_next_control_eval.py \
    --cache "$NEXT_AMASS" \
    --dataset-label "AMASS-smoothacc-nextcontrol-after-AMASS-${MODE}" \
    --output-json "$RUN_ROOT/eval_amass_after_pretrain.json" \
    "${amass_eval_max_args[@]}" \
    "${eval_frame_args[@]}" \
    "${version_args_amass[@]}"
fi

if [[ ! -f "$RUN_ROOT/eval_dip_test_after_amass_pretrain.json" ]]; then
  "$PY" pl_next_control_eval.py \
    --cache "$NEXT_DIP_TEST" \
    --dataset-label "DIP-IMU-test-smoothacc-nextcontrol-after-AMASS-${MODE}" \
    --output-json "$RUN_ROOT/eval_dip_test_after_amass_pretrain.json" \
    "${eval_max_args[@]}" \
    "${eval_frame_args[@]}" \
    "${version_args_amass[@]}"
fi

if [[ ! -f "$RUN_ROOT/eval_totalcapture_test_after_amass_pretrain.json" ]]; then
  "$PY" pl_next_control_eval.py \
    --cache "$NEXT_TC_TEST" \
    --dataset-label "TotalCapture-test-smoothacc-nextcontrol-after-AMASS-${MODE}" \
    --output-json "$RUN_ROOT/eval_totalcapture_test_after_amass_pretrain.json" \
    "${eval_max_args[@]}" \
    "${eval_frame_args[@]}" \
    "${version_args_amass[@]}"
fi

if [[ ! -f "$RUN_ROOT/dip_finetune/best_current_gR1.pt" ]]; then
  "$PY" pl_next_control_train.py \
    --train-cache "$NEXT_DIP_TRAIN" \
    --val-cache "$NEXT_DIP_VAL" \
    --output-dir "$RUN_ROOT/dip_finetune" \
    --experiment-name "${EXPERIMENT_LABEL}_dip_${MODE}" \
    --init-checkpoint "$RUN_ROOT/amass_pretrain/best_current_gR1.pt" \
    --epochs "$EPOCHS_DIP" \
    --window "$WINDOW" \
    --batch-size "$BATCH_SIZE" \
    --val-window-length "$VAL_WINDOW" \
    --max-train-sequences "$MAX_TRAIN_SEQS" \
    --max-val-sequences "$MAX_VAL_SEQS" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --early-stop-min-delta "${DIP_EARLY_STOP_MIN_DELTA:-0.00000005}" \
    --early-stop-patience "${DIP_EARLY_STOP_PATIENCE:-10}" \
    --lr "${DIP_LR:-5e-5}" \
    "${COMMON_TRAIN_ARGS[@]}"
fi

version_args_dip=(
  --version "official_PL_smoothacc=official"
  --version "newpl_v4_init36_smoothacc=$NEWPL_V4"
  --version "newpl_v5_raw_dip_on_smoothinput=$RAW_V5_DIP"
  --version "${CANDIDATE_PREFIX}_amass_current_gR1=$RUN_ROOT/amass_pretrain/best_current_gR1.pt"
  --version "${CANDIDATE_PREFIX}_dip_current_gR1=$RUN_ROOT/dip_finetune/best_current_gR1.pt"
  --version "${CANDIDATE_PREFIX}_dip_next_gR1=$RUN_ROOT/dip_finetune/best_next_gR1.pt"
  --version "${CANDIDATE_PREFIX}_dip_gravity_control=$RUN_ROOT/dip_finetune/best_gravity_control.pt"
  --version "${CANDIDATE_PREFIX}_dip_balanced=$RUN_ROOT/dip_finetune/best_current_module_metric.pt"
  --version "${CANDIDATE_PREFIX}_dip_last=$RUN_ROOT/dip_finetune/last.pt"
)
if [[ -f "$RAW_V6_DIP" ]]; then
  version_args_dip+=(--version "newpl_v6_raw_dip_on_smoothinput=$RAW_V6_DIP")
fi

if [[ ! -f "$RUN_ROOT/eval_dip_test_after_dip_finetune.json" ]]; then
  "$PY" pl_next_control_eval.py \
    --cache "$NEXT_DIP_TEST" \
    --dataset-label "DIP-IMU-test-smoothacc-nextcontrol-after-DIP-${MODE}" \
    --output-json "$RUN_ROOT/eval_dip_test_after_dip_finetune.json" \
    "${eval_max_args[@]}" \
    "${eval_frame_args[@]}" \
    "${version_args_dip[@]}"
fi

if [[ ! -f "$RUN_ROOT/eval_totalcapture_test_after_dip_finetune.json" ]]; then
  "$PY" pl_next_control_eval.py \
    --cache "$NEXT_TC_TEST" \
    --dataset-label "TotalCapture-test-smoothacc-nextcontrol-after-DIP-${MODE}" \
    --output-json "$RUN_ROOT/eval_totalcapture_test_after_dip_finetune.json" \
    "${eval_max_args[@]}" \
    "${eval_frame_args[@]}" \
    "${version_args_dip[@]}"
fi

"$PY" scripts/summarize_newpl_v6_next_control_smoothacc_gR1.py --root "$RUN_ROOT" --output "$RUN_ROOT/summary.json"

cat > "$RUN_ROOT/run_summary_${MODE}.json" <<JSON
{
  "status": "ok",
  "mode": "$MODE",
  "exp": "$EXP",
  "run_root": "$RUN_ROOT",
  "smooth_window": $SMOOTH_WINDOW,
  "batch_size": $BATCH_SIZE,
  "window": $WINDOW,
  "summary": "$RUN_ROOT/summary.json",
  "amass_eval": "$RUN_ROOT/eval_amass_after_pretrain.json",
  "dip_after_amass_eval": "$RUN_ROOT/eval_dip_test_after_amass_pretrain.json",
  "totalcapture_after_amass_eval": "$RUN_ROOT/eval_totalcapture_test_after_amass_pretrain.json",
  "dip_after_dip_eval": "$RUN_ROOT/eval_dip_test_after_dip_finetune.json",
  "totalcapture_after_dip_eval": "$RUN_ROOT/eval_totalcapture_test_after_dip_finetune.json"
}
JSON

echo "done: $RUN_ROOT"
