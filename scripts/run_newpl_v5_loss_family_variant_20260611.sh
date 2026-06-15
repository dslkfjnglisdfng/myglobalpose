#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?usage: $0 VARIANT}"

ENV_DIR="${ENV_DIR:-/home/lingfeng/.conda/envs/globalpose-gpu}"
PY="${PY:-$ENV_DIR/bin/python}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

ROOT="${ROOT:-data/experiments/newpl_v5_loss_family_ablation_20260611}"
CACHE_ROOT="${CACHE_ROOT:-data/experiments/newpl_v5_official_protocol_20260607/caches}"
GT_CTRL_ROOT="${GT_CTRL_ROOT:-data/dataset_work/GTControlCache}"
EVAL_ROOT="$ROOT/eval"

AMASS_PL_CACHE="$CACHE_ROOT/pl_amass_official_init36/pl_curve_cache_manifest.json"
DIP_TRAIN_PL_CACHE="$CACHE_ROOT/pl_dip_train_official_init36/pl_curve_cache_manifest.json"
DIP_VAL_PL_CACHE="$CACHE_ROOT/pl_dip_val_official_init36/pl_curve_cache_manifest.json"
DIP_TEST_RAW="$CACHE_ROOT/dip_test_with_offset_r/baseline_cache_manifest.json"
TC_TEST_RAW="$CACHE_ROOT/tc_test_official_with_offset_r/baseline_cache_manifest.json"

AMASS_CTRL="$GT_CTRL_ROOT/amass_train/gt_control_cache_manifest.json"
DIP_TRAIN_CTRL="$GT_CTRL_ROOT/dip_train/gt_control_cache_manifest.json"
DIP_VAL_CTRL="$GT_CTRL_ROOT/dip_val/gt_control_cache_manifest.json"

NEWPL_V4="${NEWPL_V4:-data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt}"
NEWPL_V5_AMASS="${NEWPL_V5_AMASS:-data/experiments/newpl_v5_official_protocol_20260607_tuned/amass_pretrain/best_loss.pt}"
NEWPL_V5_DIP="${NEWPL_V5_DIP:-data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt}"

mkdir -p "$EVAL_ROOT"

for required in "$AMASS_PL_CACHE" "$DIP_TRAIN_PL_CACHE" "$DIP_VAL_PL_CACHE" "$DIP_TEST_RAW" "$TC_TEST_RAW" "$AMASS_CTRL" "$DIP_TRAIN_CTRL" "$DIP_VAL_CTRL" "$NEWPL_V4" "$NEWPL_V5_AMASS" "$NEWPL_V5_DIP"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 2
  fi
done

CONTROL_ARGS=(
  --gt-control-pRB-weight 0.0
  --gt-control-gR1-weight 0.0
  --control-point-prior-weight 0.0
  --tail-update-prior-weight 0.0
)
QDOT_ARGS=(
  --pRB-dot-weight 0.0
  --gR1-dot-weight 0.0
)
QDDOT_ARGS=(
  --pRB-ddot-weight 0.0
  --gR1-ddot-weight 0.0
  --pRB-ddot-smooth-weight 0.0
)

case "$VARIANT" in
  q_only)
    ;;
  q_control)
    CONTROL_ARGS=(--gt-control-pRB-weight 0.3 --gt-control-gR1-weight 0.1 --control-point-prior-weight 0.3 --tail-update-prior-weight 0.005)
    ;;
  q_qdot)
    QDOT_ARGS=(--pRB-dot-weight 0.03 --gR1-dot-weight 0.03)
    ;;
  q_qddot)
    QDDOT_ARGS=(--pRB-ddot-weight 0.0003 --gR1-ddot-weight 0.001 --pRB-ddot-smooth-weight 0.000001)
    ;;
  q_qdot_qddot)
    QDOT_ARGS=(--pRB-dot-weight 0.03 --gR1-dot-weight 0.03)
    QDDOT_ARGS=(--pRB-ddot-weight 0.0003 --gR1-ddot-weight 0.001 --pRB-ddot-smooth-weight 0.000001)
    ;;
  q_control_qdot)
    CONTROL_ARGS=(--gt-control-pRB-weight 0.3 --gt-control-gR1-weight 0.1 --control-point-prior-weight 0.3 --tail-update-prior-weight 0.005)
    QDOT_ARGS=(--pRB-dot-weight 0.03 --gR1-dot-weight 0.03)
    ;;
  q_control_qddot)
    CONTROL_ARGS=(--gt-control-pRB-weight 0.3 --gt-control-gR1-weight 0.1 --control-point-prior-weight 0.3 --tail-update-prior-weight 0.005)
    QDDOT_ARGS=(--pRB-ddot-weight 0.0003 --gR1-ddot-weight 0.001 --pRB-ddot-smooth-weight 0.000001)
    ;;
  q_control_qdot_qddot)
    CONTROL_ARGS=(--gt-control-pRB-weight 0.3 --gt-control-gR1-weight 0.1 --control-point-prior-weight 0.3 --tail-update-prior-weight 0.005)
    QDOT_ARGS=(--pRB-dot-weight 0.03 --gR1-dot-weight 0.03)
    QDDOT_ARGS=(--pRB-ddot-weight 0.0003 --gR1-ddot-weight 0.001 --pRB-ddot-smooth-weight 0.000001)
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

COMMON_TRAIN_ARGS=(
  --init-size 36
  --window 61
  --hidden-size 512
  --tail-length 4
  --residual-scale 0.005
  --dropout 0.4
  --grad-clip 1.0
  --disable-ik-distill
  --pRB-weight 1.0
  --gR1-weight 1.0
  --baseline-pRB-weight 0.0
  --baseline-gR1-weight 0.0
  --gR-smooth-weight 0.0
  --selection-metric pl_physical
  --early-stop-min-delta 0.00000005
  --val-window-length 61
)

LOSS_ARGS=("${CONTROL_ARGS[@]}" "${QDOT_ARGS[@]}" "${QDDOT_ARGS[@]}")

train_stage() {
  local stage="$1"
  local train_cache="$2"
  local val_cache="$3"
  local train_ctrl="$4"
  local val_ctrl="$5"
  local out_dir="$6"
  local epochs="$7"
  local lr="$8"
  local batch_size="$9"
  local max_train="${10}"
  local max_val="${11}"
  local init_ckpt="${12}"
  shift 12
  local extra=("$@")
  if [[ -f "$out_dir/best_loss.pt" && -f "$out_dir/train_result.json" ]]; then
    echo "skip existing train stage: $out_dir"
    return
  fi
  mkdir -p "$out_dir"
  local cmd=(
    "$PY" pl_curve_train.py
    --train-cache "$train_cache"
    --val-cache "$val_cache"
    --train-gt-control-cache "$train_ctrl"
    --val-gt-control-cache "$val_ctrl"
    --output-dir "$out_dir"
    --experiment-name "newpl_v5_loss_family_${VARIANT}_${stage}"
    --epochs "$epochs"
    --lr "$lr"
    --batch-size "$batch_size"
    --max-train-sequences "$max_train"
    --max-val-sequences "$max_val"
    --early-stop-patience 12
    "${COMMON_TRAIN_ARGS[@]}"
    "${LOSS_ARGS[@]}"
    "${extra[@]}"
  )
  if [[ -n "$init_ckpt" ]]; then
    cmd+=(--init-checkpoint "$init_ckpt")
  fi
  "${cmd[@]}"
}

eval_stage() {
  local stage="$1"
  local variant_eval_root="$EVAL_ROOT/$VARIANT"
  mkdir -p "$variant_eval_root"
  local dip_out="$variant_eval_root/after_${stage}_dip_test.json"
  local tc_out="$variant_eval_root/after_${stage}_tc_test.json"
  local versions=(
    --version official_PL=official
    --version newpl_v4_init36="$NEWPL_V4"
    --version newpl_v5_ref_amass="$NEWPL_V5_AMASS"
    --version "${VARIANT}_amass_best=$ROOT/$VARIANT/amass_pretrain/best_loss.pt"
    --version "${VARIANT}_amass_last=$ROOT/$VARIANT/amass_pretrain/last.pt"
  )
  if [[ "$stage" == "dip" ]]; then
    versions+=(
      --version newpl_v5_ref_dip="$NEWPL_V5_DIP"
      --version "${VARIANT}_dip_best=$ROOT/$VARIANT/dip_finetune/best_loss.pt"
      --version "${VARIANT}_dip_last=$ROOT/$VARIANT/dip_finetune/last.pt"
    )
  fi
  "$PY" newpl_root_eval.py \
    --cache "$DIP_TEST_RAW" \
    --output-json "$dip_out" \
    --dataset dip \
    --dataset-label DIP-IMU-test \
    --imu-input-mode official \
    "${versions[@]}"
  "$PY" newpl_root_eval.py \
    --cache "$TC_TEST_RAW" \
    --output-json "$tc_out" \
    --dataset totalcapture \
    --dataset-label TotalCapture-test-official-input \
    --imu-input-mode official \
    "${versions[@]}"
}

train_stage smoke "$DIP_TRAIN_PL_CACHE" "$DIP_VAL_PL_CACHE" "$DIP_TRAIN_CTRL" "$DIP_VAL_CTRL" \
  "$ROOT/$VARIANT/smoke" 1 1e-5 16 16 2 ""
train_stage amass_pretrain "$AMASS_PL_CACHE" "$AMASS_PL_CACHE" "$AMASS_CTRL" "$AMASS_CTRL" \
  "$ROOT/$VARIANT/amass_pretrain" 80 1e-4 256 0 20 ""
eval_stage amass
train_stage dip_finetune "$DIP_TRAIN_PL_CACHE" "$DIP_VAL_PL_CACHE" "$DIP_TRAIN_CTRL" "$DIP_VAL_CTRL" \
  "$ROOT/$VARIANT/dip_finetune" 40 5e-6 16 0 6 "$ROOT/$VARIANT/amass_pretrain/best_loss.pt"
eval_stage dip

cat > "$ROOT/$VARIANT/done.json" <<JSON
{
  "status": "ok",
  "variant": "$VARIANT",
  "root": "$ROOT/$VARIANT",
  "amass_eval_dip": "$EVAL_ROOT/$VARIANT/after_amass_dip_test.json",
  "amass_eval_tc": "$EVAL_ROOT/$VARIANT/after_amass_tc_test.json",
  "dip_eval_dip": "$EVAL_ROOT/$VARIANT/after_dip_dip_test.json",
  "dip_eval_tc": "$EVAL_ROOT/$VARIANT/after_dip_tc_test.json"
}
JSON
