#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=${ENV_DIR:-/home/lingfeng/.conda/envs/globalpose-gpu}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

PY="$ENV_DIR/bin/python"
ROOT=${ROOT:-data/experiments/rjs_imu_proxy_newpl_quick_20260608_v1}
CACHE_ROOT=${CACHE_ROOT:-data/experiments/rjs_sensitive_newpl_20260608_feature_smoke/caches}
EVAL_DIR="$ROOT/eval"
LOG_DIR="$ROOT/logs"
mkdir -p "$EVAL_DIR" "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

AMASS_EPOCHS=${AMASS_EPOCHS:-5}
DIP_EPOCHS=${DIP_EPOCHS:-1}
AMASS_IMU_PROXY_WEIGHT=${AMASS_IMU_PROXY_WEIGHT:-0.05}
DIP_IMU_PROXY_WEIGHT=${DIP_IMU_PROXY_WEIGHT:-0.0}
IMU_PROXY_ACC_SCALE=${IMU_PROXY_ACC_SCALE:-30.0}
EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-4}

AMASS_CACHE="$CACHE_ROOT/pl_amass_offset_aware/pl_curve_cache_manifest.json"
DIP_TRAIN_CACHE="$CACHE_ROOT/pl_dip_train_offset_aware/pl_curve_cache_manifest.json"
DIP_VAL_CACHE="$CACHE_ROOT/pl_dip_val_offset_aware/pl_curve_cache_manifest.json"
DIP_TEST_CACHE="$CACHE_ROOT/pl_dip_test_offset_aware/pl_curve_cache_manifest.json"
TC_TEST_CACHE="$CACHE_ROOT/pl_tc_test_offset_aware/pl_curve_cache_manifest.json"

AMASS_OUT="$ROOT/amass_rjs_sensitive"
DIP_OUT="$ROOT/dip_finetune"

echo "started $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ROOT=$ROOT CACHE_ROOT=$CACHE_ROOT"
echo "AMASS_EPOCHS=$AMASS_EPOCHS DIP_EPOCHS=$DIP_EPOCHS AMASS_IMU_PROXY_WEIGHT=$AMASS_IMU_PROXY_WEIGHT DIP_IMU_PROXY_WEIGHT=$DIP_IMU_PROXY_WEIGHT"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

for path in "$AMASS_CACHE" "$DIP_TRAIN_CACHE" "$DIP_VAL_CACHE" "$DIP_TEST_CACHE" "$TC_TEST_CACHE"; do
  if [ ! -f "$path" ]; then
    echo "missing required cache: $path" >&2
    exit 2
  fi
done

if [ ! -f "$AMASS_OUT/train_result.json" ]; then
  "$PY" pl_curve_train.py \
    --train-cache "$AMASS_CACHE" \
    --val-cache "$AMASS_CACHE" \
    --output-dir "$AMASS_OUT" \
    --experiment-name rjs_imu_proxy_newpl_amass_stage_a \
    --epochs "$AMASS_EPOCHS" \
    --window 61 \
    --lr 1e-4 \
    --hidden-size 512 \
    --batch-size 128 \
    --dropout 0.15 \
    --disable-ik-distill \
    --model-variant offset_aware \
    --input-size 156 \
    --init-size 36 \
    --film-scale 0.3 \
    --offset-embed-size 128 \
    --selection-metric pl_and_control_physical \
    --pRB-weight 1.0 \
    --gR1-weight 1.0 \
    --baseline-pRB-weight 0.05 \
    --baseline-gR1-weight 0.0 \
    --gt-control-pRB-weight 0.3 \
    --gt-control-gR1-weight 0.1 \
    --pRB-ddot-smooth-weight 1e-6 \
    --offset-consistency-weight 0.1 \
    --offset-consistency-target full_pl \
    --offset-contrast-weight 0.2 \
    --offset-contrast-margin 0.001 \
    --offset-contrast-mode roll_random \
    --offset-contrast-target full_pl \
    --imu-proxy-offset-acc-weight "$AMASS_IMU_PROXY_WEIGHT" \
    --imu-proxy-acc-scale "$IMU_PROXY_ACC_SCALE" \
    --early-stop-patience 8 \
    --early-stop-min-delta 5e-8 \
    --max-train-sequences 32 \
    --max-val-sequences 8
fi

if [ ! -f "$DIP_OUT/train_result.json" ]; then
  "$PY" pl_curve_train.py \
    --train-cache "$DIP_TRAIN_CACHE" \
    --val-cache "$DIP_VAL_CACHE" \
    --output-dir "$DIP_OUT" \
    --experiment-name rjs_imu_proxy_newpl_dip_stage_b \
    --epochs "$DIP_EPOCHS" \
    --window 61 \
    --lr 5e-6 \
    --hidden-size 512 \
    --batch-size 12 \
    --dropout 0.05 \
    --disable-ik-distill \
    --model-variant offset_aware \
    --input-size 156 \
    --init-size 36 \
    --film-scale 0.3 \
    --offset-embed-size 128 \
    --init-checkpoint "$AMASS_OUT/best_loss.pt" \
    --selection-metric pl_and_control_physical \
    --pRB-weight 1.0 \
    --gR1-weight 1.0 \
    --baseline-pRB-weight 0.05 \
    --baseline-gR1-weight 0.0 \
    --gt-control-pRB-weight 0.3 \
    --gt-control-gR1-weight 0.1 \
    --pRB-ddot-smooth-weight 1e-6 \
    --imu-proxy-offset-acc-weight "$DIP_IMU_PROXY_WEIGHT" \
    --imu-proxy-acc-scale "$IMU_PROXY_ACC_SCALE" \
    --early-stop-patience 8 \
    --early-stop-min-delta 5e-8 \
    --max-train-sequences 8 \
    --max-val-sequences 4
fi

for ckpt_name in best last; do
  case "$ckpt_name" in
    best) ckpt_path="$DIP_OUT/best_loss.pt" ;;
    last) ckpt_path="$DIP_OUT/last.pt" ;;
  esac
  for split in dip_val dip_test tc_test; do
    case "$split" in
      dip_val) cache="$DIP_VAL_CACHE" ;;
      dip_test) cache="$DIP_TEST_CACHE" ;;
      tc_test) cache="$TC_TEST_CACHE" ;;
    esac
    "$PY" pl_curve_pl_accuracy_eval.py \
      --pl-cache "$cache" \
      --checkpoint "$ckpt_path" \
      --output-json "$EVAL_DIR/${ckpt_name}_${split}_module_pl_accuracy.json" \
      --max-sequences "$EVAL_MAX_SEQS"
  done
  for split in dip_test tc_test; do
    case "$split" in
      dip_test) cache="$DIP_TEST_CACHE" ;;
      tc_test) cache="$TC_TEST_CACHE" ;;
    esac
    "$PY" pl_curve_offset_swap_eval.py \
      --pl-cache "$cache" \
      --checkpoint "$ckpt_path" \
      --output-json "$EVAL_DIR/${ckpt_name}_${split}_offset_swap.json" \
      --max-sequences "$EVAL_MAX_SEQS" \
      --swap-feature-offset \
      --variants good,zero,roll_sensors,other_sequence,negate
  done
done

"$PY" scripts/summarize_rjs_sensitive_newpl.py --root "$ROOT"
echo "finished $(date --iso-8601=seconds)"
