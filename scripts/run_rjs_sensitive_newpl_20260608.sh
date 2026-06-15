#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=${ENV_DIR:-/home/lingfeng/.conda/envs/globalpose-gpu}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

PY="$ENV_DIR/bin/python"
ROOT=${ROOT:-data/experiments/rjs_sensitive_newpl_20260608}
CACHE_ROOT="$ROOT/caches"
EVAL_DIR="$ROOT/eval"
FULL_DIR="$ROOT/full_eval"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$EVAL_DIR" "$FULL_DIR" "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

SMOKE=${SMOKE:-0}
RUN_FULL_IF_PASS=${RUN_FULL_IF_PASS:-0}
AMASS_IMU_PROXY_WEIGHT=${AMASS_IMU_PROXY_WEIGHT:-0.0}
DIP_IMU_PROXY_WEIGHT=${DIP_IMU_PROXY_WEIGHT:-0.0}
IMU_PROXY_ACC_SCALE=${IMU_PROXY_ACC_SCALE:-30.0}

AMASS_SRC=${AMASS_SRC:-data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay_processed_fields/baseline_cache_manifest.json}
DIP_TRAIN_SRC=${DIP_TRAIN_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json}
DIP_VAL_SRC=${DIP_VAL_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json}
DIP_TEST_SRC=${DIP_TEST_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json}
TC_TEST_SRC=${TC_TEST_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json}

TC_S4_CACHE=${TC_S4_CACHE:-data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json}
TC_S5_CACHE=${TC_S5_CACHE:-data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_test_official_neural_only_offset_r/baseline_cache_manifest.json}

AMASS_CACHE="$CACHE_ROOT/pl_amass_offset_aware/pl_curve_cache_manifest.json"
DIP_TRAIN_CACHE="$CACHE_ROOT/pl_dip_train_offset_aware/pl_curve_cache_manifest.json"
DIP_VAL_CACHE="$CACHE_ROOT/pl_dip_val_offset_aware/pl_curve_cache_manifest.json"
DIP_TEST_CACHE="$CACHE_ROOT/pl_dip_test_offset_aware/pl_curve_cache_manifest.json"
TC_TEST_CACHE="$CACHE_ROOT/pl_tc_test_offset_aware/pl_curve_cache_manifest.json"

AMASS_OUT="$ROOT/amass_rjs_sensitive"
DIP_OUT="$ROOT/dip_finetune"

echo "started $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ROOT=$ROOT SMOKE=$SMOKE RUN_FULL_IF_PASS=$RUN_FULL_IF_PASS"
echo "AMASS_IMU_PROXY_WEIGHT=$AMASS_IMU_PROXY_WEIGHT DIP_IMU_PROXY_WEIGHT=$DIP_IMU_PROXY_WEIGHT IMU_PROXY_ACC_SCALE=$IMU_PROXY_ACC_SCALE"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

if [ "$SMOKE" = "1" ]; then
  CACHE_MAX_SEQS=32
  AMASS_EPOCHS=1
  DIP_EPOCHS=1
  AMASS_MAX_TRAIN=32
  AMASS_MAX_VAL=8
  DIP_MAX_TRAIN=8
  DIP_MAX_VAL=4
  EVAL_MAX_SEQS=4
else
  CACHE_MAX_SEQS=0
  AMASS_EPOCHS=80
  DIP_EPOCHS=40
  AMASS_MAX_TRAIN=0
  AMASS_MAX_VAL=20
  DIP_MAX_TRAIN=0
  DIP_MAX_VAL=0
  EVAL_MAX_SEQS=0
fi

ensure_pl_cache() {
  local src="$1"
  local out_dir="$2"
  if [ ! -f "$out_dir/pl_curve_cache_manifest.json" ]; then
    "$PY" pl_curve_cache.py \
      --input-cache "$src" \
      --output-dir "$out_dir" \
      --shard-size 100 \
      --imu-input-mode official \
      --feature-mode offset_aware \
      --max-sequences "$CACHE_MAX_SEQS"
  fi
}

ensure_pl_cache "$AMASS_SRC" "$CACHE_ROOT/pl_amass_offset_aware"
ensure_pl_cache "$DIP_TRAIN_SRC" "$CACHE_ROOT/pl_dip_train_offset_aware"
ensure_pl_cache "$DIP_VAL_SRC" "$CACHE_ROOT/pl_dip_val_offset_aware"
ensure_pl_cache "$DIP_TEST_SRC" "$CACHE_ROOT/pl_dip_test_offset_aware"
ensure_pl_cache "$TC_TEST_SRC" "$CACHE_ROOT/pl_tc_test_offset_aware"

if [ ! -f "$AMASS_OUT/train_result.json" ]; then
  "$PY" pl_curve_train.py \
    --train-cache "$AMASS_CACHE" \
    --val-cache "$AMASS_CACHE" \
    --output-dir "$AMASS_OUT" \
    --experiment-name rjs_sensitive_newpl_amass_stage_a \
    --epochs "$AMASS_EPOCHS" \
    --window 61 \
    --lr 1e-4 \
    --hidden-size 512 \
    --batch-size 256 \
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
    --early-stop-patience 12 \
    --early-stop-min-delta 5e-8 \
    --max-train-sequences "$AMASS_MAX_TRAIN" \
    --max-val-sequences "$AMASS_MAX_VAL"
fi

if [ ! -f "$DIP_OUT/train_result.json" ]; then
  "$PY" pl_curve_train.py \
    --train-cache "$DIP_TRAIN_CACHE" \
    --val-cache "$DIP_VAL_CACHE" \
    --output-dir "$DIP_OUT" \
    --experiment-name rjs_sensitive_newpl_dip_stage_b \
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
    --early-stop-patience 10 \
    --early-stop-min-delta 5e-8 \
    --max-train-sequences "$DIP_MAX_TRAIN" \
    --max-val-sequences "$DIP_MAX_VAL"
fi

run_module_eval() {
  local ckpt_name="$1"
  local ckpt_path="$2"
  local split="$3"
  local cache="$4"
  "$PY" pl_curve_pl_accuracy_eval.py \
    --pl-cache "$cache" \
    --checkpoint "$ckpt_path" \
    --output-json "$EVAL_DIR/${ckpt_name}_${split}_module_pl_accuracy.json" \
    --max-sequences "$EVAL_MAX_SEQS"
}

run_swap_eval() {
  local ckpt_name="$1"
  local ckpt_path="$2"
  local split="$3"
  local cache="$4"
  "$PY" pl_curve_offset_swap_eval.py \
    --pl-cache "$cache" \
    --checkpoint "$ckpt_path" \
    --output-json "$EVAL_DIR/${ckpt_name}_${split}_offset_swap.json" \
    --max-sequences "$EVAL_MAX_SEQS" \
    --swap-feature-offset \
    --variants good,zero,roll_sensors,other_sequence,negate
}

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
    run_module_eval "$ckpt_name" "$ckpt_path" "$split" "$cache"
  done
  for split in dip_test tc_test; do
    case "$split" in
      dip_test) cache="$DIP_TEST_CACHE" ;;
      tc_test) cache="$TC_TEST_CACHE" ;;
    esac
    run_swap_eval "$ckpt_name" "$ckpt_path" "$split" "$cache"
  done
done

"$PY" scripts/summarize_rjs_sensitive_newpl.py --root "$ROOT"

if [ "$RUN_FULL_IF_PASS" = "1" ]; then
  SHOULD_RUN_FULL="$("$PY" - "$ROOT/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("1" if d["gates"]["full_pipeline_allowed"]["passed"] else "0")
PY
)"
  if [ "$SHOULD_RUN_FULL" = "1" ]; then
    "$PY" pl_curve_eval.py \
      --val-cache "$DIP_TEST_SRC" \
      --checkpoint "$DIP_OUT/best_loss.pt" \
      --output-json "$FULL_DIR/dip_test_full_pipeline.json" \
      --imu-input-mode official
    "$PY" pl_curve_eval.py \
      --val-cache "$TC_S4_CACHE" \
      --checkpoint "$DIP_OUT/best_loss.pt" \
      --output-json "$FULL_DIR/tc_s4_full_pipeline.json" \
      --imu-input-mode official
    "$PY" pl_curve_eval.py \
      --val-cache "$TC_S5_CACHE" \
      --checkpoint "$DIP_OUT/best_loss.pt" \
      --output-json "$FULL_DIR/tc_s5_full_pipeline.json" \
      --imu-input-mode official
  else
    echo "full pipeline skipped because summary gates did not pass"
  fi
fi

echo "finished $(date --iso-8601=seconds)"
