#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ENV_DIR:-}" ]]; then
  if [[ -x /home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python ]]; then
    ENV_DIR=/home/lingfeng/remote-envs/globalpose-gpu-py310
  else
    ENV_DIR=/home/lingfeng/.conda/envs/globalpose-gpu
  fi
fi
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
PY="$ENV_DIR/bin/python"

ROOT=${ROOT:-data/experiments/newpl_offset_v6_control_only_20260609}
CACHE_ROOT="$ROOT/caches"
EVAL_DIR="$ROOT/eval"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$EVAL_DIR" "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

SMOKE=${SMOKE:-0}
AMASS_BATCH=${AMASS_BATCH:-256}
DIP_BATCH=${DIP_BATCH:-32}
AMASS_IMU_PROXY_WEIGHT=${AMASS_IMU_PROXY_WEIGHT:-0.0}
DIP_IMU_PROXY_WEIGHT=${DIP_IMU_PROXY_WEIGHT:-0.0}
IMU_PROXY_ACC_SCALE=${IMU_PROXY_ACC_SCALE:-30.0}
REUSE_PL_CACHE_ROOT=${REUSE_PL_CACHE_ROOT:-data/experiments/offset_aware_newpl_20260607_longrun_v1/caches}

AMASS_SRC=${AMASS_SRC:-data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay_processed_fields/baseline_cache_manifest.json}
DIP_TRAIN_SRC=${DIP_TRAIN_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json}
DIP_VAL_SRC=${DIP_VAL_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json}
DIP_TEST_SRC=${DIP_TEST_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json}
TC_TEST_SRC=${TC_TEST_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json}

AMASS_GT_CONTROL=${AMASS_GT_CONTROL:-data/dataset_work/GTControlCache/amass_train/gt_control_cache_manifest.json}
DIP_TRAIN_GT_CONTROL=${DIP_TRAIN_GT_CONTROL:-data/dataset_work/GTControlCache/dip_train/gt_control_cache_manifest.json}
DIP_VAL_GT_CONTROL=${DIP_VAL_GT_CONTROL:-data/dataset_work/GTControlCache/dip_val/gt_control_cache_manifest.json}

AMASS_OFFICIAL_PL_CACHE=${AMASS_OFFICIAL_PL_CACHE:-data/experiments/newpl_v5_official_protocol_20260607/caches/pl_amass_official_init36/pl_curve_cache_manifest.json}
DIP_TEST_OFFICIAL_PL_CACHE=${DIP_TEST_OFFICIAL_PL_CACHE:-data/experiments/newpl_v5_official_protocol_20260607/caches/pl_dip_test_official_init36/pl_curve_cache_manifest.json}
TC_TEST_OFFICIAL_PL_CACHE=${TC_TEST_OFFICIAL_PL_CACHE:-data/experiments/newpl_v5_official_protocol_20260607/caches/pl_tc_test_official_init36/pl_curve_cache_manifest.json}

NEWPL_V4_INIT36=${NEWPL_V4_INIT36:-}
NEWPL_V5_DIP=${NEWPL_V5_DIP:-data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt}
CANONICAL_DIP=${CANONICAL_DIP:-data/experiments/newpl_diff_vs_control_ablation_20260608/canonical_control/dip_finetune/best_loss.pt}

AMASS_CACHE="$CACHE_ROOT/pl_amass_offset_aware/pl_curve_cache_manifest.json"
DIP_TRAIN_CACHE="$CACHE_ROOT/pl_dip_train_offset_aware/pl_curve_cache_manifest.json"
DIP_VAL_CACHE="$CACHE_ROOT/pl_dip_val_offset_aware/pl_curve_cache_manifest.json"
DIP_TEST_CACHE="$CACHE_ROOT/pl_dip_test_offset_aware/pl_curve_cache_manifest.json"
TC_TEST_CACHE="$CACHE_ROOT/pl_tc_test_offset_aware/pl_curve_cache_manifest.json"

if [[ -n "$REUSE_PL_CACHE_ROOT" && -f "$REUSE_PL_CACHE_ROOT/pl_amass_offset_aware/pl_curve_cache_manifest.json" ]]; then
  echo "Reusing offset-aware PL caches from $REUSE_PL_CACHE_ROOT"
  AMASS_CACHE="$REUSE_PL_CACHE_ROOT/pl_amass_offset_aware/pl_curve_cache_manifest.json"
  DIP_TRAIN_CACHE="$REUSE_PL_CACHE_ROOT/pl_dip_train_offset_aware/pl_curve_cache_manifest.json"
  DIP_VAL_CACHE="$REUSE_PL_CACHE_ROOT/pl_dip_val_offset_aware/pl_curve_cache_manifest.json"
  DIP_TEST_CACHE="$REUSE_PL_CACHE_ROOT/pl_dip_test_offset_aware/pl_curve_cache_manifest.json"
  TC_TEST_CACHE="$REUSE_PL_CACHE_ROOT/pl_tc_test_offset_aware/pl_curve_cache_manifest.json"
  SKIP_PL_CACHE_BUILD=1
else
  SKIP_PL_CACHE_BUILD=0
fi

AMASS_OUT="$ROOT/amass_pretrain"
DIP_OUT="$ROOT/dip_finetune"

echo "started $(date --iso-8601=seconds)"
echo "ROOT=$ROOT SMOKE=$SMOKE CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "AMASS_BATCH=$AMASS_BATCH DIP_BATCH=$DIP_BATCH"
echo "AMASS_IMU_PROXY_WEIGHT=$AMASS_IMU_PROXY_WEIGHT DIP_IMU_PROXY_WEIGHT=$DIP_IMU_PROXY_WEIGHT IMU_PROXY_ACC_SCALE=$IMU_PROXY_ACC_SCALE"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "mem", torch.cuda.get_device_properties(0).total_memory)
PY

if [[ "$SMOKE" == "1" ]]; then
  CACHE_MAX_SEQS=24
  AMASS_EPOCHS=1
  DIP_EPOCHS=1
  AMASS_MAX_TRAIN=24
  AMASS_MAX_VAL=6
  DIP_MAX_TRAIN=8
  DIP_MAX_VAL=4
  EVAL_MAX_SEQS=3
  SWAP_MAX_SEQS=3
else
  CACHE_MAX_SEQS=0
  AMASS_EPOCHS=${AMASS_EPOCHS:-80}
  DIP_EPOCHS=${DIP_EPOCHS:-40}
  AMASS_MAX_TRAIN=0
  AMASS_MAX_VAL=${AMASS_MAX_VAL:-20}
  DIP_MAX_TRAIN=0
  DIP_MAX_VAL=0
  EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-50}
  SWAP_MAX_SEQS=${SWAP_MAX_SEQS:-20}
fi

ensure_pl_cache() {
  local src="$1"
  local out_dir="$2"
  if [[ ! -f "$out_dir/pl_curve_cache_manifest.json" ]]; then
    "$PY" pl_curve_cache.py \
      --input-cache "$src" \
      --output-dir "$out_dir" \
      --shard-size 100 \
      --imu-input-mode official \
      --feature-mode offset_aware \
      --max-sequences "$CACHE_MAX_SEQS"
  fi
}

if [[ "$SKIP_PL_CACHE_BUILD" != "1" ]]; then
  ensure_pl_cache "$AMASS_SRC" "$CACHE_ROOT/pl_amass_offset_aware"
  ensure_pl_cache "$DIP_TRAIN_SRC" "$CACHE_ROOT/pl_dip_train_offset_aware"
  ensure_pl_cache "$DIP_VAL_SRC" "$CACHE_ROOT/pl_dip_val_offset_aware"
  ensure_pl_cache "$DIP_TEST_SRC" "$CACHE_ROOT/pl_dip_test_offset_aware"
  ensure_pl_cache "$TC_TEST_SRC" "$CACHE_ROOT/pl_tc_test_offset_aware"
fi

COMMON_LOSS_ARGS=(
  --disable-ik-distill
  --selection-metric control_physical
  --pRB-weight 1.0
  --gR1-weight 1.0
  --baseline-pRB-weight 0.05
  --baseline-gR1-weight 0.0
  --gt-control-pRB-weight 0.3
  --gt-control-gR1-weight 0.1
  --pRB-dot-weight 0.0
  --pRB-ddot-weight 0.0
  --pRB-ddot-smooth-weight 0.0
  --gR1-dot-weight 0.0
  --gR1-ddot-weight 0.0
)

if [[ ! -f "$AMASS_OUT/train_result.json" ]]; then
  "$PY" pl_curve_train.py \
    --train-cache "$AMASS_CACHE" \
    --val-cache "$AMASS_CACHE" \
    --output-dir "$AMASS_OUT" \
    --experiment-name newpl_offset_v6_control_only_amass \
    --epochs "$AMASS_EPOCHS" \
    --window 61 \
    --lr 1e-4 \
    --hidden-size 512 \
    --batch-size "$AMASS_BATCH" \
    --dropout 0.15 \
    --model-variant offset_aware \
    --input-size 156 \
    --init-size 36 \
    --film-scale 0.3 \
    --offset-embed-size 128 \
    --imu-proxy-offset-acc-weight "$AMASS_IMU_PROXY_WEIGHT" \
    --imu-proxy-acc-scale "$IMU_PROXY_ACC_SCALE" \
    --train-gt-control-cache "$AMASS_GT_CONTROL" \
    --val-gt-control-cache "$AMASS_GT_CONTROL" \
    --early-stop-patience 12 \
    --early-stop-min-delta 5e-8 \
    --max-train-sequences "$AMASS_MAX_TRAIN" \
    --max-val-sequences "$AMASS_MAX_VAL" \
    --val-window-length 61 \
    "${COMMON_LOSS_ARGS[@]}"
fi

if [[ ! -f "$DIP_OUT/train_result.json" ]]; then
  "$PY" pl_curve_train.py \
    --train-cache "$DIP_TRAIN_CACHE" \
    --val-cache "$DIP_VAL_CACHE" \
    --output-dir "$DIP_OUT" \
    --experiment-name newpl_offset_v6_control_only_dip \
    --epochs "$DIP_EPOCHS" \
    --window 61 \
    --lr 5e-6 \
    --hidden-size 512 \
    --batch-size "$DIP_BATCH" \
    --dropout 0.05 \
    --model-variant offset_aware \
    --input-size 156 \
    --init-size 36 \
    --film-scale 0.3 \
    --offset-embed-size 128 \
    --init-checkpoint "$AMASS_OUT/best_loss.pt" \
    --imu-proxy-offset-acc-weight "$DIP_IMU_PROXY_WEIGHT" \
    --imu-proxy-acc-scale "$IMU_PROXY_ACC_SCALE" \
    --train-gt-control-cache "$DIP_TRAIN_GT_CONTROL" \
    --val-gt-control-cache "$DIP_VAL_GT_CONTROL" \
    --early-stop-patience 10 \
    --early-stop-min-delta 5e-8 \
    --max-train-sequences "$DIP_MAX_TRAIN" \
    --max-val-sequences "$DIP_MAX_VAL" \
    --val-window-length 61 \
    "${COMMON_LOSS_ARGS[@]}"
fi

run_eval() {
  local version="$1"
  local ckpt="$2"
  local dataset="$3"
  local cache="$4"
  local out="$EVAL_DIR/${version}_${dataset}.json"
  if [[ -n "$ckpt" && -f "$ckpt" && ! -f "$out" ]]; then
    "$PY" pl_curve_pl_accuracy_eval.py \
      --pl-cache "$cache" \
      --checkpoint "$ckpt" \
      --output-json "$out" \
      --max-sequences "$EVAL_MAX_SEQS"
  fi
}

run_swap_eval() {
  local version="$1"
  local ckpt="$2"
  local dataset="$3"
  local cache="$4"
  local out="$EVAL_DIR/${version}_${dataset}_offset_swap.json"
  if [[ -f "$ckpt" && ! -f "$out" ]]; then
    "$PY" pl_curve_offset_swap_eval.py \
      --pl-cache "$cache" \
      --checkpoint "$ckpt" \
      --output-json "$out" \
      --max-sequences "$SWAP_MAX_SEQS" \
      --swap-feature-offset \
      --variants good,zero,roll_sensors,other_sequence,negate
  fi
}

for dataset in amass dip_test tc_test; do
  case "$dataset" in
    amass)
      official_cache="$AMASS_OFFICIAL_PL_CACHE"
      offset_cache="$AMASS_CACHE"
      ;;
    dip_test)
      official_cache="$DIP_TEST_OFFICIAL_PL_CACHE"
      offset_cache="$DIP_TEST_CACHE"
      ;;
    tc_test)
      official_cache="$TC_TEST_OFFICIAL_PL_CACHE"
      offset_cache="$TC_TEST_CACHE"
      ;;
  esac
  run_eval newpl_v4_init36 "$NEWPL_V4_INIT36" "$dataset" "$official_cache"
  run_eval newpl_v5_dip_best "$NEWPL_V5_DIP" "$dataset" "$official_cache"
  run_eval canonical_control_dip_best "$CANONICAL_DIP" "$dataset" "$official_cache"
  run_eval newpl_offset_v6_best "$DIP_OUT/best_loss.pt" "$dataset" "$offset_cache"
  run_eval newpl_offset_v6_last "$DIP_OUT/last.pt" "$dataset" "$offset_cache"
done

for dataset in dip_test tc_test; do
  case "$dataset" in
    dip_test) offset_cache="$DIP_TEST_CACHE" ;;
    tc_test) offset_cache="$TC_TEST_CACHE" ;;
  esac
  run_swap_eval newpl_offset_v6_best "$DIP_OUT/best_loss.pt" "$dataset" "$offset_cache"
  run_swap_eval newpl_offset_v6_last "$DIP_OUT/last.pt" "$dataset" "$offset_cache"
done

"$PY" scripts/summarize_newpl_offset_v6.py --root "$ROOT"
echo "finished $(date --iso-8601=seconds)"
