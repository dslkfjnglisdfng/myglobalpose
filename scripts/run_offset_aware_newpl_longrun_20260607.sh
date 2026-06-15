#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=/home/lingfeng/.conda/envs/globalpose-gpu
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

ROOT=data/experiments/offset_aware_newpl_20260607_longrun_v1
CACHE_ROOT="$ROOT/caches"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

PY="$ENV_DIR/bin/python"

AMASS_SRC=data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay_processed_fields/baseline_cache_manifest.json
DIP_TRAIN_SRC=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json
DIP_VAL_SRC=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json

AMASS_CACHE="$CACHE_ROOT/pl_amass_offset_aware"
DIP_TRAIN_CACHE="$CACHE_ROOT/pl_dip_train_offset_aware"
DIP_VAL_CACHE="$CACHE_ROOT/pl_dip_val_offset_aware"
AMASS_OUT="$ROOT/amass_pretrain"
DIP_OUT="$ROOT/dip_finetune"

echo "started $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

if [ ! -f "$AMASS_CACHE/pl_curve_cache_manifest.json" ]; then
  "$PY" pl_curve_cache.py \
    --input-cache "$AMASS_SRC" \
    --output-dir "$AMASS_CACHE" \
    --shard-size 100 \
    --imu-input-mode official \
    --feature-mode offset_aware
fi

if [ ! -f "$DIP_TRAIN_CACHE/pl_curve_cache_manifest.json" ]; then
  "$PY" pl_curve_cache.py \
    --input-cache "$DIP_TRAIN_SRC" \
    --output-dir "$DIP_TRAIN_CACHE" \
    --shard-size 100 \
    --imu-input-mode official \
    --feature-mode offset_aware
fi

if [ ! -f "$DIP_VAL_CACHE/pl_curve_cache_manifest.json" ]; then
  "$PY" pl_curve_cache.py \
    --input-cache "$DIP_VAL_SRC" \
    --output-dir "$DIP_VAL_CACHE" \
    --shard-size 100 \
    --imu-input-mode official \
    --feature-mode offset_aware
fi

if [ ! -f "$AMASS_OUT/train_result.json" ]; then
  "$PY" pl_curve_train.py \
    --train-cache "$AMASS_CACHE/pl_curve_cache_manifest.json" \
    --val-cache "$AMASS_CACHE/pl_curve_cache_manifest.json" \
    --output-dir "$AMASS_OUT" \
    --experiment-name offset_aware_newpl_amass_pretrain \
    --epochs 80 \
    --window 61 \
    --lr 1e-4 \
    --hidden-size 512 \
    --batch-size 256 \
    --disable-ik-distill \
    --model-variant offset_aware \
    --init-size 36 \
    --selection-metric control_physical \
    --gt-control-pRB-weight 0.3 \
    --gt-control-gR1-weight 0.1 \
    --pRB-ddot-smooth-weight 1e-6 \
    --early-stop-patience 12 \
    --early-stop-min-delta 5e-8 \
    --max-val-sequences 20 \
    --offset-consistency-weight 0.1 \
    --offset-consistency-target full_pl
fi

if [ ! -f "$DIP_OUT/train_result.json" ]; then
  "$PY" pl_curve_train.py \
    --train-cache "$DIP_TRAIN_CACHE/pl_curve_cache_manifest.json" \
    --val-cache "$DIP_VAL_CACHE/pl_curve_cache_manifest.json" \
    --output-dir "$DIP_OUT" \
    --experiment-name offset_aware_newpl_dip_finetune \
    --epochs 40 \
    --window 61 \
    --lr 5e-6 \
    --hidden-size 512 \
    --batch-size 12 \
    --disable-ik-distill \
    --model-variant offset_aware \
    --init-size 36 \
    --init-checkpoint "$AMASS_OUT/best_loss.pt" \
    --selection-metric control_physical \
    --gt-control-pRB-weight 0.3 \
    --gt-control-gR1-weight 0.1 \
    --pRB-ddot-smooth-weight 1e-6 \
    --early-stop-patience 10 \
    --early-stop-min-delta 5e-8
fi

echo "finished $(date --iso-8601=seconds)"
