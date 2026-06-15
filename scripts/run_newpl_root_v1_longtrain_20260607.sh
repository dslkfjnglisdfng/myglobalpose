#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=/home/lingfeng/.conda/envs/globalpose-gpu
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

PY="$ENV_DIR/bin/python"
ROOT="data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect"

AMASS_CACHE="data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json"
TC_TRAIN="data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json"
TC_TEST="data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json"
DIP_TRAIN="data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json"
DIP_TEST="data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only_with_offset_r/baseline_cache_manifest.json"
NEWPL_V4="data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt"

mkdir -p "$ROOT/eval"

"$PY" newpl_root_train.py \
  --train-cache "$AMASS_CACHE" \
  --val-cache "$AMASS_CACHE" \
  --output-dir "$ROOT/amass_pretrain" \
  --experiment-name newpl_root_v1_amass_pretrain \
  --dataset amass \
  --imu-input-mode official \
  --root-vel-mode gt \
  --epochs 60 \
  --window 61 \
  --batch-size 2048 \
  --selection-metric control_root_physical \
  --lr 1e-4 \
  --max-val-sequences 10 \
  --early-stop-min-delta 1e-5 \
  --early-stop-patience 8 \
  --init-checkpoint "$NEWPL_V4"

"$PY" newpl_root_eval.py \
  --cache "$AMASS_CACHE" \
  --output-json "$ROOT/eval/amass_module_metrics.json" \
  --dataset amass \
  --dataset-label AMASS-val20 \
  --imu-input-mode official \
  --root-vel-gt \
  --max-eval-sequences 20 \
  --version official_PL=official \
  --version newpl_v4_init36="$NEWPL_V4" \
  --version newpl_root_v1_amass_pretrain="$ROOT/amass_pretrain/best_loss.pt"

"$PY" newpl_root_train.py \
  --train-cache "$TC_TRAIN" \
  --val-cache "$TC_TEST" \
  --output-dir "$ROOT/tc_finetune" \
  --experiment-name newpl_root_v1_tc_finetune \
  --dataset totalcapture \
  --imu-input-mode processed \
  --root-vel-mode gt \
  --epochs 20 \
  --window 61 \
  --batch-size 64 \
  --selection-metric control_root_physical \
  --lr 1e-5 \
  --early-stop-min-delta 1e-5 \
  --early-stop-patience 6 \
  --init-checkpoint "$ROOT/amass_pretrain/best_loss.pt"

"$PY" newpl_root_eval.py \
  --cache "$TC_TEST" \
  --output-json "$ROOT/eval/tc_test_module_metrics.json" \
  --dataset totalcapture \
  --dataset-label TotalCapture-test \
  --imu-input-mode processed \
  --root-vel-gt \
  --version official_PL=official \
  --version newpl_v4_init36="$NEWPL_V4" \
  --version newpl_root_v1_amass_pretrain="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_root_v1_tc_finetune="$ROOT/tc_finetune/best_loss.pt"

"$PY" newpl_root_train.py \
  --train-cache "$DIP_TRAIN" \
  --val-cache "$DIP_TEST" \
  --output-dir "$ROOT/dip_finetune" \
  --experiment-name newpl_root_v1_dip_finetune \
  --dataset dip \
  --imu-input-mode official \
  --root-vel-mode none \
  --freeze-root-head \
  --allow-zero-offset-init \
  --epochs 20 \
  --window 61 \
  --batch-size 64 \
  --selection-metric control_physical \
  --lr 1e-5 \
  --early-stop-min-delta 1e-5 \
  --early-stop-patience 6 \
  --init-checkpoint "$ROOT/amass_pretrain/best_loss.pt"

"$PY" newpl_root_eval.py \
  --cache "$DIP_TEST" \
  --output-json "$ROOT/eval/dip_test_module_metrics.json" \
  --dataset dip \
  --dataset-label DIP-IMU-test \
  --imu-input-mode official \
  --version official_PL=official \
  --version newpl_v4_init36="$NEWPL_V4" \
  --version newpl_root_v1_amass_pretrain="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_root_v1_dip_finetune="$ROOT/dip_finetune/best_loss.pt"
