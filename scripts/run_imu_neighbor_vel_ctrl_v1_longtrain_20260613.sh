#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/experiments/imu_neighbor_vel_ctrl_v1_longtrain_20260613}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

AMASS_CACHE="data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json"
TC_TRAIN_CACHE="data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json"
TC_VAL_CACHE="data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json"
TC_TEST_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json"
DIP_TRAIN_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json"
DIP_VAL_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json"
DIP_TEST_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json"

AMASS_BATCH="${AMASS_BATCH:-1536}"
TC_BATCH="${TC_BATCH:-128}"
DIP_BATCH="${DIP_BATCH:-64}"
WINDOW="${WINDOW:-61}"
HIDDEN="${HIDDEN:-512}"
LAYERS="${LAYERS:-2}"
DROPOUT="${DROPOUT:-0.2}"

mkdir -p "$ROOT/logs" "$ROOT/eval"
LOG="$ROOT/logs/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "[imu_neighbor_vel_ctrl_v1_longtrain] start $(date --iso-8601=seconds)"
echo "root=$ROOT"
echo "gpu=$GPU"
echo "amass_batch=$AMASS_BATCH tc_batch=$TC_BATCH dip_batch=$DIP_BATCH window=$WINDOW hidden=$HIDDEN layers=$LAYERS"
echo "DIP policy: no world velocity/root velocity/acceleration GT; distill/smooth only."
echo "No full-pipeline 11 metrics are run by this script."

if [ -e "$ROOT/amass_pretrain/train_result.json" ]; then
  echo "Refusing to overwrite existing AMASS output under $ROOT/amass_pretrain"
  exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_vel_ctrl_train.py \
  --train-cache "$AMASS_CACHE" \
  --val-cache "$AMASS_CACHE" \
  --output-dir "$ROOT/amass_pretrain" \
  --dataset amass \
  --imu-input-mode official \
  --world-gt-mode auto \
  --epochs "${AMASS_EPOCHS:-80}" \
  --window "$WINDOW" \
  --batch-size "$AMASS_BATCH" \
  --lr "${AMASS_LR:-1e-4}" \
  --hidden-size "$HIDDEN" \
  --num-layers "$LAYERS" \
  --dropout "$DROPOUT" \
  --max-val-sequences "${AMASS_VAL_SEQS:-64}" \
  --early-stop-patience "${AMASS_PATIENCE:-12}" \
  --early-stop-min-delta "${AMASS_MIN_DELTA:-1e-5}"

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_vel_ctrl_eval.py \
  --cache "$AMASS_CACHE" \
  --checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --output-json "$ROOT/eval/eval_amass_after_amass_best.json" \
  --dataset amass \
  --imu-input-mode official \
  --world-gt-mode auto \
  --max-sequences "${AMASS_EVAL_SEQS:-128}"

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_vel_ctrl_eval.py \
  --cache "$TC_TEST_CACHE" \
  --checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --output-json "$ROOT/eval/eval_totalcapture_test_after_amass_best.json" \
  --dataset totalcapture \
  --imu-input-mode official \
  --world-gt-mode auto

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_vel_ctrl_eval.py \
  --cache "$DIP_TEST_CACHE" \
  --checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --output-json "$ROOT/eval/eval_dip_test_after_amass_best.json" \
  --dataset dip \
  --imu-input-mode official \
  --world-gt-mode auto

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_vel_ctrl_train.py \
  --train-cache "$TC_TRAIN_CACHE" \
  --val-cache "$TC_VAL_CACHE" \
  --output-dir "$ROOT/totalcapture_finetune" \
  --dataset totalcapture \
  --imu-input-mode official \
  --world-gt-mode auto \
  --init-checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --epochs "${TC_EPOCHS:-60}" \
  --window "$WINDOW" \
  --batch-size "$TC_BATCH" \
  --lr "${TC_LR:-1e-5}" \
  --hidden-size "$HIDDEN" \
  --num-layers "$LAYERS" \
  --dropout "$DROPOUT" \
  --early-stop-patience "${TC_PATIENCE:-12}" \
  --early-stop-min-delta "${TC_MIN_DELTA:-1e-5}"

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_vel_ctrl_eval.py \
  --cache "$TC_TEST_CACHE" \
  --checkpoint "$ROOT/totalcapture_finetune/best_loss.pt" \
  --output-json "$ROOT/eval/eval_totalcapture_test_after_tc_finetune_best.json" \
  --dataset totalcapture \
  --imu-input-mode official \
  --world-gt-mode auto

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_vel_ctrl_train.py \
  --train-cache "$DIP_TRAIN_CACHE" \
  --val-cache "$DIP_VAL_CACHE" \
  --output-dir "$ROOT/dip_distill" \
  --dataset dip \
  --imu-input-mode official \
  --world-gt-mode auto \
  --init-checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --teacher-checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --epochs "${DIP_EPOCHS:-20}" \
  --window "$WINDOW" \
  --batch-size "$DIP_BATCH" \
  --lr "${DIP_LR:-5e-6}" \
  --hidden-size "$HIDDEN" \
  --num-layers "$LAYERS" \
  --dropout "$DROPOUT" \
  --early-stop-patience "${DIP_PATIENCE:-8}" \
  --early-stop-min-delta "${DIP_MIN_DELTA:-1e-6}"

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_vel_ctrl_eval.py \
  --cache "$DIP_TEST_CACHE" \
  --checkpoint "$ROOT/dip_distill/best_loss.pt" \
  --output-json "$ROOT/eval/eval_dip_test_after_dip_distill_best.json" \
  --dataset dip \
  --imu-input-mode official \
  --world-gt-mode auto

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_vel_ctrl_eval.py \
  --cache "$TC_TEST_CACHE" \
  --checkpoint "$ROOT/dip_distill/best_loss.pt" \
  --output-json "$ROOT/eval/eval_totalcapture_test_after_dip_distill_best.json" \
  --dataset totalcapture \
  --imu-input-mode official \
  --world-gt-mode auto

python - <<'PY' "$ROOT"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {
    "root": str(root),
    "status": "completed",
    "amass_best": str(root / "amass_pretrain" / "best_loss.pt"),
    "tc_best": str(root / "totalcapture_finetune" / "best_loss.pt"),
    "dip_distill_best": str(root / "dip_distill" / "best_loss.pt"),
    "eval": {},
}
for path in sorted((root / "eval").glob("*.json")):
    data = json.loads(path.read_text())
    agg = data.get("aggregate", {})
    summary["eval"][path.name] = {
        "world_gt_status": agg.get("world_gt_status"),
        "velocity_L2_mps": agg.get("velocity_L2_mps"),
        "acceleration_L2_mps2": agg.get("acceleration_L2_mps2"),
        "root_velocity_L2_mps": agg.get("root_velocity_L2_mps"),
        "root_acceleration_L2_mps2": agg.get("root_acceleration_L2_mps2"),
        "baseline_velocity_source": agg.get("baseline_velocity_source"),
        "baseline_root_velocity_L2_mps": agg.get("baseline_root_velocity_L2_mps"),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[imu_neighbor_vel_ctrl_v1_longtrain] done $(date --iso-8601=seconds)"
