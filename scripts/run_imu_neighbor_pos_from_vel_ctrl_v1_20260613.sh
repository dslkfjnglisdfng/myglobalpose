#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

AMASS_CACHE="data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json"
TC_TRAIN_CACHE="data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json"
TC_VAL_CACHE="data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json"
TC_TEST_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json"
DIP_TRAIN_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json"
DIP_VAL_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json"
DIP_TEST_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json"

VELOCITY_CKPT="${VELOCITY_CKPT:-data/experiments/imu_neighbor_vel_ctrl_v1_longtrain_20260613/amass_pretrain/best_loss.pt}"
WINDOW="${WINDOW:-61}"
HIDDEN="${HIDDEN:-512}"
LAYERS="${LAYERS:-2}"
DROPOUT="${DROPOUT:-0.2}"
PREFLIGHT="${PREFLIGHT:-1}"

mkdir -p "$ROOT/logs" "$ROOT/eval" "$ROOT/preflight"
LOG="$ROOT/logs/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "[imu_neighbor_pos_from_vel_ctrl_v1] start $(date --iso-8601=seconds)"
echo "root=$ROOT"
echo "gpu=$GPU"
echo "velocity_ckpt=$VELOCITY_CKPT"
echo "window=$WINDOW hidden=$HIDDEN layers=$LAYERS dropout=$DROPOUT"
echo "DIP policy: no DIP trans, no DIP world/root velocity GT; only pose-derived root-relative position supervision."
echo "No full-pipeline 11 metrics are run by this script."

if [ ! -f "$VELOCITY_CKPT" ]; then
  echo "Missing velocity checkpoint: $VELOCITY_CKPT"
  exit 2
fi

if [ -e "$ROOT/amass_pretrain/train_result.json" ]; then
  echo "Refusing to overwrite existing AMASS output under $ROOT/amass_pretrain"
  exit 2
fi

choose_batch() {
  local name="$1"
  local dataset="$2"
  local train_cache="$3"
  local val_cache="$4"
  local candidates="$5"
  local selected=""
  if [ "$PREFLIGHT" != "1" ]; then
    echo "$6"
    return 0
  fi
  for bs in $candidates; do
    local out="$ROOT/preflight/${name}_bs${bs}"
    local preflight_log="$ROOT/preflight/${name}_bs${bs}.log"
    if [ -e "$out/train_result.json" ]; then
      echo "preflight $name batch=$bs already ok" >&2
      selected="$bs"
      continue
    fi
    echo "preflight $name batch=$bs" >&2
    mkdir -p "$out"
    if CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_pos_from_vel_ctrl_train.py \
      --train-cache "$train_cache" \
      --val-cache "$val_cache" \
      --output-dir "$out" \
      --dataset "$dataset" \
      --velocity-checkpoint "$VELOCITY_CKPT" \
      --imu-input-mode official \
      --epochs 1 \
      --window "$WINDOW" \
      --batch-size "$bs" \
      --lr 1e-4 \
      --hidden-size "$HIDDEN" \
      --num-layers "$LAYERS" \
      --dropout "$DROPOUT" \
      --max-train-sequences "$bs" \
      --max-val-sequences 4 \
      --gt-vel-mix-start 0.0 \
      --gt-vel-mix-final 0.0 \
      --gt-vel-mix-epochs 1 \
      --val-gt-vel-mix-ratio 0.0 >"$preflight_log" 2>&1; then
      selected="$bs"
    else
      echo "preflight $name batch=$bs failed; keeping selected=${selected:-$6}; see $preflight_log" >&2
      break
    fi
  done
  echo "${selected:-$6}"
}

AMASS_BATCH="${AMASS_BATCH:-$(choose_batch amass amass "$AMASS_CACHE" "$AMASS_CACHE" "512 1024 1536" 512)}"
TC_BATCH="${TC_BATCH:-$(choose_batch totalcapture totalcapture "$TC_TRAIN_CACHE" "$TC_VAL_CACHE" "128 256 512" 128)}"
DIP_BATCH="${DIP_BATCH:-$(choose_batch dip dip "$DIP_TRAIN_CACHE" "$DIP_VAL_CACHE" "128 256 512" 128)}"
echo "selected batches: amass=$AMASS_BATCH totalcapture=$TC_BATCH dip=$DIP_BATCH"

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_pos_from_vel_ctrl_train.py \
  --train-cache "$AMASS_CACHE" \
  --val-cache "$AMASS_CACHE" \
  --output-dir "$ROOT/amass_pretrain" \
  --dataset amass \
  --velocity-checkpoint "$VELOCITY_CKPT" \
  --imu-input-mode official \
  --epochs "${AMASS_EPOCHS:-80}" \
  --window "$WINDOW" \
  --batch-size "$AMASS_BATCH" \
  --lr "${AMASS_LR:-1e-4}" \
  --hidden-size "$HIDDEN" \
  --num-layers "$LAYERS" \
  --dropout "$DROPOUT" \
  --max-val-sequences "${AMASS_VAL_SEQS:-64}" \
  --gt-vel-mix-start "${AMASS_GT_VEL_MIX_START:-0.75}" \
  --gt-vel-mix-final "${AMASS_GT_VEL_MIX_FINAL:-0.0}" \
  --gt-vel-mix-epochs "${AMASS_GT_VEL_MIX_EPOCHS:-30}" \
  --val-gt-vel-mix-ratio 0.0 \
  --early-stop-patience "${AMASS_PATIENCE:-12}" \
  --early-stop-min-delta "${AMASS_MIN_DELTA:-1e-5}"

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_pos_from_vel_ctrl_eval.py \
  --cache "$AMASS_CACHE" \
  --checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --velocity-checkpoint "$VELOCITY_CKPT" \
  --output-json "$ROOT/eval/eval_amass_after_amass_best.json" \
  --dataset amass \
  --imu-input-mode official \
  --gt-vel-input-ratio 0.0 \
  --max-sequences "${AMASS_EVAL_SEQS:-128}"

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_pos_from_vel_ctrl_eval.py \
  --cache "$TC_TEST_CACHE" \
  --checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --velocity-checkpoint "$VELOCITY_CKPT" \
  --output-json "$ROOT/eval/eval_totalcapture_test_after_amass_best.json" \
  --dataset totalcapture \
  --imu-input-mode official \
  --gt-vel-input-ratio 0.0

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_pos_from_vel_ctrl_eval.py \
  --cache "$DIP_TEST_CACHE" \
  --checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --velocity-checkpoint "$VELOCITY_CKPT" \
  --output-json "$ROOT/eval/eval_dip_test_after_amass_best.json" \
  --dataset dip \
  --imu-input-mode official \
  --gt-vel-input-ratio 0.0

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_pos_from_vel_ctrl_train.py \
  --train-cache "$TC_TRAIN_CACHE" \
  --val-cache "$TC_VAL_CACHE" \
  --output-dir "$ROOT/totalcapture_finetune" \
  --dataset totalcapture \
  --velocity-checkpoint "$VELOCITY_CKPT" \
  --imu-input-mode official \
  --init-checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --epochs "${TC_EPOCHS:-60}" \
  --window "$WINDOW" \
  --batch-size "$TC_BATCH" \
  --lr "${TC_LR:-1e-5}" \
  --hidden-size "$HIDDEN" \
  --num-layers "$LAYERS" \
  --dropout "$DROPOUT" \
  --gt-vel-mix-start "${TC_GT_VEL_MIX_START:-0.25}" \
  --gt-vel-mix-final "${TC_GT_VEL_MIX_FINAL:-0.0}" \
  --gt-vel-mix-epochs "${TC_GT_VEL_MIX_EPOCHS:-10}" \
  --val-gt-vel-mix-ratio 0.0 \
  --early-stop-patience "${TC_PATIENCE:-12}" \
  --early-stop-min-delta "${TC_MIN_DELTA:-1e-5}"

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_pos_from_vel_ctrl_eval.py \
  --cache "$TC_TEST_CACHE" \
  --checkpoint "$ROOT/totalcapture_finetune/best_loss.pt" \
  --velocity-checkpoint "$VELOCITY_CKPT" \
  --output-json "$ROOT/eval/eval_totalcapture_test_after_tc_finetune_best.json" \
  --dataset totalcapture \
  --imu-input-mode official \
  --gt-vel-input-ratio 0.0

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_pos_from_vel_ctrl_train.py \
  --train-cache "$DIP_TRAIN_CACHE" \
  --val-cache "$DIP_VAL_CACHE" \
  --output-dir "$ROOT/dip_finetune" \
  --dataset dip \
  --velocity-checkpoint "$VELOCITY_CKPT" \
  --imu-input-mode official \
  --init-checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
  --epochs "${DIP_EPOCHS:-30}" \
  --window "$WINDOW" \
  --batch-size "$DIP_BATCH" \
  --lr "${DIP_LR:-5e-6}" \
  --hidden-size "$HIDDEN" \
  --num-layers "$LAYERS" \
  --dropout "$DROPOUT" \
  --gt-vel-mix-start 0.0 \
  --gt-vel-mix-final 0.0 \
  --gt-vel-mix-epochs 1 \
  --val-gt-vel-mix-ratio 0.0 \
  --early-stop-patience "${DIP_PATIENCE:-8}" \
  --early-stop-min-delta "${DIP_MIN_DELTA:-1e-6}"

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_pos_from_vel_ctrl_eval.py \
  --cache "$DIP_TEST_CACHE" \
  --checkpoint "$ROOT/dip_finetune/best_loss.pt" \
  --velocity-checkpoint "$VELOCITY_CKPT" \
  --output-json "$ROOT/eval/eval_dip_test_after_dip_finetune_best.json" \
  --dataset dip \
  --imu-input-mode official \
  --gt-vel-input-ratio 0.0

CUDA_VISIBLE_DEVICES="$GPU" python imu_neighbor_pos_from_vel_ctrl_eval.py \
  --cache "$TC_TEST_CACHE" \
  --checkpoint "$ROOT/dip_finetune/best_loss.pt" \
  --velocity-checkpoint "$VELOCITY_CKPT" \
  --output-json "$ROOT/eval/eval_totalcapture_test_after_dip_finetune_best.json" \
  --dataset totalcapture \
  --imu-input-mode official \
  --gt-vel-input-ratio 0.0

python - <<'PY' "$ROOT"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {
    "root": str(root),
    "status": "completed",
    "module": "imu_neighbor_pos_from_vel_ctrl_v1",
    "amass_best": str(root / "amass_pretrain" / "best_loss.pt"),
    "tc_best": str(root / "totalcapture_finetune" / "best_loss.pt"),
    "dip_best": str(root / "dip_finetune" / "best_loss.pt"),
    "eval": {},
}
for path in sorted((root / "eval").glob("*.json")):
    data = json.loads(path.read_text())
    agg = data.get("aggregate", {})
    summary["eval"][path.name] = {
        "pos_R_L1_cm": agg.get("pos_R_L1_cm"),
        "pos_R_L2_cm": agg.get("pos_R_L2_cm"),
        "vel_R_L2_cm_s": agg.get("vel_R_L2_cm_s"),
        "acc_R_L2_cm_s2": agg.get("acc_R_L2_cm_s2"),
        "segment_length_error_cm": agg.get("segment_length_error_cm"),
        "baseline_source": agg.get("baseline_source"),
        "baseline_pos_R_L2_cm": agg.get("baseline_pos_R_L2_cm"),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[imu_neighbor_pos_from_vel_ctrl_v1] done $(date --iso-8601=seconds)"
