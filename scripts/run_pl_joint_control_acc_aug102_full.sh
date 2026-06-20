#!/usr/bin/env bash
set -euo pipefail

ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
PY=${PY:-$ENV/bin/python}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
ROOT=${ROOT:-data/experiments/pl_joint_control_acc_aug102_v1_full_${STAMP}}
CACHE_ROOT="$ROOT/cache"
TRAIN_ROOT="$ROOT/train"
EVAL_ROOT="$ROOT/eval"
LOG_ROOT="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$TRAIN_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

if [[ "${NO_TEE:-0}" != "1" ]]; then
  exec > >(tee -a "$LOG_ROOT/run.log") 2>&1
fi

echo "ROOT=$ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
date -Is

AMASS_RAW=${AMASS_RAW:-data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json}
DIP_TRAIN_RAW=${DIP_TRAIN_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json}
DIP_VAL_RAW=${DIP_VAL_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json}
DIP_TEST_RAW=${DIP_TEST_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json}

AMASS_GT=${AMASS_GT:-/home/lingfeng/projects/data/dataset_work/amass/gt_control/train/manifest.json}
DIP_TRAIN_GT=${DIP_TRAIN_GT:-/home/lingfeng/projects/data/dataset_work/dip/gt_control/train/manifest.json}
DIP_VAL_GT=${DIP_VAL_GT:-/home/lingfeng/projects/data/dataset_work/dip/gt_control/val/manifest.json}
DIP_TEST_GT=${DIP_TEST_GT:-/home/lingfeng/projects/data/dataset_work/dip/gt_control/test/manifest.json}

for required in "$AMASS_RAW" "$DIP_TRAIN_RAW" "$DIP_VAL_RAW" "$DIP_TEST_RAW" "$AMASS_GT" "$DIP_TRAIN_GT" "$DIP_VAL_GT" "$DIP_TEST_GT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done

build_cache() {
  local split="$1"
  local raw="$2"
  local gt="$3"
  local out="$CACHE_ROOT/$split"
  local manifest="$out/pl_curve_cache_manifest.json"
  if [[ ! -f "$manifest" ]]; then
    "$PY" pl_joint_control_acc_aug102_cache.py build \
      --input-cache "$raw" \
      --gt-control-cache "$gt" \
      --output-dir "$out" \
      --shard-size "${SHARD_SIZE:-100}" \
      --device "${CACHE_DEVICE:-cuda:0}"
  fi
  "$PY" pl_joint_control_acc_aug102_cache.py validate \
    --cache "$manifest" \
    --output-json "$EVAL_ROOT/cache_validation_${split}.json"
}

build_cache amass "$AMASS_RAW" "$AMASS_GT"
build_cache dip_train "$DIP_TRAIN_RAW" "$DIP_TRAIN_GT"
build_cache dip_val "$DIP_VAL_RAW" "$DIP_VAL_GT"
build_cache dip_test "$DIP_TEST_RAW" "$DIP_TEST_GT"

AMASS_DIR="$TRAIN_ROOT/amass_pretrain"
DIP_DIR="$TRAIN_ROOT/dip_finetune"

if [[ ! -f "$AMASS_DIR/best_loss.pt" ]]; then
  "$PY" pl_joint_control_acc_aug102_train.py \
    --train-cache "$CACHE_ROOT/amass/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/amass/pl_curve_cache_manifest.json" \
    --output-dir "$AMASS_DIR" \
    --experiment-name pl_joint_control_acc_aug102_v1_amass_pretrain \
    --epochs "${AMASS_EPOCHS:-80}" \
    --window "${WINDOW:-61}" \
    --batch-size "${AMASS_BATCH_SIZE:-256}" \
    --max-train-sequences "${AMASS_MAX_TRAIN_SEQUENCES:-0}" \
    --max-val-sequences "${AMASS_MAX_VAL_SEQUENCES:-20}" \
    --val-window-length "${WINDOW:-61}" \
    --hidden-size "${HIDDEN_SIZE:-512}" \
    --tail-length "${TAIL_LENGTH:-4}" \
    --residual-scale "${RESIDUAL_SCALE:-0.005}" \
    --dropout "${DROPOUT:-0.4}" \
    --grad-clip "${GRAD_CLIP:-1.0}" \
    --lr "${AMASS_LR:-1e-4}"
fi

"$PY" pl_joint_control_acc_aug102_eval.py \
  --cache "$CACHE_ROOT/dip_test/pl_curve_cache_manifest.json" \
  --checkpoint "$AMASS_DIR/best_loss.pt" \
  --output-json "$EVAL_ROOT/dip_test_after_amass.json" \
  --output-summary "$EVAL_ROOT/dip_test_after_amass.md" \
  --split dip_test_after_amass

if [[ ! -f "$DIP_DIR/best_loss.pt" ]]; then
  "$PY" pl_joint_control_acc_aug102_train.py \
    --train-cache "$CACHE_ROOT/dip_train/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/dip_val/pl_curve_cache_manifest.json" \
    --output-dir "$DIP_DIR" \
    --experiment-name pl_joint_control_acc_aug102_v1_dip_finetune \
    --epochs "${DIP_EPOCHS:-40}" \
    --window "${WINDOW:-61}" \
    --batch-size "${DIP_BATCH_SIZE:-12}" \
    --max-train-sequences "${DIP_MAX_TRAIN_SEQUENCES:-0}" \
    --max-val-sequences "${DIP_MAX_VAL_SEQUENCES:-0}" \
    --val-window-length "${WINDOW:-61}" \
    --hidden-size "${HIDDEN_SIZE:-512}" \
    --tail-length "${TAIL_LENGTH:-4}" \
    --residual-scale "${RESIDUAL_SCALE:-0.005}" \
    --dropout "${DROPOUT:-0.4}" \
    --grad-clip "${GRAD_CLIP:-1.0}" \
    --lr "${DIP_LR:-5e-6}" \
    --init-checkpoint "$AMASS_DIR/best_loss.pt"
fi

"$PY" pl_joint_control_acc_aug102_eval.py \
  --cache "$CACHE_ROOT/dip_test/pl_curve_cache_manifest.json" \
  --checkpoint "$DIP_DIR/best_loss.pt" \
  --output-json "$EVAL_ROOT/dip_test_after_dip.json" \
  --output-summary "$EVAL_ROOT/dip_test_after_dip.md" \
  --split dip_test_after_dip

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cache = json.loads((root / "cache/dip_test/pl_curve_cache_manifest.json").read_text())
amass = json.loads((root / "train/amass_pretrain/train_result.json").read_text())
dip = json.loads((root / "train/dip_finetune/train_result.json").read_text())
evals = {
    "dip_test_after_amass": json.loads((root / "eval/dip_test_after_amass.json").read_text()),
    "dip_test_after_dip": json.loads((root / "eval/dip_test_after_dip.json").read_text()),
}
summary = {
    "status": "ok",
    "root": str(root),
    "experiment": "pl_joint_control_acc_aug102_v1",
    "target_contract": cache["target_contract"],
    "feature_layout": cache["feature_layout"],
    "amass_best_loss": amass["best_loss"],
    "amass_best_epoch": amass["best_epoch"],
    "dip_best_loss": dip["best_loss"],
    "dip_best_epoch": dip["best_epoch"],
    "eval": {name: payload["summary"] for name, payload in evals.items()},
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "# PL Joint Control Acc-Aug102 v1 Full",
    "",
    f"- Root: `{root}`",
    f"- Target: {summary['target_contract']}",
    f"- Input layout: `{summary['feature_layout']}`",
    f"- AMASS best: epoch `{summary['amass_best_epoch']}`, loss `{summary['amass_best_loss']}`",
    f"- DIP best: epoch `{summary['dip_best_epoch']}`, loss `{summary['dip_best_loss']}`",
    "",
    "| split | joint_pos_l2_m | joint_vel_l2_mps | joint_acc_l2_mps2 | gravity_angle_deg |",
    "|---|---:|---:|---:|---:|",
]
for name, row in summary["eval"].items():
    lines.append(
        f"| {name} | {row['joint_pos_l2_m']:.6f} | {row['joint_vel_l2_mps']:.6f} | "
        f"{row['joint_acc_l2_mps2']:.6f} | {row['gravity_angle_deg']:.6f} |"
    )
lines += [
    "",
    "Scope: module-level joint-target PL control. No IK/full-pipeline/S4 evaluation is included.",
]
(root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"status": "ok", "summary": str(root / "SUMMARY.md")}, indent=2))
PY

date -Is
echo "done: $ROOT"
