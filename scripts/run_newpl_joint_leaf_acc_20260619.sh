#!/usr/bin/env bash
set -euo pipefail

ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
PY="$ENV/bin/python"

ROOT=${ROOT:-data/experiments/newpl_joint_leaf_acc_20260619/full}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/caches}
EVAL_ROOT="$ROOT/eval"
LOG_ROOT="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

if [[ "${NO_TEE:-0}" != "1" ]]; then
  exec > >(tee -a "$LOG_ROOT/run.log") 2>&1
fi

echo "ROOT=$ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
date -Is

AMASS_RAW=data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
DIP_TRAIN_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json
DIP_VAL_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json
DIP_TEST_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
TC_TEST_RAW=data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json

AMASS_GT=/home/lingfeng/projects/data/dataset_work/amass/gt_control/train/manifest.json
DIP_TRAIN_GT=/home/lingfeng/projects/data/dataset_work/dip/gt_control/train/manifest.json
DIP_VAL_GT=/home/lingfeng/projects/data/dataset_work/dip/gt_control/val/manifest.json
DIP_TEST_GT=/home/lingfeng/projects/data/dataset_work/dip/gt_control/test/manifest.json
TC_TEST_GT=/home/lingfeng/projects/data/dataset_work/totalcapture/gt_control/test/manifest.json

for required in "$AMASS_RAW" "$DIP_TRAIN_RAW" "$DIP_VAL_RAW" "$DIP_TEST_RAW" "$TC_TEST_RAW" "$AMASS_GT" "$DIP_TRAIN_GT" "$DIP_VAL_GT" "$DIP_TEST_GT" "$TC_TEST_GT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done

if [[ "${SMOKE:-0}" == "1" ]]; then
  CACHE_MAX=${CACHE_MAX:-2}
  CACHE_MAX_FRAMES=${CACHE_MAX_FRAMES:-120}
  AMASS_EPOCHS=${AMASS_EPOCHS:-1}
  DIP_EPOCHS=${DIP_EPOCHS:-1}
  AMASS_BATCH=${AMASS_BATCH:-16}
  DIP_BATCH=${DIP_BATCH:-8}
  AMASS_MAX_TRAIN=${AMASS_MAX_TRAIN:-2}
  AMASS_MAX_VAL=${AMASS_MAX_VAL:-2}
  DIP_MAX_TRAIN=${DIP_MAX_TRAIN:-2}
  DIP_MAX_VAL=${DIP_MAX_VAL:-2}
  EVAL_MAX=${EVAL_MAX:-2}
else
  CACHE_MAX=${CACHE_MAX:-0}
  CACHE_MAX_FRAMES=${CACHE_MAX_FRAMES:-0}
  AMASS_EPOCHS=${AMASS_EPOCHS:-80}
  DIP_EPOCHS=${DIP_EPOCHS:-40}
  AMASS_BATCH=${AMASS_BATCH:-256}
  DIP_BATCH=${DIP_BATCH:-12}
  AMASS_MAX_TRAIN=${AMASS_MAX_TRAIN:-0}
  AMASS_MAX_VAL=${AMASS_MAX_VAL:-20}
  DIP_MAX_TRAIN=${DIP_MAX_TRAIN:-0}
  DIP_MAX_VAL=${DIP_MAX_VAL:-0}
  EVAL_MAX=${EVAL_MAX:-0}
fi

cache_max_args=()
if [[ "$CACHE_MAX" != "0" ]]; then
  cache_max_args+=(--max-sequences "$CACHE_MAX")
fi
if [[ "$CACHE_MAX_FRAMES" != "0" ]]; then
  cache_max_args+=(--max-frames "$CACHE_MAX_FRAMES")
fi

eval_max_args=()
if [[ "$EVAL_MAX" != "0" ]]; then
  eval_max_args+=(--max-sequences "$EVAL_MAX")
fi

ensure_joint_cache() {
  local mode="$1"
  local split="$2"
  local input_manifest="$3"
  local gt_manifest="$4"
  local out_dir="$CACHE_ROOT/${mode}/${split}"
  if [[ ! -f "$out_dir/pl_curve_cache_manifest.json" ]]; then
    "$PY" pl_joint_leaf_acc_cache.py build \
      --input-cache "$input_manifest" \
      --gt-control-cache "$gt_manifest" \
      --output-dir "$out_dir" \
      --feature-mode "$mode" \
      --shard-size 100 \
      --device "${CACHE_DEVICE:-cuda:0}" \
      "${cache_max_args[@]}"
  fi
}

validate_split() {
  local split="$1"
  local out_json="$EVAL_ROOT/cache_validation_${split}.json"
  "$PY" pl_joint_leaf_acc_cache.py validate \
    --manifests \
    "$CACHE_ROOT/baseline_jointtarget_84D/$split/pl_curve_cache_manifest.json" \
    "$CACHE_ROOT/acc_root_102D/$split/pl_curve_cache_manifest.json" \
    "$CACHE_ROOT/acc_mixed_102D/$split/pl_curve_cache_manifest.json" \
    --output-json "$out_json"
}

for mode in baseline_jointtarget_84D acc_root_102D acc_mixed_102D; do
  ensure_joint_cache "$mode" amass "$AMASS_RAW" "$AMASS_GT"
  ensure_joint_cache "$mode" dip_train "$DIP_TRAIN_RAW" "$DIP_TRAIN_GT"
  ensure_joint_cache "$mode" dip_val "$DIP_VAL_RAW" "$DIP_VAL_GT"
  ensure_joint_cache "$mode" dip_test "$DIP_TEST_RAW" "$DIP_TEST_GT"
  ensure_joint_cache "$mode" tc_test "$TC_TEST_RAW" "$TC_TEST_GT"
done

for split in amass dip_train dip_val dip_test tc_test; do
  validate_split "$split"
done

COMMON_TRAIN_ARGS=(
  --init-size 36
  --window 61
  --hidden-size 512
  --tail-length 4
  --residual-scale 0.005
  --dropout 0.4
  --grad-clip 1.0
  --disable-ik-distill
  --baseline-pRB-weight 0.0
  --baseline-gR1-weight 0.0
  --gt-control-pRB-weight 0.3
  --gt-control-gR1-weight 0.1
  --pRB-ddot-smooth-weight 0.000001
  --gR1-dot-weight 0.03
  --gR1-ddot-weight 0.001
  --selection-metric control_physical
)

train_mode() {
  local mode="$1"
  local mode_root="$ROOT/$mode"
  local input_size=84
  if [[ "$mode" != "baseline_jointtarget_84D" ]]; then
    input_size=102
  fi
  if [[ ! -f "$mode_root/amass_pretrain/best_loss.pt" ]]; then
    "$PY" pl_curve_train.py \
      --train-cache "$CACHE_ROOT/$mode/amass/pl_curve_cache_manifest.json" \
      --val-cache "$CACHE_ROOT/$mode/amass/pl_curve_cache_manifest.json" \
      --output-dir "$mode_root/amass_pretrain" \
      --experiment-name "${mode}_amass_pretrain" \
      --epochs "$AMASS_EPOCHS" \
      --lr 1e-4 \
      --batch-size "$AMASS_BATCH" \
      --max-train-sequences "$AMASS_MAX_TRAIN" \
      --max-val-sequences "$AMASS_MAX_VAL" \
      --val-window-length 61 \
      --early-stop-min-delta 0.00000005 \
      --early-stop-patience 12 \
      --input-size "$input_size" \
      "${COMMON_TRAIN_ARGS[@]}"
  fi

  "$PY" scripts/eval_newpl_joint_leaf_acc.py \
    --cache "$CACHE_ROOT/$mode/dip_test/pl_curve_cache_manifest.json" \
    --checkpoint "$mode_root/amass_pretrain/best_loss.pt" \
    --output-json "$EVAL_ROOT/${mode}_dip_test_after_amass.json" \
    --output-summary "$EVAL_ROOT/${mode}_dip_test_after_amass.md" \
    "${eval_max_args[@]}"

  "$PY" scripts/eval_newpl_joint_leaf_acc.py \
    --cache "$CACHE_ROOT/$mode/tc_test/pl_curve_cache_manifest.json" \
    --checkpoint "$mode_root/amass_pretrain/best_loss.pt" \
    --output-json "$EVAL_ROOT/${mode}_tc_test_after_amass.json" \
    --output-summary "$EVAL_ROOT/${mode}_tc_test_after_amass.md" \
    "${eval_max_args[@]}"

  if [[ ! -f "$mode_root/dip_finetune/best_loss.pt" ]]; then
    "$PY" pl_curve_train.py \
      --train-cache "$CACHE_ROOT/$mode/dip_train/pl_curve_cache_manifest.json" \
      --val-cache "$CACHE_ROOT/$mode/dip_val/pl_curve_cache_manifest.json" \
      --output-dir "$mode_root/dip_finetune" \
      --experiment-name "${mode}_amass_to_dip_finetune" \
      --epochs "$DIP_EPOCHS" \
      --lr 5e-6 \
      --batch-size "$DIP_BATCH" \
      --max-train-sequences "$DIP_MAX_TRAIN" \
      --max-val-sequences "$DIP_MAX_VAL" \
      --val-window-length 61 \
      --init-checkpoint "$mode_root/amass_pretrain/best_loss.pt" \
      --early-stop-min-delta 0.00000005 \
      --early-stop-patience 10 \
      --input-size "$input_size" \
      "${COMMON_TRAIN_ARGS[@]}"
  fi

  "$PY" scripts/eval_newpl_joint_leaf_acc.py \
    --cache "$CACHE_ROOT/$mode/dip_test/pl_curve_cache_manifest.json" \
    --checkpoint "$mode_root/dip_finetune/best_loss.pt" \
    --output-json "$EVAL_ROOT/${mode}_dip_test_after_dip.json" \
    --output-summary "$EVAL_ROOT/${mode}_dip_test_after_dip.md" \
    "${eval_max_args[@]}"

  "$PY" scripts/eval_newpl_joint_leaf_acc.py \
    --cache "$CACHE_ROOT/$mode/tc_test/pl_curve_cache_manifest.json" \
    --checkpoint "$mode_root/dip_finetune/best_loss.pt" \
    --output-json "$EVAL_ROOT/${mode}_tc_test_after_dip.json" \
    --output-summary "$EVAL_ROOT/${mode}_tc_test_after_dip.md" \
    "${eval_max_args[@]}"
}

for mode in baseline_jointtarget_84D acc_root_102D acc_mixed_102D; do
  train_mode "$mode"
done

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
eval_root = root / "eval"
modes = ["baseline_jointtarget_84D", "acc_root_102D", "acc_mixed_102D"]
stages = ["dip_test_after_amass", "tc_test_after_amass", "dip_test_after_dip", "tc_test_after_dip"]
summary = {
    "status": "ok",
    "root": str(root),
    "contract": "joint-leaf NewPL module-only; no IK/full-pipeline",
    "modes": modes,
    "stages": {},
}
for stage in stages:
    rows = {}
    for mode in modes:
        path = eval_root / f"{mode}_{stage}.json"
        payload = json.loads(path.read_text())
        rows[mode] = payload["summary"]
    baseline = rows["baseline_jointtarget_84D"]
    deltas = {}
    for mode in ["acc_root_102D", "acc_mixed_102D"]:
        deltas[f"{mode}-baseline_jointtarget_84D"] = {
            "p_leaf_joint_R_l2_cm": rows[mode]["p_leaf_joint_R_l2_cm"] - baseline["p_leaf_joint_R_l2_cm"],
            "gR1_angle_deg": rows[mode]["gR1_angle_deg"] - baseline["gR1_angle_deg"],
        }
    deltas["acc_root_102D-acc_mixed_102D"] = {
        "p_leaf_joint_R_l2_cm": rows["acc_root_102D"]["p_leaf_joint_R_l2_cm"] - rows["acc_mixed_102D"]["p_leaf_joint_R_l2_cm"],
        "gR1_angle_deg": rows["acc_root_102D"]["gR1_angle_deg"] - rows["acc_mixed_102D"]["gR1_angle_deg"],
    }
    summary["stages"][stage] = {"rows": rows, "deltas": deltas}

(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "# Joint-Leaf Acceleration NewPL Summary",
    "",
    "Scope: module-level joint-leaf PL only; no IK/full-pipeline.",
    "",
]
for stage, payload in summary["stages"].items():
    lines += [f"## {stage}", "", "| mode | p_leaf_joint_R L2 cm | gR1 deg | base p_leaf_joint_R L2 cm |", "|---|---:|---:|---:|"]
    for mode in modes:
        row = payload["rows"][mode]
        lines.append(f"| {mode} | {row['p_leaf_joint_R_l2_cm']:.6f} | {row['gR1_angle_deg']:.6f} | {row['base_p_leaf_joint_R_l2_cm']:.6f} |")
    lines += ["", "| delta | p_leaf_joint_R L2 cm | gR1 deg |", "|---|---:|---:|"]
    for name, row in payload["deltas"].items():
        lines.append(f"| {name} | {row['p_leaf_joint_R_l2_cm']:.6f} | {row['gR1_angle_deg']:.6f} |")
    lines.append("")
(root / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"status": "ok", "summary": str(root / "summary.json")}, indent=2))
PY

date -Is
echo "done: $ROOT"
