#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=/home/lingfeng/.conda/envs/globalpose-gpu
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

ROOT=data/experiments/rjs_sensitive_newpl_quick_20260608_v1
CACHE_ROOT=data/experiments/offset_aware_newpl_20260607_longrun_v1/caches
LOG_DIR="$ROOT/logs"
EVAL_DIR="$ROOT/eval"
mkdir -p "$LOG_DIR" "$EVAL_DIR"
RUN_LOG="$LOG_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

PY="$ENV_DIR/bin/python"

AMASS_CACHE="$CACHE_ROOT/pl_amass_offset_aware/pl_curve_cache_manifest.json"
DIP_TRAIN_CACHE="$CACHE_ROOT/pl_dip_train_offset_aware/pl_curve_cache_manifest.json"
DIP_VAL_CACHE="$CACHE_ROOT/pl_dip_val_offset_aware/pl_curve_cache_manifest.json"
DIP_TEST_CACHE="$CACHE_ROOT/pl_dip_test_offset_aware/pl_curve_cache_manifest.json"
TC_TEST_CACHE="$CACHE_ROOT/pl_tc_test_offset_aware/pl_curve_cache_manifest.json"

AMASS_OUT="$ROOT/stage_a_amass_rjs_sensitive"
DIP_OUT="$ROOT/stage_b_dip_finetune"

echo "started $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
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
    --experiment-name rjs_sensitive_newpl_amass_stage_a \
    --epochs 30 \
    --window 61 \
    --lr 1e-4 \
    --hidden-size 512 \
    --batch-size 128 \
    --dropout 0.15 \
    --disable-ik-distill \
    --model-variant offset_aware \
    --init-size 36 \
    --selection-metric pl_and_control_physical \
    --pRB-weight 1.0 \
    --gR1-weight 1.0 \
    --baseline-pRB-weight 0.05 \
    --baseline-gR1-weight 0.0 \
    --gt-control-pRB-weight 0.3 \
    --gt-control-gR1-weight 0.1 \
    --control-point-prior-weight 0.05 \
    --tail-update-prior-weight 0.001 \
    --pRB-dot-weight 0.03 \
    --pRB-ddot-smooth-weight 1e-6 \
    --gR1-dot-weight 0.03 \
    --gR1-ddot-weight 0.001 \
    --offset-consistency-weight 0.2 \
    --offset-consistency-target full_pl \
    --offset-contrast-weight 0.2 \
    --offset-contrast-margin 0.0005 \
    --offset-contrast-mode roll_random \
    --offset-contrast-target full_pl \
    --offset-init-dropout-prob 0.05 \
    --offset-init-noise-std 0.01 \
    --early-stop-patience 8 \
    --early-stop-min-delta 5e-8 \
    --max-val-sequences 20
fi

if [ ! -f "$DIP_OUT/train_result.json" ]; then
  "$PY" pl_curve_train.py \
    --train-cache "$DIP_TRAIN_CACHE" \
    --val-cache "$DIP_VAL_CACHE" \
    --output-dir "$DIP_OUT" \
    --experiment-name rjs_sensitive_newpl_dip_stage_b \
    --epochs 30 \
    --window 61 \
    --lr 5e-6 \
    --hidden-size 512 \
    --batch-size 12 \
    --dropout 0.08 \
    --disable-ik-distill \
    --model-variant offset_aware \
    --init-size 36 \
    --init-checkpoint "$AMASS_OUT/best_loss.pt" \
    --selection-metric pl_and_control_physical \
    --pRB-weight 1.0 \
    --gR1-weight 1.0 \
    --baseline-pRB-weight 0.08 \
    --baseline-gR1-weight 0.0 \
    --gt-control-pRB-weight 0.3 \
    --gt-control-gR1-weight 0.1 \
    --control-point-prior-weight 0.05 \
    --tail-update-prior-weight 0.001 \
    --pRB-dot-weight 0.03 \
    --pRB-ddot-smooth-weight 1e-6 \
    --gR1-dot-weight 0.03 \
    --gR1-ddot-weight 0.001 \
    --early-stop-patience 8 \
    --early-stop-min-delta 5e-8
fi

for split in dip_val dip_test tc_test; do
  case "$split" in
    dip_val) cache="$DIP_VAL_CACHE" ;;
    dip_test) cache="$DIP_TEST_CACHE" ;;
    tc_test) cache="$TC_TEST_CACHE" ;;
  esac
  "$PY" pl_curve_pl_accuracy_eval.py \
    --pl-cache "$cache" \
    --checkpoint "$DIP_OUT/best_loss.pt" \
    --output-json "$EVAL_DIR/${split}_module_pl_accuracy_best.json"
  "$PY" pl_curve_offset_swap_eval.py \
    --pl-cache "$cache" \
    --checkpoint "$DIP_OUT/best_loss.pt" \
    --output-json "$EVAL_DIR/${split}_offset_swap_best.json" \
    --swap-feature-offset
done

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("data/experiments/rjs_sensitive_newpl_quick_20260608_v1")
eval_dir = root / "eval"
rows = []
for split in ("dip_val", "dip_test", "tc_test"):
    mod = json.loads((eval_dir / f"{split}_module_pl_accuracy_best.json").read_text())
    swap = json.loads((eval_dir / f"{split}_offset_swap_best.json").read_text())
    leaf = mod["aggregate"]["leaf_position_error_cm"]
    grav = mod["aggregate"]["gravity_angle_deg"]
    delta = swap["aggregate"]["delta_vs_good"]
    rows.append({
        "split": split,
        "num_sequences": mod["aggregate"]["num_sequences"],
        "all_finite": mod["aggregate"]["all_finite"],
        "original_leaf_cm_mean": leaf["original"]["mean"],
        "new_leaf_cm_mean": leaf["new"]["mean"],
        "delta_leaf_cm_mean": leaf["delta_new_minus_original"]["mean"],
        "original_gR1_deg_mean": grav["original"]["mean"],
        "new_gR1_deg_mean": grav["new"]["mean"],
        "delta_gR1_deg_mean": grav["delta_new_minus_original"]["mean"],
        "roll_minus_good_pRB_cm_mean": delta["roll_sensors"]["pRB_cm"]["mean"],
        "roll_minus_good_gR1_deg_mean": delta["roll_sensors"]["gR1_deg"]["mean"],
        "negate_minus_good_pRB_cm_mean": delta["negate"]["pRB_cm"]["mean"],
        "negate_minus_good_gR1_deg_mean": delta["negate"]["gR1_deg"]["mean"],
        "zero_minus_good_pRB_cm_mean": delta["zero"]["pRB_cm"]["mean"],
        "zero_minus_good_gR1_deg_mean": delta["zero"]["gR1_deg"]["mean"],
    })
summary = {
    "status": "ok",
    "root": str(root),
    "checkpoint": str(root / "stage_b_dip_finetune" / "best_loss.pt"),
    "coordinate_contract": "r_JS is joint-local IMU origin relative to mapped joint J; DIP uses pseudo-rJS only.",
    "acceptance": {
        "sensitivity": "wrong offset should worsen pRB by >=0.02 cm or gR1 by >=0.05 deg versus good",
        "dip_module": "DIP pRB should not be worse than official_PL target 6.419 cm before full-pipeline promotion",
    },
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "finished $(date --iso-8601=seconds)"
