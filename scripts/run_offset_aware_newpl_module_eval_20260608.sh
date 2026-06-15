#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=/home/lingfeng/.conda/envs/globalpose-gpu
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

ROOT=data/experiments/offset_aware_newpl_20260607_longrun_v1
CACHE_ROOT="$ROOT/caches"
EVAL_DIR="$ROOT/module_eval"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$EVAL_DIR" "$LOG_DIR"
RUN_LOG="$LOG_DIR/module_eval.log"
exec > >(tee -a "$RUN_LOG") 2>&1

PY="$ENV_DIR/bin/python"
CKPT="$ROOT/dip_finetune/best_loss.pt"

DIP_TEST_SRC=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
TC_TEST_SRC=data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json

DIP_TEST_CACHE="$CACHE_ROOT/pl_dip_test_offset_aware"
TC_TEST_CACHE="$CACHE_ROOT/pl_tc_test_offset_aware"

echo "started $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ ! -f "$DIP_TEST_CACHE/pl_curve_cache_manifest.json" ]; then
  "$PY" pl_curve_cache.py \
    --input-cache "$DIP_TEST_SRC" \
    --output-dir "$DIP_TEST_CACHE" \
    --shard-size 100 \
    --imu-input-mode official \
    --feature-mode offset_aware
fi

if [ ! -f "$TC_TEST_CACHE/pl_curve_cache_manifest.json" ]; then
  "$PY" pl_curve_cache.py \
    --input-cache "$TC_TEST_SRC" \
    --output-dir "$TC_TEST_CACHE" \
    --shard-size 100 \
    --imu-input-mode official \
    --feature-mode offset_aware
fi

"$PY" pl_curve_pl_accuracy_eval.py \
  --pl-cache "$CACHE_ROOT/pl_dip_val_offset_aware/pl_curve_cache_manifest.json" \
  --checkpoint "$CKPT" \
  --output-json "$EVAL_DIR/dip_val_module_pl_accuracy.json"

"$PY" pl_curve_pl_accuracy_eval.py \
  --pl-cache "$DIP_TEST_CACHE/pl_curve_cache_manifest.json" \
  --checkpoint "$CKPT" \
  --output-json "$EVAL_DIR/dip_test_module_pl_accuracy.json"

"$PY" pl_curve_pl_accuracy_eval.py \
  --pl-cache "$TC_TEST_CACHE/pl_curve_cache_manifest.json" \
  --checkpoint "$CKPT" \
  --output-json "$EVAL_DIR/tc_test_module_pl_accuracy.json"

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("data/experiments/offset_aware_newpl_20260607_longrun_v1/module_eval")
rows = []
for name in ("dip_val", "dip_test", "tc_test"):
    path = root / f"{name}_module_pl_accuracy.json"
    data = json.loads(path.read_text())
    agg = data["aggregate"]
    leaf = agg["leaf_position_error_cm"]
    grav = agg["gravity_angle_deg"]
    rows.append({
        "split": name,
        "num_sequences": agg["num_sequences"],
        "num_frames": agg["num_frames"],
        "all_finite": agg["all_finite"],
        "original_leaf_cm_mean": leaf["original"]["mean"],
        "new_leaf_cm_mean": leaf["new"]["mean"],
        "delta_leaf_cm_mean": leaf["delta_new_minus_original"]["mean"],
        "original_gR1_deg_mean": grav["original"]["mean"],
        "new_gR1_deg_mean": grav["new"]["mean"],
        "delta_gR1_deg_mean": grav["delta_new_minus_original"]["mean"],
    })
summary = {
    "checkpoint": "data/experiments/offset_aware_newpl_20260607_longrun_v1/dip_finetune/best_loss.pt",
    "metric_contract": {
        "leaf_position_error_cm": "decoded PL pRB[15] leaf vectors vs GT pRB[15], L2 norm in cm; lower is better",
        "gravity_angle_deg": "decoded PL gR1 direction vs GT gR1, angle in degrees; lower is better",
        "delta": "new offset-aware PL minus original frozen PL base; negative is improvement",
    },
    "rows": rows,
}
(root / "module_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "finished $(date --iso-8601=seconds)"
