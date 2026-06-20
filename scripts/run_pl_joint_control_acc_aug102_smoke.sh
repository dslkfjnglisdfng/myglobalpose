#!/usr/bin/env bash
set -euo pipefail

ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
PY=${PY:-$ENV/bin/python}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
ROOT=${ROOT:-data/experiments/pl_joint_control_acc_aug102_v1_smoke_${STAMP}}
CACHE_DIR="$ROOT/cache/dip_val"
TRAIN_DIR="$ROOT/train"
EVAL_DIR="$ROOT/eval"
mkdir -p "$CACHE_DIR" "$TRAIN_DIR" "$EVAL_DIR"

RAW_CACHE=${RAW_CACHE:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json}
GT_CONTROL=${GT_CONTROL:-/home/lingfeng/projects/data/dataset_work/dip/gt_control/val/manifest.json}

"$PY" pl_joint_control_acc_aug102_cache.py build \
  --input-cache "$RAW_CACHE" \
  --gt-control-cache "$GT_CONTROL" \
  --output-dir "$CACHE_DIR" \
  --max-sequences "${CACHE_MAX_SEQUENCES:-1}" \
  --max-frames "${CACHE_MAX_FRAMES:-120}" \
  --shard-size 10 \
  --device "${CACHE_DEVICE:-cuda:0}"

"$PY" pl_joint_control_acc_aug102_cache.py validate \
  --cache "$CACHE_DIR/pl_curve_cache_manifest.json" \
  --output-json "$ROOT/cache_validation.json"

"$PY" pl_joint_control_acc_aug102_train.py \
  --train-cache "$CACHE_DIR/pl_curve_cache_manifest.json" \
  --val-cache "$CACHE_DIR/pl_curve_cache_manifest.json" \
  --output-dir "$TRAIN_DIR" \
  --experiment-name pl_joint_control_acc_aug102_v1_smoke \
  --epochs "${EPOCHS:-1}" \
  --window "${WINDOW:-61}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --max-train-sequences 1 \
  --max-val-sequences 1 \
  --val-window-length "${WINDOW:-61}" \
  --hidden-size "${HIDDEN_SIZE:-128}" \
  --dropout 0.0 \
  --lr 1e-4

"$PY" pl_joint_control_acc_aug102_eval.py \
  --cache "$CACHE_DIR/pl_curve_cache_manifest.json" \
  --checkpoint "$TRAIN_DIR/best_loss.pt" \
  --output-json "$EVAL_DIR/metrics.json" \
  --output-summary "$EVAL_DIR/metrics.md" \
  --split smoke \
  --max-sequences 1

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cache_manifest = json.loads((root / "cache/dip_val/pl_curve_cache_manifest.json").read_text())
train_result = json.loads((root / "train/train_result.json").read_text())
eval_payload = json.loads((root / "eval/metrics.json").read_text())
s = eval_payload["summary"]
lines = [
    "# PL Joint Control Acc-Aug102 v1",
    "",
    "## 1. Question",
    "Smoke-test a new joint-target NewPL control module using frozen joint acceleration as an auxiliary input.",
    "",
    "## 2. Why joint target instead of vertex target",
    "The target is SMPL joint-based root-frame leaf positions, not the legacy IMU-vertex pRB target.",
    "",
    "## 3. Frozen acceleration source",
    f"`{cache_manifest['frozen_acc_source']['checkpoint_path']}`",
    "",
    "## 4. Input layout",
    f"`{cache_manifest['feature_layout']}`",
    "",
    "## 5. Target definition",
    cache_manifest["target_contract"],
    "",
    "## 6. Control-point decoder design",
    "The module predicts 18D control points and decodes joint_pRB/gR plus joint_pRB_dot/joint_pRB_ddot with UniformCubicBSpline.",
    "",
    "## 7. Loss weights",
    "`pRB=1.0, gR1=0.3, gt_control_pRB=0.5, gt_control_gR1=0.1, pRB_dot=0.5, pRB_ddot=0.1, pRB_ddot_smooth=0.001, gR_smooth=0.001, control_point_prior=0.001, tail_update_prior=0.001`",
    "",
    "## 8. Smoke cache results",
    f"`feature_dim={cache_manifest['sanity'].get('feature_dim')}`, `target_dim={cache_manifest['sanity'].get('target_dim')}`, `joint_minus_vertex_l2_mean_m={cache_manifest['sanity'].get('joint_minus_vertex_l2_mean_m'):.6f}`",
    "",
    "## 9. Smoke training results",
    f"`status={train_result['status']}`, `best_epoch={train_result['best_epoch']}`, `best_loss={train_result['best_loss']}`",
    "",
    "## 10. Eval metrics",
    "",
    "| split | joint_pos_l2_m | joint_vel_l2_mps | joint_acc_l2_mps2 | gravity_angle_deg |",
    "|---|---:|---:|---:|---:|",
    f"| smoke | {s['joint_pos_l2_m']:.6f} | {s['joint_vel_l2_mps']:.6f} | {s['joint_acc_l2_mps2']:.6f} | {s['gravity_angle_deg']:.6f} |",
    "",
    "## 11. Interpretation",
    "This is a smoke result only. It verifies cache construction, forward/backward training, spline derivatives, and metric reporting.",
    "",
    "## 12. Limitations",
    "No full training or full-pipeline evaluation was run.",
    "",
    "## 13. Artifacts",
    f"- Cache manifest: `{root / 'cache/dip_val/pl_curve_cache_manifest.json'}`",
    f"- Train result: `{root / 'train/train_result.json'}`",
    f"- Eval metrics: `{root / 'eval/metrics.json'}`",
]
(root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"status": "ok", "summary": str(root / "SUMMARY.md")}, indent=2))
PY

echo "$ROOT"
