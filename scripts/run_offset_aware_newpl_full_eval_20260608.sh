#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=/home/lingfeng/.conda/envs/globalpose-gpu
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

ROOT=data/experiments/offset_aware_newpl_20260607_longrun_v1
OUT_DIR="$ROOT/full_eval"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"
RUN_LOG="$LOG_DIR/full_eval.log"
exec > >(tee -a "$RUN_LOG") 2>&1

PY="$ENV_DIR/bin/python"
CKPT="$ROOT/dip_finetune/best_loss.pt"

DIP_TEST_CACHE=data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
TC_S4_CACHE=data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json
TC_S5_CACHE=data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json

echo "started $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "checkpoint=$CKPT"

"$PY" pl_curve_eval.py \
  --val-cache "$DIP_TEST_CACHE" \
  --checkpoint "$CKPT" \
  --output-json "$OUT_DIR/dip_test_full_pipeline.json" \
  --imu-input-mode official

"$PY" pl_curve_eval.py \
  --val-cache "$TC_S4_CACHE" \
  --checkpoint "$CKPT" \
  --output-json "$OUT_DIR/tc_s4_full_pipeline.json" \
  --imu-input-mode official

"$PY" pl_curve_eval.py \
  --val-cache "$TC_S5_CACHE" \
  --checkpoint "$CKPT" \
  --output-json "$OUT_DIR/tc_s5_full_pipeline.json" \
  --imu-input-mode official

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("data/experiments/offset_aware_newpl_20260607_longrun_v1/full_eval")
metric_order = [
    "L SIP Err (deg)",
    "L Angle Err (deg)",
    "L Joint Err (cm)",
    "L Vertex Err (cm)",
    "G SIP Err (deg)",
    "G Angle Err (deg)",
    "G Joint Err (cm)",
    "G Vertex Err (cm)",
    "Root Jitter (km/s^3)",
    "Joint Jitter (km/s^3)",
]
files = {
    "dip_test": root / "dip_test_full_pipeline.json",
    "tc_s4": root / "tc_s4_full_pipeline.json",
    "tc_s5": root / "tc_s5_full_pipeline.json",
}
rows = []
for split, path in files.items():
    data = json.loads(path.read_text())
    agg = data["aggregate"]
    base = agg["baseline_metrics"]
    model = agg["model_metrics"]
    row = {
        "split": split,
        "status": data["status"],
        "num_sequences": agg["num_sequences"],
        "all_finite": data["all_finite"],
        "baseline_score": sum(base[name]["mean"] for name in metric_order),
        "model_score": data["score"],
        "delta_score": data["score"] - sum(base[name]["mean"] for name in metric_order),
    }
    for name in metric_order:
        short = {
            "L SIP Err (deg)": "local_sip_deg",
            "L Angle Err (deg)": "local_angle_deg",
            "L Joint Err (cm)": "local_joint_cm",
            "L Vertex Err (cm)": "local_vertex_cm",
            "G SIP Err (deg)": "global_sip_deg",
            "G Angle Err (deg)": "global_angle_deg",
            "G Joint Err (cm)": "global_joint_cm",
            "G Vertex Err (cm)": "global_vertex_cm",
            "Root Jitter (km/s^3)": "root_jitter",
            "Joint Jitter (km/s^3)": "joint_jitter",
        }[name]
        row[f"baseline_{short}"] = base[name]["mean"]
        row[f"model_{short}"] = model[name]["mean"]
        row[f"delta_{short}"] = model[name]["mean"] - base[name]["mean"]
    rows.append(row)

summary = {
    "checkpoint": "data/experiments/offset_aware_newpl_20260607_longrun_v1/dip_finetune/best_loss.pt",
    "metric_contract": "Full GPNet streaming evaluation through pl_curve_eval.py. Baseline is frozen official GPNet output from the same cache when present; model replaces only PL with offset-aware NewPL. Lower is better; delta=model-baseline.",
    "rows": rows,
}
(root / "full_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "finished $(date --iso-8601=seconds)"
