#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_DIR="${ENV_DIR:-/home/lingfeng/.conda/envs/globalpose-gpu}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

EXP="${EXP:-data/experiments/newpl_v6_gR1_only_swap_20260614}"
OUT="$EXP/eval"
LOG_DIR="$EXP/logs"
mkdir -p "$OUT" "$LOG_DIR"

CKPT="${CKPT:-/tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/dip_finetune/best_current_gR1.pt}"

run_eval() {
  local label="$1"
  local raw_cache="$2"
  local gr1_cache="$3"
  local output_json="$4"
  echo "[run] $label"
  "$ENV_DIR/bin/python" pl_gr1_only_swap_eval.py \
    --raw-cache "$raw_cache" \
    --gr1-cache "$gr1_cache" \
    --v6-gR1-checkpoint "$CKPT" \
    --output-json "$output_json" \
    --dataset-label "$label" \
    --imu-input-mode official \
    | tee "$LOG_DIR/${label}.log"
}

run_eval \
  "dip_raw_official" \
  "data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json" \
  "data/experiments/newpl_v5_smoothacc_20260612/caches/pl_dip_test_smoothacc_init36/pl_curve_cache_manifest.json" \
  "$OUT/dip_raw_official.json"

run_eval \
  "tc_raw_official" \
  "data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json" \
  "data/experiments/newpl_v5_smoothacc_20260612/caches/pl_tc_test_smoothacc_init36/pl_curve_cache_manifest.json" \
  "$OUT/tc_raw_official.json"

run_eval \
  "dip_smoothacc" \
  "data/experiments/newpl_v5_smoothacc_20260612/caches/raw_dip_test_smooth_w9/baseline_cache_manifest.json" \
  "data/experiments/newpl_v5_smoothacc_20260612/caches/pl_dip_test_smoothacc_init36/pl_curve_cache_manifest.json" \
  "$OUT/dip_smoothacc.json"

run_eval \
  "tc_smoothacc" \
  "data/experiments/newpl_v5_smoothacc_20260612/caches/raw_tc_test_smooth_w9/baseline_cache_manifest.json" \
  "data/experiments/newpl_v5_smoothacc_20260612/caches/pl_tc_test_smoothacc_init36/pl_curve_cache_manifest.json" \
  "$OUT/tc_smoothacc.json"

"$ENV_DIR/bin/python" - <<'PY'
import json
from pathlib import Path

exp = Path("data/experiments/newpl_v6_gR1_only_swap_20260614")
items = []
for path in sorted((exp / "eval").glob("*.json")):
    data = json.loads(path.read_text())
    agg = data.get("aggregate", {})
    trace = agg.get("trace_metrics", {})
    delta_trace = trace.get("delta_swap_minus_official", {})
    base = agg.get("baseline_metrics", {})
    model = agg.get("model_metrics", {})
    def mean(section, key):
        return section.get(key, {}).get("mean")
    items.append({
        "json": str(path),
        "dataset_label": data.get("dataset_label"),
        "status": data.get("status"),
        "num_sequences": agg.get("num_sequences"),
        "all_finite": data.get("all_finite"),
        "pRB_fixed_check_passed": data.get("pRB_fixed_check_passed"),
        "pRB_fixed_max_abs_delta": trace.get("pRB_fixed_max_abs_delta"),
        "official_score": data.get("official_score"),
        "swap_score": data.get("swap_score"),
        "score_delta_swap_minus_official": data.get("score_delta_swap_minus_official"),
        "official_gR1_angle_deg": trace.get("official", {}).get("gR1_angle_deg"),
        "swap_gR1_angle_deg": trace.get("swap", {}).get("gR1_angle_deg"),
        "delta_gR1_angle_deg": delta_trace.get("gR1_angle_deg"),
        "official_gR2_angle_deg": trace.get("official", {}).get("gR2_angle_deg"),
        "swap_gR2_angle_deg": trace.get("swap", {}).get("gR2_angle_deg"),
        "delta_gR2_angle_deg": delta_trace.get("gR2_angle_deg"),
        "official_L_Angle": mean(base, "L Angle Err (deg)"),
        "swap_L_Angle": mean(model, "L Angle Err (deg)"),
        "official_G_Angle": mean(base, "G Angle Err (deg)"),
        "swap_G_Angle": mean(model, "G Angle Err (deg)"),
        "official_L_Joint": mean(base, "L Joint Err (cm)"),
        "swap_L_Joint": mean(model, "L Joint Err (cm)"),
        "official_G_Joint": mean(base, "G Joint Err (cm)"),
        "swap_G_Joint": mean(model, "G Joint Err (cm)"),
        "official_Joint_Jitter": mean(base, "Joint Jitter (km/s^3)"),
        "swap_Joint_Jitter": mean(model, "Joint Jitter (km/s^3)"),
    })

summary = {
    "experiment": "newpl_v6_gR1_only_swap_20260614",
    "contract": "official pRB[15] fixed, replace only gR1[3] with newpl_v6_gR1nextonly_smoothacc best_current_gR1.",
    "checkpoint": "/tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/dip_finetune/best_current_gR1.pt",
    "items": items,
}
(exp / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
