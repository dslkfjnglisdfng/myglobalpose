#!/usr/bin/env bash
set -euo pipefail

ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
PY="$ENV/bin/python"

ROOT=${ROOT:-data/experiments/newpl_v5_delay_eval_20260607}
CACHE_ROOT=${CACHE_ROOT:-data/experiments/newpl_v5_official_protocol_20260607/caches}
V5_ROOT=${V5_ROOT:-data/experiments/newpl_v5_official_protocol_20260607_tuned}
mkdir -p "$ROOT/smoke" "$ROOT/eval"

AMASS_BEST="$V5_ROOT/amass_pretrain/best_loss.pt"
DIP_BEST="$V5_ROOT/dip_finetune/best_loss.pt"
DIP_CACHE="$CACHE_ROOT/pl_dip_test_official_init36/pl_curve_cache_manifest.json"
TC_CACHE="$CACHE_ROOT/pl_tc_test_official_init36/pl_curve_cache_manifest.json"

run_eval() {
  local cache="$1"
  local dataset="$2"
  local label="$3"
  local out="$4"
  local max_seq="${5:-0}"
  local max_arg=()
  if [[ "$max_seq" != "0" ]]; then
    max_arg=(--max-eval-sequences "$max_seq")
  fi
  "$PY" pl_curve_delay_eval.py \
    --cache "$cache" \
    --output-json "$out" \
    --dataset-label "$label" \
    --delay-mode future_output \
    "${max_arg[@]}" \
    --version official_PL=official \
    --version newpl_v5_amass_delay0="$AMASS_BEST",delay=0 \
    --version newpl_v5_amass_delay1="$AMASS_BEST",delay=1 \
    --version newpl_v5_amass_delay2="$AMASS_BEST",delay=2 \
    --version newpl_v5_dip_delay0="$DIP_BEST",delay=0 \
    --version newpl_v5_dip_delay1="$DIP_BEST",delay=1 \
    --version newpl_v5_dip_delay2="$DIP_BEST",delay=2
}

if [[ "${RUN_SMOKE:-0}" == "1" ]]; then
  run_eval "$DIP_CACHE" dip DIP-IMU-test "$ROOT/smoke/dip_test_delay_smoke.json" 2
  run_eval "$TC_CACHE" totalcapture TotalCapture-test-official-input "$ROOT/smoke/tc_test_delay_smoke.json" 2
fi

run_eval "$DIP_CACHE" dip DIP-IMU-test "$ROOT/eval/dip_test_delay_module_metrics.json" 0
run_eval "$TC_CACHE" totalcapture TotalCapture-test-official-input "$ROOT/eval/tc_test_delay_module_metrics.json" 0

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
jsons = {
    "DIP-IMU-test": root / "eval" / "dip_test_delay_module_metrics.json",
    "TotalCapture-test-official-input": root / "eval" / "tc_test_delay_module_metrics.json",
}
summary = {
    "status": "ok",
    "root": str(root),
    "delay_mode": "future_output: evaluate pred[t+delay] against GT[t]",
    "jsons": {k: str(v) for k, v in jsons.items()},
    "pl_output_tables": {},
    "per_leaf_tables": {},
    "temporal_tables": {},
    "conclusions": {},
}

def version_rows(data):
    return {v["name"]: v["aggregate"] for v in data["versions"]}

def best_delay(rows, prefix, metric):
    candidates = [(name, agg) for name, agg in rows.items() if name.startswith(prefix)]
    return min(candidates, key=lambda item: float(item[1][metric]))

def table_line(row):
    return (
        f"| {row['Dataset']} | {row['Version']} | {row['pRB L1 cm ↓']} | "
        f"{row['pRB L2 cm ↓']} | {row['gR1 angle deg ↓']} | {row['Notes']} |"
    )

for dataset, path in jsons.items():
    data = json.loads(path.read_text())
    summary["pl_output_tables"][dataset] = data["pl_output_comparison_table"]
    summary["per_leaf_tables"][dataset] = data["per_leaf_table"]
    rows = version_rows(data)
    temporal = []
    for version in data["versions"]:
        agg = version["aggregate"]
        temporal.append({
            "Dataset": dataset,
            "Version": version["name"],
            "delay": int(agg.get("pl_output_delay_frames", 0)),
            "frames": int(agg.get("evaluated_frames", 0)),
            "pRB temporal velocity error cm/frame": f"{agg.get('pRB_temporal_velocity_error_cm_per_frame', 0.0):.6f}",
            "pRB smooth jitter cm": f"{agg.get('pRB_smooth_jitter_cm', 0.0):.6f}",
            "gR1 temporal angle velocity error deg/frame": f"{agg.get('gR1_temporal_angle_velocity_error_deg_per_frame', 0.0):.6f}",
            "gR1 smooth jitter": f"{agg.get('gR1_smooth_jitter', 0.0):.6f}",
        })
    summary["temporal_tables"][dataset] = temporal
    official = rows["official_PL"]
    amass_best_name, amass_best = best_delay(rows, "newpl_v5_amass_delay", "pRB_L2_cm")
    dip_best_name, dip_best = best_delay(rows, "newpl_v5_dip_delay", "pRB_L2_cm")
    summary["conclusions"][dataset] = {
        "best_amass_by_pRB_L2": amass_best_name,
        "best_dip_by_pRB_L2": dip_best_name,
        "best_amass_beats_official_pRB_L2": bool(amass_best["pRB_L2_cm"] < official["pRB_L2_cm"]),
        "best_dip_beats_official_pRB_L2": bool(dip_best["pRB_L2_cm"] < official["pRB_L2_cm"]),
        "best_amass_gR1_delta_vs_official_deg": amass_best["gR1_angle_deg"] - official["gR1_angle_deg"],
        "best_dip_gR1_delta_vs_official_deg": dip_best["gR1_angle_deg"] - official["gR1_angle_deg"],
    }

(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# NewPL v5 Delay Evaluation",
    "",
    "Delay mode: `future_output`, i.e. evaluate `pred[t+delay]` against `GT[t]`.",
    "",
]
for dataset in jsons:
    lines += [
        f"## {dataset}",
        "",
        "| Dataset | Version | pRB L1 cm | pRB L2 cm | gR1 angle deg | Notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines += [table_line(row) for row in summary["pl_output_tables"][dataset]]
    lines += [
        "",
        "| Dataset | Version | delay | frames | pRB vel err | pRB jitter | gR1 vel err | gR1 jitter |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["temporal_tables"][dataset]:
        lines.append(
            f"| {row['Dataset']} | {row['Version']} | {row['delay']} | {row['frames']} | "
            f"{row['pRB temporal velocity error cm/frame']} | {row['pRB smooth jitter cm']} | "
            f"{row['gR1 temporal angle velocity error deg/frame']} | {row['gR1 smooth jitter']} |"
        )
    lines.append("")
    lines.append("Conclusion:")
    for key, value in summary["conclusions"][dataset].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

(root / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"status": "ok", "summary_json": str(root / "summary.json"), "summary_md": str(root / "summary.md")}, indent=2))
PY
