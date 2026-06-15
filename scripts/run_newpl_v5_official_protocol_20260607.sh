#!/usr/bin/env bash
set -euo pipefail

ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
PY="$ENV/bin/python"

ROOT=${ROOT:-data/experiments/newpl_v5_official_protocol_20260607_tuned}
CACHE_ROOT=${CACHE_ROOT:-data/experiments/newpl_v5_official_protocol_20260607/caches}
EVAL_ROOT="$ROOT/eval"
mkdir -p "$CACHE_ROOT" "$EVAL_ROOT"

AMASS_RAW=data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
DIP_TRAIN_RAW=data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json
DIP_VAL_RAW=data/dataset_work/L4Cache/prephysics_pose_velocity_dip_val_globalpose_neural_only/baseline_cache_manifest.json
DIP_TEST_RAW=data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json
TC_TEST_RAW=data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_test_official_neural_only/baseline_cache_manifest.json

DIP_TRAIN_OFFSET_SOURCE=data/dataset_work/SensorOffset/full_diagnostic_v1/dip_train.pt
DIP_VAL_OFFSET_SOURCE=data/dataset_work/SensorOffset/full_diagnostic_v1/dip_val.pt
DIP_TEST_OFFSET_SOURCE=data/experiments/dip_official_protocol_check_20260607/dip_test_offsets.pt
TC_TEST_OFFSET_SOURCE=data/dataset_work/TotalCapture_globalpose_official/test_with_offsets.pt

NEWPL_V4=data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt

write_offset_alias() {
  local src="$1"
  local dst="$2"
  "$PY" - "$src" "$dst" <<'PY'
import sys
from pathlib import Path
import torch

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = torch.load(src, map_location="cpu")
if "imu_offset_r" in data:
    offset = data["imu_offset_r"]
elif "r_JS" in data:
    offset = data["r_JS"]
elif "offset" in data:
    offset = data["offset"]
else:
    raise KeyError(f"{src} has no imu_offset_r/r_JS/offset field")
if not torch.is_tensor(offset):
    offset = torch.stack(offset)
out = {
    "name": list(data["name"]),
    "r_JS": offset.float().clone(),
    "imu_offset_r": offset.float().clone(),
    "source": str(src),
    "note": "Alias for NewPL init36 offset_r only; not real DIP/TC offset supervision.",
}
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(out, dst)
print({"status": "ok", "alias": str(dst), "num_sequences": len(out["name"])})
PY
}

ensure_enriched_cache() {
  local raw_manifest="$1"
  local offset_source="$2"
  local out_dir="$3"
  local alias_path="$4"
  if [[ ! -f "$out_dir/baseline_cache_manifest.json" ]]; then
    write_offset_alias "$offset_source" "$alias_path"
    "$PY" l4_enrich_cache_with_offsets.py \
      --cache-manifest "$raw_manifest" \
      --processed-dataset "$alias_path" \
      --output-dir "$out_dir"
  fi
}

ensure_pl_cache() {
  local input_manifest="$1"
  local output_dir="$2"
  local imu_mode="$3"
  if [[ ! -f "$output_dir/pl_curve_cache_manifest.json" ]]; then
    "$PY" pl_curve_cache.py \
      --input-cache "$input_manifest" \
      --output-dir "$output_dir" \
      --imu-input-mode "$imu_mode" \
      --shard-size 100
  fi
}

ensure_enriched_cache "$DIP_TRAIN_RAW" "$DIP_TRAIN_OFFSET_SOURCE" "$CACHE_ROOT/dip_train_with_offset_r" "$CACHE_ROOT/dip_train_offset_alias.pt"
ensure_enriched_cache "$DIP_VAL_RAW" "$DIP_VAL_OFFSET_SOURCE" "$CACHE_ROOT/dip_val_with_offset_r" "$CACHE_ROOT/dip_val_offset_alias.pt"
ensure_enriched_cache "$DIP_TEST_RAW" "$DIP_TEST_OFFSET_SOURCE" "$CACHE_ROOT/dip_test_with_offset_r" "$CACHE_ROOT/dip_test_offset_alias.pt"
ensure_enriched_cache "$TC_TEST_RAW" "$TC_TEST_OFFSET_SOURCE" "$CACHE_ROOT/tc_test_official_with_offset_r" "$CACHE_ROOT/tc_test_offset_alias.pt"

ensure_pl_cache "$AMASS_RAW" "$CACHE_ROOT/pl_amass_official_init36" official
ensure_pl_cache "$CACHE_ROOT/dip_train_with_offset_r/baseline_cache_manifest.json" "$CACHE_ROOT/pl_dip_train_official_init36" official
ensure_pl_cache "$CACHE_ROOT/dip_val_with_offset_r/baseline_cache_manifest.json" "$CACHE_ROOT/pl_dip_val_official_init36" official
ensure_pl_cache "$CACHE_ROOT/dip_test_with_offset_r/baseline_cache_manifest.json" "$CACHE_ROOT/pl_dip_test_official_init36" official
ensure_pl_cache "$CACHE_ROOT/tc_test_official_with_offset_r/baseline_cache_manifest.json" "$CACHE_ROOT/pl_tc_test_official_init36" official

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

if [[ ! -f "$ROOT/smoke/best_loss.pt" ]]; then
  "$PY" pl_curve_train.py \
    --train-cache "$CACHE_ROOT/pl_dip_train_official_init36/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/pl_dip_val_official_init36/pl_curve_cache_manifest.json" \
    --output-dir "$ROOT/smoke" \
    --experiment-name newpl_v5_official_protocol_smoke \
    --epochs 1 \
    --lr 1e-5 \
    --batch-size 16 \
    --max-train-sequences 16 \
    --max-val-sequences 2 \
    "${COMMON_TRAIN_ARGS[@]}"
fi

if [[ ! -f "$ROOT/amass_pretrain/best_loss.pt" ]]; then
  "$PY" pl_curve_train.py \
    --train-cache "$CACHE_ROOT/pl_amass_official_init36/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/pl_amass_official_init36/pl_curve_cache_manifest.json" \
    --output-dir "$ROOT/amass_pretrain" \
    --experiment-name newpl_v5_amass_pretrain_official_protocol \
    --epochs 80 \
    --lr 1e-4 \
    --batch-size 256 \
    --max-val-sequences 20 \
    --early-stop-min-delta 0.00000005 \
    --early-stop-patience 12 \
    "${COMMON_TRAIN_ARGS[@]}"
fi

"$PY" newpl_root_eval.py \
  --cache "$CACHE_ROOT/dip_test_with_offset_r/baseline_cache_manifest.json" \
  --output-json "$EVAL_ROOT/dip_test_after_amass_pretrain.json" \
  --dataset dip \
  --dataset-label DIP-IMU-test \
  --imu-input-mode official \
  --version official_PL=official \
  --version newpl_v4_init36="$NEWPL_V4" \
  --version newpl_v5_amass_best="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_v5_amass_last="$ROOT/amass_pretrain/last.pt"

"$PY" newpl_root_eval.py \
  --cache "$CACHE_ROOT/tc_test_official_with_offset_r/baseline_cache_manifest.json" \
  --output-json "$EVAL_ROOT/tc_test_after_amass_pretrain.json" \
  --dataset totalcapture \
  --dataset-label TotalCapture-test-official-input \
  --imu-input-mode official \
  --version official_PL=official \
  --version newpl_v4_init36="$NEWPL_V4" \
  --version newpl_v5_amass_best="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_v5_amass_last="$ROOT/amass_pretrain/last.pt"

if [[ ! -f "$ROOT/dip_finetune/best_loss.pt" ]]; then
  "$PY" pl_curve_train.py \
    --train-cache "$CACHE_ROOT/pl_dip_train_official_init36/pl_curve_cache_manifest.json" \
    --val-cache "$CACHE_ROOT/pl_dip_val_official_init36/pl_curve_cache_manifest.json" \
    --output-dir "$ROOT/dip_finetune" \
    --experiment-name newpl_v5_amass_to_dip_finetune_official_protocol \
    --epochs 40 \
    --lr 5e-6 \
    --batch-size 12 \
    --init-checkpoint "$ROOT/amass_pretrain/best_loss.pt" \
    --early-stop-min-delta 0.00000005 \
    --early-stop-patience 10 \
    "${COMMON_TRAIN_ARGS[@]}"
fi

"$PY" newpl_root_eval.py \
  --cache "$CACHE_ROOT/dip_test_with_offset_r/baseline_cache_manifest.json" \
  --output-json "$EVAL_ROOT/dip_test_after_dip_finetune.json" \
  --dataset dip \
  --dataset-label DIP-IMU-test \
  --imu-input-mode official \
  --version official_PL=official \
  --version newpl_v4_init36="$NEWPL_V4" \
  --version newpl_v5_amass_best="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_v5_amass_last="$ROOT/amass_pretrain/last.pt" \
  --version newpl_v5_dip_best="$ROOT/dip_finetune/best_loss.pt" \
  --version newpl_v5_dip_last="$ROOT/dip_finetune/last.pt"

"$PY" newpl_root_eval.py \
  --cache "$CACHE_ROOT/tc_test_official_with_offset_r/baseline_cache_manifest.json" \
  --output-json "$EVAL_ROOT/tc_test_after_dip_finetune.json" \
  --dataset totalcapture \
  --dataset-label TotalCapture-test-official-input \
  --imu-input-mode official \
  --version official_PL=official \
  --version newpl_v4_init36="$NEWPL_V4" \
  --version newpl_v5_amass_best="$ROOT/amass_pretrain/best_loss.pt" \
  --version newpl_v5_amass_last="$ROOT/amass_pretrain/last.pt" \
  --version newpl_v5_dip_best="$ROOT/dip_finetune/best_loss.pt" \
  --version newpl_v5_dip_last="$ROOT/dip_finetune/last.pt"

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
eval_root = root / "eval"
summary = {
    "status": "ok",
    "root": str(root),
    "protocol": "AMASS pretrain -> DIP train fine-tune -> DIP test and TotalCapture official-input test",
    "tc_train_used": False,
    "dip_trans_loss_used": False,
    "full_pipeline_11_metrics": "not measured",
    "jsons": {},
}
for name in [
    "dip_test_after_amass_pretrain",
    "tc_test_after_amass_pretrain",
    "dip_test_after_dip_finetune",
    "tc_test_after_dip_finetune",
]:
    path = eval_root / f"{name}.json"
    data = json.loads(path.read_text())
    summary["jsons"][name] = str(path)
    summary[name] = {
        "pl_output_comparison_table": data.get("pl_output_comparison_table", []),
        "per_leaf_table": data.get("per_leaf_table", []),
    }

out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
md = root / "summary.md"
lines = [
    "# NewPL v5 Official-Protocol Summary",
    "",
    "Protocol: AMASS pretrain -> DIP train fine-tune -> DIP test and TotalCapture official-input test.",
    "",
    "TC train used: false.",
    "DIP trans/root-velocity loss used: false.",
    "Full-pipeline 11 metrics: not measured.",
    "",
]
for key in ["dip_test_after_amass_pretrain", "tc_test_after_amass_pretrain", "dip_test_after_dip_finetune", "tc_test_after_dip_finetune"]:
    lines.append(f"## {key}")
    lines.append("")
    lines.append("| Dataset | Version | pRB L1 cm | pRB L2 cm | gR1 angle deg | Notes |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in summary[key]["pl_output_comparison_table"]:
        lines.append(
            f"| {row['Dataset']} | {row['Version']} | {row['pRB L1 cm ↓']} | {row['pRB L2 cm ↓']} | {row['gR1 angle deg ↓']} | {row['Notes']} |"
        )
    lines.append("")
md.write_text("\n".join(lines) + "\n")
print(json.dumps({"status": "ok", "summary_json": str(out), "summary_md": str(md)}, indent=2))
PY
