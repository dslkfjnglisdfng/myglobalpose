#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=${ENV_DIR:-/home/lingfeng/.conda/envs/globalpose-gpu}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

PY="$ENV_DIR/bin/python"
ROOT=${ROOT:-data/experiments/bone_aux_newpl_20260608}
CACHE_ROOT="$ROOT/caches"
EVAL_DIR="$ROOT/eval"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$EVAL_DIR" "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

SMOKE=${SMOKE:-0}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}

AMASS_SRC=${AMASS_SRC:-data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay_processed_fields/baseline_cache_manifest.json}
DIP_TRAIN_SRC=${DIP_TRAIN_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json}
DIP_VAL_SRC=${DIP_VAL_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json}
DIP_TEST_SRC=${DIP_TEST_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json}
TC_TEST_SRC=${TC_TEST_SRC:-data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json}

AMASS_CACHE="$CACHE_ROOT/pl_amass_v4_offset_aware/pl_curve_cache_manifest.json"
DIP_TRAIN_CACHE="$CACHE_ROOT/pl_dip_train_v4_offset_aware/pl_curve_cache_manifest.json"
DIP_VAL_CACHE="$CACHE_ROOT/pl_dip_val_v4_offset_aware/pl_curve_cache_manifest.json"
DIP_TEST_CACHE="$CACHE_ROOT/pl_dip_test_v4_offset_aware/pl_curve_cache_manifest.json"
TC_TEST_CACHE="$CACHE_ROOT/pl_tc_test_v4_offset_aware/pl_curve_cache_manifest.json"

AMASS_OUT="$ROOT/amass_bone_aux"
DIP_OUT="$ROOT/dip_finetune"

echo "started $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ROOT=$ROOT SMOKE=$SMOKE"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

if [ "$SMOKE" = "1" ]; then
  CACHE_MAX_SEQS=${CACHE_MAX_SEQS:-8}
  AMASS_EPOCHS=${AMASS_EPOCHS:-1}
  DIP_EPOCHS=${DIP_EPOCHS:-1}
  AMASS_MAX_TRAIN=${AMASS_MAX_TRAIN:-8}
  AMASS_MAX_VAL=${AMASS_MAX_VAL:-4}
  DIP_MAX_TRAIN=${DIP_MAX_TRAIN:-4}
  DIP_MAX_VAL=${DIP_MAX_VAL:-2}
  EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-2}
  TRAIN_BATCH=${TRAIN_BATCH:-8}
  DIP_BATCH=${DIP_BATCH:-2}
  HIDDEN_SIZE=${HIDDEN_SIZE:-128}
else
  CACHE_MAX_SEQS=${CACHE_MAX_SEQS:-0}
  AMASS_EPOCHS=${AMASS_EPOCHS:-80}
  DIP_EPOCHS=${DIP_EPOCHS:-40}
  AMASS_MAX_TRAIN=${AMASS_MAX_TRAIN:-0}
  AMASS_MAX_VAL=${AMASS_MAX_VAL:-20}
  DIP_MAX_TRAIN=${DIP_MAX_TRAIN:-0}
  DIP_MAX_VAL=${DIP_MAX_VAL:-0}
  EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-0}
  TRAIN_BATCH=${TRAIN_BATCH:-256}
  DIP_BATCH=${DIP_BATCH:-12}
  HIDDEN_SIZE=${HIDDEN_SIZE:-512}
fi

ensure_bone_cache() {
  local src="$1"
  local out_dir="$2"
  local manifest="$out_dir/pl_curve_cache_manifest.json"
  local ok=0
  if [ -f "$manifest" ]; then
    ok="$("$PY" - "$manifest" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(1 if d.get("type") == "pl_curve_cache_v4" and "bone6d_target" in d.get("fields", {}) else 0)
PY
)"
  fi
  if [ "$ok" != "1" ]; then
    rm -rf "$out_dir"
    "$PY" pl_curve_cache.py \
      --input-cache "$src" \
      --output-dir "$out_dir" \
      --shard-size 100 \
      --imu-input-mode official \
      --feature-mode offset_aware \
      --max-sequences "$CACHE_MAX_SEQS"
  fi
}

ensure_bone_cache "$AMASS_SRC" "$CACHE_ROOT/pl_amass_v4_offset_aware"
ensure_bone_cache "$DIP_TRAIN_SRC" "$CACHE_ROOT/pl_dip_train_v4_offset_aware"
ensure_bone_cache "$DIP_VAL_SRC" "$CACHE_ROOT/pl_dip_val_v4_offset_aware"
ensure_bone_cache "$DIP_TEST_SRC" "$CACHE_ROOT/pl_dip_test_v4_offset_aware"
ensure_bone_cache "$TC_TEST_SRC" "$CACHE_ROOT/pl_tc_test_v4_offset_aware"

COMMON_LOSS_ARGS=(
  --pRB-weight 1.0
  --gR1-weight 1.0
  --baseline-pRB-weight 0.05
  --baseline-gR1-weight 0.0
  --gt-control-pRB-weight 0.3
  --gt-control-gR1-weight 0.1
  --pRB-ddot-smooth-weight 1e-6
  --bone6d-weight 0.05
  --bone-geo-weight 0.2
  --gt-control-bone6d-weight 0.2
  --gt-control-bone-geo-weight 0.2
  --bone6d-dot-weight 0.03
  --bone6d-ddot-weight 0.001
  --bone-control-point-prior-weight 0.01
  --bone-tail-update-prior-weight 0.005
)

INIT_ARGS=()
if [ -n "$INIT_CHECKPOINT" ]; then
  INIT_ARGS=(--init-checkpoint "$INIT_CHECKPOINT")
fi

if [ ! -f "$AMASS_OUT/train_result.json" ]; then
  "$PY" pl_curve_train.py \
    --train-cache "$AMASS_CACHE" \
    --val-cache "$AMASS_CACHE" \
    --output-dir "$AMASS_OUT" \
    --experiment-name bone_aux_newpl_amass_stage_a \
    --epochs "$AMASS_EPOCHS" \
    --window 61 \
    --lr 1e-4 \
    --hidden-size "$HIDDEN_SIZE" \
    --batch-size "$TRAIN_BATCH" \
    --dropout 0.15 \
    --disable-ik-distill \
    --model-variant offset_aware \
    --input-size 156 \
    --init-size 36 \
    --film-scale 0.3 \
    --offset-embed-size 128 \
    --bone-aux-dim 30 \
    --bone-residual-scale 0.05 \
    --selection-metric pl_control_bone_physical \
    "${COMMON_LOSS_ARGS[@]}" \
    "${INIT_ARGS[@]}" \
    --early-stop-patience 12 \
    --early-stop-min-delta 5e-8 \
    --max-train-sequences "$AMASS_MAX_TRAIN" \
    --max-val-sequences "$AMASS_MAX_VAL"
fi

if [ ! -f "$DIP_OUT/train_result.json" ]; then
  "$PY" pl_curve_train.py \
    --train-cache "$DIP_TRAIN_CACHE" \
    --val-cache "$DIP_VAL_CACHE" \
    --output-dir "$DIP_OUT" \
    --experiment-name bone_aux_newpl_dip_stage_b \
    --epochs "$DIP_EPOCHS" \
    --window 61 \
    --lr 5e-6 \
    --hidden-size "$HIDDEN_SIZE" \
    --batch-size "$DIP_BATCH" \
    --dropout 0.05 \
    --disable-ik-distill \
    --model-variant offset_aware \
    --input-size 156 \
    --init-size 36 \
    --film-scale 0.3 \
    --offset-embed-size 128 \
    --bone-aux-dim 30 \
    --bone-residual-scale 0.03 \
    --init-checkpoint "$AMASS_OUT/best_loss.pt" \
    --selection-metric pl_control_bone_physical \
    "${COMMON_LOSS_ARGS[@]}" \
    --early-stop-patience 10 \
    --early-stop-min-delta 5e-8 \
    --max-train-sequences "$DIP_MAX_TRAIN" \
    --max-val-sequences "$DIP_MAX_VAL"
fi

run_module_eval() {
  local ckpt_name="$1"
  local ckpt_path="$2"
  local split="$3"
  local cache="$4"
  "$PY" pl_curve_pl_accuracy_eval.py \
    --pl-cache "$cache" \
    --checkpoint "$ckpt_path" \
    --output-json "$EVAL_DIR/${ckpt_name}_${split}_module_pl_bone_accuracy.json" \
    --max-sequences "$EVAL_MAX_SEQS"
}

for ckpt_name in best last; do
  case "$ckpt_name" in
    best) ckpt_path="$DIP_OUT/best_loss.pt" ;;
    last) ckpt_path="$DIP_OUT/last.pt" ;;
  esac
  for split in dip_val dip_test tc_test; do
    case "$split" in
      dip_val) cache="$DIP_VAL_CACHE" ;;
      dip_test) cache="$DIP_TEST_CACHE" ;;
      tc_test) cache="$TC_TEST_CACHE" ;;
    esac
    run_module_eval "$ckpt_name" "$ckpt_path" "$split" "$cache"
  done
done

"$PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "eval").glob("*_module_pl_bone_accuracy.json")):
    data = json.load(open(path))
    agg = data.get("aggregate", {})
    bone = agg.get("bone_orientation_angle_deg", {})
    leaf = agg.get("leaf_position_error_cm", {})
    rows.append({
        "file": path.name,
        "status": data.get("status"),
        "num_sequences": agg.get("num_sequences"),
        "leaf_new_cm_mean": leaf.get("new", {}).get("mean"),
        "bone_base_deg_mean": bone.get("base", {}).get("mean"),
        "bone_new_deg_mean": bone.get("new", {}).get("mean"),
        "bone_delta_deg_mean": bone.get("delta_new_minus_base", {}).get("mean"),
    })
summary = {"root": str(root), "rows": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = ["# Bone Aux NewPL Summary", "", "| file | status | seq | leaf new cm | bone base deg | bone new deg | bone delta deg |", "|---|---:|---:|---:|---:|---:|---:|"]
for row in rows:
    def fmt(v):
        return "" if v is None else f"{float(v):.6f}"
    lines.append(f"| {row['file']} | {row['status']} | {row['num_sequences']} | {fmt(row['leaf_new_cm_mean'])} | {fmt(row['bone_base_deg_mean'])} | {fmt(row['bone_new_deg_mean'])} | {fmt(row['bone_delta_deg_mean'])} |")
(root / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "finished $(date --iso-8601=seconds)"
