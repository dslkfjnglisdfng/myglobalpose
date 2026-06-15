#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=${ENV_DIR:-/home/lingfeng/.conda/envs/globalpose-gpu}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

PY="$ENV_DIR/bin/python"
ROOT=${ROOT:-data/experiments/bone_aux_newpl_20260608_v4_winner_rjs_dip_ft}
CACHE_ROOT="$ROOT/caches"
EVAL_DIR="$ROOT/eval"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$EVAL_DIR" "$LOG_DIR"

RUN_LOG="$LOG_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

SMOKE=${SMOKE:-0}

INIT_CHECKPOINT=${INIT_CHECKPOINT:-data/experiments/bone_aux_newpl_20260608_v4/amass_bone_aux/best_loss.pt}
DIP_TRAIN_SRC=${DIP_TRAIN_SRC:-data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json}
DIP_VAL_SRC=${DIP_VAL_SRC:-data/dataset_work/L4Cache/prephysics_pose_velocity_dip_val_globalpose_neural_only/baseline_cache_manifest.json}
DIP_TEST_SRC=${DIP_TEST_SRC:-data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json}

DIP_TRAIN_RJS=${DIP_TRAIN_RJS:-data/experiments/footlock_transpose_rjs_20260608/dip_train_footlock_transpose_rjs.pt}
DIP_VAL_RJS=${DIP_VAL_RJS:-data/experiments/footlock_transpose_rjs_20260608/dip_val_footlock_transpose_rjs.pt}
DIP_TEST_RJS=${DIP_TEST_RJS:-data/experiments/footlock_transpose_rjs_20260608/dip_test_footlock_transpose_rjs.pt}

DIP_TRAIN_CACHE="$CACHE_ROOT/dip_train_winner_rjs_bone_aux/pl_curve_cache_manifest.json"
DIP_VAL_CACHE="$CACHE_ROOT/dip_val_winner_rjs_bone_aux/pl_curve_cache_manifest.json"
DIP_TEST_CACHE="$CACHE_ROOT/dip_test_winner_rjs_bone_aux/pl_curve_cache_manifest.json"
DIP_OUT="$ROOT/dip_finetune"

echo "started $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ROOT=$ROOT SMOKE=$SMOKE"
echo "INIT_CHECKPOINT=$INIT_CHECKPOINT"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

if [ "$SMOKE" = "1" ]; then
  CACHE_MAX_SEQS=${CACHE_MAX_SEQS:-4}
  DIP_EPOCHS=${DIP_EPOCHS:-1}
  DIP_MAX_TRAIN=${DIP_MAX_TRAIN:-4}
  DIP_MAX_VAL=${DIP_MAX_VAL:-2}
  EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-2}
  DIP_BATCH=${DIP_BATCH:-2}
  HIDDEN_SIZE=${HIDDEN_SIZE:-128}
else
  CACHE_MAX_SEQS=${CACHE_MAX_SEQS:-0}
  DIP_EPOCHS=${DIP_EPOCHS:-40}
  DIP_MAX_TRAIN=${DIP_MAX_TRAIN:-0}
  DIP_MAX_VAL=${DIP_MAX_VAL:-0}
  EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-0}
  DIP_BATCH=${DIP_BATCH:-12}
  HIDDEN_SIZE=${HIDDEN_SIZE:-512}
fi

ensure_cache() {
  local src="$1"
  local offset_cache="$2"
  local out_dir="$3"
  local manifest="$out_dir/pl_curve_cache_manifest.json"
  local ok=0
  if [ -f "$manifest" ]; then
    ok="$("$PY" - "$manifest" "$offset_cache" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(1 if (
    d.get("type") == "pl_curve_cache_v4"
    and d.get("feature_mode") == "offset_aware"
    and d.get("offset_cache") == sys.argv[2]
    and "bone6d_target" in d.get("fields", {})
) else 0)
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
      --offset-cache "$offset_cache" \
      --max-sequences "$CACHE_MAX_SEQS"
  fi
}

ensure_cache "$DIP_TRAIN_SRC" "$DIP_TRAIN_RJS" "$CACHE_ROOT/dip_train_winner_rjs_bone_aux"
ensure_cache "$DIP_VAL_SRC" "$DIP_VAL_RJS" "$CACHE_ROOT/dip_val_winner_rjs_bone_aux"
ensure_cache "$DIP_TEST_SRC" "$DIP_TEST_RJS" "$CACHE_ROOT/dip_test_winner_rjs_bone_aux"

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

if [ ! -f "$DIP_OUT/train_result.json" ]; then
  "$PY" pl_curve_train.py \
    --train-cache "$DIP_TRAIN_CACHE" \
    --val-cache "$DIP_VAL_CACHE" \
    --output-dir "$DIP_OUT" \
    --experiment-name bone_aux_v4_winner_rjs_dip_finetune \
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
    --bone-residual-scale 0.05 \
    --init-checkpoint "$INIT_CHECKPOINT" \
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
  for split in dip_val dip_test; do
    case "$split" in
      dip_val) cache="$DIP_VAL_CACHE" ;;
      dip_test) cache="$DIP_TEST_CACHE" ;;
    esac
    run_module_eval "$ckpt_name" "$ckpt_path" "$split" "$cache"
  done
done

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "eval").glob("*_module_pl_bone_accuracy.json")):
    data = json.load(open(path))
    agg = data.get("aggregate", {})
    leaf = agg.get("leaf_position_error_cm", {})
    gravity = agg.get("gravity_angle_deg", {})
    bone = agg.get("bone_orientation_angle_deg", {})
    rows.append({
        "file": path.name,
        "status": data.get("status"),
        "num_sequences": agg.get("num_sequences"),
        "pRB_base_cm": leaf.get("original", {}).get("mean"),
        "pRB_new_cm": leaf.get("new", {}).get("mean"),
        "pRB_delta_cm": leaf.get("delta_new_minus_original", {}).get("mean"),
        "gR1_base_deg": gravity.get("original", {}).get("mean"),
        "gR1_new_deg": gravity.get("new", {}).get("mean"),
        "gR1_delta_deg": gravity.get("delta_new_minus_original", {}).get("mean"),
        "bone_base_deg": bone.get("base", {}).get("mean"),
        "bone_new_deg": bone.get("new", {}).get("mean"),
        "bone_delta_deg": bone.get("delta_new_minus_base", {}).get("mean"),
    })

summary = {
    "root": str(root),
    "init_checkpoint": str(root / "dip_finetune" / "config.json"),
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "# Bone Aux v4 Winner-rJS DIP Fine-tune Summary",
    "",
    "| file | status | seq | pRB base cm | pRB new cm | delta cm | gR1 base deg | gR1 new deg | delta deg | bone base deg | bone new deg | delta deg |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    def fmt(value):
        return "" if value is None else f"{float(value):.6f}"
    lines.append(
        f"| {row['file']} | {row['status']} | {row['num_sequences']} | "
        f"{fmt(row['pRB_base_cm'])} | {fmt(row['pRB_new_cm'])} | {fmt(row['pRB_delta_cm'])} | "
        f"{fmt(row['gR1_base_deg'])} | {fmt(row['gR1_new_deg'])} | {fmt(row['gR1_delta_deg'])} | "
        f"{fmt(row['bone_base_deg'])} | {fmt(row['bone_new_deg'])} | {fmt(row['bone_delta_deg'])} |"
    )
(root / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "finished $(date --iso-8601=seconds)"
