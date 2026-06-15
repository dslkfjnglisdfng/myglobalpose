#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/lingfeng/.conda/envs/globalpose-gpu/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi
export LD_LIBRARY_PATH="/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}"

EXP="${EXP:-data/experiments/newpl_v6_next_control_tail4_20260611}"
CACHE_DIR="$EXP/caches"
LOG_DIR="$EXP/logs"
RUN_ROOT="$EXP/$MODE"
if [[ -n "${RUN_SUFFIX:-}" ]]; then
  RUN_ROOT="${EXP}/${MODE}_${RUN_SUFFIX}"
fi
NEXT_CACHE_DIR="$CACHE_DIR/$MODE"
mkdir -p "$CACHE_DIR" "$NEXT_CACHE_DIR" "$LOG_DIR" "$RUN_ROOT"

V4_CKPT="${V4_CKPT:-data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt}"
BATCH_SIZE="${BATCH_SIZE:-512}"
WINDOW="${WINDOW:-81}"
if ! [[ "$WINDOW" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid WINDOW=$WINDOW; using WINDOW=81 for non-empty training windows." >&2
  WINDOW=81
fi

PL_AMASS="data/experiments/newpl_v5_official_protocol_20260607/caches/pl_amass_official_init36/pl_curve_cache_manifest.json"
PL_DIP_TRAIN="data/experiments/newpl_v5_official_protocol_20260607/caches/pl_dip_train_official_init36/pl_curve_cache_manifest.json"
PL_DIP_VAL="data/experiments/newpl_v5_official_protocol_20260607/caches/pl_dip_val_official_init36/pl_curve_cache_manifest.json"
PL_DIP_TEST="data/experiments/newpl_v5_official_protocol_20260607/caches/pl_dip_test_official_init36/pl_curve_cache_manifest.json"
PL_TC_TEST="data/experiments/newpl_v5_official_protocol_20260607/caches/pl_tc_test_official_init36/pl_curve_cache_manifest.json"

GT_AMASS="data/dataset_work/GTControlCache/amass_train/gt_control_cache_manifest.json"
GT_DIP_TRAIN="data/dataset_work/GTControlCache/dip_train/gt_control_cache_manifest.json"
GT_DIP_VAL="data/dataset_work/GTControlCache/dip_val/gt_control_cache_manifest.json"
GT_DIP_TEST="data/dataset_work/GTControlCache/dip_test/gt_control_cache_manifest.json"
GT_TC_TEST="data/dataset_work/GTControlCache/totalcapture_test/gt_control_cache_manifest.json"

MAX_CACHE_SEQS=0
EPOCHS_AMASS="${EPOCHS_AMASS:-80}"
EPOCHS_DIP="${EPOCHS_DIP:-40}"
MAX_TRAIN_VAL_SEQS="${MAX_TRAIN_VAL_SEQS:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MAX_EVAL_SEQS=0
if [[ "$MODE" == "smoke" ]]; then
  MAX_CACHE_SEQS="${SMOKE_MAX_CACHE_SEQS:-4}"
  EPOCHS_AMASS="${SMOKE_EPOCHS_AMASS:-1}"
  EPOCHS_DIP="${SMOKE_EPOCHS_DIP:-1}"
  MAX_TRAIN_VAL_SEQS="${SMOKE_MAX_EVAL_SEQS:-4}"
  MAX_EVAL_SEQS="${SMOKE_MAX_EVAL_SEQS:-4}"
  BATCH_SIZE="${SMOKE_BATCH_SIZE:-4}"
elif [[ "$MODE" != "full" ]]; then
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

build_next_cache_if_missing() {
  local pl_cache="$1"
  local gt_cache="$2"
  local out_dir="$3"
  local manifest="$out_dir/pl_next_control_cache_manifest.json"
  if [[ -f "$manifest" ]]; then
    local cache_type
    cache_type="$("$PY" - "$manifest" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f).get('type', ''))
PY
)"
    if [[ "$cache_type" == "pl_next_control_cache_v2" ]]; then
      return
    fi
    echo "Existing cache at $out_dir is type=$cache_type; writing v2 cache into a fresh experiment root is required." >&2
    exit 3
  fi
  "$PY" pl_next_control_cache.py \
    --pl-cache "$pl_cache" \
    --gt-control-cache "$gt_cache" \
    --output-dir "$out_dir" \
    --shard-size 100 \
    --max-sequences "$MAX_CACHE_SEQS"
}

build_next_cache_if_missing "$PL_AMASS" "$GT_AMASS" "$NEXT_CACHE_DIR/next_amass_train"
build_next_cache_if_missing "$PL_DIP_TRAIN" "$GT_DIP_TRAIN" "$NEXT_CACHE_DIR/next_dip_train"
build_next_cache_if_missing "$PL_DIP_VAL" "$GT_DIP_VAL" "$NEXT_CACHE_DIR/next_dip_val"
build_next_cache_if_missing "$PL_DIP_TEST" "$GT_DIP_TEST" "$NEXT_CACHE_DIR/next_dip_test"
build_next_cache_if_missing "$PL_TC_TEST" "$GT_TC_TEST" "$NEXT_CACHE_DIR/next_tc_test"

NEXT_AMASS="$NEXT_CACHE_DIR/next_amass_train/pl_next_control_cache_manifest.json"
NEXT_DIP_TRAIN="$NEXT_CACHE_DIR/next_dip_train/pl_next_control_cache_manifest.json"
NEXT_DIP_VAL="$NEXT_CACHE_DIR/next_dip_val/pl_next_control_cache_manifest.json"
NEXT_DIP_TEST="$NEXT_CACHE_DIR/next_dip_test/pl_next_control_cache_manifest.json"
NEXT_TC_TEST="$NEXT_CACHE_DIR/next_tc_test/pl_next_control_cache_manifest.json"

"$PY" pl_next_control_train.py \
  --train-cache "$NEXT_AMASS" \
  --val-cache "$NEXT_AMASS" \
  --output-dir "$RUN_ROOT/amass_pretrain" \
  --experiment-name "newpl_v6_next_control_tail4_amass_pretrain_${MODE}" \
  --epochs "$EPOCHS_AMASS" \
  --window "$WINDOW" \
  --batch-size "$BATCH_SIZE" \
  --val-window-length 512 \
  --max-val-sequences "$MAX_TRAIN_VAL_SEQS" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --lr 1e-4

"$PY" pl_next_control_eval.py \
  --cache "$NEXT_AMASS" \
  --dataset-label "AMASS module ${MODE}" \
  --output-json "$RUN_ROOT/eval_amass_after_pretrain.json" \
  --max-eval-sequences "$MAX_EVAL_SEQS" \
  --version "official PL baseline=official" \
  --version "newpl_v4_init36 baseline=$V4_CKPT" \
  --version "newpl_v6_next_control_amass=$RUN_ROOT/amass_pretrain/best_next_module_metric.pt" \
  --version "newpl_v6_next_control_amass_control=$RUN_ROOT/amass_pretrain/best_control_metric.pt"

"$PY" pl_next_control_eval.py \
  --cache "$NEXT_TC_TEST" \
  --dataset-label "TotalCapture test after AMASS ${MODE}" \
  --output-json "$RUN_ROOT/eval_totalcapture_test_after_amass_pretrain.json" \
  --max-eval-sequences "$MAX_EVAL_SEQS" \
  --version "official PL baseline=official" \
  --version "newpl_v4_init36 baseline=$V4_CKPT" \
  --version "newpl_v6_next_control_amass=$RUN_ROOT/amass_pretrain/best_next_module_metric.pt" \
  --version "newpl_v6_next_control_amass_control=$RUN_ROOT/amass_pretrain/best_control_metric.pt"

"$PY" pl_next_control_train.py \
  --train-cache "$NEXT_DIP_TRAIN" \
  --val-cache "$NEXT_DIP_VAL" \
  --output-dir "$RUN_ROOT/dip_finetune" \
  --experiment-name "newpl_v6_next_control_tail4_dip_finetune_${MODE}" \
  --init-checkpoint "$RUN_ROOT/amass_pretrain/best_next_module_metric.pt" \
  --epochs "$EPOCHS_DIP" \
  --window "$WINDOW" \
  --batch-size "$BATCH_SIZE" \
  --val-window-length 512 \
  --max-val-sequences "$MAX_TRAIN_VAL_SEQS" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --lr 5e-5

"$PY" pl_next_control_eval.py \
  --cache "$NEXT_DIP_TEST" \
  --dataset-label "DIP-IMU test after DIP ${MODE}" \
  --output-json "$RUN_ROOT/eval_dip_test_after_dip_finetune.json" \
  --max-eval-sequences "$MAX_EVAL_SEQS" \
  --version "official PL baseline=official" \
  --version "newpl_v4_init36 baseline=$V4_CKPT" \
  --version "newpl_v6_next_control_amass=$RUN_ROOT/amass_pretrain/best_next_module_metric.pt" \
  --version "newpl_v6_next_control_dip=$RUN_ROOT/dip_finetune/best_next_module_metric.pt" \
  --version "newpl_v6_next_control_dip_control=$RUN_ROOT/dip_finetune/best_control_metric.pt"

"$PY" pl_next_control_eval.py \
  --cache "$NEXT_TC_TEST" \
  --dataset-label "TotalCapture test after DIP ${MODE}" \
  --output-json "$RUN_ROOT/eval_totalcapture_test_after_dip_finetune.json" \
  --max-eval-sequences "$MAX_EVAL_SEQS" \
  --version "official PL baseline=official" \
  --version "newpl_v4_init36 baseline=$V4_CKPT" \
  --version "newpl_v6_next_control_amass=$RUN_ROOT/amass_pretrain/best_next_module_metric.pt" \
  --version "newpl_v6_next_control_dip=$RUN_ROOT/dip_finetune/best_next_module_metric.pt" \
  --version "newpl_v6_next_control_dip_control=$RUN_ROOT/dip_finetune/best_control_metric.pt"

cat > "$RUN_ROOT/run_summary_${MODE}.json" <<JSON
{
  "status": "ok",
  "mode": "$MODE",
  "batch_size": $BATCH_SIZE,
  "window": $WINDOW,
  "amass_eval": "$RUN_ROOT/eval_amass_after_pretrain.json",
  "totalcapture_after_amass_eval": "$RUN_ROOT/eval_totalcapture_test_after_amass_pretrain.json",
  "dip_eval": "$RUN_ROOT/eval_dip_test_after_dip_finetune.json",
  "totalcapture_after_dip_eval": "$RUN_ROOT/eval_totalcapture_test_after_dip_finetune.json"
}
JSON
