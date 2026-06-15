#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ENV:-}" ]]; then
  if [[ -x /home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python ]]; then
    ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310
  else
    ENV=/home/lingfeng/.conda/envs/globalpose-gpu
  fi
fi
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
PY="$ENV/bin/python"

ROOT=${ROOT:-data/experiments/ik2_q75_ctrl_v1_20260614}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/caches}
EVAL_ROOT="$ROOT/eval"
LOG_ROOT="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

AMASS_RAW=${AMASS_RAW:-data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json}
DIP_TRAIN_RAW=${DIP_TRAIN_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json}
DIP_VAL_RAW=${DIP_VAL_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json}
DIP_TEST_RAW=${DIP_TEST_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json}
TC_TEST_RAW=${TC_TEST_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json}

SMOKE_ONLY=${SMOKE_ONLY:-0}
RUN_SMOKE=${RUN_SMOKE:-1}
RUN_MODULE_EVAL=${RUN_MODULE_EVAL:-1}
RUN_FULL_DIAG=${RUN_FULL_DIAG:-1}
MODULE_EVAL_MAX_SEQUENCES=${MODULE_EVAL_MAX_SEQUENCES:-0}
FULL_DIAG_MAX_SEQUENCES=${FULL_DIAG_MAX_SEQUENCES:-2}
FULL_DIAG_MAX_FRAMES=${FULL_DIAG_MAX_FRAMES:-300}
CACHE_MAX=${CACHE_MAX:-0}
CACHE_SHARD_SIZE=${CACHE_SHARD_SIZE:-25}
CUTOFF_HZ=${CUTOFF_HZ:-20}
FILTER_FS=${FILTER_FS:-60}
AMASS_EPOCHS=${AMASS_EPOCHS:-80}
DIP_EPOCHS=${DIP_EPOCHS:-60}
AMASS_BATCH=${AMASS_BATCH:-32}
DIP_BATCH=${DIP_BATCH:-16}

ensure_cache() {
  local input_cache="$1"
  local out_dir="$2"
  local max_sequences="${3:-0}"
  if [[ ! -f "$out_dir/ik2_q75_ctrl_cache_manifest.json" ]]; then
    local args=(
      ik2_q75_ctrl_cache.py
      --input-cache "$input_cache"
      --output-dir "$out_dir"
      --shard-size "$CACHE_SHARD_SIZE"
      --cutoff-hz "$CUTOFF_HZ"
      --filter-fs "$FILTER_FS"
    )
    if [[ "$max_sequences" != "0" ]]; then
      args+=(--max-sequences "$max_sequences")
    elif [[ "$CACHE_MAX" != "0" ]]; then
      args+=(--max-sequences "$CACHE_MAX")
    fi
    "$PY" "${args[@]}"
  fi
}

train_stage() {
  local preset="$1"
  local train_cache="$2"
  local val_cache="$3"
  local out_dir="$4"
  local name="$5"
  local epochs="$6"
  local lr="$7"
  local min_lr="$8"
  local batch="$9"
  local init="${10:-}"
  if [[ ! -f "$out_dir/train_result.json" ]]; then
    local args=(
      ik2_q75_ctrl_train.py
      --train-cache "$train_cache"
      --val-cache "$val_cache"
      --output-dir "$out_dir"
      --experiment-name "$name"
      --epochs "$epochs"
      --lr "$lr"
      --min-lr "$min_lr"
      --warmup-epochs 2
      --batch-size "$batch"
      --window 61
      --hidden-size 512
      --residual-scale 0.05
      --dropout 0.2
      --offset-init-scale 0.1
      --grad-clip 1.0
      --weight-decay 0.0001
      --early-stop-min-delta 0.00000005
      --early-stop-patience 12
      --loss-preset "$preset"
      --selection-metric fk_pva
    )
    if [[ -n "$init" ]]; then
      args+=(--init-checkpoint "$init")
    fi
    "$PY" "${args[@]}"
  fi
}

eval_module() {
  local checkpoint="$1"
  local cache="$2"
  local out_json="$3"
  local split="$4"
  local version="$5"
  if [[ ! -f "$out_json" ]]; then
    local args=(
      ik2_q75_ctrl_eval.py
      --checkpoint "$checkpoint"
      --cache "$cache"
      --output-json "$out_json"
      --split-label "$split"
      --version-name "$version"
      --module-only
    )
    if [[ "$MODULE_EVAL_MAX_SEQUENCES" != "0" ]]; then
      args+=(--max-eval-sequences "$MODULE_EVAL_MAX_SEQUENCES")
    fi
    "$PY" "${args[@]}"
  fi
}

eval_full_diag() {
  local checkpoint="$1"
  local cache="$2"
  local out_json="$3"
  local split="$4"
  local version="$5"
  if [[ ! -f "$out_json" ]]; then
    "$PY" ik2_q75_ctrl_eval.py \
      --checkpoint "$checkpoint" \
      --cache "$cache" \
      --output-json "$out_json" \
      --split-label "$split" \
      --version-name "$version" \
      --max-eval-sequences "$FULL_DIAG_MAX_SEQUENCES" \
      --max-smoke-frames "$FULL_DIAG_MAX_FRAMES"
  fi
}

ensure_cache "$DIP_VAL_RAW" "$CACHE_ROOT/smoke_dip_val" 1
if [[ "$RUN_SMOKE" == "1" ]]; then
  train_stage pos_dominant \
    "$CACHE_ROOT/smoke_dip_val/ik2_q75_ctrl_cache_manifest.json" \
    "$CACHE_ROOT/smoke_dip_val/ik2_q75_ctrl_cache_manifest.json" \
    "$ROOT/smoke_pos_dominant" \
    ik2_q75_ctrl_v1_smoke_pos_dominant \
    1 5e-5 1e-6 1 ""
  eval_module \
    "$ROOT/smoke_pos_dominant/best_fk_pva.pt" \
    "$CACHE_ROOT/smoke_dip_val/ik2_q75_ctrl_cache_manifest.json" \
    "$EVAL_ROOT/smoke_module.json" \
    DIP-val-smoke \
    ik2_q75_ctrl_v1_smoke
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo '{"status":"smoke_ok"}'
  exit 0
fi

ensure_cache "$AMASS_RAW" "$CACHE_ROOT/amass_train"
ensure_cache "$DIP_TRAIN_RAW" "$CACHE_ROOT/dip_train"
ensure_cache "$DIP_VAL_RAW" "$CACHE_ROOT/dip_val"
ensure_cache "$DIP_TEST_RAW" "$CACHE_ROOT/dip_test"
ensure_cache "$TC_TEST_RAW" "$CACHE_ROOT/tc_test"

for preset in pos_dominant balanced acc_stronger; do
  train_stage "$preset" \
    "$CACHE_ROOT/amass_train/ik2_q75_ctrl_cache_manifest.json" \
    "$CACHE_ROOT/amass_train/ik2_q75_ctrl_cache_manifest.json" \
    "$ROOT/${preset}/stage_a_amass_pretrain" \
    "ik2_q75_ctrl_v1_${preset}_amass_pretrain" \
    "$AMASS_EPOCHS" 1e-4 1e-6 "$AMASS_BATCH" ""

  train_stage "$preset" \
    "$CACHE_ROOT/dip_train/ik2_q75_ctrl_cache_manifest.json" \
    "$CACHE_ROOT/dip_val/ik2_q75_ctrl_cache_manifest.json" \
    "$ROOT/${preset}/stage_b_dip_finetune" \
    "ik2_q75_ctrl_v1_${preset}_dip_finetune" \
    "$DIP_EPOCHS" 2e-5 2e-7 "$DIP_BATCH" \
    "$ROOT/${preset}/stage_a_amass_pretrain/best_fk_pva.pt"

  if [[ "$RUN_MODULE_EVAL" == "1" ]]; then
    eval_module \
      "$ROOT/${preset}/stage_a_amass_pretrain/best_fk_pva.pt" \
      "$CACHE_ROOT/dip_test/ik2_q75_ctrl_cache_manifest.json" \
      "$EVAL_ROOT/${preset}_dip_test_after_amass_module.json" \
      DIP-test \
      "ik2_q75_ctrl_v1_${preset}_amass"
    eval_module \
      "$ROOT/${preset}/stage_b_dip_finetune/best_fk_pva.pt" \
      "$CACHE_ROOT/dip_test/ik2_q75_ctrl_cache_manifest.json" \
      "$EVAL_ROOT/${preset}_dip_test_after_dip_module.json" \
      DIP-test \
      "ik2_q75_ctrl_v1_${preset}_dip"
    eval_module \
      "$ROOT/${preset}/stage_b_dip_finetune/best_fk_pva.pt" \
      "$CACHE_ROOT/tc_test/ik2_q75_ctrl_cache_manifest.json" \
      "$EVAL_ROOT/${preset}_tc_test_after_dip_module.json" \
      TotalCapture-test \
      "ik2_q75_ctrl_v1_${preset}_dip"
  fi

  if [[ "$RUN_FULL_DIAG" == "1" ]]; then
    eval_full_diag \
      "$ROOT/${preset}/stage_b_dip_finetune/best_fk_pva.pt" \
      "$CACHE_ROOT/dip_test/ik2_q75_ctrl_cache_manifest.json" \
      "$EVAL_ROOT/${preset}_dip_test_full_diag.json" \
      DIP-test \
      "ik2_q75_ctrl_v1_${preset}_dip"
    eval_full_diag \
      "$ROOT/${preset}/stage_b_dip_finetune/best_fk_pva.pt" \
      "$CACHE_ROOT/tc_test/ik2_q75_ctrl_cache_manifest.json" \
      "$EVAL_ROOT/${preset}_tc_test_full_diag.json" \
      TotalCapture-test \
      "ik2_q75_ctrl_v1_${preset}_dip"
  fi
done

echo "done: $ROOT"
