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

ROOT=${ROOT:-data/experiments/newpose_ctrl_v1_20260608}
CACHE_ROOT="$ROOT/caches"
EVAL_ROOT="$ROOT/eval"
LOG_ROOT="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

AMASS_RAW=${AMASS_RAW:-data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json}
DIP_TRAIN_RAW=${DIP_TRAIN_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json}
DIP_VAL_RAW=${DIP_VAL_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json}
DIP_TEST_RAW=${DIP_TEST_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json}
TC_TEST_RAW=${TC_TEST_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json}

NEWPL_AMASS=${NEWPL_AMASS:-data/experiments/newpl_v5_official_protocol_20260607_tuned/amass_pretrain/best_loss.pt}
NEWPL_DIP=${NEWPL_DIP:-data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt}

SMOKE_ONLY=${SMOKE_ONLY:-0}
RUN_MODULE_EVAL=${RUN_MODULE_EVAL:-1}
RUN_FULL_EVAL=${RUN_FULL_EVAL:-0}
MODULE_EVAL_MAX_SEQUENCES=${MODULE_EVAL_MAX_SEQUENCES:-0}
FULL_EVAL_MAX_SEQUENCES=${FULL_EVAL_MAX_SEQUENCES:-0}
EVAL_MODULE_ROOT="$ROOT/eval_module"
mkdir -p "$EVAL_MODULE_ROOT"

ensure_newpose_cache() {
  local input_cache="$1"
  local output_dir="$2"
  local pl_checkpoint="$3"
  local max_sequences="${4:-0}"
  if [[ ! -f "$output_dir/newpose_ctrl_cache_manifest.json" ]]; then
    local args=(
      newpose_ctrl_cache.py
      --input-cache "$input_cache"
      --output-dir "$output_dir"
      --pl-checkpoint "$pl_checkpoint"
      --imu-input-mode official
      --shard-size 25
    )
    if [[ "$max_sequences" != "0" ]]; then
      args+=(--max-sequences "$max_sequences")
    fi
    "$PY" "${args[@]}"
  fi
}

train_newpose() {
  local train_cache="$1"
  local val_cache="$2"
  local out_dir="$3"
  local name="$4"
  local epochs="$5"
  local lr="$6"
  local min_lr="$7"
  local batch="$8"
  local init="${9:-}"
  local max_val_sequences="${10:-0}"
  if [[ ! -f "$out_dir/train_result.json" ]]; then
    local args=(
      newpose_ctrl_train.py
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
      --tail-length 4
      --residual-scale 0.1
      --dropout 0.2
      --offset-init-scale 0.1
      --grad-clip 1.0
      --weight-decay 0.0001
      --early-stop-min-delta 0.00000005
      --early-stop-patience 10
      --selection-metric control_pose_physical
      --state-geodesic-weight 0.2
      --parent-relative-weight 0.0
      --distill-ik2-weight 0.0
      --control-point-prior-weight 0.0
      --tail-update-prior-weight 0.0
    )
    if [[ -n "$init" ]]; then
      args+=(--init-checkpoint "$init")
    fi
    if [[ "$max_val_sequences" != "0" ]]; then
      args+=(--max-val-sequences "$max_val_sequences" --val-window 300)
    fi
    "$PY" "${args[@]}"
  fi
}

eval_newpose() {
  local checkpoint="$1"
  local cache="$2"
  local out_json="$3"
  local split="$4"
  local version="$5"
  local max_eval_sequences="${6:-0}"
  if [[ ! -f "$out_json" ]]; then
    local args=(
      newpose_ctrl_eval.py
      --checkpoint "$checkpoint"
      --newpose-cache "$cache"
      --output-json "$out_json"
      --split-label "$split"
      --version-name "$version"
    )
    if [[ "$max_eval_sequences" != "0" ]]; then
      args+=(--max-eval-sequences "$max_eval_sequences")
    fi
    "$PY" "${args[@]}"
  fi
}

eval_newpose_module() {
  local checkpoint="$1"
  local cache="$2"
  local out_json="$3"
  local split="$4"
  local version="$5"
  if [[ ! -f "$out_json" ]]; then
    local args=(
      newpose_ctrl_eval.py
      --checkpoint "$checkpoint"
      --newpose-cache "$cache"
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

eval_pl_baseline() {
  local raw_cache="$1"
  local out_json="$2"
  local checkpoint="${3:-}"
  if [[ ! -f "$out_json" ]]; then
    local args=(
      pl_curve_eval.py
      --val-cache "$raw_cache"
      --output-json "$out_json"
      --imu-input-mode official
      --skip-baseline-rerun
    )
    if [[ -n "$checkpoint" ]]; then
      args+=(--checkpoint "$checkpoint")
    fi
    if [[ "$FULL_EVAL_MAX_SEQUENCES" != "0" ]]; then
      args+=(--max-eval-sequences "$FULL_EVAL_MAX_SEQUENCES")
    fi
    "$PY" "${args[@]}"
  fi
}

eval_baseline_module() {
  local raw_cache="$1"
  local out_json="$2"
  local version="$3"
  local split="$4"
  local checkpoint="${5:-}"
  if [[ ! -f "$out_json" ]]; then
    local args=(
      newpose_baseline_ik2_module_eval.py
      --val-cache "$raw_cache"
      --output-json "$out_json"
      --version-name "$version"
      --split-label "$split"
      --imu-input-mode official
    )
    if [[ -n "$checkpoint" ]]; then
      args+=(--checkpoint "$checkpoint")
    fi
    if [[ "$MODULE_EVAL_MAX_SEQUENCES" != "0" ]]; then
      args+=(--max-eval-sequences "$MODULE_EVAL_MAX_SEQUENCES")
    fi
    "$PY" "${args[@]}"
  fi
}

# Smoke: cache, train, module eval, and short full-pipeline eval.
ensure_newpose_cache "$DIP_VAL_RAW" "$CACHE_ROOT/smoke_dip_val_newpl_dip" "$NEWPL_DIP" 1
train_newpose \
  "$CACHE_ROOT/smoke_dip_val_newpl_dip/newpose_ctrl_cache_manifest.json" \
  "$CACHE_ROOT/smoke_dip_val_newpl_dip/newpose_ctrl_cache_manifest.json" \
  "$ROOT/smoke_train" \
  newpose_ctrl_v1_smoke \
  1 5e-5 1e-6 1 "" 1
if [[ ! -f "$EVAL_ROOT/smoke_module.json" ]]; then
  "$PY" newpose_ctrl_eval.py \
    --checkpoint "$ROOT/smoke_train/best_loss.pt" \
    --newpose-cache "$CACHE_ROOT/smoke_dip_val_newpl_dip/newpose_ctrl_cache_manifest.json" \
    --output-json "$EVAL_ROOT/smoke_module.json" \
    --split-label DIP-val-smoke \
    --version-name newpose_ctrl_v1_smoke \
    --module-only \
    --max-eval-sequences 1 \
    --max-smoke-frames 120
fi
if [[ ! -f "$EVAL_ROOT/smoke_full_10f.json" ]]; then
  "$PY" newpose_ctrl_eval.py \
    --checkpoint "$ROOT/smoke_train/best_loss.pt" \
    --newpose-cache "$CACHE_ROOT/smoke_dip_val_newpl_dip/newpose_ctrl_cache_manifest.json" \
    --output-json "$EVAL_ROOT/smoke_full_10f.json" \
    --split-label DIP-val-smoke \
    --version-name newpose_ctrl_v1_smoke_full_10f \
    --max-eval-sequences 1 \
    --max-smoke-frames 10
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo '{"status":"smoke_ok"}'
  exit 0
fi

# Official-like route:
# A: AMASS pretrain using NewPL AMASS stream.
# B: DIP-IMU train fine-tune using NewPL DIP stream.
# Eval: DIP test and TotalCapture test, with both NewPL-AMASS and NewPL-DIP upstream caches where needed.
ensure_newpose_cache "$AMASS_RAW" "$CACHE_ROOT/amass_train_newpl_amass" "$NEWPL_AMASS"
ensure_newpose_cache "$DIP_TRAIN_RAW" "$CACHE_ROOT/dip_train_newpl_dip" "$NEWPL_DIP"
ensure_newpose_cache "$DIP_VAL_RAW" "$CACHE_ROOT/dip_val_newpl_dip" "$NEWPL_DIP"
ensure_newpose_cache "$DIP_TEST_RAW" "$CACHE_ROOT/dip_test_newpl_amass" "$NEWPL_AMASS"
ensure_newpose_cache "$TC_TEST_RAW" "$CACHE_ROOT/tc_test_newpl_amass" "$NEWPL_AMASS"
ensure_newpose_cache "$DIP_TEST_RAW" "$CACHE_ROOT/dip_test_newpl_dip" "$NEWPL_DIP"
ensure_newpose_cache "$TC_TEST_RAW" "$CACHE_ROOT/tc_test_newpl_dip" "$NEWPL_DIP"

train_newpose \
  "$CACHE_ROOT/amass_train_newpl_amass/newpose_ctrl_cache_manifest.json" \
  "$CACHE_ROOT/amass_train_newpl_amass/newpose_ctrl_cache_manifest.json" \
  "$ROOT/stage_a_amass_pretrain" \
  newpose_ctrl_v1_stage_a_amass_pretrain \
  50 1e-4 1e-6 128 "" 50

train_newpose \
  "$CACHE_ROOT/dip_train_newpl_dip/newpose_ctrl_cache_manifest.json" \
  "$CACHE_ROOT/dip_val_newpl_dip/newpose_ctrl_cache_manifest.json" \
  "$ROOT/stage_b_dip_finetune" \
  newpose_ctrl_v1_stage_b_dip_finetune \
  40 2e-5 2e-7 12 \
  "$ROOT/stage_a_amass_pretrain/best_loss.pt" \
  6

if [[ "$RUN_MODULE_EVAL" == "1" ]]; then
  eval_baseline_module "$DIP_TEST_RAW" "$EVAL_MODULE_ROOT/dip_official_gpnet_module.json" official_gpnet DIP-IMU-test
  eval_baseline_module "$TC_TEST_RAW" "$EVAL_MODULE_ROOT/tc_official_gpnet_module.json" official_gpnet TotalCapture-test
  eval_baseline_module "$DIP_TEST_RAW" "$EVAL_MODULE_ROOT/dip_newpl_v5_amass_official_ik2_module.json" newpl_v5_amass_official_ik2 DIP-IMU-test "$NEWPL_AMASS"
  eval_baseline_module "$TC_TEST_RAW" "$EVAL_MODULE_ROOT/tc_newpl_v5_amass_official_ik2_module.json" newpl_v5_amass_official_ik2 TotalCapture-test "$NEWPL_AMASS"
  eval_baseline_module "$DIP_TEST_RAW" "$EVAL_MODULE_ROOT/dip_newpl_v5_dip_official_ik2_module.json" newpl_v5_dip_official_ik2 DIP-IMU-test "$NEWPL_DIP"
  eval_baseline_module "$TC_TEST_RAW" "$EVAL_MODULE_ROOT/tc_newpl_v5_dip_official_ik2_module.json" newpl_v5_dip_official_ik2 TotalCapture-test "$NEWPL_DIP"

  eval_newpose_module "$ROOT/stage_a_amass_pretrain/best_loss.pt" "$CACHE_ROOT/dip_test_newpl_amass/newpose_ctrl_cache_manifest.json" "$EVAL_MODULE_ROOT/dip_stage_a_best_module.json" DIP-IMU-test newpose_ctrl_v1_stage_a_best
  eval_newpose_module "$ROOT/stage_a_amass_pretrain/best_loss.pt" "$CACHE_ROOT/tc_test_newpl_amass/newpose_ctrl_cache_manifest.json" "$EVAL_MODULE_ROOT/tc_stage_a_best_module.json" TotalCapture-test newpose_ctrl_v1_stage_a_best
  eval_newpose_module "$ROOT/stage_b_dip_finetune/best_loss.pt" "$CACHE_ROOT/dip_test_newpl_dip/newpose_ctrl_cache_manifest.json" "$EVAL_MODULE_ROOT/dip_stage_b_best_module.json" DIP-IMU-test newpose_ctrl_v1_stage_b_best
  eval_newpose_module "$ROOT/stage_b_dip_finetune/best_loss.pt" "$CACHE_ROOT/tc_test_newpl_dip/newpose_ctrl_cache_manifest.json" "$EVAL_MODULE_ROOT/tc_stage_b_best_module.json" TotalCapture-test newpose_ctrl_v1_stage_b_best
  eval_newpose_module "$ROOT/stage_b_dip_finetune/last.pt" "$CACHE_ROOT/dip_test_newpl_dip/newpose_ctrl_cache_manifest.json" "$EVAL_MODULE_ROOT/dip_stage_b_last_module.json" DIP-IMU-test newpose_ctrl_v1_stage_b_last
  eval_newpose_module "$ROOT/stage_b_dip_finetune/last.pt" "$CACHE_ROOT/tc_test_newpl_dip/newpose_ctrl_cache_manifest.json" "$EVAL_MODULE_ROOT/tc_stage_b_last_module.json" TotalCapture-test newpose_ctrl_v1_stage_b_last
fi

if [[ "$RUN_FULL_EVAL" == "1" ]]; then
  eval_pl_baseline "$DIP_TEST_RAW" "$EVAL_ROOT/dip_official_gpnet.json" ""
  eval_pl_baseline "$TC_TEST_RAW" "$EVAL_ROOT/tc_official_gpnet.json" ""
  eval_pl_baseline "$DIP_TEST_RAW" "$EVAL_ROOT/dip_newpl_v5_amass_official_ik2.json" "$NEWPL_AMASS"
  eval_pl_baseline "$TC_TEST_RAW" "$EVAL_ROOT/tc_newpl_v5_amass_official_ik2.json" "$NEWPL_AMASS"
  eval_pl_baseline "$DIP_TEST_RAW" "$EVAL_ROOT/dip_newpl_v5_dip_official_ik2.json" "$NEWPL_DIP"
  eval_pl_baseline "$TC_TEST_RAW" "$EVAL_ROOT/tc_newpl_v5_dip_official_ik2.json" "$NEWPL_DIP"

  eval_newpose "$ROOT/stage_a_amass_pretrain/best_loss.pt" "$CACHE_ROOT/dip_test_newpl_amass/newpose_ctrl_cache_manifest.json" "$EVAL_ROOT/dip_stage_a_best.json" DIP-IMU-test newpose_ctrl_v1_stage_a_best "$FULL_EVAL_MAX_SEQUENCES"
  eval_newpose "$ROOT/stage_a_amass_pretrain/best_loss.pt" "$CACHE_ROOT/tc_test_newpl_amass/newpose_ctrl_cache_manifest.json" "$EVAL_ROOT/tc_stage_a_best.json" TotalCapture-test newpose_ctrl_v1_stage_a_best "$FULL_EVAL_MAX_SEQUENCES"
  eval_newpose "$ROOT/stage_b_dip_finetune/best_loss.pt" "$CACHE_ROOT/dip_test_newpl_dip/newpose_ctrl_cache_manifest.json" "$EVAL_ROOT/dip_stage_b_best.json" DIP-IMU-test newpose_ctrl_v1_stage_b_best "$FULL_EVAL_MAX_SEQUENCES"
  eval_newpose "$ROOT/stage_b_dip_finetune/best_loss.pt" "$CACHE_ROOT/tc_test_newpl_dip/newpose_ctrl_cache_manifest.json" "$EVAL_ROOT/tc_stage_b_best.json" TotalCapture-test newpose_ctrl_v1_stage_b_best "$FULL_EVAL_MAX_SEQUENCES"
  eval_newpose "$ROOT/stage_b_dip_finetune/last.pt" "$CACHE_ROOT/dip_test_newpl_dip/newpose_ctrl_cache_manifest.json" "$EVAL_ROOT/dip_stage_b_last.json" DIP-IMU-test newpose_ctrl_v1_stage_b_last "$FULL_EVAL_MAX_SEQUENCES"
  eval_newpose "$ROOT/stage_b_dip_finetune/last.pt" "$CACHE_ROOT/tc_test_newpl_dip/newpose_ctrl_cache_manifest.json" "$EVAL_ROOT/tc_stage_b_last.json" TotalCapture-test newpose_ctrl_v1_stage_b_last "$FULL_EVAL_MAX_SEQUENCES"
fi

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("data/experiments/newpose_ctrl_v1_20260608")
eval_root = root / "eval"
evals = {}
for path in sorted(eval_root.glob("*.json")):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        data = {"status": "failed_to_read", "error": str(exc)}
    evals[path.name] = {
        "status": data.get("status"),
        "version_name": data.get("version_name"),
        "split_label": data.get("split_label"),
        "score": data.get("score"),
        "all_finite": data.get("all_finite"),
        "module_aggregate": data.get("module_aggregate"),
    }
summary = {
    "status": "ok",
    "root": str(root),
    "protocol": "newpose_ctrl_v1: NewPL stream -> pose control points; AMASS pretrain -> DIP fine-tune -> DIP/TC eval.",
    "frame_input": "official IMU[90]+RRB_after_pl[45]+pRB/gR1[18]+last PL control[18]+gR0[3] = 174D; offset_r/r_JS is init-only.",
    "output": "RRJ_control[90]+gR_pose_control[3] = 93D pose-control state.",
    "checkpoints": {
        "stage_a_best": str(root / "stage_a_amass_pretrain" / "best_loss.pt"),
        "stage_b_best": str(root / "stage_b_dip_finetune" / "best_loss.pt"),
        "stage_b_last": str(root / "stage_b_dip_finetune" / "last.pt"),
    },
    "evals": evals,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"status": "ok", "summary": str(root / "summary.json")}, indent=2))
PY
