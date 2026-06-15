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

ROOT=${ROOT:-data/experiments/newik1_v10_official_protocol_last_control_20260607}
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

FEATURE_MODE=${FEATURE_MODE:-last_control}
SMOKE_ONLY=${SMOKE_ONLY:-0}
RUN_FULL_EVAL=${RUN_FULL_EVAL:-1}

ensure_control_cache() {
  local input_cache="$1"
  local output_dir="$2"
  local mode="$3"
  local pl_checkpoint="${4:-}"
  local max_sequences="${5:-0}"
  if [[ ! -f "$output_dir/newik1_control_cache_manifest.json" ]]; then
    local args=(
      newik1_control_cache.py
      --input-cache "$input_cache"
      --output-dir "$output_dir"
      --mode "$mode"
      --imu-input-mode official
      --feature-mode "$FEATURE_MODE"
      --tail-len 4
      --shard-size 100
    )
    if [[ "$mode" == "pl1_streaming" ]]; then
      args+=(--pl-checkpoint "$pl_checkpoint")
    fi
    if [[ "$max_sequences" != "0" ]]; then
      args+=(--max-sequences "$max_sequences")
    fi
    "$PY" "${args[@]}"
  fi
}

train_ik1() {
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
      newik1_control_train.py
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
      --residual-scale 0.005
      --dropout 0.2
      --grad-clip 1.0
      --weight-decay 0.0001
      --early-stop-min-delta 0.00000005
      --early-stop-patience 10
      --pRJ-weight 2.0
      --gR2-weight 1.0
      --pRJ-dot-weight 0.01
      --pRJ-ddot-weight 0.0003
      --gR2-dot-weight 0.03
      --gR2-ddot-weight 0.001
      --control-pRJ-weight 0.1
      --control-gR2-weight 0.1
      --control-pRJ-dot-weight 0.0
      --control-gR2-dot-weight 0.0
      --control-pRJ-ddot-weight 0.0
      --control-gR2-ddot-weight 0.0
      --bone-length-weight 0.0
      --control-point-prior-weight 0.0
      --tail-update-prior-weight 0.0
    )
    if [[ -n "$init" ]]; then
      args+=(--init-checkpoint "$init")
    fi
    if [[ "$max_val_sequences" != "0" ]]; then
      args+=(--max-val-sequences "$max_val_sequences")
    fi
    "$PY" "${args[@]}"
  fi
}

eval_streaming() {
  local raw_cache="$1"
  local out_json="$2"
  local split="$3"
  local version="$4"
  local pl_checkpoint="${5:-}"
  local ik1_checkpoint="${6:-}"
  local max_eval_sequences="${7:-0}"
  if [[ ! -f "$out_json" ]]; then
    local args=(
      newik1_real_streaming_audit.py
      --val-cache "$raw_cache"
      --output-json "$out_json"
      --split-label "$split"
      --version-name "$version"
      --imu-input-mode official
    )
    if [[ -n "$pl_checkpoint" ]]; then
      args+=(--pl-checkpoint "$pl_checkpoint")
    fi
    if [[ -n "$ik1_checkpoint" ]]; then
      args+=(--ik1-checkpoint "$ik1_checkpoint" --ik1-backend auto_control_point)
    else
      args+=(--ik1-backend original)
    fi
    if [[ "$max_eval_sequences" != "0" ]]; then
      args+=(--max-eval-sequences "$max_eval_sequences")
    fi
    "$PY" "${args[@]}"
  fi
}

# Fast end-to-end smoke: checks cache generation, train loop, checkpoint load,
# and real-streaming audit without consuming test split for selection.
ensure_control_cache "$DIP_VAL_RAW" "$CACHE_ROOT/smoke_teacher_forced_dip_val" teacher_forced "" 2
train_ik1 \
  "$CACHE_ROOT/smoke_teacher_forced_dip_val/newik1_control_cache_manifest.json" \
  "$CACHE_ROOT/smoke_teacher_forced_dip_val/newik1_control_cache_manifest.json" \
  "$ROOT/smoke_train" \
  newik1_v10_smoke \
  1 1e-5 1e-7 2
eval_streaming "$DIP_VAL_RAW" "$EVAL_ROOT/smoke_dip_val.json" DIP-val-smoke newik1_v10_smoke "$NEWPL_DIP" "$ROOT/smoke_train/best_loss.pt" 1

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo '{"status":"smoke_ok"}'
  exit 0
fi

# Official-like NewIK1 route:
# A: clean AMASS teacher-forced pretrain.
# B: AMASS PL-streaming adaptation using the AMASS-pretrained NewPL.
# C: DIP train PL-streaming fine-tune using the DIP-finetuned NewPL.
ensure_control_cache "$AMASS_RAW" "$CACHE_ROOT/teacher_forced_amass_last_control" teacher_forced
ensure_control_cache "$AMASS_RAW" "$CACHE_ROOT/pl_streaming_amass_last_control_newpl_v5_amass" pl1_streaming "$NEWPL_AMASS"
ensure_control_cache "$DIP_TRAIN_RAW" "$CACHE_ROOT/pl_streaming_dip_train_last_control_newpl_v5_dip" pl1_streaming "$NEWPL_DIP"
ensure_control_cache "$DIP_VAL_RAW" "$CACHE_ROOT/pl_streaming_dip_val_last_control_newpl_v5_dip" pl1_streaming "$NEWPL_DIP"

train_ik1 \
  "$CACHE_ROOT/teacher_forced_amass_last_control/newik1_control_cache_manifest.json" \
  "$CACHE_ROOT/teacher_forced_amass_last_control/newik1_control_cache_manifest.json" \
  "$ROOT/stage_a_amass_teacher_forced" \
  newik1_v10_stage_a_amass_teacher_forced \
  50 1e-4 1e-6 128 "" 50

train_ik1 \
  "$CACHE_ROOT/pl_streaming_amass_last_control_newpl_v5_amass/newik1_control_cache_manifest.json" \
  "$CACHE_ROOT/pl_streaming_amass_last_control_newpl_v5_amass/newik1_control_cache_manifest.json" \
  "$ROOT/stage_b_amass_pl_streaming" \
  newik1_v10_stage_b_amass_pl_streaming \
  20 2e-5 2e-7 128 \
  "$ROOT/stage_a_amass_teacher_forced/best_loss.pt" 50

train_ik1 \
  "$CACHE_ROOT/pl_streaming_dip_train_last_control_newpl_v5_dip/newik1_control_cache_manifest.json" \
  "$CACHE_ROOT/pl_streaming_dip_val_last_control_newpl_v5_dip/newik1_control_cache_manifest.json" \
  "$ROOT/stage_c_dip_pl_streaming" \
  newik1_v10_stage_c_dip_pl_streaming \
  40 5e-6 5e-8 12 \
  "$ROOT/stage_b_amass_pl_streaming/best_loss.pt"

if [[ "$RUN_FULL_EVAL" == "1" ]]; then
  eval_streaming "$DIP_TEST_RAW" "$EVAL_ROOT/dip_official_gpnet.json" DIP-IMU-test official_gpnet "" ""
  eval_streaming "$TC_TEST_RAW" "$EVAL_ROOT/tc_official_gpnet.json" TotalCapture-test official_gpnet "" ""

  eval_streaming "$DIP_TEST_RAW" "$EVAL_ROOT/dip_newpl_v5_amass_official_ik1.json" DIP-IMU-test newpl_v5_amass_official_ik1 "$NEWPL_AMASS" ""
  eval_streaming "$TC_TEST_RAW" "$EVAL_ROOT/tc_newpl_v5_amass_official_ik1.json" TotalCapture-test newpl_v5_amass_official_ik1 "$NEWPL_AMASS" ""
  eval_streaming "$DIP_TEST_RAW" "$EVAL_ROOT/dip_stage_a_best.json" DIP-IMU-test newik1_v10_stage_a_best "$NEWPL_AMASS" "$ROOT/stage_a_amass_teacher_forced/best_loss.pt"
  eval_streaming "$TC_TEST_RAW" "$EVAL_ROOT/tc_stage_a_best.json" TotalCapture-test newik1_v10_stage_a_best "$NEWPL_AMASS" "$ROOT/stage_a_amass_teacher_forced/best_loss.pt"
  eval_streaming "$DIP_TEST_RAW" "$EVAL_ROOT/dip_stage_b_best.json" DIP-IMU-test newik1_v10_stage_b_best "$NEWPL_AMASS" "$ROOT/stage_b_amass_pl_streaming/best_loss.pt"
  eval_streaming "$TC_TEST_RAW" "$EVAL_ROOT/tc_stage_b_best.json" TotalCapture-test newik1_v10_stage_b_best "$NEWPL_AMASS" "$ROOT/stage_b_amass_pl_streaming/best_loss.pt"

  eval_streaming "$DIP_TEST_RAW" "$EVAL_ROOT/dip_newpl_v5_dip_official_ik1.json" DIP-IMU-test newpl_v5_dip_official_ik1 "$NEWPL_DIP" ""
  eval_streaming "$TC_TEST_RAW" "$EVAL_ROOT/tc_newpl_v5_dip_official_ik1.json" TotalCapture-test newpl_v5_dip_official_ik1 "$NEWPL_DIP" ""
  eval_streaming "$DIP_TEST_RAW" "$EVAL_ROOT/dip_stage_c_best.json" DIP-IMU-test newik1_v10_stage_c_best "$NEWPL_DIP" "$ROOT/stage_c_dip_pl_streaming/best_loss.pt"
  eval_streaming "$TC_TEST_RAW" "$EVAL_ROOT/tc_stage_c_best.json" TotalCapture-test newik1_v10_stage_c_best "$NEWPL_DIP" "$ROOT/stage_c_dip_pl_streaming/best_loss.pt"
  eval_streaming "$DIP_TEST_RAW" "$EVAL_ROOT/dip_stage_c_last.json" DIP-IMU-test newik1_v10_stage_c_last "$NEWPL_DIP" "$ROOT/stage_c_dip_pl_streaming/last.pt"
  eval_streaming "$TC_TEST_RAW" "$EVAL_ROOT/tc_stage_c_last.json" TotalCapture-test newik1_v10_stage_c_last "$NEWPL_DIP" "$ROOT/stage_c_dip_pl_streaming/last.pt"
fi

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {
    "status": "ok",
    "root": str(root),
    "protocol": "NewIK1 last-control official-like route: AMASS teacher-forced -> AMASS PL-streaming -> DIP PL-streaming; eval DIP test and TotalCapture test.",
    "feature_mode": "last_control",
    "dip_trans_loss_used": False,
    "totalcapture_train_used": False,
    "checkpoints": {
        "stage_a_best": str(root / "stage_a_amass_teacher_forced" / "best_loss.pt"),
        "stage_b_best": str(root / "stage_b_amass_pl_streaming" / "best_loss.pt"),
        "stage_c_best": str(root / "stage_c_dip_pl_streaming" / "best_loss.pt"),
        "stage_c_last": str(root / "stage_c_dip_pl_streaming" / "last.pt"),
    },
    "evals": {},
}
for p in sorted((root / "eval").glob("*.json")):
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        summary["evals"][p.name] = {"status": "parse_failed", "error": str(exc)}
        continue
    summary["evals"][p.name] = {
        "status": data.get("status"),
        "version_name": data.get("version_name"),
        "split_label": data.get("split_label"),
        "score": data.get("score"),
        "all_finite": data.get("all_finite"),
        "ik1_module_aggregate": data.get("ik1_module_aggregate"),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "# NewIK1 v10 Official-Protocol Last-Control Summary",
    "",
    f"Protocol: {summary['protocol']}",
    "",
    f"DIP trans loss used: {summary['dip_trans_loss_used']}.",
    f"TotalCapture train used: {summary['totalcapture_train_used']}.",
    "",
]
for name, item in summary["evals"].items():
    lines.append(f"- {name}: status={item.get('status')}, version={item.get('version_name')}, split={item.get('split_label')}, score={item.get('score')}")
(root / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"status": "ok", "summary": str(root / "summary.json")}, indent=2))
PY
