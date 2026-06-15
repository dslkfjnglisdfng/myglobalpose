#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ENV_DIR:-}" ]]; then
  if [[ -x /home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python ]]; then
    ENV_DIR=/home/lingfeng/remote-envs/globalpose-gpu-py310
  else
    ENV_DIR=/home/lingfeng/.conda/envs/globalpose-gpu
  fi
fi
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
PY="$ENV_DIR/bin/python"

ROOT=${ROOT:-data/experiments/newik1_v11_canonical_control_20260609}
CACHE_ROOT="$ROOT/caches"
EVAL_DIR="$ROOT/eval"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE_ROOT" "$EVAL_DIR" "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

SMOKE=${SMOKE:-0}
AMASS_BATCH=${AMASS_BATCH:-128}
DIP_BATCH=${DIP_BATCH:-32}
FEATURE_MODE=${FEATURE_MODE:-last_control}
AMASS_STREAM_MAX_SEQS=${AMASS_STREAM_MAX_SEQS:-50}
REUSE_TEACHER_FORCED_AMASS_CACHE=${REUSE_TEACHER_FORCED_AMASS_CACHE:-data/experiments/newik1_v10_official_protocol_last_control_20260607/caches/teacher_forced_amass_last_control/newik1_control_cache_manifest.json}

AMASS_RAW=${AMASS_RAW:-data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json}
DIP_TRAIN_RAW=${DIP_TRAIN_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json}
DIP_VAL_RAW=${DIP_VAL_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json}
DIP_TEST_RAW=${DIP_TEST_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json}
TC_TEST_RAW=${TC_TEST_RAW:-data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json}

CANONICAL_AMASS=${CANONICAL_AMASS:-data/experiments/newpl_diff_vs_control_ablation_20260608/canonical_control/amass_pretrain/best_loss.pt}
CANONICAL_DIP=${CANONICAL_DIP:-data/experiments/newpl_diff_vs_control_ablation_20260608/canonical_control/dip_finetune/best_loss.pt}
NEWIK1_V10_STAGE_C=${NEWIK1_V10_STAGE_C:-data/experiments/newik1_v10_official_protocol_last_control_20260607/stage_c_dip_pl_streaming/best_loss.pt}

echo "started $(date --iso-8601=seconds)"
echo "ROOT=$ROOT SMOKE=$SMOKE CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "AMASS_BATCH=$AMASS_BATCH DIP_BATCH=$DIP_BATCH FEATURE_MODE=$FEATURE_MODE"
echo "CANONICAL_AMASS=$CANONICAL_AMASS"
echo "CANONICAL_DIP=$CANONICAL_DIP"
echo "AMASS_STREAM_MAX_SEQS=$AMASS_STREAM_MAX_SEQS"
echo "REUSE_TEACHER_FORCED_AMASS_CACHE=$REUSE_TEACHER_FORCED_AMASS_CACHE"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "mem", torch.cuda.get_device_properties(0).total_memory)
PY

if [[ "$SMOKE" == "1" ]]; then
  CACHE_MAX_SEQS=8
  STAGE_A_EPOCHS=1
  STAGE_B_EPOCHS=1
  STAGE_C_EPOCHS=1
  AMASS_MAX_VAL=4
  EVAL_MAX_SEQS=3
else
  CACHE_MAX_SEQS=0
  STAGE_A_EPOCHS=${STAGE_A_EPOCHS:-50}
  STAGE_B_EPOCHS=${STAGE_B_EPOCHS:-20}
  STAGE_C_EPOCHS=${STAGE_C_EPOCHS:-40}
  AMASS_MAX_VAL=${AMASS_MAX_VAL:-50}
  EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-20}
fi

ensure_control_cache() {
  local input_cache="$1"
  local output_dir="$2"
  local mode="$3"
  local pl_checkpoint="${4:-}"
  local max_sequences="${5:-$CACHE_MAX_SEQS}"
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
  local max_val="${10:-0}"
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
      --val-window-length 61
      --selection-metric ik1_control_physical
      --hidden-size 512
      --tail-length 4
      --residual-scale 0.005
      --dropout 0.2
      --grad-clip 1.0
      --weight-decay 0.0001
      --early-stop-min-delta 0.00000005
      --early-stop-patience 10
      --pRJ-weight 2.0
      --leaf-pRJ-weight 1.0
      --gR2-weight 1.0
      --pRJ-dot-weight 0.0
      --pRJ-ddot-weight 0.0
      --gR2-dot-weight 0.0
      --gR2-ddot-weight 0.0
      --control-pRJ-weight 0.2
      --control-gR2-weight 0.2
      --control-pRJ-dot-weight 0.0
      --control-gR2-dot-weight 0.0
      --control-pRJ-ddot-weight 0.0
      --control-gR2-ddot-weight 0.0
      --gt-control-pRJ-weight 0.3
      --gt-control-gR2-weight 0.1
      --gt-control-leaf-pRJ-weight 0.2
      --bone-length-weight 0.1
      --control-point-prior-weight 0.05
      --tail-update-prior-weight 0.001
    )
    if [[ -n "$init" ]]; then
      args+=(--init-checkpoint "$init")
    fi
    if [[ "$max_val" != "0" ]]; then
      args+=(--max-val-sequences "$max_val")
    fi
    "$PY" "${args[@]}"
  fi
}

run_local_diag() {
  local version="${1:-}"
  local ckpt="${2:-}"
  local dataset="${3:-}"
  local cache="${4:-}"
  if [[ -z "$version" || -z "$ckpt" || -z "$dataset" || -z "$cache" ]]; then
    echo "skip local diag with incomplete args: version=${version:-unset} dataset=${dataset:-unset}"
    return 0
  fi
  local out="$EVAL_DIR/${version}_${dataset}_local_diag.json"
  if [[ -f "$ckpt" && ! -f "$out" ]]; then
    "$PY" newik1_local_diagnostic.py \
      --cache "$cache" \
      --ik1-checkpoint "$ckpt" \
      --output-json "$out" \
      --max-sequences "$EVAL_MAX_SEQS"
  fi
}

if [[ -f "$REUSE_TEACHER_FORCED_AMASS_CACHE" ]]; then
  echo "Reusing teacher-forced AMASS IK1 cache: $REUSE_TEACHER_FORCED_AMASS_CACHE"
  TEACHER_FORCED_AMASS_CACHE="$REUSE_TEACHER_FORCED_AMASS_CACHE"
else
  ensure_control_cache "$AMASS_RAW" "$CACHE_ROOT/teacher_forced_amass_last_control" teacher_forced "" "$CACHE_MAX_SEQS"
  TEACHER_FORCED_AMASS_CACHE="$CACHE_ROOT/teacher_forced_amass_last_control/newik1_control_cache_manifest.json"
fi
ensure_control_cache "$AMASS_RAW" "$CACHE_ROOT/pl_streaming_amass_last_control_canonical_amass" pl1_streaming "$CANONICAL_AMASS" "$AMASS_STREAM_MAX_SEQS"
ensure_control_cache "$DIP_TRAIN_RAW" "$CACHE_ROOT/pl_streaming_dip_train_last_control_canonical_dip" pl1_streaming "$CANONICAL_DIP" "$CACHE_MAX_SEQS"
ensure_control_cache "$DIP_VAL_RAW" "$CACHE_ROOT/pl_streaming_dip_val_last_control_canonical_dip" pl1_streaming "$CANONICAL_DIP" "$CACHE_MAX_SEQS"
ensure_control_cache "$DIP_TEST_RAW" "$CACHE_ROOT/pl_streaming_dip_test_last_control_canonical_dip" pl1_streaming "$CANONICAL_DIP" "$CACHE_MAX_SEQS"
ensure_control_cache "$TC_TEST_RAW" "$CACHE_ROOT/pl_streaming_tc_test_last_control_canonical_dip" pl1_streaming "$CANONICAL_DIP" "$CACHE_MAX_SEQS"

train_ik1 \
  "$TEACHER_FORCED_AMASS_CACHE" \
  "$TEACHER_FORCED_AMASS_CACHE" \
  "$ROOT/stage_a_amass_teacher_forced" \
  newik1_v11_stage_a_amass_teacher_forced \
  "$STAGE_A_EPOCHS" 1e-4 1e-6 "$AMASS_BATCH" "" "$AMASS_MAX_VAL"

train_ik1 \
  "$CACHE_ROOT/pl_streaming_amass_last_control_canonical_amass/newik1_control_cache_manifest.json" \
  "$CACHE_ROOT/pl_streaming_amass_last_control_canonical_amass/newik1_control_cache_manifest.json" \
  "$ROOT/stage_b_amass_pl_streaming" \
  newik1_v11_stage_b_amass_pl_streaming \
  "$STAGE_B_EPOCHS" 2e-5 2e-7 "$AMASS_BATCH" \
  "$ROOT/stage_a_amass_teacher_forced/best_loss.pt" "$AMASS_MAX_VAL"

train_ik1 \
  "$CACHE_ROOT/pl_streaming_dip_train_last_control_canonical_dip/newik1_control_cache_manifest.json" \
  "$CACHE_ROOT/pl_streaming_dip_val_last_control_canonical_dip/newik1_control_cache_manifest.json" \
  "$ROOT/stage_c_dip_pl_streaming" \
  newik1_v11_stage_c_dip_pl_streaming \
  "$STAGE_C_EPOCHS" 5e-6 5e-8 "$DIP_BATCH" \
  "$ROOT/stage_b_amass_pl_streaming/best_loss.pt" 0

for dataset in amass dip_val dip_test tc_test; do
  case "$dataset" in
    amass) cache="$CACHE_ROOT/pl_streaming_amass_last_control_canonical_amass/newik1_control_cache_manifest.json" ;;
    dip_val) cache="$CACHE_ROOT/pl_streaming_dip_val_last_control_canonical_dip/newik1_control_cache_manifest.json" ;;
    dip_test) cache="$CACHE_ROOT/pl_streaming_dip_test_last_control_canonical_dip/newik1_control_cache_manifest.json" ;;
    tc_test) cache="$CACHE_ROOT/pl_streaming_tc_test_last_control_canonical_dip/newik1_control_cache_manifest.json" ;;
  esac
  run_local_diag newik1_v11_best "$ROOT/stage_c_dip_pl_streaming/best_loss.pt" "$dataset" "$cache"
  run_local_diag newik1_v11_last "$ROOT/stage_c_dip_pl_streaming/last.pt" "$dataset" "$cache"
  run_local_diag newik1_v10_stage_c_best "$NEWIK1_V10_STAGE_C" "$dataset" "$cache"
done

"$PY" scripts/summarize_newik1_v11.py --root "$ROOT"
echo "finished $(date --iso-8601=seconds)"
