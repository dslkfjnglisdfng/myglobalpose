#!/usr/bin/env bash
set -euo pipefail
ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PL_CKPT=data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/best_loss.pt
ROOT=data/experiments/newik1_official_input_20260604
LOGDIR=logs/newik1_official_input_20260604
mkdir -p "$ROOT/caches" "$LOGDIR"
run_step() {
  local name="$1"
  local output="$2"
  shift 2
  if [ -e "$output" ]; then
    echo "$(date -Is) SKIP $name output_exists=$output"
    return 0
  fi
  echo "$(date -Is) START $name"
  "$@" 2>&1 | tee "$LOGDIR/${name}.log"
  echo "$(date -Is) DONE $name"
}
run_step cache_teacher_forced_amass "$ROOT/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json" \
  python newik1_official_input_cache.py \
    --input-cache data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged/baseline_cache_manifest.json \
    --output-dir "$ROOT/caches/teacher_forced_amass" \
    --mode teacher_forced --imu-input-mode auto --shard-size 50
run_step cache_teacher_forced_tc_train "$ROOT/caches/teacher_forced_tc_train/newik1_official_input_cache_manifest.json" \
  python newik1_official_input_cache.py \
    --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json \
    --output-dir "$ROOT/caches/teacher_forced_tc_train" \
    --mode teacher_forced --imu-input-mode processed --shard-size 50
run_step cache_teacher_forced_tc_val "$ROOT/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json" \
  python newik1_official_input_cache.py \
    --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json \
    --output-dir "$ROOT/caches/teacher_forced_tc_val" \
    --mode teacher_forced --imu-input-mode processed --shard-size 50
run_step train_teacher_forced_amass "$ROOT/teacher_forced_amass/train_result.json" \
  python newik1_official_input_train.py \
    --train-cache "$ROOT/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json" \
    --val-cache "$ROOT/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json" \
    --output-dir "$ROOT/teacher_forced_amass" \
    --experiment-name newik1_official_input_teacher_forced_amass \
    --epochs 20 --lr 1e-4 --batch-size 16 --window 61 --max-val-sequences 5
run_step train_teacher_forced_tc_finetune "$ROOT/teacher_forced_tc_finetune/train_result.json" \
  python newik1_official_input_train.py \
    --train-cache "$ROOT/caches/teacher_forced_tc_train/newik1_official_input_cache_manifest.json" \
    --val-cache "$ROOT/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json" \
    --output-dir "$ROOT/teacher_forced_tc_finetune" \
    --experiment-name newik1_official_input_teacher_forced_tc_finetune \
    --epochs 10 --lr 3e-6 --batch-size 8 --window 61 \
    --init-checkpoint "$ROOT/teacher_forced_amass/best_loss.pt"
run_step cache_pl1_streaming_tc_train "$ROOT/caches/pl1_streaming_tc_train/newik1_official_input_cache_manifest.json" \
  python newik1_official_input_cache.py \
    --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json \
    --output-dir "$ROOT/caches/pl1_streaming_tc_train" \
    --mode pl1_streaming --imu-input-mode processed --pl-checkpoint "$PL_CKPT" --shard-size 50
run_step cache_pl1_streaming_tc_val "$ROOT/caches/pl1_streaming_tc_val/newik1_official_input_cache_manifest.json" \
  python newik1_official_input_cache.py \
    --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json \
    --output-dir "$ROOT/caches/pl1_streaming_tc_val" \
    --mode pl1_streaming --imu-input-mode processed --pl-checkpoint "$PL_CKPT" --shard-size 50
run_step train_pl1_streaming_tc_finetune "$ROOT/pl1_streaming_tc_finetune/train_result.json" \
  python newik1_official_input_train.py \
    --train-cache "$ROOT/caches/pl1_streaming_tc_train/newik1_official_input_cache_manifest.json" \
    --val-cache "$ROOT/caches/pl1_streaming_tc_val/newik1_official_input_cache_manifest.json" \
    --output-dir "$ROOT/pl1_streaming_tc_finetune" \
    --experiment-name newik1_official_input_pl1_streaming_tc_finetune \
    --epochs 10 --lr 3e-6 --batch-size 8 --window 61 \
    --init-checkpoint "$ROOT/teacher_forced_tc_finetune/best_loss.pt"
run_step eval_pl1_streaming_tc_val "$ROOT/eval_pl1_streaming_tc_val.json" \
  python newik1_official_input_eval.py \
    --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json \
    --output-json "$ROOT/eval_pl1_streaming_tc_val.json" \
    --pl-checkpoint "$PL_CKPT" \
    --ik1-checkpoint "$ROOT/pl1_streaming_tc_finetune/best_loss.pt" \
    --imu-input-mode processed
python - <<'PY'
import json
from pathlib import Path
root = Path("data/experiments/newik1_official_input_20260604")
summary = {"status": "ok"}
for rel in ["teacher_forced_amass/train_result.json", "teacher_forced_tc_finetune/train_result.json", "pl1_streaming_tc_finetune/train_result.json", "eval_pl1_streaming_tc_val.json"]:
    p = root / rel
    if p.exists():
        data = json.loads(p.read_text())
        summary[rel] = {k: data.get(k) for k in ("status", "best_loss", "best_epoch", "score", "all_finite", "error")}
print(json.dumps(summary, indent=2), flush=True)
PY
