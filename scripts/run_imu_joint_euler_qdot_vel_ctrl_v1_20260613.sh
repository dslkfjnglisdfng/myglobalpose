#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613}"
VARIANTS_RAW="${VARIANTS:-A_qctrl_main B_qdot_qddot_strong C_vel_acc_strong D_all_balanced}"

AMASS_CACHE="data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json"
TC_TRAIN_CACHE="data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json"
TC_VAL_CACHE="data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json"
TC_TEST_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json"
DIP_TRAIN_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json"
DIP_VAL_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json"
DIP_TEST_CACHE="data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json"

WINDOW="${WINDOW:-61}"
HIDDEN="${HIDDEN:-512}"
LAYERS="${LAYERS:-3}"
DROPOUT="${DROPOUT:-0.4}"
PREFLIGHT="${PREFLIGHT:-1}"
SHARED_PRECOMPUTE="${SHARED_PRECOMPUTE:-1}"
KEEP_LAST="${KEEP_LAST:-1}"
AMASS_BATCH_CANDIDATES="${AMASS_BATCH_CANDIDATES:-512 768 1024}"
TC_BATCH_CANDIDATES="${TC_BATCH_CANDIDATES:-128 256 512}"
DIP_BATCH_CANDIDATES="${DIP_BATCH_CANDIDATES:-128 256 512}"

AMASS_EPOCHS="${AMASS_EPOCHS:-80}"
TC_EPOCHS="${TC_EPOCHS:-60}"
DIP_EPOCHS="${DIP_EPOCHS:-30}"
AMASS_LR="${AMASS_LR:-1e-4}"
TC_LR="${TC_LR:-1e-5}"
DIP_LR="${DIP_LR:-5e-6}"

AMASS_MAX_TRAIN_SEQS="${AMASS_MAX_TRAIN_SEQS:-0}"
TC_MAX_TRAIN_SEQS="${TC_MAX_TRAIN_SEQS:-0}"
DIP_MAX_TRAIN_SEQS="${DIP_MAX_TRAIN_SEQS:-0}"
AMASS_VAL_SEQS="${AMASS_VAL_SEQS:-64}"
TC_VAL_SEQS="${TC_VAL_SEQS:-0}"
DIP_VAL_SEQS="${DIP_VAL_SEQS:-0}"
AMASS_EVAL_SEQS="${AMASS_EVAL_SEQS:-128}"
TC_EVAL_SEQS="${TC_EVAL_SEQS:-0}"
DIP_EVAL_SEQS="${DIP_EVAL_SEQS:-0}"

mkdir -p "$ROOT/logs" "$ROOT/preflight"
MAIN_LOG="$ROOT/logs/run.log"
exec > >(tee -a "$MAIN_LOG") 2>&1
PRECOMP_DIR="$ROOT/precomputed"

echo "[imu_joint_euler_qdot_vel_ctrl_v1] start $(date --iso-8601=seconds)"
echo "root=$ROOT"
echo "variants=$VARIANTS_RAW"
echo "window=$WINDOW hidden=$HIDDEN layers=$LAYERS dropout=$DROPOUT"
echo "shared_precompute=$SHARED_PRECOMPUTE"
echo "keep_last=$KEEP_LAST"
echo "input contract: official world/model-frame aM[18]+wM[18]+root-frame R_rootIMU_sensorIMU_flat[54]=90D"
echo "rotation input: R_rootIMU_sensorIMU = RMB[root_imu=5]^T @ RMB[sensor]; aM/wM are left in their existing project frame."
echo "output contract: q_RJ_euler_control[18]+qdot_RJ_euler_control[18]+vel_RJ_control[18]=54D"
echo "target contract: R_RJ=R_WR^T R_WJ, p_RJ=(p_WJ-p_WR) @ R_WR; velocities/accelerations finite-diff in root frame."
echo "DIP policy: no DIP trans, no world/root velocity GT, no full-pipeline 11 metrics."

for path in "$AMASS_CACHE" "$TC_TRAIN_CACHE" "$TC_VAL_CACHE" "$TC_TEST_CACHE" "$DIP_TRAIN_CACHE" "$DIP_VAL_CACHE" "$DIP_TEST_CACHE"; do
  if [ ! -f "$path" ]; then
    echo "Missing required cache manifest: $path" >&2
    exit 2
  fi
done

if [ -n "${GPU_LIST:-}" ]; then
  IFS=', ' read -r -a GPUS <<< "$GPU_LIST"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d')
fi
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
fi
echo "visible/planned GPUs=${GPUS[*]}"

run_train() {
  local gpu="$1"
  local dataset="$2"
  local variant="$3"
  local train_cache="$4"
  local val_cache="$5"
  local output_dir="$6"
  local epochs="$7"
  local batch_size="$8"
  local lr="$9"
  local max_train="${10}"
  local max_val="${11}"
  local init_checkpoint="${12:-}"
  local precomputed_train="${13:-}"
  local precomputed_val="${14:-}"

  if [ -f "$output_dir/train_result.json" ] && [ -f "$output_dir/best_loss.pt" ]; then
    echo "skip train existing: $output_dir"
    return 0
  fi
  if [ -d "$output_dir" ] && [ "$(find "$output_dir" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ]; then
    echo "Refusing to overwrite partial/non-empty training directory: $output_dir" >&2
    exit 2
  fi
  mkdir -p "$output_dir"
  local init_args=()
  if [ -n "$init_checkpoint" ]; then
    init_args=(--init-checkpoint "$init_checkpoint")
  fi
  local precompute_args=()
  if [ -n "$precomputed_train" ]; then
    precompute_args+=(--precomputed-train-records "$precomputed_train")
  fi
  if [ -n "$precomputed_val" ]; then
    precompute_args+=(--precomputed-val-records "$precomputed_val")
  fi
  local last_args=()
  if [ "$KEEP_LAST" != "1" ]; then
    last_args=(--no-save-last)
  fi
  echo "train dataset=$dataset variant=$variant gpu=$gpu batch=$batch_size epochs=$epochs lr=$lr out=$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" python imu_joint_euler_qdot_vel_ctrl_train.py \
    --train-cache "$train_cache" \
    --val-cache "$val_cache" \
    --output-dir "$output_dir" \
    --dataset "$dataset" \
    --variant "$variant" \
    --imu-input-mode official \
    --epochs "$epochs" \
    --window "$WINDOW" \
    --batch-size "$batch_size" \
    --lr "$lr" \
    --hidden-size "$HIDDEN" \
    --num-layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --max-train-sequences "$max_train" \
    --max-val-sequences "$max_val" \
    --early-stop-patience "${EARLY_STOP_PATIENCE:-12}" \
    --early-stop-min-delta "${EARLY_STOP_MIN_DELTA:-1e-5}" \
    "${precompute_args[@]}" \
    "${init_args[@]}" \
    "${last_args[@]}"
  if [ "$KEEP_LAST" != "1" ]; then
    rm -f "$output_dir/last.pt"
  fi
}

run_eval() {
  local gpu="$1"
  local dataset="$2"
  local cache="$3"
  local checkpoint="$4"
  local output_json="$5"
  local max_sequences="$6"
  if [ -f "$output_json" ]; then
    echo "skip eval existing: $output_json"
    return 0
  fi
  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint for eval: $checkpoint" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$output_json")"
  local max_args=()
  if [ "$max_sequences" != "0" ]; then
    max_args=(--max-sequences "$max_sequences")
  fi
  echo "eval dataset=$dataset gpu=$gpu ckpt=$checkpoint out=$output_json"
  CUDA_VISIBLE_DEVICES="$gpu" python imu_joint_euler_qdot_vel_ctrl_eval.py \
    --cache "$cache" \
    --checkpoint "$checkpoint" \
    --output-json "$output_json" \
    --dataset "$dataset" \
    --dataset-label "$dataset" \
    --imu-input-mode official \
    "${max_args[@]}"
}

choose_batch() {
  local gpu="$1"
  local name="$2"
  local dataset="$3"
  local train_cache="$4"
  local val_cache="$5"
  local candidates="$6"
  local fallback="$7"
  local precomputed_train="${8:-}"
  local precomputed_val="${9:-}"
  if [ "$PREFLIGHT" != "1" ]; then
    echo "$fallback"
    return 0
  fi
  local selected=""
  for bs in $candidates; do
    local out="$ROOT/preflight/${name}_bs${bs}"
    local log="$ROOT/preflight/${name}_bs${bs}.log"
    if [ -f "$out/train_result.json" ]; then
      selected="$bs"
      echo "preflight $name batch=$bs already ok" >&2
      continue
    fi
    if [ -d "$out" ]; then
      rm -rf "$out"
    fi
    echo "preflight $name batch=$bs on gpu=$gpu" >&2
    local precompute_args=()
    if [ -n "$precomputed_train" ]; then
      precompute_args+=(--precomputed-train-records "$precomputed_train")
    fi
    if [ -n "$precomputed_val" ]; then
      precompute_args+=(--precomputed-val-records "$precomputed_val")
    fi
    if CUDA_VISIBLE_DEVICES="$gpu" python imu_joint_euler_qdot_vel_ctrl_train.py \
      --train-cache "$train_cache" \
      --val-cache "$val_cache" \
      --output-dir "$out" \
      --dataset "$dataset" \
      --variant D_all_balanced \
      --imu-input-mode official \
      --epochs 1 \
      --window "$WINDOW" \
      --batch-size "$bs" \
      --lr 1e-4 \
      --hidden-size "$HIDDEN" \
      --num-layers "$LAYERS" \
      --dropout "$DROPOUT" \
      --max-train-sequences "$bs" \
      --max-val-sequences 2 \
      "${precompute_args[@]}" >"$log" 2>&1; then
      selected="$bs"
    else
      echo "preflight $name batch=$bs failed; selected=${selected:-$fallback}; see $log" >&2
      break
    fi
  done
  echo "${selected:-$fallback}"
}

ensure_precompute() {
  local gpu="$1"
  local name="$2"
  local dataset="$3"
  local cache="$4"
  local output="$5"
  if [ -f "$output" ]; then
    echo "skip precompute existing: $output"
    return 0
  fi
  local job_dir="$ROOT/precompute_jobs/$name"
  rm -rf "$job_dir"
  mkdir -p "$job_dir"
  echo "precompute name=$name dataset=$dataset gpu=$gpu cache=$cache output=$output"
  CUDA_VISIBLE_DEVICES="$gpu" python imu_joint_euler_qdot_vel_ctrl_train.py \
    --train-cache "$cache" \
    --val-cache "$cache" \
    --output-dir "$job_dir" \
    --dataset "$dataset" \
    --variant D_all_balanced \
    --imu-input-mode official \
    --epochs 1 \
    --window "$WINDOW" \
    --batch-size 2 \
    --lr 1e-4 \
    --hidden-size "$HIDDEN" \
    --num-layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --precompute-only \
    --write-precomputed-train-records "$output"
}

PREFLIGHT_GPU="${GPUS[0]}"
mkdir -p "$PRECOMP_DIR"
AMASS_PRECOMP="$PRECOMP_DIR/amass_train_val.pt"
TC_TRAIN_PRECOMP="$PRECOMP_DIR/totalcapture_train.pt"
TC_VAL_PRECOMP="$PRECOMP_DIR/totalcapture_val.pt"
DIP_TRAIN_PRECOMP="$PRECOMP_DIR/dip_train.pt"
DIP_VAL_PRECOMP="$PRECOMP_DIR/dip_val.pt"
if [ "$SHARED_PRECOMPUTE" = "1" ]; then
  ensure_precompute "$PREFLIGHT_GPU" amass amass "$AMASS_CACHE" "$AMASS_PRECOMP"
  ensure_precompute "$PREFLIGHT_GPU" totalcapture_train totalcapture "$TC_TRAIN_CACHE" "$TC_TRAIN_PRECOMP"
  ensure_precompute "$PREFLIGHT_GPU" totalcapture_val totalcapture "$TC_VAL_CACHE" "$TC_VAL_PRECOMP"
  ensure_precompute "$PREFLIGHT_GPU" dip_train dip "$DIP_TRAIN_CACHE" "$DIP_TRAIN_PRECOMP"
  ensure_precompute "$PREFLIGHT_GPU" dip_val dip "$DIP_VAL_CACHE" "$DIP_VAL_PRECOMP"
  rm -rf "$ROOT/precompute_jobs"
else
  echo "shared precompute disabled; each stage will build compact records in memory and avoid large .pt writes."
  AMASS_PRECOMP=""
  TC_TRAIN_PRECOMP=""
  TC_VAL_PRECOMP=""
  DIP_TRAIN_PRECOMP=""
  DIP_VAL_PRECOMP=""
fi

AMASS_BATCH="${AMASS_BATCH:-$(choose_batch "$PREFLIGHT_GPU" amass amass "$AMASS_CACHE" "$AMASS_CACHE" "$AMASS_BATCH_CANDIDATES" 512 "$AMASS_PRECOMP" "$AMASS_PRECOMP")}"
TC_BATCH="${TC_BATCH:-$(choose_batch "$PREFLIGHT_GPU" totalcapture totalcapture "$TC_TRAIN_CACHE" "$TC_VAL_CACHE" "$TC_BATCH_CANDIDATES" 128 "$TC_TRAIN_PRECOMP" "$TC_VAL_PRECOMP")}"
DIP_BATCH="${DIP_BATCH:-$(choose_batch "$PREFLIGHT_GPU" dip dip "$DIP_TRAIN_CACHE" "$DIP_VAL_CACHE" "$DIP_BATCH_CANDIDATES" 128 "$DIP_TRAIN_PRECOMP" "$DIP_VAL_PRECOMP")}"
echo "selected batches: amass=$AMASS_BATCH totalcapture=$TC_BATCH dip=$DIP_BATCH"
rm -rf "$ROOT/preflight"

run_variant_flow() {
  local variant="$1"
  local gpu="$2"
  local variant_root="$ROOT/$variant"
  local eval_dir="$variant_root/eval"
  mkdir -p "$variant_root/logs" "$eval_dir"
  echo "[variant $variant] start gpu=$gpu $(date --iso-8601=seconds)"

  run_train "$gpu" amass "$variant" "$AMASS_CACHE" "$AMASS_CACHE" \
    "$variant_root/amass_pretrain" "$AMASS_EPOCHS" "$AMASS_BATCH" "$AMASS_LR" "$AMASS_MAX_TRAIN_SEQS" "$AMASS_VAL_SEQS" "" "$AMASS_PRECOMP" "$AMASS_PRECOMP"
  local amass_ckpt="$variant_root/amass_pretrain/best_loss.pt"
  run_eval "$gpu" amass "$AMASS_CACHE" "$amass_ckpt" "$eval_dir/eval_amass_after_amass_best.json" "$AMASS_EVAL_SEQS"
  run_eval "$gpu" totalcapture "$TC_TEST_CACHE" "$amass_ckpt" "$eval_dir/eval_totalcapture_test_after_amass_best.json" "$TC_EVAL_SEQS"
  run_eval "$gpu" dip "$DIP_TEST_CACHE" "$amass_ckpt" "$eval_dir/eval_dip_test_after_amass_best.json" "$DIP_EVAL_SEQS"

  run_train "$gpu" totalcapture "$variant" "$TC_TRAIN_CACHE" "$TC_VAL_CACHE" \
    "$variant_root/totalcapture_finetune" "$TC_EPOCHS" "$TC_BATCH" "$TC_LR" "$TC_MAX_TRAIN_SEQS" "$TC_VAL_SEQS" "$amass_ckpt" "$TC_TRAIN_PRECOMP" "$TC_VAL_PRECOMP"
  local tc_ckpt="$variant_root/totalcapture_finetune/best_loss.pt"
  run_eval "$gpu" totalcapture "$TC_TEST_CACHE" "$tc_ckpt" "$eval_dir/eval_totalcapture_test_after_tc_finetune_best.json" "$TC_EVAL_SEQS"

  run_train "$gpu" dip "$variant" "$DIP_TRAIN_CACHE" "$DIP_VAL_CACHE" \
    "$variant_root/dip_finetune" "$DIP_EPOCHS" "$DIP_BATCH" "$DIP_LR" "$DIP_MAX_TRAIN_SEQS" "$DIP_VAL_SEQS" "$amass_ckpt" "$DIP_TRAIN_PRECOMP" "$DIP_VAL_PRECOMP"
  local dip_ckpt="$variant_root/dip_finetune/best_loss.pt"
  run_eval "$gpu" dip "$DIP_TEST_CACHE" "$dip_ckpt" "$eval_dir/eval_dip_test_after_dip_finetune_best.json" "$DIP_EVAL_SEQS"
  run_eval "$gpu" totalcapture "$TC_TEST_CACHE" "$dip_ckpt" "$eval_dir/eval_totalcapture_test_after_dip_finetune_best.json" "$TC_EVAL_SEQS"

  echo "[variant $variant] done $(date --iso-8601=seconds)"
}

read -r -a VARIANTS <<< "$VARIANTS_RAW"
gpu_count="${#GPUS[@]}"
status=0
for ((i = 0; i < ${#VARIANTS[@]}; i += gpu_count)); do
  pids=()
  batch_variants=()
  for ((j = 0; j < gpu_count && i + j < ${#VARIANTS[@]}; j++)); do
    variant="${VARIANTS[$((i + j))]}"
    gpu="${GPUS[$j]}"
    variant_log="$ROOT/logs/${variant}.log"
    (run_variant_flow "$variant" "$gpu") > >(tee -a "$variant_log") 2>&1 &
    pids+=("$!")
    batch_variants+=("$variant")
  done
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "variant failed: ${batch_variants[$idx]}" >&2
      status=1
    fi
  done
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done

python - <<'PY' "$ROOT"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {
    "status": "completed",
    "root": str(root),
    "contract": "input aM[18]+wM[18]+R_rootIMU_sensorIMU_flat[54], where R_rootIMU_sensorIMU=RMB[root_imu=5]^T@RMB[sensor]; outputs q/euler, qdot/euler, vel root-relative controls for joints [18,19,4,5,15,0]",
    "variants": {},
}
metric_keys = [
    "rotation_geodesic_deg",
    "q_euler_L1_deg",
    "q_euler_L2_deg",
    "qdot_from_q_L2_deg_s",
    "qdot_head_L2_deg_s",
    "qddot_from_q_L2_deg_s2",
    "qddot_head_L2_deg_s2",
    "vel_RJ_L1_cm_s",
    "vel_RJ_L2_cm_s",
    "acc_RJ_L2_cm_s2",
    "velocity_direction_angle_deg",
    "baseline_rotation_geodesic_deg",
    "baseline_vel_RJ_L2_cm_s",
    "baseline_acc_RJ_L2_cm_s2",
    "input_RMB_root_relative_rotation_geodesic_deg",
]
skip_dirs = {"logs", "preflight", "precomputed", "precompute_jobs"}
for variant_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in skip_dirs):
    item = {
        "checkpoints": {
            "amass_best": str(variant_dir / "amass_pretrain" / "best_loss.pt"),
            "tc_best": str(variant_dir / "totalcapture_finetune" / "best_loss.pt"),
            "dip_best": str(variant_dir / "dip_finetune" / "best_loss.pt"),
        },
        "eval": {},
    }
    for path in sorted((variant_dir / "eval").glob("*.json")):
        data = json.loads(path.read_text())
        agg = data.get("aggregate", {})
        item["eval"][path.name] = {key: agg.get(key) for key in metric_keys if key in agg}
    summary["variants"][variant_dir.name] = item
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[imu_joint_euler_qdot_vel_ctrl_v1] done $(date --iso-8601=seconds)"
