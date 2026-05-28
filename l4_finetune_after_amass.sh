#!/usr/bin/env bash
set -euo pipefail

cd /home/lingfeng/projects/GlobalposeMy/GlobalPose

PY=${PY:-/home/lingfeng/.conda/envs/globalpose-gpu/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

if [[ -z "${AMASS_BEST:-}" ]]; then
  echo "Set AMASS_BEST to the validation-selected AMASS best.pt before running." >&2
  exit 2
fi

COMMON_ARGS=(
  --epochs 40
  --batch-size 512
  --hidden-size 256
  --num-layers 3
  --tail-length 4
  --residual-scale 0.005
  --velocity-residual-scale 0.005
  --lr 3e-5
  --grad-clip 1.0
  --early-stop-patience 15
  --delta-v-root-max-catastrophic 1.0
  --strict-safety-gate
  --init-checkpoint "${AMASS_BEST}"
)

run_dip() {
  "${PY}" -u l4_train_prephysics_pose_velocity.py \
    --cache data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json \
    --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_dip_val_globalpose_neural_only/baseline_cache_manifest.json \
    --output-dir data/experiments/l4_prephysics_pose_velocity_dip_finetune_v1 \
    --disable-root-velocity-loss \
    --disable-root-translation-loss \
    "${COMMON_ARGS[@]}"
}

run_totalcapture() {
  "${PY}" -u l4_train_prephysics_pose_velocity.py \
    --cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only/baseline_cache_manifest.json \
    --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only/baseline_cache_manifest.json \
    --output-dir data/experiments/l4_prephysics_pose_velocity_totalcapture_finetune_v1 \
    "${COMMON_ARGS[@]}"
}

case "${1:-all}" in
  dip)
    run_dip
    ;;
  totalcapture)
    run_totalcapture
    ;;
  all)
    run_dip
    run_totalcapture
    ;;
  *)
    echo "Usage: AMASS_BEST=/path/to/best.pt $0 [dip|totalcapture|all]" >&2
    exit 2
    ;;
esac
