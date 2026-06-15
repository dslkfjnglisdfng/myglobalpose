#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=${ENV_DIR:-/home/lingfeng/.conda/envs/globalpose-gpu}
PY="$ENV_DIR/bin/python"
OUTPUT_ROOT=${OUTPUT_ROOT:-data/dataset_work/GTControlCache}
SHARD_SIZE=${SHARD_SIZE:-32}
FK_BATCH_SIZE=${FK_BATCH_SIZE:-2048}

export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

"$PY" scripts/build_gt_control_cache.py \
  --all-defaults \
  --output-root "$OUTPUT_ROOT" \
  --shard-size "$SHARD_SIZE" \
  --fk-batch-size "$FK_BATCH_SIZE" \
  "$@"
