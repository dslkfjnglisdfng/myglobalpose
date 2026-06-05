#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.codex_train_resume.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ -z "${REPO_DIR:-}" ]]; then
  echo "REPO_DIR is empty in $ENV_FILE" >&2
  exit 2
fi

cd "$REPO_DIR"
mkdir -p logs status metrics scripts

nohup bash scripts/run_train_then_codex_resume.sh > logs/launcher.out 2>&1 &
pid=$!
echo "$pid" > train.pid

echo "pid: $pid"
echo "log directory: $REPO_DIR/logs"
echo "status: $REPO_DIR/status/latest.json"
echo "launcher log: $REPO_DIR/logs/launcher.out"
echo "training log: $REPO_DIR/logs/<RUN_NAME>.log"
echo "on finish: codex exec --cd \"$REPO_DIR\" resume ${CODEX_RESUME_ARGS:---last} ..."
