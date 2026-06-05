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

CODEX_RESUME_ARGS="${CODEX_RESUME_ARGS:---last}"

cd "$REPO_DIR"
mkdir -p logs status metrics scripts

RUN_NAME="${RUN_NAME:-train_$(date +%Y%m%d_%H%M%S)}"
LOG="logs/${RUN_NAME}.log"
TAIL_LOG="logs/latest_tail.txt"
STATUS_JSON="status/latest.json"
RESUME_LOG="logs/${RUN_NAME}.codex_resume.log"
final_status_written=0

write_status() {
  local status="$1"
  local exit_code="${2:-}"
  local end_time="${3:-}"
  python - "$STATUS_JSON" "$status" "$RUN_NAME" "$(date -Is)" "$LOG" "$TAIL_LOG" "$exit_code" "$end_time" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
status = sys.argv[2]
run_name = sys.argv[3]
timestamp = sys.argv[4]
log = sys.argv[5]
tail = sys.argv[6]
exit_code = sys.argv[7]
end_time = sys.argv[8]

previous = {}
if status_path.exists():
    try:
        previous = json.loads(status_path.read_text())
    except Exception:
        previous = {}

data = {
    "status": status,
    "run_name": run_name,
    "start": previous.get("start", timestamp),
    "log": log,
}

if status != "running":
    data["exit_code"] = int(exit_code)
    data["end"] = end_time or timestamp
    data["tail"] = tail

optional_paths = {
    "metrics_latest": Path("metrics/latest.json"),
    "summary": Path("summary.txt"),
}
for key, path in optional_paths.items():
    if path.exists():
        data[key] = str(path)

candidate_files = []
for root in (Path("data/experiments"), Path("outputs"), Path("checkpoints")):
    if not root.exists():
        continue
    for pattern in ("*.pt", "*.pth", "*.ckpt", "train_result.json", "result.json", "events.out.tfevents*", "wandb/latest-run"):
        candidate_files.extend(root.rglob(pattern))

existing = [p for p in candidate_files if p.exists()]
existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
if existing:
    data["recent_artifacts"] = [str(p) for p in existing[:20]]

status_path.parent.mkdir(parents=True, exist_ok=True)
status_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
PY
}

write_status "running"

on_unexpected_exit() {
  local rc=$?
  if [[ "$final_status_written" -eq 0 ]]; then
    tail -n 120 "$LOG" > "$TAIL_LOG" 2>/dev/null || true
    write_status "failed" "$rc" "$(date -Is)" || true
  fi
}
trap on_unexpected_exit EXIT

exit_code=0
if [[ -z "${TRAIN_CMD:-}" ]]; then
  {
    echo "TRAIN_CMD is empty in $ENV_FILE."
    echo "Edit .codex_train_resume.env and set TRAIN_CMD before launching a real training run."
  } | tee "$LOG"
  exit_code=2
else
  set +e
  bash -lc "$TRAIN_CMD" 2>&1 | tee "$LOG"
  exit_code=${PIPESTATUS[0]}
  set -e
fi

tail -n 120 "$LOG" > "$TAIL_LOG" || true

if [[ "$exit_code" -eq 0 ]]; then
  write_status "done" "$exit_code" "$(date -Is)"
else
  write_status "failed" "$exit_code" "$(date -Is)"
fi
final_status_written=1

RESUME_PROMPT="训练脚本已经结束。请根据当前结果继续：先读取 status/latest.json、logs/latest_tail.txt，以及存在的 metrics/latest.json、summary.txt、eval 结果或 checkpoint 信息；判断成功/失败和关键指标；如果成功就执行下一步合理的分析/文档更新/后续实验决策，如果失败就定位报错并给出修复方案。不要重新启动训练，不要使用 tail -f/watch/while true/sleep 循环，不要持续监控进程。"

# shellcheck disable=SC2086
codex exec --cd "$REPO_DIR" resume $CODEX_RESUME_ARGS "$RESUME_PROMPT" > "$RESUME_LOG" 2>&1 || true

exit "$exit_code"
