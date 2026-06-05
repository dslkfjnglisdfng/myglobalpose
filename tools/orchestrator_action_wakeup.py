#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {"read_error": str(exc)}


def status_line(label: str, data: dict[str, Any]) -> str | None:
    summary = data.get("summary") or {}
    failed = summary.get("failed") or []
    blocked = summary.get("blocked") or []
    terminal = bool(summary.get("terminal")) or data.get("status") == "terminal"
    if failed or blocked:
        return f"{label} 训练需要处理：failed={failed} blocked={blocked}。请根据日志诊断后决定下一步。"
    if terminal:
        return f"{label} 训练结束，请根据训练结果进行下一步。watch_status={data.get('watch_status')} state_file={data.get('state_file')}"
    if data.get("status") in {"orchestrator_exited_nonterminal", "watch_timeout"}:
        return f"{label} watcher 异常停止：status={data.get('status')}。请读取 watch_status 和 orchestrator_stdout 决定下一步。"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit one actionable wakeup line for long orchestrator training batches.")
    parser.add_argument("--watch-status", action="append", required=True, help="Path to watch_status.json. Can be repeated.")
    parser.add_argument("--label", action="append", default=[], help="Optional label matching each --watch-status.")
    parser.add_argument("--selection-file", action="append", default=[], help="Optional completion artifact such as pl1_selection.json.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="0 means no timeout.")
    args = parser.parse_args()

    watch_paths = [Path(p) for p in args.watch_status]
    labels = list(args.label)
    while len(labels) < len(watch_paths):
        labels.append(watch_paths[len(labels)].parent.name)
    selection_paths = [Path(p) for p in args.selection_file]
    start = time.time()
    seen: set[str] = set()

    while True:
        for selection in selection_paths:
            if selection.exists():
                key = f"selection:{selection}"
                if key not in seen:
                    print(f"训练结束，请根据训练结果进行下一步。选择结果已生成：{selection}", flush=True)
                    return 0
        for label, path in zip(labels, watch_paths):
            data = read_json(path)
            if not data:
                continue
            line = status_line(label, data)
            if line:
                key = f"{label}:{line}"
                if key not in seen:
                    print(line, flush=True)
                    return 0 if "训练结束" in line else 2
        if args.timeout_seconds and time.time() - start > args.timeout_seconds:
            print(f"长训练唤醒脚本超时：{datetime.now().isoformat(timespec='seconds')}。请手动读取 watch_status。", flush=True)
            return 3
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
