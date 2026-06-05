#!/usr/bin/env python3
"""Background watcher for long-running experiment orchestrator batches.

This wrapper is intentionally project-agnostic. It launches the normal
`tools/experiment_orchestrator.py` runner in a detached supervisor process,
monitors the runner state file, and writes durable wakeup artifacts:

- watch_status.json: machine-readable status for resume.
- next_action.md: short human-readable next step.
- wakeup_prompt.txt: copyable prompt for the next Codex/Claude turn.

It does not kill jobs, overwrite task outputs, or replace the orchestrator's
dependency/GPU scheduling. It only owns long-task waiting outside the chat
foreground.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import experiment_orchestrator as orch


TERMINAL_STATUSES = {"completed", "failed", "blocked", "skipped"}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"read_error": str(exc)}


def default_run_dir(task_file: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/experiments/orchestrator_watchers") / f"{task_file.stem}_{stamp}"


def project_root() -> Path:
    return Path.cwd()


def process_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def summarize_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not state or "tasks" not in state:
        return {
            "task_count": 0,
            "counts": {},
            "terminal": False,
            "failed": [],
            "blocked": [],
            "running": [],
            "pending": [],
        }
    tasks = state.get("tasks", {})
    counts: Dict[str, int] = {}
    for row in tasks.values():
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "task_count": len(tasks),
        "counts": counts,
        "terminal": bool(tasks) and all(row.get("status") in TERMINAL_STATUSES for row in tasks.values()),
        "failed": [tid for tid, row in tasks.items() if row.get("status") == "failed"],
        "blocked": [tid for tid, row in tasks.items() if row.get("status") == "blocked"],
        "running": [tid for tid, row in tasks.items() if row.get("status") == "running"],
        "pending": [tid for tid, row in tasks.items() if row.get("status") == "pending"],
    }


def load_task_names(task_file: Path) -> Dict[str, str]:
    try:
        data = orch.load_task_file(task_file)
    except Exception:
        return {}
    return {task["id"]: task.get("name", task["id"]) for task in data.get("tasks", [])}


def build_wakeup_prompt(status: Dict[str, Any]) -> str:
    summary = status.get("summary", {})
    failed = ", ".join(summary.get("failed", [])) or "none"
    blocked = ", ".join(summary.get("blocked", [])) or "none"
    running = ", ".join(summary.get("running", [])) or "none"
    pending = ", ".join(summary.get("pending", [])) or "none"
    return (
        "Continue the long-running orchestrator batch from watcher artifacts.\n\n"
        f"Task file: {status.get('task_file')}\n"
        f"State file: {status.get('state_file')}\n"
        f"Watcher run dir: {status.get('run_dir')}\n"
        f"Orchestrator stdout: {status.get('orchestrator_stdout')}\n"
        f"Watcher status: {status.get('watch_status')}\n"
        f"Terminal: {summary.get('terminal')}\n"
        f"Counts: {summary.get('counts')}\n"
        f"Failed: {failed}\n"
        f"Blocked: {blocked}\n"
        f"Running: {running}\n"
        f"Pending: {pending}\n\n"
        "Instructions for the next agent:\n"
        "1. Read watch_status.json and the orchestrator state file first.\n"
        "2. Do not relaunch completed tasks or overwrite existing outputs/logs.\n"
        "3. If terminal=true, parse outputs and summarize results.\n"
        "4. If failed/blocked tasks exist, inspect their task logs and report the retry plan.\n"
        "5. If tasks are still running, report status and keep foreground waiting short.\n"
    )


def write_status_files(run_dir: Path, status: Dict[str, Any]) -> None:
    status["watch_status"] = str(run_dir / "watch_status.json")
    write_json(run_dir / "watch_status.json", status)
    summary = status.get("summary", {})
    if summary.get("terminal"):
        if summary.get("failed") or summary.get("blocked"):
            next_action = "Batch reached a terminal state with failures or blocked tasks. Inspect logs before retrying.\n"
        else:
            next_action = "Batch completed. Parse outputs and summarize the results; do not rerun completed tasks.\n"
    else:
        next_action = "Batch is still running or waiting. Read the state file before deciding whether to resume.\n"
    (run_dir / "next_action.md").write_text(
        "# Next Action\n\n"
        + next_action
        + "\n"
        + f"- State file: `{status.get('state_file')}`\n"
        + f"- Orchestrator stdout: `{status.get('orchestrator_stdout')}`\n"
        + f"- Watcher status: `{run_dir / 'watch_status.json'}`\n"
    )
    (run_dir / "wakeup_prompt.txt").write_text(build_wakeup_prompt(status))


def launch(args: argparse.Namespace) -> None:
    task_file = Path(args.task_file)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(task_file)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty watcher run dir: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--watch",
        "--task-file",
        str(task_file),
        "--run-dir",
        str(run_dir),
        "--poll-seconds",
        str(args.poll_seconds),
        "--watch-poll-seconds",
        str(args.watch_poll_seconds),
        "--max-watch-minutes",
        str(args.max_watch_minutes),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.state_file:
        cmd.extend(["--state-file", args.state_file])
    if args.project_status:
        cmd.extend(["--project-status", args.project_status])
    if args.force:
        cmd.append("--force")
    if args.allow_same_user_gpu_share:
        cmd.append("--allow-same-user-gpu-share")
    if args.max_gpu_utilization is not None:
        cmd.extend(["--max-gpu-utilization", str(args.max_gpu_utilization)])
    watcher_log = (run_dir / "watcher.log").open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=project_root(),
        stdout=watcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    (run_dir / "watcher.pid").write_text(str(proc.pid) + "\n")
    write_json(run_dir / "launch.json", {"pid": proc.pid, "command": cmd, "launched_at": now(), "cwd": str(project_root())})
    status = {
        "status": "watcher_launched",
        "launched_at": now(),
        "task_file": str(task_file),
        "run_dir": str(run_dir),
        "watcher_pid": proc.pid,
        "watcher_log": str(run_dir / "watcher.log"),
        "state_file": args.state_file,
        "orchestrator_stdout": str(run_dir / "orchestrator_stdout.log"),
        "summary": {"terminal": False, "counts": {"watcher_launched": 1}},
    }
    write_status_files(run_dir, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))


def watch(args: argparse.Namespace) -> None:
    task_file = Path(args.task_file)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = orch.load_task_file(task_file)
    state_file = Path(args.state_file) if args.state_file else Path(data.get("state_file") or orch.default_state_file(task_file))
    stdout_path = run_dir / "orchestrator_stdout.log"
    cmd = [
        sys.executable,
        "tools/experiment_orchestrator.py",
        "--task-file",
        str(task_file),
        "--run",
        "--poll-seconds",
        str(args.poll_seconds),
        "--state-file",
        str(state_file),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.project_status:
        cmd.extend(["--project-status", args.project_status])
    if args.force:
        cmd.append("--force")
    if args.allow_same_user_gpu_share:
        cmd.append("--allow-same-user-gpu-share")
    if args.max_gpu_utilization is not None:
        cmd.extend(["--max-gpu-utilization", str(args.max_gpu_utilization)])
    with stdout_path.open("w") as stdout:
        proc = subprocess.Popen(
            cmd,
            cwd=project_root(),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        (run_dir / "orchestrator.pid").write_text(str(proc.pid) + "\n")
        start = time.time()
        task_names = load_task_names(task_file)
        while True:
            state = read_json(state_file)
            summary = summarize_state(state)
            status = {
                "status": "terminal" if summary["terminal"] else "running",
                "updated_at": now(),
                "started_at": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
                "task_file": str(task_file),
                "state_file": str(state_file),
                "run_dir": str(run_dir),
                "orchestrator_pid": proc.pid,
                "orchestrator_running": proc.poll() is None and process_running(proc.pid),
                "orchestrator_return_code": proc.poll(),
                "orchestrator_command": cmd,
                "orchestrator_stdout": str(stdout_path),
                "task_names": task_names,
                "summary": summary,
            }
            write_status_files(run_dir, status)
            if summary["terminal"]:
                return
            if proc.poll() is not None:
                status["status"] = "orchestrator_exited_nonterminal"
                status["summary"] = summarize_state(read_json(state_file))
                status["orchestrator_return_code"] = proc.returncode
                write_status_files(run_dir, status)
                return
            if args.max_watch_minutes and (time.time() - start) > float(args.max_watch_minutes) * 60:
                status["status"] = "watch_timeout"
                status["timeout_minutes"] = args.max_watch_minutes
                write_status_files(run_dir, status)
                return
            time.sleep(args.watch_poll_seconds)


def show_status(args: argparse.Namespace) -> None:
    status_path = Path(args.run_dir) / "watch_status.json"
    data = read_json(status_path)
    if data is None:
        raise SystemExit(f"No watch_status.json found in {args.run_dir}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch and monitor long orchestrator batches in the background.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true", help="Start a detached watcher and return immediately.")
    mode.add_argument("--watch", action="store_true", help="Internal mode used by --launch.")
    mode.add_argument("--status", action="store_true", help="Print an existing watcher status.")
    parser.add_argument("--task-file", help="YAML or JSON orchestrator task file.")
    parser.add_argument("--run-dir", help="Watcher artifact directory.")
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--project-status", default="PROJECT_STATUS.md")
    parser.add_argument("--poll-seconds", type=int, default=15, help="Poll interval passed to experiment_orchestrator.py.")
    parser.add_argument("--watch-poll-seconds", type=int, default=30, help="Watcher status update interval.")
    parser.add_argument("--max-watch-minutes", type=float, default=0, help="0 means no watcher timeout.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-same-user-gpu-share", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=30)
    args = parser.parse_args()
    if args.status:
        if not args.run_dir:
            raise SystemExit("--status requires --run-dir")
        show_status(args)
        return
    if not args.task_file:
        raise SystemExit("--task-file is required for --launch/--watch")
    if args.launch:
        launch(args)
    else:
        if not args.run_dir:
            raise SystemExit("--watch requires --run-dir")
        watch(args)


if __name__ == "__main__":
    main()
