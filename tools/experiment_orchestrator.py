#!/usr/bin/env python3
"""Reusable parallel experiment orchestrator.

This runner is a project-local implementation used by the general
`experiment-orchestrator` Codex skill for training, audit, validation, cache,
parsing, benchmark, and documentation tasks.
It is intentionally task-file driven: experiment design still happens outside
this script, while this script handles dependency scheduling, GPU allocation,
logging, state tracking, result parsing, and PROJECT_STATUS.md writeback.

Streaming-compatible cache rule for future mainline training caches:

For each sequence:
1. reset / initialize RNN state;
2. call official GPNet.rnn_initialize(init_pose, init_velocity);
3. run frame-by-frame forward;
4. save upstream module streaming outputs;
5. downstream training must consume upstream streaming outputs.

Do not use this old batch/cache contract as a new mainline training cache:

    gpnet.plnet([(pl_input, pl_target[0])])

That batch/cache form is allowed only for historical diagnostics or explicit
ablation controls.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - YAML is optional at runtime.
    yaml = None


TASK_STATUSES = {"pending", "running", "completed", "failed", "blocked", "skipped"}

S4_KEYS = [
    ("Score", "score"),
    ("Local SIP", "L SIP Err (deg)"),
    ("Local Angle", "L Angle Err (deg)"),
    ("Local Joint", "L Joint Err (cm)"),
    ("Local Mesh", "L Vertex Err (cm)"),
    ("Global SIP", "G SIP Err (deg)"),
    ("Global Angle", "G Angle Err (deg)"),
    ("Global Joint", "G Joint Err (cm)"),
    ("Global Mesh", "G Vertex Err (cm)"),
    ("Root Jitter", "Root Jitter (km/s^3)"),
    ("Joint Jitter", "Joint Jitter (km/s^3)"),
]

REQUIRED_TASK_KEYS = (
    "id",
    "name",
    "type",
    "command",
    "dependencies",
    "gpu_required",
    "estimated_gpu_mem_gb",
    "priority",
    "outputs",
    "log_path",
    "summary_parser",
    "project_status_section",
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_task_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not available; use a JSON task file.")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise RuntimeError(f"Task file must contain an object: {path}")
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError(f"No tasks found in {path}.")
    normalize_and_validate_tasks(tasks)
    return data


def normalize_and_validate_tasks(tasks: List[Dict[str, Any]]) -> None:
    seen = set()
    for task in tasks:
        missing = [key for key in REQUIRED_TASK_KEYS if key not in task]
        if missing:
            raise KeyError(f"Task {task.get('id')} missing required keys: {missing}")
        task.setdefault("working_dir", ".")
        task.setdefault("env", {})
        task.setdefault("allow_parallel", True)
        task.setdefault("max_runtime_minutes", None)
        task.setdefault("retry", {"max_retries": 0, "retry_on_failure": False})
        task.setdefault("notes", "")
        task.setdefault("tags", [])
        task.setdefault("status", "pending")
        task.setdefault("skip_if_outputs_exist", False)
        if task["id"] in seen:
            raise RuntimeError(f"Duplicate task id: {task['id']}")
        if task["status"] not in TASK_STATUSES:
            raise RuntimeError(f"Invalid status for {task['id']}: {task['status']}")
        seen.add(task["id"])
    for task in tasks:
        for dep in task.get("dependencies", []):
            if dep not in seen:
                raise RuntimeError(f"Task {task['id']} depends on unknown task {dep}")
    check_dependency_cycles(tasks)


def check_dependency_cycles(tasks: List[Dict[str, Any]]) -> None:
    deps = {task["id"]: list(task.get("dependencies", [])) for task in tasks}
    visiting = set()
    visited = set()

    def visit(tid: str, stack: List[str]) -> None:
        if tid in visited:
            return
        if tid in visiting:
            raise RuntimeError("Dependency cycle: " + " -> ".join(stack + [tid]))
        visiting.add(tid)
        for dep in deps.get(tid, []):
            visit(dep, stack + [tid])
        visiting.remove(tid)
        visited.add(tid)

    for tid in deps:
        visit(tid, [])


def default_state_file(task_file: Path) -> Path:
    return Path("data/experiments/orchestrator_states") / f"{task_file.stem}.json"


def initial_state(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "created_at": now(),
        "updated_at": now(),
        "tasks": {
            task["id"]: {
                "status": task.get("status", "pending"),
                "pid": None,
                "gpu": None,
                "start_time": None,
                "end_time": None,
                "return_code": None,
                "log_path": task["log_path"],
                "outputs": list(task.get("outputs", [])),
                "error": None,
                "attempts": 0,
            }
            for task in tasks
        },
    }


def load_state(path: Path, tasks: List[Dict[str, Any]], resume: bool) -> Dict[str, Any]:
    state = initial_state(tasks)
    if resume and path.exists():
        saved = json.loads(path.read_text())
        saved_tasks = saved.get("tasks", {})
        for task in tasks:
            tid = task["id"]
            if tid in saved_tasks:
                state["tasks"][tid].update(saved_tasks[tid])
                if state["tasks"][tid]["status"] == "running":
                    pid = state["tasks"][tid].get("pid")
                    if not process_running(pid):
                        state["tasks"][tid]["status"] = "failed"
                        state["tasks"][tid]["error"] = "resume found stale running task pid"
    for task in tasks:
        tid = task["id"]
        if task.get("skip_if_outputs_exist", False) and task_outputs_complete(task):
            state["tasks"][tid]["status"] = "completed"
            state["tasks"][tid]["error"] = None
    state["updated_at"] = now()
    return state


def process_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def list_gpus() -> List[Dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    query = "index,memory.used,memory.total,utilization.gpu,temperature.gpu"
    result = run_cmd([
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ])
    if result.returncode != 0:
        return []
    uuid_result = run_cmd([
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ])
    uuid_to_index: Dict[str, int] = {}
    if uuid_result.returncode == 0:
        for line in uuid_result.stdout.splitlines():
            if line.strip():
                idx, uuid = [part.strip() for part in line.split(",", 1)]
                uuid_to_index[uuid] = int(idx)
    proc_result = run_cmd([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ])
    procs: Dict[int, List[Dict[str, Any]]] = {}
    if proc_result.returncode == 0:
        for line in proc_result.stdout.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            uuid, pid, mem = parts[:3]
            idx = uuid_to_index.get(uuid)
            if idx is None:
                continue
            owner = None
            cmdline = ""
            ps = run_cmd(["ps", "-o", "user=,cmd=", "-p", pid])
            if ps.returncode == 0 and ps.stdout.strip():
                bits = ps.stdout.strip().split(maxsplit=1)
                owner = bits[0]
                cmdline = bits[1] if len(bits) > 1 else ""
            procs.setdefault(idx, []).append({
                "pid": int(pid),
                "owner": owner,
                "used_memory_mib": int(float(mem)),
                "cmd": cmdline,
            })
    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        idx, used, total, util, temp = [part.strip() for part in line.split(",")]
        used_i = int(float(used))
        total_i = int(float(total))
        gpus.append({
            "index": int(idx),
            "used_mib": used_i,
            "total_mib": total_i,
            "free_mib": total_i - used_i,
            "utilization": int(float(util)),
            "temperature": int(float(temp)),
            "processes": procs.get(int(idx), []),
        })
    return gpus


def select_gpu(
    task: Dict[str, Any],
    gpus: List[Dict[str, Any]],
    reserved: Iterable[int],
    allow_same_user_share: bool = False,
    max_utilization: int = 30,
) -> Optional[int]:
    if not task.get("gpu_required", False):
        return None
    need_mib = int(float(task.get("estimated_gpu_mem_gb", 0)) * 1024)
    current_user = os.environ.get("USER")
    reserved_set = set(reserved)
    candidates = []
    for gpu in gpus:
        if gpu["index"] in reserved_set:
            continue
        foreign = [
            proc for proc in gpu.get("processes", [])
            if proc.get("owner") and proc.get("owner") != current_user
        ]
        if foreign:
            continue
        if gpu.get("processes") and not allow_same_user_share:
            continue
        if gpu["free_mib"] < need_mib:
            continue
        if gpu["utilization"] > max_utilization and gpu.get("processes"):
            continue
        candidates.append(gpu)
    if not candidates:
        return None
    candidates.sort(key=lambda g: (-g["free_mib"], g["utilization"], g["index"]))
    return candidates[0]["index"]


def output_conflicts(tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    owners: Dict[str, str] = {}
    conflicts = []
    for task in tasks:
        for out in task.get("outputs", []):
            if out in owners:
                conflicts.append({"path": out, "first_task": owners[out], "second_task": task["id"]})
            owners[out] = task["id"]
    return conflicts


def output_dir_conflicts(tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    owners: Dict[str, str] = {}
    conflicts = []
    for task in tasks:
        dirs = {str(Path(out).parent) for out in task.get("outputs", [])}
        for dpath in dirs:
            if dpath in owners:
                conflicts.append({"directory": dpath, "first_task": owners[dpath], "second_task": task["id"]})
            owners[dpath] = task["id"]
    return conflicts


def existing_outputs(task: Dict[str, Any]) -> List[str]:
    return [out for out in task.get("outputs", []) if Path(out).exists()]


def task_outputs_complete(task: Dict[str, Any]) -> bool:
    outputs = task.get("outputs", [])
    return bool(outputs) and all(Path(out).exists() for out in outputs)


def log_conflicts(tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    owners: Dict[str, str] = {}
    conflicts = []
    for task in tasks:
        log_path = task.get("log_path")
        if not log_path:
            continue
        if log_path in owners:
            conflicts.append({"path": log_path, "first_task": owners[log_path], "second_task": task["id"]})
        owners[log_path] = task["id"]
    return conflicts


def runnable_for_preflight(task: Dict[str, Any], state: Optional[Dict[str, Any]]) -> bool:
    if state is None:
        return True
    status = state.get("tasks", {}).get(task["id"], {}).get("status", task.get("status", "pending"))
    return status not in {"completed", "skipped", "blocked"}


def preflight_conflicts(tasks: List[Dict[str, Any]], force: bool, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    checked_tasks = [task for task in tasks if runnable_for_preflight(task, state)]
    existing_output_rows = []
    existing_log_rows = []
    for task in checked_tasks:
        if not force and not task.get("skip_if_outputs_exist", False):
            for out in existing_outputs(task):
                existing_output_rows.append({"task": task["id"], "path": out})
        log_path = Path(task["log_path"])
        if log_path.exists() and not force and not task.get("append_log", False):
            existing_log_rows.append({"task": task["id"], "path": str(log_path)})
    return {
        "duplicate_outputs": output_conflicts(checked_tasks),
        "duplicate_output_directories": output_dir_conflicts(checked_tasks),
        "duplicate_logs": log_conflicts(checked_tasks),
        "existing_outputs": existing_output_rows,
        "existing_logs": existing_log_rows,
    }


def conflict_errors(conflicts: Dict[str, Any]) -> List[str]:
    errors = []
    if conflicts["duplicate_outputs"]:
        errors.append(f"duplicate output paths: {conflicts['duplicate_outputs']}")
    if conflicts["duplicate_output_directories"]:
        errors.append(f"duplicate output directories: {conflicts['duplicate_output_directories']}")
    if conflicts["duplicate_logs"]:
        errors.append(f"duplicate log paths: {conflicts['duplicate_logs']}")
    if conflicts["existing_outputs"]:
        errors.append(f"existing outputs: {conflicts['existing_outputs']}")
    if conflicts["existing_logs"]:
        errors.append(f"existing logs: {conflicts['existing_logs']}")
    return errors


def dependency_graph(tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    return {task["id"]: list(task.get("dependencies", [])) for task in tasks}


def downstream_blocked(tasks: List[Dict[str, Any]], failed_id: str) -> List[str]:
    blocked = []
    changed = True
    failed_or_blocked = {failed_id}
    while changed:
        changed = False
        for task in tasks:
            tid = task["id"]
            if tid in failed_or_blocked:
                continue
            if any(dep in failed_or_blocked for dep in task.get("dependencies", [])):
                failed_or_blocked.add(tid)
                blocked.append(tid)
                changed = True
    return blocked


def ready_tasks(tasks: List[Dict[str, Any]], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    ready = []
    statuses = {tid: row["status"] for tid, row in state["tasks"].items()}
    failed_like = {tid for tid, status in statuses.items() if status in {"failed", "blocked"}}
    for task in sorted(tasks, key=lambda t: (t["priority"], t["id"])):
        tid = task["id"]
        if statuses.get(tid, task.get("status", "pending")) != "pending":
            continue
        if any(dep in failed_like for dep in task.get("dependencies", [])):
            state["tasks"][tid]["status"] = "blocked"
            state["tasks"][tid]["error"] = "dependency failed or blocked"
            continue
        if all(statuses.get(dep) == "completed" for dep in task.get("dependencies", [])):
            ready.append(task)
    return ready


def parse_s4(path: Path) -> str:
    data = json.loads(path.read_text())
    model_metrics = data.get("aggregate", {}).get("model_metrics", {})
    rows = []
    for label, key in S4_KEYS:
        if label == "Score":
            value = data.get("score")
        else:
            item = model_metrics.get(key)
            value = item.get("mean") if isinstance(item, dict) else item
        rows.append((label, value))
    return markdown_metric_table(rows)


def parse_train_log(path: Path) -> str:
    data = json.loads(path.read_text())
    rows = [
        ("status", data.get("status")),
        ("best_epoch", data.get("best_epoch")),
        ("best_loss", data.get("best_loss")),
    ]
    history = data.get("history") or []
    if history:
        last = history[-1]
        validation_loss = (last.get("validation") or {}).get("loss")
        if isinstance(validation_loss, dict):
            validation_loss = validation_loss.get("loss")
        rows.append(("last_epoch", last.get("epoch")))
        rows.append(("last_val_loss", validation_loss))
    weights = data.get("weights")
    text = markdown_metric_table(rows)
    if weights:
        text += "\nWeights:\n\n" + markdown_metric_table(sorted(weights.items()))
    return text


def parse_streaming_pl(path: Path) -> str:
    data = json.loads(path.read_text())
    lines = ["| Run | Input | pRB frame cm | gR1 frame deg | gRdot L2 | gRddot L2 | Frames |", "|---|---|---:|---:|---:|---:|---:|"]
    for run in data.get("runs", []):
        fw = run.get("summary", {}).get("frame_weighted", {})
        lines.append(
            "| {run} | {input_mode} | {pRB:.6f} | {gR1:.6f} | {gdot:.6f} | {gddot:.6f} | {frames} |".format(
                run=run.get("run"),
                input_mode=run.get("input_mode"),
                pRB=safe_float(((fw.get("pRB_mean_cm") or {}).get("mean"))),
                gR1=safe_float(((fw.get("gR1_mean_deg") or {}).get("mean"))),
                gdot=safe_float(((fw.get("gRdot_l2") or {}).get("mean"))),
                gddot=safe_float(((fw.get("gRddot_l2") or {}).get("mean"))),
                frames=run.get("summary", {}).get("num_frames"),
            )
        )
    return "\n".join(lines)


def parse_generic_json(path: Path) -> str:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        rows = []
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                rows.append((key, value))
        if rows:
            return markdown_metric_table(rows)
    return "Generic JSON parsed. See output path for full content."


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def markdown_metric_table(rows: Iterable[Tuple[str, Any]]) -> str:
    lines = ["| metric | value |", "|---|---:|"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def parse_summary(task: Dict[str, Any]) -> str:
    parser = (task.get("summary_parser") or "none").strip()
    if parser == "none":
        return "Task completed. No parser requested; see outputs and log."
    outputs = [Path(out) for out in task.get("outputs", [])]
    json_outputs = [out for out in outputs if out.suffix == ".json" and out.exists()]
    if parser == "parse_train_log":
        target = next((out for out in outputs if out.name == "train_result.json" and out.exists()), None)
        target = target or (json_outputs[0] if json_outputs else None)
        return parse_train_log(target) if target else "Task completed, but train_result.json was not found."
    if parser == "parse_s4_metrics":
        return parse_s4(json_outputs[0]) if json_outputs else "Task completed, but S4 JSON was not found."
    if parser == "parse_streaming_pl_audit":
        return parse_streaming_pl(json_outputs[0]) if json_outputs else "Task completed, but streaming PL JSON was not found."
    if parser == "parse_generic_json":
        return parse_generic_json(json_outputs[0]) if json_outputs else "Task completed, but JSON output was not found."
    return f"Task completed, parser `{parser}` is not implemented. See outputs and log."


def log_tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(read_text(path).splitlines()[-n:])


def append_project_status(
    project_status_path: Path,
    task: Dict[str, Any],
    state_row: Dict[str, Any],
    summary: str,
    missing_outputs: List[str],
    blocked_downstream: List[str],
) -> None:
    section = task.get("project_status_section", "Parallel Experiment Orchestrator Skill")
    block = []
    block.append(f"\n### Orchestrator Task: {task['id']}\n\n")
    block.append(f"Name: {task['name']}\n\n")
    block.append(f"Status: {state_row['status']}\n\n")
    block.append(f"Type: {task['type']}\n\n")
    block.append(f"Start: {state_row.get('start_time')}\n\n")
    block.append(f"End: {state_row.get('end_time')}\n\n")
    block.append(f"GPU: {state_row.get('gpu') if state_row.get('gpu') is not None else 'CPU'}\n\n")
    block.append(f"PID: {state_row.get('pid')}\n\n")
    block.append(f"Return code: {state_row.get('return_code')}\n\n")
    block.append(f"Command: `{task['command']}`\n\n")
    block.append(f"Log: `{task['log_path']}`\n\n")
    block.append("Outputs:\n\n")
    for out in task.get("outputs", []):
        block.append(f"- `{out}`\n")
    if missing_outputs:
        block.append("\nMissing outputs:\n\n")
        for out in missing_outputs:
            block.append(f"- `{out}`\n")
    if blocked_downstream:
        block.append("\nBlocked downstream tasks:\n\n")
        for tid in blocked_downstream:
            block.append(f"- `{tid}`\n")
    if state_row.get("error"):
        block.append(f"\nError: {state_row['error']}\n")
    block.append("\nSummary:\n\n")
    block.append(summary.rstrip() + "\n")
    text = read_text(project_status_path) or "# GlobalPose Project Status\n"
    header = f"## {section}"
    if header not in text:
        text = text.rstrip() + f"\n\n{header}\n"
    insert_at = text.find(header) + len(header)
    next_header = text.find("\n## ", insert_at)
    if next_header == -1:
        text = text.rstrip() + "\n" + "".join(block)
    else:
        text = text[:next_header].rstrip() + "\n" + "".join(block) + "\n" + text[next_header:].lstrip("\n")
    project_status_path.write_text(text)


def dry_run(
    tasks: List[Dict[str, Any]],
    state: Dict[str, Any],
    force: bool,
    allow_same_user_share: bool,
    max_gpu_utilization: int,
) -> Dict[str, Any]:
    conflicts = preflight_conflicts(tasks, force=force, state=state)
    ready = ready_tasks(tasks, state)
    gpus = list_gpus()
    reserved = set()
    planned = []
    for task in ready:
        gpu = select_gpu(task, gpus, reserved, allow_same_user_share, max_gpu_utilization)
        if task.get("gpu_required") and gpu is None:
            planned.append({"task": task["id"], "action": "wait_for_gpu"})
        else:
            if gpu is not None:
                reserved.add(gpu)
            planned.append({"task": task["id"], "action": "start", "gpu": gpu if gpu is not None else "CPU"})
    pending_wait = []
    for task in tasks:
        row = state["tasks"][task["id"]]
        if row["status"] == "pending" and task not in ready:
            pending_wait.append({"task": task["id"], "dependencies": task.get("dependencies", [])})
    return {
        "task_count": len(tasks),
        "dependency_graph": dependency_graph(tasks),
        "statuses": {tid: row["status"] for tid, row in state["tasks"].items()},
        "ready_tasks": [task["id"] for task in ready],
        "pending_waiting_for_dependencies": pending_wait,
        "gpus": gpus,
        "planned": planned,
        "conflicts": conflicts,
        "conflict_errors": conflict_errors(conflicts),
        "will_execute_commands": False,
    }


def task_working_dir(task: Dict[str, Any]) -> Path:
    return Path(task.get("working_dir") or ".").resolve()


def launch_task(task: Dict[str, Any], gpu: Optional[int], force: bool) -> Tuple[subprocess.Popen[str], Any]:
    log_path = Path(task["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and not force and not task.get("append_log", False):
        raise RuntimeError(f"Refusing to overwrite existing log: {log_path}")
    mode = "a" if task.get("append_log", False) else "w"
    log_file = log_path.open(mode)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd()}:{env.get('PYTHONPATH', '')}"
    for key, value in (task.get("env") or {}).items():
        if key == "CUDA_VISIBLE_DEVICES" and value == "auto":
            continue
        env[key] = str(value)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = task["command"]
    log_file.write(f"# task_id={task['id']}\n# start={now()}\n# gpu={gpu if gpu is not None else 'CPU'}\n# command={command}\n\n")
    log_file.flush()
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=task_working_dir(task),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        executable="/bin/bash",
    )
    return proc, log_file


def run_scheduler(
    tasks: List[Dict[str, Any]],
    state: Dict[str, Any],
    state_file: Path,
    project_status_path: Path,
    poll_seconds: int,
    force: bool,
    allow_same_user_share: bool,
    max_gpu_utilization: int,
) -> Dict[str, Any]:
    conflicts = preflight_conflicts(tasks, force=force, state=state)
    errors = conflict_errors(conflicts)
    if errors:
        raise RuntimeError("Preflight conflicts:\n" + "\n".join(errors))
    by_id = {task["id"]: task for task in tasks}
    running: Dict[str, Dict[str, Any]] = {}
    log_handles: Dict[str, Any] = {}
    write_json(state_file, state)
    try:
        while True:
            for tid, item in list(running.items()):
                proc = item["proc"]
                task = by_id[tid]
                max_minutes = task.get("max_runtime_minutes")
                if proc.poll() is None and max_minutes:
                    elapsed = time.time() - item["start_ts"]
                    if elapsed > float(max_minutes) * 60:
                        proc.send_signal(signal.SIGTERM)
                        state["tasks"][tid]["error"] = f"max_runtime_minutes exceeded: {max_minutes}"
                if proc.poll() is None:
                    continue
                log_handles[tid].close()
                running.pop(tid)
                row = state["tasks"][tid]
                row["return_code"] = proc.returncode
                row["end_time"] = now()
                missing = [out for out in task.get("outputs", []) if not Path(out).exists()]
                ok = proc.returncode == 0 and not missing
                row["status"] = "completed" if ok else "failed"
                if missing:
                    row["error"] = f"missing outputs: {missing}"
                elif proc.returncode != 0:
                    row["error"] = f"return code {proc.returncode}"
                blocked = []
                if not ok:
                    blocked = downstream_blocked(tasks, tid)
                    for blocked_tid in blocked:
                        if state["tasks"][blocked_tid]["status"] == "pending":
                            state["tasks"][blocked_tid]["status"] = "blocked"
                            state["tasks"][blocked_tid]["error"] = f"dependency {tid} failed"
                summary = parse_summary(task) if ok else "Task failed.\n\nLog tail:\n\n```text\n" + log_tail(Path(task["log_path"])) + "\n```"
                append_project_status(project_status_path, task, row, summary, missing, blocked)
                state["updated_at"] = now()
                write_json(state_file, state)
            if all(row["status"] in {"completed", "failed", "blocked", "skipped"} for row in state["tasks"].values()):
                return state
            gpus = list_gpus()
            active_gpu = {item["gpu"] for item in running.values() if item["gpu"] is not None}
            for task in ready_tasks(tasks, state):
                tid = task["id"]
                if not task.get("allow_parallel", True) and running:
                    continue
                gpu = select_gpu(task, gpus, active_gpu, allow_same_user_share, max_gpu_utilization)
                if task.get("gpu_required") and gpu is None:
                    continue
                if gpu is not None:
                    active_gpu.add(gpu)
                for out in task.get("outputs", []):
                    Path(out).parent.mkdir(parents=True, exist_ok=True)
                proc, handle = launch_task(task, gpu, force=force)
                row = state["tasks"][tid]
                row["status"] = "running"
                row["pid"] = proc.pid
                row["gpu"] = gpu
                row["start_time"] = now()
                row["end_time"] = None
                row["return_code"] = None
                row["error"] = None
                row["attempts"] = int(row.get("attempts") or 0) + 1
                row["log_path"] = task["log_path"]
                running[tid] = {"proc": proc, "gpu": gpu, "start_ts": time.time()}
                log_handles[tid] = handle
                print(json.dumps({"started": tid, "pid": proc.pid, "gpu": gpu if gpu is not None else "CPU", "log": task["log_path"]}), flush=True)
                state["updated_at"] = now()
                write_json(state_file, state)
            time.sleep(poll_seconds)
    finally:
        for handle in log_handles.values():
            try:
                handle.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel experiment orchestrator.")
    parser.add_argument("--task-file", required=True, help="YAML or JSON task file.")
    parser.add_argument("--dry-run", action="store_true", help="Print scheduling plan without executing commands.")
    parser.add_argument("--run", action="store_true", help="Execute pending tasks.")
    parser.add_argument("--resume", action="store_true", help="Resume from the state file.")
    parser.add_argument("--state-file", default=None, help="Override state file path.")
    parser.add_argument("--project-status", default="PROJECT_STATUS.md", help="Project status markdown path.")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing logs/outputs. Default is false.")
    parser.add_argument("--allow-same-user-gpu-share", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=30)
    parser.add_argument("--write-skill-doc", action="store_true", help="Deprecated no-op; skill docs live in ~/.codex/skills/experiment-orchestrator.")
    args = parser.parse_args()
    if args.dry_run and args.run:
        raise SystemExit("Use only one of --dry-run or --run.")
    task_file = Path(args.task_file)
    data = load_task_file(task_file)
    tasks = data["tasks"]
    state_file = Path(args.state_file) if args.state_file else Path(data.get("state_file") or default_state_file(task_file))
    project_status_path = Path(args.project_status)
    allow_same_user_share = bool(data.get("allow_same_user_gpu_share", False) or args.allow_same_user_gpu_share)
    if args.write_skill_doc and not args.dry_run and not args.run:
        print(json.dumps({"note": "--write-skill-doc is deprecated; no project document was modified."}, indent=2))
        return
    state = load_state(state_file, tasks, resume=args.resume)
    if args.dry_run:
        print(json.dumps(dry_run(tasks, state, args.force, allow_same_user_share, args.max_gpu_utilization), indent=2, ensure_ascii=False))
        return
    if not args.run:
        raise SystemExit("Specify --dry-run, --run, or --write-skill-doc.")
    final_state = run_scheduler(
        tasks=tasks,
        state=state,
        state_file=state_file,
        project_status_path=project_status_path,
        poll_seconds=args.poll_seconds,
        force=args.force,
        allow_same_user_share=allow_same_user_share,
        max_gpu_utilization=args.max_gpu_utilization,
    )
    print(json.dumps({"state_file": str(state_file), "tasks": final_state["tasks"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
