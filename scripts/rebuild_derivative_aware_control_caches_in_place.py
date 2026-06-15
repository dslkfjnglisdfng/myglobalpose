import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_TOKENS = (
    "smoke",
    "gt_control_derivative_audit_20260608",
)


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def is_full_cache(path):
    text = str(path)
    return not any(token in text for token in SKIP_TOKENS)


def infer_shard_size(manifest, default):
    cache_files = manifest.get("cache_files") or []
    sizes = [int(item.get("num_sequences", 0)) for item in cache_files if int(item.get("num_sequences", 0)) > 0]
    return max(sizes) if sizes else default


def newik1_command(manifest_path, manifest):
    output_dir = manifest_path.parent
    source = manifest.get("source_cache")
    if not source:
        raise ValueError(f"{manifest_path} missing source_cache")
    mode = manifest.get("mode")
    if mode not in ("teacher_forced", "pl1_streaming"):
        raise ValueError(f"{manifest_path} has unsupported mode={mode!r}")
    feature_mode = manifest.get("feature_mode")
    if feature_mode not in ("control_tail", "last_control"):
        feature_mode = "last_control" if int(manifest.get("input_size", 120)) == 63 else "control_tail"
    cmd = [
        sys.executable,
        "newik1_control_cache.py",
        "--input-cache",
        source,
        "--output-dir",
        str(output_dir),
        "--mode",
        mode,
        "--imu-input-mode",
        manifest.get("imu_input_mode", "auto"),
        "--feature-mode",
        feature_mode,
        "--tail-len",
        str(int(manifest.get("tail_len", 4))),
        "--shard-size",
        str(infer_shard_size(manifest, 50)),
    ]
    pl_checkpoint = manifest.get("pl_checkpoint")
    if mode == "pl1_streaming":
        if not pl_checkpoint:
            raise ValueError(f"{manifest_path} pl1_streaming missing pl_checkpoint")
        cmd.extend(["--pl-checkpoint", pl_checkpoint])
    return cmd


def newpose_command(manifest_path, manifest):
    output_dir = manifest_path.parent
    source = manifest.get("source_cache")
    pl_checkpoint = manifest.get("pl_checkpoint")
    if not source or not pl_checkpoint:
        raise ValueError(f"{manifest_path} missing source_cache or pl_checkpoint")
    cmd = [
        sys.executable,
        "newpose_ctrl_cache.py",
        "--input-cache",
        source,
        "--output-dir",
        str(output_dir),
        "--pl-checkpoint",
        pl_checkpoint,
        "--imu-input-mode",
        manifest.get("imu_input_mode", "official"),
        "--shard-size",
        str(infer_shard_size(manifest, 100)),
    ]
    if manifest.get("include_official_distill", False):
        cmd.append("--include-official-distill")
    return cmd


def discover(root):
    tasks = []
    for manifest_path in sorted(root.glob("data/**/newik1_control_cache_manifest.json")):
        if not is_full_cache(manifest_path):
            continue
        manifest = load_manifest(manifest_path)
        tasks.append(("newik1", manifest_path, newik1_command(manifest_path, manifest)))
    for manifest_path in sorted(root.glob("data/**/newpose_ctrl_cache_manifest.json")):
        if not is_full_cache(manifest_path):
            continue
        manifest = load_manifest(manifest_path)
        tasks.append(("newpose", manifest_path, newpose_command(manifest_path, manifest)))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Rebuild historical NewIK1/NewPose control-tail caches in place with the current derivative-aware control fit.")
    parser.add_argument("--run", action="store_true", help="Execute rebuilds. Without this flag, only print the task list.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = discover(ROOT)
    if args.limit:
        tasks = tasks[: args.limit]
    print(json.dumps({
        "status": "dry_run" if not args.run else "running",
        "num_tasks": len(tasks),
        "tasks": [
            {
                "kind": kind,
                "manifest": str(path),
                "command": " ".join(cmd),
            }
            for kind, path, cmd in tasks
        ],
    }, indent=2), flush=True)
    if not args.run:
        return
    completed = []
    for idx, (kind, manifest_path, cmd) in enumerate(tasks, start=1):
        print(json.dumps({
            "event": "start",
            "index": idx,
            "num_tasks": len(tasks),
            "kind": kind,
            "manifest": str(manifest_path),
        }), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        completed.append(str(manifest_path))
        print(json.dumps({
            "event": "done",
            "index": idx,
            "manifest": str(manifest_path),
        }), flush=True)
    print(json.dumps({"status": "ok", "completed": completed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
