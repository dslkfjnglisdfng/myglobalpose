"""Build a versioned AMASS cache with temporally consistent FK RMB and wM."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import load_cache_files
from pl_va_state import causal_world_angular_velocity_from_rmb_sequence

IMU_JOINTS = (18, 19, 4, 5, 15, 0)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def fk_rmb_and_consistent_w(pose, model, device, dt):
    global_r = model.forward_kinematics_R(pose.to(device).float())[:, IMU_JOINTS]
    rmb = global_r.cpu()
    w_m = causal_world_angular_velocity_from_rmb_sequence(rmb, lag=1, beta=1.0, dt=dt)
    return rmb, w_m


def build(source_manifest, output_dir, fps=60.0, max_sequences=0):
    output_dir.mkdir(parents=True, exist_ok=False)
    files, _ = load_cache_files(source_manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = art.ParametricModel("models/SMPL_male.pkl", device=device)
    shards, total_sequences, total_frames = [], 0, 0
    for shard_idx, source_file in enumerate(files):
        data = torch.load(source_file, map_location="cpu", weights_only=False)
        count = len(data["name"])
        if max_sequences:
            count = min(count, max_sequences - total_sequences)
        if count <= 0:
            break
        out = {key: value[:count] if isinstance(value, list) else value for key, value in data.items()}
        out["RMB"], out["wM"] = [], []
        for i in range(count):
            rmb, w_m = fk_rmb_and_consistent_w(data["pose_gt"][i], model, device, 1.0 / fps)
            out["RMB"].append(rmb)
            out["wM"].append(w_m)
            print(f"[{total_sequences + i + 1}] {data['name'][i]}", flush=True)
        path = output_dir / f"baseline_cache_shard{shard_idx:05d}.pt"
        torch.save(out, path)
        frames = sum(len(x) for x in out["RMB"])
        shards.append({"path": str(path), "source_path": str(source_file), "num_sequences": count,
                       "num_frames": frames, "sha256": sha256(path)})
        total_sequences += count
        total_frames += frames
        if max_sequences and total_sequences >= max_sequences:
            break
    manifest = {
        "cache_type": "pl_va_amass_fk_rmb_w_consistent_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest), "source_manifest_sha256": sha256(source_manifest),
        "producer": "pl_va_state_amass_consistent_cache.py::FK pose_gt global IMU-joint rotations",
        "fps": fps, "dt": 1.0 / fps, "num_sequences": total_sequences, "num_frames": total_frames,
        "sensor_order": ["left_forearm", "right_forearm", "left_lower_leg", "right_lower_leg", "head", "pelvis"],
        "imu_joints": list(IMU_JOINTS),
        "RMB": "R_M_B from SMPL GT pose FK global rotations; no independent per-frame nR",
        "wM": "Log(RMB[t] RMB[t-1]^T)/dt in model/world frame; first frame zero",
        "preserved_fields": "all source fields except RMB and wM copied unchanged",
        "cache_files": shards,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": " ".join(__import__("sys").argv),
    }
    (output_dir / "baseline_cache_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--max-sequences", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(build(a.source_manifest, a.output_dir, a.fps, a.max_sequences), indent=2, default=str))


if __name__ == "__main__":
    main()
