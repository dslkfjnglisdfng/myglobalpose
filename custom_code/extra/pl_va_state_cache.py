"""Build auditable sequence-level PL-VA-State-V1 caches."""

import argparse
import json
from pathlib import Path

import torch
import articulate as art

from l4_train_diverse_short import load_cache_files
from pl_curve import pl_target_from_pose
from pl_curve import pl_input_feature
from pl_va_state import centered_derivative_targets, pl_va_feature_sequence


def iter_records(manifest):
    files, source = load_cache_files(manifest)
    for file in files:
        data = torch.load(file, map_location="cpu")
        for i, name in enumerate(data["name"]):
            yield str(file), str(name), {k: data[k][i] for k in ("pose_gt", "aM", "wM", "RMB")}
    return source


def build_cache(input_cache, output_dir, max_sequences=0, max_frames=0, fps=60.0,
                cutoff_hz=4.0, order=2, shard_size=50):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = art.ParametricModel("models/SMPL_male.pkl", vert_mask=(1961, 5424, 1176, 4662, 411, 3021), device=device)
    shard, files, count, frames, shard_idx = [], [], 0, 0, 0

    def flush():
        nonlocal shard, shard_idx
        if not shard:
            return
        path = output_dir / f"pl_va_state_shard{shard_idx:05d}.pt"
        torch.save(shard, path)
        files.append({"path": str(path), "num_sequences": len(shard), "num_frames": sum(x["length"] for x in shard)})
        shard, shard_idx = [], shard_idx + 1

    for source_file, name, record in iter_records(input_cache):
        if max_sequences and count >= max_sequences:
            break
        length = min(len(record["pose_gt"]), len(record["aM"]), len(record["RMB"]))
        if max_frames:
            length = min(length, max_frames)
        if length < 3:
            continue
        pose, a_m, w_m, rmb = (record[k][:length].float() for k in ("pose_gt", "aM", "wM", "RMB"))
        feature = pl_va_feature_sequence(a_m, rmb, dt=1.0 / fps, fs=fps, cutoff_hz=cutoff_hz, order=order)
        legacy_feature = torch.stack([pl_input_feature(a, w, r) for a, w, r in zip(a_m, w_m, rmb)])
        target = pl_target_from_pose(pose.to(device), body).float().cpu()
        target[:, 15:18] = art.math.normalize_tensor(target[:, 15:18], avoid_nan=True)
        v_gt, a_gt, derivative_mask = centered_derivative_targets(target[:, :15], 1.0 / fps)
        item = {"name": name, "source_file": source_file, "feature": feature, "p_gt": target[:, :15],
                "g_gt": target[:, 15:18], "v_gt": v_gt, "a_gt": a_gt,
                "valid_mask": torch.ones(length, dtype=torch.bool), "derivative_mask": derivative_mask,
                "init_legacy": target[0], "length": length, "legacy_feature": legacy_feature}
        if not all(torch.isfinite(v).all() for v in item.values() if torch.is_tensor(v)):
            raise RuntimeError(f"non-finite cache tensors for {name}")
        shard.append(item); count += 1; frames += length
        if len(shard) >= shard_size:
            flush()
    flush()
    manifest = {"type": "pl_va_state_v1_cache", "source_cache": str(input_cache), "fps": fps,
                "dt": 1.0 / fps, "num_sequences": count, "num_frames": frames, "cache_files": files,
                "input_layout": "aRB_raw[18]+wRB_rmbdiff[18]+RRB[45]+gR0[3]+aRB_smooth[18]",
                "target_layout": "pRB_vertex[15], vRB=d(pRB)/dt, aRB=d2(pRB)/dt2, gR1[3]",
                "derivatives": "centered interior; one-sided endpoints; mask recorded",
                "angular_velocity": {"source": "k2_so3_curve.SO3CurveStateDecoder._angular_motion",
                    "relative_order": "RMB_t^T @ RMB_t-1", "sign": "negative axis-angle / dt", "first_frame": "zero"},
                "smoothing": {"causal": True, "filter": "Butterworth SOS", "cutoff_hz": cutoff_hz, "order": order}}
    manifest["target_fk_device"] = str(device)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-cache", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-sequences", type=int, default=0); p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--fps", type=float, default=60.0); p.add_argument("--cutoff-hz", type=float, default=4.0)
    p.add_argument("--order", type=int, default=2); p.add_argument("--shard-size", type=int, default=50)
    a = p.parse_args(); print(json.dumps(build_cache(a.input_cache, a.output_dir, a.max_sequences, a.max_frames,
                                                     a.fps, a.cutoff_hz, a.order, a.shard_size), indent=2))


if __name__ == "__main__": main()
