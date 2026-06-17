import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import articulate as art
from acc_curve import acc_curve_features
from l4_q75_utils import q75_to_pose_tran
from l4_sensor_offset_utils import DT, IMU_JOINTS, COORDINATE_CONTRACT, official_imu_fields, smooth_centered
from l4_train_diverse_short import load_cache_files


DEFAULT_PRESETS = {
    "amass_train": {
        "input_cache": "data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json",
        "dataset": "AMASS",
        "split": "train",
    },
    "dip_train": {
        "input_cache": "data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json",
        "dataset": "DIP-IMU",
        "split": "train",
    },
    "dip_val": {
        "input_cache": "data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json",
        "dataset": "DIP-IMU",
        "split": "val",
    },
    "dip_test": {
        "input_cache": "data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json",
        "dataset": "DIP-IMU",
        "split": "test",
    },
}


def load_pose_tran(data, seq_idx):
    if data.get("pose_gt") and data["pose_gt"]:
        return data["pose_gt"][seq_idx].float(), data.get("tran_gt", [None])[seq_idx]
    if data.get("q75_gt") and data["q75_gt"]:
        return q75_to_pose_tran(data["q75_gt"][seq_idx].float())
    if data.get("pose") and data["pose"]:
        pose = art.math.axis_angle_to_rotation_matrix(data["pose"][seq_idx].float()).view(-1, 24, 3, 3)
        tran = data["tran"][seq_idx].float() if data.get("tran") and data["tran"] else None
        return pose, tran
    raise KeyError("source sequence has no pose_gt/q75_gt/pose field")


def load_offset(data, seq_idx):
    for key in ("offset_r", "imu_offset_r", "r_JS"):
        if data.get(key) and data[key]:
            return data[key][seq_idx].float()
    raise KeyError("source sequence has no offset_r/imu_offset_r/r_JS field")


@torch.no_grad()
def sensor_site_acc_from_pose(pose, tran, offset_r, body_model, device, batch_size):
    pose = pose.to(device)
    if tran is None:
        tran = pose.new_zeros(pose.shape[0], 3)
    tran = tran.float().to(device)
    offset_r = offset_r.to(device)
    positions = []
    for start in range(0, pose.shape[0], batch_size):
        p = pose[start:start + batch_size]
        t = tran[start:start + batch_size]
        grot, joints = body_model.forward_kinematics(p, tran=t, calc_mesh=False)[:2]
        p_wj = joints[:, IMU_JOINTS]
        r = offset_r.view(1, 6, 3, 1).expand(p.shape[0], -1, -1, -1)
        p_ws = p_wj + grot[:, IMU_JOINTS].matmul(r).squeeze(-1)
        positions.append(p_ws.detach().cpu())
    pos = torch.cat(positions, dim=0).float()
    acc = torch.full_like(pos, float("nan"))
    if pos.shape[0] >= 3:
        acc[1:-1] = (pos[:-2] - 2.0 * pos[1:-1] + pos[2:]) / (DT ** 2)
    return acc


def build_sequence(data, seq_idx, name, args, body_model, device):
    pose, tran = load_pose_tran(data, seq_idx)
    aM_raw, wM, RMB = official_imu_fields(data, seq_idx)
    offset_r = load_offset(data, seq_idx)
    n = min(pose.shape[0], aM_raw.shape[0], wM.shape[0], RMB.shape[0])
    if args.max_frames:
        n = min(n, int(args.max_frames))
    pose = pose[:n]
    tran = None if tran is None else tran[:n]
    if args.force_zero_tran:
        tran = torch.zeros(n, 3, dtype=pose.dtype)
    aM_raw, wM, RMB = aM_raw[:n], wM[:n], RMB[:n]
    aM_smooth = smooth_centered(aM_raw, args.smooth_window, mode=args.smoothing_mode)
    aFK = sensor_site_acc_from_pose(pose, tran, offset_r, body_model, device, args.fk_batch_size)
    aFK_smooth = smooth_centered(aFK, args.smooth_window, mode=args.smoothing_mode)
    valid = torch.isfinite(aFK_smooth).all(dim=(-1, -2))
    trim = int(args.trim)
    if trim > 0 and valid.shape[0] > 2 * trim:
        valid[:trim] = False
        valid[-trim:] = False
    aFK_smooth = torch.nan_to_num(aFK_smooth, nan=0.0)
    feature = acc_curve_features(aM_raw, aM_smooth, wM, RMB)
    if feature.shape[-1] != 108:
        raise RuntimeError(f"bad AccCurve feature dim {feature.shape[-1]}")
    tensors = (feature, aM_raw, aM_smooth, aFK_smooth, wM, RMB)
    if not all(torch.isfinite(x).all() for x in tensors):
        raise RuntimeError(f"non-finite AccCurve tensors for {name}")
    return {
        "name": str(name),
        "num_frames": int(n),
        "feature": feature.cpu(),
        "aM_raw": aM_raw.reshape(n, 18).cpu(),
        "aM_smooth": aM_smooth.reshape(n, 18).cpu(),
        "aFK_smooth": aFK_smooth.reshape(n, 18).cpu(),
        "valid_mask": valid.cpu(),
        "wM": wM.reshape(n, 18).cpu(),
        "RMB": RMB.cpu(),
        "offset_r": offset_r.cpu(),
    }


def empty_shard():
    return {
        "name": [],
        "num_frames": [],
        "feature": [],
        "aM_raw": [],
        "aM_smooth": [],
        "aFK_smooth": [],
        "valid_mask": [],
        "wM": [],
        "RMB": [],
        "offset_r": [],
    }


def append_record(shard, record):
    for key in shard:
        shard[key].append(record[key])


def prepare_output_dir(path, overwrite):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; pass --overwrite to rebuild intentionally.")
    if overwrite and path.exists():
        for child in path.iterdir():
            if child.is_dir():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def build_cache(args, preset_name=None):
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    files, source_manifest = load_cache_files(args.input_cache)
    device = torch.device(args.device)
    body_model = art.ParametricModel("models/SMPL_male.pkl", device=device)
    cache_files = []
    shard = empty_shard()
    total_sequences = 0
    total_frames = 0
    total_valid = 0
    started = time.time()

    def flush():
        nonlocal shard
        if not shard["name"]:
            return
        out = output_dir / f"acc_curve_cache_shard{len(cache_files):05d}.pt"
        torch.save(shard, out)
        cache_files.append({
            "path": str(out),
            "num_sequences": len(shard["name"]),
            "num_frames": int(sum(shard["num_frames"])),
        })
        shard = empty_shard()

    for cache_file in files:
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        for seq_idx, name in enumerate(data["name"]):
            if args.max_sequences and total_sequences >= args.max_sequences:
                break
            record = build_sequence(data, seq_idx, name, args, body_model, device)
            append_record(shard, record)
            total_sequences += 1
            total_frames += int(record["num_frames"])
            total_valid += int(record["valid_mask"].sum())
            if len(shard["name"]) >= args.shard_size:
                flush()
            if args.progress_every and total_sequences % args.progress_every == 0:
                print(json.dumps({
                    "processed_sequences": total_sequences,
                    "processed_frames": total_frames,
                    "valid_frames": total_valid,
                    "elapsed_sec": round(time.time() - started, 3),
                }))
        if args.max_sequences and total_sequences >= args.max_sequences:
            break
    flush()
    manifest = {
        "type": "acc_curve_cache_v1",
        "preset": preset_name,
        "dataset": args.dataset,
        "split": args.split,
        "source_cache": str(args.input_cache),
        "source_manifest": source_manifest,
        "cache_files": cache_files,
        "num_sequences": int(total_sequences),
        "num_frames": int(total_frames),
        "valid_frames": int(total_valid),
        "feature_dim": 108,
        "target_dim": 18,
        "dt": DT,
        "smooth_window": int(args.smooth_window),
        "smoothing_mode": args.smoothing_mode,
        "trim": int(args.trim),
        "force_zero_tran": bool(args.force_zero_tran),
        "target_translation_contract": (
            "tran forced to zero even if source cache contains tran"
            if args.force_zero_tran
            else "source tran is used when available; otherwise zero translation is used"
        ),
        "coordinate_contract": {
            "input_frame": "model/world frame M from GlobalPose aM/wM/RMB cache fields",
            "target_frame": (
                "model/world frame with root translation removed; p_WS = p_WJ + R_WJ @ rJS"
                if args.force_zero_tran
                else "same model/world frame M; aFK_smooth is ddot(p_WJ + R_WJ @ r_JS)"
            ),
            "offset": COORDINATE_CONTRACT,
        },
        "feature_layout": "aM_raw[18] + aM_smooth[18] + raw_minus_smooth[18] + wM[18] + RMB_6d[36]",
        "target_layout": "aFK_smooth[18], six sensor-site accelerations in m/s^2",
        "base_layout": "aM_smooth[18], same units and frame as target",
        "valid_mask": "finite centered-difference FK acceleration frames, with configured trim removed",
        "created_at_unix": time.time(),
    }
    manifest_path = output_dir / "acc_curve_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "num_sequences": total_sequences,
        "num_frames": total_frames,
        "valid_frames": total_valid,
        "elapsed_sec": round(time.time() - started, 3),
    }, indent=2))
    return manifest_path


def resolve_preset_args(base_args, preset_name):
    preset = DEFAULT_PRESETS[preset_name]
    args = argparse.Namespace(**vars(base_args))
    args.input_cache = preset["input_cache"]
    args.output_dir = str(Path(base_args.output_root) / preset_name)
    args.dataset = preset["dataset"]
    args.split = preset["split"]
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Build AccCurve input/target/base caches.")
    parser.add_argument("--preset", choices=sorted(DEFAULT_PRESETS))
    parser.add_argument("--all-defaults", action="store_true")
    parser.add_argument("--input-cache")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-root", default="code/outputs/smooth_acc_cache_amass_dip_20260617")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--split", default="custom")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--smoothing-mode", default="centered_moving_average")
    parser.add_argument("--trim", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--fk-batch-size", type=int, default=2048)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--force-zero-tran", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be odd")
    if args.all_defaults:
        for preset in DEFAULT_PRESETS:
            build_cache(resolve_preset_args(args, preset), preset)
        return
    if args.preset:
        build_cache(resolve_preset_args(args, args.preset), args.preset)
        return
    if not args.input_cache or not args.output_dir:
        raise SystemExit("--input-cache and --output-dir are required without --preset")
    build_cache(args, None)


if __name__ == "__main__":
    main()
