import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from l4_sensor_offset_utils import GRAVITY_WORLD, SENSOR_NAMES, fk_imu_joints_and_vertices


DEFAULT_RANGES = torch.tensor(
    [
        [0.05, 0.03, 0.03],  # left_forearm
        [0.05, 0.03, 0.03],  # right_forearm
        [0.05, 0.03, 0.03],  # left_lower_leg
        [0.05, 0.03, 0.03],  # right_lower_leg
        [0.04, 0.04, 0.04],  # head
        [0.06, 0.04, 0.06],  # pelvis
    ],
    dtype=torch.float32,
)


def backup_once(path, suffix):
    backup = Path(str(path) + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def stack_if_list(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, list) and x and torch.is_tensor(x[0]):
        return torch.stack([_.float() for _ in x])
    raise TypeError(type(x))


def stable_sequence_seed(base_seed, name):
    digest = hashlib.sha1(str(name).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((int(base_seed) + offset) % (2**63 - 1))


def centered_second_difference_position(p_ws, fps):
    p_ws = p_ws.float()
    a = p_ws[:-2] - 2.0 * p_ws[1:-1] + p_ws[2:]
    a0 = 2.0 * p_ws[0] - 5.0 * p_ws[1] + 4.0 * p_ws[2] - p_ws[3]
    a1 = 2.0 * p_ws[-1] - 5.0 * p_ws[-2] + 4.0 * p_ws[-3] - p_ws[-4]
    return torch.cat((a0.unsqueeze(0), a, a1.unsqueeze(0)), dim=0) * float(fps) * float(fps)


def make_T(R, r):
    T = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(6, 1, 1)
    T[:, :3, :3] = R.float()
    T[:, :3, 3] = r.float()
    return T


def sample_offset(center, generator, ranges, min_norm, max_norm):
    noise = (torch.rand((6, 3), generator=generator) * 2.0 - 1.0) * ranges
    r = center.float() + noise.float()
    norm = r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    over = norm.squeeze(-1) > max_norm
    if over.any():
        r[over] = r[over] / norm[over] * max_norm
    under = r.norm(dim=-1) < min_norm
    if under.any():
        direction = r[under] / r[under].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        r[under] = direction * min_norm
    return r.float()


def resynthesize_raw_style_fields(aM, wM, RMB):
    gravity = GRAVITY_WORLD.view(1, 1, 3).to(aM.device)
    aS = RMB.transpose(-1, -2).matmul((aM - gravity).unsqueeze(-1)).squeeze(-1)
    wS = RMB.transpose(-1, -2).matmul(wM.unsqueeze(-1)).squeeze(-1)
    return aS.cpu().float(), wS.cpu().float()


def resynthesize_sequence(data, idx, args, ranges):
    name = data["name"][idx]
    pose = data["pose"][idx].float()
    tran = data["tran"][idx].float()
    old_aM = data["aM"][idx].float()
    old_wM = data["wM"][idx].float()
    old_RMB = data["RMB"][idx].float()
    if pose.shape[0] < 4:
        raise ValueError(f"{name}: need at least 4 frames for GlobalPose finite differences")

    if "original_imu_offset_r" in data:
        center = data["original_imu_offset_r"][idx].float()
    elif "imu_offset_r" in data:
        center = data["imu_offset_r"][idx].float()
    else:
        center = data["r_JS"][idx].float()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_sequence_seed(args.seed, name))
    r_new = sample_offset(center, generator, ranges, args.min_offset_norm, args.max_offset_norm)

    p_wj, R_wj, _ = fk_imu_joints_and_vertices(pose, tran, device=args.device)
    p_ws = p_wj + R_wj.matmul(r_new.view(1, 6, 3, 1)).squeeze(-1)
    aM_new = centered_second_difference_position(p_ws, args.fps).cpu().float()
    aS_new, wS_check = resynthesize_raw_style_fields(aM_new, old_wM, old_RMB)

    recon = centered_second_difference_position(
        p_wj + R_wj.matmul(r_new.view(1, 6, 3, 1)).squeeze(-1),
        args.fps,
    ).cpu().float()
    recon_err = (recon - aM_new).norm(dim=-1).max().item()

    a_diff = (aM_new - old_aM).norm(dim=-1)
    w_diff = (wS_check - data["wS"][idx].float()).norm(dim=-1) if "wS" in data else torch.zeros_like(a_diff)
    return {
        "r_new": r_new,
        "aM_new": aM_new,
        "aS_new": aS_new,
        "wS_new": wS_check,
        "recon_err": float(recon_err),
        "a_diff_mean": float(a_diff.mean().item()),
        "a_diff_median": float(a_diff.median().item()),
        "a_diff_p95": float(torch.quantile(a_diff.reshape(-1), 0.95).item()),
        "a_diff_max": float(a_diff.max().item()),
        "wS_change_max": float(w_diff.max().item()),
    }


def process_shard(path, args, ranges):
    data = torch.load(path, map_location="cpu")
    required = ("name", "pose", "tran", "aM", "wM", "RMB", "aS")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} missing required fields: {missing}")
    if "imu_offset_r" not in data and "r_JS" not in data:
        raise KeyError(f"{path} missing imu_offset_r/r_JS; run offset append first")

    backup = backup_once(path, ".bak_before_single_random_placement")
    n = len(data["name"])
    original_count = len(data["pose"])
    if not args.no_store_original and "original_imu_offset_r" not in data:
        data["original_imu_offset_r"] = [_.clone().float() for _ in data["imu_offset_r"]]

    r_list = []
    T_list = []
    a_diff_mean = []
    a_diff_median = []
    a_diff_p95 = []
    a_diff_max = []
    recon_err = []
    w_change_max = []

    for idx in range(n):
        item = resynthesize_sequence(data, idx, args, ranges)
        r_new = item["r_new"]
        r_list.append(r_new)
        if "imu_offset_R" in data:
            T_list.append(make_T(data["imu_offset_R"][idx], r_new))
        data["aM"][idx] = item["aM_new"]
        data["aS"][idx] = item["aS_new"]
        if not args.keep_existing_wS:
            data["wS"][idx] = item["wS_new"]
        a_diff_mean.append(item["a_diff_mean"])
        a_diff_median.append(item["a_diff_median"])
        a_diff_p95.append(item["a_diff_p95"])
        a_diff_max.append(item["a_diff_max"])
        recon_err.append(item["recon_err"])
        w_change_max.append(item["wS_change_max"])
        if args.verbose:
            print(f"[AMASS] {path.name} {idx + 1}/{n} {data['name'][idx]}")

    data["imu_offset_r"] = r_list
    data["r_JS"] = r_list
    if "imu_offset_R" in data:
        data["imu_offset_T"] = T_list
        data["T_JS"] = T_list
    data["placement_sampling_config"] = {
        "mode": "single_random_placement_per_sequence",
        "seed": int(args.seed),
        "fps": float(args.fps),
        "sensor_names": list(SENSOR_NAMES),
        "joint_local_uniform_ranges_m": ranges.tolist(),
        "min_offset_norm_m": float(args.min_offset_norm),
        "max_offset_norm_m": float(args.max_offset_norm),
        "resynthesized_fields": ["aM", "aS"] if args.keep_existing_wS else ["aM", "aS", "wS"],
        "unchanged_fields": ["pose", "tran", "RMB", "wM"],
        "acceleration_convention": "aM is world/model-frame second derivative of p_WS using GlobalPose IMUSimulator finite-difference endpoints; aS=RMB^T(aM-g).",
    }
    torch.save(data, path)

    r = torch.stack(r_list)
    return {
        "path": str(path),
        "backup": str(backup),
        "original_sequence_count": int(original_count),
        "final_sequence_count": int(len(data["pose"])),
        "imu_offset_r_shape": list(r.shape),
        "offset_norm_min": float(r.norm(dim=-1).min().item()),
        "offset_norm_median": float(r.norm(dim=-1).median().item()),
        "offset_norm_max": float(r.norm(dim=-1).max().item()),
        "aM_diff_mean": float(torch.tensor(a_diff_mean).mean().item()),
        "aM_diff_median": float(torch.tensor(a_diff_median).median().item()),
        "aM_diff_p95_median": float(torch.tensor(a_diff_p95).median().item()),
        "aM_diff_max": float(torch.tensor(a_diff_max).max().item()),
        "reconstruction_error_max": float(torch.tensor(recon_err).max().item()),
        "wS_change_max": float(torch.tensor(w_change_max).max().item()),
    }


def verify_shard(path):
    data = torch.load(path, map_location="cpu")
    n = len(data["pose"])
    r = stack_if_list(data["imu_offset_r"])
    assert r.shape == (n, 6, 3), (path, r.shape, n)
    assert len(data["aM"]) == n
    assert len(data["aS"]) == n
    return {
        "num_sequences": int(n),
        "r_shape": list(r.shape),
        "aM_first_shape": list(data["aM"][0].shape),
        "aS_first_shape": list(data["aS"][0].shape),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Resynthesize AMASS aM using one random joint-local IMU placement per sequence.")
    parser.add_argument("--amass-glob", default="data/dataset_work/AMASS/globalpose_synth_shard*.pt")
    parser.add_argument("--summary-path", default="data/dataset_work/SensorOffset/amass_single_random_placement_resynthesis_summary.json")
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--min-offset-norm", type=float, default=0.02)
    parser.add_argument("--max-offset-norm", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-store-original", action="store_true")
    parser.add_argument("--keep-existing-wS", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(Path().glob(args.amass_glob))
    if args.max_shards:
        paths = paths[: args.max_shards]
    if not paths:
        raise RuntimeError(f"No AMASS shards matched {args.amass_glob}")
    ranges = DEFAULT_RANGES.float()
    summary = {
        "config": {
            "seed": int(args.seed),
            "fps": float(args.fps),
            "sensor_names": list(SENSOR_NAMES),
            "joint_local_uniform_ranges_m": ranges.tolist(),
            "min_offset_norm_m": float(args.min_offset_norm),
            "max_offset_norm_m": float(args.max_offset_norm),
            "data_volume_increased": False,
            "totalcapture_touched": False,
        },
        "shards": [],
    }
    for path in paths:
        item = process_shard(path, args, ranges)
        item["verify"] = verify_shard(path)
        summary["shards"].append(item)
        print(
            f"[AMASS] wrote {path} n={item['final_sequence_count']} "
            f"norm med={item['offset_norm_median']:.4f} "
            f"aM diff med={item['aM_diff_median']:.4f}",
            flush=True,
        )

    norms = []
    diff_mean = []
    diff_median = []
    diff_p95 = []
    diff_max = []
    recon = []
    original_n = 0
    final_n = 0
    for item in summary["shards"]:
        data = torch.load(item["path"], map_location="cpu")
        norms.append(stack_if_list(data["imu_offset_r"]).norm(dim=-1))
        diff_mean.append(item["aM_diff_mean"])
        diff_median.append(item["aM_diff_median"])
        diff_p95.append(item["aM_diff_p95_median"])
        diff_max.append(item["aM_diff_max"])
        recon.append(item["reconstruction_error_max"])
        original_n += item["original_sequence_count"]
        final_n += item["final_sequence_count"]
    norms = torch.cat(norms, dim=0)
    summary["aggregate"] = {
        "num_shards": len(summary["shards"]),
        "original_sequence_count": int(original_n),
        "final_sequence_count": int(final_n),
        "offset_norm_min": float(norms.min().item()),
        "offset_norm_median": float(norms.median().item()),
        "offset_norm_max": float(norms.max().item()),
        "aM_diff_mean_mean": float(torch.tensor(diff_mean).mean().item()),
        "aM_diff_median_median": float(torch.tensor(diff_median).median().item()),
        "aM_diff_p95_median": float(torch.tensor(diff_p95).median().item()),
        "aM_diff_max": float(torch.tensor(diff_max).max().item()),
        "reconstruction_error_max": float(torch.tensor(recon).max().item()),
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
