import argparse
import json
import shutil
from pathlib import Path

import torch

from l4_rawlike_se3_calibration import robust_rotation_mean
from l4_sensor_offset_utils import IMU_JOINTS, fk_imu_joints_and_vertices


def backup_once(path):
    backup = Path(str(path) + ".bak_before_imu_offset")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def stack_if_list(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, list) and x and torch.is_tensor(x[0]):
        return torch.stack(x)
    raise TypeError(type(x))


def make_T(R, r):
    R = R.float()
    r = r.float()
    eye = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(R.shape[0], 1, 1)
    eye[:, :3, :3] = R
    eye[:, :3, 3] = r
    return eye


def compute_amass_offsets_for_sequence(pose, tran, joint, v_imu, RMB, device):
    _, R_wj, _ = fk_imu_joints_and_vertices(pose.float(), tran.float(), device=device)
    p_wj = joint[:, IMU_JOINTS].float()
    p_ws = v_imu.float()
    r_frames = R_wj.transpose(-1, -2).matmul((p_ws - p_wj).unsqueeze(-1)).squeeze(-1)
    r = r_frames.median(dim=0).values.float()
    if RMB is not None:
        R_samples = R_wj.transpose(-1, -2).matmul(RMB.float())
        R = torch.stack([robust_rotation_mean(R_samples[:, s]) for s in range(6)]).float()
    else:
        R = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(6, 1, 1)
    return r, R, make_T(R, r)


def append_amass_shard(path, args):
    data = torch.load(path, map_location="cpu")
    required = ("pose", "tran", "joint", "v_imu", "name")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} missing {missing}")
    backup = backup_once(path)
    r_list, R_list, T_list = [], [], []
    for i, name in enumerate(data["name"]):
        r, R, T = compute_amass_offsets_for_sequence(
            data["pose"][i],
            data["tran"][i],
            data["joint"][i],
            data["v_imu"][i],
            data["RMB"][i] if "RMB" in data else None,
            args.device,
        )
        r_list.append(r)
        R_list.append(R)
        T_list.append(T)
        if args.verbose:
            print(f"[AMASS] {path.name} {i + 1}/{len(data['name'])} {name}")
    data["imu_offset_r"] = r_list
    data["r_JS"] = r_list
    data["imu_offset_R"] = R_list
    data["R_JS"] = R_list
    data["imu_offset_T"] = T_list
    data["T_JS"] = T_list
    torch.save(data, path)
    norms = torch.stack(r_list).norm(dim=-1)
    return {
        "path": str(path),
        "backup": str(backup),
        "num_sequences": len(data["name"]),
        "imu_offset_r_shape": [len(r_list), 6, 3],
        "imu_offset_R_shape": [len(R_list), 6, 3, 3],
        "offset_norm_min": float(norms.min().item()),
        "offset_norm_median": float(norms.median().item()),
        "offset_norm_max": float(norms.max().item()),
    }


def load_totalcapture_offset_cache(path):
    cache = torch.load(path, map_location="cpu")
    names = list(cache["sequence_id"])
    mapping = {}
    for i, name in enumerate(names):
        R = cache["R_JS"][i].float()
        r = cache["r_JS"][i].float()
        mapping[name] = {
            "r": r,
            "R": R,
            "T": make_T(R, r),
        }
    return mapping


def append_totalcapture(path, offset_map):
    data = torch.load(path, map_location="cpu")
    if "name" not in data:
        raise KeyError(f"{path} missing name")
    missing = [name for name in data["name"] if name not in offset_map]
    if missing:
        raise RuntimeError(f"{path} missing offsets for sequences: {missing}")
    backup = backup_once(path)
    r_list = [offset_map[name]["r"] for name in data["name"]]
    R_list = [offset_map[name]["R"] for name in data["name"]]
    T_list = [offset_map[name]["T"] for name in data["name"]]
    data["imu_offset_r"] = r_list
    data["r_JS"] = r_list
    data["imu_offset_R"] = R_list
    data["R_JS"] = R_list
    data["imu_offset_T"] = T_list
    data["T_JS"] = T_list
    torch.save(data, path)
    norms = torch.stack(r_list).norm(dim=-1)
    return {
        "path": str(path),
        "backup": str(backup),
        "num_sequences": len(data["name"]),
        "imu_offset_r_shape": [len(r_list), 6, 3],
        "imu_offset_R_shape": [len(R_list), 6, 3, 3],
        "offset_norm_min": float(norms.min().item()),
        "offset_norm_median": float(norms.median().item()),
        "offset_norm_max": float(norms.max().item()),
        "missing_sequences": missing,
    }


def verify_file(path):
    data = torch.load(path, map_location="cpu")
    n = len(data["name"])
    r = stack_if_list(data["imu_offset_r"])
    R = stack_if_list(data["imu_offset_R"])
    T = stack_if_list(data["imu_offset_T"])
    assert r.shape == (n, 6, 3), (path, r.shape, n)
    assert R.shape == (n, 6, 3, 3), (path, R.shape, n)
    assert T.shape == (n, 6, 4, 4), (path, T.shape, n)
    norms = r.norm(dim=-1)
    return {
        "path": str(path),
        "num_sequences": n,
        "r_shape": list(r.shape),
        "R_shape": list(R.shape),
        "T_shape": list(T.shape),
        "offset_norm_min": float(norms.min().item()),
        "offset_norm_median": float(norms.median().item()),
        "offset_norm_max": float(norms.max().item()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Append IMU offset fields to processed AMASS and TotalCapture caches.")
    parser.add_argument("--amass-glob", default="data/dataset_work/AMASS/globalpose_synth_shard*.pt")
    parser.add_argument("--totalcapture-train", default="data/dataset_work/TotalCapture_globalpose_official/train.pt")
    parser.add_argument("--totalcapture-val", default="data/dataset_work/TotalCapture_globalpose_official/val.pt")
    parser.add_argument("--totalcapture-offset-cache", default="data/dataset_work/SensorOffset/rawlike_se3_candidate_a_v1/totalcapture_full_sequence_se3_cache.pt")
    parser.add_argument("--summary-path", default="data/dataset_work/SensorOffset/append_imu_offsets_summary.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-amass-shards", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    amass_paths = sorted(Path().glob(args.amass_glob))
    if args.max_amass_shards:
        amass_paths = amass_paths[: args.max_amass_shards]
    if not amass_paths:
        raise RuntimeError(f"No AMASS shards matched {args.amass_glob}")
    summary = {
        "amass": [],
        "totalcapture": [],
        "fields_added": ["imu_offset_r", "r_JS", "imu_offset_R", "R_JS", "imu_offset_T", "T_JS"],
        "missing_sequences": [],
    }
    for path in amass_paths:
        item = append_amass_shard(path, args)
        item["verify"] = verify_file(path)
        summary["amass"].append(item)
        print(
            f"[AMASS] wrote {path} n={item['num_sequences']} "
            f"norm med={item['offset_norm_median']:.4f}",
            flush=True,
        )
    offset_map = load_totalcapture_offset_cache(Path(args.totalcapture_offset_cache))
    for path in (Path(args.totalcapture_train), Path(args.totalcapture_val)):
        item = append_totalcapture(path, offset_map)
        item["verify"] = verify_file(path)
        summary["totalcapture"].append(item)
        summary["missing_sequences"].extend(item["missing_sequences"])
        print(
            f"[TotalCapture] wrote {path} n={item['num_sequences']} "
            f"norm med={item['offset_norm_median']:.4f}",
            flush=True,
        )
    amass_norms = []
    for item in summary["amass"]:
        v = torch.load(item["path"], map_location="cpu")["imu_offset_r"]
        amass_norms.append(stack_if_list(v).norm(dim=-1))
    amass_norms = torch.cat(amass_norms, dim=0)
    tc_norms = []
    for item in summary["totalcapture"]:
        v = torch.load(item["path"], map_location="cpu")["imu_offset_r"]
        tc_norms.append(stack_if_list(v).norm(dim=-1))
    tc_norms = torch.cat(tc_norms, dim=0)
    summary["aggregate"] = {
        "amass_num_shards": len(summary["amass"]),
        "amass_num_sequences": int(amass_norms.shape[0]),
        "amass_offset_norm_min": float(amass_norms.min().item()),
        "amass_offset_norm_median": float(amass_norms.median().item()),
        "amass_offset_norm_max": float(amass_norms.max().item()),
        "totalcapture_num_sequences": int(tc_norms.shape[0]),
        "totalcapture_offset_norm_min": float(tc_norms.min().item()),
        "totalcapture_offset_norm_median": float(tc_norms.median().item()),
        "totalcapture_offset_norm_max": float(tc_norms.max().item()),
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
