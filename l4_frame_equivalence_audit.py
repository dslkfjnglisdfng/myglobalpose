import argparse
import json
from pathlib import Path

import torch

from l4_rawlike_se3_calibration import robust_rotation_mean, rotation_angle_deg
from l4_sensor_offset_utils import SENSOR_NAMES, fk_imu_joints_and_vertices, load_dataset_file


def tensor_median(x):
    x = torch.as_tensor(x).float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan")
    return float(x.median().item())


def q(x, p):
    x = torch.as_tensor(x).float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x, p).item())


def compute_rmb(data, seq_idx):
    if "RMB" in data:
        return data["RMB"][seq_idx].float()
    RIM = data["RIM"][seq_idx].float()
    RIS = data["RIS"][seq_idx].float()
    RSB = data["RSB"][seq_idx].float()
    return RIM.transpose(1, 2).matmul(RIS).matmul(RSB)


def audit_sequence(data, seq_idx, args, source_label):
    pose = data["pose"][seq_idx].float()
    tran = data["tran"][seq_idx].float()
    RMB = compute_rmb(data, seq_idx)
    n = min(pose.shape[0], tran.shape[0], RMB.shape[0])
    if args.max_frames > 0:
        n = min(n, args.max_frames)
    pose, tran, RMB = pose[:n], tran[:n], RMB[:n]
    _, R_WJ, _ = fk_imu_joints_and_vertices(pose, tran, device=args.device)
    name = str(data["name"][seq_idx]) if "name" in data else f"{source_label}_{seq_idx}"
    rows = []
    for s, sensor_name in enumerate(SENSOR_NAMES):
        R_j = R_WJ[:, s]
        R_b = RMB[:, s]
        valid = torch.isfinite(R_j).all(dim=(-1, -2)) & torch.isfinite(R_b).all(dim=(-1, -2))
        R_j = R_j[valid]
        R_b = R_b[valid]
        if R_j.shape[0] < 8:
            continue
        delta = R_j.transpose(-1, -2).matmul(R_b)
        delta_mean = robust_rotation_mean(delta)
        raw_angle = rotation_angle_deg(delta)
        residual = rotation_angle_deg(delta_mean.T.view(1, 3, 3).matmul(delta))
        rows.append(
            {
                "sequence_id": name,
                "source_label": source_label,
                "sensor_id": s,
                "sensor_name": sensor_name,
                "num_frames": int(R_j.shape[0]),
                "raw_angle_median_deg": tensor_median(raw_angle),
                "raw_angle_p75_deg": q(raw_angle, 0.75),
                "raw_angle_p95_deg": q(raw_angle, 0.95),
                "fixed_delta_residual_median_deg": tensor_median(residual),
                "fixed_delta_residual_p75_deg": q(residual, 0.75),
                "fixed_delta_residual_p95_deg": q(residual, 0.95),
                "fixed_delta_angle_deg": float(rotation_angle_deg(delta_mean).item()),
            }
        )
    return rows


def summarize(rows):
    out = {
        "num_sequence_sensor_entries": len(rows),
        "overall": {},
        "per_sensor": [],
    }
    for key in (
        "raw_angle_median_deg",
        "raw_angle_p75_deg",
        "raw_angle_p95_deg",
        "fixed_delta_residual_median_deg",
        "fixed_delta_residual_p75_deg",
        "fixed_delta_residual_p95_deg",
        "fixed_delta_angle_deg",
    ):
        out["overall"][key] = tensor_median([r[key] for r in rows])
    for sensor in SENSOR_NAMES:
        rs = [r for r in rows if r["sensor_name"] == sensor]
        entry = {"sensor": sensor, "num_entries": len(rs)}
        for key in out["overall"]:
            entry[key] = tensor_median([r[key] for r in rs])
        if entry["fixed_delta_residual_p95_deg"] <= 5.0:
            entry["frame_equivalence"] = "equivalent_up_to_fixed_rotation"
        elif entry["fixed_delta_residual_p95_deg"] <= 10.0:
            entry["frame_equivalence"] = "mostly_equivalent_with_dynamic_noise"
        else:
            entry["frame_equivalence"] = "not_equivalent"
        out["per_sensor"].append(entry)
    out["decision"] = (
        "R_WJ_AND_RMB_EQUIVALENT_UP_TO_FIXED_ROTATION"
        if out["overall"]["fixed_delta_residual_p95_deg"] <= 5.0
        else "R_WJ_AND_RMB_HAVE_DYNAMIC_DIFFERENCES"
    )
    return out


def process_source(path, label, args):
    data = load_dataset_file(path)
    count = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    rows = []
    for i in range(count):
        rows.extend(audit_sequence(data, i, args, label))
        print(f"[{label} {i + 1}/{count}]", flush=True)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Audit FK R_WJ vs official TotalCapture body orientation RMB.")
    parser.add_argument("--output-dir", default="data/dataset_work/SensorOffset/frame_equivalence_audit")
    parser.add_argument("--official-train", default="data/dataset_work/TotalCapture_globalpose_official/train.pt")
    parser.add_argument("--official-val", default="data/dataset_work/TotalCapture_globalpose_official/val.pt")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(process_source(args.official_train, "official_train_source", args))
    rows.extend(process_source(args.official_val, "official_val_source", args))
    summary = {
        "config": vars(args),
        "contract": {
            "R_WJ": "SMPL FK global rotation returned by ParametricModel.forward_kinematics as grot[:, IMU_JOINTS]",
            "RMB": "official TotalCapture body/bone-to-model/world orientation, computed as RIM^T RIS RSB",
            "delta": "R_WJ^T RMB; fixed_delta_residual removes the robust per-sequence/sensor mean delta",
        },
        "summary": summarize(rows),
        "paths": {
            "rows": str(out_dir / "frame_equivalence_rows.json"),
            "summary": str(out_dir / "frame_equivalence_summary.json"),
        },
    }
    (out_dir / "frame_equivalence_rows.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "frame_equivalence_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
