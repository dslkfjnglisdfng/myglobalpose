import argparse
import json
from pathlib import Path

import torch

from l4_train_diverse_short import load_records
from l4_train_loss_ablation import (
    DEVICE,
    DT,
    GRAVITY_M,
    IMU_JOINTS,
    adjust_acc_gravity,
    acceleration_residual_stats,
    offset_sensor_positions_from_pose,
)


def tensor_stats(residual):
    stats = acceleration_residual_stats(residual)
    out = {key: float(value.detach().cpu()) for key, value in stats.items()}
    out["smooth_l1"] = float(torch.nn.functional.smooth_l1_loss(
        residual,
        residual.new_zeros(residual.shape),
    ).detach().cpu())
    return out


def average_dicts(items):
    if not items:
        return {}
    keys = items[0].keys()
    return {key: sum(float(item[key]) for item in items) / len(items) for key in keys}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Audit gravity convention for offset-aware IMU acceleration proxy.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--view-filter", default="offset_aug", help="Substring required in record name; empty disables filtering.")
    args = parser.parse_args()

    records, manifest = load_records(args.cache)
    if args.view_filter:
        records = [record for record in records if args.view_filter in str(record.get("name", ""))]
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise RuntimeError("No records matched the requested cache/view filter.")

    modes = ("none", "plus_g", "minus_g")
    rows = []
    aggregates = {mode: [] for mode in modes}
    for record in records:
        pose = record["pose_gt"].to(DEVICE)
        tran = record["tran_gt"].to(DEVICE)
        if pose.shape[0] < 3:
            continue
        p_sensor = offset_sensor_positions_from_pose(pose, tran, record.get("offset_r"))
        acc_proxy = (p_sensor[2:] - 2.0 * p_sensor[1:-1] + p_sensor[:-2]) / (DT * DT)
        target = record["aM"][1:-1].to(DEVICE)
        row = {
            "name": record["name"],
            "frames": int(acc_proxy.shape[0]),
        }
        for mode in modes:
            residual = adjust_acc_gravity(acc_proxy, mode) - target
            stats = tensor_stats(residual)
            row[mode] = stats
            aggregates[mode].append(stats)
        rows.append(row)

    summary = {mode: average_dicts(aggregates[mode]) for mode in modes}
    selected_mode = min(summary, key=lambda mode: summary[mode]["imu_proxy_offset_acc_rms"])
    result = {
        "cache": args.cache,
        "cache_manifest": manifest,
        "num_records": len(rows),
        "view_filter": args.view_filter,
        "dt": DT,
        "gravity_model_frame": [float(v) for v in GRAVITY_M.detach().cpu()],
        "imu_joints": list(IMU_JOINTS),
        "formula": "p_S(t)=p_J(t)+R_J(t)@r_JS; a_proxy=(p_S[t+1]-2*p_S[t]+p_S[t-1]) / dt^2",
        "modes": {
            "none": "compare a_proxy vs aM",
            "plus_g": "compare a_proxy + g vs aM",
            "minus_g": "compare a_proxy - g vs aM",
        },
        "summary": summary,
        "selected_gravity_mode": selected_mode,
        "rows": rows,
        "test_set_used": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "output": str(output),
        "num_records": len(rows),
        "selected_gravity_mode": selected_mode,
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
