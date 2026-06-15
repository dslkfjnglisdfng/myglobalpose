import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imu_position_offset import OFFSET_POSITION_CONTRACT, load_offset_cache, prepare_sequence
from l4_sensor_offset_utils import GRAVITY_WORLD
from l4_rawlike_se3_calibration import matvec


def finite_stats(values):
    values = torch.as_tensor(values).float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def sequence_lever_arm_residual(seq, offset):
    offset = offset.float().view(6, 3)
    rows = []
    per_sensor_residuals = []
    per_sensor_zero = []
    for sensor_idx in range(6):
        R_ws_t = seq["R_wj"][:, sensor_idx].matmul(seq["R_JS"][sensor_idx]).transpose(-1, -2)
        joint_acc_sensor = matvec(R_ws_t, seq["ddot_p_wj"][:, sensor_idx] - GRAVITY_WORLD.view(1, 3))
        lever_sensor = seq["ddot_R_wj"][:, sensor_idx].matmul(offset[sensor_idx].view(3, 1)).squeeze(-1)
        pred = joint_acc_sensor + lever_sensor
        residual = (seq["aM"][:, sensor_idx] - pred).norm(dim=-1)
        zero_residual = (seq["aM"][:, sensor_idx] - joint_acc_sensor).norm(dim=-1)
        valid = torch.isfinite(residual) & torch.isfinite(zero_residual)
        residual = residual[valid]
        zero_residual = zero_residual[valid]
        improvement = zero_residual - residual
        rows.append({
            "sensor_idx": sensor_idx,
            "offset_norm_m": float(offset[sensor_idx].norm()),
            "num_valid_frames": int(valid.sum()),
            "residual_mps2": finite_stats(residual),
            "zero_residual_mps2": finite_stats(zero_residual),
            "improvement_mps2": finite_stats(improvement),
        })
        per_sensor_residuals.append(residual.mean() if residual.numel() else torch.tensor(float("nan")))
        per_sensor_zero.append(zero_residual.mean() if zero_residual.numel() else torch.tensor(float("nan")))
    residuals = torch.stack(per_sensor_residuals)
    zero = torch.stack(per_sensor_zero)
    return {
        "sensor_rows": rows,
        "residual_mps2_mean_over_sensors": float(residuals[torch.isfinite(residuals)].mean()) if torch.isfinite(residuals).any() else None,
        "zero_residual_mps2_mean_over_sensors": float(zero[torch.isfinite(zero)].mean()) if torch.isfinite(zero).any() else None,
        "improvement_mps2_mean_over_sensors": float((zero - residuals)[torch.isfinite(zero - residuals)].mean()) if torch.isfinite(zero - residuals).any() else None,
    }


def load_offsets(label_path_items):
    caches = {}
    for item in label_path_items:
        if "=" not in item:
            raise ValueError(f"Expected label=path, got {item}")
        label, path = item.split("=", 1)
        caches[label] = {
            "path": path,
            "offsets": load_offset_cache(path),
        }
    return caches


def eval_offsets(args):
    data = torch.load(args.input, map_location="cpu")
    count = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    caches = load_offsets(args.offset_cache)
    rows = []
    missing = []
    for seq_idx in range(count):
        seq = prepare_sequence(
            data,
            seq_idx,
            device=args.device,
            smooth_window=args.smooth_window,
            derivative_mode=args.derivative_mode,
        )
        seq_row = {"name": seq["name"], "num_frames": int(seq["pose"].shape[0]), "methods": {}}
        for label, cache in caches.items():
            offset = cache["offsets"].get(seq["name"])
            if offset is None:
                missing.append({"method": label, "name": seq["name"]})
                continue
            seq_row["methods"][label] = sequence_lever_arm_residual(seq, offset)
        rows.append(seq_row)
        print(json.dumps({"idx": seq_idx + 1, "count": count, "name": seq["name"]}), flush=True)

    aggregate = {}
    for label in caches:
        residual = []
        zero_residual = []
        improvement = []
        norms = []
        for row in rows:
            method = row["methods"].get(label)
            if not method:
                continue
            value = method["residual_mps2_mean_over_sensors"]
            zero_value = method["zero_residual_mps2_mean_over_sensors"]
            improvement_value = method["improvement_mps2_mean_over_sensors"]
            if value is not None:
                residual.append(value)
            if zero_value is not None:
                zero_residual.append(zero_value)
            if improvement_value is not None:
                improvement.append(improvement_value)
            for sensor_row in method["sensor_rows"]:
                norms.append(sensor_row["offset_norm_m"])
        aggregate[label] = {
            "cache_path": caches[label]["path"],
            "sequence_residual_mps2": finite_stats(residual),
            "sequence_zero_residual_mps2": finite_stats(zero_residual),
            "sequence_improvement_mps2": finite_stats(improvement),
            "offset_norm_m": finite_stats(norms),
        }

    result = {
        "status": "ok",
        "input": str(args.input),
        "dataset": args.dataset,
        "num_sequences": count,
        "coordinate_contract": OFFSET_POSITION_CONTRACT,
        "diagnostic_note": (
            "This evaluates forward lever-arm acceleration consistency only. "
            "It is not real offset GT accuracy, and DIP trans is not used as trusted supervision."
        ),
        "config": {
            "smooth_window": args.smooth_window,
            "derivative_mode": args.derivative_mode,
            "device": args.device,
            "max_sequences": args.max_sequences,
        },
        "missing": missing,
        "aggregate": aggregate,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate IMU position offsets by forward lever-arm acceleration consistency.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", default="other")
    parser.add_argument("--offset-cache", action="append", required=True, help="label=path to an offset cache .pt file")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--derivative-mode", choices=("legacy", "centered", "strict_centered"), default="centered")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = eval_offsets(args)
    print(json.dumps({
        "status": result["status"],
        "output_json": str(args.output_json),
        "num_sequences": result["num_sequences"],
        "aggregate": result["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
