"""Frame audit for cached, legacy, new causal, and centered RMB angular velocity."""

import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import load_cache_files
from pl_va_state import (ANGULAR_VELOCITY_EMA_BETA, ANGULAR_VELOCITY_LAG,
                         ANGULAR_VELOCITY_METHOD, LEAF_NAMES,
                         causal_world_angular_velocity_from_rmb_sequence,
                         legacy_body_omega_to_world_frame,
                         legacy_lag1_body_angular_velocity_from_rmb_sequence)

SENSORS = LEAF_NAMES + ("pelvis_root",)
AXES = ("x", "y", "z")


def stats(x, y):
    d = x - y
    xf, yf = x.flatten(), y.flatten()
    xc, yc = xf - xf.mean(), yf - yf.mean()
    return {"rmse": float(torch.sqrt((d * d).mean())), "mean_l2": float(d.norm(dim=-1).mean()),
            "pearson": float((xc * yc).sum() / (xc.norm() * yc.norm()).clamp_min(1e-12)),
            "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=-1).mean())}


def detailed_stats(x, y):
    return {"overall": stats(x, y),
            "per_sensor": {name: stats(x[:, i], y[:, i]) for i, name in enumerate(SENSORS)},
            "per_axis": {axis: stats(x[..., i:i + 1], y[..., i:i + 1]) for i, axis in enumerate(AXES)}}


def centered_world_omega(rmb, dt=1 / 60):
    out = torch.zeros(rmb.shape[:-2] + (3,), dtype=rmb.dtype, device=rmb.device)
    if len(rmb) > 2:
        delta = rmb[2:].matmul(rmb[:-2].transpose(-1, -2))
        out[1:-1] = art.math.rotation_matrix_to_axis_angle(delta.reshape(-1, 3, 3)).reshape(
            delta.shape[:-2] + (3,)) / (2 * dt)
        out[0], out[-1] = out[1], out[-2]
    return out


def angular_step_quantiles(rmb):
    if len(rmb) < 2:
        return {}
    delta = rmb[1:].matmul(rmb[:-1].transpose(-1, -2))
    step = art.math.rotation_matrix_to_axis_angle(delta.reshape(-1, 3, 3)).reshape(delta.shape[:-2] + (3,)).norm(dim=-1)
    flat = step.flatten()
    return {"p50_rad": float(torch.quantile(flat, .50)), "p95_rad": float(torch.quantile(flat, .95)),
            "p99_rad": float(torch.quantile(flat, .99)), "max_rad": float(flat.max())}


def audit(manifest_path, max_sequences=4):
    files, _ = load_cache_files(manifest_path)
    values = {"cached_measured_wM": [], "legacy_lag1_noema_world": [],
              "new_lag2_ema03_world": [], "offline_centered_world": []}
    names, step_rows = [], []
    for file in files:
        data = torch.load(file, map_location="cpu")
        for i, name in enumerate(data["name"]):
            rmb, measured = data["RMB"][i].float(), data["wM"][i].float()
            legacy_body = legacy_lag1_body_angular_velocity_from_rmb_sequence(rmb)
            values["cached_measured_wM"].append(measured)
            values["legacy_lag1_noema_world"].append(legacy_body_omega_to_world_frame(legacy_body, rmb))
            values["new_lag2_ema03_world"].append(causal_world_angular_velocity_from_rmb_sequence(rmb))
            values["offline_centered_world"].append(centered_world_omega(rmb))
            names.append(str(name))
            group = "offset_overlay" if "offset" in str(name).lower() else "original"
            step_rows.append({"sequence": str(name), "group": group, **angular_step_quantiles(rmb)})
            if len(names) >= max_sequences:
                break
        if len(names) >= max_sequences:
            break
    cat = {key: torch.cat(rows) for key, rows in values.items()}
    measured = cat["cached_measured_wM"]
    comparisons = {key + "_vs_cached_measured_wM": detailed_stats(value[2:], measured[2:])
                   for key, value in cat.items() if key != "cached_measured_wM"}
    comparisons["new_lag2_ema03_world_vs_offline_centered_fk_rotation"] = detailed_stats(
        cat["new_lag2_ema03_world"][2:], cat["offline_centered_world"][2:])
    return {"manifest": str(manifest_path), "sequences": names,
            "frame_contract": "RMB=R_M_B; all comparisons in model/world M; PL conversion is wRB=wM@RMB_root",
            "method": ANGULAR_VELOCITY_METHOD, "lag": ANGULAR_VELOCITY_LAG,
            "ema_beta": ANGULAR_VELOCITY_EMA_BETA, "first_frames": "t=0,1 strict zero",
            "relative_rotation_order": "RMB_t @ RMB_t-2^T", "comparisons": comparisons,
            "angular_step_distribution_per_sequence": step_rows,
            "fk_reference": "not present in this GlobalPose cache; run the linked FK audit for FK comparisons"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", nargs=2, metavar=("LABEL", "MANIFEST"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-sequences", type=int, default=4)
    a = p.parse_args()
    result = {label: audit(path, a.max_sequences) for label, path in a.dataset}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
