import argparse
import json
from pathlib import Path

import torch
import articulate as art

from l4_tail_update_qstate import UniformCubicBSpline
from l4_train_diverse_short import DEVICE, load_records
from net import GPNet
from pl_curve import fit_uniform_cubic_spline_controls, normalize_gravity, pl_target_from_pose
from pl_curve_train import load_pl_curve_records


def summarize(values):
    values = torch.as_tensor(values).float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "count": 0}
    return {
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "std": float(finite.std(unbiased=False)) if finite.numel() > 1 else 0.0,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "count": int(finite.numel()),
    }


def gravity_angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred.float(), avoid_nan=True)
    target = art.math.normalize_tensor(target.float(), avoid_nan=True)
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def temporal_metrics(pred, target):
    pred_g = art.math.normalize_tensor(pred[..., 15:].float(), avoid_nan=True)
    target_g = art.math.normalize_tensor(target[..., 15:].float(), avoid_nan=True)
    out = {
        "gRdot_l2": torch.empty(0),
        "gRddot_l2": torch.empty(0),
    }
    if pred_g.shape[0] >= 2:
        out["gRdot_l2"] = ((pred_g[1:] - pred_g[:-1]) - (target_g[1:] - target_g[:-1])).norm(dim=-1).cpu()
    if pred_g.shape[0] >= 3:
        pred_ddot = pred_g[2:] - 2.0 * pred_g[1:-1] + pred_g[:-2]
        target_ddot = target_g[2:] - 2.0 * target_g[1:-1] + target_g[:-2]
        out["gRddot_l2"] = (pred_ddot - target_ddot).norm(dim=-1).cpu()
    return out


@torch.no_grad()
def audit(records, max_sequences=0):
    spline = UniformCubicBSpline().to(DEVICE)
    body_model = None
    rows = []
    selected = records[:max_sequences] if max_sequences else records
    for record in selected:
        if "pl_target" in record:
            target = normalize_gravity(record["pl_target"].float().to(DEVICE))
        else:
            if body_model is None:
                probe = GPNet().eval().to(DEVICE)
                body_model = art.ParametricModel("models/SMPL_male.pkl", vert_mask=probe.v_imu, device=DEVICE)
            pose = record["pose_gt"].float().to(DEVICE)
            target = normalize_gravity(pl_target_from_pose(pose, body_model).float())
        target_for_control = torch.cat((target[..., :15], target[..., 15:]), dim=-1)
        controls = fit_uniform_cubic_spline_controls(target_for_control)
        decoded = normalize_gravity(spline(controls.unsqueeze(0))[0])
        leaf = (decoded[..., :15].reshape(-1, 5, 3) - target[..., :15].reshape(-1, 5, 3)).norm(dim=-1) * 100.0
        grav = gravity_angle_deg(decoded[..., 15:], target[..., 15:])
        temporal = temporal_metrics(decoded, target)
        rows.append({
            "name": str(record["name"]),
            "num_frames": int(target.shape[0]),
            "target_shape": list(target.shape),
            "control_shape": list(controls.shape),
            "finite": bool(torch.isfinite(target).all() and torch.isfinite(controls).all() and torch.isfinite(decoded).all()),
            "pRB_error_cm_mean": float(leaf.mean().cpu()),
            "gR1_error_deg_mean": float(grav.mean().cpu()),
            "gRdot_l2_mean": float(temporal["gRdot_l2"].mean()) if temporal["gRdot_l2"].numel() else None,
            "gRddot_l2_mean": float(temporal["gRddot_l2"].mean()) if temporal["gRddot_l2"].numel() else None,
            "control_norm_mean": float(controls.norm(dim=-1).mean().cpu()),
            "control_gR_norm_mean": float(controls[..., 15:].norm(dim=-1).mean().cpu()),
        })
    p = torch.tensor([row["pRB_error_cm_mean"] for row in rows])
    g = torch.tensor([row["gR1_error_deg_mean"] for row in rows])
    gd = torch.tensor([row["gRdot_l2_mean"] for row in rows if row["gRdot_l2_mean"] is not None])
    gdd = torch.tensor([row["gRddot_l2_mean"] for row in rows if row["gRddot_l2_mean"] is not None])
    return {
        "status": "ok",
        "num_sequences": len(rows),
        "num_frames": int(sum(row["num_frames"] for row in rows)),
        "all_finite": all(row["finite"] for row in rows),
        "summary": {
            "decode_CGT_pRB_error_cm": summarize(p),
            "decode_CGT_gR1_error_deg": summarize(g),
            "decode_CGT_gRdot_l2": summarize(gd),
            "decode_CGT_gRddot_l2": summarize(gdd),
        },
        "rows": rows,
        "method": {
            "target": "concat(target_pRB, normalize(target_gR1))",
            "control_fit": "solve tridiagonal UniformCubicBSpline(C_GT) = target",
            "decode": "UniformCubicBSpline(C_GT), then normalize decoded gR1 for metrics",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Audit GT PL spline-control fitting quality.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=0)
    args = parser.parse_args()
    records, manifest = load_pl_curve_records(args.cache)
    result = audit(records, max_sequences=args.max_sequences)
    result["cache"] = str(args.cache)
    result["manifest"] = manifest
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "status": result["status"],
        "num_sequences": result["num_sequences"],
        "num_frames": result["num_frames"],
        "all_finite": result["all_finite"],
        "summary": result["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
