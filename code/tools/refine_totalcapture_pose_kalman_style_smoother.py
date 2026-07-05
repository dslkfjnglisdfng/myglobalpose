#!/usr/bin/env python3
"""Simplified TotalCapture Kalman-style pose smoother.

This is an offline window optimization, not an EKF. It follows the smooth
protocol audit result and optimizes only smooth/low-frequency IMU acceleration,
never raw acceleration.
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import articulate as art  # noqa: E402
from l4_sensor_offset_utils import (  # noqa: E402
    FPS,
    GRAVITY_WORLD,
    IMU_JOINTS,
    SENSOR_NAMES,
    body_model,
    first_derivative,
    moving_average,
    second_derivative,
    sensor_to_joint_map,
)


DEFAULT_DATASET = ROOT / "data/dataset_work/TotalCapture_globalpose_official/test.pt"
DEFAULT_RJS = ROOT / "code/outputs/totalcapture_accfit_rjs_synthesis_20260704_171103/rjs_accfit_global.pt"
DEFAULT_OUT = ROOT / "code/outputs"
LOWER_BODY_JOINTS = (1, 2, 4, 5, 7, 8)
ROOT_JOINT = 0
FOOT_JOINTS = (10, 11)
HELDOUT_SENSOR_IDS = (2, 3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--rjs-path", type=Path, default=DEFAULT_RJS)
    p.add_argument("--rjs-method", default="savgol9_p3_fd")
    p.add_argument("--rjs-field", default="r_JS_projected")
    p.add_argument("--measurement", choices=("centered_ma21", "centered_ma15", "lowpass_3hz", "lowpass_5hz"), default="centered_ma21")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-sequences", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--trim", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lambda-acc", type=float, default=1.0)
    p.add_argument("--lambda-gyro", type=float, default=0.02)
    p.add_argument("--lambda-pose", type=float, default=500.0)
    p.add_argument("--lambda-tran", type=float, default=8000.0)
    p.add_argument("--lambda-vel", type=float, default=0.05)
    p.add_argument("--lambda-qdd", type=float, default=0.01)
    p.add_argument("--lambda-jerk", type=float, default=0.01)
    p.add_argument("--lambda-foot", type=float, default=0.02)
    p.add_argument("--max-pose-delta-deg", type=float, default=3.0)
    p.add_argument("--max-tran-delta", type=float, default=0.01)
    p.add_argument("--delta-smooth-window", type=int, default=21)
    return p.parse_args()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def matvec(R, v):
    return torch.matmul(R, v.unsqueeze(-1)).squeeze(-1)


def finite_slice(n, trim):
    trim = min(int(trim), max(1, (int(n) - 3) // 2))
    return slice(trim, int(n) - trim), trim


def load_rjs(path, method, field):
    payload = torch.load(path, map_location="cpu")
    if field not in payload or method not in payload[field]:
        raise KeyError(f"{path} missing {field}[{method}]")
    rjs = payload[field][method].float()
    R_JS = payload.get("R_JS", {}).get(method, torch.eye(3).repeat(rjs.shape[0], 1, 1)).float()
    if rjs.shape != (6, 3):
        raise ValueError(f"Expected global rJS [6,3], got {tuple(rjs.shape)}")
    return rjs, R_JS, payload.get("config", {})


def lowpass_zero_phase(x, cutoff_hz):
    from scipy.signal import butter, sosfiltfilt

    x = fill_nonfinite(x.float())
    flat = x.detach().cpu().reshape(x.shape[0], -1).numpy()
    sos = butter(2, float(cutoff_hz), btype="lowpass", fs=FPS, output="sos")
    y = sosfiltfilt(sos, flat, axis=0)
    return torch.from_numpy(y.copy()).reshape_as(x).to(dtype=x.dtype, device=x.device)


def fill_nonfinite(x):
    x = x.float().clone()
    flat = x.reshape(x.shape[0], -1)
    idx = torch.arange(x.shape[0])
    for col in range(flat.shape[1]):
        y = flat[:, col]
        finite = torch.isfinite(y)
        if bool(finite.all()):
            continue
        if int(finite.sum()) == 0:
            y.zero_()
            continue
        finite_idx = idx[finite]
        finite_y = y[finite]
        y[: int(finite_idx[0])] = finite_y[0]
        y[int(finite_idx[-1]) + 1 :] = finite_y[-1]
        bad = ~torch.isfinite(y)
        if bool(bad.any()):
            y[bad] = torch.from_numpy(np.interp(idx[bad].numpy(), finite_idx.numpy(), finite_y.numpy())).to(dtype=y.dtype)
    return flat.reshape_as(x)


def smooth_measurement(x, name):
    if name == "centered_ma21":
        return moving_average(fill_nonfinite(x), 21)
    if name == "centered_ma15":
        return moving_average(fill_nonfinite(x), 15)
    if name == "lowpass_3hz":
        return lowpass_zero_phase(x, 3.0)
    if name == "lowpass_5hz":
        return lowpass_zero_phase(x, 5.0)
    raise ValueError(name)


def differentiable_fk(pose_aa, tran, device):
    pose_R = art.math.axis_angle_to_rotation_matrix(pose_aa.reshape(-1, 3)).view(-1, 24, 3, 3)
    model = body_model(device)
    grot, joint = model.forward_kinematics(pose_R.to(device), None, tran.to(device), calc_mesh=False)
    return grot, joint


def predict(pose, tran, rjs, R_JS, R_WS_obs, measurement):
    R_all, J_all = differentiable_fk(pose, tran, pose.device)
    R_WJ = R_all[:, list(IMU_JOINTS)]
    p_WJ = J_all[:, list(IMU_JOINTS)]
    p_WS = p_WJ + matvec(R_WJ, rjs.to(pose.device).view(1, 6, 3))
    R_WS_fk = R_WJ.matmul(R_JS.to(pose.device).view(1, 6, 3, 3))
    acc_world = second_derivative(p_WS, fps=FPS, mode="centered")
    acc_sensor = matvec(R_WS_obs.transpose(-1, -2), acc_world - GRAVITY_WORLD.to(pose.device).view(1, 1, 3))
    return {
        "joints": J_all,
        "R_WS_fk": R_WS_fk,
        "acc_sensor": acc_sensor,
        "acc_sensor_smooth": smooth_measurement(acc_sensor, measurement),
    }


def angular_velocity_sensor(R_WS):
    R_dot = first_derivative(R_WS, fps=FPS, mode="centered")
    omega_hat_world = R_dot.matmul(R_WS.transpose(-1, -2))
    omega_hat_world = 0.5 * (omega_hat_world - omega_hat_world.transpose(-1, -2))
    omega_world = torch.stack(
        (omega_hat_world[..., 2, 1], omega_hat_world[..., 0, 2], omega_hat_world[..., 1, 0]),
        dim=-1,
    )
    return matvec(R_WS.transpose(-1, -2), omega_world)


def q_derivatives(pose, tran):
    q = torch.cat([tran, pose.reshape(pose.shape[0], -1)], dim=-1)
    qd = first_derivative(q, fps=FPS, mode="centered")
    qdd = second_derivative(q, fps=FPS, mode="centered")
    return q, qd, qdd


def init_vars(n, device):
    return {
        "delta_tran": torch.zeros(n, 3, device=device, requires_grad=True),
        "delta_pose": torch.zeros(n, 1 + len(LOWER_BODY_JOINTS), 3, device=device, requires_grad=True),
    }


def clean_pose(pose0, vars_):
    pose = pose0.clone().view(pose0.shape[0], 24, 3)
    pose[:, [ROOT_JOINT, *LOWER_BODY_JOINTS]] += vars_["delta_pose"]
    return pose.reshape_as(pose0)


def clamp_vars(vars_, args):
    with torch.no_grad():
        for key in ("delta_tran", "delta_pose"):
            vars_[key].nan_to_num_(0.0, posinf=0.0, neginf=0.0)
            if vars_[key].shape[0] >= int(args.delta_smooth_window):
                vars_[key].copy_(moving_average(vars_[key], int(args.delta_smooth_window)))
        vars_["delta_tran"].clamp_(-float(args.max_tran_delta), float(args.max_tran_delta))
        max_rad = math.radians(float(args.max_pose_delta_deg))
        norm = vars_["delta_pose"].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        vars_["delta_pose"].mul_((max_rad / norm).clamp_max(1.0))


def rmse(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def mean_norm(x):
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x).all(axis=-1)
    return float(np.linalg.norm(x[valid], axis=-1).mean()) if np.any(valid) else float("nan")


def contact_metric(joints):
    foot = joints[:, list(FOOT_JOINTS)]
    vel = first_derivative(foot, fps=FPS, mode="centered")
    sl, _ = finite_slice(foot.shape[0], 2)
    foot, vel = foot[sl], vel[sl]
    gate = foot[..., 1] <= (foot[..., 1].min() + 0.08)
    horiz = vel[..., [0, 2]].norm(dim=-1)
    return float(horiz[gate].mean().detach().cpu()) if int(gate.sum()) else float("nan")


def roughness(qdd):
    jerk = first_derivative(qdd, fps=FPS, mode="centered")
    return float(torch.nanmean(jerk.square()).sqrt().detach().cpu())


def prepare_sequence(data, idx, args):
    pose = data["pose"][idx].float()
    tran = data["tran"][idx].float()
    aS = data["aS"][idx].float()
    wS = data["wS"][idx].float()
    RIS = data["RIS"][idx].float()
    RIM = data["RIM"][idx].float()
    n = min(pose.shape[0], tran.shape[0], aS.shape[0], wS.shape[0], RIS.shape[0])
    if args.max_frames:
        n = min(n, int(args.max_frames))
    pose, tran, aS, wS, RIS, RIM = pose[:n], tran[:n], aS[:n], wS[:n], RIS[:n], RIM[:n]
    R_WS_obs = RIM.transpose(1, 2).unsqueeze(0).matmul(RIS)
    return {
        "name": str(data["name"][idx]),
        "n": int(n),
        "pose": pose,
        "tran": tran,
        "aS": aS,
        "wS": wS,
        "R_WS_obs": R_WS_obs.float(),
    }


def optimize_sequence(seq, rjs, R_JS, args):
    device = torch.device(args.device)
    pose0 = seq["pose"].to(device)
    tran0 = seq["tran"].to(device)
    aS0 = seq["aS"].to(device)
    wS = seq["wS"].to(device)
    R_WS_obs = seq["R_WS_obs"].to(device)
    aS_smooth = smooth_measurement(aS0, args.measurement)
    sl, _ = finite_slice(seq["n"], args.trim)
    vars_ = init_vars(seq["n"], device)
    opt = torch.optim.Adam(list(vars_.values()), lr=float(args.lr))
    last = {}
    for it in range(int(args.iterations)):
        opt.zero_grad(set_to_none=True)
        pose_clean = clean_pose(pose0, vars_)
        tran_clean = tran0 + vars_["delta_tran"]
        pred = predict(pose_clean, tran_clean, rjs, R_JS, R_WS_obs, args.measurement)
        gyro_pred = angular_velocity_sensor(pred["R_WS_fk"])
        _q, qd, qdd = q_derivatives(pose_clean, tran_clean)
        acc_res = pred["acc_sensor_smooth"][sl] - aS_smooth[sl]
        gyro_res = gyro_pred[sl] - wS[sl]
        jerk = first_derivative(qdd, fps=FPS, mode="centered")
        foot = pred["joints"][:, list(FOOT_JOINTS)]
        foot_vel = first_derivative(foot, fps=FPS, mode="centered")[sl]
        foot_h = foot[sl, :, 1]
        foot_gate = (foot_h <= (foot_h.detach().min() + 0.08)).float().detach()
        foot_loss = (foot_gate * foot_vel[..., [0, 2]].norm(dim=-1).square()).mean()
        loss = (
            float(args.lambda_acc) * torch.nanmean(acc_res.square())
            + float(args.lambda_gyro) * torch.nanmean(gyro_res.square())
            + float(args.lambda_pose) * vars_["delta_pose"].square().mean()
            + float(args.lambda_tran) * vars_["delta_tran"].square().mean()
            + float(args.lambda_vel) * torch.nanmean(qd[sl].square())
            + float(args.lambda_qdd) * torch.nanmean(qdd[sl].square())
            + float(args.lambda_jerk) * torch.nanmean(jerk[sl].square())
            + float(args.lambda_foot) * foot_loss
        )
        loss.backward()
        for var in vars_.values():
            if var.grad is not None:
                var.grad.nan_to_num_(0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(list(vars_.values()), 5.0)
        opt.step()
        clamp_vars(vars_, args)
        if it == int(args.iterations) - 1 or it % 20 == 0:
            last = {"iteration": it + 1, "loss": float(loss.detach().cpu())}
    with torch.no_grad():
        pose_clean = clean_pose(pose0, vars_)
        tran_clean = tran0 + vars_["delta_tran"]
        pred0 = predict(pose0, tran0, rjs, R_JS, R_WS_obs, args.measurement)
        pred1 = predict(pose_clean, tran_clean, rjs, R_JS, R_WS_obs, args.measurement)
        q0, qd0, qdd0 = q_derivatives(pose0, tran0)
        q1, qd1, qdd1 = q_derivatives(pose_clean, tran_clean)
    return {
        "pose_clean": pose_clean.detach().cpu(),
        "tran_clean": tran_clean.detach().cpu(),
        "qd_clean": qd1.detach().cpu(),
        "qdd_clean": qdd1.detach().cpu(),
        "pred_before": pred0,
        "pred_after": pred1,
        "q0": q0.detach().cpu(),
        "qd0": qd0.detach().cpu(),
        "qdd0": qdd0.detach().cpu(),
        "q1": q1.detach().cpu(),
        "qd1": qd1.detach().cpu(),
        "qdd1": qdd1.detach().cpu(),
        "debug": last,
    }


def metric_rows(seq, result, args):
    sl, _ = finite_slice(seq["n"], args.trim)
    target = smooth_measurement(seq["aS"], args.measurement)[sl].numpy()
    before = result["pred_before"]["acc_sensor_smooth"].detach().cpu()[sl].numpy()
    after = result["pred_after"]["acc_sensor_smooth"].detach().cpu()[sl].numpy()
    rows = []
    for group, ids in {"all": list(range(6)), "heldout_lower_leg": list(HELDOUT_SENSOR_IDS)}.items():
        rb = before[:, ids] - target[:, ids]
        ra = after[:, ids] - target[:, ids]
        rows.append(
            {
                "sequence_id": seq["name"],
                "sensor_group": group,
                "measurement": args.measurement,
                "before_acc_rmse": rmse(rb),
                "after_acc_rmse": rmse(ra),
                "delta_acc_rmse": rmse(ra) - rmse(rb),
                "before_acc_l2": mean_norm(rb.reshape(-1, 3)),
                "after_acc_l2": mean_norm(ra.reshape(-1, 3)),
            }
        )
    gyro0 = angular_velocity_sensor(result["pred_before"]["R_WS_fk"]).detach().cpu()[sl].numpy()
    gyro1 = angular_velocity_sensor(result["pred_after"]["R_WS_fk"]).detach().cpu()[sl].numpy()
    wS = seq["wS"][sl].numpy()
    pose_delta = (result["pose_clean"].view(-1, 24, 3) - seq["pose"].view(-1, 24, 3)).norm(dim=-1) * (180 / math.pi)
    tran_delta = (result["tran_clean"] - seq["tran"]).norm(dim=-1)
    rows.append(
        {
            "sequence_id": seq["name"],
            "sensor_group": "diagnostics",
            "measurement": args.measurement,
            "before_gyro_rmse": rmse(gyro0 - wS),
            "after_gyro_rmse": rmse(gyro1 - wS),
            "delta_gyro_rmse": rmse(gyro1 - wS) - rmse(gyro0 - wS),
            "mean_pose_delta_deg": float(pose_delta.mean()),
            "max_pose_delta_deg": float(pose_delta.max()),
            "mean_tran_delta": float(tran_delta.mean()),
            "max_tran_delta": float(tran_delta.max()),
            "before_jerk": roughness(result["qdd0"]),
            "after_jerk": roughness(result["qdd1"]),
            "delta_jerk": roughness(result["qdd1"]) - roughness(result["qdd0"]),
            "before_foot_sliding": contact_metric(result["pred_before"]["joints"].detach().cpu()),
            "after_foot_sliding": contact_metric(result["pred_after"]["joints"].detach().cpu()),
            "delta_foot_sliding": contact_metric(result["pred_after"]["joints"].detach().cpu()) - contact_metric(result["pred_before"]["joints"].detach().cpu()),
        }
    )
    return rows


def aggregate(rows):
    out = []
    for group in sorted({r["sensor_group"] for r in rows}):
        rs = [r for r in rows if r["sensor_group"] == group]
        keys = sorted({k for r in rs for k in r if k not in ("sequence_id", "sensor_group", "measurement")})
        item = {"sensor_group": group, "measurement": rs[0]["measurement"], "num_sequences": len({r["sequence_id"] for r in rs})}
        for key in keys:
            vals = [float(r[key]) for r in rs if r.get(key) not in ("", None) and math.isfinite(float(r[key]))]
            if vals:
                item[key] = float(np.mean(vals))
        out.append(item)
    return out


def plots(out, rows):
    seq_rows = [r for r in rows if r["sensor_group"] in ("all", "heldout_lower_leg")]
    labels = [f"{r['sequence_id']}:{r['sensor_group']}" for r in seq_rows]
    before = [float(r["before_acc_rmse"]) for r in seq_rows]
    after = [float(r["after_acc_rmse"]) for r in seq_rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(max(7, len(labels) * 0.9), 4))
    plt.bar(x - 0.2, before, 0.4, label="before")
    plt.bar(x + 0.2, after, 0.4, label="after")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("smooth acc RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "plots/acc_rmse_before_after.png", dpi=150)
    plt.close()


def summary_md(out, overall, args):
    all_row = next((r for r in overall if r["sensor_group"] == "all"), {})
    held_row = next((r for r in overall if r["sensor_group"] == "heldout_lower_leg"), {})
    diag = next((r for r in overall if r["sensor_group"] == "diagnostics"), {})
    verdict = "diagnostic"
    if (
        all_row.get("delta_acc_rmse", 1.0) < 0
        and held_row.get("delta_acc_rmse", 1.0) <= 0
        and diag.get("delta_gyro_rmse", 1.0) <= 0
        and diag.get("delta_jerk", 1.0) <= 0
        and diag.get("delta_foot_sliding", 1.0) <= 0
    ):
        verdict = "passes conservative gate"
    lines = [
        "# TotalCapture Kalman-Style Smoother",
        "",
        "Offline window optimization only; no EKF, no raw-acc objective, no network changes.",
        "",
        f"- Measurement: `{args.measurement}` sensor-frame specific force.",
        f"- rJS: `{args.rjs_path}` field `{args.rjs_field}` method `{args.rjs_method}`.",
        f"- Verdict: `{verdict}`.",
        "",
        "| group | before acc RMSE | after acc RMSE | delta acc RMSE |",
        "|---|---:|---:|---:|",
    ]
    for row in overall:
        if "before_acc_rmse" in row:
            lines.append(f"| {row['sensor_group']} | {row['before_acc_rmse']:.6f} | {row['after_acc_rmse']:.6f} | {row['delta_acc_rmse']:.6f} |")
    lines += [
        "",
        "Diagnostics are in `kalman_style_refinement_summary.csv`; refined trajectories are in `refined_sequences.pt`.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out = args.output_dir or (DEFAULT_OUT / f"totalcapture_kalman_style_smoother_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    data = torch.load(args.dataset, map_location="cpu")
    rjs, R_JS, rjs_config = load_rjs(args.rjs_path, args.rjs_method, args.rjs_field)
    count = len(data["name"])
    if args.max_sequences:
        count = min(count, int(args.max_sequences))
    rows, payload = [], []
    for idx in range(count):
        seq = prepare_sequence(data, idx, args)
        print(f"[opt] {seq['name']} frames={seq['n']}", flush=True)
        result = optimize_sequence(seq, rjs, R_JS, args)
        rows.extend(metric_rows(seq, result, args))
        payload.append(
            {
                "sequence_id": seq["name"],
                "pose_clean": result["pose_clean"],
                "tran_clean": result["tran_clean"],
                "qd_clean": result["qd_clean"],
                "qdd_clean": result["qdd_clean"],
                "debug": result["debug"],
            }
        )
    overall = aggregate(rows)
    write_csv(out / "kalman_style_refinement_summary.csv", overall)
    write_csv(out / "per_sequence_summary.csv", rows)
    write_json(
        out / "config.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "fps": FPS,
            "gravity_world": GRAVITY_WORLD.tolist(),
            "sensor_to_joint_map": sensor_to_joint_map(),
            "rjs_config": rjs_config,
            "contract": "R_WJ maps joint-local to world; r_JS is joint-local sensor origin; acc target is matched smooth sensor-frame specific force.",
        },
    )
    torch.save({"config": vars(args), "sequences": payload}, out / "refined_sequences.pt")
    plots(out, rows)
    summary_md(out, overall, args)
    print(json.dumps({"status": "ok", "output_dir": str(out), "summary": str(out / "SUMMARY.md")}, indent=2))


if __name__ == "__main__":
    main()
