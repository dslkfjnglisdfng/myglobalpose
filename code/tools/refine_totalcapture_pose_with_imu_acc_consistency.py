#!/usr/bin/env python3
"""TotalCapture IMU-aware pose label refinement diagnostic.

This script refines TotalCapture pose/tran near the original labels by fitting
FK-derived IMU-site specific force to observed IMU acceleration. It is a data
diagnostic only: no network training and no PL/IK/VR changes.
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
    official_imu_fields,
    pose_to_rotation_matrices,
    moving_average,
    savgol_smooth,
    second_derivative,
    sensor_to_joint_map,
)


DEFAULT_DATASET = ROOT / "data/dataset_work/TotalCapture_globalpose_official/test.pt"
DEFAULT_RJS = ROOT / "code/outputs/totalcapture_accfit_rjs_synthesis_20260704_171103/rjs_accfit_global.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "code/outputs"
LOWER_BODY_JOINTS = (1, 2, 4, 5, 7, 8)
ROOT_JOINT = 0
FOOT_JOINTS = (10, 11)
FIT_SENSOR_DEFAULT = ("left_forearm", "right_forearm", "head")
HELDOUT_SENSOR_DEFAULT = ("left_lower_leg", "right_lower_leg")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--rjs-path", default=str(DEFAULT_RJS))
    p.add_argument("--rjs-method", default="savgol9_p3_fd")
    p.add_argument("--rjs-field", default="r_JS_projected")
    p.add_argument("--mode", choices=("A", "B", "C", "all"), default="C")
    p.add_argument("--output-dir", default="")
    p.add_argument("--max-sequences", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--trim", type=int, default=4)
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--device", default="cpu")
    p.add_argument("--robust-loss", choices=("huber", "charbonnier"), default="huber")
    p.add_argument("--huber-delta", type=float, default=1.5)
    p.add_argument("--charbonnier-eps", type=float, default=1e-3)
    p.add_argument("--fit-sensors", default=",".join(FIT_SENSOR_DEFAULT))
    p.add_argument("--heldout-sensors", default=",".join(HELDOUT_SENSOR_DEFAULT))
    p.add_argument("--lambda-acc", type=float, default=1.0)
    p.add_argument("--lambda-gyro", type=float, default=0.05)
    p.add_argument("--lambda-pose", type=float, default=200.0)
    p.add_argument("--lambda-tran", type=float, default=5000.0)
    p.add_argument("--lambda-vel", type=float, default=0.02)
    p.add_argument("--lambda-acc-smooth", type=float, default=0.005)
    p.add_argument("--lambda-jerk", type=float, default=0.001)
    p.add_argument("--lambda-contact", type=float, default=0.005)
    p.add_argument("--max-pose-delta-deg", type=float, default=8.0)
    p.add_argument("--max-tran-delta", type=float, default=0.02)
    p.add_argument("--delta-smooth-window", type=int, default=9)
    p.add_argument("--save-frame-level", action="store_true")
    return p.parse_args()


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2))


def sensor_ids(names_csv):
    names = [x.strip() for x in names_csv.split(",") if x.strip()]
    lookup = {name: i for i, name in enumerate(SENSOR_NAMES)}
    unknown = [x for x in names if x not in lookup]
    if unknown:
        raise ValueError(f"Unknown sensors {unknown}; choices={list(SENSOR_NAMES)}")
    return [lookup[x] for x in names]


def robust_loss(x, args):
    if args.robust_loss == "huber":
        ax = x.abs()
        d = float(args.huber_delta)
        return torch.where(ax <= d, 0.5 * ax.square(), d * (ax - 0.5 * d))
    return torch.sqrt(x.square() + float(args.charbonnier_eps) ** 2) - float(args.charbonnier_eps)


def rmse_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x * x)))


def mean_norm_np(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    valid = np.isfinite(x).all(axis=-1)
    if not np.any(valid):
        return float("nan")
    return float(np.linalg.norm(x[valid], axis=-1).mean())


def matvec(R, v):
    return torch.matmul(R, v.unsqueeze(-1)).squeeze(-1)


def finite_slice(n, trim):
    trim = min(int(trim), max(0, (int(n) - 3) // 2))
    return slice(trim, int(n) - trim), trim


def load_rjs(path, method, field):
    payload = torch.load(path, map_location="cpu")
    if field not in payload or method not in payload[field]:
        raise KeyError(f"{path} missing {field}[{method}]")
    rjs = payload[field][method].float()
    R_JS = payload.get("R_JS", {}).get(method, torch.eye(3).repeat(rjs.shape[0], 1, 1)).float()
    if rjs.dim() != 2 or rjs.shape != (6, 3):
        raise ValueError(f"Expected global rJS shape (6,3), got {tuple(rjs.shape)}")
    return rjs, R_JS, payload.get("config", {})


def output_dir(args):
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = DEFAULT_OUTPUT_ROOT / f"totalcapture_imu_aware_pose_refinement_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    return out


def differentiable_fk(pose_aa, tran, device):
    if pose_aa.dim() == 4 and pose_aa.shape[-2:] == (3, 3):
        pose_R = pose_aa
    else:
        pose_R = art.math.axis_angle_to_rotation_matrix(pose_aa.reshape(-1, 3)).view(-1, 24, 3, 3)
    model = body_model(device)
    grot, joint = model.forward_kinematics(pose_R.to(device), None, tran.to(device), calc_mesh=False)
    return grot, joint


def rotation_matrix_to_axis_angle_torch(R):
    R = R.float()
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1.0) * 0.5
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos)
    vee = torch.stack(
        (
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ),
        dim=-1,
    )
    scale = theta / (2.0 * torch.sin(theta)).clamp_min(1e-6)
    return vee * scale.unsqueeze(-1)


def pose_delta_deg(R_orig, R_clean):
    rel = R_orig.transpose(-1, -2).matmul(R_clean)
    aa = rotation_matrix_to_axis_angle_torch(rel.reshape(-1, 3, 3)).view(*rel.shape[:-2], 3)
    return aa.norm(dim=-1) * (180.0 / math.pi)


def angular_velocity_sensor(R_WS, fps):
    R_dot = first_derivative(R_WS, fps=fps, mode="centered")
    omega_hat_world = R_dot.matmul(R_WS.transpose(-1, -2))
    omega_hat_world = 0.5 * (omega_hat_world - omega_hat_world.transpose(-1, -2))
    omega_world = torch.stack(
        (
            omega_hat_world[..., 2, 1],
            omega_hat_world[..., 0, 2],
            omega_hat_world[..., 1, 0],
        ),
        dim=-1,
    )
    return matvec(R_WS.transpose(-1, -2), omega_world)


def q_derivatives(pose_aa, tran):
    q = torch.cat([tran, pose_aa.reshape(pose_aa.shape[0], -1)], dim=-1)
    qd = first_derivative(q, fps=FPS, mode="centered")
    qdd = second_derivative(q, fps=FPS, mode="centered")
    return q, qd, qdd


def make_clean_pose_axis(pose_orig, delta_vars, mode):
    if mode == "A":
        return pose_orig
    original_shape = pose_orig.shape
    clean = pose_orig.clone().view(pose_orig.shape[0], 24, 3)
    joints = [ROOT_JOINT, *LOWER_BODY_JOINTS]
    clean[:, joints] = clean[:, joints] + delta_vars["delta_pose"]
    return clean.reshape(original_shape)


def make_clean_pose_rot(pose_orig, delta_vars, mode):
    R_orig = pose_to_rotation_matrices(pose_orig)
    if mode == "A":
        return R_orig
    joints = [ROOT_JOINT, *LOWER_BODY_JOINTS]
    delta = delta_vars["delta_pose"]
    R_delta = art.math.axis_angle_to_rotation_matrix(delta.reshape(-1, 3)).view(delta.shape[0], len(joints), 3, 3)
    R_clean = R_orig.clone()
    R_clean[:, joints] = R_orig[:, joints].matmul(R_delta)
    return R_clean


def init_vars(n, mode, device):
    vars_ = {"delta_tran": torch.zeros(n, 3, device=device, requires_grad=True)}
    if mode != "A":
        vars_["delta_pose"] = torch.zeros(n, 1 + len(LOWER_BODY_JOINTS), 3, device=device, requires_grad=True)
    return vars_


def clamp_vars(vars_, args):
    with torch.no_grad():
        window = int(args.delta_smooth_window)
        vars_["delta_tran"].nan_to_num_(0.0, posinf=0.0, neginf=0.0)
        if window > 1 and vars_["delta_tran"].shape[0] >= window:
            vars_["delta_tran"].copy_(moving_average(vars_["delta_tran"], window))
        vars_["delta_tran"].clamp_(-float(args.max_tran_delta), float(args.max_tran_delta))
        if "delta_pose" in vars_:
            vars_["delta_pose"].nan_to_num_(0.0, posinf=0.0, neginf=0.0)
            if window > 1 and vars_["delta_pose"].shape[0] >= window:
                vars_["delta_pose"].copy_(moving_average(vars_["delta_pose"], window))
            max_rad = math.radians(float(args.max_pose_delta_deg))
            norm = vars_["delta_pose"].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = (max_rad / norm).clamp_max(1.0)
            vars_["delta_pose"].mul_(scale)


def predict_imu(pose_aa, tran, rJS, R_JS, device):
    R_WJ_all, p_WJ_all = differentiable_fk(pose_aa, tran, device)
    R_WJ = R_WJ_all[:, list(IMU_JOINTS)]
    p_WJ = p_WJ_all[:, list(IMU_JOINTS)]
    p_WS = p_WJ + matvec(R_WJ, rJS.to(device).view(1, 6, 3))
    R_WS_fk = R_WJ.matmul(R_JS.to(device).view(1, 6, 3, 3))
    acc_world = second_derivative(p_WS, fps=FPS, mode="centered")
    acc_sensor = None
    return {
        "R_WJ_all": R_WJ_all,
        "p_WJ_all": p_WJ_all,
        "R_WS_fk": R_WS_fk,
        "p_WS": p_WS,
        "acc_world": acc_world,
        "acc_sensor": acc_sensor,
    }


def sensor_specific_force(acc_world, R_WS_obs):
    return matvec(R_WS_obs.transpose(-1, -2), acc_world - GRAVITY_WORLD.to(acc_world.device).view(1, 1, 3))


def contact_metrics(joints):
    foot = joints[:, list(FOOT_JOINTS)]
    vel = first_derivative(foot, fps=FPS, mode="centered")
    acc = second_derivative(foot, fps=FPS, mode="centered")
    sl, _ = finite_slice(foot.shape[0], 2)
    foot = foot[sl]
    vel = vel[sl]
    acc = acc[sl]
    h = foot[..., 1]
    gate = h <= (h.min() + 0.08)
    horiz_vel = vel[..., [0, 2]].norm(dim=-1)
    horiz_acc = acc[..., [0, 2]].norm(dim=-1)
    if int(gate.sum()) == 0:
        return {"foot_sliding": float("nan"), "stance_foot_velocity": float("nan"), "stance_foot_acceleration": float("nan")}
    return {
        "foot_sliding": float(horiz_vel[gate].mean().detach().cpu()),
        "stance_foot_velocity": float(horiz_vel[gate].mean().detach().cpu()),
        "stance_foot_acceleration": float(horiz_acc[gate].mean().detach().cpu()),
    }


def smoothness_metrics(qd, qdd):
    jerk = first_derivative(qdd, fps=FPS, mode="centered")
    return {
        "qd_roughness": float(torch.nanmean(qd.square()).sqrt().detach().cpu()),
        "qdd_roughness": float(torch.nanmean(qdd.square()).sqrt().detach().cpu()),
        "jerk": float(torch.nanmean(jerk.square()).sqrt().detach().cpu()),
        "high_frequency_energy": high_frequency_energy(qdd.detach().cpu().numpy()),
    }


def high_frequency_energy(x):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 8:
        return float("nan")
    x = np.nan_to_num(x - np.nanmean(x, axis=0, keepdims=True))
    spec = np.fft.rfft(x.reshape(x.shape[0], -1), axis=0)
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / FPS)
    total = np.mean(np.abs(spec) ** 2)
    high = np.mean(np.abs(spec[freqs >= 10.0]) ** 2) if np.any(freqs >= 10.0) else 0.0
    return float(high / max(total, 1e-12))


def optimize_sequence(seq, rJS, R_JS, mode, args, fit_ids):
    device = torch.device(args.device)
    pose0 = seq["pose"].to(device)
    tran0 = seq["tran"].to(device)
    aS = seq["aS"].to(device)
    wS = seq["wS"].to(device)
    R_WS_obs = seq["R_WS_obs"].to(device)
    sl, trim = finite_slice(seq["n"], args.trim)
    opt_mode = "B" if mode == "C" else mode
    vars_ = init_vars(seq["n"], opt_mode, device)
    opt = torch.optim.Adam(list(vars_.values()), lr=float(args.lr))
    last = {}
    for it in range(int(args.iterations)):
        opt.zero_grad(set_to_none=True)
        pose_clean_axis = make_clean_pose_axis(pose0, vars_, opt_mode)
        tran_clean = tran0 + vars_["delta_tran"]
        pred = predict_imu(pose_clean_axis, tran_clean, rJS, R_JS, device)
        accS_pred = sensor_specific_force(pred["acc_world"], R_WS_obs)
        gyro_pred = angular_velocity_sensor(pred["R_WS_fk"], FPS)
        acc_res = accS_pred[sl][:, fit_ids] - aS[sl][:, fit_ids]
        gyro_res = gyro_pred[sl][:, fit_ids] - wS[sl][:, fit_ids]
        q, qd, qdd = q_derivatives(pose_clean_axis, tran_clean)
        loss_acc = robust_loss(acc_res, args).mean()
        loss_gyro = gyro_res.square().mean()
        loss_tran = vars_["delta_tran"].square().mean()
        loss_pose = torch.zeros((), device=device)
        if "delta_pose" in vars_:
            loss_pose = vars_["delta_pose"].square().mean()
        loss_vel = qd[sl].square().mean()
        loss_acc_smooth = qdd[sl].square().mean()
        jerk = first_derivative(qdd, fps=FPS, mode="centered")
        loss_jerk = jerk[sl].square().mean()
        contact = torch.zeros((), device=device)
        if float(args.lambda_contact) > 0.0:
            foot = pred["p_WJ_all"][:, list(FOOT_JOINTS)]
            foot_vel = first_derivative(foot, fps=FPS, mode="centered")
            foot = foot[sl]
            foot_vel = foot_vel[sl]
            h = foot[..., 1]
            gate = (h <= (h.detach().min() + 0.08)).float().detach()
            contact = (gate * foot_vel[..., [0, 2]].norm(dim=-1).square()).mean()
        loss = (
            float(args.lambda_acc) * loss_acc
            + float(args.lambda_gyro) * loss_gyro
            + float(args.lambda_pose) * loss_pose
            + float(args.lambda_tran) * loss_tran
            + float(args.lambda_vel) * loss_vel
            + float(args.lambda_acc_smooth) * loss_acc_smooth
            + float(args.lambda_jerk) * loss_jerk
            + float(args.lambda_contact) * contact
        )
        loss.backward()
        for var in vars_.values():
            if var.grad is not None:
                var.grad.nan_to_num_(0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(list(vars_.values()), max_norm=10.0)
        opt.step()
        clamp_vars(vars_, args)
        if it == int(args.iterations) - 1 or it % 20 == 0:
            last = {
                "loss": float(loss.detach().cpu()),
                "loss_acc": float(loss_acc.detach().cpu()),
                "loss_gyro": float(loss_gyro.detach().cpu()),
                "loss_pose": float(loss_pose.detach().cpu()),
                "loss_tran": float(loss_tran.detach().cpu()),
                "iteration": it + 1,
            }
    with torch.no_grad():
        pose_clean_axis = make_clean_pose_axis(pose0, vars_, opt_mode)
        tran_clean = tran0 + vars_["delta_tran"]
        pred = predict_imu(pose_clean_axis, tran_clean, rJS, R_JS, device)
        accS_pred = sensor_specific_force(pred["acc_world"], R_WS_obs)
        q, qd, qdd = q_derivatives(pose_clean_axis, tran_clean)
    return {
        "pose_clean": pose_clean_axis.detach().cpu(),
        "tran_clean": tran_clean.detach().cpu(),
        "qd_clean": qd.detach().cpu(),
        "qdd_clean": qdd.detach().cpu(),
        "acc_clean": accS_pred.detach().cpu(),
        "acc_world_clean": pred["acc_world"].detach().cpu(),
        "joints_clean": pred["p_WJ_all"].detach().cpu(),
        "R_WS_fk": pred["R_WS_fk"].detach().cpu(),
        "debug": last,
    }


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
    with torch.no_grad():
        R0, j0 = differentiable_fk(pose.to(args.device), tran.to(args.device), torch.device(args.device))
        q0, qd0, qdd0 = q_derivatives(pose.to(args.device), tran.to(args.device))
    return {
        "idx": idx,
        "name": str(data["name"][idx]) if "name" in data else f"seq_{idx}",
        "n": int(n),
        "pose": pose,
        "tran": tran,
        "aS": aS,
        "wS": wS,
        "R_WS_obs": R_WS_obs.float(),
        "joints_orig": j0.detach().cpu().float(),
        "R_orig_all": R0.detach().cpu().float(),
        "qd_orig": qd0.detach().cpu().float(),
        "qdd_orig": qdd0.detach().cpu().float(),
    }


def original_prediction(seq, rJS, R_JS, args):
    with torch.no_grad():
        pred = predict_imu(seq["pose"].to(args.device), seq["tran"].to(args.device), rJS, R_JS, torch.device(args.device))
        accS = sensor_specific_force(pred["acc_world"], seq["R_WS_obs"].to(args.device))
    return {
        "acc": accS.detach().cpu(),
        "acc_world": pred["acc_world"].detach().cpu(),
        "joints": pred["p_WJ_all"].detach().cpu(),
        "R_WS_fk": pred["R_WS_fk"].detach().cpu(),
    }


def sensor_metric_rows(seq, mode, original, refined, args, fit_ids, heldout_ids):
    rows = []
    sl, _ = finite_slice(seq["n"], args.trim)
    obs_raw = seq["aS"][sl]
    before_raw = original["acc"][sl]
    after_raw = refined["acc_clean"][sl]
    variants = {
        "raw": (obs_raw, before_raw, after_raw),
        "savgol9_p3": (savgol_smooth(obs_raw, 9, 3), savgol_smooth(before_raw, 9, 3), savgol_smooth(after_raw, 9, 3)),
        "savgol15_p3": (savgol_smooth(obs_raw, 15, 3), savgol_smooth(before_raw, 15, 3), savgol_smooth(after_raw, 15, 3)),
    }
    for signal_filter, (obs_t, before_t, after_t) in variants.items():
        obs = obs_t.numpy()
        before = before_t.numpy()
        after = after_t.numpy()
        for sid, name in enumerate(SENSOR_NAMES):
            rb = before[:, sid] - obs[:, sid]
            ra = after[:, sid] - obs[:, sid]
            group = "fit" if sid in fit_ids else "heldout" if sid in heldout_ids else "other"
            for label, residual in (("original", rb), ("refined", ra)):
                rows.append(
                    {
                        "mode": mode,
                        "sequence_id": seq["name"],
                        "signal_filter": signal_filter,
                        "sensor_id": sid,
                        "sensor_name": name,
                        "sensor_group": group,
                        "variant": label,
                        "acc_rmse": rmse_np(residual),
                        "acc_mean_norm": mean_norm_np(residual),
                    }
                )
            rows.append(
                {
                    "mode": mode,
                    "sequence_id": seq["name"],
                    "signal_filter": signal_filter,
                    "sensor_id": sid,
                    "sensor_name": name,
                    "sensor_group": group,
                    "variant": "delta",
                    "acc_rmse": rmse_np(ra) - rmse_np(rb),
                    "acc_mean_norm": mean_norm_np(ra) - mean_norm_np(rb),
                }
            )
    return rows


def sequence_summary(seq, mode, original, refined, args, fit_ids, heldout_ids):
    sl, _ = finite_slice(seq["n"], args.trim)
    rows = []
    groups = {
        "all": list(range(6)),
        "fit": fit_ids,
        "heldout": heldout_ids,
    }
    variants = {
        "raw": (seq["aS"][sl], original["acc"][sl], refined["acc_clean"][sl]),
        "savgol9_p3": (
            savgol_smooth(seq["aS"][sl], 9, 3),
            savgol_smooth(original["acc"][sl], 9, 3),
            savgol_smooth(refined["acc_clean"][sl], 9, 3),
        ),
        "savgol15_p3": (
            savgol_smooth(seq["aS"][sl], 15, 3),
            savgol_smooth(original["acc"][sl], 15, 3),
            savgol_smooth(refined["acc_clean"][sl], 15, 3),
        ),
    }
    for signal_filter, (obs_t, before_t, after_t) in variants.items():
        obs = obs_t.numpy()
        before = before_t.numpy()
        after = after_t.numpy()
        for group, ids in groups.items():
            rb = before[:, ids] - obs[:, ids]
            ra = after[:, ids] - obs[:, ids]
            rows.append(
                {
                    "mode": mode,
                    "sequence_id": seq["name"],
                    "signal_filter": signal_filter,
                    "sensor_group": group,
                    "original_acc_rmse": rmse_np(rb),
                    "refined_acc_rmse": rmse_np(ra),
                    "delta_acc_rmse": rmse_np(ra) - rmse_np(rb),
                }
            )
    return rows


def gyro_summary(seq, mode, original, refined, args):
    sl, _ = finite_slice(seq["n"], args.trim)
    gyro_orig = angular_velocity_sensor(original["R_WS_fk"], FPS)[sl].numpy()
    gyro_ref = angular_velocity_sensor(refined["R_WS_fk"], FPS)[sl].numpy()
    obs = seq["wS"][sl].numpy()
    return {
        "mode": mode,
        "sequence_id": seq["name"],
        "original_gyro_rmse": rmse_np(gyro_orig - obs),
        "refined_gyro_rmse": rmse_np(gyro_ref - obs),
        "delta_gyro_rmse": rmse_np(gyro_ref - obs) - rmse_np(gyro_orig - obs),
    }


def pose_summary(seq, mode, refined):
    p0 = seq["pose"].view(seq["pose"].shape[0], 24, 3)
    p1 = refined["pose_clean"].view(refined["pose_clean"].shape[0], 24, 3)
    deg = (p1 - p0).norm(dim=-1).detach().cpu() * (180.0 / math.pi)
    trans = (refined["tran_clean"] - seq["tran"]).norm(dim=-1)
    lower = deg[:, list(LOWER_BODY_JOINTS)]
    rows = []
    rows.append(
        {
            "mode": mode,
            "sequence_id": seq["name"],
            "joint_id": "ALL",
            "joint_name": "ALL",
            "mean_pose_delta_deg": float(deg.mean()),
            "max_pose_delta_deg": float(deg.max()),
            "lower_body_pose_delta_deg": float(lower.mean()),
            "mean_trans_delta": float(trans.mean()),
            "max_trans_delta": float(trans.max()),
        }
    )
    for jid in range(24):
        rows.append(
            {
                "mode": mode,
                "sequence_id": seq["name"],
                "joint_id": jid,
                "joint_name": f"joint_{jid}",
                "mean_pose_delta_deg": float(deg[:, jid].mean()),
                "max_pose_delta_deg": float(deg[:, jid].max()),
                "lower_body_pose_delta_deg": float(deg[:, jid].mean()) if jid in LOWER_BODY_JOINTS else 0.0,
                "mean_trans_delta": "",
                "max_trans_delta": "",
            }
        )
    return rows


def aggregate_overall(per_seq_rows, gyro_rows, pose_rows, smooth_rows, contact_rows):
    rows = []
    modes = sorted({r["mode"] for r in per_seq_rows})
    for mode in modes:
        seq_all = [r for r in per_seq_rows if r["mode"] == mode and r["sensor_group"] == "all" and r.get("signal_filter") == "raw"]
        fit = [r for r in per_seq_rows if r["mode"] == mode and r["sensor_group"] == "fit" and r.get("signal_filter") == "raw"]
        held = [r for r in per_seq_rows if r["mode"] == mode and r["sensor_group"] == "heldout" and r.get("signal_filter") == "raw"]
        pose_all = [r for r in pose_rows if r["mode"] == mode and r["joint_id"] == "ALL"]
        gy = [r for r in gyro_rows if r["mode"] == mode]
        sm_o = [r for r in smooth_rows if r["mode"] == mode and r["variant"] == "original"]
        sm_r = [r for r in smooth_rows if r["mode"] == mode and r["variant"] == "refined"]
        co_o = [r for r in contact_rows if r["mode"] == mode and r["variant"] == "original"]
        co_r = [r for r in contact_rows if r["mode"] == mode and r["variant"] == "refined"]
        def avg(rows_, key):
            vals = [float(r[key]) for r in rows_ if r.get(key) not in ("", None) and math.isfinite(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        rows.append(
            {
                "mode": mode,
                "original_acc_rmse": avg(seq_all, "original_acc_rmse"),
                "refined_acc_rmse": avg(seq_all, "refined_acc_rmse"),
                "delta_acc_rmse": avg(seq_all, "delta_acc_rmse"),
                "fit_sensor_delta_acc_rmse": avg(fit, "delta_acc_rmse"),
                "heldout_sensor_delta_acc_rmse": avg(held, "delta_acc_rmse"),
                "mean_pose_delta_deg": avg(pose_all, "mean_pose_delta_deg"),
                "max_pose_delta_deg": avg(pose_all, "max_pose_delta_deg"),
                "lower_body_pose_delta_deg": avg(pose_all, "lower_body_pose_delta_deg"),
                "mean_trans_delta": avg(pose_all, "mean_trans_delta"),
                "max_trans_delta": avg(pose_all, "max_trans_delta"),
                "original_gyro_rmse": avg(gy, "original_gyro_rmse"),
                "refined_gyro_rmse": avg(gy, "refined_gyro_rmse"),
                "delta_gyro_rmse": avg(gy, "delta_gyro_rmse"),
                "original_jerk": avg(sm_o, "jerk"),
                "refined_jerk": avg(sm_r, "jerk"),
                "delta_jerk": avg(sm_r, "jerk") - avg(sm_o, "jerk"),
                "original_foot_sliding": avg(co_o, "foot_sliding"),
                "refined_foot_sliding": avg(co_r, "foot_sliding"),
                "delta_foot_sliding": avg(co_r, "foot_sliding") - avg(co_o, "foot_sliding"),
            }
        )
    return rows


def plot_bar(path, rows, key_before, key_after, label):
    if not rows:
        return
    labels = [r.get("mode", r.get("sensor_name", "")) for r in rows]
    before = [float(r[key_before]) for r in rows]
    after = [float(r[key_after]) for r in rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(max(5, len(labels) * 0.8), 4))
    plt.bar(x - 0.18, before, width=0.36, label="original")
    plt.bar(x + 0.18, after, width=0.36, label="refined")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel(label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_plots(out, overall, per_sensor, pose_rows, trans_examples, gyro_rows, heldout_rows, example):
    plot_bar(out / "plots/acc_rmse_before_after.png", overall, "original_acc_rmse", "refined_acc_rmse", "acc RMSE")
    plot_bar(out / "plots/gyro_residual_before_after.png", gyro_rows, "original_gyro_rmse", "refined_gyro_rmse", "gyro RMSE")
    plot_bar(out / "plots/heldout_sensor_rmse.png", heldout_rows, "original_acc_rmse", "refined_acc_rmse", "held-out acc RMSE")
    joint_rows = [r for r in pose_rows if r["joint_id"] != "ALL"]
    if joint_rows:
        by_joint = {}
        for r in joint_rows:
            by_joint.setdefault(int(r["joint_id"]), []).append(float(r["mean_pose_delta_deg"]))
        xs = sorted(by_joint)
        ys = [np.mean(by_joint[x]) for x in xs]
        plt.figure(figsize=(8, 4))
        plt.bar(xs, ys)
        plt.xlabel("SMPL joint id")
        plt.ylabel("mean pose delta deg")
        plt.tight_layout()
        plt.savefig(out / "plots/pose_delta_per_joint.png", dpi=150)
        plt.close()
    if trans_examples:
        t = np.arange(len(trans_examples[0])) / FPS
        plt.figure(figsize=(8, 4))
        for i, v in enumerate(trans_examples[:3]):
            plt.plot(t, v, label=f"seq{i}")
        plt.ylabel("translation delta norm (m)")
        plt.xlabel("time (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "plots/trans_delta_timeseries.png", dpi=150)
        plt.close()
    if example:
        t = np.arange(example["obs"].shape[0]) / FPS
        plt.figure(figsize=(9, 4))
        plt.plot(t, example["obs"][:, 0], label="obs x")
        plt.plot(t, example["before"][:, 0], label="original x")
        plt.plot(t, example["after"][:, 0], label="refined x")
        plt.xlabel("time (s)")
        plt.ylabel("specific force")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "plots/example_acc_timeseries_before_after.png", dpi=150)
        plt.close()
    jerk_rows = [r for r in overall if "original_jerk" in r]
    plot_bar(out / "plots/jerk_before_after.png", jerk_rows, "original_jerk", "refined_jerk", "jerk")


def summary_md(out, overall, args, fit_ids, heldout_ids):
    lines = [
        "# TotalCapture IMU-aware Pose Refinement",
        "",
        "Diagnostic only: no network training and no PL/IK/VR changes.",
        "",
        "## Coordinate Contract",
        "",
        "- `R_WJ` maps joint-local vectors into world coordinates.",
        "- `r_JS` is the IMU sensor origin relative to mapped joint origin, expressed in joint-local coordinates.",
        "- `R_WS = R_WJ @ R_JS` maps sensor-frame vectors into world coordinates.",
        "- `aS_pred = R_WS_obs^T @ (d2(p_WJ + R_WJ @ r_JS)/dt2 - gravity_world)`.",
        "",
        "## Default Sensor Split",
        "",
        f"- Fit sensors: {', '.join(SENSOR_NAMES[i] for i in fit_ids)}",
        f"- Held-out sensors: {', '.join(SENSOR_NAMES[i] for i in heldout_ids)}",
        "",
        "## Overall Metrics",
        "",
        "| mode | acc RMSE orig | acc RMSE refined | held-out delta | pose mean deg | trans mean m | gyro delta | jerk delta | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in overall:
        verdict = "diagnostic"
        if (
            r["delta_acc_rmse"] < 0
            and r["heldout_sensor_delta_acc_rmse"] <= 0
            and r["delta_gyro_rmse"] <= 0
            and r["delta_jerk"] <= 0
            and r["mean_pose_delta_deg"] <= 2.0
            and r["mean_trans_delta"] <= 0.02
        ):
            verdict = "passes conservative gate"
        elif r["delta_acc_rmse"] < 0 and r["heldout_sensor_delta_acc_rmse"] > 0:
            verdict = "overfit risk"
        lines.append(
            f"| {r['mode']} | {r['original_acc_rmse']:.6f} | {r['refined_acc_rmse']:.6f} | "
            f"{r['heldout_sensor_delta_acc_rmse']:.6f} | {r['mean_pose_delta_deg']:.6f} | "
            f"{r['mean_trans_delta']:.6f} | {r['delta_gyro_rmse']:.6f} | {r['delta_jerk']:.6f} | {verdict} |"
        )
    lines += [
        "",
        "Success requires acc residual improvement, no held-out/gyro/smoothness/contact regression, and small pose/tran deviation.",
        "",
        "## Config",
        "",
        f"- dataset: `{args.dataset}`",
        f"- rjs: `{args.rjs_path}`",
        f"- rjs method/field: `{args.rjs_method}` / `{args.rjs_field}`",
        f"- robust loss: `{args.robust_loss}`",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def run_mode(mode, sequences, rJS, R_JS, args, fit_ids, heldout_ids):
    per_sensor_rows, per_sequence_rows, pose_rows = [], [], []
    smooth_rows, gyro_rows, contact_rows = [], [], []
    refined_payload, trans_examples = [], []
    example = None
    for seq in sequences:
        print(f"[{mode}] optimize {seq['name']} frames={seq['n']}", flush=True)
        original = original_prediction(seq, rJS, R_JS, args)
        refined = optimize_sequence(seq, rJS, R_JS, mode, args, fit_ids)
        per_sensor_rows.extend(sensor_metric_rows(seq, mode, original, refined, args, fit_ids, heldout_ids))
        per_sequence_rows.extend(sequence_summary(seq, mode, original, refined, args, fit_ids, heldout_ids))
        pose_rows.extend(pose_summary(seq, mode, refined))
        gyro_rows.append(gyro_summary(seq, mode, original, refined, args))
        smooth_orig = smoothness_metrics(seq["qd_orig"], seq["qdd_orig"])
        smooth_ref = smoothness_metrics(refined["qd_clean"], refined["qdd_clean"])
        smooth_rows.append({"mode": mode, "sequence_id": seq["name"], "variant": "original", **smooth_orig})
        smooth_rows.append({"mode": mode, "sequence_id": seq["name"], "variant": "refined", **smooth_ref})
        contact_rows.append({"mode": mode, "sequence_id": seq["name"], "variant": "original", **contact_metrics(seq["joints_orig"])})
        contact_rows.append({"mode": mode, "sequence_id": seq["name"], "variant": "refined", **contact_metrics(refined["joints_clean"])})
        trans_examples.append((refined["tran_clean"] - seq["tran"]).norm(dim=-1).numpy())
        if example is None:
            sl, _ = finite_slice(seq["n"], args.trim)
            example = {
                "obs": seq["aS"][sl, 0].numpy(),
                "before": original["acc"][sl, 0].numpy(),
                "after": refined["acc_clean"][sl, 0].numpy(),
            }
        refined_payload.append(
            {
                "mode": mode,
                "sequence_id": seq["name"],
                "pose_clean": refined["pose_clean"],
                "tran_clean": refined["tran_clean"],
                "qd_clean": refined["qd_clean"],
                "qdd_clean": refined["qdd_clean"],
                "acc_clean": refined["acc_clean"],
                "debug": refined["debug"],
            }
        )
    return {
        "per_sensor": per_sensor_rows,
        "per_sequence": per_sequence_rows,
        "pose": pose_rows,
        "smooth": smooth_rows,
        "gyro": gyro_rows,
        "contact": contact_rows,
        "payload": refined_payload,
        "trans_examples": trans_examples,
        "example": example,
    }


def main():
    args = parse_args()
    out = output_dir(args)
    data = torch.load(args.dataset, map_location="cpu")
    rJS, R_JS, rjs_config = load_rjs(args.rjs_path, args.rjs_method, args.rjs_field)
    fit_ids = sensor_ids(args.fit_sensors)
    heldout_ids = sensor_ids(args.heldout_sensors)
    count = len(data["pose"])
    if args.max_sequences:
        count = min(count, int(args.max_sequences))
    sequences = [prepare_sequence(data, i, args) for i in range(count)]
    modes = ("A", "B", "C") if args.mode == "all" else (args.mode,)

    all_result = {
        "per_sensor": [],
        "per_sequence": [],
        "pose": [],
        "smooth": [],
        "gyro": [],
        "contact": [],
        "payload": [],
        "trans_examples": [],
    }
    example = None
    for mode in modes:
        result = run_mode(mode, sequences, rJS, R_JS, args, fit_ids, heldout_ids)
        for key in ("per_sensor", "per_sequence", "pose", "smooth", "gyro", "contact", "payload", "trans_examples"):
            all_result[key].extend(result[key])
        example = example or result["example"]

    overall = aggregate_overall(
        all_result["per_sequence"],
        all_result["gyro"],
        all_result["pose"],
        all_result["smooth"],
        all_result["contact"],
    )
    heldout_rows = [
        r
        for r in all_result["per_sensor"]
        if r["sensor_group"] == "heldout" and r["variant"] in ("original", "refined") and r.get("signal_filter") == "raw"
    ]
    heldout_wide = []
    for r in all_result["per_sensor"]:
        if r["sensor_group"] == "heldout" and r["variant"] == "original" and r.get("signal_filter") == "raw":
            match = next(
                x
                for x in all_result["per_sensor"]
                if x["mode"] == r["mode"]
                and x["sequence_id"] == r["sequence_id"]
                and x["sensor_id"] == r["sensor_id"]
                and x.get("signal_filter") == "raw"
                and x["variant"] == "refined"
            )
            heldout_wide.append(
                {
                    "mode": r["mode"],
                    "sequence_id": r["sequence_id"],
                    "sensor_name": r["sensor_name"],
                    "original_acc_rmse": r["acc_rmse"],
                    "refined_acc_rmse": match["acc_rmse"],
                    "delta_acc_rmse": match["acc_rmse"] - r["acc_rmse"],
                }
            )

    config = {
        "args": vars(args),
        "output_dir": str(out),
        "fps": FPS,
        "gravity_world": GRAVITY_WORLD.tolist(),
        "sensor_to_joint_map": sensor_to_joint_map(),
        "lower_body_joints": list(LOWER_BODY_JOINTS),
        "fit_sensors": [SENSOR_NAMES[i] for i in fit_ids],
        "heldout_sensors": [SENSOR_NAMES[i] for i in heldout_ids],
        "rjs_config": rjs_config,
        "transform_convention": {
            "R_WJ": "maps joint-local vectors into world coordinates",
            "R_JS": "maps sensor-frame vectors into joint-local coordinates",
            "R_WS": "R_WJ @ R_JS, maps sensor-frame vectors into world coordinates",
            "r_JS": "IMU sensor origin relative to mapped joint origin, expressed in joint-local coordinates",
        },
    }
    write_json(out / "config.json", config)
    write_csv(out / "summary_overall.csv", overall)
    write_csv(out / "summary_per_sensor.csv", all_result["per_sensor"])
    write_csv(out / "summary_per_sequence.csv", all_result["per_sequence"])
    write_csv(out / "pose_delta_summary.csv", all_result["pose"])
    write_csv(out / "smoothness_summary.csv", all_result["smooth"])
    write_csv(out / "heldout_sensor_summary.csv", heldout_wide)
    write_csv(out / "gyro_summary.csv", all_result["gyro"])
    write_csv(out / "contact_summary.csv", all_result["contact"])
    torch.save({"config": config, "sequences": all_result["payload"]}, out / "refined_sequences.pt")
    if args.save_frame_level:
        frame_rows = []
        for item in all_result["payload"]:
            frame_rows.append({"mode": item["mode"], "sequence_id": item["sequence_id"], "frames": item["pose_clean"].shape[0]})
        write_csv(out / "frame_level_manifest.csv", frame_rows)
    make_plots(out, overall, all_result["per_sensor"], all_result["pose"], all_result["trans_examples"], all_result["gyro"], heldout_wide, example)
    summary_md(out, overall, args, fit_ids, heldout_ids)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
