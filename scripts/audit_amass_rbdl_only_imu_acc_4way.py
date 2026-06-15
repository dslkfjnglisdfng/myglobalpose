import argparse
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RBDL_PYTHON = Path("/home/lingfeng/rbdl/build/python")
if RBDL_PYTHON.exists() and str(RBDL_PYTHON) not in sys.path:
    sys.path.insert(0, str(RBDL_PYTHON))

try:
    from articulate.utils.rbdl import RBDLModel
except ImportError as exc:
    raise ImportError(
        "Failed to import RBDL. Run with the GlobalPose GPU conda environment, for example: "
        'ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; '
        'export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; '
        '"$ENV/bin/python" scripts/audit_amass_rbdl_only_imu_acc_4way.py'
    ) from exc
from l4_q75_utils import q75_to_pose_tran
from l4_tail_update_qstate import UniformCubicBSpline
from l4_train_diverse_short import load_cache_files
from pip_physics_backend import PIP_PHYSICS_MODEL_FILE, smpl_to_pip_rbdl
from pl_curve import fit_uniform_cubic_spline_controls_position_only


FPS = 60.0
DT = 1.0 / FPS
IMU_LINKS = (
    "imu_left_forearm",
    "imu_right_forearm",
    "imu_left_knee",
    "imu_right_knee",
    "imu_head",
    "imu_root",
)


class BodyRef:
    def __init__(self, value):
        self.value = int(value)


def wrap_angle_delta(delta):
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def unwrap_q_angles(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[:, 3:] = np.unwrap(q[:, 3:], axis=0)
    return q


def finite_difference_first(x, mode="central", fps=FPS):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    if x.shape[0] < 2:
        return out
    if mode == "central":
        out[1:-1] = (x[2:] - x[:-2]) * (0.5 * fps)
        out[0] = (x[1] - x[0]) * fps
        out[-1] = (x[-1] - x[-2]) * fps
    elif mode == "five_point":
        if x.shape[0] < 5:
            return finite_difference_first(x, "central", fps)
        out[2:-2] = (-x[4:] + 8.0 * x[3:-1] - 8.0 * x[1:-3] + x[:-4]) * (fps / 12.0)
        out[:2] = finite_difference_first(x, "central", fps)[:2]
        out[-2:] = finite_difference_first(x, "central", fps)[-2:]
    else:
        raise ValueError(f"Unsupported first-difference mode: {mode}")
    return out


def finite_difference_second(x, mode="three_point", fps=FPS):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    if x.shape[0] < 3:
        return out
    if mode == "three_point":
        out[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) * (fps ** 2)
        out[0] = out[1]
        out[-1] = out[-2]
    elif mode == "five_point":
        if x.shape[0] < 5:
            return finite_difference_second(x, "three_point", fps)
        out[2:-2] = (
            -x[4:] + 16.0 * x[3:-1] - 30.0 * x[2:-2] + 16.0 * x[1:-3] - x[:-4]
        ) * ((fps ** 2) / 12.0)
        out[:2] = finite_difference_second(x, "three_point", fps)[:2]
        out[-2:] = finite_difference_second(x, "three_point", fps)[-2:]
    else:
        raise ValueError(f"Unsupported second-difference mode: {mode}")
    return out


def finite_difference_q(q, fps=FPS):
    q = np.asarray(q, dtype=np.float64)
    qdot = np.zeros_like(q)
    qddot = np.zeros_like(q)
    if q.shape[0] < 2:
        return qdot, qddot
    delta = q[1:] - q[:-1]
    delta[:, 3:] = wrap_angle_delta(delta[:, 3:])
    qdot[1:-1] = (delta[1:] + delta[:-1]) * (0.5 * fps)
    qdot[0] = delta[0] * fps
    qdot[-1] = delta[-1] * fps
    if q.shape[0] >= 3:
        second = q[2:] - 2.0 * q[1:-1] + q[:-2]
        second[:, 3:] = wrap_angle_delta(q[2:, 3:] - q[1:-1, 3:]) + wrap_angle_delta(q[:-2, 3:] - q[1:-1, 3:])
        qddot[1:-1] = second * (fps ** 2)
        qddot[0] = qddot[1]
        qddot[-1] = qddot[-2]
    return qdot, qddot


def spline_decode(samples):
    tensor = torch.from_numpy(np.asarray(samples, dtype=np.float64)).float()
    controls = fit_uniform_cubic_spline_controls_position_only(tensor)
    spline = UniformCubicBSpline(DT)
    q, qdot, qddot = spline(controls, return_derivatives=True)
    return (
        controls.detach().cpu().double().numpy(),
        q.detach().cpu().double().numpy(),
        qdot.detach().cpu().double().numpy(),
        qddot.detach().cpu().double().numpy(),
    )


def spline_operators(n, dt=DT):
    s = np.zeros((n, n), dtype=np.float64)
    d1 = np.zeros((n, n), dtype=np.float64)
    d2 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        left = max(i - 1, 0)
        right = min(i + 1, n - 1)
        s[i, left] += 1.0 / 6.0
        s[i, i] += 4.0 / 6.0
        s[i, right] += 1.0 / 6.0
        d1[i, right] += 1.0 / (2.0 * dt)
        d1[i, left] -= 1.0 / (2.0 * dt)
        d2[i, left] += 1.0 / (dt ** 2)
        d2[i, i] -= 2.0 / (dt ** 2)
        d2[i, right] += 1.0 / (dt ** 2)
    return s, d1, d2


def derivative_aware_spline_decode(samples, vel_target, acc_target, wp, wv, wa, wr):
    samples = np.asarray(samples, dtype=np.float64)
    original_shape = samples.shape
    n = samples.shape[0]
    flat = samples.reshape(n, -1)
    vel_flat = np.asarray(vel_target, dtype=np.float64).reshape(n, -1)
    acc_flat = np.asarray(acc_target, dtype=np.float64).reshape(n, -1)
    s, d1, d2 = spline_operators(n, DT)
    lhs = wp * (s.T @ s) + wv * (d1.T @ d1) + wa * (d2.T @ d2)
    if wr > 0.0:
        lhs = lhs + wr * np.eye(n, dtype=np.float64)
    rhs = wp * (s.T @ flat) + wv * (d1.T @ vel_flat) + wa * (d2.T @ acc_flat)
    controls_flat = np.linalg.solve(lhs, rhs)
    controls = controls_flat.reshape(original_shape)
    tensor = torch.from_numpy(controls_flat).float()
    spline = UniformCubicBSpline(DT)
    q, qdot, qddot = spline(tensor, return_derivatives=True)
    return (
        controls,
        q.detach().cpu().double().numpy().reshape(original_shape),
        qdot.detach().cpu().double().numpy().reshape(original_shape),
        qddot.detach().cpu().double().numpy().reshape(original_shape),
    )


def stats(arrays):
    values = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    x = np.concatenate(values)
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def vector_l1(a, b):
    return np.mean(np.abs(a - b), axis=-1)


def vector_l2(a, b):
    return np.linalg.norm(a - b, axis=-1)


def direction_angle_deg(a, b, eps=1e-8):
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    mask = (an > eps) & (bn > eps)
    out = np.full(an.shape, np.nan, dtype=np.float64)
    dot = np.sum(a[mask] * b[mask], axis=-1) / (an[mask] * bn[mask])
    out[mask] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return out


def per_sensor_mean_l2(a, b):
    l2 = vector_l2(a, b)
    return [float(v) for v in np.nanmean(l2.reshape(-1, l2.shape[-1]), axis=0)]


def trim_slice(n_frames, trim):
    if n_frames > 2 * trim:
        return slice(trim, n_frames - trim)
    return slice(0, n_frames)


def ensure_imu_bodies(model, links):
    bodies = []
    missing = []
    for link in links:
        body_id = int(model.model.GetBodyId(link))
        if body_id == 2**32 - 1:
            missing.append(link)
        else:
            bodies.append(BodyRef(body_id))
    if missing:
        raise KeyError(f"RBDL model missing required IMU links: {missing}")
    return bodies


def rbdl_site_positions(model, bodies, q):
    positions = np.zeros((q.shape[0], len(bodies), 3), dtype=np.float64)
    for t in range(q.shape[0]):
        for s, body in enumerate(bodies):
            positions[t, s] = model.calc_body_to_base_coordinates(q[t], body, np.zeros(3))
    return positions


def rbdl_site_motion(model, bodies, q, qdot, qddot):
    vel = np.zeros((q.shape[0], len(bodies), 3), dtype=np.float64)
    acc = np.zeros_like(vel)
    for t in range(q.shape[0]):
        for s, body in enumerate(bodies):
            vel[t, s] = model.calc_point_velocity(q[t], qdot[t], body, np.zeros(3))
            acc[t, s] = model.calc_point_acceleration(q[t], qdot[t], qddot[t], body, np.zeros(3))
    return vel, acc


def record_q(record):
    if "pose_gt" in record and "tran_gt" in record:
        pose = record["pose_gt"].float()
        tran = record["tran_gt"].float()
    elif "q75_gt" in record:
        pose, tran = q75_to_pose_tran(record["q75_gt"].float())
    else:
        raise KeyError(f"{record.get('name')} has neither pose_gt/tran_gt nor q75_gt")
    return smpl_to_pip_rbdl(pose.detach().cpu().numpy(), tran.detach().cpu().numpy())


def load_records(cache_path, max_sequences=0, max_frames=0):
    files, manifest = load_cache_files(cache_path)
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location="cpu")
        names = data.get("name")
        if names is None:
            raise KeyError(f"{cache_file} missing name")
        for seq_idx, name in enumerate(names):
            record = {"name": str(name), "cache_file": str(cache_file)}
            for key in ("pose_gt", "tran_gt", "q75_gt", "aM"):
                if key in data and data[key]:
                    value = data[key][seq_idx].float()
                    if max_frames:
                        value = value[:max_frames]
                    record[key] = value
            if "pose_gt" not in record and "q75_gt" not in record:
                raise KeyError(f"{cache_file}:{name} missing pose_gt/tran_gt and q75_gt")
            if ("pose_gt" in record and record["pose_gt"].shape[0] >= 5) or (
                "q75_gt" in record and record["q75_gt"].shape[0] >= 5
            ):
                records.append(record)
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def compare_pair(acc_source, acc_target, vel_source, vel_target, sl):
    acc_source = acc_source[sl]
    acc_target = acc_target[sl]
    vel_source = vel_source[sl]
    vel_target = vel_target[sl]
    return {
        "acc_l1_m_s2": vector_l1(acc_source, acc_target),
        "acc_l2_m_s2": vector_l2(acc_source, acc_target),
        "acc_angle_deg": direction_angle_deg(acc_source, acc_target),
        "vel_l2_m_s": vector_l2(vel_source, vel_target),
        "per_sensor_acc_l2_m_s2": per_sensor_mean_l2(acc_source, acc_target),
    }


def aggregate_groups(rows, group_names):
    aggregate = {}
    for group in group_names:
        group_rows = [row["groups"][group] for row in rows if group in row["groups"]]
        aggregate[group] = {
            "acc_l1_m_s2": stats([row["acc_l1_m_s2"] for row in group_rows]),
            "acc_l2_m_s2": stats([row["acc_l2_m_s2"] for row in group_rows]),
            "acc_angle_deg": stats([row["acc_angle_deg"] for row in group_rows]),
            "vel_l2_m_s": stats([row["vel_l2_m_s"] for row in group_rows]),
            "per_sensor_acc_l2_m_s2": [
                float(v)
                for v in np.nanmean(
                    np.asarray([row["per_sensor_acc_l2_m_s2"] for row in group_rows], dtype=np.float64),
                    axis=0,
                )
            ]
            if group_rows
            else [],
        }
    return aggregate


def aggregate_diagnostics(rows):
    keys = (
        "q_ctrl_reconstruction_l2",
        "q_derivfit_reconstruction_l2",
        "rbdl_position_ctrl_reconstruction_l2_m",
        "rbdl_position_derivfit_reconstruction_l2_m",
        "rbdl_position_ctrl_acc_vs_fd_l2_m_s2",
        "rbdl_position_derivfit_acc_vs_fd_l2_m_s2",
        "rbdl_position_ctrl_vel_vs_fd_l2_m_s",
        "rbdl_position_derivfit_vel_vs_fd_l2_m_s",
        "pos_fd3_vs_fd5_acc_l2_m_s2",
    )
    out = {}
    for key in keys:
        out[key] = stats([row["diagnostic_arrays"][key] for row in rows if key in row["diagnostic_arrays"]])
    out["aM_reference_only_present_count"] = int(
        sum(1 for row in rows if row["diagnostic_arrays"].get("aM_reference_only_present", False))
    )
    return out


def conclusion_from_aggregate(aggregate):
    ranked = sorted(
        (
            (group, values["acc_l2_m_s2"]["mean"], values["acc_angle_deg"]["mean"], values["vel_l2_m_s"]["mean"])
            for group, values in aggregate.items()
            if values["acc_l2_m_s2"]["mean"] is not None
        ),
        key=lambda item: item[1],
    )
    if not ranked:
        return {"status": "not_available"}
    best_group, best_l2, best_angle, best_vel = ranked[0]
    recommendation = {
        "q_fd_vs_pos_fd": "Use finite-difference qdot/qddot when the acceleration target is finite-difference RBDL IMU-link position acceleration.",
        "q_fd_vs_pos_ctrl": "Unexpected cross-source winner; inspect derivative definitions before using this as training supervision.",
        "q_ctrl_vs_pos_fd": "Unexpected cross-source winner; inspect derivative definitions before using this as training supervision.",
        "q_ctrl_vs_pos_ctrl": "Use control-curve qdot/qddot when the acceleration target is control-curve RBDL IMU-link position acceleration.",
        "q_derivfit_vs_pos_derivfit": "Use derivative-aware control fitting when both q controls and position controls are trained to match finite-difference velocity/acceleration targets.",
        "q_derivfit_vs_pos_fd": "Derivative-aware q controls are close to the finite-difference position target; compare against q_fd_vs_pos_fd before using them for acceleration supervision.",
        "q_fd_vs_pos_derivfit": "Derivative-aware position controls are close to the finite-difference q target; compare against q_fd_vs_pos_fd before using them for acceleration supervision.",
    }[best_group]
    return {
        "status": "ok",
        "best_group_by_acc_l2": best_group,
        "best_acc_l2_m_s2_mean": float(best_l2),
        "best_acc_angle_deg_mean": float(best_angle),
        "best_vel_l2_m_s_mean": float(best_vel),
        "ranking_by_acc_l2": [
            {
                "group": group,
                "acc_l2_m_s2_mean": float(acc_l2),
                "acc_angle_deg_mean": float(angle),
                "vel_l2_m_s_mean": float(vel),
            }
            for group, acc_l2, angle, vel in ranked
        ],
        "recommendation": recommendation,
        "interpretation": (
            "The comparison is source-consistency, not cached-aM accuracy. Matching q-derivative and "
            "position-target derivative families should be preferred; crossed derivative families inject a "
            "large synthetic mismatch."
        ),
    }


def summarize_row_for_json(row):
    out = {
        "name": row["name"],
        "cache_file": row["cache_file"],
        "num_frames": row["num_frames"],
        "trim": row["trim"],
        "groups": {},
        "diagnostics": row["diagnostics"],
    }
    for group, values in row["groups"].items():
        out["groups"][group] = {
            "acc_l1_m_s2": stats([values["acc_l1_m_s2"]]),
            "acc_l2_m_s2": stats([values["acc_l2_m_s2"]]),
            "acc_angle_deg": stats([values["acc_angle_deg"]]),
            "vel_l2_m_s": stats([values["vel_l2_m_s"]]),
            "per_sensor_acc_l2_m_s2": values["per_sensor_acc_l2_m_s2"],
        }
    return out


def audit_record(record, model, bodies, args):
    q_raw = record_q(record)
    q_unwrapped = unwrap_q_angles(q_raw)
    if args.max_frames:
        q_unwrapped = q_unwrapped[: args.max_frames]
    n = q_unwrapped.shape[0]
    sl = trim_slice(n, args.trim)

    p_rbdl = rbdl_site_positions(model, bodies, q_unwrapped)
    v_pos_fd = finite_difference_first(p_rbdl, mode=args.position_velocity_fd_mode)
    a_pos_fd = finite_difference_second(p_rbdl, mode=args.position_acc_fd_mode)
    _, p_ctrl_recon, v_pos_ctrl, a_pos_ctrl = spline_decode(p_rbdl.reshape(n, -1))
    p_ctrl_recon = p_ctrl_recon.reshape(p_rbdl.shape)
    v_pos_ctrl = v_pos_ctrl.reshape(p_rbdl.shape)
    a_pos_ctrl = a_pos_ctrl.reshape(p_rbdl.shape)
    _, p_deriv_recon, v_pos_deriv, a_pos_deriv = derivative_aware_spline_decode(
        p_rbdl,
        v_pos_fd,
        a_pos_fd,
        args.derivfit_position_weight,
        args.derivfit_velocity_weight,
        args.derivfit_acceleration_weight,
        args.derivfit_ridge_weight,
    )

    qdot_fd, qddot_fd = finite_difference_q(q_unwrapped)
    q_ctrl_input = q_unwrapped.copy()
    _, q_ctrl, qdot_ctrl, qddot_ctrl = spline_decode(q_ctrl_input)
    _, q_deriv, qdot_deriv, qddot_deriv = derivative_aware_spline_decode(
        q_unwrapped,
        qdot_fd,
        qddot_fd,
        args.derivfit_position_weight,
        args.derivfit_velocity_weight,
        args.derivfit_acceleration_weight,
        args.derivfit_ridge_weight,
    )

    v_q_fd, a_q_fd = rbdl_site_motion(model, bodies, q_unwrapped, qdot_fd, qddot_fd)
    v_q_ctrl, a_q_ctrl = rbdl_site_motion(model, bodies, q_ctrl, qdot_ctrl, qddot_ctrl)
    v_q_deriv, a_q_deriv = rbdl_site_motion(model, bodies, q_deriv, qdot_deriv, qddot_deriv)

    groups = {
        "q_fd_vs_pos_fd": compare_pair(a_q_fd, a_pos_fd, v_q_fd, v_pos_fd, sl),
        "q_fd_vs_pos_ctrl": compare_pair(a_q_fd, a_pos_ctrl, v_q_fd, v_pos_ctrl, sl),
        "q_ctrl_vs_pos_fd": compare_pair(a_q_ctrl, a_pos_fd, v_q_ctrl, v_pos_fd, sl),
        "q_ctrl_vs_pos_ctrl": compare_pair(a_q_ctrl, a_pos_ctrl, v_q_ctrl, v_pos_ctrl, sl),
        "q_derivfit_vs_pos_derivfit": compare_pair(a_q_deriv, a_pos_deriv, v_q_deriv, v_pos_deriv, sl),
        "q_derivfit_vs_pos_fd": compare_pair(a_q_deriv, a_pos_fd, v_q_deriv, v_pos_fd, sl),
        "q_fd_vs_pos_derivfit": compare_pair(a_q_fd, a_pos_deriv, v_q_fd, v_pos_deriv, sl),
    }
    diagnostics = {
        "q_ctrl_reconstruction_l2": vector_l2(q_ctrl[sl], q_unwrapped[sl]),
        "q_derivfit_reconstruction_l2": vector_l2(q_deriv[sl], q_unwrapped[sl]),
        "rbdl_position_ctrl_reconstruction_l2_m": vector_l2(p_ctrl_recon[sl], p_rbdl[sl]),
        "rbdl_position_derivfit_reconstruction_l2_m": vector_l2(p_deriv_recon[sl], p_rbdl[sl]),
        "rbdl_position_ctrl_acc_vs_fd_l2_m_s2": vector_l2(a_pos_ctrl[sl], a_pos_fd[sl]),
        "rbdl_position_derivfit_acc_vs_fd_l2_m_s2": vector_l2(a_pos_deriv[sl], a_pos_fd[sl]),
        "rbdl_position_ctrl_vel_vs_fd_l2_m_s": vector_l2(v_pos_ctrl[sl], v_pos_fd[sl]),
        "rbdl_position_derivfit_vel_vs_fd_l2_m_s": vector_l2(v_pos_deriv[sl], v_pos_fd[sl]),
        "pos_fd3_vs_fd5_acc_l2_m_s2": vector_l2(
            finite_difference_second(p_rbdl, "three_point")[sl],
            finite_difference_second(p_rbdl, "five_point")[sl],
        ),
        "aM_reference_only_present": "aM" in record,
    }
    return {
        "name": record["name"],
        "cache_file": record["cache_file"],
        "num_frames": int(n),
        "trim": int(args.trim),
        "groups": groups,
        "diagnostic_arrays": diagnostics,
        "diagnostics": {
            "q_ctrl_reconstruction_l2": stats([diagnostics["q_ctrl_reconstruction_l2"]]),
            "q_derivfit_reconstruction_l2": stats([diagnostics["q_derivfit_reconstruction_l2"]]),
            "rbdl_position_ctrl_reconstruction_l2_m": stats([diagnostics["rbdl_position_ctrl_reconstruction_l2_m"]]),
            "rbdl_position_derivfit_reconstruction_l2_m": stats([diagnostics["rbdl_position_derivfit_reconstruction_l2_m"]]),
            "rbdl_position_ctrl_acc_vs_fd_l2_m_s2": stats([diagnostics["rbdl_position_ctrl_acc_vs_fd_l2_m_s2"]]),
            "rbdl_position_derivfit_acc_vs_fd_l2_m_s2": stats([diagnostics["rbdl_position_derivfit_acc_vs_fd_l2_m_s2"]]),
            "rbdl_position_ctrl_vel_vs_fd_l2_m_s": stats([diagnostics["rbdl_position_ctrl_vel_vs_fd_l2_m_s"]]),
            "rbdl_position_derivfit_vel_vs_fd_l2_m_s": stats([diagnostics["rbdl_position_derivfit_vel_vs_fd_l2_m_s"]]),
            "pos_fd3_vs_fd5_acc_l2_m_s2": stats([diagnostics["pos_fd3_vs_fd5_acc_l2_m_s2"]]),
            "aM_reference_only_present": diagnostics["aM_reference_only_present"],
        },
    }


def group_notes(group):
    notes = {
        "q_fd_vs_pos_fd": "RBDL qdot/qddot from wrapped finite differences compared with finite-difference acceleration of RBDL IMU-link positions.",
        "q_fd_vs_pos_ctrl": "RBDL qdot/qddot from wrapped finite differences compared with control-curve acceleration of RBDL IMU-link positions.",
        "q_ctrl_vs_pos_fd": "RBDL qdot/qddot from fitted q controls compared with finite-difference acceleration of RBDL IMU-link positions.",
        "q_ctrl_vs_pos_ctrl": "RBDL qdot/qddot from fitted q controls compared with control-curve acceleration of RBDL IMU-link positions.",
        "q_derivfit_vs_pos_derivfit": "RBDL qdot/qddot from derivative-aware q controls compared with derivative-aware control acceleration of RBDL IMU-link positions.",
        "q_derivfit_vs_pos_fd": "RBDL qdot/qddot from derivative-aware q controls compared with finite-difference acceleration of RBDL IMU-link positions.",
        "q_fd_vs_pos_derivfit": "RBDL qdot/qddot from wrapped finite differences compared with derivative-aware control acceleration of RBDL IMU-link positions.",
    }
    return notes[group]


def markdown_table(result):
    lines = [
        "# AMASS RBDL-only IMU acceleration derivative audit",
        "",
        "SMPL FK is not used for IMU acceleration synthesis. AMASS pose/tran is converted to PIP/RBDL q, and RBDL IMU links are the only source of IMU site position, velocity, and acceleration.",
        "",
        f"Cache: `{result['cache']}`",
        f"Sequences: `{result['num_records']}`",
        f"Position FD mode: `{result['position_acc_fd_mode']}`",
        f"Derivative-aware fit weights: `wp={result['derivfit_weights']['position']}, wv={result['derivfit_weights']['velocity']}, wa={result['derivfit_weights']['acceleration']}, wr={result['derivfit_weights']['ridge']}`",
        f"Trim: `{result['trim']}` frames",
        "",
        "| q source | position acc target | acc L1 m/s^2 ↓ | acc L2 m/s^2 ↓ | angle deg ↓ | vel L2 m/s ↓ | Notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    labels = {
        "q_fd_vs_pos_fd": ("finite-diff qdot/qddot", "RBDL position finite diff"),
        "q_fd_vs_pos_ctrl": ("finite-diff qdot/qddot", "RBDL position control curve"),
        "q_ctrl_vs_pos_fd": ("control-curve qdot/qddot", "RBDL position finite diff"),
        "q_ctrl_vs_pos_ctrl": ("control-curve qdot/qddot", "RBDL position control curve"),
        "q_derivfit_vs_pos_derivfit": ("derivative-aware qdot/qddot", "RBDL position derivative-aware curve"),
        "q_derivfit_vs_pos_fd": ("derivative-aware qdot/qddot", "RBDL position finite diff"),
        "q_fd_vs_pos_derivfit": ("finite-diff qdot/qddot", "RBDL position derivative-aware curve"),
    }
    for group, (q_label, target_label) in labels.items():
        agg = result["aggregate"][group]
        lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
                q_label,
                target_label,
                agg["acc_l1_m_s2"]["mean"],
                agg["acc_l2_m_s2"]["mean"],
                agg["acc_angle_deg"]["mean"],
                agg["vel_l2_m_s"]["mean"],
                group_notes(group),
            )
        )
    lines.extend([
        "",
        "## Per-sensor acc L2 m/s^2",
        "",
        "| group | left_forearm | right_forearm | left_knee | right_knee | head | root |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in labels:
        values = result["aggregate"][group]["per_sensor_acc_l2_m_s2"]
        lines.append("| {} | {} |".format(group, " | ".join(f"{v:.6f}" for v in values)))
    diag = result.get("diagnostic_aggregate", {})
    if diag:
        lines.extend([
            "",
            "## Direct curve-vs-finite-difference diagnostics",
            "",
            "| metric | mean | median | p95 |",
            "|---|---:|---:|---:|",
        ])
        for key in (
            "rbdl_position_ctrl_acc_vs_fd_l2_m_s2",
            "rbdl_position_derivfit_acc_vs_fd_l2_m_s2",
            "rbdl_position_ctrl_vel_vs_fd_l2_m_s",
            "rbdl_position_derivfit_vel_vs_fd_l2_m_s",
            "rbdl_position_ctrl_reconstruction_l2_m",
            "rbdl_position_derivfit_reconstruction_l2_m",
        ):
            value = diag[key]
            lines.append(
                "| {} | {:.6f} | {:.6f} | {:.6f} |".format(
                    key,
                    value["mean"],
                    value["median"],
                    value["p95"],
                )
            )
    conclusion = result.get("conclusion", {})
    if conclusion.get("status") == "ok":
        lines.extend([
            "",
            "## Conclusion",
            "",
            f"Best group by mean acceleration L2: `{conclusion['best_group_by_acc_l2']}`.",
            "",
            conclusion["recommendation"],
            "",
            conclusion["interpretation"],
        ])
    lines.extend([
        "",
        "## Contract",
        "",
        "- `cached aM` is not used as GT.",
        "- RBDL base/world-frame linear acceleration is compared in `m/s^2`.",
        "- Gravity is not added in the main kinematic comparison.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="RBDL-only 2x2 audit for AMASS IMU-site acceleration derivative sources.")
    parser.add_argument("--cache", type=Path, default=Path("data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged/baseline_cache_manifest.json"))
    parser.add_argument("--output-json", type=Path, default=Path("data/experiments/gt_control_derivative_audit_20260608/amass_rbdl_only_imu_acc_4way.json"))
    parser.add_argument("--output-md", type=Path, default=Path("data/experiments/gt_control_derivative_audit_20260608/amass_rbdl_only_imu_acc_4way.md"))
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--trim", type=int, default=2)
    parser.add_argument("--position-acc-fd-mode", choices=("three_point", "five_point"), default="three_point")
    parser.add_argument("--position-velocity-fd-mode", choices=("central", "five_point"), default="central")
    parser.add_argument("--derivfit-position-weight", type=float, default=1.0)
    parser.add_argument("--derivfit-velocity-weight", type=float, default=0.03)
    parser.add_argument("--derivfit-acceleration-weight", type=float, default=0.0003)
    parser.add_argument("--derivfit-ridge-weight", type=float, default=1e-6)
    args = parser.parse_args()

    model = RBDLModel(str(PIP_PHYSICS_MODEL_FILE), update_kinematics_by_hand=False)
    bodies = ensure_imu_bodies(model, IMU_LINKS)
    records, manifest = load_records(args.cache, max_sequences=args.max_records, max_frames=args.max_frames)
    group_names = (
        "q_fd_vs_pos_fd",
        "q_fd_vs_pos_ctrl",
        "q_ctrl_vs_pos_fd",
        "q_ctrl_vs_pos_ctrl",
        "q_derivfit_vs_pos_derivfit",
        "q_derivfit_vs_pos_fd",
        "q_fd_vs_pos_derivfit",
    )
    rows = []
    failures = []
    for record in records:
        try:
            rows.append(audit_record(record, model, bodies, args))
        except Exception as exc:
            failures.append({
                "name": record.get("name"),
                "cache_file": record.get("cache_file"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            if not args.max_records or args.max_records <= 2:
                raise
    aggregate = aggregate_groups(rows, group_names)
    result = {
        "status": "ok" if rows else "failed",
        "cache": str(args.cache),
        "cache_manifest": manifest,
        "num_records": len(rows),
        "num_failures": len(failures),
        "failures": failures,
        "fps": FPS,
        "dt": DT,
        "trim": int(args.trim),
        "position_acc_fd_mode": args.position_acc_fd_mode,
        "position_velocity_fd_mode": args.position_velocity_fd_mode,
        "derivfit_weights": {
            "position": float(args.derivfit_position_weight),
            "velocity": float(args.derivfit_velocity_weight),
            "acceleration": float(args.derivfit_acceleration_weight),
            "ridge": float(args.derivfit_ridge_weight),
        },
        "rbdl_model": str(PIP_PHYSICS_MODEL_FILE),
        "imu_links": list(IMU_LINKS),
        "contract": {
            "smpl_fk_used_for_imu_acceleration_synthesis": False,
            "rbdl_physical_model_is_only_imu_site_source": True,
            "cached_aM_used_as_gt": False,
            "amass_pose_tran_role": "Only converted to PIP/RBDL Euler q; no SMPL FK IMU acceleration target is synthesized.",
            "frame": "RBDL base/world frame",
            "unit": "m/s^2 for acceleration, m/s for velocity",
            "gravity_in_main_comparison": "not added",
        },
        "aggregate": aggregate,
        "diagnostic_aggregate": aggregate_diagnostics(rows),
        "conclusion": conclusion_from_aggregate(aggregate),
        "rows": [summarize_row_for_json(row) for row in rows],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_table(result))
    print(json.dumps({
        "status": result["status"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "num_records": result["num_records"],
        "num_failures": result["num_failures"],
        "aggregate": result["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
