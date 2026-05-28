import argparse
import json
from pathlib import Path

import torch

from l4_estimate_sensor_offsets import window_ranges
from l4_sensor_offset_utils import (
    FPS,
    GRAVITY_WORLD,
    SENSOR_NAMES,
    first_derivative,
    fk_imu_joints_and_vertices,
    load_dataset_file,
    make_metadata,
    second_derivative,
    smooth_centered,
    skew_matrix,
    vee_skew,
)


OLD_LS_CACHE = Path("data/dataset_work/SensorOffset/totalcapture_only_v2/selected_cache")


def tensor_median(x):
    x = torch.as_tensor(x).float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan")
    return float(x.median().item())


def nanmean(x):
    x = torch.as_tensor(x).float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return torch.tensor(float("nan"))
    return x.mean()


def robust_rotation_mean(rotations):
    rotations = rotations.float()
    valid = torch.isfinite(rotations).all(dim=(-1, -2))
    rotations = rotations[valid]
    if rotations.numel() == 0:
        return torch.eye(3)
    mean_matrix = rotations.mean(dim=0)
    u, _, vh = torch.linalg.svd(mean_matrix)
    r = u.matmul(vh)
    if torch.det(r) < 0:
        u[:, -1] *= -1.0
        r = u.matmul(vh)
    return r.float()


def rotation_angle_deg(R):
    R = R.float()
    trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def axis_angle_to_matrix(axis_angle):
    theta = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = axis_angle / theta
    k = skew_matrix(axis)
    eye = torch.eye(3, dtype=axis_angle.dtype).expand(axis_angle.shape[:-1] + (3, 3))
    sin = torch.sin(theta)[..., None]
    cos = torch.cos(theta)[..., None]
    return eye + sin * k + (1.0 - cos) * k.matmul(k)


def matvec(R, v):
    return R.matmul(v.unsqueeze(-1)).squeeze(-1)


def align_by_dt(*arrays, dt_frames):
    dt_frames = int(dt_frames)
    if dt_frames > 0:
        return [arr[:-dt_frames] for arr in arrays[:-1]] + [arrays[-1][dt_frames:]]
    if dt_frames < 0:
        k = -dt_frames
        return [arr[k:] for arr in arrays[:-1]] + [arrays[-1][:-k]]
    return list(arrays)


def fit_regularized_linear(A, y, ridge=1e-4, bias_ridge=1000.0, fit_bias=True):
    A = A.float()
    y = y.float()
    if A.shape[0] < 8:
        return None
    n = A.shape[0]
    if fit_bias:
        I = torch.eye(3, dtype=A.dtype).expand(n, 3, 3)
        M = torch.cat((A, I), dim=-1).reshape(-1, 6)
        reg = torch.diag(torch.tensor([ridge, ridge, ridge, bias_ridge, bias_ridge, bias_ridge], dtype=A.dtype))
    else:
        M = A.reshape(-1, 3)
        reg = torch.eye(3, dtype=A.dtype) * ridge
    target = y.reshape(-1)
    lhs = M.T.matmul(M) + reg
    rhs = M.T.matmul(target)
    try:
        sol = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        sol = torch.linalg.lstsq(lhs, rhs).solution
    if fit_bias:
        r = sol[:3]
        b = sol[3:]
    else:
        r = sol
        b = torch.zeros(3, dtype=A.dtype)
    pred = A.matmul(r.view(3, 1)).squeeze(-1) + b.view(1, 3)
    s = torch.linalg.svdvals(A.reshape(-1, 3))
    cond = s.max() / s.min().clamp_min(1e-12)
    return {
        "r": r.float(),
        "bias": b.float(),
        "pred": pred.float(),
        "target": y.float(),
        "condition_number": cond.float(),
        "singular_values": s.float(),
        "observability_score": s.min().float(),
        "num_valid_frames": int(A.shape[0]),
    }


def prepare_rawlike_sequence(data, seq_idx, args):
    pose = data["pose"][seq_idx].float()
    tran = data["tran"][seq_idx].float()
    RIM = data["RIM"][seq_idx].float()
    RIS = data["RIS"][seq_idx].float()
    aS = data["aS"][seq_idx].float()
    wS = data["wS"][seq_idx].float()
    n = min(pose.shape[0], tran.shape[0], RIS.shape[0], aS.shape[0], wS.shape[0])
    if args.max_frames > 0:
        n = min(n, args.max_frames)
    pose, tran = pose[:n], tran[:n]
    RIS, aS, wS = RIS[:n], aS[:n], wS[:n]
    p_wj, R_wj, _ = fk_imu_joints_and_vertices(pose, tran, device=args.device)
    R_WS_obs = RIM.transpose(1, 2).matmul(RIS)
    if args.smooth_window > 1 and args.smoothing_mode not in ("none", "identity"):
        p_wj = smooth_centered(p_wj, args.smooth_window, args.smoothing_mode)
        R_wj = smooth_centered(R_wj, args.smooth_window, args.smoothing_mode)
        aS = smooth_centered(aS, args.smooth_window, args.smoothing_mode)
        wS = smooth_centered(wS, args.smooth_window, args.smoothing_mode)
    R_dot = first_derivative(R_wj, fps=FPS, mode=args.derivative_mode)
    omega_hat = R_dot.matmul(R_wj.transpose(-1, -2))
    omega_hat = 0.5 * (omega_hat - omega_hat.transpose(-1, -2))
    omega_wj = vee_skew(omega_hat)
    return {
        "name": str(data["name"][seq_idx]) if "name" in data else f"seq_{seq_idx}",
        "p_wj": p_wj,
        "R_wj": R_wj,
        "R_WS_obs": R_WS_obs,
        "aS": aS,
        "wS": wS,
        "ddot_p_wj": second_derivative(p_wj, fps=FPS, mode=args.derivative_mode),
        "ddot_R_wj": second_derivative(R_wj, fps=FPS, mode=args.derivative_mode),
        "omega_wj": omega_wj,
    }


def make_synthetic_sequence(data, seq_idx, args, R_JS_gt, r_JS_gt):
    pose = data["pose"][seq_idx].float()
    tran = data["tran"][seq_idx].float()
    n = min(pose.shape[0], tran.shape[0])
    if args.max_frames > 0:
        n = min(n, args.max_frames)
    pose, tran = pose[:n], tran[:n]
    p_wj, R_wj, _ = fk_imu_joints_and_vertices(pose, tran, device=args.device)
    if args.smooth_window > 1 and args.smoothing_mode not in ("none", "identity"):
        p_wj = smooth_centered(p_wj, args.smooth_window, args.smoothing_mode)
        R_wj = smooth_centered(R_wj, args.smooth_window, args.smoothing_mode)
    R_WS = R_wj.matmul(R_JS_gt.view(1, 6, 3, 3))
    p_WS = p_wj + matvec(R_wj, r_JS_gt.view(1, 6, 3))
    R_dot = first_derivative(R_wj, fps=FPS, mode=args.derivative_mode)
    omega_hat = R_dot.matmul(R_wj.transpose(-1, -2))
    omega_hat = 0.5 * (omega_hat - omega_hat.transpose(-1, -2))
    omega_wj = vee_skew(omega_hat)
    omega_s = matvec(R_WS.transpose(-1, -2), omega_wj)
    ddot_p_WS = second_derivative(p_WS, fps=FPS, mode=args.derivative_mode)
    aS = matvec(R_WS.transpose(-1, -2), ddot_p_WS - GRAVITY_WORLD.view(1, 1, 3))
    return {
        "name": str(data["name"][seq_idx]) if "name" in data else f"synthetic_{seq_idx}",
        "p_wj": p_wj,
        "R_wj": R_wj,
        "R_WS_obs": R_WS,
        "aS": aS,
        "wS": omega_s,
        "ddot_p_wj": second_derivative(p_wj, fps=FPS, mode=args.derivative_mode),
        "ddot_R_wj": second_derivative(R_wj, fps=FPS, mode=args.derivative_mode),
        "omega_wj": omega_wj,
        "R_JS_gt": R_JS_gt,
        "r_JS_gt": r_JS_gt,
    }


def fit_sensor_candidate(seq, sensor_idx, start, end, dt_frames, args):
    s = sensor_idx
    R_wj = seq["R_wj"][start:end, s]
    R_WS_obs = seq["R_WS_obs"][start:end, s]
    aS = seq["aS"][start:end, s]
    wS = seq["wS"][start:end, s]
    ddot_p = seq["ddot_p_wj"][start:end, s]
    ddot_R = seq["ddot_R_wj"][start:end, s]
    omega_wj = seq["omega_wj"][start:end, s]
    if dt_frames != 0:
        R_wj, ddot_p, ddot_R, omega_wj, R_WS_obs = align_by_dt(
            R_wj, ddot_p, ddot_R, omega_wj, R_WS_obs, dt_frames=dt_frames
        )
        obs_keep = R_WS_obs.shape[0]
        if dt_frames > 0:
            aS = aS[dt_frames : dt_frames + obs_keep]
            wS = wS[dt_frames : dt_frames + obs_keep]
        else:
            aS = aS[:obs_keep]
            wS = wS[:obs_keep]
    valid_rot = torch.isfinite(R_wj).all(dim=(-1, -2)) & torch.isfinite(R_WS_obs).all(dim=(-1, -2))
    if valid_rot.sum() < 8:
        return None
    R_samples = R_wj[valid_rot].transpose(-1, -2).matmul(R_WS_obs[valid_rot])
    R_JS = robust_rotation_mean(R_samples)
    R_WS_pred = R_wj.matmul(R_JS.view(1, 3, 3))
    orient_fit = rotation_angle_deg(R_WS_obs.transpose(-1, -2).matmul(R_WS_pred))
    orient_zero = rotation_angle_deg(R_WS_obs.transpose(-1, -2).matmul(R_wj))
    w_pred_no_bias = matvec(R_JS.T.matmul(R_wj.transpose(-1, -2)), omega_wj)
    gyro_valid = torch.isfinite(wS).all(dim=-1) & torch.isfinite(w_pred_no_bias).all(dim=-1)
    gyro_res = wS[gyro_valid] - w_pred_no_bias[gyro_valid]
    gyro_bias = gyro_res.sum(dim=0) / (gyro_res.shape[0] + args.gyro_bias_ridge) if gyro_res.numel() else torch.zeros(3)
    gyro_zero = (wS[gyro_valid] - matvec(R_wj[gyro_valid].transpose(-1, -2), omega_wj[gyro_valid])).norm(dim=-1)
    gyro_fit = (gyro_res - gyro_bias.view(1, 3)).norm(dim=-1)
    R_WS_fit_T = R_WS_pred.transpose(-1, -2)
    c = matvec(R_WS_fit_T, ddot_p - GRAVITY_WORLD.view(1, 3))
    A = R_WS_fit_T.matmul(ddot_R)
    acc_valid = torch.isfinite(A).all(dim=(-1, -2)) & torch.isfinite(c).all(dim=-1) & torch.isfinite(aS).all(dim=-1)
    A_valid = A[acc_valid]
    c_valid = c[acc_valid]
    aS_valid = aS[acc_valid]
    fit = fit_regularized_linear(
        A_valid,
        aS_valid - c_valid,
        ridge=args.ridge,
        bias_ridge=args.acc_bias_ridge,
        fit_bias=args.fit_acc_bias,
    )
    if fit is None:
        return None
    acc_zero = (aS_valid - c_valid).norm(dim=-1)
    acc_residual_vec = aS_valid - (c_valid + fit["pred"])
    acc_fit = acc_residual_vec.norm(dim=-1)
    gravity_residual_mean = acc_residual_vec.mean(dim=0)
    return {
        "R_JS": R_JS.float(),
        "r_JS": fit["r"].float(),
        "b_aS": fit["bias"].float(),
        "b_gS": gyro_bias.float(),
        "orientation_residual_zero_deg": nanmean(orient_zero).float(),
        "orientation_residual_fit_deg": nanmean(orient_fit).float(),
        "orientation_improvement": ((nanmean(orient_zero) - nanmean(orient_fit)) / nanmean(orient_zero).clamp_min(1e-12)).float(),
        "gyro_residual_zero": nanmean(gyro_zero).float(),
        "gyro_residual_fit": nanmean(gyro_fit).float(),
        "gyro_improvement": ((nanmean(gyro_zero) - nanmean(gyro_fit)) / nanmean(gyro_zero).clamp_min(1e-12)).float(),
        "acc_residual_zero": nanmean(acc_zero).float(),
        "acc_residual_fit": nanmean(acc_fit).float(),
        "acc_residual_mean_sensor": gravity_residual_mean.float(),
        "gravity_residual_mean_norm": gravity_residual_mean.norm().float(),
        "acc_improvement": ((nanmean(acc_zero) - nanmean(acc_fit)) / nanmean(acc_zero).clamp_min(1e-12)).float(),
        "condition_number": fit["condition_number"].float(),
        "observability_score": fit["observability_score"].float(),
        "singular_values": fit["singular_values"].float(),
        "num_valid_frames": fit["num_valid_frames"],
    }


def stack_metric(results, key, default=float("nan")):
    out = []
    for item in results:
        if item is None:
            out.append(torch.tensor(default))
        else:
            out.append(torch.as_tensor(item[key]).float())
    return torch.stack(out)


def aggregate_windows(window_results):
    valid = [r for r in window_results if r is not None]
    if not valid:
        return None, torch.tensor(float("nan")), torch.tensor(float("nan"))
    r = torch.stack([x["r_JS"] for x in valid])
    R = torch.stack([x["R_JS"] for x in valid])
    r_med = r.median(dim=0).values
    R_med = robust_rotation_mean(R)
    r_cons = (r - r_med.view(1, 3)).norm(dim=-1).median()
    R_cons = rotation_angle_deg(R_med.T.view(1, 3, 3).matmul(R)).median()
    return {"r_JS": r_med.float(), "R_JS": R_med.float()}, r_cons.float(), R_cons.float()


def estimate_sequence(seq, args, dt_frames=0, compute_windows=True):
    n = seq["aS"].shape[0] - abs(int(dt_frames))
    ranges = window_ranges(n, args.window_size, args.stride) if compute_windows and n >= args.min_window_frames else [(0, n)]
    records = []
    for s in range(6):
        full = fit_sensor_candidate(seq, s, 0, seq["aS"].shape[0], dt_frames, args)
        win = [fit_sensor_candidate(seq, s, a, b, dt_frames, args) for a, b in ranges] if compute_windows else []
        _, r_cons, R_cons = aggregate_windows(win)
        if full is None:
            records.append(None)
            continue
        full["window_consistency_m"] = r_cons
        full["rotation_window_consistency_deg"] = R_cons
        full["num_windows"] = int(len([x for x in win if x is not None]))
        records.append(full)
    return records


def dt_sensitivity_sequence(seq, args):
    dt_values = [int(x) for x in args.dt_values.split(",") if x.strip()]
    by_dt = []
    for dt in dt_values:
        rec = estimate_sequence(seq, args, dt_frames=dt, compute_windows=False)
        by_dt.append(
            {
                "dt": int(dt),
                "acc_residual_fit": stack_metric(rec, "acc_residual_fit"),
                "acc_improvement": stack_metric(rec, "acc_improvement"),
                "offset_norm": stack_metric(rec, "r_JS").norm(dim=-1),
                "r_JS": stack_metric(rec, "r_JS"),
            }
        )
    fit = torch.stack([x["acc_residual_fit"] for x in by_dt])
    best = torch.argmin(torch.where(torch.isfinite(fit), fit, torch.full_like(fit, float("inf"))), dim=0)
    dt_tensor = torch.tensor(dt_values, dtype=torch.long)
    r_by_dt = torch.stack([x["r_JS"] for x in by_dt])
    zero_idx = dt_values.index(0) if 0 in dt_values else None
    if zero_idx is None:
        pm1_change = torch.full((6,), float("nan"))
    else:
        changes = []
        for dt in (-1, 1):
            if dt in dt_values:
                idx = dt_values.index(dt)
                changes.append((r_by_dt[idx] - r_by_dt[zero_idx]).norm(dim=-1))
        pm1_change = torch.stack(changes).median(dim=0).values if changes else torch.full((6,), float("nan"))
    return {
        "dt_values": dt_tensor,
        "dt_best": dt_tensor[best],
        "per_dt_acc_residual_fit": fit,
        "per_dt_acc_improvement": torch.stack([x["acc_improvement"] for x in by_dt]),
        "per_dt_offset_norm": torch.stack([x["offset_norm"] for x in by_dt]),
        "pm1_offset_change": pm1_change,
    }


def quality_mask(records, args):
    return (
        torch.isfinite(stack_metric(records, "acc_improvement"))
        & torch.isfinite(stack_metric(records, "gyro_improvement"))
        & (stack_metric(records, "r_JS").norm(dim=-1) <= args.quality_max_offset_norm)
        & (stack_metric(records, "acc_improvement") >= args.quality_min_acc_improvement)
        & (stack_metric(records, "gyro_improvement") >= args.quality_min_gyro_improvement)
        & (stack_metric(records, "condition_number") <= args.quality_max_condition)
        & (stack_metric(records, "window_consistency_m") <= args.quality_max_window_consistency)
        & (stack_metric(records, "orientation_residual_fit_deg") <= args.quality_max_orientation_deg)
    )


def build_split_output(data, args, dataset_name, split, source_path):
    count = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    names, seq_records, dt_records, masks = [], [], [], []
    for i in range(count):
        seq = prepare_rawlike_sequence(data, i, args)
        rec = estimate_sequence(seq, args, dt_frames=0)
        dt = dt_sensitivity_sequence(seq, args) if args.dt_sensitivity else None
        names.append(seq["name"])
        seq_records.append(rec)
        dt_records.append(dt)
        masks.append(quality_mask(rec, args))
        print(
            f"[{split} {i + 1}/{count}] {seq['name']} "
            f"acc_imp={tensor_median(stack_metric(rec, 'acc_improvement')):.4f} "
            f"gyro_imp={tensor_median(stack_metric(rec, 'gyro_improvement')):.4f} "
            f"norm={tensor_median(stack_metric(rec, 'r_JS').norm(dim=-1)):.4f}",
            flush=True,
        )
    output = {
        "name": names,
        "R_JS": torch.stack([stack_metric(r, "R_JS") for r in seq_records]),
        "r_JS": torch.stack([stack_metric(r, "r_JS") for r in seq_records]),
        "offset_norm": torch.stack([stack_metric(r, "r_JS").norm(dim=-1) for r in seq_records]),
        "b_aS": torch.stack([stack_metric(r, "b_aS") for r in seq_records]),
        "b_gS": torch.stack([stack_metric(r, "b_gS") for r in seq_records]),
        "orientation_residual_zero_deg": torch.stack([stack_metric(r, "orientation_residual_zero_deg") for r in seq_records]),
        "orientation_residual_fit_deg": torch.stack([stack_metric(r, "orientation_residual_fit_deg") for r in seq_records]),
        "orientation_improvement": torch.stack([stack_metric(r, "orientation_improvement") for r in seq_records]),
        "gyro_residual_zero": torch.stack([stack_metric(r, "gyro_residual_zero") for r in seq_records]),
        "gyro_residual_fit": torch.stack([stack_metric(r, "gyro_residual_fit") for r in seq_records]),
        "gyro_improvement": torch.stack([stack_metric(r, "gyro_improvement") for r in seq_records]),
        "acc_residual_zero": torch.stack([stack_metric(r, "acc_residual_zero") for r in seq_records]),
        "acc_residual_fit": torch.stack([stack_metric(r, "acc_residual_fit") for r in seq_records]),
        "acc_residual_mean_sensor": torch.stack([stack_metric(r, "acc_residual_mean_sensor") for r in seq_records]),
        "gravity_residual_mean_norm": torch.stack([stack_metric(r, "gravity_residual_mean_norm") for r in seq_records]),
        "acc_improvement": torch.stack([stack_metric(r, "acc_improvement") for r in seq_records]),
        "condition_number": torch.stack([stack_metric(r, "condition_number") for r in seq_records]),
        "observability_score": torch.stack([stack_metric(r, "observability_score") for r in seq_records]),
        "window_consistency_m": torch.stack([stack_metric(r, "window_consistency_m") for r in seq_records]),
        "rotation_window_consistency_deg": torch.stack([stack_metric(r, "rotation_window_consistency_deg") for r in seq_records]),
        "quality_mask": torch.stack(masks),
        "sequence_records": seq_records,
        "dt_sensitivity": dt_records,
        "metadata": make_metadata(dataset_name, split, source_path, args),
    }
    output["metadata"]["se3_contract"] = {
        "R_JS": "maps sensor-frame vectors into joint-local frame vectors",
        "r_JS": "sensor origin relative to mapped joint origin, expressed in joint-local frame",
        "orientation_prediction": "R_WS = R_WJ @ R_JS",
        "acceleration_prediction": "aS = R_WS^T @ (ddot(p_WJ) + ddot(R_WJ) @ r_JS - g_W) + b_aS",
    }
    return output


def fixed_synthetic_truth():
    axis_angle = torch.tensor(
        [
            [0.08, -0.03, 0.05],
            [-0.06, -0.04, -0.04],
            [0.05, 0.06, -0.03],
            [-0.05, 0.06, 0.03],
            [0.03, -0.02, 0.08],
            [0.02, 0.03, -0.05],
        ],
        dtype=torch.float32,
    )
    r = torch.tensor(
        [
            [0.08, -0.02, 0.03],
            [-0.08, -0.02, 0.03],
            [0.04, -0.18, 0.02],
            [-0.04, -0.18, 0.02],
            [0.00, 0.12, 0.03],
            [0.00, 0.08, -0.02],
        ],
        dtype=torch.float32,
    )
    return axis_angle_to_matrix(axis_angle), r


def build_synthetic_output(data, args, source_path):
    R_gt, r_gt = fixed_synthetic_truth()
    count = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    names, recs, dt_records = [], [], []
    for i in range(count):
        seq = make_synthetic_sequence(data, i, args, R_gt, r_gt)
        rec = estimate_sequence(seq, args, dt_frames=0)
        dt = dt_sensitivity_sequence(seq, args) if args.dt_sensitivity else None
        names.append(seq["name"])
        recs.append(rec)
        dt_records.append(dt)
        r_err = (stack_metric(rec, "r_JS") - r_gt).norm(dim=-1)
        R_err = rotation_angle_deg(R_gt.transpose(-1, -2).matmul(stack_metric(rec, "R_JS")))
        print(
            f"[synthetic {i + 1}/{count}] {seq['name']} "
            f"rot_med={tensor_median(R_err):.4f}deg r_mean={float(r_err.mean()):.5f}m",
            flush=True,
        )
    R_fit = torch.stack([stack_metric(r, "R_JS") for r in recs])
    r_fit = torch.stack([stack_metric(r, "r_JS") for r in recs])
    output = {
        "name": names,
        "R_JS": R_fit,
        "r_JS": r_fit,
        "R_JS_gt": R_gt,
        "r_JS_gt": r_gt,
        "rotation_error_deg": rotation_angle_deg(R_gt.view(1, 6, 3, 3).transpose(-1, -2).matmul(R_fit)),
        "translation_error_m": (r_fit - r_gt.view(1, 6, 3)).norm(dim=-1),
        "gyro_improvement": torch.stack([stack_metric(r, "gyro_improvement") for r in recs]),
        "acc_improvement": torch.stack([stack_metric(r, "acc_improvement") for r in recs]),
        "dt_sensitivity": dt_records,
        "metadata": make_metadata("synthetic_amass_rawlike", "sanity", source_path, args),
    }
    return output


def summarize_output(output):
    summary = {
        "num_sequences": len(output["name"]),
        "median_offset_norm": tensor_median(output.get("offset_norm", output["r_JS"].norm(dim=-1))),
        "median_acc_improvement": tensor_median(output["acc_improvement"]),
        "median_gyro_improvement": tensor_median(output["gyro_improvement"]),
        "median_condition_number": tensor_median(output.get("condition_number", torch.full_like(output["acc_improvement"], float("nan")))),
        "median_window_consistency_m": tensor_median(output.get("window_consistency_m", torch.full_like(output["acc_improvement"], float("nan")))),
        "quality_mask_fraction": float(output.get("quality_mask", torch.ones_like(output["acc_improvement"], dtype=torch.bool)).float().mean().item()),
    }
    if output.get("dt_sensitivity") and output["dt_sensitivity"][0] is not None:
        best = torch.stack([x["dt_best"] for x in output["dt_sensitivity"]])
        summary["dt0_best_fraction"] = float((best == 0).float().mean().item())
        summary["best_dt_distribution"] = {str(int(v)): int((best == int(v)).sum().item()) for v in sorted(best.unique().tolist())}
        summary["median_pm1_offset_change_m"] = tensor_median(torch.stack([x["pm1_offset_change"] for x in output["dt_sensitivity"]]))
    return summary


def summarize_synthetic(output):
    summary = {
        "num_sequences": len(output["name"]),
        "median_rotation_error_deg": tensor_median(output["rotation_error_deg"]),
        "mean_translation_error_m": float(output["translation_error_m"].mean().item()),
        "max_translation_error_m": float(output["translation_error_m"].max().item()),
        "median_acc_improvement": tensor_median(output["acc_improvement"]),
        "median_gyro_improvement": tensor_median(output["gyro_improvement"]),
    }
    if output.get("dt_sensitivity") and output["dt_sensitivity"][0] is not None:
        best = torch.stack([x["dt_best"] for x in output["dt_sensitivity"]])
        summary["dt0_best_fraction"] = float((best == 0).float().mean().item())
        summary["best_dt_distribution"] = {str(int(v)): int((best == int(v)).sum().item()) for v in sorted(best.unique().tolist())}
    return summary


def summarize_old_cache(split):
    path = OLD_LS_CACHE / f"official_{split}__A_centered_smooth5__offset_cache_v2.pt"
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = torch.load(path, map_location="cpu")
    out = {
        "path": str(path),
        "exists": True,
        "median_offset_norm": tensor_median(data["offset_norm"]),
        "median_acc_improvement": tensor_median(data["residual_improvement"]),
        "median_window_consistency_m": tensor_median(data["window_consistency"]),
        "quality_mask_fraction": float(data["quality_mask"].float().mean().item()),
    }
    if data.get("dt_sensitivity"):
        best = torch.stack([x["dt_best"] for x in data["dt_sensitivity"]])
        out["dt0_best_fraction"] = float((best == 0).float().mean().item())
        out["best_dt_distribution"] = {str(int(v)): int((best == int(v)).sum().item()) for v in sorted(best.unique().tolist())}
    return out


def write_summary(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def to_float(x):
    return float(torch.as_tensor(x).float().item())


def tensor_to_list(x):
    return torch.as_tensor(x).detach().cpu().tolist()


def dt_best_distribution(dt_sensitivity):
    best = torch.stack([x["dt_best"] for x in dt_sensitivity])
    return {str(int(v)): int((best == int(v)).sum().item()) for v in sorted(best.unique().tolist())}


def combine_source_outputs(source_outputs, out_dir, args):
    sequence_id = []
    source_label = []
    combined = {}
    tensor_keys = [
        "R_JS",
        "r_JS",
        "offset_norm",
        "b_aS",
        "b_gS",
        "orientation_residual_zero_deg",
        "orientation_residual_fit_deg",
        "orientation_improvement",
        "gyro_residual_zero",
        "gyro_residual_fit",
        "gyro_improvement",
        "acc_residual_zero",
        "acc_residual_fit",
        "acc_residual_mean_sensor",
        "gravity_residual_mean_norm",
        "acc_improvement",
        "condition_number",
        "observability_score",
        "window_consistency_m",
        "rotation_window_consistency_deg",
        "quality_mask",
    ]
    for label, output in source_outputs:
        sequence_id.extend(output["name"])
        source_label.extend([label] * len(output["name"]))
        for key in tensor_keys:
            combined.setdefault(key, []).append(output[key])
    cache = {
        "sequence_id": sequence_id,
        "source_label": source_label,
        "sensor_names": list(SENSOR_NAMES),
        "processing_mode": "dataset_processing_per_sequence",
        "cache_contract": "se3_cache[sequence_id, sensor_id] = T_JS. Source labels identify files only; they are not train/validation semantics.",
        "metadata": {
            "config": vars(args),
            "quality_rule": {
                "offset_norm_max_m": args.quality_max_offset_norm,
                "acc_improvement_min": args.quality_min_acc_improvement,
                "gyro_improvement_min": args.quality_min_gyro_improvement,
                "condition_number_max": args.quality_max_condition,
                "window_consistency_max_m": args.quality_max_window_consistency,
                "orientation_residual_max_deg": args.quality_max_orientation_deg,
            },
        },
    }
    for key, parts in combined.items():
        cache[key] = torch.cat(parts, dim=0)
    cache_path = out_dir / "totalcapture_full_sequence_se3_cache.pt"
    torch.save(cache, cache_path)

    residual_rows = []
    window_rows = []
    quality_rows = []
    bad_entries = []
    dt_best = torch.cat([torch.stack([x["dt_best"] for x in out["dt_sensitivity"]]) for _, out in source_outputs], dim=0)
    for seq_idx, seq in enumerate(sequence_id):
        for sensor_idx, sensor_name in enumerate(SENSOR_NAMES):
            q = bool(cache["quality_mask"][seq_idx, sensor_idx].item())
            row_base = {
                "sequence_id": seq,
                "source_label": source_label[seq_idx],
                "sensor_id": sensor_idx,
                "sensor_name": sensor_name,
            }
            residual_rows.append(
                {
                    **row_base,
                    "orientation_residual_fit_deg": to_float(cache["orientation_residual_fit_deg"][seq_idx, sensor_idx]),
                    "gyro_residual_zero": to_float(cache["gyro_residual_zero"][seq_idx, sensor_idx]),
                    "gyro_residual_fit": to_float(cache["gyro_residual_fit"][seq_idx, sensor_idx]),
                    "gyro_improvement": to_float(cache["gyro_improvement"][seq_idx, sensor_idx]),
                    "acc_residual_zero": to_float(cache["acc_residual_zero"][seq_idx, sensor_idx]),
                    "acc_residual_fit": to_float(cache["acc_residual_fit"][seq_idx, sensor_idx]),
                    "acc_improvement": to_float(cache["acc_improvement"][seq_idx, sensor_idx]),
                    "gravity_residual_mean_norm": to_float(cache["gravity_residual_mean_norm"][seq_idx, sensor_idx]),
                    "b_aS_norm": to_float(cache["b_aS"][seq_idx, sensor_idx].norm()),
                    "b_gS_norm": to_float(cache["b_gS"][seq_idx, sensor_idx].norm()),
                    "dt_best": int(dt_best[seq_idx, sensor_idx].item()),
                }
            )
            window_rows.append(
                {
                    **row_base,
                    "window_consistency_m": to_float(cache["window_consistency_m"][seq_idx, sensor_idx]),
                    "rotation_window_consistency_deg": to_float(cache["rotation_window_consistency_deg"][seq_idx, sensor_idx]),
                }
            )
            quality_reason = []
            if to_float(cache["offset_norm"][seq_idx, sensor_idx]) > args.quality_max_offset_norm:
                quality_reason.append("offset_norm")
            if to_float(cache["acc_improvement"][seq_idx, sensor_idx]) < args.quality_min_acc_improvement:
                quality_reason.append("acc_improvement")
            if to_float(cache["gyro_improvement"][seq_idx, sensor_idx]) < args.quality_min_gyro_improvement:
                quality_reason.append("gyro_improvement")
            if to_float(cache["condition_number"][seq_idx, sensor_idx]) > args.quality_max_condition:
                quality_reason.append("condition_number")
            if to_float(cache["window_consistency_m"][seq_idx, sensor_idx]) > args.quality_max_window_consistency:
                quality_reason.append("window_consistency")
            if to_float(cache["orientation_residual_fit_deg"][seq_idx, sensor_idx]) > args.quality_max_orientation_deg:
                quality_reason.append("orientation_residual")
            quality_entry = {
                **row_base,
                "quality_mask": q,
                "quality_fail_reasons": quality_reason,
                "offset_norm": to_float(cache["offset_norm"][seq_idx, sensor_idx]),
                "condition_number": to_float(cache["condition_number"][seq_idx, sensor_idx]),
            }
            quality_rows.append(quality_entry)
            if not q:
                bad_entries.append(quality_entry)

    processing_summary = {
        "processing_mode": "dataset_processing_per_sequence",
        "num_sequences": len(sequence_id),
        "num_sequence_sensor_entries": int(cache["quality_mask"].numel()),
        "quality_mask_fraction": float(cache["quality_mask"].float().mean().item()),
        "median_offset_norm_m": tensor_median(cache["offset_norm"]),
        "median_acc_improvement": tensor_median(cache["acc_improvement"]),
        "median_gyro_improvement": tensor_median(cache["gyro_improvement"]),
        "median_orientation_residual_fit_deg": tensor_median(cache["orientation_residual_fit_deg"]),
        "median_window_consistency_m": tensor_median(cache["window_consistency_m"]),
        "median_rotation_window_consistency_deg": tensor_median(cache["rotation_window_consistency_deg"]),
        "median_acc_bias_norm": tensor_median(cache["b_aS"].norm(dim=-1)),
        "median_gyro_bias_norm": tensor_median(cache["b_gS"].norm(dim=-1)),
        "median_gravity_residual_mean_norm": tensor_median(cache["gravity_residual_mean_norm"]),
        "dt0_best_fraction": float((dt_best == 0).float().mean().item()),
        "best_dt_distribution": {str(int(v)): int((dt_best == int(v)).sum().item()) for v in sorted(dt_best.unique().tolist())},
        "bad_entries": bad_entries,
        "paths": {
            "full_sequence_cache": str(cache_path),
            "per_sequence_sensor_residual_report": str(out_dir / "totalcapture_full_sequence_residual_report.json"),
            "window_consistency_report": str(out_dir / "totalcapture_full_sequence_window_consistency_report.json"),
            "quality_report": str(out_dir / "totalcapture_full_sequence_quality_report.json"),
        },
    }
    write_summary(out_dir / "totalcapture_full_sequence_residual_report.json", residual_rows)
    write_summary(out_dir / "totalcapture_full_sequence_window_consistency_report.json", window_rows)
    write_summary(out_dir / "totalcapture_full_sequence_quality_report.json", {"summary": processing_summary, "entries": quality_rows})
    write_summary(out_dir / "totalcapture_dataset_processing_summary.json", processing_summary)
    return cache, processing_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Raw-like TotalCapture per-sequence SE(3) dataset-processing prototype.")
    parser.add_argument("--output-dir", default="data/dataset_work/SensorOffset/rawlike_se3_candidate_a_v1")
    parser.add_argument("--amass-input", default="data/dataset_work/AMASS/globalpose_synth_shard00000.pt")
    parser.add_argument("--official-train", default="data/dataset_work/TotalCapture_globalpose_official/train.pt")
    parser.add_argument("--official-val", default="data/dataset_work/TotalCapture_globalpose_official/val.pt")
    parser.add_argument("--max-synthetic-sequences", type=int, default=3)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=180)
    parser.add_argument("--stride", type=int, default=90)
    parser.add_argument("--min-window-frames", type=int, default=90)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--smoothing-mode", default="moving_average", choices=("none", "moving_average", "centered_moving_average", "savgol"))
    parser.add_argument("--derivative-mode", default="centered", choices=("legacy", "centered", "strict_centered"))
    parser.add_argument("--dt-sensitivity", action="store_true")
    parser.add_argument("--dt-values", default="-3,-2,-1,0,1,2,3")
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--fit-acc-bias", action="store_true")
    parser.add_argument("--acc-bias-ridge", type=float, default=1000.0)
    parser.add_argument("--gyro-bias-ridge", type=float, default=1000.0)
    parser.add_argument("--quality-max-offset-norm", type=float, default=0.5)
    parser.add_argument("--quality-min-acc-improvement", type=float, default=0.05)
    parser.add_argument("--quality-min-gyro-improvement", type=float, default=0.05)
    parser.add_argument("--quality-max-condition", type=float, default=1e8)
    parser.add_argument("--quality-max-window-consistency", type=float, default=0.15)
    parser.add_argument("--quality-max-orientation-deg", type=float, default=20.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combine-existing-only", action="store_true", help="Build full-sequence dataset-processing reports from existing per-source caches.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.combine_existing_only:
        train_output = torch.load(out_dir / "totalcapture_official_train_rawlike_se3.pt", map_location="cpu")
        val_output = torch.load(out_dir / "totalcapture_official_val_rawlike_se3.pt", map_location="cpu")
        _, processing_summary = combine_source_outputs(
            [("official_train_source", train_output), ("official_val_source", val_output)], out_dir, args
        )
        print(json.dumps(processing_summary, indent=2))
        return

    amass_args = argparse.Namespace(**vars(args))
    amass_args.max_sequences = args.max_synthetic_sequences
    amass = load_dataset_file(args.amass_input)
    synthetic = build_synthetic_output(amass, amass_args, args.amass_input)
    torch.save(synthetic, out_dir / "synthetic_se3_sanity.pt")

    train = load_dataset_file(args.official_train)
    train_output = build_split_output(train, args, "totalcapture_rawlike_se3", "official_train", args.official_train)
    torch.save(train_output, out_dir / "totalcapture_official_train_rawlike_se3.pt")

    val = load_dataset_file(args.official_val)
    val_output = build_split_output(val, args, "totalcapture_rawlike_se3", "official_val", args.official_val)
    torch.save(val_output, out_dir / "totalcapture_official_val_rawlike_se3.pt")
    _, processing_summary = combine_source_outputs(
        [("official_train_source", train_output), ("official_val_source", val_output)], out_dir, args
    )

    summary = {
        "config": vars(args),
        "paths": {
            "output_dir": str(out_dir),
            "synthetic": str(out_dir / "synthetic_se3_sanity.pt"),
            "official_train": str(out_dir / "totalcapture_official_train_rawlike_se3.pt"),
            "official_val": str(out_dir / "totalcapture_official_val_rawlike_se3.pt"),
        },
        "synthetic": summarize_synthetic(synthetic),
        "totalcapture_official_train": summarize_output(train_output),
        "totalcapture_official_val": summarize_output(val_output),
        "old_globalpose_aM_ls": {
            "official_train": summarize_old_cache("train"),
            "official_val": summarize_old_cache("val"),
        },
        "dataset_processing": processing_summary,
    }
    write_summary(out_dir / "rawlike_se3_candidate_a_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
