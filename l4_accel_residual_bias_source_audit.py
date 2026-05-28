import argparse
import json
from pathlib import Path

import torch

from l4_accel_residual_explanation_no_dt import build_linear_system, configs, fit_config, prepare_rawlike_sequence_ext
from l4_rawlike_se3_calibration import matvec, rotation_angle_deg
from l4_rawlike_se3_v3_accel_batch import estimate_rjs_from_orientation
from l4_sensor_offset_utils import GRAVITY_WORLD, SENSOR_NAMES, load_dataset_file, moving_average


NO_DT_SUMMARY = Path("data/dataset_work/SensorOffset/accel_residual_explanation_no_dt/accel_residual_explanation_no_dt_summary.json")


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


def mean(x):
    x = torch.as_tensor(x).float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan")
    return float(x.mean().item())


def to_float(x):
    return float(torch.as_tensor(x).float().item())


def fit_scalar_scale(obs, pred, baseline_zero):
    valid = torch.isfinite(obs).all(dim=-1) & torch.isfinite(pred).all(dim=-1)
    obs = obs[valid]
    pred = pred[valid]
    baseline_zero = baseline_zero[valid]
    if obs.shape[0] < 8:
        return None
    # obs ~= s * pred + b
    M = torch.cat((pred.reshape(-1, 1), torch.eye(3).repeat(obs.shape[0], 1, 1).reshape(-1, 3)), dim=1)
    sol = torch.linalg.lstsq(M, obs.reshape(-1)).solution
    s = sol[0]
    b = sol[1:4]
    fit = s * pred + b.view(1, 3)
    res = (obs - fit).norm(dim=-1)
    zero = baseline_zero.norm(dim=-1).mean().clamp_min(1e-12)
    return {
        "scale": to_float(s),
        "bias_norm": to_float(b.norm()),
        "residual_fit": to_float(res.mean()),
        "improvement": to_float((zero - res.mean()) / zero),
    }


def fit_diag_scale(obs, pred, baseline_zero):
    valid = torch.isfinite(obs).all(dim=-1) & torch.isfinite(pred).all(dim=-1)
    obs = obs[valid]
    pred = pred[valid]
    baseline_zero = baseline_zero[valid]
    if obs.shape[0] < 8:
        return None
    scales = []
    bias = []
    fit_axes = []
    for axis in range(3):
        M = torch.stack((pred[:, axis], torch.ones_like(pred[:, axis])), dim=1)
        sol = torch.linalg.lstsq(M, obs[:, axis]).solution
        scales.append(sol[0])
        bias.append(sol[1])
        fit_axes.append(M.matmul(sol))
    fit = torch.stack(fit_axes, dim=1)
    s = torch.stack(scales)
    b = torch.stack(bias)
    res = (obs - fit).norm(dim=-1)
    zero = baseline_zero.norm(dim=-1).mean().clamp_min(1e-12)
    return {
        "scale_x": to_float(s[0]),
        "scale_y": to_float(s[1]),
        "scale_z": to_float(s[2]),
        "scale_spread": to_float((s - 1.0).abs().max()),
        "bias_norm": to_float(b.norm()),
        "residual_fit": to_float(res.mean()),
        "improvement": to_float((zero - res.mean()) / zero),
    }


def low_high_frequency(obs, pred, zero_pred, window):
    obs_lp = moving_average(obs, window)
    pred_lp = moving_average(pred, window)
    zero_lp = moving_average(zero_pred, window)
    obs_hp = obs - obs_lp
    pred_hp = pred - pred_lp
    zero_hp = zero_pred - zero_lp
    low_res = (obs_lp - pred_lp).norm(dim=-1)
    low_zero = (obs_lp - zero_lp).norm(dim=-1).mean().clamp_min(1e-12)
    high_res = (obs_hp - pred_hp).norm(dim=-1)
    high_zero = (obs_hp - zero_hp).norm(dim=-1).mean().clamp_min(1e-12)
    full_res = (obs - pred).norm(dim=-1).mean().clamp_min(1e-12)
    return {
        "low_improvement": to_float((low_zero - low_res.mean()) / low_zero),
        "high_improvement": to_float((high_zero - high_res.mean()) / high_zero),
        "low_residual": to_float(low_res.mean()),
        "high_residual": to_float(high_res.mean()),
        "high_to_full_residual_ratio": to_float(high_res.mean() / full_res),
    }


def action_name(sequence_id):
    parts = sequence_id.split("_")
    if len(parts) < 2:
        return "unknown"
    return "".join([c for c in parts[1] if not c.isdigit()]) or parts[1]


def audit_sequence(data, seq_idx, label, args):
    seq = prepare_rawlike_sequence_ext(data, seq_idx, args)
    rows = []
    cfg = configs(args)["l2_bias"]
    for s, sensor_name in enumerate(SENSOR_NAMES):
        fit = fit_config(seq, s, args, cfg)
        if fit is None:
            continue
        R_JS = estimate_rjs_from_orientation(seq, s)
        A, y, R_WS = build_linear_system(seq, s, R_JS, frame="joint")
        valid = torch.isfinite(A).all(dim=(-1, -2)) & torch.isfinite(y).all(dim=-1)
        A = A[valid]
        y = y[valid]
        aS = seq["aS"][:, s][valid]
        R_WS = R_WS[valid]
        c = aS - y
        pred_no_bias = c + A.matmul(fit["r_JS"].view(3, 1)).squeeze(-1)
        pred = pred_no_bias + fit["b_aS"].view(1, 3)
        residual = aS - pred
        residual_norm = residual.norm(dim=-1)
        zero_vec = aS - c

        R_obs = seq["R_WS_obs"][:, s][valid]
        orient_angle = rotation_angle_deg(R_obs.transpose(-1, -2).matmul(R_WS))
        g_obs = matvec(R_obs.transpose(-1, -2), -GRAVITY_WORLD.view(1, 3).expand(R_obs.shape[0], 3))
        g_pred = matvec(R_WS.transpose(-1, -2), -GRAVITY_WORLD.view(1, 3).expand(R_WS.shape[0], 3))
        g_proj_err = (g_obs - g_pred).norm(dim=-1)
        g_dir = g_pred / g_pred.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        residual_g_frac = (residual * g_dir).sum(dim=-1).abs() / residual_norm.clamp_min(1e-12)

        motion_score = seq["wS"][:, s][valid].norm(dim=-1) + A.reshape(A.shape[0], -1).norm(dim=-1) / 50.0
        low_motion = motion_score <= torch.quantile(motion_score[torch.isfinite(motion_score)], args.low_motion_quantile)
        low_res = residual[low_motion]
        low_res_norm = low_res.norm(dim=-1)
        low_res_g_frac = residual_g_frac[low_motion]

        scalar = fit_scalar_scale(aS, pred_no_bias, zero_vec)
        diag = fit_diag_scale(aS, pred_no_bias, zero_vec)
        freq = low_high_frequency(aS, pred, c, args.lowpass_window)

        acc_mag = zero_vec.norm(dim=-1)
        hi_acc = acc_mag >= torch.quantile(acc_mag[torch.isfinite(acc_mag)], 0.75)
        rows.append(
            {
                "sequence_id": seq["name"],
                "source_label": label,
                "action": action_name(seq["name"]),
                "sensor_id": s,
                "sensor_name": sensor_name,
                "acc_improvement": to_float(fit["acc_improvement"]),
                "acc_residual_fit": to_float(fit["acc_residual_fit"]),
                "offset_norm": to_float(fit["offset_norm"]),
                "b_aS_x": to_float(fit["b_aS"][0]),
                "b_aS_y": to_float(fit["b_aS"][1]),
                "b_aS_z": to_float(fit["b_aS"][2]),
                "b_aS_norm": to_float(fit["bias_norm"]),
                "orientation_residual_deg": to_float(fit["orientation_residual_fit_deg"]),
                "gravity_projection_error_median": tensor_median(g_proj_err),
                "gravity_projection_error_p95": q(g_proj_err, 0.95),
                "gravity_projection_error_bound_from_angle_median": float(2 * 9.8 * torch.sin(torch.deg2rad(orient_angle).median() * 0.5).item()),
                "residual_gravity_axis_abs_fraction_median": tensor_median(residual_g_frac),
                "low_motion_residual_mean_norm": to_float(low_res.mean(dim=0).norm()) if low_res.numel() else float("nan"),
                "low_motion_residual_std_norm": to_float(low_res.std(dim=0).norm()) if low_res.shape[0] > 1 else float("nan"),
                "low_motion_residual_gravity_axis_fraction": tensor_median(low_res_g_frac),
                "scalar_scale": scalar["scale"],
                "scalar_scale_improvement": scalar["improvement"],
                "scalar_scale_bias_norm": scalar["bias_norm"],
                "diag_scale_x": diag["scale_x"],
                "diag_scale_y": diag["scale_y"],
                "diag_scale_z": diag["scale_z"],
                "diag_scale_spread": diag["scale_spread"],
                "diag_scale_improvement": diag["improvement"],
                "diag_scale_bias_norm": diag["bias_norm"],
                "low_freq_improvement": freq["low_improvement"],
                "high_freq_improvement": freq["high_improvement"],
                "low_freq_residual": freq["low_residual"],
                "high_freq_residual": freq["high_residual"],
                "high_to_full_residual_ratio": freq["high_to_full_residual_ratio"],
                "high_acc_residual_median": tensor_median(residual_norm[hi_acc]),
                "low_acc_residual_median": tensor_median(residual_norm[~hi_acc]),
            }
        )
    return rows


def process_source(path, label, args):
    data = load_dataset_file(path)
    count = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    rows = []
    for i in range(count):
        rows.extend(audit_sequence(data, i, label, args))
        print(f"[{label} {i + 1}/{count}]", flush=True)
    return rows


def summarize(rows, key):
    return tensor_median([r[key] for r in rows if key in r])


def p95(rows, key):
    return q([r[key] for r in rows if key in r], 0.95)


def per_sensor(rows):
    no_dt = {}
    if NO_DT_SUMMARY.exists():
        s = json.loads(NO_DT_SUMMARY.read_text())
        no_dt = {x["sensor"]: x for x in s.get("per_sensor", [])}
    out = []
    for sensor in SENSOR_NAMES:
        rs = [r for r in rows if r["sensor_name"] == sensor]
        prev = no_dt.get(sensor, {})
        acc = summarize(rs, "acc_improvement")
        low = summarize(rs, "low_freq_improvement")
        high = summarize(rs, "high_freq_improvement")
        scale_gain = summarize(rs, "diag_scale_improvement") - acc
        upper_gap = prev.get("short_window_upper_acc_improvement", float("nan")) - acc
        if scale_gain > 0.08 or abs(summarize(rs, "diag_scale_spread")) > 0.25:
            factor = "scale_axis_calibration"
        elif low - high > 0.15 or summarize(rs, "high_to_full_residual_ratio") > 0.65:
            factor = "filtering_mismatch"
        elif upper_gap > 0.08:
            factor = "soft_tissue_strap"
        elif summarize(rs, "gravity_projection_error_median") > 0.8:
            factor = "gravity_projection"
        elif summarize(rs, "b_aS_norm") > 0.5:
            factor = "accelerometer_bias"
        elif acc < 0.30:
            factor = "static_rJS_limit"
        else:
            factor = "mixed_static_limit"
        out.append(
            {
                "sensor": sensor,
                "acc_improvement": acc,
                "b_aS_norm_median": summarize(rs, "b_aS_norm"),
                "b_aS_norm_p95": p95(rs, "b_aS_norm"),
                "gravity_projection_error_median": summarize(rs, "gravity_projection_error_median"),
                "residual_gravity_axis_fraction": summarize(rs, "residual_gravity_axis_abs_fraction_median"),
                "diag_scale_improvement": summarize(rs, "diag_scale_improvement"),
                "diag_scale_spread": summarize(rs, "diag_scale_spread"),
                "low_freq_improvement": low,
                "high_freq_improvement": high,
                "high_to_full_residual_ratio": summarize(rs, "high_to_full_residual_ratio"),
                "window_upper_gap": upper_gap,
                "likely_limiting_factor": factor,
            }
        )
    return out


def action_summary(rows):
    out = []
    for action in sorted(set(r["action"] for r in rows)):
        rs = [r for r in rows if r["action"] == action]
        out.append(
            {
                "action": action,
                "entries": len(rs),
                "median_acc_improvement": summarize(rs, "acc_improvement"),
                "median_residual_fit": summarize(rs, "acc_residual_fit"),
                "median_high_acc_residual": summarize(rs, "high_acc_residual_median"),
            }
        )
    return sorted(out, key=lambda x: x["median_residual_fit"], reverse=True)


def parse_args():
    parser = argparse.ArgumentParser(description="TotalCapture acceleration residual bias/source audit.")
    parser.add_argument("--output-dir", default="data/dataset_work/SensorOffset/accel_residual_bias_source_audit")
    parser.add_argument("--official-train", default="data/dataset_work/TotalCapture_globalpose_official/train.pt")
    parser.add_argument("--official-val", default="data/dataset_work/TotalCapture_globalpose_official/val.pt")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--smoothing-mode", default="moving_average", choices=("none", "moving_average", "centered_moving_average", "savgol"))
    parser.add_argument("--derivative-mode", default="centered", choices=("legacy", "centered", "strict_centered"))
    parser.add_argument("--window-size", type=int, default=180)
    parser.add_argument("--stride", type=int, default=90)
    parser.add_argument("--short-window-size", type=int, default=60)
    parser.add_argument("--short-stride", type=int, default=30)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--acc-bias-ridge", type=float, default=1000.0)
    parser.add_argument("--robust-param", type=float, default=1.5)
    parser.add_argument("--irls-iters", type=int, default=4)
    parser.add_argument("--min-retained-ratio", type=float, default=0.85)
    parser.add_argument("--low-motion-quantile", type=float, default=0.25)
    parser.add_argument("--lowpass-window", type=int, default=15)
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
        "paths": {
            "rows": str(out_dir / "accel_residual_bias_source_rows.json"),
            "summary": str(out_dir / "accel_residual_bias_source_summary.json"),
        },
        "overall": {
            "acc_improvement": summarize(rows, "acc_improvement"),
            "b_aS_norm_median": summarize(rows, "b_aS_norm"),
            "b_aS_norm_p95": p95(rows, "b_aS_norm"),
            "gravity_projection_error_median": summarize(rows, "gravity_projection_error_median"),
            "gravity_projection_error_p95": p95(rows, "gravity_projection_error_p95"),
            "residual_gravity_axis_fraction": summarize(rows, "residual_gravity_axis_abs_fraction_median"),
            "low_motion_residual_mean_norm": summarize(rows, "low_motion_residual_mean_norm"),
            "scalar_scale_improvement": summarize(rows, "scalar_scale_improvement"),
            "scalar_scale_median": summarize(rows, "scalar_scale"),
            "diag_scale_improvement": summarize(rows, "diag_scale_improvement"),
            "diag_scale_spread": summarize(rows, "diag_scale_spread"),
            "low_freq_improvement": summarize(rows, "low_freq_improvement"),
            "high_freq_improvement": summarize(rows, "high_freq_improvement"),
            "high_to_full_residual_ratio": summarize(rows, "high_to_full_residual_ratio"),
        },
        "per_sensor": per_sensor(rows),
        "action_summary_by_residual": action_summary(rows),
    }
    (out_dir / "accel_residual_bias_source_rows.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "accel_residual_bias_source_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
