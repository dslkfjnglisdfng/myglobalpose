import argparse
import json
from pathlib import Path

import torch

from l4_accel_residual_explanation_no_dt import build_linear_system, configs, fit_config, prepare_rawlike_sequence_ext
from l4_rawlike_se3_v3_accel_batch import estimate_rjs_from_orientation
from l4_sensor_offset_utils import SENSOR_NAMES, load_dataset_file, moving_average


GRAVITY_NORM = 9.8


def finite(x):
    x = torch.as_tensor(x).float()
    return x[torch.isfinite(x)]


def rms(x):
    x = finite(x)
    if x.numel() == 0:
        return float("nan")
    return float(torch.sqrt((x * x).mean()).item())


def med(x):
    x = finite(x)
    if x.numel() == 0:
        return float("nan")
    return float(x.median().item())


def mean(x):
    x = finite(x)
    if x.numel() == 0:
        return float("nan")
    return float(x.mean().item())


def quantile(x, p):
    x = finite(x)
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x, p).item())


def action_name(sequence_id):
    parts = sequence_id.split("_")
    if len(parts) < 2:
        return "unknown"
    return "".join([c for c in parts[1] if not c.isdigit()]) or parts[1]


def grade(value):
    if value < 0.5:
        return "GOOD"
    if value < 1.0:
        return "ACCEPTABLE"
    if value < 2.0:
        return "MODERATE"
    return "BAD"


def vector_axis_stats(v):
    return {
        "x": mean(v[:, 0]),
        "y": mean(v[:, 1]),
        "z": mean(v[:, 2]),
    }, {
        "x": float(v[:, 0].std().item()),
        "y": float(v[:, 1].std().item()),
        "z": float(v[:, 2].std().item()),
    }


def residual_metrics(aS, pred, c, residual, lowpass_window):
    res_norm = residual.norm(dim=-1)
    zero_vec = aS - c
    zero_norm = zero_vec.norm(dim=-1)
    pred_norm = pred.norm(dim=-1)
    obs_norm = aS.norm(dim=-1)
    axis_mean, axis_std = vector_axis_stats(residual)
    obs_rms = torch.sqrt((obs_norm * obs_norm).mean()).clamp_min(1e-12)
    pred_rms = torch.sqrt((pred_norm * pred_norm).mean()).clamp_min(1e-12)
    obs_med = obs_norm.median().clamp_min(1e-12)
    fit_rms = torch.sqrt((res_norm * res_norm).mean())
    zero_rms = torch.sqrt((zero_norm * zero_norm).mean())
    return {
        "residual_zero_rms": float(zero_rms.item()),
        "residual_fit_rms": float(fit_rms.item()),
        "residual_fit_mean_abs": mean(res_norm),
        "residual_fit_median_abs": med(res_norm),
        "residual_fit_p90": quantile(res_norm, 0.90),
        "residual_fit_p95": quantile(res_norm, 0.95),
        "residual_fit_clipped_max_p99": quantile(res_norm, 0.99),
        "residual_fit_per_axis_mean": axis_mean,
        "residual_fit_per_axis_std": axis_std,
        "residual_fit_rms_over_g": float((fit_rms / GRAVITY_NORM).item()),
        "residual_fit_rms_over_obs_rms": float((fit_rms / obs_rms).item()),
        "residual_fit_rms_over_pred_rms": float((fit_rms / pred_rms).item()),
        "residual_fit_rms_over_obs_median_norm": float((fit_rms / obs_med).item()),
        "residual_improvement_rms": float(((zero_rms - fit_rms) / zero_rms.clamp_min(1e-12)).item()),
        "acceptability": grade(float(fit_rms.item())),
    }


def low_high_metrics(aS, pred, c, residual, lowpass_window):
    a_lp = moving_average(aS, lowpass_window)
    p_lp = moving_average(pred, lowpass_window)
    c_lp = moving_average(c, lowpass_window)
    r_lp = a_lp - p_lp
    a_hp = aS - a_lp
    p_hp = pred - p_lp
    c_hp = c - c_lp
    r_hp = a_hp - p_hp
    low_rms = torch.sqrt((r_lp.norm(dim=-1) ** 2).mean())
    high_rms = torch.sqrt((r_hp.norm(dim=-1) ** 2).mean())
    low_obs_rms = torch.sqrt((a_lp.norm(dim=-1) ** 2).mean()).clamp_min(1e-12)
    high_obs_rms = torch.sqrt((a_hp.norm(dim=-1) ** 2).mean()).clamp_min(1e-12)
    low_zero_rms = torch.sqrt(((a_lp - c_lp).norm(dim=-1) ** 2).mean()).clamp_min(1e-12)
    high_zero_rms = torch.sqrt(((a_hp - c_hp).norm(dim=-1) ** 2).mean()).clamp_min(1e-12)
    return {
        "low_freq_residual_fit_rms": float(low_rms.item()),
        "high_freq_residual_fit_rms": float(high_rms.item()),
        "low_freq_residual_fit_over_g": float((low_rms / GRAVITY_NORM).item()),
        "high_freq_residual_fit_over_g": float((high_rms / GRAVITY_NORM).item()),
        "low_freq_residual_fit_over_low_obs_rms": float((low_rms / low_obs_rms).item()),
        "high_freq_residual_fit_over_high_obs_rms": float((high_rms / high_obs_rms).item()),
        "low_freq_residual_improvement_rms": float(((low_zero_rms - low_rms) / low_zero_rms).item()),
        "high_freq_residual_improvement_rms": float(((high_zero_rms - high_rms) / high_zero_rms).item()),
        "high_freq_ratio_of_fit_rms": float((high_rms / torch.sqrt((residual.norm(dim=-1) ** 2).mean()).clamp_min(1e-12)).item()),
    }


def audit_sequence(data, seq_idx, label, args):
    seq = prepare_rawlike_sequence_ext(data, seq_idx, args)
    rows = []
    cfg = configs(args)["l2_bias"]
    for s, sensor_name in enumerate(SENSOR_NAMES):
        fit = fit_config(seq, s, args, cfg)
        if fit is None:
            continue
        R_JS = estimate_rjs_from_orientation(seq, s)
        A, y, _ = build_linear_system(seq, s, R_JS, frame="joint")
        valid = torch.isfinite(A).all(dim=(-1, -2)) & torch.isfinite(y).all(dim=-1)
        A = A[valid]
        y = y[valid]
        aS = seq["aS"][:, s][valid]
        c = aS - y
        pred = c + A.matmul(fit["r_JS"].view(3, 1)).squeeze(-1) + fit["b_aS"].view(1, 3)
        residual = aS - pred
        row = {
            "sequence_id": seq["name"],
            "source_label": label,
            "action": action_name(seq["name"]),
            "sensor_id": s,
            "sensor_name": sensor_name,
            "num_frames": int(aS.shape[0]),
            "residual_improvement": float(fit["acc_improvement"].item()),
        }
        row.update(residual_metrics(aS, pred, c, residual, args.lowpass_window))
        row.update(low_high_metrics(aS, pred, c, residual, args.lowpass_window))
        rows.append(row)
    return rows


def process_source(path, label, args):
    data = load_dataset_file(path)
    count = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    rows = []
    for i in range(count):
        rows.extend(audit_sequence(data, i, label, args))
        print(f"[{label} {i + 1}/{count}]", flush=True)
    return rows


def summarize(rows, keys):
    out = {}
    for key in keys:
        out[key] = med([r[key] for r in rows])
    return out


def summarize_group(rows, group_key):
    keys = [
        "residual_fit_rms",
        "residual_fit_p95",
        "residual_fit_rms_over_g",
        "residual_fit_rms_over_obs_rms",
        "low_freq_residual_fit_rms",
        "high_freq_residual_fit_rms",
        "low_freq_residual_fit_over_g",
        "high_freq_residual_fit_over_g",
        "low_freq_residual_fit_over_low_obs_rms",
        "high_freq_residual_fit_over_high_obs_rms",
        "residual_improvement",
        "high_freq_ratio_of_fit_rms",
    ]
    output = []
    for value in sorted(set(r[group_key] for r in rows)):
        group = [r for r in rows if r[group_key] == value]
        item = {group_key: value, "entries": len(group)}
        item.update(summarize(group, keys))
        item["acceptability"] = grade(item["residual_fit_rms"])
        if item["residual_fit_rms"] < 1.0 and item["high_freq_residual_fit_rms"] < 1.0:
            item["reliability_label"] = "reliable_absolute"
        elif item["low_freq_residual_fit_rms"] < 1.0 and item["high_freq_residual_fit_rms"] >= 1.0:
            item["reliability_label"] = "low_freq_only"
        elif item["residual_fit_rms"] < 2.0:
            item["reliability_label"] = "usable_with_mask"
        else:
            item["reliability_label"] = "mask_required"
        output.append(item)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="TotalCapture static SE(3) absolute acceleration residual audit.")
    parser.add_argument("--output-dir", default="data/dataset_work/SensorOffset/accel_absolute_residual_audit")
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
    overall_keys = [
        "residual_zero_rms",
        "residual_fit_rms",
        "residual_fit_mean_abs",
        "residual_fit_median_abs",
        "residual_fit_p90",
        "residual_fit_p95",
        "residual_fit_clipped_max_p99",
        "residual_fit_rms_over_g",
        "residual_fit_rms_over_obs_rms",
        "residual_fit_rms_over_pred_rms",
        "residual_fit_rms_over_obs_median_norm",
        "low_freq_residual_fit_rms",
        "high_freq_residual_fit_rms",
        "low_freq_residual_fit_over_g",
        "high_freq_residual_fit_over_g",
        "low_freq_residual_fit_over_low_obs_rms",
        "high_freq_residual_fit_over_high_obs_rms",
        "residual_improvement",
        "high_freq_ratio_of_fit_rms",
    ]
    summary = {
        "config": vars(args),
        "paths": {
            "rows": str(out_dir / "accel_absolute_residual_rows.json"),
            "summary": str(out_dir / "accel_absolute_residual_summary.json"),
        },
        "thresholds": {
            "GOOD": "residual_fit_rms < 0.5 m/s^2 or < 5% gravity",
            "ACCEPTABLE": "0.5-1.0 m/s^2 or 5%-10% gravity",
            "MODERATE": "1.0-2.0 m/s^2 or 10%-20% gravity",
            "BAD": ">2.0 m/s^2 or >20% gravity",
        },
        "overall": summarize(rows, overall_keys),
        "per_sensor": summarize_group(rows, "sensor_name"),
        "per_action": summarize_group(rows, "action"),
    }
    summary["overall"]["acceptability"] = grade(summary["overall"]["residual_fit_rms"])
    (out_dir / "accel_absolute_residual_rows.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "accel_absolute_residual_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
