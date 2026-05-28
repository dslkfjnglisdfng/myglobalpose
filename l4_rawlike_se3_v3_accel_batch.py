import argparse
import json
from pathlib import Path

import torch

from l4_estimate_sensor_offsets import window_ranges
from l4_rawlike_se3_calibration import (
    align_by_dt,
    dt_best_distribution,
    fixed_synthetic_truth,
    make_synthetic_sequence,
    matvec,
    prepare_rawlike_sequence,
    robust_rotation_mean,
    rotation_angle_deg,
    stack_metric,
    summarize_old_cache,
    tensor_median,
)
from l4_sensor_offset_utils import (
    GRAVITY_WORLD,
    SENSOR_NAMES,
    load_dataset_file,
    make_metadata,
)


CANDIDATE_A_DIR = Path("data/dataset_work/SensorOffset/rawlike_se3_candidate_a_v1")


def nanmean(x):
    x = torch.as_tensor(x).float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return torch.tensor(float("nan"))
    return x.mean()


def to_float(x):
    return float(torch.as_tensor(x).float().item())


def fit_acc_linear_v3(A, y, args):
    A = A.float()
    y = y.float()
    if A.shape[0] < 8:
        return None
    n = A.shape[0]
    eye = torch.eye(3, dtype=A.dtype).expand(n, 3, 3)
    if args.fit_acc_scale:
        # aS ~= c + A r + b + (s - 1) * aS.  The scale term is heavily
        # regularized and disabled by default; it is diagnostic-only.
        scale_col = y.reshape(-1, 1)
        M = torch.cat((A, eye), dim=-1).reshape(-1, 6)
        M = torch.cat((M, scale_col), dim=-1)
        reg_diag = [
            args.ridge,
            args.ridge,
            args.ridge,
            args.acc_bias_ridge,
            args.acc_bias_ridge,
            args.acc_bias_ridge,
            args.acc_scale_ridge,
        ]
    else:
        M = torch.cat((A, eye), dim=-1).reshape(-1, 6)
        reg_diag = [
            args.ridge,
            args.ridge,
            args.ridge,
            args.acc_bias_ridge,
            args.acc_bias_ridge,
            args.acc_bias_ridge,
        ]
    reg = torch.diag(torch.tensor(reg_diag, dtype=A.dtype))
    target = y.reshape(-1)
    lhs = M.T.matmul(M) + reg
    rhs = M.T.matmul(target)
    try:
        sol = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        sol = torch.linalg.lstsq(lhs, rhs).solution
    r = sol[:3]
    b = sol[3:6]
    scale_delta = sol[6] if args.fit_acc_scale else torch.tensor(0.0, dtype=A.dtype)
    pred = A.matmul(r.view(3, 1)).squeeze(-1) + b.view(1, 3)
    if args.fit_acc_scale:
        pred = pred + scale_delta * y
    s = torch.linalg.svdvals(A.reshape(-1, 3))
    return {
        "r_JS": r.float(),
        "b_aS": b.float(),
        "acc_scale": (1.0 + scale_delta).float(),
        "pred": pred.float(),
        "condition_number": (s.max() / s.min().clamp_min(1e-12)).float(),
        "observability_score": s.min().float(),
        "singular_values": s.float(),
        "num_valid_frames": int(A.shape[0]),
    }


def estimate_rjs_from_orientation(seq, sensor_idx):
    R_wj = seq["R_wj"][:, sensor_idx]
    R_WS_obs = seq["R_WS_obs"][:, sensor_idx]
    valid = torch.isfinite(R_wj).all(dim=(-1, -2)) & torch.isfinite(R_WS_obs).all(dim=(-1, -2))
    if valid.sum() < 8:
        return torch.eye(3)
    return robust_rotation_mean(R_wj[valid].transpose(-1, -2).matmul(R_WS_obs[valid]))


def fit_sensor_fixed_R(seq, sensor_idx, R_JS, start, end, dt_frames, args):
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
        n_keep = R_wj.shape[0]
        if dt_frames > 0:
            aS = aS[dt_frames : dt_frames + n_keep]
            wS = wS[dt_frames : dt_frames + n_keep]
        else:
            aS = aS[:n_keep]
            wS = wS[:n_keep]

    R_WS_pred = R_wj.matmul(R_JS.view(1, 3, 3))
    orient_fit = rotation_angle_deg(R_WS_obs.transpose(-1, -2).matmul(R_WS_pred))
    orient_zero = rotation_angle_deg(R_WS_obs.transpose(-1, -2).matmul(R_wj))
    w_pred = matvec(R_JS.T.matmul(R_wj.transpose(-1, -2)), omega_wj)
    gyro_valid = torch.isfinite(wS).all(dim=-1) & torch.isfinite(w_pred).all(dim=-1)
    gyro_res = wS[gyro_valid] - w_pred[gyro_valid]
    gyro_bias = gyro_res.sum(dim=0) / (gyro_res.shape[0] + args.gyro_bias_ridge) if gyro_res.numel() else torch.zeros(3)
    gyro_zero = (wS[gyro_valid] - matvec(R_wj[gyro_valid].transpose(-1, -2), omega_wj[gyro_valid])).norm(dim=-1)
    gyro_fit = (gyro_res - gyro_bias.view(1, 3)).norm(dim=-1)

    R_WS_T = R_WS_pred.transpose(-1, -2)
    c = matvec(R_WS_T, ddot_p - GRAVITY_WORLD.view(1, 3))
    A = R_WS_T.matmul(ddot_R)
    valid = torch.isfinite(A).all(dim=(-1, -2)) & torch.isfinite(c).all(dim=-1) & torch.isfinite(aS).all(dim=-1)
    A_valid = A[valid]
    c_valid = c[valid]
    aS_valid = aS[valid]
    fit = fit_acc_linear_v3(A_valid, aS_valid - c_valid, args)
    if fit is None:
        return None
    acc_zero = (aS_valid - c_valid).norm(dim=-1)
    acc_fit_vec = aS_valid - (c_valid + fit["pred"])
    acc_fit = acc_fit_vec.norm(dim=-1)
    return {
        "R_JS": R_JS.float(),
        "r_JS": fit["r_JS"],
        "b_aS": fit["b_aS"],
        "b_gS": gyro_bias.float(),
        "acc_scale": fit["acc_scale"],
        "orientation_residual_zero_deg": nanmean(orient_zero).float(),
        "orientation_residual_fit_deg": nanmean(orient_fit).float(),
        "gyro_residual_zero": nanmean(gyro_zero).float(),
        "gyro_residual_fit": nanmean(gyro_fit).float(),
        "gyro_improvement": ((nanmean(gyro_zero) - nanmean(gyro_fit)) / nanmean(gyro_zero).clamp_min(1e-12)).float(),
        "acc_residual_zero": nanmean(acc_zero).float(),
        "acc_residual_fit": nanmean(acc_fit).float(),
        "acc_improvement": ((nanmean(acc_zero) - nanmean(acc_fit)) / nanmean(acc_zero).clamp_min(1e-12)).float(),
        "acc_residual_mean_sensor": acc_fit_vec.mean(dim=0).float(),
        "gravity_residual_mean_norm": acc_fit_vec.mean(dim=0).norm().float(),
        "condition_number": fit["condition_number"],
        "observability_score": fit["observability_score"],
        "singular_values": fit["singular_values"],
        "num_valid_frames": fit["num_valid_frames"],
    }


def aggregate_window_consistency(seq, sensor_idx, R_JS, dt_frames, args):
    n = seq["aS"].shape[0] - abs(int(dt_frames))
    if n < args.min_window_frames:
        return torch.tensor(float("nan"))
    ranges = window_ranges(n, args.window_size, args.stride)
    records = [fit_sensor_fixed_R(seq, sensor_idx, R_JS, a, b, dt_frames, args) for a, b in ranges]
    valid = [r for r in records if r is not None]
    if not valid:
        return torch.tensor(float("nan"))
    r = torch.stack([x["r_JS"] for x in valid])
    return (r - r.median(dim=0).values.view(1, 3)).norm(dim=-1).median().float()


def estimate_sequence_v3(seq, args):
    R_fixed = [estimate_rjs_from_orientation(seq, s) for s in range(6)]
    dt_values = [int(x) for x in args.dt_values.split(",") if x.strip()]
    candidates = []
    for dt in dt_values:
        records = [fit_sensor_fixed_R(seq, s, R_fixed[s], 0, seq["aS"].shape[0], dt, args) for s in range(6)]
        candidates.append({"dt": dt, "records": records, "median_acc_fit": tensor_median(stack_metric(records, "acc_residual_fit"))})
    best_dt = min(candidates, key=lambda x: x["median_acc_fit"])["dt"]
    records = next(x["records"] for x in candidates if x["dt"] == best_dt)
    for s, rec in enumerate(records):
        if rec is not None:
            rec["window_consistency_m"] = aggregate_window_consistency(seq, s, R_fixed[s], best_dt, args)
            rec["sequence_dt"] = torch.tensor(float(best_dt))
    dt_fit = torch.stack([stack_metric(x["records"], "acc_residual_fit") for x in candidates])
    dt_imp = torch.stack([stack_metric(x["records"], "acc_improvement") for x in candidates])
    dt_norm = torch.stack([stack_metric(x["records"], "r_JS").norm(dim=-1) for x in candidates])
    dt_tensor = torch.tensor(dt_values, dtype=torch.long)
    best_per_sensor = dt_tensor[torch.argmin(torch.where(torch.isfinite(dt_fit), dt_fit, torch.full_like(dt_fit, float("inf"))), dim=0)]
    zero_idx = dt_values.index(0) if 0 in dt_values else None
    if zero_idx is None:
        pm1 = torch.full((6,), float("nan"))
    else:
        r_by_dt = torch.stack([stack_metric(x["records"], "r_JS") for x in candidates])
        changes = []
        for dt in (-1, 1):
            if dt in dt_values:
                changes.append((r_by_dt[dt_values.index(dt)] - r_by_dt[zero_idx]).norm(dim=-1))
        pm1 = torch.stack(changes).median(dim=0).values if changes else torch.full((6,), float("nan"))
    return records, {
        "dt_values": dt_tensor,
        "sequence_dt": int(best_dt),
        "dt_best": best_per_sensor,
        "per_dt_acc_residual_fit": dt_fit,
        "per_dt_acc_improvement": dt_imp,
        "per_dt_offset_norm": dt_norm,
        "pm1_offset_change": pm1,
    }


def quality_mask(records, args):
    return (
        torch.isfinite(stack_metric(records, "acc_improvement"))
        & (stack_metric(records, "r_JS").norm(dim=-1) <= args.quality_max_offset_norm)
        & (stack_metric(records, "acc_improvement") >= args.quality_min_acc_improvement)
        & (stack_metric(records, "gyro_improvement") >= args.quality_min_gyro_improvement)
        & (stack_metric(records, "condition_number") <= args.quality_max_condition)
        & (stack_metric(records, "window_consistency_m") <= args.quality_max_window_consistency)
        & (stack_metric(records, "b_aS").norm(dim=-1) <= args.quality_max_acc_bias_norm)
    )


def build_source_output(data, args, label, source_path):
    count = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    names, seq_records, dt_records, masks = [], [], [], []
    for i in range(count):
        seq = prepare_rawlike_sequence(data, i, args)
        records, dt = estimate_sequence_v3(seq, args)
        names.append(seq["name"])
        seq_records.append(records)
        dt_records.append(dt)
        masks.append(quality_mask(records, args))
        print(
            f"[{label} {i + 1}/{count}] {seq['name']} "
            f"seq_dt={dt['sequence_dt']} "
            f"acc_imp={tensor_median(stack_metric(records, 'acc_improvement')):.4f} "
            f"norm={tensor_median(stack_metric(records, 'r_JS').norm(dim=-1)):.4f} "
            f"b_a={tensor_median(stack_metric(records, 'b_aS').norm(dim=-1)):.4f}",
            flush=True,
        )
    out = {
        "name": names,
        "source_label": label,
        "R_JS": torch.stack([stack_metric(r, "R_JS") for r in seq_records]),
        "r_JS": torch.stack([stack_metric(r, "r_JS") for r in seq_records]),
        "offset_norm": torch.stack([stack_metric(r, "r_JS").norm(dim=-1) for r in seq_records]),
        "b_aS": torch.stack([stack_metric(r, "b_aS") for r in seq_records]),
        "b_gS": torch.stack([stack_metric(r, "b_gS") for r in seq_records]),
        "acc_scale": torch.stack([stack_metric(r, "acc_scale") for r in seq_records]),
        "orientation_residual_fit_deg": torch.stack([stack_metric(r, "orientation_residual_fit_deg") for r in seq_records]),
        "gyro_residual_zero": torch.stack([stack_metric(r, "gyro_residual_zero") for r in seq_records]),
        "gyro_residual_fit": torch.stack([stack_metric(r, "gyro_residual_fit") for r in seq_records]),
        "gyro_improvement": torch.stack([stack_metric(r, "gyro_improvement") for r in seq_records]),
        "acc_residual_zero": torch.stack([stack_metric(r, "acc_residual_zero") for r in seq_records]),
        "acc_residual_fit": torch.stack([stack_metric(r, "acc_residual_fit") for r in seq_records]),
        "acc_improvement": torch.stack([stack_metric(r, "acc_improvement") for r in seq_records]),
        "acc_residual_mean_sensor": torch.stack([stack_metric(r, "acc_residual_mean_sensor") for r in seq_records]),
        "gravity_residual_mean_norm": torch.stack([stack_metric(r, "gravity_residual_mean_norm") for r in seq_records]),
        "condition_number": torch.stack([stack_metric(r, "condition_number") for r in seq_records]),
        "observability_score": torch.stack([stack_metric(r, "observability_score") for r in seq_records]),
        "window_consistency_m": torch.stack([stack_metric(r, "window_consistency_m") for r in seq_records]),
        "quality_mask": torch.stack(masks),
        "dt_sensitivity": dt_records,
        "sequence_dt": torch.tensor([x["sequence_dt"] for x in dt_records], dtype=torch.long),
        "sequence_records": seq_records,
        "metadata": make_metadata("totalcapture_rawlike_se3_v3", label, source_path, args),
    }
    out["metadata"]["v3_contract"] = {
        "processing_mode": "dataset_processing_per_sequence",
        "R_JS": "fixed from full-sequence orientation robust mean at dt=0; maps sensor-frame vectors into joint-local vectors",
        "r_JS": "refit with acceleration residual and one sequence-level integer dt shared by all six sensors",
        "bias_policy": "sequence/sensor-level b_aS with strong L2 ridge; no per-window dt",
        "acc_scale_policy": "disabled by default; available only as diagnostic with strong ridge",
    }
    return out


def summarize_output(out):
    summary = {
        "num_sequences": len(out["name"]),
        "median_offset_norm_m": tensor_median(out["offset_norm"]),
        "median_acc_improvement": tensor_median(out["acc_improvement"]),
        "median_acc_residual_fit": tensor_median(out["acc_residual_fit"]),
        "median_gyro_improvement": tensor_median(out["gyro_improvement"]),
        "median_window_consistency_m": tensor_median(out["window_consistency_m"]),
        "quality_mask_fraction": float(out["quality_mask"].float().mean().item()),
        "median_acc_bias_norm": tensor_median(out["b_aS"].norm(dim=-1)),
        "median_gravity_residual_mean_norm": tensor_median(out["gravity_residual_mean_norm"]),
        "sequence_dt_distribution": {str(int(v)): int((out["sequence_dt"] == int(v)).sum().item()) for v in sorted(out["sequence_dt"].unique().tolist())},
    }
    best = torch.stack([x["dt_best"] for x in out["dt_sensitivity"]])
    summary["dt0_best_fraction_per_sensor"] = float((best == 0).float().mean().item())
    summary["best_dt_distribution_per_sensor"] = {str(int(v)): int((best == int(v)).sum().item()) for v in sorted(best.unique().tolist())}
    summary["median_pm1_offset_change_m"] = tensor_median(torch.stack([x["pm1_offset_change"] for x in out["dt_sensitivity"]]))
    return summary


def summarize_synthetic(out, R_gt, r_gt):
    r_err = (out["r_JS"] - r_gt.view(1, 6, 3)).norm(dim=-1)
    R_err = rotation_angle_deg(R_gt.view(1, 6, 3, 3).transpose(-1, -2).matmul(out["R_JS"]))
    return {
        "num_sequences": len(out["name"]),
        "median_rotation_error_deg": tensor_median(R_err),
        "mean_translation_error_m": float(r_err.mean().item()),
        "max_translation_error_m": float(r_err.max().item()),
        "median_acc_improvement": tensor_median(out["acc_improvement"]),
        "median_gyro_improvement": tensor_median(out["gyro_improvement"]),
        "sequence_dt_distribution": {str(int(v)): int((out["sequence_dt"] == int(v)).sum().item()) for v in sorted(out["sequence_dt"].unique().tolist())},
    }


def combine_outputs(outputs, out_dir, args):
    cache = {
        "sequence_id": sum([list(o["name"]) for o in outputs], []),
        "source_label": sum([[o["source_label"]] * len(o["name"]) for o in outputs], []),
        "sensor_names": list(SENSOR_NAMES),
        "processing_mode": "dataset_processing_per_sequence",
        "cache_contract": "v3 se3_cache[sequence_id, sensor_id] = T_JS. Source labels identify files only.",
        "metadata": {"config": vars(args)},
    }
    keys = [
        "R_JS",
        "r_JS",
        "offset_norm",
        "b_aS",
        "b_gS",
        "acc_scale",
        "orientation_residual_fit_deg",
        "gyro_residual_zero",
        "gyro_residual_fit",
        "gyro_improvement",
        "acc_residual_zero",
        "acc_residual_fit",
        "acc_improvement",
        "acc_residual_mean_sensor",
        "gravity_residual_mean_norm",
        "condition_number",
        "observability_score",
        "window_consistency_m",
        "quality_mask",
    ]
    for key in keys:
        cache[key] = torch.cat([o[key] for o in outputs], dim=0)
    cache["sequence_dt"] = torch.cat([o["sequence_dt"] for o in outputs], dim=0)
    cache_path = out_dir / "totalcapture_full_sequence_se3_v3_cache.pt"
    torch.save(cache, cache_path)
    rows = []
    bad = []
    for i, seq in enumerate(cache["sequence_id"]):
        for s, sensor in enumerate(SENSOR_NAMES):
            row = {
                "sequence_id": seq,
                "source_label": cache["source_label"][i],
                "sensor_id": s,
                "sensor_name": sensor,
                "sequence_dt": int(cache["sequence_dt"][i].item()),
                "offset_norm": to_float(cache["offset_norm"][i, s]),
                "acc_improvement": to_float(cache["acc_improvement"][i, s]),
                "acc_residual_fit": to_float(cache["acc_residual_fit"][i, s]),
                "gyro_improvement": to_float(cache["gyro_improvement"][i, s]),
                "window_consistency_m": to_float(cache["window_consistency_m"][i, s]),
                "b_aS_norm": to_float(cache["b_aS"][i, s].norm()),
                "gravity_residual_mean_norm": to_float(cache["gravity_residual_mean_norm"][i, s]),
                "quality_mask": bool(cache["quality_mask"][i, s].item()),
            }
            rows.append(row)
            if not row["quality_mask"]:
                bad.append(row)
    summary = {
        "num_sequences": len(cache["sequence_id"]),
        "num_sequence_sensor_entries": int(cache["quality_mask"].numel()),
        "median_offset_norm_m": tensor_median(cache["offset_norm"]),
        "median_acc_improvement": tensor_median(cache["acc_improvement"]),
        "median_acc_residual_fit": tensor_median(cache["acc_residual_fit"]),
        "median_gyro_improvement": tensor_median(cache["gyro_improvement"]),
        "median_window_consistency_m": tensor_median(cache["window_consistency_m"]),
        "quality_mask_fraction": float(cache["quality_mask"].float().mean().item()),
        "median_acc_bias_norm": tensor_median(cache["b_aS"].norm(dim=-1)),
        "median_gravity_residual_mean_norm": tensor_median(cache["gravity_residual_mean_norm"]),
        "sequence_dt_distribution": {str(int(v)): int((cache["sequence_dt"] == int(v)).sum().item()) for v in sorted(cache["sequence_dt"].unique().tolist())},
        "bad_entries": bad,
        "paths": {
            "cache": str(cache_path),
            "per_sequence_sensor_report": str(out_dir / "totalcapture_full_sequence_se3_v3_report.json"),
        },
    }
    (out_dir / "totalcapture_full_sequence_se3_v3_report.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "totalcapture_full_sequence_se3_v3_processing_summary.json").write_text(json.dumps(summary, indent=2))
    return cache, summary


def load_candidate_a_summary():
    path = CANDIDATE_A_DIR / "rawlike_se3_candidate_a_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"exists": False, "path": str(path)}


def run_synthetic(args):
    data = load_dataset_file(args.amass_input)
    R_gt, r_gt = fixed_synthetic_truth()
    synth_args = argparse.Namespace(**vars(args))
    synth_args.max_sequences = args.max_synthetic_sequences
    count = min(len(data["pose"]), synth_args.max_sequences)
    names, records, dts = [], [], []
    for i in range(count):
        seq = make_synthetic_sequence(data, i, synth_args, R_gt, r_gt)
        rec, dt = estimate_sequence_v3(seq, synth_args)
        names.append(seq["name"])
        records.append(rec)
        dts.append(dt)
        print(
            f"[synthetic {i + 1}/{count}] {seq['name']} seq_dt={dt['sequence_dt']} "
            f"r_err_mean={float((stack_metric(rec, 'r_JS') - r_gt).norm(dim=-1).mean()):.6f}",
            flush=True,
        )
    out = {
        "name": names,
        "source_label": "synthetic_sanity",
        "R_JS": torch.stack([stack_metric(r, "R_JS") for r in records]),
        "r_JS": torch.stack([stack_metric(r, "r_JS") for r in records]),
        "offset_norm": torch.stack([stack_metric(r, "r_JS").norm(dim=-1) for r in records]),
        "b_aS": torch.stack([stack_metric(r, "b_aS") for r in records]),
        "b_gS": torch.stack([stack_metric(r, "b_gS") for r in records]),
        "acc_scale": torch.stack([stack_metric(r, "acc_scale") for r in records]),
        "acc_improvement": torch.stack([stack_metric(r, "acc_improvement") for r in records]),
        "gyro_improvement": torch.stack([stack_metric(r, "gyro_improvement") for r in records]),
        "window_consistency_m": torch.stack([stack_metric(r, "window_consistency_m") for r in records]),
        "quality_mask": torch.stack([quality_mask(r, args) for r in records]),
        "dt_sensitivity": dts,
        "sequence_dt": torch.tensor([x["sequence_dt"] for x in dts], dtype=torch.long),
    }
    return out, R_gt, r_gt


def per_sensor_summary(cache):
    out = []
    for i, sensor in enumerate(SENSOR_NAMES):
        out.append(
            {
                "sensor": sensor,
                "median_acc_improvement": tensor_median(cache["acc_improvement"][:, i]),
                "median_offset_norm_m": tensor_median(cache["offset_norm"][:, i]),
                "median_window_consistency_m": tensor_median(cache["window_consistency_m"][:, i]),
                "median_acc_bias_norm": tensor_median(cache["b_aS"][:, i].norm(dim=-1)),
                "quality_fraction": float(cache["quality_mask"][:, i].float().mean().item()),
            }
        )
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="TotalCapture raw-like SE(3) v3 acceleration residual batch diagnostic.")
    parser.add_argument("--output-dir", default="data/dataset_work/SensorOffset/rawlike_se3_v3_accel_batch")
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
    parser.add_argument("--dt-values", default="-3,-2,-1,0,1,2,3")
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--acc-bias-ridge", type=float, default=1000.0)
    parser.add_argument("--gyro-bias-ridge", type=float, default=1000.0)
    parser.add_argument("--fit-acc-scale", action="store_true")
    parser.add_argument("--acc-scale-ridge", type=float, default=100000.0)
    parser.add_argument("--quality-max-offset-norm", type=float, default=0.5)
    parser.add_argument("--quality-min-acc-improvement", type=float, default=0.05)
    parser.add_argument("--quality-min-gyro-improvement", type=float, default=0.05)
    parser.add_argument("--quality-max-condition", type=float, default=1e8)
    parser.add_argument("--quality-max-window-consistency", type=float, default=0.15)
    parser.add_argument("--quality-max-acc-bias-norm", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic, R_gt, r_gt = run_synthetic(args)
    torch.save(synthetic, out_dir / "synthetic_se3_v3_sanity.pt")

    train = build_source_output(load_dataset_file(args.official_train), args, "official_train_source", args.official_train)
    torch.save(train, out_dir / "totalcapture_official_train_se3_v3.pt")
    val = build_source_output(load_dataset_file(args.official_val), args, "official_val_source", args.official_val)
    torch.save(val, out_dir / "totalcapture_official_val_se3_v3.pt")
    cache, processing_summary = combine_outputs([train, val], out_dir, args)

    summary = {
        "config": vars(args),
        "paths": {
            "output_dir": str(out_dir),
            "synthetic": str(out_dir / "synthetic_se3_v3_sanity.pt"),
            "cache": str(out_dir / "totalcapture_full_sequence_se3_v3_cache.pt"),
            "report": str(out_dir / "totalcapture_full_sequence_se3_v3_report.json"),
        },
        "synthetic": summarize_synthetic(synthetic, R_gt, r_gt),
        "candidate_a": load_candidate_a_summary(),
        "old_globalpose_aM_ls": {
            "official_train": summarize_old_cache("train"),
            "official_val": summarize_old_cache("val"),
        },
        "totalcapture_official_train_source": summarize_output(train),
        "totalcapture_official_val_source": summarize_output(val),
        "dataset_processing": processing_summary,
        "per_sensor": per_sensor_summary(cache),
    }
    (out_dir / "rawlike_se3_v3_accel_batch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
