import argparse
import json
from pathlib import Path

import torch

from l4_estimate_sensor_offsets import window_ranges
from l4_rawlike_se3_calibration import (
    fixed_synthetic_truth,
    make_synthetic_sequence,
    matvec,
    prepare_rawlike_sequence,
    rotation_angle_deg,
    stack_metric,
    tensor_median,
)
from l4_rawlike_se3_v3_accel_batch import estimate_rjs_from_orientation
from l4_sensor_offset_utils import (
    FPS,
    GRAVITY_WORLD,
    SENSOR_NAMES,
    first_derivative,
    load_dataset_file,
    second_derivative,
    smooth_centered,
    vee_skew,
)


CANDIDATE_A_DIR = Path("data/dataset_work/SensorOffset/rawlike_se3_candidate_a_v1")


def to_float(x):
    return float(torch.as_tensor(x).float().item())


def finite_median(x):
    return tensor_median(torch.as_tensor(x).float())


def robust_weights(residual_norm, loss, param):
    r = residual_norm.float().clamp_min(1e-12)
    if loss == "l2":
        return torch.ones_like(r)
    if loss == "huber":
        return torch.minimum(torch.ones_like(r), torch.tensor(float(param), dtype=r.dtype) / r)
    if loss == "cauchy":
        c = float(param)
        return 1.0 / (1.0 + (r / c) ** 2)
    if loss == "tukey":
        c = float(param)
        z = r / c
        w = (1.0 - z.square()).clamp_min(0.0).square()
        return torch.where(z < 1.0, w, torch.zeros_like(w))
    raise ValueError(f"Unknown robust loss: {loss}")


def excitation_weights(A, enabled):
    if not enabled:
        return None
    e = A.reshape(A.shape[0], -1).norm(dim=-1)
    med = e[e.isfinite()].median().clamp_min(1e-6)
    # Keep weak-motion frames, but down-weight them instead of dropping them.
    return (e / med).clamp(0.25, 4.0)


def weighted_linear_fit(A, y, args, loss="l2", fit_bias=True, bias_ridge=None, excitation=False, trim_ratio=1.0):
    A = A.float()
    y = y.float()
    valid = torch.isfinite(A).all(dim=(-1, -2)) & torch.isfinite(y).all(dim=-1)
    A = A[valid]
    y = y[valid]
    if A.shape[0] < 12:
        return None
    n = A.shape[0]
    bias_ridge = args.acc_bias_ridge if bias_ridge is None else bias_ridge
    base_w = torch.ones(n, dtype=A.dtype)
    ew = excitation_weights(A, excitation)
    if ew is not None:
        base_w = base_w * ew
    keep = torch.ones(n, dtype=torch.bool)
    r = torch.zeros(3, dtype=A.dtype)
    b = torch.zeros(3, dtype=A.dtype)
    for _ in range(args.irls_iters):
        Ak = A[keep]
        yk = y[keep]
        wk = base_w[keep]
        if fit_bias:
            eye = torch.eye(3, dtype=A.dtype).expand(Ak.shape[0], 3, 3)
            M = torch.cat((Ak, eye), dim=-1).reshape(-1, 6)
            reg = torch.diag(torch.tensor([args.ridge] * 3 + [bias_ridge] * 3, dtype=A.dtype))
        else:
            M = Ak.reshape(-1, 3)
            reg = torch.eye(3, dtype=A.dtype) * args.ridge
        frame_pred = Ak.matmul(r.view(3, 1)).squeeze(-1) + (b.view(1, 3) if fit_bias else 0.0)
        rw = robust_weights((yk - frame_pred).norm(dim=-1), loss, args.robust_param) * wk
        sw = rw.sqrt().repeat_interleave(3)
        Mw = M * sw.view(-1, 1)
        yw = yk.reshape(-1) * sw
        lhs = Mw.T.matmul(Mw) + reg
        rhs = Mw.T.matmul(yw)
        try:
            sol = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            sol = torch.linalg.lstsq(lhs, rhs).solution
        r = sol[:3]
        b = sol[3:6] if fit_bias else torch.zeros(3, dtype=A.dtype)
        if trim_ratio < 1.0:
            res = (y - (A.matmul(r.view(3, 1)).squeeze(-1) + b.view(1, 3))).norm(dim=-1)
            threshold = torch.quantile(res[res.isfinite()], float(trim_ratio))
            keep = res <= threshold
            if keep.float().mean() < args.min_retained_ratio:
                keep = torch.ones(n, dtype=torch.bool)
                break
    pred = A.matmul(r.view(3, 1)).squeeze(-1) + b.view(1, 3)
    residual_vec = y - pred
    residual_norm = residual_vec.norm(dim=-1)
    zero_norm = y.norm(dim=-1)
    s = torch.linalg.svdvals(A[keep].reshape(-1, 3)) if keep.any() else torch.full((3,), float("nan"))
    return {
        "r_JS": r.float(),
        "b_aS": b.float(),
        "pred": pred.float(),
        "residual_vec": residual_vec.float(),
        "acc_residual_zero": zero_norm.mean().float(),
        "acc_residual_fit": residual_norm[keep].mean().float(),
        "acc_improvement": ((zero_norm[keep].mean() - residual_norm[keep].mean()) / zero_norm[keep].mean().clamp_min(1e-12)).float(),
        "retained_ratio": keep.float().mean().float(),
        "mean_weight": base_w[keep].mean().float(),
        "condition_number": (s.max() / s.min().clamp_min(1e-12)).float(),
        "gravity_residual_mean_norm": residual_vec[keep].mean(dim=0).norm().float(),
    }


def prepare_rawlike_sequence_ext(data, seq_idx, args):
    seq = prepare_rawlike_sequence(data, seq_idx, args)
    n = seq["aS"].shape[0]
    seq["RSB"] = data["RSB"][seq_idx].float()
    R_WB = seq["R_WS_obs"].matmul(seq["RSB"].view(1, 6, 3, 3))
    if args.smooth_window > 1 and args.smoothing_mode not in ("none", "identity"):
        R_WB = smooth_centered(R_WB, args.smooth_window, args.smoothing_mode)
    R_dot = first_derivative(R_WB, fps=FPS, mode=args.derivative_mode)
    omega_hat = R_dot.matmul(R_WB.transpose(-1, -2))
    omega_hat = 0.5 * (omega_hat - omega_hat.transpose(-1, -2))
    seq["R_WB_proxy"] = R_WB[:n]
    seq["ddot_R_WB_proxy"] = second_derivative(R_WB, fps=FPS, mode=args.derivative_mode)[:n]
    seq["omega_wb_proxy"] = vee_skew(omega_hat)[:n]
    return seq


def build_linear_system(seq, sensor_idx, R_local_sensor, frame="joint", start=0, end=None):
    s = sensor_idx
    end = seq["aS"].shape[0] if end is None else end
    if frame == "joint":
        R_frame = seq["R_wj"][start:end, s]
        ddot_R_frame = seq["ddot_R_wj"][start:end, s]
        ddot_p = seq["ddot_p_wj"][start:end, s]
        R_FS = R_local_sensor
    elif frame == "segment_proxy":
        R_frame = seq["R_WB_proxy"][start:end, s]
        ddot_R_frame = seq["ddot_R_WB_proxy"][start:end, s]
        ddot_p = seq["ddot_p_wj"][start:end, s]
        R_FS = seq["RSB"][s].T
    else:
        raise ValueError(frame)
    R_WS = R_frame.matmul(R_FS.view(1, 3, 3))
    R_WS_T = R_WS.transpose(-1, -2)
    c = matvec(R_WS_T, ddot_p - GRAVITY_WORLD.view(1, 3))
    A = R_WS_T.matmul(ddot_R_frame)
    y = seq["aS"][start:end, s] - c
    return A, y, R_WS


def gyro_orientation_metrics(seq, sensor_idx, R_JS):
    s = sensor_idx
    R_wj = seq["R_wj"][:, s]
    R_WS_obs = seq["R_WS_obs"][:, s]
    R_WS_pred = R_wj.matmul(R_JS.view(1, 3, 3))
    orient = rotation_angle_deg(R_WS_obs.transpose(-1, -2).matmul(R_WS_pred))
    w_pred = matvec(R_JS.T.matmul(R_wj.transpose(-1, -2)), seq["omega_wj"][:, s])
    valid = torch.isfinite(seq["wS"][:, s]).all(dim=-1) & torch.isfinite(w_pred).all(dim=-1)
    gyro_fit = (seq["wS"][valid, s] - w_pred[valid]).norm(dim=-1)
    gyro_zero = (seq["wS"][valid, s] - matvec(R_wj[valid].transpose(-1, -2), seq["omega_wj"][valid, s])).norm(dim=-1)
    return {
        "orientation_residual_fit_deg": orient[torch.isfinite(orient)].mean().float(),
        "gyro_residual_zero": gyro_zero.mean().float(),
        "gyro_residual_fit": gyro_fit.mean().float(),
        "gyro_improvement": ((gyro_zero.mean() - gyro_fit.mean()) / gyro_zero.mean().clamp_min(1e-12)).float(),
    }


def fit_config(seq, sensor_idx, args, config, start=0, end=None):
    R_JS = estimate_rjs_from_orientation(seq, sensor_idx)
    A, y, _ = build_linear_system(seq, sensor_idx, R_JS, frame=config["frame"], start=start, end=end)
    fit = weighted_linear_fit(
        A,
        y,
        args,
        loss=config["loss"],
        fit_bias=config["fit_bias"],
        bias_ridge=config["bias_ridge"],
        excitation=config["excitation"],
        trim_ratio=config["trim_ratio"],
    )
    if fit is None:
        return None
    metrics = gyro_orientation_metrics(seq, sensor_idx, R_JS)
    out = {**metrics, **fit}
    out["offset_norm"] = out["r_JS"].norm().float()
    out["bias_norm"] = out["b_aS"].norm().float()
    out["R_JS"] = R_JS.float()
    return out


def window_upper_bound(seq, sensor_idx, args, config, window_size, stride):
    ranges = window_ranges(seq["aS"].shape[0], window_size, stride)
    records = [fit_config(seq, sensor_idx, args, config, a, b) for a, b in ranges]
    records = [r for r in records if r is not None]
    if not records:
        return None
    r = torch.stack([x["r_JS"] for x in records])
    imp = torch.stack([x["acc_improvement"] for x in records])
    retained = torch.stack([x["retained_ratio"] for x in records])
    return {
        "acc_improvement": imp.median().float(),
        "window_consistency_m": (r - r.median(dim=0).values.view(1, 3)).norm(dim=-1).median().float(),
        "retained_ratio": retained.median().float(),
        "num_windows": torch.tensor(float(len(records))),
    }


def rotation_window_consistency(seq, sensor_idx, args):
    ranges = window_ranges(seq["aS"].shape[0], args.window_size, args.stride)
    full = estimate_rjs_from_orientation(seq, sensor_idx)
    vals = []
    for a, b in ranges:
        sub = {
            "R_wj": seq["R_wj"][a:b],
            "R_WS_obs": seq["R_WS_obs"][a:b],
        }
        R = estimate_rjs_from_orientation(sub, sensor_idx)
        vals.append(rotation_angle_deg(full.T.matmul(R)))
    return torch.stack(vals).median().float() if vals else torch.tensor(float("nan"))


def configs(args):
    return {
        "l2_no_bias": {"loss": "l2", "fit_bias": False, "bias_ridge": args.acc_bias_ridge, "excitation": False, "trim_ratio": 1.0, "frame": "joint"},
        "l2_bias": {"loss": "l2", "fit_bias": True, "bias_ridge": args.acc_bias_ridge, "excitation": False, "trim_ratio": 1.0, "frame": "joint"},
        "huber_bias": {"loss": "huber", "fit_bias": True, "bias_ridge": args.acc_bias_ridge, "excitation": False, "trim_ratio": 1.0, "frame": "joint"},
        "cauchy_bias": {"loss": "cauchy", "fit_bias": True, "bias_ridge": args.acc_bias_ridge, "excitation": False, "trim_ratio": 1.0, "frame": "joint"},
        "cauchy_excitation": {"loss": "cauchy", "fit_bias": True, "bias_ridge": args.acc_bias_ridge, "excitation": True, "trim_ratio": 1.0, "frame": "joint"},
        "cauchy_trim90": {"loss": "cauchy", "fit_bias": True, "bias_ridge": args.acc_bias_ridge, "excitation": False, "trim_ratio": 0.90, "frame": "joint"},
        "free_bias_upper": {"loss": "l2", "fit_bias": True, "bias_ridge": 1e-6, "excitation": False, "trim_ratio": 1.0, "frame": "joint"},
        "segment_proxy_cauchy": {"loss": "cauchy", "fit_bias": True, "bias_ridge": args.acc_bias_ridge, "excitation": False, "trim_ratio": 1.0, "frame": "segment_proxy"},
    }


def process_source(data, args, label):
    cfgs = configs(args)
    count = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    rows = []
    for i in range(count):
        seq = prepare_rawlike_sequence_ext(data, i, args)
        for s, sensor_name in enumerate(SENSOR_NAMES):
            row = {
                "sequence_id": seq["name"],
                "source_label": label,
                "sensor_id": s,
                "sensor_name": sensor_name,
            }
            for name, cfg in cfgs.items():
                fit = fit_config(seq, s, args, cfg)
                if fit is None:
                    continue
                prefix = name + "__"
                for key in (
                    "acc_improvement",
                    "acc_residual_fit",
                    "offset_norm",
                    "bias_norm",
                    "retained_ratio",
                    "condition_number",
                    "gravity_residual_mean_norm",
                    "gyro_improvement",
                    "orientation_residual_fit_deg",
                ):
                    row[prefix + key] = to_float(fit[key])
            main_fit = fit_config(seq, s, args, cfgs["cauchy_bias"])
            if main_fit is not None:
                row["main_window_upper180__acc_improvement"] = to_float(
                    window_upper_bound(seq, s, args, cfgs["cauchy_bias"], args.window_size, args.stride)["acc_improvement"]
                )
                row["main_window_upper180__window_consistency_m"] = to_float(
                    window_upper_bound(seq, s, args, cfgs["cauchy_bias"], args.window_size, args.stride)["window_consistency_m"]
                )
                short = window_upper_bound(seq, s, args, cfgs["cauchy_bias"], args.short_window_size, args.short_stride)
                row["short_window_upper__acc_improvement"] = to_float(short["acc_improvement"])
                row["short_window_upper__window_consistency_m"] = to_float(short["window_consistency_m"])
                row["rotation_window_consistency_deg"] = to_float(rotation_window_consistency(seq, s, args))
            rows.append(row)
        print(f"[{label} {i + 1}/{count}] {seq['name']}", flush=True)
    return rows


def summarize_rows(rows, key):
    vals = torch.tensor([r[key] for r in rows if key in r and torch.isfinite(torch.tensor(r[key]))], dtype=torch.float32)
    return finite_median(vals)


def per_sensor_summary(rows, selected="cauchy_bias"):
    out = []
    for sensor in SENSOR_NAMES:
        rs = [r for r in rows if r["sensor_name"] == sensor]
        acc = summarize_rows(rs, selected + "__acc_improvement")
        gyro = summarize_rows(rs, selected + "__gyro_improvement")
        orient = summarize_rows(rs, selected + "__orientation_residual_fit_deg")
        norm = summarize_rows(rs, selected + "__offset_norm")
        bias = summarize_rows(rs, selected + "__bias_norm")
        win = summarize_rows(rs, "main_window_upper180__window_consistency_m")
        rotwin = summarize_rows(rs, "rotation_window_consistency_deg")
        seg = summarize_rows(rs, "segment_proxy_cauchy__acc_improvement")
        upper = summarize_rows(rs, "main_window_upper180__acc_improvement")
        short = summarize_rows(rs, "short_window_upper__acc_improvement")
        if acc >= 0.45 and win <= 0.04 and norm <= 0.30:
            label = "reliable"
        elif acc >= 0.25 and win <= 0.06 and norm <= 0.35:
            label = "usable_with_mask"
        elif acc >= 0.10:
            label = "weak"
        else:
            label = "reject"
        out.append(
            {
                "sensor": sensor,
                "median_acc_improvement": acc,
                "median_gyro_improvement": gyro,
                "median_orientation_residual_deg": orient,
                "median_offset_norm_m": norm,
                "median_window_consistency_m": win,
                "median_rotation_window_consistency_deg": rotwin,
                "median_bias_norm": bias,
                "segment_proxy_acc_improvement": seg,
                "window_upper_acc_improvement": upper,
                "short_window_upper_acc_improvement": short,
                "reliability": label,
            }
        )
    return out


def load_candidate_a_summary():
    path = CANDIDATE_A_DIR / "rawlike_se3_candidate_a_summary.json"
    return json.loads(path.read_text()) if path.exists() else {"exists": False, "path": str(path)}


def run_synthetic(args):
    data = load_dataset_file(args.amass_input)
    R_gt, r_gt = fixed_synthetic_truth()
    n = min(len(data["pose"]), args.max_synthetic_sequences)
    rows = []
    for i in range(n):
        seq = make_synthetic_sequence(data, i, args, R_gt, r_gt)
        for s in range(6):
            fit = fit_config(seq, s, args, configs(args)["cauchy_bias"])
            rows.append(
                {
                    "translation_error_m": to_float((fit["r_JS"] - r_gt[s]).norm()),
                    "acc_improvement": to_float(fit["acc_improvement"]),
                    "gyro_improvement": to_float(fit["gyro_improvement"]),
                }
            )
        print(f"[synthetic {i + 1}/{n}] {seq['name']}", flush=True)
    return {
        "num_sequences": n,
        "mean_translation_error_m": float(torch.tensor([r["translation_error_m"] for r in rows]).mean().item()),
        "max_translation_error_m": float(torch.tensor([r["translation_error_m"] for r in rows]).max().item()),
        "median_acc_improvement": summarize_rows(rows, "acc_improvement"),
        "median_gyro_improvement": summarize_rows(rows, "gyro_improvement"),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic = run_synthetic(args)
    train_rows = process_source(load_dataset_file(args.official_train), args, "official_train_source")
    val_rows = process_source(load_dataset_file(args.official_val), args, "official_val_source")
    rows = train_rows + val_rows
    summary = {
        "config": vars(args),
        "policy": {
            "dt_policy": "main experiments are dt=0 only; dt sensitivity is retained only as historical risk context",
            "no_per_window_dt": True,
            "no_per_sequence_dt": True,
        },
        "paths": {
            "output_dir": str(out_dir),
            "per_sequence_sensor_report": str(out_dir / "accel_residual_explanation_no_dt_rows.json"),
            "summary": str(out_dir / "accel_residual_explanation_no_dt_summary.json"),
        },
        "synthetic": synthetic,
        "candidate_a": load_candidate_a_summary(),
        "overall": {},
        "per_sensor": per_sensor_summary(rows),
    }
    for name in configs(args).keys():
        summary["overall"][name] = {
            "median_acc_improvement": summarize_rows(rows, name + "__acc_improvement"),
            "median_acc_residual_fit": summarize_rows(rows, name + "__acc_residual_fit"),
            "median_offset_norm_m": summarize_rows(rows, name + "__offset_norm"),
            "median_bias_norm": summarize_rows(rows, name + "__bias_norm"),
            "median_retained_ratio": summarize_rows(rows, name + "__retained_ratio"),
            "median_gravity_residual_mean_norm": summarize_rows(rows, name + "__gravity_residual_mean_norm"),
        }
    summary["upper_bounds"] = {
        "window180_cauchy_bias_median_acc_improvement": summarize_rows(rows, "main_window_upper180__acc_improvement"),
        "window180_cauchy_bias_median_window_consistency_m": summarize_rows(rows, "main_window_upper180__window_consistency_m"),
        "short_window_cauchy_bias_median_acc_improvement": summarize_rows(rows, "short_window_upper__acc_improvement"),
        "short_window_cauchy_bias_median_window_consistency_m": summarize_rows(rows, "short_window_upper__window_consistency_m"),
        "free_bias_median_acc_improvement": summary["overall"]["free_bias_upper"]["median_acc_improvement"],
        "free_bias_median_bias_norm": summary["overall"]["free_bias_upper"]["median_bias_norm"],
    }
    (out_dir / "accel_residual_explanation_no_dt_rows.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "accel_residual_explanation_no_dt_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="TotalCapture acceleration residual explanation without dt correction.")
    parser.add_argument("--output-dir", default="data/dataset_work/SensorOffset/accel_residual_explanation_no_dt")
    parser.add_argument("--amass-input", default="data/dataset_work/AMASS/globalpose_synth_shard00000.pt")
    parser.add_argument("--official-train", default="data/dataset_work/TotalCapture_globalpose_official/train.pt")
    parser.add_argument("--official-val", default="data/dataset_work/TotalCapture_globalpose_official/val.pt")
    parser.add_argument("--max-synthetic-sequences", type=int, default=3)
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
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
