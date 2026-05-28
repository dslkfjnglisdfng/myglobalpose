import argparse
import json
from pathlib import Path

import torch

from l4_estimate_sensor_offsets import window_ranges
from l4_rawlike_se3_calibration import robust_rotation_mean, rotation_angle_deg, matvec, tensor_median
from l4_sensor_offset_utils import (
    FPS,
    SENSOR_NAMES,
    first_derivative,
    fk_imu_joints_and_vertices,
    load_dataset_file,
    smooth_centered,
    vee_skew,
)


def nanmean(x):
    x = torch.as_tensor(x).float()
    x = x[torch.isfinite(x)]
    return x.mean() if x.numel() else torch.tensor(float("nan"))


def percentile(x, q):
    x = torch.as_tensor(x).float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x, q).item())


def summarize_tensor(x):
    return {
        "median": tensor_median(x),
        "p25": percentile(x, 0.25),
        "p75": percentile(x, 0.75),
        "p95": percentile(x, 0.95),
    }


def to_float(x):
    return float(torch.as_tensor(x).float().item())


def prepare_sequence(data, seq_idx, args):
    pose = data["pose"][seq_idx].float()
    tran = data["tran"][seq_idx].float()
    RIM = data["RIM"][seq_idx].float()
    RSB = data["RSB"][seq_idx].float()
    RIS = data["RIS"][seq_idx].float()
    wS = data["wS"][seq_idx].float()
    n = min(pose.shape[0], tran.shape[0], RIS.shape[0], wS.shape[0])
    if args.max_frames > 0:
        n = min(n, args.max_frames)
    pose, tran = pose[:n], tran[:n]
    RIS, wS = RIS[:n], wS[:n]
    _, R_wj, _ = fk_imu_joints_and_vertices(pose, tran, device=args.device)
    R_WS_obs = RIM.transpose(1, 2).matmul(RIS)
    if args.smooth_window > 1 and args.smoothing_mode not in ("none", "identity"):
        R_wj = smooth_centered(R_wj, args.smooth_window, args.smoothing_mode)
        wS = smooth_centered(wS, args.smooth_window, args.smoothing_mode)
    R_dot = first_derivative(R_wj, fps=FPS, mode=args.derivative_mode)
    omega_hat = R_dot.matmul(R_wj.transpose(-1, -2))
    omega_hat = 0.5 * (omega_hat - omega_hat.transpose(-1, -2))
    omega_wj = vee_skew(omega_hat)
    return {
        "name": str(data["name"][seq_idx]) if "name" in data else f"seq_{seq_idx}",
        "R_wj": R_wj,
        "R_WS_obs": R_WS_obs,
        "R_JS_official": RSB.transpose(-1, -2),
        "wS": wS,
        "omega_wj": omega_wj,
    }


def align_by_dt(kin, obs, dt):
    dt = int(dt)
    if dt > 0:
        return kin[:-dt], obs[dt:]
    if dt < 0:
        k = -dt
        return kin[k:], obs[:-k]
    return kin, obs


def gyro_residual(R_wj, omega_wj, wS, R_JS):
    pred = matvec(R_JS.T.matmul(R_wj.transpose(-1, -2)), omega_wj)
    valid = torch.isfinite(pred).all(dim=-1) & torch.isfinite(wS).all(dim=-1)
    if valid.sum() < 3:
        return torch.tensor(float("nan"))
    return (wS[valid] - pred[valid]).norm(dim=-1).mean()


def orientation_residual(R_wj, R_WS_obs, R_JS):
    pred = R_wj.matmul(R_JS.view(1, 3, 3))
    valid = torch.isfinite(pred).all(dim=(-1, -2)) & torch.isfinite(R_WS_obs).all(dim=(-1, -2))
    if valid.sum() < 3:
        return torch.tensor(float("nan"))
    return rotation_angle_deg(R_WS_obs[valid].transpose(-1, -2).matmul(pred[valid])).mean()


def estimate_rjs_from_orientation(R_wj, R_WS_obs):
    valid = torch.isfinite(R_wj).all(dim=(-1, -2)) & torch.isfinite(R_WS_obs).all(dim=(-1, -2))
    if valid.sum() < 3:
        return torch.eye(3)
    return robust_rotation_mean(R_wj[valid].transpose(-1, -2).matmul(R_WS_obs[valid]))


def sensor_audit(seq, sensor_idx, R_JS_est, args):
    s = sensor_idx
    R_wj = seq["R_wj"][:, s]
    R_obs = seq["R_WS_obs"][:, s]
    wS = seq["wS"][:, s]
    omega = seq["omega_wj"][:, s]
    R_off = seq["R_JS_official"][s]
    R_identity = torch.eye(3)

    orient_zero = orientation_residual(R_wj, R_obs, R_identity)
    orient_off = orientation_residual(R_wj, R_obs, R_off)
    orient_est = orientation_residual(R_wj, R_obs, R_JS_est)

    gyro_zero = gyro_residual(R_wj, omega, wS, R_identity)
    gyro_off = gyro_residual(R_wj, omega, wS, R_off)
    gyro_est = gyro_residual(R_wj, omega, wS, R_JS_est)

    n = R_wj.shape[0]
    windows = window_ranges(n, args.window_size, args.stride) if n >= args.min_window_frames else [(0, n)]
    win_off_res = []
    win_est_rot = []
    for a, b in windows:
        win_off_res.append(orientation_residual(R_wj[a:b], R_obs[a:b], R_off))
        win_est_rot.append(estimate_rjs_from_orientation(R_wj[a:b], R_obs[a:b]))
    win_off_res = torch.stack(win_off_res)
    win_est_rot = torch.stack(win_est_rot)
    R_win_mean = robust_rotation_mean(win_est_rot)
    win_est_cons = rotation_angle_deg(R_win_mean.T.view(1, 3, 3).matmul(win_est_rot)).median()
    win_off_vs_est = rotation_angle_deg(R_off.T.view(1, 3, 3).matmul(win_est_rot)).median()

    dt_values = [int(x) for x in args.dt_values.split(",") if x.strip()]
    dt_fit = []
    for dt in dt_values:
        R_dt, R_obs_dt = align_by_dt(R_wj, R_obs, dt)
        omega_dt, wS_dt = align_by_dt(omega, wS, dt)
        # Keep orientation alignment available for debugging, but choose dt by gyro residual.
        _ = orientation_residual(R_dt, R_obs_dt, R_off)
        dt_fit.append(gyro_residual(R_dt, omega_dt, wS_dt, R_off))
    dt_fit = torch.stack(dt_fit)
    best_idx = torch.argmin(torch.where(torch.isfinite(dt_fit), dt_fit, torch.full_like(dt_fit, float("inf"))))
    dt_best = dt_values[int(best_idx)]

    return {
        "R_JS_official": R_off.float(),
        "R_JS_estimated": R_JS_est.float(),
        "official_vs_estimated_deg": rotation_angle_deg(R_off.T.matmul(R_JS_est)).float(),
        "orientation_residual_zero_deg": orient_zero.float(),
        "orientation_residual_official_deg": orient_off.float(),
        "orientation_residual_estimated_deg": orient_est.float(),
        "orientation_improvement_official": ((orient_zero - orient_off) / orient_zero.clamp_min(1e-12)).float(),
        "orientation_improvement_estimated": ((orient_zero - orient_est) / orient_zero.clamp_min(1e-12)).float(),
        "gyro_residual_zero": gyro_zero.float(),
        "gyro_residual_official": gyro_off.float(),
        "gyro_residual_estimated": gyro_est.float(),
        "gyro_improvement_official": ((gyro_zero - gyro_off) / gyro_zero.clamp_min(1e-12)).float(),
        "gyro_improvement_estimated": ((gyro_zero - gyro_est) / gyro_zero.clamp_min(1e-12)).float(),
        "window_orientation_residual_official_median_deg": win_off_res.median().float(),
        "window_orientation_residual_official_std_deg": win_off_res.std().float() if win_off_res.numel() > 1 else torch.tensor(0.0),
        "window_estimated_RJS_consistency_deg": win_est_cons.float(),
        "window_official_vs_estimated_RJS_median_deg": win_off_vs_est.float(),
        "gyro_dt_best_official": int(dt_best),
        "gyro_dt_residual_fit_official": dt_fit.float(),
    }


def load_estimated_cache(path):
    if not path or not Path(path).exists():
        return None
    return torch.load(path, map_location="cpu")


def find_estimated_rjs(cache, sequence_id, source_label, sensor_idx):
    if cache is None:
        return None
    for i, (seq, src) in enumerate(zip(cache["sequence_id"], cache["source_label"])):
        if seq == sequence_id and src == source_label:
            return cache["R_JS"][i, sensor_idx].float()
    for i, seq in enumerate(cache["sequence_id"]):
        if seq == sequence_id:
            return cache["R_JS"][i, sensor_idx].float()
    return None


def build_source_output(data, source_label, args, estimated_cache=None):
    count = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    names = []
    records = []
    for i in range(count):
        seq = prepare_sequence(data, i, args)
        names.append(seq["name"])
        sensor_records = []
        for s in range(6):
            est = find_estimated_rjs(estimated_cache, seq["name"], source_label, s)
            if est is None:
                est = estimate_rjs_from_orientation(seq["R_wj"][:, s], seq["R_WS_obs"][:, s])
            sensor_records.append(sensor_audit(seq, s, est, args))
        records.append(sensor_records)
        print(
            f"[{source_label} {i + 1}/{count}] {seq['name']} "
            f"official_ori={tensor_median(stack_metric(sensor_records, 'orientation_residual_official_deg')):.3f}deg "
            f"official_gyro={tensor_median(stack_metric(sensor_records, 'gyro_improvement_official')):.3f}",
            flush=True,
        )
    return {
        "sequence_id": names,
        "source_label": source_label,
        "records": records,
        **stack_records(records),
    }


def stack_metric(records, key):
    return torch.stack([torch.as_tensor(r[key]).float() for r in records])


def stack_records(records):
    keys = [
        "R_JS_official",
        "R_JS_estimated",
        "official_vs_estimated_deg",
        "orientation_residual_zero_deg",
        "orientation_residual_official_deg",
        "orientation_residual_estimated_deg",
        "orientation_improvement_official",
        "orientation_improvement_estimated",
        "gyro_residual_zero",
        "gyro_residual_official",
        "gyro_residual_estimated",
        "gyro_improvement_official",
        "gyro_improvement_estimated",
        "window_orientation_residual_official_median_deg",
        "window_orientation_residual_official_std_deg",
        "window_estimated_RJS_consistency_deg",
        "window_official_vs_estimated_RJS_median_deg",
        "gyro_dt_best_official",
    ]
    out = {}
    for key in keys:
        out[key] = torch.stack([torch.stack([torch.as_tensor(r[key]).float() for r in seq]) for seq in records])
    out["gyro_dt_residual_fit_official"] = torch.stack(
        [torch.stack([r["gyro_dt_residual_fit_official"] for r in seq]) for seq in records]
    )
    return out


def combine_outputs(outputs):
    combined = {
        "sequence_id": [],
        "source_label": [],
        "sensor_names": list(SENSOR_NAMES),
        "records": [],
    }
    keys = [key for key in outputs[0].keys() if key not in ("sequence_id", "source_label", "records")]
    for output in outputs:
        combined["sequence_id"].extend(output["sequence_id"])
        combined["source_label"].extend([output["source_label"]] * len(output["sequence_id"]))
        combined["records"].extend(output["records"])
        for key in keys:
            combined.setdefault(key, []).append(output[key])
    for key in keys:
        combined[key] = torch.cat(combined[key], dim=0)
    return combined


def bad_entries(output, args):
    bad = []
    for i, seq in enumerate(output["sequence_id"]):
        for s, sensor in enumerate(SENSOR_NAMES):
            reasons = []
            if to_float(output["orientation_residual_official_deg"][i, s]) > args.bad_orientation_deg:
                reasons.append("orientation_residual")
            if to_float(output["gyro_improvement_official"][i, s]) < args.bad_gyro_improvement:
                reasons.append("gyro_improvement")
            if to_float(output["window_orientation_residual_official_std_deg"][i, s]) > args.bad_window_std_deg:
                reasons.append("window_orientation_residual_std")
            if reasons:
                bad.append(
                    {
                        "sequence_id": seq,
                        "source_label": output["source_label"][i],
                        "sensor_id": s,
                        "sensor_name": sensor,
                        "reasons": reasons,
                        "orientation_residual_official_deg": to_float(output["orientation_residual_official_deg"][i, s]),
                        "gyro_improvement_official": to_float(output["gyro_improvement_official"][i, s]),
                        "official_vs_estimated_deg": to_float(output["official_vs_estimated_deg"][i, s]),
                    }
                )
    return bad


def sensor_summary(output):
    rows = []
    for s, sensor in enumerate(SENSOR_NAMES):
        rows.append(
            {
                "sensor_id": s,
                "sensor_name": sensor,
                "orientation_residual_official_deg": summarize_tensor(output["orientation_residual_official_deg"][:, s]),
                "orientation_residual_estimated_deg": summarize_tensor(output["orientation_residual_estimated_deg"][:, s]),
                "gyro_improvement_official": summarize_tensor(output["gyro_improvement_official"][:, s]),
                "gyro_improvement_estimated": summarize_tensor(output["gyro_improvement_estimated"][:, s]),
                "official_vs_estimated_deg": summarize_tensor(output["official_vs_estimated_deg"][:, s]),
                "window_estimated_RJS_consistency_deg": summarize_tensor(output["window_estimated_RJS_consistency_deg"][:, s]),
            }
        )
    return rows


def sequence_sensor_rows(output):
    rows = []
    for i, seq in enumerate(output["sequence_id"]):
        for s, sensor in enumerate(SENSOR_NAMES):
            rows.append(
                {
                    "sequence_id": seq,
                    "source_label": output["source_label"][i],
                    "sensor_id": s,
                    "sensor_name": sensor,
                    "orientation_residual_official_deg": to_float(output["orientation_residual_official_deg"][i, s]),
                    "orientation_residual_estimated_deg": to_float(output["orientation_residual_estimated_deg"][i, s]),
                    "gyro_residual_zero": to_float(output["gyro_residual_zero"][i, s]),
                    "gyro_residual_official": to_float(output["gyro_residual_official"][i, s]),
                    "gyro_residual_estimated": to_float(output["gyro_residual_estimated"][i, s]),
                    "gyro_improvement_official": to_float(output["gyro_improvement_official"][i, s]),
                    "gyro_improvement_estimated": to_float(output["gyro_improvement_estimated"][i, s]),
                    "official_vs_estimated_deg": to_float(output["official_vs_estimated_deg"][i, s]),
                    "window_orientation_residual_official_median_deg": to_float(output["window_orientation_residual_official_median_deg"][i, s]),
                    "window_orientation_residual_official_std_deg": to_float(output["window_orientation_residual_official_std_deg"][i, s]),
                    "window_estimated_RJS_consistency_deg": to_float(output["window_estimated_RJS_consistency_deg"][i, s]),
                    "gyro_dt_best_official": int(output["gyro_dt_best_official"][i, s].item()),
                }
            )
    return rows


def summarize(output, args):
    dt_best = output["gyro_dt_best_official"].long()
    bad = bad_entries(output, args)
    official_ori = output["orientation_residual_official_deg"]
    estimated_ori = output["orientation_residual_estimated_deg"]
    official_gyro = output["gyro_improvement_official"]
    estimated_gyro = output["gyro_improvement_estimated"]
    diff = output["official_vs_estimated_deg"]
    if tensor_median(official_ori) <= args.strong_orientation_deg and tensor_median(official_gyro) >= args.strong_gyro_improvement:
        decision = "OFFICIAL_SO3_STRONG"
    elif tensor_median(estimated_ori) + args.better_margin_deg < tensor_median(official_ori) or tensor_median(estimated_gyro) > tensor_median(official_gyro) + args.better_margin_gyro:
        decision = "ESTIMATED_RJS_BETTER"
    else:
        decision = "ROTATION_PREPROCESSING_SUSPICIOUS"
    return {
        "num_sequences": len(output["sequence_id"]),
        "num_sequence_sensor_entries": int(official_ori.numel()),
        "orientation_residual_official_deg": summarize_tensor(official_ori),
        "orientation_residual_estimated_deg": summarize_tensor(estimated_ori),
        "gyro_improvement_official": summarize_tensor(official_gyro),
        "gyro_improvement_estimated": summarize_tensor(estimated_gyro),
        "gyro_residual_zero": summarize_tensor(output["gyro_residual_zero"]),
        "gyro_residual_official": summarize_tensor(output["gyro_residual_official"]),
        "gyro_residual_estimated": summarize_tensor(output["gyro_residual_estimated"]),
        "official_vs_estimated_RJS_deg": summarize_tensor(diff),
        "window_orientation_residual_official_std_deg": summarize_tensor(output["window_orientation_residual_official_std_deg"]),
        "window_estimated_RJS_consistency_deg": summarize_tensor(output["window_estimated_RJS_consistency_deg"]),
        "gyro_dt0_best_fraction_official": float((dt_best == 0).float().mean().item()),
        "gyro_best_dt_distribution_official": {str(int(v)): int((dt_best == int(v)).sum().item()) for v in sorted(dt_best.unique().tolist())},
        "sensor_summary": sensor_summary(output),
        "bad_entries": bad,
        "decision": decision,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Audit GlobalPose official TotalCapture IMU SO(3) explainability.")
    parser.add_argument("--output-dir", default="data/dataset_work/SensorOffset/official_so3_explainability_v1")
    parser.add_argument("--official-train", default="data/dataset_work/TotalCapture_globalpose_official/train.pt")
    parser.add_argument("--official-val", default="data/dataset_work/TotalCapture_globalpose_official/val.pt")
    parser.add_argument("--estimated-cache", default="data/dataset_work/SensorOffset/rawlike_se3_candidate_a_v1/totalcapture_full_sequence_se3_cache.pt")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=180)
    parser.add_argument("--stride", type=int, default=90)
    parser.add_argument("--min-window-frames", type=int, default=90)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--smoothing-mode", default="moving_average", choices=("none", "moving_average", "centered_moving_average", "savgol"))
    parser.add_argument("--derivative-mode", default="centered", choices=("legacy", "centered", "strict_centered"))
    parser.add_argument("--dt-values", default="-3,-2,-1,0,1,2,3")
    parser.add_argument("--strong-orientation-deg", type=float, default=10.0)
    parser.add_argument("--strong-gyro-improvement", type=float, default=0.70)
    parser.add_argument("--better-margin-deg", type=float, default=2.0)
    parser.add_argument("--better-margin-gyro", type=float, default=0.05)
    parser.add_argument("--bad-orientation-deg", type=float, default=20.0)
    parser.add_argument("--bad-gyro-improvement", type=float, default=0.50)
    parser.add_argument("--bad-window-std-deg", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    estimated_cache = load_estimated_cache(args.estimated_cache)
    train = build_source_output(load_dataset_file(args.official_train), "official_train_source", args, estimated_cache)
    val = build_source_output(load_dataset_file(args.official_val), "official_val_source", args, estimated_cache)
    combined = combine_outputs([train, val])
    summary = summarize(combined, args)
    payload = {
        "config": vars(args),
        "chain_contract": {
            "RIS": "R_I_S: sensor frame to TotalCapture IMU-reference/world frame, from per-frame IMU quaternion.",
            "RIM": "R_I_M: model/world frame to TotalCapture IMU-reference/world frame after SMPL global-frame correction; RIM^T maps I to M.",
            "RSB": "R_S_B: body/bone frame to sensor frame after SMPL bone-frame correction; RSB^T maps sensor to body/bone.",
            "RMB": "R_M_B = RIM^T RIS RSB, body/bone frame to model/world frame, used by GlobalPose as official orientation input.",
            "official_sensor_to_body_so3": "R_B_S = RSB^T. In this joint-local audit it is compared to estimated R_JS because the mapped FK joint frame is the available body proxy.",
        },
        "summary": summary,
        "paths": {
            "pt": str(out_dir / "official_so3_explainability_v1.pt"),
            "summary_json": str(out_dir / "official_so3_explainability_summary.json"),
            "sequence_sensor_json": str(out_dir / "official_so3_sequence_sensor_report.json"),
        },
    }
    torch.save({**combined, "summary": summary, "config": vars(args)}, out_dir / "official_so3_explainability_v1.pt")
    write_json(out_dir / "official_so3_explainability_summary.json", payload)
    write_json(out_dir / "official_so3_sequence_sensor_report.json", sequence_sensor_rows(combined))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
