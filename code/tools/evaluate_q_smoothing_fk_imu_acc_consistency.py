#!/usr/bin/env python3
"""Evaluate q-smoothing effects on TotalCapture FK/IMU acceleration consistency.

This diagnostic smooths only q(t) = [translation, axis-angle pose]. It then
recomputes FK sensor-site positions from the processed q trajectory and obtains
FK acceleration by strict centered second differences. Predicted FK acceleration
is never post-smoothed; only the observed IMU target is low-pass/MA smoothed.
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

from compare_totalcapture_imu_acc_vs_vertex_diff_acc import (  # noqa: E402
    aggregate_rows,
    matvec,
    vector_metrics,
    write_csv,
    write_json,
)
from l4_sensor_offset_utils import (  # noqa: E402
    FPS,
    GRAVITY_WORLD,
    IMU_JOINTS,
    SENSOR_NAMES,
    fk_imu_joints_and_vertices,
    first_derivative,
    moving_average,
    savgol_smooth,
    second_derivative,
    sensor_to_joint_map,
)


DATASET_PATH = ROOT / "data/dataset_work/TotalCapture_globalpose_official/test.pt"
DEFAULT_RJS = ROOT / "code/outputs/totalcapture_accfit_rjs_synthesis_20260704_171103/rjs_accfit_global.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "code/outputs"
Q_METHODS = (
    "raw_q",
    "savgol_q_w9_p3",
    "savgol_q_w15_p3",
    "bspline_q_knot10",
    "bspline_q_knot15",
    "bspline_q_knot21",
    "bspline_q_knot30",
)
TARGET_FILTERS = ("centered_ma21", "lowpass_5hz")
LOWER_LEG_IDS = (2, 3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    p.add_argument("--rjs-path", type=Path, default=DEFAULT_RJS)
    p.add_argument("--rjs-method", default="savgol9_p3_fd")
    p.add_argument("--rjs-field", default="r_JS_projected")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-sequences", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--trim", type=int, default=12)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_dataset(path):
    data = torch.load(path, map_location="cpu")
    missing = [k for k in ("name", "pose", "tran", "aS", "RIS", "RIM") if k not in data]
    if missing:
        raise KeyError(f"{path} missing required fields: {missing}")
    return data


def load_rjs(path, field, method):
    payload = torch.load(path, map_location="cpu")
    if field not in payload or method not in payload[field]:
        raise KeyError(f"{path} missing {field}[{method}]")
    rjs = payload[field][method].float()
    if rjs.shape != (6, 3):
        raise ValueError(f"Expected r_JS [6,3], got {tuple(rjs.shape)}")
    return rjs, payload


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return to_builtin(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json_lf(path, payload):
    path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n")


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
            y[bad] = torch.from_numpy(np.interp(idx[bad].numpy(), finite_idx.numpy(), finite_y.numpy())).to(y)
    return flat.reshape_as(x)


def lowpass_zero_phase(x, cutoff_hz=5.0):
    from scipy.signal import butter, sosfiltfilt

    x = fill_nonfinite(x)
    if x.shape[0] < 8:
        return x
    original_shape = x.shape
    flat = x.detach().cpu().reshape(x.shape[0], -1).numpy()
    sos = butter(2, float(cutoff_hz), btype="lowpass", fs=FPS, output="sos")
    y = sosfiltfilt(sos, flat, axis=0)
    return torch.from_numpy(y.copy()).reshape(original_shape).to(dtype=x.dtype)


def target_filter(aS, name):
    if name == "centered_ma21":
        return moving_average(fill_nonfinite(aS), 21)
    if name == "lowpass_5hz":
        return lowpass_zero_phase(aS, 5.0)
    raise ValueError(name)


def bspline_smooth_q(q, knot_count):
    try:
        from scipy.interpolate import make_interp_spline
    except ImportError as exc:
        raise RuntimeError("scipy.interpolate is required for B-spline q methods") from exc
    n = q.shape[0]
    if n < 2:
        return q.float(), {"requested_knot_count": int(knot_count), "used_knot_count": n, "spline_order": 0}
    used = max(2, min(int(knot_count), n))
    idx = np.unique(np.rint(np.linspace(0, n - 1, used)).astype(np.int64))
    if idx[0] != 0:
        idx = np.r_[0, idx]
    if idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    used = int(idx.size)
    order = min(3, used - 1)
    x = np.arange(n, dtype=np.float64)
    y = q.detach().cpu().float().reshape(n, -1).numpy()
    spline = make_interp_spline(idx.astype(np.float64), y[idx], k=order, axis=0)
    out = spline(x)
    return torch.from_numpy(out.copy()).reshape_as(q).to(dtype=q.dtype), {
        "requested_knot_count": int(knot_count),
        "used_knot_count": used,
        "spline_order": int(order),
    }


def process_q(tran, pose, method):
    pose3 = pose.float().reshape(pose.shape[0], 24, 3)
    q = torch.cat([tran.float(), pose3.reshape(pose.shape[0], -1)], dim=-1)
    meta = {"method": method, "postprocesses_fk_acceleration": False}
    if method == "raw_q":
        q_out = q
    elif method == "savgol_q_w9_p3":
        q_out = savgol_smooth(q, 9, 3)
    elif method == "savgol_q_w15_p3":
        q_out = savgol_smooth(q, 15, 3)
    elif method.startswith("bspline_q_knot"):
        q_out, spline_meta = bspline_smooth_q(q, int(method.rsplit("knot", 1)[1]))
        meta.update(spline_meta)
    else:
        raise ValueError(method)
    return q_out[:, :3].contiguous(), q_out[:, 3:].reshape_as(pose3).contiguous(), meta


def finite_slice(n, trim):
    trim = min(int(trim), max(1, (int(n) - 3) // 2))
    return slice(trim, int(n) - trim), trim


def prepare_sequence(data, idx, args):
    n = min(
        data["pose"][idx].shape[0],
        data["tran"][idx].shape[0],
        data["aS"][idx].shape[0],
        data["RIS"][idx].shape[0],
    )
    if args.max_frames:
        n = min(n, int(args.max_frames))
    if n < max(16, args.trim * 2 + 3):
        raise RuntimeError(f"{data['name'][idx]} too short after max_frames: {n}")
    pose = data["pose"][idx][:n].float()
    tran = data["tran"][idx][:n].float()
    aS = data["aS"][idx][:n].float()
    RIS = data["RIS"][idx][:n].float()
    RIM = data["RIM"][idx].float()
    if RIM.dim() == 3:
        RIM_t = RIM.unsqueeze(0)
    elif RIM.dim() == 4:
        RIM_t = RIM[:n]
    else:
        raise ValueError(f"Unsupported RIM shape for {data['name'][idx]}: {tuple(RIM.shape)}")
    R_WS_obs = RIM_t.transpose(-1, -2).matmul(RIS)
    return {
        "sequence_index": idx,
        "sequence_id": str(data["name"][idx]),
        "n": int(n),
        "time": np.arange(n, dtype=np.float64) / FPS,
        "pose": pose,
        "tran": tran,
        "aS": aS,
        "R_WS_obs": R_WS_obs.float(),
    }


def q_deviation(seq, tran_processed, pose_processed):
    pose_delta = (pose_processed.reshape(-1, 24, 3) - seq["pose"].reshape(-1, 24, 3)).norm(dim=-1) * (180.0 / math.pi)
    tran_delta = (tran_processed - seq["tran"]).norm(dim=-1)
    return {
        "mean_tran_delta_m": float(tran_delta.mean()),
        "max_tran_delta_m": float(tran_delta.max()),
        "mean_pose_delta_deg": float(pose_delta.mean()),
        "max_pose_delta_deg": float(pose_delta.max()),
        "q_deviation_acceptable": bool(float(tran_delta.mean()) <= 0.02 and float(pose_delta.mean()) <= 2.0),
    }


def prediction_for_method(seq, rjs, method, args):
    tran_p, pose_p, meta = process_q(seq["tran"], seq["pose"], method)
    _qd = first_derivative(torch.cat([tran_p, pose_p.reshape(pose_p.shape[0], -1)], dim=-1), fps=FPS, mode="centered")
    _qdd = second_derivative(torch.cat([tran_p, pose_p.reshape(pose_p.shape[0], -1)], dim=-1), fps=FPS, mode="centered")
    p_wj, R_wj, _p_wv = fk_imu_joints_and_vertices(pose_p, tran_p, device=args.device)
    p_ws = p_wj + matvec(R_wj, rjs.view(1, 6, 3))
    acc_world = second_derivative(p_ws, fps=FPS, mode="centered")
    acc_sensor = matvec(seq["R_WS_obs"].transpose(-1, -2), acc_world - GRAVITY_WORLD.view(1, 1, 3))
    return acc_sensor.float(), q_deviation(seq, tran_p, pose_p), meta


def sequence_rows(seq, rjs, args):
    sl, used_trim = finite_slice(seq["n"], args.trim)
    rows, q_rows, method_meta = [], [], []
    targets = {name: target_filter(seq["aS"], name) for name in TARGET_FILTERS}
    for method in Q_METHODS:
        pred, qdev, meta = prediction_for_method(seq, rjs, method, args)
        method_meta.append({"sequence_id": seq["sequence_id"], **meta})
        q_rows.append({"sequence_id": seq["sequence_id"], "method": method, **qdev})
        pred_np = pred[sl].numpy()
        valid_pred = np.isfinite(pred_np).all(axis=-1)
        for filt, target_full in targets.items():
            target_np = target_full[sl].numpy()
            valid = valid_pred & np.isfinite(target_np).all(axis=-1)
            for sid, sensor_name in enumerate(SENSOR_NAMES):
                v = valid[:, sid]
                if not np.any(v):
                    continue
                row = {
                    "method": method,
                    "target_filter": filt,
                    "sequence_id": seq["sequence_id"],
                    "sensor_id": sid,
                    "sensor_name": sensor_name,
                    "sensor_group": "lower_leg" if sid in LOWER_LEG_IDS else "other",
                    "mapped_joint_id": int(IMU_JOINTS[sid]),
                    "trim": used_trim,
                }
                row.update(vector_metrics(pred_np[:, sid][v], target_np[:, sid][v]))
                row["_pred"] = pred_np[:, sid][v]
                row["_target"] = target_np[:, sid][v]
                rows.append(row)
    return rows, q_rows, method_meta


def compact_rows(rows):
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def aggregate_lower_leg(rows):
    return aggregate_rows([r for r in rows if int(r["sensor_id"]) in LOWER_LEG_IDS], ["method", "target_filter"])


def best_method(overall):
    centered = [r for r in overall if r["target_filter"] == "centered_ma21"]
    return min(centered, key=lambda r: float(r["rmse"]))


def method_lookup(rows):
    return {(r["method"], r["target_filter"]): r for r in rows}


def qdev_lookup(q_rows):
    grouped = {}
    for row in q_rows:
        grouped.setdefault(row["method"], []).append(row)
    out = {}
    for method, rows in grouped.items():
        out[method] = {
            "method": method,
            "mean_tran_delta_m": float(np.mean([float(r["mean_tran_delta_m"]) for r in rows])),
            "max_tran_delta_m": float(np.max([float(r["max_tran_delta_m"]) for r in rows])),
            "mean_pose_delta_deg": float(np.mean([float(r["mean_pose_delta_deg"]) for r in rows])),
            "max_pose_delta_deg": float(np.max([float(r["max_pose_delta_deg"]) for r in rows])),
        }
        out[method]["q_deviation_acceptable"] = (
            out[method]["mean_tran_delta_m"] <= 0.02 and out[method]["mean_pose_delta_deg"] <= 2.0
        )
    return out


def gate_rows(overall, lower, q_rows):
    best = best_method(overall)
    overall_map = method_lookup(overall)
    lower_map = method_lookup(lower)
    qmap = qdev_lookup(q_rows)
    raw_overall = overall_map[("raw_q", "centered_ma21")]
    raw_lower = lower_map[("raw_q", "centered_ma21")]
    rows = []
    for method in Q_METHODS:
        row = overall_map[(method, "centered_ma21")]
        lower_row = lower_map[(method, "centered_ma21")]
        qdev = qmap[method]
        is_bspline = method.startswith("bspline")
        within_5 = float(row["rmse"]) <= float(best["rmse"]) * 1.05
        lower_not_worse_raw = float(lower_row["rmse"]) <= float(raw_lower["rmse"])
        can_refine = (is_bspline and (method == best["method"] or within_5) and lower_not_worse_raw and qdev["q_deviation_acceptable"])
        rows.append(
            {
                "method": method,
                "target_filter": "centered_ma21",
                "overall_rmse": float(row["rmse"]),
                "best_overall_rmse": float(best["rmse"]),
                "delta_rmse_vs_raw": float(row["rmse"]) - float(raw_overall["rmse"]),
                "rmse_ratio_vs_best": float(row["rmse"]) / max(float(best["rmse"]), 1e-12),
                "lower_leg_rmse": float(lower_row["rmse"]),
                "lower_leg_delta_vs_raw": float(lower_row["rmse"]) - float(raw_lower["rmse"]),
                **qdev,
                "bspline_refinement_supported": bool(can_refine),
            }
        )
    return rows


def plot_overall(overall, out_dir):
    for filt in TARGET_FILTERS:
        rows = [r for r in overall if r["target_filter"] == filt]
        values = [float(r["rmse"]) for r in rows]
        labels = [r["method"] for r in rows]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(np.arange(len(labels)), values)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title(f"q smoothing FK/IMU consistency ({filt})")
        fig.tight_layout()
        fig.savefig(out_dir / f"overall_rmse_{filt}.png", dpi=160)
        plt.close(fig)


def plot_qdev(q_summary, out_dir):
    labels = [r["method"] for r in q_summary]
    pose = [float(r["mean_pose_delta_deg"]) for r in q_summary]
    tran_cm = [float(r["mean_tran_delta_m"]) * 100.0 for r in q_summary]
    fig, ax1 = plt.subplots(figsize=(12, 4))
    x = np.arange(len(labels))
    ax1.bar(x - 0.18, pose, width=0.36, label="pose deg")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, tran_cm, width=0.36, color="tab:orange", label="tran cm")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("mean pose delta (deg)")
    ax2.set_ylabel("mean translation delta (cm)")
    ax1.set_title("q deviation from raw trajectory")
    fig.tight_layout()
    fig.savefig(out_dir / "q_deviation.png", dpi=160)
    plt.close(fig)


def write_summary_md(path, overall, lower, q_summary, gate, config):
    best = best_method(overall)
    lowpass_best = min([r for r in overall if r["target_filter"] == "lowpass_5hz"], key=lambda r: float(r["rmse"]))
    supported = [r["method"] for r in gate if r["bspline_refinement_supported"]]
    lines = [
        "# q Smoothing FK/IMU Acceleration Consistency",
        "",
        "Evaluation only: no pose optimization, no FK-acceleration post-smoothing, no model training, no dataset cache generation.",
        "",
        "## Coordinate Contract",
        "",
        "- `R_WJ` maps joint-local vectors into world coordinates.",
        "- `r_JS` is the sensor-site offset from mapped joint J to sensor S, expressed in the joint frame.",
        "- `p_WS = p_WJ + R_WJ @ r_JS`.",
        "- `a_pred_world = d2(p_WS)/dt2` uses strict centered finite differences.",
        "- `a_pred_sensor = R_WS_obs^T @ (a_pred_world - gravity_world)`.",
        "- Predicted FK acceleration is not smoothed. Only `aS` targets use `centered_ma21` and `lowpass_5hz`.",
        "",
        "## Main Result",
        "",
        "| target | best method | RMSE | mean L2 | corr | cosine | residual p95 |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| centered_ma21 | {best['method']} | {best['rmse']:.6f} | {best['mean_l2_error']:.6f} | {best['pearson_correlation']:.6f} | {best['cosine_similarity']:.6f} | {best['residual_p95']:.6f} |",
        f"| lowpass_5hz | {lowpass_best['method']} | {lowpass_best['rmse']:.6f} | {lowpass_best['mean_l2_error']:.6f} | {lowpass_best['pearson_correlation']:.6f} | {lowpass_best['cosine_similarity']:.6f} | {lowpass_best['residual_p95']:.6f} |",
        "",
        "## B-spline Refinement Gate",
        "",
        f"- Supported methods: `{', '.join(supported) if supported else 'none'}`.",
        "- Gate: B-spline is best or within 5% of best centered_ma21 RMSE, lower-leg RMSE is not worse than raw, and q deviation is acceptable.",
        "- q deviation acceptable means mean pose delta <= 2 deg and mean translation delta <= 2 cm.",
        "",
        "## Outputs",
        "",
        "- `summary.json`: machine-readable result summary.",
        "- `config.json`: paths, filters, methods, frame contract, and spline-order metadata.",
        "- `overall_summary.csv`: method-level metrics.",
        "- `per_sequence_summary.csv`: per-sequence metrics.",
        "- `per_sensor_summary.csv`: per-sensor metrics.",
        "- `lower_leg_summary.csv`: lower-leg-only method metrics.",
        "- `q_deviation_summary.csv`: q displacement from raw trajectory.",
        "- `refinement_gate.csv`: B-spline control-point-refinement gate.",
        "- `figures/*.png`: overall RMSE and q-deviation plots.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"q_smoothing_fk_imu_acc_consistency_{stamp}")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset_path)
    rjs, rjs_payload = load_rjs(args.rjs_path, args.rjs_field, args.rjs_method)
    count = len(data["name"])
    if args.max_sequences:
        count = min(count, int(args.max_sequences))

    rows, q_rows, method_meta = [], [], []
    for idx in range(count):
        seq = prepare_sequence(data, idx, args)
        print(f"[eval] {idx + 1}/{count} {seq['sequence_id']} frames={seq['n']}", flush=True)
        seq_rows, seq_q_rows, seq_method_meta = sequence_rows(seq, rjs, args)
        rows.extend(seq_rows)
        q_rows.extend(seq_q_rows)
        method_meta.extend(seq_method_meta)

    overall = aggregate_rows(rows, ["method", "target_filter"])
    per_sequence = aggregate_rows(rows, ["method", "target_filter", "sequence_id"])
    per_sensor = aggregate_rows(rows, ["method", "target_filter", "sensor_id", "sensor_name", "sensor_group"])
    lower = aggregate_lower_leg(rows)
    q_summary = list(qdev_lookup(q_rows).values())
    gate = gate_rows(overall, lower, q_rows)
    summary = {
        "best_method_centered_ma21": best_method(overall),
        "best_method_lowpass_5hz": min([r for r in overall if r["target_filter"] == "lowpass_5hz"], key=lambda r: float(r["rmse"])),
        "bspline_refinement_supported_methods": [r["method"] for r in gate if r["bspline_refinement_supported"]],
        "claim_boundary": "diagnostic only; q(t) smoothing was tested, predicted FK acceleration was not post-smoothed.",
    }

    config = {
        "args": vars(args),
        "dataset_path": args.dataset_path,
        "rjs_path": args.rjs_path,
        "rjs_field": args.rjs_field,
        "rjs_method": args.rjs_method,
        "fps": FPS,
        "gravity_world": GRAVITY_WORLD.tolist(),
        "q_methods": Q_METHODS,
        "target_filters": TARGET_FILTERS,
        "lower_leg_sensor_ids": LOWER_LEG_IDS,
        "sensor_to_joint_map": sensor_to_joint_map(),
        "rjs_source_config": rjs_payload.get("config", {}),
        "spline_metadata": method_meta,
        "contract": {
            "T_AB_convention": "R_WJ maps coordinates from joint frame J into world frame W.",
            "q": "q(t) = [tran_W, pose_axis_angle_24x3]. qd and qdd are derived from the same processed q(t).",
            "r_JS": "sensor origin S relative to mapped joint J, expressed in joint-local coordinates.",
            "p_WS": "p_WJ + R_WJ @ r_JS",
            "a_pred_world": "strict centered second difference of p_WS at 60 Hz",
            "a_pred_sensor": "R_WS_obs^T @ (a_pred_world - gravity_world)",
            "prediction_smoothing": "none",
            "target_smoothing": "centered_ma21(aS) and zero-phase lowpass_5hz(aS)",
        },
    }

    write_csv(out_dir / "overall_summary.csv", compact_rows(overall))
    write_csv(out_dir / "per_sequence_summary.csv", compact_rows(per_sequence))
    write_csv(out_dir / "per_sensor_summary.csv", compact_rows(per_sensor))
    write_csv(out_dir / "lower_leg_summary.csv", compact_rows(lower))
    write_csv(out_dir / "q_deviation_summary.csv", q_summary)
    write_csv(out_dir / "refinement_gate.csv", gate)
    write_json_lf(out_dir / "config.json", config)
    write_json_lf(out_dir / "summary.json", summary)
    plot_overall(overall, fig_dir)
    plot_qdev(q_summary, fig_dir)
    write_summary_md(out_dir / "summary.md", overall, lower, q_summary, gate, config)

    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "summary": str(out_dir / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
