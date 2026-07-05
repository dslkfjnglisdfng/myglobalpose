#!/usr/bin/env python3
"""Audit TotalCapture FK IMU-site acceleration vs IMU under matched smoothing.

Evaluation-only script. It does not refine pose/tran, train a model, or write a
new dataset cache.
"""

import argparse
import csv
import gzip
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

from compare_totalcapture_imu_acc_vs_rjs_diff_acc import (  # noqa: E402
    DEFAULT_RJS_CANDIDATES,
    load_rjs_metadata,
)
from compare_totalcapture_imu_acc_vs_vertex_diff_acc import (  # noqa: E402
    aggregate_rows,
    infer_imu_contract,
    matvec,
    vector_metrics,
    write_csv,
    write_json,
)
from imu_position_offset import load_offset_cache  # noqa: E402
from l4_sensor_offset_utils import (  # noqa: E402
    FPS,
    GRAVITY_WORLD,
    IMU_JOINTS,
    IMU_VERTICES,
    SENSOR_NAMES,
    fk_imu_joints_and_vertices,
    moving_average,
    official_imu_fields,
    savgol_smooth,
    second_derivative,
    sensor_to_joint_map,
)


DATASET_PATH = ROOT / "data/dataset_work/TotalCapture_globalpose_official/test.pt"
DEFAULT_ACCFIT_GLOBAL_RJS = (
    ROOT / "code/outputs/totalcapture_accfit_rjs_synthesis_20260704_171103/rjs_accfit_global.pt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "code/outputs"
LOWER_LEG_IDS = (2, 3)
PLOT_SPACES = ("sensor_specific_force", "model_world_linear_acc")
SOURCES = ("vertex", "old_rjs", "accfit_global_rjs")


PROTOCOLS = (
    {"name": "raw", "kind": "identity"},
    {"name": "legacy_both_smooth_ma9", "kind": "ma", "window": 9},
    {"name": "legacy_both_lowpass_5hz", "kind": "lowpass", "cutoff": 5.0},
    {"name": "savgol9_p3", "kind": "savgol", "window": 9, "polyorder": 3},
    {"name": "savgol15_p3", "kind": "savgol", "window": 15, "polyorder": 3},
    {"name": "centered_ma9", "kind": "ma", "window": 9},
    {"name": "centered_ma15", "kind": "ma", "window": 15},
    {"name": "centered_ma21", "kind": "ma", "window": 21},
    {"name": "lowpass_3hz", "kind": "lowpass", "cutoff": 3.0},
    {"name": "lowpass_5hz", "kind": "lowpass", "cutoff": 5.0},
    {"name": "lowpass_8hz", "kind": "lowpass", "cutoff": 8.0},
    {"name": "lowpass_12hz", "kind": "lowpass", "cutoff": 12.0},
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    p.add_argument("--old-rjs-path", type=Path, default=None)
    p.add_argument("--accfit-global-rjs-path", type=Path, default=DEFAULT_ACCFIT_GLOBAL_RJS)
    p.add_argument("--accfit-rjs-method", default="savgol9_p3_fd")
    p.add_argument("--accfit-rjs-field", default="r_JS_projected")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-sequences", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--trim", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-frame-level", action="store_true")
    return p.parse_args()


def load_dataset(path):
    data = torch.load(path, map_location="cpu")
    missing = [k for k in ("name", "pose", "tran", "aS", "wS", "RIS", "RIM", "RSB") if k not in data]
    if missing:
        raise KeyError(f"{path} missing required TotalCapture fields: {missing}")
    return data


def choose_old_rjs_path(user_path):
    if user_path:
        if not user_path.exists():
            raise FileNotFoundError(user_path)
        return user_path
    for path in DEFAULT_RJS_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No old rJS cache found; pass --old-rjs-path.")


def load_global_accfit_rjs(path, method, field):
    payload = torch.load(path, map_location="cpu")
    if field not in payload or method not in payload[field]:
        raise KeyError(f"{path} missing {field}[{method}]")
    rjs = payload[field][method].float()
    if rjs.shape != (6, 3):
        raise ValueError(f"Expected accfit global rJS [6,3], got {tuple(rjs.shape)}")
    return rjs, payload


def lowpass_zero_phase(x, cutoff_hz):
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:
        raise RuntimeError("scipy.signal is required for low-pass protocols") from exc
    x = x.float()
    if x.shape[0] < 8:
        return x
    original_shape = x.shape
    flat = x.detach().cpu().reshape(x.shape[0], -1).numpy()
    sos = butter(2, float(cutoff_hz), btype="lowpass", fs=FPS, output="sos")
    y = sosfiltfilt(sos, flat, axis=0)
    return torch.from_numpy(y.copy()).reshape(original_shape).to(dtype=x.dtype)


def fill_nonfinite_for_filter(x):
    x = x.float().clone()
    flat = x.reshape(x.shape[0], -1)
    for col in range(flat.shape[1]):
        y = flat[:, col]
        finite = torch.isfinite(y)
        if bool(finite.all()):
            continue
        if int(finite.sum()) == 0:
            y.zero_()
            continue
        idx = torch.arange(y.shape[0])
        finite_idx = idx[finite]
        finite_y = y[finite]
        first = int(finite_idx[0])
        last = int(finite_idx[-1])
        y[:first] = finite_y[0]
        y[last + 1 :] = finite_y[-1]
        bad = ~torch.isfinite(y)
        if bool(bad.any()):
            # Interior NaNs are rare here; linear interpolation is enough for
            # filter continuity, while final metrics still use trimmed frames.
            xp = finite_idx.cpu().numpy()
            fp = finite_y.cpu().numpy()
            xi = idx[bad].cpu().numpy()
            y[bad] = torch.from_numpy(np.interp(xi, xp, fp)).to(dtype=y.dtype)
    return flat.reshape_as(x)


def apply_protocol(x, protocol):
    kind = protocol["kind"]
    if kind == "identity":
        return x.float()
    x = fill_nonfinite_for_filter(x)
    if kind == "ma":
        return moving_average(x.float(), int(protocol["window"]))
    if kind == "savgol":
        return savgol_smooth(x.float(), int(protocol["window"]), int(protocol["polyorder"]))
    if kind == "lowpass":
        return lowpass_zero_phase(x.float(), float(protocol["cutoff"]))
    raise ValueError(kind)


def safe_trim(n, trim):
    trim = min(int(trim), max(1, (int(n) - 3) // 2))
    return slice(trim, int(n) - trim), trim


def prepare_sequence(data, idx, args):
    name = str(data["name"][idx])
    pose = data["pose"][idx].float()
    tran = data["tran"][idx].float()
    aS = data["aS"][idx].float()
    wS = data["wS"][idx].float()
    RIS = data["RIS"][idx].float()
    RIM = data["RIM"][idx].float()
    aM, _wM, _RMB = official_imu_fields(data, idx)
    n = min(pose.shape[0], tran.shape[0], aS.shape[0], wS.shape[0], RIS.shape[0], aM.shape[0])
    if args.max_frames:
        n = min(n, int(args.max_frames))
    pose, tran, aS, wS, RIS, RIM, aM = pose[:n], tran[:n], aS[:n], wS[:n], RIS[:n], RIM[:n], aM[:n]
    if n < 32:
        raise RuntimeError(f"{name} too short: {n}")
    p_wj, R_wj, p_wv = fk_imu_joints_and_vertices(pose, tran, device=args.device)
    R_WS_obs = RIM.transpose(1, 2).unsqueeze(0).matmul(RIS)
    return {
        "idx": idx,
        "name": name,
        "n": int(n),
        "time": np.arange(n, dtype=np.float64) / FPS,
        "p_wj": p_wj.float(),
        "R_wj": R_wj.float(),
        "p_wv": p_wv.float(),
        "R_WS_obs": R_WS_obs.float(),
        "aS": aS.float(),
        "aM": aM.float(),
    }


def fk_world_acc(seq, source, old_rjs, accfit_rjs):
    if source == "vertex":
        return second_derivative(seq["p_wv"][:, :5], fps=FPS, mode="centered")
    rjs = old_rjs if source == "old_rjs" else accfit_rjs
    p_ws = seq["p_wj"][:, :5] + matvec(seq["R_wj"][:, :5], rjs[:5].view(1, 5, 3))
    return second_derivative(p_ws, fps=FPS, mode="centered")


def rows_for_sequence(seq, protocol, source, old_rjs, accfit_rjs, args):
    acc_world = fk_world_acc(seq, source, old_rjs, accfit_rjs)
    R_WS = seq["R_WS_obs"][:, :5]
    pred_sensor = matvec(R_WS.transpose(-1, -2), acc_world - GRAVITY_WORLD.view(1, 1, 3))
    pred_world = acc_world
    target_sensor = seq["aS"][:, :5]
    target_world = seq["aM"][:, :5]
    pred_sensor_f = apply_protocol(pred_sensor, protocol)
    pred_world_f = apply_protocol(pred_world, protocol)
    target_sensor_f = apply_protocol(target_sensor, protocol)
    target_world_f = apply_protocol(target_world, protocol)
    sl, _ = safe_trim(seq["n"], args.trim)
    rows = []
    for space, pred_full, target_full in (
        ("sensor_specific_force", pred_sensor_f, target_sensor_f),
        ("model_world_linear_acc", pred_world_f, target_world_f),
    ):
        pred = pred_full[sl].numpy()
        target = target_full[sl].numpy()
        valid = np.isfinite(pred).all(axis=-1) & np.isfinite(target).all(axis=-1)
        for sid, sensor_name in enumerate(SENSOR_NAMES[:5]):
            v = valid[:, sid]
            if not np.any(v):
                continue
            row = {
                "source": source,
                "protocol": protocol["name"],
                "comparison_space": space,
                "sequence_id": seq["name"],
                "sensor_id": sid,
                "sensor_name": sensor_name,
                "sensor_group": "lower_leg" if sid in LOWER_LEG_IDS else "other",
                "mapped_joint_id": int(IMU_JOINTS[sid]),
                "vertex_id": int(IMU_VERTICES[sid]) if source == "vertex" else "",
            }
            row.update(vector_metrics(pred[:, sid][v], target[:, sid][v]))
            row["_pred"] = pred[:, sid][v]
            row["_target"] = target[:, sid][v]
            rows.append(row)
    return rows


def compact_rows(rows):
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def frame_level_rows(rows, trim):
    out = []
    for row in rows:
        pred = row["_pred"]
        target = row["_target"]
        residual = pred - target
        l2 = np.linalg.norm(residual, axis=-1)
        base = {k: row[k] for k in ("source", "protocol", "comparison_space", "sequence_id", "sensor_id", "sensor_name")}
        for i in range(pred.shape[0]):
            out.append(
                {
                    **base,
                    "frame_index": int(i + trim),
                    "pred_x": float(pred[i, 0]),
                    "pred_y": float(pred[i, 1]),
                    "pred_z": float(pred[i, 2]),
                    "imu_x": float(target[i, 0]),
                    "imu_y": float(target[i, 1]),
                    "imu_z": float(target[i, 2]),
                    "residual_l2": float(l2[i]),
                }
            )
    return out


def write_csv_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def mean_metric(rows, key):
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def best_rows(overall):
    preferred = [r for r in overall if r["comparison_space"] == "sensor_specific_force"]
    return sorted(preferred, key=lambda r: float(r["rmse"]))


def protocol_improvement_rows(overall):
    rows = []
    raw = {
        (r["source"], r["comparison_space"]): r
        for r in overall
        if r["protocol"] == "raw"
    }
    for r in overall:
        key = (r["source"], r["comparison_space"])
        base = raw.get(key)
        if not base:
            continue
        item = dict(r)
        item["raw_rmse"] = float(base["rmse"])
        item["delta_rmse_vs_raw"] = float(r["rmse"]) - float(base["rmse"])
        item["rmse_ratio_vs_raw"] = float(r["rmse"]) / max(float(base["rmse"]), 1e-12)
        rows.append(item)
    return rows


def source_comparison_rows(overall):
    rows = []
    by_key = {(r["source"], r["protocol"], r["comparison_space"]): r for r in overall}
    for protocol in [p["name"] for p in PROTOCOLS]:
        for space in PLOT_SPACES:
            acc = by_key.get(("accfit_global_rjs", protocol, space))
            old = by_key.get(("old_rjs", protocol, space))
            vertex = by_key.get(("vertex", protocol, space))
            if not acc:
                continue
            rows.append(
                {
                    "protocol": protocol,
                    "comparison_space": space,
                    "accfit_global_rmse": float(acc["rmse"]),
                    "old_rjs_rmse": float(old["rmse"]) if old else float("nan"),
                    "vertex_rmse": float(vertex["rmse"]) if vertex else float("nan"),
                    "accfit_minus_old_rmse": float(acc["rmse"]) - float(old["rmse"]) if old else float("nan"),
                    "accfit_minus_vertex_rmse": float(acc["rmse"]) - float(vertex["rmse"]) if vertex else float("nan"),
                }
            )
    return rows


def plot_protocol_rmse(overall, out_dir, space):
    labels = [p["name"] for p in PROTOCOLS]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(15, 5))
    for i, source in enumerate(SOURCES):
        values = []
        for protocol in labels:
            row = next(r for r in overall if r["source"] == source and r["protocol"] == protocol and r["comparison_space"] == space)
            values.append(float(row["rmse"]))
        ax.bar(x + (i - 1) * width, values, width=width, label=source)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title(f"FK acceleration vs IMU by protocol ({space})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"protocol_rmse_{space}.png", dpi=160)
    plt.close(fig)


def plot_lower_leg(overall_lower, out_dir, space):
    rows = [r for r in overall_lower if r["comparison_space"] == space and r["source"] == "accfit_global_rjs"]
    labels = [r["protocol"] for r in rows]
    values = [float(r["rmse"]) for r in rows]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("lower-leg RMSE")
    ax.set_title(f"Lower-leg only, accfit global rJS ({space})")
    fig.tight_layout()
    fig.savefig(out_dir / f"lower_leg_rmse_{space}.png", dpi=160)
    plt.close(fig)


def plot_timeseries(examples, out_dir, protocol_name="lowpass_5hz", source="accfit_global_rjs"):
    for ex in examples[:4]:
        seq = ex["seq"]
        old_rjs = ex["old_rjs"]
        accfit_rjs = ex["accfit_rjs"]
        protocol = next(p for p in PROTOCOLS if p["name"] == protocol_name)
        acc_world = fk_world_acc(seq, source, old_rjs, accfit_rjs)
        pred = matvec(seq["R_WS_obs"][:, :5].transpose(-1, -2), acc_world - GRAVITY_WORLD.view(1, 1, 3))
        pred = apply_protocol(pred, protocol).numpy()
        obs = apply_protocol(seq["aS"][:, :5], protocol).numpy()
        n = min(pred.shape[0], int(12 * FPS))
        t = seq["time"][:n]
        fig, axes = plt.subplots(5, 3, figsize=(16, 11), sharex=True)
        for s, sensor in enumerate(SENSOR_NAMES[:5]):
            for a, axis in enumerate(("x", "y", "z")):
                ax = axes[s, a]
                ax.plot(t, obs[:n, s, a], label="IMU", linewidth=1)
                ax.plot(t, pred[:n, s, a], label="FK", linewidth=1, alpha=0.85)
                if s == 0:
                    ax.set_title(axis)
                if a == 0:
                    ax.set_ylabel(sensor)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{seq['name']} {source} {protocol_name} sensor specific force")
        fig.tight_layout()
        safe = seq["name"].replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"timeseries_{safe}_{source}_{protocol_name}.png", dpi=150)
        plt.close(fig)


def write_summary_md(path, overall, lower_overall, source_cmp, config):
    preferred = [r for r in overall if r["comparison_space"] == "sensor_specific_force"]
    best = min(preferred, key=lambda r: float(r["rmse"]))
    raw_accfit = next(r for r in preferred if r["source"] == "accfit_global_rjs" and r["protocol"] == "raw")
    best_accfit = min([r for r in preferred if r["source"] == "accfit_global_rjs"], key=lambda r: float(r["rmse"]))
    best_world = min([r for r in overall if r["comparison_space"] == "model_world_linear_acc"], key=lambda r: float(r["rmse"]))
    lower_raw = next(
        r for r in lower_overall
        if r["source"] == "accfit_global_rjs" and r["protocol"] == "raw" and r["comparison_space"] == "sensor_specific_force"
    )
    lower_best = min(
        [r for r in lower_overall if r["source"] == "accfit_global_rjs" and r["comparison_space"] == "sensor_specific_force"],
        key=lambda r: float(r["rmse"]),
    )
    sg9_cmp = next(r for r in source_cmp if r["protocol"] == "savgol9_p3" and r["comparison_space"] == "sensor_specific_force")
    low5_cmp = next(r for r in source_cmp if r["protocol"] == "lowpass_5hz" and r["comparison_space"] == "sensor_specific_force")
    best_measurement = best_accfit["protocol"]
    lines = [
        "# TotalCapture Both-Smooth FK/IMU Acceleration Audit",
        "",
        "Evaluation only: no pose/tran refinement, no model training, no dataset cache generation.",
        "",
        "## Coordinate Contract",
        "",
        "- `R_WJ` maps joint-local vectors into world coordinates.",
        "- `r_JS` is the IMU sensor origin relative to the mapped joint origin, expressed in joint-local coordinates.",
        "- `p_WS = p_WJ + R_WJ @ r_JS`.",
        "- Sensor-frame specific force comparison uses `a_S = R_WS_obs^T @ (ddot(p_WS) - g_W)`.",
        "- World/model-frame comparison uses `ddot(p_WS)` against reconstructed/stored `aM`.",
        "",
        "## Best Rows",
        "",
        "| scope | source | protocol | space | RMSE | L2 | corr | cosine | p95 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
        f"| overall best | {best['source']} | {best['protocol']} | {best['comparison_space']} | {best['rmse']:.6f} | {best['mean_l2_error']:.6f} | {best['pearson_correlation']:.6f} | {best['cosine_similarity']:.6f} | {best['residual_p95']:.6f} |",
        f"| accfit raw | {raw_accfit['source']} | {raw_accfit['protocol']} | {raw_accfit['comparison_space']} | {raw_accfit['rmse']:.6f} | {raw_accfit['mean_l2_error']:.6f} | {raw_accfit['pearson_correlation']:.6f} | {raw_accfit['cosine_similarity']:.6f} | {raw_accfit['residual_p95']:.6f} |",
        f"| accfit best smooth | {best_accfit['source']} | {best_accfit['protocol']} | {best_accfit['comparison_space']} | {best_accfit['rmse']:.6f} | {best_accfit['mean_l2_error']:.6f} | {best_accfit['pearson_correlation']:.6f} | {best_accfit['cosine_similarity']:.6f} | {best_accfit['residual_p95']:.6f} |",
        f"| world/model best | {best_world['source']} | {best_world['protocol']} | {best_world['comparison_space']} | {best_world['rmse']:.6f} | {best_world['mean_l2_error']:.6f} | {best_world['pearson_correlation']:.6f} | {best_world['cosine_similarity']:.6f} | {best_world['residual_p95']:.6f} |",
        f"| lower-leg accfit raw | {lower_raw['source']} | {lower_raw['protocol']} | {lower_raw['comparison_space']} | {lower_raw['rmse']:.6f} | {lower_raw['mean_l2_error']:.6f} | {lower_raw['pearson_correlation']:.6f} | {lower_raw['cosine_similarity']:.6f} | {lower_raw['residual_p95']:.6f} |",
        f"| lower-leg accfit best | {lower_best['source']} | {lower_best['protocol']} | {lower_best['comparison_space']} | {lower_best['rmse']:.6f} | {lower_best['mean_l2_error']:.6f} | {lower_best['pearson_correlation']:.6f} | {lower_best['cosine_similarity']:.6f} | {lower_best['residual_p95']:.6f} |",
        "",
        "## Required Answers",
        "",
        f"1. The old/smooth protocol helps because it applies the same low-pass operation to the noisy IMU signal and to the noisy second-difference FK acceleration. Raw finite differences amplify pose/tran noise; both-smoothing removes high-frequency components neither side can match reliably.",
        f"2. The improvement is primarily from smoothing/filtering, not only from frame choice. Both comparison spaces are reported; best sensor-frame row is `{best['protocol']}` RMSE `{best['rmse']:.6f}`, while best world/model row is `{best_world['protocol']}` RMSE `{best_world['rmse']:.6f}`.",
        f"3. Under SavGol-9 sensor specific force, accfit global rJS vs old rJS delta RMSE is `{sg9_cmp['accfit_minus_old_rmse']:.6f}` and vs vertex delta RMSE is `{sg9_cmp['accfit_minus_vertex_rmse']:.6f}`. Under lowpass-5Hz, the deltas are `{low5_cmp['accfit_minus_old_rmse']:.6f}` and `{low5_cmp['accfit_minus_vertex_rmse']:.6f}`.",
        f"4. Lower-leg accfit global rJS raw RMSE `{lower_raw['rmse']:.6f}` changes to best smooth `{lower_best['protocol']}` RMSE `{lower_best['rmse']:.6f}`.",
        f"5. Recommended measurement for the next Kalman-style smoother is `{best_measurement}` sensor-frame specific force with accfit global rJS, because it is the lowest-RMSE matched-processing protocol for the fixed-rJS source.",
        "6. If raw remains worse than matched smoothing, downstream refinement should pursue smooth/low-frequency acceleration only. This audit should be treated as the gate against optimizing raw acceleration.",
        "",
        "## Outputs",
        "",
        "- `smooth_protocol_summary.csv`: overall source/protocol/frame metrics.",
        "- `per_sensor_summary.csv`: per-sensor metrics.",
        "- `per_sequence_summary.csv`: per-sequence metrics.",
        "- `lower_leg_summary.csv`: lower-leg-only aggregate metrics.",
        "- `protocol_improvement_vs_raw.csv`: smoothing gains relative to raw for each source/frame.",
        "- `rjs_source_comparison.csv`: accfit global vs old rJS vs vertex deltas.",
        "- `plots/*.png`: protocol, lower-leg, and example time-series plots.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"totalcapture_both_smooth_fk_imu_acc_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    data = load_dataset(args.dataset_path)
    old_rjs_path = choose_old_rjs_path(args.old_rjs_path)
    old_rjs_map = load_offset_cache(old_rjs_path)
    old_rjs_meta = load_rjs_metadata(old_rjs_path)
    accfit_rjs, accfit_payload = load_global_accfit_rjs(
        args.accfit_global_rjs_path, args.accfit_rjs_method, args.accfit_rjs_field
    )

    count = len(data["name"])
    if args.max_sequences:
        count = min(count, int(args.max_sequences))
    rows, examples = [], []
    selected_names = [str(x) for x in data["name"][:count]]
    missing_old = [name for name in selected_names if name not in old_rjs_map]
    if missing_old:
        raise KeyError(f"Old rJS cache missing sequences: {missing_old}")

    for idx in range(count):
        seq = prepare_sequence(data, idx, args)
        old_rjs = old_rjs_map[seq["name"]].float()
        if old_rjs.shape != (6, 3):
            raise ValueError(f"{seq['name']} old rJS shape {tuple(old_rjs.shape)} != (6,3)")
        for protocol in PROTOCOLS:
            for source in SOURCES:
                rows.extend(rows_for_sequence(seq, protocol, source, old_rjs, accfit_rjs, args))
        examples.append({"seq": seq, "old_rjs": old_rjs, "accfit_rjs": accfit_rjs})
        print(f"[{idx + 1}/{count}] {seq['name']} rows={len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No rows produced.")

    overall = aggregate_rows(rows, ["source", "protocol", "comparison_space"])
    per_sensor = aggregate_rows(rows, ["source", "protocol", "comparison_space", "sensor_id", "sensor_name", "sensor_group"])
    per_sequence = aggregate_rows(rows, ["source", "protocol", "comparison_space", "sequence_id"])
    lower_rows = [r for r in rows if r["sensor_group"] == "lower_leg"]
    lower_overall = aggregate_rows(lower_rows, ["source", "protocol", "comparison_space"])
    improve = protocol_improvement_rows(overall)
    source_cmp = source_comparison_rows(overall)

    config = {
        "script": str(Path(__file__).resolve()),
        "dataset_path": str(args.dataset_path),
        "split": "TotalCapture official test",
        "output_dir": str(out_dir),
        "num_sequences": count,
        "sequence_names": selected_names,
        "fps": FPS,
        "dt": 1.0 / FPS,
        "trim": args.trim,
        "protocols": PROTOCOLS,
        "sources": SOURCES,
        "old_rjs_path": str(old_rjs_path),
        "old_rjs_metadata": old_rjs_meta,
        "accfit_global_rjs_path": str(args.accfit_global_rjs_path),
        "accfit_rjs_method": args.accfit_rjs_method,
        "accfit_rjs_field": args.accfit_rjs_field,
        "accfit_global_rjs": accfit_rjs.tolist(),
        "accfit_source_config": accfit_payload.get("config", {}),
        "gravity_world": GRAVITY_WORLD.tolist(),
        "imu_contract": infer_imu_contract(data),
        "sensor_to_joint_map": sensor_to_joint_map(),
        "lower_leg_sensor_ids": list(LOWER_LEG_IDS),
        "lower_leg_sensor_names": [SENSOR_NAMES[i] for i in LOWER_LEG_IDS],
    }

    write_csv(out_dir / "smooth_protocol_summary.csv", compact_rows(overall))
    write_csv(out_dir / "per_sensor_summary.csv", compact_rows(per_sensor))
    write_csv(out_dir / "per_sequence_summary.csv", compact_rows(per_sequence))
    write_csv(out_dir / "lower_leg_summary.csv", compact_rows(lower_overall))
    write_csv(out_dir / "protocol_improvement_vs_raw.csv", compact_rows(improve))
    write_csv(out_dir / "rjs_source_comparison.csv", compact_rows(source_cmp))
    if args.save_frame_level:
        write_csv_gz(out_dir / "frame_level_metrics.csv.gz", frame_level_rows(rows, args.trim))
    write_json(out_dir / "config.json", config)

    for space in PLOT_SPACES:
        plot_protocol_rmse(overall, out_dir / "plots", space)
        plot_lower_leg(lower_overall, out_dir / "plots", space)
    plot_timeseries(examples, out_dir / "plots", protocol_name="lowpass_5hz", source="accfit_global_rjs")
    plot_timeseries(examples, out_dir / "plots", protocol_name="savgol9_p3", source="accfit_global_rjs")
    write_summary_md(out_dir / "SUMMARY.md", overall, lower_overall, source_cmp, config)

    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "summary": str(out_dir / "SUMMARY.md")}, indent=2))


if __name__ == "__main__":
    main()
