#!/usr/bin/env python3
"""Compare TotalCapture IMU acceleration against SMPL IMU-vertex acceleration."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l4_sensor_offset_utils import (  # noqa: E402
    FPS,
    GRAVITY_WORLD,
    IMU_VERTICES,
    SENSOR_NAMES,
    fk_imu_joints_and_vertices,
    official_imu_fields,
    savgol_smooth,
    sensor_to_joint_map,
)


SPLITS = {
    "train": ROOT / "data/dataset_work/TotalCapture_globalpose_official/train.pt",
    "val": ROOT / "data/dataset_work/TotalCapture_globalpose_official/val.pt",
    "test": ROOT / "data/dataset_work/TotalCapture_globalpose_official/test.pt",
}
METHODS = ("raw_fd", "savgol9_p3_fd", "savgol15_p3_fd")
SPACES = ("sensor_specific_force", "model_world_linear_acc")
AXES = ("x", "y", "z")


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, torch.Tensor):
        return to_builtin(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json(path, payload):
    path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n")


def finite_second_centered(pos, fps):
    acc = torch.full_like(pos, float("nan"))
    if pos.shape[0] >= 3:
        acc[1:-1] = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) * (float(fps) ** 2)
    return acc


def vertex_acc_versions(pos, fps):
    return {
        "raw_fd": finite_second_centered(pos, fps),
        "savgol9_p3_fd": finite_second_centered(savgol_smooth(pos, 9, 3), fps),
        "savgol15_p3_fd": finite_second_centered(savgol_smooth(pos, 15, 3), fps),
    }


def matvec(mat, vec):
    return mat.matmul(vec.unsqueeze(-1)).squeeze(-1)


def corr_1d(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if a.size == 0 or den < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def vector_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred).all(axis=-1) & np.isfinite(target).all(axis=-1)
    pred = pred[valid]
    target = target[valid]
    if pred.size == 0:
        return {}
    residual = pred - target
    l2 = np.linalg.norm(residual, axis=-1)
    mag_residual = np.linalg.norm(pred, axis=-1) - np.linalg.norm(target, axis=-1)
    out = {
        "num_frames": int(pred.shape[0]),
        "mean_l2_error": float(l2.mean()),
        "rmse": float(math.sqrt(np.mean(residual ** 2))),
        "pearson_correlation": corr_1d(pred, target),
        "cosine_similarity": cosine(pred, target),
        "magnitude_mae": float(np.mean(np.abs(mag_residual))),
        "residual_mean": float(l2.mean()),
        "residual_std": float(l2.std()),
        "residual_p95": float(np.quantile(l2, 0.95)),
    }
    for i, axis in enumerate(AXES):
        r = residual[:, i]
        out[f"{axis}_mae"] = float(np.mean(np.abs(r)))
        out[f"{axis}_rmse"] = float(math.sqrt(np.mean(r ** 2)))
        out[f"{axis}_bias"] = float(np.mean(r))
        out[f"{axis}_pearson_correlation"] = corr_1d(pred[:, i], target[:, i])
    return out


def aggregate_rows(rows, group_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, {"pred": [], "target": []})
        grouped[key]["pred"].append(row["_pred"])
        grouped[key]["target"].append(row["_target"])
    out = []
    for key, values in sorted(grouped.items()):
        pred = np.concatenate(values["pred"], axis=0)
        target = np.concatenate(values["target"], axis=0)
        item = {k: v for k, v in zip(group_keys, key)}
        item.update(vector_metrics(pred, target))
        out.append(item)
    return out


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key.startswith("_"):
                    continue
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_dataset(path):
    data = torch.load(path, map_location="cpu")
    missing = [k for k in ("name", "pose", "tran", "aS", "RIS", "RIM", "RSB", "wS") if k not in data]
    if missing:
        raise KeyError(f"{path} missing required TotalCapture fields: {missing}")
    return data


def infer_imu_contract(data):
    has_world = all(k in data for k in ("aM", "wM", "RMB"))
    return {
        "raw_field_aS": "TotalCapture raw accelerometer field in sensor frame; preprocessing stores specific force.",
        "sensor_like_formula": "acc_sensor_like = R_WS^T @ (acc_vertex_world - gravity_world), where R_WS = RIM^T @ RIS.",
        "official_model_world_formula": "aM = RIM^T @ RIS @ aS + gravity_world.",
        "official_model_world_available_in_file": has_world,
        "comparison_spaces": {
            "sensor_specific_force": "compare raw aS to sensor-frame specific force from vertex acceleration.",
            "model_world_linear_acc": "compare official/reconstructed aM to world/model-frame vertex linear acceleration.",
        },
    }


def process_sequence(data, idx, args):
    name = str(data["name"][idx])
    pose = data["pose"][idx].float()
    tran = data["tran"][idx].float()
    aS = data["aS"][idx].float()
    RIS = data["RIS"][idx].float()
    RIM = data["RIM"][idx].float()
    aM, _wM, _RMB = official_imu_fields(data, idx)
    n = min(pose.shape[0], tran.shape[0], aS.shape[0], RIS.shape[0], aM.shape[0])
    if args.max_frames:
        n = min(n, int(args.max_frames))
    pose, tran, aS, RIS, aM = pose[:n], tran[:n], aS[:n], RIS[:n], aM[:n]
    if n < 16:
        return [], None

    _, _R_wj, p_wv6 = fk_imu_joints_and_vertices(pose, tran, device=args.device)
    p_wv = p_wv6[:, :5]
    R_WS = RIM.transpose(1, 2).unsqueeze(0).matmul(RIS)[:, :5]
    acc_by_method = vertex_acc_versions(p_wv, FPS)
    gravity = GRAVITY_WORLD.view(1, 1, 3)
    trim = max(1, int(args.trim))
    frame_slice = slice(trim, n - trim)
    rows = []
    examples = {
        "name": name,
        "time": np.arange(n, dtype=np.float64) / FPS,
        "imu_sensor": aS[:, :5].numpy(),
        "imu_world": aM[:, :5].numpy(),
        "vertex_world": {},
        "vertex_sensor": {},
    }
    for method, acc_world_full in acc_by_method.items():
        acc_sensor_full = matvec(R_WS.transpose(-1, -2), acc_world_full - gravity)
        examples["vertex_world"][method] = acc_world_full.numpy()
        examples["vertex_sensor"][method] = acc_sensor_full.numpy()
        for space, pred_full, target_full in (
            ("sensor_specific_force", acc_sensor_full, aS[:, :5]),
            ("model_world_linear_acc", acc_world_full, aM[:, :5]),
        ):
            pred = pred_full[frame_slice].numpy()
            target = target_full[frame_slice].numpy()
            valid = np.isfinite(pred).all(axis=-1) & np.isfinite(target).all(axis=-1)
            for sensor_idx, sensor_name in enumerate(SENSOR_NAMES[:5]):
                p = pred[:, sensor_idx]
                t = target[:, sensor_idx]
                v = valid[:, sensor_idx]
                if not np.any(v):
                    continue
                row = {
                    "sequence_id": name,
                    "method": method,
                    "comparison_space": space,
                    "sensor_id": sensor_idx,
                    "sensor_name": sensor_name,
                    "vertex_id": int(IMU_VERTICES[sensor_idx]),
                }
                row.update(vector_metrics(p[v], t[v]))
                row["_pred"] = p[v]
                row["_target"] = t[v]
                rows.append(row)
    return rows, examples


def compact_rows(rows):
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def frame_level_rows(rows, trim):
    out = []
    for row in rows:
        pred = row["_pred"]
        target = row["_target"]
        residual = pred - target
        l2 = np.linalg.norm(residual, axis=-1)
        pred_mag = np.linalg.norm(pred, axis=-1)
        target_mag = np.linalg.norm(target, axis=-1)
        base = {
            "sequence_id": row["sequence_id"],
            "method": row["method"],
            "comparison_space": row["comparison_space"],
            "sensor_id": row["sensor_id"],
            "sensor_name": row["sensor_name"],
            "vertex_id": row["vertex_id"],
        }
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
                    "residual_x": float(residual[i, 0]),
                    "residual_y": float(residual[i, 1]),
                    "residual_z": float(residual[i, 2]),
                    "residual_l2": float(l2[i]),
                    "pred_magnitude": float(pred_mag[i]),
                    "imu_magnitude": float(target_mag[i]),
                    "magnitude_residual": float(pred_mag[i] - target_mag[i]),
                }
            )
    return out


def select_rows(rows, space):
    return [r for r in rows if r["comparison_space"] == space]


def method_labels():
    return {"raw_fd": "raw", "savgol9_p3_fd": "SavGol-9", "savgol15_p3_fd": "SavGol-15"}


def plot_error_bars(rows, out_dir, space):
    labels = method_labels()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    sensors = list(SENSOR_NAMES[:5])
    x = np.arange(len(sensors))
    width = 0.25
    for mi, method in enumerate(METHODS):
        rs = [r for r in rows if r["comparison_space"] == space and r["method"] == method]
        by_sensor = {r["sensor_name"]: r for r in aggregate_rows(rs, ["sensor_name"])}
        offset = (mi - 1) * width
        axes[0].bar(x + offset, [by_sensor[s]["rmse"] for s in sensors], width, label=labels[method])
        axes[1].bar(x + offset, [by_sensor[s]["mean_l2_error"] for s in sensors], width, label=labels[method])
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("mean L2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sensors, rotation=20, ha="right")
    axes[0].legend()
    fig.suptitle(f"Acceleration error by sensor ({space})")
    fig.tight_layout()
    fig.savefig(out_dir / "error_bar_rmse.png", dpi=160)
    plt.close(fig)


def plot_corr_bars(rows, out_dir, space):
    labels = method_labels()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    sensors = list(SENSOR_NAMES[:5])
    x = np.arange(len(sensors))
    width = 0.25
    for mi, method in enumerate(METHODS):
        rs = [r for r in rows if r["comparison_space"] == space and r["method"] == method]
        by_sensor = {r["sensor_name"]: r for r in aggregate_rows(rs, ["sensor_name"])}
        offset = (mi - 1) * width
        axes[0].bar(x + offset, [by_sensor[s]["pearson_correlation"] for s in sensors], width, label=labels[method])
        axes[1].bar(x + offset, [by_sensor[s]["cosine_similarity"] for s in sensors], width, label=labels[method])
    axes[0].set_ylabel("Pearson corr")
    axes[1].set_ylabel("cosine")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sensors, rotation=20, ha="right")
    axes[0].legend()
    fig.suptitle(f"Acceleration agreement by sensor ({space})")
    fig.tight_layout()
    fig.savefig(out_dir / "corr_bar.png", dpi=160)
    plt.close(fig)


def plot_timeseries(examples, out_dir, method, space, max_examples):
    for ex in examples[:max_examples]:
        target = ex["imu_sensor"] if space == "sensor_specific_force" else ex["imu_world"]
        pred = ex["vertex_sensor"][method] if space == "sensor_specific_force" else ex["vertex_world"][method]
        n = min(target.shape[0], int(12 * FPS))
        t = ex["time"][:n]
        fig, axes = plt.subplots(5, 3, figsize=(16, 11), sharex=True)
        for s, sensor in enumerate(SENSOR_NAMES[:5]):
            for a, axis in enumerate(AXES):
                ax = axes[s, a]
                ax.plot(t, target[:n, s, a], label="IMU", linewidth=1.0)
                ax.plot(t, pred[:n, s, a], label="vertex diff", linewidth=1.0, alpha=0.8)
                if s == 0:
                    ax.set_title(axis)
                if a == 0:
                    ax.set_ylabel(sensor)
        axes[0, 0].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"{ex['name']} {method} {space}")
        fig.tight_layout()
        safe = ex["name"].replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"timeseries_examples_{safe}_{method}_{space}.png", dpi=150)
        plt.close(fig)


def sample_pairs(rows, method, space, max_points=8000):
    pairs = []
    for r in rows:
        if r["method"] == method and r["comparison_space"] == space:
            pred = r["_pred"].reshape(-1, 3)
            target = r["_target"].reshape(-1, 3)
            pairs.append((pred, target, r["sensor_name"]))
    return pairs[:]


def plot_scatter(rows, out_dir, method, space):
    pairs = sample_pairs(rows, method, space)
    fig, axes = plt.subplots(5, 3, figsize=(15, 13), sharex=False, sharey=False)
    rng = np.random.default_rng(0)
    for s, sensor in enumerate(SENSOR_NAMES[:5]):
        sensor_pairs = [(p, t) for p, t, name in pairs if name == sensor]
        if not sensor_pairs:
            continue
        pred = np.concatenate([p for p, _t in sensor_pairs], axis=0)
        target = np.concatenate([t for _p, t in sensor_pairs], axis=0)
        idx = np.arange(pred.shape[0])
        if idx.size > 3000:
            idx = rng.choice(idx, size=3000, replace=False)
        for a, axis in enumerate(AXES):
            ax = axes[s, a]
            x = pred[idx, a]
            y = target[idx, a]
            ax.scatter(x, y, s=3, alpha=0.25)
            lo = float(np.nanmin([x.min(), y.min()]))
            hi = float(np.nanmax([x.max(), y.max()]))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
            if s == 0:
                ax.set_title(axis)
            if a == 0:
                ax.set_ylabel(sensor)
    fig.suptitle(f"IMU vs vertex diff scatter ({method}, {space})")
    fig.tight_layout()
    fig.savefig(out_dir / f"scatter_{method}_{space}.png", dpi=150)
    plt.close(fig)


def plot_residual_hist(rows, out_dir, method, space):
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    for s, sensor in enumerate(SENSOR_NAMES[:5]):
        residuals = []
        for r in rows:
            if r["method"] == method and r["comparison_space"] == space and r["sensor_name"] == sensor:
                residuals.append(np.linalg.norm(r["_pred"] - r["_target"], axis=-1))
        if residuals:
            axes[s].hist(np.concatenate(residuals), bins=80, alpha=0.85)
        axes[s].set_ylabel(sensor)
    axes[-1].set_xlabel("L2 residual")
    fig.suptitle(f"Residual distribution ({method}, {space})")
    fig.tight_layout()
    fig.savefig(out_dir / f"residual_hist_{method}_{space}.png", dpi=150)
    plt.close(fig)


def plot_boxplot(rows, out_dir, method, space):
    data = []
    labels = []
    for sensor in SENSOR_NAMES[:5]:
        residuals = []
        for r in rows:
            if r["method"] == method and r["comparison_space"] == space and r["sensor_name"] == sensor:
                residuals.append(np.linalg.norm(r["_pred"] - r["_target"], axis=-1))
        if residuals:
            data.append(np.concatenate(residuals))
            labels.append(sensor)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("L2 residual")
    ax.set_title(f"Residuals by sensor ({method}, {space})")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_residuals.png", dpi=160)
    plt.close(fig)


def write_summary_md(path, overall, per_sensor, per_sequence, config):
    preferred = config["preferred_comparison_space"]
    rows = [r for r in overall if r["comparison_space"] == preferred]
    best = min(rows, key=lambda r: r["rmse"])
    worst_sensor = max(
        [r for r in per_sensor if r["comparison_space"] == preferred and r["method"] == best["method"]],
        key=lambda r: r["rmse"],
    )
    best_sensor = min(
        [r for r in per_sensor if r["comparison_space"] == preferred and r["method"] == best["method"]],
        key=lambda r: r["rmse"],
    )
    seq_worst = max(
        [r for r in per_sequence if r["comparison_space"] == preferred and r["method"] == best["method"]],
        key=lambda r: r["rmse"],
    )
    method_lines = []
    for method in METHODS:
        row = next(r for r in rows if r["method"] == method)
        method_lines.append(
            f"| {method} | {row['mean_l2_error']:.4f} | {row['rmse']:.4f} | "
            f"{row['pearson_correlation']:.4f} | {row['cosine_similarity']:.4f} | {row['magnitude_mae']:.4f} |"
        )
    bias_sorted = sorted(
        [r for r in per_sensor if r["comparison_space"] == preferred and r["method"] == best["method"]],
        key=lambda r: abs(r["x_bias"]) + abs(r["y_bias"]) + abs(r["z_bias"]),
        reverse=True,
    )
    bias_note = bias_sorted[0]
    support = (
        "not supported as a direct supervision/explainability target without calibration"
        if best["rmse"] > 5.0 or best["pearson_correlation"] < 0.5
        else "partially supported, but should still be gated by per-sensor calibration checks"
    )
    lines = [
        "# TotalCapture IMU Acceleration vs Vertex Difference Acceleration",
        "",
        "## Setup",
        "",
        f"- Dataset path: `{config['dataset_path']}`",
        f"- Split: `{config['split']}`; sequences: {config['num_sequences']}",
        f"- FPS: {config['fps']}; gravity: {config['gravity_world']}",
        f"- Preferred comparison: `{preferred}`",
        f"- IMU acceleration contract: {config['imu_contract']['raw_field_aS']}",
        f"- Sensor-like formula: `{config['imu_contract']['sensor_like_formula']}`",
        f"- World/model formula: `{config['imu_contract']['official_model_world_formula']}`",
        "- Vertex position source: SMPL FK with `tran`, using `fk_imu_joints_and_vertices`; this is a full world trajectory, not root-relative positions.",
        "",
        "## Vertex IDs",
        "",
        "| sensor | body part | vertex id |",
        "|---|---:|---:|",
    ]
    for name, vertex in zip(SENSOR_NAMES[:5], IMU_VERTICES[:5]):
        lines.append(f"| {name} | {name} | {int(vertex)} |")
    lines.extend(
        [
            "",
            "## Overall Metrics",
            "",
            "| method | mean L2 error | RMSE | Pearson corr | cosine | magnitude MAE |",
            "|---|---:|---:|---:|---:|---:|",
            *method_lines,
            "",
            "## Required Answers",
            "",
            f"1. Average difference on TotalCapture: best preferred-space method is `{best['method']}` with mean L2 {best['mean_l2_error']:.4f} m/s^2 and RMSE {best['rmse']:.4f} m/s^2.",
            f"2. Closest method: `{best['method']}`. See `summary_overall.csv` for raw/SavGol-9/SavGol-15 in both comparison spaces.",
            f"3. With `{best['method']}`, largest sensor error is `{worst_sensor['sensor_name']}` vertex {worst_sensor['vertex_id']} RMSE {worst_sensor['rmse']:.4f}; smallest is `{best_sensor['sensor_name']}` vertex {best_sensor['vertex_id']} RMSE {best_sensor['rmse']:.4f}. Worst sequence is `{seq_worst['sequence_id']}` RMSE {seq_worst['rmse']:.4f}.",
            f"4. Error diagnosis: largest aggregate bias is on `{bias_note['sensor_name']}` with bias ({bias_note['x_bias']:.4f}, {bias_note['y_bias']:.4f}, {bias_note['z_bias']:.4f}); compare this with residual p95 {bias_note['residual_p95']:.4f}, magnitude MAE {bias_note['magnitude_mae']:.4f}, and correlations in the CSVs/plots to separate bias, scale/noise, and axis-specific failure.",
            f"5. Conclusion: this diagnostic says vertex finite-difference acceleration is {support}.",
            "6. If unsupported, most likely causes are coordinate/gravity convention mismatch, non-equivalence between SMPL vertex and real IMU mount, soft-tissue or strap motion, and finite-difference noise. The script explicitly tests both sensor-specific-force and world/model-frame comparisons so a large gap in both spaces is not just a missing gravity-addition issue.",
            "",
            "## Outputs",
            "",
            "- `summary_overall.csv`",
            "- `summary_per_sensor.csv`",
            "- `summary_per_sequence.csv`",
            "- `frame_level_metrics.csv`",
            "- `config.json`",
            "- `vertex_ids.json`",
            "- `error_bar_rmse.png`, `corr_bar.png`, `timeseries_examples_*.png`, `scatter_*.png`, `residual_hist_*.png`, `boxplot_residuals.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--trim", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeseries-method", choices=METHODS, default="savgol9_p3_fd")
    parser.add_argument("--plot-space", choices=SPACES, default="sensor_specific_force")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset_path or SPLITS[args.split]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (ROOT / "code/outputs" / f"totalcapture_imu_vs_vertex_diff_acc_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset_path)
    count = len(data["name"])
    if args.max_sequences:
        count = min(count, int(args.max_sequences))
    rows = []
    examples = []
    for idx in range(count):
        seq_rows, example = process_sequence(data, idx, args)
        rows.extend(seq_rows)
        if example is not None:
            examples.append(example)
        print(f"[{idx + 1}/{count}] {data['name'][idx]} rows={len(seq_rows)}", flush=True)

    if not rows:
        raise RuntimeError("No comparable rows were produced.")

    overall = aggregate_rows(rows, ["comparison_space", "method"])
    per_sensor = aggregate_rows(rows, ["comparison_space", "method", "sensor_id", "sensor_name", "vertex_id"])
    per_sequence = aggregate_rows(rows, ["comparison_space", "method", "sequence_id"])
    frame_rows = frame_level_rows(rows, args.trim)
    config = {
        "script": str(Path(__file__).resolve()),
        "dataset_path": str(dataset_path),
        "split": args.split,
        "output_dir": str(out_dir),
        "num_sequences": count,
        "sequence_names": [str(x) for x in data["name"][:count]],
        "fps": FPS,
        "dt": 1.0 / FPS,
        "trim": args.trim,
        "gravity_world": GRAVITY_WORLD.tolist(),
        "preferred_comparison_space": args.plot_space,
        "methods": list(METHODS),
        "imu_contract": infer_imu_contract(data),
        "source_fields": {
            key: [tuple(v.shape) if torch.is_tensor(v) else type(v).__name__ for v in data[key][: min(2, len(data[key]))]]
            for key in sorted(data)
            if isinstance(data[key], list)
        },
        "sensor_to_joint_map": sensor_to_joint_map(),
        "world_position_note": "FK vertex positions are generated with the sequence root translation `tran`; no root-relative subtraction is applied.",
    }
    vertex_ids = {
        "source": "l4_sensor_offset_utils.IMU_VERTICES",
        "all_imu_vertices": list(IMU_VERTICES),
        "compared_leaf_vertices": [
            {"sensor_id": i, "sensor_name": SENSOR_NAMES[i], "body_part": SENSOR_NAMES[i], "vertex_id": int(IMU_VERTICES[i])}
            for i in range(5)
        ],
        "pelvis_root_vertex_not_compared": {"sensor_name": SENSOR_NAMES[5], "vertex_id": int(IMU_VERTICES[5])},
    }

    write_csv(out_dir / "summary_overall.csv", overall)
    write_csv(out_dir / "summary_per_sensor.csv", per_sensor)
    write_csv(out_dir / "summary_per_sequence.csv", per_sequence)
    write_csv(out_dir / "frame_level_metrics.csv", frame_rows)
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "vertex_ids.json", vertex_ids)
    plot_error_bars(rows, out_dir, args.plot_space)
    plot_corr_bars(rows, out_dir, args.plot_space)
    plot_timeseries(examples, out_dir, args.timeseries_method, args.plot_space, max_examples=min(5, len(examples)))
    for method in METHODS:
        plot_scatter(rows, out_dir, method, args.plot_space)
        plot_residual_hist(rows, out_dir, method, args.plot_space)
    plot_boxplot(rows, out_dir, args.timeseries_method, args.plot_space)
    write_summary_md(out_dir / "SUMMARY.md", overall, per_sensor, per_sequence, config)
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "summary": str(out_dir / "SUMMARY.md")}, indent=2))


if __name__ == "__main__":
    main()
