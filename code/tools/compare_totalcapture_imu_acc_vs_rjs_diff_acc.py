#!/usr/bin/env python3
"""Compare TotalCapture IMU acceleration against r_JS IMU-site acceleration."""

import argparse
import csv
import json
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
    AXES,
    METHODS,
    SPACES,
    aggregate_rows,
    finite_second_centered,
    frame_level_rows,
    infer_imu_contract,
    matvec,
    plot_boxplot,
    plot_corr_bars,
    plot_error_bars,
    plot_residual_hist,
    plot_scatter,
    plot_timeseries,
    vector_metrics,
    vertex_acc_versions,
    write_csv,
    write_json,
)
from imu_position_offset import load_offset_cache  # noqa: E402
from l4_sensor_offset_utils import (  # noqa: E402
    FPS,
    GRAVITY_WORLD,
    IMU_JOINTS,
    SENSOR_NAMES,
    fk_imu_joints_and_vertices,
    official_imu_fields,
    sensor_to_joint_map,
)


DATASET_PATH = ROOT / "data/dataset_work/TotalCapture_globalpose_official/test.pt"
DEFAULT_RJS_CANDIDATES = (
    ROOT / "data/experiments/footlock_transpose_rjs_smoothacc_20260609/totalcapture_test_footlock_transpose_rjs.pt",
    ROOT / "data/dataset_work/TotalCapture_globalpose_official/test_with_offsets.pt",
)
DEFAULT_VERTEX_BASELINE = ROOT / "code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409"


def load_dataset(path):
    data = torch.load(path, map_location="cpu")
    missing = [k for k in ("name", "pose", "tran", "aS", "RIS", "RIM", "RSB", "wS") if k not in data]
    if missing:
        raise KeyError(f"{path} missing required TotalCapture fields: {missing}")
    return data


def choose_rjs_path(user_path):
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path, "user_argument"
    for path in DEFAULT_RJS_CANDIDATES:
        if path.exists():
            return path, "default_auto_search"
    raise FileNotFoundError("No default rJS cache found; pass --rjs-path.")


def tensor_or_list(value):
    if torch.is_tensor(value):
        return value.float()
    if isinstance(value, list) and value and torch.is_tensor(value[0]):
        return torch.stack([v.float() for v in value])
    return None


def load_rjs_metadata(path):
    cache = torch.load(path, map_location="cpu")
    names = [str(x) for x in (cache.get("name") or cache.get("sequence_id"))]
    rows = {str(r.get("name", r.get("sequence_id", ""))): r for r in cache.get("rows", []) if isinstance(r, dict)}
    offset_tensor = None
    offset_field = None
    for key in ("offset", "r_JS", "imu_offset_r"):
        if key in cache:
            offset_tensor = tensor_or_list(cache[key])
            offset_field = key
            break
    if offset_tensor is None:
        raise KeyError(f"{path} has no offset/r_JS/imu_offset_r tensor/list.")
    return {
        "path": str(path),
        "names": names,
        "offset_field": offset_field,
        "offset_tensor": offset_tensor,
        "rows": rows,
        "summary": cache.get("summary", {}),
        "config": cache.get("config", {}),
        "method": cache.get("method", ""),
        "source_path": cache.get("source_path", ""),
        "coordinate_contract": cache.get("coordinate_contract", ""),
    }


def row_list_value(row, key, sensor_idx, default=None):
    value = row.get(key, default)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)) and sensor_idx < len(value):
        return value[sensor_idx]
    return default


def make_rjs_offsets_json(dataset, rjs_map, meta, selected_names):
    per_sequence = []
    sequence_specific = len(set(selected_names)) == len(selected_names)
    for name in selected_names:
        if name not in rjs_map:
            raise KeyError(f"rJS cache missing sequence {name}")
        offsets = rjs_map[name].float()
        if offsets.shape != (6, 3):
            raise ValueError(f"{name} rJS shape={tuple(offsets.shape)}, expected [6,3]")
        row = meta["rows"].get(name, {})
        sensors = []
        for s, sensor_name in enumerate(SENSOR_NAMES):
            r = offsets[s]
            sensors.append(
                {
                    "sensor_id": s,
                    "sensor_name": sensor_name,
                    "mapped_joint_id": int(IMU_JOINTS[s]),
                    "r_JS": [float(x) for x in r.tolist()],
                    "r_JS_norm": float(r.norm().item()),
                    "confidence": row_list_value(row, "confidence", s),
                    "fallback_reason": row_list_value(row, "fallback_reason", s, ""),
                    "fit_improvement": row_list_value(row, "fit_improvement", s),
                    "num_fit_frames": row_list_value(row, "num_fit_frames", s),
                    "num_fit_windows": row_list_value(row, "num_fit_windows", s),
                }
            )
        per_sequence.append(
            {
                "sequence_id": name,
                "has_sequence_specific_rJS": True,
                "method_source": row.get("method_source", meta.get("method", "")),
                "contact_selection_mode": row.get("contact_selection_mode", ""),
                "contact_window_count": row.get("contact_window_count", None),
                "contact_side_window_count": row.get("contact_side_window_count", None),
                "sensors": sensors,
            }
        )
    return {
        "rjs_path": meta["path"],
        "source_path": meta.get("source_path", ""),
        "offset_field_used": meta["offset_field"],
        "sequence_specific": sequence_specific,
        "fallback_or_global_note": "sequence-specific r_JS was found for every TotalCapture test sequence",
        "coordinate_contract": meta.get("coordinate_contract", ""),
        "sensor_to_joint_map": sensor_to_joint_map(),
        "summary": meta.get("summary", {}),
        "config": meta.get("config", {}),
        "per_sequence": per_sequence,
    }


def process_sequence(data, idx, rjs_map, args):
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
    if name not in rjs_map:
        raise KeyError(f"rJS cache missing sequence {name}")
    rjs = rjs_map[name].float()
    if rjs.shape != (6, 3):
        raise ValueError(f"{name} rJS shape={tuple(rjs.shape)}, expected [6,3]")

    p_wj, R_wj, _p_wv = fk_imu_joints_and_vertices(pose, tran, device=args.device)
    p_rjs = p_wj[:, :5] + matvec(R_wj[:, :5], rjs[:5].view(1, 5, 3))
    R_WS = RIM.transpose(1, 2).unsqueeze(0).matmul(RIS)[:, :5]
    acc_by_method = vertex_acc_versions(p_rjs, FPS)
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
                    "vertex_id": "",
                    "mapped_joint_id": int(IMU_JOINTS[sensor_idx]),
                }
                row.update(vector_metrics(p[v], t[v]))
                row["_pred"] = p[v]
                row["_target"] = t[v]
                rows.append(row)
    return rows, examples


def read_vertex_baseline(path):
    csv_path = Path(path) / "summary_per_sensor.csv"
    if not csv_path.exists():
        return None
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["comparison_space"] == "sensor_specific_force" and row["method"] == "savgol9_p3_fd":
                rows.append(row)
    return rows


def plot_vertex_vs_rjs(vertex_rows, rjs_rows, out_dir):
    if not vertex_rows:
        return False
    rjs = {
        r["sensor_name"]: r
        for r in aggregate_rows(
            [x for x in rjs_rows if x["comparison_space"] == "sensor_specific_force" and x["method"] == "savgol9_p3_fd"],
            ["sensor_name"],
        )
    }
    vertex = {r["sensor_name"]: r for r in vertex_rows}
    sensors = list(SENSOR_NAMES[:5])
    x = np.arange(len(sensors))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, [float(vertex[s]["rmse"]) for s in sensors], width, label="vertex SavGol-9")
    ax.bar(x + width / 2, [float(rjs[s]["rmse"]) for s in sensors], width, label="rJS SavGol-9")
    ax.set_xticks(x)
    ax.set_xticklabels(sensors, rotation=20, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Vertex vs rJS acceleration RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "vertex_vs_rjs_rmse_bar.png", dpi=160)
    plt.close(fig)
    return True


def write_summary_md(path, overall, per_sensor, per_sequence, config, vertex_rows):
    preferred = config["preferred_comparison_space"]
    rows = [r for r in overall if r["comparison_space"] == preferred]
    best = min(rows, key=lambda r: r["rmse"])
    rjs_sg9 = next(r for r in rows if r["method"] == "savgol9_p3_fd")
    vertex_map = {r["sensor_name"]: r for r in (vertex_rows or [])}
    rjs_sensor_sg9 = {
        r["sensor_name"]: r
        for r in per_sensor
        if r["comparison_space"] == preferred and r["method"] == "savgol9_p3_fd"
    }
    comparable = bool(vertex_map)
    improvements = []
    for sensor, rjs_row in rjs_sensor_sg9.items():
        if sensor in vertex_map:
            improvements.append((sensor, float(vertex_map[sensor]["rmse"]) - float(rjs_row["rmse"])))
    best_improvements = sorted(improvements, key=lambda x: x[1], reverse=True)
    lower_leg = [x for x in improvements if x[0] in ("left_lower_leg", "right_lower_leg")]
    worst_sensor = max(
        [r for r in per_sensor if r["comparison_space"] == preferred and r["method"] == best["method"]],
        key=lambda r: r["rmse"],
    )
    best_sensor = min(
        [r for r in per_sensor if r["comparison_space"] == preferred and r["method"] == best["method"]],
        key=lambda r: r["rmse"],
    )
    method_lines = []
    for method in METHODS:
        row = next(r for r in rows if r["method"] == method)
        method_lines.append(
            f"| {method} | {row['mean_l2_error']:.4f} | {row['rmse']:.4f} | "
            f"{row['pearson_correlation']:.4f} | {row['cosine_similarity']:.4f} | {row['magnitude_mae']:.4f} |"
        )
    compare_lines = []
    if comparable:
        compare_lines = ["| sensor | vertex SavGol-9 RMSE | rJS SavGol-9 RMSE | delta vertex-rJS |", "|---|---:|---:|---:|"]
        for sensor in SENSOR_NAMES[:5]:
            v = float(vertex_map[sensor]["rmse"])
            r = float(rjs_sensor_sg9[sensor]["rmse"])
            compare_lines.append(f"| {sensor} | {v:.4f} | {r:.4f} | {v - r:.4f} |")
        verdict = "better" if rjs_sg9["rmse"] < float(next(r for r in config["vertex_overall"] if r["method"] == "savgol9_p3_fd")["rmse"]) else "not better"
    else:
        compare_lines = ["Vertex baseline directory was not readable; direct vertex-vs-rJS comparison was not produced."]
        verdict = "unknown"
    lower_leg_text = (
        ", ".join(f"{s}: delta {d:.4f} m/s^2" for s, d in lower_leg)
        if lower_leg
        else "not available"
    )
    positive_improvements = [x for x in best_improvements if x[1] > 0.0]
    improvement_text = (
        ", ".join(f"{s} {d:.4f}" for s, d in positive_improvements[:3])
        if positive_improvements
        else "none; all matched sensors regressed or were unavailable"
    )
    support = (
        "supported as a stronger target than vertex acceleration"
        if comparable and verdict == "better" and rjs_sg9["rmse"] < 2.5
        else "not clearly supported as a stronger target than the vertex baseline"
    )
    lines = [
        "# TotalCapture IMU Acceleration vs rJS Difference Acceleration",
        "",
        "## Setup",
        "",
        f"- Dataset path: `{config['dataset_path']}`",
        f"- rJS path: `{config['rjs_path']}`",
        f"- rJS path selection: `{config['rjs_path_selection']}`",
        f"- Vertex baseline dir: `{config['vertex_baseline_dir']}`",
        f"- Split: `{config['split']}`; sequences: {config['num_sequences']}",
        f"- FPS: {config['fps']}; gravity: {config['gravity_world']}",
        f"- Preferred comparison: `{preferred}`",
        f"- rJS contract: `{config['rjs_contract']}`",
        "- Position source: `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS`, using FK joint world positions/rotations with `tran`; no root-relative subtraction is applied.",
        "",
        "## Overall Metrics",
        "",
        "| method | mean L2 error | RMSE | Pearson corr | cosine | magnitude MAE |",
        "|---|---:|---:|---:|---:|---:|",
        *method_lines,
        "",
        "## Vertex vs rJS SavGol-9",
        "",
        *compare_lines,
        "",
        "## Required Answers",
        "",
        f"1. rJS average difference on TotalCapture: best method `{best['method']}` has mean L2 {best['mean_l2_error']:.4f} m/s^2 and RMSE {best['rmse']:.4f} m/s^2.",
        f"2. Best rJS method: `{best['method']}`.",
        f"3. rJS is `{verdict}` than the five-vertex baseline under sensor-specific SavGol-9 comparison.",
        f"4. Sensors that improved: {improvement_text}.",
        f"5. Lower-leg change: {lower_leg_text}.",
        "6. If rJS did not improve, likely causes include footlock/pseudo-constraint rJS not being a GT mount, mapped joint plus fixed offset still missing soft-tissue/strap motion, coordinate convention mismatch in R_WJ/R_WS/r_JS, or time/filtering differences.",
        f"7. Conclusion: rJS position acceleration is {support}.",
        "",
        f"Best rJS sensor: `{best_sensor['sensor_name']}` RMSE {best_sensor['rmse']:.4f}; worst rJS sensor: `{worst_sensor['sensor_name']}` RMSE {worst_sensor['rmse']:.4f}.",
        "",
        "## Outputs",
        "",
        "- `summary_overall.csv`, `summary_per_sensor.csv`, `summary_per_sequence.csv`, `frame_level_metrics.csv`",
        "- `config.json`, `rjs_offsets.json`, `SUMMARY.md`",
        "- `error_bar_rmse.png`, `corr_bar.png`, `timeseries_examples_*.png`, `scatter_*.png`, `residual_hist_*.png`, `boxplot_residuals.png`, `vertex_vs_rjs_rmse_bar.png`",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--rjs-path", type=Path, default=None)
    parser.add_argument("--vertex-baseline-dir", type=Path, default=DEFAULT_VERTEX_BASELINE)
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
    rjs_path, rjs_selection = choose_rjs_path(args.rjs_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (ROOT / "code/outputs" / f"totalcapture_imu_vs_rjs_diff_acc_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset_path)
    rjs_map = load_offset_cache(rjs_path)
    rjs_meta = load_rjs_metadata(rjs_path)
    count = len(data["name"])
    if args.max_sequences:
        count = min(count, int(args.max_sequences))
    selected_names = [str(x) for x in data["name"][:count]]
    missing = [name for name in selected_names if name not in rjs_map]
    if missing:
        raise RuntimeError(f"rJS cache {rjs_path} missing TotalCapture sequences: {missing}")

    rows = []
    examples = []
    for idx in range(count):
        seq_rows, example = process_sequence(data, idx, rjs_map, args)
        rows.extend(seq_rows)
        if example is not None:
            examples.append(example)
        print(f"[{idx + 1}/{count}] {data['name'][idx]} rows={len(seq_rows)}", flush=True)
    if not rows:
        raise RuntimeError("No comparable rows were produced.")

    overall = aggregate_rows(rows, ["comparison_space", "method"])
    per_sensor = aggregate_rows(rows, ["comparison_space", "method", "sensor_id", "sensor_name", "mapped_joint_id"])
    per_sequence = aggregate_rows(rows, ["comparison_space", "method", "sequence_id"])
    frame_rows = frame_level_rows(rows, args.trim)
    vertex_rows = read_vertex_baseline(args.vertex_baseline_dir)
    vertex_overall = []
    vo_path = Path(args.vertex_baseline_dir) / "summary_overall.csv"
    if vo_path.exists():
        with vo_path.open(newline="") as f:
            vertex_overall = list(csv.DictReader(f))

    config = {
        "script": str(Path(__file__).resolve()),
        "dataset_path": str(args.dataset_path),
        "split": "test",
        "output_dir": str(out_dir),
        "rjs_path": str(rjs_path),
        "rjs_path_selection": rjs_selection,
        "vertex_baseline_dir": str(args.vertex_baseline_dir),
        "num_sequences": count,
        "sequence_names": selected_names,
        "fps": FPS,
        "dt": 1.0 / FPS,
        "trim": args.trim,
        "gravity_world": GRAVITY_WORLD.tolist(),
        "preferred_comparison_space": args.plot_space,
        "methods": list(METHODS),
        "imu_contract": infer_imu_contract(data),
        "rjs_contract": rjs_meta.get("coordinate_contract") or "r_JS is joint-local; p_WS=p_WJ+R_WJ@r_JS.",
        "rjs_source_summary": rjs_meta.get("summary", {}),
        "rjs_source_config": rjs_meta.get("config", {}),
        "sensor_to_joint_map": sensor_to_joint_map(),
        "world_position_note": "FK joint positions use the sequence root translation `tran`; no root-relative subtraction is applied.",
        "vertex_baseline_loaded": bool(vertex_rows),
        "vertex_overall": vertex_overall,
    }
    rjs_offsets = make_rjs_offsets_json(data, rjs_map, rjs_meta, selected_names)

    write_csv(out_dir / "summary_overall.csv", overall)
    write_csv(out_dir / "summary_per_sensor.csv", per_sensor)
    write_csv(out_dir / "summary_per_sequence.csv", per_sequence)
    write_csv(out_dir / "frame_level_metrics.csv", frame_rows)
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "rjs_offsets.json", rjs_offsets)
    plot_error_bars(rows, out_dir, args.plot_space)
    plot_corr_bars(rows, out_dir, args.plot_space)
    plot_timeseries(examples, out_dir, args.timeseries_method, args.plot_space, max_examples=min(5, len(examples)))
    for method in METHODS:
        plot_scatter(rows, out_dir, method, args.plot_space)
        plot_residual_hist(rows, out_dir, method, args.plot_space)
    plot_boxplot(rows, out_dir, args.timeseries_method, args.plot_space)
    plot_vertex_vs_rjs(vertex_rows, rows, out_dir)
    write_summary_md(out_dir / "SUMMARY.md", overall, per_sensor, per_sequence, config, vertex_rows)
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "summary": str(out_dir / "SUMMARY.md")}, indent=2))


if __name__ == "__main__":
    main()
