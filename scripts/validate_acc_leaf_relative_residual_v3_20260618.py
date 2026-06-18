#!/usr/bin/env python3
"""Validate leaf-only acceleration residual audit caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l4_sensor_offset_utils import SENSOR_NAMES


ROOT_INDEX = 5
LEAF_INDICES = (0, 1, 2, 3, 4)
LEAF_SENSOR_NAMES = tuple(SENSOR_NAMES[i] for i in LEAF_INDICES)
SUMMARY_SENSOR_ORDER = ("head", "left_forearm", "left_lower_leg", "right_forearm", "right_lower_leg")
EXPERIMENT = "acc_leaf_relative_residual_v3_20260618"
COMPARISONS = {
    "raw_leaf_relative": ("aIMU_leaf_rel_raw", "aGT_leaf_rel_raw"),
    "smooth_leaf_relative": ("aIMU_leaf_rel_smooth", "aGT_leaf_rel_smooth"),
}


def corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    if a.numel() < 2:
        return float("nan")
    av = a - a.mean()
    bv = b - b.mean()
    den = av.norm() * bv.norm()
    return float((av * bv).sum() / den) if float(den) > 1e-12 else float("nan")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.reshape(-1, 3), b.reshape(-1, 3), dim=-1, eps=1e-8).mean())


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    err = a - b
    mag_err = (a.norm(dim=-1) - b.norm(dim=-1)).abs()
    axis_mae = err.abs().mean(dim=(0, 1))
    axis_rmse = err.square().mean(dim=(0, 1)).sqrt()
    return {
        "mse": float(err.square().mean()),
        "rmse": float(err.square().mean().sqrt()),
        "mae": float(err.abs().mean()),
        "l2": float(err.norm(dim=-1).mean()),
        "corr": corrcoef(a, b),
        "cosine": cosine(a, b),
        "mag_mae": float(mag_err.mean()),
        "axis_mae_x": float(axis_mae[0]),
        "axis_mae_y": float(axis_mae[1]),
        "axis_mae_z": float(axis_mae[2]),
        "axis_rmse_x": float(axis_rmse[0]),
        "axis_rmse_y": float(axis_rmse[1]),
        "axis_rmse_z": float(axis_rmse[2]),
        "energy_imu": float(a.norm(dim=-1).mean()),
        "energy_gt": float(b.norm(dim=-1).mean()),
    }


def validate_record(record: dict, path: str) -> None:
    required = (
        "aIMU_leaf_rel_raw",
        "aGT_leaf_rel_raw",
        "aIMU_leaf_rel_smooth",
        "aGT_leaf_rel_smooth",
        "aM_raw",
        "aGT_abs_raw",
        "aM_root_raw",
        "aGT_root_raw",
        "wM",
        "RMB",
        "rJS",
        "valid_mask",
        "meta",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(f"{path} missing required fields: {missing}")
    for comp, (a_key, b_key) in COMPARISONS.items():
        a = record[a_key]
        b = record[b_key]
        if a.shape != b.shape:
            raise ValueError(f"{path} {comp} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        if a.shape[-2:] != (5, 3):
            raise ValueError(f"{path} {comp} expected [T,5,3], got {tuple(a.shape)}")
    if record["aM_raw"].shape[-2:] != (6, 3) or record["aGT_abs_raw"].shape[-2:] != (6, 3):
        raise ValueError(f"{path} debug acceleration fields must be [T,6,3]")
    if record["aM_root_raw"].shape[-1:] != (3,) or record["aGT_root_raw"].shape[-1:] != (3,):
        raise ValueError(f"{path} root debug fields must be [T,3]")
    valid = record["valid_mask"].bool()
    if not bool(valid.any()):
        raise ValueError(f"{path} has no valid frames")


def sequence_metrics(record: dict, path: str) -> Tuple[List[dict], dict]:
    validate_record(record, path)
    valid = record["valid_mask"].bool()
    meta = record["meta"]
    rows = []
    debug = {
        "path": path,
        "dataset": meta["dataset"],
        "split": meta.get("split", ""),
        "sequence_name": meta["sequence_name"],
        "valid_frames": int(valid.sum()),
        "num_frames": int(valid.numel()),
        "valid_frame_range": meta.get("valid_frame_range"),
        "root_index": ROOT_INDEX,
        "leaf_indices": list(LEAF_INDICES),
        "root_excluded_from_metrics": True,
        "finite_leaf_raw": bool(torch.isfinite(record["aIMU_leaf_rel_raw"][valid]).all() and torch.isfinite(record["aGT_leaf_rel_raw"][valid]).all()),
        "finite_leaf_smooth": bool(torch.isfinite(record["aIMU_leaf_rel_smooth"][valid]).all() and torch.isfinite(record["aGT_leaf_rel_smooth"][valid]).all()),
    }
    for comparison, (a_key, b_key) in COMPARISONS.items():
        a = record[a_key][valid].float()
        b = record[b_key][valid].float()
        rows.append({
            "comparison": comparison,
            "dataset": meta["dataset"],
            "split": meta.get("split", ""),
            "sequence_name": meta["sequence_name"],
            "sensor": "overall_leaf_only",
            "valid_frames": int(valid.sum()),
            "num_sequences": 1,
            **metrics(a, b),
        })
        for sensor_idx, sensor in enumerate(LEAF_SENSOR_NAMES):
            rows.append({
                "comparison": comparison,
                "dataset": meta["dataset"],
                "split": meta.get("split", ""),
                "sequence_name": meta["sequence_name"],
                "sensor": sensor,
                "valid_frames": int(valid.sum()),
                "num_sequences": 1,
                **metrics(a[:, sensor_idx:sensor_idx + 1], b[:, sensor_idx:sensor_idx + 1]),
            })
    return rows, debug


def add_group(groups: dict, key: tuple, a: torch.Tensor, b: torch.Tensor, frames: int) -> None:
    groups[key].append((a, b, frames))


def aggregate_rows(records: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for rec in records:
        valid = rec["valid_mask"].bool()
        meta = rec["meta"]
        dataset = meta["dataset"]
        split = meta.get("split", "")
        valid_frames = int(valid.sum())
        for comparison, (a_key, b_key) in COMPARISONS.items():
            a_all = rec[a_key][valid].float()
            b_all = rec[b_key][valid].float()
            for dataset_key, split_key in ((dataset, split), ("ALL", "ALL_SPLITS")):
                add_group(groups, (comparison, "dataset_split", dataset_key, split_key, "overall_leaf_only"), a_all, b_all, valid_frames)
                for sensor_idx, sensor in enumerate(LEAF_SENSOR_NAMES):
                    add_group(
                        groups,
                        (comparison, "sensor", dataset_key, split_key, sensor),
                        a_all[:, sensor_idx:sensor_idx + 1],
                        b_all[:, sensor_idx:sensor_idx + 1],
                        valid_frames,
                    )
            for dataset_key in (dataset, "ALL"):
                add_group(groups, (comparison, "dataset", dataset_key, "", "overall_leaf_only"), a_all, b_all, valid_frames)
                for sensor_idx, sensor in enumerate(LEAF_SENSOR_NAMES):
                    add_group(
                        groups,
                        (comparison, "sensor_dataset", dataset_key, "", sensor),
                        a_all[:, sensor_idx:sensor_idx + 1],
                        b_all[:, sensor_idx:sensor_idx + 1],
                        valid_frames,
                    )
    rows = []
    seen = set()
    for (comparison, group_type, dataset, split, sensor), parts in sorted(groups.items()):
        dedupe_key = (comparison, group_type, dataset, split, sensor)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        a = torch.cat([part[0] for part in parts], dim=0)
        b = torch.cat([part[1] for part in parts], dim=0)
        rows.append({
            "comparison": comparison,
            "group_type": group_type,
            "dataset": dataset,
            "split": split,
            "sensor": sensor,
            "valid_frames": int(sum(part[2] for part in parts)),
            "num_sequences": int(len(parts)),
            **metrics(a, b),
        })
    return rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{float(value):.6f}"


def find_row(rows: List[dict], comparison: str, dataset: str, sensor: str = "overall_leaf_only") -> dict:
    group_type = "dataset" if sensor == "overall_leaf_only" else "sensor_dataset"
    for row in rows:
        if (
            row["comparison"] == comparison
            and row["dataset"] == dataset
            and row["sensor"] == sensor
            and row["group_type"] == group_type
        ):
            return row
    raise KeyError((comparison, dataset, sensor))


def main_result_table(rows: List[dict]) -> str:
    lines = [
        "| Dataset | Formulation | L2 | RMSE | MAE | Corr | Cosine | Mag MAE | Valid frames |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("AMASS", "DIP", "TotalCapture", "ALL"):
        for comparison in ("raw_leaf_relative", "smooth_leaf_relative"):
            row = find_row(rows, comparison, dataset)
            lines.append(
                f"| {dataset} | {comparison} | {fmt(row['l2'])} | {fmt(row['rmse'])} | "
                f"{fmt(row['mae'])} | {fmt(row['corr'])} | {fmt(row['cosine'])} | "
                f"{fmt(row['mag_mae'])} | {int(row['valid_frames'])} |"
            )
    return "\n".join(lines)


def sensor_table(rows: List[dict]) -> str:
    lines = [
        "| Sensor | Raw L2 | Raw RMSE | Raw Corr | Smooth L2 | Smooth RMSE | Smooth Corr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sensor in SUMMARY_SENSOR_ORDER:
        raw = find_row(rows, "raw_leaf_relative", "ALL", sensor)
        smooth = find_row(rows, "smooth_leaf_relative", "ALL", sensor)
        lines.append(
            f"| {sensor} | {fmt(raw['l2'])} | {fmt(raw['rmse'])} | {fmt(raw['corr'])} | "
            f"{fmt(smooth['l2'])} | {fmt(smooth['rmse'])} | {fmt(smooth['corr'])} |"
        )
    return "\n".join(lines)


def build_summary(result: dict) -> str:
    aggregate = result["aggregate"]
    raw_all = find_row(aggregate, "raw_leaf_relative", "ALL")
    smooth_all = find_row(aggregate, "smooth_leaf_relative", "ALL")
    l2_delta = raw_all["l2"] - smooth_all["l2"]
    rmse_delta = raw_all["rmse"] - smooth_all["rmse"]
    corr_delta = smooth_all["corr"] - raw_all["corr"]
    reduces = smooth_all["l2"] < raw_all["l2"] and smooth_all["rmse"] < raw_all["rmse"]
    lines = [
        "# Acc Leaf-Relative Residual v3 20260618",
        "",
        f"Experiment: `{EXPERIMENT}`",
        "",
        "## Contract",
        "",
        "- root index = 5",
        "- leaf indices = 0..4",
        "- root is used only as reference acceleration",
        "- root is not included in residual/loss/metric",
        "- frame = model/world frame M",
        "- no sensor-local rotation",
        "- GT FK uses tran=0",
        "- diff method = centered second difference, dt=1/60",
        "- smooth method = centered moving average, window=9",
        "",
        "## Main Result Table",
        "",
        main_result_table(aggregate),
        "",
        "## Per-Sensor Compact Table",
        "",
        sensor_table(aggregate),
        "",
        "## Required Judgment",
        "",
        f"- Smooth L2 delta vs raw: `{fmt(l2_delta)}`; smooth RMSE delta vs raw: `{fmt(rmse_delta)}`.",
        f"- Smooth corr delta vs raw: `{fmt(corr_delta)}`.",
        (
            "- smoothing is necessary before comparing IMU/FK acceleration residuals."
            if reduces
            else "- smoothing did not reduce both L2 and RMSE in this audit."
        ),
        (
            "- root-relative smoothed acceleration is a cleaner explainability target."
            if corr_delta > 0.05
            else "- smooth_leaf_relative corr does not improve substantially enough by the >0.05 rule."
        ),
        "- If residual remains large after smoothing, measured IMU acceleration still contains noise / bias / soft-tissue / convention mismatch not explained by zero-trans FK.",
        "",
        "## Explicit Non-Claims",
        "",
        "- This is not AccCurve training.",
        "- This is not PL/NewPL training.",
        "- This is not full-pipeline evaluation.",
        "- Root channel is not evaluated as residual.",
        "- Do not claim downstream pose improvement.",
        "",
        "## Artifacts",
        "",
        "- `cache_manifest.json`: leaf-relative cache file list and contract.",
        "- `metrics.json`: aggregate metrics and checks.",
        "- `per_sequence.csv`: per-sequence leaf-only metrics.",
        "- `debug.json`: root-reference/debug checks.",
    ]
    return "\n".join(lines) + "\n"


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError(f"{path} is not {EXPERIMENT}")
    if manifest.get("root_index") != ROOT_INDEX:
        raise ValueError(f"{path} root_index must be {ROOT_INDEX}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate leaf-relative acceleration residual audit v3.")
    parser.add_argument("--root", type=Path, default=Path("data/experiments/acc_leaf_relative_residual_v3_20260618"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=0)
    args = parser.parse_args()

    root = args.root
    manifest_path = args.manifest or root / "cache_manifest.json"
    manifest = load_manifest(manifest_path)
    seq_rows = []
    debug = []
    records = []
    for idx, item in enumerate(manifest["cache_files"]):
        if args.max_sequences and idx >= args.max_sequences:
            break
        rec = torch.load(item["path"], map_location="cpu", weights_only=False)
        rows, dbg = sequence_metrics(rec, item["path"])
        seq_rows.extend(rows)
        debug.append(dbg)
        records.append(rec)
    aggregate = aggregate_rows(records)
    checks = {
        "shape_consistency": all(
            rec["aIMU_leaf_rel_raw"].shape == rec["aGT_leaf_rel_raw"].shape == rec["aIMU_leaf_rel_smooth"].shape == rec["aGT_leaf_rel_smooth"].shape
            and rec["aIMU_leaf_rel_raw"].shape[-2:] == (5, 3)
            for rec in records
        ),
        "root_excluded_from_metrics": True,
        "root_index": ROOT_INDEX,
        "leaf_indices": list(LEAF_INDICES),
        "num_sequences": len(records),
        "valid_frames": int(sum(int(rec["valid_mask"].bool().sum()) for rec in records)),
    }
    result = {
        "experiment": EXPERIMENT,
        "manifest": str(manifest_path),
        "num_sequences": len(records),
        "valid_frames": checks["valid_frames"],
        "checks": checks,
        "aggregate": aggregate,
        "debug": debug,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (root / "debug.json").write_text(json.dumps(debug, indent=2, sort_keys=True) + "\n")
    write_csv(root / "per_sequence.csv", seq_rows)
    (root / "summary.md").write_text(build_summary(result))
    print(json.dumps({
        "summary": str(root / "summary.md"),
        "metrics": str(root / "metrics.json"),
        "per_sequence": str(root / "per_sequence.csv"),
        "debug": str(root / "debug.json"),
        "num_sequences": len(records),
        "valid_frames": checks["valid_frames"],
    }, indent=2))


if __name__ == "__main__":
    main()
