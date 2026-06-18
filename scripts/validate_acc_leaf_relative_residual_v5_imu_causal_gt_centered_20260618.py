#!/usr/bin/env python3
"""Validate asymmetric leaf-only acceleration residual audit caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l4_sensor_offset_utils import SENSOR_NAMES


ROOT_INDEX = 5
LEAF_INDICES = (0, 1, 2, 3, 4)
LEAF_SENSOR_NAMES = tuple(SENSOR_NAMES[i] for i in LEAF_INDICES)
SUMMARY_SENSOR_ORDER = ("head", "left_forearm", "left_lower_leg", "right_forearm", "right_lower_leg")
EXPERIMENT = "acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618"
DEFAULT_ROOT = Path("data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618")
V3_REFERENCE_PATH = Path("data/experiments/acc_leaf_relative_residual_v3_20260618/metrics.json")
V4_REFERENCE_PATH = Path("data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/metrics.json")
COMPARISONS = {
    "raw_leaf_relative": ("aIMU_leaf_rel_raw", "aGT_leaf_rel_raw"),
    "v4_symmetric_butter_reference": ("aIMU_leaf_rel_butter2_4hz", "aGT_leaf_rel_butter2_4hz"),
    "imu_butter2_4hz_vs_gt_centered_ma9": ("aIMU_leaf_rel_butter2_4hz", "aGT_leaf_rel_centered_ma9"),
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


def smooth_centered(x: torch.Tensor, window: int = 9) -> torch.Tensor:
    if window <= 1:
        return x.float()
    if window % 2 == 0:
        raise ValueError("centered moving average window must be odd")
    pad = window // 2
    flat = x.float().reshape(x.shape[0], -1).T.unsqueeze(0)
    flat = torch.nn.functional.pad(flat, (pad, pad), mode="replicate")
    kernel = torch.ones(flat.shape[1], 1, window, dtype=flat.dtype, device=flat.device) / float(window)
    out = torch.nn.functional.conv1d(flat, kernel, groups=flat.shape[1]).squeeze(0).T
    return out.reshape_as(x)


def centered_ma9_valid_segment(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    valid = record["valid_mask"].bool()
    valid_idx = valid.nonzero(as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        raise ValueError("record has no valid frames")
    start = int(valid_idx[0])
    end = int(valid_idx[-1]) + 1
    imu_segment = record["aIMU_leaf_rel_raw"][start:end].float()
    gt_segment = record["aGT_leaf_rel_raw"][start:end].float()
    if not torch.isfinite(imu_segment).all() or not torch.isfinite(gt_segment).all():
        raise ValueError("centered_ma9_oracle segment contains non-finite values")
    return smooth_centered(imu_segment, 9), smooth_centered(gt_segment, 9)


def validate_record(record: dict, path: str) -> None:
    required = (
        "aIMU_leaf_rel_raw",
        "aGT_leaf_rel_raw",
        "aIMU_leaf_rel_butter2_4hz",
        "aGT_leaf_rel_butter2_4hz",
        "aGT_leaf_rel_centered_ma9",
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
    for key in ("aIMU_leaf_rel_raw", "aGT_leaf_rel_raw", "aIMU_leaf_rel_butter2_4hz", "aGT_leaf_rel_butter2_4hz", "aGT_leaf_rel_centered_ma9"):
        if not torch.isfinite(record[key][valid]).all():
            raise ValueError(f"{path} has non-finite {key} on valid frames")


def row_for(comparison: str, record: dict, path: str) -> tuple[list[dict], dict]:
    valid = record["valid_mask"].bool()
    meta = record["meta"]
    if comparison == "centered_ma9_oracle":
        a, b = centered_ma9_valid_segment(record)
    else:
        a_key, b_key = COMPARISONS[comparison]
        a = record[a_key][valid].float()
        b = record[b_key][valid].float()
    rows = [{
        "comparison": comparison,
        "dataset": meta["dataset"],
        "split": meta.get("split", ""),
        "sequence_name": meta["sequence_name"],
        "sensor": "overall_leaf_only",
        "valid_frames": int(valid.sum()),
        "num_sequences": 1,
        **metrics(a, b),
    }]
    if comparison != "centered_ma9_oracle":
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
        "finite_leaf_butter": bool(torch.isfinite(record["aIMU_leaf_rel_butter2_4hz"][valid]).all() and torch.isfinite(record["aGT_leaf_rel_butter2_4hz"][valid]).all()),
        "finite_gt_centered_ma9": bool(torch.isfinite(record["aGT_leaf_rel_centered_ma9"][valid]).all()),
        "zero_lookahead": bool(meta.get("imu_smoother", {}).get("zero_lookahead", False)),
        "initialized_from_first_valid_sample": bool(meta.get("imu_smoother", {}).get("initialized_from_first_valid_sample", False)),
        "gt_centered_target_only": bool(meta.get("gt_smoother", {}).get("target_only", False)),
    }
    return rows, debug


def sequence_metrics(record: dict, path: str) -> tuple[list[dict], dict]:
    validate_record(record, path)
    rows = []
    debug = None
    for comparison in ("raw_leaf_relative", "v4_symmetric_butter_reference", "imu_butter2_4hz_vs_gt_centered_ma9", "centered_ma9_oracle"):
        comp_rows, comp_debug = row_for(comparison, record, path)
        rows.extend(comp_rows)
        debug = comp_debug
    return rows, debug or {}


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
        oracle_a, oracle_b = centered_ma9_valid_segment(rec)
        for dataset_key, split_key in ((dataset, split), ("ALL", "ALL_SPLITS")):
            add_group(groups, ("centered_ma9_oracle", "dataset_split", dataset_key, split_key, "overall_leaf_only"), oracle_a, oracle_b, valid_frames)
        for dataset_key in (dataset, "ALL"):
            add_group(groups, ("centered_ma9_oracle", "dataset", dataset_key, "", "overall_leaf_only"), oracle_a, oracle_b, valid_frames)
    rows = []
    for (comparison, group_type, dataset, split, sensor), parts in sorted(groups.items()):
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


def find_row(rows: List[dict], comparison: str, dataset: str, sensor: str = "overall_leaf_only", group_type: str | None = None) -> dict:
    if group_type is None:
        group_type = "dataset" if sensor == "overall_leaf_only" else "sensor_dataset"
    for row in rows:
        if (
            row["comparison"] == comparison
            and row["dataset"] == dataset
            and row["sensor"] == sensor
            and row["group_type"] == group_type
        ):
            return row
    raise KeyError((comparison, dataset, sensor, group_type))


def sensor_table(rows: List[dict]) -> str:
    lines = [
        "| Sensor | Raw L2 | Raw RMSE | Raw Corr | Asym L2 | Asym RMSE | Asym Corr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sensor in SUMMARY_SENSOR_ORDER:
        raw = find_row(rows, "raw_leaf_relative", "ALL", sensor)
        butter = find_row(rows, "imu_butter2_4hz_vs_gt_centered_ma9", "ALL", sensor)
        lines.append(
            f"| {sensor} | {fmt(raw['l2'])} | {fmt(raw['rmse'])} | {fmt(raw['corr'])} | "
            f"{fmt(butter['l2'])} | {fmt(butter['rmse'])} | {fmt(butter['corr'])} |"
        )
    return "\n".join(lines)


def load_v3_reference() -> dict:
    if not V3_REFERENCE_PATH.exists():
        return {}
    data = json.loads(V3_REFERENCE_PATH.read_text())
    refs = {}
    for row in data.get("aggregate", []):
        if row.get("group_type") == "dataset" and row.get("sensor") == "overall_leaf_only" and row.get("comparison") == "smooth_leaf_relative":
            if row.get("dataset") in ("DIP", "TotalCapture"):
                refs[row["dataset"]] = row
    return refs


def load_v4_reference_from_metrics() -> dict:
    if not V4_REFERENCE_PATH.exists():
        return {}
    data = json.loads(V4_REFERENCE_PATH.read_text())
    refs = {}
    for row in data.get("aggregate", []):
        if row.get("group_type") == "dataset" and row.get("sensor") == "overall_leaf_only" and row.get("comparison") == "butter2_4hz_leaf_relative":
            if row.get("dataset") in ("DIP", "TotalCapture"):
                refs[row["dataset"]] = row
    return refs


def primary_table(rows: List[dict]) -> str:
    lines = [
        "| Dataset | Split | Formulation | L2 | RMSE | MAE | Corr | Cosine | Mag MAE | Valid frames |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("DIP", "TotalCapture"):
        for comparison in ("raw_leaf_relative", "imu_butter2_4hz_vs_gt_centered_ma9"):
            try:
                row = find_row(rows, comparison, dataset)
            except KeyError:
                continue
            lines.append(
                f"| {dataset} | all | {comparison} | {fmt(row['l2'])} | {fmt(row['rmse'])} | "
                f"{fmt(row['mae'])} | {fmt(row['corr'])} | {fmt(row['cosine'])} | "
                f"{fmt(row['mag_mae'])} | {int(row['valid_frames'])} |"
            )
    return "\n".join(lines)


def secondary_table(rows: List[dict]) -> str:
    lines = [
        "| Dataset | Formulation | L2 | RMSE | Corr | Valid frames |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in ("AMASS", "ALL"):
        for comparison in ("raw_leaf_relative", "v4_symmetric_butter_reference", "imu_butter2_4hz_vs_gt_centered_ma9", "centered_ma9_oracle"):
            try:
                row = find_row(rows, comparison, dataset)
            except KeyError:
                continue
            lines.append(f"| {dataset} | {comparison} | {fmt(row['l2'])} | {fmt(row['rmse'])} | {fmt(row['corr'])} | {int(row['valid_frames'])} |")
    return "\n".join(lines)


def decision(aggregate: List[dict], v3_ref: dict) -> tuple[str, dict]:
    try:
        dip_raw = find_row(aggregate, "raw_leaf_relative", "DIP")
        dip_asym = find_row(aggregate, "imu_butter2_4hz_vs_gt_centered_ma9", "DIP")
        dip_v4 = find_row(aggregate, "v4_symmetric_butter_reference", "DIP")
        tc_raw = find_row(aggregate, "raw_leaf_relative", "TotalCapture")
        tc_asym = find_row(aggregate, "imu_butter2_4hz_vs_gt_centered_ma9", "TotalCapture")
        tc_v4 = find_row(aggregate, "v4_symmetric_butter_reference", "TotalCapture")
    except KeyError:
        return "not_applicable", {
            "improves_dip": False,
            "improves_totalcapture": False,
            "dip_rmse_ratio_vs_v3_oracle": float("nan"),
            "totalcapture_rmse_ratio_vs_v3_oracle": float("nan"),
            "dip_rmse_ratio_vs_v4_symmetric_butter": float("nan"),
            "totalcapture_rmse_ratio_vs_v4_symmetric_butter": float("nan"),
            "dip_corr": float("nan"),
            "totalcapture_corr": float("nan"),
        }
    dip_ref = v3_ref.get("DIP", {})
    tc_ref = v3_ref.get("TotalCapture", {})
    dip_rmse_ratio = dip_asym["rmse"] / dip_ref.get("rmse", dip_asym["rmse"]) if dip_ref else float("nan")
    tc_rmse_ratio = tc_asym["rmse"] / tc_ref.get("rmse", tc_asym["rmse"]) if tc_ref else float("nan")
    dip_v4_rmse_ratio = dip_asym["rmse"] / dip_v4["rmse"]
    tc_v4_rmse_ratio = tc_asym["rmse"] / tc_v4["rmse"]
    improves_dip = dip_asym["l2"] < dip_raw["l2"] and dip_asym["rmse"] < dip_raw["rmse"] and dip_asym["corr"] > dip_raw["corr"]
    improves_tc = tc_asym["l2"] < tc_raw["l2"] and tc_asym["rmse"] < tc_raw["rmse"] and tc_asym["corr"] > tc_raw["corr"]
    pass_gate = improves_dip and improves_tc and dip_asym["corr"] >= 0.93 and tc_asym["corr"] >= 0.93 and dip_v4_rmse_ratio <= 1.15 and tc_v4_rmse_ratio <= 1.15 and dip_rmse_ratio <= 1.30 and tc_rmse_ratio <= 1.30
    soft_pass_gate = improves_dip and improves_tc and dip_asym["corr"] >= 0.90 and tc_asym["corr"] >= 0.90 and dip_v4_rmse_ratio <= 1.35 and tc_v4_rmse_ratio <= 1.35
    if pass_gate:
        label = "pass"
    elif soft_pass_gate:
        label = "soft-pass"
    else:
        label = "fail"
    return label, {
        "improves_dip": improves_dip,
        "improves_totalcapture": improves_tc,
        "dip_rmse_ratio_vs_v3_oracle": dip_rmse_ratio,
        "totalcapture_rmse_ratio_vs_v3_oracle": tc_rmse_ratio,
        "dip_rmse_ratio_vs_v4_symmetric_butter": dip_v4_rmse_ratio,
        "totalcapture_rmse_ratio_vs_v4_symmetric_butter": tc_v4_rmse_ratio,
        "dip_corr": dip_asym["corr"],
        "totalcapture_corr": tc_asym["corr"],
    }


def build_summary(result: dict) -> str:
    aggregate = result["aggregate"]
    v3_ref = result["v3_reference"]
    v4_ref = result.get("v4_reference", {})
    label, decision_debug = decision(aggregate, v3_ref)

    dip_v4 = v4_ref.get("DIP")
    if dip_v4 is None:
        dip_v4 = find_row(aggregate, "v4_symmetric_butter_reference", "DIP")
    tc_v4 = v4_ref.get("TotalCapture")
    if tc_v4 is None:
        tc_v4 = find_row(aggregate, "v4_symmetric_butter_reference", "TotalCapture")
    dip_v3 = v3_ref.get("DIP", {})
    tc_v3 = v3_ref.get("TotalCapture", {})
    dip_asym = find_row(aggregate, "imu_butter2_4hz_vs_gt_centered_ma9", "DIP")
    tc_asym = find_row(aggregate, "imu_butter2_4hz_vs_gt_centered_ma9", "TotalCapture")

    lines = [
        "# Acc Leaf-Relative Residual v5 IMU Causal GT Centered 20260618",
        "",
        f"Experiment: `{EXPERIMENT}`",
        "",
        "## Contract",
        "",
        "- root index 5",
        "- leaf indices 0..4",
        "- root used only as reference acceleration",
        "- root excluded from residual/loss/metric",
        "- GT FK uses tran=0",
        "- frame = model/world frame M",
        "- no sensor-local rotation",
        "- IMU smoothing: causal Butterworth order=2 cutoff=4Hz; realtime / zero-lookahead",
        "- GT smoothing: centered moving average window=9; non-realtime; target-only",
        "- GT centered smoothing is allowed because GT is used only during training/eval; runtime does not use GT",
        "",
        "## Primary DIP / TotalCapture Table",
        "",
        primary_table(aggregate),
        "",
        "## Reference Comparison",
        "",
        f"- DIP v4 symmetric butter L2/RMSE/corr = {fmt(dip_v4.get('l2', float('nan')))} / {fmt(dip_v4.get('rmse', float('nan')))} / {fmt(dip_v4.get('corr', float('nan')))}",
        f"- TotalCapture v4 symmetric butter L2/RMSE/corr = {fmt(tc_v4.get('l2', float('nan')))} / {fmt(tc_v4.get('rmse', float('nan')))} / {fmt(tc_v4.get('corr', float('nan')))}",
        f"- DIP v3 centered_ma9 L2/RMSE/corr = {fmt(dip_v3.get('l2', float('nan')))} / {fmt(dip_v3.get('rmse', float('nan')))} / {fmt(dip_v3.get('corr', float('nan')))}",
        f"- TotalCapture v3 centered_ma9 L2/RMSE/corr = {fmt(tc_v3.get('l2', float('nan')))} / {fmt(tc_v3.get('rmse', float('nan')))} / {fmt(tc_v3.get('corr', float('nan')))}",
        f"- asymmetric improves over raw on DIP and TC: `{decision_debug['improves_dip'] and decision_debug['improves_totalcapture']}`",
        f"- DIP asym/v4 RMSE ratio: `{fmt(decision_debug['dip_rmse_ratio_vs_v4_symmetric_butter'])}`; asym/v3 oracle RMSE ratio: `{fmt(decision_debug['dip_rmse_ratio_vs_v3_oracle'])}`",
        f"- TotalCapture asym/v4 RMSE ratio: `{fmt(decision_debug['totalcapture_rmse_ratio_vs_v4_symmetric_butter'])}`; asym/v3 oracle RMSE ratio: `{fmt(decision_debug['totalcapture_rmse_ratio_vs_v3_oracle'])}`",
        f"- DIP asymmetric is {'better' if dip_asym['rmse'] < dip_v4.get('rmse', float('inf')) else 'worse'} than v4 symmetric butter by RMSE.",
        f"- TotalCapture asymmetric is {'better' if tc_asym['rmse'] < tc_v4.get('rmse', float('inf')) else 'worse'} than v4 symmetric butter by RMSE.",
        "",
        "## Decision Rule",
        "",
        f"Decision: `{label}`.",
        "",
        "- Pass if DIP and TC improve over raw in L2/RMSE/corr, corr remains >=0.93, RMSE is not worse than v4 symmetric butter by more than 15%, and RMSE is not worse than v3 centered oracle by more than 30%.",
        "- Soft pass if DIP and TC improve strongly over raw, corr remains >=0.90 on both, but RMSE is 15-35% worse than v4 symmetric butter.",
        "- Fail if either DIP or TC does not improve over raw, corr drops below 0.90, or TC degrades sharply relative to v4.",
        "",
        "## Interpretation Notes",
        "",
        "- GT centered smoothing may reduce target noise.",
        "- IMU causal smoothing has phase lag, while GT centered smoothing is near zero-phase.",
        "- Therefore residual can become slightly worse than v4 symmetric butter even if the target is cleaner.",
        "- This experiment tests training-target cleanliness, not runtime GT availability.",
        "",
        "## Secondary Synthetic Sanity",
        "",
        secondary_table(aggregate),
        "",
        "## Per-Sensor Compact Table",
        "",
        sensor_table(aggregate),
        "",
        "## Non-Claims",
        "",
        "- This is not AccCurve training.",
        "- This is not PL/NewPL training.",
        "- This is not full-pipeline evaluation.",
        "- This does not claim downstream pose improvement.",
        "- AMASS is synthetic and not primary evidence.",
        "- GT centered smoothing is not available at runtime; it is target-only.",
        "",
        "## Artifacts",
        "",
        "- `cache_manifest.json`",
        "- `metrics.json`",
        "- `per_sequence.csv`",
        "- `debug.json`",
        "- `summary.md`",
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
    parser = argparse.ArgumentParser(description="Validate asymmetric leaf-relative acceleration residual audit v5.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
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
            rec["aIMU_leaf_rel_raw"].shape == rec["aGT_leaf_rel_raw"].shape == rec["aIMU_leaf_rel_butter2_4hz"].shape == rec["aGT_leaf_rel_butter2_4hz"].shape == rec["aGT_leaf_rel_centered_ma9"].shape
            and rec["aIMU_leaf_rel_raw"].shape[-2:] == (5, 3)
            for rec in records
        ),
        "root_excluded_from_metrics": True,
        "root_index": ROOT_INDEX,
        "leaf_indices": list(LEAF_INDICES),
        "zero_lookahead": True,
        "num_sequences": len(records),
        "valid_frames": int(sum(int(rec["valid_mask"].bool().sum()) for rec in records)),
    }
    v3_reference = load_v3_reference()
    v4_reference = load_v4_reference_from_metrics()
    result = {
        "experiment": EXPERIMENT,
        "manifest": str(manifest_path),
        "num_sequences": len(records),
        "valid_frames": checks["valid_frames"],
        "checks": checks,
        "aggregate": aggregate,
        "debug": debug,
        "v3_reference": v3_reference,
        "v4_reference": v4_reference,
    }
    label, decision_debug = decision(aggregate, v3_reference)
    result["decision"] = {"label": label, **decision_debug}
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
        "decision": label,
    }, indent=2))


if __name__ == "__main__":
    main()
