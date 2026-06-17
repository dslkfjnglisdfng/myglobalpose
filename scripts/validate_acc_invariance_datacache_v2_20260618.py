#!/usr/bin/env python3
"""Validate and summarize acc invariance v2 root-IMU-relative caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l4_sensor_offset_utils import SENSOR_NAMES


ROOT_INDEX = 5
EXPERIMENT = "acc_invariance_datacache_v2_rebuild_20260618"


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
    axis = err.abs().mean(dim=(0, 1))
    return {
        "mse": float(err.square().mean()),
        "rmse": float(err.square().mean().sqrt()),
        "mae": float(err.abs().mean()),
        "l2": float(err.norm(dim=-1).mean()),
        "corr": corrcoef(a, b),
        "cosine": cosine(a, b),
        "mag_mae": float(mag_err.mean()),
        "axis_mae_x": float(axis[0]),
        "axis_mae_y": float(axis[1]),
        "axis_mae_z": float(axis[2]),
        "energy_a": float(a.norm(dim=-1).mean()),
        "energy_b": float(b.norm(dim=-1).mean()),
    }


def sequence_metrics(record: dict, path: str) -> Tuple[List[dict], dict]:
    aM_rel = record["aM_rel"].float()
    aGT_rel = record["aGT_rel"].float()
    aM_raw = record["aM_raw"].float()
    aM_smooth = record["aM_smooth"].float()
    aGT_raw = record["aGT_raw"].float()
    valid = record["valid_mask"].bool()
    meta = record["meta"]
    if aM_rel.shape != aGT_rel.shape:
        raise ValueError(f"{path} shape mismatch: aM_rel={tuple(aM_rel.shape)} aGT_rel={tuple(aGT_rel.shape)}")
    if aM_rel.shape[-2:] != (6, 3):
        raise ValueError(f"{path} expected [T,6,3], got {tuple(aM_rel.shape)}")
    if not bool(valid.any()):
        raise ValueError(f"{path} has no valid frames")
    rows = []
    comparisons = {
        "raw absolute": (aM_raw[valid], aGT_raw[valid], "aM", "FK absolute zero-trans"),
        "zero-trans old": (aM_smooth[valid], aGT_raw[valid], "aM_smooth", "FK zero-trans absolute"),
        "v2 relative (NEW)": (aM_rel[valid], aGT_rel[valid], "aM_rel", "aGT_rel"),
    }
    root_abs = float(aM_rel[valid, ROOT_INDEX].abs().mean())
    leakage_raw_corr = corrcoef(aM_smooth[valid], aGT_rel[valid])
    rel_corr = corrcoef(aM_rel[valid], aGT_rel[valid])
    debug = {
        "path": path,
        "dataset": meta["dataset"],
        "split": meta.get("split"),
        "sequence_name": meta["sequence_name"],
        "valid_frames": int(valid.sum()),
        "valid_frame_range": meta.get("valid_frame_range"),
        "root_abs_mean_aM_rel": root_abs,
        "corr_aM_smooth_vs_aGT_rel": leakage_raw_corr,
        "corr_aM_rel_vs_aGT_rel": rel_corr,
        "leakage_test_pass": bool(rel_corr > leakage_raw_corr),
    }
    for comp, (a, b, inp, target) in comparisons.items():
        overall = metrics(a, b)
        rows.append({
            "comparison": comp,
            "input": inp,
            "target": target,
            "dataset": meta["dataset"],
            "split": meta.get("split", ""),
            "sequence_name": meta["sequence_name"],
            "sensor": "overall",
            "valid_frames": int(valid.sum()),
            **overall,
            "root_abs_mean_aM_rel": root_abs,
            "leakage_raw_corr": leakage_raw_corr,
            "leakage_rel_corr": rel_corr,
        })
        for sensor_idx, sensor in enumerate(SENSOR_NAMES):
            sensor_metrics = metrics(a[:, sensor_idx:sensor_idx + 1], b[:, sensor_idx:sensor_idx + 1])
            rows.append({
                "comparison": comp,
                "input": inp,
                "target": target,
                "dataset": meta["dataset"],
                "split": meta.get("split", ""),
                "sequence_name": meta["sequence_name"],
                "sensor": sensor,
                "valid_frames": int(valid.sum()),
                **sensor_metrics,
                "root_abs_mean_aM_rel": root_abs,
                "leakage_raw_corr": leakage_raw_corr,
                "leakage_rel_corr": rel_corr,
            })
    return rows, debug


def aggregate_rows(records: List[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for rec in records:
        valid = rec["valid_mask"].bool()
        grouped[("raw absolute", rec["meta"]["dataset"], "overall")].append((rec["aM_raw"][valid].float(), rec["aGT_raw"][valid].float()))
        grouped[("zero-trans old", rec["meta"]["dataset"], "overall")].append((rec["aM_smooth"][valid].float(), rec["aGT_raw"][valid].float()))
        grouped[("v2 relative (NEW)", rec["meta"]["dataset"], "overall")].append((rec["aM_rel"][valid].float(), rec["aGT_rel"][valid].float()))
        for i, sensor in enumerate(SENSOR_NAMES):
            grouped[("raw absolute", rec["meta"]["dataset"], sensor)].append((rec["aM_raw"][valid, i:i + 1].float(), rec["aGT_raw"][valid, i:i + 1].float()))
            grouped[("zero-trans old", rec["meta"]["dataset"], sensor)].append((rec["aM_smooth"][valid, i:i + 1].float(), rec["aGT_raw"][valid, i:i + 1].float()))
            grouped[("v2 relative (NEW)", rec["meta"]["dataset"], sensor)].append((rec["aM_rel"][valid, i:i + 1].float(), rec["aGT_rel"][valid, i:i + 1].float()))
    rows = []
    for (comp, dataset, sensor), pairs in sorted(grouped.items()):
        a = torch.cat([p[0] for p in pairs], dim=0)
        b = torch.cat([p[1] for p in pairs], dim=0)
        rows.append({
            "comparison": comp,
            "dataset": dataset,
            "sensor": sensor,
            "valid_frames": int(a.shape[0]),
            **metrics(a, b),
        })
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
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


def table(rows: List[dict], dataset: str = "ALL") -> str:
    selected = [r for r in rows if r["sensor"] == "overall" and (dataset == "ALL" or r["dataset"] == dataset)]
    lines = [
        "| formulation | input | target | dataset | L2 | RMSE | corr |",
        "|------------|-------|--------|---------|---:|-----:|-----:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['comparison']} | {row.get('input', '') or row['comparison']} | {row.get('target', '') or ''} | "
            f"{row['dataset']} | {fmt(row['l2'])} | {fmt(row['rmse'])} | {fmt(row['corr'])} |"
        )
    return "\n".join(lines)


def sensor_gain_table(rows: List[dict]) -> str:
    by = {(r["comparison"], r["sensor"]): r for r in rows if r["dataset"] == "ALL" or r["dataset"]}
    lines = [
        "| sensor | raw-raw L2 | smooth-smooth L2 | v2 relative L2 | gain vs raw |",
        "|---|---:|---:|---:|---:|",
    ]
    # Prefer overall aggregated across each dataset is shown in the detailed CSV;
    # this compact table aggregates all records manually below.
    return "\n".join(lines)


def build_summary(result: dict) -> str:
    aggregate = result["aggregate"]
    debug = result["debug_root_leakage"]
    rel = [r for r in aggregate if r["comparison"] == "v2 relative (NEW)" and r["sensor"] == "overall"]
    raw = [r for r in aggregate if r["comparison"] == "raw absolute" and r["sensor"] == "overall"]
    rel_mean_corr = sum(r["corr"] for r in rel) / max(len(rel), 1)
    raw_mean_corr = sum(r["corr"] for r in raw) / max(len(raw), 1)
    leak_pass = sum(1 for d in debug if d["leakage_test_pass"])
    leak_total = len(debug)
    decision = (
        "acceleration learning is root-invariant and the dataset is physically consistent"
        if rel_mean_corr > raw_mean_corr
        else "root leakage or FK inconsistency exists"
    )
    lines = [
        "# Acc Invariance Datacache v2 Rebuild 20260618",
        "",
        f"Experiment: `{EXPERIMENT}`",
        "",
        "## Contract",
        "",
        "- IMU input: `aM_rel[:, i, :] = aM_smooth[:, i, :] - aM_smooth[:, 5, :]`.",
        "- GT target: SMPL FK sensor-site position with `tran=0`, centered second difference, then root-IMU subtraction.",
        "- Root index: `5` (`pelvis`).",
        "- Frame: M/world-frame vectors; root acceleration subtraction removes translation leakage but does not rotate into sensor-local coordinates.",
        "- Difference: central second difference `(p[t-1] - 2p[t] + p[t+1]) / dt^2`, `dt=1/60`.",
        "",
        "## Command",
        "",
        "```bash",
        "python scripts/build_acc_invariance_datacache_v2_20260618.py --overwrite",
        "python scripts/validate_acc_invariance_datacache_v2_20260618.py",
        "```",
        "",
        "## Formulation Comparison",
        "",
        table(aggregate),
        "",
        "## Validator Checks",
        "",
        f"- Shape consistency: `{result['checks']['shape_consistency']}`.",
        f"- Root invariance max mean `|aM_rel[:,5]|`: `{fmt(result['checks']['root_invariance_max_abs_mean'])}`.",
        f"- Leakage test pass: `{leak_pass}/{leak_total}` sequences have `corr(aM_rel,aGT_rel) > corr(aM_smooth,aGT_rel)`.",
        f"- Mean corr(v2 relative): `{fmt(rel_mean_corr)}`; mean corr(raw absolute): `{fmt(raw_mean_corr)}`.",
        "",
        "## Required Judgment",
        "",
        f"IF `corr(v2 relative) > corr(raw absolute)`, THEN `{decision}`.",
        "",
        "## Artifacts",
        "",
        "- `cache_manifest.json`: cache file list and frame contract.",
        "- `metrics.json`: aggregate metrics and validator checks.",
        "- `per_sequence.csv`: per-sequence/per-sensor metrics.",
        "- `debug_root_leakage.json`: root invariance and leakage diagnostics.",
        "",
        "## Conclusion",
        "",
        f"Conclusion: {decision}.",
    ]
    return "\n".join(lines) + "\n"


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError(f"{path} is not {EXPERIMENT}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate acc invariance datacache v2.")
    parser.add_argument("--root", type=Path, default=Path("data/experiments/acc_invariance_datacache_v2_20260618"))
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
    # Add a true all-dataset aggregate for compact comparison.
    for comp in ("raw absolute", "zero-trans old", "v2 relative (NEW)"):
        for sensor in ("overall", *SENSOR_NAMES):
            a_parts, b_parts = [], []
            for rec in records:
                valid = rec["valid_mask"].bool()
                if comp == "raw absolute":
                    a, b = rec["aM_raw"][valid].float(), rec["aGT_raw"][valid].float()
                elif comp == "zero-trans old":
                    a, b = rec["aM_smooth"][valid].float(), rec["aGT_raw"][valid].float()
                else:
                    a, b = rec["aM_rel"][valid].float(), rec["aGT_rel"][valid].float()
                if sensor != "overall":
                    idx = list(SENSOR_NAMES).index(sensor)
                    a, b = a[:, idx:idx + 1], b[:, idx:idx + 1]
                a_parts.append(a)
                b_parts.append(b)
            if a_parts:
                a = torch.cat(a_parts, dim=0)
                b = torch.cat(b_parts, dim=0)
                aggregate.append({"comparison": comp, "dataset": "ALL", "sensor": sensor, "valid_frames": int(a.shape[0]), **metrics(a, b)})
    checks = {
        "shape_consistency": all(torch.load(item["path"], map_location="cpu", weights_only=False)["aM_rel"].shape == torch.load(item["path"], map_location="cpu", weights_only=False)["aGT_rel"].shape for item in manifest["cache_files"][: len(records)]),
        "root_invariance_max_abs_mean": max((d["root_abs_mean_aM_rel"] for d in debug), default=float("nan")),
        "leakage_pass_sequences": sum(1 for d in debug if d["leakage_test_pass"]),
        "leakage_total_sequences": len(debug),
    }
    result = {
        "experiment": EXPERIMENT,
        "manifest": str(manifest_path),
        "num_sequences": len(records),
        "checks": checks,
        "aggregate": aggregate,
        "debug_root_leakage": debug,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (root / "debug_root_leakage.json").write_text(json.dumps(debug, indent=2, sort_keys=True) + "\n")
    write_csv(root / "per_sequence.csv", seq_rows)
    (root / "summary.md").write_text(build_summary(result))
    print(json.dumps({
        "summary": str(root / "summary.md"),
        "metrics": str(root / "metrics.json"),
        "per_sequence": str(root / "per_sequence.csv"),
        "debug": str(root / "debug_root_leakage.json"),
        "num_sequences": len(records),
    }, indent=2))


if __name__ == "__main__":
    main()
