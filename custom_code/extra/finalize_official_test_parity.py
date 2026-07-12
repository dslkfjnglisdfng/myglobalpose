"""Build parity, official comparison, per-sequence statistics, and samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


VARIANTS = ("baseline_original", "current_g0", "g2_vr_swap")
POSE_METRICS = (
    "L SIP Err", "L Angle Err", "L Joint Err", "L Vertex Err",
    "G SIP Err", "G Angle Err", "G Joint Err", "G Vertex Err",
)
JITTER_METRICS = ("Root Jitter", "Joint Jitter")


def load_json(path):
    return json.loads(path.read_text())


def prediction_stats(a, b):
    pose_diffs = [(x - y).abs() for x, y in zip(a["pose"], b["pose"])]
    tran_diffs = [(x - y).abs() for x, y in zip(a["tran"], b["tran"])]
    rotation_diffs = []
    for x, y, element_diff in zip(a["pose"], b["pose"], pose_diffs):
        rel = x.transpose(-1, -2) @ y
        angle = torch.rad2deg(torch.acos(((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
        angle = torch.where(element_diff.amax(dim=(-1, -2)) == 0, torch.zeros_like(angle), angle)
        rotation_diffs.append(angle)
    return {
        "pose_max_abs_diff": max(x.max().item() for x in pose_diffs),
        "pose_mean_abs_diff": sum(x.sum().item() for x in pose_diffs) / sum(x.numel() for x in pose_diffs),
        "pose_rotation_max_diff_deg": max(x.max().item() for x in rotation_diffs),
        "pose_rotation_mean_diff_deg": sum(x.sum().item() for x in rotation_diffs) / sum(x.numel() for x in rotation_diffs),
        "tran_max_abs_diff_m": max(x.max().item() for x in tran_diffs),
        "tran_mean_abs_diff_m": sum(x.sum().item() for x in tran_diffs) / sum(x.numel() for x in tran_diffs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()

    metrics = {}
    for variant in VARIANTS:
        metrics[variant] = {}
        sample = {}
        for dataset in ("dip", "totalcapture"):
            item = load_json(root / variant / f"{dataset}_metrics.json")
            metrics[variant][dataset] = item
            pred = torch.load(root / variant / f"{dataset}_run" / "predictions.pt", map_location="cpu")
            sample[dataset] = {
                "sequence": item["sequences"][0],
                "pose_prediction": pred["pose"][0],
                "tran_prediction": pred["tran"][0],
            }
        (root / variant / "metrics.json").write_text(json.dumps(metrics[variant], indent=2) + "\n")
        torch.save(sample, root / variant / "predictions_sample.pt")

    parity = {"thresholds": load_json(root / "config.json")["parity_gate"], "datasets": {}}
    for dataset in ("dip", "totalcapture"):
        bpred = torch.load(root / "baseline_original" / f"{dataset}_run" / "predictions.pt", map_location="cpu")
        gpred = torch.load(root / "current_g0" / f"{dataset}_run" / "predictions.pt", map_location="cpu")
        pred_stats = prediction_stats(bpred, gpred)
        aggregate_diffs = {
            key: abs(metrics["current_g0"][dataset]["aggregate"][key]["mean"] - value["mean"])
            for key, value in metrics["baseline_original"][dataset]["aggregate"].items()
        }
        pred_stats["official_aggregate_abs_diffs"] = aggregate_diffs
        pred_stats["official_aggregate_max_abs_diff"] = max(aggregate_diffs.values())
        pred_stats["passed"] = (
            pred_stats["pose_rotation_mean_diff_deg"] < parity["thresholds"]["pose_rotation_mean_deg"]
            and pred_stats["pose_rotation_max_diff_deg"] < parity["thresholds"]["pose_rotation_max_deg"]
            and pred_stats["tran_max_abs_diff_m"] < parity["thresholds"]["translation_max_abs_m"]
            and pred_stats["official_aggregate_max_abs_diff"] < parity["thresholds"]["official_aggregate_metric_abs"]
        )
        parity["datasets"][dataset] = pred_stats
    parity["passed"] = all(row["passed"] for row in parity["datasets"].values())
    (root / "g0_baseline_parity.json").write_text(json.dumps(parity, indent=2) + "\n")

    comparison_rows = []
    sequence_rows = []
    sequence_stats = {}
    for dataset in ("dip", "totalcapture"):
        metric_names = list(POSE_METRICS + JITTER_METRICS)
        if dataset == "totalcapture":
            metric_names += [f"Translation {i}m" for i in range(1, 8)]
        for metric in metric_names:
            b = metrics["baseline_original"][dataset]["aggregate"][metric]["mean"]
            g0 = metrics["current_g0"][dataset]["aggregate"][metric]["mean"]
            g2 = metrics["g2_vr_swap"][dataset]["aggregate"][metric]["mean"]
            comparison_rows.append({
                "dataset": dataset, "metric": metric, "baseline_original": b,
                "current_g0": g0, "g2_vr_swap": g2, "g2_minus_baseline": g2 - b,
                "g2_relative_change_percent": (g2 - b) / b * 100 if b else None,
            })
            changes = []
            named = []
            bseq = metrics["baseline_original"][dataset]["per_sequence"]
            g2seq = metrics["g2_vr_swap"][dataset]["per_sequence"]
            for rb, rg in zip(bseq, g2seq):
                change = rg[metric] - rb[metric]
                changes.append(change)
                named.append((change, rb["sequence"]))
                sequence_rows.append({
                    "dataset": dataset, "sequence": rb["sequence"], "metric": metric,
                    "baseline_original": rb[metric], "g2_vr_swap": rg[metric],
                    "g2_minus_baseline": change,
                    "g2_relative_change_percent": change / rb[metric] * 100 if rb[metric] else None,
                    "outcome": "better" if change < -1e-12 else "worse" if change > 1e-12 else "tie",
                })
            ordered = sorted(changes)
            median = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered)//2-1] + ordered[len(ordered)//2]) / 2
            best = min(named)
            worst = max(named)
            sequence_stats[f"{dataset}|{metric}"] = {
                "better": sum(x < -1e-12 for x in changes),
                "worse": sum(x > 1e-12 for x in changes),
                "tie": sum(abs(x) <= 1e-12 for x in changes),
                "mean_change": sum(changes) / len(changes), "median_change": median,
                "best_sequence": best[1], "best_change": best[0],
                "worst_sequence": worst[1], "worst_change": worst[0],
            }

    for filename, rows in (("official_metric_comparison.csv", comparison_rows), ("per_sequence_comparison.csv", sequence_rows)):
        with (root / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "per_sequence_statistics.json").write_text(json.dumps(sequence_stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
