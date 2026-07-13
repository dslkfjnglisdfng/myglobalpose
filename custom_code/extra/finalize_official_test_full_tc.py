"""Finalize full-TC parity, paper reproduction, G2 comparisons, and summary."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch


VARIANTS = ("baseline_original", "current_g0", "g2_vr_swap")
CALIBRATIONS = ("officalib", "dipcalib")
POSE_METRICS = (
    "L SIP Err", "L Angle Err", "L Joint Err", "L Vertex Err",
    "G SIP Err", "G Angle Err", "G Joint Err", "G Vertex Err",
)
JITTER_METRICS = ("Root Jitter", "Joint Jitter")
TRANSLATION_METRICS = tuple(f"Translation {i}m" for i in range(1, 8))
PAPER = {
    "officalib": {
        "L SIP Err": 10.17, "L Angle Err": 10.16, "L Joint Err": 4.31, "L Vertex Err": 4.96,
        "G SIP Err": 10.87, "G Angle Err": 10.55, "G Joint Err": 4.31, "G Vertex Err": 5.02,
        "Root Jitter": 0.21, "Joint Jitter": 0.37, "Translation Drift 7m percent": 4.68,
    },
    "dipcalib": {
        "L SIP Err": 9.81, "L Angle Err": 9.99, "L Joint Err": 4.25, "L Vertex Err": 4.94,
        "G SIP Err": 10.24, "G Angle Err": 10.15, "G Joint Err": 4.18, "G Vertex Err": 4.87,
        "Root Jitter": 0.20, "Joint Jitter": 0.35, "Translation Drift 7m percent": 3.74,
    },
}


def load_json(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_stats(a: dict, b: dict) -> dict:
    pose_sum = tran_sum = rot_sum = 0.0
    pose_n = tran_n = rot_n = 0
    pose_max = tran_max = rot_max = 0.0
    per_sequence = []
    for i, (pose_a, pose_b, tran_a, tran_b) in enumerate(zip(a["pose"], b["pose"], a["tran"], b["tran"])):
        pose_diff, tran_diff = (pose_a - pose_b).abs(), (tran_a - tran_b).abs()
        rel = pose_a.transpose(-1, -2) @ pose_b
        rot = torch.rad2deg(torch.acos(((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
        rot = torch.where(pose_diff.amax(dim=(-1, -2)) == 0, torch.zeros_like(rot), rot)
        row = {
            "sequence_index": i,
            "pose_max_abs_diff": float(pose_diff.max()),
            "pose_mean_abs_diff": float(pose_diff.mean()),
            "pose_rotation_max_diff_deg": float(rot.max()),
            "pose_rotation_mean_diff_deg": float(rot.mean()),
            "translation_max_abs_diff_m": float(tran_diff.max()),
            "translation_mean_abs_diff_m": float(tran_diff.mean()),
        }
        per_sequence.append(row)
        pose_max, tran_max, rot_max = max(pose_max, row["pose_max_abs_diff"]), max(tran_max, row["translation_max_abs_diff_m"]), max(rot_max, row["pose_rotation_max_diff_deg"])
        pose_sum += float(pose_diff.sum()); pose_n += pose_diff.numel()
        tran_sum += float(tran_diff.sum()); tran_n += tran_diff.numel()
        rot_sum += float(rot.sum()); rot_n += rot.numel()
    return {
        "pose_max_abs_diff": pose_max, "pose_mean_abs_diff": pose_sum / pose_n,
        "pose_rotation_max_diff_deg": rot_max, "pose_rotation_mean_diff_deg": rot_sum / rot_n,
        "translation_max_abs_diff_m": tran_max, "translation_mean_abs_diff_m": tran_sum / tran_n,
        "per_sequence": per_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--previous-s5-root", type=Path, required=True)
    args = parser.parse_args()
    root, s5_root = args.output_root.resolve(), args.previous_s5_root.resolve()
    config, inventory = load_json(root / "config.json"), load_json(root / "dataset_inventory.json")
    metrics = {variant: {cal: load_json(root / variant / f"{cal}_metrics.json") for cal in CALIBRATIONS} for variant in VARIANTS}

    parity = {"thresholds": config["parity_gate"], "calibrations": {}}
    for cal in CALIBRATIONS:
        b_path = root / "baseline_original" / f"{cal}_run/predictions.pt"
        g_path = root / "current_g0" / f"{cal}_run/predictions.pt"
        bpred, gpred = torch.load(b_path, map_location="cpu"), torch.load(g_path, map_location="cpu")
        stats = prediction_stats(bpred, gpred)
        diffs = {key: abs(metrics["current_g0"][cal]["aggregate"][key]["mean"] - value["mean"]) for key, value in metrics["baseline_original"][cal]["aggregate"].items()}
        stats["official_aggregate_abs_diffs"] = diffs
        stats["official_aggregate_max_abs_diff"] = max(diffs.values())
        t = config["parity_gate"]
        stats["passed"] = stats["pose_rotation_mean_diff_deg"] < t["pose_rotation_mean_deg"] and stats["pose_rotation_max_diff_deg"] < t["pose_rotation_max_deg"] and stats["translation_max_abs_diff_m"] < t["translation_max_abs_m"] and stats["official_aggregate_max_abs_diff"] < t["official_aggregate_metric_abs"]
        parity["calibrations"][cal] = stats
    parity["passed"] = all(item["passed"] for item in parity["calibrations"].values())
    (root / "g0_baseline_full_tc_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    if not parity["passed"]:
        raise RuntimeError("full-TC G0 parity gate failed; refusing G2 conclusion")

    comparison, translation, per_sequence, sequence_stats, paper_rows = [], [], [], {}, []
    for cal in CALIBRATIONS:
        names = inventory[cal]["sequences"]
        all_metrics = POSE_METRICS + JITTER_METRICS + TRANSLATION_METRICS + ("Translation Drift 7m percent",)
        for metric in all_metrics:
            b = metrics["baseline_original"][cal]["aggregate"][metric]["mean"]
            g0 = metrics["current_g0"][cal]["aggregate"][metric]["mean"]
            g2 = metrics["g2_vr_swap"][cal]["aggregate"][metric]["mean"]
            comparison.append({
                "calibration": cal, "metric": metric, "paper_ours": PAPER[cal].get(metric),
                "baseline_original": b, "current_g0": g0, "g2_vr_swap": g2,
                "g2_minus_baseline": g2 - b, "g2_relative_change_percent": (g2 - b) / b * 100 if b else None,
            })
            if metric in PAPER[cal]:
                paper_value = PAPER[cal][metric]
                paper_rows.append({
                    "calibration": cal, "metric": metric, "paper_ours": paper_value,
                    "g0_high_precision": g0, "g0_rounded_2dp": f"{g0:.2f}",
                    "paper_rounded_2dp": f"{paper_value:.2f}", "rounded_match": round(g0, 2) == round(paper_value, 2),
                    "high_precision_delta": g0 - paper_value,
                })

        bseq, g2seq = metrics["baseline_original"][cal]["per_sequence"], metrics["g2_vr_swap"][cal]["per_sequence"]
        for metric in POSE_METRICS + JITTER_METRICS + TRANSLATION_METRICS:
            changes, named = [], []
            for row_b, row_g, name_info in zip(bseq, g2seq, names):
                b, g2 = row_b[metric], row_g[metric]
                if b is None or g2 is None:
                    continue
                delta = g2 - b
                seq_name = name_info["recovered_name"]
                changes.append(delta); named.append((delta, seq_name))
                per_sequence.append({
                    "calibration": cal, "sequence_index": name_info["index"], "source_name": name_info["source_name"],
                    "sequence": seq_name, "frames": name_info["frames"], "metric": metric,
                    "baseline_original": b, "g2_vr_swap": g2, "g2_minus_baseline": delta,
                    "g2_relative_change_percent": delta / b * 100 if b else None,
                    "outcome": "better" if delta < -1e-12 else "worse" if delta > 1e-12 else "tie",
                })
            sequence_stats[f"{cal}|{metric}"] = {
                "win_count": sum(x < -1e-12 for x in changes), "loss_count": sum(x > 1e-12 for x in changes),
                "tie_count": sum(abs(x) <= 1e-12 for x in changes), "evaluated_sequence_count": len(changes),
                "mean_delta": statistics.mean(changes), "median_delta": statistics.median(changes),
                "best_sequence": min(named)[1], "best_delta": min(named)[0],
                "worst_sequence": max(named)[1], "worst_delta": max(named)[0],
            }
        for window in range(1, 8):
            metric = f"Translation {window}m"
            row = next(item for item in comparison if item["calibration"] == cal and item["metric"] == metric)
            stats = sequence_stats[f"{cal}|{metric}"]
            translation.append({
                "calibration": cal, "window_m": window, "baseline_original": row["baseline_original"],
                "current_g0": row["current_g0"], "g2_vr_swap": row["g2_vr_swap"],
                "g2_minus_baseline": row["g2_minus_baseline"], "g2_relative_change_percent": row["g2_relative_change_percent"],
                "sequence_wins": stats["win_count"], "sequence_losses": stats["loss_count"], "sequence_ties": stats["tie_count"],
            })

    write_csv(root / "official_metric_comparison.csv", comparison)
    write_csv(root / "translation_window_comparison.csv", translation)
    write_csv(root / "per_sequence_comparison.csv", per_sequence)
    write_csv(root / "paper_reproduction_comparison.csv", paper_rows)
    (root / "per_sequence_statistics.json").write_text(json.dumps(sequence_stats, indent=2) + "\n")

    s5_metrics = {variant: load_json(s5_root / variant / "totalcapture_metrics.json") for variant in VARIANTS}
    s5_rows = []
    for metric in POSE_METRICS + JITTER_METRICS + TRANSLATION_METRICS:
        s5_b = s5_metrics["baseline_original"]["aggregate"][metric]["mean"]
        s5_g2 = s5_metrics["g2_vr_swap"]["aggregate"][metric]["mean"]
        for cal in CALIBRATIONS:
            full_b = metrics["baseline_original"][cal]["aggregate"][metric]["mean"]
            full_g2 = metrics["g2_vr_swap"][cal]["aggregate"][metric]["mean"]
            s5_rows.append({
                "calibration": cal, "metric": metric, "s5_baseline": s5_b, "s5_g2": s5_g2,
                "s5_delta": s5_g2 - s5_b, "s5_relative_percent": (s5_g2 - s5_b) / s5_b * 100,
                "full_baseline": full_b, "full_g2": full_g2, "full_delta": full_g2 - full_b,
                "full_relative_percent": (full_g2 - full_b) / full_b * 100,
            })
    write_csv(root / "s5_vs_full_tc_comparison.csv", s5_rows)

    for variant in VARIANTS:
        combined = {cal: torch.load(root / variant / f"{cal}_run/predictions.pt", map_location="cpu") for cal in CALIBRATIONS}
        torch.save(combined, root / variant / "predictions.pt")

    rounded_matches = {cal: sum(row["rounded_match"] for row in paper_rows if row["calibration"] == cal) for cal in CALIBRATIONS}
    translation_lines = []
    for cal in CALIBRATIONS:
        parts = []
        for window in range(1, 8):
            row = next(item for item in translation if item["calibration"] == cal and item["window_m"] == window)
            parts.append(f"{window}m {row['baseline_original']:.6f}->{row['g2_vr_swap']:.6f} ({row['g2_relative_change_percent']:+.2f}%, wins {row['sequence_wins']}/{inventory[cal]['sequence_count']})")
        translation_lines.append(f"- {cal}: " + "; ".join(parts))
    pose_direction = {}
    for cal in CALIBRATIONS:
        rows = [row for row in comparison if row["calibration"] == cal and row["metric"] in POSE_METRICS]
        pose_direction[cal] = (sum(row["g2_minus_baseline"] < 0 for row in rows), sum(row["g2_minus_baseline"] > 0 for row in rows))
    summary = f"""# Full TotalCapture official-test G2 audit

## Protocol and inventory

- Baseline commit: `{config['baseline_commit']}`; current source commit: `{config['current_source_commit']}`.
- Weight SHA-256 before/after: see `weights_sha256_before.txt` and `weights_sha256_after.txt`; both must equal `{config['weights_sha256_expected']}`.
- Official calibration: {inventory['officalib']['sequence_count']} sequences, {inventory['officalib']['total_frames']} frames.
- DIP calibration: {inventory['dipcalib']['sequence_count']} sequences, {inventory['dipcalib']['total_frames']} frames.
- Every formal run passed the complete release dictionary unchanged to original `test.compare_realimu`, which traverses `range(len(data['pose']))`. No s5, subject, motion-name, `--sequence`, or `max_sequences` filtering was used.
- Pose uses original `test.MotionEvaluator`; translation uses the original start/end frame-pair construction and sequence-first aggregation.

## G0 parity

Full-sequence parity passed: **{parity['passed']}**.

- Official calibration: pose max/mean `{parity['calibrations']['officalib']['pose_max_abs_diff']:.3g}` / `{parity['calibrations']['officalib']['pose_mean_abs_diff']:.3g}`, rotation max/mean `{parity['calibrations']['officalib']['pose_rotation_max_diff_deg']:.3g}` / `{parity['calibrations']['officalib']['pose_rotation_mean_diff_deg']:.3g}` deg, translation max/mean `{parity['calibrations']['officalib']['translation_max_abs_diff_m']:.3g}` / `{parity['calibrations']['officalib']['translation_mean_abs_diff_m']:.3g}` m, aggregate max diff `{parity['calibrations']['officalib']['official_aggregate_max_abs_diff']:.3g}`.
- DIP calibration: pose max/mean `{parity['calibrations']['dipcalib']['pose_max_abs_diff']:.3g}` / `{parity['calibrations']['dipcalib']['pose_mean_abs_diff']:.3g}`, rotation max/mean `{parity['calibrations']['dipcalib']['pose_rotation_max_diff_deg']:.3g}` / `{parity['calibrations']['dipcalib']['pose_rotation_mean_diff_deg']:.3g}` deg, translation max/mean `{parity['calibrations']['dipcalib']['translation_max_abs_diff_m']:.3g}` / `{parity['calibrations']['dipcalib']['translation_mean_abs_diff_m']:.3g}` m, aggregate max diff `{parity['calibrations']['dipcalib']['official_aggregate_max_abs_diff']:.3g}`.

## Paper reproduction sanity check

- Official calibration rounded matches: {rounded_matches['officalib']}/{len(PAPER['officalib'])} paper values.
- DIP calibration rounded matches: {rounded_matches['dipcalib']}/{len(PAPER['dipcalib'])} paper values.
- High-precision values, two-decimal rendering, and per-metric match flags are in `paper_reproduction_comparison.csv`. A mismatch is reported, not corrected by changing formulas.

## G2 results

- Pose directional counts among eight aggregate metrics (better/worse): official calibration {pose_direction['officalib'][0]}/{pose_direction['officalib'][1]}; DIP calibration {pose_direction['dipcalib'][0]}/{pose_direction['dipcalib'][1]}.
{chr(10).join(translation_lines)}
- Official-calibration jitter: root {metrics['baseline_original']['officalib']['aggregate']['Root Jitter']['mean']:.6f}->{metrics['g2_vr_swap']['officalib']['aggregate']['Root Jitter']['mean']:.6f}; joint {metrics['baseline_original']['officalib']['aggregate']['Joint Jitter']['mean']:.6f}->{metrics['g2_vr_swap']['officalib']['aggregate']['Joint Jitter']['mean']:.6f}.
- DIP-calibration jitter: root {metrics['baseline_original']['dipcalib']['aggregate']['Root Jitter']['mean']:.6f}->{metrics['g2_vr_swap']['dipcalib']['aggregate']['Root Jitter']['mean']:.6f}; joint {metrics['baseline_original']['dipcalib']['aggregate']['Joint Jitter']['mean']:.6f}->{metrics['g2_vr_swap']['dipcalib']['aggregate']['Joint Jitter']['mean']:.6f}.

## s5 subset versus full TotalCapture

`s5_vs_full_tc_comparison.csv` compares the previous four-sequence official-calibration subset against both complete 45-sequence caches. The full-data sequence wins/losses, mean/median deltas, and best/worst sequences are in `per_sequence_statistics.json`; these determine whether the prior s5 conclusion generalizes or reflected sampling bias.

## Conclusion boundary

G0 parity passed, so G2 can be compared directly with the official baseline. The defensible conclusion must be category-specific: pose, official 1-7 m translation, and root/joint jitter are reported separately in the tables above. This report does **not** use custom root RMSE, drift, or foot-slip diagnostics to claim official improvement, and it does not use the unconditional phrase “G2 overall outperforms GlobalPose.”
"""
    (root / "SUMMARY.md").write_text(summary)


if __name__ == "__main__":
    main()
