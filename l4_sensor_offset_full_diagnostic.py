import argparse
import json
import subprocess
from pathlib import Path

import torch

from l4_sensor_offset_diagnostic_report import summarize_payload
from l4_sensor_offset_utils import SENSOR_NAMES, sensor_to_joint_map


TC_SPLIT_PREFIXES = {
    "train": ("s1_", "s2_", "s3_"),
    "val": ("s4_",),
}


class ReportArgs:
    def __init__(self, max_offset_norm, max_condition, min_improvement):
        self.max_offset_norm = max_offset_norm
        self.max_condition = max_condition
        self.min_improvement = min_improvement


def run(cmd, cwd):
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def subset(data, indices):
    out = {}
    for key, value in data.items():
        if isinstance(value, list):
            out[key] = [value[idx] for idx in indices]
        else:
            out[key] = value
    return out


def prepare_totalcapture_dipcalib_splits(root, output_dir):
    source_path = root / "data/test_datasets/totalcapture_dipcalib.pt"
    official_path = root / "data/test_datasets/totalcapture_officalib.pt"
    source = torch.load(source_path, map_location="cpu")
    official = torch.load(official_path, map_location="cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_input": str(source_path),
        "official_name_source": str(official_path),
        "note": "totalcapture_dipcalib.pt stores numeric names; split assignment is copied by index from totalcapture_officalib.pt.",
        "splits": {},
    }
    for split, prefixes in TC_SPLIT_PREFIXES.items():
        indices = [idx for idx, name in enumerate(official["name"]) if str(name).startswith(prefixes)]
        split_data = subset(source, indices)
        split_data["name"] = [str(official["name"][idx]) + "_dipcalib" for idx in indices]
        out_path = output_dir / f"{split}.pt"
        torch.save(split_data, out_path)
        manifest["splits"][split] = {
            "path": str(out_path),
            "num_sequences": len(split_data["pose"]),
            "num_frames": int(sum(seq.shape[0] for seq in split_data["pose"])),
            "subjects": list(prefixes),
            "names": split_data["name"],
        }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def fmt_stat(stats, key="median"):
    value = stats.get(key)
    return "n/a" if value is None else f"{value:.4f}"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(path, summaries, command_log, cache_paths, report_json_path):
    rows = []
    for summary in summaries:
        rows.append(
            [
                f"{summary['dataset']} / {summary['split']}",
                summary["num_sequences"],
                fmt_stat(summary["overall"]["offset_norm"]),
                fmt_stat(summary["overall"]["residual_improvement"]),
                fmt_stat(summary["overall"]["condition_number"]),
                fmt_stat(summary["overall"]["window_consistency"]),
                f"{summary['overall']['quality_mask_fraction']:.3f}",
                len(summary["bad_entries"]),
            ]
        )

    sensor_sections = []
    for summary in summaries:
        sensor_rows = []
        for sensor_name, stats in summary["per_sensor"].items():
            sensor_rows.append(
                [
                    sensor_name,
                    fmt_stat(stats["offset_norm"], "mean"),
                    fmt_stat(stats["offset_norm"]),
                    fmt_stat(stats["offset_norm"], "std"),
                    fmt_stat(stats["offset_norm"], "p05"),
                    fmt_stat(stats["offset_norm"], "p25"),
                    fmt_stat(stats["offset_norm"], "p75"),
                    fmt_stat(stats["offset_norm"], "p95"),
                    f"{stats['offset_norm_gt_0p4_ratio']:.3f}" if stats["offset_norm_gt_0p4_ratio"] is not None else "n/a",
                    f"{stats['offset_norm_gt_0p5_ratio']:.3f}" if stats["offset_norm_gt_0p5_ratio"] is not None else "n/a",
                    fmt_stat(stats["residual_improvement"]),
                    f"{stats['residual_improvement_lt_0_ratio']:.3f}" if stats["residual_improvement_lt_0_ratio"] is not None else "n/a",
                    fmt_stat(stats["condition_number"]),
                    fmt_stat(stats["observability_score"]),
                    fmt_stat(stats["window_consistency"]),
                    stats["quality_count"],
                ]
            )
        sensor_sections.append(
            f"### {summary['dataset']} / {summary['split']}\n\n"
            + markdown_table(
                [
                    "sensor",
                    "norm mean",
                    "norm med",
                    "norm std",
                    "norm p05",
                    "norm p25",
                    "norm p75",
                    "norm p95",
                    ">0.4m",
                    ">0.5m",
                    "impr med",
                    "impr<0",
                    "cond med",
                    "obs med",
                    "win-cons med",
                    "quality",
                ],
                sensor_rows,
            )
        )

    dt_sections = []
    for summary in summaries:
        if "dt_sensitivity" not in summary:
            continue
        dt_rows = [
            [
                item["dt"],
                f"{item['median_residual_fit']:.4f}",
                f"{item['median_residual_improvement']:.4f}",
                f"{item['median_offset_norm']:.4f}",
                f"{item['median_offset_deviation_from_dt0']:.4f}",
            ]
            for item in summary["dt_sensitivity"]["by_dt"]
        ]
        dt_sections.append(
            f"### {summary['dataset']} / {summary['split']}\n\n"
            + markdown_table(
                ["dt frames", "median residual_fit", "median improvement", "median offset norm", "median |r_dt-r_0|"],
                dt_rows,
            )
            + f"\n\nBest dt distribution: `{summary['dt_sensitivity']['best_dt_distribution']}`. "
            + f"dt=0 best fraction: `{summary['dt_sensitivity']['dt0_is_best_fraction']}`."
        )

    bad_sections = []
    for summary in summaries:
        bad = summary["bad_entries"][:30]
        bad_rows = [
            [
                item["sequence"],
                item["sensor"],
                ",".join(item["reasons"]),
                item["offset_norm"],
                item["condition_number"],
                item["residual_improvement"],
            ]
            for item in bad
        ]
        bad_sections.append(
            f"### {summary['dataset']} / {summary['split']}\n\n"
            + (markdown_table(["sequence", "sensor", "reasons", "norm", "cond", "improvement"], bad_rows) if bad_rows else "No bad entries under report thresholds.")
        )

    text = f"""# L4 Sensor Offset Full Diagnostic Report

Generated: 2026-05-26

## Current Implementation Overview

This diagnostic estimates joint-local IMU position offsets `r_JS[6,3]`, where `T_JS` is used only for position here: `r_JS` is the IMU origin relative to mapped joint `J`, expressed in the joint-local frame. The forward model is `p_WS(t)=p_WJ(t)+R_WJ(t) r_JS`; the acceleration least-squares model is `a_WS_pred=ddot(p_WJ)+ddot(R_WJ) r_JS`.

Sensor map: `{sensor_to_joint_map()}`.

The estimator uses SMPL FK at 60 FPS. Real DIP/TotalCapture raw fields are converted by `official_imu_fields`: `RMB=RIM^T RIS RSB`, `aM=RIM^T RIS aS + g`, `wM=RIM^T RIS wS`, with `g=[0,-9.8,0]`. DIP preprocessing used identity `RIM/RSB`, `RIS=ori`, `aS=ori^T(acc-g)`, zero translation. TotalCapture official calibration used dataset `RIM/RSB/RIS/aS/wS/tran/pose`.

Second derivatives are central finite differences at 60 FPS after optional moving-average smoothing. This run uses the command-line smoothing/window settings recorded in each `.pt` cache metadata. Sequence offsets are coordinate-wise medians over accepted windows; quality masks are stricter post-hoc gates and are not connected to L4.

## Commands

```text
{chr(10).join(command_log)}
```

## Dataset-Level Summary

{markdown_table(["dataset/split", "seq", "offset med", "impr med", "cond med", "win-cons med", "quality frac", "bad entries"], rows)}

## Per-Sensor Offset / Residual / Observability / Consistency

{chr(10).join(sensor_sections)}

## DT Sensitivity

{chr(10).join(dt_sections)}

## Bad Sequence / Sensor Lists

Showing first 30 entries per split under report thresholds.

{chr(10).join(bad_sections)}

## Cache And Report Artifacts

Offset caches:

```text
{chr(10).join(str(p) for p in cache_paths)}
```

Report JSON: `{report_json_path}`.

## Recommendation For L4 Offset-Input Ablation

Do not connect this to L4 unless the full metrics show stable positive real-data improvement, physically plausible per-sensor norms, acceptable window consistency, non-pathological observability, and no strong dt sensitivity. Use the generated `quality_mask` per sequence/sensor as the only candidate mask if a later ablation is approved.
"""
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Run full offline L4 sensor-offset diagnostics and write markdown.")
    parser.add_argument("--output-dir", default="data/dataset_work/SensorOffset/full_diagnostic_v1")
    parser.add_argument("--window-size", type=int, default=180)
    parser.add_argument("--stride", type=int, default=90)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=0.0)
    parser.add_argument("--max-offset-norm", type=float, default=0.6)
    parser.add_argument("--min-improvement", type=float, default=-0.05)
    parser.add_argument("--quality-max-offset-norm", type=float, default=0.5)
    parser.add_argument("--quality-min-improvement", type=float, default=0.05)
    parser.add_argument("--quality-max-window-consistency", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_totalcapture_dipcalib_splits(root, out_dir / "totalcapture_dipcalib_splits")

    jobs = [
        ("data/dataset_work/AMASS/globalpose_synth_shard00000.pt", "amass", "synthetic_sanity", "amass_synthetic_sanity.pt", ["--synthetic-sanity", "--max-sequences", "3"]),
        ("data/dataset_work/DIP_IMU_globalpose/train.pt", "dip", "train", "dip_train.pt", []),
        ("data/dataset_work/DIP_IMU_globalpose/val.pt", "dip", "val", "dip_val.pt", []),
        ("data/dataset_work/TotalCapture_globalpose_official/train.pt", "totalcapture", "official_train", "totalcapture_official_train.pt", []),
        ("data/dataset_work/TotalCapture_globalpose_official/val.pt", "totalcapture", "official_val", "totalcapture_official_val.pt", []),
        (str(out_dir / "totalcapture_dipcalib_splits/train.pt"), "totalcapture", "dipcalib_train", "totalcapture_dipcalib_train.pt", []),
        (str(out_dir / "totalcapture_dipcalib_splits/val.pt"), "totalcapture", "dipcalib_val", "totalcapture_dipcalib_val.pt", []),
    ]

    cache_paths = []
    command_log = []
    for input_path, dataset, split, output_name, extra in jobs:
        output_path = out_dir / output_name
        cmd = [
            "python",
            "l4_estimate_sensor_offsets.py",
            "--input",
            input_path,
            "--dataset",
            dataset,
            "--split",
            split,
            "--output",
            str(output_path),
            "--window-size",
            str(args.window_size),
            "--stride",
            str(args.stride),
            "--smooth-window",
            str(args.smooth_window),
            "--ridge",
            str(args.ridge),
            "--huber-delta",
            str(args.huber_delta),
            "--max-offset-norm",
            str(args.max_offset_norm),
            "--min-improvement",
            str(args.min_improvement),
            "--quality-max-offset-norm",
            str(args.quality_max_offset_norm),
            "--quality-min-improvement",
            str(args.quality_min_improvement),
            "--quality-max-window-consistency",
            str(args.quality_max_window_consistency),
            "--dt-sensitivity",
            "--device",
            args.device,
        ] + extra
        command_log.append(" ".join(cmd))
        run(cmd, root)
        cache_paths.append(output_path)

    report_json = out_dir / "full_diagnostic_report.json"
    report_cmd = [
        "python",
        "l4_sensor_offset_diagnostic_report.py",
        *[str(p) for p in cache_paths],
        "--output-json",
        str(report_json),
        "--max-offset-norm",
        str(args.quality_max_offset_norm),
        "--min-improvement",
        str(args.quality_min_improvement),
    ]
    command_log.append(" ".join(report_cmd))
    run(report_cmd, root)

    report_args = ReportArgs(args.quality_max_offset_norm, 1e8, args.quality_min_improvement)
    summaries = [summarize_payload(path, report_args) for path in cache_paths]
    print("Markdown source of truth: SENSOR_OFFSET_PLAN_AND_REVIEW.md")


if __name__ == "__main__":
    main()
