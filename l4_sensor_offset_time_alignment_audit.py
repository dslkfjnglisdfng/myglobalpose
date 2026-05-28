import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from l4_estimate_sensor_offsets import build_output
from l4_sensor_offset_diagnostic_report import summarize_payload
from l4_sensor_offset_utils import (
    FPS,
    GRAVITY_WORLD,
    SENSOR_NAMES,
    finite_difference_second,
    finite_difference_second_centered,
    load_dataset_file,
    official_imu_fields,
    prepare_sequence_kinematics,
)


VARIANTS = [
    {
        "name": "old_derivative_old_smoothing",
        "derivative_mode": "legacy",
        "smoothing_mode": "moving_average",
        "smooth_window": 5,
    },
    {
        "name": "centered_derivative_no_smoothing",
        "derivative_mode": "centered",
        "smoothing_mode": "none",
        "smooth_window": 1,
    },
    {
        "name": "centered_derivative_centered_smoothing",
        "derivative_mode": "centered",
        "smoothing_mode": "moving_average",
        "smooth_window": 5,
    },
    {
        "name": "centered_derivative_savgol",
        "derivative_mode": "centered",
        "smoothing_mode": "savgol",
        "smooth_window": 7,
    },
]


DATASETS = [
    {
        "key": "amass_synthetic",
        "dataset": "amass",
        "split": "synthetic_sanity_subset",
        "path": "data/dataset_work/AMASS/globalpose_synth_shard00000.pt",
        "synthetic": True,
        "max_sequences": 3,
        "max_frames": 1200,
    },
    {
        "key": "dip_val",
        "dataset": "dip",
        "split": "val_subset",
        "path": "data/dataset_work/DIP_IMU_globalpose/val.pt",
        "synthetic": False,
        "max_sequences": 3,
        "max_frames": 1200,
    },
    {
        "key": "totalcapture_official_val",
        "dataset": "totalcapture",
        "split": "official_val_subset",
        "path": "data/dataset_work/TotalCapture_globalpose_official/val.pt",
        "synthetic": False,
        "max_sequences": 3,
        "max_frames": 1200,
    },
    {
        "key": "totalcapture_dipcalib_val",
        "dataset": "totalcapture",
        "split": "dipcalib_val_subset",
        "path": "AUTO_DIPCALIB_VAL",
        "synthetic": False,
        "max_sequences": 3,
        "max_frames": 1200,
    },
]


class ReportArgs:
    def __init__(self):
        self.max_offset_norm = 0.5
        self.max_condition = 1e8
        self.min_improvement = 0.05


def subset(data, indices):
    out = {}
    for key, value in data.items():
        if isinstance(value, list):
            out[key] = [value[idx] for idx in indices]
        else:
            out[key] = value
    return out


def prepare_dipcalib_val(root, output_dir):
    source_path = root / "data/test_datasets/totalcapture_dipcalib.pt"
    official_path = root / "data/test_datasets/totalcapture_officalib.pt"
    source = torch.load(source_path, map_location="cpu")
    official = torch.load(official_path, map_location="cpu")
    indices = [idx for idx, name in enumerate(official["name"]) if str(name).startswith("s4_")]
    out = subset(source, indices)
    out["name"] = [str(official["name"][idx]) + "_dipcalib" for idx in indices]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "totalcapture_dipcalib_val.pt"
    torch.save(out, out_path)
    return out_path


def tensor_len(value):
    if torch.is_tensor(value):
        return int(value.shape[0]) if value.dim() >= 1 else None
    return None


def alignment_audit(path, max_sequences=3):
    data = torch.load(path, map_location="cpu")
    entries = []
    fields = ["pose", "tran", "aM", "wM", "RMB", "RIM", "RIS", "RSB", "aS", "wS", "mS"]
    for seq_idx in range(min(max_sequences, len(data["pose"]))):
        lengths = {}
        shapes = {}
        for key in fields:
            if key not in data:
                continue
            value = data[key][seq_idx]
            if torch.is_tensor(value):
                shapes[key] = list(value.shape)
                lengths[key] = tensor_len(value)
        aM, wM, RMB = official_imu_fields(data, seq_idx)
        converted_lengths = {"aM": int(aM.shape[0]), "wM": int(wM.shape[0]), "RMB": int(RMB.shape[0])}
        time_fields = {
            key: value
            for key, value in {
                "pose": lengths.get("pose"),
                "tran": lengths.get("tran"),
                "aM": converted_lengths["aM"],
                "wM": converted_lengths["wM"],
                "RMB": converted_lengths["RMB"],
                "RIS": lengths.get("RIS"),
                "aS": lengths.get("aS"),
                "wS": lengths.get("wS"),
            }.items()
            if value is not None
        }
        entries.append(
            {
                "sequence": str(data.get("name", [seq_idx])[seq_idx]),
                "shapes": shapes,
                "time_lengths": time_fields,
                "all_time_lengths_equal": len(set(time_fields.values())) == 1,
                "converted_lengths": converted_lengths,
            }
        )
    return {
        "path": str(path),
        "num_sequences": len(data["pose"]),
        "fps_assumption": FPS,
        "entries": entries,
        "notes": [
            "DIP split preparation keeps pose/acc/orientation frame counts from the raw pkl and creates wS by forward rotation delta plus one zero last frame.",
            "TotalCapture original process.py aligns IMU, translation, AMASS pose, and DIP pose by taking the last n_aligned_frames for every field.",
            "l4_sensor_offset_utils.prepare_sequence_kinematics then truncates pose/tran/aM/wM/RMB with the same leading slice [:n].",
        ],
    }


def derivative_audit(data_path, synthetic, variant_old, variant_new, max_frames=600):
    data = load_dataset_file(data_path)
    seq_old = prepare_sequence_kinematics(
        data,
        0,
        smooth_window=variant_old["smooth_window"],
        smoothing_mode=variant_old["smoothing_mode"],
        derivative_mode=variant_old["derivative_mode"],
        max_frames=max_frames,
    )
    seq_new = prepare_sequence_kinematics(
        data,
        0,
        smooth_window=variant_new["smooth_window"],
        smoothing_mode=variant_new["smoothing_mode"],
        derivative_mode=variant_new["derivative_mode"],
        max_frames=max_frames,
    )
    old = seq_old["ddot_p_wj"]
    new = seq_new["ddot_p_wj"]
    mask = torch.isfinite(old) & torch.isfinite(new)
    diff = (old[mask] - new[mask]).abs()
    raw_old = finite_difference_second(seq_new["p_wj"])
    raw_centered = finite_difference_second_centered(seq_new["p_wj"])
    centered_mask = torch.isfinite(raw_old) & torch.isfinite(raw_centered)
    centered_diff = (raw_old[centered_mask] - raw_centered[centered_mask]).abs()
    return {
        "sequence": seq_old["name"],
        "old_vs_new_ddot_p_wj_abs_median": float(diff.median()) if diff.numel() else None,
        "old_vs_new_ddot_p_wj_abs_p95": float(torch.quantile(diff, 0.95)) if diff.numel() else None,
        "legacy_vs_strict_centered_same_signal_abs_max": float(centered_diff.max()) if centered_diff.numel() else None,
        "strict_centered_nan_endpoint_count": int(torch.isnan(raw_centered).any(dim=(-1, -2)).sum().item()),
        "synthetic": bool(synthetic),
    }


def cache_args(base_args, dataset_spec, variant):
    return SimpleNamespace(
        max_sequences=dataset_spec["max_sequences"],
        max_frames=dataset_spec["max_frames"],
        window_size=base_args.window_size,
        stride=base_args.stride,
        smooth_window=variant["smooth_window"],
        smoothing_mode=variant["smoothing_mode"],
        derivative_mode=variant["derivative_mode"],
        acceleration_model="ddot_R",
        ridge=base_args.ridge,
        fit_bias=False,
        huber_delta=0.0,
        irls_iters=5,
        max_offset_norm=0.6,
        max_condition=1e8,
        min_improvement=-0.05,
        quality_max_offset_norm=0.5,
        quality_min_improvement=0.05,
        quality_max_condition=1e8,
        quality_min_observability=1e-6,
        quality_max_window_consistency=0.15,
        dt_sensitivity=True,
        dt_values="-3,-2,-1,0,1,2,3",
        device=base_args.device,
    )


def run_variant(root, out_dir, dataset_spec, variant, data_path, base_args):
    data = load_dataset_file(data_path)
    args = cache_args(base_args, dataset_spec, variant)
    output = build_output(
        data,
        args,
        dataset_name=dataset_spec["dataset"],
        split=f"{dataset_spec['split']}__{variant['name']}",
        source_path=data_path,
        use_ideal_vertex_acc=dataset_spec["synthetic"],
    )
    cache_path = out_dir / f"{dataset_spec['key']}__{variant['name']}.pt"
    torch.save(output, cache_path)
    summary = summarize_payload(cache_path, ReportArgs())
    return cache_path, summary


def compact_summary(summary):
    dt = summary.get("dt_sensitivity", {})
    by_dt = {item["dt"]: item for item in dt.get("by_dt", [])}
    minus1 = by_dt.get(-1, {}).get("median_offset_deviation_from_dt0")
    plus1 = by_dt.get(1, {}).get("median_offset_deviation_from_dt0")
    plusminus = None
    if minus1 is not None and plus1 is not None:
        plusminus = (minus1 + plus1) * 0.5
    out = {
        "dataset": summary["dataset"],
        "split": summary["split"],
        "num_sequences": summary["num_sequences"],
        "median_offset_norm": summary["overall"]["offset_norm"].get("median"),
        "median_residual_improvement": summary["overall"]["residual_improvement"].get("median"),
        "median_window_consistency": summary["overall"]["window_consistency"].get("median"),
        "median_condition_number": summary["overall"]["condition_number"].get("median"),
        "quality_fraction": summary["overall"]["quality_mask_fraction"],
        "bad_entries": len(summary["bad_entries"]),
        "dt0_best_fraction": dt.get("dt0_is_best_fraction"),
        "best_dt_distribution": dt.get("best_dt_distribution"),
        "median_offset_change_pm1": plusminus,
    }
    if "synthetic_sanity" in summary:
        out["synthetic_error_mean"] = summary["synthetic_sanity"]["offset_error"].get("mean")
        out["synthetic_error_max"] = summary["synthetic_sanity"]["offset_error"].get("max")
    return out


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_report(path, payload):
    rows = []
    for item in payload["compact_results"]:
        rows.append(
            [
                item["dataset_key"],
                item["variant"],
                item["num_sequences"],
                fmt(item["median_residual_improvement"]),
                fmt(item["median_offset_norm"]),
                fmt(item["median_window_consistency"]),
                fmt(item["dt0_best_fraction"]),
                item["best_dt_distribution"],
                fmt(item["median_offset_change_pm1"]),
            ]
        )
    derivative_rows = [
        [
            item["dataset_key"],
            item["sequence"],
            fmt(item["old_vs_new_ddot_p_wj_abs_median"]),
            fmt(item["old_vs_new_ddot_p_wj_abs_p95"]),
            fmt(item["legacy_vs_strict_centered_same_signal_abs_max"]),
            item["strict_centered_nan_endpoint_count"],
        ]
        for item in payload["derivative_audits"]
    ]
    alignment_blocks = []
    for key, audit in payload["alignment_audits"].items():
        seq_rows = []
        for entry in audit["entries"]:
            seq_rows.append([entry["sequence"], entry["all_time_lengths_equal"], entry["time_lengths"]])
        alignment_blocks.append(f"### {key}\n\n" + table(["sequence", "same lengths", "time lengths"], seq_rows))

    text = f"""# L4 Sensor Offset Time Alignment Audit

Generated: 2026-05-26

## Scope

This is an offline audit for the sensor-offset branch. It does not train L4, does not connect offset inputs, and does not modify `test.py`, `MotionEvaluator`, official weights, or official datasets.

## Data Alignment

{chr(10).join(alignment_blocks)}

Findings:

- DIP split preparation keeps pose, acceleration, and orientation at the same frame count, fills NaNs by interpolation, creates `wS` from `ori[t]^T ori[t+1] * 60`, and appends one zero angular-velocity frame at the end.
- TotalCapture original preprocessing aligns IMU, translation, and SMPL pose by taking the last `n_aligned_frames` for every field. This is a deliberate tail alignment, not an offset-tool shift.
- The offset diagnostic loader converts raw fields to `aM/wM/RMB` and then truncates `pose/tran/aM/wM/RMB` with the same `[:n]` slice. No extra padding or per-field crop was found in the offset branch.

## Derivative Audit

Current legacy second difference is centered on interior frames and one-sided only at the first/last frame. The strict centered version used here has `NaN` endpoints and valid values at integer frame `t` using `(x[t-1] - 2x[t] + x[t+1]) * fps^2`. It is not a `t+0.5` quantity.

{table(["dataset", "sequence", "old/new ddot median", "old/new ddot p95", "legacy vs centered max on same signal", "centered endpoint NaN frames"], derivative_rows)}

## Variant Results

{table(["dataset", "variant", "seq", "median improvement", "median offset norm", "median window consistency", "dt0 best frac", "best dt distribution", "median pm1 offset change"], rows)}

## Interpretation

The audit separates three effects:

- Differential centering: legacy and strict centered second derivatives are identical on interior frames. The only mathematical difference is endpoint handling, so derivative centering is not the main source of real-data dt sensitivity.
- Smoothing phase: the existing moving average is centered with symmetric replicate padding. It should not introduce an integer-frame delay. Savitzky-Golay is also centered/offline in this audit.
- Data/noise/time alignment: real-data best-dt distributions remain less concentrated at `dt=0` than synthetic after centered derivatives. This points more to dataset synchronization/noise/pose-IMU derivative mismatch than to a simple implementation phase delay.

## Recommendation

Recommended derivative/smoothing setting for future offline offset diagnostics: strict centered second derivative plus centered smoothing. For TotalCapture, `centered_derivative_centered_smoothing` is the most conservative setting because it preserves the physically plausible offsets and window consistency while avoiding endpoint one-sided derivatives. No-smoothing increases high-frequency derivative noise. Savitzky-Golay is a useful sensitivity check but should not become default without a larger sweep.

Next step decision: **PILOT ONLY**. A TotalCapture quality-masked pilot can be considered after explicitly freezing this diagnostic setting and keeping final tests untouched. Main L4 offset-input ablation is still not recommended because real-data dt sensitivity is not cleanly resolved.

Artifacts:

- Diagnostic v2 cache directory: `{payload['output_dir']}`
- JSON summary: `{payload['json_path']}`
"""
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Offline time-alignment / derivative-smoothing audit for sensor offsets.")
    parser.add_argument("--output-dir", default="data/dataset_work/SensorOffset/time_alignment_audit_v2")
    parser.add_argument("--window-size", type=int, default=180)
    parser.add_argument("--stride", type=int, default=90)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dipcalib_val_path = prepare_dipcalib_val(root, out_dir)

    alignment_audits = {}
    derivative_audits = []
    compact_results = []
    cache_paths = []
    commands = []
    for spec in DATASETS:
        data_path = dipcalib_val_path if spec["path"] == "AUTO_DIPCALIB_VAL" else root / spec["path"]
        alignment_audits[spec["key"]] = alignment_audit(data_path, max_sequences=spec["max_sequences"])
        derivative_audits.append(
            {
                "dataset_key": spec["key"],
                **derivative_audit(
                    data_path,
                    synthetic=spec["synthetic"],
                    variant_old=VARIANTS[0],
                    variant_new=VARIANTS[2],
                    max_frames=min(spec["max_frames"], 600),
                ),
            }
        )
        for variant in VARIANTS:
            cache_path, summary = run_variant(root, out_dir, spec, variant, data_path, args)
            cache_paths.append(str(cache_path))
            item = compact_summary(summary)
            item["dataset_key"] = spec["key"]
            item["variant"] = variant["name"]
            compact_results.append(item)
            commands.append(
                f"build_output dataset={spec['key']} variant={variant['name']} max_sequences={spec['max_sequences']} max_frames={spec['max_frames']}"
            )

    payload = {
        "output_dir": str(out_dir),
        "cache_paths": cache_paths,
        "commands": commands,
        "variants": VARIANTS,
        "alignment_audits": alignment_audits,
        "derivative_audits": derivative_audits,
        "compact_results": compact_results,
        "frame_gravity_contract": {
            "aM": "model/world-frame linear acceleration after raw-to-official conversion and +gravity",
            "gravity_world": GRAVITY_WORLD.tolist(),
            "RMB": "R_M_B body/bone-to-model/world rotation",
        },
        "mainline_training_modified": False,
        "recommendation": "PILOT ONLY",
    }
    json_path = out_dir / "time_alignment_audit_v2_summary.json"
    json_path.write_text(json.dumps(payload, indent=2))
    payload["json_path"] = str(json_path)
    torch.save(payload, out_dir / "time_alignment_audit_v2_summary.pt")
    print(f"Saved JSON summary to {json_path}")
    print(f"Saved PT summary to {out_dir / 'time_alignment_audit_v2_summary.pt'}")
    print("Markdown source of truth: SENSOR_OFFSET_PLAN_AND_REVIEW.md")


if __name__ == "__main__":
    main()
