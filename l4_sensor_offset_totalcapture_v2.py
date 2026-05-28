import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

from l4_estimate_sensor_offsets import build_output
from l4_sensor_offset_diagnostic_report import summarize_payload
from l4_sensor_offset_full_diagnostic import prepare_totalcapture_dipcalib_splits
from l4_sensor_offset_utils import (
    FPS,
    GRAVITY_WORLD,
    SENSOR_NAMES,
    load_dataset_file,
    official_imu_fields,
    prepare_sequence_kinematics,
    sensor_to_joint_map,
)


TOTALCAPTURE_SPLITS = [
    {
        "key": "official_train",
        "dataset": "totalcapture",
        "split": "official_train",
        "path": "data/dataset_work/TotalCapture_globalpose_official/train.pt",
        "source_kind": "official calibration",
    },
    {
        "key": "official_val",
        "dataset": "totalcapture",
        "split": "official_val",
        "path": "data/dataset_work/TotalCapture_globalpose_official/val.pt",
        "source_kind": "official calibration",
    },
    {
        "key": "dipcalib_train",
        "dataset": "totalcapture",
        "split": "dipcalib_train",
        "path": "AUTO_DIPCALIB_TRAIN",
        "source_kind": "DIP calibration",
    },
    {
        "key": "dipcalib_val",
        "dataset": "totalcapture",
        "split": "dipcalib_val",
        "path": "AUTO_DIPCALIB_VAL",
        "source_kind": "DIP calibration",
    },
]


CONFIGS = [
    {
        "name": "A_old_smooth5",
        "derivative_mode": "legacy",
        "smoothing_mode": "moving_average",
        "smooth_window": 5,
        "acceleration_model": "ddot_R",
    },
    {
        "name": "A_centered_none",
        "derivative_mode": "centered",
        "smoothing_mode": "none",
        "smooth_window": 1,
        "acceleration_model": "ddot_R",
    },
    {
        "name": "A_centered_smooth5",
        "derivative_mode": "centered",
        "smoothing_mode": "moving_average",
        "smooth_window": 5,
        "acceleration_model": "ddot_R",
    },
    {
        "name": "A_centered_savgol7",
        "derivative_mode": "centered",
        "smoothing_mode": "savgol",
        "smooth_window": 7,
        "acceleration_model": "ddot_R",
    },
    {
        "name": "B_centered_smooth5",
        "derivative_mode": "centered",
        "smoothing_mode": "moving_average",
        "smooth_window": 5,
        "acceleration_model": "alpha_omega",
    },
    {
        "name": "B_centered_savgol7",
        "derivative_mode": "centered",
        "smoothing_mode": "savgol",
        "smooth_window": 7,
        "acceleration_model": "alpha_omega",
    },
]


class ReportArgs:
    def __init__(self):
        self.max_offset_norm = 0.6
        self.max_condition = 1e8
        self.min_improvement = -0.05


def subset(data, indices):
    out = {}
    for key, value in data.items():
        out[key] = [value[idx] for idx in indices] if isinstance(value, list) else value
    return out


def tc_args(base_args, config, max_sequences=0):
    return SimpleNamespace(
        max_sequences=max_sequences,
        max_frames=0,
        window_size=base_args.window_size,
        stride=base_args.stride,
        smooth_window=config["smooth_window"],
        smoothing_mode=config["smoothing_mode"],
        derivative_mode=config["derivative_mode"],
        acceleration_model=config["acceleration_model"],
        ridge=base_args.ridge,
        fit_bias=False,
        huber_delta=0.0,
        irls_iters=5,
        max_offset_norm=base_args.max_offset_norm,
        max_condition=base_args.max_condition,
        min_improvement=base_args.min_improvement,
        quality_max_offset_norm=base_args.quality_max_offset_norm,
        quality_min_improvement=base_args.quality_min_improvement,
        quality_max_condition=base_args.quality_max_condition,
        quality_min_observability=base_args.quality_min_observability,
        quality_max_window_consistency=base_args.quality_max_window_consistency,
        dt_sensitivity=True,
        dt_values="-3,-2,-1,0,1,2,3",
        device=base_args.device,
    )


def stats_tensor(x):
    x = x.detach().cpu().float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"count": 0}
    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "median": float(torch.quantile(x, 0.5)),
        "p25": float(torch.quantile(x, 0.25)),
        "p75": float(torch.quantile(x, 0.75)),
        "max": float(x.max()),
    }


def alignment_audit(path, label):
    data = torch.load(path, map_location="cpu")
    fields = ["pose", "tran", "RIM", "RIS", "RSB", "aS", "wS", "mS", "aM", "wM", "RMB"]
    entries = []
    mismatches = []
    for seq_idx in range(len(data["pose"])):
        lengths = {}
        shapes = {}
        for key in fields:
            if key not in data:
                continue
            value = data[key][seq_idx]
            if torch.is_tensor(value):
                shapes[key] = list(value.shape)
                if value.dim() > 0:
                    lengths[key] = int(value.shape[0])
        aM, wM, RMB = official_imu_fields(data, seq_idx)
        lengths["converted_aM"] = int(aM.shape[0])
        lengths["converted_wM"] = int(wM.shape[0])
        lengths["converted_RMB"] = int(RMB.shape[0])
        time_lengths = {
            key: lengths[key]
            for key in ("pose", "tran", "aS", "wS", "mS", "RIS", "converted_aM", "converted_wM", "converted_RMB")
            if key in lengths
        }
        same = len(set(time_lengths.values())) == 1
        if not same:
            mismatches.append(str(data.get("name", [seq_idx])[seq_idx]))
        entries.append(
            {
                "sequence": str(data.get("name", [seq_idx])[seq_idx]),
                "time_lengths": time_lengths,
                "all_time_lengths_equal": same,
                "shapes": shapes,
            }
        )
    return {
        "label": label,
        "path": str(path),
        "num_sequences": len(data["pose"]),
        "fps_assumption": FPS,
        "all_sequences_equal": len(mismatches) == 0,
        "mismatched_sequences": mismatches,
        "entries": entries,
    }


def frame_gravity_audit(path, max_sequences=5):
    data = torch.load(path, map_location="cpu")
    seq_summaries = []
    for seq_idx in range(min(max_sequences, len(data["pose"]))):
        aM, wM, RMB = official_imu_fields(data, seq_idx)
        raw_aM_without_gravity = None
        if all(key in data for key in ("RIM", "RIS", "aS")):
            RIM = data["RIM"][seq_idx].float()
            RIS = data["RIS"][seq_idx].float()
            aS = data["aS"][seq_idx].float()
            raw_aM_without_gravity = RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1)
        seq_summaries.append(
            {
                "sequence": str(data.get("name", [seq_idx])[seq_idx]),
                "aM_mean": aM.mean(dim=(0, 1)).tolist(),
                "aM_norm": stats_tensor(aM.norm(dim=-1)),
                "wM_norm": stats_tensor(wM.norm(dim=-1)),
                "RMB_det_mean": float(torch.linalg.det(RMB).mean()),
                "raw_aM_without_gravity_mean": raw_aM_without_gravity.mean(dim=(0, 1)).tolist()
                if raw_aM_without_gravity is not None
                else None,
            }
        )
    return {
        "gravity_world": GRAVITY_WORLD.tolist(),
        "conversion": "RMB=RIM^T RIS RSB, aM=RIM^T RIS aS + g, wM=RIM^T RIS wS",
        "aM_contract": "model/world-frame linear acceleration including gravity, consistent with FK second derivative for world positions after +g conversion",
        "RMB_contract": "R_M_B, body/bone-to-model/world rotation as used by GlobalPose forward_frame",
        "sample_sequences": seq_summaries,
    }


def run_config(root, out_dir, split_spec, config, base_args, max_sequences=0):
    data_path = root / split_spec["path"]
    data = load_dataset_file(data_path)
    args = tc_args(base_args, config, max_sequences=max_sequences)
    payload = build_output(
        data,
        args,
        dataset_name=split_spec["dataset"],
        split=f"{split_spec['split']}__{config['name']}",
        source_path=data_path,
        use_ideal_vertex_acc=False,
    )
    cache_path = out_dir / "comparison" / f"{split_spec['key']}__{config['name']}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    summary = summarize_payload(cache_path, ReportArgs())
    return cache_path, summary


def run_synthetic(root, out_dir, config, base_args):
    data_path = root / "data/dataset_work/AMASS/globalpose_synth_shard00000.pt"
    data = load_dataset_file(data_path)
    args = tc_args(base_args, config, max_sequences=3)
    args.max_frames = 1200
    payload = build_output(
        data,
        args,
        dataset_name="amass",
        split=f"synthetic_sanity__{config['name']}",
        source_path=data_path,
        use_ideal_vertex_acc=True,
    )
    cache_path = out_dir / "comparison" / f"synthetic__{config['name']}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return cache_path, summarize_payload(cache_path, ReportArgs())


def compact(summary):
    dt = summary.get("dt_sensitivity", {})
    by_dt = {item["dt"]: item for item in dt.get("by_dt", [])}
    pm1 = None
    if -1 in by_dt and 1 in by_dt:
        pm1 = 0.5 * (
            by_dt[-1]["median_offset_deviation_from_dt0"] + by_dt[1]["median_offset_deviation_from_dt0"]
        )
    out = {
        "dataset": summary["dataset"],
        "split": summary["split"],
        "num_sequences": summary["num_sequences"],
        "median_offset_norm": summary["overall"]["offset_norm"].get("median"),
        "median_residual_improvement": summary["overall"]["residual_improvement"].get("median"),
        "median_condition_number": summary["overall"]["condition_number"].get("median"),
        "median_window_consistency": summary["overall"]["window_consistency"].get("median"),
        "quality_fraction": summary["overall"]["quality_mask_fraction"],
        "bad_entries": len(summary["bad_entries"]),
        "dt0_best_fraction": dt.get("dt0_is_best_fraction"),
        "best_dt_distribution": dt.get("best_dt_distribution"),
        "median_offset_change_pm1": pm1,
        "per_sensor": {
            name: {
                "median_improvement": stats["residual_improvement"].get("median"),
                "median_window_consistency": stats["window_consistency"].get("median"),
                "bad_ratio": stats["outlier_count"] / max(1, summary["num_sequences"]),
            }
            for name, stats in summary["per_sensor"].items()
        },
    }
    if "synthetic_sanity" in summary:
        out["synthetic_error_mean"] = summary["synthetic_sanity"]["offset_error"].get("mean")
        out["synthetic_error_max"] = summary["synthetic_sanity"]["offset_error"].get("max")
    return out


def selected_config_name(compact_results):
    official = [
        item
        for item in compact_results
        if item["config"] in ("A_centered_smooth5", "B_centered_smooth5", "A_centered_savgol7", "B_centered_savgol7")
        and item["split_key"] in ("official_train", "official_val")
    ]
    grouped = {}
    for item in official:
        grouped.setdefault(item["config"], []).append(item)
    scores = []
    for name, items in grouped.items():
        if len(items) < 2:
            continue
        min_quality = min(item["quality_fraction"] for item in items)
        med_impr = sum(item["median_residual_improvement"] for item in items) / len(items)
        med_win = sum(item["median_window_consistency"] for item in items) / len(items)
        dt0 = sum(item["dt0_best_fraction"] for item in items) / len(items)
        # Keep residual and stability primary; dt0 is diagnostic but not optimized alone.
        score = med_impr - 0.5 * med_win + 0.05 * dt0 + 0.05 * min_quality
        scores.append((score, name))
    return sorted(scores, reverse=True)[0][1] if scores else "A_centered_smooth5"


def fixed_dt_policy_summary(compact_results, selected):
    rows = []
    for item in compact_results:
        if item["config"] != selected or item["split_key"] not in {"official_train", "official_val", "dipcalib_train", "dipcalib_val"}:
            continue
        dist = {int(k): int(v) for k, v in item["best_dt_distribution"].items()}
        total = sum(dist.values())
        best_dt = max(dist, key=dist.get) if dist else 0
        rows.append(
            {
                "split_key": item["split_key"],
                "sequence_sensor_entries": total,
                "mode_best_dt": int(best_dt),
                "mode_fraction": float(dist[best_dt] / total) if total else None,
                "dt0_fraction": item["dt0_best_fraction"],
                "best_dt_distribution": dist,
                "median_offset_change_pm1": item["median_offset_change_pm1"],
                "recommended_dt_policy": "dt0"
                if item["median_offset_change_pm1"] is not None and item["median_offset_change_pm1"] < 0.02
                else "diagnostic_only_no_forced_correction",
            }
        )
    return rows


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def write_audit_report(path, payload):
    alignment_rows = []
    for key, audit in payload["alignment_audit"].items():
        alignment_rows.append(
            [
                key,
                audit["num_sequences"],
                audit["all_sequences_equal"],
                ",".join(audit["mismatched_sequences"][:10]) if audit["mismatched_sequences"] else "none",
            ]
        )
    text = f"""# L4 Sensor Offset TotalCapture Audit

Generated: 2026-05-26

## Scope

This report is TotalCapture-only. DIP is kept only as historical diagnostic context and is not used for this round's method choice, tuning, or recommendation.

## Data Chain

{table(["split", "seq", "same pose/tran/IMU lengths", "mismatches"], alignment_rows)}

TotalCapture preprocessing in `process.py` aligns IMU, root translation, AMASS pose, and DIP pose by taking the last `n_aligned_frames` for every field. The split files preserve this sequence data. The offset loader then converts raw fields to `aM/wM/RMB` and applies the same leading `[:n]` crop to `pose/tran/aM/wM/RMB`; this branch does not add hidden per-field shift or padding.

## Frame And Gravity Contract

- Sensor map: `{sensor_to_joint_map()}`.
- Raw-to-model conversion: `{payload['frame_gravity_audit']['conversion']}`.
- `aM`: {payload['frame_gravity_audit']['aM_contract']}.
- `RMB`: {payload['frame_gravity_audit']['RMB_contract']}.
- Gravity: `{payload['frame_gravity_audit']['gravity_world']}`.

Conclusion: TotalCapture `aM` can be compared to FK-derived world acceleration only after the existing `+g` conversion. It should be treated as world/model-frame linear acceleration including gravity under the same convention used by GlobalPose.

## Official vs DIP-Calibration

Both variants come from the same TotalCapture source sequence order. Official train/val uses `totalcapture_officalib.pt`; DIP-calibration uses `totalcapture_dipcalib.pt` but copies split membership and names by index from the official file. Keep both as separate diagnostic variants because the calibration differs and the dt behavior is not identical. Do not merge their statistics.

## DT Sensitivity Interpretation

The low `dt=0` best fraction is unlikely to be caused by an offset-branch crop/padding bug or by a one-frame smoothing delay. Current evidence points to a mixture of dataset synchronization residuals, IMU/pose calibration mismatch, action-dependent excitation, and acceleration derivative noise/model mismatch.
"""
    path.write_text(text)


def write_v2_report(path, payload):
    rows = []
    for item in payload["compact_results"]:
        rows.append(
            [
                item["split_key"],
                item["config"],
                item["num_sequences"],
                fmt(item["median_residual_improvement"]),
                fmt(item["median_offset_norm"]),
                fmt(item["median_condition_number"]),
                fmt(item["median_window_consistency"]),
                fmt(item["quality_fraction"]),
                item["bad_entries"],
                fmt(item["dt0_best_fraction"]),
                item["best_dt_distribution"],
                fmt(item["median_offset_change_pm1"]),
            ]
        )
    synth_rows = [
        [
            item["config"],
            fmt(item.get("synthetic_error_mean")),
            fmt(item.get("synthetic_error_max")),
            fmt(item.get("median_residual_improvement")),
        ]
        for item in payload["synthetic_results"]
    ]
    policy_rows = [
        [
            item["split_key"],
            item["mode_best_dt"],
            fmt(item["mode_fraction"]),
            fmt(item["dt0_fraction"]),
            item["best_dt_distribution"],
            fmt(item["median_offset_change_pm1"]),
            item["recommended_dt_policy"],
        ]
        for item in payload["time_alignment_policy"]
    ]
    selected_rows = [
        [
            item["split_key"],
            fmt(item["median_residual_improvement"]),
            fmt(item["median_offset_norm"]),
            fmt(item["median_window_consistency"]),
            fmt(item["quality_fraction"]),
            fmt(item["dt0_best_fraction"]),
            item["best_dt_distribution"],
            fmt(item["median_offset_change_pm1"]),
        ]
        for item in payload["compact_results"]
        if item["config"] == payload["selected_config"]
    ]
    sensor_blocks = []
    for item in payload["compact_results"]:
        if item["config"] != payload["selected_config"]:
            continue
        sensor_rows = []
        for sensor_name in SENSOR_NAMES:
            stats = item["per_sensor"][sensor_name]
            sensor_rows.append(
                [
                    sensor_name,
                    fmt(stats["median_improvement"]),
                    fmt(stats["median_window_consistency"]),
                    fmt(stats["bad_ratio"]),
                ]
            )
        sensor_blocks.append(
            f"### {item['split_key']}\n\n"
            + table(["sensor", "median residual improvement", "median window consistency", "bad ratio"], sensor_rows)
        )
    text = f"""# L4 Sensor Offset TotalCapture V2 Report

Generated: 2026-05-26

## Scope

This is a TotalCapture-only offset extraction v2. DIP is intentionally excluded from method selection because v1 showed TotalCapture has stronger residual improvement, more stable quality masks, and a different dt-sensitivity profile.

## V1 TotalCapture Baseline

- Official train: median norm about 0.181 m, median residual improvement about 0.408, median condition about 1.769, median window consistency about 0.034 m, quality about 0.986, dt0 best about 0.435.
- Official val: median norm about 0.201 m, median residual improvement about 0.351, median condition about 1.606, median window consistency about 0.052 m, quality about 0.933, dt0 best about 0.467.
- DIP-calibration train/val: residual improvement about 0.34-0.43 with high quality fraction, but low dt0 concentration.

## Synthetic Sanity

{table(["config", "mean offset error", "max offset error", "median improvement"], synth_rows)}

## TotalCapture Config Comparison

{table(["split", "config", "seq", "impr med", "norm med", "cond med", "win-cons med", "quality", "bad", "dt0 frac", "best dt dist", "pm1 offset change"], rows)}

## Time-Alignment Aware LS

{table(["split", "mode best dt", "mode fraction", "dt0 fraction", "best dt distribution", "pm1 offset change", "policy"], policy_rows)}

Per-window free dt is intentionally not used. The diagnostic compares fixed dt=0, split-level mode dt, and sequence/sensor best-dt distributions from the same cache. The selected v2 cache keeps `dt=0` as the offset estimate and stores dt sensitivity summaries because the best-dt distributions are not clean enough for a forced correction across all TotalCapture splits.

## Acceleration Model A vs B

Model A uses `ddot(R_WJ) r_JS`. Model B uses `alpha_W x (R_WJ r_JS) + omega_W x (omega_W x (R_WJ r_JS))`, implemented as `(skew(alpha_W) R_WJ + skew(omega_W)^2 R_WJ) r_JS`. They are theoretically equivalent for a smooth rotation trajectory. Numerical differences come from differentiating rotation matrices twice versus differentiating angular velocity/acceleration, smoothing interactions, and any mismatch between SMPL FK rotations and IMU-derived angular velocity conventions.

## Selected V2 Strategy

Selected config: `{payload['selected_config']}`.

{table(["split", "impr med", "norm med", "win-cons med", "quality", "dt0 frac", "best dt dist", "pm1 offset change"], selected_rows)}

## Selected Per-Sensor Metrics

{chr(10).join(sensor_blocks)}

Cache path: `{payload['selected_manifest']}`.

## Recommendation

Next recommendation: **{payload['recommendation']}**.

Rationale: TotalCapture offsets remain physically plausible and quality-mask fractions remain high, but dt sensitivity is still not clean enough for a mainline L4 ablation. A TotalCapture quality-masked pilot is acceptable only with the selected v2 cache, unchanged final/test split, and clear reporting that the offset input is TotalCapture-only diagnostic/pilot evidence.
"""
    path.write_text(text)


def write_iteration_log(path, payload):
    text = f"""# L4 Sensor Offset Iteration Log

## 2026-05-26 TotalCapture-Only V2

- Scope narrowed to TotalCapture only. DIP is no longer used for method selection or tuning.
- Added `acceleration_model=alpha_omega` diagnostic path while preserving the original `ddot_R` model.
- Ran TotalCapture official train/val and DIP-calibration train/val comparison under derivative/smoothing/model variants.
- Generated selected TotalCapture-only v2 cache under `{payload['output_dir']}`.
- Mainline training code changed: NO. No L4 training was started, and `test.py`, `MotionEvaluator`, official weights, and official datasets were not modified.
- Recommendation: {payload['recommendation']}.
"""
    path.write_text(text)


def copy_selected_caches(root, out_dir, selected, cache_map):
    selected_dir = out_dir / "selected_cache"
    selected_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selected_config": selected,
        "sensor_to_joint": sensor_to_joint_map(),
        "frame_gravity_notes": {
            "aM": "model/world-frame acceleration including gravity after RIM^T RIS aS + g",
            "RMB": "R_M_B body/bone-to-model/world rotation",
            "gravity_world": GRAVITY_WORLD.tolist(),
        },
        "split_caches": {},
    }
    for split_key in ("official_train", "official_val", "dipcalib_train", "dipcalib_val"):
        source = cache_map[(split_key, selected)]
        dest = selected_dir / f"{split_key}__{selected}__offset_cache_v2.pt"
        shutil.copy2(source, dest)
        manifest["split_caches"][split_key] = str(dest)
    manifest_path = selected_dir / "totalcapture_offset_cache_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="TotalCapture-only sensor offset extraction v2.")
    parser.add_argument("--output-dir", default="data/dataset_work/SensorOffset/totalcapture_only_v2")
    parser.add_argument("--window-size", type=int, default=180)
    parser.add_argument("--stride", type=int, default=90)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--max-offset-norm", type=float, default=0.6)
    parser.add_argument("--max-condition", type=float, default=1e8)
    parser.add_argument("--min-improvement", type=float, default=-0.05)
    parser.add_argument("--quality-max-offset-norm", type=float, default=0.5)
    parser.add_argument("--quality-min-improvement", type=float, default=0.05)
    parser.add_argument("--quality-max-condition", type=float, default=1e8)
    parser.add_argument("--quality-min-observability", type=float, default=1e-6)
    parser.add_argument("--quality-max-window-consistency", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-sequences", type=int, default=0, help="Debug cap only; default runs full TotalCapture splits.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dipcalib_manifest = prepare_totalcapture_dipcalib_splits(root, out_dir / "totalcapture_dipcalib_splits")
    split_paths = {
        "AUTO_DIPCALIB_TRAIN": "data/dataset_work/SensorOffset/totalcapture_only_v2/totalcapture_dipcalib_splits/train.pt",
        "AUTO_DIPCALIB_VAL": "data/dataset_work/SensorOffset/totalcapture_only_v2/totalcapture_dipcalib_splits/val.pt",
    }
    split_specs = []
    for spec in TOTALCAPTURE_SPLITS:
        spec = dict(spec)
        spec["path"] = split_paths.get(spec["path"], spec["path"])
        split_specs.append(spec)

    alignment = {
        spec["key"]: alignment_audit(root / spec["path"], spec["key"])
        for spec in split_specs
    }
    frame_gravity = frame_gravity_audit(root / split_specs[0]["path"])

    compact_results = []
    cache_map = {}
    for spec in split_specs:
        for config in CONFIGS:
            cache_path, summary = run_config(root, out_dir, spec, config, args, max_sequences=args.max_sequences)
            cache_map[(spec["key"], config["name"])] = cache_path
            item = compact(summary)
            item["split_key"] = spec["key"]
            item["config"] = config["name"]
            compact_results.append(item)

    synthetic_results = []
    for config in [c for c in CONFIGS if c["name"] in ("A_centered_smooth5", "B_centered_smooth5")]:
        _, summary = run_synthetic(root, out_dir, config, args)
        item = compact(summary)
        item["config"] = config["name"]
        synthetic_results.append(item)

    selected = selected_config_name(compact_results)
    policy = fixed_dt_policy_summary(compact_results, selected)
    selected_manifest = copy_selected_caches(root, out_dir, selected, cache_map)
    recommendation = "PILOT ONLY"

    payload = {
        "output_dir": str(out_dir),
        "dipcalib_manifest": dipcalib_manifest,
        "alignment_audit": alignment,
        "frame_gravity_audit": frame_gravity,
        "configs": CONFIGS,
        "compact_results": compact_results,
        "synthetic_results": synthetic_results,
        "selected_config": selected,
        "time_alignment_policy": policy,
        "selected_manifest": str(selected_manifest),
        "recommendation": recommendation,
        "mainline_training_code_changed": False,
    }
    summary_json = out_dir / "totalcapture_v2_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2))
    torch.save(payload, out_dir / "totalcapture_v2_summary.pt")

    print(f"Saved summary JSON to {summary_json}")
    print(f"Saved selected cache manifest to {selected_manifest}")
    print(f"Selected config: {selected}")
    print(f"Recommendation: {recommendation}")
    print("Markdown source of truth: SENSOR_OFFSET_PLAN_AND_REVIEW.md")


if __name__ == "__main__":
    main()
