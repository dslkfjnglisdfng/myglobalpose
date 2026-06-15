import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imu_position_offset import (
    OFFSET_POSITION_CONTRACT,
    FOOTLOCK_POSITION_CONTRACT,
    TransPoseContactEstimator,
    build_output,
    load_offset_cache,
    plausibility_project,
    prepare_sequence,
    solve_footlock_transpose_offset,
)


def finite_stats(x):
    x = torch.as_tensor(x).float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p95": float(torch.quantile(x, 0.95)),
    }


def serializable_config(args):
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def resolve_cache_path(path, manifest_path=None):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [path, ROOT / path]
    if manifest_path is not None:
        candidates.append(Path(manifest_path).parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _extend_dataset(merged, shard):
    for key, value in shard.items():
        if isinstance(value, list):
            if not value:
                continue
            merged.setdefault(key, []).extend(value)
        elif torch.is_tensor(value) and value.dim() > 0:
            merged.setdefault(key, []).extend([value[idx] for idx in range(value.shape[0])])
        else:
            merged[key] = value


def normalize_dataset_fields(data):
    out = dict(data)
    if "pose" not in out:
        if "pose_gt" in out:
            out["pose"] = out["pose_gt"]
        elif "pose_prephysics" in out:
            out["pose"] = out["pose_prephysics"]
        elif "pose_baseline" in out:
            out["pose"] = out["pose_baseline"]
    if "tran" not in out:
        if "tran_gt" in out:
            out["tran"] = out["tran_gt"]
        elif "tran_baseline" in out:
            out["tran"] = out["tran_baseline"]
        elif "pose" in out:
            out["tran"] = [torch.zeros(item.shape[0], 3, dtype=torch.float32) for item in out["pose"]]
    if "name" not in out and "pose" in out:
        out["name"] = [f"seq_{idx}" for idx in range(len(out["pose"]))]
    missing = [key for key in ("pose", "tran", "aM", "wM", "RMB") if key not in out]
    if missing:
        raise KeyError(f"Input cache missing required fields after alias normalization: {missing}")
    return out


def load_input_dataset(input_path):
    input_path = Path(input_path)
    if input_path.suffix == ".json":
        manifest = json.loads(input_path.read_text())
        merged = {"source_manifest": str(input_path), "manifest": manifest}
        for item in manifest["cache_files"]:
            shard_path = resolve_cache_path(item["path"], manifest_path=input_path)
            shard = torch.load(shard_path, map_location="cpu")
            _extend_dataset(merged, shard)
        return normalize_dataset_fields(merged)
    return normalize_dataset_fields(torch.load(input_path, map_location="cpu"))


def method_offset(args, seq, contact_model, fallback_offsets=None):
    if args.method != "footlock_transpose_v1":
        raise ValueError(
            f"{args.method} is retired. The only active real-data r_JS route is footlock_transpose_v1."
        )
    if contact_model is None:
        raise RuntimeError("--method footlock_transpose_v1 requires a TransPose contact model")
    fallback_offset = None if fallback_offsets is None else fallback_offsets.get(seq["name"])
    result = solve_footlock_transpose_offset(
        seq,
        contact_model,
        device=args.device,
        ridge=args.ridge,
        fit_sensor_bias=not args.no_fit_sensor_bias,
        max_offset_norm=args.max_offset_norm,
        contact_threshold=args.contact_threshold,
        contact_margin=args.contact_margin,
        min_contact_frames=args.min_contact_frames,
        max_contact_frames=args.max_contact_frames,
        contact_selection_mode=args.contact_selection_mode,
        contact_height_margin=args.contact_height_margin,
        transpose_prob_low=args.transpose_prob_low,
        transpose_prob_high=args.transpose_prob_high,
        min_fit_frames=args.min_fit_frames,
        min_fit_improvement=args.min_fit_improvement,
        max_condition_number=args.max_condition_number,
        fallback_offset=fallback_offset,
        derivative_mode=args.derivative_mode,
        smooth_window=args.smooth_window,
    )
    result["offset"] = plausibility_project(result["offset"], max_norm=args.max_offset_norm)
    return result


def build(args):
    data = load_input_dataset(args.input)
    count = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    weights = Path(args.transpose_weights)
    if not weights.is_absolute():
        weights = Path(args.transpose_root) / weights
    contact_model = TransPoseContactEstimator(weights, device=args.device)
    fallback_offsets = load_offset_cache(args.fallback_offset_cache) if args.fallback_offset_cache else None
    names = []
    offsets = []
    rows = []
    for seq_idx in range(count):
        seq = prepare_sequence(
            data,
            seq_idx,
            device=args.device,
            smooth_window=args.smooth_window,
            derivative_mode=args.derivative_mode,
            max_frames=args.max_frames,
        )
        result = method_offset(
            args,
            seq,
            contact_model=contact_model,
            fallback_offsets=fallback_offsets,
        )
        offset = result["offset"].float()
        names.append(seq["name"])
        offsets.append(offset)
        row = {
            "name": seq["name"],
            "num_frames": int(seq["pose"].shape[0]),
            "method_source": result["source"],
            "offset_norm_m": [float(x) for x in offset.norm(dim=-1)],
            "offset_norm_median": float(offset.norm(dim=-1).median()),
            "finite": bool(torch.isfinite(offset).all()),
        }
        for key in ("residual_zero", "residual_fit", "condition_number", "solver_residual_fit", "confidence", "fit_improvement", "num_fit_frames", "num_fit_windows", "contact_probability_mean"):
            if key in result:
                row[key] = [float(x) if torch.isfinite(x) else None for x in result[key].float()]
        for key in ("fallback_reason", "contact_selection_mode", "contact_height_margin", "contact_window_count", "contact_side_window_count", "contact_windows", "contact_input", "fit_input", "coordinate_contract"):
            if key in result:
                row[key] = result[key]
        rows.append(row)
        print(json.dumps({"idx": seq_idx + 1, "count": count, "name": seq["name"], "offset_norm_median": row["offset_norm_median"]}), flush=True)
    output = build_output(
        names,
        offsets,
        rows,
        args.method,
        args.input,
        extra={
            "config": serializable_config(args),
            "dataset_note": (
                "DIP-IMU has zero/unsupported global trans in GlobalPose preprocessing; "
                "footlock_transpose_v1 uses stance-window pseudo-r_JS and must not be treated as true GT."
                if args.dataset == "dip"
                else "TotalCapture offsets are diagnostic unless the split is explicitly training-only."
            ),
            "footlock_contract": FOOTLOCK_POSITION_CONTRACT if args.method == "footlock_transpose_v1" else None,
        },
    )
    offset_tensor = output["offset"]
    output["summary"] = {
        "num_sequences": len(names),
        "offset_norm_m": finite_stats(offset_tensor.norm(dim=-1)),
        "all_finite": bool(torch.isfinite(offset_tensor).all()),
        "coordinate_contract": OFFSET_POSITION_CONTRACT,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps({
            "output": str(args.output),
            "method": args.method,
            "summary": output["summary"],
            "config": serializable_config(args),
            "rows": rows,
        }, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description="Build footlock_transpose_v1 sequence-level IMU position offset caches.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--dataset", choices=("dip", "totalcapture", "amass", "other"), default="other")
    parser.add_argument("--method", choices=("footlock_transpose_v1",), default="footlock_transpose_v1")
    parser.add_argument("--transpose-root", type=Path, default=Path("/home/lingfeng/projects/TransPose"))
    parser.add_argument("--transpose-weights", type=Path, default=Path("data/weights.pt"))
    parser.add_argument("--contact-threshold", type=float, default=0.85)
    parser.add_argument("--contact-margin", type=float, default=0.15)
    parser.add_argument("--contact-selection-mode", choices=("transpose_winner",), default="transpose_winner")
    parser.add_argument("--contact-height-margin", type=float, default=0.08)
    parser.add_argument("--transpose-prob-low", type=float, default=0.5)
    parser.add_argument("--transpose-prob-high", type=float, default=0.9)
    parser.add_argument("--min-contact-frames", type=int, default=24)
    parser.add_argument("--max-contact-frames", type=int, default=180)
    parser.add_argument("--min-fit-frames", type=int, default=48)
    parser.add_argument("--fallback-offset-cache", default="")
    parser.add_argument("--min-fit-improvement", type=float, default=0.05)
    parser.add_argument("--max-condition-number", type=float, default=1e5)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--derivative-mode", choices=("legacy", "centered", "strict_centered"), default="centered")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--no-fit-sensor-bias", action="store_true")
    parser.add_argument("--max-offset-norm", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = build(args)
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    main()
