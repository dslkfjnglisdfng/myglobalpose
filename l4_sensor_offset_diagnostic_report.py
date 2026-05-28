import argparse
import json
from pathlib import Path

import torch

from l4_sensor_offset_utils import SENSOR_NAMES


def finite_stats(x):
    x = x.detach().cpu().float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"count": 0}
    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)) if x.numel() > 1 else 0.0,
        "min": float(x.min()),
        "p05": float(torch.quantile(x, 0.05)),
        "p25": float(torch.quantile(x, 0.25)),
        "median": float(torch.quantile(x, 0.50)),
        "p75": float(torch.quantile(x, 0.75)),
        "p95": float(torch.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def finite_ratio(mask, valid):
    valid = valid.detach().cpu().bool()
    mask = mask.detach().cpu().bool() & valid
    denom = int(valid.sum().item())
    return float(mask.sum().item() / denom) if denom else None


def window_tensor(payload, field, sensor_idx):
    values = []
    for seq_records in payload.get("window_records", []):
        for record in seq_records[sensor_idx]:
            if record.get("valid", False) and field in record:
                values.append(record[field])
    if not values:
        return torch.empty(0)
    return torch.stack([v if torch.is_tensor(v) else torch.tensor(v) for v in values])


def sequence_medians(payload):
    out = []
    for seq_idx, name in enumerate(payload["name"]):
        out.append(
            {
                "sequence": str(name),
                "median_offset_norm": float(torch.nanmedian(payload["offset_norm"][seq_idx])),
                "median_residual_improvement": float(torch.nanmedian(payload["residual_improvement"][seq_idx])),
                "median_condition_number": float(torch.nanmedian(payload["condition_number"][seq_idx])),
                "quality_fraction": float(payload.get("quality_mask", torch.zeros_like(payload["outlier_mask"]))[seq_idx].float().mean()),
            }
        )
    return out


def dt_summary(payload):
    if "dt_sensitivity" not in payload:
        return None
    dt_values = payload["dt_sensitivity"][0]["dt_values"].long()
    residual_fit = torch.stack([item["residual_fit"] for item in payload["dt_sensitivity"]])
    residual_improvement = torch.stack([item["residual_improvement"] for item in payload["dt_sensitivity"]])
    offset_norm = torch.stack([item["offset_norm"] for item in payload["dt_sensitivity"]])
    offset_deviation = torch.stack([item["offset_deviation_from_zero"] for item in payload["dt_sensitivity"]])
    dt_best = torch.stack([item["dt_best"] for item in payload["dt_sensitivity"]])
    by_dt = []
    for idx, dt in enumerate(dt_values.tolist()):
        by_dt.append(
            {
                "dt": int(dt),
                "median_residual_fit": float(torch.nanmedian(residual_fit[:, idx])),
                "median_residual_improvement": float(torch.nanmedian(residual_improvement[:, idx])),
                "median_offset_norm": float(torch.nanmedian(offset_norm[:, idx])),
                "median_offset_deviation_from_dt0": float(torch.nanmedian(offset_deviation[:, idx])),
            }
        )
    best_distribution = {
        str(int(dt)): int((dt_best == int(dt)).sum().item())
        for dt in dt_values.tolist()
    }
    zero_idx = (dt_values == 0).nonzero(as_tuple=False)
    dt0_is_global_best_fraction = None
    if zero_idx.numel():
        dt0_is_global_best_fraction = float((dt_best == 0).float().mean())
    return {
        "by_dt": by_dt,
        "best_dt_distribution": best_distribution,
        "dt0_is_best_fraction": dt0_is_global_best_fraction,
    }


def bad_entries(payload, max_offset_norm, max_condition, min_improvement):
    bad = []
    names = payload["name"]
    offset_norm = payload["offset_norm"]
    condition = payload["condition_number"]
    improvement = payload["residual_improvement"]
    outlier = payload["outlier_mask"]
    for seq_idx, name in enumerate(names):
        for sensor_idx, sensor_name in enumerate(SENSOR_NAMES):
            reasons = []
            if bool(outlier[seq_idx, sensor_idx]):
                reasons.append("outlier_mask")
            if torch.isfinite(offset_norm[seq_idx, sensor_idx]) and offset_norm[seq_idx, sensor_idx] > max_offset_norm:
                reasons.append("offset_norm")
            if torch.isfinite(condition[seq_idx, sensor_idx]) and condition[seq_idx, sensor_idx] > max_condition:
                reasons.append("condition_number")
            if torch.isfinite(improvement[seq_idx, sensor_idx]) and improvement[seq_idx, sensor_idx] < min_improvement:
                reasons.append("residual_improvement")
            if not torch.isfinite(offset_norm[seq_idx, sensor_idx]):
                reasons.append("nonfinite_offset")
            if reasons:
                bad.append(
                    {
                        "sequence": str(name),
                        "sensor": sensor_name,
                        "sensor_index": sensor_idx,
                        "reasons": reasons,
                        "offset_norm": float(offset_norm[seq_idx, sensor_idx]) if torch.isfinite(offset_norm[seq_idx, sensor_idx]) else None,
                        "condition_number": float(condition[seq_idx, sensor_idx]) if torch.isfinite(condition[seq_idx, sensor_idx]) else None,
                        "residual_improvement": float(improvement[seq_idx, sensor_idx]) if torch.isfinite(improvement[seq_idx, sensor_idx]) else None,
                    }
                )
    return bad


def summarize_payload(path, args):
    payload = torch.load(path, map_location="cpu")
    summary = {
        "path": str(path),
        "dataset": payload.get("metadata", {}).get("dataset", "unknown"),
        "split": payload.get("metadata", {}).get("split", "unknown"),
        "num_sequences": len(payload["name"]),
        "offset_shape": list(payload["offset"].shape),
        "sensor_to_joint": payload.get("metadata", {}).get("sensor_to_joint", {}),
        "overall": {
            "offset_norm": finite_stats(payload["offset_norm"]),
            "residual_zero": finite_stats(payload["residual_zero"]),
            "residual_fit": finite_stats(payload["residual_fit"]),
            "residual_improvement": finite_stats(payload["residual_improvement"]),
            "condition_number": finite_stats(payload["condition_number"]),
            "observability_score": finite_stats(payload["observability_score"]),
            "window_consistency": finite_stats(payload["window_consistency"]),
            "quality_mask_fraction": float(payload.get("quality_mask", torch.zeros_like(payload["outlier_mask"])).float().mean()),
        },
        "per_sensor": {},
        "per_sequence": sequence_medians(payload),
        "bad_entries": bad_entries(payload, args.max_offset_norm, args.max_condition, args.min_improvement),
    }
    for sensor_idx, sensor_name in enumerate(SENSOR_NAMES):
        valid_norm = torch.isfinite(payload["offset_norm"][:, sensor_idx])
        window_offsets = window_tensor(payload, "offset", sensor_idx)
        window_norm = window_tensor(payload, "offset_norm", sensor_idx)
        if window_offsets.numel() > 0:
            aggregate = payload["offset"][:, sensor_idx]
            deviations = []
            for seq_idx, seq_records in enumerate(payload.get("window_records", [])):
                for record in seq_records[sensor_idx]:
                    if record.get("valid", False) and "offset" in record:
                        deviations.append((record["offset"] - aggregate[seq_idx]).norm())
            window_deviation = torch.stack(deviations) if deviations else torch.empty(0)
        else:
            window_deviation = torch.empty(0)
        summary["per_sensor"][sensor_name] = {
            "offset_norm": finite_stats(payload["offset_norm"][:, sensor_idx]),
            "offset_norm_gt_0p4_ratio": finite_ratio(payload["offset_norm"][:, sensor_idx] > 0.4, valid_norm),
            "offset_norm_gt_0p5_ratio": finite_ratio(payload["offset_norm"][:, sensor_idx] > 0.5, valid_norm),
            "residual_zero": finite_stats(payload["residual_zero"][:, sensor_idx]),
            "residual_fit": finite_stats(payload["residual_fit"][:, sensor_idx]),
            "residual_improvement": finite_stats(payload["residual_improvement"][:, sensor_idx]),
            "residual_improvement_lt_0_ratio": finite_ratio(
                payload["residual_improvement"][:, sensor_idx] < 0,
                torch.isfinite(payload["residual_improvement"][:, sensor_idx]),
            ),
            "residual_improvement_abs_lt_0p02_ratio": finite_ratio(
                payload["residual_improvement"][:, sensor_idx].abs() < 0.02,
                torch.isfinite(payload["residual_improvement"][:, sensor_idx]),
            ),
            "condition_number": finite_stats(payload["condition_number"][:, sensor_idx]),
            "observability_score": finite_stats(payload["observability_score"][:, sensor_idx]),
            "window_consistency": finite_stats(payload["window_consistency"][:, sensor_idx]),
            "window_offset_norm": finite_stats(window_norm),
            "window_to_sequence_offset_deviation": finite_stats(window_deviation),
            "outlier_count": int(payload["outlier_mask"][:, sensor_idx].sum().item()),
            "quality_count": int(payload.get("quality_mask", torch.zeros_like(payload["outlier_mask"]))[:, sensor_idx].sum().item()),
        }
    left_right_pairs = [("left_forearm", "right_forearm"), ("left_lower_leg", "right_lower_leg")]
    summary["left_right_symmetry"] = {
        f"{left}_vs_{right}": {
            "median_norm_delta": summary["per_sensor"][left]["offset_norm"].get("median", float("nan"))
            - summary["per_sensor"][right]["offset_norm"].get("median", float("nan")),
        }
        for left, right in left_right_pairs
    }
    dts = dt_summary(payload)
    if dts is not None:
        summary["dt_sensitivity"] = dts
    if "synthetic_offset_error" in payload:
        summary["synthetic_sanity"] = {
            "offset_error": finite_stats(payload["synthetic_offset_error"]),
            "per_sensor_error": {
                sensor_name: finite_stats(payload["synthetic_offset_error"][:, sensor_idx])
                for sensor_idx, sensor_name in enumerate(SENSOR_NAMES)
            },
        }
    return summary


def print_summary(summary):
    print(f"\n=== {summary['dataset']} / {summary['split']} ===")
    print(f"path: {summary['path']}")
    print(f"offset_shape: {summary['offset_shape']} sequences={summary['num_sequences']}")
    print("overall offset_norm:", summary["overall"]["offset_norm"])
    print("overall residual_improvement:", summary["overall"]["residual_improvement"])
    print("overall condition_number:", summary["overall"]["condition_number"])
    print("overall observability_score:", summary["overall"]["observability_score"])
    if "synthetic_sanity" in summary:
        print("synthetic offset_error:", summary["synthetic_sanity"]["offset_error"])
    print(f"bad entries: {len(summary['bad_entries'])}")
    for sensor_name, stats in summary["per_sensor"].items():
        print(
            f"  {sensor_name}: norm_median={stats['offset_norm'].get('median')} "
            f"improve_median={stats['residual_improvement'].get('median')} "
            f"cond_median={stats['condition_number'].get('median')} "
            f"outliers={stats['outlier_count']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize l4 sensor offset diagnostic caches.")
    parser.add_argument("inputs", nargs="+", help="One or more offset diagnostic .pt files.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--max-offset-norm", type=float, default=0.25)
    parser.add_argument("--max-condition", type=float, default=1e8)
    parser.add_argument("--min-improvement", type=float, default=-0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = [summarize_payload(Path(path), args) for path in args.inputs]
    for summary in summaries:
        print_summary(summary)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"reports": summaries}, indent=2))
        print(f"\nSaved report JSON to {out}")


if __name__ == "__main__":
    main()
