#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def mean(block, key="mean"):
    if isinstance(block, dict):
        return block.get(key)
    return None


def metric_row(path, version, dataset, notes):
    data = read_json(path)
    if data is None:
        return {
            "dataset": dataset,
            "version": version,
            "status": "not available",
            "path": str(path),
            "notes": notes,
        }
    if data.get("status") != "ok":
        return {
            "dataset": dataset,
            "version": version,
            "status": data.get("status", "failed"),
            "path": str(path),
            "notes": data.get("error", notes),
        }
    agg = data["aggregate"]
    leaf = agg["leaf_position_error_cm"]
    grav = agg["gravity_angle_deg"]
    return {
        "dataset": dataset,
        "version": version,
        "status": "ok",
        "path": str(path),
        "num_sequences": agg.get("num_sequences"),
        "num_frames": agg.get("num_frames"),
        "all_finite": agg.get("all_finite"),
        "pRB_L2_cm": mean(leaf["new"]),
        "pRB_L2_delta_vs_official_cm": mean(leaf["delta_new_minus_original"]),
        "gR1_angle_deg": mean(grav["new"]),
        "gR1_delta_vs_official_deg": mean(grav["delta_new_minus_original"]),
        "per_leaf_pRB_L2_cm": {
            leaf_name: mean(values["new_cm"])
            for leaf_name, values in leaf.get("by_leaf", {}).items()
        },
        "notes": notes,
    }


def official_row(path, dataset):
    data = read_json(path)
    if data is None or data.get("status") != "ok":
        return {
            "dataset": dataset,
            "version": "official PL baseline",
            "status": "not available",
            "path": str(path),
            "notes": "requires one successful PL eval JSON to read original baseline fields",
        }
    agg = data["aggregate"]
    leaf = agg["leaf_position_error_cm"]
    grav = agg["gravity_angle_deg"]
    return {
        "dataset": dataset,
        "version": "official PL baseline",
        "status": "ok",
        "path": str(path),
        "num_sequences": agg.get("num_sequences"),
        "num_frames": agg.get("num_frames"),
        "all_finite": agg.get("all_finite"),
        "pRB_L2_cm": mean(leaf["original"]),
        "pRB_L2_delta_vs_official_cm": 0.0,
        "gR1_angle_deg": mean(grav["original"]),
        "gR1_delta_vs_official_deg": 0.0,
        "per_leaf_pRB_L2_cm": {
            leaf_name: mean(values["original_cm"])
            for leaf_name, values in leaf.get("by_leaf", {}).items()
        },
        "notes": "original GPNet PL output from the same split/cache",
    }


def swap_row(path, version, dataset):
    data = read_json(path)
    if data is None:
        return {
            "dataset": dataset,
            "version": version,
            "status": "not available",
            "path": str(path),
        }
    if data.get("status") != "ok":
        return {
            "dataset": dataset,
            "version": version,
            "status": data.get("status", "failed"),
            "path": str(path),
            "notes": data.get("error"),
        }
    delta = data["aggregate"]["delta_vs_good"]
    row = {
        "dataset": dataset,
        "version": version,
        "status": "ok",
        "path": str(path),
        "swap_feature_offset": bool(data.get("swap_feature_offset")),
    }
    for variant, values in delta.items():
        row[f"{variant}_minus_good_pRB_cm"] = mean(values["pRB_cm"])
        row[f"{variant}_minus_good_gR1_deg"] = mean(values["gR1_deg"])
    return row


def table_lines(rows):
    lines = [
        "| Dataset | Version | pRB L2 cm ↓ | Δ pRB vs official | gR1 angle deg ↓ | Δ gR1 vs official | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {version} | {pRB} | {dp} | {g} | {dg} | {notes} |".format(
                dataset=row.get("dataset"),
                version=row.get("version"),
                pRB=format_value(row.get("pRB_L2_cm")),
                dp=format_value(row.get("pRB_L2_delta_vs_official_cm")),
                g=format_value(row.get("gR1_angle_deg")),
                dg=format_value(row.get("gR1_delta_vs_official_deg")),
                notes=row.get("notes", row.get("status", "")),
            )
        )
    return lines


def format_value(value):
    if value is None:
        return "not available"
    if isinstance(value, str):
        return value
    return f"{float(value):.6f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize NewPL offset-aware v6 control-only evals.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    eval_dir = root / "eval"
    rows = []
    swaps = []
    for dataset in ("amass", "dip_test", "tc_test"):
        ref_path = eval_dir / f"newpl_offset_v6_best_{dataset}.json"
        rows.append(official_row(ref_path, dataset))
        rows.append(metric_row(eval_dir / f"newpl_v4_init36_{dataset}.json", "newpl_v4_init36 baseline", dataset, "not available if checkpoint path is unset"))
        rows.append(metric_row(eval_dir / f"newpl_v5_dip_best_{dataset}.json", "newpl_v5_dip_best", dataset, "v5 official-protocol baseline"))
        rows.append(metric_row(eval_dir / f"canonical_control_dip_best_{dataset}.json", "canonical_control_dip_best", dataset, "canonical GT-control baseline"))
        rows.append(metric_row(ref_path, "newpl_offset_v6_best", dataset, "control-only offset-aware"))
        rows.append(metric_row(eval_dir / f"newpl_offset_v6_last_{dataset}.json", "newpl_offset_v6_last", dataset, "control-only offset-aware last"))
    for dataset in ("dip_test", "tc_test"):
        swaps.append(swap_row(eval_dir / f"newpl_offset_v6_best_{dataset}_offset_swap.json", "newpl_offset_v6_best", dataset))
        swaps.append(swap_row(eval_dir / f"newpl_offset_v6_last_{dataset}_offset_swap.json", "newpl_offset_v6_last", dataset))

    result = {
        "status": "ok",
        "root": str(root),
        "loss_contract": "control-only: derivative and second-derivative loss weights are disabled; GT fitted control-point losses select best checkpoint.",
        "coordinate_contract": "r_JS is the IMU origin relative to mapped joint J, expressed in the joint-local frame. p_WS=p_WJ+R_WJ@r_JS.",
        "rows": rows,
        "offset_sensitivity": swaps,
        "tables": {
            "pl_output_comparison": table_lines(rows),
        },
    }
    output_json = args.output_json or root / "summary.json"
    output_md = args.output_md or root / "summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    output_md.write_text(
        "\n".join([
            "# NewPL Offset v6 Control-Only Summary",
            "",
            f"Root: `{root}`",
            "",
            "## PL Output Comparison",
            "",
            *table_lines(rows),
            "",
            "## Offset Sensitivity",
            "",
            "See `summary.json` for correct/zero/rolled/random offset diagnostics.",
            "",
        ]) + "\n"
    )
    print(json.dumps({"status": "ok", "summary_json": str(output_json), "summary_md": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
