#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def diag_rows(path, version, dataset, notes):
    data = read_json(path)
    if data is None:
        return [{
            "dataset": dataset,
            "version": version,
            "status": "not available",
            "path": str(path),
            "notes": notes,
        }]
    if data.get("status") != "ok":
        return [{
            "dataset": dataset,
            "version": version,
            "status": data.get("status", "failed"),
            "path": str(path),
            "notes": data.get("error", notes),
        }]
    agg = data["aggregate"]
    baseline = agg["baseline_ik1_on_newpl"]
    model = agg["newik1"]
    delta = agg["delta_newik1_minus_baseline"]
    return [
        {
            "dataset": dataset,
            "version": "official IK1 baseline",
            "status": "ok",
            "path": str(path),
            "pRJ_L2_cm": baseline.get("pRJ_cm_l2"),
            "leaf_pRJ_L2_cm": baseline.get("leaf_pRJ_cm_l2"),
            "gR2_angle_deg": baseline.get("gR2_angle_deg"),
            "state_L2": baseline.get("state_l2"),
            "notes": "ik1_base from the same canonical-control NewPL cache",
        },
        {
            "dataset": dataset,
            "version": version,
            "status": "ok",
            "path": str(path),
            "pRJ_L2_cm": model.get("pRJ_cm_l2"),
            "leaf_pRJ_L2_cm": model.get("leaf_pRJ_cm_l2"),
            "gR2_angle_deg": model.get("gR2_angle_deg"),
            "state_L2": model.get("state_l2"),
            "delta_pRJ_L2_cm": delta.get("pRJ_cm_l2"),
            "delta_leaf_pRJ_L2_cm": delta.get("leaf_pRJ_cm_l2"),
            "delta_gR2_angle_deg": delta.get("gR2_angle_deg"),
            "notes": notes,
        },
    ]


def format_value(value):
    if value is None:
        return "not available"
    return f"{float(value):.6f}"


def table_lines(rows):
    lines = [
        "| Dataset | Version | pRJ L2 cm ↓ | leaf pRJ L2 cm ↓ | gR2 angle deg ↓ | Δ pRJ vs official | Δ gR2 vs official | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {version} | {p} | {leaf} | {g} | {dp} | {dg} | {notes} |".format(
                dataset=row.get("dataset"),
                version=row.get("version"),
                p=format_value(row.get("pRJ_L2_cm")),
                leaf=format_value(row.get("leaf_pRJ_L2_cm")),
                g=format_value(row.get("gR2_angle_deg")),
                dp=format_value(row.get("delta_pRJ_L2_cm")),
                dg=format_value(row.get("delta_gR2_angle_deg")),
                notes=row.get("notes", row.get("status", "")),
            )
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description="Summarize NewIK1 v11 canonical-control module diagnostics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    eval_dir = root / "eval"
    rows = []
    for dataset in ("amass", "dip_val", "dip_test", "tc_test"):
        rows.extend(diag_rows(eval_dir / f"newik1_v11_best_{dataset}_local_diag.json", "newik1_v11_best", dataset, "control-only canonical-control IK1"))
        rows.extend(diag_rows(eval_dir / f"newik1_v11_last_{dataset}_local_diag.json", "newik1_v11_last", dataset, "control-only canonical-control IK1 last"))
        rows.extend(diag_rows(eval_dir / f"newik1_v10_stage_c_best_{dataset}_local_diag.json", "newik1_v10_stage_c_best", dataset, "historical v10 checkpoint on this cache"))

    result = {
        "status": "ok",
        "root": str(root),
        "loss_contract": "control-only IK1: derivative and second-derivative loss weights are disabled; best checkpoint is selected by ik1/control physical output terms.",
        "upstream": "canonical_control NewPL checkpoints are used to generate PL-streaming cache.",
        "rows": rows,
        "tables": {"ik1_module_comparison": table_lines(rows)},
    }
    output_json = args.output_json or root / "summary.json"
    output_md = args.output_md or root / "summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    output_md.write_text(
        "\n".join([
            "# NewIK1 v11 Canonical-Control Summary",
            "",
            f"Root: `{root}`",
            "",
            "## IK1 Module Comparison",
            "",
            *table_lines(rows),
            "",
        ]) + "\n"
    )
    print(json.dumps({"status": "ok", "summary_json": str(output_json), "summary_md": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
