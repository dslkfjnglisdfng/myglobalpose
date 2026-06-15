import argparse
import json
from pathlib import Path


RAW_BASELINE_ROOT = Path("data/experiments/newpl_v5_official_protocol_20260607_tuned")


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_float(value):
    if value in (None, "not available", "not measured"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rows_by_version(rows):
    return {row["Version"]: row for row in rows or []}


def eval_payload(root, name):
    path = root / "eval" / f"{name}.json"
    data = load_json(path)
    return {"path": str(path), "data": data}


def compact_row(row):
    if not row:
        return None
    return {
        "version": row.get("Version"),
        "pRB_L1_cm": metric_float(row.get("pRB L1 cm \u2193")),
        "pRB_L2_cm": metric_float(row.get("pRB L2 cm \u2193")),
        "gR1_angle_deg": metric_float(row.get("gR1 angle deg \u2193")),
        "notes": row.get("Notes", ""),
    }


def delta(new_row, base_row, key):
    new_value = None if not new_row else new_row.get(key)
    base_value = None if not base_row else base_row.get(key)
    if new_value is None or base_value is None:
        return None
    return new_value - base_value


def comparison_block(dataset_name, stage_name, raw_rows, smooth_rows, smooth_version, raw_version):
    raw = rows_by_version(raw_rows)
    smooth = rows_by_version(smooth_rows)
    raw_official = compact_row(raw.get("official_PL"))
    smooth_official = compact_row(smooth.get("official_PL_smoothacc"))
    raw_v5 = compact_row(raw.get(raw_version))
    smooth_v5 = compact_row(smooth.get(smooth_version))
    smooth_v4 = compact_row(smooth.get("newpl_v4_init36_smoothacc"))
    return {
        "dataset": dataset_name,
        "stage": stage_name,
        "raw_official": raw_official,
        "smooth_official": smooth_official,
        "raw_v5_reference": raw_v5,
        "smoothacc_v5": smooth_v5,
        "smoothacc_v4": smooth_v4,
        "smooth_official_delta_vs_raw_official": {
            "pRB_L1_cm": delta(smooth_official, raw_official, "pRB_L1_cm"),
            "pRB_L2_cm": delta(smooth_official, raw_official, "pRB_L2_cm"),
            "gR1_angle_deg": delta(smooth_official, raw_official, "gR1_angle_deg"),
        },
        "smoothacc_v5_delta_vs_raw_v5": {
            "pRB_L1_cm": delta(smooth_v5, raw_v5, "pRB_L1_cm"),
            "pRB_L2_cm": delta(smooth_v5, raw_v5, "pRB_L2_cm"),
            "gR1_angle_deg": delta(smooth_v5, raw_v5, "gR1_angle_deg"),
        },
        "smoothacc_v5_delta_vs_smooth_official": {
            "pRB_L1_cm": delta(smooth_v5, smooth_official, "pRB_L1_cm"),
            "pRB_L2_cm": delta(smooth_v5, smooth_official, "pRB_L2_cm"),
            "gR1_angle_deg": delta(smooth_v5, smooth_official, "gR1_angle_deg"),
        },
        "smoothacc_v5_delta_vs_smooth_v4": {
            "pRB_L1_cm": delta(smooth_v5, smooth_v4, "pRB_L1_cm"),
            "pRB_L2_cm": delta(smooth_v5, smooth_v4, "pRB_L2_cm"),
            "gR1_angle_deg": delta(smooth_v5, smooth_v4, "gR1_angle_deg"),
        },
    }


def format_value(value, suffix=""):
    if value is None:
        return "not available"
    return f"{value:.6f}{suffix}"


def format_delta(value, suffix=""):
    if value is None:
        return "not available"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.6f}{suffix}"


def row_line(label, row):
    if not row:
        return f"- {label}: not available"
    return (
        f"- {label}: pRB L1 {format_value(row['pRB_L1_cm'], ' cm')}, "
        f"pRB L2 {format_value(row['pRB_L2_cm'], ' cm')}, "
        f"gR1 {format_value(row['gR1_angle_deg'], ' deg')}"
    )


def block_to_lines(block):
    lines = [f"## {block['dataset']} {block['stage']}", ""]
    lines.append(row_line("raw official_PL", block["raw_official"]))
    lines.append(row_line("smooth official_PL_smoothacc", block["smooth_official"]))
    lines.append(row_line("raw NewPL v5 reference", block["raw_v5_reference"]))
    lines.append(row_line("smooth-acc NewPL v5", block["smoothacc_v5"]))
    lines.append(row_line("smooth-acc newpl_v4_init36", block["smoothacc_v4"]))
    lines.append("")
    d = block["smooth_official_delta_vs_raw_official"]
    lines.append(
        "- smooth official vs raw official delta: "
        f"pRB L2 {format_delta(d['pRB_L2_cm'], ' cm')}, "
        f"gR1 {format_delta(d['gR1_angle_deg'], ' deg')}"
    )
    d = block["smoothacc_v5_delta_vs_raw_v5"]
    lines.append(
        "- smooth-acc NewPL v5 vs raw NewPL v5 delta: "
        f"pRB L2 {format_delta(d['pRB_L2_cm'], ' cm')}, "
        f"gR1 {format_delta(d['gR1_angle_deg'], ' deg')}"
    )
    d = block["smoothacc_v5_delta_vs_smooth_official"]
    lines.append(
        "- smooth-acc NewPL v5 vs smooth official delta: "
        f"pRB L2 {format_delta(d['pRB_L2_cm'], ' cm')}, "
        f"gR1 {format_delta(d['gR1_angle_deg'], ' deg')}"
    )
    d = block["smoothacc_v5_delta_vs_smooth_v4"]
    lines.append(
        "- smooth-acc NewPL v5 vs smooth newpl_v4 delta: "
        f"pRB L2 {format_delta(d['pRB_L2_cm'], ' cm')}, "
        f"gR1 {format_delta(d['gR1_angle_deg'], ' deg')}"
    )
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Summarize NewPL v5 smooth-acc experiment.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--raw-baseline-summary", default=str(RAW_BASELINE_ROOT / "summary.json"))
    args = parser.parse_args()

    root = Path(args.root)
    raw_summary = load_json(args.raw_baseline_summary) or {}
    evals = {
        "amass_after_amass_pretrain_smoothinput": eval_payload(root, "amass_after_amass_pretrain_smoothinput"),
        "dip_test_after_amass_pretrain_smoothinput": eval_payload(root, "dip_test_after_amass_pretrain_smoothinput"),
        "tc_test_after_amass_pretrain_smoothinput": eval_payload(root, "tc_test_after_amass_pretrain_smoothinput"),
        "dip_test_after_dip_finetune_smoothinput": eval_payload(root, "dip_test_after_dip_finetune_smoothinput"),
        "tc_test_after_dip_finetune_smoothinput": eval_payload(root, "tc_test_after_dip_finetune_smoothinput"),
    }

    blocks = []
    if evals["dip_test_after_amass_pretrain_smoothinput"]["data"]:
        blocks.append(comparison_block(
            "DIP-IMU test",
            "after AMASS pretrain",
            raw_summary.get("dip_test_after_amass_pretrain", {}).get("pl_output_comparison_table", []),
            evals["dip_test_after_amass_pretrain_smoothinput"]["data"].get("pl_output_comparison_table", []),
            "newpl_v5_smoothacc_amass_best",
            "newpl_v5_amass_best",
        ))
    if evals["tc_test_after_amass_pretrain_smoothinput"]["data"]:
        blocks.append(comparison_block(
            "TotalCapture test",
            "after AMASS pretrain",
            raw_summary.get("tc_test_after_amass_pretrain", {}).get("pl_output_comparison_table", []),
            evals["tc_test_after_amass_pretrain_smoothinput"]["data"].get("pl_output_comparison_table", []),
            "newpl_v5_smoothacc_amass_best",
            "newpl_v5_amass_best",
        ))
    if evals["dip_test_after_dip_finetune_smoothinput"]["data"]:
        blocks.append(comparison_block(
            "DIP-IMU test",
            "after DIP fine-tune",
            raw_summary.get("dip_test_after_dip_finetune", {}).get("pl_output_comparison_table", []),
            evals["dip_test_after_dip_finetune_smoothinput"]["data"].get("pl_output_comparison_table", []),
            "newpl_v5_smoothacc_dip_best",
            "newpl_v5_dip_best",
        ))
    if evals["tc_test_after_dip_finetune_smoothinput"]["data"]:
        blocks.append(comparison_block(
            "TotalCapture test",
            "after DIP fine-tune",
            raw_summary.get("tc_test_after_dip_finetune", {}).get("pl_output_comparison_table", []),
            evals["tc_test_after_dip_finetune_smoothinput"]["data"].get("pl_output_comparison_table", []),
            "newpl_v5_smoothacc_dip_best",
            "newpl_v5_dip_best",
        ))

    summary = {
        "status": "ok" if blocks else "incomplete",
        "root": str(root),
        "protocol": "AMASS pretrain with smoothed aM -> DIP train fine-tune with smoothed aM -> DIP/TotalCapture test with smoothed aM",
        "input_contract": "aM is centered moving-average smoothed from official aM with window=9; aM_raw is preserved; wM/RMB and targets are unchanged.",
        "loss_contract": "same NewPL v5 PL loss and control_physical checkpoint selection; GT control caches are used to avoid repeated fitting.",
        "dip_trans_loss_used": False,
        "full_pipeline_11_metrics": "not measured",
        "raw_baseline_summary": str(args.raw_baseline_summary),
        "eval_jsons": {name: payload["path"] for name, payload in evals.items()},
        "comparisons": blocks,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# NewPL v5 Smooth-Acc Summary",
        "",
        "Protocol: AMASS pretrain -> DIP-IMU fine-tune -> DIP-IMU and TotalCapture module-level test.",
        "",
        "Input contract: replace official `aM` with centered moving-average smoothed `aM` (window 9); keep `aM_raw` for audit; keep `wM/RMB/targets/offset_r` unchanged.",
        "",
        "Loss contract: same NewPL v5 loss and `control_physical` best-checkpoint selection; GT control caches are used for speed.",
        "",
        "Full-pipeline 11 metrics: not measured.",
        "DIP translation/root velocity loss: not used.",
        "",
    ]
    for block in blocks:
        lines.extend(block_to_lines(block))
    (root / "summary.md").write_text("\n".join(lines).rstrip() + "\n")
    print(json.dumps({"status": summary["status"], "summary_json": str(root / "summary.json"), "summary_md": str(root / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
