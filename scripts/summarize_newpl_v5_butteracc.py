import argparse
import json
from pathlib import Path


RAW_BASELINE_SUMMARY = Path("data/experiments/newpl_v5_official_protocol_20260607_tuned/summary.json")
CENTERED_SMOOTH_SUMMARY = Path("data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/summary.json")


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


def cutoff_tag(cutoff):
    text = f"{float(cutoff):g}"
    return "fc" + text.replace(".", "p")


def get_raw_row(raw_summary, key, version):
    rows = rows_by_version(raw_summary.get(key, {}).get("pl_output_comparison_table", []))
    return compact_row(rows.get(version))


def get_eval_row(root, name, version):
    data = load_json(root / "eval" / f"{name}.json")
    if not data:
        return None
    return compact_row(rows_by_version(data.get("pl_output_comparison_table", [])).get(version))


def select_cutoff(root, cutoffs, raw_summary, force_cutoff_hz="", tc_margin_cm=0.10):
    raw_tc_official = get_raw_row(raw_summary, "tc_test_after_dip_finetune", "official_PL")
    raw_tc_pRB = None if not raw_tc_official else raw_tc_official["pRB_L2_cm"]
    rows = []
    for cutoff in cutoffs:
        tag = cutoff_tag(cutoff)
        dip = get_eval_row(root, f"input_only_dip_test_{tag}", f"official_PL_butter_{tag}")
        tc = get_eval_row(root, f"input_only_tc_test_{tag}", f"official_PL_butter_{tag}")
        allowed = False
        reason = "not evaluated"
        if raw_tc_pRB is not None and tc and tc["pRB_L2_cm"] is not None:
            allowed = tc["pRB_L2_cm"] <= raw_tc_pRB + float(tc_margin_cm)
            reason = (
                "passes TC pRB guard"
                if allowed
                else f"TC pRB exceeds raw official by {tc['pRB_L2_cm'] - raw_tc_pRB:.6f} cm"
            )
        rows.append({
            "cutoff_hz": float(cutoff),
            "tag": tag,
            "dip_official_butter": dip,
            "tc_official_butter": tc,
            "passes_tc_guard": allowed,
            "selection_reason": reason,
        })
    if force_cutoff_hz:
        selected = float(force_cutoff_hz)
        return {
            "selected_cutoff_hz": selected,
            "selected_tag": cutoff_tag(selected),
            "status": "forced",
            "reason": f"forced by FORCE_CUTOFF_HZ={force_cutoff_hz}",
            "raw_tc_official_pRB_L2_cm": raw_tc_pRB,
            "tc_margin_cm": float(tc_margin_cm),
            "input_only_rows": rows,
        }
    candidates = [row for row in rows if row["passes_tc_guard"] and row["dip_official_butter"]]
    if not candidates:
        return {
            "selected_cutoff_hz": None,
            "selected_tag": "",
            "status": "no_candidate",
            "reason": "no cutoff passed the TotalCapture pRB guard",
            "raw_tc_official_pRB_L2_cm": raw_tc_pRB,
            "tc_margin_cm": float(tc_margin_cm),
            "input_only_rows": rows,
        }
    candidates.sort(key=lambda row: (
        row["dip_official_butter"]["pRB_L2_cm"],
        row["dip_official_butter"]["gR1_angle_deg"],
        abs(row["cutoff_hz"] - 10.0),
    ))
    best = candidates[0]
    return {
        "selected_cutoff_hz": best["cutoff_hz"],
        "selected_tag": best["tag"],
        "status": "selected",
        "reason": "lowest DIP pRB among cutoffs passing TotalCapture pRB guard",
        "raw_tc_official_pRB_L2_cm": raw_tc_pRB,
        "tc_margin_cm": float(tc_margin_cm),
        "input_only_rows": rows,
    }


def final_block(root, tag, raw_summary):
    if not tag:
        return {}
    versions = {
        "raw_official_dip": get_raw_row(raw_summary, "dip_test_after_dip_finetune", "official_PL"),
        "raw_official_tc": get_raw_row(raw_summary, "tc_test_after_dip_finetune", "official_PL"),
        "raw_v5_dip": get_raw_row(raw_summary, "dip_test_after_dip_finetune", "newpl_v5_dip_best"),
        "raw_v5_tc": get_raw_row(raw_summary, "tc_test_after_dip_finetune", "newpl_v5_dip_best"),
        "butter_official_dip": get_eval_row(root, f"dip_test_after_dip_finetune_butter_{tag}", f"official_PL_butter_{tag}"),
        "butter_official_tc": get_eval_row(root, f"tc_test_after_dip_finetune_butter_{tag}", f"official_PL_butter_{tag}"),
        "butter_v4_dip": get_eval_row(root, f"dip_test_after_dip_finetune_butter_{tag}", f"newpl_v4_init36_butter_{tag}"),
        "butter_v4_tc": get_eval_row(root, f"tc_test_after_dip_finetune_butter_{tag}", f"newpl_v4_init36_butter_{tag}"),
        "butter_raw_v5_dip": get_eval_row(root, f"dip_test_after_dip_finetune_butter_{tag}", f"newpl_v5_raw_dip_butter_{tag}"),
        "butter_raw_v5_tc": get_eval_row(root, f"tc_test_after_dip_finetune_butter_{tag}", f"newpl_v5_raw_dip_butter_{tag}"),
        "butter_trained_dip": get_eval_row(root, f"dip_test_after_dip_finetune_butter_{tag}", f"newpl_v5_butteracc_dip_best_{tag}"),
        "butter_trained_tc": get_eval_row(root, f"tc_test_after_dip_finetune_butter_{tag}", f"newpl_v5_butteracc_dip_best_{tag}"),
        "butter_trained_amass_dip": get_eval_row(root, f"dip_test_after_amass_pretrain_butter_{tag}", f"newpl_v5_butteracc_amass_best_{tag}"),
        "butter_trained_amass_tc": get_eval_row(root, f"tc_test_after_amass_pretrain_butter_{tag}", f"newpl_v5_butteracc_amass_best_{tag}"),
    }
    return versions


def format_row(label, row):
    if not row:
        return f"- {label}: not available"
    return (
        f"- {label}: pRB L1 {row['pRB_L1_cm']:.6f} cm, "
        f"pRB L2 {row['pRB_L2_cm']:.6f} cm, "
        f"gR1 {row['gR1_angle_deg']:.6f} deg"
    )


def write_summary_md(path, selection, final_rows):
    lines = [
        "# NewPL v5 ButterAcc Summary",
        "",
        "Protocol: causal Butterworth low-pass on `aM` only, then AMASS pretrain -> DIP fine-tune -> module-level DIP/TotalCapture eval.",
        "",
        "Realtime contract: causal zero-lookahead filter, `lookahead_frames=0`, `latency_ms=0`; `wM/RMB` unchanged; `aM_raw` kept for audit.",
        "",
        f"Selection status: {selection['status']}.",
        f"Selected cutoff: {selection['selected_cutoff_hz'] if selection['selected_cutoff_hz'] is not None else 'none'} Hz.",
        f"Reason: {selection['reason']}.",
        "",
        "## Input-only sweep",
        "",
    ]
    for row in selection["input_only_rows"]:
        dip = row["dip_official_butter"]
        tc = row["tc_official_butter"]
        dip_text = "not available" if not dip else f"DIP pRB L2 {dip['pRB_L2_cm']:.6f} cm, gR1 {dip['gR1_angle_deg']:.6f} deg"
        tc_text = "not available" if not tc else f"TC pRB L2 {tc['pRB_L2_cm']:.6f} cm, gR1 {tc['gR1_angle_deg']:.6f} deg"
        lines.append(f"- {row['tag']}: {dip_text}; {tc_text}; guard={row['passes_tc_guard']} ({row['selection_reason']})")
    if final_rows:
        lines.extend(["", "## Final trained checkpoint", ""])
        for key, label in [
            ("raw_official_dip", "DIP raw official_PL"),
            ("butter_official_dip", "DIP Butter official_PL"),
            ("butter_v4_dip", "DIP Butter newpl_v4_init36"),
            ("butter_raw_v5_dip", "DIP Butter raw newpl_v5_dip"),
            ("butter_trained_dip", "DIP Butter trained newpl_v5_butteracc"),
            ("raw_official_tc", "TC raw official_PL"),
            ("butter_official_tc", "TC Butter official_PL"),
            ("butter_v4_tc", "TC Butter newpl_v4_init36"),
            ("butter_raw_v5_tc", "TC Butter raw newpl_v5_dip"),
            ("butter_trained_tc", "TC Butter trained newpl_v5_butteracc"),
        ]:
            lines.append(format_row(label, final_rows.get(key)))
    path.write_text("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize NewPL v5 ButterAcc experiment and choose cutoff.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--cutoffs", default="8 10 12")
    parser.add_argument("--force-cutoff-hz", default="")
    parser.add_argument("--tc-margin-cm", type=float, default=0.10)
    parser.add_argument("--raw-baseline-summary", default=str(RAW_BASELINE_SUMMARY))
    parser.add_argument("--centered-summary", default=str(CENTERED_SMOOTH_SUMMARY))
    args = parser.parse_args()
    root = Path(args.root)
    raw_summary = load_json(args.raw_baseline_summary)
    if raw_summary is None:
        raise FileNotFoundError(args.raw_baseline_summary)
    cutoffs = [float(item) for item in args.cutoffs.replace(",", " ").split()]
    selection = select_cutoff(
        root,
        cutoffs,
        raw_summary,
        force_cutoff_hz=args.force_cutoff_hz,
        tc_margin_cm=args.tc_margin_cm,
    )
    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    final_rows = final_block(root, selection["selected_tag"], raw_summary)
    summary = {
        "status": "ok",
        "root": str(root),
        "protocol": "causal Butterworth aM -> AMASS pretrain -> DIP fine-tune -> module eval",
        "filter_contract": {
            "mode": "causal_butterworth",
            "fs_hz": 60.0,
            "order": 2,
            "lookahead_frames": 0,
            "latency_ms": 0.0,
            "aM_raw_preserved": True,
            "wM_RMB_unchanged": True,
        },
        "dip_trans_loss_used": False,
        "full_pipeline_11_metrics": "not measured",
        "selection": selection,
        "final_rows": final_rows,
        "raw_baseline_summary": str(args.raw_baseline_summary),
        "centered_summary": str(args.centered_summary),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_summary_md(root / "summary.md", selection, final_rows)
    print(json.dumps({
        "status": "ok",
        "selection_json": str(selection_path),
        "summary_json": str(root / "summary.json"),
        "selected_cutoff_hz": selection["selected_cutoff_hz"],
        "selected_tag": selection["selected_tag"],
        "selection_status": selection["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
