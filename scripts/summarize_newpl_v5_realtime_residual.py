import argparse
import json
from pathlib import Path


RAW_BASELINE_ROOT = Path("data/experiments/newpl_v5_official_protocol_20260607_tuned")


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_from_eval(payload, which):
    if not payload or payload.get("status") != "ok":
        return None
    aggregate = payload.get("aggregate", {})
    leaf = aggregate.get("leaf_position_error_cm", {}).get(which, {})
    grav = aggregate.get("gravity_angle_deg", {}).get(which, {})
    return {
        "pRB_L2_cm": leaf.get("mean"),
        "pRB_L2_cm_p95": leaf.get("p95"),
        "gR1_angle_deg": grav.get("mean"),
        "gR1_angle_deg_p95": grav.get("p95"),
        "num_sequences": aggregate.get("num_sequences"),
        "num_frames": aggregate.get("num_frames"),
        "all_finite": aggregate.get("all_finite"),
    }


def raw_official_row(raw_summary, key):
    rows = raw_summary.get(key, {}).get("pl_output_comparison_table", [])
    for row in rows:
        if row.get("Version") == "official_PL":
            return {
                "pRB_L2_cm": _float(row.get("pRB L2 cm \u2193")),
                "gR1_angle_deg": _float(row.get("gR1 angle deg \u2193")),
            }
    return None


def raw_summary_row(raw_summary, key, version):
    rows = raw_summary.get(key, {}).get("pl_output_comparison_table", [])
    for row in rows:
        if row.get("Version") == version:
            return {
                "pRB_L2_cm": _float(row.get("pRB L2 cm \u2193")),
                "gR1_angle_deg": _float(row.get("gR1 angle deg \u2193")),
            }
    return None


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def delta(candidate, baseline, key):
    if not candidate or not baseline:
        return None
    a = candidate.get(key)
    b = baseline.get(key)
    if a is None or b is None:
        return None
    return a - b


def fmt(value, unit=""):
    if value is None:
        return "not available"
    return f"{float(value):.6f}{unit}"


def fmt_delta(value, unit=""):
    if value is None:
        return "not available"
    sign = "+" if value > 0 else ""
    return f"{sign}{float(value):.6f}{unit}"


def eval_payload(root, filename):
    path = root / "eval" / filename
    return {"path": str(path), "data": load_json(path)}


def build_block(root, raw_summary, dataset, stage, candidate_file, raw_key, raw_ref_version):
    payload = eval_payload(root, candidate_file)
    candidate = metric_from_eval(payload["data"], "new")
    official = metric_from_eval(payload["data"], "original") or raw_official_row(raw_summary, raw_key)
    raw_ref = raw_summary_row(raw_summary, raw_key, raw_ref_version)
    return {
        "dataset": dataset,
        "stage": stage,
        "candidate_eval_json": payload["path"],
        "official_raw": official,
        "raw_reference": raw_ref,
        "raw_reference_source": str(RAW_BASELINE_ROOT / "summary.json"),
        "raw_reference_note": (
            "Historical reference from a different experiment root/cache. "
            "Use only as context unless the checkpoint is re-evaluated on the same cache/protocol."
        ),
        "candidate": candidate,
        "candidate_delta_vs_official": {
            "pRB_L2_cm": delta(candidate, official, "pRB_L2_cm"),
            "gR1_angle_deg": delta(candidate, official, "gR1_angle_deg"),
        },
        "candidate_delta_vs_raw_reference": {
            "pRB_L2_cm": delta(candidate, raw_ref, "pRB_L2_cm"),
            "gR1_angle_deg": delta(candidate, raw_ref, "gR1_angle_deg"),
        },
    }


def block_lines(block):
    lines = [f"## {block['dataset']} {block['stage']}", ""]
    official = block["official_raw"]
    raw_ref = block["raw_reference"]
    candidate = block["candidate"]
    lines.append(
        f"- raw official PL: pRB L2 {fmt(official.get('pRB_L2_cm') if official else None, ' cm')}, "
        f"gR1 {fmt(official.get('gR1_angle_deg') if official else None, ' deg')}"
    )
    lines.append(
        f"- historical raw-reference NewPL: pRB L2 {fmt(raw_ref.get('pRB_L2_cm') if raw_ref else None, ' cm')}, "
        f"gR1 {fmt(raw_ref.get('gR1_angle_deg') if raw_ref else None, ' deg')} "
        "(different experiment root/cache; context only)"
    )
    lines.append(
        f"- realtime residual NewPL: pRB L2 {fmt(candidate.get('pRB_L2_cm') if candidate else None, ' cm')}, "
        f"gR1 {fmt(candidate.get('gR1_angle_deg') if candidate else None, ' deg')}"
    )
    d = block["candidate_delta_vs_official"]
    lines.append(
        f"- candidate vs raw official: pRB L2 {fmt_delta(d['pRB_L2_cm'], ' cm')}, "
        f"gR1 {fmt_delta(d['gR1_angle_deg'], ' deg')}"
    )
    d = block["candidate_delta_vs_raw_reference"]
    lines.append(
        f"- candidate vs historical raw reference: pRB L2 {fmt_delta(d['pRB_L2_cm'], ' cm')}, "
        f"gR1 {fmt_delta(d['gR1_angle_deg'], ' deg')} (not a fairness comparison)"
    )
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Summarize NewPL realtime smooth+residual experiment.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--raw-baseline-summary", default=str(RAW_BASELINE_ROOT / "summary.json"))
    parser.add_argument("--filter-mode", default="causal_iir")
    parser.add_argument("--cutoff-hz", type=float, default=20.0)
    args = parser.parse_args()

    root = Path(args.root)
    raw_summary = load_json(args.raw_baseline_summary) or {}
    blocks = [
        build_block(
            root,
            raw_summary,
            "AMASS proxy",
            "after AMASS pretrain",
            "amass_after_amass_pretrain_realtime_residual.json",
            "amass_after_amass_pretrain",
            "newpl_v5_amass_best",
        ),
        build_block(
            root,
            raw_summary,
            "DIP-IMU test",
            "after AMASS pretrain",
            "dip_test_after_amass_pretrain_realtime_residual.json",
            "dip_test_after_amass_pretrain",
            "newpl_v5_amass_best",
        ),
        build_block(
            root,
            raw_summary,
            "TotalCapture test",
            "after AMASS pretrain",
            "tc_test_after_amass_pretrain_realtime_residual.json",
            "tc_test_after_amass_pretrain",
            "newpl_v5_amass_best",
        ),
        build_block(
            root,
            raw_summary,
            "DIP-IMU test",
            "after DIP fine-tune",
            "dip_test_after_dip_finetune_realtime_residual.json",
            "dip_test_after_dip_finetune",
            "newpl_v5_dip_best",
        ),
        build_block(
            root,
            raw_summary,
            "TotalCapture test",
            "after DIP fine-tune",
            "tc_test_after_dip_finetune_realtime_residual.json",
            "tc_test_after_dip_finetune",
            "newpl_v5_dip_best",
        ),
    ]
    summary = {
        "status": "ok",
        "root": str(root),
        "variant": "newpl_v5_realtime_smooth_residual",
        "input_contract": (
            "102D frame input: aRB_smooth[18] + aRB_residual_raw_minus_smooth[18] + "
            "wRB[18] + RRB[45] + gR0[3]. Output remains pRB[15]+gR1[3]."
        ),
        "filter": {
            "mode": args.filter_mode,
            "cutoff_hz": float(args.cutoff_hz),
            "lookahead_frames": 0,
        },
        "protocol": "AMASS pretrain -> DIP train fine-tune -> module-level DIP/TotalCapture test; no TotalCapture fine-tune.",
        "full_pipeline_11_metrics": "not measured",
        "dip_trans_loss_used": False,
        "raw_baseline_summary": str(args.raw_baseline_summary),
        "metric_namespace_note": (
            "raw_reference rows are imported from the raw_baseline_summary as historical context. "
            "They are not same-cache fairness baselines for the realtime smooth+residual cache."
        ),
        "comparisons": blocks,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# NewPL v5 Realtime Smooth+Residual Summary",
        "",
        "Protocol: AMASS pretrain -> DIP-IMU fine-tune -> DIP-IMU and TotalCapture module-level test.",
        "",
        "Input contract: `aRB_smooth[18] + aRB_residual_raw_minus_smooth[18] + wRB[18] + RRB[45] + gR0[3] = 102D`.",
        f"Filter: `{args.filter_mode}`, cutoff `{args.cutoff_hz:g} Hz`, zero lookahead.",
        "",
        "Full-pipeline 11 metrics: not measured.",
        "DIP translation/root velocity loss: not used.",
        "Raw-reference NewPL rows are historical context from a different experiment root/cache; they are not same-cache fairness comparisons.",
        "",
    ]
    for block in blocks:
        lines.extend(block_lines(block))
    (root / "summary.md").write_text("\n".join(lines).rstrip() + "\n")
    print(json.dumps({
        "status": "ok",
        "summary_json": str(root / "summary.json"),
        "summary_md": str(root / "summary.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
