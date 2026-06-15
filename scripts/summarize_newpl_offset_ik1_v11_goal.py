#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SECTION_TITLE = "NewPL-offset v6 and NewIK1 v11 control-only training (2026-06-09)"


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {"status": "not available", "path": str(path)}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"status": "parse_failed", "path": str(path), "error": str(exc)}


def first_metric(summary, dataset, version, key):
    for row in summary.get("rows", []):
        if row.get("dataset") == dataset and row.get("version") == version:
            return row.get(key)
    return None


def fmt(value):
    if value is None:
        return "not available"
    if isinstance(value, str):
        return value
    return f"{float(value):.6f}"


def metric(summary, dataset, version, key):
    return first_metric(summary, dataset, version, key)


def delta_text(value, ref, unit):
    if value is None or ref is None:
        return "not available"
    delta = float(value) - float(ref)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.6f} {unit}"


def pl_table(main_summary, acc_summary):
    lines = [
        "| Dataset | Version | pRB L2 cm ↓ | gR1 angle deg ↓ | Notes |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for dataset in ("amass", "dip_test", "tc_test"):
        for version, source, notes in (
            ("official PL baseline", main_summary, "official baseline from cache"),
            ("newpl_v4_init36 baseline", main_summary, "not available if checkpoint path is unset"),
            ("newpl_v5_dip_best", main_summary, "prior best official-route NewPL"),
            ("canonical_control_dip_best", main_summary, "canonical control target baseline"),
            ("newpl_offset_v6_best", main_summary, "control-only offset-aware"),
            ("newpl_offset_v6_acc_aux_best", acc_summary, "offset-aware with IMU acceleration auxiliary"),
        ):
            lookup_version = "newpl_offset_v6_best" if version == "newpl_offset_v6_acc_aux_best" else version
            lines.append(
                f"| {dataset} | {version} | "
                f"{fmt(first_metric(source, dataset, lookup_version, 'pRB_L2_cm'))} | "
                f"{fmt(first_metric(source, dataset, lookup_version, 'gR1_angle_deg'))} | "
                f"{notes} |"
            )
    return lines


def offset_sensitivity_lines(summary, label):
    rows = summary.get("offset_sensitivity", [])
    if not rows:
        return [f"- {label}: offset sensitivity not available."]
    lines = []
    for dataset in ("dip_test", "tc_test"):
        row = next((r for r in rows if r.get("dataset") == dataset and r.get("version") == "newpl_offset_v6_best"), None)
        if row is None:
            lines.append(f"- {label} {dataset}: offset sensitivity not available.")
            continue
        lines.append(
            f"- {label} {dataset}: zero-offset minus true-offset pRB "
            f"{fmt(row.get('zero_minus_good_pRB_cm'))} cm, gR1 "
            f"{fmt(row.get('zero_minus_good_gR1_deg'))} deg; this is too small to prove strong offset usage."
        )
    return lines


def ik1_table(summary):
    lines = [
        "| Dataset | Version | pRJ L2 cm ↓ | leaf pRJ L2 cm ↓ | gR2 angle deg ↓ | Notes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for dataset in ("amass", "dip_val", "dip_test", "tc_test"):
        for version in ("official IK1 baseline", "newik1_v10_stage_c_best", "newik1_v11_best", "newik1_v11_last"):
            lines.append(
                f"| {dataset} | {version} | "
                f"{fmt(first_metric(summary, dataset, version, 'pRJ_L2_cm'))} | "
                f"{fmt(first_metric(summary, dataset, version, 'leaf_pRJ_L2_cm'))} | "
                f"{fmt(first_metric(summary, dataset, version, 'gR2_angle_deg'))} | "
                f"canonical-control NewPL cache |"
            )
    return lines


def measured_conclusions(main_summary, acc_summary, ik1_summary):
    lines = [
        "## Measured Conclusions",
        "",
        "### NewPL-offset v6",
        "",
    ]
    for dataset in ("amass", "dip_test", "tc_test"):
        v6_p = metric(main_summary, dataset, "newpl_offset_v6_best", "pRB_L2_cm")
        off_p = metric(main_summary, dataset, "official PL baseline", "pRB_L2_cm")
        v5_p = metric(main_summary, dataset, "newpl_v5_dip_best", "pRB_L2_cm")
        canon_p = metric(main_summary, dataset, "canonical_control_dip_best", "pRB_L2_cm")
        v6_g = metric(main_summary, dataset, "newpl_offset_v6_best", "gR1_angle_deg")
        off_g = metric(main_summary, dataset, "official PL baseline", "gR1_angle_deg")
        v5_g = metric(main_summary, dataset, "newpl_v5_dip_best", "gR1_angle_deg")
        canon_g = metric(main_summary, dataset, "canonical_control_dip_best", "gR1_angle_deg")
        lines.append(
            f"- {dataset}: `newpl_offset_v6_best` pRB delta vs official/v5/canonical = "
            f"{delta_text(v6_p, off_p, 'cm')} / {delta_text(v6_p, v5_p, 'cm')} / {delta_text(v6_p, canon_p, 'cm')}; "
            f"gR1 delta = {delta_text(v6_g, off_g, 'deg')} / {delta_text(v6_g, v5_g, 'deg')} / {delta_text(v6_g, canon_g, 'deg')}."
        )
    lines.extend([
        "- Verdict: `newpl_offset_v6_best` is close to the canonical-control baseline but does not clearly beat the prior `newpl_v5_dip_best` or the official PL baseline across pRB and gR1. Do not promote it as the selected PL mainline.",
        "- `newpl_offset_v6_acc_aux_best` is not adopted: it gives a small AMASS pRB gain, but DIP/TC pRB and gR1 are not consistently better than the control-only branch or the prior PL baselines.",
        "",
        "### Offset / IMU Acceleration Validation",
        "",
        *offset_sensitivity_lines(main_summary, "control-only"),
        *offset_sensitivity_lines(acc_summary, "acc-aux"),
        "- Verdict: the current offset-aware NewPL variants do not yet demonstrate meaningful dependence on `r_JS`; the IMU acceleration auxiliary option remains an ablation, not a selected change.",
        "",
        "### NewIK1 v11",
        "",
    ])
    for dataset in ("dip_test", "tc_test"):
        v11_p = metric(ik1_summary, dataset, "newik1_v11_best", "pRJ_L2_cm")
        off_p = metric(ik1_summary, dataset, "official IK1 baseline", "pRJ_L2_cm")
        v10_p = metric(ik1_summary, dataset, "newik1_v10_stage_c_best", "pRJ_L2_cm")
        v11_leaf = metric(ik1_summary, dataset, "newik1_v11_best", "leaf_pRJ_L2_cm")
        off_leaf = metric(ik1_summary, dataset, "official IK1 baseline", "leaf_pRJ_L2_cm")
        v11_g = metric(ik1_summary, dataset, "newik1_v11_best", "gR2_angle_deg")
        off_g = metric(ik1_summary, dataset, "official IK1 baseline", "gR2_angle_deg")
        lines.append(
            f"- {dataset}: `newik1_v11_best` pRJ delta vs official/v10 = "
            f"{delta_text(v11_p, off_p, 'cm')} / {delta_text(v11_p, v10_p, 'cm')}; "
            f"leaf pRJ delta vs official = {delta_text(v11_leaf, off_leaf, 'cm')}; "
            f"gR2 delta vs official = {delta_text(v11_g, off_g, 'deg')}."
        )
    lines.extend([
        "- Verdict: v11 improves pRJ on DIP test and TotalCapture test, and improves TotalCapture leaf pRJ, but it regresses gR2 versus the official IK1 baseline and has mixed leaf behavior on DIP. Keep it as diagnostic evidence; do not connect it to the full pipeline until gR2 is fixed or a downstream run justifies the tradeoff.",
        "",
        "### Protocol Notes",
        "",
        "- No full-pipeline 11 metrics were run for this goal.",
        "- DIP trans/root velocity/global trajectory GT was not used.",
        "- `newpl_v4_init36 baseline` rows are marked `not available` in this specific canonical-control summary because the checkpoint path was not provided to the runner; historical v4 results remain recorded in earlier sections.",
    ])
    return lines


def upsert_section(path, heading, body, level="##"):
    path = Path(path)
    text = path.read_text(errors="replace") if path.exists() else ""
    marker = f"{level} {heading}"
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in heading).strip("-")
    begin = f"<!-- BEGIN {slug} -->"
    end = f"<!-- END {slug} -->"
    block = f"{begin}\n{marker}\n\n{body.strip()}\n{end}\n"
    if begin in text and end in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start].rstrip() + "\n\n" + block + "\n" + text[stop:].lstrip()
    elif marker in text:
        # Legacy migration for this summary. The body contains nested headings,
        # so the old same-level heading scan could leave duplicate tail blocks.
        start = text.index(marker)
        text = text[:start].rstrip() + "\n\n" + block + "\n"
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Summarize and write back the NewPL-offset/IK1 v11 goal.")
    parser.add_argument("--newpl-main-summary", type=Path, required=True)
    parser.add_argument("--newpl-acc-summary", type=Path, required=True)
    parser.add_argument("--ik1-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--write-docs", action="store_true")
    args = parser.parse_args()

    main_summary = read_json(args.newpl_main_summary)
    acc_summary = read_json(args.newpl_acc_summary)
    ik1_summary = read_json(args.ik1_summary)
    pl_lines = pl_table(main_summary, acc_summary)
    ik1_lines = ik1_table(ik1_summary)
    result = {
        "status": "ok",
        "newpl_main_summary": str(args.newpl_main_summary),
        "newpl_acc_summary": str(args.newpl_acc_summary),
        "ik1_summary": str(args.ik1_summary),
        "pl_table": pl_lines,
        "ik1_table": ik1_lines,
        "measured_conclusions": measured_conclusions(main_summary, acc_summary, ik1_summary),
        "conclusion_policy": [
            "newpl_offset_v6 must beat or match official/v5/canonical on pRB and not degrade gR1 before promotion.",
            "acc auxiliary is accepted only if pRB/gR1/control metrics improve, not merely acceleration reconstruction.",
            "newik1_v11 is accepted only if pRJ/leaf_pRJ/gR2 improve over official IK1 baseline and v10 on canonical-control caches.",
            "DIP trans/root velocity is not used.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    md = "\n".join([
        f"# {SECTION_TITLE}",
        "",
        "## PL Module Comparison",
        "",
        *pl_lines,
        "",
        "## IK1 Module Comparison",
        "",
        *ik1_lines,
        "",
        "## Contracts",
        "",
        "- NewPL output remains `pRB[15] + gR1[3]`.",
        "- IK1 output remains `pRJ[69] + gR2[3]`.",
        "- Derivative and second-derivative loss terms are disabled; fitted GT control-point losses are used instead.",
        "- `r_JS` is the IMU origin relative to mapped joint `J`, expressed in the joint-local frame; `p_WS=p_WJ+R_WJ@r_JS`.",
        "- DIP trans/root velocity/global trajectory GT is not used.",
        "",
        *measured_conclusions(main_summary, acc_summary, ik1_summary),
        "",
    ])
    args.output_md.write_text(md)
    if args.write_docs:
        upsert_section("EXPERIMENT_LOG.md", SECTION_TITLE, md, level="##")
        upsert_section("PROJECT_STATUS.md", "NewPL-offset v6 / NewIK1 v11 control-only", md, level="###")
        upsert_section("RECENT_REPLACEMENT_VERSIONS.md", "NewPL-offset v6 / NewIK1 v11 control-only", md, level="###")
    print(json.dumps({"status": "ok", "output_json": str(args.output_json), "output_md": str(args.output_md), "docs_written": bool(args.write_docs)}, indent=2))


if __name__ == "__main__":
    main()
