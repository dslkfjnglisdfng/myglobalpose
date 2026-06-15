#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SPLITS = ("dip_val", "dip_test", "tc_test")
CHECKPOINTS = ("best", "last")
WRONG_VARIANTS = ("zero", "roll_sensors", "other_sequence", "negate")


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_mean(block, key):
    return block[key]["mean"]


def module_row(root, ckpt, split):
    eval_dir = root / "eval"
    path = eval_dir / f"{ckpt}_{split}_module_pl_accuracy.json"
    if not path.exists() and ckpt == "best":
        path = eval_dir / f"{split}_module_pl_accuracy_best.json"
    data = read_json(path)
    if data is None or data.get("status") != "ok":
        return None
    agg = data["aggregate"]
    leaf = agg["leaf_position_error_cm"]
    grav = agg["gravity_angle_deg"]
    return {
        "checkpoint": ckpt,
        "split": split,
        "path": str(path),
        "num_sequences": agg["num_sequences"],
        "num_frames": agg["num_frames"],
        "all_finite": agg["all_finite"],
        "original_leaf_cm_mean": metric_mean(leaf, "original"),
        "new_leaf_cm_mean": metric_mean(leaf, "new"),
        "delta_leaf_cm_mean": metric_mean(leaf, "delta_new_minus_original"),
        "original_gR1_deg_mean": metric_mean(grav, "original"),
        "new_gR1_deg_mean": metric_mean(grav, "new"),
        "delta_gR1_deg_mean": metric_mean(grav, "delta_new_minus_original"),
    }


def swap_row(root, ckpt, split):
    eval_dir = root / "eval"
    path = eval_dir / f"{ckpt}_{split}_offset_swap.json"
    if not path.exists() and ckpt == "best":
        path = eval_dir / f"{split}_offset_swap_best.json"
    data = read_json(path)
    if data is None or data.get("status") != "ok":
        return None
    delta = data["aggregate"]["delta_vs_good"]
    row = {
        "checkpoint": ckpt,
        "split": split,
        "path": str(path),
        "num_sequences": data["num_sequences"],
        "swap_feature_offset": bool(data.get("swap_feature_offset", False)),
    }
    for variant in WRONG_VARIANTS:
        if variant not in delta:
            continue
        row[f"{variant}_minus_good_pRB_cm_mean"] = metric_mean(delta[variant], "pRB_cm")
        row[f"{variant}_minus_good_gR1_deg_mean"] = metric_mean(delta[variant], "gR1_deg")
        row[f"{variant}_minus_good_pl_gt_loss_mean"] = metric_mean(delta[variant], "pl_gt_loss")
    return row


def train_summary(root, rel):
    path = root / rel / "train_result.json"
    data = read_json(path)
    if data is None:
        return {"path": str(path), "status": "missing"}
    return {
        "path": str(path),
        "status": data.get("status"),
        "best_epoch": data.get("best_epoch"),
        "best_loss": data.get("best_loss"),
        "selection_metric": data.get("selection_metric"),
        "stopped_early": data.get("stopped_early"),
        "stop_epoch": data.get("stop_epoch"),
        "num_train_sequences": data.get("num_train_sequences"),
        "num_val_sequences": data.get("num_val_sequences"),
    }


def sensitivity_gate(row, p_threshold, g_threshold):
    if row is None:
        return False, "missing"
    if not row.get("swap_feature_offset", False):
        return False, "missing feature-level rJS perturbation; result is init-feature-only"
    best = {"variant": None, "pRB_cm": float("-inf"), "gR1_deg": float("-inf")}
    for variant in WRONG_VARIANTS:
        p = row.get(f"{variant}_minus_good_pRB_cm_mean")
        g = row.get(f"{variant}_minus_good_gR1_deg_mean")
        if p is not None and p > best["pRB_cm"]:
            best["pRB_cm"] = p
            best["variant"] = variant
        if g is not None and g > best["gR1_deg"]:
            best["gR1_deg"] = g
    passed = best["pRB_cm"] >= p_threshold or best["gR1_deg"] >= g_threshold
    reason = (
        f"max wrong-minus-good pRB={best['pRB_cm']:.6g} cm, "
        f"gR1={best['gR1_deg']:.6g} deg"
    )
    return passed, reason


def threshold_reason(row, metric_key, threshold, unit):
    if row is None:
        return "missing"
    value = row[metric_key]
    op = "<=" if value <= threshold else ">"
    return f"{metric_key}={value:.6f} {unit} {op} {threshold:.6f}"


def write_markdown(path, summary):
    lines = [
        "# rJS-sensitive NewPL Summary",
        "",
        f"Root: `{summary['root']}`",
        f"Coordinate contract: {summary['coordinate_contract']}",
        "",
        "## Gates",
        "",
        "| Gate | Passed | Reason |",
        "| --- | --- | --- |",
    ]
    for key, value in summary["gates"].items():
        lines.append(f"| {key} | {value['passed']} | {value['reason']} |")
    lines.extend(["", "## Module Metrics", "", "| checkpoint | split | new leaf cm | delta leaf cm | new gR1 deg | delta gR1 deg |"])
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in summary["module_rows"]:
        lines.append(
            f"| {row['checkpoint']} | {row['split']} | "
            f"{row['new_leaf_cm_mean']:.6f} | {row['delta_leaf_cm_mean']:.6f} | "
            f"{row['new_gR1_deg_mean']:.6f} | {row['delta_gR1_deg_mean']:.6f} |"
        )
    lines.extend(["", "## Sensitivity Metrics", "", "| checkpoint | split | feature rJS | zero pRB | roll pRB | other pRB | negate pRB | max gR1 |"])
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["swap_rows"]:
        max_g = max(
            row.get(f"{variant}_minus_good_gR1_deg_mean", float("-inf"))
            for variant in WRONG_VARIANTS
        )
        lines.append(
            f"| {row['checkpoint']} | {row['split']} | {row.get('swap_feature_offset', False)} | "
            f"{row.get('zero_minus_good_pRB_cm_mean', 0.0):.6g} | "
            f"{row.get('roll_sensors_minus_good_pRB_cm_mean', 0.0):.6g} | "
            f"{row.get('other_sequence_minus_good_pRB_cm_mean', 0.0):.6g} | "
            f"{row.get('negate_minus_good_pRB_cm_mean', 0.0):.6g} | "
            f"{max_g:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize rJS-sensitive NewPL training/evaluation gates.")
    parser.add_argument("--root", type=Path, default=Path("data/experiments/rjs_sensitive_newpl_20260608"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--sensitivity-prb-threshold", type=float, default=0.02)
    parser.add_argument("--sensitivity-gr-threshold", type=float, default=0.05)
    parser.add_argument("--dip-prb-threshold", type=float, default=6.419)
    parser.add_argument("--tc-prb-threshold", type=float, default=6.5573)
    args = parser.parse_args()

    root = args.root
    output_json = args.output_json or root / "summary.json"
    output_md = args.output_md or root / "summary.md"

    module_rows = [
        row for ckpt in CHECKPOINTS for split in SPLITS
        if (row := module_row(root, ckpt, split)) is not None
    ]
    swap_rows = [
        row for ckpt in CHECKPOINTS for split in ("dip_test", "tc_test")
        if (row := swap_row(root, ckpt, split)) is not None
    ]

    best_dip_swap = next((r for r in swap_rows if r["checkpoint"] == "best" and r["split"] == "dip_test"), None)
    best_tc_swap = next((r for r in swap_rows if r["checkpoint"] == "best" and r["split"] == "tc_test"), None)
    dip_sens_pass, dip_sens_reason = sensitivity_gate(best_dip_swap, args.sensitivity_prb_threshold, args.sensitivity_gr_threshold)
    tc_sens_pass, tc_sens_reason = sensitivity_gate(best_tc_swap, args.sensitivity_prb_threshold, args.sensitivity_gr_threshold)

    best_dip_module = next((r for r in module_rows if r["checkpoint"] == "best" and r["split"] == "dip_test"), None)
    best_tc_module = next((r for r in module_rows if r["checkpoint"] == "best" and r["split"] == "tc_test"), None)
    dip_module_pass = bool(best_dip_module and best_dip_module["new_leaf_cm_mean"] <= args.dip_prb_threshold)
    tc_module_pass = bool(best_tc_module and best_tc_module["new_leaf_cm_mean"] <= args.tc_prb_threshold)

    gates = {
        "sensitivity_dip_best": {"passed": dip_sens_pass, "reason": dip_sens_reason},
        "sensitivity_tc_best": {"passed": tc_sens_pass, "reason": tc_sens_reason},
        "module_dip_best": {
            "passed": dip_module_pass,
            "reason": threshold_reason(best_dip_module, "new_leaf_cm_mean", args.dip_prb_threshold, "cm"),
        },
        "module_tc_best": {
            "passed": tc_module_pass,
            "reason": threshold_reason(best_tc_module, "new_leaf_cm_mean", args.tc_prb_threshold, "cm"),
        },
    }
    gates["full_pipeline_allowed"] = {
        "passed": all(gates[key]["passed"] for key in ("sensitivity_dip_best", "sensitivity_tc_best", "module_dip_best", "module_tc_best")),
        "reason": "requires sensitivity and module gates to pass",
    }

    summary = {
        "status": "ok",
        "root": str(root),
        "coordinate_contract": "r_JS is joint-local IMU origin relative to mapped joint J; DIP uses pseudo-rJS only, not GT.",
        "training": {
            "stage_a": train_summary(root, "amass_rjs_sensitive"),
            "stage_b": train_summary(root, "dip_finetune"),
        },
        "thresholds": {
            "sensitivity_prb_cm": args.sensitivity_prb_threshold,
            "sensitivity_gR1_deg": args.sensitivity_gr_threshold,
            "dip_prb_cm": args.dip_prb_threshold,
            "tc_prb_cm": args.tc_prb_threshold,
        },
        "gates": gates,
        "module_rows": module_rows,
        "swap_rows": swap_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(output_md, summary)
    print(json.dumps({"status": "ok", "summary_json": str(output_json), "summary_md": str(output_md), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
