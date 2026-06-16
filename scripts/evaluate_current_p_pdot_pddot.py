import argparse
import csv
import json
from pathlib import Path

import torch

from pl_curve import normalize_gravity
from pl_next_control_cache import _central_acceleration, _central_velocity, _shift_next
from pl_next_control_eval import angle_deg, load_version, run_model
from pl_next_control_train import load_next_records


LEAF_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot", "head")
SHIFTS = (-2, -1, 0, 1, 2)


def finite_float(value):
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return value


class VectorMetricAccumulator:
    def __init__(self):
        self.l1_sum = 0.0
        self.l1_count = 0
        self.leaf_l2_sum = [0.0 for _ in LEAF_NAMES]
        self.leaf_l2_count = [0 for _ in LEAF_NAMES]

    def add(self, pred, target, mask, scale):
        if mask is None:
            mask = torch.ones(pred.shape[0], dtype=torch.bool)
        mask = mask.bool().cpu()
        if not bool(mask.any()):
            return
        err = (pred[..., :15] - target[..., :15]).cpu()
        vec_mask = mask.unsqueeze(-1).expand_as(err)
        l1_values = err.abs().masked_select(vec_mask)
        self.l1_sum += float(l1_values.sum() * scale)
        self.l1_count += int(l1_values.numel())
        leaf = err.reshape(err.shape[:-1] + (5, 3)).norm(dim=-1)
        for leaf_idx in range(5):
            values = leaf[..., leaf_idx].masked_select(mask)
            self.leaf_l2_sum[leaf_idx] += float(values.sum() * scale)
            self.leaf_l2_count[leaf_idx] += int(values.numel())

    def result(self, prefix, unit):
        total_leaf_sum = sum(self.leaf_l2_sum)
        total_leaf_count = sum(self.leaf_l2_count)
        per_leaf = {
            LEAF_NAMES[idx]: (
                self.leaf_l2_sum[idx] / self.leaf_l2_count[idx]
                if self.leaf_l2_count[idx]
                else None
            )
            for idx in range(5)
        }
        return {
            f"{prefix}_L1_{unit}": self.l1_sum / self.l1_count if self.l1_count else None,
            f"{prefix}_L2_{unit}": total_leaf_sum / total_leaf_count if total_leaf_count else None,
            f"{prefix}_per_leaf_L2_{unit}": per_leaf,
            f"{prefix}_num_leaf_values": total_leaf_count,
        }


class AngleAccumulator:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, pred, target, mask):
        if mask is None:
            mask = torch.ones(pred.shape[0], dtype=torch.bool)
        mask = mask.bool().cpu()
        if not bool(mask.any()):
            return
        values = angle_deg(pred[..., 15:], target[..., 15:]).cpu().masked_select(mask)
        self.sum += float(values.sum())
        self.count += int(values.numel())

    def result(self, key):
        return {key: self.sum / self.count if self.count else None}


class L2Accumulator:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, pred, target, mask, scale):
        if not bool(mask.any()):
            return
        err = (pred[..., :15] - target[..., :15]).cpu()
        leaf = err.reshape(err.shape[:-1] + (5, 3)).norm(dim=-1)
        values = leaf.masked_select(mask.unsqueeze(-1).expand_as(leaf))
        self.sum += float(values.sum() * scale)
        self.count += int(values.numel())

    def mean(self):
        return self.sum / self.count if self.count else None


def metric_row(pred, target, mask, scale, prefix, unit):
    acc = VectorMetricAccumulator()
    acc.add(pred, target, mask, scale)
    return acc.result(prefix, unit)


def angle_row(pred, target, mask, key):
    acc = AngleAccumulator()
    acc.add(pred, target, mask)
    return acc.result(key)


def make_current_derivative_mask(length):
    mask = torch.zeros(length, dtype=torch.bool)
    if length > 2:
        mask[1:-1] = True
    return mask


def frame_counts(records):
    current = sum(int(record["pl_target"].shape[0]) for record in records)
    current_derivative = sum(int(make_current_derivative_mask(record["pl_target"].shape[0]).sum()) for record in records)
    nxt = sum(int(record["valid_next_mask"].bool().sum()) for record in records)
    return current, current_derivative, nxt


def shifted_mask(length, shift, base_mask):
    idx = torch.arange(length)
    shifted = idx + int(shift)
    return base_mask.bool() & (shifted >= 0) & (shifted < length)


def shifted_target(target, shift):
    idx = torch.arange(target.shape[0]) + int(shift)
    return target[idx.clamp(0, target.shape[0] - 1)]


def evaluate_alignment(pred_p, pred_pd, pred_pdd, record, accumulators):
    length = int(record["pl_target"].shape[0])
    all_mask = torch.ones(length, dtype=torch.bool)
    deriv_mask = make_current_derivative_mask(length)
    metrics = (
        ("current_p", pred_p, record["pl_target"], all_mask, 100.0),
        ("current_pdot_fd", pred_pd, record["gt_pldot"], deriv_mask, 100.0),
        ("current_pddot_fd", pred_pdd, record["gt_plddot"], deriv_mask, 100.0),
    )
    for metric, pred, target, base_mask, scale in metrics:
        for shift in SHIFTS:
            mask = shifted_mask(length, shift, base_mask)
            accumulators[(metric, shift)].add(pred, shifted_target(target, shift), mask, scale)


def seq_metric_values(pred_p, pred_pd, pred_pdd, pred_next_p, pred_next_pd, pred_next_pdd, record):
    length = int(record["pl_target"].shape[0])
    current_mask = torch.ones(length, dtype=torch.bool)
    deriv_mask = make_current_derivative_mask(length)
    next_mask = record["valid_next_mask"].bool()
    row = {}
    row.update(metric_row(pred_p, record["pl_target"], current_mask, 100.0, "current_p", "cm"))
    row.update(metric_row(pred_pd, record["gt_pldot"], deriv_mask, 100.0, "current_pdot", "cm_s"))
    row.update(metric_row(pred_pdd, record["gt_plddot"], deriv_mask, 100.0, "current_pddot", "cm_s2"))
    row.update(angle_row(pred_p, record["pl_target"], current_mask, "current_gR1_angle_deg"))
    row.update(metric_row(pred_next_p, record["pl_target_next"], next_mask, 100.0, "next_p", "cm"))
    row.update(metric_row(pred_next_pd, record["gt_pldot_next"], next_mask, 100.0, "next_pdot", "cm_s"))
    row.update(metric_row(pred_next_pdd, record["gt_plddot_next"], next_mask, 100.0, "next_pddot", "cm_s2"))
    row.update(angle_row(pred_next_p, record["pl_target_next"], next_mask, "next_gR1_angle_deg"))
    return row


@torch.no_grad()
def evaluate_dataset(dataset_label, cache, versions, dt, max_eval_sequences=0, max_frames_per_sequence=0):
    records, manifest = load_next_records(cache, max_sequences=max_eval_sequences)
    if max_frames_per_sequence and max_frames_per_sequence > 0:
        from pl_next_control_eval import truncate_record

        records = [truncate_record(record, max_frames_per_sequence) for record in records]
    current_frames, current_derivative_frames, next_frames = frame_counts(records)
    version_results = []
    per_sequence_rows = []
    per_leaf_rows = []
    alignment_rows = []

    for version in versions:
        current_p_acc = VectorMetricAccumulator()
        current_pd_acc = VectorMetricAccumulator()
        current_pdd_acc = VectorMetricAccumulator()
        current_g_acc = AngleAccumulator()
        next_p_acc = VectorMetricAccumulator()
        next_pd_acc = VectorMetricAccumulator()
        next_pdd_acc = VectorMetricAccumulator()
        next_g_acc = AngleAccumulator()
        alignment_acc = {
            (metric, shift): L2Accumulator()
            for metric in ("current_p", "current_pdot_fd", "current_pddot_fd")
            for shift in SHIFTS
        }
        next_source = None
        current_derivative_source = "central finite difference of output['pl'][..., :15], first/last frames excluded"

        for record in records:
            output = run_model(version, record)
            pred_current = normalize_gravity(output["pl"]).detach().cpu()
            pred_current_pd = _central_velocity(pred_current, dt=dt)
            pred_current_pdd = _central_acceleration(pred_current, dt=dt)
            if "next_pl" in output:
                pred_next = normalize_gravity(output["next_pl"]).detach().cpu()
                pred_next_pd = output["next_pldot"].detach().cpu()
                pred_next_pdd = output["next_plddot"].detach().cpu()
                next_source = "output['next_pl'], output['next_pldot'], output['next_plddot'] decoded from predicted next control"
            else:
                pred_next = pred_current
                pred_next_pd = _shift_next(pred_current_pd)
                pred_next_pdd = _shift_next(pred_current_pdd)
                next_source = "baseline persistence: current output reused as next p; shifted current finite differences for next pdot/pddot"

            length = int(record["pl_target"].shape[0])
            current_mask = torch.ones(length, dtype=torch.bool)
            deriv_mask = make_current_derivative_mask(length)
            next_mask = record["valid_next_mask"].bool()

            current_p_acc.add(pred_current, record["pl_target"], current_mask, 100.0)
            current_pd_acc.add(pred_current_pd, record["gt_pldot"], deriv_mask, 100.0)
            current_pdd_acc.add(pred_current_pdd, record["gt_plddot"], deriv_mask, 100.0)
            current_g_acc.add(pred_current, record["pl_target"], current_mask)
            next_p_acc.add(pred_next, record["pl_target_next"], next_mask, 100.0)
            next_pd_acc.add(pred_next_pd, record["gt_pldot_next"], next_mask, 100.0)
            next_pdd_acc.add(pred_next_pdd, record["gt_plddot_next"], next_mask, 100.0)
            next_g_acc.add(pred_next, record["pl_target_next"], next_mask)
            evaluate_alignment(pred_current, pred_current_pd, pred_current_pdd, record, alignment_acc)

            seq_values = seq_metric_values(
                pred_current,
                pred_current_pd,
                pred_current_pdd,
                pred_next,
                pred_next_pd,
                pred_next_pdd,
                record,
            )
            per_sequence_rows.append({
                "dataset": dataset_label,
                "version": version["name"],
                "sequence": record["name"],
                "source_frames": length,
                "current_position_frames": int(current_mask.sum()),
                "current_derivative_frames": int(deriv_mask.sum()),
                "next_frames": int(next_mask.sum()),
                "current_mask_used": "all current frames for position/gR1",
                "current_derivative_mask_used": "exclude first and last frames",
                "next_mask_used": "valid_next_mask",
                **seq_values,
            })

        current = {}
        current.update(current_p_acc.result("current_p", "cm"))
        current.update(current_pd_acc.result("current_pdot", "cm_s"))
        current.update(current_pdd_acc.result("current_pddot", "cm_s2"))
        current.update(current_g_acc.result("current_gR1_angle_deg"))
        nxt = {}
        nxt.update(next_p_acc.result("next_p", "cm"))
        nxt.update(next_pd_acc.result("next_pdot", "cm_s"))
        nxt.update(next_pdd_acc.result("next_pddot", "cm_s2"))
        nxt.update(next_g_acc.result("next_gR1_angle_deg"))

        for frame_scope, metrics in (("current", current), ("next", nxt)):
            for metric_name, unit_key in (
                (f"{frame_scope}_p", "cm"),
                (f"{frame_scope}_pdot", "cm_s"),
                (f"{frame_scope}_pddot", "cm_s2"),
            ):
                leaf_key = f"{metric_name}_per_leaf_L2_{unit_key}"
                if leaf_key not in metrics:
                    continue
                per_leaf = metrics[leaf_key]
                per_leaf_rows.append({
                    "dataset": dataset_label,
                    "version": version["name"],
                    "frame_scope": frame_scope,
                    "metric": metric_name,
                    "unit": unit_key.replace("_", "/"),
                    "mask_used": (
                        "all current frames"
                        if metric_name == "current_p"
                        else "current_derivative_mask excluding first/last"
                        if frame_scope == "current"
                        else "valid_next_mask"
                    ),
                    "num_evaluated_frames": current_frames if metric_name == "current_p" else current_derivative_frames if frame_scope == "current" else next_frames,
                    **{leaf: per_leaf[leaf] for leaf in LEAF_NAMES},
                    "mean_over_leaves": metrics[f"{metric_name}_L2_{unit_key}"],
                })

        alignment = {}
        for metric in ("current_p", "current_pdot_fd", "current_pddot_fd"):
            shift_values = {shift: alignment_acc[(metric, shift)].mean() for shift in SHIFTS}
            best_shift = min((s for s in SHIFTS if shift_values[s] is not None), key=lambda s: shift_values[s])
            row = {
                "dataset": dataset_label,
                "version": version["name"],
                "metric": metric,
                "unit": "cm" if metric == "current_p" else "cm/s" if metric == "current_pdot_fd" else "cm/s^2",
                "best_shift": best_shift,
                "warning": "WARNING: best temporal alignment is not zero; current-frame accuracy is time-shifted." if best_shift != 0 else "",
            }
            for shift in SHIFTS:
                row[f"shift_{shift:+d}"] = shift_values[shift]
            alignment_rows.append(row)
            alignment[metric] = row

        version_results.append({
            "name": version["name"],
            "kind": version["kind"],
            "path": version.get("path"),
            "notes": version.get("notes"),
            "current_frame": current,
            "next_frame": nxt,
            "alignment_sweep": alignment,
            "current_derivative_source": current_derivative_source,
            "next_source": next_source,
            "num_sequences": len(records),
            "current_eval_frames": current_frames,
            "current_derivative_eval_frames": current_derivative_frames,
            "next_eval_frames": next_frames,
            "masks": {
                "current_position": "all current frames",
                "current_derivative_mask": "valid frames excluding first/last frames needed by finite difference",
                "next": "valid_next_mask",
            },
        })

    return {
        "status": "ok",
        "dataset_label": dataset_label,
        "cache": cache,
        "manifest": manifest,
        "num_sequences": len(records),
        "current_eval_frames": current_frames,
        "current_derivative_eval_frames": current_derivative_frames,
        "next_eval_frames": next_frames,
        "evaluation_contract": {
            "current_output": "output['pl'] = pRB_t[15] + gR1_t[3]",
            "next_output": "output['next_pl'] = predicted pRB_{t+1}[15] + gR1_{t+1}[3]",
            "next_derivatives": "output['next_pldot'] and output['next_plddot'] are decoded from predicted next control via spline when available",
            "current_position_gt": "pl_target[..., :15]",
            "current_velocity_gt": "gt_pldot[..., :15]",
            "current_acceleration_gt": "gt_plddot[..., :15]",
            "current_position_pred": "output['pl'][..., :15]",
            "current_velocity_pred": "central finite difference of output['pl'][..., :15]",
            "current_acceleration_pred": "central finite difference acceleration of output['pl'][..., :15]",
            "current_selection_warning": "The p_pdot_pddot_strong experiment selected by validation normalized next p/pdot/pddot composite; it does not by itself prove current-frame p/pdot/pddot accuracy.",
            "units": {
                "position": "L1 cm, L2 cm",
                "velocity": "L1 cm/s, L2 cm/s",
                "acceleration": "L1 cm/s^2, L2 cm/s^2",
            },
        },
        "versions": version_results,
        "per_sequence_rows": per_sequence_rows,
        "per_leaf_rows": per_leaf_rows,
        "alignment_rows": alignment_rows,
    }


def table_value(value):
    if value is None:
        return "not available"
    return f"{float(value):.6f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(out)


def current_table_rows(dataset_result):
    rows = []
    for version in dataset_result["versions"]:
        cur = version["current_frame"]
        rows.append({
            "dataset": dataset_result["dataset_label"],
            "version": version["name"],
            "current p L2 cm": table_value(cur["current_p_L2_cm"]),
            "current pdot L2 cm/s": table_value(cur["current_pdot_L2_cm_s"]),
            "current pddot L2 cm/s^2": table_value(cur["current_pddot_L2_cm_s2"]),
            "current gR1 deg": table_value(cur["current_gR1_angle_deg"]),
        })
    return rows


def next_table_rows(dataset_result):
    rows = []
    for version in dataset_result["versions"]:
        nxt = version["next_frame"]
        rows.append({
            "dataset": dataset_result["dataset_label"],
            "version": version["name"],
            "next p L2 cm": table_value(nxt["next_p_L2_cm"]),
            "next pdot L2 cm/s": table_value(nxt["next_pdot_L2_cm_s"]),
            "next pddot L2 cm/s^2": table_value(nxt["next_pddot_L2_cm_s2"]),
            "next gR1 deg": table_value(nxt["next_gR1_angle_deg"]),
        })
    return rows


def alignment_table_rows(dataset_result):
    rows = []
    for row in dataset_result["alignment_rows"]:
        rows.append({
            "dataset": row["dataset"],
            "version": row["version"],
            "metric": row["metric"],
            "best shift": row["best_shift"],
            "shift -2": table_value(row["shift_-2"]),
            "shift -1": table_value(row["shift_-1"]),
            "shift 0": table_value(row["shift_+0"]),
            "shift +1": table_value(row["shift_+1"]),
            "shift +2": table_value(row["shift_+2"]),
            "warning": row["warning"],
        })
    return rows


def decide(results, strong_name):
    failures = []
    for result in results:
        strong = next(v for v in result["versions"] if v["name"] == strong_name)
        baselines = [v for v in result["versions"] if v["name"] != strong_name and not v["name"].endswith("_dip_last")]
        for metric in ("current_p_L2_cm", "current_pdot_L2_cm_s", "current_pddot_L2_cm_s2"):
            strong_value = strong["current_frame"][metric]
            best_baseline = min(v["current_frame"][metric] for v in baselines if v["current_frame"][metric] is not None)
            if strong_value is None or strong_value > best_baseline:
                failures.append({
                    "dataset": result["dataset_label"],
                    "metric": metric,
                    "strong": strong_value,
                    "best_baseline": best_baseline,
                })
    if failures:
        return "diagnostic only", failures
    return "promote", []


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, results, strong_name, decision, failures):
    lines = [
        "# Current-frame p/pdot/pddot evaluation",
        "",
        "This pass answers whether the current-frame PL/IMU pRB position, velocity, and acceleration estimates are accurate. It does not use the normalized training composite as evidence of current-frame accuracy.",
        "",
        "## Semantics",
        "",
        "```text",
        "current output:",
        "  output[\"pl\"] = pRB_t[15] + gR1_t[3]",
        "",
        "next output:",
        "  output[\"next_pl\"] = predicted pRB_{t+1}[15] + gR1_{t+1}[3]",
        "",
        "next derivatives:",
        "  output[\"next_pldot\"], output[\"next_plddot\"]",
        "  are decoded from predicted next control via spline.",
        "",
        "Current experiment selection:",
        "  selected by validation normalized next p/pdot/pddot composite.",
        "",
        "Therefore this experiment does not by itself prove current-frame p/pdot/pddot accuracy.",
        "```",
        "",
        "## Masks and Units",
        "",
        "- current position: `output[\"pl\"][..., :15]` vs `pl_target[..., :15]`, all current frames, L1/L2 in cm.",
        "- current velocity: central finite difference of `output[\"pl\"][..., :15]` vs `gt_pldot[..., :15]`, first/last frames excluded, L1/L2 in cm/s.",
        "- current acceleration: central finite difference acceleration of `output[\"pl\"][..., :15]` vs `gt_plddot[..., :15]`, first/last frames excluded, L1/L2 in cm/s^2.",
        "- next position/velocity/acceleration: next-frame predictions against `pl_target_next`, `gt_pldot_next`, and `gt_plddot_next` using `valid_next_mask`.",
        "",
    ]
    for result in results:
        lines.extend([
            f"## {result['dataset_label']}",
            "",
            f"- num sequences: `{result['num_sequences']}`",
            f"- current_eval_frames: `{result['current_eval_frames']}`",
            f"- current_derivative_eval_frames: `{result['current_derivative_eval_frames']}`",
            f"- next_eval_frames: `{result['next_eval_frames']}`",
            "",
            "### Table 1: current-frame accuracy",
            "",
            table(
                ["version", "current p L2 cm", "current pdot L2 cm/s", "current pddot L2 cm/s^2", "current gR1 deg"],
                [{k: v for k, v in row.items() if k != "dataset"} for row in current_table_rows(result)],
            ),
            "",
            "### Table 2: next-frame accuracy",
            "",
            table(
                ["version", "next p L2 cm", "next pdot L2 cm/s", "next pddot L2 cm/s^2", "next gR1 deg"],
                [{k: v for k, v in row.items() if k != "dataset"} for row in next_table_rows(result)],
            ),
            "",
            "### Table 3: alignment sweep",
            "",
            table(
                ["version", "metric", "best shift", "shift -2", "shift -1", "shift 0", "shift +1", "shift +2", "warning"],
                [{k: v for k, v in row.items() if k != "dataset"} for row in alignment_table_rows(result)],
            ),
            "",
        ])
    lines.extend([
        "## Decision",
        "",
        f"Decision: `{decision}`.",
        "",
    ])
    if failures:
        lines.extend([
            "The strong checkpoint fails the requested current-frame non-regression gate:",
            "",
            table(
                ["dataset", "metric", "strong", "best_baseline"],
                [
                    {
                        "dataset": f["dataset"],
                        "metric": f["metric"],
                        "strong": table_value(f["strong"]),
                        "best_baseline": table_value(f["best_baseline"]),
                    }
                    for f in failures
                ],
            ),
            "",
        ])
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate current-frame and next-frame p/pdot/pddot metrics for NewPL next-control experiments.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dip-cache", required=True)
    parser.add_argument("--totalcapture-cache", required=True)
    parser.add_argument("--version", action="append", required=True, help="NAME=official or NAME=/path/to/checkpoint.pt")
    parser.add_argument("--strong-version", required=True)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--max-eval-sequences", type=int, default=0)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    args = parser.parse_args()

    versions = [load_version(spec) for spec in args.version]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        ("DIP test", args.dip_cache, output_dir / "eval_current_p_pdot_pddot_dip.json"),
        ("TotalCapture test", args.totalcapture_cache, output_dir / "eval_current_p_pdot_pddot_totalcapture.json"),
    ]
    results = []
    all_per_sequence = []
    all_per_leaf = []
    all_alignment = []
    for label, cache, json_path in datasets:
        result = evaluate_dataset(
            label,
            cache,
            versions,
            dt=args.dt,
            max_eval_sequences=args.max_eval_sequences,
            max_frames_per_sequence=args.max_frames_per_sequence,
        )
        json_result = {key: value for key, value in result.items() if key not in ("per_sequence_rows", "per_leaf_rows", "alignment_rows")}
        json_path.write_text(json.dumps(json_result, indent=2) + "\n")
        results.append(result)
        all_per_sequence.extend(result["per_sequence_rows"])
        all_per_leaf.extend(result["per_leaf_rows"])
        all_alignment.extend(result["alignment_rows"])

    write_csv(output_dir / "per_sequence_current_p_pdot_pddot.csv", all_per_sequence)
    write_csv(output_dir / "per_leaf_current_p_pdot_pddot.csv", all_per_leaf)
    write_csv(output_dir / "alignment_sweep.csv", all_alignment)
    decision, failures = decide(results, args.strong_version)
    write_summary(output_dir / "summary_current_p_pdot_pddot.md", results, args.strong_version, decision, failures)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "decision": decision,
        "outputs": [
            str(output_dir / "eval_current_p_pdot_pddot_dip.json"),
            str(output_dir / "eval_current_p_pdot_pddot_totalcapture.json"),
            str(output_dir / "summary_current_p_pdot_pddot.md"),
            str(output_dir / "per_sequence_current_p_pdot_pddot.csv"),
            str(output_dir / "per_leaf_current_p_pdot_pddot.csv"),
            str(output_dir / "alignment_sweep.csv"),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
