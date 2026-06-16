import argparse
import csv
import json
from pathlib import Path

import torch

from pl_curve import normalize_gravity
from pl_next_control_cache import _central_acceleration, _central_velocity, _decode_control_derivatives
from pl_next_control_eval import load_version, run_model
from pl_next_control_train import load_next_records


LEAF_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot", "head")
SHIFTS = (-2, -1, 0, 1, 2)


def current_derivative_mask(n):
    mask = torch.zeros(n, dtype=torch.bool)
    if n > 2:
        mask[1:-1] = True
    return mask


def shifted_mask(n, shift, base_mask):
    idx = torch.arange(n) + int(shift)
    return base_mask.bool() & (idx >= 0) & (idx < n)


def shifted_target(x, shift):
    idx = (torch.arange(x.shape[0]) + int(shift)).clamp(0, x.shape[0] - 1)
    return x[idx]


class LeafL2:
    def __init__(self):
        self.sum = torch.zeros(5, dtype=torch.float64)
        self.count = torch.zeros(5, dtype=torch.float64)

    def add(self, pred, target, mask, scale=100.0):
        if not bool(mask.any()):
            return
        err = (pred[..., :15] - target[..., :15]).detach().cpu()
        leaf = err.reshape(err.shape[:-1] + (5, 3)).norm(dim=-1) * float(scale)
        mask = mask.detach().cpu().bool()
        for i in range(5):
            vals = leaf[..., i].masked_select(mask)
            self.sum[i] += vals.double().sum()
            self.count[i] += vals.numel()

    def mean(self):
        total_count = self.count.sum().item()
        if total_count == 0:
            return None
        return float(self.sum.sum().item() / total_count)

    def per_leaf(self):
        out = {}
        for i, name in enumerate(LEAF_NAMES):
            out[name] = float(self.sum[i].item() / self.count[i].item()) if self.count[i].item() else None
        return out


def l2_value(pred, target, mask, scale=100.0):
    acc = LeafL2()
    acc.add(pred, target, mask, scale=scale)
    return acc.mean()


def sequence_l2(pred, target, mask, scale=100.0):
    return l2_value(pred, target, mask, scale=scale)


def classify(findings):
    if findings["actual_dt_mismatch"]:
        return "C. velocity error mainly comes from dt/unit mismatch"
    if findings["derivative_target_mismatch"]:
        return "D. velocity error mainly comes from derivative target definition mismatch"
    if findings["next_head_alignment_mismatch"]:
        return "B. next-head velocity/acceleration error mainly comes from temporal alignment/source mismatch; current finite-difference velocity uses the audited current-frame definition"
    if findings["current_alignment_mismatch"]:
        return "B. current finite-difference velocity/acceleration error mainly comes from temporal alignment mismatch"
    if findings["boundary_mask_issue"]:
        return "E. velocity error mainly comes from boundary/mask issue"
    return "A. velocity error is a real model/target-tracking issue under the audited metric definitions"


def model_predictions(version, record, dt):
    out = run_model(version, record)
    current_p = normalize_gravity(out["pl"]).detach().cpu()
    current_fd_v = _central_velocity(current_p, dt=dt)
    current_fd_a = _central_acceleration(current_p, dt=dt)
    if "next_pl" in out:
        next_p = normalize_gravity(out["next_pl"]).detach().cpu()
        next_head_v = out["next_pldot"].detach().cpu()
        next_head_a = out["next_plddot"].detach().cpu()
    else:
        next_p = current_p
        next_head_v = _central_velocity(next_p, dt=dt)
        next_head_a = _central_acceleration(next_p, dt=dt)
    next_fd_v = _central_velocity(next_p, dt=dt)
    next_fd_a = _central_acceleration(next_p, dt=dt)
    return {
        "current_p": current_p,
        "current_fd_velocity": current_fd_v,
        "current_fd_acceleration": current_fd_a,
        "next_p": next_p,
        "next_head_velocity": next_head_v,
        "next_head_acceleration": next_head_a,
        "next_position_fd_velocity": next_fd_v,
        "next_position_fd_acceleration": next_fd_a,
    }


def audit_dataset(label, cache, versions, dt_arg=None, max_sequences=0, max_frames=0):
    records, manifest = load_next_records(cache, max_sequences=max_sequences)
    if max_frames:
        from pl_next_control_eval import truncate_record
        records = [truncate_record(r, max_frames) for r in records]
    manifest_dt = float(manifest.get("dt", 1.0 / 60.0)) if manifest else 1.0 / 60.0
    eval_dt = float(dt_arg if dt_arg is not None else manifest_dt)
    dt_sweep = [("1/60", 1.0 / 60.0), ("1", 1.0), ("manifest", manifest_dt)]

    mask_summary = {
        "num_sequences": len(records),
        "num_frames": sum(int(r["pl_target"].shape[0]) for r in records),
        "valid_next_frames": sum(int(r["valid_next_mask"].bool().sum()) for r in records),
        "current_derivative_valid_frames": sum(int(current_derivative_mask(r["pl_target"].shape[0]).sum()) for r in records),
        "excluded_boundary_frames": sum(max(0, min(2, int(r["pl_target"].shape[0]))) for r in records),
    }

    gt_decode = {k: LeafL2() for k in ("decoded_pl", "decoded_dot", "decoded_ddot")}
    fd_consistency = {(name, metric): LeafL2() for name, _ in dt_sweep for metric in ("fd_gt_vel", "fd_gt_acc")}
    for r in records:
        n = int(r["pl_target"].shape[0])
        all_mask = torch.ones(n, dtype=torch.bool)
        deriv_mask = current_derivative_mask(n)
        dec_p, dec_v, dec_a = _decode_control_derivatives(r["pl_target_control"], dt=manifest_dt)
        gt_decode["decoded_pl"].add(dec_p, r["pl_target"], all_mask)
        gt_decode["decoded_dot"].add(dec_v, r["gt_pldot"], all_mask)
        gt_decode["decoded_ddot"].add(dec_a, r["gt_plddot"], all_mask)
        for name, dt in dt_sweep:
            fd_v = _central_velocity(r["pl_target"], dt=dt)
            fd_a = _central_acceleration(r["pl_target"], dt=dt)
            fd_consistency[(name, "fd_gt_vel")].add(fd_v, r["gt_pldot"], deriv_mask)
            fd_consistency[(name, "fd_gt_acc")].add(fd_a, r["gt_plddot"], deriv_mask)

    version_results = []
    per_seq = []
    per_leaf_rows = []
    alignment_rows = []
    for version in versions:
        metric_acc = {
            "current_fd_velocity": LeafL2(),
            "current_fd_acceleration": LeafL2(),
            "next_head_velocity": LeafL2(),
            "next_head_acceleration": LeafL2(),
            "next_position_fd_velocity": LeafL2(),
            "next_position_fd_acceleration": LeafL2(),
        }
        align_acc = {
            (metric, shift): LeafL2()
            for metric in ("current_fd_velocity", "current_fd_acceleration", "next_head_velocity", "next_head_acceleration")
            for shift in SHIFTS
        }
        for r in records:
            pred = model_predictions(version, r, eval_dt)
            n = int(r["pl_target"].shape[0])
            deriv_mask = current_derivative_mask(n)
            next_mask = r["valid_next_mask"].bool()
            next_fd_mask = next_mask & deriv_mask

            metric_acc["current_fd_velocity"].add(pred["current_fd_velocity"], r["gt_pldot"], deriv_mask)
            metric_acc["current_fd_acceleration"].add(pred["current_fd_acceleration"], r["gt_plddot"], deriv_mask)
            metric_acc["next_head_velocity"].add(pred["next_head_velocity"], r["gt_pldot_next"], next_mask)
            metric_acc["next_head_acceleration"].add(pred["next_head_acceleration"], r["gt_plddot_next"], next_mask)
            metric_acc["next_position_fd_velocity"].add(pred["next_position_fd_velocity"], r["gt_pldot_next"], next_fd_mask)
            metric_acc["next_position_fd_acceleration"].add(pred["next_position_fd_acceleration"], r["gt_plddot_next"], next_fd_mask)

            seq = {
                "dataset": label,
                "version": version["name"],
                "sequence": r["name"],
                "num_frames": n,
                "valid_next_frames": int(next_mask.sum()),
                "current_derivative_valid_frames": int(deriv_mask.sum()),
                "current_fd_velocity_L2_cm_s": sequence_l2(pred["current_fd_velocity"], r["gt_pldot"], deriv_mask),
                "current_fd_acceleration_L2_cm_s2": sequence_l2(pred["current_fd_acceleration"], r["gt_plddot"], deriv_mask),
                "next_head_velocity_L2_cm_s": sequence_l2(pred["next_head_velocity"], r["gt_pldot_next"], next_mask),
                "next_head_acceleration_L2_cm_s2": sequence_l2(pred["next_head_acceleration"], r["gt_plddot_next"], next_mask),
                "next_position_fd_velocity_L2_cm_s": sequence_l2(pred["next_position_fd_velocity"], r["gt_pldot_next"], next_fd_mask),
                "next_position_fd_acceleration_L2_cm_s2": sequence_l2(pred["next_position_fd_acceleration"], r["gt_plddot_next"], next_fd_mask),
            }
            per_seq.append(seq)

            for metric, target_key, base_mask in (
                ("current_fd_velocity", "gt_pldot", deriv_mask),
                ("current_fd_acceleration", "gt_plddot", deriv_mask),
                ("next_head_velocity", "gt_pldot_next", next_mask),
                ("next_head_acceleration", "gt_plddot_next", next_mask),
            ):
                target = r[target_key]
                for shift in SHIFTS:
                    mask = shifted_mask(n, shift, base_mask)
                    align_acc[(metric, shift)].add(pred[metric], shifted_target(target, shift), mask)

        metrics = {k + "_L2": v.mean() for k, v in metric_acc.items()}
        per_leaf = {k: v.per_leaf() for k, v in metric_acc.items()}
        for metric, leaves in per_leaf.items():
            row = {"dataset": label, "version": version["name"], "metric": metric}
            row.update(leaves)
            row["mean"] = metric_acc[metric].mean()
            per_leaf_rows.append(row)

        alignment = {}
        for metric in ("current_fd_velocity", "current_fd_acceleration", "next_head_velocity", "next_head_acceleration"):
            vals = {shift: align_acc[(metric, shift)].mean() for shift in SHIFTS}
            best_shift = min((s for s in SHIFTS if vals[s] is not None), key=lambda s: vals[s])
            row = {
                "dataset": label,
                "version": version["name"],
                "metric": metric,
                "best_shift": best_shift,
                "warning": "WARNING: best temporal alignment is not zero" if best_shift != 0 else "",
            }
            for shift in SHIFTS:
                row[f"shift_{shift:+d}"] = vals[shift]
            alignment_rows.append(row)
            alignment[metric] = row

        version_results.append({
            "name": version["name"],
            "kind": version["kind"],
            "path": version.get("path"),
            "metrics": metrics,
            "per_leaf": per_leaf,
            "alignment": alignment,
        })

    gt_decode_result = {k + "_L2": v.mean() for k, v in gt_decode.items()}
    fd_result = {name: {metric + "_L2": fd_consistency[(name, metric)].mean() for metric in ("fd_gt_vel", "fd_gt_acc")} for name, _ in dt_sweep}
    current_alignment_mismatch = any(
        row["best_shift"] != 0
        for row in alignment_rows
        if row["metric"] in ("current_fd_velocity", "current_fd_acceleration")
    )
    next_head_alignment_mismatch = any(
        row["best_shift"] != 0
        for row in alignment_rows
        if row["metric"] in ("next_head_velocity", "next_head_acceleration")
    )
    findings = {
        "derivative_target_mismatch": (
            (gt_decode_result["decoded_dot_L2"] or 0.0) > 1e-3
            or (gt_decode_result["decoded_ddot_L2"] or 0.0) > 1e-2
        ),
        "dt_unit_sensitivity": False,
        "actual_dt_mismatch": abs(eval_dt - manifest_dt) > 1e-9,
        "current_alignment_mismatch": current_alignment_mismatch,
        "next_head_alignment_mismatch": next_head_alignment_mismatch,
        "boundary_mask_issue": False,
    }
    if fd_result["1/60"]["fd_gt_vel_L2"] and fd_result["1"]["fd_gt_vel_L2"]:
        ratio = fd_result["1"]["fd_gt_vel_L2"] / max(fd_result["1/60"]["fd_gt_vel_L2"], 1e-9)
        findings["dt_unit_sensitivity"] = ratio > 10.0 or ratio < 0.1
    conclusion = classify(findings)

    return {
        "dataset": label,
        "cache": cache,
        "manifest_dt": manifest_dt,
        "eval_dt": eval_dt,
        "mask_summary": mask_summary,
        "gt_control_decode_self_consistency": gt_decode_result,
        "finite_difference_consistency_by_dt": fd_result,
        "versions": version_results,
        "alignment_rows": alignment_rows,
        "per_leaf_rows": per_leaf_rows,
        "per_sequence_rows": per_seq,
        "findings": findings,
        "conclusion": conclusion,
    }


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


def fmt(x):
    if x is None:
        return "NA"
    return f"{float(x):.6f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def write_md(path, results):
    lines = ["# Velocity / acceleration metric audit", ""]
    lines += [
        "This audit separates current finite-difference velocity, next-head velocity, and finite differences of next position.",
        "It checks GT control decode self-consistency, finite-difference target consistency, dt/unit effects, masks, alignment, per-leaf, and per-sequence breakdowns.",
        "Velocity units are cm/s and acceleration units are cm/s^2. All values are mean leaf-vector L2 over evaluated frames and leaves.",
        "",
    ]
    for result in results:
        lines += [f"## {result['dataset']}", ""]
        ms = result["mask_summary"]
        lines += [
            f"- manifest dt: `{result['manifest_dt']}`",
            f"- eval dt: `{result['eval_dt']}`",
            f"- num_frames: `{ms['num_frames']}`",
            f"- valid_next_frames: `{ms['valid_next_frames']}`",
            f"- current_derivative_valid_frames: `{ms['current_derivative_valid_frames']}`",
            f"- excluded_boundary_frames: `{ms['excluded_boundary_frames']}`",
            "",
            "### GT self-consistency",
            "",
            md_table(
                ["check", "L2"],
                [
                    {"check": k, "L2": fmt(v)}
                    for k, v in result["gt_control_decode_self_consistency"].items()
                ],
            ),
            "",
            "### Finite-difference consistency by dt",
            "",
            md_table(
                ["dt", "fd_gt_vel L2 cm/s", "fd_gt_acc L2 cm/s^2"],
                [
                    {"dt": k, "fd_gt_vel L2 cm/s": fmt(v["fd_gt_vel_L2"]), "fd_gt_acc L2 cm/s^2": fmt(v["fd_gt_acc_L2"])}
                    for k, v in result["finite_difference_consistency_by_dt"].items()
                ],
            ),
            "",
            "### Model velocity / acceleration metrics",
            "",
            md_table(
                [
                    "version",
                    "current_fd_velocity L2 cm/s",
                    "current_fd_acceleration L2 cm/s^2",
                    "next_head_velocity L2 cm/s",
                    "next_head_acceleration L2 cm/s^2",
                    "next_position_fd_velocity L2 cm/s",
                    "next_position_fd_acceleration L2 cm/s^2",
                ],
                [
                    {
                        "version": v["name"],
                        "current_fd_velocity L2 cm/s": fmt(v["metrics"]["current_fd_velocity_L2"]),
                        "current_fd_acceleration L2 cm/s^2": fmt(v["metrics"]["current_fd_acceleration_L2"]),
                        "next_head_velocity L2 cm/s": fmt(v["metrics"]["next_head_velocity_L2"]),
                        "next_head_acceleration L2 cm/s^2": fmt(v["metrics"]["next_head_acceleration_L2"]),
                        "next_position_fd_velocity L2 cm/s": fmt(v["metrics"]["next_position_fd_velocity_L2"]),
                        "next_position_fd_acceleration L2 cm/s^2": fmt(v["metrics"]["next_position_fd_acceleration_L2"]),
                    }
                    for v in result["versions"]
                ],
            ),
            "",
            "### Alignment sweep",
            "",
            md_table(
                ["version", "metric", "best_shift", "shift_-2", "shift_-1", "shift_+0", "shift_+1", "shift_+2", "warning"],
                [
                    {
                        "version": row["version"],
                        "metric": row["metric"],
                        "best_shift": row["best_shift"],
                        "shift_-2": fmt(row["shift_-2"]),
                        "shift_-1": fmt(row["shift_-1"]),
                        "shift_+0": fmt(row["shift_+0"]),
                        "shift_+1": fmt(row["shift_+1"]),
                        "shift_+2": fmt(row["shift_+2"]),
                        "warning": row["warning"],
                    }
                    for row in result["alignment_rows"]
                ],
            ),
            "",
            "### Conclusion",
            "",
            md_table(
                ["finding", "value"],
                [{"finding": k, "value": v} for k, v in result["findings"].items()],
            ),
            "",
            result["conclusion"],
            "",
            "Interpretation: `dt_unit_sensitivity=True` means a wrong `dt=1` would change FD velocity/acceleration scale drastically. It is not evidence of an actual dt bug unless `actual_dt_mismatch=True`.",
            "",
        ]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache", action="append", required=True, help="LABEL=manifest")
    parser.add_argument("--version", action="append", required=True, help="NAME=official or NAME=checkpoint.pt")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--max-eval-sequences", type=int, default=0)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    args = parser.parse_args()

    versions = [load_version(spec) for spec in args.version]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_align, all_leaf, all_seq = [], [], []
    for spec in args.cache:
        label, path = spec.split("=", 1)
        result = audit_dataset(
            label,
            path,
            versions,
            dt_arg=args.dt,
            max_sequences=args.max_eval_sequences,
            max_frames=args.max_frames_per_sequence,
        )
        results.append(result)
        all_align += result["alignment_rows"]
        all_leaf += result["per_leaf_rows"]
        all_seq += result["per_sequence_rows"]

    json_safe = []
    for result in results:
        json_safe.append({k: v for k, v in result.items() if k not in ("alignment_rows", "per_leaf_rows", "per_sequence_rows")})
    (out_dir / "velocity_metric_audit.json").write_text(json.dumps({"status": "ok", "datasets": json_safe}, indent=2) + "\n")
    write_md(out_dir / "velocity_metric_audit.md", results)
    write_csv(out_dir / "velocity_alignment_sweep.csv", all_align)
    write_csv(out_dir / "velocity_per_leaf.csv", all_leaf)
    write_csv(out_dir / "velocity_per_sequence.csv", all_seq)
    print(json.dumps({"status": "ok", "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
