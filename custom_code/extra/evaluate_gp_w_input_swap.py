#!/usr/bin/env python3
"""Zero-shot G0-G3 angular-velocity input swap on official GP weights."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import articulate as art

sys.path.insert(0, str(Path(__file__).resolve().parent))

from custom_code.modified_official.net import GPNet
from gp_w_input_swap import causal_w_sequence
from l4_train_diverse_short import DEVICE, load_records
from official_processed_module_audit import LEAF_NAMES, build_targets, gravity_angle_deg, metric_stats
from test import MotionEvaluator


VARIANTS = {
    "G0_official": (False, False),
    "G1_pl_swap": (True, False),
    "G2_vr_swap": (False, True),
    "G3_all_swap": (True, True),
}
MOTION_NAMES = [
    "local_sip_deg", "local_angle_deg", "local_joint_cm", "local_mesh_cm",
    "global_sip_deg", "global_angle_deg", "global_joint_cm", "global_mesh_cm",
    "root_jitter_km_s3", "joint_jitter_km_s3",
]
LEAF_JOINTS = (17, 18, 4, 5, 15)  # pRJ indices after removing root


def stats(x):
    return metric_stats(torch.as_tensor(x).float())


def vector_metrics(pred, target, scale=1.0):
    d = (pred.float() - target.float()).reshape(pred.shape[0], -1, 3)
    return {
        "l1": stats(d.abs().mean((-1, -2)) * scale),
        "l2": stats(d.norm(dim=-1).mean(-1) * scale),
        "rmse": stats(d.square().mean((-1, -2)).sqrt() * scale),
    }


def temporal_l2(pred, target, order):
    if order == 1:
        d = (pred[1:] - pred[:-1]) - (target[1:] - target[:-1])
    else:
        d = (pred[2:] - 2 * pred[1:-1] + pred[:-2]) - (target[2:] - 2 * target[1:-1] + target[:-2])
    return stats(d.reshape(len(d), -1, 3).norm(dim=-1).mean(-1) * 100.0)


def motion_metrics(pose_p, pose_t, tran_p, tran_t):
    evaluator = MotionEvaluator()
    weighted = None
    total = 0
    for start in range(0, len(pose_p), 512):
        end = min(len(pose_p), start + 512)
        if end - start < 4 and start:
            start = end - 4
        chunk = evaluator(
            pose_p[start:end].to(DEVICE), pose_t[start:end].to(DEVICE),
            tran_p[start:end].to(DEVICE), tran_t[start:end].to(DEVICE),
        ).cpu()
        weight = end - start
        weighted = chunk * weight if weighted is None else weighted + chunk * weight
        total += weight
    values = weighted / total
    root_jitter = ((tran_p[3:] - 3 * tran_p[2:-1] + 3 * tran_p[1:-2] - tran_p[:-3]) * (60 ** 3)).norm(dim=1)
    if root_jitter.numel():
        values[8] = torch.tensor([root_jitter.mean(), root_jitter.std()]) / 1000.0
    return {name: {"mean": float(values[i, 0]), "std": float(values[i, 1])} for i, name in enumerate(MOTION_NAMES)}


def root_metrics(net, pose_p, pose_t, tran_p, tran_t):
    e = tran_p - tran_t
    ea = (tran_p - tran_p[0]) - (tran_t - tran_t[0])
    vp, vt = (tran_p[1:] - tran_p[:-1]) * 60.0, (tran_t[1:] - tran_t[:-1]) * 60.0
    step = (tran_p[1:] - tran_p[:-1]).norm(dim=-1)
    body = net.body_model
    jp = body.forward_kinematics(pose_p)[1] + tran_p[:, None]
    jt = body.forward_kinematics(pose_t)[1] + tran_t[:, None]
    feet = (10, 11, 22, 23)
    vjp, vjt = (jp[1:, feet] - jp[:-1, feet]) * 60.0, (jt[1:, feet] - jt[:-1, feet]) * 60.0
    floor = jt[..., 1].amin()
    contact = (vjt.norm(dim=-1) < 0.15) & (jt[1:, feet, 1] < floor + 0.08)
    cv = vjp.norm(dim=-1)[contact]
    slip = vjp[..., (0, 2)].norm(dim=-1)[contact]
    return {
        "root_translation_rmse_m": float(e.square().sum(-1).mean().sqrt()),
        "root_translation_first_frame_aligned_rmse_m": float(ea.square().sum(-1).mean().sqrt()),
        "root_trajectory_drift_m": float(ea[-1].norm()),
        "root_velocity_rmse_m_s": float((vp - vt).square().sum(-1).mean().sqrt()),
        "root_velocity_pred_speed_mean_m_s": float(vp.norm(dim=-1).mean()),
        "root_velocity_gt_speed_mean_m_s": float(vt.norm(dim=-1).mean()),
        "root_velocity_pred_speed_std_m_s": float(vp.norm(dim=-1).std(unbiased=False)),
        "max_frame_root_step_m": float(step.max()),
        "contact_velocity_mean_m_s": float(cv.mean()) if cv.numel() else None,
        "foot_slip_mean_m_s": float(slip.mean()) if slip.numel() else None,
        "contact_frames": int(contact.sum()),
    }


@torch.no_grad()
def run_variant(record, variant):
    swap_pl, swap_vr = VARIANTS[variant]
    a, cached_w, rmb = record["aM"], record["wM"], record["RMB"]
    causal_w = causal_w_sequence(rmb)
    net = GPNet().eval().to(DEVICE)
    net.rnn_initialize(record["pose_gt"][0])
    pose = torch.zeros_like(record["pose_gt"])
    tran = torch.zeros_like(record["tran_gt"])
    pl, ik1 = [], []
    for t in range(len(a)):
        pose[t], tran[t] = net.forward_frame(
            a[t].to(DEVICE), cached_w[t].to(DEVICE), rmb[t].to(DEVICE),
            w_pl_override=causal_w[t].to(DEVICE) if swap_pl else None,
            w_vr_override=causal_w[t].to(DEVICE) if swap_vr else None,
        )
        d = net.last_w_source_debug
        pl.append(torch.cat((d["pRB"], d["gR1"])))
        ik1.append(torch.cat((d["pRJ"], d["gR2"])))
    targets = build_targets(record, net)
    pl, ik1 = torch.stack(pl), torch.stack(ik1)
    pl_target, ik1_target = targets["pl_target"], targets["ik1_target"]
    module = {
        "PL": {
            **{f"pRB_{k}_cm": v for k, v in vector_metrics(pl[:, :15], pl_target[:, :15], 100.0).items()},
            "gR1_angle_deg": stats(gravity_angle_deg(pl[:, 15:], pl_target[:, 15:])),
            "per_leaf": {
                name: vector_metrics(pl[:, i * 3:(i + 1) * 3], pl_target[:, i * 3:(i + 1) * 3], 100.0)
                for i, name in enumerate(LEAF_NAMES)
            },
        },
        "IK1": {
            **{f"pRJ_{k}_cm": v for k, v in vector_metrics(ik1[:, :69], ik1_target[:, :69], 100.0).items()},
            "gR2_angle_deg": stats(gravity_angle_deg(ik1[:, 69:], ik1_target[:, 69:])),
            "pRJ_first_difference_l2_cm_per_frame": temporal_l2(ik1[:, :69], ik1_target[:, :69], 1),
            "pRJ_second_difference_l2_cm_per_frame2": temporal_l2(ik1[:, :69], ik1_target[:, :69], 2),
            "per_leaf": {
                name: vector_metrics(ik1[:, j * 3:(j + 1) * 3], ik1_target[:, j * 3:(j + 1) * 3], 100.0)
                for name, j in zip(LEAF_NAMES, LEAF_JOINTS)
            },
        },
    }
    return {
        "dataset": None, "sequence": record["name"], "variant": variant, "frames": len(a),
        "motion": motion_metrics(pose, record["pose_gt"], tran, record["tran_gt"]),
        "root": root_metrics(net, pose, record["pose_gt"], tran, record["tran_gt"]),
        "module": module,
    }


def flatten_means(prefix, value, out):
    if isinstance(value, dict):
        if "mean" in value and set(value).issuperset({"mean"}):
            out[prefix] = value["mean"]
        else:
            for k, v in value.items():
                flatten_means(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(value, (int, float)) or value is None:
        out[prefix] = value


def aggregate(rows):
    result = {}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        flat = []
        for row in subset:
            x = {}; flatten_means("", {"motion": row["motion"], "root": row["root"], "module": row["module"]}, x); flat.append(x)
        result[variant] = {k: float(np.average([x[k] for x in flat if x.get(k) is not None], weights=[subset[i]["frames"] for i, x in enumerate(flat) if x.get(k) is not None])) for k in sorted(set().union(*(x.keys() for x in flat)))}
    return result


def write_comparison(dataset, rows, aggregate_rows, path):
    records = []
    for row in rows:
        flat = {}; flatten_means("", {"motion": row["motion"], "root": row["root"], "module": row["module"]}, flat)
        official = next(r for r in rows if r["sequence"] == row["sequence"] and r["variant"] == "G0_official")
        base = {}; flatten_means("", {"motion": official["motion"], "root": official["root"], "module": official["module"]}, base)
        for metric, value in flat.items():
            b = base.get(metric); delta = None if value is None or b is None else value - b
            pct = None if delta is None or abs(b) < 1e-12 else delta / b * 100.0
            records.append({"dataset": dataset, "sequence": row["sequence"], "variant": row["variant"], "metric": metric, "value": value, "delta_vs_official": delta, "percent_delta_vs_official": pct})
    for variant, flat in aggregate_rows.items():
        base = aggregate_rows["G0_official"]
        for metric, value in flat.items():
            b = base.get(metric); delta = None if value is None or b is None else value - b
            pct = None if delta is None or abs(b) < 1e-12 else delta / b * 100.0
            records.append({"dataset": dataset, "sequence": "aggregate", "variant": variant, "metric": metric, "value": value, "delta_vs_official": delta, "percent_delta_vs_official": pct})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)


def input_audit(records):
    imu_root = Path("/home/lingfeng/projects/imu_acc_explainability")
    sys.path[:0] = [str(imu_root / "code"), str(imu_root / "code/tools"), str(imu_root / "scripts")]
    import audit_rbdl_offset_smooth_acc_explain as rbdl_audit
    import audit_w_vs_fk_simple as w_base
    from fit_tc_transpose_n6_fk_rjs import transpose_n_q_derivatives
    from pip_physics_backend import smpl_to_pip_rbdl
    model = rbdl_audit.load_rbdl(); bodies = rbdl_audit.rbdl_bodies(model)[:5]
    pairs = {k: [] for k in ("cached_wM_vs_FK_w", "causal_RMB_w_vs_FK_w", "cached_wM_vs_causal_RMB_w")}
    per_sequence = {}
    for rec in records:
        q = smpl_to_pip_rbdl(rec["pose_gt"].numpy(), rec["tran_gt"].numpy())
        q, qdot, _, _ = transpose_n_q_derivatives(q, 8, 1 / 60)
        fk, source = w_base.fk_angular_velocity(model, bodies, q, qdot, 8)
        cached = rec["wM"][8:-8, :5].numpy(); causal = causal_w_sequence(rec["RMB"]) [8:-8, :5].numpy()
        n = min(len(fk), len(cached), len(causal)); fk, cached, causal = fk[:n], cached[:n], causal[:n]
        seq = {}
        for name, x, y in (("cached_wM_vs_FK_w", cached, fk), ("causal_RMB_w_vs_FK_w", causal, fk), ("cached_wM_vs_causal_RMB_w", cached, causal)):
            d = x - y; xf, yf = x.reshape(-1, 3), y.reshape(-1, 3)
            cos = np.sum(xf * yf, 1) / np.maximum(np.linalg.norm(xf, axis=1) * np.linalg.norm(yf, axis=1), 1e-12)
            m = {"rmse": float(np.sqrt(np.mean(d ** 2))), "pearson": float(np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1]), "mean_l2": float(np.linalg.norm(d, axis=-1).mean()), "cosine": float(cos.mean())}
            seq[name] = m; pairs[name].append((x, y))
        seq["fk_source"] = source; per_sequence[rec["name"]] = seq
    overall = {}
    for name, chunks in pairs.items():
        x = np.concatenate([c[0] for c in chunks]); y = np.concatenate([c[1] for c in chunks]); d = x - y; xf, yf = x.reshape(-1, 3), y.reshape(-1, 3)
        cos = np.sum(xf * yf, 1) / np.maximum(np.linalg.norm(xf, axis=1) * np.linalg.norm(yf, axis=1), 1e-12)
        overall[name] = {"rmse": float(np.sqrt(np.mean(d ** 2))), "pearson": float(np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1]), "mean_l2": float(np.linalg.norm(d, axis=-1).mean()), "cosine": float(cos.mean())}
    return {"contract": "FK w = calc_space_Jacobian(q, body)[:3] @ qdot; qdot centered n=8 reference only", "overall": overall, "per_sequence": per_sequence}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", choices=("dip", "totalcapture"), required=True); p.add_argument("--cache", required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--max-sequences", type=int, default=0); p.add_argument("--max-frames", type=int, default=0); p.add_argument("--sequence", action="append", default=[]); p.add_argument("--input-audit", action="store_true"); p.add_argument("--audit-only", action="store_true"); args = p.parse_args()
    records, manifest = load_records(args.cache, args.max_sequences)
    if args.sequence:
        wanted = set(args.sequence)
        records = [rec for rec in records if rec["name"] in wanted]
        missing = wanted - {rec["name"] for rec in records}
        if missing:
            raise ValueError(f"Unknown sequences: {sorted(missing)}")
    if args.max_frames:
        records = [{k: (v[:args.max_frames] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] >= args.max_frames else v) for k, v in rec.items()} for rec in records]
    if args.audit_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "input_signal_audit.json").write_text(json.dumps(input_audit(records), indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for rec in records:
        for variant in VARIANTS:
            row = run_variant(rec, variant); row["dataset"] = args.dataset; rows.append(row)
            print(json.dumps({"sequence": rec["name"], "variant": variant, "frames": row["frames"]}), flush=True)
    agg = aggregate(rows)
    for row in rows:
        d = args.output_dir / row["sequence"]; d.mkdir(exist_ok=True); (d / f"{row['variant']}.json").write_text(json.dumps(row, indent=2))
    (args.output_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    write_comparison(args.dataset, rows, agg, args.output_dir / "comparison.csv")
    if args.input_audit:
        (args.output_dir.parent / "input_signal_audit.json").write_text(json.dumps(input_audit(records), indent=2))
    (args.output_dir / "run_metadata.json").write_text(json.dumps({"dataset": args.dataset, "cache": args.cache, "manifest": manifest, "num_sequences": len(records), "variants": VARIANTS}, indent=2, default=str))


if __name__ == "__main__":
    main()
