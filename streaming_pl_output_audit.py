import argparse
import json
from pathlib import Path

import torch
import articulate as art

from l4_train_diverse_short import DEVICE, load_records
from net import GPNet
from pl_curve import PLCurveModule, normalize_gravity, pl_input_feature, pl_target_from_pose


LEAF_NAMES = ("L_LowArm", "R_LowArm", "L_LowLeg", "R_LowLeg", "Head")
DEFAULT_PROCESSED_CACHE = Path(
    "data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/"
    "baseline_cache_manifest.json"
)
DEFAULT_OFFICIAL_CACHE = Path(
    "data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/"
    "baseline_cache_manifest.json"
)
DEFAULT_OLD_PROCESSED = Path(
    "data/experiments/pl_curve_v2_processed_no_baseline/tc_finetune_10ep/best_loss.pt"
)


def summarize(values):
    values = torch.as_tensor(values).float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "count": 0}
    return {
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "std": float(finite.std(unbiased=False)) if finite.numel() > 1 else 0.0,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "count": int(finite.numel()),
    }


def build_pl_curve(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get("config", {})
    model = PLCurveModule(
        init_size=int(config.get("init_size", 18)),
        hidden_size=int(config.get("hidden_size", 512)),
        tail_update=int(config.get("tail_length", 4)),
        residual_scale=float(config.get("residual_scale", 0.005)),
        dropout=float(config.get("dropout", 0.4)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def selected_imu(record, mode):
    if mode == "official":
        return record["aM"].float(), record["wM"].float(), record["RMB"].float()
    if mode == "processed":
        return record["l4_aM"].float(), record["l4_wM"].float(), record["l4_RMB"].float()
    raise ValueError(mode)


def gravity_angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred.float(), avoid_nan=True)
    target = art.math.normalize_tensor(target.float(), avoid_nan=True)
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def leaf_error_cm(pred, target):
    pred = pred[..., :15].reshape(pred.shape[:-1] + (5, 3))
    target = target[..., :15].reshape(target.shape[:-1] + (5, 3))
    return (pred - target).norm(dim=-1) * 100.0


def temporal_metrics(pred, target):
    pred_g = art.math.normalize_tensor(pred[..., 15:].float(), avoid_nan=True)
    target_g = art.math.normalize_tensor(target[..., 15:].float(), avoid_nan=True)
    out = {
        "gRdot_l2": pred.new_zeros(0).cpu(),
        "gRdot_smooth_l1": pred.new_zeros(0).cpu(),
        "gRddot_l2": pred.new_zeros(0).cpu(),
        "gRddot_smooth_l1": pred.new_zeros(0).cpu(),
    }
    if pred_g.shape[0] >= 2:
        pd = pred_g[1:] - pred_g[:-1]
        td = target_g[1:] - target_g[:-1]
        out["gRdot_l2"] = (pd - td).norm(dim=-1).cpu()
        out["gRdot_smooth_l1"] = torch.nn.functional.smooth_l1_loss(pd, td, reduction="none").mean(dim=-1).cpu()
    if pred_g.shape[0] >= 3:
        pdd = pred_g[2:] - 2.0 * pred_g[1:-1] + pred_g[:-2]
        tdd = target_g[2:] - 2.0 * target_g[1:-1] + target_g[:-2]
        out["gRddot_l2"] = (pdd - tdd).norm(dim=-1).cpu()
        out["gRddot_smooth_l1"] = torch.nn.functional.smooth_l1_loss(pdd, tdd, reduction="none").mean(dim=-1).cpu()
    return out


@torch.no_grad()
def run_streaming(records, input_mode, checkpoint=None):
    curve = None
    config = None
    if checkpoint is not None:
        curve, config = build_pl_curve(checkpoint)
    probe = GPNet().eval().to(DEVICE)
    body_model = art.ParametricModel("models/SMPL_male.pkl", vert_mask=probe.v_imu, device=DEVICE)
    rows = []
    for record in records:
        pose = record["pose_gt"].float()
        target = normalize_gravity(pl_target_from_pose(pose.to(DEVICE), body_model).float()).cpu()
        a, w, R = selected_imu(record, input_mode)
        net = GPNet(
            pl_backend="curve_v1" if curve is not None else "original",
            pl_curve_module=curve,
        ).eval().to(DEVICE)
        net.rnn_initialize(pose[0], offset_r=record.get("offset_r"))
        outputs = []
        for idx in range(pose.shape[0]):
            pl_input = pl_input_feature(a[idx], w[idx], R[idx]).to(DEVICE)
            pl_raw, _ = net._run_pl_stage(pl_input)
            outputs.append(normalize_gravity(pl_raw.detach().cpu()))
        pred = torch.stack(outputs)
        leaf = leaf_error_cm(pred, target)
        grav = gravity_angle_deg(pred[..., 15:], target[..., 15:])
        temporal = temporal_metrics(pred, target)
        rows.append({
            "name": str(record["name"]),
            "num_frames": int(pose.shape[0]),
            "finite": bool(torch.isfinite(pred).all() and torch.isfinite(target).all()),
            "pRB_error_cm": leaf,
            "gR1_error_deg": grav,
            "temporal": temporal,
        })
    return rows, config


def aggregate_rows(rows):
    leaf_all = torch.cat([row["pRB_error_cm"].reshape(-1) for row in rows])
    g_all = torch.cat([row["gR1_error_deg"].reshape(-1) for row in rows])
    p_seq = torch.stack([row["pRB_error_cm"].mean() for row in rows])
    g_seq = torch.stack([row["gR1_error_deg"].mean() for row in rows])
    per_leaf = {}
    for leaf_idx, leaf_name in enumerate(LEAF_NAMES):
        per_leaf[leaf_name] = summarize(torch.cat([row["pRB_error_cm"][:, leaf_idx] for row in rows]))
    temporal_keys = ("gRdot_l2", "gRdot_smooth_l1", "gRddot_l2", "gRddot_smooth_l1")
    temporal_frame = {}
    temporal_seq = {}
    for key in temporal_keys:
        vals = [row["temporal"][key].reshape(-1) for row in rows if row["temporal"][key].numel()]
        temporal_frame[key] = summarize(torch.cat(vals)) if vals else summarize(torch.empty(0))
        seq_vals = [row["temporal"][key].mean() for row in rows if row["temporal"][key].numel()]
        temporal_seq[key] = summarize(torch.stack(seq_vals)) if seq_vals else summarize(torch.empty(0))
    return {
        "num_sequences": len(rows),
        "num_frames": int(sum(row["num_frames"] for row in rows)),
        "all_finite": all(row["finite"] for row in rows),
        "frame_weighted": {
            "pRB_mean_cm": summarize(leaf_all),
            "gR1_mean_deg": summarize(g_all),
            "pRB_per_leaf_cm": per_leaf,
            **temporal_frame,
        },
        "sequence_equal": {
            "pRB_mean_cm": summarize(p_seq),
            "gR1_mean_deg": summarize(g_seq),
            **temporal_seq,
        },
    }


def compact_rows(rows):
    return [
        {
            "name": row["name"],
            "num_frames": row["num_frames"],
            "finite": row["finite"],
            "pRB_frame_mean_cm": float(row["pRB_error_cm"].mean()),
            "gR1_frame_mean_deg": float(row["gR1_error_deg"].mean()),
            "gRdot_l2_mean": float(row["temporal"]["gRdot_l2"].mean()) if row["temporal"]["gRdot_l2"].numel() else None,
            "gRddot_l2_mean": float(row["temporal"]["gRddot_l2"].mean()) if row["temporal"]["gRddot_l2"].numel() else None,
        }
        for row in rows
    ]


def main():
    parser = argparse.ArgumentParser(description="Streaming-compatible PL output audit.")
    parser.add_argument("--processed-cache", type=Path, default=DEFAULT_PROCESSED_CACHE)
    parser.add_argument("--official-cache", type=Path, default=DEFAULT_OFFICIAL_CACHE)
    parser.add_argument("--old-processed-checkpoint", type=Path, default=DEFAULT_OLD_PROCESSED)
    parser.add_argument("--new-best-checkpoint", type=Path)
    parser.add_argument("--new-last-checkpoint", type=Path)
    parser.add_argument("--official-checkpoint", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "started",
        "audit_contract": "streaming-compatible: GPNet.rnn_initialize per sequence, then _run_pl_stage frame by frame.",
        "target_definition": "pl_target_from_pose with SMPL male vert_mask=GPNet.v_imu; normalized gR1.",
        "metric_definition": {
            "pRB": "L2 leaf error in cm.",
            "gR1": "angle between normalized gravity directions in degrees.",
            "gRdot/gRddot": "finite differences of normalized gR1, no dt scaling.",
        },
    }
    try:
        processed_records, processed_manifest = load_records(args.processed_cache)
        official_records, official_manifest = load_records(args.official_cache)
        run_specs = [
            ("original_streaming_processed", "processed", processed_records, None),
            ("old_plcurve_v2_processed_no_baseline_streaming", "processed", processed_records, args.old_processed_checkpoint),
        ]
        if args.new_best_checkpoint:
            run_specs.append(("new_gRdyn_best_streaming", "processed", processed_records, args.new_best_checkpoint))
        if args.new_last_checkpoint:
            run_specs.append(("new_gRdyn_last_streaming", "processed", processed_records, args.new_last_checkpoint))
        run_specs.append(("original_streaming_official", "official", official_records, None))
        if args.official_checkpoint:
            run_specs.append(("plcurve_official_checkpoint_streaming", "official", official_records, args.official_checkpoint))
        runs = []
        for name, input_mode, records, checkpoint in run_specs:
            rows, config = run_streaming(records, input_mode=input_mode, checkpoint=checkpoint)
            summary = aggregate_rows(rows)
            runs.append({
                "run": name,
                "input_mode": input_mode,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "checkpoint_config": config,
                "summary": summary,
                "rows": compact_rows(rows),
            })
            print(json.dumps({
                "run": name,
                "pRB_frame_cm": summary["frame_weighted"]["pRB_mean_cm"]["mean"],
                "gR1_frame_deg": summary["frame_weighted"]["gR1_mean_deg"]["mean"],
                "gRdot_l2": summary["frame_weighted"]["gRdot_l2"]["mean"],
                "gRddot_l2": summary["frame_weighted"]["gRddot_l2"]["mean"],
                "num_frames": summary["num_frames"],
            }), flush=True)
        result.update({
            "status": "ok",
            "processed_cache": str(args.processed_cache),
            "official_cache": str(args.official_cache),
            "processed_manifest": processed_manifest,
            "official_manifest": official_manifest,
            "runs": runs,
            "all_finite": all(run["summary"]["all_finite"] for run in runs),
            "training_run": False,
            "s5_run": False,
            "official_weights_modified": False,
            "motion_evaluator_modified": False,
            "test_py_modified": False,
        })
    except Exception as exc:
        import traceback
        result.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result.get("status"),
        "output_json": str(args.output_json),
        "all_finite": result.get("all_finite"),
        "error_type": result.get("error_type"),
        "error": result.get("error"),
    }, indent=2), flush=True)
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
