import argparse
import json
from pathlib import Path

import torch
import articulate as art

from l4_train_diverse_short import DEVICE, load_cache_files, load_records
from net import GPNet
from pl_curve import PLCurveModule, normalize_gravity, pl_input_feature, pl_target_from_pose


LEAF_NAMES = ("L_LowArm", "R_LowArm", "L_LowLeg", "R_LowLeg", "Head")
DEFAULT_OFFICIAL_BASE_CACHE = Path(
    "data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/"
    "baseline_cache_manifest.json"
)
DEFAULT_PROCESSED_BASE_CACHE = Path(
    "data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/"
    "baseline_cache_manifest.json"
)
DEFAULT_OFFICIAL_PL_CACHE = Path(
    "data/dataset_work/L4Cache/pl_curve_v1_totalcapture_val_official_neural_only_offset_r/"
    "pl_curve_cache_manifest.json"
)
DEFAULT_PROCESSED_PL_CACHE = Path(
    "data/dataset_work/L4Cache/pl_curve_v2_processed_no_baseline_tc_val_Roffset_A/"
    "pl_curve_cache_manifest.json"
)
DEFAULT_PL_V3_CHECKPOINT = Path(
    "data/experiments/pl_curve_v3_official_no_baseline/tc_finetune_10ep/best_loss.pt"
)
DEFAULT_PL_V2_CHECKPOINT = Path(
    "data/experiments/pl_curve_v2_processed_no_baseline/tc_finetune_10ep/best_loss.pt"
)
DEFAULT_OUTPUT_JSON = Path(
    "data/experiments/unified_pl_output_audit/unified_pl_output_audit_s4.json"
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


def gravity_angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred.float(), avoid_nan=True)
    target = art.math.normalize_tensor(target.float(), avoid_nan=True)
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def leaf_error_cm(pred, target):
    pred = pred[..., :15].reshape(pred.shape[:-1] + (5, 3))
    target = target[..., :15].reshape(target.shape[:-1] + (5, 3))
    return (pred - target).norm(dim=-1) * 100.0


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


def load_pl_cache_records(cache_path):
    files, manifest = load_cache_files(cache_path)
    if manifest is None or manifest.get("type") not in ("pl_curve_cache_v1", "pl_curve_cache_v2"):
        raise RuntimeError(f"Expected pl_curve_cache_v1/v2 manifest at {cache_path}.")
    has_init = manifest.get("type") == "pl_curve_cache_v2"
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location="cpu")
        for seq_idx, name in enumerate(data["name"]):
            record = {
                "name": str(name),
                "pl_input": data["pl_input"][seq_idx].float(),
                "pl_target": normalize_gravity(data["pl_target"][seq_idx].float()),
                "pl_base": normalize_gravity(data["pl_base"][seq_idx].float()),
                "num_frames": int(data["pl_input"][seq_idx].shape[0]),
            }
            if has_init:
                record["pl_init_feature"] = data["pl_init_feature"][seq_idx].float()
            records.append(record)
    return records, manifest


def selected_imu(record, mode):
    if mode == "official":
        return record["aM"].float(), record["wM"].float(), record["RMB"].float()
    if mode == "processed":
        return record["l4_aM"].float(), record["l4_wM"].float(), record["l4_RMB"].float()
    raise ValueError(mode)


def make_pl_inputs_from_record(record, mode):
    a, w, R = selected_imu(record, mode)
    return torch.stack([pl_input_feature(a[i], w[i], R[i]) for i in range(a.shape[0])]).float()


@torch.no_grad()
def streaming_original_outputs(base_cache, mode):
    records, manifest = load_records(base_cache)
    probe = GPNet().eval().to(DEVICE)
    body_model = art.ParametricModel("models/SMPL_male.pkl", vert_mask=probe.v_imu, device=DEVICE)
    out = []
    for record in records:
        pose = record["pose_gt"].float()
        target = normalize_gravity(pl_target_from_pose(pose.to(DEVICE), body_model).float()).cpu()
        pl_input = make_pl_inputs_from_record(record, mode)
        net = GPNet().eval().to(DEVICE)
        net.rnn_initialize(pose[0])
        outputs = []
        for i in range(pl_input.shape[0]):
            pl_raw, _ = net._run_pl_stage(pl_input[i].to(DEVICE))
            outputs.append(normalize_gravity(pl_raw.detach().cpu()))
        out.append({
            "name": str(record["name"]),
            "num_frames": int(pose.shape[0]),
            "pl_target": target,
            "pl_output": torch.stack(outputs),
            "pl_input": pl_input,
        })
    return out, manifest


@torch.no_grad()
def cache_original_outputs(pl_cache):
    records, manifest = load_pl_cache_records(pl_cache)
    return [
        {
            "name": row["name"],
            "num_frames": row["num_frames"],
            "pl_target": row["pl_target"],
            "pl_output": row["pl_base"],
            "pl_input": row["pl_input"],
        }
        for row in records
    ], manifest


@torch.no_grad()
def curve_outputs(pl_cache, checkpoint):
    model, config = build_pl_curve(checkpoint)
    records, manifest = load_pl_cache_records(pl_cache)
    out = []
    for row in records:
        pl_input = row["pl_input"].to(DEVICE)
        target = row["pl_target"].to(DEVICE)
        base = row["pl_base"].to(DEVICE)
        init_feature = row.get("pl_init_feature")
        if init_feature is not None:
            init_feature = init_feature.to(DEVICE)
        elif model.init_size != 18:
            raise RuntimeError(f'PL init dim {model.init_size} requires pl_init_feature for {row["name"]}.')
        pred = normalize_gravity(model.forward_sequence(
            pl_input,
            base,
            init_output=target[0] if init_feature is None else None,
            init_feature=init_feature,
        )["pl"]).cpu()
        out.append({
            "name": row["name"],
            "num_frames": row["num_frames"],
            "pl_target": row["pl_target"],
            "pl_output": pred,
            "pl_input": row["pl_input"],
        })
    return out, manifest, config


def summarize_run(rows):
    leaf_values = []
    gravity_values = []
    sequence_leaf_means = []
    sequence_gravity_means = []
    target_diffs = []
    per_leaf_values = {name: [] for name in LEAF_NAMES}
    for row in rows:
        leaf = leaf_error_cm(row["pl_output"], row["pl_target"])
        grav = gravity_angle_deg(row["pl_output"][..., 15:], row["pl_target"][..., 15:])
        leaf_values.append(leaf.reshape(-1))
        gravity_values.append(grav.reshape(-1))
        sequence_leaf_means.append(leaf.mean())
        sequence_gravity_means.append(grav.mean())
        for leaf_idx, leaf_name in enumerate(LEAF_NAMES):
            per_leaf_values[leaf_name].append(leaf[:, leaf_idx])
        target_diffs.append(torch.zeros(1))
    leaf_all = torch.cat(leaf_values)
    grav_all = torch.cat(gravity_values)
    per_leaf = {
        name: summarize(torch.cat(values))
        for name, values in per_leaf_values.items()
    }
    return {
        "num_sequences": len(rows),
        "num_frames": int(sum(row["num_frames"] for row in rows)),
        "all_finite": bool(all(
            torch.isfinite(row["pl_output"]).all()
            and torch.isfinite(row["pl_target"]).all()
            and torch.isfinite(row["pl_input"]).all()
            for row in rows
        )),
        "frame_weighted": {
            "pRB_mean_cm": summarize(leaf_all),
            "gR1_mean_deg": summarize(grav_all),
            "pRB_per_leaf_cm": per_leaf,
        },
        "sequence_equal": {
            "pRB_mean_cm": summarize(torch.stack(sequence_leaf_means)),
            "gR1_mean_deg": summarize(torch.stack(sequence_gravity_means)),
        },
    }


def compact_rows(rows):
    compact = []
    for row in rows:
        leaf = leaf_error_cm(row["pl_output"], row["pl_target"])
        grav = gravity_angle_deg(row["pl_output"][..., 15:], row["pl_target"][..., 15:])
        compact.append({
            "name": row["name"],
            "num_frames": row["num_frames"],
            "pRB_mean_cm_frame_weighted_within_sequence": float(leaf.mean()),
            "gR1_mean_deg_frame_weighted_within_sequence": float(grav.mean()),
            "finite": bool(
                torch.isfinite(row["pl_output"]).all()
                and torch.isfinite(row["pl_target"]).all()
                and torch.isfinite(row["pl_input"]).all()
            ),
        })
    return compact


def main():
    parser = argparse.ArgumentParser(description="Unified PL output metric audit for S4.")
    parser.add_argument("--official-base-cache", type=Path, default=DEFAULT_OFFICIAL_BASE_CACHE)
    parser.add_argument("--processed-base-cache", type=Path, default=DEFAULT_PROCESSED_BASE_CACHE)
    parser.add_argument("--official-pl-cache", type=Path, default=DEFAULT_OFFICIAL_PL_CACHE)
    parser.add_argument("--processed-pl-cache", type=Path, default=DEFAULT_PROCESSED_PL_CACHE)
    parser.add_argument("--pl-v3-checkpoint", type=Path, default=DEFAULT_PL_V3_CHECKPOINT)
    parser.add_argument("--pl-v2-checkpoint", type=Path, default=DEFAULT_PL_V2_CHECKPOINT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "started",
        "target_definition": (
            "pl_target_from_pose: SMPL male vert_mask=GPNet.v_imu; "
            "pRB=(verts[:5]-verts[5:]) @ pose_root, flattened [5,3]; "
            "gR1=-pose_root[:,1], normalized for metric."
        ),
        "leaf_order": list(LEAF_NAMES),
        "metric_definition": {
            "pRB": "L2 norm over 3D leaf vector, meters*100 to cm.",
            "gR1": "acos of normalized gravity direction dot product, degrees.",
            "frame_weighted": "concatenate all frames/leaves before taking mean.",
            "sequence_equal": "mean per sequence first, then unweighted mean over sequences.",
        },
        "notes": [
            "No training.",
            "Official weights are loaded by GPNet from data/weights.pt and not modified.",
            "MotionEvaluator and test.py are not used or modified.",
            "Streaming original PL uses GPNet.rnn_initialize(pose_gt[0]) and _run_pl_stage frame by frame.",
            "Batch/cache original PL uses RNNWithInit.forward([(pl_input, pl_target[0])]) as stored in pl_curve_cache pl_base.",
        ],
    }
    try:
        runs = []
        run_specs = [
            ("original_streaming_official", "official", "streaming", args.official_base_cache, None, None),
            ("original_streaming_processed", "processed", "streaming", args.processed_base_cache, None, None),
            ("original_batch_cache_official", "official", "batch_cache", args.official_pl_cache, None, None),
            ("original_batch_cache_processed", "processed", "batch_cache", args.processed_pl_cache, None, None),
            ("plcurve_v3_official_no_baseline", "official", "plcurve", args.official_pl_cache, args.pl_v3_checkpoint, None),
            ("plcurve_v2_processed_no_baseline", "processed", "plcurve", args.processed_pl_cache, args.pl_v2_checkpoint, None),
        ]
        for name, input_mode, output_source, cache, checkpoint, _ in run_specs:
            config = None
            if output_source == "streaming":
                rows, manifest = streaming_original_outputs(cache, input_mode)
            elif output_source == "batch_cache":
                rows, manifest = cache_original_outputs(cache)
            elif output_source == "plcurve":
                rows, manifest, config = curve_outputs(cache, checkpoint)
            else:
                raise ValueError(output_source)
            summary = summarize_run(rows)
            runs.append({
                "run": name,
                "input_mode": input_mode,
                "output_source": output_source,
                "cache": str(cache),
                "checkpoint": str(checkpoint) if checkpoint else None,
                "checkpoint_config": config,
                "manifest_summary": {
                    "type": manifest.get("type") if isinstance(manifest, dict) else None,
                    "cache_type": manifest.get("cache_type") if isinstance(manifest, dict) else None,
                    "imu_input_mode": manifest.get("imu_input_mode") if isinstance(manifest, dict) else None,
                    "num_sequences": manifest.get("num_sequences") if isinstance(manifest, dict) else None,
                    "num_frames": manifest.get("num_frames") if isinstance(manifest, dict) else None,
                    "source_cache": manifest.get("source_cache") if isinstance(manifest, dict) else None,
                },
                "summary": summary,
                "rows": compact_rows(rows),
            })
            print(json.dumps({
                "run": name,
                "input_mode": input_mode,
                "output_source": output_source,
                "pRB_frame_weighted_cm": summary["frame_weighted"]["pRB_mean_cm"]["mean"],
                "gR1_frame_weighted_deg": summary["frame_weighted"]["gR1_mean_deg"]["mean"],
                "pRB_sequence_equal_cm": summary["sequence_equal"]["pRB_mean_cm"]["mean"],
                "gR1_sequence_equal_deg": summary["sequence_equal"]["gR1_mean_deg"]["mean"],
                "num_frames": summary["num_frames"],
            }), flush=True)
        result.update({
            "status": "ok",
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
