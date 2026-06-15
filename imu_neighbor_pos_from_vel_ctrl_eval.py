import argparse
import json
from pathlib import Path

import torch

import articulate as art
from imu_neighbor_pos_from_vel_ctrl import (
    DT,
    IMUNeighborPositionFromVelocityModule,
    average_metric_dicts,
    mix_velocity_inputs,
    model_contract,
    neighbor_position_targets_from_pose,
    position_input_features,
    position_metric_dict,
    velocity_pack_keys,
)
from imu_neighbor_pos_from_vel_ctrl_train import (
    build_imu_feature,
    gt_velocity_pack,
    load_velocity_model,
    predict_velocity_pack,
)
from l4_train_diverse_short import DEVICE, load_records


def load_position_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    if checkpoint.get("model_type") != "imu_neighbor_pos_from_vel_ctrl_v1":
        raise ValueError(f"{path} is not an imu_neighbor_pos_from_vel_ctrl_v1 checkpoint.")
    config = checkpoint.get("config", {})
    model = IMUNeighborPositionFromVelocityModule(
        hidden_size=int(config.get("hidden_size", 512)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
        dt=float(config.get("dt", DT)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def build_feature_for_record(record, velocity_model, body_model, args):
    imu_feature = build_imu_feature(record, args.imu_input_mode, args.allow_zero_offset)
    pred_velocity = predict_velocity_pack(velocity_model, imu_feature)
    gt_velocity = None
    if args.gt_vel_input_ratio > 0.0:
        gt_velocity = gt_velocity_pack(record, body_model, args.dataset, args.dt)
        if gt_velocity is None:
            raise RuntimeError(f"{record.get('name')} has no allowed GT velocity input for ratio={args.gt_vel_input_ratio}.")
    velocity_input = mix_velocity_inputs(pred_velocity, gt_velocity, args.gt_vel_input_ratio)
    return position_input_features(imu_feature, velocity_input).to(DEVICE), {
        "pred_velocity_keys": list(velocity_pack_keys()),
        "gt_velocity_input_used": gt_velocity is not None and args.gt_vel_input_ratio > 0.0,
    }


def baseline_for_record(record, body_model, dt):
    if "pose_baseline" in record:
        return neighbor_position_targets_from_pose(record["pose_baseline"], body_model, DEVICE, dt=dt), "pose_baseline FK root-relative"
    if "pose_prephysics" in record:
        return neighbor_position_targets_from_pose(record["pose_prephysics"], body_model, DEVICE, dt=dt), "pose_prephysics FK root-relative"
    return None, "pose baseline not available"


@torch.no_grad()
def evaluate(args):
    if args.dataset == "dip" and args.gt_vel_input_ratio > 0.0:
        raise RuntimeError("DIP cannot use GT world velocity input; set --gt-vel-input-ratio 0.")
    model, checkpoint = load_position_model(args.checkpoint)
    velocity_model, velocity_checkpoint = load_velocity_model(args.velocity_checkpoint)
    records, manifest = load_records(args.cache, max_sequences=args.max_sequences)
    body_model = art.ParametricModel("models/SMPL_male.pkl", device=DEVICE)
    rows = []
    for record in records:
        features, feature_info = build_feature_for_record(record, velocity_model, body_model, args)
        output = model.forward_sequence(features)
        target = neighbor_position_targets_from_pose(record["pose_gt"], body_model, DEVICE, dt=args.dt)
        target = {key: value.to(DEVICE) for key, value in target.items()}
        baseline_target, baseline_source = baseline_for_record(record, body_model, args.dt)
        if baseline_target is not None:
            baseline_target = {key: value.to(DEVICE) for key, value in baseline_target.items()}
        metrics = position_metric_dict(output, target, baseline_target=baseline_target, baseline_source=baseline_source)
        rows.append({
            "name": record["name"],
            "num_frames": int(record["pose_gt"].shape[0]),
            "feature_info": feature_info,
            "metrics": metrics,
        })
    aggregate = average_metric_dicts([row["metrics"] for row in rows])
    result = {
        "status": "ok",
        "contract": model_contract(),
        "dataset": args.dataset,
        "dataset_label": args.dataset_label or args.dataset,
        "cache": args.cache,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "velocity_checkpoint": args.velocity_checkpoint,
        "velocity_checkpoint_epoch": velocity_checkpoint.get("epoch"),
        "gt_velocity_input_ratio": args.gt_vel_input_ratio,
        "dip_policy": "no DIP trans or world/root velocity GT used" if args.dataset == "dip" else "",
        "baseline_comparison": {
            "official_PL": "not applicable: cache does not contain official PL output for this 33D neighbor-node layout",
            "newpl_v4_init36": "not applicable: cache does not contain newpl_v4 output for this 33D neighbor-node layout",
            "pose_prephysics": aggregate.get("baseline_source", "pose baseline not available"),
            "velocity_integration": "not measured: diagnostic only and not the main baseline for root-relative position",
        },
        "num_sequences": len(rows),
        "aggregate": aggregate,
        "rows": rows,
        "source_manifest": manifest,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate imu_neighbor_pos_from_vel_ctrl_v1 module outputs.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--velocity-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", choices=("amass", "totalcapture", "dip"), required=True)
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--imu-input-mode", choices=("official", "processed", "auto"), default="official")
    parser.add_argument("--gt-vel-input-ratio", type=float, default=0.0)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--allow-zero-offset", action="store_true")
    parser.add_argument("--dt", type=float, default=DT)
    args = parser.parse_args()
    result = evaluate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output_json": str(output_path), "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
