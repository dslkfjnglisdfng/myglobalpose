import argparse
import json
from pathlib import Path

import torch

import articulate as art
from imu_joint_euler_qdot_vel_ctrl import (
    DT,
    EULER_SEQ,
    IMUJointEulerQdotVelControlModule,
    IMU_JOINTS,
    average_metric_dicts,
    imu_rootframe_features,
    metric_dict,
    model_contract,
    root_relative_targets_from_pose,
    rotation_geodesic_deg,
)
from l4_train_diverse_short import DEVICE, load_records


def selected_imu_fields(record, mode):
    if mode == "official":
        return record["aM"], record["wM"], record["RMB"]
    has_l4 = all(key in record for key in ("l4_aM", "l4_wM", "l4_RMB"))
    if mode == "processed":
        if not has_l4:
            raise KeyError(f"processed mode requires l4_aM/l4_wM/l4_RMB in {record.get('name')}")
        return record["l4_aM"], record["l4_wM"], record["l4_RMB"]
    if mode == "auto":
        return (record["l4_aM"], record["l4_wM"], record["l4_RMB"]) if has_l4 else (record["aM"], record["wM"], record["RMB"])
    raise ValueError(f"Unsupported imu_input_mode={mode!r}")


def build_features(record, mode):
    aM, wM, RMB = selected_imu_fields(record, mode)
    return imu_rootframe_features(aM.float(), wM.float(), RMB.float())


def target_for_record(record, body_model, args, pose_key="pose_gt"):
    return root_relative_targets_from_pose(
        record[pose_key],
        body_model,
        DEVICE,
        dt=args.dt,
        euler_seq=args.euler_seq,
    )


def init_from_target(target):
    return torch.cat((target["q_euler"][0], target["qdot_euler"][0], target["vel_RJ"][0]), dim=-1).to(DEVICE)


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    if checkpoint.get("model_type") != "imu_joint_euler_qdot_vel_ctrl_v1":
        raise ValueError(f"{path} is not an imu_joint_euler_qdot_vel_ctrl_v1 checkpoint.")
    config = checkpoint.get("config", {})
    model = IMUJointEulerQdotVelControlModule(
        hidden_size=int(config.get("hidden_size", 512)),
        num_layers=int(config.get("num_layers", 3)),
        dropout=float(config.get("dropout", 0.4)),
        dt=float(config.get("dt", DT)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def input_rmb_rotation_baseline(record, target, mode, euler_seq=EULER_SEQ):
    _, _, RMB = selected_imu_fields(record, mode)
    R = RMB.float()
    R_root = R[:, 5]
    R_sel = R[:, list(range(6))]
    R_rel = R_root.transpose(-1, -2).unsqueeze(1).matmul(R_sel)
    euler = art.math.rotation_matrix_to_euler_angle(R_rel.reshape(-1, 3, 3), seq=euler_seq).view(-1, 6, 3)
    target_q = target["q_euler"].reshape(-1, 6, 3)
    geo = rotation_geodesic_deg(euler, target_q, seq=euler_seq).view(-1, 6)
    return {
        "input_RMB_root_relative_rotation_geodesic_deg": float(geo.mean()),
        "input_RMB_per_joint_rotation_geodesic_deg": [float(v) for v in geo.mean(dim=0)],
        "input_RMB_note": "R_rootIMU^T R_sensorIMU baseline; sensor orientation is not guaranteed to equal joint orientation.",
    }


@torch.no_grad()
def evaluate(args):
    model, checkpoint = load_model(args.checkpoint)
    records, manifest = load_records(args.cache, max_sequences=args.max_sequences)
    body_model = art.ParametricModel("models/SMPL_male.pkl", device=DEVICE)
    rows = []
    for record in records:
        features = build_features(record, args.imu_input_mode).to(DEVICE)
        target = target_for_record(record, body_model, args)
        output = model.forward_sequence(features, init_state=init_from_target(target))
        baseline = target_for_record(record, body_model, args, pose_key="pose_prephysics") if "pose_prephysics" in record else None
        metrics = metric_dict(output, target, baseline_target=baseline)
        metrics.update(input_rmb_rotation_baseline(record, target, args.imu_input_mode, args.euler_seq))
        rows.append({"name": record["name"], "num_frames": int(record["pose_gt"].shape[0]), "metrics": metrics})
    aggregate = average_metric_dicts([row["metrics"] for row in rows])
    return {
        "status": "ok",
        "contract": model_contract(),
        "dataset": args.dataset,
        "dataset_label": args.dataset_label or args.dataset,
        "cache": args.cache,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "variant": checkpoint.get("config", {}).get("variant"),
        "imu_input_mode": args.imu_input_mode,
        "dip_policy": "DIP trans is not used; targets are pose-derived root-relative q/vel/acc." if args.dataset == "dip" else "",
        "num_sequences": len(rows),
        "aggregate": aggregate,
        "rows": rows,
        "source_manifest": manifest,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate imu_joint_euler_qdot_vel_ctrl_v1.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", choices=("amass", "totalcapture", "dip"), required=True)
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--imu-input-mode", choices=("official", "processed", "auto"), default="official")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--euler-seq", default=EULER_SEQ)
    args = parser.parse_args()
    result = evaluate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output_json": str(output_path), "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
