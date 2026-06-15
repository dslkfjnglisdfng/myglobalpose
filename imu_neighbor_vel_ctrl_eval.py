import argparse
import json
from pathlib import Path

import torch

import articulate as art
from imu_neighbor_vel_ctrl import (
    DT,
    IMUNeighborVelocityControlModule,
    average_metric_dicts,
    imu_neighbor_features,
    metric_dict,
    model_contract,
    neighbor_velocity_targets_from_pose_tran,
    world_gt_available,
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


def offset_for_record(record, allow_zero_offset=False):
    if "offset_r" in record:
        return record["offset_r"].float()
    if "imu_offset_r" in record:
        return record["imu_offset_r"].float()
    if allow_zero_offset:
        return torch.zeros(6, 3)
    raise KeyError(f"{record.get('name')} missing offset_r/r_JS required by imu_neighbor_vel_ctrl_v1")


def build_features(record, mode, allow_zero_offset=False):
    aM, wM, RMB = selected_imu_fields(record, mode)
    return imu_neighbor_features(aM.float(), wM.float(), RMB.float(), offset_for_record(record, allow_zero_offset))


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    if checkpoint.get("model_type") != "imu_neighbor_vel_ctrl_v1":
        raise ValueError(f"{path} is not an imu_neighbor_vel_ctrl_v1 checkpoint.")
    config = checkpoint.get("config", {})
    model = IMUNeighborVelocityControlModule(
        hidden_size=int(config.get("hidden_size", 512)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
        dt=float(config.get("dt", DT)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def target_for_record(record, body_model, dataset, world_gt_mode, dt):
    gt = world_gt_available(dataset, world_gt_mode)
    if not gt:
        return None, False
    if "tran_gt" not in record:
        raise KeyError(f"{record.get('name')} missing tran_gt for world-frame velocity GT")
    return neighbor_velocity_targets_from_pose_tran(record["pose_gt"], record["tran_gt"], body_model, DEVICE, dt=dt), True


def baseline_for_record(record, body_model, target, gt, dt):
    if not gt or target is None:
        return None, None, None
    if "pose_baseline" in record and "tran_baseline" in record:
        baseline = neighbor_velocity_targets_from_pose_tran(record["pose_baseline"], record["tran_baseline"], body_model, DEVICE, dt=dt)
        return baseline, baseline["vel_W"][..., -3:], "pose_baseline/tran_baseline finite difference"
    if "v_root_vr" in record:
        v_root = record["v_root_vr"].float()
        if v_root.shape[-1] == 3:
            return None, v_root, "v_root_vr official VR root velocity"
    return None, None, None


@torch.no_grad()
def evaluate(args):
    model, checkpoint = load_model(args.checkpoint)
    records, manifest = load_records(args.cache, max_sequences=args.max_sequences)
    body_model = art.ParametricModel("models/SMPL_male.pkl", device=DEVICE)
    rows = []
    for record in records:
        features = build_features(record, args.imu_input_mode, args.allow_zero_offset).to(DEVICE)
        output = model.forward_sequence(features)
        target, gt = target_for_record(record, body_model, args.dataset, args.world_gt_mode, args.dt)
        if target is not None:
            target = {key: value.to(DEVICE) for key, value in target.items()}
        baseline, baseline_root_velocity, baseline_root_source = baseline_for_record(record, body_model, target, gt, args.dt)
        if baseline is not None:
            baseline = {key: value.to(DEVICE) for key, value in baseline.items()}
        if baseline_root_velocity is not None:
            baseline_root_velocity = baseline_root_velocity.to(DEVICE)
        metrics = metric_dict(
            output,
            target,
            gt,
            baseline_output=baseline,
            baseline_root_velocity=baseline_root_velocity,
            baseline_root_source=baseline_root_source,
        )
        rows.append({"name": record["name"], "num_frames": int(record["pose_gt"].shape[0]), "metrics": metrics})
    aggregate = average_metric_dicts([row["metrics"] for row in rows])
    result = {
        "status": "ok",
        "contract": model_contract(),
        "dataset": args.dataset,
        "dataset_label": args.dataset_label or args.dataset,
        "cache": args.cache,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "world_gt_available": world_gt_available(args.dataset, args.world_gt_mode),
        "world_gt_note": "world velocity GT not available" if not world_gt_available(args.dataset, args.world_gt_mode) else "",
        "baseline_note": aggregate.get("baseline_velocity_source", "baseline velocity not available"),
        "num_sequences": len(rows),
        "aggregate": aggregate,
        "rows": rows,
        "source_manifest": manifest,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate imu_neighbor_vel_ctrl_v1 module outputs.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", choices=("amass", "totalcapture", "dip"), required=True)
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--imu-input-mode", choices=("official", "processed", "auto"), default="official")
    parser.add_argument("--world-gt-mode", choices=("auto", "available", "unavailable"), default="auto")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--allow-zero-offset", action="store_true")
    parser.add_argument("--dt", type=float, default=DT)
    args = parser.parse_args()
    if args.dataset == "dip" and args.world_gt_mode == "available":
        raise RuntimeError("DIP world velocity/acceleration GT is forbidden; use --world-gt-mode auto or unavailable.")
    result = evaluate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output_json": str(output_path), "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
