import argparse
import json
import math
import random
from pathlib import Path

import torch
import tqdm

import articulate as art
from imu_offset_net import (
    IMUOffsetNet,
    OFFSET_COORDINATE_CONTRACT,
    make_checkpoint,
    offset_input_feature,
    offset_regularizers,
)
from imu_offset_infer import selected_imu_fields, trim_sequence
from l4_sensor_offset_utils import IMU_JOINTS
from l4_train_diverse_short import load_records
from pl_curve import normalize_gravity, pl_target_from_pose


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    config = checkpoint.get("config", {})
    model = IMUOffsetNet(
        version=checkpoint.get("version", config.get("version", "offset_v1_mlp_frame")),
        input_size=int(config.get("feature_dim", config.get("input_size", 108))),
        hidden_size=int(config.get("hidden_size", 256)),
        prior_offset=torch.tensor(config.get("prior_offset_median", torch.zeros(6, 3))).float(),
        residual_scale=float(config.get("residual_scale", 0.05)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint, config


@torch.no_grad()
def attach_pl_outputs(records, max_frames=0):
    body = art.ParametricModel("models/SMPL_male.pkl", vert_mask=torch.tensor([1961, 5424, 1176, 4662, 411, 3021]), device=DEVICE)
    for record in tqdm.tqdm(records, desc="pl outputs"):
        pose = record["pose_prephysics"].float()
        if max_frames:
            pose = pose[:max_frames]
        chunks = []
        for start in range(0, pose.shape[0], 2048):
            p = pose[start : start + 2048].to(DEVICE)
            chunks.append(normalize_gravity(pl_target_from_pose(p, body)).detach().cpu())
        record["pl_output"] = torch.cat(chunks, dim=0)
    return records


def split_windows(records, window, windows_per_sequence, seed):
    rng = random.Random(seed)
    windows = []
    for rec in records:
        n = min(rec["pl_output"].shape[0], rec["aM"].shape[0], rec["wM"].shape[0], rec["RMB"].shape[0])
        if n <= window:
            starts = [0]
        else:
            starts = [rng.randint(0, n - window) for _ in range(max(1, windows_per_sequence))]
        windows.extend((rec, s, min(n, s + window)) for s in starts)
    rng.shuffle(windows)
    return windows


def feature_for(record, start, end):
    return offset_input_feature(
        record["pl_output"][start:end],
        record["aM"][start:end],
        record["wM"][start:end],
        record["RMB"][start:end],
    )


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def pose_kinematics(record, start, end):
    pose = record["pose_gt"][start:end].to(DEVICE)
    joints = record.get("_imu_joints_gt")
    rotations = record.get("_imu_rotations_gt")
    if joints is None or rotations is None:
        body = art.ParametricModel("models/SMPL_male.pkl", device=DEVICE)
        grot, joint = body.forward_kinematics(record["pose_gt"].to(DEVICE))[:2]
        joints = joint[:, IMU_JOINTS].detach().cpu()
        rotations = grot[:, IMU_JOINTS].detach().cpu()
        record["_imu_joints_gt"] = joints
        record["_imu_rotations_gt"] = rotations
    return joints[start:end].to(DEVICE), rotations[start:end].to(DEVICE)


def finite_second_like(x):
    if x.shape[0] < 3:
        return torch.zeros_like(x)
    acc = torch.zeros_like(x)
    acc[1:-1] = (x[:-2] - 2.0 * x[1:-1] + x[2:]) * (60.0 ** 2)
    acc[0] = acc[1]
    acc[-1] = acc[-2]
    return acc


def consistency_losses(record, start, end, pred_offset):
    frame_offset = pred_offset[0]
    seq_offset = frame_offset.mean(dim=0)
    a_meas = record["aM"][start:end].to(DEVICE)
    joints, rotations = pose_kinematics(record, start, end)
    p_sensor = joints + rotations.matmul(frame_offset.unsqueeze(-1)).squeeze(-1)
    sensor_acc_proxy = finite_second_like(p_sensor)
    # No DIP trans is used here. This is a pose-derived, translation-free
    # acceleration proxy; subtracting each sequence mean avoids treating the
    # unavailable root translation acceleration as supervision.
    centered_meas = a_meas - a_meas.mean(dim=0, keepdim=True)
    centered_proxy = sensor_acc_proxy - sensor_acc_proxy.mean(dim=0, keepdim=True)
    acc_proxy = torch.nn.functional.smooth_l1_loss(centered_meas, centered_proxy)
    regs = offset_regularizers(pred_offset)
    return {
        "pose_acc_proxy": acc_proxy,
        "temporal_smooth": regs["temporal_smooth"],
        "magnitude": regs["magnitude"],
        "offset_sequence_std": pred_offset[0].std(dim=0).norm(dim=-1).mean(),
        "offset_mean_norm": seq_offset.norm(dim=-1).mean(),
    }


@torch.no_grad()
def evaluate(model, records, args):
    model.eval()
    rows = []
    for rec, start, end in split_windows(records, args.window, args.val_windows_per_sequence, args.seed + 999):
        feature = feature_for(rec, start, end).unsqueeze(0).to(DEVICE)
        pred = model(feature)
        losses = consistency_losses(rec, start, end, pred)
        rows.append({k: float(v.detach().cpu()) for k, v in losses.items()})
    total = max(1, len(rows))
    return {
        "num_windows": len(rows),
        "offset_gt_available": False,
        "offset_l1_cm": "not available",
        "offset_l2_cm": "not available",
        "pose_acc_proxy": sum(row["pose_acc_proxy"] for row in rows) / total,
        "temporal_smooth": sum(row["temporal_smooth"] for row in rows) / total,
        "magnitude": sum(row["magnitude"] for row in rows) / total,
        "offset_sequence_std": sum(row["offset_sequence_std"] for row in rows) / total,
        "offset_mean_norm": sum(row["offset_mean_norm"] for row in rows) / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage B DIP fine-tune for IMUOffsetNet without trans or real offset GT.")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, default=Path("data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json"))
    parser.add_argument("--val-cache", type=Path, default=Path("data/dataset_work/L4Cache/prephysics_pose_velocity_dip_val_globalpose_neural_only/baseline_cache_manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--windows-per-sequence", type=int, default=1)
    parser.add_argument("--val-windows-per-sequence", type=int, default=1)
    parser.add_argument("--max-train-sequences", type=int, default=0)
    parser.add_argument("--max-val-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--acc-weight", type=float, default=1.0)
    parser.add_argument("--smooth-weight", type=float, default=1e-3)
    parser.add_argument("--magnitude-weight", type=float, default=1e-2)
    parser.add_argument("--std-weight", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    model, init_checkpoint, init_config = load_model(args.init_checkpoint)
    train_records, train_manifest = load_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, max_sequences=args.max_val_sequences)
    # Enforce DIP official input contract: selected_imu_fields must resolve to
    # the raw official aM/wM/RMB fields and never l4_* processed fields.
    for record in train_records + val_records:
        aM, wM, RMB, source = selected_imu_fields({"aM": [record["aM"]], "wM": [record["wM"]], "RMB": [record["RMB"]]}, 0, "official")
        record["aM"], record["wM"], record["RMB"] = trim_sequence(aM, wM, RMB, max_frames=args.max_frames)[0]
        if source != "official":
            raise RuntimeError("DIP fine-tune must use official IMU fields")
        if args.max_frames:
            original_frames = record["pose_gt"].shape[0]
            for key, value in list(record.items()):
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == original_frames:
                    record[key] = value[: args.max_frames]
    attach_pl_outputs(train_records + val_records, max_frames=args.max_frames)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json_safe(vars(args).copy())
    config.update(
        {
            "stage": "B_DIP_official_input_consistency_finetune",
            "coordinate_contract": OFFSET_COORDINATE_CONTRACT,
            "offset_gt_available": False,
            "forbidden_losses": ["trans_loss", "offset_gt_loss"],
            "imu_input_contract": "DIP uses official baseline aM/wM/RMB only; processed l4_* IMU is not used.",
            "pl_input_contract": "pose_prephysics -> PL output proxy; no pose_gt PL teacher forcing.",
            "train_sequences": len(train_records),
            "val_sequences": len(val_records),
            "init_checkpoint_config": json_safe(init_config),
            "train_manifest": train_manifest,
            "val_manifest": val_manifest,
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    initial_val = evaluate(model, val_records, args)
    (args.output_dir / "initial_val.json").write_text(json.dumps({
        "stage": "B_DIP_official_input_consistency_finetune",
        "checkpoint": str(args.init_checkpoint),
        "offset_gt_available": False,
        "offset_l1_cm": "not available",
        "offset_l2_cm": "not available",
        "val": initial_val,
    }, indent=2) + "\n")
    best = math.inf
    best_epoch = 0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        windows = split_windows(train_records, args.window, args.windows_per_sequence, args.seed + epoch)
        totals = {}
        for rec, start, end in tqdm.tqdm(windows, desc=f"dip ft epoch {epoch}"):
            pred = model(feature_for(rec, start, end).unsqueeze(0).to(DEVICE))
            losses = consistency_losses(rec, start, end, pred)
            loss = (
                args.acc_weight * losses["pose_acc_proxy"]
                + args.smooth_weight * losses["temporal_smooth"]
                + args.magnitude_weight * losses["magnitude"]
                + args.std_weight * losses["offset_sequence_std"]
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite DIP fine-tune loss on {rec['name']}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        val = evaluate(model, val_records, args)
        train = {key: value / max(1, len(windows)) for key, value in totals.items()}
        item = {"epoch": epoch, "train": train, "val": val}
        (args.output_dir / "train_log.jsonl").open("a").write(json.dumps(item) + "\n")
        selection = float(val["pose_acc_proxy"] + args.magnitude_weight * val["magnitude"])
        if selection < best:
            best = selection
            best_epoch = epoch
            torch.save(make_checkpoint(model, config, epoch, step, selection, optimizer), args.output_dir / "best_loss.pt")
        torch.save(make_checkpoint(model, config, epoch, step, selection, optimizer), args.output_dir / "last.pt")
        print(json.dumps({"epoch": epoch, "selection": selection, "val": val}, indent=2), flush=True)
    result = {
        "status": "ok",
        "stage": "B_DIP_official_input_consistency_finetune",
        "offset_gt_available": False,
        "offset_l1_cm": "not available",
        "offset_l2_cm": "not available",
        "initial_val": initial_val,
        "best_epoch": best_epoch,
        "best_selection": best,
        "last_val": evaluate(model, val_records, args),
        "checkpoints": {"best_loss": str(args.output_dir / "best_loss.pt"), "last": str(args.output_dir / "last.pt")},
        "config": config,
    }
    (args.output_dir / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "best_selection", "last_val", "checkpoints")}, indent=2))


if __name__ == "__main__":
    main()
