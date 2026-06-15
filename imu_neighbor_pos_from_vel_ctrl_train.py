import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from imu_neighbor_pos_from_vel_ctrl import (
    DT,
    IMUNeighborPositionFromVelocityModule,
    default_loss_weights,
    mix_velocity_inputs,
    model_contract,
    neighbor_position_loss,
    neighbor_position_targets_from_pose,
    position_input_features,
    selection_value,
    velocity_pack_keys,
)
from imu_neighbor_vel_ctrl import (
    IMUNeighborVelocityControlModule,
    imu_neighbor_features,
    neighbor_velocity_targets_from_pose_tran,
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
    raise KeyError(f"{record.get('name')} missing offset_r/r_JS required by imu_neighbor_pos_from_vel_ctrl_v1")


def record_length(record):
    if "seq_len" in record:
        return int(record["seq_len"])
    return int(record["pose_gt"].shape[0])


def slice_record(record, start, length):
    seq_len = record_length(record)
    if length <= 0 or seq_len <= length:
        return record
    start = min(max(0, int(start)), seq_len - length)
    end = start + length
    out = {}
    for key, value in record.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == seq_len:
            out[key] = value[start:end]
        elif isinstance(value, dict):
            sliced = {}
            changed = False
            for sub_key, sub_value in value.items():
                if torch.is_tensor(sub_value) and sub_value.ndim > 0 and sub_value.shape[0] == seq_len:
                    sliced[sub_key] = sub_value[start:end]
                    changed = True
                else:
                    sliced[sub_key] = sub_value
            out[key] = sliced if changed else value
        else:
            out[key] = value
    out["seq_len"] = end - start
    out["name"] = f"{record['name']}[{start}:{end}]"
    return out


def build_imu_feature(record, mode, allow_zero_offset=False):
    if "imu_feature" in record:
        return record["imu_feature"].float()
    aM, wM, RMB = selected_imu_fields(record, mode)
    return imu_neighbor_features(aM.float(), wM.float(), RMB.float(), offset_for_record(record, allow_zero_offset))


def load_velocity_model(path):
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def predict_velocity_pack(velocity_model, imu_feature):
    out = velocity_model.forward_sequence(imu_feature.to(DEVICE))
    return {key: out[key].detach().cpu() for key in velocity_pack_keys()}


def gt_velocity_pack(record, body_model, dataset, dt):
    if dataset == "dip":
        return None
    if "tran_gt" not in record:
        return None
    target = neighbor_velocity_targets_from_pose_tran(record["pose_gt"], record["tran_gt"], body_model, DEVICE, dt=dt)
    return {key: target[key].detach().cpu() for key in velocity_pack_keys()}


def precompute_records(records, body_model, velocity_model, args, split_name):
    compact_records = []
    gt_velocity_sequences = 0
    for record in records:
        imu_feature = build_imu_feature(record, args.imu_input_mode, args.allow_zero_offset).detach().cpu()
        pred_velocity = predict_velocity_pack(velocity_model, imu_feature)
        gt_velocity = gt_velocity_pack(record, body_model, args.dataset, args.dt)
        if gt_velocity is not None:
            gt_velocity_sequences += 1
        position_target = neighbor_position_targets_from_pose(record["pose_gt"], body_model, DEVICE, dt=args.dt)
        compact = {
            "name": record["name"],
            "seq_len": int(imu_feature.shape[0]),
            "imu_feature": imu_feature,
            "pred_velocity": pred_velocity,
            "position_target": {key: value.detach().cpu() for key, value in position_target.items()},
            "gt_velocity_input_available": gt_velocity is not None,
        }
        if gt_velocity is not None:
            compact["gt_velocity"] = gt_velocity
        compact_records.append(compact)
    return compact_records, {
        "split": split_name,
        "num_sequences": len(compact_records),
        "gt_velocity_input_sequences": gt_velocity_sequences,
        "feature_dim": 189,
        "target_dim": 33,
        "mode": "in_memory_compact_precompute",
        "dip_trans_policy": "not used" if args.dataset == "dip" else "not applicable",
    }


def build_position_feature(record, gt_velocity_ratio):
    imu_feature = record["imu_feature"].float()
    velocity_input = mix_velocity_inputs(record["pred_velocity"], record.get("gt_velocity"), gt_velocity_ratio)
    return position_input_features(imu_feature, velocity_input)


def stack_dict(items):
    return {key: torch.stack([item[key] for item in items], dim=1).to(DEVICE) for key in items[0]}


def run_records(model, records, weights, gt_velocity_ratio, train_mode):
    features, targets, names = [], [], []
    for record in records:
        features.append(build_position_feature(record, gt_velocity_ratio).to(DEVICE))
        targets.append(record["position_target"])
        names.append(record["name"])
    features_tbd = torch.stack(features, dim=1).to(DEVICE)
    target = stack_dict(targets)
    out = model.forward_sequence(features_tbd)
    loss, components = neighbor_position_loss(out, target, weights)
    row = {
        "name": "|".join(str(name) for name in names),
        "loss": float(loss.detach()),
        "gt_velocity_ratio": float(gt_velocity_ratio),
    }
    row.update({key: float(value.detach()) for key, value in components.items()})
    if not train_mode:
        row["selection_value"] = selection_value(row)
    return loss, row


def average(rows):
    if not rows:
        return {}
    numeric = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, bool)):
                numeric.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in numeric.items()}


def epoch_gt_velocity_ratio(args, epoch):
    if args.dataset == "dip":
        return 0.0
    if args.gt_vel_mix_epochs <= 0:
        return float(args.gt_vel_mix_final)
    progress = min(1.0, max(0.0, float(epoch - 1) / float(args.gt_vel_mix_epochs)))
    return float(args.gt_vel_mix_start) + (float(args.gt_vel_mix_final) - float(args.gt_vel_mix_start)) * progress


def eval_model(model, records, weights, args):
    model.eval()
    rows = []
    selected = records[: args.max_val_sequences] if args.max_val_sequences else records
    with torch.no_grad():
        for record in selected:
            _, row = run_records(model, [record], weights, args.val_gt_vel_mix_ratio, train_mode=False)
            rows.append(row)
    return {"num_sequences": len(rows), "loss": average(rows), "rows": rows}


def save_checkpoint(path, model, optimizer, args, epoch, step, selected_loss, weights):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "epoch": int(epoch),
            "step": int(step),
            "selection_value": float(selected_loss),
            "weights": weights,
            "model_type": "imu_neighbor_pos_from_vel_ctrl_v1",
            "contract": model_contract(),
            "velocity_checkpoint": args.velocity_checkpoint,
        },
        path,
    )


def load_model_from_checkpoint(path):
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
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train imu_neighbor_pos_from_vel_ctrl_v1.")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("amass", "totalcapture", "dip"), required=True)
    parser.add_argument("--velocity-checkpoint", required=True)
    parser.add_argument("--imu-input-mode", choices=("official", "processed", "auto"), default="official")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--window", type=int, default=61)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-sequences", type=int, default=0)
    parser.add_argument("--max-val-sequences", type=int, default=0)
    parser.add_argument("--allow-zero-offset", action="store_true")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--gt-vel-mix-start", type=float, default=0.0)
    parser.add_argument("--gt-vel-mix-final", type=float, default=0.0)
    parser.add_argument("--gt-vel-mix-epochs", type=int, default=1)
    parser.add_argument("--val-gt-vel-mix-ratio", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    args = parser.parse_args()

    if args.batch_size <= 1:
        raise RuntimeError("Long module training must be batched; choose --batch-size > 1.")
    if args.dataset == "dip" and (
        args.gt_vel_mix_start > 0.0 or args.gt_vel_mix_final > 0.0 or args.val_gt_vel_mix_ratio > 0.0
    ):
        raise RuntimeError("DIP cannot use GT world velocity input; set all GT velocity mix ratios to 0.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    train_records, train_manifest = load_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, max_sequences=args.max_val_sequences)
    train_records = [record for record in train_records if record_length(record) >= args.window]
    if not train_records:
        raise RuntimeError(f"No train sequence has at least window={args.window} frames.")

    body_model = art.ParametricModel("models/SMPL_male.pkl", device=DEVICE)
    velocity_model, velocity_checkpoint = load_velocity_model(args.velocity_checkpoint)
    train_records, train_precompute = precompute_records(train_records, body_model, velocity_model, args, "train")
    val_records, val_precompute = precompute_records(val_records, body_model, velocity_model, args, "val")
    precompute_summary = {
        "enabled": True,
        "train": train_precompute,
        "val": val_precompute,
        "velocity_checkpoint_epoch": velocity_checkpoint.get("epoch"),
    }
    (output_dir / "precompute_summary.json").write_text(json.dumps(precompute_summary, indent=2) + "\n")

    weights = default_loss_weights(args.dataset)
    model = IMUNeighborPositionFromVelocityModule(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dt=args.dt,
    ).to(DEVICE)
    init_load = None
    if args.init_checkpoint:
        init_model, checkpoint = load_model_from_checkpoint(args.init_checkpoint)
        model.load_state_dict(init_model.state_dict())
        init_load = {"path": args.init_checkpoint, "epoch": checkpoint.get("epoch")}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_value = float("inf")
    best_epoch = 0
    stale_epochs = 0
    step = 0
    history = []
    log_path = output_dir / "train_log.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_rows = []
        gt_ratio = epoch_gt_velocity_ratio(args, epoch)
        for batch_start in range(0, len(train_records), args.batch_size):
            step += 1
            batch_records = []
            for offset, source in enumerate(train_records[batch_start:batch_start + args.batch_size]):
                max_start = max(0, record_length(source) - args.window)
                start = (step + offset) % (max_start + 1) if max_start > 0 else 0
                batch_records.append(slice_record(source, start, args.window))
            loss, row = run_records(model, batch_records, weights, gt_ratio, train_mode=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {row['name']}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            row.update({"epoch": epoch, "step": step})
            train_rows.append(row)
        validation = eval_model(model, val_records, weights, args)
        selected = selection_value(validation["loss"])
        improved = selected < best_value if best_value == float("inf") else (best_value - selected) > args.early_stop_min_delta
        if improved:
            best_value = selected
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(output_dir / "best_loss.pt", model, optimizer, args, epoch, step, selected, weights)
        else:
            stale_epochs += 1
        save_checkpoint(output_dir / "last.pt", model, optimizer, args, epoch, step, selected, weights)
        epoch_row = {
            "epoch": epoch,
            "step": step,
            "gt_velocity_ratio": gt_ratio,
            "train_loss": average(train_rows),
            "validation": validation,
            "selection_value": selected,
            "best_value": best_value,
            "best_epoch": best_epoch,
            "improved": improved,
        }
        history.append(epoch_row)
        with log_path.open("a") as f:
            f.write(json.dumps(epoch_row) + "\n")
        if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
            break

    result = {
        "status": "ok",
        "contract": model_contract(),
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "precompute": precompute_summary,
        "init_checkpoint_load": init_load,
        "velocity_checkpoint": args.velocity_checkpoint,
        "weights": weights,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "history": history,
        "checkpoints": {
            "best": str(output_dir / "best_loss.pt"),
            "last": str(output_dir / "last.pt"),
        },
    }
    (output_dir / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "best_epoch": best_epoch, "best_value": best_value, "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
