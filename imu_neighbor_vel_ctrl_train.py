import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from imu_neighbor_vel_ctrl import (
    DT,
    IMUNeighborVelocityControlModule,
    default_loss_weights,
    imu_neighbor_features,
    metric_dict,
    model_contract,
    neighbor_velocity_loss,
    neighbor_velocity_targets_from_pose_tran,
    selection_value,
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


def build_features(record, mode, allow_zero_offset=False):
    if "neighbor_features" in record:
        return record["neighbor_features"].float()
    aM, wM, RMB = selected_imu_fields(record, mode)
    return imu_neighbor_features(aM.float(), wM.float(), RMB.float(), offset_for_record(record, allow_zero_offset))


def build_target(record, body_model, dataset, world_gt_mode, dt):
    available = world_gt_available(dataset, world_gt_mode)
    if not available:
        return None, False
    if "neighbor_target" in record:
        return record["neighbor_target"], bool(record.get("neighbor_world_gt", True))
    if "tran_gt" not in record:
        raise KeyError(f"{record.get('name')} missing tran_gt for world-frame velocity GT")
    target = neighbor_velocity_targets_from_pose_tran(record["pose_gt"], record["tran_gt"], body_model, DEVICE, dt=dt)
    return target, True


def precompute_records(records, body_model, args, split_name):
    compact_records = []
    world_gt = world_gt_available(args.dataset, args.world_gt_mode)
    for record in records:
        feature = build_features(record, args.imu_input_mode, args.allow_zero_offset).detach().cpu()
        target, gt = build_target(record, body_model, args.dataset, args.world_gt_mode, args.dt)
        compact = {
            "name": record["name"],
            "seq_len": int(feature.shape[0]),
            "neighbor_features": feature,
            "neighbor_world_gt": bool(gt),
        }
        if target is not None:
            compact["neighbor_target"] = {key: value.detach().cpu() for key, value in target.items()}
        compact_records.append(compact)
    return compact_records, {
        "split": split_name,
        "num_sequences": len(compact_records),
        "world_gt_requested": bool(world_gt),
        "world_gt_sequences": int(sum(1 for record in compact_records if record.get("neighbor_world_gt"))),
        "feature_dim": 90,
        "target_dim": 33,
        "mode": "in_memory_compact_precompute",
    }


def stack_target(targets):
    if targets[0] is None:
        return None
    return {key: torch.stack([target[key] for target in targets], dim=1).to(DEVICE) for key in targets[0]}


@torch.no_grad()
def run_teacher(teacher, features):
    if teacher is None:
        return None
    teacher.eval()
    return teacher.forward_sequence(features)


def run_records(model, teacher, records, body_model, weights, args, train_mode):
    features, targets, gt_flags, names = [], [], [], []
    for record in records:
        features.append(build_features(record, args.imu_input_mode, args.allow_zero_offset).to(DEVICE))
        target, gt = build_target(record, body_model, args.dataset, args.world_gt_mode, args.dt)
        targets.append(target)
        gt_flags.append(gt)
        names.append(record["name"])
    features_tbd = torch.stack(features, dim=1).to(DEVICE)
    target = stack_target(targets)
    world_gt = all(gt_flags)
    out = model.forward_sequence(features_tbd)
    teacher_out = run_teacher(teacher, features_tbd)
    loss, components = neighbor_velocity_loss(out, target, weights, world_gt=world_gt, teacher_output=teacher_out)
    row = {
        "name": "|".join(str(name) for name in names),
        "loss": float(loss.detach()),
        "world_gt_available": bool(world_gt),
    }
    row.update({key: float(value.detach()) for key, value in components.items()})
    if not train_mode:
        row["metrics"] = metric_dict(out, target, world_gt)
    return loss, row


def average(rows, key="loss"):
    if not rows:
        return {}
    numeric = {}
    for row in rows:
        for item_key, value in row.items():
            if isinstance(value, (int, float, bool)):
                numeric.setdefault(item_key, []).append(float(value))
    return {item_key: sum(values) / len(values) for item_key, values in numeric.items()}


def eval_model(model, teacher, records, body_model, weights, args):
    model.eval()
    rows = []
    selected = records[: args.max_val_sequences] if args.max_val_sequences else records
    with torch.no_grad():
        for record in selected:
            _, row = run_records(model, teacher, [record], body_model, weights, args, train_mode=False)
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
            "model_type": "imu_neighbor_vel_ctrl_v1",
            "contract": model_contract(),
        },
        path,
    )


def load_model_from_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    config = checkpoint.get("config", {})
    model = IMUNeighborVelocityControlModule(
        hidden_size=int(config.get("hidden_size", 512)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
        dt=float(config.get("dt", DT)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train imu_neighbor_vel_ctrl_v1 module-level velocity controls.")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("amass", "totalcapture", "dip"), required=True)
    parser.add_argument("--imu-input-mode", choices=("official", "processed", "auto"), default="official")
    parser.add_argument("--world-gt-mode", choices=("auto", "available", "unavailable"), default="auto")
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
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--no-precompute", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    args = parser.parse_args()

    if args.dataset == "dip" and args.world_gt_mode == "available":
        raise RuntimeError("DIP world velocity/acceleration GT is forbidden; use --world-gt-mode auto or unavailable.")
    if args.batch_size <= 1:
        raise RuntimeError("Long module training must be batched; choose --batch-size > 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    train_records, train_manifest = load_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, max_sequences=args.max_val_sequences)
    train_records = [record for record in train_records if record_length(record) >= args.window]
    if not train_records:
        raise RuntimeError(f"No train sequence has at least window={args.window} frames.")

    gt_enabled = world_gt_available(args.dataset, args.world_gt_mode)
    weights = default_loss_weights(gt_enabled)
    body_model = art.ParametricModel("models/SMPL_male.pkl", device=DEVICE)
    precompute_summary = {"enabled": not args.no_precompute}
    if not args.no_precompute:
        train_records, train_precompute = precompute_records(train_records, body_model, args, "train")
        val_records, val_precompute = precompute_records(val_records, body_model, args, "val")
        precompute_summary.update({"train": train_precompute, "val": val_precompute})
        (output_dir / "precompute_summary.json").write_text(json.dumps(precompute_summary, indent=2) + "\n")
    model = IMUNeighborVelocityControlModule(
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
    teacher = None
    if args.teacher_checkpoint:
        teacher, _ = load_model_from_checkpoint(args.teacher_checkpoint)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
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
        for batch_start in range(0, len(train_records), args.batch_size):
            step += 1
            batch_records = []
            for offset, source in enumerate(train_records[batch_start:batch_start + args.batch_size]):
                max_start = max(0, record_length(source) - args.window)
                start = (step + offset) % (max_start + 1) if max_start > 0 else 0
                batch_records.append(slice_record(source, start, args.window))
            loss, row = run_records(model, teacher, batch_records, body_model, weights, args, train_mode=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {row['name']}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            row.update({"epoch": epoch, "step": step})
            train_rows.append(row)
        validation = eval_model(model, teacher, val_records, body_model, weights, args)
        selected = selection_value(validation["loss"], world_gt=gt_enabled)
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
        "world_gt_available": gt_enabled,
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
