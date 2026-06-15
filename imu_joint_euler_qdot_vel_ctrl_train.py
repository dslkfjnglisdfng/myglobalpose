import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from imu_joint_euler_qdot_vel_ctrl import (
    DT,
    IMUJointEulerQdotVelControlModule,
    INPUT_ROTATION_FRAME,
    VARIANT_WEIGHTS,
    imu_joint_loss,
    imu_rootframe_features,
    model_contract,
    root_relative_targets_from_pose,
    selection_value,
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


def build_features(record, mode):
    if "imu_joint_features" in record:
        return record["imu_joint_features"].float()
    aM, wM, RMB = selected_imu_fields(record, mode)
    return imu_rootframe_features(aM.float(), wM.float(), RMB.float())


def build_target(record, body_model, args):
    if "imu_joint_target" in record:
        return record["imu_joint_target"]
    return root_relative_targets_from_pose(
        record["pose_gt"],
        body_model,
        DEVICE,
        dt=args.dt,
        euler_seq=args.euler_seq,
    )


def precompute_records(records, body_model, args, split_name):
    compact_records = []
    for record in records:
        feature = build_features(record, args.imu_input_mode).detach().cpu()
        target = build_target(record, body_model, args)
        compact_records.append({
            "name": record["name"],
            "seq_len": int(feature.shape[0]),
            "imu_joint_features": feature,
            "imu_joint_target": {key: value.detach().cpu() for key, value in target.items()},
        })
    return compact_records, {
        "split": split_name,
        "num_sequences": len(compact_records),
        "feature_dim": 90,
        "target_dim": 54,
        "mode": "in_memory_compact_precompute",
        "input_rotation_frame_id": INPUT_ROTATION_FRAME,
        "input_rotation_frame": "root IMU frame, R_rootIMU_sensorIMU = RMB[root_imu=5]^T @ RMB[sensor]",
        "dip_trans_policy": "DIP trans is not used; targets are pose-derived root-relative q/vel/acc.",
    }


def save_precomputed_records(path, records, summary, source_cache, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "records": records,
            "summary": summary,
            "source_cache": source_cache,
            "imu_input_mode": args.imu_input_mode,
            "dt": args.dt,
            "euler_seq": args.euler_seq,
            "contract": model_contract(),
        },
        tmp_path,
    )
    tmp_path.replace(path)


def load_precomputed_records(path, split_name, max_sequences=0):
    data = torch.load(path, map_location="cpu")
    summary_raw = data.get("summary", {})
    if summary_raw.get("input_rotation_frame_id") != INPUT_ROTATION_FRAME:
        raise RuntimeError(
            f"Precomputed records at {path} use input_rotation_frame_id="
            f"{summary_raw.get('input_rotation_frame_id')!r}; expected {INPUT_ROTATION_FRAME!r}. "
            "Regenerate precompute records for the root-frame RMB input."
        )
    records = data["records"]
    if max_sequences:
        records = records[:max_sequences]
    summary = dict(summary_raw)
    summary.update({
        "split": split_name,
        "num_sequences": len(records),
        "mode": "disk_compact_precompute",
        "path": str(path),
        "source_cache": data.get("source_cache"),
    })
    return records, summary


def summarize_manifest(manifest):
    if not manifest:
        return {}
    summary = {}
    for key in (
        "cache_type",
        "source_input",
        "source_original_cache",
        "source_augmented_input",
        "num_records",
        "num_sequences",
        "num_pairs",
        "num_frames",
        "skipped_count",
    ):
        if key in manifest:
            summary[key] = manifest[key]
    if "cache_files" in manifest:
        summary["cache_files_count"] = len(manifest.get("cache_files") or [])
    return summary


def stack_target(targets):
    return {key: torch.stack([target[key] for target in targets], dim=1).to(DEVICE) for key in targets[0]}


def init_from_stacked_target(target):
    return torch.cat((target["q_euler"][0], target["qdot_euler"][0], target["vel_RJ"][0]), dim=-1)


def run_records(model, records, body_model, weights, args, train_mode):
    features, targets, names = [], [], []
    for record in records:
        features.append(build_features(record, args.imu_input_mode).to(DEVICE))
        targets.append(build_target(record, body_model, args))
        names.append(record["name"])
    features_tbd = torch.stack(features, dim=1).to(DEVICE)
    target = stack_target(targets)
    init_state = init_from_stacked_target(target)
    output = model.forward_sequence(features_tbd, init_state=init_state)
    loss, components = imu_joint_loss(output, target, weights)
    row = {
        "name": "|".join(str(name) for name in names),
        "loss": float(loss.detach()),
    }
    row.update({key: float(value.detach()) for key, value in components.items()})
    return loss, row


def average(rows):
    numeric = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, bool)):
                numeric.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in numeric.items()}


@torch.no_grad()
def eval_model(model, records, body_model, weights, args):
    model.eval()
    selected = records[: args.max_val_sequences] if args.max_val_sequences else records
    rows = []
    for record in selected:
        _, row = run_records(model, [record], body_model, weights, args, train_mode=False)
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
            "model_type": "imu_joint_euler_qdot_vel_ctrl_v1",
            "contract": model_contract(),
        },
        path,
    )


def load_model_from_checkpoint(path, args):
    checkpoint = torch.load(path, map_location=DEVICE)
    config = checkpoint.get("config", {})
    model = IMUJointEulerQdotVelControlModule(
        hidden_size=int(config.get("hidden_size", args.hidden_size)),
        num_layers=int(config.get("num_layers", args.num_layers)),
        dropout=float(config.get("dropout", args.dropout)),
        dt=float(config.get("dt", args.dt)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train imu_joint_euler_qdot_vel_ctrl_v1.")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("amass", "totalcapture", "dip"), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_WEIGHTS.keys()), required=True)
    parser.add_argument("--imu-input-mode", choices=("official", "processed", "auto"), default="official")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--window", type=int, default=61)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--euler-seq", default="XYZ")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-sequences", type=int, default=0)
    parser.add_argument("--max-val-sequences", type=int, default=0)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--no-precompute", action="store_true")
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument("--precomputed-train-records", default="")
    parser.add_argument("--precomputed-val-records", default="")
    parser.add_argument("--write-precomputed-train-records", default="")
    parser.add_argument("--write-precomputed-val-records", default="")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--no-save-last", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 1:
        raise RuntimeError("Long module training must be batched; choose --batch-size > 1.")
    if args.dataset == "dip":
        print("DIP policy: no trans/root/world velocity GT is used; targets are pose-derived root-relative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    body_model = art.ParametricModel("models/SMPL_male.pkl", device=DEVICE)
    precompute_summary = {"enabled": not args.no_precompute}

    if args.precompute_only:
        raw_train_records, train_manifest = load_records(args.train_cache, max_sequences=args.max_train_sequences)
        raw_train_records = [record for record in raw_train_records if record_length(record) >= args.window]
        if not raw_train_records:
            raise RuntimeError(f"No train sequence has at least window={args.window} frames.")
        train_records, train_precompute = precompute_records(raw_train_records, body_model, args, "train")
        if not args.write_precomputed_train_records:
            raise RuntimeError("--precompute-only requires --write-precomputed-train-records.")
        save_precomputed_records(args.write_precomputed_train_records, train_records, train_precompute, args.train_cache, args)
        result = {
            "status": "ok",
            "mode": "precompute_only",
            "train_manifest": summarize_manifest(train_manifest),
            "train_precompute": train_precompute,
            "train_precomputed_records": args.write_precomputed_train_records,
        }
        if args.write_precomputed_val_records:
            if args.val_cache == args.train_cache and args.write_precomputed_val_records == args.write_precomputed_train_records:
                val_records, val_precompute, val_manifest = train_records, train_precompute, train_manifest
            else:
                raw_val_records, val_manifest = load_records(args.val_cache, max_sequences=args.max_val_sequences)
                val_records, val_precompute = precompute_records(raw_val_records, body_model, args, "val")
                save_precomputed_records(args.write_precomputed_val_records, val_records, val_precompute, args.val_cache, args)
            result.update({
                "val_manifest": summarize_manifest(val_manifest),
                "val_precompute": val_precompute,
                "val_precomputed_records": args.write_precomputed_val_records,
            })
        (output_dir / "precompute_only_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({
            "status": "ok",
            "mode": "precompute_only",
            "train_precomputed_records": args.write_precomputed_train_records,
            "num_train_sequences": train_precompute.get("num_sequences"),
            "val_precomputed_records": args.write_precomputed_val_records,
        }))
        return

    if args.precomputed_train_records:
        train_records, train_precompute = load_precomputed_records(
            args.precomputed_train_records,
            "train",
            max_sequences=args.max_train_sequences,
        )
        train_manifest = {"precomputed_records": args.precomputed_train_records}
    else:
        train_records, train_manifest = load_records(args.train_cache, max_sequences=args.max_train_sequences)
        train_records = [record for record in train_records if record_length(record) >= args.window]
        if not args.no_precompute:
            train_records, train_precompute = precompute_records(train_records, body_model, args, "train")
        else:
            train_precompute = {"split": "train", "mode": "disabled"}
    if args.precomputed_val_records:
        val_records, val_precompute = load_precomputed_records(
            args.precomputed_val_records,
            "val",
            max_sequences=args.max_val_sequences,
        )
        val_manifest = {"precomputed_records": args.precomputed_val_records}
    else:
        val_records, val_manifest = load_records(args.val_cache, max_sequences=args.max_val_sequences)
        if not args.no_precompute:
            val_records, val_precompute = precompute_records(val_records, body_model, args, "val")
        else:
            val_precompute = {"split": "val", "mode": "disabled"}
    train_records = [record for record in train_records if record_length(record) >= args.window]
    if not train_records:
        raise RuntimeError(f"No train sequence has at least window={args.window} frames.")
    if not args.no_precompute or args.precomputed_train_records or args.precomputed_val_records:
        precompute_summary.update({"train": train_precompute, "val": val_precompute})
        (output_dir / "precompute_summary.json").write_text(json.dumps(precompute_summary, indent=2) + "\n")

    weights = VARIANT_WEIGHTS[args.variant]
    model = IMUJointEulerQdotVelControlModule(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dt=args.dt,
    ).to(DEVICE)
    init_load = None
    if args.init_checkpoint:
        init_model, checkpoint = load_model_from_checkpoint(args.init_checkpoint, args)
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
        for batch_start in range(0, len(train_records), args.batch_size):
            step += 1
            batch_records = []
            for offset, source in enumerate(train_records[batch_start:batch_start + args.batch_size]):
                max_start = max(0, record_length(source) - args.window)
                start = (step + offset) % (max_start + 1) if max_start > 0 else 0
                batch_records.append(slice_record(source, start, args.window))
            loss, row = run_records(model, batch_records, body_model, weights, args, train_mode=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {row['name']}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            row.update({"epoch": epoch, "step": step})
            train_rows.append(row)
        validation = eval_model(model, val_records, body_model, weights, args)
        selected = selection_value(validation["loss"])
        improved = selected < best_value if best_value == float("inf") else (best_value - selected) > args.early_stop_min_delta
        if improved:
            best_value = selected
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(output_dir / "best_loss.pt", model, optimizer, args, epoch, step, selected, weights)
        else:
            stale_epochs += 1
        if not args.no_save_last:
            save_checkpoint(output_dir / "last.pt", model, optimizer, args, epoch, step, selected, weights)
        epoch_row = {
            "epoch": epoch,
            "step": step,
            "train_loss": average(train_rows),
            "validation": {
                "num_sequences": validation["num_sequences"],
                "loss": validation["loss"],
            },
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
        "variant": args.variant,
        "weights": weights,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "history": history,
        "checkpoints": {"best": str(output_dir / "best_loss.pt"), "last": str(output_dir / "last.pt")},
    }
    (output_dir / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "variant": args.variant, "best_epoch": best_epoch, "best_value": best_value, "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
