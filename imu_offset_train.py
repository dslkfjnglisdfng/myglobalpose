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
    acceleration_consistency_error,
    make_checkpoint,
    offset_input_feature,
    offset_regularizers,
    offset_supervised_loss,
)
from pl_curve import normalize_gravity, pl_target_from_pose


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stack_if_list(x):
    if torch.is_tensor(x):
        return x.float()
    if isinstance(x, list) and x and torch.is_tensor(x[0]):
        return torch.stack([item.float() for item in x])
    raise TypeError(type(x))


def load_amass_records(paths, max_sequences=0, max_frames=0):
    records = []
    for path in paths:
        data = torch.load(path, map_location="cpu")
        required = ("name", "pose", "tran", "aM", "wM", "RMB")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{path} missing required fields: {missing}")
        offset_key = "imu_offset_r" if "imu_offset_r" in data else "r_JS"
        if offset_key not in data:
            raise KeyError(f"{path} missing imu_offset_r/r_JS synthetic offset GT")
        for idx, name in enumerate(data["name"]):
            pose = data["pose"][idx].float()
            tran = data["tran"][idx].float()
            aM = data["aM"][idx].float()
            wM = data["wM"][idx].float()
            RMB = data["RMB"][idx].float()
            n = min(pose.shape[0], tran.shape[0], aM.shape[0], wM.shape[0], RMB.shape[0])
            if max_frames:
                n = min(n, max_frames)
            if n < 4:
                continue
            records.append(
                {
                    "name": str(name),
                    "source_path": str(path),
                    "pose": pose[:n],
                    "tran": tran[:n],
                    "aM": aM[:n],
                    "wM": wM[:n],
                    "RMB": RMB[:n],
                    "offset": data[offset_key][idx].float(),
                    "num_frames": int(n),
                }
            )
            if max_sequences and len(records) >= max_sequences:
                return records
    return records


def split_records(records, val_ratio, seed):
    idxs = list(range(len(records)))
    random.Random(seed).shuffle(idxs)
    n_val = max(1, int(round(len(records) * val_ratio))) if len(records) > 1 else 1
    val_ids = set(idxs[:n_val])
    train = [rec for i, rec in enumerate(records) if i not in val_ids]
    val = [rec for i, rec in enumerate(records) if i in val_ids]
    return train or val, val


def build_pl_targets(records):
    model = art.ParametricModel("models/SMPL_male.pkl", vert_mask=torch.tensor([1961, 5424, 1176, 4662, 411, 3021]), device=DEVICE)
    for rec in tqdm.tqdm(records, desc="pl targets"):
        pose = art.math.axis_angle_to_rotation_matrix(rec["pose"].to(DEVICE)).view(-1, 24, 3, 3)
        target = normalize_gravity(pl_target_from_pose(pose, model)).detach().cpu()
        rec["pl_output"] = target
    return records


def sample_windows(records, window, windows_per_sequence, seed):
    rng = random.Random(seed)
    out = []
    for rec in records:
        n = rec["num_frames"]
        if n <= window:
            starts = [0]
        else:
            starts = [rng.randint(0, n - window) for _ in range(max(1, windows_per_sequence))]
        for start in starts:
            end = min(n, start + window)
            out.append((rec, start, end))
    rng.shuffle(out)
    return out


def batch_feature(rec, start, end):
    return offset_input_feature(
        rec["pl_output"][start:end],
        rec["aM"][start:end],
        rec["wM"][start:end],
        rec["RMB"][start:end],
    )


def offset_errors_cm(pred_offset, target_offset):
    if pred_offset.dim() == 4:
        pred = pred_offset.mean(dim=1)
    elif pred_offset.dim() == 3:
        pred = pred_offset.mean(dim=0, keepdim=True)
    else:
        pred = pred_offset.view(1, *pred_offset.shape)
    target = target_offset.to(pred.device, pred.dtype)
    if target.dim() == 2:
        target = target.view(1, *target.shape).expand_as(pred)
    err = (pred - target).norm(dim=-1) * 100.0
    return {
        "offset_l1_cm": float((pred - target).abs().mean().mul(100.0).detach().cpu()),
        "offset_l2_cm": float(err.mean().detach().cpu()),
        "offset_l2_cm_per_sensor": [float(x) for x in err.mean(dim=0).detach().cpu()],
        "temporal_stability_cm": float(pred_offset.std(dim=1).norm(dim=-1).mean().mul(100.0).detach().cpu()) if pred_offset.dim() == 4 and pred_offset.shape[1] > 1 else 0.0,
    }


@torch.no_grad()
def evaluate(model, records, window, windows_per_sequence, seed, acc_device):
    model.eval()
    rows = []
    windows = sample_windows(records, window, windows_per_sequence, seed)
    for rec, start, end in windows:
        feature = batch_feature(rec, start, end).unsqueeze(0).to(DEVICE)
        target = rec["offset"].to(DEVICE)
        pred = model(feature)
        errs = offset_errors_cm(pred, target)
        pred_seq = pred[0].mean(dim=0).detach().cpu()
        target_acc = acceleration_consistency_error(
            rec["pose"][start:end],
            rec["tran"][start:end],
            target.detach().cpu(),
            rec["aM"][start:end],
            device=acc_device,
        )
        pred_acc = acceleration_consistency_error(
            rec["pose"][start:end],
            rec["tran"][start:end],
            pred_seq,
            rec["aM"][start:end],
            device=acc_device,
        )
        rows.append(
            {
                "name": rec["name"],
                "start": int(start),
                "end": int(end),
                "num_frames": int(end - start),
                **errs,
                "acc_consistency_gt_offset_mean": target_acc["acc_consistency_mean"],
                "acc_consistency_pred_offset_mean": pred_acc["acc_consistency_mean"],
                "finite": bool(torch.isfinite(pred).all()),
            }
        )
    total = max(1, len(rows))
    aggregate = {
        "num_windows": len(rows),
        "num_sequences": len({row["name"] for row in rows}),
        "offset_gt_available": True,
        "offset_l1_cm": sum(row["offset_l1_cm"] for row in rows) / total,
        "offset_l2_cm": sum(row["offset_l2_cm"] for row in rows) / total,
        "temporal_stability_cm": sum(row["temporal_stability_cm"] for row in rows) / total,
        "acc_consistency_gt_offset_mean": sum(row["acc_consistency_gt_offset_mean"] for row in rows) / total,
        "acc_consistency_pred_offset_mean": sum(row["acc_consistency_pred_offset_mean"] for row in rows) / total,
        "all_finite": all(row["finite"] for row in rows),
    }
    if rows:
        sensors = len(rows[0]["offset_l2_cm_per_sensor"])
        aggregate["offset_l2_cm_per_sensor"] = [
            sum(row["offset_l2_cm_per_sensor"][idx] for row in rows) / total for idx in range(sensors)
        ]
    return rows, aggregate


def train(args):
    paths = sorted(Path().glob(args.amass_glob))
    if args.max_shards:
        paths = paths[: args.max_shards]
    if not paths:
        raise RuntimeError(f"No AMASS shards matched {args.amass_glob}")
    records = load_amass_records(paths, max_sequences=args.max_sequences, max_frames=args.max_frames)
    if not records:
        raise RuntimeError("No AMASS records loaded")
    build_pl_targets(records)
    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    prior = torch.stack([rec["offset"] for rec in train_records]).median(dim=0).values
    feature_dim = offset_input_feature(train_records[0]["pl_output"][:1], train_records[0]["aM"][:1], train_records[0]["wM"][:1], train_records[0]["RMB"][:1]).shape[-1]
    model = IMUOffsetNet(
        version=args.version,
        input_size=feature_dim,
        hidden_size=args.hidden_size,
        prior_offset=prior,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "feature_dim": int(feature_dim),
            "coordinate_contract": OFFSET_COORDINATE_CONTRACT,
            "device": str(DEVICE),
            "train_sequences": len(train_records),
            "val_sequences": len(val_records),
            "prior_offset_median": prior.tolist(),
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best_loss = math.inf
    best_epoch = 0
    step = 0
    train_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        windows = sample_windows(train_records, args.window, args.windows_per_sequence, args.seed + epoch)
        total = 0.0
        random.Random(args.seed + epoch).shuffle(windows)
        for rec, start, end in tqdm.tqdm(windows, desc=f"train epoch {epoch}"):
            feature = batch_feature(rec, start, end).unsqueeze(0).to(DEVICE)
            target = rec["offset"].to(DEVICE)
            pred = model(feature)
            loss_main = offset_supervised_loss(pred, target)
            regs = offset_regularizers(pred, prior if args.version == "offset_v3_residual_prior" else None)
            loss = loss_main + args.magnitude_weight * regs["magnitude"] + args.smooth_weight * regs["temporal_smooth"] + args.prior_weight * regs["prior_l2"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch={epoch} sequence={rec['name']} start={start} end={end}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            total += float(loss.detach())
        rows, val = evaluate(model, val_records, args.window, args.val_windows_per_sequence, args.seed + 1000 + epoch, args.acc_device)
        if not math.isfinite(val["offset_l2_cm"]):
            raise RuntimeError(f"Non-finite validation offset_l2_cm at epoch={epoch}")
        train_loss = total / max(1, len(windows))
        item = {"epoch": epoch, "train_loss": train_loss, "val": val}
        train_log.append(item)
        (output_dir / "train_log.jsonl").open("a").write(json.dumps(item) + "\n")
        if val["offset_l2_cm"] < best_loss:
            best_loss = val["offset_l2_cm"]
            best_epoch = epoch
            torch.save(make_checkpoint(model, config, epoch, step, best_loss, optimizer), output_dir / "best_loss.pt")
        torch.save(make_checkpoint(model, config, epoch, step, val["offset_l2_cm"], optimizer), output_dir / "last.pt")
        print(json.dumps({"epoch": epoch, "train_loss": train_loss, "val_offset_l2_cm": val["offset_l2_cm"], "best_offset_l2_cm": best_loss}, indent=2), flush=True)
    val_rows, val_agg = evaluate(model, val_records, args.window, args.val_windows_per_sequence, args.seed + 2000, args.acc_device)
    result = {
        "status": "ok",
        "version": args.version,
        "coordinate_contract": OFFSET_COORDINATE_CONTRACT,
        "stage": "A_synthetic_supervised",
        "offset_gt_available": True,
        "best_epoch": int(best_epoch),
        "best_offset_l2_cm": float(best_loss),
        "last_val": val_agg,
        "val_rows": val_rows,
        "checkpoints": {
            "best_loss": str(output_dir / "best_loss.pt"),
            "last": str(output_dir / "last.pt"),
        },
        "config": config,
    }
    (output_dir / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Stage A synthetic supervised training for IMUOffsetNet.")
    parser.add_argument("--version", choices=("offset_v1_mlp_frame", "offset_v2_temporal_rnn", "offset_v3_residual_prior"), required=True)
    parser.add_argument("--amass-glob", default="data/dataset_work/AMASS/globalpose_synth_shard*.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--windows-per-sequence", type=int, default=2)
    parser.add_argument("--val-windows-per-sequence", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--magnitude-weight", type=float, default=0.0)
    parser.add_argument("--smooth-weight", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--acc-device", default="cpu")
    return parser.parse_args()


def main():
    torch.manual_seed(20260607)
    result = train(parse_args())
    print(json.dumps({k: result.get(k) for k in ("status", "version", "best_offset_l2_cm", "best_epoch", "checkpoints")}, indent=2))


if __name__ == "__main__":
    main()
