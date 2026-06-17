import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import shlex
import sys
import time
from pathlib import Path

import torch

from acc_curve import PLStyleAccCurveModule


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRIC_SENSOR_NAMES = ("left_forearm", "right_forearm", "left_lower_leg", "right_lower_leg", "head", "pelvis")


def load_cache_files(cache_path):
    path = Path(cache_path)
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        return [Path(item["path"]) for item in manifest["cache_files"]], manifest
    return [path], None


def load_acc_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location="cpu")
        required = ("name", "feature", "aM_smooth", "aFK_smooth", "valid_mask")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{cache_file} missing required fields: {missing}")
        for idx, name in enumerate(data["name"]):
            records.append({
                "name": str(name),
                "feature": data["feature"][idx].float(),
                "base": data["aM_smooth"][idx].float(),
                "target": data["aFK_smooth"][idx].float(),
                "valid_mask": data["valid_mask"][idx].bool(),
                "num_frames": int(data["num_frames"][idx]),
            })
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def stable_val_key(name):
    return int(hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:8], 16) / float(16 ** 8)


def split_amass_hash(records, val_ratio):
    train, val = [], []
    for record in records:
        (val if stable_val_key(record["name"]) < val_ratio else train).append(record)
    if not val and len(records) > 1:
        val = [records[0]]
        train = records[1:]
    if not train:
        train = val
    return train, val


def make_windows(records, window, stride):
    windows = []
    for ridx, record in enumerate(records):
        n = int(record["feature"].shape[0])
        if n <= 0:
            continue
        if n <= window:
            starts = [0]
        else:
            starts = list(range(0, n - window + 1, stride))
            if starts[-1] != n - window:
                starts.append(n - window)
        for start in starts:
            end = min(n, start + window)
            if bool(record["valid_mask"][start:end].any()):
                windows.append((ridx, start, end))
    return windows


class AccCurveWindowDataset(torch.utils.data.Dataset):
    def __init__(self, records, window, stride, norm=None):
        self.records = records
        self.window = int(window)
        self.stride = int(stride)
        self.windows = make_windows(records, self.window, self.stride)
        self.norm = norm

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ridx, start, end = self.windows[idx]
        record = self.records[ridx]
        feature = record["feature"][start:end]
        if self.norm is not None:
            feature = (feature - self.norm["mean"]) / self.norm["std"]
        base = record["base"][start:end]
        target = record["target"][start:end]
        valid = record["valid_mask"][start:end]
        length = int(end - start)
        return {
            "feature": feature,
            "base": base,
            "target": target,
            "valid_mask": valid,
            "length": length,
            "name": record["name"],
            "start": int(start),
        }


def collate_windows(batch):
    max_len = max(item["length"] for item in batch)
    bsz = len(batch)
    feature = torch.zeros(max_len, bsz, 108)
    base = torch.zeros(max_len, bsz, 18)
    target = torch.zeros(max_len, bsz, 18)
    valid = torch.zeros(max_len, bsz, dtype=torch.bool)
    lengths = []
    names = []
    starts = []
    for i, item in enumerate(batch):
        n = item["length"]
        feature[:n, i] = item["feature"]
        base[:n, i] = item["base"]
        target[:n, i] = item["target"]
        valid[:n, i] = item["valid_mask"]
        lengths.append(n)
        names.append(item["name"])
        starts.append(item["start"])
    return {
        "feature": feature,
        "base": base,
        "target": target,
        "valid_mask": valid,
        "lengths": torch.tensor(lengths),
        "names": names,
        "starts": starts,
    }


def fit_feature_norm(records):
    total = 0
    sum_x = torch.zeros(108, dtype=torch.float64)
    sum_x2 = torch.zeros(108, dtype=torch.float64)
    for record in records:
        x = record["feature"].reshape(-1, 108).double()
        total += x.shape[0]
        sum_x += x.sum(dim=0)
        sum_x2 += x.square().sum(dim=0)
    mean = (sum_x / max(1, total)).float()
    var = (sum_x2 / max(1, total) - mean.double().square()).clamp_min(1e-12).float()
    std = var.sqrt().clamp_min(1e-6)
    return {"mean": mean, "std": std, "count": int(total)}


def masked_mse(pred, target, mask):
    mask = mask.to(pred.device)
    if not bool(mask.any()):
        return pred.new_zeros(())
    diff = pred[mask] - target[mask]
    return diff.square().mean()


def masked_l2_mean(pred, target, mask):
    mask = mask.to(pred.device)
    if not bool(mask.any()):
        return float("nan")
    return float((pred[mask] - target[mask]).reshape(-1, 6, 3).norm(dim=-1).mean().detach().cpu())


def run_batch(model, batch):
    feature = batch["feature"].to(DEVICE)
    base_input = batch["base"].to(DEVICE)
    target = batch["target"].to(DEVICE)
    valid = batch["valid_mask"].to(DEVICE)
    out = model.forward_sequence(feature, base_input)
    pred = out["pred_aM"]
    base = out["base"]
    loss = masked_mse(pred, target, valid)
    base_loss = masked_mse(base, target, valid)
    ratio = (masked_l2_mean(pred, target, valid) / max(masked_l2_mean(base, target, valid), 1e-12))
    components = {
        "loss": loss,
        "base_mse": base_loss.detach(),
        "pred_base_ratio": pred.new_tensor(ratio),
        "control_point_prior_t": out["control_point_prior_t"],
        "new_delta_norm": out["new_delta_norm"],
        "tail_delta_norm": out["tail_delta_norm"],
        "residual_std": out["residual"][valid].std() if bool(valid.any()) else pred.new_zeros(()),
    }
    return loss, components


def average_rows(rows):
    if not rows:
        return {}
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and math.isfinite(float(v))})
    return {key: sum(float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))) / max(1, sum(1 for row in rows if key in row and math.isfinite(float(row[key])))) for key in keys}


@torch.no_grad()
def eval_windows(model, loader):
    model.eval()
    rows = []
    for batch in loader:
        _, comp = run_batch(model, batch)
        rows.append({k: float(v.detach().cpu()) for k, v in comp.items()})
    avg = average_rows(rows)
    if "pred_base_ratio" in avg:
        avg["val_pred_base_ratio"] = avg["pred_base_ratio"]
    return avg


def train_stage(model, train_records, val_records, norm, args, output_dir, stage_name, init_optimizer=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds = AccCurveWindowDataset(train_records, args.window, args.stride, norm=norm)
    val_ds = AccCurveWindowDataset(val_records, args.window, args.stride, norm=norm)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"{stage_name} has train_windows={len(train_ds)} val_windows={len(val_ds)}")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if init_optimizer else None
    best_value = float("inf")
    best_epoch = 0
    start_epoch = 1
    log_path = output_dir / "train_log.jsonl"
    if getattr(args, "resume", False):
        best_path = output_dir / "best_loss.pt"
        if not best_path.exists():
            best_path = output_dir / "best.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=DEVICE)
            best_value = float(best_ckpt.get("selection_value", float("inf")))
            best_epoch = int(best_ckpt.get("epoch", 0))
        last_path = output_dir / "last_loss.pt"
        if not last_path.exists():
            last_path = output_dir / "last.pt"
        if last_path.exists():
            last_ckpt = torch.load(last_path, map_location=DEVICE)
            model.load_state_dict(last_ckpt["model_state_dict"])
            if optimizer is not None and last_ckpt.get("optimizer_state_dict") is not None:
                try:
                    optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
                except ValueError:
                    pass
            start_epoch = int(last_ckpt.get("epoch", 0)) + 1
    dataset_summary = {
        "stage": stage_name,
        "train_sequences": len(train_records),
        "val_sequences": len(val_records),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "window": int(args.window),
        "stride": int(args.stride),
        "batch_size": int(args.batch_size),
        "steps_per_epoch": len(train_loader),
        "norm_count": int(norm["count"]),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2) + "\n")
    print(json.dumps(dataset_summary, indent=2))
    if start_epoch > args.epochs:
        return {"best_epoch": best_epoch, "best_selection": best_value, "dataset_summary": dataset_summary}
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        model.train()
        train_rows = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, comp = run_batch(model, batch)
            reg = args.control_prior_weight * comp["control_point_prior_t"]
            total = loss + reg
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            row = {k: float(v.detach().cpu()) for k, v in comp.items()}
            row["total_loss"] = float(total.detach().cpu())
            train_rows.append(row)
        val_avg = eval_windows(model, val_loader)
        train_avg = average_rows(train_rows)
        selection = float(val_avg.get("val_pred_base_ratio", val_avg.get("loss", float("inf"))))
        elapsed = time.time() - started
        row = {
            "stage": stage_name,
            "epoch": epoch,
            "seconds": elapsed,
            "train": train_avg,
            "val": val_avg,
            "selection": selection,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row))
        save_checkpoint(output_dir / "last.pt", model, optimizer, args, epoch, selection, norm, stage_name)
        save_checkpoint(output_dir / "last_loss.pt", model, optimizer, args, epoch, selection, norm, stage_name)
        if selection < best_value:
            best_value = selection
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, args, epoch, selection, norm, stage_name)
            save_checkpoint(output_dir / "best_loss.pt", model, optimizer, args, epoch, selection, norm, stage_name)
    return {"best_epoch": best_epoch, "best_selection": best_value, "dataset_summary": dataset_summary}


def save_checkpoint(path, model, optimizer, args, epoch, selection, norm, stage_name):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": vars(args),
        "epoch": int(epoch),
        "selection_value": float(selection),
        "feature_norm": {"mean": norm["mean"], "std": norm["std"], "count": norm["count"]},
        "stage": stage_name,
        "model_type": "acc_curve_v1",
        "output_keys": {
            "pred_aM_curve": "[T,B,18] absolute sensor-site acceleration in m/s^2",
            "base": "[T,B,18] decoded aM_smooth baseline",
            "residual": "pred_aM_curve - base",
        },
        "contract": "108D model/world-frame IMU feature -> absolute 18D sensor-site acceleration in m/s^2",
    }, path)


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    model = PLStyleAccCurveModule(
        hidden_size=int(cfg.get("hidden_size", 512)),
        residual_scale=float(cfg.get("residual_scale", 1.0)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    norm = ckpt["feature_norm"]
    norm = {"mean": norm["mean"].float(), "std": norm["std"].float(), "count": int(norm["count"])}
    return model, norm, ckpt


@torch.no_grad()
def predict_record(model, record, norm):
    mean = norm["mean"].to(record["feature"].device)
    std = norm["std"].to(record["feature"].device)
    feature = ((record["feature"] - mean) / std).to(DEVICE)
    base = record["base"].to(DEVICE)
    out = model.forward_sequence(feature, base)
    return out["pred_aM"].detach().cpu(), out["base"].detach().cpu()


def corrcoef(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    if a.numel() < 2:
        return float("nan")
    av = a - a.mean()
    bv = b - b.mean()
    den = av.norm() * bv.norm()
    if float(den) <= 1e-12:
        return float("nan")
    return float((av * bv).sum() / den)


def eval_record_metrics(pred, base, target, valid, name):
    valid = valid.bool()
    if not bool(valid.any()):
        return {"name": name, "valid_frames": 0}
    pred_v = pred[valid].reshape(-1, 6, 3)
    base_v = base[valid].reshape(-1, 6, 3)
    target_v = target[valid].reshape(-1, 6, 3)
    pred_err = pred_v - target_v
    base_err = base_v - target_v
    residual = pred_v - base_v
    row = {
        "name": name,
        "valid_frames": int(valid.sum()),
        "pred_l2": float(pred_err.norm(dim=-1).mean()),
        "base_l2": float(base_err.norm(dim=-1).mean()),
        "pred_rmse": float(pred_err.square().mean().sqrt()),
        "base_rmse": float(base_err.square().mean().sqrt()),
        "pred_base_ratio": float(pred_err.norm(dim=-1).mean() / base_err.norm(dim=-1).mean().clamp_min(1e-12)),
        "corr": corrcoef(pred_v, target_v),
        "residual_std": float(residual.std()),
        "residual_p95": float(torch.quantile(residual.norm(dim=-1).reshape(-1), 0.95)),
    }
    per_sensor = pred_err.norm(dim=-1).mean(dim=0)
    base_per_sensor = base_err.norm(dim=-1).mean(dim=0)
    for idx, sensor in enumerate(METRIC_SENSOR_NAMES):
        row[f"pred_l2_{sensor}"] = float(per_sensor[idx])
        row[f"base_l2_{sensor}"] = float(base_per_sensor[idx])
    axis_pred = pred_err.abs().mean(dim=(0, 1))
    axis_base = base_err.abs().mean(dim=(0, 1))
    for idx, axis in enumerate(("x", "y", "z")):
        row[f"pred_axis_mae_{axis}"] = float(axis_pred[idx])
        row[f"base_axis_mae_{axis}"] = float(axis_base[idx])
    return row


def evaluate_checkpoint(checkpoint, cache, output_dir, split_name, max_sequences=0):
    model, norm, ckpt = load_model(checkpoint)
    records, manifest = load_acc_records(cache, max_sequences=max_sequences)
    rows = []
    model.eval()
    for record in records:
        pred, base = predict_record(model, record, norm)
        rows.append(eval_record_metrics(pred, base, record["target"], record["valid_mask"], record["name"]))
    aggregate = average_rows(rows)
    result = {
        "split": split_name,
        "checkpoint": str(checkpoint),
        "cache": str(cache),
        "num_sequences": len(records),
        "aggregate": aggregate,
        "rows": rows,
        "manifest": manifest,
        "checkpoint_selection": ckpt.get("selection_value"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split_name}_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    csv_path = output_dir / f"{split_name}_per_sequence.csv"
    if rows:
        keys = sorted({k for row in rows for k in row})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    md = [
        f"# AccCurve {split_name} Eval",
        "",
        f"checkpoint: `{checkpoint}`",
        f"cache: `{cache}`",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in ("pred_l2", "base_l2", "pred_rmse", "base_rmse", "pred_base_ratio", "corr", "residual_std", "residual_p95"):
        if key in aggregate:
            md.append(f"| {key} | {aggregate[key]:.6f} |")
    (output_dir / f"{split_name}_eval.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"split": split_name, "aggregate": aggregate}, indent=2))
    return result


def smoke_zero_init(records, norm, args):
    model = PLStyleAccCurveModule(hidden_size=args.hidden_size, residual_scale=args.residual_scale, dropout=args.dropout).to(DEVICE)
    ds = AccCurveWindowDataset(records, args.window, args.stride, norm=norm)
    batch = collate_windows([ds[0]])
    with torch.no_grad():
        out = model.forward_sequence(batch["feature"].to(DEVICE), batch["base"].to(DEVICE))
    diff = (out["pred_aM"].cpu() - out["base"].cpu()).abs().max()
    return float(diff)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate PL-style AccCurve residual module.")
    parser.add_argument("--mode", choices=("train_full", "eval"), default="train_full")
    parser.add_argument("--amass-cache", default="code/outputs/smooth_acc_cache_amass_dip_20260617/amass_train/acc_curve_cache_manifest.json")
    parser.add_argument("--dip-train-cache", default="code/outputs/smooth_acc_cache_amass_dip_20260617/dip_train/acc_curve_cache_manifest.json")
    parser.add_argument("--dip-val-cache", default="code/outputs/smooth_acc_cache_amass_dip_20260617/dip_val/acc_curve_cache_manifest.json")
    parser.add_argument("--dip-test-cache", default="code/outputs/smooth_acc_cache_amass_dip_20260617/dip_test/acc_curve_cache_manifest.json")
    parser.add_argument("--output-dir", default="data/experiments/acc_curve_v1_20260617")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dip-epochs", type=int, default=20)
    parser.add_argument("--window", type=int, default=240)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--control-prior-weight", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amass-val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-amass-sequences", type=int, default=0)
    parser.add_argument("--max-dip-train-sequences", type=int, default=0)
    parser.add_argument("--max-eval-sequences", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif not args.resume:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to rebuild intentionally.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    if args.mode == "eval":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for eval")
        eval_dir = output_dir / "eval"
        evaluate_checkpoint(args.checkpoint, args.dip_val_cache, eval_dir, "dip_val", args.max_eval_sequences)
        evaluate_checkpoint(args.checkpoint, args.dip_test_cache, eval_dir, "dip_test", args.max_eval_sequences)
        return

    amass_records, _ = load_acc_records(args.amass_cache, args.max_amass_sequences)
    dip_train_records, _ = load_acc_records(args.dip_train_cache, args.max_dip_train_sequences)
    dip_val_records, _ = load_acc_records(args.dip_val_cache)
    dip_test_records, _ = load_acc_records(args.dip_test_cache)
    amass_train, amass_val = split_amass_hash(amass_records, args.amass_val_ratio)
    norm = fit_feature_norm(amass_train)
    zero_diff = smoke_zero_init(amass_train, norm, args)
    if zero_diff > 1e-5:
        raise RuntimeError(f"zero-init AccCurve output is not close to base: max_abs={zero_diff}")
    model = PLStyleAccCurveModule(hidden_size=args.hidden_size, residual_scale=args.residual_scale, dropout=args.dropout).to(DEVICE)
    amass_args = argparse.Namespace(**vars(args))
    amass_args.epochs = args.epochs
    amass_result = train_stage(model, amass_train, amass_val, norm, amass_args, output_dir / "amass_pretrain", "amass_pretrain")
    amass_best = output_dir / "amass_pretrain" / "best_loss.pt"
    if amass_best.exists():
        model.load_state_dict(torch.load(amass_best, map_location=DEVICE)["model_state_dict"])
    dip_args = argparse.Namespace(**vars(args))
    dip_args.epochs = args.dip_epochs
    dip_result = train_stage(model, dip_train_records, dip_val_records, norm, dip_args, output_dir / "dip_finetune", "dip_finetune")
    final_ckpt = output_dir / "dip_finetune" / "best_loss.pt"
    eval_dir = output_dir / "eval"
    val_result = evaluate_checkpoint(final_ckpt, args.dip_val_cache, eval_dir, "dip_val", args.max_eval_sequences)
    test_result = evaluate_checkpoint(final_ckpt, args.dip_test_cache, eval_dir, "dip_test", args.max_eval_sequences)
    summary = {
        "zero_init_max_abs_pred_minus_base": zero_diff,
        "amass_pretrain": amass_result,
        "dip_finetune": dip_result,
        "dip_val": val_result["aggregate"],
        "dip_test": test_result["aggregate"],
        "final_checkpoint": str(final_ckpt),
    }
    (output_dir / "train_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
