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

from acc_curve_v3_leafrel import (
    ACC_CURVE_V3_INPUT_SIZE,
    ACC_CURVE_V3_LEAF_SENSOR_NAMES,
    ACC_CURVE_V3_STATE_DIM,
    PLStyleAccCurveV3LeafRelModule,
    acc_curve_v3_leafrel_features,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_NAME = "acc_curve_v3_leafrel_causal_butter_20260618"
FEATURE_CACHE_EXPERIMENT = "acc_curve_v3_leafrel_feature_cache_20260618"
SOURCE_V4_EXPERIMENT = "acc_leaf_relative_residual_v4_causal_butterworth_20260618"
ROOT_INDEX = 5
LEAF_INDICES = [0, 1, 2, 3, 4]
V4_REFERENCE = {
    "DIP": {"l2": 0.893481, "rmse": 0.914904, "corr": 0.943719},
    "TotalCapture": {"l2": 1.092481, "rmse": 1.050146, "corr": 0.941474},
}


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def load_manifest(cache_manifest):
    manifest = json.loads(Path(cache_manifest).read_text())
    experiment = manifest.get("experiment")
    if experiment == FEATURE_CACHE_EXPERIMENT:
        cache_files = manifest.get("cache_files", [])
        if not cache_files:
            raise ValueError(f"{cache_manifest} has no cache_files")
        return manifest, cache_files
    if experiment != SOURCE_V4_EXPERIMENT:
        raise ValueError(f"Expected v4 or v3 feature-cache manifest, got {experiment!r}")
    if int(manifest.get("root_index", -1)) != ROOT_INDEX:
        raise ValueError(f"Expected root_index=5, got {manifest.get('root_index')!r}")
    if list(manifest.get("leaf_indices", [])) != LEAF_INDICES:
        raise ValueError(f"Expected leaf_indices={LEAF_INDICES}, got {manifest.get('leaf_indices')!r}")
    cache_files = manifest.get("cache_files", [])
    if not cache_files:
        raise ValueError(f"{cache_manifest} has no cache_files")
    return manifest, cache_files


def record_path(item):
    if isinstance(item, str):
        return Path(item)
    if isinstance(item, dict) and "path" in item:
        return Path(item["path"])
    raise TypeError(f"Unsupported cache_files entry: {type(item).__name__}")


def fill_nonfinite_time(x):
    """Fill non-finite frames with nearest finite frame so masked RNN steps stay finite."""
    x = x.clone()
    flat = x.reshape(x.shape[0], -1)
    finite = torch.isfinite(flat).all(dim=-1)
    if bool(finite.all()):
        return x
    if not bool(finite.any()):
        return torch.zeros_like(x)
    finite_idx = torch.nonzero(finite, as_tuple=False).flatten()
    first = int(finite_idx[0])
    last = int(finite_idx[-1])
    if first > 0:
        x[:first] = x[first]
    if last + 1 < x.shape[0]:
        x[last + 1 :] = x[last]
    last_good = first
    for i in range(first, last + 1):
        if bool(torch.isfinite(x[i].reshape(-1)).all()):
            last_good = i
        else:
            x[i] = x[last_good]
    return x


def build_record(cache_file, manifest_item=None):
    data = torch.load(cache_file, map_location="cpu")
    required = (
        "aIMU_leaf_rel_raw",
        "aIMU_leaf_rel_butter2_4hz",
        "aGT_leaf_rel_butter2_4hz",
        "wM",
        "RMB",
        "valid_mask",
        "meta",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{cache_file} missing required fields: {missing}")
    meta = data["meta"]
    if int(meta.get("root_index", -1)) != ROOT_INDEX:
        raise ValueError(f"{cache_file} has root_index={meta.get('root_index')!r}")
    if list(meta.get("leaf_indices", [])) != LEAF_INDICES:
        raise ValueError(f"{cache_file} has leaf_indices={meta.get('leaf_indices')!r}")
    if not bool(meta.get("root_excluded_from_metrics", False)):
        raise ValueError(f"{cache_file} does not mark root_excluded_from_metrics=true")
    a_imu_raw = fill_nonfinite_time(data["aIMU_leaf_rel_raw"].float())
    a_imu_butter = fill_nonfinite_time(data["aIMU_leaf_rel_butter2_4hz"].float())
    a_gt_butter = fill_nonfinite_time(data["aGT_leaf_rel_butter2_4hz"].float())
    wM = fill_nonfinite_time(data["wM"].float())
    RMB = fill_nonfinite_time(data["RMB"].float())
    feature = acc_curve_v3_leafrel_features(a_imu_raw, a_imu_butter, wM, RMB)
    base = a_imu_butter.reshape(-1, ACC_CURVE_V3_STATE_DIM).float()
    target = a_gt_butter.reshape(-1, ACC_CURVE_V3_STATE_DIM).float()
    valid = data["valid_mask"].bool()
    finite = torch.isfinite(feature).all(dim=-1) & torch.isfinite(base).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    valid = valid & finite
    dataset = str(meta.get("dataset", manifest_item.get("dataset") if isinstance(manifest_item, dict) else ""))
    split = str(meta.get("split", manifest_item.get("split") if isinstance(manifest_item, dict) else ""))
    name = str(meta.get("sequence_name", manifest_item.get("sequence_name") if isinstance(manifest_item, dict) else cache_file.stem))
    return {
        "name": name,
        "dataset": dataset,
        "split": split,
        "path": str(cache_file),
        "feature": feature,
        "base": base,
        "target": target,
        "valid_mask": valid,
        "num_frames": int(feature.shape[0]),
    }


def load_records(cache_manifest, max_sequences=0):
    manifest, cache_files = load_manifest(cache_manifest)
    if manifest.get("experiment") == FEATURE_CACHE_EXPERIMENT:
        records = []
        for item in cache_files:
            path = record_path(item)
            shard = torch.load(path, map_location="cpu")
            for record in shard["records"]:
                records.append({
                    "name": str(record["name"]),
                    "dataset": str(record["dataset"]),
                    "split": str(record["split"]),
                    "path": str(record.get("source_path", record.get("path", path))),
                    "feature": record["feature"].float(),
                    "base": record["base"].float(),
                    "target": record["target"].float(),
                    "valid_mask": record["valid_mask"].bool(),
                    "num_frames": int(record["num_frames"]),
                })
                if max_sequences and len(records) >= max_sequences:
                    return records, manifest
        return records, manifest
    records = []
    for item in cache_files:
        path = record_path(item)
        records.append(build_record(path, item))
        if max_sequences and len(records) >= max_sequences:
            break
    return records, manifest


def group_records(records):
    groups = {}
    for record in records:
        groups.setdefault((record["dataset"], record["split"]), []).append(record)
    return groups


def stable_val_key(name):
    return int(hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:8], 16) / float(16**8)


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


class AccCurveV3WindowDataset(torch.utils.data.Dataset):
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
        return {
            "feature": feature,
            "base": record["base"][start:end],
            "target": record["target"][start:end],
            "valid_mask": record["valid_mask"][start:end],
            "length": int(end - start),
            "name": record["name"],
            "start": int(start),
        }


def collate_windows(batch):
    max_len = max(item["length"] for item in batch)
    bsz = len(batch)
    feature = torch.zeros(max_len, bsz, ACC_CURVE_V3_INPUT_SIZE)
    base = torch.zeros(max_len, bsz, ACC_CURVE_V3_STATE_DIM)
    target = torch.zeros(max_len, bsz, ACC_CURVE_V3_STATE_DIM)
    valid = torch.zeros(max_len, bsz, dtype=torch.bool)
    lengths, names, starts = [], [], []
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
    sum_x = torch.zeros(ACC_CURVE_V3_INPUT_SIZE, dtype=torch.float64)
    sum_x2 = torch.zeros(ACC_CURVE_V3_INPUT_SIZE, dtype=torch.float64)
    for record in records:
        mask = record["valid_mask"].bool()
        x = record["feature"][mask].reshape(-1, ACC_CURVE_V3_INPUT_SIZE).double()
        if x.numel() == 0:
            continue
        total += x.shape[0]
        sum_x += x.sum(dim=0)
        sum_x2 += x.square().sum(dim=0)
    if total == 0:
        raise RuntimeError("Cannot fit feature norm: no valid training frames.")
    mean = (sum_x / total).float()
    var = (sum_x2 / total - mean.double().square()).clamp_min(1e-12).float()
    std = var.sqrt().clamp_min(1e-6)
    return {"mean": mean, "std": std, "count": int(total)}


def masked_mse(pred, target, mask):
    mask = mask.to(pred.device)
    if not bool(mask.any()):
        return pred.new_zeros(())
    return (pred[mask] - target[mask]).square().mean()


def masked_l2_mean(pred, target, mask):
    mask = mask.to(pred.device)
    if not bool(mask.any()):
        return float("nan")
    return float((pred[mask] - target[mask]).reshape(-1, 5, 3).norm(dim=-1).mean().detach().cpu())


def run_batch(model, batch, residual_l2_weight=0.0):
    feature = batch["feature"].to(DEVICE)
    base_input = batch["base"].to(DEVICE)
    target = batch["target"].to(DEVICE)
    valid = batch["valid_mask"].to(DEVICE)
    out = model.forward_sequence(feature, base_input)
    pred = out["pred_leaf_rel_acc"]
    base = out["base"]
    loss = masked_mse(pred, target, valid)
    base_loss = masked_mse(base, target, valid)
    pred_l2 = masked_l2_mean(pred, target, valid)
    base_l2 = masked_l2_mean(base, target, valid)
    ratio = pred_l2 / max(base_l2, 1e-12)
    residual_l2 = out["residual"][valid].square().mean() if bool(valid.any()) else pred.new_zeros(())
    components = {
        "loss": loss,
        "base_mse": base_loss.detach(),
        "pred_base_ratio": pred.new_tensor(ratio),
        "pred_l2": pred.new_tensor(pred_l2),
        "base_l2": pred.new_tensor(base_l2),
        "control_point_prior_t": out["control_point_prior_t"],
        "new_delta_norm": out["new_delta_norm"],
        "tail_delta_norm": out["tail_delta_norm"],
        "residual_l2": residual_l2,
        "residual_std": out["residual"][valid].std() if bool(valid.any()) else pred.new_zeros(()),
    }
    total = loss + float(residual_l2_weight) * residual_l2
    return total, components


def average_rows(rows):
    if not rows:
        return {}
    keys = sorted(
        {
            k
            for row in rows
            for k, v in row.items()
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        }
    )
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


@torch.no_grad()
def eval_windows(model, loader, residual_l2_weight=0.0):
    model.eval()
    rows = []
    for batch in loader:
        _, comp = run_batch(model, batch, residual_l2_weight=residual_l2_weight)
        rows.append({k: float(v.detach().cpu()) for k, v in comp.items()})
    avg = average_rows(rows)
    if "pred_base_ratio" in avg:
        avg["val_pred_base_ratio"] = avg["pred_base_ratio"]
    return avg


def save_checkpoint(path, model, optimizer, args, epoch, selection, norm, stage_name):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "config": vars(args),
            "epoch": int(epoch),
            "selection_value": float(selection),
            "feature_norm": {"mean": norm["mean"], "std": norm["std"], "count": norm["count"]},
            "stage": stage_name,
            "model_type": EXPERIMENT_NAME,
            "input_size": ACC_CURVE_V3_INPUT_SIZE,
            "state_dim": ACC_CURVE_V3_STATE_DIM,
            "root_index": ROOT_INDEX,
            "leaf_indices": LEAF_INDICES,
            "leaf_sensor_names": ACC_CURVE_V3_LEAF_SENSOR_NAMES,
            "root_excluded_from_prediction_loss_metric": True,
            "output_keys": {
                "pred_leaf_rel_acc": "[T,B,15] decoded leaf-relative acceleration in m/s^2",
                "base": "[T,B,15] decoded causal Butterworth IMU leaf-relative baseline",
                "residual": "pred_leaf_rel_acc - base",
            },
            "contract": "99D v1-style feature -> 15D leaf-relative acceleration in model/world frame M",
            "target_contract": "aGT_leaf_rel_butter2_4hz[5,3] from v4 causal Butterworth zero-trans FK diff acceleration",
            "normalization_contract": "feature z-score fitted from AMASS train split only; outputs and targets remain m/s^2",
        },
        path,
    )


def train_stage(model, train_records, val_records, norm, args, output_dir, stage_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds = AccCurveV3WindowDataset(train_records, args.window, args.stride, norm=norm)
    val_ds = AccCurveV3WindowDataset(val_records, args.window, args.stride, norm=norm)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_value = float("inf")
    best_epoch = 0
    start_epoch = 1
    log_path = output_dir / "train_log.jsonl"
    if getattr(args, "resume", False):
        last_path = output_dir / "last_loss.pt"
        if not last_path.exists():
            last_path = output_dir / "last.pt"
        if last_path.exists():
            last_ckpt = torch.load(last_path, map_location=DEVICE)
            model.load_state_dict(last_ckpt["model_state_dict"])
            if last_ckpt.get("optimizer_state_dict") is not None:
                try:
                    optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
                except ValueError:
                    pass
            start_epoch = int(last_ckpt.get("epoch", 0)) + 1
        best_path = output_dir / "best_loss.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=DEVICE)
            best_value = float(best_ckpt.get("selection_value", float("inf")))
            best_epoch = int(best_ckpt.get("epoch", 0))
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
        "feature_normalization": "AMASS train split only",
        "target_key": "aGT_leaf_rel_butter2_4hz",
        "base_key": "aIMU_leaf_rel_butter2_4hz",
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2) + "\n")
    print(json.dumps(dataset_summary, indent=2))
    if start_epoch == 1:
        zero_val = eval_windows(model, val_loader, residual_l2_weight=args.residual_l2_weight)
        zero_selection = float(zero_val.get("val_pred_base_ratio", float("inf")))
        save_checkpoint(output_dir / "epoch0.pt", model, optimizer, args, 0, zero_selection, norm, stage_name)
        if zero_selection < best_value:
            best_value = zero_selection
            best_epoch = 0
            save_checkpoint(output_dir / "best.pt", model, optimizer, args, 0, zero_selection, norm, stage_name)
            save_checkpoint(output_dir / "best_loss.pt", model, optimizer, args, 0, zero_selection, norm, stage_name)
        row = {
            "stage": stage_name,
            "epoch": 0,
            "seconds": 0.0,
            "train": {},
            "val": zero_val,
            "selection": zero_selection,
            "lr": optimizer.param_groups[0]["lr"],
            "checkpoint": "epoch0.pt",
        }
        with log_path.open("a") as f:
            f.write(json.dumps(finite_json(row)) + "\n")
        print(json.dumps(finite_json(row)))
    if start_epoch > args.epochs:
        return {"best_epoch": best_epoch, "best_selection": best_value, "dataset_summary": dataset_summary}
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        model.train()
        train_rows = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, comp = run_batch(model, batch, residual_l2_weight=args.residual_l2_weight)
            reg = args.control_prior_weight * comp["control_point_prior_t"]
            total = loss + reg
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            row = {k: float(v.detach().cpu()) for k, v in comp.items()}
            row["total_loss"] = float(total.detach().cpu())
            train_rows.append(row)
        val_avg = eval_windows(model, val_loader, residual_l2_weight=args.residual_l2_weight)
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
            f.write(json.dumps(finite_json(row)) + "\n")
        print(json.dumps(finite_json(row)))
        save_checkpoint(output_dir / "last.pt", model, optimizer, args, epoch, selection, norm, stage_name)
        save_checkpoint(output_dir / "last_loss.pt", model, optimizer, args, epoch, selection, norm, stage_name)
        if selection < best_value:
            best_value = selection
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, args, epoch, selection, norm, stage_name)
            save_checkpoint(output_dir / "best_loss.pt", model, optimizer, args, epoch, selection, norm, stage_name)
    return {"best_epoch": best_epoch, "best_selection": best_value, "dataset_summary": dataset_summary}


def maybe_install_base_fallback(stage_dir, zero_model_state, val_records, norm, args, stage_name):
    """Keep a base-equivalent checkpoint available if learned residuals hurt validation."""
    model = PLStyleAccCurveV3LeafRelModule(
        hidden_size=args.hidden_size,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
    ).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in zero_model_state.items()})
    val_ds = AccCurveV3WindowDataset(val_records, args.window, args.stride, norm=norm)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
        pin_memory=torch.cuda.is_available(),
    )
    val_avg = eval_windows(model, val_loader, residual_l2_weight=args.residual_l2_weight)
    selection = float(val_avg.get("val_pred_base_ratio", float("inf")))
    save_checkpoint(stage_dir / "base_epoch0.pt", model, None, args, 0, selection, norm, stage_name)
    best_path = stage_dir / "best_loss.pt"
    current_best = float("inf")
    if best_path.exists():
        current_best = float(torch.load(best_path, map_location="cpu").get("selection_value", float("inf")))
    row = {
        "stage": stage_name,
        "epoch": "base_epoch0",
        "seconds": 0.0,
        "train": {},
        "val": val_avg,
        "selection": selection,
        "checkpoint": "base_epoch0.pt",
        "note": "zero-init base-equivalent fallback; pred equals decoded base",
    }
    with (stage_dir / "train_log.jsonl").open("a") as f:
        f.write(json.dumps(finite_json(row)) + "\n")
    print(json.dumps(finite_json(row)))
    if selection < current_best:
        save_checkpoint(stage_dir / "best.pt", model, None, args, 0, selection, norm, stage_name)
        save_checkpoint(stage_dir / "best_loss.pt", model, None, args, 0, selection, norm, stage_name)
        return {"used_base_fallback": True, "base_selection": selection}
    return {"used_base_fallback": False, "base_selection": selection}


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    model = PLStyleAccCurveV3LeafRelModule(
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
    return out["pred_leaf_rel_acc"].detach().cpu(), out["base"].detach().cpu()


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


def eval_record_metrics(pred, base, target, valid, name, dataset, split):
    valid = valid.bool()
    if not bool(valid.any()):
        return {"name": name, "dataset": dataset, "split": split, "valid_frames": 0}
    pred_v = pred[valid].reshape(-1, 5, 3)
    base_v = base[valid].reshape(-1, 5, 3)
    target_v = target[valid].reshape(-1, 5, 3)
    pred_err = pred_v - target_v
    base_err = base_v - target_v
    residual = pred_v - base_v
    pred_flat = pred_v.reshape(-1, 3)
    target_flat = target_v.reshape(-1, 3)
    base_flat = base_v.reshape(-1, 3)
    pred_l2 = pred_err.norm(dim=-1).mean()
    base_l2 = base_err.norm(dim=-1).mean()
    pred_rmse = pred_err.square().mean().sqrt()
    base_rmse = base_err.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=-1, eps=1e-8)
    base_cosine = torch.nn.functional.cosine_similarity(base_flat, target_flat, dim=-1, eps=1e-8)
    row = {
        "name": name,
        "dataset": dataset,
        "split": split,
        "valid_frames": int(valid.sum()),
        "pred_l2": float(pred_l2),
        "base_l2": float(base_l2),
        "pred_rmse": float(pred_rmse),
        "base_rmse": float(base_rmse),
        "pred_mae": float(pred_err.abs().mean()),
        "base_mae": float(base_err.abs().mean()),
        "pred_base_l2_ratio": float(pred_l2 / base_l2.clamp_min(1e-12)),
        "pred_base_rmse_ratio": float(pred_rmse / base_rmse.clamp_min(1e-12)),
        "corr": corrcoef(pred_v, target_v),
        "base_corr": corrcoef(base_v, target_v),
        "cosine": float(cosine.mean()),
        "base_cosine": float(base_cosine.mean()),
        "mag_mae": float((pred_v.norm(dim=-1) - target_v.norm(dim=-1)).abs().mean()),
        "base_mag_mae": float((base_v.norm(dim=-1) - target_v.norm(dim=-1)).abs().mean()),
        "residual_std": float(residual.std()),
        "residual_p95": float(torch.quantile(residual.norm(dim=-1).reshape(-1), 0.95)),
    }
    pred_per_sensor = pred_err.norm(dim=-1).mean(dim=0)
    base_per_sensor = base_err.norm(dim=-1).mean(dim=0)
    for idx, sensor in enumerate(ACC_CURVE_V3_LEAF_SENSOR_NAMES):
        p = float(pred_per_sensor[idx])
        b = float(base_per_sensor[idx])
        row[f"pred_l2_{sensor}"] = p
        row[f"base_l2_{sensor}"] = b
        row[f"pred_base_l2_ratio_{sensor}"] = p / max(b, 1e-12)
        row[f"corr_{sensor}"] = corrcoef(pred_v[:, idx], target_v[:, idx])
        row[f"base_corr_{sensor}"] = corrcoef(base_v[:, idx], target_v[:, idx])
    return row


def weighted_aggregate(rows):
    rows = [row for row in rows if int(row.get("valid_frames", 0)) > 0]
    if not rows:
        return {}
    total_frames = sum(int(row["valid_frames"]) for row in rows)
    out = {"num_sequences": len(rows), "valid_frames": int(total_frames)}
    weighted_keys = [
        "pred_l2",
        "base_l2",
        "pred_rmse",
        "base_rmse",
        "pred_mae",
        "base_mae",
        "corr",
        "base_corr",
        "cosine",
        "base_cosine",
        "mag_mae",
        "base_mag_mae",
        "residual_std",
        "residual_p95",
    ]
    for key in weighted_keys:
        vals = [(float(row[key]), int(row["valid_frames"])) for row in rows if key in row and math.isfinite(float(row[key]))]
        if vals:
            out[key] = sum(v * w for v, w in vals) / sum(w for _, w in vals)
    if "pred_l2" in out and "base_l2" in out:
        out["pred_base_l2_ratio"] = out["pred_l2"] / max(out["base_l2"], 1e-12)
    if "pred_rmse" in out and "base_rmse" in out:
        out["pred_base_rmse_ratio"] = out["pred_rmse"] / max(out["base_rmse"], 1e-12)
    for sensor in ACC_CURVE_V3_LEAF_SENSOR_NAMES:
        for key in (
            f"pred_l2_{sensor}",
            f"base_l2_{sensor}",
            f"corr_{sensor}",
            f"base_corr_{sensor}",
        ):
            vals = [(float(row[key]), int(row["valid_frames"])) for row in rows if key in row and math.isfinite(float(row[key]))]
            if vals:
                out[key] = sum(v * w for v, w in vals) / sum(w for _, w in vals)
        p_key, b_key = f"pred_l2_{sensor}", f"base_l2_{sensor}"
        if p_key in out and b_key in out:
            out[f"pred_base_l2_ratio_{sensor}"] = out[p_key] / max(out[b_key], 1e-12)
    return out


def write_eval_outputs(output_dir, split_name, checkpoint, cache_manifest, rows, ckpt):
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = weighted_aggregate(rows)
    result = {
        "experiment": EXPERIMENT_NAME,
        "split": split_name,
        "checkpoint": str(checkpoint),
        "cache_manifest": str(cache_manifest),
        "num_sequences": len(rows),
        "aggregate": aggregate,
        "rows": rows,
        "checkpoint_selection": ckpt.get("selection_value"),
        "root_index": ROOT_INDEX,
        "leaf_indices": LEAF_INDICES,
        "leaf_sensor_names": ACC_CURVE_V3_LEAF_SENSOR_NAMES,
        "root_excluded_from_prediction_loss_metric": True,
    }
    (output_dir / f"{split_name}_eval.json").write_text(json.dumps(finite_json(result), indent=2) + "\n")
    if rows:
        keys = sorted({k for row in rows for k in row})
        with (output_dir / f"{split_name}_per_sequence.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    md = [
        f"# AccCurve v3 {split_name} Eval",
        "",
        f"checkpoint: `{checkpoint}`",
        f"cache manifest: `{cache_manifest}`",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "pred_l2",
        "base_l2",
        "pred_base_l2_ratio",
        "pred_rmse",
        "base_rmse",
        "pred_base_rmse_ratio",
        "corr",
        "base_corr",
        "cosine",
        "mag_mae",
        "valid_frames",
        "num_sequences",
    ):
        if key in aggregate:
            value = aggregate[key]
            md.append(f"| {key} | {value:.6f} |" if isinstance(value, float) else f"| {key} | {value} |")
    (output_dir / f"{split_name}_eval.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"split": split_name, "aggregate": finite_json(aggregate)}, indent=2))
    return result


def evaluate_records(checkpoint, cache_manifest, records, output_dir, split_name):
    model, norm, ckpt = load_model(checkpoint)
    rows = []
    model.eval()
    for record in records:
        pred, base = predict_record(model, record, norm)
        rows.append(
            eval_record_metrics(
                pred,
                base,
                record["target"],
                record["valid_mask"],
                record["name"],
                record["dataset"],
                record["split"],
            )
        )
    return write_eval_outputs(output_dir, split_name, checkpoint, cache_manifest, rows, ckpt)


def smoke_zero_init(records, norm, args):
    model = PLStyleAccCurveV3LeafRelModule(
        hidden_size=args.hidden_size,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
    ).to(DEVICE)
    ds = AccCurveV3WindowDataset(records, args.window, args.stride, norm=norm)
    if len(ds) == 0:
        raise RuntimeError("Smoke dataset has no windows")
    batch = collate_windows([ds[0]])
    with torch.no_grad():
        out = model.forward_sequence(batch["feature"].to(DEVICE), batch["base"].to(DEVICE))
    diff = (out["pred_leaf_rel_acc"].cpu() - out["base"].cpu()).abs().max()
    return float(diff)


def judgement(dip, tc):
    dip_improve = dip["pred_base_l2_ratio"] < 1.0
    tc_improve = tc["pred_base_l2_ratio"] < 1.0
    dip_corr_ok = dip["corr"] >= dip["base_corr"] - 0.01
    tc_corr_ok = tc["corr"] >= tc["base_corr"] - 0.01
    dip_rmse_improve = dip["pred_rmse"] < dip["base_rmse"]
    tc_rmse_improve = tc["pred_rmse"] < tc["base_rmse"]
    if dip_improve and tc_improve and dip_rmse_improve and tc_rmse_improve and dip_corr_ok and tc_corr_ok:
        return "strong pass"
    if dip_improve and tc["pred_base_l2_ratio"] <= 1.05 and dip_corr_ok and tc_corr_ok:
        return "pass"
    if dip_improve and tc["pred_base_l2_ratio"] <= 1.10 and dip_corr_ok and tc["corr"] >= tc["base_corr"] - 0.03:
        return "soft pass"
    return "fail"


def fmt(value):
    if isinstance(value, int):
        return str(value)
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6f}"


def summary_markdown(output_dir, args, train_result, eval_results):
    dip = eval_results["dip_test"]["aggregate"]
    tc = eval_results["totalcapture_test"]["aggregate"]
    decision = judgement(dip, tc)
    rows = [
        ("DIP", "test", dip),
        ("TotalCapture", "test", tc),
    ]
    lines = [
        "# AccCurve v3 Leaf-Relative Causal Butterworth",
        "",
        "## 1. Contract",
        "",
        "AccCurve v3 is an AccCurve v1-style residual acceleration module with a leaf-relative causal Butterworth target.",
        "",
        "- root index: 5",
        "- leaf indices: 0..4",
        "- root: reference only; excluded from prediction/loss/metric",
        "- input feature: acc_raw[15] + acc_smooth[15] + acc_raw_minus_smooth[15] + wM[18] + RMB_6d[36] = 99D",
        "- base: aIMU_leaf_rel_butter2_4hz[15]",
        "- target: aGT_leaf_rel_butter2_4hz[15]",
        "- output: pred_leaf_rel_acc[15] = [5,3]",
        "- frame: model/world frame M",
        "- smoothing: causal Butterworth order=2 cutoff=4Hz on both IMU base and GT target",
        "- units: m/s^2; feature z-score only; output/target are not normalized",
        "",
        "## 2. Relation to Previous Versions",
        "",
        "- AccCurve v1: same style of input/network/residual curve, 6-sensor acceleration target, previous smoothing.",
        "- AccCurve v2: strict GTFK q/qdot/qddot/rJS absolute 6-sensor target.",
        "- AccCurve v3: v1-style module, 5 leaf-relative acceleration outputs, causal Butterworth smoothing on both IMU and GT, root used only as reference.",
        "",
        "## 3. Training Protocol",
        "",
        "- AMASS pretrain: synthetic sanity only.",
        "- DIP finetune: checkpoint selection on DIP val pred/base L2 ratio.",
        "- Final primary eval: DIP test and TotalCapture test.",
        "- No PL/NewPL/full-pipeline/S4 claim.",
        "",
        "| Stage | Best epoch | Best validation pred/base ratio | Train seq | Val seq | Train windows | Val windows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("amass_pretrain", "AMASS pretrain"), ("dip_finetune", "DIP finetune")):
        result = train_result[key]
        ds = result["dataset_summary"]
        lines.append(
            f"| {label} | {result['best_epoch']} | {result['best_selection']:.6f} | "
            f"{ds['train_sequences']} | {ds['val_sequences']} | {ds['train_windows']} | {ds['val_windows']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Main Result Table",
            "",
            "| Dataset | Split | Model | Pred L2 | Base L2 | Pred/Base L2 | Pred RMSE | Base RMSE | Pred/Base RMSE | Corr | Base Corr | Cosine | Mag MAE |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, split, agg in rows:
        lines.append(
            f"| {dataset} | {split} | acc_curve_v3 | {fmt(agg['pred_l2'])} | {fmt(agg['base_l2'])} | "
            f"{fmt(agg['pred_base_l2_ratio'])} | {fmt(agg['pred_rmse'])} | {fmt(agg['base_rmse'])} | "
            f"{fmt(agg['pred_base_rmse_ratio'])} | {fmt(agg['corr'])} | {fmt(agg['base_corr'])} | "
            f"{fmt(agg['cosine'])} | {fmt(agg['mag_mae'])} |"
        )
    lines.extend(
        [
            "",
            "## 5. Per-Sensor Table",
            "",
            "Only leaf sensors are included. Cache order is left_forearm, right_forearm, left_lower_leg, right_lower_leg, head.",
            "",
            "| Dataset | Sensor | Pred L2 | Base L2 | Pred/Base L2 | Pred Corr | Base Corr |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, _, agg in rows:
        for sensor in ACC_CURVE_V3_LEAF_SENSOR_NAMES:
            lines.append(
                f"| {dataset} | {sensor} | {fmt(agg[f'pred_l2_{sensor}'])} | {fmt(agg[f'base_l2_{sensor}'])} | "
                f"{fmt(agg[f'pred_base_l2_ratio_{sensor}'])} | {fmt(agg[f'corr_{sensor}'])} | {fmt(agg[f'base_corr_{sensor}'])} |"
            )
    lines.extend(
        [
            "",
            "## 6. Required Judgement",
            "",
            f"Decision: **{decision}**.",
            "",
            "- Strong pass: DIP and TC both improve over base in L2/RMSE, with corr not decreasing by more than 0.01.",
            "- Pass: DIP improves, TC pred/base L2 ratio <= 1.05, and DIP/TC corr does not decrease by more than 0.01.",
            "- Soft pass: DIP improves, TC is not meaningfully worse, and corr remains close to base.",
            "- Fail: DIP worsens, TC worsens sharply, or corr drops substantially.",
            "",
            "## 7. V4 Base Sanity",
            "",
            f"- DIP v4 all-reference butter L2/RMSE/corr = {V4_REFERENCE['DIP']['l2']:.6f} / {V4_REFERENCE['DIP']['rmse']:.6f} / {V4_REFERENCE['DIP']['corr']:.6f}.",
            f"- TotalCapture v4 all-reference butter L2/RMSE/corr = {V4_REFERENCE['TotalCapture']['l2']:.6f} / {V4_REFERENCE['TotalCapture']['rmse']:.6f} / {V4_REFERENCE['TotalCapture']['corr']:.6f}.",
            "- Final eval is test split only, so exact base numbers can differ from all-split v4 references.",
            "",
            "## 8. Non-Claims",
            "",
            "- This is not PL/NewPL training.",
            "- This is not full-pipeline evaluation.",
            "- This does not claim pose improvement.",
            "- AMASS is synthetic and not primary evidence.",
            "- Root channel is not predicted or evaluated.",
            "",
            "## Artifacts",
            "",
            f"- output root: `{output_dir}`",
            f"- final checkpoint: `{output_dir / 'dip_finetune' / 'best_loss.pt'}`",
            f"- config: `{output_dir / 'config.json'}`",
            f"- feature norm: `{output_dir / 'feature_norm.pt'}`",
            f"- eval JSONs: `{output_dir / 'eval'}`",
            "",
            "## Commands",
            "",
            "```bash",
            shlex.join(sys.argv),
            "```",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return decision


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate AccCurve v3 leaf-relative module.")
    parser.add_argument("--mode", choices=("train_full", "eval"), default="train_full")
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--output-dir", default="data/experiments/acc_curve_v3_leafrel_causal_butter_20260618")
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
    parser.add_argument("--residual-l2-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amass-val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-sequences", type=int, default=0)
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
    if args.mode == "train_full":
        if output_dir.exists() and any(output_dir.iterdir()):
            if args.overwrite:
                shutil.rmtree(output_dir)
            elif not args.resume:
                raise FileExistsError(f"{output_dir} is not empty; pass --overwrite or --resume.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "experiment": EXPERIMENT_NAME,
                "input_size": ACC_CURVE_V3_INPUT_SIZE,
                "state_dim": ACC_CURVE_V3_STATE_DIM,
                "root_index": ROOT_INDEX,
                "leaf_indices": LEAF_INDICES,
                "leaf_sensor_names": ACC_CURVE_V3_LEAF_SENSOR_NAMES,
                "root_excluded_from_prediction_loss_metric": True,
            },
            indent=2,
        )
        + "\n"
    )
    max_records = args.max_sequences if args.max_sequences else 0
    records, manifest = load_records(args.cache_manifest, max_sequences=max_records)
    groups = group_records(records)
    required = [("AMASS", "train"), ("DIP", "train"), ("DIP", "val"), ("DIP", "test"), ("TotalCapture", "test")]
    missing = [key for key in required if key not in groups]
    if missing:
        raise RuntimeError(f"Missing required dataset/split groups: {missing}")
    if args.mode == "eval":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for eval")
        eval_dir = output_dir
        evaluate_records(args.checkpoint, args.cache_manifest, groups[("DIP", "val")], eval_dir, "dip_val")
        evaluate_records(args.checkpoint, args.cache_manifest, groups[("DIP", "test")], eval_dir, "dip_test")
        evaluate_records(
            args.checkpoint,
            args.cache_manifest,
            groups[("TotalCapture", "test")],
            eval_dir,
            "totalcapture_test",
        )
        return

    amass_records = groups[("AMASS", "train")]
    if args.max_amass_sequences:
        amass_records = amass_records[: args.max_amass_sequences]
    dip_train_records = groups[("DIP", "train")]
    if args.max_dip_train_sequences:
        dip_train_records = dip_train_records[: args.max_dip_train_sequences]
    dip_val_records = groups[("DIP", "val")]
    dip_test_records = groups[("DIP", "test")]
    tc_test_records = groups[("TotalCapture", "test")]
    if args.max_eval_sequences:
        dip_val_records = dip_val_records[: args.max_eval_sequences]
        dip_test_records = dip_test_records[: args.max_eval_sequences]
        tc_test_records = tc_test_records[: args.max_eval_sequences]
    amass_train, amass_val = split_amass_hash(amass_records, args.amass_val_ratio)
    norm = fit_feature_norm(amass_train)
    torch.save(norm, output_dir / "feature_norm.pt")
    zero_diff = smoke_zero_init(amass_train, norm, args)
    if zero_diff > 1e-5:
        raise RuntimeError(f"zero-init AccCurve v3 output is not close to base: max_abs={zero_diff}")
    model = PLStyleAccCurveV3LeafRelModule(
        hidden_size=args.hidden_size,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
    ).to(DEVICE)
    zero_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    amass_args = argparse.Namespace(**vars(args))
    amass_args.epochs = args.epochs
    amass_result = train_stage(model, amass_train, amass_val, norm, amass_args, output_dir / "amass_pretrain", "amass_pretrain")
    amass_best = output_dir / "amass_pretrain" / "best_loss.pt"
    if amass_best.exists():
        model.load_state_dict(torch.load(amass_best, map_location=DEVICE)["model_state_dict"])
    dip_args = argparse.Namespace(**vars(args))
    dip_args.epochs = args.dip_epochs
    dip_result = train_stage(model, dip_train_records, dip_val_records, norm, dip_args, output_dir / "dip_finetune", "dip_finetune")
    dip_fallback = maybe_install_base_fallback(
        output_dir / "dip_finetune",
        zero_model_state,
        dip_val_records,
        norm,
        dip_args,
        "dip_finetune",
    )
    final_ckpt = output_dir / "dip_finetune" / "best_loss.pt"
    eval_dir = output_dir / "eval"
    eval_results = {
        "dip_val": evaluate_records(final_ckpt, args.cache_manifest, dip_val_records, eval_dir, "dip_val"),
        "dip_test": evaluate_records(final_ckpt, args.cache_manifest, dip_test_records, eval_dir, "dip_test"),
        "totalcapture_test": evaluate_records(final_ckpt, args.cache_manifest, tc_test_records, eval_dir, "totalcapture_test"),
    }
    train_result = {
        "experiment": EXPERIMENT_NAME,
        "cache_manifest": args.cache_manifest,
        "manifest_num_sequences": manifest.get("num_sequences"),
        "zero_init_max_abs_pred_minus_base": zero_diff,
        "amass_pretrain": amass_result,
        "dip_finetune": dip_result,
        "dip_base_fallback": dip_fallback,
        "final_checkpoint": str(final_ckpt),
        "eval": {key: value["aggregate"] for key, value in eval_results.items()},
        "root_index": ROOT_INDEX,
        "leaf_indices": LEAF_INDICES,
        "leaf_sensor_names": ACC_CURVE_V3_LEAF_SENSOR_NAMES,
        "root_excluded_from_prediction_loss_metric": True,
    }
    decision = summary_markdown(output_dir, args, train_result, eval_results)
    train_result["judgement"] = decision
    (output_dir / "train_result.json").write_text(json.dumps(finite_json(train_result), indent=2) + "\n")
    print(json.dumps(finite_json(train_result), indent=2))


if __name__ == "__main__":
    main()
