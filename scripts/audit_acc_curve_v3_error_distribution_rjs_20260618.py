#!/usr/bin/env python3
"""Audit why AccCurve v3 leaf-relative causal Butterworth correction fails on TC.

This is a diagnostic-only script.  It keeps the AccCurve v3 contract:
root IMU index 5 is used only as the reference acceleration, while prediction,
loss, residual, and correction metrics are computed on the five leaf sensors.
All accelerations are model/world-frame M vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import articulate as art
from acc_curve_v3_leafrel import (
    ACC_CURVE_V3_LEAF_SENSOR_NAMES,
    PLStyleAccCurveV3LeafRelModule,
    acc_curve_v3_leafrel_features,
)
from l4_q75_utils import q75_to_pose_tran
from l4_sensor_offset_utils import DT, IMU_JOINTS, SENSOR_NAMES


EXPERIMENT = "acc_curve_v3_error_distribution_rjs_audit_20260618"
ROOT_INDEX = 5
LEAF_INDICES = [0, 1, 2, 3, 4]
LEAF_SENSOR_NAMES = list(ACC_CURVE_V3_LEAF_SENSOR_NAMES)
ALL_GROUPS = [
    ("AMASS", "train"),
    ("DIP", "train"),
    ("DIP", "val"),
    ("DIP", "test"),
    ("TotalCapture", "train"),
    ("TotalCapture", "val"),
    ("TotalCapture", "test"),
]
PRIMARY_DISTANCE_PAIRS = [
    (("DIP", "train"), ("DIP", "test")),
    (("DIP", "train"), ("TotalCapture", "test")),
    (("DIP", "train"), ("TotalCapture", "train")),
    (("DIP", "train"), ("TotalCapture", "val")),
    (("DIP", "train"), ("TotalCapture", "test")),
    (("DIP", "all"), ("TotalCapture", "all")),
]
V3_RECAP = {
    "DIP_test": {
        "pred_l2": 0.990334,
        "base_l2": 1.196030,
        "pred_base_l2_ratio": 0.828017,
        "pred_corr": 0.958276,
        "base_corr": 0.943321,
    },
    "TotalCapture_test": {
        "pred_l2": 1.365116,
        "base_l2": 1.052403,
        "pred_base_l2_ratio": 1.297142,
        "pred_corr": 0.923813,
        "base_corr": 0.946864,
    },
}


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2, sort_keys=True) + "\n")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; use --overwrite")
    if overwrite and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("experiment") != "acc_leaf_relative_residual_v4_causal_butterworth_20260618":
        raise ValueError(f"Expected v4 cache manifest, got {manifest.get('experiment')!r}")
    if int(manifest.get("root_index", -1)) != ROOT_INDEX:
        raise ValueError(f"Expected root_index=5, got {manifest.get('root_index')!r}")
    if list(manifest.get("leaf_indices", [])) != LEAF_INDICES:
        raise ValueError(f"Expected leaf_indices={LEAF_INDICES}, got {manifest.get('leaf_indices')!r}")
    return manifest


def cache_entries(manifest: dict) -> List[dict]:
    entries = list(manifest.get("cache_files", []))
    if not entries:
        raise ValueError("v4 cache manifest has no cache_files")
    return entries


def group_key(dataset: str, split: str) -> str:
    return f"{dataset}/{split}"


def all_key(dataset: str) -> str:
    return f"{dataset}/all"


def load_record(entry: dict) -> dict:
    data = torch.load(entry["path"], map_location="cpu")
    meta = data["meta"]
    if int(meta.get("root_index", -1)) != ROOT_INDEX:
        raise ValueError(f"{entry['path']} has root_index={meta.get('root_index')!r}")
    if list(meta.get("leaf_indices", [])) != LEAF_INDICES:
        raise ValueError(f"{entry['path']} has leaf_indices={meta.get('leaf_indices')!r}")
    if not bool(meta.get("root_excluded_from_metrics", False)):
        raise ValueError(f"{entry['path']} does not mark root_excluded_from_metrics=true")
    return data


def valid_leaf_tensor(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.bool() & torch.isfinite(x).all(dim=(-1, -2))
    return x[mask].float()


def flatten_leaf_samples(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).float()


def sample_rows(x: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.randperm(x.shape[0], generator=g)[:max_rows]
    return x[idx]


def sample_np_time(x: np.ndarray, max_rows: int = 5000) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[0] <= max_rows:
        return x
    idx = np.linspace(0, x.shape[0] - 1, max_rows).astype(np.int64)
    return x[idx]


def corrcoef_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return float("nan")
    den = np.linalg.norm(a[mask]) * np.linalg.norm(b[mask])
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a[mask], b[mask]) / den)


def skew_kurt_np(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")
    mu = x.mean()
    std = x.std()
    if std <= 1e-12:
        return 0.0, -3.0
    z = (x - mu) / std
    return float(np.mean(z**3)), float(np.mean(z**4) - 3.0)


def quantile_np(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def axis_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1])
    out = {}
    for axis, name in enumerate(("x", "y", "z")):
        col = flat[:, axis]
        out[f"mean_{name}"] = float(np.nanmean(col)) if col.size else float("nan")
        out[f"std_{name}"] = float(np.nanstd(col)) if col.size else float("nan")
        out[f"rms_{name}"] = float(np.sqrt(np.nanmean(col**2))) if col.size else float("nan")
        out[f"mae_{name}"] = float(np.nanmean(np.abs(col))) if col.size else float("nan")
        out[f"p95_abs_{name}"] = quantile_np(np.abs(col), 0.95)
    return out


def distribution_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    vectors = x.reshape(-1, 3)
    norms = np.linalg.norm(vectors, axis=-1)
    flat = vectors.reshape(-1)
    skewness, kurtosis = skew_kurt_np(flat)
    finite_vectors = vectors[np.isfinite(vectors).all(axis=1)]
    cov = np.cov(finite_vectors.T).tolist() if finite_vectors.shape[0] >= 2 else None
    return {
        "count": int(vectors.shape[0]),
        "mean": float(np.nanmean(flat)),
        "std": float(np.nanstd(flat)),
        "rms": float(np.sqrt(np.nanmean(flat**2))),
        "mae": float(np.nanmean(np.abs(flat))),
        "l2": float(np.nanmean(norms)),
        "p50": quantile_np(norms, 0.50),
        "p75": quantile_np(norms, 0.75),
        "p90": quantile_np(norms, 0.90),
        "p95": quantile_np(norms, 0.95),
        "p99": quantile_np(norms, 0.99),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "max_abs": float(np.nanmax(np.abs(flat))),
        "covariance_xyz": cov,
        "energy": float(np.nanmean(norms)),
        **axis_stats(vectors),
    }


def high_frequency_ratio(x: torch.Tensor, valid: torch.Tensor, fps: float = 60.0, cutoff_hz: float = 4.0) -> float:
    y = valid_leaf_tensor(x, valid)
    if y.shape[0] < 8:
        return float("nan")
    flat = y.reshape(y.shape[0], -1).double()
    flat = flat - flat.mean(dim=0, keepdim=True)
    spec = torch.fft.rfft(flat, dim=0)
    power = spec.abs().square()
    freqs = torch.fft.rfftfreq(flat.shape[0], d=1.0 / fps)
    total = power.sum().item()
    if total <= 1e-12:
        return 0.0
    high = power[freqs >= cutoff_hz].sum().item()
    return float(high / total)


def autocorr_lag(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] <= lag:
        return float("nan")
    return corrcoef_np(x[:-lag], x[lag:])


def best_crosscorr_lag(base: np.ndarray, target: np.ndarray, max_lag: int = 10) -> dict:
    base = np.asarray(base, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    best = {"lag": 0, "corr": float("nan")}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = base[-lag:], target[: lag or None]
        elif lag > 0:
            a, b = base[:-lag], target[lag:]
        else:
            a, b = base, target
        c = corrcoef_np(a, b)
        if math.isfinite(c) and (not math.isfinite(best["corr"]) or c > best["corr"]):
            best = {"lag": int(lag), "corr": float(c)}
    return best


def diag_frechet(mu1, std1, mu2, std2) -> float:
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    std1 = np.asarray(std1, dtype=np.float64)
    std2 = np.asarray(std2, dtype=np.float64)
    return float(np.sum((mu1 - mu2) ** 2) + np.sum((std1 - std2) ** 2))


def rbf_mmd(x: np.ndarray, y: np.ndarray, max_n: int = 5000) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    y = y[np.isfinite(y).all(axis=1)]
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    if x.shape[0] > max_n:
        rng = np.random.default_rng(1234)
        x = x[rng.choice(x.shape[0], size=max_n, replace=False)]
    if y.shape[0] > max_n:
        rng = np.random.default_rng(5678)
        y = y[rng.choice(y.shape[0], size=max_n, replace=False)]
    z = np.concatenate([x[: min(1000, len(x))], y[: min(1000, len(y))]], axis=0)
    if z.shape[0] < 2:
        return float("nan")
    diffs = z[: min(512, z.shape[0]), None, :] - z[None, : min(512, z.shape[0]), :]
    d2 = np.sum(diffs * diffs, axis=-1)
    sigma2 = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    sigma2 = max(sigma2, 1e-6)

    def kmean(a, b):
        total = 0.0
        count = 0
        chunk = 1024
        for i in range(0, a.shape[0], chunk):
            d = a[i:i + chunk, None, :] - b[None, :, :]
            total += np.exp(-np.sum(d * d, axis=-1) / (2.0 * sigma2)).sum()
            count += d.shape[0] * d.shape[1]
        return total / max(count, 1)

    return float(kmean(x, x) + kmean(y, y) - 2.0 * kmean(x, y))


def wasserstein_per_dim(x: np.ndarray, y: np.ndarray) -> List[float]:
    try:
        from scipy.stats import wasserstein_distance
    except Exception:
        return []
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = []
    for dim in range(x.shape[1]):
        a = x[:, dim]
        b = y[:, dim]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        out.append(float(wasserstein_distance(a, b)) if a.size and b.size else float("nan"))
    return out


class RunningRJS:
    def __init__(self):
        self.values = []
        self.rows = []

    def add(self, rjs: torch.Tensor, dataset: str, split: str, name: str):
        arr = rjs.detach().cpu().float().numpy()
        self.values.append(arr)
        for sidx, sensor in enumerate(SENSOR_NAMES):
            v = arr[sidx]
            self.rows.append({
                "dataset": dataset,
                "split": split,
                "sequence": name,
                "sensor": sensor,
                "sensor_index": sidx,
                "x": float(v[0]),
                "y": float(v[1]),
                "z": float(v[2]),
                "norm": float(np.linalg.norm(v)),
            })


class SampleStore:
    def __init__(self, max_rows: int, seed: int):
        self.max_rows = int(max_rows)
        self.seed = int(seed)
        self.data = None
        self.count = 0

    def add(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.detach().cpu().float()
        self.count += int(x.shape[0])
        if self.data is None:
            self.data = sample_rows(x, self.max_rows, self.seed + self.count)
            return
        merged = torch.cat((self.data, x), dim=0)
        self.data = sample_rows(merged, self.max_rows, self.seed + self.count)

    def tensor(self) -> torch.Tensor:
        if self.data is None:
            return torch.empty(0, 15)
        return self.data


def rjs_group_stats(values: List[np.ndarray]) -> dict:
    if not values:
        return {}
    x = np.stack(values, axis=0)  # N,6,3
    out = {"num_sequences": int(x.shape[0])}
    for sidx, sensor in enumerate(SENSOR_NAMES):
        v = x[:, sidx, :]
        n = np.linalg.norm(v, axis=-1)
        out[sensor] = {
            "xyz_mean": np.mean(v, axis=0).tolist(),
            "xyz_std": np.std(v, axis=0).tolist(),
            "norm_mean": float(np.mean(n)),
            "norm_std": float(np.std(n)),
            "norm_p50": float(np.quantile(n, 0.50)),
            "norm_p90": float(np.quantile(n, 0.90)),
            "norm_p95": float(np.quantile(n, 0.95)),
            "norm_min": float(np.min(n)),
            "norm_max": float(np.max(n)),
        }
    flat = x.reshape(x.shape[0], -1)
    out["flat_mean"] = np.mean(flat, axis=0).tolist()
    out["flat_std"] = np.std(flat, axis=0).tolist()
    return out


def mahalanobis(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    d = x - mean
    return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", d, inv_cov, d), 0.0))


class SourcePoseLoader:
    def __init__(self):
        self.cache = {}

    def get(self, source_file: str, sequence_name: str):
        if source_file not in self.cache:
            data = torch.load(source_file, map_location="cpu")
            names = [str(n) for n in data["name"]]
            self.cache[source_file] = (data, {name: idx for idx, name in enumerate(names)})
        data, name_to_idx = self.cache[source_file]
        if sequence_name not in name_to_idx:
            raise KeyError(f"{sequence_name!r} not found in {source_file}")
        idx = name_to_idx[sequence_name]
        return load_pose_from_source(data, idx)


def load_pose_from_source(data: dict, seq_idx: int) -> torch.Tensor:
    if data.get("pose_gt") and data["pose_gt"]:
        return data["pose_gt"][seq_idx].float()
    if data.get("pose") and data["pose"]:
        pose = data["pose"][seq_idx].float()
        if pose.dim() == 2 and pose.shape[-1] == 72:
            pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        return pose.float()
    if data.get("q75_gt") and data["q75_gt"]:
        pose, _ = q75_to_pose_tran(data["q75_gt"][seq_idx].float())
        return pose.float()
    raise KeyError("missing pose_gt/pose/q75_gt")


@torch.no_grad()
def fk_joint_and_offset(
    pose: torch.Tensor,
    rjs: torch.Tensor,
    body_model: art.ParametricModel,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pose = pose.to(device=device, dtype=torch.float32)
    rjs = rjs.to(device=device, dtype=torch.float32)
    joints_out, offset_out, site_out = [], [], []
    zero = None
    for start in range(0, pose.shape[0], batch_size):
        p = pose[start:start + batch_size]
        if zero is None or zero.shape[0] != p.shape[0]:
            zero = torch.zeros(p.shape[0], 3, device=device, dtype=p.dtype)
        grot, joints = body_model.forward_kinematics(p, tran=zero, calc_mesh=False)[:2]
        p_wj = joints[:, IMU_JOINTS]
        r = rjs.view(1, 6, 3, 1).expand(p.shape[0], -1, -1, -1)
        offset = grot[:, IMU_JOINTS].matmul(r).squeeze(-1)
        site = p_wj + offset
        joints_out.append(p_wj.detach().cpu())
        offset_out.append(offset.detach().cpu())
        site_out.append(site.detach().cpu())
    return (
        torch.cat(joints_out, dim=0).float(),
        torch.cat(offset_out, dim=0).float(),
        torch.cat(site_out, dim=0).float(),
    )


def centered_second_difference(x: torch.Tensor) -> torch.Tensor:
    acc = torch.full_like(x, float("nan"))
    if x.shape[0] >= 3:
        acc[1:-1] = (x[:-2] - 2.0 * x[1:-1] + x[2:]) / (DT**2)
    return acc


def leaf_relative(acc: torch.Tensor) -> torch.Tensor:
    leaf = torch.as_tensor(LEAF_INDICES, dtype=torch.long)
    return acc[:, leaf] - acc[:, ROOT_INDEX:ROOT_INDEX + 1]


def fill_nonfinite_time(x: torch.Tensor) -> torch.Tensor:
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


def load_acc_curve_model(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    model = PLStyleAccCurveV3LeafRelModule(
        hidden_size=int(cfg.get("hidden_size", 512)),
        residual_scale=float(cfg.get("residual_scale", 1.0)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm = ckpt["feature_norm"]
    norm = {
        "mean": norm["mean"].float().to(device),
        "std": norm["std"].float().to(device),
        "count": int(norm["count"]),
    }
    return model, norm, ckpt


@torch.no_grad()
def predict_record(data: dict, model, norm: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    a_raw = fill_nonfinite_time(data["aIMU_leaf_rel_raw"].float())
    a_butter = fill_nonfinite_time(data["aIMU_leaf_rel_butter2_4hz"].float())
    wM = fill_nonfinite_time(data["wM"].float())
    RMB = fill_nonfinite_time(data["RMB"].float())
    feature = acc_curve_v3_leafrel_features(a_raw, a_butter, wM, RMB).to(device)
    feature = (feature - norm["mean"]) / norm["std"]
    base = a_butter.reshape(-1, 15).to(device)
    out = model.forward_sequence(feature, base)
    pred = out["pred_leaf_rel_acc"].detach().cpu().reshape(-1, 5, 3)
    decoded_base = out["base"].detach().cpu().reshape(-1, 5, 3)
    return pred, decoded_base


def correction_metrics(pred: torch.Tensor, base: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    mask = valid.bool() & torch.isfinite(pred).all(dim=(-1, -2)) & torch.isfinite(base).all(dim=(-1, -2)) & torch.isfinite(target).all(dim=(-1, -2))
    if not bool(mask.any()):
        return {"valid_frames": 0}
    pred = pred[mask]
    base = base[mask]
    target = target[mask]
    c_true = target - base
    c_pred = pred - base
    base_err = base - target
    pred_err = pred - target
    true_norm = c_true.norm(dim=-1)
    pred_norm = c_pred.norm(dim=-1)
    corr = corrcoef_np(c_pred.numpy(), c_true.numpy())
    cos = torch.nn.functional.cosine_similarity(c_pred.reshape(-1, 3), c_true.reshape(-1, 3), dim=-1, eps=1e-8).mean()
    sign = (torch.sign(c_pred) == torch.sign(c_true)).float().mean()
    over = (pred_norm > 1.5 * true_norm.clamp_min(1e-12)).float().mean()
    harmful = (pred_err.norm(dim=-1) > base_err.norm(dim=-1)).float().mean()
    return {
        "valid_frames": int(mask.sum()),
        "correction_l2_true": float(true_norm.mean()),
        "correction_l2_pred": float(pred_norm.mean()),
        "correction_ratio": float(pred_norm.mean() / true_norm.mean().clamp_min(1e-12)),
        "correction_error_l2": float((c_pred - c_true).norm(dim=-1).mean()),
        "correction_cosine": float(cos),
        "correction_corr": corr,
        "sign_agreement": float(sign),
        "overcorrection_rate": float(over),
        "harmful_rate": float(harmful),
        "pred_l2": float(pred_err.norm(dim=-1).mean()),
        "base_l2": float(base_err.norm(dim=-1).mean()),
    }


def weighted_average_dict(rows: List[dict], weight_key: str = "valid_frames") -> dict:
    rows = [r for r in rows if int(r.get(weight_key, 0)) > 0]
    if not rows:
        return {}
    total = sum(int(r[weight_key]) for r in rows)
    out = {"num_sequences": len(rows), weight_key: int(total)}
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k != weight_key})
    for key in keys:
        vals = [(float(r[key]), int(r[weight_key])) for r in rows if key in r and math.isfinite(float(r[key]))]
        if vals:
            out[key] = sum(v * w for v, w in vals) / sum(w for _, w in vals)
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([finite_json(row) for row in rows])


def compute_audit(args: argparse.Namespace) -> dict:
    out_root = Path(args.output_root)
    prepare_output(out_root, args.overwrite)
    manifest = load_manifest(Path(args.v4_cache_manifest))
    entries = cache_entries(manifest)
    device = torch.device(args.device)
    model_device = torch.device(args.model_device)
    body_model = art.ParametricModel("models/SMPL_male.pkl", device=device)
    pose_loader = SourcePoseLoader()
    model, norm, ckpt = load_acc_curve_model(Path(args.acc_curve_v3_checkpoint), model_device)

    residual_samples = defaultdict(lambda: defaultdict(lambda: SampleStore(args.sample_frames_per_group, 1234)))
    sensor_rows = []
    seq_signal_rows = []
    rjs_values = defaultdict(list)
    rjs_flat_rows = []
    rjs_collector = defaultdict(RunningRJS)
    rjs_contrib_rows = []
    corr_rows = []
    harmful_rows = []

    started = time.time()
    if args.datasets:
        keep = set(args.datasets)
        entries = [entry for entry in entries if str(entry.get("dataset")) in keep]
    if args.max_records > 0:
        entries = entries[: args.max_records]
    for idx, entry in enumerate(entries, start=1):
        data = load_record(entry)
        meta = data["meta"]
        dataset = str(meta["dataset"])
        split = str(meta["split"])
        name = str(meta["sequence_name"])
        gkeys = [group_key(dataset, split), all_key(dataset), "ALL/all"]
        valid = data["valid_mask"].bool()
        base = data["aIMU_leaf_rel_butter2_4hz"].float()
        target = data["aGT_leaf_rel_butter2_4hz"].float()
        finite = valid & torch.isfinite(base).all(dim=(-1, -2)) & torch.isfinite(target).all(dim=(-1, -2))
        e_base = base - target
        c_true = target - base

        for gk in gkeys:
            residual_samples[gk]["e_base"].add(flatten_leaf_samples(e_base[finite]))
            residual_samples[gk]["c_true"].add(flatten_leaf_samples(c_true[finite]))
            rjs_values[gk].append(data["rJS"].float().numpy())
            rjs_collector[gk].add(data["rJS"], dataset, split, name)

        for sidx, sensor in enumerate(LEAF_SENSOR_NAMES):
            e_s_full = e_base[finite, sidx].numpy()
            c_s_full = c_true[finite, sidx].numpy()
            e_s = sample_np_time(e_s_full, args.sequence_stat_sample_frames)
            c_s = sample_np_time(c_s_full, args.sequence_stat_sample_frames)
            for label, arr in (("e_base", e_s), ("c_true", c_s)):
                row = {
                    "dataset": dataset,
                    "split": split,
                    "group": group_key(dataset, split),
                    "sequence": name,
                    "sensor": sensor,
                    "sensor_index": sidx,
                    "quantity": label,
                    "sequence_stat_sample_frames": int(arr.shape[0]),
                    "high_frequency_energy_ratio": None,
                }
                row.update(distribution_stats(arr))
                for lag in (1, 2, 4, 8):
                    row[f"autocorr_lag{lag}"] = autocorr_lag(arr.reshape(arr.shape[0], -1), lag)
                sensor_rows.append(row)
            base_s = sample_np_time(base[finite, sidx].numpy(), args.sequence_stat_sample_frames)
            target_s = sample_np_time(target[finite, sidx].numpy(), args.sequence_stat_sample_frames)
            cc = best_crosscorr_lag(base_s, target_s, max_lag=10)
            seq_signal_rows.append({
                "dataset": dataset,
                "split": split,
                "sequence": name,
                "sensor": sensor,
                "crosscorr_best_lag": cc["lag"],
                "crosscorr_best_corr": cc["corr"],
            })

        rjs = data["rJS"].float().numpy()
        for sidx, sensor in enumerate(SENSOR_NAMES):
            v = rjs[sidx]
            rjs_flat_rows.append({
                "dataset": dataset,
                "split": split,
                "group": group_key(dataset, split),
                "sequence": name,
                "sensor": sensor,
                "sensor_index": sidx,
                "x": float(v[0]),
                "y": float(v[1]),
                "z": float(v[2]),
                "norm": float(np.linalg.norm(v)),
                "root_part_of_prediction_loss_metric": bool(sidx != ROOT_INDEX),
            })

        if dataset not in ("DIP", "TotalCapture"):
            if args.progress_every and (idx % args.progress_every == 0 or idx == len(entries)):
                elapsed = time.time() - started
                print(f"[{idx:04d}/{len(entries)}] {dataset}/{split} {name} elapsed={elapsed:.1f}s", flush=True)
            continue

        try:
            pose = pose_loader.get(meta["source_file"], name)
            n = min(int(pose.shape[0]), int(base.shape[0]))
            p_joint, p_offset, p_site = fk_joint_and_offset(
                pose[:n], data["rJS"].float(), body_model, device, args.fk_batch_size
            )
            a_joint_rel = leaf_relative(centered_second_difference(p_joint))
            a_offset_rel = leaf_relative(centered_second_difference(p_offset))
            a_site_rel = leaf_relative(centered_second_difference(p_site))
            cmask = finite[:n] & torch.isfinite(a_site_rel).all(dim=(-1, -2)) & torch.isfinite(a_offset_rel).all(dim=(-1, -2)) & torch.isfinite(a_joint_rel).all(dim=(-1, -2))
            if bool(cmask.any()):
                for sidx, sensor in enumerate(LEAF_SENSOR_NAMES):
                    site = a_site_rel[cmask, sidx]
                    off = a_offset_rel[cmask, sidx]
                    joint = a_joint_rel[cmask, sidx]
                    site_l2 = site.norm(dim=-1).mean().clamp_min(1e-12)
                    row = {
                        "dataset": dataset,
                        "split": split,
                        "group": group_key(dataset, split),
                        "sequence": name,
                        "sensor": sensor,
                        "sensor_index": sidx,
                        "valid_frames": int(cmask.sum()),
                        "offset_contribution_ratio": float(off.norm(dim=-1).mean() / site_l2),
                        "joint_contribution_ratio": float(joint.norm(dim=-1).mean() / site_l2),
                        "site_l2": float(site_l2),
                        "offset_l2": float(off.norm(dim=-1).mean()),
                        "joint_l2": float(joint.norm(dim=-1).mean()),
                        "corr_offset_site": corrcoef_np(off.numpy(), site.numpy()),
                        "corr_joint_site": corrcoef_np(joint.numpy(), site.numpy()),
                    }
                    rjs_contrib_rows.append(row)
        except Exception as exc:
            rjs_contrib_rows.append({
                "dataset": dataset,
                "split": split,
                "group": group_key(dataset, split),
                "sequence": name,
                "sensor": "__sequence_error__",
                "valid_frames": 0,
                "error": str(exc),
            })

        pred, decoded_base = predict_record(data, model, norm, model_device)
        target_leaf = data["aGT_leaf_rel_butter2_4hz"].float()
        corr_metrics = correction_metrics(pred, decoded_base, target_leaf, finite)
        corr_metrics.update({"dataset": dataset, "split": split, "group": group_key(dataset, split), "sequence": name})
        corr_rows.append(corr_metrics)
        if corr_metrics.get("valid_frames", 0) > 0:
            harmful_rows.append({
                "dataset": dataset,
                "split": split,
                "sequence": name,
                "valid_frames": corr_metrics["valid_frames"],
                "harmful_rate": corr_metrics["harmful_rate"],
                "overcorrection_rate": corr_metrics["overcorrection_rate"],
                "correction_cosine": corr_metrics["correction_cosine"],
                "correction_corr": corr_metrics["correction_corr"],
                "pred_l2": corr_metrics["pred_l2"],
                "base_l2": corr_metrics["base_l2"],
            })

        if args.progress_every and (idx % args.progress_every == 0 or idx == len(entries)):
            elapsed = time.time() - started
            print(f"[{idx:04d}/{len(entries)}] {dataset}/{split} {name} elapsed={elapsed:.1f}s", flush=True)

    residual_by_group = {}
    group_arrays = {}
    for gk, by_quantity in residual_samples.items():
        residual_by_group[gk] = {}
        group_arrays[gk] = {}
        for quantity, store in by_quantity.items():
            arr = store.tensor()
            group_arrays[gk][quantity] = arr.numpy()
            residual_by_group[gk][quantity] = distribution_stats(arr.reshape(-1, 5, 3).numpy())
            residual_by_group[gk][quantity]["sampled_frames"] = int(arr.shape[0])

    distance = {}
    pair_seen = set()
    for left, right in PRIMARY_DISTANCE_PAIRS:
        lk = group_key(*left) if left[1] != "all" else all_key(left[0])
        rk = group_key(*right) if right[1] != "all" else all_key(right[0])
        pkey = f"{lk}__vs__{rk}"
        if pkey in pair_seen or lk not in group_arrays or rk not in group_arrays:
            continue
        pair_seen.add(pkey)
        distance[pkey] = {}
        for quantity in ("e_base", "c_true"):
            x = group_arrays[lk][quantity]
            y = group_arrays[rk][quantity]
            if x.size == 0 or y.size == 0:
                continue
            mx, my = np.mean(x, axis=0), np.mean(y, axis=0)
            sx, sy = np.std(x, axis=0), np.std(y, axis=0)
            distance[pkey][quantity] = {
                "left_group": lk,
                "right_group": rk,
                "mean_difference_norm": float(np.linalg.norm(mx - my)),
                "std_ratio_mean": float(np.mean((sy + 1e-12) / (sx + 1e-12))),
                "diag_gaussian_frechet": diag_frechet(mx, sx, my, sy),
                "mmd_rbf": rbf_mmd(x, y),
                "wasserstein_per_dim": wasserstein_per_dim(x, y),
                "cosine_of_mean_correction_vector": cosine_np(mx, my),
                "left_samples": int(x.shape[0]),
                "right_samples": int(y.shape[0]),
            }

    rjs_by_group = {gk: rjs_group_stats(vals) for gk, vals in rjs_values.items()}
    dip_train = np.stack(rjs_values.get("DIP/train", []), axis=0).reshape(-1, 18) if rjs_values.get("DIP/train") else None
    tc_train = np.stack(rjs_values.get("TotalCapture/train", []), axis=0).reshape(-1, 18) if rjs_values.get("TotalCapture/train") else None
    refs = {}
    for ref_name, ref in (("DIP_train", dip_train), ("TotalCapture_train", tc_train)):
        if ref is not None and ref.shape[0] >= 2:
            cov = np.cov(ref.T) + np.eye(ref.shape[1]) * 1e-6
            refs[ref_name] = {"mean": ref.mean(axis=0), "inv_cov": np.linalg.pinv(cov)}

    rjs_outliers = []
    for gk, vals in rjs_values.items():
        if not vals:
            continue
        x = np.stack(vals, axis=0).reshape(len(vals), 18)
        for ref_name, ref in refs.items():
            d = mahalanobis(x, ref["mean"], ref["inv_cov"])
            for i, dist in enumerate(d):
                rjs_outliers.append({"group": gk, "ref": ref_name, "sequence_index_in_group": i, "mahalanobis": float(dist)})
        flat = x.reshape(-1)
        mu, sd = float(np.mean(flat)), float(np.std(flat) + 1e-12)
        for i, seq in enumerate(x):
            z = float(np.max(np.abs((seq - mu) / sd)))
            rjs_outliers.append({"group": gk, "ref": "global_scalar_z", "sequence_index_in_group": i, "max_abs_z": z})

    rjs_distance = {}
    group_rjs_flat = {gk: np.stack(vals, axis=0).reshape(len(vals), 18) for gk, vals in rjs_values.items() if vals}
    rjs_pairs = [("DIP/train", "TotalCapture/test"), ("DIP/train", "TotalCapture/train"), ("DIP/all", "TotalCapture/all"), ("DIP/train", "AMASS/train")]
    for lk, rk in rjs_pairs:
        if lk not in group_rjs_flat or rk not in group_rjs_flat:
            continue
        x, y = group_rjs_flat[lk], group_rjs_flat[rk]
        rjs_distance[f"{lk}__vs__{rk}"] = {
            "mean_difference_norm": float(np.linalg.norm(x.mean(axis=0) - y.mean(axis=0))),
            "std_ratio_mean": float(np.mean((y.std(axis=0) + 1e-12) / (x.std(axis=0) + 1e-12))),
            "diag_gaussian_frechet": diag_frechet(x.mean(axis=0), x.std(axis=0), y.mean(axis=0), y.std(axis=0)),
            "mmd_rbf": rbf_mmd(x, y, max_n=5000),
            "cosine_of_mean_rjs": cosine_np(x.mean(axis=0), y.mean(axis=0)),
            "left_sequences": int(x.shape[0]),
            "right_sequences": int(y.shape[0]),
        }

    contrib_by_group = {}
    contrib_by_sensor_rows = []
    for key in sorted({(r.get("dataset"), r.get("split"), r.get("sensor")) for r in rjs_contrib_rows if r.get("valid_frames", 0)}):
        dataset, split, sensor = key
        rows = [r for r in rjs_contrib_rows if r.get("dataset") == dataset and r.get("split") == split and r.get("sensor") == sensor and r.get("valid_frames", 0)]
        agg = weighted_average_dict(rows)
        agg.update({"dataset": dataset, "split": split, "group": group_key(dataset, split), "sensor": sensor})
        contrib_by_sensor_rows.append(agg)
    for gk in sorted({r["group"] for r in contrib_by_sensor_rows}):
        rows = [r for r in contrib_by_sensor_rows if r["group"] == gk]
        contrib_by_group[gk] = weighted_average_dict(rows)

    corr_by_group = {}
    corr_by_sensor_rows = []
    for gk in sorted({r["group"] for r in corr_rows}):
        corr_by_group[gk] = weighted_average_dict([r for r in corr_rows if r["group"] == gk])
    for gk in sorted({r["group"] for r in corr_rows}):
        # Per-sensor correction transfer is recomputed from sequence-level rows only as group-level
        # sequence metrics; exact sensor breakdown is not needed for harmful sequence ranking.
        row = dict(corr_by_group[gk])
        row["group"] = gk
        corr_by_sensor_rows.append(row)

    write_json(out_root / "residual_distribution_by_group.json", residual_by_group)
    write_csv(out_root / "residual_distribution_by_sensor.csv", sensor_rows + seq_signal_rows)
    write_json(out_root / "residual_distribution_distance.json", distance)
    write_json(out_root / "rjs_stats_by_group.json", rjs_by_group)
    write_csv(out_root / "rjs_stats_by_sensor.csv", rjs_flat_rows)
    write_csv(out_root / "rjs_outlier_sequences.csv", sorted(rjs_outliers, key=lambda r: max(float(r.get("mahalanobis", 0.0)), float(r.get("max_abs_z", 0.0))), reverse=True))
    write_json(out_root / "rjs_dataset_distance.json", rjs_distance)
    write_json(out_root / "rjs_acc_contribution_by_group.json", contrib_by_group)
    write_csv(out_root / "rjs_acc_contribution_by_sensor.csv", contrib_by_sensor_rows)
    write_json(out_root / "correction_transfer_by_group.json", corr_by_group)
    write_csv(out_root / "correction_transfer_by_sensor.csv", corr_by_sensor_rows)
    write_csv(out_root / "harmful_sequences.csv", sorted(harmful_rows, key=lambda r: float(r.get("harmful_rate", 0.0)), reverse=True))

    summary_payload = {
        "experiment": EXPERIMENT,
        "v3_recap": V3_RECAP,
        "residual_distance": distance,
        "rjs_distance": rjs_distance,
        "rjs_contribution": contrib_by_group,
        "correction_transfer": corr_by_group,
        "elapsed_sec": time.time() - started,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_selection": ckpt.get("selection_value"),
        "root_index": ROOT_INDEX,
        "leaf_indices": LEAF_INDICES,
        "root_excluded_from_all_residual_correction_metrics": True,
    }
    write_json(out_root / "debug.json", summary_payload)
    write_summary(out_root / "summary.md", summary_payload)
    return summary_payload


def fmt(x, digits=6):
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "n/a"
    return f"{float(x):.{digits}f}"


def top_shift_sensors(sensor_csv_rows: List[dict], dataset="TotalCapture", split="test", quantity="e_base", n=3) -> List[str]:
    rows = [r for r in sensor_csv_rows if r.get("dataset") == dataset and r.get("split") == split and r.get("quantity") == quantity]
    by_sensor = defaultdict(list)
    for row in rows:
        try:
            value = float(row.get("l2", "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            by_sensor[row["sensor"]].append(value)
    ranked = sorted(
        ((sensor, sum(values) / len(values)) for sensor, values in by_sensor.items() if values),
        key=lambda item: item[1],
        reverse=True,
    )
    return [f"{sensor} L2={fmt(value)}" for sensor, value in ranked[:n]]


def decide(summary: dict) -> Tuple[str, str]:
    dist = summary["residual_distance"].get("DIP/train__vs__TotalCapture/test", {})
    e_fd = dist.get("e_base", {}).get("diag_gaussian_frechet", float("nan"))
    c_fd = dist.get("c_true", {}).get("diag_gaussian_frechet", float("nan"))
    rjs_fd = summary["rjs_distance"].get("DIP/train__vs__TotalCapture/test", {}).get("diag_gaussian_frechet", float("nan"))
    tc_corr = summary["correction_transfer"].get("TotalCapture/test", {})
    tc_harm = tc_corr.get("harmful_rate", float("nan"))
    tc_over = tc_corr.get("overcorrection_rate", float("nan"))
    tc_cos = tc_corr.get("correction_cosine", float("nan"))
    dip_corr = summary["correction_transfer"].get("DIP/test", {})
    dip_harm = dip_corr.get("harmful_rate", float("nan"))
    if math.isfinite(tc_harm) and tc_harm > 0.50 and (not math.isfinite(dip_harm) or tc_harm > dip_harm + 0.10):
        return "likely_model_overfit", "use base only for TC-like distribution; then test residual_scale smaller or a sensor-specific residual gate"
    if math.isfinite(rjs_fd) and math.isfinite(e_fd) and rjs_fd > e_fd:
        return "likely_rjs_target_mismatch", "rJS correction / rJS normalization"
    if math.isfinite(e_fd) and math.isfinite(c_fd) and (e_fd > 1.0 or c_fd > 1.0):
        return "likely_error_distribution_shift", "train with DIP+TC mixed real-IMU and add domain-adaptive normalization"
    if math.isfinite(tc_over) and tc_over > 0.30 and math.isfinite(tc_cos) and tc_cos < 0.3:
        return "likely_model_overfit", "residual_scale smaller"
    return "mixed", "train with DIP+TC mixed real-IMU plus sensor-specific residual gate"


def write_summary(path: Path, summary: dict) -> None:
    out_root = path.parent
    sensor_rows = []
    sensor_csv = out_root / "residual_distribution_by_sensor.csv"
    if sensor_csv.exists():
        with sensor_csv.open() as f:
            sensor_rows = list(csv.DictReader(f))
    diagnosis, recommendation = decide(summary)
    dist = summary["residual_distance"].get("DIP/train__vs__TotalCapture/test", {})
    e_dist = dist.get("e_base", {})
    c_dist = dist.get("c_true", {})
    rjs_dist = summary["rjs_distance"].get("DIP/train__vs__TotalCapture/test", {})
    dip_corr = summary["correction_transfer"].get("DIP/test", {})
    tc_corr = summary["correction_transfer"].get("TotalCapture/test", {})
    dip_contrib = summary["rjs_contribution"].get("DIP/test", {})
    tc_contrib = summary["rjs_contribution"].get("TotalCapture/test", {})
    shifted = top_shift_sensors(sensor_rows)

    lines = [
        "# AccCurve v3 Error Distribution and rJS Audit",
        "",
        "## 1. Main Question",
        "",
        "Is AccCurve v3 failing on TotalCapture because IMU acceleration error distributions shift, rJS/offset_r target construction differs, or the learned correction is DIP-specific?",
        "",
        "Contract: root index 5 is reference only and excluded from all residual/correction metrics; leaf indices are 0..4; frame is model/world frame M; IMU base and GT target both use causal Butterworth order=2 cutoff=4Hz.",
        "",
        "## 2. AccCurve v3 Recap",
        "",
        "| Dataset | Pred L2 | Base L2 | Pred/Base L2 | Pred Corr | Base Corr |",
        "|---|---:|---:|---:|---:|---:|",
        f"| DIP test | {fmt(V3_RECAP['DIP_test']['pred_l2'])} | {fmt(V3_RECAP['DIP_test']['base_l2'])} | {fmt(V3_RECAP['DIP_test']['pred_base_l2_ratio'])} | {fmt(V3_RECAP['DIP_test']['pred_corr'])} | {fmt(V3_RECAP['DIP_test']['base_corr'])} |",
        f"| TotalCapture test | {fmt(V3_RECAP['TotalCapture_test']['pred_l2'])} | {fmt(V3_RECAP['TotalCapture_test']['base_l2'])} | {fmt(V3_RECAP['TotalCapture_test']['pred_base_l2_ratio'])} | {fmt(V3_RECAP['TotalCapture_test']['pred_corr'])} | {fmt(V3_RECAP['TotalCapture_test']['base_corr'])} |",
        "",
        "DIP improves, while TotalCapture worsens.",
        "",
        "## 3. Error Distribution Comparison",
        "",
        "| Pair | Quantity | Mean diff norm | Std ratio | Diag Gaussian FD | MMD RBF | Mean-vector cosine |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| DIP train vs TC test | e_base | {fmt(e_dist.get('mean_difference_norm'))} | {fmt(e_dist.get('std_ratio_mean'))} | {fmt(e_dist.get('diag_gaussian_frechet'))} | {fmt(e_dist.get('mmd_rbf'))} | {fmt(e_dist.get('cosine_of_mean_correction_vector'))} |",
        f"| DIP train vs TC test | c_true | {fmt(c_dist.get('mean_difference_norm'))} | {fmt(c_dist.get('std_ratio_mean'))} | {fmt(c_dist.get('diag_gaussian_frechet'))} | {fmt(c_dist.get('mmd_rbf'))} | {fmt(c_dist.get('cosine_of_mean_correction_vector'))} |",
        "",
        "Largest shifted TC test sensors by base residual L2: " + (", ".join(shifted) if shifted else "n/a") + ".",
        "",
        "## 4. rJS Comparison",
        "",
        "| Pair | Mean diff norm | Std ratio | Diag Gaussian FD | MMD RBF | Mean rJS cosine |",
        "|---|---:|---:|---:|---:|---:|",
        f"| DIP train vs TC test | {fmt(rjs_dist.get('mean_difference_norm'))} | {fmt(rjs_dist.get('std_ratio_mean'))} | {fmt(rjs_dist.get('diag_gaussian_frechet'))} | {fmt(rjs_dist.get('mmd_rbf'))} | {fmt(rjs_dist.get('cosine_of_mean_rjs'))} |",
        "",
        "Root/pelvis rJS is audited for offset distribution only; it is not part of AccCurve v3 prediction/loss/metric.",
        "",
        "## 5. rJS Acceleration Contribution",
        "",
        "| Dataset | Split | Offset contribution ratio | Joint contribution ratio | Offset-site corr | Joint-site corr |",
        "|---|---|---:|---:|---:|---:|",
        f"| DIP | test | {fmt(dip_contrib.get('offset_contribution_ratio'))} | {fmt(dip_contrib.get('joint_contribution_ratio'))} | {fmt(dip_contrib.get('corr_offset_site'))} | {fmt(dip_contrib.get('corr_joint_site'))} |",
        f"| TotalCapture | test | {fmt(tc_contrib.get('offset_contribution_ratio'))} | {fmt(tc_contrib.get('joint_contribution_ratio'))} | {fmt(tc_contrib.get('corr_offset_site'))} | {fmt(tc_contrib.get('corr_joint_site'))} |",
        "",
        "## 6. Correction Transfer",
        "",
        "| Dataset | Split | c_pred/c_true ratio | Correction error L2 | Correction cosine | Correction corr | Overcorrection | Harmful rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| DIP | test | {fmt(dip_corr.get('correction_ratio'))} | {fmt(dip_corr.get('correction_error_l2'))} | {fmt(dip_corr.get('correction_cosine'))} | {fmt(dip_corr.get('correction_corr'))} | {fmt(dip_corr.get('overcorrection_rate'))} | {fmt(dip_corr.get('harmful_rate'))} |",
        f"| TotalCapture | test | {fmt(tc_corr.get('correction_ratio'))} | {fmt(tc_corr.get('correction_error_l2'))} | {fmt(tc_corr.get('correction_cosine'))} | {fmt(tc_corr.get('correction_corr'))} | {fmt(tc_corr.get('overcorrection_rate'))} | {fmt(tc_corr.get('harmful_rate'))} |",
        "",
        "## 7. Final Diagnosis",
        "",
        f"`{diagnosis}`",
        "",
        "## 8. Next Recommendation",
        "",
        recommendation,
        "",
        "## 9. Non-Claims",
        "",
        "- This is diagnostic only.",
        "- No PL/NewPL/full-pipeline claim.",
        "- AMASS is synthetic sanity only.",
        "- Root channel is not predicted, supervised, or evaluated as a residual/correction target.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-cache-manifest", required=True)
    parser.add_argument("--acc-curve-v3-checkpoint", required=True)
    parser.add_argument("--output-root", default="data/experiments/acc_curve_v3_error_distribution_rjs_audit_20260618")
    parser.add_argument("--sample-frames-per-group", type=int, default=200000)
    parser.add_argument("--fk-batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-records", type=int, default=0, help="debug only; 0 means all records")
    parser.add_argument("--datasets", nargs="*", default=None, help="debug/filter only; default uses all datasets")
    parser.add_argument("--sequence-stat-sample-frames", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_audit(args)
    print(json.dumps(finite_json({
        "experiment": EXPERIMENT,
        "output_root": args.output_root,
        "elapsed_sec": result["elapsed_sec"],
        "diagnosis_and_recommendation": decide(result),
    }), indent=2))


if __name__ == "__main__":
    main()
