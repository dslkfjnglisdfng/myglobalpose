#!/usr/bin/env python3
"""Evaluate AccCurve v1 full-trans reconstruction with root acceleration.

This is AccCurve v1 historical target namespace only:

  p_WS_zero = p_WJ + R_WJ @ rJS
  p_WS_full = trans + p_WJ + R_WJ @ rJS

No AccCurve training, PL/NewPL training, IK/VR/full-pipeline evaluation, or S4
claim is made here. The script evaluates acceleration tensors only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acc_curve import PLStyleAccCurveModule
from l4_sensor_offset_utils import FPS, finite_difference_second, smooth_centered


EXPERIMENT_NAME = "acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618"
DEFAULT_ZERO_TRANS_CACHE = Path(
    "code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test/acc_curve_cache_manifest.json"
)
DEFAULT_FULL_TRANS_CACHE = Path(
    "code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test/acc_curve_cache_manifest.json"
)
DEFAULT_SOURCE_CACHE = Path(
    "data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/"
    "baseline_cache_manifest.json"
)
DEFAULT_CHECKPOINT = Path("data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt")
DEFAULT_OUTPUT_ROOT = Path(f"data/experiments/{EXPERIMENT_NAME}")

SENSOR_NAMES = ("left_forearm", "right_forearm", "left_lower_leg", "right_lower_leg", "head", "pelvis")
ALL6 = (0, 1, 2, 3, 4, 5)
LEAF_ONLY = (0, 1, 2, 3, 4)

TC_FULL_TRANS_PREVIOUS = {
    "base_l2": 0.873843,
    "base_rmse": 0.693060,
    "base_corr": 0.974734,
    "pred_l2": 2.091960,
    "pred_rmse": 1.539445,
    "pred_corr": 0.866428,
    "pred_base_ratio": 2.393977,
}
TC_ZERO_TRANS_PREVIOUS = {
    "base_l2": 1.832642,
    "pred_l2": 1.415560,
    "pred_base_ratio": 0.772415,
    "pred_corr": 0.945382,
}


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("type") != "acc_curve_cache_v1":
        raise ValueError(f"{path} is not acc_curve_cache_v1: {manifest.get('type')!r}")
    if manifest.get("feature_dim") != 108 or manifest.get("target_dim") != 18:
        raise ValueError(f"{path} has unexpected dims: {manifest.get('feature_dim')} / {manifest.get('target_dim')}")
    if str(manifest.get("smoothing_mode")) != "centered_moving_average":
        raise ValueError(f"{path} smoothing mode must be centered_moving_average")
    if int(manifest.get("smooth_window", -1)) != 9:
        raise ValueError(f"{path} smooth_window must be 9")
    return manifest


def cache_files(manifest: dict) -> List[Path]:
    return [Path(item["path"] if isinstance(item, dict) else item) for item in manifest["cache_files"]]


def load_cache_records(manifest_path: Path) -> Tuple[Dict[str, dict], dict]:
    manifest = load_manifest(manifest_path)
    records: Dict[str, dict] = {}
    for cache_file in cache_files(manifest):
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        required = ("name", "num_frames", "feature", "aM_smooth", "aFK_smooth", "valid_mask")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{cache_file} missing fields: {missing}")
        for idx, name in enumerate(data["name"]):
            key = str(name)
            records[key] = {
                "name": key,
                "num_frames": int(data["num_frames"][idx]),
                "feature": data["feature"][idx].float(),
                "base": data["aM_smooth"][idx].float().reshape(-1, 6, 3),
                "target": data["aFK_smooth"][idx].float().reshape(-1, 6, 3),
                "valid_mask": data["valid_mask"][idx].bool(),
            }
    return records, manifest


def load_source_trans(source_manifest_path: Path) -> Dict[str, torch.Tensor]:
    manifest = json.loads(source_manifest_path.read_text())
    out: Dict[str, torch.Tensor] = {}
    for item in manifest["cache_files"]:
        cache_file = Path(item["path"] if isinstance(item, dict) else item)
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        if "tran_gt" not in data:
            raise KeyError(f"{cache_file} missing tran_gt")
        for idx, name in enumerate(data["name"]):
            out[str(name)] = data["tran_gt"][idx].float()
    return out


def load_model(checkpoint: Path, device: torch.device) -> Tuple[PLStyleAccCurveModule, dict, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model = PLStyleAccCurveModule(
        hidden_size=int(config.get("hidden_size", 512)),
        residual_scale=float(config.get("residual_scale", 1.0)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm = ckpt["feature_norm"]
    feature_norm = {
        "mean": torch.as_tensor(norm["mean"], dtype=torch.float32, device=device),
        "std": torch.as_tensor(norm["std"], dtype=torch.float32, device=device).clamp_min(1e-6),
        "count": int(norm["count"]),
    }
    if ckpt.get("target_key") and ckpt["target_key"] != "aFK_smooth":
        raise ValueError(f"unexpected target_key in checkpoint: {ckpt.get('target_key')!r}")
    return model, feature_norm, ckpt


@torch.no_grad()
def predict_zero(model: PLStyleAccCurveModule, norm: dict, record: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    feature = record["feature"].to(device)
    base = record["base"].reshape(record["base"].shape[0], 18).to(device)
    feature_norm = (feature - norm["mean"]) / norm["std"]
    out = model.forward_sequence(feature_norm, base)
    return out["pred_aM"].detach().cpu().reshape(-1, 6, 3), out["base"].detach().cpu().reshape(-1, 6, 3)


def root_acc_from_trans(tran: torch.Tensor, window: int, mode: str) -> torch.Tensor:
    raw = finite_difference_second(tran.float(), fps=FPS)
    return smooth_centered(raw, window, mode=mode).float()


def flatten_valid(x: torch.Tensor, valid: torch.Tensor, sensors: Sequence[int]) -> torch.Tensor:
    return x[valid][:, list(sensors), :].reshape(-1, 3)


def corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
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


def candidate_metrics(candidate: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, sensors: Sequence[int]) -> dict:
    cand_v = candidate[valid][:, list(sensors), :]
    targ_v = target[valid][:, list(sensors), :]
    err = cand_v - targ_v
    err_norm = err.norm(dim=-1)
    cosine = torch.nn.functional.cosine_similarity(
        cand_v.reshape(-1, 3), targ_v.reshape(-1, 3), dim=-1, eps=1e-8
    )
    return {
        "valid_frames": int(valid.sum()),
        "l2": float(err_norm.mean()),
        "rmse": float(err.square().mean().sqrt()),
        "mae": float(err.abs().mean()),
        "corr": corrcoef(cand_v, targ_v),
        "cosine": float(cosine.mean()),
        "mag_mae": float((cand_v.norm(dim=-1) - targ_v.norm(dim=-1)).abs().mean()),
        "residual_std": float(err.std()),
        "residual_p95": float(torch.quantile(err_norm.reshape(-1), 0.95)),
    }


def per_sensor_metrics(row_name: str, candidate: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> List[dict]:
    rows = []
    for sensor_idx, sensor_name in enumerate(SENSOR_NAMES):
        m = candidate_metrics(candidate, target, valid, (sensor_idx,))
        m.update({"row": row_name, "sensor": sensor_name, "sensor_idx": sensor_idx})
        rows.append(m)
    return rows


def per_axis_metrics(candidate: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, sensors: Sequence[int]) -> dict:
    err = candidate[valid][:, list(sensors), :] - target[valid][:, list(sensors), :]
    out = {}
    for idx, axis in enumerate(("x", "y", "z")):
        out[f"axis_{axis}_mae"] = float(err[..., idx].abs().mean())
        out[f"axis_{axis}_rmse"] = float(err[..., idx].square().mean().sqrt())
    return out


def aggregate_metric(
    row_name: str,
    candidate_by_name: Dict[str, torch.Tensor],
    target_by_name: Dict[str, torch.Tensor],
    mask_by_name: Dict[str, torch.Tensor],
    sensors: Sequence[int],
    baseline_l2: float | None,
    scope: str,
) -> dict:
    candidates = []
    targets = []
    masks = []
    for name in candidate_by_name:
        valid = mask_by_name[name]
        candidates.append(candidate_by_name[name][valid])
        targets.append(target_by_name[name][valid])
        masks.append(torch.ones(int(valid.sum()), dtype=torch.bool))
    candidate = torch.cat(candidates, dim=0)
    target = torch.cat(targets, dim=0)
    valid = torch.cat(masks, dim=0)
    m = candidate_metrics(candidate, target, valid, sensors)
    m.update(per_axis_metrics(candidate, target, valid, sensors))
    m["row"] = row_name
    m["scope"] = scope
    m["pred_base_ratio"] = float(m["l2"] / baseline_l2) if baseline_l2 and baseline_l2 > 0.0 else None
    return m


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    preferred = [
        "row",
        "scope",
        "name",
        "sensor",
        "sensor_idx",
        "valid_frames",
        "l2",
        "rmse",
        "mae",
        "corr",
        "cosine",
        "mag_mae",
        "pred_base_ratio",
        "residual_std",
        "residual_p95",
    ]
    keys = preferred + sorted({k for row in rows for k in row if k not in preferred})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sanity_stats(errors: List[torch.Tensor], valid_masks: List[torch.Tensor]) -> dict:
    vals = torch.cat([e[v].reshape(-1) for e, v in zip(errors, valid_masks)], dim=0)
    return {
        "max_abs": float(vals.abs().max()),
        "mean_abs": float(vals.abs().mean()),
        "rmse": float(vals.square().mean().sqrt()),
    }


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    if overwrite and path.exists():
        for child in path.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    (path / "logs").mkdir(parents=True, exist_ok=True)


def build_summary(result: dict) -> str:
    rows = {(r["row"], r["scope"]): r for r in result["main_rows"]}
    baseline = rows[("baseline_full", "all6")]
    wrong = rows[("wrong_pred_zero_vs_full", "all6")]
    corrected = rows[("correct_pred_zero_plus_gt_root_trans", "all6")]
    optional = rows.get(("optional_pred_zero_plus_imu_root_est", "all6"))
    zero = rows[("zero_trans_sanity", "all6")]
    sanity = result["decomposition_sanity"]["cache_consistent_root"]
    beats = corrected["l2"] < baseline["l2"]
    verdict = (
        "AccCurve v1 was not actually failing TotalCapture acceleration generalization under the "
        "full-trans target once GT root translational acceleration is added back; the old full-trans "
        "failure was target-mismatched."
        if beats
        else "Target mismatch explains part but not all of the old full-trans failure; the corrected "
        "prediction still does not beat the aM_smooth full-trans baseline."
    )
    optional_line = ""
    if optional:
        optional_line = (
            f"| optional_pred_zero_plus_imu_root_est | {optional['l2']:.6f} | {optional['rmse']:.6f} | "
            f"{optional['mae']:.6f} | {optional['corr']:.6f} | {optional['cosine']:.6f} | "
            f"{optional['mag_mae']:.6f} | {optional['pred_base_ratio']:.6f} |\n"
        )
    return f"""# {EXPERIMENT_NAME}

## Main Question

Was the old TotalCapture full-trans failure caused by comparing the zero-trans AccCurve v1 prediction against full-trans ground truth?

## Historical Recap

- DIP trained target was effectively zero-trans: `smooth(diff_acc(p_WJ + R_WJ @ rJS))`.
- Old TC full-trans eval directly compared `pred_zero` against `GT_full` and got pred/base ratio `{TC_FULL_TRANS_PREVIOUS['pred_base_ratio']:.6f}`.
- TC zero-trans eval got pred/base ratio `{TC_ZERO_TRANS_PREVIOUS['pred_base_ratio']:.6f}`.

## Definitions

- `pred_zero`: AccCurve v1 prediction from the zero-trans v1 cache.
- `a_root_trans_smooth`: cache-consistent `GT_full - GT_zero`, checked against `smooth(diff_acc(tran_gt))`.
- `pred_full_reconstructed = pred_zero + a_root_trans_smooth`.
- `GT_full`: full-trans `aFK_smooth` from the TotalCapture v1 cache.
- `aM_smooth baseline`: AccCurve v1 spline-decoded base stream from `aM_smooth`, matching the historical v1 evaluator.

## Decomposition Sanity

| Check | max abs | mean abs | RMSE |
|---|---:|---:|---:|
| `GT_full - (GT_zero + cache_root_acc)` | {sanity['max_abs']:.9f} | {sanity['mean_abs']:.9f} | {sanity['rmse']:.9f} |
| `cache_root_acc - smooth(diff_acc(tran_gt))` | {result['decomposition_sanity']['source_tran_vs_cache_root']['max_abs']:.9f} | {result['decomposition_sanity']['source_tran_vs_cache_root']['mean_abs']:.9f} | {result['decomposition_sanity']['source_tran_vs_cache_root']['rmse']:.9f} |

## Main Table

All 6 sensors are primary for v1 historical compatibility.

| Row | L2 | RMSE | MAE | Corr | Cosine | Mag MAE | Pred/Base ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_full | {baseline['l2']:.6f} | {baseline['rmse']:.6f} | {baseline['mae']:.6f} | {baseline['corr']:.6f} | {baseline['cosine']:.6f} | {baseline['mag_mae']:.6f} | {baseline['pred_base_ratio']:.6f} |
| wrong_pred_zero_vs_full | {wrong['l2']:.6f} | {wrong['rmse']:.6f} | {wrong['mae']:.6f} | {wrong['corr']:.6f} | {wrong['cosine']:.6f} | {wrong['mag_mae']:.6f} | {wrong['pred_base_ratio']:.6f} |
| correct_pred_zero_plus_gt_root_trans | {corrected['l2']:.6f} | {corrected['rmse']:.6f} | {corrected['mae']:.6f} | {corrected['corr']:.6f} | {corrected['cosine']:.6f} | {corrected['mag_mae']:.6f} | {corrected['pred_base_ratio']:.6f} |
{optional_line}| zero_trans_sanity | {zero['l2']:.6f} | {zero['rmse']:.6f} | {zero['mae']:.6f} | {zero['corr']:.6f} | {zero['cosine']:.6f} | {zero['mag_mae']:.6f} | {zero['pred_base_ratio']:.6f} |

## Interpretation

- Corrected prediction beats baseline_full: `{beats}`.
- {verdict}

## Current Project Implication

AccCurve v1 should be described as a root-translation-free sensor-site acceleration predictor, not a full absolute acceleration predictor. Future target construction should explicitly define whether root translational acceleration is included. For current leaf-relative work, use root-reference/translation-free formulation deliberately.

## Non-Claims

- No PL/NewPL/full-pipeline claim.
- No S4 metrics.
- No retraining.
- This is a historical AccCurve v1 evaluation correction only.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-trans-cache", type=Path, default=DEFAULT_ZERO_TRANS_CACHE)
    parser.add_argument("--full-trans-cache", type=Path, default=DEFAULT_FULL_TRANS_CACHE)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output(args.output_root, args.overwrite)
    command = " ".join([sys.executable, *sys.argv])
    (args.output_root / "exact_command.txt").write_text(
        f"cd {Path.cwd()}\n"
        f"export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${{LD_LIBRARY_PATH:-}}\n"
        f"{command}\n"
    )
    for path in (args.zero_trans_cache, args.full_trans_cache, args.source_cache, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"start: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"experiment: {EXPERIMENT_NAME}")
    print(f"output_root: {args.output_root}")

    zero_records, zero_manifest = load_cache_records(args.zero_trans_cache)
    full_records, full_manifest = load_cache_records(args.full_trans_cache)
    source_trans = load_source_trans(args.source_cache)
    names = list(zero_records)
    if names != list(full_records):
        raise ValueError("zero/full cache record order differs")
    missing_trans = [name for name in names if name not in source_trans]
    if missing_trans:
        raise KeyError(f"missing source tran_gt for {missing_trans}")

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model, norm, ckpt = load_model(args.checkpoint, device)

    feature_diffs = []
    base_diffs = []
    valid_mask_diffs = []
    pred_zero_by_name: Dict[str, torch.Tensor] = {}
    base_zero_decoded_by_name: Dict[str, torch.Tensor] = {}
    base_full_by_name: Dict[str, torch.Tensor] = {}
    target_full_by_name: Dict[str, torch.Tensor] = {}
    target_zero_by_name: Dict[str, torch.Tensor] = {}
    root_cache_by_name: Dict[str, torch.Tensor] = {}
    root_source_by_name: Dict[str, torch.Tensor] = {}
    valid_full_by_name: Dict[str, torch.Tensor] = {}
    valid_zero_by_name: Dict[str, torch.Tensor] = {}
    per_sequence_rows = []
    per_sensor_rows = []

    for idx, name in enumerate(names):
        z = zero_records[name]
        f = full_records[name]
        if z["num_frames"] != f["num_frames"]:
            raise ValueError(f"{name} frame mismatch: {z['num_frames']} vs {f['num_frames']}")
        feature_diffs.append((z["feature"] - f["feature"]).abs().max())
        base_diffs.append((z["base"] - f["base"]).abs().max())
        valid_mask_diffs.append((z["valid_mask"] ^ f["valid_mask"]).sum())
        pred_zero, base_decoded = predict_zero(model, norm, z, device)
        base_eval = base_decoded
        root_cache = f["target"] - z["target"]
        root_source = root_acc_from_trans(
            source_trans[name],
            int(full_manifest["smooth_window"]),
            str(full_manifest["smoothing_mode"]),
        )
        pred_zero_by_name[name] = pred_zero
        base_zero_decoded_by_name[name] = base_decoded
        base_full_by_name[name] = f["base"]
        target_full_by_name[name] = f["target"]
        target_zero_by_name[name] = z["target"]
        root_cache_by_name[name] = root_cache[:, 0, :]
        root_source_by_name[name] = root_source
        valid_full_by_name[name] = f["valid_mask"]
        valid_zero_by_name[name] = z["valid_mask"]
        valid_full = f["valid_mask"]
        full_recon = pred_zero + root_cache[:, :1, :]
        imu_root_est = base_eval[:, 5:6, :] - pred_zero[:, 5:6, :]
        pred_imuroot = pred_zero + imu_root_est
        seq_candidates = {
            "baseline_full": (base_eval, f["target"], valid_full, None),
            "wrong_pred_zero_vs_full": (pred_zero, f["target"], valid_full, None),
            "correct_pred_zero_plus_gt_root_trans": (full_recon, f["target"], valid_full, None),
            "optional_pred_zero_plus_imu_root_est": (pred_imuroot, f["target"], valid_full, None),
            "zero_trans_sanity": (pred_zero, z["target"], z["valid_mask"], None),
        }
        base_full_l2 = candidate_metrics(base_eval, f["target"], valid_full, ALL6)["l2"]
        base_zero_l2 = candidate_metrics(base_eval, z["target"], z["valid_mask"], ALL6)["l2"]
        for row_name, (candidate, target, valid, _) in seq_candidates.items():
            baseline_l2 = base_zero_l2 if row_name == "zero_trans_sanity" else base_full_l2
            for scope, sensors in (("all6", ALL6), ("leaf_only_0_4", LEAF_ONLY)):
                m = candidate_metrics(candidate, target, valid, sensors)
                m.update(per_axis_metrics(candidate, target, valid, sensors))
                m.update({
                    "row": row_name,
                    "scope": scope,
                    "name": name,
                    "pred_base_ratio": float(m["l2"] / baseline_l2),
                })
                per_sequence_rows.append(m)
            if row_name in (
                "baseline_full",
                "wrong_pred_zero_vs_full",
                "correct_pred_zero_plus_gt_root_trans",
                "optional_pred_zero_plus_imu_root_est",
                "zero_trans_sanity",
            ):
                per_sensor_rows.extend(per_sensor_metrics(row_name, candidate, target, valid))
        print(json.dumps({"idx": idx + 1, "count": len(names), "name": name, "valid": int(valid_full.sum())}))

    pred_full_gtroot_by_name = {
        name: pred_zero_by_name[name] + root_cache_by_name[name][:, None, :] for name in names
    }
    pred_full_imuroot_by_name = {
        name: pred_zero_by_name[name] + (base_zero_decoded_by_name[name][:, 5:6, :] - pred_zero_by_name[name][:, 5:6, :])
        for name in names
    }

    baseline_full_l2_all = aggregate_metric(
        "baseline_full", base_zero_decoded_by_name, target_full_by_name, valid_full_by_name, ALL6, 1.0, "all6"
    )["l2"]
    baseline_full_l2_leaf = aggregate_metric(
        "baseline_full", base_zero_decoded_by_name, target_full_by_name, valid_full_by_name, LEAF_ONLY, 1.0, "leaf_only_0_4"
    )["l2"]
    baseline_zero_l2_all = aggregate_metric(
        "baseline_zero", base_zero_decoded_by_name, target_zero_by_name, valid_zero_by_name, ALL6, 1.0, "all6"
    )["l2"]
    baseline_zero_l2_leaf = aggregate_metric(
        "baseline_zero", base_zero_decoded_by_name, target_zero_by_name, valid_zero_by_name, LEAF_ONLY, 1.0, "leaf_only_0_4"
    )["l2"]

    main_rows = []
    for scope, sensors, full_base_l2, zero_base_l2 in (
        ("all6", ALL6, baseline_full_l2_all, baseline_zero_l2_all),
        ("leaf_only_0_4", LEAF_ONLY, baseline_full_l2_leaf, baseline_zero_l2_leaf),
    ):
        main_rows.append(aggregate_metric("baseline_full", base_zero_decoded_by_name, target_full_by_name, valid_full_by_name, sensors, full_base_l2, scope))
        main_rows.append(aggregate_metric("wrong_pred_zero_vs_full", pred_zero_by_name, target_full_by_name, valid_full_by_name, sensors, full_base_l2, scope))
        main_rows.append(aggregate_metric("correct_pred_zero_plus_gt_root_trans", pred_full_gtroot_by_name, target_full_by_name, valid_full_by_name, sensors, full_base_l2, scope))
        main_rows.append(aggregate_metric("optional_pred_zero_plus_imu_root_est", pred_full_imuroot_by_name, target_full_by_name, valid_full_by_name, sensors, full_base_l2, scope))
        main_rows.append(aggregate_metric("zero_trans_sanity", pred_zero_by_name, target_zero_by_name, valid_zero_by_name, sensors, zero_base_l2, scope))

    cache_root_errors = [
        target_full_by_name[n] - (target_zero_by_name[n] + root_cache_by_name[n][:, None, :]) for n in names
    ]
    source_root_errors = [
        root_cache_by_name[n] - root_source_by_name[n] for n in names
    ]
    valid_masks = [valid_full_by_name[n] for n in names]
    decomposition_sanity = {
        "cache_consistent_root": sanity_stats(cache_root_errors, valid_masks),
        "source_tran_vs_cache_root": sanity_stats(source_root_errors, valid_masks),
        "max_abs_feature_zero_minus_full": float(torch.stack(feature_diffs).max()),
        "max_abs_aM_smooth_zero_minus_full": float(torch.stack(base_diffs).max()),
        "valid_mask_xor_frames": int(torch.stack([x.to(torch.int64) for x in valid_mask_diffs]).sum()),
        "root_acc_primary_for_correct_eval": "cache_consistent_gt_root = aFK_full_smooth - aFK_zero_smooth",
        "source_tran_root_acc_contract": "finite_difference_second(tran_gt, fps=60) then centered_moving_average window=9",
    }

    result = {
        "experiment": EXPERIMENT_NAME,
        "output_root": str(args.output_root),
        "checkpoint": str(args.checkpoint),
        "zero_trans_cache": str(args.zero_trans_cache),
        "full_trans_cache": str(args.full_trans_cache),
        "source_cache": str(args.source_cache),
        "contracts": {
            "namespace": "AccCurve v1 historical target namespace",
            "zero_trans_target": "smooth(diff_acc(p_WJ + R_WJ @ rJS))",
            "full_trans_target": "smooth(diff_acc(trans + p_WJ + R_WJ @ rJS))",
            "corrected_eval": "pred_zero + a_root_trans_smooth vs GT_full",
            "non_claims": ["no training", "no PL/NewPL/full-pipeline claim", "no S4 metrics"],
        },
        "checkpoint_selection": ckpt.get("selection_value"),
        "feature_norm_count": norm["count"],
        "num_sequences": len(names),
        "main_rows": main_rows,
        "decomposition_sanity": decomposition_sanity,
        "tc_full_trans_previous": TC_FULL_TRANS_PREVIOUS,
        "tc_zero_trans_previous": TC_ZERO_TRANS_PREVIOUS,
    }

    (args.output_root / "fulltrans_reconstruction_eval.json").write_text(json.dumps(jsonable(result), indent=2) + "\n")
    write_csv(args.output_root / "fulltrans_reconstruction_eval.csv", main_rows)
    write_csv(args.output_root / "per_sequence_metrics.csv", per_sequence_rows)
    write_csv(args.output_root / "per_sensor_metrics.csv", per_sensor_rows)
    (args.output_root / "decomposition_sanity.json").write_text(json.dumps(jsonable(decomposition_sanity), indent=2) + "\n")
    (args.output_root / "summary.md").write_text(build_summary(result))
    print(json.dumps(jsonable({"main_rows": main_rows, "decomposition_sanity": decomposition_sanity}), indent=2))
    print(f"summary: {args.output_root / 'summary.md'}")
    print(f"end: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
