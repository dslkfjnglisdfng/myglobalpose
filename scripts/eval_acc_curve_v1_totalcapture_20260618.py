#!/usr/bin/env python3
"""Evaluate AccCurve v1 on TotalCapture smooth(diff_acc(p_WS)) targets.

This script is intentionally separate from acc_curve_train.py because the
current train/eval entry point is strict v2 GTFK-only. AccCurve v1 uses the
historical target namespace:

  p_WS = p_WJ + R_WJ @ rJS
  smooth(diff_acc(p_WS)) -> aFK_smooth[6,3]

No model training, PL training, IK/VR/full-pipeline evaluation, or S4 metrics
are performed here.
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
from typing import Dict, Iterable, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acc_curve import PLStyleAccCurveModule


DEFAULT_OUTPUT_ROOT = Path("data/experiments/acc_curve_v1_totalcapture_eval_20260618")
DEFAULT_CHECKPOINT = Path("data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt")
DEFAULT_CACHE = Path("code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test/acc_curve_cache_manifest.json")
DEFAULT_EXPERIMENT_NAME = "acc_curve_v1_totalcapture_eval_20260618"

SENSOR_NAMES = ("sensor_0", "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5")
DIP_V1_HISTORICAL = {
    "base_l2": 2.368697,
    "pred_l2": 1.202067,
    "pred_base_ratio": 0.622049,
    "pred_rmse": 0.930242,
    "base_rmse": 1.733464,
    "corr": 0.940837,
}
TC_FULL_TRANS_PREVIOUS = {
    "base_l2": 0.873843,
    "pred_l2": 2.091960,
    "pred_base_ratio": 2.393977,
    "base_rmse": 0.693060,
    "pred_rmse": 1.539445,
    "corr": 0.866428,
}


def load_cache_files(manifest_path: Path) -> Tuple[List[Path], dict]:
    manifest = json.loads(manifest_path.read_text())
    files = [Path(item["path"] if isinstance(item, dict) else item) for item in manifest["cache_files"]]
    return files, manifest


def validate_v1_manifest(manifest_path: Path, manifest: dict) -> None:
    if manifest.get("type") != "acc_curve_cache_v1":
        raise ValueError(f"{manifest_path} type must be acc_curve_cache_v1, got {manifest.get('type')!r}")
    target_layout = str(manifest.get("target_layout", ""))
    if "aFK_smooth[18]" not in target_layout:
        raise ValueError(f"{manifest_path} target_layout must be aFK_smooth[18], got {target_layout!r}")
    coordinate = manifest.get("coordinate_contract", {})
    input_frame = str(coordinate.get("input_frame", ""))
    target_frame = str(coordinate.get("target_frame", ""))
    if "model/world frame M" not in input_frame:
        raise ValueError(f"{manifest_path} input_frame is not model/world frame M: {input_frame!r}")
    valid_target_frame = (
        ("same model/world frame M" in target_frame and "ddot(p_WJ + R_WJ @ r_JS)" in target_frame)
        or ("root translation removed" in target_frame and "p_WS = p_WJ + R_WJ @ rJS" in target_frame)
    )
    if not valid_target_frame:
        raise ValueError(f"{manifest_path} target_frame is not v1 diff-pos target: {target_frame!r}")
    if "GTFK" in json.dumps(manifest):
        raise ValueError(f"{manifest_path} unexpectedly mentions GTFK; this evaluator is v1 diff-pos only.")


def load_records(manifest_path: Path, max_sequences: int = 0) -> Tuple[List[dict], dict]:
    files, manifest = load_cache_files(manifest_path)
    validate_v1_manifest(manifest_path, manifest)
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        required = ("name", "num_frames", "feature", "aM_smooth", "aFK_smooth", "valid_mask")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{cache_file} missing required v1 fields: {missing}")
        for idx, name in enumerate(data["name"]):
            records.append({
                "name": str(name),
                "num_frames": int(data["num_frames"][idx]),
                "feature": data["feature"][idx].float(),
                "base": data["aM_smooth"][idx].float(),
                "target": data["aFK_smooth"][idx].float(),
                "valid_mask": data["valid_mask"][idx].bool(),
            })
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


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
    feature_norm = ckpt["feature_norm"]
    norm = {
        "mean": torch.as_tensor(feature_norm["mean"], dtype=torch.float32, device=device),
        "std": torch.as_tensor(feature_norm["std"], dtype=torch.float32, device=device).clamp_min(1e-6),
        "count": int(feature_norm["count"]),
    }
    if ckpt.get("target_key") and ckpt["target_key"] != "aFK_smooth":
        raise ValueError(f"Unexpected v1 checkpoint target_key: {ckpt.get('target_key')!r}")
    return model, norm, ckpt


@torch.no_grad()
def predict_record(model: PLStyleAccCurveModule, norm: dict, record: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    feature = record["feature"].to(device)
    base = record["base"].to(device)
    feature_norm = (feature - norm["mean"]) / norm["std"]
    output = model.forward_sequence(feature_norm, base)
    return output["pred_aM"].detach().cpu(), output["base"].detach().cpu()


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


def eval_metrics(pred: torch.Tensor, base: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, name: str = "") -> dict:
    valid = valid.bool()
    if not bool(valid.any()):
        return {"name": name, "valid_frames": 0}
    pred_v = pred[valid].reshape(-1, 6, 3)
    base_v = base[valid].reshape(-1, 6, 3)
    target_v = target[valid].reshape(-1, 6, 3)
    pred_err = pred_v - target_v
    base_err = base_v - target_v
    residual = pred_v - base_v
    cosine = torch.nn.functional.cosine_similarity(pred_v.reshape(-1, 3), target_v.reshape(-1, 3), dim=-1, eps=1e-8)
    pred_l2 = pred_err.norm(dim=-1).mean()
    base_l2 = base_err.norm(dim=-1).mean()
    row = {
        "name": name,
        "valid_frames": int(valid.sum()),
        "pred_l2": float(pred_l2),
        "base_l2": float(base_l2),
        "pred_base_ratio": float(pred_l2 / base_l2.clamp_min(1e-12)),
        "pred_rmse": float(pred_err.square().mean().sqrt()),
        "base_rmse": float(base_err.square().mean().sqrt()),
        "corr": corrcoef(pred_v, target_v),
        "cosine": float(cosine.mean()),
        "mag_mae": float((pred_v.norm(dim=-1) - target_v.norm(dim=-1)).abs().mean()),
        "residual_std": float(residual.std()),
        "residual_p95": float(torch.quantile(residual.norm(dim=-1).reshape(-1), 0.95)),
    }
    pred_sensor = pred_err.norm(dim=-1).mean(dim=0)
    base_sensor = base_err.norm(dim=-1).mean(dim=0)
    for idx, sensor in enumerate(SENSOR_NAMES):
        row[f"pred_l2_{sensor}"] = float(pred_sensor[idx])
        row[f"base_l2_{sensor}"] = float(base_sensor[idx])
    pred_axis_mae = pred_err.abs().mean(dim=(0, 1))
    base_axis_mae = base_err.abs().mean(dim=(0, 1))
    pred_axis_rmse = pred_err.square().mean(dim=(0, 1)).sqrt()
    base_axis_rmse = base_err.square().mean(dim=(0, 1)).sqrt()
    for idx, axis in enumerate(("x", "y", "z")):
        row[f"pred_axis_mae_{axis}"] = float(pred_axis_mae[idx])
        row[f"base_axis_mae_{axis}"] = float(base_axis_mae[idx])
        row[f"pred_axis_rmse_{axis}"] = float(pred_axis_rmse[idx])
        row[f"base_axis_rmse_{axis}"] = float(base_axis_rmse[idx])
    return row


def aggregate_from_tensors(preds: List[torch.Tensor], bases: List[torch.Tensor], targets: List[torch.Tensor], masks: List[torch.Tensor]) -> dict:
    pred = torch.cat([p[m] for p, m in zip(preds, masks)], dim=0)
    base = torch.cat([b[m] for b, m in zip(bases, masks)], dim=0)
    target = torch.cat([t[m] for t, m in zip(targets, masks)], dim=0)
    mask = torch.ones(pred.shape[0], dtype=torch.bool)
    metrics = eval_metrics(pred, base, target, mask, name="overall")
    metrics.pop("name", None)
    return metrics


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = [
        "name",
        "valid_frames",
        "pred_l2",
        "base_l2",
        "pred_base_ratio",
        "pred_rmse",
        "base_rmse",
        "corr",
        "cosine",
        "mag_mae",
    ]
    extra = sorted(k for row in rows for k in row if k not in keys)
    fieldnames = keys + extra
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def target_label(manifest: dict) -> tuple[str, str]:
    if manifest.get("force_zero_tran"):
        return "forced zero", "smooth(diff_acc(p_WS_zero_trans))"
    return "source tran", "smooth(diff_acc(trans + p_WS))"


def result_table(aggregate: dict, manifest: dict) -> str:
    target_translation, target = target_label(manifest)
    return "\n".join([
        "| Dataset | Target translation | Target | Acc source | L2 error | RMSE | pred/base ratio | corr | valid frames |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
        (
            f"| TotalCapture test | {target_translation} | {target} | aM_smooth | "
            f"{aggregate['base_l2']:.6f} | {aggregate['base_rmse']:.6f} | 1.000000 | "
            f"{aggregate['base_corr']:.6f} | {int(aggregate['valid_frames'])} |"
        ),
        (
            f"| TotalCapture test | {target_translation} | {target} | AccCurve v1 pred | "
            f"{aggregate['pred_l2']:.6f} | {aggregate['pred_rmse']:.6f} | {aggregate['pred_base_ratio']:.6f} | "
            f"{aggregate['corr']:.6f} | {int(aggregate['valid_frames'])} |"
        ),
    ])


def build_summary(summary: dict) -> str:
    aggregate = summary["aggregate"]
    manifest = summary["manifest"]
    dip = DIP_V1_HISTORICAL
    prev = TC_FULL_TRANS_PREVIOUS
    target_translation, target = target_label(manifest)
    zero_trans = bool(manifest.get("force_zero_tran"))
    if aggregate["pred_base_ratio"] < 1.0:
        verdict = "TC zero-trans pred/base ratio is below 1, so v1's DIP-style zero-trans acceleration target can generalize to TC; the previous full-trans failure is mainly consistent with target translation mismatch."
        recommendation = "Continue considering zero-trans v1 acceleration as a NewPL retrain input candidate, but still gate it with same-cache PL module metrics before any full-pipeline claim."
    else:
        verdict = "TC zero-trans pred/base ratio is still above or equal to 1, so aligning target translation is not enough; v1 residual does not generalize across datasets."
        recommendation = "Do not continue v1 acceleration replacement as a NewPL retrain input without revising the acceleration module or adding a stronger gate/fallback."
    return f"""# {summary['experiment']}

## Purpose

Evaluate AccCurve v1 on TotalCapture test before using v1 acceleration to retrain NewPL. This is acceleration-level evaluation only: no AccCurve training, no PL training, no IK/VR/full-pipeline evaluation, and no S4 metrics.

## Contract

- Experiment root: `{summary['experiment_root']}`
- Cache root: `{summary['cache_root']}`
- Checkpoint: `{summary['checkpoint']}`
- Cache manifest: `{summary['cache_manifest']}`
- Target translation: `{target_translation}`
- Target: `{target}`
- Target key: `aFK_smooth[18]`
- Frame: model/world frame M for input, base, prediction, and target
- This is v1 target namespace, not strict GTFK v2.
- Force zero tran: `{zero_trans}`

## TotalCapture Results

{result_table(aggregate, manifest)}

## Three-Way Comparison

| Eval | Target translation | Target | Base L2 | Pred L2 | Pred/Base | Base RMSE | Pred RMSE | Pred Corr |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DIP historical | zero/none | smooth(diff_acc(p_WS_zero_trans)) | {dip['base_l2']:.6f} | {dip['pred_l2']:.6f} | {dip['pred_base_ratio']:.6f} | {dip['base_rmse']:.6f} | {dip['pred_rmse']:.6f} | {dip['corr']:.6f} |
| TC previous | source tran | smooth(diff_acc(trans + p_WS)) | {prev['base_l2']:.6f} | {prev['pred_l2']:.6f} | {prev['pred_base_ratio']:.6f} | {prev['base_rmse']:.6f} | {prev['pred_rmse']:.6f} | {prev['corr']:.6f} |
| TC zero-trans | forced zero | smooth(diff_acc(p_WS_zero_trans)) | {aggregate['base_l2']:.6f} | {aggregate['pred_l2']:.6f} | {aggregate['pred_base_ratio']:.6f} | {aggregate['base_rmse']:.6f} | {aggregate['pred_rmse']:.6f} | {aggregate['corr']:.6f} |

## Conclusion

- TotalCapture pred/base ratio: `{aggregate['pred_base_ratio']:.6f}`.
- DIP historical pred/base ratio: `{dip['pred_base_ratio']:.6f}`.
- {verdict}
- {recommendation}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-sequences", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    (output_root / "exact_command.txt").write_text(
        f"cd {Path.cwd()}\n"
        f"export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${{LD_LIBRARY_PATH:-}}\n"
        f"{command}\n"
    )
    for path in (args.checkpoint, args.cache):
        if not path.exists():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    print(f"start: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"cache: {args.cache}")
    print(f"output_root: {output_root}")
    records, manifest = load_records(args.cache, max_sequences=args.max_sequences)
    model, norm, ckpt = load_model(args.checkpoint, device)
    rows = []
    preds: List[torch.Tensor] = []
    bases: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    total_frames = 0
    for idx, record in enumerate(records):
        pred, base = predict_record(model, norm, record, device)
        if pred.shape != record["target"].shape or pred.shape[-1] != 18:
            raise AssertionError(f"{record['name']} bad pred shape {tuple(pred.shape)}")
        row = eval_metrics(pred, base, record["target"], record["valid_mask"], record["name"])
        rows.append(row)
        preds.append(pred)
        bases.append(base)
        targets.append(record["target"])
        masks.append(record["valid_mask"])
        total_frames += int(record["num_frames"])
        print(f"[{idx + 1}/{len(records)}] {record['name']}: frames={record['num_frames']} valid={row['valid_frames']} ratio={row['pred_base_ratio']:.6f}")
    aggregate = aggregate_from_tensors(preds, bases, targets, masks)
    base_corr = corrcoef(
        torch.cat([b[m] for b, m in zip(bases, masks)], dim=0),
        torch.cat([t[m] for t, m in zip(targets, masks)], dim=0),
    )
    aggregate["base_corr"] = base_corr
    aggregate["num_sequences"] = len(records)
    aggregate["num_frames"] = int(total_frames)
    result = {
        "experiment": args.experiment_name,
        "experiment_root": str(output_root),
        "cache_root": str(args.cache.parent),
        "checkpoint": str(args.checkpoint),
        "cache_manifest": str(args.cache),
        "target": target_label(manifest)[1],
        "target_translation": target_label(manifest)[0],
        "target_key": "aFK_smooth",
        "target_contract": "p_WS = p_WJ + R_WJ @ rJS; centered smooth window=9 of finite-difference acceleration",
        "frame": "model/world frame M",
        "not_evaluated": ["PL retrain", "IK/VR/full pipeline", "S4"],
        "feature_norm_source": "checkpoint feature_norm fitted during v1 training; no TC fitting",
        "manifest": manifest,
        "checkpoint_selection": ckpt.get("selection_value"),
        "aggregate": aggregate,
        "rows": rows,
        "dip_v1_historical": DIP_V1_HISTORICAL,
        "tc_full_trans_previous": TC_FULL_TRANS_PREVIOUS,
    }
    (output_root / "tc_test_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(output_root / "tc_test_per_sequence.csv", rows)
    (output_root / "summary.md").write_text(build_summary(result))
    print(result_table(aggregate, manifest))
    print(f"result: {output_root / 'tc_test_eval.json'}")
    print(f"end: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
