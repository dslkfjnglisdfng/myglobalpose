#!/usr/bin/env python3
"""Evaluate AccCurve predictions as frozen official PL acceleration input.

This is an evaluation-only experiment. It never trains PL and only replaces the
first 18D acceleration block of the legacy 84D PL input:

  aRB[18] + wRB[18] + RRB[45] + gR0[3].

AccCurve predictions are model/world-frame accelerations. They are converted to
the PL root frame with the original GlobalPose contract before PL forward:

  aRB = acc_M @ RMB_root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import articulate as art
from acc_curve import PLStyleAccCurveModule
from l4_train_diverse_short import load_records
from pl_curve import pl_input_feature, pl_target_from_pose


DEFAULT_ROOT = Path("data/experiments/acc_curve_pl_input_eval_20260617")
DEFAULT_DIP_TEST_CACHE = Path(
    "data/experiments/newpl_v5_official_protocol_20260607/caches/"
    "dip_test_with_offset_r/baseline_cache_manifest.json"
)
DEFAULT_PL_WEIGHTS = Path("data/weights.pt")
DEFAULT_V1_CKPT = Path("data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt")
DEFAULT_V2_CKPT = Path(
    "data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/"
    "dip_finetune/best_loss.pt"
)
DEFAULT_V1_CACHE = Path(
    "code/outputs/smooth_acc_cache_amass_dip_20260617/"
    "dip_test/acc_curve_cache_manifest.json"
)
DEFAULT_V2_CACHE = Path(
    "code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/"
    "dip_test/acc_curve_gtfk_cache_manifest.json"
)

VARIANTS = (
    ("official_raw_acc", "raw aM", "none"),
    ("smooth_acc", "smooth(aM)", "none"),
    ("acc_curve_v1_pred", "AccCurve v1 pred", "smooth(diff_acc(p_WS))"),
    ("acc_curve_v2_gtfk_pred", "AccCurve v2 pred", "smooth(GTFKacc(q,qdot,qddot,rJS))"),
)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def load_acc_curve_model(checkpoint_path: Path, device: torch.device) -> Tuple[PLStyleAccCurveModule, Dict[str, torch.Tensor], dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = PLStyleAccCurveModule(
        hidden_size=int(config.get("hidden_size", 512)),
        dropout=float(config.get("dropout", 0.1)),
        residual_scale=float(config.get("residual_scale", 1.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    feature_norm = checkpoint["feature_norm"]
    norm = {
        "mean": torch.as_tensor(feature_norm["mean"], dtype=torch.float32, device=device),
        "std": torch.as_tensor(feature_norm["std"], dtype=torch.float32, device=device).clamp_min(1e-6),
    }
    return model, norm, checkpoint


def iter_cache_files(manifest_path: Path) -> Tuple[List[Path], dict]:
    manifest = json.loads(manifest_path.read_text())
    files = []
    for item in manifest["cache_files"]:
        path = Path(item["path"] if isinstance(item, dict) else item)
        files.append(path)
    return files, manifest


@torch.no_grad()
def load_acc_curve_predictions(
    manifest_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    max_sequences: int = 0,
) -> Tuple[Dict[str, dict], dict, dict]:
    model, norm, checkpoint = load_acc_curve_model(checkpoint_path, device)
    files, manifest = iter_cache_files(manifest_path)
    predictions: Dict[str, dict] = {}
    for cache_file in files:
        data = torch.load(cache_file, map_location="cpu")
        for idx, name in enumerate(data["name"]):
            if max_sequences and len(predictions) >= max_sequences:
                return predictions, manifest, checkpoint
            feature = data["feature"][idx].float().to(device)
            base = data["aM_smooth"][idx].float().to(device)
            feature_norm = (feature - norm["mean"]) / norm["std"]
            output = model.forward_sequence(feature_norm, base)
            pred = output["pred_aM"].detach().cpu().float().reshape(-1, 6, 3)
            if pred.dim() != 3 or pred.shape[1:] != (6, 3):
                raise AssertionError(f"{checkpoint_path} prediction for {name} has shape {tuple(pred.shape)}")
            predictions[name] = {
                "pred": pred,
                "aM_raw": data["aM_raw"][idx].float().reshape(-1, 6, 3),
                "aM_smooth": data["aM_smooth"][idx].float().reshape(-1, 6, 3),
                "feature": data["feature"][idx].float(),
                "valid_mask": data["valid_mask"][idx].bool(),
            }
    return predictions, manifest, checkpoint


def build_pl_features(acc_m: torch.Tensor, w_m: torch.Tensor, r_mb: torch.Tensor) -> torch.Tensor:
    if acc_m.dim() == 2:
        acc_m = acc_m.reshape(-1, 6, 3)
    if w_m.dim() == 2:
        w_m = w_m.reshape(-1, 6, 3)
    root = r_mb[:, 5]
    a_rb = acc_m.matmul(root)
    w_rb = w_m.matmul(root)
    r_rb = root.transpose(1, 2).unsqueeze(1).matmul(r_mb[:, :5])
    g_r0 = -r_mb[:, 5, 1]
    return torch.cat((a_rb.reshape(acc_m.shape[0], 18), w_rb.reshape(acc_m.shape[0], 18), r_rb.reshape(acc_m.shape[0], 45), g_r0), dim=-1).float()


def check_official_feature_equivalence(record: dict, vectorized: torch.Tensor, max_frames: int = 0) -> float:
    limit = vectorized.shape[0] if not max_frames else min(vectorized.shape[0], max_frames)
    legacy = torch.stack([
        pl_input_feature(record["aM"][i], record["wM"][i], record["RMB"][i]).float()
        for i in range(limit)
    ])
    return float((legacy - vectorized[:limit]).abs().max().item())


class OfficialPLRunner(torch.nn.Module):
    def __init__(self, weights_path: Path, device: torch.device):
        super().__init__()
        from articulate.utils.torch import RNNWithInit

        self.plnet = RNNWithInit(
            input_linear=False,
            input_size=84,
            output_size=18,
            hidden_size=512,
            num_rnn_layer=3,
            dropout=0.4,
        )
        weights = torch.load(weights_path, map_location="cpu")
        pl_state = {key[len("plnet.") :]: value for key, value in weights.items() if key.startswith("plnet.")}
        missing, unexpected = self.plnet.load_state_dict(pl_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"PL state mismatch: missing={missing}, unexpected={unexpected}")
        self.body_model = art.ParametricModel("models/SMPL_male.pkl", vert_mask=(1961, 5424, 1176, 4662, 411, 3021))
        self.to(device)
        self.device = device
        self.pl1hc = None
        self.eval()

    @torch.no_grad()
    def rnn_initialize(self, init_pose: torch.Tensor) -> None:
        init_pose = init_pose.detach().cpu().view(1, 24, 3, 3)
        _, _, verts = self.body_model.forward_kinematics(init_pose, calc_mesh=True)
        p_rl = (verts[0, :5] - verts[0, 5:]).mm(init_pose[0, 0]).reshape(-1)
        g_r = -init_pose[0, 0, :, 1]
        x1 = torch.cat((p_rl, g_r), dim=0).to(self.device)
        self.pl1hc = [
            hidden.contiguous()
            for hidden in self.plnet.init_net(x1)
            .view(1, 2, self.plnet.num_layers, self.plnet.hidden_size)
            .permute(1, 2, 0, 3)
        ]

    @torch.no_grad()
    def forward_sequence(self, features: torch.Tensor, init_pose: torch.Tensor) -> torch.Tensor:
        self.rnn_initialize(init_pose)
        outputs = []
        for frame in features.to(self.device):
            x, self.pl1hc = self.plnet.rnn(frame.view(1, 1, -1), self.pl1hc)
            pl_out = self.plnet.linear2(x.squeeze())
            pl_out = torch.cat((pl_out[:15], art.math.normalize_tensor(pl_out[15:])), dim=0)
            outputs.append(pl_out.detach().cpu())
        return torch.stack(outputs, dim=0)


def sequence_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    pred = pred[mask]
    target = target[mask]
    if pred.numel() == 0:
        raise ValueError("No valid frames for metrics.")
    p_err = pred[:, :15].reshape(-1, 5, 3) - target[:, :15].reshape(-1, 5, 3)
    pred_g = art.math.normalize_tensor(pred[:, 15:])
    target_g = art.math.normalize_tensor(target[:, 15:])
    dot = (pred_g * target_g).sum(dim=-1).clamp(-1.0, 1.0)
    return {
        "valid_frames": int(mask.sum().item()),
        "pRB_l2_cm": float(p_err.norm(dim=-1).mean().item() * 100.0),
        "pRB_rmse_cm": float(torch.sqrt((p_err.square()).mean()).item() * 100.0),
        "pRB_mae_cm": float(p_err.abs().mean().item() * 100.0),
        "gR1_deg": float(torch.rad2deg(torch.acos(dot)).mean().item()),
        "gR1_cos_loss": float((1.0 - dot).mean().item()),
    }


def aggregate_metrics(preds: List[torch.Tensor], targets: List[torch.Tensor], masks: List[torch.Tensor]) -> dict:
    pred = torch.cat([p[m] for p, m in zip(preds, masks)], dim=0)
    target = torch.cat([t[m] for t, m in zip(targets, masks)], dim=0)
    mask = torch.ones(pred.shape[0], dtype=torch.bool)
    metrics = sequence_metrics(pred, target, mask)
    metrics["num_sequences"] = len(preds)
    metrics["num_frames"] = int(sum(p.shape[0] for p in preds))
    return metrics


def accel_debug_stats(name: str, blocks: Dict[str, torch.Tensor]) -> dict:
    out = {"sequence": name, "frame": "PL root-frame acceleration block aRB[18]"}
    official = blocks["official_raw_acc"]
    smooth = blocks["smooth_acc"]
    for variant, block in blocks.items():
        out[variant] = {
            "mean": float(block.mean().item()),
            "std": float(block.std(unbiased=False).item()),
            "rmse_vs_official_raw_acc": float(torch.sqrt((block - official).square().mean()).item()),
            "rmse_vs_smooth_acc": float(torch.sqrt((block - smooth).square().mean()).item()),
        }
    return out


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_result_table(rows: List[dict]) -> str:
    lines = [
        "| Variant | Acc source | Target used by AccCurve | PL pRB L2 cm | PL pRB RMSE cm | PL gR1 deg | valid frames |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['acc_source']} | {row['target_used_by_acc_curve']} | "
            f"{row['pRB_l2_cm']:.6f} | {row['pRB_rmse_cm']:.6f} | {row['gR1_deg']:.6f} | "
            f"{int(row['valid_frames'])} |"
        )
    return "\n".join(lines)


def build_summary_md(summary: dict, rows: List[dict]) -> str:
    official = next(row for row in rows if row["variant"] == "official_raw_acc")

    def improvement_line(variant: str) -> str:
        row = next(row for row in rows if row["variant"] == variant)
        p_better = row["pRB_l2_cm"] < official["pRB_l2_cm"]
        g_better = row["gR1_deg"] < official["gR1_deg"]
        return (
            f"- `{variant}`: pRB {'improves' if p_better else 'does not improve'} "
            f"({row['pRB_l2_cm']:.6f} vs {official['pRB_l2_cm']:.6f} cm); "
            f"gR1 {'improves' if g_better else 'does not improve'} "
            f"({row['gR1_deg']:.6f} vs {official['gR1_deg']:.6f} deg)."
        )

    transfer = (
        "At least one AccCurve variant improved both PL pRB and gR1 against official raw acceleration."
        if any(
            row["variant"].startswith("acc_curve")
            and row["pRB_l2_cm"] < official["pRB_l2_cm"]
            and row["gR1_deg"] < official["gR1_deg"]
            for row in rows
        )
        else "Acceleration-level improvement did not transfer into a simultaneous PL pRB and gR1 improvement against official raw acceleration."
    )
    return f"""# EXP-20260617-acc_curve_pl_input_eval

## Purpose

Evaluate whether AccCurve v1/v2 acceleration predictions help the frozen official baseline PL module when used only as the PL acceleration input. No PL weights are trained or modified.

## Protocol

- Experiment root: `{summary['experiment_root']}`
- Evaluator: `{summary['evaluator']}`
- Frozen baseline PL checkpoint: `{summary['baseline_pl_checkpoint']}` (`GPNet.plnet` weights only)
- DIP test cache/protocol: `{summary['dip_test_cache']}`
- AccCurve v1 checkpoint: `{summary['acc_curve_v1_checkpoint']}`
- AccCurve v2 checkpoint: `{summary['acc_curve_v2_checkpoint']}`
- AccCurve v1 target: `smooth(diff_acc(p_WS))`
- AccCurve v2 target: `smooth(GTFKacc(q,qdot,qddot,rJS))`

Only the first 18D acceleration block of the legacy PL feature is replaced. The other 66D (`wRB[18] + RRB[45] + gR0[3]`) are asserted identical across variants. AccCurve outputs are model/world-frame accelerations and are converted to PL root frame with `aRB = acc_M @ RMB_root` before PL forward.

DIP test is used only for evaluation. It is not used for training, normalization fitting, or checkpoint selection.

## Validation

- Official raw vectorized 84D feature vs `pl_input_feature`: max abs diff `{summary['assertions']['official_feature_max_abs_diff']:.8g}`.
- Non-acceleration 66D feature block max abs diff across variants: `{summary['assertions']['non_acc_block_max_abs_diff']:.8g}`.
- AccCurve v1/v2 predictions were asserted as `[T,6,3]` for every evaluated sequence.
- Debug acceleration stats: `{summary['debug_json']}`.

## Results

{format_result_table(rows)}

## Conclusion

{improvement_line('smooth_acc')}
{improvement_line('acc_curve_v1_pred')}
{improvement_line('acc_curve_v2_gtfk_pred')}

{transfer}

This is a standalone PL module-input evaluation. It does not claim full-pipeline motion quality improvement and does not mix v1/v2 acceleration-level RMSE target namespaces.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dip-test-cache", type=Path, default=DEFAULT_DIP_TEST_CACHE)
    parser.add_argument("--baseline-pl-weights", type=Path, default=DEFAULT_PL_WEIGHTS)
    parser.add_argument("--acc-curve-v1-checkpoint", type=Path, default=DEFAULT_V1_CKPT)
    parser.add_argument("--acc-curve-v2-checkpoint", type=Path, default=DEFAULT_V2_CKPT)
    parser.add_argument("--acc-curve-v1-cache", type=Path, default=DEFAULT_V1_CACHE)
    parser.add_argument("--acc-curve-v2-cache", type=Path, default=DEFAULT_V2_CACHE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--feature-check-max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    command = " ".join([os.environ.get("_", sys.executable), *sys.argv])
    (output_root / "exact_command.txt").write_text(
        f"cd {Path.cwd()}\n"
        f"export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${{LD_LIBRARY_PATH:-}}\n"
        f"{command}\n"
    )

    for path in (
        args.dip_test_cache,
        args.baseline_pl_weights,
        args.acc_curve_v1_checkpoint,
        args.acc_curve_v2_checkpoint,
        args.acc_curve_v1_cache,
        args.acc_curve_v2_cache,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    print(f"start: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"device: {device}")
    print(f"output_root: {output_root}")
    print(f"dip_test_cache: {args.dip_test_cache}")
    print(f"baseline_pl_weights: {args.baseline_pl_weights}")

    records, baseline_manifest = load_records(args.dip_test_cache, max_sequences=args.max_sequences)
    records_by_name = {record["name"]: record for record in records}
    print(f"loaded DIP test records: {len(records)}")

    v1_preds, v1_manifest, v1_ckpt = load_acc_curve_predictions(args.acc_curve_v1_cache, args.acc_curve_v1_checkpoint, device, args.max_sequences)
    v2_preds, v2_manifest, v2_ckpt = load_acc_curve_predictions(args.acc_curve_v2_cache, args.acc_curve_v2_checkpoint, device, args.max_sequences)
    missing_v1 = sorted(set(records_by_name) - set(v1_preds))
    missing_v2 = sorted(set(records_by_name) - set(v2_preds))
    if missing_v1 or missing_v2:
        raise KeyError(f"Missing AccCurve predictions: v1={missing_v1[:5]}, v2={missing_v2[:5]}")

    runner = OfficialPLRunner(args.baseline_pl_weights, device)
    target_cache: Dict[str, torch.Tensor] = {}
    per_variant_preds: Dict[str, List[torch.Tensor]] = defaultdict(list)
    per_variant_targets: Dict[str, List[torch.Tensor]] = defaultdict(list)
    per_variant_masks: Dict[str, List[torch.Tensor]] = defaultdict(list)
    per_sequence_rows: List[dict] = []
    official_feature_max_abs_diff = 0.0
    non_acc_block_max_abs_diff = 0.0
    first_debug = None
    total_frames = 0

    for seq_idx, record in enumerate(records):
        name = record["name"]
        a_m_raw = record["aM"].float().reshape(-1, 6, 3)
        w_m = record["wM"].float().reshape(-1, 6, 3)
        r_mb = record["RMB"].float()
        t = a_m_raw.shape[0]
        total_frames += t
        for label, source in (("v1", v1_preds[name]), ("v2", v2_preds[name])):
            if source["pred"].shape != (t, 6, 3):
                raise AssertionError(f"{label} pred for {name} shape {tuple(source['pred'].shape)} != {(t, 6, 3)}")
        smooth = v2_preds[name]["aM_smooth"]
        smooth_diff = float((smooth - v1_preds[name]["aM_smooth"]).abs().max().item())
        if smooth_diff > 1e-4:
            raise AssertionError(f"v1/v2 aM_smooth mismatch for {name}: {smooth_diff}")
        if smooth.shape != (t, 6, 3):
            raise AssertionError(f"smooth shape for {name}: {tuple(smooth.shape)}")

        variant_acc = {
            "official_raw_acc": a_m_raw,
            "smooth_acc": smooth,
            "acc_curve_v1_pred": v1_preds[name]["pred"],
            "acc_curve_v2_gtfk_pred": v2_preds[name]["pred"],
        }
        variant_features = {
            key: build_pl_features(acc, w_m, r_mb)
            for key, acc in variant_acc.items()
        }
        official_feature_diff = check_official_feature_equivalence(
            record,
            variant_features["official_raw_acc"],
            max_frames=args.feature_check_max_frames,
        )
        official_feature_max_abs_diff = max(official_feature_max_abs_diff, official_feature_diff)
        if official_feature_diff >= 1e-5:
            raise AssertionError(
                f"official_raw_acc feature differs from pl_input_feature for {name}: {official_feature_diff}"
            )

        ref_non_acc = variant_features["official_raw_acc"][:, 18:]
        for variant, feature in variant_features.items():
            diff = float((feature[:, 18:] - ref_non_acc).abs().max().item())
            non_acc_block_max_abs_diff = max(non_acc_block_max_abs_diff, diff)
            if diff != 0.0:
                raise AssertionError(f"non-acc 66D block changed for {name}/{variant}: {diff}")

        if first_debug is None:
            first_debug = accel_debug_stats(
                name,
                {key: feature[:, :18] for key, feature in variant_features.items()},
            )

        target = pl_target_from_pose(record["pose_gt"].float(), runner.body_model).float()
        target[:, 15:] = art.math.normalize_tensor(target[:, 15:])
        finite_mask = torch.isfinite(target).all(dim=-1)
        target_cache[name] = target
        for variant, feature in variant_features.items():
            pred = runner.forward_sequence(feature, record["pose_gt"][0].float())
            metrics = sequence_metrics(pred, target, finite_mask)
            row = {
                "sequence": name,
                "variant": variant,
                **metrics,
            }
            per_sequence_rows.append(row)
            per_variant_preds[variant].append(pred)
            per_variant_targets[variant].append(target)
            per_variant_masks[variant].append(finite_mask)
        print(f"[{seq_idx + 1}/{len(records)}] {name}: frames={t}")

    result_rows: List[dict] = []
    for variant, acc_source, target_used in VARIANTS:
        metrics = aggregate_metrics(per_variant_preds[variant], per_variant_targets[variant], per_variant_masks[variant])
        result_rows.append({
            "variant": variant,
            "acc_source": acc_source,
            "target_used_by_acc_curve": target_used,
            **metrics,
        })

    official = next(row for row in result_rows if row["variant"] == "official_raw_acc")
    for row in result_rows:
        row["pRB_l2_delta_vs_official_cm"] = row["pRB_l2_cm"] - official["pRB_l2_cm"]
        row["gR1_deg_delta_vs_official"] = row["gR1_deg"] - official["gR1_deg"]
        row["pRB_l2_improves_vs_official"] = bool(row["pRB_l2_cm"] < official["pRB_l2_cm"])
        row["gR1_deg_improves_vs_official"] = bool(row["gR1_deg"] < official["gR1_deg"])

    debug_path = output_root / "debug_first_sequence_acceleration_blocks.json"
    debug_path.write_text(json.dumps(_jsonable(first_debug), indent=2) + "\n")
    write_csv(
        output_root / "dip_test_pl_input_eval.csv",
        result_rows,
        [
            "variant",
            "acc_source",
            "target_used_by_acc_curve",
            "pRB_l2_cm",
            "pRB_rmse_cm",
            "pRB_mae_cm",
            "gR1_deg",
            "gR1_cos_loss",
            "num_sequences",
            "num_frames",
            "valid_frames",
            "pRB_l2_delta_vs_official_cm",
            "gR1_deg_delta_vs_official",
            "pRB_l2_improves_vs_official",
            "gR1_deg_improves_vs_official",
        ],
    )
    write_csv(
        output_root / "per_sequence_metrics.csv",
        per_sequence_rows,
        ["sequence", "variant", "valid_frames", "pRB_l2_cm", "pRB_rmse_cm", "pRB_mae_cm", "gR1_deg", "gR1_cos_loss"],
    )

    summary = {
        "experiment": "EXP-20260617-acc_curve_pl_input_eval",
        "experiment_root": str(output_root),
        "evaluator": "scripts/eval_pl_with_acc_curve_input_20260617.py",
        "baseline_pl_checkpoint": str(args.baseline_pl_weights),
        "dip_test_cache": str(args.dip_test_cache),
        "dip_test_protocol": "newpl_v5_official_protocol_20260607 DIP test with offset_r baseline cache; evaluation-only",
        "acc_curve_v1_checkpoint": str(args.acc_curve_v1_checkpoint),
        "acc_curve_v2_checkpoint": str(args.acc_curve_v2_checkpoint),
        "acc_curve_v1_cache": str(args.acc_curve_v1_cache),
        "acc_curve_v2_cache": str(args.acc_curve_v2_cache),
        "num_sequences": len(records),
        "num_frames": total_frames,
        "assertions": {
            "official_feature_max_abs_diff": official_feature_max_abs_diff,
            "non_acc_block_max_abs_diff": non_acc_block_max_abs_diff,
            "acc_curve_prediction_shape": "[T,6,3] for v1 and v2",
            "dip_test_no_train_norm_or_checkpoint_selection": True,
        },
        "target_contract": "PL target from pl_target_from_pose(pose_gt, SMPL_male vert_mask=GPNet.v_imu): pRB[15]+gR1[3]",
        "frame_contract": "AccCurve pred is model/world-frame M acceleration; PL input uses root-frame aRB = acc_M @ RMB_root.",
        "debug_json": str(debug_path),
        "variants": result_rows,
        "source_manifests": {
            "baseline": baseline_manifest,
            "acc_curve_v1": v1_manifest,
            "acc_curve_v2": v2_manifest,
        },
        "checkpoint_contracts": {
            "acc_curve_v1": v1_ckpt.get("contract"),
            "acc_curve_v2": v2_ckpt.get("target_contract") or v2_ckpt.get("contract"),
        },
    }
    (output_root / "result_summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    (output_root / "summary.md").write_text(build_summary_md(summary, result_rows))
    print(format_result_table(result_rows))
    print(f"result_summary: {output_root / 'result_summary.json'}")
    print(f"end: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
