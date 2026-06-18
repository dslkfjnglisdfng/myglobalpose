#!/usr/bin/env python3
"""Build leaf-relative acceleration residual audit caches.

Contract:
  IMU residual input: aIMU_leaf_rel = aM_leaf - aM_root
  GT residual target: aGT_leaf_rel = diff_acc(FK_zero_trans leaf site)
                    - diff_acc(FK_zero_trans root site)

The GT path uses SMPL forward kinematics with tran forced to zero for every
dataset.  Both IMU and GT are model/world-frame vectors.  The root IMU at index
5 is used only as the reference acceleration for the five leaf sensors and is
not included in residual/loss/metric tensors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import articulate as art
from l4_q75_utils import q75_to_pose_tran
from l4_sensor_offset_utils import (
    COORDINATE_CONTRACT,
    DT,
    FPS,
    IMU_JOINTS,
    SENSOR_NAMES,
    official_imu_fields,
    smooth_centered,
)
from l4_train_diverse_short import load_cache_files


EXPERIMENT = "acc_leaf_relative_residual_v3_20260618"
DEFAULT_OUTPUT_ROOT = Path("data/experiments/acc_leaf_relative_residual_v3_20260618")
ROOT_INDEX = 5
LEAF_INDICES = (0, 1, 2, 3, 4)

DEFAULT_SOURCES = {
    "AMASS": [
        {
            "split": "train",
            "path": "data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json",
        },
    ],
    "DIP": [
        {
            "split": "train",
            "path": "data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json",
        },
        {
            "split": "val",
            "path": "data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json",
        },
        {
            "split": "test",
            "path": "data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json",
        },
    ],
    "TotalCapture": [
        {
            "split": "train",
            "path": "data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json",
        },
        {
            "split": "val",
            "path": "data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json",
        },
        {
            "split": "test",
            "path": "data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json",
        },
    ],
}


def sanitize_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "__", str(name)).strip("._")
    return text[:180] or "sequence"


def load_pose(data: dict, seq_idx: int) -> Tuple[torch.Tensor, str]:
    if data.get("pose_gt") and data["pose_gt"]:
        return data["pose_gt"][seq_idx].float(), "pose_gt"
    if data.get("pose") and data["pose"]:
        pose = data["pose"][seq_idx].float()
        if pose.dim() == 2 and pose.shape[-1] == 72:
            pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        return pose.float(), "pose"
    if data.get("q75_gt") and data["q75_gt"]:
        pose, _ = q75_to_pose_tran(data["q75_gt"][seq_idx].float())
        return pose.float(), "q75_gt->pose"
    raise KeyError("missing pose_gt/pose/q75_gt")


def load_offset(data: dict, seq_idx: int) -> Tuple[torch.Tensor, str]:
    for key in ("offset_r", "imu_offset_r", "r_JS"):
        if data.get(key) and data[key]:
            offset = data[key][seq_idx].float()
            if offset.shape != (6, 3):
                raise ValueError(f"{key} shape={tuple(offset.shape)}, expected (6,3)")
            return offset, key
    raise KeyError("missing offset_r/imu_offset_r/r_JS")


def load_imu(data: dict, seq_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aM_raw, wM, RMB = official_imu_fields(data, seq_idx)
    return aM_raw.float(), wM.float(), RMB.float()


@torch.no_grad()
def sensor_sites_zero_translation(
    pose: torch.Tensor,
    offset_r: torch.Tensor,
    body_model: art.ParametricModel,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    pose = pose.to(device=device, dtype=torch.float32)
    offset_r = offset_r.to(device=device, dtype=torch.float32)
    out = []
    zero = None
    for start in range(0, pose.shape[0], batch_size):
        p = pose[start:start + batch_size]
        if zero is None or zero.shape[0] != p.shape[0]:
            zero = torch.zeros(p.shape[0], 3, device=device, dtype=p.dtype)
        grot, joints = body_model.forward_kinematics(p, tran=zero, calc_mesh=False)[:2]
        p_wj = joints[:, IMU_JOINTS]
        r = offset_r.view(1, 6, 3, 1).expand(p.shape[0], -1, -1, -1)
        p_ws = p_wj + grot[:, IMU_JOINTS].matmul(r).squeeze(-1)
        out.append(p_ws.detach().cpu())
    return torch.cat(out, dim=0).float()


def centered_second_difference(x: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    acc = torch.full_like(x, float("nan"))
    valid = torch.zeros(x.shape[0], dtype=torch.bool)
    if x.shape[0] >= 3:
        acc[1:-1] = (x[:-2] - 2.0 * x[1:-1] + x[2:]) / (float(dt) ** 2)
        valid[1:-1] = True
    return acc, valid


def metric_values(a: torch.Tensor, b: torch.Tensor) -> dict:
    err = a - b
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    av = flat_a - flat_a.mean()
    bv = flat_b - flat_b.mean()
    den = av.norm() * bv.norm()
    corr = float((av * bv).sum() / den) if float(den) > 1e-12 else float("nan")
    return {
        "l2": float(err.norm(dim=-1).mean()),
        "rmse": float(err.square().mean().sqrt()),
        "mae": float(err.abs().mean()),
        "corr": corr,
    }


def build_record(
    data: dict,
    seq_idx: int,
    dataset: str,
    split: str,
    source_manifest: str,
    source_file: str,
    args: argparse.Namespace,
    body_model: art.ParametricModel,
    device: torch.device,
) -> dict:
    name = str(data["name"][seq_idx])
    pose, pose_source = load_pose(data, seq_idx)
    offset_r, offset_source = load_offset(data, seq_idx)
    aM_raw, wM, RMB = load_imu(data, seq_idx)
    n = min(pose.shape[0], aM_raw.shape[0], wM.shape[0], RMB.shape[0])
    if args.max_frames:
        n = min(n, int(args.max_frames))
    if n < 3:
        raise ValueError(f"{name} has only {n} aligned frames")
    pose = pose[:n]
    aM_raw = aM_raw[:n]
    wM = wM[:n]
    RMB = RMB[:n]
    p_ws = sensor_sites_zero_translation(pose, offset_r, body_model, device, args.fk_batch_size)
    aGT_abs_raw, diff_valid = centered_second_difference(p_ws, DT)
    leaf = torch.as_tensor(LEAF_INDICES, dtype=torch.long)
    aIMU_leaf_rel_raw = aM_raw[:, leaf] - aM_raw[:, ROOT_INDEX:ROOT_INDEX + 1]
    aGT_leaf_rel_raw = aGT_abs_raw[:, leaf] - aGT_abs_raw[:, ROOT_INDEX:ROOT_INDEX + 1]
    aIMU_leaf_rel_smooth = smooth_centered(aIMU_leaf_rel_raw, args.smooth_window, mode=args.smoothing_mode)
    aGT_leaf_rel_smooth = smooth_centered(aGT_leaf_rel_raw, args.smooth_window, mode=args.smoothing_mode)
    valid = (
        diff_valid
        & torch.isfinite(aIMU_leaf_rel_raw).all(dim=(-1, -2))
        & torch.isfinite(aGT_leaf_rel_raw).all(dim=(-1, -2))
    )
    if args.trim > 0 and valid.shape[0] > 2 * args.trim:
        valid[:args.trim] = False
        valid[-args.trim:] = False
    if not bool(valid.any()):
        raise ValueError(f"{name} has no valid frames after centered diff and trim")
    aGT_abs_raw = torch.nan_to_num(aGT_abs_raw, nan=0.0)
    aGT_leaf_rel_raw = torch.nan_to_num(aGT_leaf_rel_raw, nan=0.0)
    aGT_leaf_rel_smooth = torch.nan_to_num(aGT_leaf_rel_smooth, nan=0.0)
    meta = {
        "experiment": EXPERIMENT,
        "dataset": dataset,
        "split": split,
        "sequence_name": name,
        "source_manifest": source_manifest,
        "source_file": source_file,
        "fps": FPS,
        "dt": DT,
        "root_index": ROOT_INDEX,
        "root_sensor": SENSOR_NAMES[ROOT_INDEX],
        "leaf_indices": list(LEAF_INDICES),
        "leaf_sensors": [SENSOR_NAMES[i] for i in LEAF_INDICES],
        "formulation": "leaf_relative_acceleration_residual",
        "zero_translation": True,
        "pose_source": pose_source,
        "imuacc_source": "aM_raw from official_imu_fields",
        "offset_source": offset_source,
        "sensor_names": list(SENSOR_NAMES),
        "sensor_joint_mapping": list(IMU_JOINTS),
        "offset_contract": COORDINATE_CONTRACT,
        "frame_contract": "M/world-frame acceleration vectors; no sensor-local rotation is applied. Root index 5 is reference only and excluded from residual metrics.",
        "diff_method": "central second difference: (p[t-1] - 2*p[t] + p[t+1]) / dt^2",
        "valid_frame_range": [int(valid.nonzero()[0]), int(valid.nonzero()[-1])],
        "valid_frames": int(valid.sum()),
        "num_frames": int(n),
        "smoothing": {
            "window": int(args.smooth_window),
            "mode": args.smoothing_mode,
            "applied_to": "leaf-relative raw residual tensors directly",
        },
    }
    return {
        "aIMU_leaf_rel_raw": aIMU_leaf_rel_raw.cpu(),
        "aGT_leaf_rel_raw": aGT_leaf_rel_raw.cpu(),
        "aIMU_leaf_rel_smooth": aIMU_leaf_rel_smooth.cpu(),
        "aGT_leaf_rel_smooth": aGT_leaf_rel_smooth.cpu(),
        "aM_raw": aM_raw.cpu(),
        "aGT_abs_raw": aGT_abs_raw.cpu(),
        "aM_root_raw": aM_raw[:, ROOT_INDEX].cpu(),
        "aGT_root_raw": aGT_abs_raw[:, ROOT_INDEX].cpu(),
        "p_ws_zero_trans": p_ws.cpu() if args.save_positions else None,
        "wM": wM.cpu(),
        "RMB": RMB.cpu(),
        "rJS": offset_r.cpu(),
        "valid_mask": valid.cpu(),
        "meta": meta,
    }


def iter_sources(args: argparse.Namespace) -> Iterable[Tuple[str, str, Path]]:
    if args.source:
        for item in args.source:
            dataset, split, path = item.split("=", 2)
            yield dataset, split, Path(path)
        return
    for dataset, items in DEFAULT_SOURCES.items():
        for item in items:
            yield dataset, item["split"], Path(item["path"])


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; use --overwrite")
    if overwrite and path.exists():
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_all(args: argparse.Namespace) -> Path:
    out_root = Path(args.output_root)
    prepare_output(out_root, args.overwrite)
    device = torch.device(args.device)
    body_model = art.ParametricModel("models/SMPL_male.pkl", device=device)
    manifest = {
        "experiment": EXPERIMENT,
        "output_root": str(out_root),
        "root_index": ROOT_INDEX,
        "root_sensor": SENSOR_NAMES[ROOT_INDEX],
        "leaf_indices": list(LEAF_INDICES),
        "leaf_sensors": [SENSOR_NAMES[i] for i in LEAF_INDICES],
        "fps": FPS,
        "dt": DT,
        "formulation": "leaf_relative_acceleration_residual",
        "zero_translation": True,
        "metric_channels": "leaf indices 0..4 only; root index 5 excluded",
        "cache_files": [],
        "datasets": {},
        "failures": [],
        "created_at_unix": time.time(),
    }
    started = time.time()
    seq_count = 0
    for dataset, split, source in iter_sources(args):
        if args.dataset and dataset not in set(args.dataset):
            continue
        files, source_manifest = load_cache_files(source)
        dataset_dir = out_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest["datasets"].setdefault(dataset, {"splits": {}, "num_sequences": 0, "num_frames": 0, "valid_frames": 0})
        manifest["datasets"][dataset]["splits"].setdefault(split, {"num_sequences": 0, "num_frames": 0, "valid_frames": 0})
        for cache_file in files:
            data = torch.load(cache_file, map_location="cpu", weights_only=False)
            for seq_idx, name in enumerate(data["name"]):
                if args.max_sequences and seq_count >= args.max_sequences:
                    break
                try:
                    rec = build_record(
                        data, seq_idx, dataset, split, str(source), str(cache_file), args, body_model, device
                    )
                except Exception as exc:
                    failure = {
                        "dataset": dataset,
                        "split": split,
                        "source": str(source),
                        "cache_file": str(cache_file),
                        "sequence": str(name),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    manifest["failures"].append(failure)
                    if not args.skip_failures:
                        write_json(out_root / "cache_manifest.json", manifest)
                        raise RuntimeError(f"cache build failed: {failure}") from exc
                    continue
                seq_count += 1
                safe = sanitize_name(rec["meta"]["sequence_name"])
                out = dataset_dir / f"{split}__{seq_count:06d}__{safe}.pt"
                if rec.get("p_ws_zero_trans") is None:
                    rec.pop("p_ws_zero_trans", None)
                torch.save(rec, out)
                entry = {
                    "path": str(out),
                    "dataset": dataset,
                    "split": split,
                    "sequence_name": rec["meta"]["sequence_name"],
                    "num_frames": rec["meta"]["num_frames"],
                    "valid_frames": rec["meta"]["valid_frames"],
                    "valid_frame_range": rec["meta"]["valid_frame_range"],
                }
                manifest["cache_files"].append(entry)
                for bucket in (manifest["datasets"][dataset], manifest["datasets"][dataset]["splits"][split]):
                    bucket["num_sequences"] += 1
                    bucket["num_frames"] += rec["meta"]["num_frames"]
                    bucket["valid_frames"] += rec["meta"]["valid_frames"]
                if args.progress_every and seq_count % args.progress_every == 0:
                    print(json.dumps({
                        "processed_sequences": seq_count,
                        "last_dataset": dataset,
                        "last_split": split,
                        "elapsed_sec": round(time.time() - started, 3),
                        "failures": len(manifest["failures"]),
                    }), flush=True)
            if args.max_sequences and seq_count >= args.max_sequences:
                break
    manifest["num_sequences"] = len(manifest["cache_files"])
    manifest["num_failures"] = len(manifest["failures"])
    manifest["elapsed_sec"] = round(time.time() - started, 3)
    write_json(out_root / "cache_manifest.json", manifest)
    print(json.dumps({
        "cache_manifest": str(out_root / "cache_manifest.json"),
        "num_sequences": manifest["num_sequences"],
        "num_failures": manifest["num_failures"],
        "elapsed_sec": manifest["elapsed_sec"],
    }, indent=2))
    return out_root / "cache_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leaf-relative acceleration residual audit cache v3.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source", action="append", help="Override source as DATASET=SPLIT=manifest.json; repeatable.")
    parser.add_argument("--dataset", action="append", choices=("AMASS", "DIP", "TotalCapture"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--smoothing-mode", default="centered_moving_average")
    parser.add_argument("--trim", type=int, default=0, help="Extra valid-frame trim after centered diff; default keeps frames 1..T-2.")
    parser.add_argument("--fk-batch-size", type=int, default=2048)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--save-positions", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be odd")
    build_all(args)


if __name__ == "__main__":
    main()
