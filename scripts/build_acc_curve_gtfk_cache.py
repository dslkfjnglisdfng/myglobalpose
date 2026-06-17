import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RBDL_PYTHON = Path("/home/lingfeng/rbdl/build/python")
if RBDL_PYTHON.exists() and str(RBDL_PYTHON) not in sys.path:
    sys.path.insert(0, str(RBDL_PYTHON))

try:
    from articulate.utils.rbdl import RBDLModel
except ImportError as exc:
    raise ImportError(
        "Failed to import RBDL. Use the GlobalPose GPU conda env and LD_LIBRARY_PATH, e.g. "
        'ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; '
        'export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; '
        '"$ENV/bin/python" scripts/build_acc_curve_gtfk_cache.py --preset dip_val'
    ) from exc

from acc_curve import acc_curve_features
from l4_q75_utils import q75_to_pose_tran
from l4_sensor_offset_utils import DT, COORDINATE_CONTRACT, official_imu_fields, smooth_centered
from l4_train_diverse_short import load_cache_files
from pip_physics_backend import PIP_PHYSICS_MODEL_FILE, smpl_to_pip_rbdl
from scripts.audit_amass_rbdl_only_imu_acc_4way import (
    IMU_LINKS,
    derivative_aware_spline_decode,
    ensure_imu_bodies,
    finite_difference_q,
    unwrap_q_angles,
)


DEFAULT_PRESETS = {
    "amass_train": {
        "input_cache": "data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json",
        "dataset": "AMASS",
        "split": "train",
    },
    "dip_train": {
        "input_cache": "data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json",
        "dataset": "DIP-IMU",
        "split": "train",
    },
    "dip_val": {
        "input_cache": "data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json",
        "dataset": "DIP-IMU",
        "split": "val",
    },
    "dip_test": {
        "input_cache": "data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json",
        "dataset": "DIP-IMU",
        "split": "test",
    },
}

TARGET_SOURCE = "GTFK(q,qdot,qddot,rJS)"
TARGET_CONTRACT = "GTFKacc(q,qdot,qddot,rJS) -> centered smooth -> aFK_gtfk_smooth[6,3]"


class BodyRef:
    def __init__(self, value):
        self.value = int(value)


def load_pose_tran(data, seq_idx):
    if data.get("pose_gt") and data["pose_gt"]:
        pose = data["pose_gt"][seq_idx].float()
        tran = data["tran_gt"][seq_idx].float()
        return pose, tran, "pose_gt/tran_gt"
    if data.get("q75_gt") and data["q75_gt"]:
        pose, tran = q75_to_pose_tran(data["q75_gt"][seq_idx].float())
        return pose, tran, "q75_gt->pose/tran"
    raise KeyError("source sequence has no pose_gt/tran_gt or q75_gt for strict q source")


def load_offset(data, seq_idx):
    for key in ("offset_r", "imu_offset_r", "r_JS"):
        if data.get(key) and data[key]:
            return data[key][seq_idx].float(), key
    raise KeyError("source sequence has no offset_r/imu_offset_r/r_JS for strict rJS source")


def q_qdot_qddot_from_pose(pose, tran, args):
    q_raw = smpl_to_pip_rbdl(
        pose.detach().cpu().numpy(),
        tran.detach().cpu().numpy(),
    )
    q = unwrap_q_angles(q_raw)
    qdot_fd, qddot_fd = finite_difference_q(q)
    if args.q_derivative_source == "finite_difference":
        return q, qdot_fd, qddot_fd, {
            "q_source": "smpl_to_pip_rbdl(pose_gt,tran_gt)",
            "qdot_qddot_source": "finite_difference_q",
        }
    if args.q_derivative_source != "derivative_aware_spline":
        raise ValueError(f"Unsupported q derivative source: {args.q_derivative_source}")
    _, q_ctrl, qdot_ctrl, qddot_ctrl = derivative_aware_spline_decode(
        q,
        qdot_fd,
        qddot_fd,
        args.derivfit_position_weight,
        args.derivfit_velocity_weight,
        args.derivfit_acceleration_weight,
        args.derivfit_ridge_weight,
    )
    return q_ctrl, qdot_ctrl, qddot_ctrl, {
        "q_source": "smpl_to_pip_rbdl(pose_gt,tran_gt)",
        "qdot_qddot_source": "derivative_aware_spline_decode(q, qdot_fd, qddot_fd)",
        "derivfit_weights": {
            "position": float(args.derivfit_position_weight),
            "velocity": float(args.derivfit_velocity_weight),
            "acceleration": float(args.derivfit_acceleration_weight),
            "ridge": float(args.derivfit_ridge_weight),
        },
    }


def rbdl_sensor_site_acceleration(model, bodies, q, qdot, qddot, r_js):
    q = np.asarray(q, dtype=np.float64)
    qdot = np.asarray(qdot, dtype=np.float64)
    qddot = np.asarray(qddot, dtype=np.float64)
    r_js = np.asarray(r_js, dtype=np.float64).reshape(6, 3)
    if q.shape != qdot.shape or q.shape != qddot.shape:
        raise ValueError(f"q/qdot/qddot shape mismatch: {q.shape}, {qdot.shape}, {qddot.shape}")
    if q.shape[-1] != model.qdot_size:
        raise ValueError(f"Expected q dim {model.qdot_size}, got {q.shape[-1]}")
    acc = np.zeros((q.shape[0], 6, 3), dtype=np.float64)
    for t in range(q.shape[0]):
        for sensor_idx, body in enumerate(bodies):
            acc[t, sensor_idx] = model.calc_point_acceleration(
                q[t],
                qdot[t],
                qddot[t],
                body,
                r_js[sensor_idx],
            )
    return torch.from_numpy(acc).float()


def build_sequence(data, seq_idx, name, args, model, bodies):
    pose, tran, pose_source = load_pose_tran(data, seq_idx)
    aM_raw, wM, RMB = official_imu_fields(data, seq_idx)
    offset_r, offset_source = load_offset(data, seq_idx)
    n = min(pose.shape[0], tran.shape[0], aM_raw.shape[0], wM.shape[0], RMB.shape[0])
    if args.max_frames:
        n = min(n, int(args.max_frames))
    pose, tran = pose[:n], tran[:n]
    aM_raw, wM, RMB = aM_raw[:n], wM[:n], RMB[:n]
    if n < 5:
        raise ValueError(f"{name} too short for strict GTFK target: frames={n}")

    q, qdot, qddot, q_meta = q_qdot_qddot_from_pose(pose, tran, args)
    if q.shape[0] != n:
        raise RuntimeError(f"{name} q length mismatch: q={q.shape[0]} n={n}")
    aFK_gtfk_raw = rbdl_sensor_site_acceleration(model, bodies, q, qdot, qddot, offset_r)
    aFK_gtfk_smooth = smooth_centered(aFK_gtfk_raw, args.smooth_window, mode=args.smoothing_mode)
    aM_smooth = smooth_centered(aM_raw, args.smooth_window, mode=args.smoothing_mode)

    valid = torch.isfinite(aFK_gtfk_smooth).all(dim=(-1, -2))
    trim = int(args.trim)
    if trim > 0 and valid.shape[0] > 2 * trim:
        valid[:trim] = False
        valid[-trim:] = False

    feature = acc_curve_features(aM_raw, aM_smooth, wM, RMB)
    if feature.shape[-1] != 108:
        raise RuntimeError(f"bad AccCurve feature dim {feature.shape[-1]}")
    tensors = (feature, aM_raw, aM_smooth, aFK_gtfk_raw, aFK_gtfk_smooth, wM, RMB)
    if not all(torch.isfinite(x).all() for x in tensors):
        raise RuntimeError(f"non-finite strict AccCurve tensors for {name}")
    if "diffpos" in TARGET_SOURCE.lower():
        raise RuntimeError("diffpos must not appear in v2 target source")

    return {
        "name": str(name),
        "num_frames": int(n),
        "feature": feature.cpu(),
        "aM_raw": aM_raw.reshape(n, 18).cpu(),
        "aM_smooth": aM_smooth.reshape(n, 18).cpu(),
        "aFK_gtfk_raw": aFK_gtfk_raw.reshape(n, 18).cpu(),
        "aFK_gtfk_smooth": aFK_gtfk_smooth.reshape(n, 18).cpu(),
        "valid_mask": valid.cpu(),
        "wM": wM.reshape(n, 18).cpu(),
        "RMB": RMB.cpu(),
        "offset_r": offset_r.cpu(),
        "rJS": offset_r.cpu(),
        "q": torch.from_numpy(q).float().cpu(),
        "qdot": torch.from_numpy(qdot).float().cpu(),
        "qddot": torch.from_numpy(qddot).float().cpu(),
        "source_metadata": {
            "pose_source": pose_source,
            "offset_source": offset_source,
            **q_meta,
            "target_source": TARGET_SOURCE,
            "target_contract": TARGET_CONTRACT,
            "rbdl_model": str(PIP_PHYSICS_MODEL_FILE),
            "rbdl_imu_links": list(IMU_LINKS),
        },
    }


def empty_shard():
    return {
        "name": [],
        "num_frames": [],
        "feature": [],
        "aM_raw": [],
        "aM_smooth": [],
        "aFK_gtfk_raw": [],
        "aFK_gtfk_smooth": [],
        "valid_mask": [],
        "wM": [],
        "RMB": [],
        "offset_r": [],
        "rJS": [],
        "q": [],
        "qdot": [],
        "qddot": [],
        "source_metadata": [],
    }


def append_record(shard, record):
    for key in shard:
        shard[key].append(record[key])


def prepare_output_dir(path, overwrite):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; pass --overwrite to rebuild intentionally.")
    if overwrite and path.exists():
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_sanity_json(output_dir, record):
    valid = record["valid_mask"].bool()
    if not bool(valid.any()):
        return None
    base = record["aM_smooth"][valid].reshape(-1, 6, 3)
    target = record["aFK_gtfk_smooth"][valid].reshape(-1, 6, 3)
    base_rmse = float((base - target).square().mean().sqrt())
    sanity = {
        "name": record["name"],
        "target_source": TARGET_SOURCE,
        "target_contract": TARGET_CONTRACT,
        "aM_smooth_vs_aFK_gtfk_smooth_RMSE": base_rmse,
        "pred_vs_aFK_gtfk_smooth_RMSE": None,
        "valid_frames": int(valid.sum()),
        "note": "pred field is filled by training/eval; cache sanity reports strict target/base alignment only.",
    }
    if "diffpos" in sanity["target_source"].lower():
        raise RuntimeError("diffpos must not appear in v2 target source")
    path = output_dir / "sanity_gtfk_target.json"
    path.write_text(json.dumps(sanity, indent=2) + "\n")
    return sanity


def build_cache(args, preset_name=None):
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    files, source_manifest = load_cache_files(args.input_cache)
    model = RBDLModel(str(PIP_PHYSICS_MODEL_FILE), update_kinematics_by_hand=False)
    bodies = ensure_imu_bodies(model, IMU_LINKS)
    cache_files = []
    shard = empty_shard()
    total_sequences = 0
    total_frames = 0
    total_valid = 0
    first_sanity = None
    failures = []
    started = time.time()

    def flush():
        nonlocal shard
        if not shard["name"]:
            return
        out = output_dir / f"acc_curve_gtfk_cache_shard{len(cache_files):05d}.pt"
        torch.save(shard, out)
        cache_files.append({
            "path": str(out),
            "num_sequences": len(shard["name"]),
            "num_frames": int(sum(shard["num_frames"])),
        })
        shard = empty_shard()

    for cache_file in files:
        data = torch.load(cache_file, map_location="cpu")
        for seq_idx, name in enumerate(data["name"]):
            if args.max_sequences and total_sequences >= args.max_sequences:
                break
            try:
                record = build_sequence(data, seq_idx, name, args, model, bodies)
            except Exception as exc:
                failure = {
                    "name": str(name),
                    "cache_file": str(cache_file),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                if not args.skip_failures:
                    raise RuntimeError(f"Failed strict GTFK cache build: {failure}") from exc
                continue
            if first_sanity is None:
                first_sanity = write_sanity_json(output_dir, record)
            append_record(shard, record)
            total_sequences += 1
            total_frames += int(record["num_frames"])
            total_valid += int(record["valid_mask"].sum())
            if len(shard["name"]) >= args.shard_size:
                flush()
            if args.progress_every and total_sequences % args.progress_every == 0:
                print(json.dumps({
                    "processed_sequences": total_sequences,
                    "processed_frames": total_frames,
                    "valid_frames": total_valid,
                    "failures": len(failures),
                    "elapsed_sec": round(time.time() - started, 3),
                }))
        if args.max_sequences and total_sequences >= args.max_sequences:
            break
    flush()
    manifest = {
        "type": "acc_curve_cache_v2_gtfk",
        "version": "acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617",
        "preset": preset_name,
        "dataset": args.dataset,
        "split": args.split,
        "source_cache": str(args.input_cache),
        "source_manifest": source_manifest,
        "cache_files": cache_files,
        "num_sequences": int(total_sequences),
        "num_frames": int(total_frames),
        "valid_frames": int(total_valid),
        "num_failures": int(len(failures)),
        "failures": failures,
        "feature_dim": 108,
        "target_dim": 18,
        "dt": DT,
        "smooth_window": int(args.smooth_window),
        "smoothing_mode": args.smoothing_mode,
        "trim": int(args.trim),
        "q_derivative_source": args.q_derivative_source,
        "target_key": "aFK_gtfk_smooth",
        "target_source": TARGET_SOURCE,
        "target_contract": TARGET_CONTRACT,
        "coordinate_contract": {
            "input_frame": "model/world frame M from GlobalPose aM/wM/RMB cache fields; no root-frame transform",
            "target_frame": "RBDL base/world frame from GTFKacc(q,qdot,qddot,rJS)",
            "offset": COORDINATE_CONTRACT,
            "RMB_6d": "rotation[..., :, :2].transpose(-1, -2).reshape(..., 6), matching PL convention",
        },
        "feature_layout": "aM_raw[18] + aM_smooth[18] + raw_minus_smooth[18] + wM[18] + RMB_6d[36]",
        "target_layout": "aFK_gtfk_smooth[18], six sensor-site accelerations in m/s^2",
        "raw_target_layout": "aFK_gtfk_raw[18], direct RBDL calc_point_acceleration before smoothing",
        "base_layout": "aM_smooth[18], same units and frame as target",
        "valid_mask": "finite strict GTFK acceleration frames, with configured trim removed",
        "forbidden_fallback": "No p_WS finite difference target; no smooth(diff_acc(p_WS)) fallback is allowed.",
        "sanity": first_sanity,
        "created_at_unix": time.time(),
    }
    if "diffpos" in str(manifest["target_source"]).lower():
        raise RuntimeError("diffpos must not appear in v2 target source")
    manifest_path = output_dir / "acc_curve_gtfk_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "num_sequences": total_sequences,
        "num_frames": total_frames,
        "valid_frames": total_valid,
        "failures": len(failures),
        "elapsed_sec": round(time.time() - started, 3),
        "target_contract": TARGET_CONTRACT,
    }, indent=2))
    return manifest_path


def resolve_preset_args(base_args, preset_name):
    preset = DEFAULT_PRESETS[preset_name]
    args = argparse.Namespace(**vars(base_args))
    args.input_cache = preset["input_cache"]
    args.output_dir = str(Path(base_args.output_root) / preset_name)
    args.dataset = preset["dataset"]
    args.split = preset["split"]
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Build strict AccCurve v2 GTFK q/qdot/qddot/rJS caches.")
    parser.add_argument("--preset", choices=sorted(DEFAULT_PRESETS))
    parser.add_argument("--all-defaults", action="store_true")
    parser.add_argument("--input-cache")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-root", default="code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--split", default="custom")
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--smoothing-mode", default="centered_moving_average")
    parser.add_argument("--trim", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--q-derivative-source", choices=("derivative_aware_spline", "finite_difference"), default="derivative_aware_spline")
    parser.add_argument("--derivfit-position-weight", type=float, default=1.0)
    parser.add_argument("--derivfit-velocity-weight", type=float, default=0.03)
    parser.add_argument("--derivfit-acceleration-weight", type=float, default=0.0003)
    parser.add_argument("--derivfit-ridge-weight", type=float, default=1e-6)
    parser.add_argument("--skip-failures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be odd")
    if args.all_defaults:
        for preset in DEFAULT_PRESETS:
            build_cache(resolve_preset_args(args, preset), preset)
        return
    if args.preset:
        build_cache(resolve_preset_args(args, args.preset), args.preset)
        return
    if not args.input_cache or not args.output_dir:
        raise SystemExit("--input-cache and --output-dir are required without --preset")
    build_cache(args, None)


if __name__ == "__main__":
    main()
