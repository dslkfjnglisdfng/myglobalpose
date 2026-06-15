#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_amass_rbdl_only_imu_acc_4way import (  # noqa: E402
    IMU_LINKS,
    derivative_aware_spline_decode,
    ensure_imu_bodies,
    finite_difference_q,
    rbdl_site_motion,
    unwrap_q_angles,
)
from articulate.utils.rbdl import RBDLModel  # noqa: E402
from imu_position_offset import load_offset_cache, prepare_sequence  # noqa: E402
from l4_rawlike_se3_calibration import matvec  # noqa: E402
from l4_sensor_offset_utils import GRAVITY_WORLD, SENSOR_NAMES, smooth_centered  # noqa: E402
from pip_physics_backend import PIP_PHYSICS_MODEL_FILE, smpl_to_pip_rbdl  # noqa: E402


FPS = 60.0
ROOT_SENSOR = 5
LOWER_SENSORS = (2, 3)
NONROOT_SENSORS = (0, 1, 2, 3, 4)
SMOOTH_ROOT = Path("data/experiments/footlock_transpose_rjs_smoothacc_20260609")
OLD_FOOTLOCK_ROOT = Path("data/experiments/footlock_transpose_rjs_20260608")


PRESETS = {
    "dip_train": {
        "dataset": "dip",
        "source": "data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json",
        "smooth_offset": SMOOTH_ROOT / "dip_train_footlock_transpose_rjs.pt",
        "old_offset": OLD_FOOTLOCK_ROOT / "dip_train_footlock_transpose_rjs.pt",
        "use_trans": False,
    },
    "dip_val": {
        "dataset": "dip",
        "source": "data/dataset_work/L4Cache/prephysics_pose_velocity_dip_val_globalpose_neural_only/baseline_cache_manifest.json",
        "smooth_offset": SMOOTH_ROOT / "dip_val_footlock_transpose_rjs.pt",
        "old_offset": OLD_FOOTLOCK_ROOT / "dip_val_footlock_transpose_rjs.pt",
        "use_trans": False,
    },
    "dip_test": {
        "dataset": "dip",
        "source": "data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json",
        "smooth_offset": SMOOTH_ROOT / "dip_test_footlock_transpose_rjs.pt",
        "old_offset": OLD_FOOTLOCK_ROOT / "dip_test_footlock_transpose_rjs.pt",
        "use_trans": False,
    },
    "totalcapture_train": {
        "dataset": "totalcapture",
        "source": "data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json",
        "smooth_offset": SMOOTH_ROOT / "totalcapture_train_footlock_transpose_rjs.pt",
        "old_offset": None,
        "use_trans": True,
    },
    "totalcapture_val": {
        "dataset": "totalcapture",
        "source": "data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json",
        "smooth_offset": SMOOTH_ROOT / "totalcapture_val_footlock_transpose_rjs.pt",
        "old_offset": None,
        "use_trans": True,
    },
    "totalcapture_test": {
        "dataset": "totalcapture",
        "source": "data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_test_official_neural_only_offset_r/baseline_cache_manifest.json",
        "smooth_offset": SMOOTH_ROOT / "totalcapture_test_footlock_transpose_rjs.pt",
        "old_offset": None,
        "use_trans": True,
    },
}


def resolve_path(path, manifest_path=None):
    path = Path(path)
    if path.is_absolute() and path.exists():
        return path
    candidates = [path, ROOT / path]
    if manifest_path is not None:
        candidates.append(Path(manifest_path).parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def tensor_item(value, idx):
    return value[idx] if isinstance(value, list) else value[idx]


def load_records(path, max_sequences=0, max_frames=0):
    path = Path(path)
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        files = [resolve_path(item["path"], manifest_path=path) for item in manifest["cache_files"]]
    else:
        manifest = {"cache_files": [{"path": str(path)}]}
        files = [path]
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location="cpu")
        names = data.get("name")
        if names is None:
            raise KeyError(f"{cache_file} missing name")
        for seq_idx, name in enumerate(names):
            record = {"name": str(name), "cache_file": str(cache_file)}
            for key in ("pose_gt", "tran_gt", "q75_gt", "aM", "wM", "RMB"):
                if key in data and data[key]:
                    value = tensor_item(data[key], seq_idx).float()
                    if int(max_frames) > 0:
                        value = value[: int(max_frames)]
                    record[key] = value
            required = ("pose_gt", "tran_gt", "aM", "wM", "RMB")
            missing = [key for key in required if key not in record]
            if missing:
                raise KeyError(f"{cache_file}:{name} missing {missing}")
            records.append(record)
            if int(max_sequences) > 0 and len(records) >= int(max_sequences):
                return records, manifest
    return records, manifest


def load_offset_detail(path):
    if path is None:
        return None
    path = resolve_path(path)
    if not path.exists():
        return None
    cache = torch.load(path, map_location="cpu")
    offsets = load_offset_cache(path)
    rows = {str(row.get("name")): row for row in cache.get("rows", []) if row.get("name") is not None}
    return {"path": str(path), "offsets": offsets, "rows": rows}


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def vector_l1(a, b):
    return np.mean(np.abs(a - b), axis=-1)


def vector_l2(a, b):
    return np.linalg.norm(a - b, axis=-1)


def direction_angle_deg(a, b, eps=1e-8):
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    mask = (an > eps) & (bn > eps)
    out = np.full(an.shape, np.nan, dtype=np.float64)
    dot = np.sum(a[mask] * b[mask], axis=-1) / (an[mask] * bn[mask])
    out[mask] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return out


def finite_mask(*arrays):
    mask = None
    for arr in arrays:
        current = np.isfinite(arr).all(axis=tuple(range(1, arr.ndim))) if arr.ndim > 1 else np.isfinite(arr)
        mask = current if mask is None else (mask & current)
    return mask


def summarize_pair(pred, target, frame_mask=None):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if frame_mask is not None:
        pred = pred[frame_mask]
        target = target[frame_mask]
    mask = finite_mask(pred, target)
    pred = pred[mask]
    target = target[mask]
    if pred.size == 0:
        return {
            "l1_mps2": stats([]),
            "l2_mps2": stats([]),
            "angle_deg": stats([]),
            "per_sensor_l2_mps2": [],
        }
    l2 = vector_l2(pred, target)
    per_sensor = []
    if pred.ndim == 3:
        per_sensor = [float(v) for v in np.nanmean(vector_l2(pred, target), axis=0)]
    return {
        "l1_mps2": stats(vector_l1(pred, target)),
        "l2_mps2": stats(l2),
        "angle_deg": stats(direction_angle_deg(pred, target)),
        "per_sensor_l2_mps2": per_sensor,
    }


def contact_mask_from_row(row, n):
    mask = np.zeros(n, dtype=bool)
    if not row:
        return mask
    for window in row.get("contact_windows", []) or []:
        start = max(0, int(window.get("start", 0)))
        end = min(n, int(window.get("end", 0)))
        if end > start:
            mask[start:end] = True
    return mask


def high_acc_masks(aM_smooth):
    rel_lower = aM_smooth[:, list(LOWER_SENSORS)] - aM_smooth[:, ROOT_SENSOR:ROOT_SENSOR + 1]
    mag = np.linalg.norm(rel_lower, axis=-1).mean(axis=-1)
    if not np.isfinite(mag).any():
        return np.zeros_like(mag, dtype=bool), np.zeros_like(mag, dtype=bool)
    threshold = np.nanquantile(mag, 0.75)
    return mag >= threshold, mag < threshold


def make_sequence_for_prepare(record, use_trans):
    pose = record["pose_gt"].float()
    tran = record["tran_gt"].float() if use_trans else torch.zeros_like(record["tran_gt"].float())
    return {
        "name": [record["name"]],
        "pose": [pose],
        "tran": [tran],
        "aM": [record["aM"].float()],
        "wM": [record["wM"].float()],
        "RMB": [record["RMB"].float()],
    }


def smpl_contract_prediction(seq, offset):
    offset = torch.as_tensor(offset).float().view(6, 3)
    preds = []
    target = seq["aM"].detach().cpu().numpy().astype(np.float64)
    for sensor_idx in range(6):
        R_ws_t = seq["R_wj"][:, sensor_idx].matmul(seq["R_JS"][sensor_idx]).transpose(-1, -2)
        joint_acc_sensor = matvec(R_ws_t, seq["ddot_p_wj"][:, sensor_idx] - GRAVITY_WORLD.view(1, 3))
        lever_sensor = seq["ddot_R_wj"][:, sensor_idx].matmul(offset[sensor_idx].view(3, 1)).squeeze(-1)
        preds.append((joint_acc_sensor + lever_sensor).detach().cpu().numpy())
    return np.stack(preds, axis=1).astype(np.float64), target


def smpl_variant_metrics(seq, offset, group_masks):
    pred, target = smpl_contract_prediction(seq, offset)
    lower = list(LOWER_SENSORS)
    nonroot = list(NONROOT_SENSORS)
    pred_lower_rel = pred[:, lower] - pred[:, ROOT_SENSOR:ROOT_SENSOR + 1]
    target_lower_rel = target[:, lower] - target[:, ROOT_SENSOR:ROOT_SENSOR + 1]
    pred_nonroot_rel = pred[:, nonroot] - pred[:, ROOT_SENSOR:ROOT_SENSOR + 1]
    target_nonroot_rel = target[:, nonroot] - target[:, ROOT_SENSOR:ROOT_SENSOR + 1]
    out = {}
    for group, mask in group_masks.items():
        out[group] = {
            "absolute_all": summarize_pair(pred, target, mask),
            "rootrel_lower": summarize_pair(pred_lower_rel, target_lower_rel, mask),
            "rootrel_nonroot": summarize_pair(pred_nonroot_rel, target_nonroot_rel, mask),
        }
    return out


def q_from_record(record, use_trans):
    pose = record["pose_gt"].detach().cpu().numpy()
    tran = record["tran_gt"].detach().cpu().numpy()
    if not use_trans:
        tran = np.zeros_like(tran)
    return smpl_to_pip_rbdl(pose, tran)


def rbdl_qcontrol_metrics(record, model, bodies, use_trans, smooth_window, group_masks):
    q_raw = q_from_record(record, use_trans=use_trans)
    q = unwrap_q_angles(q_raw)
    qdot_fd, qddot_fd = finite_difference_q(q)
    _, q_ctrl, qdot_ctrl, qddot_ctrl = derivative_aware_spline_decode(
        q,
        qdot_fd,
        qddot_fd,
        1.0,
        0.03,
        0.0003,
        1e-6,
    )
    _, a_q_ctrl = rbdl_site_motion(model, bodies, q_ctrl, qdot_ctrl, qddot_ctrl)
    aM_smooth = smooth_centered(record["aM"].float(), smooth_window).detach().cpu().numpy().astype(np.float64)
    lower = list(LOWER_SENSORS)
    nonroot = list(NONROOT_SENSORS)
    pred_lower_rel = a_q_ctrl[:, lower] - a_q_ctrl[:, ROOT_SENSOR:ROOT_SENSOR + 1]
    target_lower_rel = aM_smooth[:, lower] - aM_smooth[:, ROOT_SENSOR:ROOT_SENSOR + 1]
    pred_nonroot_rel = a_q_ctrl[:, nonroot] - a_q_ctrl[:, ROOT_SENSOR:ROOT_SENSOR + 1]
    target_nonroot_rel = aM_smooth[:, nonroot] - aM_smooth[:, ROOT_SENSOR:ROOT_SENSOR + 1]
    out = {}
    for group, mask in group_masks.items():
        out[group] = {
            "rootrel_lower": summarize_pair(pred_lower_rel, target_lower_rel, mask),
            "rootrel_nonroot": summarize_pair(pred_nonroot_rel, target_nonroot_rel, mask),
        }
    return out


def aggregate_metric(rows, variant, group, metric_family, value_key):
    values = []
    per_sensor = []
    for row in rows:
        metric = row.get("smpl_contract", {}).get(variant, {}).get(group, {}).get(metric_family)
        if not metric:
            continue
        stat = metric.get(value_key, {})
        if stat.get("mean") is not None:
            values.append(stat["mean"])
        if metric.get("per_sensor_l2_mps2"):
            per_sensor.append(metric["per_sensor_l2_mps2"])
    out = stats(values)
    if per_sensor:
        out["per_sensor_l2_mps2_mean"] = [float(v) for v in np.nanmean(np.asarray(per_sensor), axis=0)]
    return out


def aggregate_rbdl(rows, group, metric_family, value_key):
    values = []
    per_sensor = []
    for row in rows:
        metric = row.get("rbdl_qcontrol", {}).get(group, {}).get(metric_family)
        if not metric:
            continue
        stat = metric.get(value_key, {})
        if stat.get("mean") is not None:
            values.append(stat["mean"])
        if metric.get("per_sensor_l2_mps2"):
            per_sensor.append(metric["per_sensor_l2_mps2"])
    out = stats(values)
    if per_sensor:
        out["per_sensor_l2_mps2_mean"] = [float(v) for v in np.nanmean(np.asarray(per_sensor), axis=0)]
    return out


def build_aggregate(rows, variants):
    groups = ("all", "contact", "noncontact", "high_acc", "low_acc")
    aggregate = {"smpl_contract": {}, "rbdl_qcontrol": {}}
    for variant in variants:
        aggregate["smpl_contract"][variant] = {}
        for group in groups:
            aggregate["smpl_contract"][variant][group] = {
                "rootrel_lower_l2_mps2": aggregate_metric(rows, variant, group, "rootrel_lower", "l2_mps2"),
                "rootrel_nonroot_l2_mps2": aggregate_metric(rows, variant, group, "rootrel_nonroot", "l2_mps2"),
                "absolute_all_l2_mps2": aggregate_metric(rows, variant, group, "absolute_all", "l2_mps2"),
            }
    for group in groups:
        aggregate["rbdl_qcontrol"][group] = {
            "rootrel_lower_l2_mps2": aggregate_rbdl(rows, group, "rootrel_lower", "l2_mps2"),
            "rootrel_nonroot_l2_mps2": aggregate_rbdl(rows, group, "rootrel_nonroot", "l2_mps2"),
        }
    return aggregate


def compare_summary(aggregate):
    def mean_for(variant):
        return aggregate["smpl_contract"].get(variant, {}).get("all", {}).get("rootrel_lower_l2_mps2", {}).get("mean")

    zero = mean_for("zero_rJS")
    old = mean_for("old_footlock_20260608")
    smooth = mean_for("smoothacc_footlock_20260609")
    out = {
        "primary_metric": "SMPL-contract all-frame root-relative lower-body acceleration L2 m/s^2",
        "zero_rJS": zero,
        "old_footlock_20260608": old,
        "smoothacc_footlock_20260609": smooth,
        "smooth_vs_zero_delta": None if zero is None or smooth is None else smooth - zero,
        "smooth_vs_old_delta": None if old is None or smooth is None else smooth - old,
    }
    if smooth is None:
        out["verdict"] = "smoothacc_not_available"
    elif zero is not None and smooth < zero and (old is None or smooth < old):
        out["verdict"] = "smoothacc_best"
    elif zero is not None and smooth < zero:
        out["verdict"] = "smoothacc_better_than_zero_not_best"
    else:
        out["verdict"] = "smoothacc_not_better_than_zero"
    return out


def audit_split(args):
    preset = PRESETS[args.preset]
    source = Path(args.source) if args.source else Path(preset["source"])
    use_trans = bool(preset["use_trans"])
    if args.force_zero_trans:
        use_trans = False
    smooth_offset_path = Path(args.smooth_offset) if args.smooth_offset else preset["smooth_offset"]
    old_offset_path = Path(args.old_offset) if args.old_offset else preset["old_offset"]

    records, manifest = load_records(source, max_sequences=args.max_sequences, max_frames=args.max_frames)
    smooth_offsets = load_offset_detail(smooth_offset_path)
    old_offsets = load_offset_detail(old_offset_path)

    variants = ["zero_rJS", "smoothacc_footlock_20260609"]
    if old_offsets is not None:
        variants.insert(1, "old_footlock_20260608")

    model = None
    bodies = None
    if not args.skip_rbdl:
        model = RBDLModel(str(PIP_PHYSICS_MODEL_FILE), update_kinematics_by_hand=False)
        bodies = ensure_imu_bodies(model, IMU_LINKS)
    rows = []
    failures = []
    for idx, record in enumerate(records):
        try:
            seq_data = make_sequence_for_prepare(record, use_trans=use_trans)
            seq = prepare_sequence(
                seq_data,
                0,
                device=args.device,
                smooth_window=args.smooth_window,
                derivative_mode=args.derivative_mode,
            )
            n = seq["aM"].shape[0]
            new_row = smooth_offsets["rows"].get(record["name"]) if smooth_offsets else None
            contact = contact_mask_from_row(new_row, n)
            high_acc, low_acc = high_acc_masks(seq["aM"].detach().cpu().numpy())
            group_masks = {
                "all": np.ones(n, dtype=bool),
                "contact": contact,
                "noncontact": ~contact,
                "high_acc": high_acc,
                "low_acc": low_acc,
            }
            offsets = {
                "zero_rJS": torch.zeros(6, 3),
                "smoothacc_footlock_20260609": smooth_offsets["offsets"].get(record["name"]) if smooth_offsets else None,
            }
            if old_offsets is not None:
                offsets["old_footlock_20260608"] = old_offsets["offsets"].get(record["name"])
            smpl_contract = {}
            missing = []
            for variant in variants:
                offset = offsets.get(variant)
                if offset is None:
                    missing.append(variant)
                    continue
                smpl_contract[variant] = smpl_variant_metrics(seq, offset, group_masks)
            row = {
                "name": record["name"],
                "num_frames": int(n),
                "use_trans": use_trans,
                "missing_variants": missing,
                "contact_frame_count": int(contact.sum()),
                "high_acc_frame_count": int(high_acc.sum()),
                "smpl_contract": smpl_contract,
            }
            if not args.skip_rbdl:
                row["rbdl_qcontrol"] = rbdl_qcontrol_metrics(
                    record,
                    model,
                    bodies,
                    use_trans=use_trans,
                    smooth_window=args.smooth_window,
                    group_masks=group_masks,
                )
            rows.append(row)
            print(json.dumps({"idx": idx + 1, "count": len(records), "name": record["name"]}), flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"name": record.get("name"), "error": repr(exc)})
            print(json.dumps({"idx": idx + 1, "count": len(records), "name": record.get("name"), "error": repr(exc)}), flush=True)

    aggregate = build_aggregate(rows, variants)
    result = {
        "status": "ok" if not failures else "partial",
        "preset": args.preset,
        "dataset": preset["dataset"],
        "source": str(source),
        "num_records": len(rows),
        "failures": failures,
        "config": {
            "smooth_window": args.smooth_window,
            "derivative_mode": args.derivative_mode,
            "max_sequences": args.max_sequences,
            "max_frames": args.max_frames,
            "device": args.device,
            "use_trans": use_trans,
            "skip_rbdl": bool(args.skip_rbdl),
            "dip_trans_policy": "not_used" if preset["dataset"] == "dip" and not use_trans else "available_or_forced",
        },
        "offset_caches": {
            "smoothacc_footlock_20260609": None if smooth_offsets is None else smooth_offsets["path"],
            "old_footlock_20260608": None if old_offsets is None else old_offsets["path"],
            "zero_rJS": "generated_zero_offset",
        },
        "frame_contract": {
            "r_JS": "SMPL joint-local offset: p_WS = p_WJ + R_WJ @ r_JS",
            "rbdl_offset_injection": "not approved",
            "reason": "RBDL IMU link frame is not proven equivalent to SMPL mapped-joint local frame; r_JS is therefore evaluated with the SMPL-contract residual.",
            "q_control": (
                "RBDL q/qdot/qddot use derivative-aware UniformCubicBSpline control decode."
                if not args.skip_rbdl
                else "skipped in this run; use smoke/subset runs for RBDL q-control diagnostics."
            ),
        },
        "sensor_order": list(SENSOR_NAMES),
        "rbdl_imu_links": list(IMU_LINKS),
        "aggregate": aggregate,
        "comparison": compare_summary(aggregate),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_summary(result))
    return result


def fmt(value):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "not available"
    return f"{value:.6f}"


def markdown_summary(result):
    lines = [
        f"# Smoothed rJS acceleration control-curve audit: {result['preset']}",
        "",
        f"Dataset: `{result['dataset']}`",
        f"Records: `{result['num_records']}`",
        f"Source: `{result['source']}`",
        f"DIP trans policy: `{result['config']['dip_trans_policy']}`",
        "",
        "## Frame contract",
        "",
        f"- `r_JS`: {result['frame_contract']['r_JS']}",
        f"- `rbdl_offset_injection`: `{result['frame_contract']['rbdl_offset_injection']}`",
        f"- Reason: {result['frame_contract']['reason']}",
        f"- q control: {result['frame_contract']['q_control']}",
        "",
        "## Primary SMPL-contract metric",
        "",
        "Metric: all-frame root-relative lower-body acceleration L2 in m/s^2. Lower is better.",
        "",
        "| Variant | L2 mean | L2 median | L2 p95 | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for variant, notes in (
        ("zero_rJS", "zero offset baseline"),
        ("old_footlock_20260608", "historical footlock, if available"),
        ("smoothacc_footlock_20260609", "new smoothed-fit footlock"),
    ):
        metric = (
            result["aggregate"]["smpl_contract"]
            .get(variant, {})
            .get("all", {})
            .get("rootrel_lower_l2_mps2", {})
        )
        lines.append(
            f"| `{variant}` | {fmt(metric.get('mean'))} | {fmt(metric.get('median'))} | {fmt(metric.get('p95'))} | {notes} |"
        )
    comp = result["comparison"]
    lines.extend([
        "",
        "## Comparison",
        "",
        f"- Verdict: `{comp['verdict']}`",
        f"- smoothacc - zero delta: `{fmt(comp['smooth_vs_zero_delta'])}` m/s^2",
        f"- smoothacc - old delta: `{fmt(comp['smooth_vs_old_delta'])}` m/s^2",
        "",
        "## Contact and high-acc checks",
        "",
        "| Group | zero L2 | old L2 | smoothacc L2 |",
        "|---|---:|---:|---:|",
    ])
    for group in ("contact", "noncontact", "high_acc", "low_acc"):
        vals = []
        for variant in ("zero_rJS", "old_footlock_20260608", "smoothacc_footlock_20260609"):
            vals.append(
                result["aggregate"]["smpl_contract"]
                .get(variant, {})
                .get(group, {})
                .get("rootrel_lower_l2_mps2", {})
                .get("mean")
            )
        lines.append(f"| `{group}` | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} |")
    lines.extend([
        "",
        "## RBDL q-control diagnostic",
        "",
        "This uses derivative-aware control-curve q/qdot/qddot, but does not inject r_JS into RBDL.",
        "",
        "| Group | rootrel lower L2 | rootrel nonroot L2 |",
        "|---|---:|---:|",
    ])
    for group in ("all", "contact", "noncontact", "high_acc", "low_acc"):
        metric = result["aggregate"]["rbdl_qcontrol"].get(group, {})
        lines.append(
            "| `{}` | {} | {} |".format(
                group,
                fmt(metric.get("rootrel_lower_l2_mps2", {}).get("mean")),
                fmt(metric.get("rootrel_nonroot_l2_mps2", {}).get("mean")),
            )
        )
    if result.get("failures"):
        lines.extend(["", "## Failures", "", "```text"])
        for failure in result["failures"]:
            lines.append(f"{failure['name']}: {failure['error']}")
        lines.append("```")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Audit whether smoothed-footlock rJS explains smoothed IMU acceleration.")
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--smooth-offset", default="")
    parser.add_argument("--old-offset", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--derivative-mode", choices=("legacy", "centered", "strict_centered"), default="centered")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--force-zero-trans", action="store_true")
    parser.add_argument("--skip-rbdl", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = audit_split(args)
    print(json.dumps({
        "status": result["status"],
        "preset": result["preset"],
        "num_records": result["num_records"],
        "comparison": result["comparison"],
        "output_json": str(args.output_json),
    }, indent=2))


if __name__ == "__main__":
    main()
