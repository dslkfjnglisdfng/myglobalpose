#!/usr/bin/env python3
"""Synthesize TotalCapture r_JS by fitting sensor-frame IMU acceleration."""

import argparse
import csv
import gzip
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_totalcapture_imu_acc_vs_rjs_diff_acc import (  # noqa: E402
    DEFAULT_RJS_CANDIDATES,
    load_rjs_metadata,
)
from compare_totalcapture_imu_acc_vs_vertex_diff_acc import (  # noqa: E402
    AXES,
    METHODS,
    SPACES,
    aggregate_rows,
    frame_level_rows,
    infer_imu_contract,
    matvec,
    plot_boxplot,
    plot_residual_hist,
    plot_scatter,
    vector_metrics,
    write_csv,
    write_json,
)
from imu_position_offset import OFFSET_POSITION_CONTRACT, load_offset_cache  # noqa: E402
from l4_rawlike_se3_calibration import robust_rotation_mean, rotation_angle_deg  # noqa: E402
from l4_sensor_offset_utils import (  # noqa: E402
    FPS,
    GRAVITY_WORLD,
    IMU_JOINTS,
    IMU_VERTICES,
    SENSOR_NAMES,
    fk_imu_joints_and_vertices,
    official_imu_fields,
    savgol_smooth,
    second_derivative,
    sensor_to_joint_map,
)


DATASET_PATH = ROOT / "data/dataset_work/TotalCapture_globalpose_official/test.pt"
DEFAULT_VERTEX_BASELINE = ROOT / "code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409"
PRIMARY_SPACE = "sensor_specific_force"
PRIMARY_SOLUTION = "projected"
PRIMARY_BIAS = "no_bias"
ACC_FIT_METHODS = {
    "raw_fd": {"window": 1, "polyorder": 0},
    "savgol9_p3_fd": {"window": 9, "polyorder": 3},
    "savgol15_p3_fd": {"window": 15, "polyorder": 3},
}
SOURCES = ("vertex_baseline", "old_footlock_rjs", "accfit_per_sequence", "accfit_global", "accfit_loo")


def load_dataset(path):
    data = torch.load(path, map_location="cpu")
    missing = [k for k in ("name", "pose", "tran", "aS", "RIS", "RIM", "RSB", "wS") if k not in data]
    if missing:
        raise KeyError(f"{path} missing required TotalCapture fields: {missing}")
    return data


def choose_rjs_path(user_path):
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for path in DEFAULT_RJS_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No old rJS cache found; pass --old-rjs-path.")


def compact_rows(rows):
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def write_csv_lf(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key.startswith("_"):
                    continue
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def finite_float(x, default=float("nan")):
    try:
        value = float(x)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def sanitize_name(name):
    return str(name).replace("/", "_").replace(" ", "_")


def smooth_variant(x, method):
    spec = ACC_FIT_METHODS[method]
    if spec["window"] <= 1:
        return x.float()
    return savgol_smooth(x.float(), spec["window"], spec["polyorder"])


def prepare_sequences(data, count, args):
    sequences = []
    for idx in range(count):
        name = str(data["name"][idx])
        pose = data["pose"][idx].float()
        tran = data["tran"][idx].float()
        aS = data["aS"][idx].float()
        RIS = data["RIS"][idx].float()
        RIM = data["RIM"][idx].float()
        aM, _wM, _RMB = official_imu_fields(data, idx)
        n = min(pose.shape[0], tran.shape[0], aS.shape[0], RIS.shape[0], aM.shape[0])
        if args.max_frames:
            n = min(n, int(args.max_frames))
        if n < max(16, args.trim * 2 + 3):
            raise RuntimeError(f"{name} too short after max_frames: {n}")
        pose, tran, aS, RIS, aM = pose[:n], tran[:n], aS[:n], RIS[:n], aM[:n]
        p_wj, R_wj, p_wv = fk_imu_joints_and_vertices(pose, tran, device=args.device)
        R_WS_obs = RIM.transpose(1, 2).unsqueeze(0).matmul(RIS)
        seq = {
            "idx": idx,
            "name": name,
            "n": int(n),
            "time": np.arange(n, dtype=np.float64) / FPS,
            "p_wj": p_wj.float(),
            "R_wj": R_wj.float(),
            "p_wv": p_wv.float(),
            "R_WS_obs": R_WS_obs.float(),
            "aS": aS.float(),
            "aM": aM.float(),
            "variant": {},
        }
        for method in METHODS:
            p_s = smooth_variant(seq["p_wj"], method)
            R_s = smooth_variant(seq["R_wj"], method)
            v_s = smooth_variant(seq["p_wv"], method)
            seq["variant"][method] = {
                "p_wj": p_s,
                "R_wj": R_s,
                "p_wv": v_s,
                "ddot_p_wj": second_derivative(p_s, fps=FPS, mode="centered"),
                "ddot_R_wj": second_derivative(R_s, fps=FPS, mode="centered"),
                "acc_vertex_world": second_derivative(v_s[:, :5], fps=FPS, mode="centered"),
            }
        sequences.append(seq)
        print(f"[prepare {idx + 1}/{count}] {name} frames={n}", flush=True)
    return sequences


def estimate_R_JS(seq, sensor_idx, method, trim):
    R_wj = seq["variant"][method]["R_wj"][:, sensor_idx]
    R_obs = seq["R_WS_obs"][:, sensor_idx]
    valid = torch.isfinite(R_wj).all(dim=(-1, -2)) & torch.isfinite(R_obs).all(dim=(-1, -2))
    valid[:trim] = False
    valid[seq["n"] - trim :] = False
    if int(valid.sum()) < 8:
        return torch.eye(3)
    return robust_rotation_mean(R_wj[valid].transpose(-1, -2).matmul(R_obs[valid]))


def linear_terms(seq, sensor_idx, method, R_JS, trim):
    var = seq["variant"][method]
    R_wj = var["R_wj"][:, sensor_idx]
    ddot_p = var["ddot_p_wj"][:, sensor_idx]
    ddot_R = var["ddot_R_wj"][:, sensor_idx]
    aS = seq["aS"][:, sensor_idx]
    R_obs = seq["R_WS_obs"][:, sensor_idx]
    R_WS = R_wj.matmul(R_JS.view(1, 3, 3))
    R_SW = R_WS.transpose(-1, -2)
    base = matvec(R_SW, ddot_p - GRAVITY_WORLD.view(1, 3))
    A = R_SW.matmul(ddot_R)
    valid = (
        torch.isfinite(A).all(dim=(-1, -2))
        & torch.isfinite(base).all(dim=-1)
        & torch.isfinite(aS).all(dim=-1)
        & torch.isfinite(R_obs).all(dim=(-1, -2))
    )
    valid[:trim] = False
    valid[seq["n"] - trim :] = False
    orient_deg = rotation_angle_deg(R_obs.transpose(-1, -2).matmul(R_WS))
    return A[valid], aS[valid] - base[valid], base[valid], aS[valid], valid, orient_deg[valid]


def solve_ridge(A, y, ridge, fit_bias=False, bias_ridge=1e-4):
    if A.shape[0] < 8:
        raise RuntimeError("not enough valid frames for rJS fit")
    if fit_bias:
        eye = torch.eye(3, dtype=A.dtype).view(1, 3, 3).expand(A.shape[0], -1, -1)
        M = torch.cat([A, eye], dim=-1).reshape(-1, 6)
        reg = torch.diag(torch.tensor([ridge, ridge, ridge, bias_ridge, bias_ridge, bias_ridge], dtype=A.dtype))
    else:
        M = A.reshape(-1, 3)
        reg = torch.eye(3, dtype=A.dtype) * float(ridge)
    target = y.reshape(-1)
    lhs = M.T.matmul(M) + reg
    rhs = M.T.matmul(target)
    try:
        sol = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        sol = torch.linalg.lstsq(lhs, rhs).solution
    r = sol[:3] if fit_bias else sol
    bias = sol[3:] if fit_bias else torch.zeros(3, dtype=A.dtype)
    pred = A.matmul(r.view(3, 1)).squeeze(-1) + bias.view(1, 3)
    svals = torch.linalg.svdvals(A.reshape(-1, 3))
    cond = svals.max() / svals.min().clamp_min(1e-12)
    return r.float(), bias.float(), pred.float(), cond.float(), svals.float()


def project_offset(r, max_norm):
    norm = r.norm().clamp_min(1e-12)
    if float(norm) <= float(max_norm):
        return r.float(), False
    return (r * (float(max_norm) / norm)).float(), True


def fit_scope(sequences, seq_indices, method, sensor_idx, args, mode_name, test_sequence=None):
    R_list = [estimate_R_JS(sequences[i], sensor_idx, method, args.trim) for i in seq_indices]
    R_JS = robust_rotation_mean(torch.stack(R_list))
    As, ys, bases, obs, valids, orient = [], [], [], [], [], []
    for i in seq_indices:
        A, y, base, aS, valid, orient_deg = linear_terms(sequences[i], sensor_idx, method, R_JS, args.trim)
        As.append(A)
        ys.append(y)
        bases.append(base)
        obs.append(aS)
        valids.append(valid)
        orient.append(orient_deg)
    A_all = torch.cat(As, dim=0)
    y_all = torch.cat(ys, dim=0)
    base_all = torch.cat(bases, dim=0)
    obs_all = torch.cat(obs, dim=0)
    before_vec = obs_all - base_all
    r_nb, b_nb, pred_nb, cond, svals = solve_ridge(A_all, y_all, args.ridge, fit_bias=False)
    r_wb, b_wb, pred_wb, cond_wb, svals_wb = solve_ridge(A_all, y_all, args.ridge, fit_bias=True, bias_ridge=args.bias_ridge)
    r_proj, clipped = project_offset(r_nb, args.max_norm)
    pred_proj = A_all.matmul(r_proj.view(3, 1)).squeeze(-1)
    before = before_vec.norm(dim=-1)
    after_nb = (obs_all - (base_all + pred_nb)).norm(dim=-1)
    after_proj = (obs_all - (base_all + pred_proj)).norm(dim=-1)
    after_wb = (obs_all - (base_all + pred_wb)).norm(dim=-1)
    total_frames = sum(int(sequences[i]["n"] - 2 * args.trim) for i in seq_indices)
    valid_frames = int(A_all.shape[0])
    return {
        "mode": mode_name,
        "fit_sequence_indices": list(seq_indices),
        "fit_sequence_names": [sequences[i]["name"] for i in seq_indices],
        "test_sequence": test_sequence,
        "method": method,
        "sensor_id": int(sensor_idx),
        "sensor_name": SENSOR_NAMES[sensor_idx],
        "mapped_joint_id": int(IMU_JOINTS[sensor_idx]),
        "R_JS": R_JS.float(),
        "r_JS_unconstrained": r_nb.float(),
        "r_JS_projected": r_proj.float(),
        "r_JS_with_bias": r_wb.float(),
        "bias_sensor": b_wb.float(),
        "r_JS_norm": float(r_nb.norm()),
        "r_JS_projected_norm": float(r_proj.norm()),
        "r_JS_was_projected": bool(clipped),
        "bias_norm": float(b_wb.norm()),
        "condition_number": float(cond),
        "condition_number_with_bias": float(cond_wb),
        "singular_values": svals,
        "singular_values_with_bias": svals_wb,
        "residual_before_fit": float(before.mean()),
        "residual_after_fit": float(after_nb.mean()),
        "residual_after_projected_fit": float(after_proj.mean()),
        "residual_after_with_bias_fit": float(after_wb.mean()),
        "fit_improvement": float((before.mean() - after_nb.mean()) / before.mean().clamp_min(1e-12)),
        "projected_fit_improvement": float((before.mean() - after_proj.mean()) / before.mean().clamp_min(1e-12)),
        "with_bias_fit_improvement": float((before.mean() - after_wb.mean()) / before.mean().clamp_min(1e-12)),
        "num_frames": valid_frames,
        "total_candidate_frames": int(total_frames),
        "valid_frame_ratio": float(valid_frames / max(1, total_frames)),
        "orientation_residual_deg": float(torch.cat(orient).mean()) if orient else float("nan"),
    }


def fit_all_offsets(sequences, args):
    per_seq, global_fit, loo = {}, {}, {}
    all_idx = list(range(len(sequences)))
    for method in METHODS:
        per_seq[method] = []
        global_fit[method] = []
        loo[method] = []
        for seq_idx in all_idx:
            sensors = [fit_scope(sequences, [seq_idx], method, s, args, "per_sequence") for s in range(6)]
            per_seq[method].append(sensors)
        for s in range(6):
            global_fit[method].append(fit_scope(sequences, all_idx, method, s, args, "global"))
        if not args.skip_loo and len(sequences) > 1:
            for test_idx in all_idx:
                fit_idx = [i for i in all_idx if i != test_idx]
                sensors = [
                    fit_scope(
                        sequences,
                        fit_idx,
                        method,
                        s,
                        args,
                        "leave_one_sequence_out",
                        test_sequence=sequences[test_idx]["name"],
                    )
                    for s in range(6)
                ]
                loo[method].append(sensors)
        print(f"[fit] {method} complete", flush=True)
    return per_seq, global_fit, loo


def offset_tensor_from_fit(fits, field="r_JS_projected"):
    return torch.stack([torch.stack([sensor[field] for sensor in seq]) for seq in fits])


def global_tensor_from_fit(fits, field="r_JS_projected"):
    return torch.stack([sensor[field] for sensor in fits])


def old_rjs_tensor(old_map, sequences):
    return torch.stack([old_map[seq["name"]].float() for seq in sequences])


def predicted_acc_sensor(seq, method, offsets, source):
    if source == "vertex_baseline":
        acc_world = seq["variant"][method]["acc_vertex_world"]
        R_WS = seq["R_WS_obs"][:, :5]
        return matvec(R_WS.transpose(-1, -2), acc_world - GRAVITY_WORLD.view(1, 1, 3)), acc_world
    r = offsets.float()
    var = seq["variant"][method]
    p_ws = var["p_wj"][:, :5] + matvec(var["R_wj"][:, :5], r[:5].view(1, 5, 3))
    acc_world = second_derivative(p_ws, fps=FPS, mode="centered")
    R_WS = seq["R_WS_obs"][:, :5]
    acc_sensor = matvec(R_WS.transpose(-1, -2), acc_world - GRAVITY_WORLD.view(1, 1, 3))
    return acc_sensor, acc_world


def eval_source(sequences, method, source, offset_by_seq=None, global_offsets=None, loo_offsets=None, args=None):
    rows, examples = [], []
    for idx, seq in enumerate(sequences):
        if source == "vertex_baseline":
            offsets = None
        elif source == "accfit_global":
            offsets = global_offsets[method]
        elif source == "accfit_loo":
            if loo_offsets is None or method not in loo_offsets or not loo_offsets[method]:
                continue
            offsets = torch.stack([sensor["r_JS_projected"] for sensor in loo_offsets[method][idx]])
        else:
            offsets = offset_by_seq[method][idx] if isinstance(offset_by_seq, dict) else offset_by_seq[idx]
        acc_sensor_full, acc_world_full = predicted_acc_sensor(seq, method, offsets, source)
        trim = int(args.trim)
        sl = slice(trim, seq["n"] - trim)
        pred_sensor = acc_sensor_full[sl].numpy()
        target_sensor = seq["aS"][: seq["n"], :5][sl].numpy()
        pred_world = acc_world_full[sl].numpy()
        target_world = seq["aM"][: seq["n"], :5][sl].numpy()
        for space, pred, target in (
            ("sensor_specific_force", pred_sensor, target_sensor),
            ("model_world_linear_acc", pred_world, target_world),
        ):
            valid = np.isfinite(pred).all(axis=-1) & np.isfinite(target).all(axis=-1)
            for s, sensor_name in enumerate(SENSOR_NAMES[:5]):
                v = valid[:, s]
                if not np.any(v):
                    continue
                row = {
                    "source": source,
                    "solution_type": PRIMARY_SOLUTION if source.startswith("accfit") else "baseline",
                    "bias_mode": PRIMARY_BIAS if source.startswith("accfit") else "baseline",
                    "sequence_id": seq["name"],
                    "method": method,
                    "comparison_space": space,
                    "sensor_id": s,
                    "sensor_name": sensor_name,
                    "mapped_joint_id": int(IMU_JOINTS[s]),
                    "vertex_id": int(IMU_VERTICES[s]) if source == "vertex_baseline" else "",
                }
                row.update(vector_metrics(pred[:, s][v], target[:, s][v]))
                row["_pred"] = pred[:, s][v]
                row["_target"] = target[:, s][v]
                rows.append(row)
        if method in ("savgol9_p3_fd", "savgol15_p3_fd") and source in ("vertex_baseline", "old_footlock_rjs", "accfit_per_sequence", "accfit_global"):
            examples.append(
                {
                    "source": source,
                    "name": seq["name"],
                    "method": method,
                    "time": seq["time"],
                    "imu_sensor": seq["aS"][:, :5].numpy(),
                    "pred_sensor": acc_sensor_full.numpy(),
                }
            )
    return rows, examples


def evaluate_all(sequences, per_seq_fit, global_fit, loo_fit, old_offsets, args):
    accfit_seq_offsets = {
        method: [torch.stack([sensor["r_JS_projected"] for sensor in seq_fit]) for seq_fit in seq_fits]
        for method, seq_fits in per_seq_fit.items()
    }
    global_offsets = {method: torch.stack([sensor["r_JS_projected"] for sensor in sensors]) for method, sensors in global_fit.items()}
    rows, examples = [], []
    for method in METHODS:
        source_specs = (
            ("vertex_baseline", None),
            ("old_footlock_rjs", old_offsets),
            ("accfit_per_sequence", accfit_seq_offsets),
            ("accfit_global", accfit_seq_offsets),
            ("accfit_loo", accfit_seq_offsets),
        )
        for source, offsets in source_specs:
            source_rows, ex = eval_source(
                sequences,
                method,
                source,
                offset_by_seq=offsets,
                global_offsets=global_offsets,
                loo_offsets=loo_fit,
                args=args,
            )
            rows.extend(source_rows)
            examples.extend(ex)
        print(f"[eval] {method} rows={len(rows)}", flush=True)
    return rows, examples


def fit_rows(per_seq_fit, global_fit, loo_fit):
    rows = []
    for method, seqs in per_seq_fit.items():
        for seq_idx, sensors in enumerate(seqs):
            for item in sensors:
                row = fit_row(item)
                row["sequence_id"] = item["fit_sequence_names"][0]
                rows.append(row)
    for method, sensors in global_fit.items():
        for item in sensors:
            row = fit_row(item)
            row["sequence_id"] = "GLOBAL"
            rows.append(row)
    for method, seqs in loo_fit.items():
        for sensors in seqs:
            for item in sensors:
                row = fit_row(item)
                row["sequence_id"] = item["test_sequence"]
                rows.append(row)
    return rows


def fit_row(item):
    return {
        "mode": item["mode"],
        "method": item["method"],
        "sequence_id": item.get("test_sequence") or ",".join(item["fit_sequence_names"]),
        "fit_sequences": "|".join(item["fit_sequence_names"]),
        "sensor_id": item["sensor_id"],
        "sensor_name": item["sensor_name"],
        "mapped_joint_id": item["mapped_joint_id"],
        "rJS_x": float(item["r_JS_unconstrained"][0]),
        "rJS_y": float(item["r_JS_unconstrained"][1]),
        "rJS_z": float(item["r_JS_unconstrained"][2]),
        "rJS_norm": item["r_JS_norm"],
        "rJS_projected_x": float(item["r_JS_projected"][0]),
        "rJS_projected_y": float(item["r_JS_projected"][1]),
        "rJS_projected_z": float(item["r_JS_projected"][2]),
        "rJS_projected_norm": item["r_JS_projected_norm"],
        "rJS_was_projected": item["r_JS_was_projected"],
        "bias_x": float(item["bias_sensor"][0]),
        "bias_y": float(item["bias_sensor"][1]),
        "bias_z": float(item["bias_sensor"][2]),
        "bias_norm": item["bias_norm"],
        "condition_number": item["condition_number"],
        "residual_before_fit": item["residual_before_fit"],
        "residual_after_fit": item["residual_after_fit"],
        "residual_after_projected_fit": item["residual_after_projected_fit"],
        "residual_after_with_bias_fit": item["residual_after_with_bias_fit"],
        "fit_improvement": item["fit_improvement"],
        "projected_fit_improvement": item["projected_fit_improvement"],
        "with_bias_fit_improvement": item["with_bias_fit_improvement"],
        "num_frames": item["num_frames"],
        "valid_frame_ratio": item["valid_frame_ratio"],
        "orientation_residual_deg": item["orientation_residual_deg"],
    }


def fit_payload(per_seq_fit, global_fit, loo_fit, sequences, args):
    payload = {
        "name": [seq["name"] for seq in sequences],
        "method": list(METHODS),
        "r_JS_unconstrained": {m: offset_tensor_from_fit(v, "r_JS_unconstrained") for m, v in per_seq_fit.items()},
        "r_JS_projected": {m: offset_tensor_from_fit(v, "r_JS_projected") for m, v in per_seq_fit.items()},
        "R_JS": {m: offset_tensor_from_fit(v, "R_JS") for m, v in per_seq_fit.items()},
        "bias_sensor": {m: offset_tensor_from_fit(v, "bias_sensor") for m, v in per_seq_fit.items()},
        "summary_rows": fit_rows(per_seq_fit, {}, {}),
        "config": common_contract(args),
    }
    return payload


def global_payload(global_fit, args):
    return {
        "method": list(METHODS),
        "r_JS_unconstrained": {m: global_tensor_from_fit(v, "r_JS_unconstrained") for m, v in global_fit.items()},
        "r_JS_projected": {m: global_tensor_from_fit(v, "r_JS_projected") for m, v in global_fit.items()},
        "R_JS": {m: global_tensor_from_fit(v, "R_JS") for m, v in global_fit.items()},
        "bias_sensor": {m: global_tensor_from_fit(v, "bias_sensor") for m, v in global_fit.items()},
        "summary_rows": fit_rows({}, global_fit, {}),
        "config": common_contract(args),
    }


def common_contract(args):
    return {
        "fps": FPS,
        "dt": 1.0 / FPS,
        "gravity_world": GRAVITY_WORLD.tolist(),
        "ridge": args.ridge,
        "bias_ridge": args.bias_ridge,
        "max_norm": args.max_norm,
        "trim": args.trim,
        "methods": ACC_FIT_METHODS,
        "derivative_mode": "centered",
        "rjs_contract": OFFSET_POSITION_CONTRACT,
        "transform_convention": {
            "R_WJ": "maps joint-local vectors into world coordinates",
            "R_JS": "maps sensor-frame vectors into joint-local coordinates",
            "R_WS": "R_WJ @ R_JS, maps sensor-frame vectors into world coordinates",
            "r_JS": "IMU sensor origin relative to mapped joint origin, expressed in joint-local coordinates",
            "prediction": "a_S = R_WS^T @ (ddot(p_WJ) + ddot(R_WJ) @ r_JS - g_W)",
        },
    }


def rjs_offsets_json(per_seq_fit, global_fit, loo_fit, sequences, old_meta, args):
    rows = fit_rows(per_seq_fit, global_fit, loo_fit)
    return {
        "coordinate_contract": OFFSET_POSITION_CONTRACT,
        "sensor_to_joint_map": sensor_to_joint_map(),
        "old_rjs_source": old_meta,
        "fit_config": common_contract(args),
        "rows": rows,
        "per_sequence": [
            {
                "sequence_id": seq["name"],
                "sensors": [
                    {
                        "sensor_id": s,
                        "sensor_name": SENSOR_NAMES[s],
                        "mapped_joint_id": int(IMU_JOINTS[s]),
                        "r_JS_projected_by_method": {
                            m: [float(x) for x in per_seq_fit[m][i][s]["r_JS_projected"].tolist()] for m in METHODS
                        },
                    }
                    for s in range(6)
                ],
            }
            for i, seq in enumerate(sequences)
        ],
    }


def read_vertex_baseline_rows(path):
    csv_path = Path(path) / "summary_per_sensor.csv"
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("comparison_space") == PRIMARY_SPACE and row.get("method") in ("savgol9_p3_fd", "savgol15_p3_fd"):
                rows.append(row)
    return rows


def value_from_aggregate(overall, source, method, metric="rmse"):
    for row in overall:
        if row["source"] == source and row["comparison_space"] == PRIMARY_SPACE and row["method"] == method:
            return float(row[metric])
    return float("nan")


def sensor_value(per_sensor, source, method, sensor, metric="rmse"):
    for row in per_sensor:
        if (
            row["source"] == source
            and row["comparison_space"] == PRIMARY_SPACE
            and row["method"] == method
            and row["sensor_name"] == sensor
        ):
            return float(row[metric])
    return float("nan")


def plot_source_rmse_bar(per_sensor, out_dir, filename, sources, method, title):
    sensors = list(SENSOR_NAMES[:5])
    x = np.arange(len(sensors))
    width = 0.8 / len(sources)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, source in enumerate(sources):
        vals = [sensor_value(per_sensor, source, method, s) for s in sensors]
        ax.bar(x + (i - (len(sources) - 1) / 2) * width, vals, width, label=source)
    ax.set_xticks(x)
    ax.set_xticklabels(sensors, rotation=20, ha="right")
    ax.set_ylabel("RMSE (m/s^2)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def plot_fit_metric_bar(fit_summary, out_dir, metric, filename, title, method="savgol9_p3_fd", mode="per_sequence"):
    rows = [r for r in fit_summary if r["mode"] == mode and r["method"] == method]
    labels = [f"{r['sequence_id']}:{r['sensor_name']}" for r in rows if r["sensor_name"] in SENSOR_NAMES[:5]]
    vals = [float(r[metric]) for r in rows if r["sensor_name"] in SENSOR_NAMES[:5]]
    fig, ax = plt.subplots(figsize=(max(12, len(vals) * 0.35), 5))
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def plot_timeseries_examples(examples, out_dir, max_examples=4):
    used = {"old_footlock_rjs": 0, "accfit_per_sequence": 0}
    for ex in examples:
        if ex["source"] not in used or used[ex["source"]] >= max_examples:
            continue
        n = min(ex["imu_sensor"].shape[0], int(10 * FPS))
        t = ex["time"][:n]
        fig, axes = plt.subplots(5, 3, figsize=(16, 11), sharex=True)
        for s, sensor in enumerate(SENSOR_NAMES[:5]):
            for a, axis in enumerate(AXES):
                ax = axes[s, a]
                ax.plot(t, ex["imu_sensor"][:n, s, a], label="IMU", linewidth=1.0)
                ax.plot(t, ex["pred_sensor"][:n, s, a], label=ex["source"], linewidth=1.0, alpha=0.85)
                if s == 0:
                    ax.set_title(axis)
                if a == 0:
                    ax.set_ylabel(sensor)
        axes[0, 0].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"{ex['name']} {ex['source']} {ex['method']} sensor specific force")
        fig.tight_layout()
        fig.savefig(out_dir / f"timeseries_examples_{sanitize_name(ex['name'])}_{ex['source']}_{ex['method']}.png", dpi=150)
        plt.close(fig)
        used[ex["source"]] += 1
        if all(v >= max_examples for v in used.values()):
            break


def frame_level_rows_with_source(rows, trim):
    out = []
    for row in rows:
        pred = row["_pred"]
        target = row["_target"]
        residual = pred - target
        l2 = np.linalg.norm(residual, axis=-1)
        pred_mag = np.linalg.norm(pred, axis=-1)
        target_mag = np.linalg.norm(target, axis=-1)
        base = {
            "source": row["source"],
            "solution_type": row["solution_type"],
            "bias_mode": row["bias_mode"],
            "sequence_id": row["sequence_id"],
            "method": row["method"],
            "comparison_space": row["comparison_space"],
            "sensor_id": row["sensor_id"],
            "sensor_name": row["sensor_name"],
            "mapped_joint_id": row["mapped_joint_id"],
            "vertex_id": row["vertex_id"],
        }
        for i in range(pred.shape[0]):
            out.append(
                {
                    **base,
                    "frame_index": int(i + trim),
                    "pred_x": float(pred[i, 0]),
                    "pred_y": float(pred[i, 1]),
                    "pred_z": float(pred[i, 2]),
                    "imu_x": float(target[i, 0]),
                    "imu_y": float(target[i, 1]),
                    "imu_z": float(target[i, 2]),
                    "residual_x": float(residual[i, 0]),
                    "residual_y": float(residual[i, 1]),
                    "residual_z": float(residual[i, 2]),
                    "residual_l2": float(l2[i]),
                    "pred_magnitude": float(pred_mag[i]),
                    "imu_magnitude": float(target_mag[i]),
                    "magnitude_residual": float(pred_mag[i] - target_mag[i]),
                }
            )
    return out


def write_summary_md(path, overall, per_sensor, fit_summary, config):
    sg9 = "savgol9_p3_fd"
    sg15 = "savgol15_p3_fd"
    vertex_sg9 = value_from_aggregate(overall, "vertex_baseline", sg9)
    old_sg9 = value_from_aggregate(overall, "old_footlock_rjs", sg9)
    old_sg15 = value_from_aggregate(overall, "old_footlock_rjs", sg15)
    per_sg9 = value_from_aggregate(overall, "accfit_per_sequence", sg9)
    glob_sg9 = value_from_aggregate(overall, "accfit_global", sg9)
    loo_sg9 = value_from_aggregate(overall, "accfit_loo", sg9)
    per_sg15 = value_from_aggregate(overall, "accfit_per_sequence", sg15)
    glob_sg15 = value_from_aggregate(overall, "accfit_global", sg15)
    old_better = per_sg9 < old_sg9 or per_sg15 < old_sg15
    vertex_better = per_sg9 < vertex_sg9
    rb_old = sensor_value(per_sensor, "old_footlock_rjs", sg9, "right_lower_leg")
    rb_new = sensor_value(per_sensor, "accfit_per_sequence", sg9, "right_lower_leg")
    ll_old = sensor_value(per_sensor, "old_footlock_rjs", sg9, "left_lower_leg")
    ll_new = sensor_value(per_sensor, "accfit_per_sequence", sg9, "left_lower_leg")
    lf_old = sensor_value(per_sensor, "old_footlock_rjs", sg9, "left_forearm")
    lf_new = sensor_value(per_sensor, "accfit_per_sequence", sg9, "left_forearm")
    rf_old = sensor_value(per_sensor, "old_footlock_rjs", sg9, "right_forearm")
    rf_new = sensor_value(per_sensor, "accfit_per_sequence", sg9, "right_forearm")
    global_rows = [r for r in fit_summary if r["mode"] == "global" and r["method"] == sg9]
    per_rows = [r for r in fit_summary if r["mode"] == "per_sequence" and r["method"] == sg9]
    max_norm = max([float(r["rJS_projected_norm"]) for r in per_rows + global_rows] or [float("nan")])
    max_cond = max([float(r["condition_number"]) for r in per_rows + global_rows] or [float("nan")])
    med_cond = float(np.median([float(r["condition_number"]) for r in per_rows + global_rows])) if per_rows or global_rows else float("nan")
    bias_gain = np.nanmean(
        [
            float(r["with_bias_fit_improvement"]) - float(r["projected_fit_improvement"])
            for r in per_rows
            if math.isfinite(float(r["with_bias_fit_improvement"]))
        ]
    )
    rows = [r for r in overall if r["comparison_space"] == PRIMARY_SPACE]
    metric_lines = [
        "| source | SavGol-9 RMSE | SavGol-15 RMSE |",
        "|---|---:|---:|",
    ]
    for source in SOURCES:
        v9 = value_from_aggregate(rows, source, sg9)
        v15 = value_from_aggregate(rows, source, sg15)
        if math.isfinite(v9) or math.isfinite(v15):
            metric_lines.append(f"| {source} | {v9:.6f} | {v15:.6f} |")
    lines = [
        "# TotalCapture Accfit rJS Synthesis",
        "",
        "## Setup",
        "",
        f"- Dataset: `{config['dataset_path']}`",
        f"- Sequences: {', '.join(config['sequence_names'])}",
        f"- FPS: {config['fps']}; gravity: {config['gravity_world']}",
        f"- Main objective: sensor-frame specific force `a_S = R_WS^T @ (ddot(p_WJ) + ddot(R_WJ) @ r_JS - g_W)`.",
        f"- rJS contract: `{config['rjs_contract']}`",
        f"- Smoothing variants: raw FD, SavGol-9/poly3, SavGol-15/poly3; fit and evaluation use the same smoothing per variant.",
        f"- Ridge: {config['ridge']}; max_norm projection: {config['max_norm']} m; no-bias is main, with-bias is diagnostic only.",
        f"- Old rJS comparison source: `{config['old_rjs_path']}`",
        "",
        "## Overall Sensor-Specific RMSE",
        "",
        *metric_lines,
        "",
        "## Required Answers",
        "",
        f"1. Accfit rJS vs old footlock rJS: {'better' if old_better else 'not better'} overall. Per-sequence accfit SavGol-9/SavGol-15 RMSE = {per_sg9:.6f}/{per_sg15:.6f}; old footlock = {old_sg9:.6f}/{old_sg15:.6f}.",
        f"2. Accfit rJS vs 5-vertex baseline: {'better' if vertex_better else 'not better'} under the requested SavGol-9 gate. Vertex SavGol-9 RMSE = {vertex_sg9:.6f}; per-sequence accfit SavGol-9 RMSE = {per_sg9:.6f}.",
        f"3. Per-sequence vs global: {'per-sequence is better' if per_sg9 < glob_sg9 else 'global is better or tied'} on SavGol-9. Per-sequence/global/LOO RMSE = {per_sg9:.6f}/{glob_sg9:.6f}/{loo_sg9:.6f}.",
        f"4. Lower-leg: left lower leg old -> accfit SavGol-9 RMSE {ll_old:.6f} -> {ll_new:.6f}; right lower leg {rb_old:.6f} -> {rb_new:.6f}.",
        f"5. right_lower_leg large error {'decreased' if rb_new < rb_old else 'did not decrease'} versus old rJS under SavGol-9.",
        f"6. Forearm: left forearm old -> accfit {lf_old:.6f} -> {lf_new:.6f}; right forearm {rf_old:.6f} -> {rf_new:.6f}. If recovered, the old degradation was likely from footlock/pseudo offset not optimizing acceleration; if not, fixed joint-local offset is still insufficient for forearm soft-tissue/mount behavior.",
        f"7. rJS norm: maximum projected norm is {max_norm:.6f} m. Values at the 0.5 m cap indicate noise/frame-convention absorption; see `rjs_offsets.json` and `rjs_norm_bar.png`.",
        f"8. Condition number: median {med_cond:.6f}, max {max_cond:.6f}. Large outliers mean weak lever-arm observability for that sensor/sequence/motion.",
        f"9. With-bias diagnostic: mean extra fit-improvement over projected no-bias is {finite_float(bias_gain):.6f}. A large positive gap points to bias/gravity/frame convention rather than rJS position alone.",
        f"10. Supervision target decision: {'supported for follow-up acceleration explainability' if vertex_better and old_better and glob_sg9 < vertex_sg9 else 'not yet supported as a stronger fixed-rJS supervision target'}. This is a diagnostic only; no network, PL, IK, or VR module was trained or changed.",
        "",
        "## Outputs",
        "",
        "- `config.json`, `rjs_accfit_per_sequence.pt`, `rjs_accfit_global.pt`, `rjs_accfit_summary.json`, `rjs_offsets.json`",
        "- `summary_overall.csv`, `summary_per_sensor.csv`, `summary_per_sequence.csv`, `fit_summary.csv`, `frame_level_metrics.csv.gz`",
        "- requested PNGs including RMSE bars, rJS norm, fit improvement, condition number, timeseries, scatter, residual histograms, and boxplot.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--old-rjs-path", type=Path, default=None)
    parser.add_argument("--vertex-baseline-dir", type=Path, default=DEFAULT_VERTEX_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--trim", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--bias-ridge", type=float, default=1e-4)
    parser.add_argument("--max-norm", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-loo", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    old_path = choose_rjs_path(args.old_rjs_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (ROOT / "code/outputs" / f"totalcapture_accfit_rjs_synthesis_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset_path)
    count = len(data["name"])
    if args.max_sequences:
        count = min(count, int(args.max_sequences))
    sequences = prepare_sequences(data, count, args)
    old_map = load_offset_cache(old_path)
    missing = [seq["name"] for seq in sequences if seq["name"] not in old_map]
    if missing:
        raise RuntimeError(f"old rJS cache {old_path} missing sequences: {missing}")
    old_offsets = old_rjs_tensor(old_map, sequences)
    old_meta = load_rjs_metadata(old_path)

    per_seq_fit, global_fit, loo_fit = fit_all_offsets(sequences, args)
    rows, examples = evaluate_all(sequences, per_seq_fit, global_fit, loo_fit, old_offsets, args)
    if not rows:
        raise RuntimeError("No evaluation rows produced.")

    overall = aggregate_rows(rows, ["source", "solution_type", "bias_mode", "comparison_space", "method"])
    per_sensor = aggregate_rows(
        rows,
        ["source", "solution_type", "bias_mode", "comparison_space", "method", "sensor_id", "sensor_name", "mapped_joint_id", "vertex_id"],
    )
    per_sequence = aggregate_rows(rows, ["source", "solution_type", "bias_mode", "comparison_space", "method", "sequence_id"])
    fit_summary = fit_rows(per_seq_fit, global_fit, loo_fit)
    frame_rows = frame_level_rows_with_source(rows, args.trim)

    config = {
        **common_contract(args),
        "script": str(Path(__file__).resolve()),
        "dataset_path": str(args.dataset_path),
        "split": "test",
        "output_dir": str(out_dir),
        "old_rjs_path": str(old_path),
        "vertex_baseline_dir": str(args.vertex_baseline_dir),
        "num_sequences": count,
        "sequence_names": [seq["name"] for seq in sequences],
        "preferred_comparison_space": PRIMARY_SPACE,
        "main_solution": PRIMARY_SOLUTION,
        "main_bias_mode": PRIMARY_BIAS,
        "imu_contract": infer_imu_contract(data),
        "sensor_to_joint_map": sensor_to_joint_map(),
        "old_rjs_metadata": {
            "path": old_meta["path"],
            "offset_field": old_meta["offset_field"],
            "coordinate_contract": old_meta.get("coordinate_contract", ""),
            "method": old_meta.get("method", ""),
            "source_path": old_meta.get("source_path", ""),
        },
        "vertex_baseline_historical_dir": str(args.vertex_baseline_dir),
        "vertex_baseline_historical_rows_loaded": bool(read_vertex_baseline_rows(args.vertex_baseline_dir)),
    }
    write_json(out_dir / "config.json", config)
    torch.save(fit_payload(per_seq_fit, global_fit, loo_fit, sequences, args), out_dir / "rjs_accfit_per_sequence.pt")
    torch.save(global_payload(global_fit, args), out_dir / "rjs_accfit_global.pt")
    write_json(
        out_dir / "rjs_accfit_summary.json",
        {"fit_summary": fit_summary, "overall": compact_rows(overall), "per_sensor": compact_rows(per_sensor)},
    )
    write_json(out_dir / "rjs_offsets.json", rjs_offsets_json(per_seq_fit, global_fit, loo_fit, sequences, old_meta, args))
    write_csv_lf(out_dir / "summary_overall.csv", overall)
    write_csv_lf(out_dir / "summary_per_sensor.csv", per_sensor)
    write_csv_lf(out_dir / "summary_per_sequence.csv", per_sequence)
    write_csv_lf(out_dir / "fit_summary.csv", fit_summary)
    with gzip.open(out_dir / "frame_level_metrics.csv.gz", "wt", newline="") as f:
        fieldnames = [k for row in frame_rows for k in row.keys()]
        fieldnames = list(dict.fromkeys(fieldnames))
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(frame_rows)

    plot_source_rmse_bar(
        per_sensor,
        out_dir,
        "accfit_rjs_vs_vertex_rmse_bar.png",
        ("vertex_baseline", "accfit_per_sequence", "accfit_global"),
        "savgol9_p3_fd",
        "Accfit rJS vs vertex RMSE (SavGol-9 sensor force)",
    )
    plot_source_rmse_bar(
        per_sensor,
        out_dir,
        "accfit_rjs_vs_old_rjs_rmse_bar.png",
        ("old_footlock_rjs", "accfit_per_sequence", "accfit_global"),
        "savgol9_p3_fd",
        "Accfit rJS vs old footlock rJS RMSE (SavGol-9 sensor force)",
    )
    plot_fit_metric_bar(fit_summary, out_dir, "rJS_projected_norm", "rjs_norm_bar.png", "Projected rJS norm")
    plot_fit_metric_bar(fit_summary, out_dir, "projected_fit_improvement", "fit_improvement_bar.png", "Projected no-bias fit improvement")
    plot_fit_metric_bar(fit_summary, out_dir, "condition_number", "condition_number_bar.png", "Acceleration fit condition number")
    plot_timeseries_examples(examples, out_dir)
    for method in ("savgol9_p3_fd", "savgol15_p3_fd"):
        plot_scatter([r for r in rows if r["source"] in ("old_footlock_rjs", "accfit_per_sequence")], out_dir, method, PRIMARY_SPACE)
        plot_residual_hist([r for r in rows if r["source"] in ("old_footlock_rjs", "accfit_per_sequence")], out_dir, method, PRIMARY_SPACE)
    plot_boxplot([r for r in rows if r["source"] == "accfit_per_sequence"], out_dir, "savgol9_p3_fd", PRIMARY_SPACE)
    write_summary_md(out_dir / "SUMMARY.md", overall, per_sensor, fit_summary, config)
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "summary": str(out_dir / "SUMMARY.md")}, indent=2))


if __name__ == "__main__":
    main()
