#!/usr/bin/env python3
"""Bounded alternating pose smoothing and global rJS refit for TotalCapture."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from l4_rawlike_se3_calibration import robust_rotation_mean  # noqa: E402
from l4_sensor_offset_utils import (  # noqa: E402
    FPS,
    GRAVITY_WORLD,
    IMU_JOINTS,
    SENSOR_NAMES,
    moving_average,
    second_derivative,
)
from refine_totalcapture_pose_kalman_style_smoother import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_OUT,
    DEFAULT_RJS,
    aggregate,
    finite_slice,
    load_rjs,
    matvec,
    metric_rows,
    optimize_sequence,
    prepare_sequence,
    write_csv,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--initial-rjs-path", type=Path, default=DEFAULT_RJS)
    p.add_argument("--rjs-method", default="savgol9_p3_fd")
    p.add_argument("--rjs-field", default="r_JS_projected")
    p.add_argument("--measurement", default="centered_ma21")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-sequences", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--trim", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--max-norm", type=float, default=0.25)
    p.add_argument("--lambda-acc", type=float, default=1.0)
    p.add_argument("--lambda-gyro", type=float, default=0.02)
    p.add_argument("--lambda-pose", type=float, default=500.0)
    p.add_argument("--lambda-tran", type=float, default=8000.0)
    p.add_argument("--lambda-vel", type=float, default=0.05)
    p.add_argument("--lambda-qdd", type=float, default=0.01)
    p.add_argument("--lambda-jerk", type=float, default=0.01)
    p.add_argument("--lambda-foot", type=float, default=0.02)
    p.add_argument("--max-pose-delta-deg", type=float, default=3.0)
    p.add_argument("--max-tran-delta", type=float, default=0.01)
    p.add_argument("--delta-smooth-window", type=int, default=21)
    return p.parse_args()


def save_rjs(path, rjs, R_JS, args):
    torch.save(
        {
            "method": [args.rjs_method],
            args.rjs_field: {args.rjs_method: rjs.detach().cpu()},
            "R_JS": {args.rjs_method: R_JS.detach().cpu()},
            "config": {"source": "alternate_totalcapture_rjs_pose_smoothing.py", "measurement": args.measurement},
        },
        path,
    )


def refit_global_rjs(sequences, payload, args):
    new_rjs, new_R = [], []
    for sid in range(6):
        R_est = []
        As, ys = [], []
        for seq, item in zip(sequences, payload):
            pose_seq = {"pose": item["pose_clean"], "tran": item["tran_clean"]}
            # Use the same FK helper as the smoother by running one zero-iter
            # prediction through optimize prerequisites would be overkill, so
            # derive R/p from the stored cleaned trajectory via prepare path.
            from refine_totalcapture_pose_kalman_style_smoother import differentiable_fk

            R_all, J_all = differentiable_fk(item["pose_clean"].to(args.device), item["tran_clean"].to(args.device), torch.device(args.device))
            R_wj = R_all[:, list(IMU_JOINTS)].detach().cpu()
            p_wj = J_all[:, list(IMU_JOINTS)].detach().cpu()
            R_obs = seq["R_WS_obs"][:, sid]
            R_est.append(robust_rotation_mean(R_wj[:, sid].transpose(-1, -2).matmul(R_obs)))
            R_JS = R_est[-1]
            R_WS = R_wj[:, sid].matmul(R_JS.view(1, 3, 3))
            R_SW = R_WS.transpose(-1, -2)
            ddot_p = second_derivative(moving_average(p_wj[:, sid], 21), fps=FPS, mode="centered")
            ddot_R = second_derivative(moving_average(R_wj[:, sid], 21), fps=FPS, mode="centered")
            base = matvec(R_SW, ddot_p - GRAVITY_WORLD.view(1, 3))
            A = R_SW.matmul(ddot_R)
            aS = moving_average(seq["aS"][:, sid], 21)
            sl, _ = finite_slice(seq["n"], args.trim)
            valid = torch.isfinite(A[sl]).all(dim=(-1, -2)) & torch.isfinite(base[sl]).all(dim=-1) & torch.isfinite(aS[sl]).all(dim=-1)
            As.append(A[sl][valid])
            ys.append(aS[sl][valid] - base[sl][valid])
        R_JS = robust_rotation_mean(torch.stack(R_est))
        A_all = torch.cat(As, dim=0)
        y_all = torch.cat(ys, dim=0)
        lhs = A_all.reshape(-1, 3).T.matmul(A_all.reshape(-1, 3)) + torch.eye(3) * float(args.ridge)
        rhs = A_all.reshape(-1, 3).T.matmul(y_all.reshape(-1))
        r = torch.linalg.solve(lhs, rhs)
        norm = r.norm().clamp_min(1e-12)
        if float(norm) > float(args.max_norm):
            r = r * (float(args.max_norm) / norm)
        new_rjs.append(r.float())
        new_R.append(R_JS.float())
    return torch.stack(new_rjs), torch.stack(new_R)


def summary_md(path, rows):
    lines = [
        "# TotalCapture rJS/Pose Alternating Diagnostic",
        "",
        "Bounded diagnostic only. Each round fixes rJS, runs the simplified smooth pose optimizer, then refits global rJS from the cleaned pose.",
        "",
        "| round | group | before RMSE | after RMSE | delta RMSE |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        if "before_acc_rmse" in r:
            lines.append(f"| {r['round']} | {r['sensor_group']} | {r['before_acc_rmse']:.6f} | {r['after_acc_rmse']:.6f} | {r['delta_acc_rmse']:.6f} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out = args.output_dir or (DEFAULT_OUT / f"totalcapture_rjs_pose_alternating_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.dataset, map_location="cpu")
    rjs, R_JS, _config = load_rjs(args.initial_rjs_path, args.rjs_method, args.rjs_field)
    count = len(data["name"])
    if args.max_sequences:
        count = min(count, int(args.max_sequences))
    sequences = [prepare_sequence(data, i, args) for i in range(count)]
    all_rows, round_rows = [], []
    for round_idx in range(1, int(args.rounds) + 1):
        payload = []
        rows = []
        for seq in sequences:
            print(f"[round {round_idx}] smooth {seq['name']}", flush=True)
            result = optimize_sequence(seq, rjs, R_JS, args)
            rows.extend(metric_rows(seq, result, args))
            payload.append({"sequence_id": seq["name"], "pose_clean": result["pose_clean"], "tran_clean": result["tran_clean"]})
        overall = aggregate(rows)
        for row in overall:
            row["round"] = round_idx
            round_rows.append(row)
        all_rows.extend({"round": round_idx, **row} for row in rows)
        rjs, R_JS = refit_global_rjs(sequences, payload, args)
        save_rjs(out / f"round_{round_idx}_rjs_global.pt", rjs, R_JS, args)
    write_csv(out / "rjs_iteration_summary.csv", round_rows)
    write_csv(out / "per_sequence_round_summary.csv", all_rows)
    write_json(out / "config.json", {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "sensor_names": list(SENSOR_NAMES)})
    summary_md(out / "SUMMARY.md", round_rows)
    print(json.dumps({"status": "ok", "output_dir": str(out), "summary": str(out / "SUMMARY.md")}, indent=2))


if __name__ == "__main__":
    main()
