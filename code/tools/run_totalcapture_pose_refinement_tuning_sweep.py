#!/usr/bin/env python3
"""Run bounded tuning sweeps for TotalCapture IMU-aware pose refinement."""

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REFINE_SCRIPT = ROOT / "code/tools/refine_totalcapture_pose_with_imu_acc_consistency.py"
DEFAULT_OUTPUT_ROOT = ROOT / "code/outputs"
SENSOR_NAMES = (
    "left_forearm",
    "right_forearm",
    "left_lower_leg",
    "right_lower_leg",
    "head",
    "pelvis",
)
FIT_SENSOR_PRESETS = {
    "all": {
        "fit": SENSOR_NAMES,
        "heldout": (),
    },
    "forearms_head": {
        "fit": ("left_forearm", "right_forearm", "head"),
        "heldout": ("left_lower_leg", "right_lower_leg"),
    },
    "lower_legs": {
        "fit": ("left_lower_leg", "right_lower_leg"),
        "heldout": ("left_forearm", "right_forearm", "head"),
    },
}
POSE_PRIOR = {"high": 500.0, "very_high": 2000.0}
TRAN_PRIOR = {"high": 5000.0, "very_high": 20000.0}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(ROOT / "data/dataset_work/TotalCapture_globalpose_official/test.pt"))
    p.add_argument("--rjs-path", default=str(ROOT / "code/outputs/totalcapture_accfit_rjs_synthesis_20260704_171103/rjs_accfit_global.pt"))
    p.add_argument("--output-dir", default="")
    p.add_argument("--stage", choices=("stage1", "stage2", "all"), default="all")
    p.add_argument("--force-stage2", action="store_true", help="Run Mode B even if Mode A has no passing config.")
    p.add_argument("--max-sequences", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=180)
    p.add_argument("--max-configs", type=int, default=0, help="Debug cap per stage after deterministic ordering.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--iterations", default="20,50,100")
    p.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    p.add_argument("--acc-weights", default="0.1,0.3,1.0")
    p.add_argument("--pose-priors", default="high,very_high")
    p.add_argument("--tran-priors", default="high,very_high")
    p.add_argument("--jerk-weights", default="1.0,10.0,50.0")
    p.add_argument("--losses", default="huber,charbonnier")
    p.add_argument("--signal-filters", default="raw,savgol9_p3,savgol15_p3")
    p.add_argument("--fit-presets", default="all,forearms_head,lower_legs")
    p.add_argument("--max-jerk-delta", type=float, default=0.0)
    p.add_argument("--max-pose-mean-deg", type=float, default=0.25)
    p.add_argument("--max-trans-mean", type=float, default=0.002)
    p.add_argument("--max-foot-sliding-delta", type=float, default=0.0)
    p.add_argument("--summary-only", action="store_true", help="Parse existing per_config_summary directories without rerunning.")
    return p.parse_args()


def split_csv(text, cast=str):
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return [cast(x) for x in vals]


def output_dir(args):
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = DEFAULT_OUTPUT_ROOT / f"totalcapture_pose_refinement_tuning_sweep_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_config_summary").mkdir(exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    return out


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def finite_float(value):
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def mean(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def config_grid(args, mode):
    configs = []
    for it, lr, acc_w, pose_label, tran_label, jerk_w, loss, fit_preset in itertools.product(
        split_csv(args.iterations, int),
        split_csv(args.lrs, float),
        split_csv(args.acc_weights, float),
        split_csv(args.pose_priors),
        split_csv(args.tran_priors),
        split_csv(args.jerk_weights, float),
        split_csv(args.losses),
        split_csv(args.fit_presets),
    ):
        fit = FIT_SENSOR_PRESETS[fit_preset]["fit"]
        heldout = FIT_SENSOR_PRESETS[fit_preset]["heldout"]
        configs.append(
            {
                "mode": mode,
                "iterations": it,
                "lr": lr,
                "acc_weight": acc_w,
                "pose_prior_label": pose_label,
                "pose_prior_weight": POSE_PRIOR[pose_label],
                "tran_prior_label": tran_label,
                "tran_prior_weight": TRAN_PRIOR[tran_label],
                "jerk_weight": jerk_w,
                "loss": loss,
                "fit_preset": fit_preset,
                "fit_sensors": ",".join(fit),
                "heldout_sensors": ",".join(heldout),
            }
        )
    if args.max_configs:
        configs = configs[: int(args.max_configs)]
    return configs


def command_for_config(args, cfg, out_dir):
    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--dataset",
        args.dataset,
        "--rjs-path",
        args.rjs_path,
        "--mode",
        cfg["mode"],
        "--max-sequences",
        str(args.max_sequences),
        "--max-frames",
        str(args.max_frames),
        "--iterations",
        str(cfg["iterations"]),
        "--lr",
        str(cfg["lr"]),
        "--robust-loss",
        cfg["loss"],
        "--lambda-acc",
        str(cfg["acc_weight"]),
        "--lambda-pose",
        str(cfg["pose_prior_weight"]),
        "--lambda-tran",
        str(cfg["tran_prior_weight"]),
        "--lambda-jerk",
        str(cfg["jerk_weight"]),
        "--fit-sensors",
        cfg["fit_sensors"],
        "--output-dir",
        str(out_dir),
        "--device",
        args.device,
    ]
    cmd += ["--heldout-sensors", cfg["heldout_sensors"]]
    return cmd


def run_config(args, cfg, idx, root):
    cfg_id = f"{idx:04d}_{cfg['mode']}_{cfg['fit_preset']}_it{cfg['iterations']}_lr{cfg['lr']}_acc{cfg['acc_weight']}_{cfg['loss']}_j{cfg['jerk_weight']}_{cfg['pose_prior_label']}_{cfg['tran_prior_label']}"
    out_dir = root / "per_config_summary" / cfg_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg)
    cfg["config_id"] = cfg_id
    cfg["output_dir"] = str(out_dir)
    cmd = command_for_config(args, cfg, out_dir)
    cfg["command"] = " ".join(cmd)
    (out_dir / "sweep_config.json").write_text(json.dumps(cfg, indent=2))
    if args.dry_run or args.summary_only:
        return cfg, 0
    print(f"[run] {cfg_id}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out_dir / "runner_stdout.log").write_text(proc.stdout)
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
    return cfg, proc.returncode


def row_for_filter(cfg, signal_filter):
    out = Path(cfg["output_dir"])
    if not (out / "summary_per_sequence.csv").exists():
        return {**cfg, "signal_filter": signal_filter, "verdict": "fail_missing_output"}
    per_seq = read_csv(out / "summary_per_sequence.csv")
    overall = read_csv(out / "summary_overall.csv")
    row_o = next((r for r in overall if r.get("mode") == cfg["mode"]), {})

    def rows(group):
        return [r for r in per_seq if r.get("mode") == cfg["mode"] and r.get("signal_filter") == signal_filter and r.get("sensor_group") == group]

    fit_rows = rows("fit")
    held_rows = rows("heldout")
    out_row = {
        **cfg,
        "signal_filter": signal_filter,
        "original_fit_acc_rmse": mean([finite_float(r.get("original_acc_rmse")) for r in fit_rows]),
        "refined_fit_acc_rmse": mean([finite_float(r.get("refined_acc_rmse")) for r in fit_rows]),
        "fit_delta": mean([finite_float(r.get("delta_acc_rmse")) for r in fit_rows]),
        "original_heldout_rmse": mean([finite_float(r.get("original_acc_rmse")) for r in held_rows]),
        "refined_heldout_rmse": mean([finite_float(r.get("refined_acc_rmse")) for r in held_rows]),
        "heldout_delta": mean([finite_float(r.get("delta_acc_rmse")) for r in held_rows]),
        "gyro_delta": finite_float(row_o.get("delta_gyro_rmse")),
        "jerk_delta": finite_float(row_o.get("delta_jerk")),
        "pose_mean_deg": finite_float(row_o.get("mean_pose_delta_deg")),
        "pose_max_deg": finite_float(row_o.get("max_pose_delta_deg")),
        "trans_mean_m": finite_float(row_o.get("mean_trans_delta")),
        "trans_max_m": finite_float(row_o.get("max_trans_delta")),
        "foot_sliding_delta": finite_float(row_o.get("delta_foot_sliding")),
    }
    verdict, reasons = verdict_for(out_row)
    out_row["verdict"] = verdict
    out_row["fail_reasons"] = "|".join(reasons)
    return out_row


def verdict_for(row):
    reasons = []
    if not row.get("heldout_sensors"):
        reasons.append("no_heldout_split")
    checks = [
        ("fit_not_improved", finite_float(row["fit_delta"]) < 0.0),
        ("heldout_regressed", finite_float(row["heldout_delta"]) <= 0.0),
        ("gyro_regressed", finite_float(row["gyro_delta"]) <= 0.0),
        ("jerk_regressed", finite_float(row["jerk_delta"]) <= verdict_for.max_jerk_delta),
        ("pose_too_large", finite_float(row["pose_mean_deg"]) <= verdict_for.max_pose_mean_deg),
        ("trans_too_large", finite_float(row["trans_mean_m"]) <= verdict_for.max_trans_mean),
        ("foot_sliding_regressed", finite_float(row["foot_sliding_delta"]) <= verdict_for.max_foot_sliding_delta),
    ]
    for name, ok in checks:
        if not ok:
            reasons.append(name)
    return ("pass" if not reasons else "fail"), reasons


def configure_verdict_thresholds(args):
    verdict_for.max_pose_mean_deg = float(args.max_pose_mean_deg)
    verdict_for.max_trans_mean = float(args.max_trans_mean)
    verdict_for.max_foot_sliding_delta = float(args.max_foot_sliding_delta)
    verdict_for.max_jerk_delta = float(args.max_jerk_delta)


def collect_rows(args, root, configs):
    rows = []
    for cfg in configs:
        for filt in split_csv(args.signal_filters):
            rows.append(row_for_filter(cfg, filt))
    return rows


def write_plots(root, rows):
    ok_rows = [r for r in rows if math.isfinite(finite_float(r.get("fit_delta")))]
    if not ok_rows:
        return

    def scatter(path, xkey, ykey, xlabel, ylabel):
        plt.figure(figsize=(6, 4))
        colors = ["tab:green" if r.get("verdict") == "pass" else "tab:red" for r in ok_rows]
        plt.scatter([finite_float(r[xkey]) for r in ok_rows], [finite_float(r[ykey]) for r in ok_rows], c=colors, alpha=0.75)
        plt.axhline(0, color="0.5", linewidth=1)
        plt.axvline(0, color="0.5", linewidth=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    scatter(root / "plots/sweep_acc_vs_jerk.png", "fit_delta", "jerk_delta", "fit acc RMSE delta", "jerk delta")
    scatter(root / "plots/sweep_heldout_vs_fit.png", "fit_delta", "heldout_delta", "fit acc RMSE delta", "held-out acc RMSE delta")
    scatter(root / "plots/sweep_pose_delta_vs_acc.png", "pose_mean_deg", "fit_delta", "pose mean delta deg", "fit acc RMSE delta")


def write_summary(root, rows, args, stage1_passed, ran_stage2):
    passes = [r for r in rows if r.get("verdict") == "pass"]
    fails = [r for r in rows if r.get("verdict") != "pass"]
    reason_counts = Counter()
    for row in fails:
        for reason in str(row.get("fail_reasons", "")).split("|"):
            if reason:
                reason_counts[reason] += 1
    lines = [
        "# TotalCapture Pose Refinement Tuning Sweep",
        "",
        "Diagnostic only: no network training and no PL/IK/VR changes.",
        "",
        "## Scope",
        "",
        f"- dataset: `{args.dataset}`",
        f"- max sequences / frames: `{args.max_sequences}` / `{args.max_frames}`",
        f"- max configs per stage: `{args.max_configs or 'full grid'}`",
        f"- Stage 1 passed: `{stage1_passed}`",
        f"- Stage 2 ran: `{ran_stage2}`",
        "",
        "## Result",
        "",
        f"- total verdict rows: `{len(rows)}`",
        f"- pass rows: `{len(passes)}`",
        f"- failed rows: `{len(fails)}`",
    ]
    if passes:
        lines += ["", "Best pass rows are in `best_configs_by_gate.csv`."]
    else:
        lines += [
            "",
            "No configuration passed all gates in this bounded run.",
            "",
            "Failure causes to inspect:",
        ]
        for reason, count in reason_counts.most_common():
            lines.append(f"- `{reason}`: {count}")
        lines += [
            "",
            "Interpretation candidates: acc target may be too noisy, second differences may make optimization unstable, translation-only may not explain the residual, pose/tran priors may be too strong or too weak, robust-loss gradients may be poorly scaled, or the next version may need low-frequency-only acceleration loss instead of raw acceleration loss.",
        ]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    configure_verdict_thresholds(args)
    root = output_dir(args)
    all_configs = []
    all_rows = []
    idx = 0

    run_stage1 = args.stage in ("stage1", "all")
    run_stage2_requested = args.stage in ("stage2", "all")
    stage1_passed = False
    ran_stage2 = False

    if run_stage1:
        stage1_configs = config_grid(args, "A")
        stage1_run_configs = []
        for cfg in stage1_configs:
            cfg, code = run_config(args, cfg, idx, root)
            idx += 1
            cfg["returncode"] = code
            all_configs.append(cfg)
            stage1_run_configs.append(cfg)
        stage1_rows = collect_rows(args, root, stage1_run_configs)
        all_rows.extend(stage1_rows)
        stage1_passed = any(r.get("verdict") == "pass" for r in stage1_rows)

    if run_stage2_requested and (args.force_stage2 or args.stage == "stage2" or stage1_passed):
        ran_stage2 = True
        stage2_configs = config_grid(args, "B")
        stage2_run_configs = []
        for cfg in stage2_configs:
            cfg, code = run_config(args, cfg, idx, root)
            idx += 1
            cfg["returncode"] = code
            all_configs.append(cfg)
            stage2_run_configs.append(cfg)
        all_rows.extend(collect_rows(args, root, stage2_run_configs))

    passes = [r for r in all_rows if r.get("verdict") == "pass"]
    fails = [r for r in all_rows if r.get("verdict") != "pass"]
    passes.sort(key=lambda r: (finite_float(r.get("fit_delta")), finite_float(r.get("heldout_delta")), finite_float(r.get("jerk_delta"))))
    write_csv(root / "sweep_summary.csv", all_rows)
    write_csv(root / "best_configs_by_gate.csv", passes)
    write_csv(root / "failed_configs.csv", fails)
    write_plots(root, all_rows)
    write_summary(root, all_rows, args, stage1_passed, ran_stage2)
    (root / "sweep_config.json").write_text(json.dumps({"args": vars(args), "num_configs": len(all_configs)}, indent=2))
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
