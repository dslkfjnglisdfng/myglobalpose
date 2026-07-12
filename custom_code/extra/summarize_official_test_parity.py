"""Evaluate saved predictions with the original test.py MotionEvaluator and windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch


METRIC_NAMES = [
    "L SIP Err", "L Angle Err", "L Joint Err", "L Vertex Err",
    "G SIP Err", "G Angle Err", "G Joint Err", "G Vertex Err",
    "Root Jitter", "Joint Jitter",
]


def load_test(repo_root: Path):
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    return importlib.import_module("test")


def select_data(data, dataset):
    if dataset == "dip":
        return data
    indices = [i for i, name in enumerate(data["name"]) if str(name).startswith("s5_")]
    return {key: [value[i] for i in indices] if isinstance(value, list) else value for key, value in data.items()}


def translation_windows(pred, target):
    # Exact algorithm from baseline test.py compare_realimu, kept intentionally literal.
    move_distance_t = torch.zeros(target.shape[0])
    v = (target[1:] - target[:-1]).norm(dim=1)
    for j in range(len(v)):
        move_distance_t[j + 1] = move_distance_t[j] + v[j]
    result = {}
    for window_size in range(1, 8):
        frame_pairs = []
        start, end = 0, 1
        while end < len(move_distance_t):
            if move_distance_t[end] - move_distance_t[start] < window_size:
                end += 1
            else:
                if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
                    frame_pairs.append((start, end))
                start += 1
        errs = []
        for start, end in frame_pairs:
            vel_p = pred[end] - pred[start]
            vel_t = target[end] - target[start]
            errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
        result[str(window_size)] = torch.stack(errs) if errs else torch.empty(0)
    return result


def evaluate(repo_root, dataset, predictions_path):
    official_test = load_test(repo_root)
    evaluator = official_test.MotionEvaluator()
    source = (repo_root / "test.py").read_bytes()
    dataset_file = "dipimu.pt" if dataset == "dip" else "totalcapture_officalib.pt"
    data = select_data(torch.load(repo_root / "data" / "test_datasets" / dataset_file), dataset)
    predictions = torch.load(predictions_path, map_location="cpu")
    per_sequence = []
    errors = []
    window_values = {str(i): [] for i in range(1, 8)}
    for i, name in enumerate(data["name"]):
        pose_t = official_test.art.math.axis_angle_to_rotation_matrix(data["pose"][i]).view(-1, 24, 3, 3)
        pose_p = predictions["pose"][i]
        tran_p = predictions["tran"][i]
        tran_t = data["tran"][i]
        err = evaluator(pose_p, pose_t, tran_p, tran_t).cpu()
        errors.append(err)
        row = {METRIC_NAMES[j]: float(err[j, 0]) for j in range(len(METRIC_NAMES))}
        row["sequence"] = str(name)
        if dataset == "totalcapture":
            windows = translation_windows(tran_p, tran_t)
            for key, values in windows.items():
                row[f"Translation {key}m"] = float(values.mean()) if len(values) else None
                if len(values):
                    window_values[key].append(values)
        per_sequence.append(row)
    stacked = torch.stack(errors).mean(dim=0)
    aggregate = {METRIC_NAMES[j]: {"mean": float(stacked[j, 0]), "std": float(stacked[j, 1])} for j in range(len(METRIC_NAMES))}
    if dataset == "totalcapture":
        for key, values in window_values.items():
            seq_means = torch.stack([v.mean() for v in values])
            aggregate[f"Translation {key}m"] = {
                "mean": float(seq_means.mean()),
                "std": float(torch.stack([v.std() for v in values]).mean()),
            }
    return {
        "dataset": dataset,
        "test_py_sha256": hashlib.sha256(source).hexdigest(),
        "motion_evaluator_class": "test.MotionEvaluator",
        "sequences": list(data["name"]),
        "aggregate": aggregate,
        "per_sequence": per_sequence,
    }


def rotation_diff_deg(a, b):
    rel = a.transpose(-1, -2) @ b
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1, 1)))


def parity(baseline_path, current_path, names):
    baseline, current = torch.load(baseline_path, map_location="cpu"), torch.load(current_path, map_location="cpu")
    rows = {}
    for index, name in enumerate(names):
        pose_a, pose_b = baseline["pose"][index], current["pose"][index]
        tran_a, tran_b = baseline["tran"][index], current["tran"][index]
        rot = rotation_diff_deg(pose_a, pose_b)
        rows[name] = {
            "pose_max_abs_diff": float((pose_a - pose_b).abs().max()),
            "pose_mean_abs_diff": float((pose_a - pose_b).abs().mean()),
            "pose_rotation_max_diff_deg": float(rot.max()),
            "pose_rotation_mean_diff_deg": float(rot.mean()),
            "tran_max_abs_diff_m": float((tran_a - tran_b).abs().max()),
            "tran_mean_abs_diff_m": float((tran_a - tran_b).abs().mean()),
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("dip", "totalcapture"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.repo_root.resolve(), args.dataset, args.predictions.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
