"""Evaluate full-TC predictions with the unchanged official evaluator contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import torch

from run_official_test_full_tc import DATASETS, inventory


METRIC_NAMES = [
    "L SIP Err", "L Angle Err", "L Joint Err", "L Vertex Err",
    "G SIP Err", "G Angle Err", "G Joint Err", "G Vertex Err",
    "Root Jitter", "Joint Jitter",
]


def load_test(repo_root: Path):
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    return importlib.import_module("test")


def translation_windows(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    # Exact start/end pair algorithm from baseline test.py compare_realimu.
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
                if not frame_pairs or frame_pairs[-1][1] != end:
                    frame_pairs.append((start, end))
                start += 1
        errs = []
        for start, end in frame_pairs:
            vel_p = pred[end] - pred[start]
            vel_t = target[end] - target[start]
            errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
        result[str(window_size)] = torch.stack(errs) if errs else torch.empty(0)
    return result


def evaluate(repo_root: Path, calibration: str, predictions_path: Path) -> dict:
    official_test = load_test(repo_root)
    evaluator = official_test.MotionEvaluator()
    filename, dataset_name = DATASETS[calibration]
    dataset_path = repo_root / "data" / "test_datasets" / filename
    data = torch.load(dataset_path)
    predictions = torch.load(predictions_path, map_location="cpu")
    if len(predictions["pose"]) != len(data["pose"]) or len(predictions["tran"]) != len(data["pose"]):
        raise RuntimeError("prediction sequence count does not match complete release cache")

    names = data.get("name")
    per_sequence, errors = [], []
    window_values = {str(i): [] for i in range(1, 8)}
    for i in range(len(data["pose"])):
        pose_t = official_test.art.math.axis_angle_to_rotation_matrix(data["pose"][i]).view(-1, 24, 3, 3)
        pose_p, tran_p, tran_t = predictions["pose"][i], predictions["tran"][i], data["tran"][i]
        if pose_p.shape[0] != data["pose"][i].shape[0] or tran_p.shape[0] != data["pose"][i].shape[0]:
            raise RuntimeError(f"frame count mismatch at sequence {i}")
        err = evaluator(pose_p, pose_t, tran_p, tran_t).cpu()
        errors.append(err)
        row = {
            "sequence_index": i,
            "sequence": str(names[i]) if names is not None else str(i),
            "frames": int(data["pose"][i].shape[0]),
            **{METRIC_NAMES[j]: float(err[j, 0]) for j in range(len(METRIC_NAMES))},
        }
        windows = translation_windows(tran_p, tran_t)
        for key, values in windows.items():
            row[f"Translation {key}m"] = float(values.mean()) if len(values) else None
            row[f"Translation {key}m pair_count"] = int(len(values))
            if len(values):
                window_values[key].append(values)
        per_sequence.append(row)

    stacked = torch.stack(errors).mean(dim=0)
    aggregate = {METRIC_NAMES[j]: {"mean": float(stacked[j, 0]), "std": float(stacked[j, 1])} for j in range(len(METRIC_NAMES))}
    for key, values in window_values.items():
        seq_means = torch.stack([value.mean() for value in values])
        aggregate[f"Translation {key}m"] = {
            "mean": float(seq_means.mean()),
            "std": float(torch.stack([value.std() for value in values]).mean()),
            "sequence_count": len(values),
        }
    aggregate["Translation Drift 7m percent"] = {
        "mean": aggregate["Translation 7m"]["mean"] / 7 * 100,
        "std": aggregate["Translation 7m"]["std"] / 7 * 100,
    }
    return {
        "calibration": calibration,
        "dataset_name": dataset_name,
        "dataset_inventory": inventory(data, dataset_path),
        "test_py_sha256": hashlib.sha256((repo_root / "test.py").read_bytes()).hexdigest(),
        "motion_evaluator_class": "test.MotionEvaluator",
        "translation_algorithm": "test.compare_realimu exact start/end frame-pair construction",
        "aggregate": aggregate,
        "per_sequence": per_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--calibration", choices=tuple(DATASETS), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.repo_root.resolve(), args.calibration, args.predictions.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
