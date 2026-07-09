#!/usr/bin/env python3
"""Materialize the official DIPtest real-GlobalPose cache at the dataset_work path."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


GLOBALPOSE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = GLOBALPOSE_ROOT / "data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only_with_offset_r/baseline_cache_manifest.json"
DEFAULT_OUTPUT = GLOBALPOSE_ROOT / "data/dataset_work/L4Cache/prephysics_pose_velocity_diptest_official_neural_only_offset_r/baseline_cache_manifest.json"
REQUIRED_FIELDS = ("name", "v_root_vr", "stationary_prob")


def resolve(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidates = []
    if base is not None:
        candidates.append(base.parent / p)
    candidates.append(GLOBALPOSE_ROOT / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_shard(path: Path) -> tuple[int, int, str]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in REQUIRED_FIELDS if k not in data or not data[k]]
    if missing:
        raise KeyError(f"{path} missing required fields: {missing}")
    q_key = "q75_prephysics" if data.get("q75_prephysics") else "q75_baseline" if data.get("q75_baseline") else ""
    if not q_key:
        raise KeyError(f"{path} must contain q75_prephysics or q75_baseline")
    names = [str(x) for x in data["name"]]
    if len(names) != len(set(names)):
        dup = sorted({x for x in names if names.count(x) > 1})
        raise KeyError(f"{path} has duplicate DIP sequence names: {dup[:10]}")
    frame_counts = [int(x.shape[0]) for x in data[q_key]]
    for key in ("v_root_vr", "stationary_prob"):
        bad = [name for name, n, x in zip(names, frame_counts, data[key]) if int(x.shape[0]) != n]
        if bad:
            raise KeyError(f"{path} has {key} length mismatch for sequences: {bad[:10]}")
    return len(names), int(sum(frame_counts)), q_key


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = resolve(args.source_manifest)
    source = load_manifest(source_manifest)
    output_manifest = resolve(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    cache_files = []
    total_sequences = 0
    total_frames = 0
    for idx, item in enumerate(source.get("cache_files", [])):
        src_shard = resolve(item["path"], source_manifest)
        seqs, frames, q_key = validate_shard(src_shard)
        dst_name = f"baseline_cache_shard{idx:05d}.pt"
        dst_shard = output_manifest.parent / dst_name
        if src_shard.resolve() != dst_shard.resolve():
            shutil.copy2(src_shard, dst_shard)
        rel = dst_shard.relative_to(GLOBALPOSE_ROOT)
        row = dict(item)
        row.update({
            "path": str(rel),
            "source_path": str(src_shard),
            "num_sequences": seqs,
            "num_frames": frames,
            "globalpose_cache_type": q_key,
            "failures": [],
        })
        cache_files.append(row)
        total_sequences += seqs
        total_frames += frames

    if not cache_files:
        raise RuntimeError(f"No cache_files found in {source_manifest}")

    manifest = dict(source)
    manifest.update({
        "source_manifest": str(source_manifest),
        "source_input": source.get("source_input", "data/test_datasets/dipimu.pt"),
        "cache_files": cache_files,
        "num_sequences": total_sequences,
        "num_frames": total_frames,
        "dataset_split": "dip_test",
        "globalpose_cache_type": q_key,
        "v_root_vr_source": "globalpose_pipeline_v_root_vr",
        "contact_mode": "real_globalpose_stationary_prob",
        "features_gp_contract": "q_gp_root_lower + root_translation_velocity_gp + optional contact only",
        "sequence_name_alignment": "DIPtest loader names must match cache name values exactly; missing keys are fatal in the evaluator.",
        "root_velocity_evaluation": "disabled_or_diagnostic_only_for_diptest",
        "root_acceleration_evaluation": "disabled_or_diagnostic_only_for_diptest",
        "model_input_contains_qdot_fd_from_gp": False,
        "model_input_contains_qddot_fd_from_gp": False,
        "fd_diagnostic_eval_only": True,
        "fd_diagnostic_is_native_globalpose_output": False,
        "command": args.command,
    })
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "manifest": str(output_manifest),
        "num_sequences": total_sequences,
        "num_frames": total_frames,
        "source_manifest": str(source_manifest),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", default=str(DEFAULT_SOURCE))
    p.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()
    import shlex
    import sys
    args.command = " ".join(shlex.quote(x) for x in sys.argv)
    return args


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
