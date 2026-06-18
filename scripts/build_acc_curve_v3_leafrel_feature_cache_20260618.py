import argparse
import json
import shutil
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acc_curve_v3_leafrel_train import (
    ACC_CURVE_V3_INPUT_SIZE,
    ACC_CURVE_V3_LEAF_SENSOR_NAMES,
    ACC_CURVE_V3_STATE_DIM,
    FEATURE_CACHE_EXPERIMENT,
    LEAF_INDICES,
    ROOT_INDEX,
    SOURCE_V4_EXPERIMENT,
    build_record,
    load_manifest,
    record_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build reusable AccCurve v3 leaf-relative feature cache.")
    parser.add_argument(
        "--source-manifest",
        default="data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/cache_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        default="data/dataset_work/AccCurveV3LeafRelCausalButter_20260618",
    )
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=0,
        help="Optional max sequences per (dataset, split), useful for smoke caches.",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_shard(path, records, shard_index):
    payload = {
        "experiment": FEATURE_CACHE_EXPERIMENT,
        "shard_index": int(shard_index),
        "records": records,
        "input_size": ACC_CURVE_V3_INPUT_SIZE,
        "state_dim": ACC_CURVE_V3_STATE_DIM,
        "root_index": ROOT_INDEX,
        "leaf_indices": LEAF_INDICES,
        "leaf_sensor_names": ACC_CURVE_V3_LEAF_SENSOR_NAMES,
        "root_excluded_from_prediction_loss_metric": True,
        "feature_layout": (
            "aIMU_leaf_rel_raw[15] + aIMU_leaf_rel_butter2_4hz[15] + "
            "raw_minus_butter[15] + wM[18] + RMB_6d[36]"
        ),
        "base_key": "aIMU_leaf_rel_butter2_4hz",
        "target_key": "aGT_leaf_rel_butter2_4hz",
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    started = time.time()
    source_manifest, source_items = load_manifest(args.source_manifest)
    if source_manifest.get("experiment") != SOURCE_V4_EXPERIMENT:
        raise ValueError(f"Expected source v4 manifest, got {source_manifest.get('experiment')!r}")
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"{output_root} is not empty; pass --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    shard_dir = output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cache_files = []
    shard_records = []
    counts = {}
    total_valid = 0
    total_frames = 0
    failures = []
    selected_items = []
    group_seen = {}
    for item in source_items:
        if args.max_sequences and len(selected_items) >= args.max_sequences:
            break
        if args.max_per_group and isinstance(item, dict):
            key = (item.get("dataset"), item.get("split"))
            if group_seen.get(key, 0) >= args.max_per_group:
                continue
            group_seen[key] = group_seen.get(key, 0) + 1
        selected_items.append(item)
    limit = len(selected_items)
    for idx, item in enumerate(selected_items, start=1):
        try:
            path = record_path(item)
            record = build_record(path, item)
            compact = {
                "name": record["name"],
                "dataset": record["dataset"],
                "split": record["split"],
                "source_path": record["path"],
                "feature": record["feature"].contiguous(),
                "base": record["base"].contiguous(),
                "target": record["target"].contiguous(),
                "valid_mask": record["valid_mask"].contiguous(),
                "num_frames": int(record["num_frames"]),
            }
            shard_records.append(compact)
            key = (record["dataset"], record["split"])
            counts[key] = counts.get(key, 0) + 1
            total_valid += int(record["valid_mask"].sum())
            total_frames += int(record["num_frames"])
        except Exception as exc:
            failures.append({"index": idx, "item": item, "error": repr(exc)})
        if shard_records and (len(shard_records) >= args.shard_size or idx == limit):
            shard_path = shard_dir / f"acc_curve_v3_leafrel_features_shard{len(cache_files):05d}.pt"
            save_shard(shard_path, shard_records, len(cache_files))
            cache_files.append({
                "path": str(shard_path),
                "num_sequences": len(shard_records),
            })
            shard_records = []
        if args.progress_every and idx % args.progress_every == 0:
            print(json.dumps({"processed": idx, "failures": len(failures), "shards": len(cache_files)}))
    if shard_records:
        shard_path = shard_dir / f"acc_curve_v3_leafrel_features_shard{len(cache_files):05d}.pt"
        save_shard(shard_path, shard_records, len(cache_files))
        cache_files.append({"path": str(shard_path), "num_sequences": len(shard_records)})
    datasets = {}
    for (dataset, split), count in sorted(counts.items()):
        datasets.setdefault(dataset, {})[split] = count
    manifest = {
        "experiment": FEATURE_CACHE_EXPERIMENT,
        "source_experiment": source_manifest.get("experiment"),
        "source_manifest": str(args.source_manifest),
        "output_root": str(output_root),
        "created_at_unix": time.time(),
        "elapsed_sec": time.time() - started,
        "cache_files": cache_files,
        "num_sequences": sum(item["num_sequences"] for item in cache_files),
        "num_failures": len(failures),
        "failures": failures,
        "datasets": datasets,
        "total_frames": total_frames,
        "valid_frames": total_valid,
        "input_size": ACC_CURVE_V3_INPUT_SIZE,
        "state_dim": ACC_CURVE_V3_STATE_DIM,
        "root_index": ROOT_INDEX,
        "leaf_indices": LEAF_INDICES,
        "leaf_sensor_names": list(ACC_CURVE_V3_LEAF_SENSOR_NAMES),
        "root_excluded_from_prediction_loss_metric": True,
        "frame": "model/world frame M",
        "smoothing": {
            "type": "causal_butterworth",
            "order": 2,
            "cutoff_hz": 4.0,
            "fps": 60,
            "zero_lookahead": True,
        },
        "feature_layout": (
            "aIMU_leaf_rel_raw[15] + aIMU_leaf_rel_butter2_4hz[15] + "
            "raw_minus_butter[15] + wM[18] + RMB_6d[36]"
        ),
        "base_key": "aIMU_leaf_rel_butter2_4hz",
        "target_key": "aGT_leaf_rel_butter2_4hz",
        "normalization_policy": "fit feature z-score from AMASS train split only during training",
    }
    (output_root / "cache_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_root / "README.md").write_text(
        "# AccCurve v3 Leaf-Relative Causal Butterworth Feature Cache\n\n"
        "Reusable project-level cache derived from the v4 leaf-relative causal Butterworth residual audit.\n"
        "It stores 99D v1-style AccCurve features plus 15D base/target leaf-relative acceleration.\n"
        "Root IMU index 5 is reference-only and is excluded from prediction, loss, and metrics.\n"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
