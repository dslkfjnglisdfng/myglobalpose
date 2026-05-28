import argparse
import json
from pathlib import Path

import torch
import tqdm


REQUIRED_CACHE_FIELDS = (
    "name",
    "num_frames",
    "q75_prephysics",
    "pose_prephysics",
    "v_root_vr",
    "stationary_prob",
    "aM",
    "wM",
    "RMB",
    "q75_gt",
    "pose_gt",
    "tran_gt",
)
REQUIRED_AUG_FIELDS = ("name", "aM", "wM", "RMB")
OFFSET_KEYS = ("imu_offset_r", "r_JS")


def resolve_path(path, base=None):
    out = Path(path)
    if not out.is_absolute() and base is not None and not out.exists():
        out = Path(base) / out
    return out


def load_cache_specs(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    specs = []
    for item in manifest["cache_files"]:
        cache_path = resolve_path(item["path"], path.parent)
        specs.append((cache_path, item))
    return manifest, specs


def load_augmented_manifest(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    name_to_source = {}
    specs = []
    for shard_idx, item in enumerate(manifest["shards"]):
        shard_path = resolve_path(item["path"], path.parent)
        names = [str(name) for name in item.get("names", [])]
        specs.append({"index": shard_idx, "path": shard_path, "names": names, "manifest": item})
        for seq_idx, name in enumerate(names):
            name_to_source[name] = (shard_idx, seq_idx)
    return manifest, specs, name_to_source


def empty_output():
    return {
        "name": [],
        "source_name": [],
        "source_cache_name": [],
        "source_aug_name": [],
        "view_type": [],
        "pair_id": [],
        "q75_prephysics": [],
        "pose_prephysics": [],
        "v_root_vr": [],
        "stationary_prob": [],
        "aM": [],
        "wM": [],
        "RMB": [],
        "q75_gt": [],
        "pose_gt": [],
        "tran_gt": [],
        "offset_r": [],
        "imu_offset_r": [],
        "num_frames": [],
    }


def select_record(data, idx):
    return {
        key: data[key][idx]
        for key in data
        if isinstance(data[key], list) and len(data[key]) == len(data["name"])
    }


def check_required_fields(data, required, label):
    missing = [key for key in required if key not in data or not data[key]]
    if missing:
        raise KeyError(f"{label} missing required fields: {missing}")


def finite_record(record, keys):
    for key in keys:
        value = record[key]
        if torch.is_tensor(value) and not torch.isfinite(value.float()).all():
            return False, key
    return True, ""


def diff_stats(a, b):
    d = a.float() - b.float()
    flat = d.reshape(d.shape[0], -1)
    norms = flat.norm(dim=-1)
    return {
        "max_abs": float(d.abs().max()),
        "norm_mean": float(norms.mean()),
        "norm_median": float(norms.median()),
        "norm_p90": float(torch.quantile(norms, 0.90)),
        "norm_max": float(norms.max()),
    }


def offset_from_aug(aug_data, seq_idx, original=False):
    if original and "original_imu_offset_r" in aug_data:
        return aug_data["original_imu_offset_r"][seq_idx].float(), "original_imu_offset_r"
    key = next((item for item in OFFSET_KEYS if item in aug_data), None)
    if key is None:
        raise KeyError(f"augmented shard missing one of {OFFSET_KEYS}")
    return aug_data[key][seq_idx].float(), key


def append_view(out, name, source_name, source_cache_name, source_aug_name, view_type, pair_id, record, offset_r):
    out["name"].append(name)
    out["source_name"].append(source_name)
    out["source_cache_name"].append(source_cache_name)
    out["source_aug_name"].append(source_aug_name)
    out["view_type"].append(view_type)
    out["pair_id"].append(pair_id)
    for key in (
        "q75_prephysics",
        "pose_prephysics",
        "v_root_vr",
        "stationary_prob",
        "aM",
        "wM",
        "RMB",
        "q75_gt",
        "pose_gt",
        "tran_gt",
        "num_frames",
    ):
        value = record[key]
        out[key].append(value.clone() if torch.is_tensor(value) else int(value))
    offset_r = offset_r.float().clone()
    out["offset_r"].append(offset_r)
    out["imu_offset_r"].append(offset_r)


def aggregate_diffs(items):
    if not items:
        return {}
    keys = items[0].keys()
    out = {}
    for key in keys:
        vals = [float(item[key]) for item in items]
        out[key] = {
            "mean": sum(vals) / len(vals),
            "max": max(vals),
        }
    return out


def process(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest, cache_specs = load_cache_specs(args.original_cache)
    aug_manifest, aug_specs, aug_index = load_augmented_manifest(args.augmented_input)
    aug_specs_by_index = {spec["index"]: spec for spec in aug_specs}
    if args.shard_indices:
        allowed = {int(item) for item in args.shard_indices.split(",") if item.strip()}
    else:
        allowed = None

    manifest_files = []
    diagnostics = []
    skipped = []
    total_pairs = 0
    total_records = 0
    total_frames = 0
    am_diffs = []
    wm_diffs = []
    rmb_diffs = []
    pose_diffs = []
    tran_diffs = []
    q75_diffs = []
    all_offsets = []

    for cache_idx, (cache_path, cache_item) in enumerate(cache_specs):
        cache_data = torch.load(cache_path, map_location="cpu")
        check_required_fields(cache_data, REQUIRED_CACHE_FIELDS, str(cache_path))
        out = empty_output()
        needed_by_shard = {}
        for seq_idx, name in enumerate(cache_data["name"]):
            name = str(name)
            if name not in aug_index:
                skipped.append({"name": name, "reason": "missing_augmented_sequence"})
                continue
            shard_idx, aug_seq_idx = aug_index[name]
            if allowed is not None and shard_idx not in allowed:
                skipped.append({"name": name, "reason": f"augmented_shard_{shard_idx}_not_selected"})
                continue
            needed_by_shard.setdefault(shard_idx, []).append((seq_idx, aug_seq_idx, name))

        for shard_idx, items in needed_by_shard.items():
            spec = aug_specs_by_index[shard_idx]
            aug_data = torch.load(spec["path"], map_location="cpu")
            check_required_fields(aug_data, REQUIRED_AUG_FIELDS, str(spec["path"]))
            if not any(key in aug_data for key in OFFSET_KEYS):
                raise KeyError(f"{spec['path']} missing imu_offset_r/r_JS")
            for seq_idx, aug_seq_idx, source_name in tqdm.tqdm(items, desc=f"overlay shard {shard_idx}"):
                try:
                    original = select_record(cache_data, seq_idx)
                    length = int(cache_data["num_frames"][seq_idx])
                    aug_len = int(aug_data["aM"][aug_seq_idx].shape[0])
                    if length != aug_len:
                        skipped.append({
                            "name": source_name,
                            "reason": "frame_length_mismatch",
                            "cache_frames": length,
                            "augmented_frames": aug_len,
                        })
                        continue
                    original_offset, original_offset_source = offset_from_aug(aug_data, aug_seq_idx, original=True)
                    aug_offset, aug_offset_source = offset_from_aug(aug_data, aug_seq_idx, original=False)
                    if original_offset.shape != (6, 3) or aug_offset.shape != (6, 3):
                        skipped.append({
                            "name": source_name,
                            "reason": "bad_offset_shape",
                            "original_shape": list(original_offset.shape),
                            "augmented_shape": list(aug_offset.shape),
                        })
                        continue

                    aug_record = dict(original)
                    aug_record["aM"] = aug_data["aM"][aug_seq_idx].float().clone()
                    # Keep the cache wM/RMB used by the old prephysics path. The augmented
                    # shard values are checked below and should be equivalent.
                    aug_record["wM"] = original["wM"].float().clone()
                    aug_record["RMB"] = original["RMB"].float().clone()

                    required_tensor_keys = (
                        "q75_prephysics",
                        "pose_prephysics",
                        "v_root_vr",
                        "stationary_prob",
                        "aM",
                        "wM",
                        "RMB",
                        "q75_gt",
                        "pose_gt",
                        "tran_gt",
                    )
                    ok, bad_key = finite_record(original, required_tensor_keys)
                    if not ok:
                        skipped.append({"name": source_name, "reason": f"nonfinite_original_{bad_key}"})
                        continue
                    ok, bad_key = finite_record(aug_record, required_tensor_keys)
                    if not ok:
                        skipped.append({"name": source_name, "reason": f"nonfinite_augmented_{bad_key}"})
                        continue
                    if not torch.isfinite(original_offset).all() or not torch.isfinite(aug_offset).all():
                        skipped.append({"name": source_name, "reason": "nonfinite_offset"})
                        continue

                    aM_diff = diff_stats(original["aM"], aug_record["aM"])
                    wM_diff = diff_stats(original["wM"], aug_data["wM"][aug_seq_idx])
                    RMB_diff = diff_stats(original["RMB"], aug_data["RMB"][aug_seq_idx])
                    pose_diff = diff_stats(original["pose_gt"], original["pose_gt"])
                    tran_diff = diff_stats(original["tran_gt"], original["tran_gt"])
                    q75_diff = diff_stats(original["q75_gt"], original["q75_gt"])
                    if aM_diff["norm_mean"] <= args.min_am_diff:
                        skipped.append({"name": source_name, "reason": "aM_overlay_diff_too_small", "aM_norm_mean": aM_diff["norm_mean"]})
                        continue
                    if wM_diff["max_abs"] > args.max_wm_diff:
                        skipped.append({"name": source_name, "reason": "wM_diff_too_large", "wM_max_abs": wM_diff["max_abs"]})
                        continue
                    if RMB_diff["max_abs"] > args.max_rmb_diff:
                        skipped.append({"name": source_name, "reason": "RMB_diff_too_large", "RMB_max_abs": RMB_diff["max_abs"]})
                        continue

                    pair_id = source_name
                    append_view(
                        out,
                        f"{source_name}::original",
                        source_name,
                        str(cache_path),
                        str(spec["path"]),
                        "original",
                        pair_id,
                        original,
                        original_offset,
                    )
                    append_view(
                        out,
                        f"{source_name}::offset_aug_overlay",
                        source_name,
                        str(cache_path),
                        str(spec["path"]),
                        "offset_aug_overlay",
                        pair_id,
                        aug_record,
                        aug_offset,
                    )
                    total_pairs += 1
                    total_records += 2
                    total_frames += length * 2
                    am_diffs.append(aM_diff)
                    wm_diffs.append(wM_diff)
                    rmb_diffs.append(RMB_diff)
                    pose_diffs.append(pose_diff)
                    tran_diffs.append(tran_diff)
                    q75_diffs.append(q75_diff)
                    all_offsets.extend([original_offset, aug_offset])
                    diagnostics.append({
                        "pair_id": pair_id,
                        "cache_path": str(cache_path),
                        "augmented_path": str(spec["path"]),
                        "num_frames": length,
                        "original_offset_source": original_offset_source,
                        "augmented_offset_source": aug_offset_source,
                        "aM_diff": aM_diff,
                        "wM_diff": wM_diff,
                        "RMB_diff": RMB_diff,
                        "original_offset_norm_mean": float(original_offset.norm(dim=-1).mean()),
                        "augmented_offset_norm_mean": float(aug_offset.norm(dim=-1).mean()),
                    })
                except Exception as exc:
                    skipped.append({"name": source_name, "reason": f"{type(exc).__name__}: {exc}"})

        if out["name"]:
            dest_path = output_dir / cache_path.name
            torch.save(out, dest_path)
            manifest_files.append({
                "path": str(dest_path),
                "source_cache_path": str(cache_path),
                "num_pairs": len(out["name"]) // 2,
                "num_records": len(out["name"]),
                "num_frames": int(sum(out["num_frames"])),
            })

    if total_pairs == 0:
        raise RuntimeError("No K2 overlay pairs were generated.")
    skipped_ratio = len(skipped) / max(1, total_pairs + len(skipped))
    if skipped_ratio > args.max_skip_ratio:
        raise RuntimeError(f"Too many skipped sequences: skipped_ratio={skipped_ratio:.3f} > {args.max_skip_ratio:.3f}")
    am_summary = aggregate_diffs(am_diffs)
    if am_summary.get("norm_mean", {}).get("mean", 0.0) <= args.min_am_diff:
        raise RuntimeError(f"aM overlay did not change enough: {am_summary}")

    offsets = torch.stack(all_offsets)
    manifest = {
        "cache_type": "k2_paired_offset_overlay",
        "source_original_cache": args.original_cache,
        "source_original_manifest": source_manifest,
        "source_augmented_input": args.augmented_input,
        "source_augmented_manifest": {
            key: value for key, value in aug_manifest.items() if key != "shards"
        },
        "contract": {
            "prephysics": "q75_prephysics, pose_prephysics, v_root_vr, stationary_prob, q75_gt, pose_gt, and tran_gt are reused from the old neural-only cache.",
            "original_view": "Original synthetic aM/wM/RMB from the old neural-only cache plus original/default sequence-level offset_r.",
            "offset_aug_overlay_view": "Offset-augmented aM and imu_offset_r/r_JS from current AMASS processed shards overlaid onto the old neural-only pose/prephysics/target record. wM/RMB remain from the old cache after consistency checks.",
            "offset": "offset_r is sequence-level joint-local IMU installation position offset with shape [6,3], not an additive acc/gyro measurement bias and not repeated per frame.",
        },
        "cache_files": manifest_files,
        "num_pairs": total_pairs,
        "num_records": total_records,
        "num_frames": total_frames,
        "skipped_count": len(skipped),
        "offset_r_shape": [total_records, 6, 3],
        "offset_norm_mean": float(offsets.norm(dim=-1).mean()),
        "offset_norm_median": float(offsets.norm(dim=-1).median()),
        "offset_norm_p90": float(torch.quantile(offsets.norm(dim=-1).reshape(-1), 0.90)),
        "offset_norm_max": float(offsets.norm(dim=-1).max()),
        "aM_diff": am_summary,
        "wM_diff": aggregate_diffs(wm_diffs),
        "RMB_diff": aggregate_diffs(rmb_diffs),
        "pose_gt_diff": aggregate_diffs(pose_diffs),
        "tran_gt_diff": aggregate_diffs(tran_diffs),
        "q75_gt_diff": aggregate_diffs(q75_diffs),
        "test_set_used": False,
    }
    manifest_path = output_dir / "baseline_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    summary = {
        "manifest": str(manifest_path),
        "num_pairs": total_pairs,
        "num_records": total_records,
        "num_frames": total_frames,
        "skipped_count": len(skipped),
        "skipped": skipped[: args.max_skipped_report],
        "diagnostics_sample": diagnostics[: args.max_diagnostics_report],
        "aM_diff": manifest["aM_diff"],
        "wM_diff": manifest["wM_diff"],
        "RMB_diff": manifest["RMB_diff"],
        "offset_norm_mean": manifest["offset_norm_mean"],
        "offset_norm_median": manifest["offset_norm_median"],
        "offset_norm_p90": manifest["offset_norm_p90"],
        "offset_norm_max": manifest["offset_norm_max"],
    }
    summary_path = output_dir / "k2_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "summary": str(summary_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Overlay offset-augmented AMASS aM/r_JS onto an existing neural-only L4 cache.")
    parser.add_argument("--original-cache", required=True)
    parser.add_argument("--augmented-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-indices", default="")
    parser.add_argument("--min-am-diff", type=float, default=1e-4)
    parser.add_argument("--max-wm-diff", type=float, default=1e-4)
    parser.add_argument("--max-rmb-diff", type=float, default=1e-6)
    parser.add_argument("--max-skip-ratio", type=float, default=0.25)
    parser.add_argument("--max-skipped-report", type=int, default=200)
    parser.add_argument("--max-diagnostics-report", type=int, default=200)
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
