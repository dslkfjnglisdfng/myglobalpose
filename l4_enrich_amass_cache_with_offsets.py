import argparse
import json
import re
from pathlib import Path

import torch


OFFSET_KEYS = ("imu_offset_r", "r_JS", "original_imu_offset_r")


def resolve_path(path, base=None):
    p = Path(path)
    if p.is_absolute() or p.exists() or base is None:
        return p
    return Path(base) / p


def stack_if_list(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, list) and value and torch.is_tensor(value[0]):
        return torch.stack(value)
    raise TypeError(type(value))


def shard_id_from_path(path):
    match = re.search(r"shard0*(\d+)", str(path))
    if not match:
        return None
    return int(match.group(1))


def load_offset_map(processed_path):
    data = torch.load(processed_path, map_location="cpu")
    key = next((key for key in OFFSET_KEYS if key in data), None)
    if key is None:
        raise KeyError(f"{processed_path} missing one of {OFFSET_KEYS}")
    offsets = stack_if_list(data[key]).float()
    if offsets.shape[1:] != (6, 3):
        raise ValueError(f"{processed_path} {key} shape={tuple(offsets.shape)}, expected [N,6,3]")
    return {str(name): offsets[idx].clone() for idx, name in enumerate(data["name"])}, key


def processed_path_for_item(item, source_path, processed_root):
    source_processed = item.get("source_path")
    if source_processed:
        p = Path(source_processed)
        if p.exists():
            return p
    shard_id = shard_id_from_path(source_processed or source_path)
    if shard_id is None:
        raise ValueError(f"Cannot infer AMASS shard id from item={item} source_path={source_path}")
    p = processed_root / f"globalpose_synth_shard{shard_id:05d}.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def enrich_one(source_path, dest_path, offset_map):
    data = torch.load(source_path, map_location="cpu")
    missing = [str(name) for name in data["name"] if str(name) not in offset_map]
    if missing:
        raise RuntimeError(f"{source_path} missing offset entries for {len(missing)} names, first={missing[:3]}")
    enriched = dict(data)
    enriched["offset_r"] = [offset_map[str(name)].float().clone() for name in data["name"]]
    enriched["imu_offset_r"] = [offset_map[str(name)].float().clone() for name in data["name"]]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(enriched, dest_path)
    offsets = torch.stack(enriched["offset_r"])
    return {
        "source_path": str(source_path),
        "path": str(dest_path),
        "num_sequences": len(enriched["name"]),
        "num_frames": int(sum(enriched["num_frames"])) if "num_frames" in enriched else None,
        "offset_r_shape": [len(enriched["name"]), 6, 3],
        "offset_norm_mean": float(offsets.norm(dim=-1).mean()),
        "offset_norm_median": float(offsets.norm(dim=-1).median()),
        "offset_norm_max": float(offsets.norm(dim=-1).max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Add sequence-level AMASS offset_r to a merged GlobalPose L4 baseline cache.")
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--processed-root", default="data/dataset_work/AMASS")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-path", default="")
    args = parser.parse_args()

    manifest_path = Path(args.cache_manifest)
    manifest = json.loads(manifest_path.read_text())
    processed_root = Path(args.processed_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_manifest = dict(manifest)
    output_manifest["source_cache_manifest"] = str(manifest_path)
    output_manifest["offset_source_root"] = str(processed_root)
    output_manifest["offset_field_added"] = "offset_r"
    output_manifest["offset_contract"] = (
        "offset_r is sequence-level r_JS with shape [6,3], copied from the matching AMASS "
        "globalpose_synth_shardXXXXX.pt record by sequence name. It is not repeated per frame."
    )
    output_manifest["cache_files"] = []

    offset_maps = {}
    summaries = []
    for item in manifest["cache_files"]:
        source_path = resolve_path(item["path"], manifest_path.parent)
        processed_path = processed_path_for_item(item, source_path, processed_root)
        if processed_path not in offset_maps:
            offset_maps[processed_path] = load_offset_map(processed_path)
        offset_map, offset_key = offset_maps[processed_path]
        dest_path = output_dir / source_path.name
        summary = enrich_one(source_path, dest_path, offset_map)
        new_item = dict(item)
        new_item["path"] = str(dest_path)
        new_item["source_path"] = str(source_path)
        new_item["offset_source_path"] = str(processed_path)
        new_item["offset_source_key"] = offset_key
        new_item["offset_r_shape"] = summary["offset_r_shape"]
        output_manifest["cache_files"].append(new_item)
        summaries.append(summary)
        print(f"wrote {dest_path}: n={summary['num_sequences']} offset_median={summary['offset_norm_median']:.6f}")

    manifest_out = output_dir / "baseline_cache_manifest.json"
    manifest_out.write_text(json.dumps(output_manifest, indent=2) + "\n")
    summary = {
        "status": "ok",
        "output_manifest": str(manifest_out),
        "source_cache_manifest": str(manifest_path),
        "processed_root": str(processed_root),
        "num_files": len(summaries),
        "num_sequences": sum(item["num_sequences"] for item in summaries),
        "num_frames": sum(item["num_frames"] or 0 for item in summaries),
        "files": summaries,
    }
    summary_path = Path(args.summary_path) if args.summary_path else output_dir / "offset_enrich_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_out), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
