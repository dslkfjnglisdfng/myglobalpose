import argparse
import json
import shutil
from pathlib import Path

import torch


OFFSET_KEYS = ("imu_offset_r", "r_JS")


def load_cache_specs(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    specs = []
    for item in manifest["cache_files"]:
        cache_path = Path(item["path"])
        if not cache_path.is_absolute() and not cache_path.exists():
            cache_path = path.parent / cache_path
        specs.append((cache_path, item))
    return manifest, specs


def stack_if_list(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, list) and x and torch.is_tensor(x[0]):
        return torch.stack(x)
    raise TypeError(type(x))


def load_offsets(processed_path):
    data = torch.load(processed_path, map_location="cpu")
    key = next((k for k in OFFSET_KEYS if k in data), None)
    if key is None:
        raise KeyError(f"{processed_path} missing one of {OFFSET_KEYS}")
    offsets = stack_if_list(data[key]).float()
    if offsets.shape[1:] != (6, 3):
        raise ValueError(f"{processed_path} {key} has shape {tuple(offsets.shape)}, expected [N,6,3]")
    return {str(name): offsets[idx] for idx, name in enumerate(data["name"])}, key


def enrich_cache_file(source_path, dest_path, offset_map):
    data = torch.load(source_path, map_location="cpu")
    missing = [str(name) for name in data["name"] if str(name) not in offset_map]
    if missing:
        raise RuntimeError(f"{source_path} missing offsets for names: {missing}")
    enriched = dict(data)
    enriched["offset_r"] = [offset_map[str(name)].clone() for name in data["name"]]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(enriched, dest_path)
    offsets = torch.stack(enriched["offset_r"])
    return {
        "source_path": str(source_path),
        "path": str(dest_path),
        "num_sequences": len(enriched["name"]),
        "num_frames": int(sum(enriched["num_frames"])) if "num_frames" in enriched else None,
        "offset_r_shape": [len(enriched["name"]), 6, 3],
        "offset_norm_min": float(offsets.norm(dim=-1).min()),
        "offset_norm_median": float(offsets.norm(dim=-1).median()),
        "offset_norm_max": float(offsets.norm(dim=-1).max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Copy sequence-level offset_r into an existing L4 neural-only cache.")
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--processed-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-path", default="")
    args = parser.parse_args()

    source_manifest, specs = load_cache_specs(args.cache_manifest)
    offset_map, source_key = load_offsets(args.processed_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_manifest = dict(source_manifest)
    output_manifest["source_cache_manifest"] = args.cache_manifest
    output_manifest["offset_source_path"] = args.processed_dataset
    output_manifest["offset_source_key"] = source_key
    output_manifest["offset_field_added"] = "offset_r"
    output_manifest["offset_contract"] = (
        "offset_r is sequence-level r_JS with shape [6,3]: IMU origin relative to mapped joint, "
        "expressed in joint-local coordinates. It is copied from processed data and is not repeated per frame."
    )
    output_manifest["cache_files"] = []

    summaries = []
    for idx, (source_path, item) in enumerate(specs):
        dest_path = output_dir / source_path.name
        summary = enrich_cache_file(source_path, dest_path, offset_map)
        new_item = dict(item)
        new_item["path"] = str(dest_path)
        new_item["source_path"] = str(source_path)
        new_item["offset_r_shape"] = summary["offset_r_shape"]
        output_manifest["cache_files"].append(new_item)
        summaries.append(summary)
        print(f"wrote {dest_path}: n={summary['num_sequences']} median_norm={summary['offset_norm_median']:.6f}")

    manifest_path = output_dir / "baseline_cache_manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2))
    summary = {
        "output_manifest": str(manifest_path),
        "source_cache_manifest": args.cache_manifest,
        "processed_dataset": args.processed_dataset,
        "offset_source_key": source_key,
        "files": summaries,
    }
    summary_path = Path(args.summary_path) if args.summary_path else output_dir / "offset_enrich_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
