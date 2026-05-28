import argparse
import json
from pathlib import Path

import torch
import tqdm

import articulate as art
from l4_q75_utils import pose_tran_to_q75
from net import GPNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_augmented_specs(input_path):
    path = Path(input_path)
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        specs = []
        for idx, shard in enumerate(manifest["shards"]):
            shard_path = Path(shard["path"])
            if not shard_path.is_absolute() and not shard_path.exists():
                shard_path = path.parent / shard_path
            specs.append({"path": shard_path, "index": idx, "manifest": shard})
        return specs, manifest
    return [{"path": path, "index": 0, "manifest": None}], None


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


def load_original_records(cache_manifest):
    manifest, specs = load_cache_specs(cache_manifest)
    records = {}
    for cache_path, _ in specs:
        data = torch.load(cache_path, map_location="cpu")
        for seq_idx, name in enumerate(data["name"]):
            records[str(name)] = {key: data[key][seq_idx] for key in data if isinstance(data[key], list) and len(data[key]) == len(data["name"])}
            records[str(name)]["name"] = str(name)
            records[str(name)]["_cache_path"] = str(cache_path)
    return records, manifest


def trim_record(record, length):
    out = {}
    for key, value in record.items():
        if torch.is_tensor(value) and value.shape[0] >= length and key not in ("offset_r",):
            out[key] = value[:length].clone()
        else:
            out[key] = value.clone() if torch.is_tensor(value) else value
    out["num_frames"] = int(length)
    return out


def empty_cache():
    return {
        "name": [],
        "source_name": [],
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
        "num_frames": [],
    }


def append_record(cache, name, source_name, view_type, pair_id, record, offset_r):
    cache["name"].append(name)
    cache["source_name"].append(source_name)
    cache["view_type"].append(view_type)
    cache["pair_id"].append(pair_id)
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
        cache[key].append(record[key])
    cache["offset_r"].append(offset_r.float().clone())


@torch.no_grad()
def run_augmented_prephysics(net, pose_axis_angle, tran, aM, wM, RMB, max_frames, euler_seq):
    pose_gt = art.math.axis_angle_to_rotation_matrix(pose_axis_angle).view(-1, 24, 3, 3)
    tran_gt = tran.float()
    if max_frames:
        pose_gt = pose_gt[:max_frames]
        tran_gt = tran_gt[:max_frames]
        aM = aM[:max_frames]
        wM = wM[:max_frames]
        RMB = RMB[:max_frames]
    net.rnn_initialize(pose_gt[0])
    q75_gt = pose_tran_to_q75(pose_gt, tran_gt, euler_seq=euler_seq)
    q75_prephysics = []
    pose_prephysics = []
    v_root_vr = []
    stationary_prob = []
    for frame_idx in range(pose_gt.shape[0]):
        features = net.forward_prephysics_features(
            aM[frame_idx].to(DEVICE),
            wM[frame_idx].to(DEVICE),
            RMB[frame_idx].to(DEVICE),
            prephysics_tran=None,
            euler_seq=euler_seq,
        )
        q75_prephysics.append(features["q75_prephysics"])
        pose_prephysics.append(features["pose_prephysics"])
        v_root_vr.append(features["v_root_vr"])
        stationary_prob.append(features["stationary_prob"])
    return {
        "pose_gt": pose_gt.cpu(),
        "tran_gt": tran_gt.cpu(),
        "q75_gt": q75_gt.cpu(),
        "q75_prephysics": torch.stack(q75_prephysics).cpu(),
        "pose_prephysics": torch.stack(pose_prephysics).cpu(),
        "v_root_vr": torch.stack(v_root_vr).cpu(),
        "stationary_prob": torch.stack(stationary_prob).cpu(),
        "aM": aM.cpu(),
        "wM": wM.cpu(),
        "RMB": RMB.cpu(),
        "num_frames": int(pose_gt.shape[0]),
    }


def tensor_diff_stats(a, b):
    n = min(a.shape[0], b.shape[0])
    d = (a[:n].float() - b[:n].float()).reshape(n, -1)
    norms = d.norm(dim=-1)
    return {
        "max_abs": float((a[:n].float() - b[:n].float()).abs().max()),
        "norm_mean": float(norms.mean()),
        "norm_median": float(norms.median()),
        "norm_max": float(norms.max()),
    }


def process(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_records, original_manifest = load_original_records(args.original_cache)
    augmented_specs, augmented_manifest = load_augmented_specs(args.augmented_input)
    if args.shard_indices:
        wanted = {int(item) for item in args.shard_indices.split(",") if item.strip()}
        augmented_specs = [spec for spec in augmented_specs if spec["index"] in wanted]
    elif args.max_shards:
        augmented_specs = augmented_specs[: args.max_shards]

    net = GPNet(enable_l4_prephysics=False).eval().to(DEVICE)
    cache = empty_cache()
    diagnostics = []
    failures = []
    processed_pairs = 0

    for spec in augmented_specs:
        data = torch.load(spec["path"], map_location="cpu")
        nseq = len(data["name"])
        for seq_idx in tqdm.trange(nseq, desc=f"paired shard {spec['index']}"):
            if args.max_pairs and processed_pairs >= args.max_pairs:
                break
            source_name = str(data["name"][seq_idx])
            if source_name not in original_records:
                if args.require_original_match:
                    failures.append({"name": source_name, "error": "missing_original_cache_record"})
                continue
            try:
                original = original_records[source_name]
                if "original_imu_offset_r" in data:
                    original_offset = data["original_imu_offset_r"][seq_idx].float()
                    original_offset_source = "original_imu_offset_r"
                elif "imu_offset_r" in original:
                    original_offset = original["imu_offset_r"].float()
                    original_offset_source = "original_cache_imu_offset_r"
                else:
                    original_offset = torch.zeros(6, 3)
                    original_offset_source = "zero_default"

                augmented_offset_key = "imu_offset_r" if "imu_offset_r" in data else "r_JS"
                augmented_offset = data[augmented_offset_key][seq_idx].float()
                aug_record = run_augmented_prephysics(
                    net,
                    data["pose"][seq_idx].float(),
                    data["tran"][seq_idx].float(),
                    data["aM"][seq_idx].float(),
                    data["wM"][seq_idx].float(),
                    data["RMB"][seq_idx].float(),
                    args.max_frames,
                    args.euler_seq,
                )
                length = aug_record["num_frames"]
                orig_record = trim_record(original, length)

                pose_diff = tensor_diff_stats(orig_record["pose_gt"], aug_record["pose_gt"])
                tran_diff = tensor_diff_stats(orig_record["tran_gt"], aug_record["tran_gt"])
                aM_diff = tensor_diff_stats(orig_record["aM"], aug_record["aM"])
                wM_diff = tensor_diff_stats(orig_record["wM"], aug_record["wM"])
                RMB_diff = tensor_diff_stats(orig_record["RMB"], aug_record["RMB"])

                pair_id = source_name
                append_record(
                    cache,
                    f"{source_name}::original",
                    source_name,
                    "original",
                    pair_id,
                    orig_record,
                    original_offset,
                )
                append_record(
                    cache,
                    f"{source_name}::offset_aug",
                    source_name,
                    "offset_aug",
                    pair_id,
                    aug_record,
                    augmented_offset,
                )
                diagnostics.append({
                    "pair_id": pair_id,
                    "source_name": source_name,
                    "source_shard": str(spec["path"]),
                    "original_cache_path": original.get("_cache_path"),
                    "num_frames": int(length),
                    "original_offset_source": original_offset_source,
                    "augmented_offset_source": augmented_offset_key,
                    "pose_diff": pose_diff,
                    "tran_diff": tran_diff,
                    "aM_diff": aM_diff,
                    "wM_diff": wM_diff,
                    "RMB_diff": RMB_diff,
                    "original_offset_norm_mean": float(original_offset.norm(dim=-1).mean()),
                    "augmented_offset_norm_mean": float(augmented_offset.norm(dim=-1).mean()),
                })
                processed_pairs += 1
            except Exception as exc:
                failures.append({"name": source_name, "error": f"{type(exc).__name__}: {exc}"})
        if args.max_pairs and processed_pairs >= args.max_pairs:
            break

    if processed_pairs == 0:
        raise RuntimeError("No paired records were generated.")

    cache_path = output_dir / "paired_cache_shard00000.pt"
    torch.save(cache, cache_path)
    offsets = torch.stack(cache["offset_r"])
    manifest = {
        "cache_type": "paired_offset_aug_neural_only",
        "source_original_cache": args.original_cache,
        "source_original_manifest": original_manifest,
        "source_augmented_input": args.augmented_input,
        "source_augmented_manifest": augmented_manifest,
        "device": str(DEVICE),
        "euler_seq": args.euler_seq,
        "max_frames": args.max_frames,
        "max_pairs": args.max_pairs,
        "view_contract": {
            "original": "Original synthetic IMU view from the no-offset neural-only AMASS cache.",
            "offset_aug": "Current AMASS processed view with single-random-placement r_JS and resynthesized aM/aS. wM/RMB/wS are expected to match the original view.",
        },
        "offset_contract": "offset_r is sequence-level r_JS with shape [6,3]; it is not repeated per frame.",
        "cache_files": [{
            "path": str(cache_path),
            "num_pairs": processed_pairs,
            "num_records": len(cache["name"]),
            "num_frames": int(sum(cache["num_frames"])),
            "failures": failures,
        }],
        "num_pairs": processed_pairs,
        "num_records": len(cache["name"]),
        "num_frames": int(sum(cache["num_frames"])),
        "offset_r_shape": [len(cache["name"]), 6, 3],
        "offset_norm_mean": float(offsets.norm(dim=-1).mean()),
        "offset_norm_median": float(offsets.norm(dim=-1).median()),
        "offset_norm_max": float(offsets.norm(dim=-1).max()),
    }
    manifest_path = output_dir / "paired_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    summary = {
        "manifest": str(manifest_path),
        "cache_path": str(cache_path),
        "num_pairs": processed_pairs,
        "num_records": len(cache["name"]),
        "diagnostics": diagnostics,
        "failures": failures,
    }
    summary_path = output_dir / "paired_cache_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "summary": str(summary_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Generate paired original/offset-aug AMASS neural-only L4 cache.")
    parser.add_argument("--original-cache", required=True, help="Original no-offset neural-only AMASS cache manifest.")
    parser.add_argument("--augmented-input", required=True, help="Current offset-augmented AMASS processed manifest or shard.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-indices", default="")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--euler-seq", default="XYZ")
    parser.add_argument("--require-original-match", action="store_true")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
