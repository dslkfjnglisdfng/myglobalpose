import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from l4_sensor_offset_utils import causal_butterworth_lowpass_sequence, smooth_centered
from l4_train_diverse_short import load_cache_files


def clone_item(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return [clone_item(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_item(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_item(item) for key, item in value.items()}
    return value


def is_sequence_field(value, num_sequences):
    if isinstance(value, (list, tuple)):
        return len(value) == num_sequences
    return torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == num_sequences


def subset_clone(value, num_sequences, keep):
    if is_sequence_field(value, num_sequences):
        if isinstance(value, list):
            return [clone_item(item) for item in value[:keep]]
        if isinstance(value, tuple):
            return tuple(clone_item(item) for item in value[:keep])
        return value[:keep].clone()
    return clone_item(value)


def filter_one_acc(seq, args):
    if not torch.is_tensor(seq):
        seq = torch.as_tensor(seq)
    seq = seq.float()
    if args.mode == "causal_butterworth":
        filtered = causal_butterworth_lowpass_sequence(
            seq,
            fs=args.fs,
            cutoff_hz=args.cutoff_hz,
            order=args.filter_order,
        )
    else:
        filtered = smooth_centered(seq, window=args.window, mode=args.mode)
    if not torch.isfinite(filtered).all():
        raise RuntimeError("Non-finite filtered acceleration.")
    return filtered


def filter_acc_field(value, num_sequences, keep, args):
    if isinstance(value, list):
        return [filter_one_acc(value[idx], args) for idx in range(keep)]
    if isinstance(value, tuple):
        return tuple(filter_one_acc(value[idx], args) for idx in range(keep))
    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == num_sequences:
        return torch.stack([filter_one_acc(value[idx], args) for idx in range(keep)])
    raise TypeError("aM must be a sequence field.")


def iter_sequences(value):
    if isinstance(value, (list, tuple)):
        yield from value
        return
    if torch.is_tensor(value):
        for idx in range(value.shape[0]):
            yield value[idx]
        return
    raise TypeError("Expected a sequence field.")


def acc_stats(raw_a, filtered_a):
    diff_sq = 0.0
    raw_jitter_sq = 0.0
    filtered_jitter_sq = 0.0
    value_count = 0
    jitter_count = 0
    for raw, filtered in zip(iter_sequences(raw_a), iter_sequences(filtered_a)):
        raw = raw.float()
        filtered = filtered.float()
        diff = filtered - raw
        diff_sq += float(diff.square().sum())
        value_count += int(diff.numel())
        if raw.shape[0] >= 3:
            raw_dd = raw[2:] - 2.0 * raw[1:-1] + raw[:-2]
            filtered_dd = filtered[2:] - 2.0 * filtered[1:-1] + filtered[:-2]
            raw_jitter_sq += float(raw_dd.square().sum())
            filtered_jitter_sq += float(filtered_dd.square().sum())
            jitter_count += int(raw_dd.numel())
    out = {
        "raw_filtered_rms_delta": (diff_sq / max(1, value_count)) ** 0.5,
        "num_values": value_count,
    }
    if jitter_count:
        out.update({
            "raw_second_difference_rms": (raw_jitter_sq / jitter_count) ** 0.5,
            "filtered_second_difference_rms": (filtered_jitter_sq / jitter_count) ** 0.5,
            "second_difference_rms_ratio": (filtered_jitter_sq / max(raw_jitter_sq, 1e-12)) ** 0.5,
        })
    return out


def frame_count(a_value):
    if isinstance(a_value, (list, tuple)):
        return int(sum(int(item.shape[0]) for item in a_value))
    if torch.is_tensor(a_value):
        return int(sum(int(a_value[idx].shape[0]) for idx in range(a_value.shape[0])))
    return 0


def build_smooth_cache(args):
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "baseline_cache_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(json.dumps({"status": "exists", "manifest": str(manifest_path)}, indent=2))
        return json.loads(manifest_path.read_text())

    output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(args.input_cache)
    if source_manifest is None:
        source_manifest = {
            "type": "single_pt_cache",
            "cache_files": [{"path": str(files[0])}],
        }

    cache_files = []
    total_sequences = 0
    total_frames = 0
    total_diff_sq = 0.0
    total_raw_jitter_sq = 0.0
    total_filtered_jitter_sq = 0.0
    total_values = 0
    total_jitter_values = 0
    shard_idx = 0

    for cache_file in files:
        data = torch.load(cache_file, map_location="cpu")
        if "name" not in data or "aM" not in data:
            raise KeyError(f"{cache_file} must contain name and aM fields.")
        num_sequences = len(data["name"])
        if args.max_sequences and total_sequences >= args.max_sequences:
            break
        keep = num_sequences
        if args.max_sequences:
            keep = min(keep, int(args.max_sequences) - total_sequences)
        if keep <= 0:
            break

        shard = {
            key: subset_clone(value, num_sequences, keep)
            for key, value in data.items()
        }
        raw_a = subset_clone(data["aM"], num_sequences, keep)
        smooth_a = filter_acc_field(data["aM"], num_sequences, keep, args)
        shard_stats = acc_stats(raw_a, smooth_a)
        total_diff_sq += shard_stats["raw_filtered_rms_delta"] ** 2 * shard_stats["num_values"]
        total_values += shard_stats["num_values"]
        if "raw_second_difference_rms" in shard_stats:
            jitter_values = 0
            for seq in iter_sequences(smooth_a):
                if seq.shape[0] >= 3:
                    jitter_values += int((seq.shape[0] - 2) * seq[0].numel())
            total_raw_jitter_sq += shard_stats["raw_second_difference_rms"] ** 2 * jitter_values
            total_filtered_jitter_sq += shard_stats["filtered_second_difference_rms"] ** 2 * jitter_values
            total_jitter_values += jitter_values
        shard["aM_raw"] = raw_a
        shard["aM"] = smooth_a

        frames = frame_count(smooth_a)
        out_path = output_dir / f"smooth_acc_cache_shard{shard_idx:05d}.pt"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists but manifest is missing; pass --overwrite to replace.")
        torch.save(shard, out_path)
        cache_files.append({
            "path": str(out_path),
            "num_sequences": int(keep),
            "num_frames": int(frames),
            "source_path": str(cache_file),
            "acc_filter_stats": shard_stats,
        })
        shard_idx += 1
        total_sequences += int(keep)
        total_frames += int(frames)
        print(json.dumps({
            "processed_source": str(cache_file),
            "output": str(out_path),
            "num_sequences": int(keep),
            "num_frames": int(frames),
            "total_sequences": int(total_sequences),
            "acc_filter_stats": shard_stats,
        }))

    causal = args.mode == "causal_butterworth"
    global_stats = {
        "raw_filtered_rms_delta": (total_diff_sq / max(1, total_values)) ** 0.5,
        "num_values": int(total_values),
    }
    if total_jitter_values:
        global_stats.update({
            "raw_second_difference_rms": (total_raw_jitter_sq / total_jitter_values) ** 0.5,
            "filtered_second_difference_rms": (total_filtered_jitter_sq / total_jitter_values) ** 0.5,
            "second_difference_rms_ratio": (total_filtered_jitter_sq / max(total_raw_jitter_sq, 1e-12)) ** 0.5,
        })
    if causal:
        aM_contract = (
            "aM is zero-lookahead causal Butterworth low-pass filtered from source aM; "
            "aM_raw preserves the original source aM."
        )
        cache_transform = "replace_aM_with_causal_butterworth_lowpass_aM"
    else:
        aM_contract = (
            "aM is centered offline smoothed acceleration from source aM; "
            "aM_raw preserves the original source aM."
        )
        cache_transform = "replace_aM_with_smoothed_aM"
    manifest = dict(source_manifest)
    manifest.update({
        "type": "smooth_acc_baseline_cache_v1",
        "source_cache": str(args.input_cache),
        "source_manifest": source_manifest,
        "cache_files": cache_files,
        "num_sequences": int(total_sequences),
        "num_frames": int(total_frames),
        "max_sequences": int(args.max_sequences),
        "smooth_acc_cache": True,
        "cache_transform": cache_transform,
        "aM_contract": aM_contract,
        "aM_raw_field": "aM_raw",
        "acc_smooth_mode": args.mode,
        "acc_smooth_window": int(args.window),
        "offline_centered": not causal,
        "causal": causal,
        "lookahead_frames": 0 if causal else int(args.window // 2),
        "latency_ms": 0.0 if causal else float(args.window // 2) * 1000.0 / float(args.fs),
        "fs_hz": float(args.fs),
        "cutoff_hz": float(args.cutoff_hz) if causal else None,
        "filter_order": int(args.filter_order) if causal else None,
        "acc_filter_stats": global_stats,
        "fields_update": {
            "aM": "filtered source aM, same shape as original",
            "aM_raw": "original source aM for audit only",
            "wM": "unchanged from source cache",
            "RMB": "unchanged from source cache",
        },
        "notes": args.notes,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path)}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build a raw baseline-cache view with filtered aM.")
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument(
        "--mode",
        choices=("moving_average", "centered_moving_average", "savgol", "savitzky_golay", "causal_butterworth"),
        default="centered_moving_average",
    )
    parser.add_argument("--fs", type=float, default=60.0)
    parser.add_argument("--cutoff-hz", type=float, default=10.0)
    parser.add_argument("--filter-order", type=int, default=2)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--notes", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode != "causal_butterworth" and args.window % 2 == 0:
        raise ValueError("--window must be odd for centered smoothing.")
    build_smooth_cache(args)


if __name__ == "__main__":
    main()
