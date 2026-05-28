import argparse
import json
from pathlib import Path

import torch


G = torch.tensor([0.0, -9.8, 0.0])


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
        specs.append((resolve_path(item["path"], path.parent), item))
    return manifest, specs


def load_processed_map(processed_path):
    data = torch.load(processed_path, map_location="cpu")
    required = ("name", "RIM", "RIS", "RSB", "aS", "wS", "imu_offset_r", "R_JS", "T_JS")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{processed_path} missing {missing}")
    return {str(name): idx for idx, name in enumerate(data["name"])}, data


def official_inputs(data, idx):
    RIM = data["RIM"][idx].float()
    RIS = data["RIS"][idx].float()
    RSB = data["RSB"][idx].float()
    aS = data["aS"][idx].float()
    wS = data["wS"][idx].float()
    RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB)
    aM = RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + G
    wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1)
    return aM, wM, RMB


def r_offset_inputs(data, idx):
    RIM = data["RIM"][idx].float()
    RIS = data["RIS"][idx].float()
    aS = data["aS"][idx].float()
    wS = data["wS"][idx].float()
    R_JS = data["R_JS"][idx].float()
    aM = RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + G
    wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1)
    # Candidate A follows the documented TotalCapture audit convention:
    # R_JS maps sensor-frame vectors into the joint/body proxy, so body->sensor
    # calibration is R_JS^T and can replace the official RSB term.
    RSB_A = R_JS.transpose(-1, -2)
    RMB_A = RIM.transpose(1, 2).matmul(RIS).matmul(RSB_A)
    # Candidate B is the opposite direction, kept as a diagnostic ablation
    # because the estimated joint/body proxy is not the official bone frame.
    RSB_B = R_JS
    RMB_B = RIM.transpose(1, 2).matmul(RIS).matmul(RSB_B)
    return aM, wM, RMB_A, RMB_B


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


def aggregate(items):
    if not items:
        return {}
    out = {}
    for key in items[0].keys():
        vals = [float(item[key]) for item in items]
        out[key] = {"mean": sum(vals) / len(vals), "max": max(vals)}
    return out


def process_split(cache_manifest, processed_path, output_dir, mode):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest, specs = load_cache_specs(cache_manifest)
    processed_index, processed = load_processed_map(processed_path)
    manifest_files = []
    skipped = []
    r_offsets = []
    R_offsets = []
    diffs_A = []
    diffs_B = []
    official_recompute_diffs = []

    for cache_path, item in specs:
        cache = torch.load(cache_path, map_location="cpu")
        out = {key: list(value) if isinstance(value, list) else value for key, value in cache.items()}
        for key in (
            "aM_orig",
            "wM_orig",
            "RMB_orig",
            "aM_Roffset",
            "wM_Roffset",
            "RMB_Roffset",
            "aM_RT_offset",
            "wM_RT_offset",
            "RMB_RT_offset",
            "l4_aM",
            "l4_wM",
            "l4_RMB",
            "imu_offset_R",
            "orientation_offset_mode",
        ):
            out[key] = []
        out["offset_r"] = []
        out["imu_offset_r"] = []

        for seq_idx, name in enumerate(cache["name"]):
            name = str(name)
            if name not in processed_index:
                skipped.append({"name": name, "reason": "missing_processed_sequence"})
                continue
            pidx = processed_index[name]
            aM_orig, wM_orig, RMB_orig = official_inputs(processed, pidx)
            aM_corr, wM_corr, RMB_A, RMB_B = r_offset_inputs(processed, pidx)
            n = cache["aM"][seq_idx].shape[0]
            aM_orig, wM_orig, RMB_orig = aM_orig[:n], wM_orig[:n], RMB_orig[:n]
            aM_corr, wM_corr, RMB_A, RMB_B = aM_corr[:n], wM_corr[:n], RMB_A[:n], RMB_B[:n]
            official_recompute_diffs.append(diff_stats(cache["RMB"][seq_idx], RMB_orig))
            diffs_A.append(diff_stats(cache["RMB"][seq_idx], RMB_A))
            diffs_B.append(diff_stats(cache["RMB"][seq_idx], RMB_B))
            R_JS = processed["R_JS"][pidx].float()
            r_JS = processed["imu_offset_r"][pidx].float()
            if r_JS.shape != (6, 3) or R_JS.shape != (6, 3, 3):
                skipped.append({"name": name, "reason": "bad_offset_shape", "r_shape": list(r_JS.shape), "R_shape": list(R_JS.shape)})
                continue
            if mode == "A":
                l4_R = RMB_A
            elif mode == "B":
                l4_R = RMB_B
            else:
                raise ValueError(mode)
            for value in (aM_orig, wM_orig, RMB_orig, aM_corr, wM_corr, RMB_A, RMB_B, l4_R, r_JS, R_JS):
                if not torch.isfinite(value.float()).all():
                    skipped.append({"name": name, "reason": "nonfinite_corrected_field"})
                    break
            else:
                out["aM_orig"].append(aM_orig)
                out["wM_orig"].append(wM_orig)
                out["RMB_orig"].append(RMB_orig)
                out["aM_Roffset"].append(aM_corr)
                out["wM_Roffset"].append(wM_corr)
                out["RMB_Roffset"].append(RMB_A)
                out["aM_RT_offset"].append(aM_corr)
                out["wM_RT_offset"].append(wM_corr)
                out["RMB_RT_offset"].append(RMB_B)
                out["l4_aM"].append(aM_corr)
                out["l4_wM"].append(wM_corr)
                out["l4_RMB"].append(l4_R)
                out["offset_r"].append(r_JS)
                out["imu_offset_r"].append(r_JS)
                out["imu_offset_R"].append(R_JS)
                out["orientation_offset_mode"].append(f"TC_Roffset_{mode}")
                r_offsets.append(r_JS)
                R_offsets.append(R_JS)

        if skipped:
            raise RuntimeError(f"TotalCapture orientation-offset cache has skipped records; refusing ragged output: {skipped[:10]}")
        dest_path = output_dir / cache_path.name
        torch.save(out, dest_path)
        manifest_files.append({
            "path": str(dest_path),
            "source_cache_path": str(cache_path),
            "num_sequences": len(out["name"]),
            "num_frames": int(sum(out["num_frames"])),
        })

    r_stack = torch.stack(r_offsets) if r_offsets else torch.empty(0, 6, 3)
    R_stack = torch.stack(R_offsets) if R_offsets else torch.empty(0, 6, 3, 3)
    det = torch.linalg.det(R_stack) if R_stack.numel() else torch.empty(0)
    manifest = {
        "cache_type": "totalcapture_orientation_offset_ablation",
        "orientation_offset_mode": f"TC_Roffset_{mode}",
        "source_cache_manifest": cache_manifest,
        "source_processed_path": processed_path,
        "source_manifest": source_manifest,
        "R_JS_contract": {
            "field": "imu_offset_R / R_JS",
            "shape": "[6,3,3] per sequence",
            "direction_used_for_A": "R_JS maps sensor-frame vectors into the estimated joint/body proxy; A uses RSB_corr = R_JS^T.",
            "direction_used_for_B": "B intentionally uses RSB_corr = R_JS as the opposite-direction diagnostic.",
            "official_conversion": "RMB = RIM^T RIS RSB; aM = RIM^T RIS aS + g; wM = RIM^T RIS wS.",
            "aM_wM_policy": "aM and wM are not rotated by R_JS because they are already converted through RIS into the model/world frame; only RMB is changed in this orientation-calibration ablation.",
        },
        "L4_only_contract": "aM/wM/RMB remain the original official inputs for frozen GlobalPose PL/IK/VR; l4_aM/l4_wM/l4_RMB are consumed only by the L4 feature path.",
        "cache_files": manifest_files,
        "num_sequences": int(sum(item["num_sequences"] for item in manifest_files)),
        "num_frames": int(sum(item["num_frames"] for item in manifest_files)),
        "skipped": skipped,
        "offset_r_shape": [int(r_stack.shape[0]), 6, 3],
        "R_JS_shape": [int(R_stack.shape[0]), 6, 3, 3],
        "offset_norm_mean": float(r_stack.norm(dim=-1).mean()) if r_stack.numel() else None,
        "offset_norm_median": float(r_stack.norm(dim=-1).median()) if r_stack.numel() else None,
        "R_JS_det_mean": float(det.mean()) if det.numel() else None,
        "RMB_recompute_diff": aggregate(official_recompute_diffs),
        "RMB_Roffset_A_diff": aggregate(diffs_A),
        "RMB_Roffset_B_diff": aggregate(diffs_B),
        "test_set_used": False,
    }
    manifest_path = output_dir / "baseline_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description="Build TotalCapture L4-only R_JS orientation-offset ablation caches.")
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--processed-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("A", "B"), required=True)
    args = parser.parse_args()
    manifest_path, manifest = process_split(args.cache_manifest, args.processed_dataset, args.output_dir, args.mode)
    print(json.dumps({
        "manifest": str(manifest_path),
        "mode": manifest["orientation_offset_mode"],
        "num_sequences": manifest["num_sequences"],
        "num_frames": manifest["num_frames"],
        "skipped": len(manifest["skipped"]),
    }, indent=2))


if __name__ == "__main__":
    main()
