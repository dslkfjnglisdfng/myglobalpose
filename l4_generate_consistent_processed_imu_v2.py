import argparse
import json
from pathlib import Path

import torch
import articulate as art


G = torch.tensor([0.0, -9.8, 0.0])
SENSOR_NAMES = ("L_LowArm", "R_LowArm", "L_LowLeg", "R_LowLeg", "Head", "Pelvis")
IMU_TO_SMPL_JOINTS = [18, 19, 4, 5, 15, 0]


def resolve_path(path, base=None):
    out = Path(path)
    if not out.is_absolute() and base is not None and not out.exists():
        out = Path(base) / out
    return out


def load_cache_specs(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    return manifest, [(resolve_path(item["path"], path.parent), item) for item in manifest["cache_files"]]


def load_processed_map(processed_path):
    data = torch.load(processed_path, map_location="cpu")
    required = ("name", "RIM", "RIS", "RSB", "aS", "wS", "R_JS")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{processed_path} missing required raw TotalCapture fields: {missing}")
    return {str(name): idx for idx, name in enumerate(data["name"])}, data


def baseline_imu_from_raw(data, idx):
    RIM = data["RIM"][idx].float()
    RIS = data["RIS"][idx].float()
    RSB = data["RSB"][idx].float()
    aS = data["aS"][idx].float()
    wS = data["wS"][idx].float()
    RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB)
    aM = RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + G
    wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1)
    return aM, wM, RMB


def consistent_v2_from_raw(data, idx):
    RIM = data["RIM"][idx].float()
    RIS = data["RIS"][idx].float()
    aS = data["aS"][idx].float()
    wS = data["wS"][idx].float()
    R_JS = data["R_JS"][idx].float()

    # R_JS maps sensor-frame vectors into the estimated joint/body proxy.
    # The GlobalPose RSB term is body/bone-frame to sensor-frame, so candidate A
    # uses RSB_new = R_JS^T and keeps baseline aM/wM formulas unchanged.
    RSB_new = R_JS.transpose(-1, -2)
    l4_RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB_new)
    l4_aM = RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + G
    l4_wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1)
    return l4_aM, l4_wM, l4_RMB, R_JS, RSB_new


def pl_features(aM, wM, RMB):
    root = RMB[:, 5]
    aRB = torch.einsum("tsc,tcd->tsd", aM, root)
    wRB = torch.einsum("tsc,tcd->tsd", wM, root)
    RRB = torch.einsum("tji,tsjk->tsik", root, RMB[:, :5])
    gR0 = -root[:, 1]
    return aRB, wRB, RRB, gR0


def flatten_norm(x):
    return x.float().reshape(x.shape[0], -1).norm(dim=-1)


def stat_values(x):
    x = torch.as_tensor(x).float().reshape(-1)
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "rms": None, "count": 0}
    return {
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "std": float(finite.std(unbiased=False)) if finite.numel() > 1 else 0.0,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "rms": float(torch.sqrt((finite * finite).mean())),
        "count": int(finite.numel()),
    }


def diff_stats(a, b):
    d = a.float() - b.float()
    return {
        "norm": stat_values(flatten_norm(d)),
        "abs": stat_values(d.abs()),
        "rms": float(torch.sqrt((d.reshape(-1) ** 2).mean())),
        "max_abs": float(d.abs().max()),
        "allclose": bool(torch.allclose(a.float(), b.float())),
    }


def rotation_angle_deg(a, b):
    rel = a.float().transpose(-1, -2).matmul(b.float())
    trace = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def rotation_quality(R):
    R = R.float().reshape(-1, 3, 3)
    eye = torch.eye(3).expand_as(R)
    ortho = R.transpose(-1, -2).matmul(R)
    return {
        "det": stat_values(torch.det(R)),
        "orthogonality_fro": stat_values((ortho - eye).norm(dim=(-2, -1))),
    }


def per_sensor_rotation_stats(a, b):
    out = {}
    for idx, name in enumerate(SENSOR_NAMES):
        out[name] = stat_values(rotation_angle_deg(a[:, idx], b[:, idx]))
    return out


def per_sensor_vector_diff_stats(a, b):
    out = {}
    for idx, name in enumerate(SENSOR_NAMES):
        out[name] = diff_stats(a[:, idx], b[:, idx])
    return out


def safe_cat(items):
    return torch.cat([x.float() for x in items], dim=0) if items else torch.empty(0)


def process_split(cache_manifest, processed_path, v1_manifest, output_dir, split_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest, specs = load_cache_specs(cache_manifest)
    v1_manifest_data, v1_specs = load_cache_specs(v1_manifest)
    v1_by_path = {Path(path).name: path for path, _ in v1_specs}
    processed_index, processed = load_processed_map(processed_path)

    manifest_files = []
    skipped = []
    audit_chunks = []

    body_model = art.ParametricModel("models/SMPL_male.pkl")

    for cache_path, item in specs:
        cache = torch.load(cache_path, map_location="cpu")
        v1_path = v1_by_path.get(cache_path.name)
        v1_cache = torch.load(v1_path, map_location="cpu") if v1_path is not None else None
        out = {key: list(value) if isinstance(value, list) else value for key, value in cache.items()}
        for key in ("l4_aM", "l4_wM", "l4_RMB", "l4_aRB", "l4_wRB", "l4_RRB", "l4_gR0", "imu_offset_R", "R_JS", "RSB_new"):
            out[key] = []
        out["processed_imu_version"] = []

        for seq_idx, name in enumerate(cache["name"]):
            name = str(name)
            if name not in processed_index:
                skipped.append({"name": name, "reason": "missing_processed_sequence"})
                continue
            pidx = processed_index[name]
            n = cache["aM"][seq_idx].shape[0]
            off_a, off_w, off_R = cache["aM"][seq_idx].float(), cache["wM"][seq_idx].float(), cache["RMB"][seq_idx].float()
            raw_a, raw_w, raw_R = baseline_imu_from_raw(processed, pidx)
            l4_a, l4_w, l4_R, R_JS, RSB_new = consistent_v2_from_raw(processed, pidx)
            raw_a, raw_w, raw_R = raw_a[:n], raw_w[:n], raw_R[:n]
            l4_a, l4_w, l4_R = l4_a[:n], l4_w[:n], l4_R[:n]

            aRB, wRB, RRB, gR0 = pl_features(l4_a, l4_w, l4_R)
            off_aRB, off_wRB, off_RRB, off_gR0 = pl_features(off_a, off_w, off_R)

            if not all(torch.isfinite(x).all() for x in (l4_a, l4_w, l4_R, aRB, wRB, RRB, gR0, R_JS, RSB_new)):
                skipped.append({"name": name, "reason": "nonfinite_v2_field"})
                continue

            out["l4_aM"].append(l4_a)
            out["l4_wM"].append(l4_w)
            out["l4_RMB"].append(l4_R)
            out["l4_aRB"].append(aRB)
            out["l4_wRB"].append(wRB)
            out["l4_RRB"].append(RRB)
            out["l4_gR0"].append(gR0)
            out["imu_offset_R"].append(R_JS)
            out["R_JS"].append(R_JS)
            out["RSB_new"].append(RSB_new)
            out["processed_imu_version"].append("v2_consistent_candidate_A")

            v1_same = {}
            if v1_cache is not None and all(k in v1_cache for k in ("l4_aM", "l4_wM", "l4_RMB")):
                v1_same = {
                    "aM": diff_stats(l4_a, v1_cache["l4_aM"][seq_idx].float()[:n]),
                    "wM": diff_stats(l4_w, v1_cache["l4_wM"][seq_idx].float()[:n]),
                    "RMB_geodesic_deg": stat_values(rotation_angle_deg(l4_R, v1_cache["l4_RMB"][seq_idx].float()[:n])),
                }

            pose_gt = cache["pose_gt"][seq_idx].float()[:n] if "pose_gt" in cache and cache["pose_gt"] else None
            gt_R = None
            if pose_gt is not None:
                gt_R = body_model.forward_kinematics_R(pose_gt)[:, IMU_TO_SMPL_JOINTS]

            audit_chunks.append({
                "split": split_name,
                "name": name,
                "num_frames": int(n),
                "all_finite": True,
                "official_raw_recompute": {
                    "aM": diff_stats(off_a, raw_a),
                    "wM": diff_stats(off_w, raw_w),
                    "RMB_geodesic_deg": stat_values(rotation_angle_deg(off_R, raw_R)),
                },
                "official_vs_v2": {
                    "aM": diff_stats(off_a, l4_a),
                    "wM": diff_stats(off_w, l4_w),
                    "RMB_geodesic_deg": stat_values(rotation_angle_deg(off_R, l4_R)),
                    "aRB": diff_stats(off_aRB, aRB),
                    "wRB": diff_stats(off_wRB, wRB),
                    "RRB_geodesic_deg": stat_values(rotation_angle_deg(off_RRB, RRB)),
                    "gR0": diff_stats(off_gR0, gR0),
                    "aM_per_sensor": per_sensor_vector_diff_stats(off_a, l4_a),
                    "wM_per_sensor": per_sensor_vector_diff_stats(off_w, l4_w),
                    "RMB_per_sensor_geodesic_deg": per_sensor_rotation_stats(off_R, l4_R),
                },
                "v1_vs_v2": v1_same,
                "rotation_quality": {
                    "official_RMB": rotation_quality(off_R),
                    "l4_RMB": rotation_quality(l4_R),
                },
                "gt_orientation": None if gt_R is None else {
                    "official_RMB_vs_gt_deg": stat_values(rotation_angle_deg(off_R, gt_R)),
                    "l4_RMB_vs_gt_deg": stat_values(rotation_angle_deg(l4_R, gt_R)),
                    "official_per_sensor_deg": per_sensor_rotation_stats(off_R, gt_R),
                    "l4_per_sensor_deg": per_sensor_rotation_stats(l4_R, gt_R),
                },
            })

        if skipped:
            raise RuntimeError(f"Refusing ragged output for {split_name}; skipped examples: {skipped[:5]}")
        dest = output_dir / cache_path.name
        torch.save(out, dest)
        manifest_files.append({
            "path": str(dest),
            "source_cache_path": str(cache_path),
            "source_v1_cache_path": str(v1_path) if v1_path is not None else None,
            "num_sequences": len(out["name"]),
            "num_frames": int(sum(out["num_frames"])),
        })

    manifest = dict(source_manifest)
    manifest.update({
        "cache_type": "totalcapture_orientation_offset_consistent_v2",
        "processed_imu_version": "v2_consistent",
        "correction_type": "candidate_A_baseline_frame_recomputation_from_raw_sensor_fields",
        "source_cache_manifest": str(cache_manifest),
        "source_v1_manifest": str(v1_manifest),
        "source_processed_path": str(processed_path),
        "source_fields_available": ["RIM", "RIS", "RSB", "aS", "wS", "R_JS"],
        "formula_l4_RMB": "l4_RMB = RIM^T @ RIS @ RSB_new, where RSB_new = R_JS^T",
        "formula_l4_aM": "l4_aM = RIM^T @ RIS @ aS + gravity",
        "formula_l4_wM": "l4_wM = RIM^T @ RIS @ wS",
        "formula_feature_diagnostics": {
            "l4_aRB": "l4_aM @ l4_RMB[5]",
            "l4_wRB": "l4_wM @ l4_RMB[5]",
            "l4_RRB": "l4_RMB[5]^T @ l4_RMB[:5]",
            "l4_gR0": "-l4_RMB[5,1]",
        },
        "gravity_vector": [0.0, -9.8, 0.0],
        "coordinate_convention": {
            "RMB": "R_M_B = RIM^T RIS RSB, body/bone-frame axes expressed in model/world frame.",
            "aM_wM": "Stored acceleration/angular velocity are model/world-frame vectors under the baseline formulas; they do not directly depend on RSB.",
            "row_vector_network_use": "GPNet later forms root-relative features by right multiplying row vectors: aRB=aM@RMB[5], wRB=wM@RMB[5].",
            "R_JS": "R_JS maps sensor-frame vectors into estimated joint/body proxy; candidate A uses RSB_new=R_JS^T.",
        },
        "cache_files": manifest_files,
        "num_sequences": int(sum(x["num_sequences"] for x in manifest_files)),
        "num_frames": int(sum(x["num_frames"] for x in manifest_files)),
        "skipped": skipped,
    })
    manifest_path = output_dir / "baseline_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, manifest, audit_chunks


def aggregate_audit(chunks):
    def aggregate_path(path):
        values = []
        for row in chunks:
            cur = row
            ok = True
            for key in path:
                if cur is None or key not in cur:
                    ok = False
                    break
                cur = cur[key]
            if ok and isinstance(cur, dict) and "mean" in cur:
                values.append(cur["mean"])
        return stat_values(torch.tensor(values)) if values else None

    return {
        "num_sequences": len(chunks),
        "num_frames": int(sum(row["num_frames"] for row in chunks)),
        "all_finite": all(row["all_finite"] for row in chunks),
        "official_vs_v2": {
            "aM_norm_mean_by_sequence": aggregate_path(["official_vs_v2", "aM", "norm"]),
            "wM_norm_mean_by_sequence": aggregate_path(["official_vs_v2", "wM", "norm"]),
            "RMB_geodesic_mean_by_sequence": aggregate_path(["official_vs_v2", "RMB_geodesic_deg"]),
            "aRB_norm_mean_by_sequence": aggregate_path(["official_vs_v2", "aRB", "norm"]),
            "wRB_norm_mean_by_sequence": aggregate_path(["official_vs_v2", "wRB", "norm"]),
            "RRB_geodesic_mean_by_sequence": aggregate_path(["official_vs_v2", "RRB_geodesic_deg"]),
            "gR0_norm_mean_by_sequence": aggregate_path(["official_vs_v2", "gR0", "norm"]),
        },
        "v1_vs_v2": {
            "aM_norm_mean_by_sequence": aggregate_path(["v1_vs_v2", "aM", "norm"]),
            "wM_norm_mean_by_sequence": aggregate_path(["v1_vs_v2", "wM", "norm"]),
            "RMB_geodesic_mean_by_sequence": aggregate_path(["v1_vs_v2", "RMB_geodesic_deg"]),
        },
        "gt_orientation": {
            "official_RMB_vs_gt_mean_by_sequence": aggregate_path(["gt_orientation", "official_RMB_vs_gt_deg"]),
            "l4_RMB_vs_gt_mean_by_sequence": aggregate_path(["gt_orientation", "l4_RMB_vs_gt_deg"]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate baseline-consistent processed IMU v2 caches and audit JSON.")
    parser.add_argument("--train-cache-manifest", required=True)
    parser.add_argument("--train-v1-manifest", required=True)
    parser.add_argument("--train-processed-dataset", required=True)
    parser.add_argument("--train-output-dir", required=True)
    parser.add_argument("--val-cache-manifest", required=True)
    parser.add_argument("--val-v1-manifest", required=True)
    parser.add_argument("--val-processed-dataset", required=True)
    parser.add_argument("--val-output-dir", required=True)
    parser.add_argument("--audit-json", required=True)
    args = parser.parse_args()

    train_manifest_path, train_manifest, train_chunks = process_split(
        args.train_cache_manifest, args.train_processed_dataset, args.train_v1_manifest, args.train_output_dir, "train"
    )
    val_manifest_path, val_manifest, val_chunks = process_split(
        args.val_cache_manifest, args.val_processed_dataset, args.val_v1_manifest, args.val_output_dir, "val"
    )
    chunks = train_chunks + val_chunks
    result = {
        "status": "ok",
        "processed_imu_version": "v2_consistent",
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "audit_json": str(args.audit_json),
        "baseline_formula": {
            "RMB": "RIM^T @ RIS @ RSB",
            "aM": "RIM^T @ RIS @ aS + [0,-9.8,0]",
            "wM": "RIM^T @ RIS @ wS",
        },
        "adopted_v2_formula": {
            "RSB_new": "R_JS^T",
            "l4_RMB": "RIM^T @ RIS @ RSB_new",
            "l4_aM": "RIM^T @ RIS @ aS + [0,-9.8,0]",
            "l4_wM": "RIM^T @ RIS @ wS",
        },
        "interpretation": "Under the baseline coordinate definition, changing sensor-to-body orientation changes l4_RMB but does not directly change stored model/world-frame l4_aM/l4_wM; root-relative aRB/wRB do change because GPNet multiplies by l4_RMB[5].",
        "train_cache": train_manifest,
        "val_cache": val_manifest,
        "aggregate": aggregate_audit(chunks),
        "rows": chunks,
    }
    audit_path = Path(args.audit_json)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "train_manifest": result["train_manifest"],
        "val_manifest": result["val_manifest"],
        "audit_json": str(audit_path),
        "num_sequences": result["aggregate"]["num_sequences"],
        "num_frames": result["aggregate"]["num_frames"],
        "aM_norm_mean": result["aggregate"]["official_vs_v2"]["aM_norm_mean_by_sequence"]["mean"],
        "wM_norm_mean": result["aggregate"]["official_vs_v2"]["wM_norm_mean_by_sequence"]["mean"],
        "RMB_geo_mean": result["aggregate"]["official_vs_v2"]["RMB_geodesic_mean_by_sequence"]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
