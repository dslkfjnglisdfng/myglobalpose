import argparse
import json
from pathlib import Path

import torch
import tqdm

import articulate as art
from imu_offset_net import (
    IMUOffsetNet,
    OFFSET_COORDINATE_CONTRACT,
    acceleration_consistency_error,
    offset_input_feature,
)
from imu_offset_train import offset_errors_cm
from l4_train_diverse_short import load_cache_files
from pl_curve import normalize_gravity, pl_target_from_pose


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(path, base=None):
    p = Path(path)
    if p.is_absolute() or p.exists() or base is None:
        return p
    return Path(base) / p


def stack_if_list(value):
    if torch.is_tensor(value):
        return value.float()
    if isinstance(value, list) and value and torch.is_tensor(value[0]):
        return torch.stack([x.float() for x in value])
    raise TypeError(type(value))


def selected_imu_fields(data, seq_idx, imu_input_mode):
    has_l4 = all(key in data and data[key] for key in ("l4_aM", "l4_wM", "l4_RMB"))
    if imu_input_mode == "official":
        return data["aM"][seq_idx].float(), data["wM"][seq_idx].float(), data["RMB"][seq_idx].float(), "official"
    if imu_input_mode == "processed":
        if not has_l4:
            raise KeyError("processed mode requires l4_aM/l4_wM/l4_RMB fields in cache")
        return data["l4_aM"][seq_idx].float(), data["l4_wM"][seq_idx].float(), data["l4_RMB"][seq_idx].float(), "processed"
    if imu_input_mode == "auto":
        if has_l4:
            return data["l4_aM"][seq_idx].float(), data["l4_wM"][seq_idx].float(), data["l4_RMB"][seq_idx].float(), "processed"
        return data["aM"][seq_idx].float(), data["wM"][seq_idx].float(), data["RMB"][seq_idx].float(), "official"
    raise ValueError(f"Unsupported imu_input_mode={imu_input_mode}")


def pose_source(data, seq_idx, source):
    if source == "pose_prephysics":
        if "pose_prephysics" not in data or not data["pose_prephysics"]:
            raise KeyError("pose_prephysics source requested but cache does not contain pose_prephysics")
        return data["pose_prephysics"][seq_idx].float(), "pose_prephysics"
    if source == "pose_gt":
        if "pose_gt" not in data or not data["pose_gt"]:
            raise KeyError("pose_gt source requested but cache does not contain pose_gt")
        return data["pose_gt"][seq_idx].float(), "pose_gt_diagnostic"
    if source == "auto":
        if "pose_prephysics" in data and data["pose_prephysics"]:
            return data["pose_prephysics"][seq_idx].float(), "pose_prephysics"
        if "pose_gt" in data and data["pose_gt"]:
            return data["pose_gt"][seq_idx].float(), "pose_gt_diagnostic"
    raise ValueError(f"Cannot resolve PL pose source={source}")


def trim_sequence(*xs, max_frames=0):
    n = min(x.shape[0] for x in xs)
    if max_frames:
        n = min(n, max_frames)
    return [x[:n] for x in xs], n


@torch.no_grad()
def pl_output_from_pose(pose, body_model, chunk=2048):
    rows = []
    for start in range(0, pose.shape[0], chunk):
        p = pose[start : start + chunk].to(DEVICE)
        rows.append(normalize_gravity(pl_target_from_pose(p, body_model)).detach().cpu())
    return torch.cat(rows, dim=0)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    config = checkpoint.get("config", {})
    model = IMUOffsetNet(
        version=checkpoint.get("version", config.get("version", "offset_v1_mlp_frame")),
        input_size=int(config.get("feature_dim", config.get("input_size", 108))),
        hidden_size=int(config.get("hidden_size", 256)),
        prior_offset=torch.tensor(config.get("prior_offset_median", torch.zeros(6, 3))).float(),
        residual_scale=float(config.get("residual_scale", 0.05)),
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, config


@torch.no_grad()
def predict_sequence(model, pl_output, aM, wM, RMB, window, stride):
    preds = []
    n = pl_output.shape[0]
    if n <= window:
        starts = [0]
    else:
        starts = list(range(0, n - window + 1, stride))
        if starts[-1] != n - window:
            starts.append(n - window)
    for start in starts:
        end = min(n, start + window)
        feature = offset_input_feature(pl_output[start:end], aM[start:end], wM[start:end], RMB[start:end]).unsqueeze(0).to(DEVICE)
        pred = model(feature)[0].detach().cpu()
        preds.append(pred)
    frame_offsets = torch.cat(preds, dim=0)
    sequence_offset = frame_offsets.mean(dim=0)
    return sequence_offset, frame_offsets


def offset_gt_from_cache(data, seq_idx):
    for key in ("offset_r", "imu_offset_r", "r_JS"):
        if key in data and data[key]:
            value = data[key][seq_idx].float()
            if value.shape == (6, 3):
                return value, key
    return None, None


def acc_audit(data, seq_idx, offset_r, aM, max_frames, mode, dataset_label):
    if mode == "skip":
        return {"available": False, "reason": "skipped_by_user"}
    label = dataset_label.lower()
    if mode == "auto" and ("dip" in label or "dip" in str(data.get("source_input", "")).lower()):
        return {"available": False, "reason": "DIP trans is not reliable for physical acceleration consistency"}
    if "pose_gt" not in data or "tran_gt" not in data or not data["pose_gt"] or not data["tran_gt"]:
        return {"available": False, "reason": "pose_gt/tran_gt not available"}
    pose = data["pose_gt"][seq_idx].float()
    tran = data["tran_gt"][seq_idx].float()
    (pose, tran, aM), n = trim_sequence(pose, tran, aM, max_frames=max_frames)
    if n < 4:
        return {"available": False, "reason": "sequence shorter than acceleration stencil"}
    err = acceleration_consistency_error(pose, tran, offset_r, aM, device="cpu")
    return {"available": True, **err}


def infer_file(source_path, dest_path, model, body_model, args, manifest_label):
    data = torch.load(source_path, map_location="cpu")
    out = dict(data)
    predicted_offsets = []
    rows = []
    n_seq = len(data["name"])
    selected_count = n_seq if not args.max_sequences else min(n_seq, args.max_sequences)
    for seq_idx in tqdm.tqdm(range(selected_count), desc=source_path.name):
        name = str(data["name"][seq_idx])
        pose, pl_source = pose_source(data, seq_idx, args.pl_source)
        aM, wM, RMB, imu_source = selected_imu_fields(data, seq_idx, args.imu_input_mode)
        (pose, aM, wM, RMB), n = trim_sequence(pose, aM, wM, RMB, max_frames=args.max_frames)
        if n < 4:
            raise RuntimeError(f"{source_path} {name} has only {n} usable frames")
        pl_output = pl_output_from_pose(pose, body_model, chunk=args.pl_chunk)
        seq_offset, frame_offsets = predict_sequence(model, pl_output, aM, wM, RMB, args.window, args.stride)
        predicted_offsets.append(seq_offset.float())
        row = {
            "name": name,
            "num_frames_used": int(n),
            "pl_source": pl_source,
            "imu_source": imu_source,
            "offset_gt_available": False,
            "offset_gt_note": "offset GT not available for real DIP/TotalCapture caches unless --offset-gt-mode synthetic is used",
            "pred_offset_norm_mean_m": float(seq_offset.norm(dim=-1).mean()),
            "pred_offset_norm_max_m": float(seq_offset.norm(dim=-1).max()),
            "pred_temporal_stability_cm": float(frame_offsets.std(dim=0).norm(dim=-1).mean() * 100.0),
        }
        gt, gt_key = offset_gt_from_cache(data, seq_idx)
        if args.offset_gt_mode == "synthetic" or (args.offset_gt_mode == "auto" and gt is not None and "amass" in manifest_label.lower()):
            row.update(offset_errors_cm(frame_offsets.unsqueeze(0), gt))
            row["offset_gt_available"] = True
            row["offset_gt_source_key"] = gt_key
            row["offset_gt_note"] = "synthetic/cache GT used only because this source is explicitly treated as synthetic"
        audit = acc_audit(data, seq_idx, seq_offset, aM, args.max_frames, args.acc_audit_mode, manifest_label)
        row["acc_consistency"] = audit
        rows.append(row)
    if selected_count != n_seq:
        for seq_idx in range(selected_count, n_seq):
            gt, _ = offset_gt_from_cache(data, seq_idx)
            predicted_offsets.append(gt.float() if gt is not None else torch.zeros(6, 3))
    out["offset_r"] = predicted_offsets
    out["imu_offset_pred_r"] = predicted_offsets
    out["imu_offset_pred_contract"] = OFFSET_COORDINATE_CONTRACT
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dest_path)
    return {
        "source_path": str(source_path),
        "path": str(dest_path),
        "num_sequences": int(n_seq),
        "num_sequences_predicted": int(selected_count),
        "rows": rows,
    }


def aggregate_rows(rows):
    if not rows:
        return {}
    out = {
        "num_sequences": len(rows),
        "offset_gt_available": all(row.get("offset_gt_available", False) for row in rows),
        "pred_offset_norm_mean_m": sum(row["pred_offset_norm_mean_m"] for row in rows) / len(rows),
        "pred_offset_norm_max_m": max(row["pred_offset_norm_max_m"] for row in rows),
        "pred_temporal_stability_cm": sum(row["pred_temporal_stability_cm"] for row in rows) / len(rows),
    }
    gt_rows = [row for row in rows if row.get("offset_gt_available")]
    if gt_rows:
        out["offset_l1_cm"] = sum(row["offset_l1_cm"] for row in gt_rows) / len(gt_rows)
        out["offset_l2_cm"] = sum(row["offset_l2_cm"] for row in gt_rows) / len(gt_rows)
    else:
        out["offset_l1_cm"] = "not available"
        out["offset_l2_cm"] = "not available"
    acc_rows = [row["acc_consistency"] for row in rows if row.get("acc_consistency", {}).get("available")]
    out["acc_consistency_available"] = bool(acc_rows)
    out["acc_consistency_mean"] = sum(row["acc_consistency_mean"] for row in acc_rows) / len(acc_rows) if acc_rows else "not measured"
    return out


def main():
    parser = argparse.ArgumentParser(description="Infer sequence-level IMUOffsetNet offset_r and write an enriched L4 cache.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imu-input-mode", choices=("official", "processed", "auto"), required=True)
    parser.add_argument("--pl-source", choices=("pose_prephysics", "pose_gt", "auto"), default="pose_prephysics")
    parser.add_argument("--offset-gt-mode", choices=("unavailable", "synthetic", "auto"), default="unavailable")
    parser.add_argument("--acc-audit-mode", choices=("auto", "pose_gt", "skip"), default="auto")
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--pl-chunk", type=int, default=2048)
    args = parser.parse_args()

    model, checkpoint, config = load_checkpoint(args.checkpoint)
    body_model = art.ParametricModel("models/SMPL_male.pkl", vert_mask=torch.tensor([1961, 5424, 1176, 4662, 411, 3021]), device=DEVICE)
    cache_files, manifest = load_cache_files(args.input_cache)
    manifest_label = json.dumps(manifest or {"input_cache": str(args.input_cache)})
    output_manifest = dict(manifest or {})
    output_manifest.update(
        {
            "source_cache_manifest": str(args.input_cache),
            "imu_offset_checkpoint": str(args.checkpoint),
            "imu_offset_version": checkpoint.get("version", config.get("version")),
            "imu_offset_contract": OFFSET_COORDINATE_CONTRACT,
            "imu_input_mode": args.imu_input_mode,
            "pl_source": args.pl_source,
            "offset_gt_mode": args.offset_gt_mode,
            "real_data_offset_gt_policy": "DIP/TotalCapture offset GT not available; no real-data offset accuracy is reported.",
            "cache_files": [],
        }
    )
    summaries = []
    all_rows = []
    for source in cache_files:
        source_path = resolve_path(source, args.input_cache.parent)
        dest_path = args.output_dir / source_path.name
        summary = infer_file(source_path, dest_path, model, body_model, args, manifest_label)
        summaries.append({k: v for k, v in summary.items() if k != "rows"})
        all_rows.extend(summary["rows"])
        item = {"path": str(dest_path), "source_path": str(source_path), "num_sequences": summary["num_sequences"]}
        output_manifest["cache_files"].append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "baseline_cache_manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
    result = {
        "status": "ok",
        "output_manifest": str(manifest_path),
        "checkpoint": str(args.checkpoint),
        "version": checkpoint.get("version", config.get("version")),
        "coordinate_contract": OFFSET_COORDINATE_CONTRACT,
        "input_cache": str(args.input_cache),
        "imu_input_mode": args.imu_input_mode,
        "pl_source": args.pl_source,
        "offset_gt_mode": args.offset_gt_mode,
        "files": summaries,
        "aggregate": aggregate_rows(all_rows),
        "rows": all_rows,
    }
    result_path = args.output_dir / "imu_offset_infer_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path), "result": str(result_path), "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
