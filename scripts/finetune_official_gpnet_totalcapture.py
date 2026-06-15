#!/usr/bin/env python3
"""
Minimal diagnostic TotalCapture fine-tuning entry for the official GPNet.

This is a diagnostic adaptation experiment, not the official training
protocol. TotalCapture is used for both fine-tuning and testing here only to
measure whether TotalCapture adaptation itself improves the official baseline.
"""

import argparse
import json
import os
import random
import shlex
import sys
import traceback
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import articulate as art  # noqa: E402
from l4_train_diverse_short import METRIC_NAMES, metric_to_dict, score_for_checkpoint  # noqa: E402
from net import GPNet  # noqa: E402
from pl_curve import pl_target_from_pose  # noqa: E402
from test import MotionEvaluator  # noqa: E402


DATA_PATHS = {
    "totalcapture": {
        "train": Path("data/dataset_work/TotalCapture_globalpose_official/train.pt"),
        "val": Path("data/dataset_work/TotalCapture_globalpose_official/val.pt"),
        "test": Path("data/dataset_work/TotalCapture_globalpose_official/test.pt"),
        "official_test": Path("data/test_datasets/totalcapture_officalib.pt"),
    }
}
DT = 1.0 / 60.0
GRAVITY = torch.tensor([0.0, -9.8, 0.0])


def load_state_payload(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload
    return payload, {"raw_state_dict": True}


def load_gpnet(official_ckpt, device):
    net = GPNet().to(device)
    state, payload = load_state_payload(official_ckpt)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"GPNet checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return net, payload


def totalcapture_to_model_fields(seq):
    aS = seq["aS"].float()
    wS = seq["wS"].float()
    RIS = seq["RIS"].float()
    RIM = seq["RIM"].float()
    RSB = seq["RSB"].float()
    RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB)
    aM = RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + GRAVITY
    wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1)
    return aM.float(), wM.float(), RMB.float()


def pl_input_features(aM, wM, RMB):
    aRB = aM.matmul(RMB[:, 5])
    wRB = wM.matmul(RMB[:, 5])
    RRB = RMB[:, 5].transpose(1, 2).unsqueeze(1).matmul(RMB[:, :5])
    gR0 = -RMB[:, 5, 1]
    return torch.cat((aRB.reshape(aM.shape[0], 18), wRB.reshape(aM.shape[0], 18), RRB.reshape(aM.shape[0], 45), gR0), dim=-1)


def rotation_matrix_to_6d(rotation):
    return rotation[..., :, :2].reshape(rotation.shape[:-2] + (6,))


def finite_diff_velocity(tran):
    if tran.shape[0] <= 1:
        return torch.zeros_like(tran)
    vel = torch.zeros_like(tran)
    vel[1:] = (tran[1:] - tran[:-1]) / DT
    vel[0] = vel[1]
    return vel


@torch.no_grad()
def build_targets(pose_axis_angle, tran, body_model, j_reduce):
    pose = art.math.axis_angle_to_rotation_matrix(pose_axis_angle).view(-1, 24, 3, 3)
    pl_target = pl_target_from_pose(pose, body_model).float()

    pose_body = pose.clone()
    pose_body[:, 0] = torch.eye(3)
    global_pose, joints = body_model.forward_kinematics(pose_body)[:2]
    ik1_pRJ = joints[:, 1:].reshape(pose.shape[0], 69).float()
    ik1_gR2 = (-pose[:, 0, :, 1]).float()
    ik1_target = torch.cat((ik1_pRJ, ik1_gR2), dim=-1)

    ik2_target = rotation_matrix_to_6d(global_pose[:, j_reduce]).reshape(pose.shape[0], 90).float()

    vel_world = finite_diff_velocity(tran.float())
    vel_horizontal = vel_world.clone()
    vel_horizontal[:, 1] = 0.0
    vRR_H = pose[:, 0].transpose(1, 2).matmul(vel_horizontal.unsqueeze(-1)).squeeze(-1)
    vr_target = torch.zeros(pose.shape[0], 9)
    vr_target[:, 0] = vel_world[:, 1]
    vr_target[:, 1:4] = vRR_H
    return pose.float(), torch.cat((pl_target, ik1_target, ik2_target, vr_target), dim=-1)


def truncate_record(record, max_frames):
    if not max_frames or record["feature"].shape[0] <= max_frames:
        return record
    out = dict(record)
    for key in ("feature", "target", "pose_gt", "tran_gt", "aM", "wM", "RMB"):
        out[key] = record[key][:max_frames]
    return out


def split_csv(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def load_totalcapture_records(path, body_model, j_reduce, max_sequences=0, max_frames=0, include_names=None, exclude_names=None):
    data = torch.load(path, map_location="cpu")
    records = []
    include = set(include_names or [])
    exclude = set(exclude_names or [])
    for idx, name in enumerate(data["name"]):
        if include and name not in include:
            continue
        if name in exclude:
            continue
        seq = {key: data[key][idx] for key in ("aS", "wS", "RIS", "RIM", "RSB", "pose", "tran")}
        aM, wM, RMB = totalcapture_to_model_fields(seq)
        feature = pl_input_features(aM, wM, RMB)
        pose_gt, target = build_targets(seq["pose"].float(), seq["tran"].float(), body_model, j_reduce)
        record = {
            "name": name,
            "feature": feature.float(),
            "target": target.float(),
            "pose_gt": pose_gt.float(),
            "tran_gt": seq["tran"].float(),
            "aM": aM.float(),
            "wM": wM.float(),
            "RMB": RMB.float(),
        }
        records.append(truncate_record(record, max_frames))
        if max_sequences and len(records) >= max_sequences:
            break
    return records


def slice_record(record, start, length):
    if length <= 0 or record["feature"].shape[0] <= length:
        return record
    start = min(max(0, int(start)), record["feature"].shape[0] - length)
    end = start + length
    out = dict(record)
    for key in ("feature", "target", "pose_gt", "tran_gt", "aM", "wM", "RMB"):
        out[key] = record[key][start:end]
    out["name"] = f"{record['name']}[{start}:{end}]"
    return out


def make_batch(records, starts, length):
    return [slice_record(record, start, length) for record, start in zip(records, starts)]


def cosine_loss(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target.to(pred.device, pred.dtype), avoid_nan=True)
    return (1.0 - (pred * target).sum(dim=-1).clamp(-1.0, 1.0)).mean()


def rotation_loss_6d(pred, target):
    pred_R = art.math.r6d_to_rotation_matrix(pred.contiguous().reshape(pred.shape[:-1] + (15, 6)))
    target = target.to(pred.device, pred.dtype).contiguous()
    target_R = art.math.r6d_to_rotation_matrix(target.reshape(target.shape[:-1] + (15, 6)))
    rel = pred_R.transpose(-1, -2).matmul(target_R)
    trace = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


def average_dict(rows):
    out = {}
    for row in rows:
        for key, value in row.items():
            out.setdefault(key, []).append(float(value))
    return {key: sum(vals) / max(1, len(vals)) for key, vals in out.items()}


def sequence_loss(net, batch, device):
    xs = [(record["feature"].to(device), record["target"][0].to(device)) for record in batch]
    preds = net(xs, fast=True)
    losses = []
    components = []
    for pred, record in zip(preds, batch):
        target = record["target"].to(device)
        pl_pred, pl_t = pred[:, :18], target[:, :18]
        ik1_pred, ik1_t = pred[:, 18:90], target[:, 18:90]
        ik2_pred, ik2_t = pred[:, 90:180], target[:, 90:180]
        vr_pred, vr_t = pred[:, 180:189], target[:, 180:189]
        comp = {
            "pl_pRB": torch.nn.functional.smooth_l1_loss(pl_pred[:, :15], pl_t[:, :15]),
            "pl_gR1": cosine_loss(pl_pred[:, 15:], pl_t[:, 15:]),
            "ik1_pRJ": torch.nn.functional.smooth_l1_loss(ik1_pred[:, :69], ik1_t[:, :69]),
            "ik1_gR2": cosine_loss(ik1_pred[:, 69:], ik1_t[:, 69:]),
            "ik2_rot": rotation_loss_6d(ik2_pred, ik2_t),
            "vr_velocity": torch.nn.functional.smooth_l1_loss(vr_pred[:, :4], vr_t[:, :4]),
        }
        total = (
            comp["pl_pRB"]
            + comp["pl_gR1"]
            + comp["ik1_pRJ"]
            + comp["ik1_gR2"]
            + comp["ik2_rot"]
            + 0.1 * comp["vr_velocity"]
        )
        if pred.shape[0] >= 2:
            total = total + 0.01 * torch.nn.functional.smooth_l1_loss(pred[1:, :180] - pred[:-1, :180], target[1:, :180] - target[:-1, :180])
        losses.append(total)
        components.append({key: float(value.detach().cpu()) for key, value in comp.items()})
    loss = torch.stack(losses).mean()
    comps = average_dict(components)
    comps["loss"] = float(loss.detach().cpu())
    return loss, comps


@torch.no_grad()
def eval_training_loss(net, records, device, max_sequences=0):
    net.eval()
    rows = []
    selected = records[:max_sequences] if max_sequences else records
    for record in selected:
        _, comps = sequence_loss(net, [record], device)
        rows.append(comps)
    return average_dict(rows)


def save_training_checkpoint(path, net, optimizer, args, epoch, step, val_loss, trainable_modules):
    torch.save({
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        "epoch": epoch,
        "step": step,
        "validation_loss": val_loss,
        "model_type": "official_gpnet_totalcapture_finetune_diagnostic_v1",
        "trainable_modules": trainable_modules,
        "frozen_modules": [],
        "protocol_note": "This is a diagnostic adaptation experiment, not the official training protocol.",
    }, path)


def train(args):
    device = torch.device(args.device)
    net, official_payload = load_gpnet(args.official_ckpt, device)
    trainable_modules = ["plnet", "iknet.net1", "iknet.net2", "vrnet"]
    frozen_modules = []
    for param in net.parameters():
        param.requires_grad_(True)
    body_model = net.body_model
    train_path = Path(args.train_path) if args.train_path else DATA_PATHS[args.data]["train"]
    val_path = Path(args.val_path) if args.val_path else DATA_PATHS[args.data]["val"]
    train_records = load_totalcapture_records(
        train_path,
        body_model,
        net.j_reduce,
        args.max_train_sequences,
        args.max_train_frames,
        include_names=split_csv(args.train_include_names),
        exclude_names=split_csv(args.train_exclude_names),
    )
    val_records = load_totalcapture_records(
        val_path,
        body_model,
        net.j_reduce,
        args.max_val_sequences,
        args.max_val_frames,
        include_names=split_csv(args.val_include_names),
        exclude_names=split_csv(args.val_exclude_names),
    )
    if not train_records:
        raise RuntimeError(f"No train records loaded from {train_path}.")
    if not val_records:
        raise RuntimeError(f"No val records loaded from {val_path}.")
    if args.batch_size > 1:
        train_records = [record for record in train_records if record["feature"].shape[0] >= args.window]
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    best_loss = float("inf")
    best_epoch = 0
    step = 0
    history = []
    failed_reason = None
    log_path = save_dir / "train_log.jsonl"
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = list(range(len(train_records)))
        random.shuffle(order)
        rows = []
        for batch_start in range(0, len(order), args.batch_size):
            ids = order[batch_start:batch_start + args.batch_size]
            recs = [train_records[i] for i in ids]
            starts = [random.randint(0, max(0, record["feature"].shape[0] - args.window)) for record in recs]
            batch = make_batch(recs, starts, args.window)
            optimizer.zero_grad(set_to_none=True)
            loss, comps = sequence_loss(net, batch, device)
            if not torch.isfinite(loss):
                failed_reason = f"non-finite loss at epoch={epoch} step={step}"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            rows.append(comps)
        train_loss = average_dict(rows)
        val_loss = eval_training_loss(net, val_records, device, max_sequences=args.max_val_sequences)
        val_scalar = float(val_loss.get("loss", float("inf")))
        if not torch.isfinite(torch.tensor(val_scalar)):
            failed_reason = f"non-finite validation loss at epoch={epoch}"
        if failed_reason is None and val_scalar < best_loss:
            best_loss = val_scalar
            best_epoch = epoch
            save_training_checkpoint(save_dir / "best_checkpoint.pt", net, optimizer, args, epoch, step, val_loss, trainable_modules)
            torch.save(net.state_dict(), save_dir / "best_weights.pt")
        save_training_checkpoint(save_dir / "last_checkpoint.pt", net, optimizer, args, epoch, step, val_loss, trainable_modules)
        torch.save(net.state_dict(), save_dir / "last_weights.pt")
        row = {
            "epoch": epoch,
            "train": train_loss,
            "validation": val_loss,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "status": "failed" if failed_reason else "ok",
            "failed_reason": failed_reason,
        }
        history.append(row)
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if failed_reason:
            break
    result = {
        "status": "failed" if failed_reason else "ok",
        "failed_reason": failed_reason,
        "official_checkpoint": args.official_ckpt,
        "official_checkpoint_payload_keys": sorted(official_payload.keys()) if isinstance(official_payload, dict) else [],
        "data": args.data,
        "train_data_path": str(train_path),
        "val_data_path": str(val_path),
        "trainable_modules": trainable_modules,
        "frozen_modules": frozen_modules,
        "loss_contract": {
            "pl": "pRB SmoothL1 + gR1 cosine",
            "ik1": "pRJ SmoothL1 + gR2 cosine",
            "ik2": "reduced global rotation geodesic from 6D",
            "vr": "root vertical velocity + root-frame horizontal velocity SmoothL1; stationary/contact GT not measured",
        },
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "history": history,
        "checkpoints": {
            "best_checkpoint": str(save_dir / "best_checkpoint.pt"),
            "best_weights": str(save_dir / "best_weights.pt"),
            "last_checkpoint": str(save_dir / "last_checkpoint.pt"),
            "last_weights": str(save_dir / "last_weights.pt"),
        },
        "protocol_note": "This is a diagnostic adaptation experiment, not the official training protocol.",
    }
    (save_dir / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "best_epoch": best_epoch, "best_loss": best_loss, "failed_reason": failed_reason}, indent=2))
    if failed_reason:
        raise SystemExit(1)


@torch.no_grad()
def run_eval_sequence(net, record, device):
    net.rnn_initialize(record["pose_gt"][0])
    pose = torch.zeros_like(record["pose_gt"])
    tran = torch.zeros_like(record["tran_gt"])
    for frame_idx in range(record["pose_gt"].shape[0]):
        pose[frame_idx], tran[frame_idx] = net.forward_frame(
            record["aM"][frame_idx].to(device),
            record["wM"][frame_idx].to(device),
            record["RMB"][frame_idx].to(device),
        )
    return pose.cpu(), tran.cpu()


def aggregate_metric_rows(rows):
    out = {}
    for name in METRIC_NAMES:
        out[name] = {
            "mean": sum(row["model_metrics"][name]["mean"] for row in rows) / max(1, len(rows)),
            "std": sum(row["model_metrics"][name]["std"] for row in rows) / max(1, len(rows)),
        }
    return {
        "num_sequences": len(rows),
        "model_metrics": out,
        "baseline_metrics": out,
        "delta_metrics": {name: {"mean": 0.0, "std": 0.0} for name in METRIC_NAMES},
    }


def evaluate(args):
    device = torch.device(args.device)
    net, payload = load_gpnet(args.eval_ckpt or args.official_ckpt, device)
    body_model = net.body_model
    eval_path = Path(args.eval_path) if args.eval_path else DATA_PATHS[args.data][args.eval_split]
    records = load_totalcapture_records(
        eval_path,
        body_model,
        net.j_reduce,
        args.max_eval_sequences,
        include_names=split_csv(args.eval_include_names),
        exclude_names=split_csv(args.eval_exclude_names),
    )
    if not records:
        raise RuntimeError(f"No eval records loaded from {eval_path}.")
    evaluator = MotionEvaluator()
    net.eval()
    rows = []
    for record in records:
        if args.max_eval_frames and record["pose_gt"].shape[0] > args.max_eval_frames:
            record = slice_record(record, 0, args.max_eval_frames)
        pose, tran = run_eval_sequence(net, record, device)
        metric = evaluator(pose.to(device), record["pose_gt"].to(device), tran.to(device), record["tran_gt"].to(device)).cpu()
        rows.append({
            "name": record["name"],
            "model_metrics": metric_to_dict(metric),
            "finite": bool(torch.isfinite(pose).all() and torch.isfinite(tran).all()),
        })
        print(json.dumps({"sequence": record["name"], "finite": rows[-1]["finite"]}), flush=True)
    aggregate = aggregate_metric_rows(rows)
    result = {
        "status": "ok",
        "checkpoint": args.eval_ckpt or args.official_ckpt,
        "checkpoint_payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "data": args.data,
        "eval_split": args.eval_split,
        "eval_data_path": str(eval_path),
        "rows": rows,
        "aggregate": aggregate,
        "score": score_for_checkpoint(aggregate),
        "all_finite": all(row["finite"] for row in rows),
        "protocol_note": "This is a diagnostic adaptation experiment, not the official training protocol.",
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "score": result["score"], "all_finite": result["all_finite"], "output_json": str(output_json)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune or evaluate official GPNet on TotalCapture for a diagnostic adaptation experiment.")
    parser.add_argument("--action", choices=("train", "eval"), default="train")
    parser.add_argument("--official_ckpt", required=True)
    parser.add_argument("--data", choices=("totalcapture",), default="totalcapture")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", default="data/experiments/official_gpnet_totalcapture_finetune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window", type=int, default=180)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_sequences", type=int, default=0)
    parser.add_argument("--max_val_sequences", type=int, default=0)
    parser.add_argument("--max_train_frames", type=int, default=0)
    parser.add_argument("--max_val_frames", type=int, default=0)
    parser.add_argument("--train_path", default="")
    parser.add_argument("--val_path", default="")
    parser.add_argument("--train_include_names", default="")
    parser.add_argument("--train_exclude_names", default="")
    parser.add_argument("--val_include_names", default="")
    parser.add_argument("--val_exclude_names", default="")
    parser.add_argument("--eval_ckpt", default="")
    parser.add_argument("--eval_split", choices=("test", "official_test", "val"), default="test")
    parser.add_argument("--eval_path", default="")
    parser.add_argument("--eval_include_names", default="")
    parser.add_argument("--eval_exclude_names", default="")
    parser.add_argument("--output_json", default="data/experiments/official_gpnet_totalcapture_finetune/eval.json")
    parser.add_argument("--max_eval_sequences", type=int, default=0)
    parser.add_argument("--max_eval_frames", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.action == "train":
            train(args)
        else:
            evaluate(args)
    except Exception as exc:
        if args.action == "eval":
            output_json = Path(args.output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "protocol_note": "This is a diagnostic adaptation experiment, not the official training protocol.",
            }, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
