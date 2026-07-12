"""Stateful full-sequence training for PL-VA-State-V1."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from pl_va_state import (ANGULAR_VELOCITY_EMA_BETA, ANGULAR_VELOCITY_FRAME,
                         ANGULAR_VELOCITY_LAG, ANGULAR_VELOCITY_METHOD,
                         PLVAStateV1, partial_initialize_from_official)


WEIGHTS = {"p": 1.0, "v_direct": 0.25, "v_state": 0.25, "a": 0.1,
           "g": 1.0, "consistency": 0.2, "jerk": 0.01}


def load_records(manifest_path, max_sequences=0):
    manifest = json.loads(Path(manifest_path).read_text()); rows = []
    angular = manifest.get("angular_velocity", {})
    if angular.get("method") != ANGULAR_VELOCITY_METHOD:
        raise ValueError(f"refusing incompatible PL-VA cache angular velocity: {angular.get('method')!r}")
    for item in manifest["cache_files"]:
        rows.extend(torch.load(item["path"], map_location="cpu"))
        if max_sequences and len(rows) >= max_sequences: break
    return rows[:max_sequences or None], manifest


def normalization(records):
    v = torch.cat([r["v_gt"] for r in records]); a = torch.cat([r["a_gt"] for r in records])
    return {"v_mean": v.mean(0), "v_std": v.std(0).clamp_min(1e-4),
            "a_mean": a.mean(0), "a_std": a.std(0).clamp_min(1e-4), "fit_split": "train_only"}


def batches(rows, batch_size, shuffle):
    order = list(range(len(rows)))
    if shuffle: random.shuffle(order)
    for start in range(0, len(order), batch_size): yield [rows[i] for i in order[start:start + batch_size]]


def collate(rows, device):
    lengths = torch.tensor([r["length"] for r in rows], device=device); t = int(lengths.max())
    def pad(key, width):
        out = torch.zeros(len(rows), t, width, device=device)
        for i, r in enumerate(rows): out[i, :r["length"]] = r[key].to(device)
        return out
    mask = torch.arange(t, device=device)[None] < lengths[:, None]
    return {"feature": pad("feature", 102), "p": pad("p_gt", 15), "v": pad("v_gt", 15),
            "a": pad("a_gt", 15), "g": pad("g_gt", 3), "init": torch.stack([r["init_legacy"] for r in rows]).to(device),
            "mask": mask, "lengths": lengths}


def masked_huber(x, y, mask):
    value = F.smooth_l1_loss(x, y, reduction="none").mean(-1)
    return value[mask].mean()


def losses(output, batch, stats):
    mask, dt = batch["mask"], 1 / 60
    lv = masked_huber((output["vRB_direct"] - stats["v_mean"]) / stats["v_std"],
                      (batch["v"] - stats["v_mean"]) / stats["v_std"], mask)
    lvs = masked_huber((output["vRB_state"] - stats["v_mean"]) / stats["v_std"],
                       (batch["v"] - stats["v_mean"]) / stats["v_std"], mask)
    la = masked_huber((output["aRB_leaf"] - stats["a_mean"]) / stats["a_std"],
                      (batch["a"] - stats["a_mean"]) / stats["a_std"], mask)
    raw = {"p": masked_huber(output["pRB_state"], batch["p"], mask), "v_direct": lv, "v_state": lvs, "a": la,
           "g": (1 - (output["gR1"] * batch["g"]).sum(-1).clamp(-1, 1))[mask].mean()}
    pair = mask[:, 1:] & mask[:, :-1]
    dv = output["vRB_state"][:, 1:] - output["vRB_state"][:, :-1]
    da = 0.5 * dt * (output["aRB_leaf"][:, :-1] + output["aRB_leaf"][:, 1:])
    raw["consistency"] = masked_huber(dv, da, pair)
    raw["jerk"] = masked_huber(output["aRB_leaf"][:, 1:] - output["aRB_leaf"][:, :-1], torch.zeros_like(da), pair)
    weighted = {k: raw[k] * WEIGHTS[k] for k in raw}; total = sum(weighted.values())
    return total, raw, weighted


@torch.no_grad()
def validate(model, rows, batch_size, device, stats):
    model.eval(); sums = {"p": 0., "v_state": 0., "a": 0., "g": 0., "total": 0.}; n = 0
    for group in batches(rows, batch_size, False):
        b = collate(group, device); out = model.forward_sequence(b["feature"], b["init"], b["lengths"])
        total, raw, _ = losses(out, b, stats)
        for k in ("p", "v_state", "a", "g"): sums[k] += raw[k].item() * len(group)
        sums["total"] += total.item() * len(group); n += len(group)
    result = {k: v / max(n, 1) for k, v in sums.items()}
    result["selection"] = result["p"] + .25 * result["v_state"] + .1 * result["a"] + result["g"]
    return result


def gradient_report(model):
    w = model.net.linear2.weight.grad
    return {"v_head": float(w[:15].norm()) if w is not None else 0.,
            "a_head": float(w[15:30].norm()) if w is not None else 0.,
            "gravity_head": float(w[30:33].norm()) if w is not None else 0.}


def per_loss_gradient_report(model, weighted):
    parameter = model.net.linear2.weight
    report = {}
    for name, value in weighted.items():
        grad = torch.autograd.grad(value, parameter, retain_graph=True, allow_unused=True)[0]
        report[name] = 0.0 if grad is None else float(grad.norm())
    return report


def main():
    p = argparse.ArgumentParser(); p.add_argument("--train-cache", type=Path, required=True); p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-train-sequences", type=int, default=0); p.add_argument("--max-val-sequences", type=int, default=0)
    p.add_argument("--init-checkpoint", type=Path); p.add_argument("--weights", type=Path, default=Path("data/weights.pt"))
    p.add_argument("--seed", type=int, default=42); p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, train_manifest = load_records(args.train_cache, args.max_train_sequences); val, _ = load_records(args.val_cache, args.max_val_sequences)
    model = PLVAStateV1().to(device)
    init_report = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("config", {}).get("angular_velocity_method") != ANGULAR_VELOCITY_METHOD:
            raise ValueError("refusing checkpoint trained with an incompatible PL-VA angular-velocity feature")
        model.load_state_dict(checkpoint["model"]); stats = checkpoint["normalization"]
    else:
        init_report = partial_initialize_from_official(model, args.weights, args.output_dir / "initialization_report.json")
        stats = normalization(train)
    stats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in stats.items()}
    config = vars(args) | {"beta": model.beta, "cutoff_hz": model.cutoff_hz, "filter_order": model.filter_order,
                           "loss_weights": WEIGHTS, "state_reset": "once_per_full_sequence", "device": str(device),
                           "angular_velocity_method": ANGULAR_VELOCITY_METHOD,
                           "angular_velocity_frame": ANGULAR_VELOCITY_FRAME,
                           "angular_velocity_lag": ANGULAR_VELOCITY_LAG,
                           "angular_velocity_ema_beta": ANGULAR_VELOCITY_EMA_BETA}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr); best = float("inf"); history = []; smoke_grad = None; per_loss_grad = None
    for epoch in range(args.epochs):
        model.train(); epoch_rows = []
        for group in batches(train, args.batch_size, True):
            b = collate(group, device); opt.zero_grad(set_to_none=True)
            out = model.forward_sequence(b["feature"], b["init"], b["lengths"]); total, raw, weighted = losses(out, b, stats)
            if not torch.isfinite(total): raise RuntimeError("non-finite training loss")
            if per_loss_grad is None and args.epochs == 1:
                per_loss_grad = per_loss_gradient_report(model, weighted)
            total.backward(); smoke_grad = gradient_report(model); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
            epoch_rows.append({"total": total.item(), **{f"raw_{k}": v.item() for k,v in raw.items()},
                               **{f"weighted_{k}": v.item() for k,v in weighted.items()},
                               **{f"ratio_{k}": v.item()/max(total.item(),1e-12) for k,v in weighted.items()}})
        vr = validate(model, val, args.batch_size, device, stats); row = {"epoch": epoch + 1, "validation": vr,
            "train": {k: sum(x[k] for x in epoch_rows)/len(epoch_rows) for k in epoch_rows[0]}, "gradient_norm": smoke_grad}
        history.append(row); print(json.dumps(row), flush=True)
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch + 1,
                 "normalization": {k: (v.cpu() if torch.is_tensor(v) else v) for k,v in stats.items()}, "config": config, "validation": vr}
        torch.save(state, args.output_dir / "last.pt")
        if vr["selection"] < best: best = vr["selection"]; torch.save(state, args.output_dir / "best.pt")
    summary = {"status": "ok", "finite": True, "epochs": args.epochs, "best_selection": best,
               "last": history[-1], "gradient_norm": smoke_grad, "per_loss_gradient_norm": per_loss_grad,
               "initialization_report": init_report}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")


if __name__ == "__main__": main()
