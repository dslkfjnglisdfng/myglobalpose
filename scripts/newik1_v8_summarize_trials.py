#!/usr/bin/env python3
"""Summarize NewIK1 v8 loss-search trial outputs into a ranking table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path("data/experiments/newik1_v8_parallel_adaptive_loss_search")
OUT = ROOT / "summary" / "phase1_ranking.json"

BASELINES = {
    "newpl_init36_s4": 38.625657482802865,
    "v7_best_s4": 38.69478097228706,
    "official_ik1_pRJ_l2": 0.039413859534876514,
    "official_ik1_gR2_angle_deg": 25.584681532908032,
    "official_ik1_state_l2": 0.0712608013016359,
}


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"status": "read_failed", "error": str(exc)}


def metric_mean(data: Dict[str, Any], key: str) -> Optional[float]:
    item = data.get("aggregate", {}).get("model_metrics", {}).get(key)
    if isinstance(item, dict) and "mean" in item:
        return float(item["mean"])
    return None


def summarize_checkpoint(trial_dir: Path, ckpt: str) -> Dict[str, Any]:
    s4 = read_json(trial_dir / "s4" / ckpt / "result.json")
    gt = read_json(trial_dir / "module_gt" / ckpt / "result.json")
    row: Dict[str, Any] = {
        "checkpoint": ckpt,
        "s4_status": s4.get("status") if s4 else "missing",
        "module_gt_status": gt.get("status") if gt else "missing",
    }
    if s4 and s4.get("status") == "ok":
        row.update({
            "s4_score": float(s4["score"]),
            "s4_delta_vs_v7": float(s4["score"]) - BASELINES["v7_best_s4"],
            "s4_delta_vs_newpl": float(s4["score"]) - BASELINES["newpl_init36_s4"],
            "local_sip": metric_mean(s4, "L SIP Err (deg)"),
            "local_angle": metric_mean(s4, "L Angle Err (deg)"),
            "local_joint": metric_mean(s4, "L Joint Err (cm)"),
            "local_mesh": metric_mean(s4, "L Vertex Err (cm)"),
            "global_sip": metric_mean(s4, "G SIP Err (deg)"),
            "global_angle": metric_mean(s4, "G Angle Err (deg)"),
            "global_joint": metric_mean(s4, "G Joint Err (cm)"),
            "global_mesh": metric_mean(s4, "G Vertex Err (cm)"),
            "root_jitter": metric_mean(s4, "Root Jitter (km/s^3)"),
            "joint_jitter": metric_mean(s4, "Joint Jitter (km/s^3)"),
            "tail_update_norm_mean": s4.get("aggregate", {}).get("tail_update_norm_mean"),
        })
    if gt and gt.get("status") == "ok":
        agg = gt["aggregate"]
        new = agg["newik1"]
        delta = agg["delta_newik1_minus_baseline"]
        row.update({
            "pRJ_l2": float(new["pRJ_l2"]),
            "gR2_angle_deg": float(new["gR2_angle_deg"]),
            "state_l2": float(new["state_l2"]),
            "pRJ_l2_delta": float(delta["pRJ_l2"]),
            "gR2_angle_delta": float(delta["gR2_angle_deg"]),
            "state_l2_delta": float(delta["state_l2"]),
            "module_state_l2_beats_official": bool(delta["state_l2"] < 0),
        })
    return row


def decision(row: Dict[str, Any]) -> str:
    if row.get("s4_status") != "ok" or row.get("module_gt_status") != "ok":
        return "incomplete"
    s4_delta = row.get("s4_delta_vs_v7", 999.0)
    state_delta = row.get("state_l2_delta", 999.0)
    gr_delta = row.get("gR2_angle_delta", 999.0)
    prj_delta = row.get("pRJ_l2_delta", 999.0)
    if row.get("s4_score", 999.0) < BASELINES["newpl_init36_s4"] and (
        state_delta < 0 or gr_delta < 0 or prj_delta < 0
    ):
        return "candidate_mainline"
    if s4_delta < -0.02 or state_delta < -0.0003 or gr_delta < -0.2 or prj_delta < -0.0003:
        return "continue"
    if s4_delta > 0 and state_delta > 0 and gr_delta > 0 and prj_delta > 0:
        return "stop"
    return "watch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path, default=OUT)
    args = parser.parse_args()
    rows: List[Dict[str, Any]] = []
    for trial_dir in sorted(p for p in args.root.iterdir() if p.is_dir() and p.name.startswith("v8_")):
        train = read_json(trial_dir / "train" / "train_result.json") or {}
        for ckpt in ("best_loss", "last"):
            row = {"trial": trial_dir.name}
            row["best_epoch"] = train.get("best_epoch")
            row["best_loss"] = train.get("best_loss")
            weights = train.get("weights") or {}
            row["weights"] = weights
            row.update(summarize_checkpoint(trial_dir, ckpt))
            row["decision"] = decision(row)
            rows.append(row)
    rows.sort(key=lambda r: (
        0 if r["decision"] == "candidate_mainline" else 1 if r["decision"] == "continue" else 2,
        r.get("s4_score", 999.0),
        r.get("state_l2_delta", 999.0),
    ))
    result = {"baselines": BASELINES, "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output_json": str(args.output_json),
        "num_rows": len(rows),
        "top": rows[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
