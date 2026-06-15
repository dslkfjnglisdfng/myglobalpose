#!/usr/bin/env python3
"""IK1 fixed-newpl_v4_init36 auto-search queue and result utilities."""

from __future__ import annotations

import argparse
import csv
import json
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:
    yaml = None


BASELINE_S4 = 38.625657482802865
ROOT = Path("data/experiments/ik1_auto_search")
QUEUE = Path("experiments/ik1_auto_search_queue.yaml")
ROUND1_QUEUE = Path("experiments/ik1_auto_search_round1_queue.yaml")
ROUND1_RETRY1_QUEUE = Path("experiments/ik1_auto_search_round1_retry1_queue.yaml")
RESULTS = Path("experiments/ik1_auto_search_results.csv")
ENV = "/home/lingfeng/.conda/envs/globalpose-gpu"
PY = f'"{ENV}/bin/python"'
PL = "data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt"
S4_CACHE = "data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json"
S5_CACHE = "data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json"
V6A = "data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt"
TRAIN_CACHE_OFFICIAL = "data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json"
VAL_CACHE_OFFICIAL = "data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json"

SEEDS: Dict[str, Dict[str, str]] = {
    "baseline_official_ik1": {"checkpoint": "", "backend": "original", "eval": "pl_curve_eval.py"},
    "newik1_v4_official_input": {
        "checkpoint": "data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt",
        "backend": "official_input_v1",
        "eval": "newik1_official_input_eval.py",
    },
    "newik1_v6_stage_a": {
        "checkpoint": "data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt",
        "backend": "official_input_v1",
        "eval": "newik1_official_input_eval.py",
    },
    "newik1_v7_best": {
        "checkpoint": "data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt",
        "backend": "auto_control_point",
        "eval": "newik1_control_eval.py",
    },
    "newik1_v8_B4_last": {
        "checkpoint": "data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt",
        "backend": "auto_control_point",
        "eval": "newik1_control_eval.py",
    },
    "newik1_v9_C8_last": {
        "checkpoint": "data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt",
        "backend": "auto_control_point",
        "eval": "newik1_control_eval.py",
    },
}

METRIC_KEYS = [
    ("score", "score"),
    ("local_sip", "L SIP Err (deg)"),
    ("local_angle", "L Angle Err (deg)"),
    ("local_joint", "L Joint Err (cm)"),
    ("local_mesh", "L Vertex Err (cm)"),
    ("global_sip", "G SIP Err (deg)"),
    ("global_angle", "G Angle Err (deg)"),
    ("global_joint", "G Joint Err (cm)"),
    ("global_mesh", "G Vertex Err (cm)"),
    ("root_jitter", "Root Jitter (km/s^3)"),
    ("joint_jitter", "Joint Jitter (km/s^3)"),
]

MODULE_KEYS = [
    "pRJ_l1_cm",
    "pRJ_l2_cm",
    "gR2_angle_deg",
    "pRJ_dot_l2",
    "pRJ_ddot_l2",
    "gR2_dot_l2",
    "gR2_ddot_l2",
]


def bash_prefix(gpu: int) -> str:
    return (
        f'ENV={ENV}; export CUDA_VISIBLE_DEVICES={gpu}; '
        f'export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${{LD_LIBRARY_PATH:-}}"; '
    )


def eval_command(version: str, split: str, out_json: str, gpu: int) -> str:
    seed = SEEDS[version]
    cache = S4_CACHE if split == "s4" else S5_CACHE
    if version == "baseline_official_ik1":
        return (
            bash_prefix(gpu)
            + f"{PY} pl_curve_eval.py --val-cache {cache} --output-json {out_json} "
            + f"--checkpoint {PL} --imu-input-mode processed"
        )
    return (
        bash_prefix(gpu)
        + f"{PY} {seed['eval']} --val-cache {cache} --output-json {out_json} "
        + f"--pl-checkpoint {PL} --ik1-checkpoint {seed['checkpoint']} --imu-input-mode processed"
    )


def real_streaming_command(version: str, split: str, out_json: str, gpu: int) -> str:
    seed = SEEDS[version]
    cache = S4_CACHE if split == "s4" else S5_CACHE
    split_label = split.upper()
    args = (
        bash_prefix(gpu)
        + f"{PY} newik1_real_streaming_audit.py --val-cache {cache} --output-json {out_json} "
        + f"--split-label {split_label} --version-name {version} --pl-checkpoint {PL} "
        + "--imu-input-mode processed "
    )
    if version == "baseline_official_ik1":
        return args + "--ik1-backend original"
    return args + f"--ik1-checkpoint {seed['checkpoint']} --ik1-backend {seed['backend']}"


def task(
    tid: str,
    name: str,
    typ: str,
    command: str,
    output: str,
    log: str,
    gpu: int,
    route: str,
    parent_checkpoint: str,
    priority: int,
    status: str = "pending",
) -> Dict[str, Any]:
    return {
        "id": tid,
        "experiment_id": tid,
        "name": name,
        "type": typ,
        "server": socket.gethostname(),
        "gpu": gpu,
        "status": status,
        "parent_checkpoint": parent_checkpoint or "official IK1",
        "pl_upstream": "newpl_v4_init36",
        "route": route,
        "command": command,
        "command_train": command if typ == "train" else "",
        "command_eval_s4": command if typ == "eval" and "/s4/" in output else "",
        "command_eval_s5": command if typ == "eval" and "/s5/" in output else "",
        "dependencies": [],
        "gpu_required": True,
        "estimated_gpu_mem_gb": 10,
        "priority": priority,
        "working_dir": ".",
        "env": {"CUDA_VISIBLE_DEVICES": str(gpu)},
        "outputs": [output],
        "log_path": log,
        "checkpoint_path": parent_checkpoint,
        "json_path": output,
        "summary_parser": "parse_s4_metrics" if typ == "eval" else "parse_generic_json",
        "project_status_section": "EXP-ik1-auto-search",
        "allow_parallel": True,
        "skip_if_outputs_exist": True,
    }


def existing_path(version: str, split: str, real: bool) -> Path:
    root = Path("data/experiments/fixed_init36_ik1_trend")
    if real:
        return root / "real_streaming" / split / version / "result.json"
    return root / split / version / "result.json"


def target_path(version: str, split: str, real: bool) -> Path:
    if real:
        return ROOT / "round0" / "real_streaming" / split / version / "result.json"
    return ROOT / "round0" / "full_pipeline" / split / version / "result.json"


def seed_status() -> Dict[str, str]:
    out = {}
    for version, seed in SEEDS.items():
        ckpt = seed["checkpoint"]
        out[version] = "official" if not ckpt else ("found" if Path(ckpt).exists() else "not found")
    return out


def build_round0_tasks() -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    gpu = 0
    priority = 1
    for version in SEEDS:
        ckpt = SEEDS[version]["checkpoint"]
        if ckpt and not Path(ckpt).exists():
            continue
        for split in ("s4", "s5"):
            if existing_path(version, split, real=False).exists():
                continue
            out = str(target_path(version, split, real=False))
            tasks.append(task(
                f"round0_full_{split}_{version}",
                f"Round0 {split.upper()} full-pipeline {version}",
                "eval",
                eval_command(version, split, out, gpu),
                out,
                f"logs/orchestrator/ik1_auto_search/round0/full_{split}_{version}.log",
                gpu,
                "round0_full_pipeline_completion",
                ckpt,
                priority,
            ))
            gpu = 1 - gpu
            priority += 1
        for split in ("s4", "s5"):
            if existing_path(version, split, real=True).exists():
                continue
            out = str(target_path(version, split, real=True))
            tasks.append(task(
                f"round0_real_{split}_{version}",
                f"Round0 {split.upper()} real streaming IK1 audit {version}",
                "audit",
                real_streaming_command(version, split, out, gpu),
                out,
                f"logs/orchestrator/ik1_auto_search/round0/real_{split}_{version}.log",
                gpu,
                "round0_real_streaming_ik1_vs_gt",
                ckpt,
                priority,
            ))
            gpu = 1 - gpu
            priority += 1
    return tasks


def write_queue() -> None:
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "name": "ik1_auto_search",
        "state_file": "data/experiments/orchestrator_states/ik1_auto_search_queue.json",
        "allow_same_user_gpu_share": False,
        "baseline": {
            "pl_upstream": "newpl_v4_init36",
            "pl_checkpoint": PL,
            "imu_input_mode": "processed",
            "ik1_baseline": "official IK1",
            "s4_score": BASELINE_S4,
            "selection_rule": "S4/S5 full-pipeline 11 metrics first; real streaming IK1 vs GT diagnostic; AMASS/cache diagnostic only.",
        },
        "seed_checkpoints": seed_status(),
        "round": 0,
        "tasks": build_round0_tasks(),
        "planned_round1": [
            "v10_residual_pRJ_only_alpha025_from_v6a",
            "v10_residual_pRJ_only_alpha05_from_v6a",
            "v10_stage_a_low_lr_distill_official",
            "v10_ik2_input_distill_from_v6a",
        ],
    }
    if yaml is None:
        QUEUE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    else:
        QUEUE.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def py_official_train_cmd(exp: Dict[str, Any], gpu: int, round_dir: str = "round1") -> str:
    out_dir = ROOT / round_dir / exp["id"] / "train"
    args = [
        "newik1_official_input_train.py",
        "--train-cache", TRAIN_CACHE_OFFICIAL,
        "--val-cache", VAL_CACHE_OFFICIAL,
        "--output-dir", str(out_dir),
        "--experiment-name", exp["id"],
        "--epochs", str(exp.get("epochs", 3)),
        "--lr", str(exp.get("lr", "3e-6")),
        "--dropout", str(exp.get("dropout", 0.15)),
        "--batch-size", str(exp.get("batch_size", 16)),
        "--window", "61",
        "--init-checkpoint", exp.get("init", V6A),
        "--output-mode", exp.get("output_mode", "full"),
        "--residual-alpha", str(exp.get("residual_alpha", 1.0)),
    ]
    for key, value in exp.get("weights", {}).items():
        args.extend([f"--{key.replace('_', '-')}-weight", str(value)])
    return bash_prefix(gpu) + PY + " " + " ".join(args)


def round1_experiments() -> List[Dict[str, Any]]:
    return [
        {
            "id": "v10_residual_pRJ_only_alpha025_from_v6a",
            "route": "residual_pRJ_only",
            "output_mode": "residual_pRJ_only",
            "residual_alpha": 0.25,
            "lr": "3e-6",
            "epochs": 3,
            "weights": {
                "pRJ": 2.0,
                "gR2": 0.0,
                "gR2_dot": 0.0,
                "gR2_ddot": 0.0,
                "ik1_distill_pRJ": 0.5,
                "ik1_distill_gR2": 1.0,
                "ik2_input_distill": 0.1,
            },
        },
        {
            "id": "v10_residual_pRJ_only_alpha05_from_v6a",
            "route": "residual_pRJ_only",
            "output_mode": "residual_pRJ_only",
            "residual_alpha": 0.5,
            "lr": "3e-6",
            "epochs": 3,
            "weights": {
                "pRJ": 2.0,
                "gR2": 0.0,
                "gR2_dot": 0.0,
                "gR2_ddot": 0.0,
                "ik1_distill_pRJ": 0.5,
                "ik1_distill_gR2": 1.0,
                "ik2_input_distill": 0.1,
            },
        },
        {
            "id": "v10_stage_a_low_lr_distill_official",
            "route": "stage_a_conservative_finetune",
            "output_mode": "full",
            "residual_alpha": 1.0,
            "lr": "1e-6",
            "epochs": 3,
            "weights": {
                "pRJ": 1.0,
                "gR2": 0.5,
                "pRJ_dot": 0.02,
                "gR2_dot": 0.01,
                "pRJ_ddot": 0.001,
                "gR2_ddot": 0.0005,
                "ik1_distill_pRJ": 1.0,
                "ik1_distill_gR2": 1.0,
                "ik2_input_distill": 0.1,
            },
        },
        {
            "id": "v10_ik2_input_distill_from_v6a",
            "route": "downstream_aware_ik2_input_distill",
            "output_mode": "residual_pRJ_only",
            "residual_alpha": 0.25,
            "lr": "3e-6",
            "epochs": 3,
            "weights": {
                "pRJ": 1.0,
                "gR2": 0.0,
                "gR2_dot": 0.0,
                "gR2_ddot": 0.0,
                "ik1_distill_pRJ": 0.5,
                "ik1_distill_gR2": 1.0,
                "ik2_input_distill": 2.0,
            },
        },
    ]


def build_round1_tasks(round_dir: str = "round1", task_prefix: str = "round1") -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    gpu = 0
    priority = 1
    for exp in round1_experiments():
        exp_id = exp["id"]
        ckpt = str(ROOT / round_dir / exp_id / "train" / "best_loss.pt")
        train_id = f"{task_prefix}_train_{exp_id}"
        train_out = [
            ckpt,
            str(ROOT / round_dir / exp_id / "train" / "last.pt"),
            str(ROOT / round_dir / exp_id / "train" / "train_result.json"),
        ]
        train_task = task(
            train_id,
            f"{task_prefix} train {exp_id}",
            "train",
            py_official_train_cmd(exp, gpu, round_dir=round_dir),
            train_out[0],
            f"logs/orchestrator/ik1_auto_search/{round_dir}/{exp_id}/train.log",
            gpu,
            exp["route"],
            V6A,
            priority,
        )
        train_task["outputs"] = train_out
        train_task["summary_parser"] = "parse_train_log"
        train_task["command_train"] = train_task["command"]
        train_task["notes"] = json.dumps(exp, sort_keys=True)
        tasks.append(train_task)
        s4_out = str(ROOT / round_dir / exp_id / "s4" / "best_loss" / "result.json")
        eval_task = task(
            f"{task_prefix}_s4_{exp_id}",
            f"{task_prefix} S4 full-pipeline {exp_id}",
            "eval",
            eval_command_for_checkpoint(exp_id, ckpt, s4_out, gpu, split="s4"),
            s4_out,
            f"logs/orchestrator/ik1_auto_search/{round_dir}/{exp_id}/s4_best_loss.log",
            gpu,
            exp["route"],
            ckpt,
            priority + 1,
        )
        eval_task["dependencies"] = [train_id]
        tasks.append(eval_task)
        real_out = str(ROOT / round_dir / exp_id / "real_streaming" / "s4" / "best_loss" / "result.json")
        real_task = task(
            f"{task_prefix}_real_s4_{exp_id}",
            f"{task_prefix} S4 real streaming IK1 audit {exp_id}",
            "audit",
            real_streaming_command_for_checkpoint(exp_id, ckpt, real_out, gpu, split="s4"),
            real_out,
            f"logs/orchestrator/ik1_auto_search/{round_dir}/{exp_id}/real_s4_best_loss.log",
            gpu,
            exp["route"],
            ckpt,
            priority + 2,
        )
        real_task["dependencies"] = [train_id]
        real_task["summary_parser"] = "parse_generic_json"
        tasks.append(real_task)
        gpu = 1 - gpu
        priority += 10
    return tasks


def eval_command_for_checkpoint(version: str, checkpoint: str, out_json: str, gpu: int, split: str) -> str:
    cache = S4_CACHE if split == "s4" else S5_CACHE
    return (
        bash_prefix(gpu)
        + f"{PY} newik1_official_input_eval.py --val-cache {cache} --output-json {out_json} "
        + f"--pl-checkpoint {PL} --ik1-checkpoint {checkpoint} --imu-input-mode processed"
    )


def real_streaming_command_for_checkpoint(version: str, checkpoint: str, out_json: str, gpu: int, split: str) -> str:
    cache = S4_CACHE if split == "s4" else S5_CACHE
    return (
        bash_prefix(gpu)
        + f"{PY} newik1_real_streaming_audit.py --val-cache {cache} --output-json {out_json} "
        + f"--split-label {split.upper()} --version-name {version} --pl-checkpoint {PL} "
        + f"--ik1-checkpoint {checkpoint} --ik1-backend official_input_v1 --imu-input-mode processed"
    )


def write_round1_queue() -> None:
    write_round1_like_queue(
        ROUND1_QUEUE,
        name="ik1_auto_search_round1",
        state_file="data/experiments/orchestrator_states/ik1_auto_search_round1_queue.json",
        round_dir="round1",
        task_prefix="round1",
    )


def write_round1_like_queue(queue_path: Path, name: str, state_file: str, round_dir: str, task_prefix: str) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "state_file": state_file,
        "allow_same_user_gpu_share": False,
        "baseline": {
            "pl_upstream": "newpl_v4_init36",
            "pl_checkpoint": PL,
            "imu_input_mode": "processed",
            "s4_score": BASELINE_S4,
            "selection_rule": "S4/S5 full-pipeline 11 metrics first; real streaming IK1 vs GT diagnostic; AMASS/cache diagnostic only.",
        },
        "round": 1,
        "parent_seed": "newik1_v6_stage_a",
        "parent_checkpoint": V6A,
        "tasks": build_round1_tasks(round_dir=round_dir, task_prefix=task_prefix),
    }
    if yaml is None:
        queue_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    else:
        queue_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def write_round1_retry1_queue() -> None:
    write_round1_like_queue(
        ROUND1_RETRY1_QUEUE,
        name="ik1_auto_search_round1_retry1",
        state_file="data/experiments/orchestrator_states/ik1_auto_search_round1_retry1_queue.json",
        round_dir="round1_retry1",
        task_prefix="round1_retry1",
    )


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_value(data: Optional[Dict[str, Any]], csv_key: str, json_key: str) -> Any:
    if data is None:
        return "not found"
    if csv_key == "score":
        return data.get("score", "not found")
    item = data.get("aggregate", {}).get("model_metrics", {}).get(json_key)
    if isinstance(item, dict):
        return item.get("mean", "not found")
    return item if item is not None else "not found"


def module_value(data: Optional[Dict[str, Any]], key: str) -> Any:
    if data is None:
        return "not found"
    item = data.get("ik1_module_aggregate", {}).get(key)
    if isinstance(item, dict):
        return item.get("mean", "not found")
    return item if item is not None else "not found"


def best_json(version: str, split: str, real: bool) -> Optional[Path]:
    candidates = [existing_path(version, split, real), target_path(version, split, real)]
    return next((p for p in candidates if p.exists()), None)


def write_results() -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "version", "seed_checkpoint_status", "parent_checkpoint", "pl_upstream",
        "s4_json", "s5_json", "real_s4_json", "real_s5_json",
        *[f"s4_{k}" for k, _ in METRIC_KEYS],
        *[f"s5_{k}" for k, _ in METRIC_KEYS],
        *[f"real_s4_{k}" for k in MODULE_KEYS],
        *[f"real_s5_{k}" for k in MODULE_KEYS],
        "s4_score_delta_vs_pl_only", "s5_score_delta_vs_official_ik1",
    ]
    statuses = seed_status()
    rows = []
    baseline_s5 = None
    b5_path = best_json("baseline_official_ik1", "s5", False)
    if b5_path:
        baseline_s5 = load_json(b5_path).get("score")
    for version, seed in SEEDS.items():
        row: Dict[str, Any] = {
            "version": version,
            "seed_checkpoint_status": statuses[version],
            "parent_checkpoint": seed["checkpoint"] or "official IK1",
            "pl_upstream": "newpl_v4_init36",
        }
        jsons = {}
        for split in ("s4", "s5"):
            p = best_json(version, split, False)
            row[f"{split}_json"] = str(p) if p else "not found"
            jsons[split] = load_json(p) if p else None
            for csv_key, json_key in METRIC_KEYS:
                row[f"{split}_{csv_key}"] = metric_value(jsons[split], csv_key, json_key)
        for split in ("s4", "s5"):
            p = best_json(version, split, True)
            row[f"real_{split}_json"] = str(p) if p else "not found"
            data = load_json(p) if p else None
            for key in MODULE_KEYS:
                row[f"real_{split}_{key}"] = module_value(data, key)
        s4_score = row.get("s4_score")
        row["s4_score_delta_vs_pl_only"] = float(s4_score) - BASELINE_S4 if isinstance(s4_score, (int, float)) else "not found"
        s5_score = row.get("s5_score")
        row["s5_score_delta_vs_official_ik1"] = (
            float(s5_score) - float(baseline_s5)
            if isinstance(s5_score, (int, float)) and isinstance(baseline_s5, (int, float))
            else "not found"
        )
        rows.append(row)
    with RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evidence() -> str:
    rows = []
    if RESULTS.exists():
        with RESULTS.open() as f:
            rows = list(csv.DictReader(f))
    completed = []
    missing = []
    for row in rows:
        for key in ("s4_json", "s5_json", "real_s4_json", "real_s5_json"):
            item = f"{row['version']}:{key}"
            (completed if row[key] != "not found" else missing).append(item)
    ranked = sorted(
        [r for r in rows if r.get("s4_score") not in ("", "not found")],
        key=lambda r: float(r["s4_score"]),
    )
    best_s4 = ranked[0]["version"] if ranked else "not found"
    best_overall = best_s4
    lines = [
        "[IK1 AUTO SEARCH ROUND 0 EVIDENCE]",
        "",
        "Round: 0 data completion / seed audit",
        f"Completed tasks: {', '.join(completed) if completed else 'none'}",
        f"Failed tasks: not assessed here; see orchestrator state/logs after running queue",
        f"Best S4 version: {best_s4}",
        "Best S5 version: see CSV ranking; S5 official baseline is diagnostic until S4 promising",
        f"Best overall version: {best_overall}",
        f"Does it beat PL-only best: {str(bool(ranked and float(ranked[0]['s4_score']) < BASELINE_S4)).lower()}",
        "Top-4 ranking:",
    ]
    for i, row in enumerate(ranked[:4], 1):
        lines.append(f"{i}. {row['version']} S4={row['s4_score']} delta={row['s4_score_delta_vs_pl_only']}")
    lines.extend([
        "Key trend: local/real IK1 diagnostics are not used for selection; missing real S5 diagnostics must be completed before final S5 claims.",
        "Next round planned experiments: v10_residual_pRJ_only_alpha025_from_v6a; v10_residual_pRJ_only_alpha05_from_v6a; v10_stage_a_low_lr_distill_official; v10_ik2_input_distill_from_v6a",
        f"Updated documents: {QUEUE}; {RESULTS}",
        "Logs: logs/orchestrator/ik1_auto_search/round0/*.log",
        "JSONs: data/experiments/ik1_auto_search/round0/**/result.json plus fixed_init36_ik1_trend existing JSONs",
        "Checkpoints: see seed checkpoint paths in CSV",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-queue", action="store_true")
    parser.add_argument("--write-round1-queue", action="store_true")
    parser.add_argument("--write-round1-retry1-queue", action="store_true")
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--evidence", action="store_true")
    args = parser.parse_args()
    if args.write_queue:
        write_queue()
    if args.write_round1_queue:
        write_round1_queue()
    if args.write_round1_retry1_queue:
        write_round1_retry1_queue()
    if args.write_results:
        write_results()
    if args.evidence:
        print(evidence())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
