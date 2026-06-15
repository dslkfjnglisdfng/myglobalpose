#!/usr/bin/env python3
"""Generate Phase-1 task shards for NewIK1 v8 parallel loss search.

The generated tasks intentionally keep the IK1 last-control input/output
contract fixed and evaluate every short trial with NewPL-streaming module-GT
diagnostics plus full S4 streaming metrics.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List


ENV_PREFIX = (
    'ENV=/home/lingfeng/.conda/envs/globalpose-gpu; '
    'export PATH="$ENV/bin:$PATH"; '
    'export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; '
    '"$ENV/bin/python"'
)

ROOT = Path("data/experiments/newik1_v8_parallel_adaptive_loss_search")
LOG_ROOT = Path("logs/orchestrator/newik1_v8_parallel_adaptive_loss_search")
STATE_ROOT = Path("data/experiments/orchestrator_states")

TRAIN_CACHE_TC = "data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_train/newik1_control_cache_manifest.json"
VAL_CACHE_TC = "data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json"
S4_CACHE = "data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json"
PL_CHECKPOINT = "data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt"
V7_BEST = "data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt"
V5_STAGE_A = "data/experiments/newik1_last_pl_control_20260605_v2/stage_a_gt_pretrain/best_loss.pt"
V5_STAGE_C = "data/experiments/newik1_last_pl_control_20260605_v2/stage_c_tc_finetune/best_loss.pt"

BASE_WEIGHTS: Dict[str, float] = {
    "pRJ": 1.0,
    "gR2": 1.0,
    "pRJ_dot": 0.03,
    "pRJ_ddot": 0.001,
    "gR2_dot": 0.03,
    "gR2_ddot": 0.001,
    "control_pRJ": 0.1,
    "control_gR2": 0.1,
    "control_pRJ_dot": 0.003,
    "control_gR2_dot": 0.003,
    "control_pRJ_ddot": 0.0001,
    "control_gR2_ddot": 0.0001,
    "gt_control_pRJ": 0.0,
    "gt_control_gR2": 0.0,
    "bone_length": 0.0,
    "control_point_prior": 0.0,
    "tail_update_prior": 0.0,
}


def q(value: Any) -> str:
    return shlex.quote(str(value))


def pycmd(script: str, args: Iterable[Any]) -> str:
    return " ".join([ENV_PREFIX, script, *[q(arg) for arg in args]])


def weight_args(weights: Dict[str, float]) -> List[str]:
    args: List[str] = []
    for key in sorted(weights):
        args.extend([f"--{key.replace('_', '-')}-weight", str(weights[key])])
    return args


def task(
    tid: str,
    name: str,
    typ: str,
    command: str,
    deps: List[str],
    outputs: List[str],
    log_path: str,
    parser: str,
    priority: int,
    gpu: str,
) -> Dict[str, Any]:
    return {
        "id": tid,
        "name": name,
        "type": typ,
        "command": command,
        "dependencies": deps,
        "gpu_required": True,
        "estimated_gpu_mem_gb": 8,
        "priority": priority,
        "working_dir": ".",
        "env": {"CUDA_VISIBLE_DEVICES": gpu},
        "outputs": outputs,
        "log_path": log_path,
        "summary_parser": parser,
        "project_status_section": "EXP-newik1_v8_parallel_adaptive_loss_search",
        "allow_parallel": True,
    }


def build_trial_tasks(trial: Dict[str, Any], gpu: str) -> List[Dict[str, Any]]:
    trial_id = trial["id"]
    out_dir = ROOT / trial_id / "train"
    log_dir = LOG_ROOT / trial_id
    weights = dict(BASE_WEIGHTS)
    weights.update(trial.get("weights", {}))
    train_args = [
        "newik1_control_train.py",
        "--train-cache", trial.get("train_cache", TRAIN_CACHE_TC),
        "--val-cache", VAL_CACHE_TC,
        "--output-dir", out_dir,
        "--experiment-name", trial_id,
        "--epochs", trial.get("epochs", 5),
        "--lr", trial.get("lr", "2e-6"),
        "--min-lr", trial.get("min_lr", "3e-7"),
        "--warmup-epochs", trial.get("warmup_epochs", 1),
        "--dropout", trial.get("dropout", 0.02),
        "--weight-decay", trial.get("weight_decay", "5e-5"),
        "--early-stop-patience", trial.get("patience", 3),
        "--early-stop-min-delta", trial.get("min_delta", "1e-5"),
        "--batch-size", trial.get("batch_size", 8),
        "--window", 61,
        "--init-checkpoint", trial.get("init", V7_BEST),
        *weight_args(weights),
    ]
    train_id = f"{trial_id}_train"
    tasks = [
        task(
            train_id,
            f"Train {trial_id}",
            "train",
            pycmd(train_args[0], train_args[1:]),
            [],
            [
                str(out_dir / "best_loss.pt"),
                str(out_dir / "last.pt"),
                str(out_dir / "train_result.json"),
            ],
            str(log_dir / "train.log"),
            "parse_train_log",
            1,
            gpu,
        )
    ]
    for ckpt_name in ("best_loss", "last"):
        ckpt = out_dir / f"{ckpt_name}.pt"
        module_out = ROOT / trial_id / "module_gt" / ckpt_name / "result.json"
        s4_out = ROOT / trial_id / "s4" / ckpt_name / "result.json"
        tasks.append(
            task(
                f"{trial_id}_module_gt_{ckpt_name}",
                f"Module GT {trial_id} {ckpt_name}",
                "audit",
                pycmd(
                    "newik1_local_diagnostic.py",
                    [
                        "--cache", VAL_CACHE_TC,
                        "--ik1-checkpoint", ckpt,
                        "--output-json", module_out,
                    ],
                ),
                [train_id],
                [str(module_out)],
                str(log_dir / f"module_gt_{ckpt_name}.log"),
                "parse_generic_json",
                2,
                gpu,
            )
        )
        tasks.append(
            task(
                f"{trial_id}_s4_{ckpt_name}",
                f"Full S4 {trial_id} {ckpt_name}",
                "eval",
                pycmd(
                    "newik1_control_eval.py",
                    [
                        "--val-cache", S4_CACHE,
                        "--pl-checkpoint", PL_CHECKPOINT,
                        "--ik1-checkpoint", ckpt,
                        "--imu-input-mode", "processed",
                        "--output-json", s4_out,
                    ],
                ),
                [train_id],
                [str(s4_out)],
                str(log_dir / f"s4_{ckpt_name}.log"),
                "parse_s4_metrics",
                3,
                gpu,
            )
        )
    return tasks


def phase1_trials() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "A_gR2": [
            {"id": "v8_A1_gR2_x2", "weights": {"gR2": 2.0}},
            {"id": "v8_A2_gR2_x4", "weights": {"gR2": 4.0}},
            {"id": "v8_A3_gR2_half", "weights": {"gR2": 0.5}},
            {"id": "v8_A4_gR2_x2_lowdyn", "weights": {"gR2": 2.0, "gR2_dot": 0.01, "gR2_ddot": 0.0003}},
        ],
        "B_pRJ_first": [
            {"id": "v8_B1_pRJ_x2", "weights": {"pRJ": 2.0}},
            {"id": "v8_B2_pRJ_x4", "weights": {"pRJ": 4.0}},
            {"id": "v8_B3_pRJ_half", "weights": {"pRJ": 0.5}},
            {"id": "v8_B4_pRJ_x2_lowdyn", "weights": {"pRJ": 2.0, "pRJ_dot": 0.01, "pRJ_ddot": 0.0003}},
        ],
        "C_control_prior": [
            {"id": "v8_C1_control_003", "weights": {"control_pRJ": 0.03, "control_gR2": 0.03}},
            {"id": "v8_C2_no_control_dyn", "weights": {"control_pRJ_dot": 0.0, "control_gR2_dot": 0.0, "control_pRJ_ddot": 0.0, "control_gR2_ddot": 0.0}},
            {"id": "v8_C3_tail_prior", "weights": {"tail_update_prior": 0.001, "control_point_prior": 0.02}},
            {"id": "v8_C4_control_003_tail_prior", "weights": {"control_pRJ": 0.03, "control_gR2": 0.03, "tail_update_prior": 0.001}},
        ],
        "D_recipe_bone": [
            {"id": "v8_D1_v5_stage_a_start", "init": V5_STAGE_A, "weights": {}},
            {"id": "v8_D2_v5_stage_c_start", "init": V5_STAGE_C, "weights": {}},
            {"id": "v8_D3_lr_5e7", "lr": "5e-7", "min_lr": "1e-7", "weights": {}},
            {"id": "v8_D4_bone_005", "weights": {"bone_length": 0.05}},
            {"id": "v8_D5_bone_01", "weights": {"bone_length": 0.1}},
        ],
    }


def write_task_file(shard: str, trials: List[Dict[str, Any]], gpu: str) -> Path:
    tasks: List[Dict[str, Any]] = []
    for trial in trials:
        tasks.extend(build_trial_tasks(trial, gpu=gpu))
    path = Path("configs") / f"newik1_v8_parallel_adaptive_loss_search_{shard}_tasks.json"
    data = {
        "name": f"newik1_v8_parallel_adaptive_loss_search_{shard}",
        "state_file": str(STATE_ROOT / f"newik1_v8_parallel_adaptive_loss_search_{shard}.json"),
        "allow_same_user_gpu_share": False,
        "notes": (
            "Phase-1 shard for NewIK1 v8 parallel adaptive loss search. "
            "All trials preserve IK1 last-control 63D input and 72D output; "
            "module GT and S4 use NewPL-streaming evaluation, not teacher-forced metrics."
        ),
        "tasks": tasks,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def write_combined_task_file(name: str, shard_names: List[str], mapping: Dict[str, str]) -> Path:
    trials_by_shard = phase1_trials()
    tasks: List[Dict[str, Any]] = []
    for shard in shard_names:
        for trial in trials_by_shard[shard]:
            tasks.extend(build_trial_tasks(trial, gpu=mapping.get(shard, "auto")))
    path = Path("configs") / f"newik1_v8_parallel_adaptive_loss_search_{name}_tasks.json"
    data = {
        "name": f"newik1_v8_parallel_adaptive_loss_search_{name}",
        "state_file": str(STATE_ROOT / f"newik1_v8_parallel_adaptive_loss_search_{name}.json"),
        "allow_same_user_gpu_share": False,
        "notes": (
            f"Combined Phase-1 shards {', '.join(shard_names)} for coordinated multi-GPU scheduling. "
            "All trials preserve IK1 last-control 63D input and 72D output; "
            "module GT and S4 use NewPL-streaming evaluation, not teacher-forced metrics."
        ),
        "tasks": tasks,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-map", default="A_gR2:0,B_pRJ_first:1,C_control_prior:0,D_recipe_bone:1")
    args = parser.parse_args()
    mapping = {}
    for item in args.gpu_map.split(","):
        shard, gpu = item.split(":", 1)
        mapping[shard] = gpu
    written = []
    for shard, trials in phase1_trials().items():
        written.append(str(write_task_file(shard, trials, mapping.get(shard, "auto"))))
    written.append(str(write_combined_task_file("local_AB", ["A_gR2", "B_pRJ_first"], mapping)))
    written.append(str(write_combined_task_file("remote_CD", ["C_control_prior", "D_recipe_bone"], mapping)))
    print(json.dumps({"written": written}, indent=2))


if __name__ == "__main__":
    main()
