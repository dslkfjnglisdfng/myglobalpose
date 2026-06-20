#!/usr/bin/env python3
"""Evaluate joint-target NewPL control checkpoints on position, velocity, acceleration, and gravity."""

import argparse
import json
import math
import shlex
import sys
from pathlib import Path

import torch

from l4_train_diverse_short import DEVICE, load_cache_files
from pl_curve import build_pl_curve_model, normalize_gravity
from pl_joint_control_acc_aug102_cache import FEATURE_MODE, resolve_cache_file


LEAF_NAMES = ['left_forearm', 'right_forearm', 'left_lower_leg', 'right_lower_leg', 'head']
DT = 1.0 / 60.0


def load_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is None or manifest.get('type') != 'pl_joint_control_acc_aug102_cache_v1':
        raise ValueError(f'{cache_path} must be pl_joint_control_acc_aug102_cache_v1.')
    records = []
    for cache_file in files:
        data = torch.load(resolve_cache_file(cache_file), map_location='cpu', weights_only=False)
        for idx, name in enumerate(data['name']):
            n = int(data['num_frames'][idx])
            records.append({
                'name': str(name),
                'pl_input': data['pl_input'][idx, :n].float(),
                'pl_target': data['pl_target'][idx, :n].float(),
                'pl_base': data['pl_base'][idx, :n].float(),
                'pl_init_feature': data['pl_init_feature'][idx].float(),
                'pl_target_control': data['pl_target_control'][idx, :n].float(),
            })
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def load_model(checkpoint_path, manifest):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = dict(checkpoint.get('config') or {})
    config['input_size'] = 102
    config['init_size'] = int(manifest.get('init_size', 36))
    model = build_pl_curve_model(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint, config


def finite_difference_first(x, dt=DT):
    out = torch.zeros_like(x)
    if x.shape[0] > 1:
        out[1:] = (x[1:] - x[:-1]) / float(dt)
        out[0] = out[1]
    return out


def finite_difference_second(x, dt=DT):
    out = torch.zeros_like(x)
    if x.shape[0] > 2:
        out[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (float(dt) ** 2)
        out[0] = out[1]
        out[-1] = out[-2]
    return out


def gravity_angle_deg(pred, target):
    pred_g = torch.nn.functional.normalize(pred[..., 15:], dim=-1, eps=1e-8)
    gt_g = torch.nn.functional.normalize(target[..., 15:], dim=-1, eps=1e-8)
    dot = (pred_g * gt_g).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(dot) * (180.0 / math.pi)


def evaluate_record(model, record):
    features = record['pl_input'].to(DEVICE)
    base = normalize_gravity(record['pl_base']).to(DEVICE)
    init = record['pl_init_feature'].to(DEVICE)
    target = normalize_gravity(record['pl_target']).to(DEVICE)
    out = model.forward_sequence(features, base, init_feature=init)
    pred = normalize_gravity(out['pl'])
    pred_pos = pred[:, :15].reshape(-1, 5, 3)
    gt_pos = target[:, :15].reshape(-1, 5, 3)
    pred_vel = out['pldot'][:, :15].reshape(-1, 5, 3)
    pred_acc = out['plddot'][:, :15].reshape(-1, 5, 3)
    gt_vel = finite_difference_first(target[:, :15]).reshape(-1, 5, 3)
    gt_acc = finite_difference_second(target[:, :15]).reshape(-1, 5, 3)
    pos_l2 = (pred_pos - gt_pos).norm(dim=-1)
    vel_l2 = (pred_vel - gt_vel).norm(dim=-1)
    acc_l2 = (pred_acc - gt_acc).norm(dim=-1)
    g_angle = gravity_angle_deg(pred, target)
    return {
        'name': record['name'],
        'num_frames': int(pred.shape[0]),
        'joint_pos_l2_m': float(pos_l2.mean().detach().cpu()),
        'joint_vel_l2_mps': float(vel_l2.mean().detach().cpu()),
        'joint_acc_l2_mps2': float(acc_l2.mean().detach().cpu()),
        'gravity_angle_deg': float(g_angle.mean().detach().cpu()),
        'frame_leaf_mean': {
            'joint_pos_l2_m': float(pos_l2.mean().detach().cpu()),
            'joint_vel_l2_mps': float(vel_l2.mean().detach().cpu()),
            'joint_acc_l2_mps2': float(acc_l2.mean().detach().cpu()),
        },
        'frame_mean_gravity_angle_deg': float(g_angle.mean().detach().cpu()),
        'per_leaf': {
            leaf: {
                'joint_pos_l2_m': float(pos_l2[:, idx].mean().detach().cpu()),
                'joint_vel_l2_mps': float(vel_l2[:, idx].mean().detach().cpu()),
                'joint_acc_l2_mps2': float(acc_l2[:, idx].mean().detach().cpu()),
            }
            for idx, leaf in enumerate(LEAF_NAMES)
        },
    }


def average(rows, key):
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def aggregate(rows, split):
    summary = {
        'split': split,
        'joint_pos_l2_m': average(rows, 'joint_pos_l2_m'),
        'joint_vel_l2_mps': average(rows, 'joint_vel_l2_mps'),
        'joint_acc_l2_mps2': average(rows, 'joint_acc_l2_mps2'),
        'gravity_angle_deg': average(rows, 'gravity_angle_deg'),
        'sequence_mean': {
            'joint_pos_l2_m': average(rows, 'joint_pos_l2_m'),
            'joint_vel_l2_mps': average(rows, 'joint_vel_l2_mps'),
            'joint_acc_l2_mps2': average(rows, 'joint_acc_l2_mps2'),
            'gravity_angle_deg': average(rows, 'gravity_angle_deg'),
        },
        'frame_leaf_mean': {
            metric: sum(row['frame_leaf_mean'][metric] * row['num_frames'] for row in rows) / max(1, sum(row['num_frames'] for row in rows))
            for metric in ('joint_pos_l2_m', 'joint_vel_l2_mps', 'joint_acc_l2_mps2')
        },
        'frame_mean_gravity_angle_deg': sum(row['frame_mean_gravity_angle_deg'] * row['num_frames'] for row in rows) / max(1, sum(row['num_frames'] for row in rows)),
        'sequence_mean_gravity_angle_deg': average(rows, 'gravity_angle_deg'),
        'per_leaf': {},
        'num_sequences': len(rows),
        'num_frames': sum(row['num_frames'] for row in rows),
    }
    for leaf in LEAF_NAMES:
        summary['per_leaf'][leaf] = {
            metric: sum(row['per_leaf'][leaf][metric] for row in rows) / max(1, len(rows))
            for metric in ('joint_pos_l2_m', 'joint_vel_l2_mps', 'joint_acc_l2_mps2')
        }
    return summary


def write_summary(path, payload):
    s = payload['summary']
    lines = [
        '# PL Joint Control Acc-Aug102 v1 Eval',
        '',
        '| split | joint_pos_l2_m | joint_vel_l2_mps | joint_acc_l2_mps2 | gravity_angle_deg |',
        '|---|---:|---:|---:|---:|',
        f"| {s['split']} | {s['joint_pos_l2_m']:.6f} | {s['joint_vel_l2_mps']:.6f} | {s['joint_acc_l2_mps2']:.6f} | {s['gravity_angle_deg']:.6f} |",
        '',
        '## Per Leaf',
        '',
        '| leaf | joint_pos_l2_m | joint_vel_l2_mps | joint_acc_l2_mps2 |',
        '|---|---:|---:|---:|',
    ]
    for leaf in LEAF_NAMES:
        row = s['per_leaf'][leaf]
        lines.append(f"| {leaf} | {row['joint_pos_l2_m']:.6f} | {row['joint_vel_l2_mps']:.6f} | {row['joint_acc_l2_mps2']:.6f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-summary', type=Path, default=None)
    parser.add_argument('--split', default='smoke')
    parser.add_argument('--max-sequences', type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    records, manifest = load_records(args.cache, max_sequences=args.max_sequences)
    if manifest.get('feature_mode') != FEATURE_MODE:
        raise RuntimeError(f'Expected feature_mode={FEATURE_MODE}, got {manifest.get("feature_mode")}.')
    model, checkpoint, config = load_model(args.checkpoint, manifest)
    rows = [evaluate_record(model, record) for record in records]
    payload = {
        'status': 'ok',
        'cache': str(args.cache),
        'checkpoint': str(args.checkpoint),
        'split': args.split,
        'feature_mode': manifest.get('feature_mode'),
        'target_mode': manifest.get('target_mode'),
        'target_contract': manifest.get('target_contract'),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_validation_loss': checkpoint.get('validation_loss'),
        'model_config': config,
        'summary': aggregate(rows, args.split),
        'rows': rows,
        'command': shlex.join(sys.argv),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str) + '\n')
    if args.output_summary:
        write_summary(args.output_summary, payload)
    print(json.dumps({'status': 'ok', 'summary': payload['summary']}, indent=2))


if __name__ == '__main__':
    main()
