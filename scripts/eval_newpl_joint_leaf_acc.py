#!/usr/bin/env python3
"""Evaluate NewPL checkpoints on joint-leaf PL caches."""

import argparse
import json
import math
import shlex
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l4_train_diverse_short import DEVICE, load_cache_files
from pl_curve import build_pl_curve_model, normalize_gravity


LEAF_NAMES = ['left_forearm', 'right_forearm', 'left_lower_leg', 'right_lower_leg', 'head']


def resolve_cache_file(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is None or manifest.get('type') != 'pl_curve_joint_leaf_acc_cache_v1':
        raise ValueError(f'{cache_path} must be pl_curve_joint_leaf_acc_cache_v1.')
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


def load_checkpoint(path, input_size, init_size, args):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = dict(checkpoint.get('config') or {})
    config.update({
        'input_size': int(config.get('input_size') or input_size),
        'init_size': int(config.get('init_size') or init_size),
        'hidden_size': int(config.get('hidden_size', args.hidden_size)),
        'tail_length': int(config.get('tail_length', args.tail_length)),
        'residual_scale': float(config.get('residual_scale', args.residual_scale)),
        'dropout': float(config.get('dropout', args.dropout)),
        'model_variant': config.get('model_variant', args.model_variant),
    })
    model = build_pl_curve_model(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint, config


def gravity_angle_deg(pred, target):
    pred_g = torch.nn.functional.normalize(pred[..., 15:], dim=-1, eps=1e-8)
    tgt_g = torch.nn.functional.normalize(target[..., 15:], dim=-1, eps=1e-8)
    dot = (pred_g * tgt_g).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(dot) * (180.0 / math.pi)


def metric_row(name, pred, target, base, pred_control, target_control):
    diff_leaf = (pred[..., :15] - target[..., :15]).reshape(pred.shape[0], 5, 3)
    base_diff_leaf = (base[..., :15] - target[..., :15]).reshape(base.shape[0], 5, 3)
    control_diff_leaf = (pred_control[..., :15] - target_control[..., :15]).reshape(pred_control.shape[0], 5, 3)
    leaf_l2 = diff_leaf.norm(dim=-1)
    base_leaf_l2 = base_diff_leaf.norm(dim=-1)
    control_leaf_l2 = control_diff_leaf.norm(dim=-1)
    row = {
        'name': name,
        'num_frames': int(pred.shape[0]),
        'p_leaf_joint_R_l1_cm': float(diff_leaf.abs().mean().item() * 100.0),
        'p_leaf_joint_R_l2_cm': float(leaf_l2.mean().item() * 100.0),
        'base_p_leaf_joint_R_l2_cm': float(base_leaf_l2.mean().item() * 100.0),
        'control_p_leaf_joint_R_l1_cm': float(control_diff_leaf.abs().mean().item() * 100.0),
        'control_p_leaf_joint_R_l2_cm': float(control_leaf_l2.mean().item() * 100.0),
        'gR1_angle_deg': float(gravity_angle_deg(pred, target).mean().item()),
        'base_gR1_angle_deg': float(gravity_angle_deg(base, target).mean().item()),
        'control_gR1_angle_deg': float(gravity_angle_deg(pred_control, target_control).mean().item()),
        'per_leaf_l2_cm': {
            leaf: float(leaf_l2[:, idx].mean().item() * 100.0)
            for idx, leaf in enumerate(LEAF_NAMES)
        },
        'base_per_leaf_l2_cm': {
            leaf: float(base_leaf_l2[:, idx].mean().item() * 100.0)
            for idx, leaf in enumerate(LEAF_NAMES)
        },
    }
    return row


def average_rows(rows):
    keys = [
        'p_leaf_joint_R_l1_cm',
        'p_leaf_joint_R_l2_cm',
        'base_p_leaf_joint_R_l2_cm',
        'control_p_leaf_joint_R_l1_cm',
        'control_p_leaf_joint_R_l2_cm',
        'gR1_angle_deg',
        'base_gR1_angle_deg',
        'control_gR1_angle_deg',
    ]
    summary = {key: sum(row[key] for row in rows) / max(1, len(rows)) for key in keys}
    summary['per_leaf_l2_cm'] = {
        leaf: sum(row['per_leaf_l2_cm'][leaf] for row in rows) / max(1, len(rows))
        for leaf in LEAF_NAMES
    }
    summary['base_per_leaf_l2_cm'] = {
        leaf: sum(row['base_per_leaf_l2_cm'][leaf] for row in rows) / max(1, len(rows))
        for leaf in LEAF_NAMES
    }
    summary['num_sequences'] = len(rows)
    summary['num_frames'] = sum(row['num_frames'] for row in rows)
    return summary


def protocol_check_from_manifest(manifest):
    pl_base = manifest.get('pl_base_source') or manifest.get('base_mode') or 'unknown'
    pl_target = manifest.get('pl_target_source') or manifest.get('target_mode') or 'unknown'
    comparable = pl_base == 'official_pl_s1'
    return {
        'pl_base': 'official_pl_s1' if comparable else str(pl_base),
        'pl_target': str(pl_target),
        'comparable_to_v5': bool(comparable),
    }


@torch.no_grad()
def evaluate(args):
    records, manifest = load_records(args.cache, max_sequences=args.max_sequences)
    model, checkpoint, model_config = load_checkpoint(
        args.checkpoint,
        input_size=int(manifest['feature_dim']),
        init_size=int(manifest.get('init_size', 36)),
        args=args,
    )
    rows = []
    for record in records:
        features = record['pl_input'].to(DEVICE)
        base = normalize_gravity(record['pl_base']).to(DEVICE)
        target = normalize_gravity(record['pl_target']).to(DEVICE)
        init_feature = record['pl_init_feature'].to(DEVICE)
        target_control = normalize_gravity(record['pl_target_control']).to(DEVICE)
        out = model.forward_sequence(features, base, init_feature=init_feature)
        pred = normalize_gravity(out['pl']).detach().cpu()
        pred_control = normalize_gravity(out['new_control']).detach().cpu()
        rows.append(metric_row(
            record['name'],
            pred,
            target.cpu(),
            base.cpu(),
            pred_control,
            target_control.cpu(),
        ))
    summary = average_rows(rows)
    payload = {
        'status': 'ok',
        'cache': str(args.cache),
        'checkpoint': str(args.checkpoint),
        'feature_mode': manifest.get('feature_mode'),
        'manifest_type': manifest.get('type'),
        'pl_base_source': manifest.get('pl_base_source', manifest.get('base_mode', 'unknown')),
        'pl_target_source': manifest.get('pl_target_source', manifest.get('target_mode', 'unknown')),
        'evaluation_protocol_version': manifest.get(
            'evaluation_protocol_version',
            'newpl_joint_leaf_acc_unknown_base_v1',
        ),
        'protocol_check': protocol_check_from_manifest(manifest),
        'model_config': model_config,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_validation_loss': checkpoint.get('validation_loss'),
        'contract': 'module-level joint-leaf PL only; no IK/full-pipeline evaluation',
        'legacy_loss_key_pRB_means': manifest.get('flags', {}).get('legacy_loss_key_pRB_means', 'p_leaf_joint_R'),
        'summary': summary,
        'rows': rows,
        'command': shlex.join(sys.argv),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str) + '\n')
    if args.output_summary:
        write_summary(args.output_summary, payload)
    return payload


def write_summary(path, payload):
    s = payload['summary']
    lines = [
        '# Joint-Leaf NewPL Module Eval',
        '',
        f"- Cache: `{payload['cache']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Feature mode: `{payload['feature_mode']}`",
        f"- PL base source: `{payload['pl_base_source']}`",
        f"- PL target source: `{payload['pl_target_source']}`",
        f"- Protocol version: `{payload['evaluation_protocol_version']}`",
        f"- Comparable to v5: `{payload['protocol_check']['comparable_to_v5']}`",
        '- Scope: module-level decoded joint-leaf PL only; no IK/full-pipeline.',
        '',
        '| metric | value |',
        '|---|---:|',
        f"| p_leaf_joint_R_l1_cm | {s['p_leaf_joint_R_l1_cm']:.6f} |",
        f"| p_leaf_joint_R_l2_cm | {s['p_leaf_joint_R_l2_cm']:.6f} |",
        f"| base_p_leaf_joint_R_l2_cm | {s['base_p_leaf_joint_R_l2_cm']:.6f} |",
        f"| gR1_angle_deg | {s['gR1_angle_deg']:.6f} |",
        f"| base_gR1_angle_deg | {s['base_gR1_angle_deg']:.6f} |",
        f"| control_p_leaf_joint_R_l2_cm | {s['control_p_leaf_joint_R_l2_cm']:.6f} |",
        f"| control_gR1_angle_deg | {s['control_gR1_angle_deg']:.6f} |",
        '',
        '## Per Leaf L2 cm',
        '',
        '| leaf | pred | base |',
        '|---|---:|---:|',
    ]
    for leaf in LEAF_NAMES:
        lines.append(f"| {leaf} | {s['per_leaf_l2_cm'][leaf]:.6f} | {s['base_per_leaf_l2_cm'][leaf]:.6f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-summary', type=Path, default=None)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--model-variant', default='base')
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.4)
    return parser.parse_args()


def main():
    payload = evaluate(parse_args())
    print(json.dumps({'status': 'ok', 'summary': payload['summary']}, indent=2))


if __name__ == '__main__':
    main()
