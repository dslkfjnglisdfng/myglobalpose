import argparse
import json
import traceback
from pathlib import Path

import torch

from l4_train_diverse_short import DEVICE, load_cache_files
from ik1_curve import IK1_LEAF_PRJ_INDICES
from newik1_control_eval import build_newik1_control
from newik1_control_point import finite_diff, normalize_ik1


def load_newik1_records(cache_manifest, max_sequences=0):
    files, manifest = load_cache_files(cache_manifest)
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        required = ('name', 'ik1_input', 'ik1_target', 'ik1_base')
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f'{cache_file} missing required fields: {missing}')
        for seq_idx, name in enumerate(data['name']):
            records.append({
                'name': name,
                'ik1_input': data['ik1_input'][seq_idx].float(),
                'ik1_target': normalize_ik1(data['ik1_target'][seq_idx].float()),
                'ik1_base': normalize_ik1(data['ik1_base'][seq_idx].float()),
            })
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def mean_l1(pred, target):
    return float((pred - target).abs().mean())


def mean_l2(pred, target):
    return float((pred - target).square().mean().sqrt())


def gravity_angle_deg(pred, target):
    pred_g = torch.nn.functional.normalize(pred[..., 69:], dim=-1)
    target_g = torch.nn.functional.normalize(target[..., 69:], dim=-1)
    cos = (pred_g * target_g).sum(dim=-1).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos)).mean())


def temporal_l2(pred, target, order):
    if pred.shape[0] <= order:
        return 0.0
    return mean_l2(finite_diff(pred, order), finite_diff(target, order))


def temporal_l2_slice(pred, target, order, start, end):
    if pred.shape[0] <= order:
        return 0.0
    return mean_l2(finite_diff(pred[..., start:end], order), finite_diff(target[..., start:end], order))


def leaf_pRJ(output):
    pRJ = output[..., :69].reshape(output.shape[:-1] + (23, 3))
    return pRJ[..., IK1_LEAF_PRJ_INDICES, :].reshape(output.shape[:-1] + (len(IK1_LEAF_PRJ_INDICES) * 3,))


def metrics_for(pred, target):
    pred = normalize_ik1(pred)
    target = normalize_ik1(target)
    return {
        'pRJ_l1': mean_l1(pred[..., :69], target[..., :69]),
        'pRJ_l2': mean_l2(pred[..., :69], target[..., :69]),
        'pRJ_cm_l1': mean_l1(pred[..., :69], target[..., :69]) * 100.0,
        'pRJ_cm_l2': mean_l2(pred[..., :69], target[..., :69]) * 100.0,
        'leaf_pRJ_l1': mean_l1(leaf_pRJ(pred), leaf_pRJ(target)),
        'leaf_pRJ_l2': mean_l2(leaf_pRJ(pred), leaf_pRJ(target)),
        'leaf_pRJ_cm_l1': mean_l1(leaf_pRJ(pred), leaf_pRJ(target)) * 100.0,
        'leaf_pRJ_cm_l2': mean_l2(leaf_pRJ(pred), leaf_pRJ(target)) * 100.0,
        'gR2_l1': mean_l1(pred[..., 69:], target[..., 69:]),
        'gR2_angle_deg': gravity_angle_deg(pred, target),
        'pRJ_dot_l2': temporal_l2_slice(pred, target, 1, 0, 69),
        'pRJ_ddot_l2': temporal_l2_slice(pred, target, 2, 0, 69),
        'leaf_pRJ_dot_l2': temporal_l2(leaf_pRJ(pred), leaf_pRJ(target), 1),
        'leaf_pRJ_ddot_l2': temporal_l2(leaf_pRJ(pred), leaf_pRJ(target), 2),
        'gR2_dot_l2': temporal_l2_slice(pred, target, 1, 69, 72),
        'gR2_ddot_l2': temporal_l2_slice(pred, target, 2, 69, 72),
        'state_l1': mean_l1(pred, target),
        'state_l2': mean_l2(pred, target),
        'state_dot_l2': temporal_l2(pred, target, 1),
        'state_ddot_l2': temporal_l2(pred, target, 2),
    }


def weighted_average(rows, key):
    total_frames = sum(row['num_frames'] for row in rows)
    if total_frames == 0:
        return 0.0
    return sum(row['num_frames'] * row[key] for row in rows) / total_frames


def aggregate(rows, prefix):
    keys = rows[0][prefix].keys() if rows else []
    return {key: weighted_average([{'num_frames': row['num_frames'], key: row[prefix][key]} for row in rows], key) for key in keys}


@torch.no_grad()
def evaluate(records, model):
    rows = []
    for record in records:
        feature = record['ik1_input'].to(DEVICE)
        base = record['ik1_base'].to(DEVICE)
        target = record['ik1_target'].to(DEVICE)
        model_out = model.forward_sequence(feature, base)['ik1'].squeeze(1)
        baseline_metrics = metrics_for(base, target)
        model_metrics = metrics_for(model_out, target)
        delta = {key: model_metrics[key] - baseline_metrics[key] for key in baseline_metrics}
        rows.append({
            'name': record['name'],
            'num_frames': int(target.shape[0]),
            'baseline_ik1_on_newpl': baseline_metrics,
            'newik1': model_metrics,
            'delta_newik1_minus_baseline': delta,
            'newik1_better_by_state_l2': bool(delta['state_l2'] < 0),
            'finite': bool(torch.isfinite(model_out).all() and torch.isfinite(base).all() and torch.isfinite(target).all()),
        })
    baseline = aggregate(rows, 'baseline_ik1_on_newpl')
    newik1 = aggregate(rows, 'newik1')
    delta = {key: newik1[key] - baseline[key] for key in baseline}
    return rows, {
        'baseline_ik1_on_newpl': baseline,
        'newik1': newik1,
        'delta_newik1_minus_baseline': delta,
        'newik1_better_by_state_l2': bool(delta['state_l2'] < 0),
        'all_finite': all(row['finite'] for row in rows),
        'num_sequences': len(rows),
        'num_frames': sum(row['num_frames'] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare official IK1 baseline and NewIK1 outputs against GT on the same NewPL cache inputs.')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--ik1-checkpoint', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--max-sequences', type=int, default=0)
    args = parser.parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'cache': str(args.cache),
        'ik1_checkpoint': str(args.ik1_checkpoint),
        'status': 'started',
        'contract': 'Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT.',
    }
    try:
        records, manifest = load_newik1_records(args.cache, max_sequences=args.max_sequences)
        model, config = build_newik1_control(args.ik1_checkpoint)
        rows, aggregate_result = evaluate(records, model)
        result.update({
            'status': 'ok',
            'cache_manifest': manifest,
            'ik1_checkpoint_config': config,
            'rows': rows,
            'aggregate': aggregate_result,
        })
    except Exception as exc:
        result.update({
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': result.get('status'),
        'aggregate': result.get('aggregate'),
        'error_type': result.get('error_type'),
        'error': result.get('error'),
    }, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
