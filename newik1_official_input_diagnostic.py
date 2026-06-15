import argparse
import json
import traceback
from pathlib import Path

import torch

from l4_train_diverse_short import DEVICE, load_cache_files
from newik1_control_point import finite_diff, normalize_ik1
from newik1_official_input_train import MODEL_TYPE, build_model, forward_sequence


def load_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is None or manifest.get('type') != 'newik1_official_input_cache_v1':
        raise RuntimeError(f'Expected newik1_official_input_cache_v1 manifest, got {manifest.get("type") if manifest else None}.')
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        missing = [key for key in ('name', 'ik1_input', 'ik1_target', 'ik1_base') if key not in data]
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


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if checkpoint.get('model_type') != MODEL_TYPE:
        raise RuntimeError(f"Unsupported official-input IK1 checkpoint model_type={checkpoint.get('model_type')}")
    config = checkpoint.get('config', {})
    model = build_model(dropout=float(config.get('dropout', 0.4))).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def mean_l1(pred, target):
    return float((pred - target).abs().mean())


def mean_l2(pred, target):
    return float((pred - target).square().mean().sqrt())


def gravity_angle_deg(pred, target):
    pred_g = torch.nn.functional.normalize(pred[..., 69:], dim=-1)
    target_g = torch.nn.functional.normalize(target[..., 69:], dim=-1)
    cos = (pred_g * target_g).sum(dim=-1).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos)).mean())


def temporal_l2(pred, target, order, start, end):
    if pred.shape[0] <= order:
        return 0.0
    return mean_l2(finite_diff(pred[..., start:end], order), finite_diff(target[..., start:end], order))


def metrics_for(pred, target):
    pred = normalize_ik1(pred)
    target = normalize_ik1(target)
    return {
        'pRJ_l1': mean_l1(pred[..., :69], target[..., :69]),
        'pRJ_cm_l1': mean_l1(pred[..., :69], target[..., :69]) * 100.0,
        'pRJ_l2': mean_l2(pred[..., :69], target[..., :69]),
        'pRJ_cm_l2': mean_l2(pred[..., :69], target[..., :69]) * 100.0,
        'gR2_l1': mean_l1(pred[..., 69:], target[..., 69:]),
        'gR2_angle_deg': gravity_angle_deg(pred, target),
        'pRJ_dot_l2': temporal_l2(pred, target, 1, 0, 69),
        'pRJ_dot_cm_l2': temporal_l2(pred, target, 1, 0, 69) * 100.0,
        'pRJ_ddot_l2': temporal_l2(pred, target, 2, 0, 69),
        'pRJ_ddot_cm_l2': temporal_l2(pred, target, 2, 0, 69) * 100.0,
        'gR2_dot_l2': temporal_l2(pred, target, 1, 69, 72),
        'gR2_ddot_l2': temporal_l2(pred, target, 2, 69, 72),
        'state_l1': mean_l1(pred, target),
        'state_l2': mean_l2(pred, target),
    }


def weighted_average(rows, key, prefix):
    total_frames = sum(row['num_frames'] for row in rows)
    if total_frames == 0:
        return 0.0
    return sum(row['num_frames'] * row[prefix][key] for row in rows) / total_frames


def aggregate(rows, prefix):
    if not rows:
        return {}
    keys = rows[0][prefix].keys()
    return {key: weighted_average(rows, key, prefix) for key in keys}


@torch.no_grad()
def evaluate(records, model):
    rows = []
    for record in records:
        feature = record['ik1_input'].to(DEVICE)
        target = record['ik1_target'].to(DEVICE)
        base = record['ik1_base'].to(DEVICE)
        pred = forward_sequence(model, feature)
        baseline_metrics = metrics_for(base, target)
        model_metrics = metrics_for(pred, target)
        delta = {key: model_metrics[key] - baseline_metrics[key] for key in baseline_metrics}
        rows.append({
            'name': record['name'],
            'num_frames': int(target.shape[0]),
            'baseline_official_ik1': baseline_metrics,
            'newik1': model_metrics,
            'delta_newik1_minus_baseline': delta,
            'newik1_better_by_state_l2': bool(delta['state_l2'] < 0),
            'finite': bool(torch.isfinite(pred).all() and torch.isfinite(base).all() and torch.isfinite(target).all()),
        })
    baseline = aggregate(rows, 'baseline_official_ik1')
    newik1 = aggregate(rows, 'newik1')
    delta = {key: newik1[key] - baseline[key] for key in baseline}
    return rows, {
        'baseline_official_ik1': baseline,
        'newik1': newik1,
        'delta_newik1_minus_baseline': delta,
        'newik1_better_by_state_l2': bool(delta['state_l2'] < 0) if delta else False,
        'all_finite': all(row['finite'] for row in rows),
        'num_sequences': len(rows),
        'num_frames': sum(row['num_frames'] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description='Official-input NewIK1 module-output-vs-GT diagnostic.')
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
        'module': 'IK-s1',
        'input_contract': 'RRB_after_pl[45] + gR1[3] + pRB[15] = 63D',
        'output_contract': 'pRJ[69] + gR2[3] = 72D',
        'metric_contract': 'Compare NewIK1 output and cache ik1_base against ik1_target GT on the same official-shape IK1 cache.',
    }
    try:
        records, manifest = load_records(args.cache, max_sequences=args.max_sequences)
        model, config = load_model(args.ik1_checkpoint)
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
