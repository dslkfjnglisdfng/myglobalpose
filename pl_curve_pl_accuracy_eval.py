import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files
from pl_curve import PLCurveModule, normalize_gravity


LEAF_NAMES = ('L_LowArm', 'R_LowArm', 'L_LowLeg', 'R_LowLeg', 'Head')


def build_pl_curve(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get('config', {})
    model = PLCurveModule(
        init_size=int(config.get('init_size', 18)),
        hidden_size=int(config.get('hidden_size', 512)),
        tail_update=int(config.get('tail_length', 4)),
        residual_scale=float(config.get('residual_scale', 0.005)),
        dropout=float(config.get('dropout', 0.4)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def summarize(values):
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'p95': 0.0}
    return {
        'mean': float(values.mean()),
        'std': float(values.std(unbiased=False)),
        'median': float(values.median()),
        'p95': float(torch.quantile(values, 0.95)),
    }


def gravity_angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target, avoid_nan=True)
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def leaf_error_cm(pred, target):
    pred = pred[..., :15].reshape(pred.shape[:-1] + (5, 3))
    target = target[..., :15].reshape(target.shape[:-1] + (5, 3))
    return (pred - target).norm(dim=-1) * 100.0


@torch.no_grad()
def evaluate_sequence(model, record):
    pl_input = record['pl_input'].float().to(DEVICE)
    target = normalize_gravity(record['pl_target'].float()).to(DEVICE)
    original = normalize_gravity(record['pl_base'].float()).to(DEVICE)
    init_feature = record.get('pl_init_feature')
    if init_feature is not None:
        init_feature = init_feature.float().to(DEVICE)
    elif model.init_size != 18:
        raise RuntimeError(f'PL init dim {model.init_size} requires pl_init_feature for {record["name"]}.')
    new = model.forward_sequence(pl_input, original, init_output=target[0] if init_feature is None else None, init_feature=init_feature)['pl']
    new = normalize_gravity(new)

    original_gravity = gravity_angle_deg(original[..., 15:], target[..., 15:])
    new_gravity = gravity_angle_deg(new[..., 15:], target[..., 15:])
    original_leaf = leaf_error_cm(original, target)
    new_leaf = leaf_error_cm(new, target)

    return {
        'name': record['name'],
        'num_frames': int(pl_input.shape[0]),
        'shapes': {
            'pl_input': list(pl_input.shape),
            'pl_target': list(target.shape),
            'pl_base': list(original.shape),
            'pl_curve': list(new.shape),
        },
        'finite': bool(
            torch.isfinite(pl_input).all()
            and torch.isfinite(target).all()
            and torch.isfinite(original).all()
            and torch.isfinite(new).all()
        ),
        'original_gravity_angle_deg': original_gravity.cpu(),
        'new_gravity_angle_deg': new_gravity.cpu(),
        'original_leaf_error_cm': original_leaf.cpu(),
        'new_leaf_error_cm': new_leaf.cpu(),
    }


def load_pl_curve_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is None or manifest.get('type') not in ('pl_curve_cache_v1', 'pl_curve_cache_v2'):
        raise RuntimeError(f'Expected pl_curve_cache_v1/v2 manifest, got {manifest.get("type") if manifest else None}.')
    has_init = manifest.get('type') == 'pl_curve_cache_v2'
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            record = {
                'name': name,
                'pl_input': data['pl_input'][seq_idx],
                'pl_target': data['pl_target'][seq_idx],
                'pl_base': data['pl_base'][seq_idx],
            }
            if has_init:
                record['pl_init_feature'] = data['pl_init_feature'][seq_idx]
            records.append(record)
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def aggregate_rows(rows):
    original_gravity = torch.cat([row['original_gravity_angle_deg'] for row in rows])
    new_gravity = torch.cat([row['new_gravity_angle_deg'] for row in rows])
    original_leaf = torch.cat([row['original_leaf_error_cm'] for row in rows], dim=0)
    new_leaf = torch.cat([row['new_leaf_error_cm'] for row in rows], dim=0)
    leaf_by_name = {}
    for leaf_idx, leaf_name in enumerate(LEAF_NAMES):
        leaf_by_name[leaf_name] = {
            'original_cm': summarize(original_leaf[:, leaf_idx]),
            'new_cm': summarize(new_leaf[:, leaf_idx]),
            'delta_new_minus_original_cm': summarize(new_leaf[:, leaf_idx] - original_leaf[:, leaf_idx]),
        }
    return {
        'num_sequences': len(rows),
        'num_frames': int(sum(row['num_frames'] for row in rows)),
        'all_finite': all(row['finite'] for row in rows),
        'gravity_angle_deg': {
            'original': summarize(original_gravity),
            'new': summarize(new_gravity),
            'delta_new_minus_original': summarize(new_gravity - original_gravity),
        },
        'leaf_position_error_cm': {
            'original': summarize(original_leaf),
            'new': summarize(new_leaf),
            'delta_new_minus_original': summarize(new_leaf - original_leaf),
            'by_leaf': leaf_by_name,
        },
    }


def compact_row(row):
    original_gravity = row['original_gravity_angle_deg']
    new_gravity = row['new_gravity_angle_deg']
    original_leaf = row['original_leaf_error_cm']
    new_leaf = row['new_leaf_error_cm']
    return {
        'name': row['name'],
        'num_frames': row['num_frames'],
        'shapes': row['shapes'],
        'finite': row['finite'],
        'gravity_angle_deg': {
            'original': summarize(original_gravity),
            'new': summarize(new_gravity),
            'delta_new_minus_original': summarize(new_gravity - original_gravity),
        },
        'leaf_position_error_cm': {
            'original': summarize(original_leaf),
            'new': summarize(new_leaf),
            'delta_new_minus_original': summarize(new_leaf - original_leaf),
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Compare original PL-s1 and PLCurve_v1 at the PL output level.')
    parser.add_argument('--pl-cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--max-sequences', type=int, default=0)
    args = parser.parse_args()

    result = {
        'checkpoint': str(args.checkpoint),
        'pl_cache': str(args.pl_cache),
        'status': 'started',
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        model, config = build_pl_curve(args.checkpoint)
        records, manifest = load_pl_curve_records(args.pl_cache, max_sequences=args.max_sequences)
        rows = [evaluate_sequence(model, record) for record in records]
        result.update({
            'status': 'ok',
            'checkpoint_config': config,
            'manifest': manifest,
            'rows': [compact_row(row) for row in rows],
            'aggregate': aggregate_rows(rows),
        })
    except Exception as exc:
        result.update({
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
        })
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': result['status'],
        'output_json': str(args.output_json),
        'num_sequences': result.get('aggregate', {}).get('num_sequences'),
        'num_frames': result.get('aggregate', {}).get('num_frames'),
        'all_finite': result.get('aggregate', {}).get('all_finite'),
        'error_type': result.get('error_type'),
        'error': result.get('error'),
    }, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
