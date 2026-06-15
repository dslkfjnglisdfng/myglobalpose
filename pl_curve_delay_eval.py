import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files
from pl_curve import build_pl_curve_model, normalize_gravity


def load_pl_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            record = {
                'name': name,
                'pl_input': data['pl_input'][seq_idx].float(),
                'pl_target': normalize_gravity(data['pl_target'][seq_idx].float()),
                'pl_base': normalize_gravity(data['pl_base'][seq_idx].float()),
            }
            if 'pl_init_feature' in data:
                record['pl_init_feature'] = data['pl_init_feature'][seq_idx].float()
            records.append(record)
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def parse_version_spec(spec):
    name, value = spec.split('=', 1)
    delay = 0
    if ',delay=' in value:
        value, delay_text = value.rsplit(',delay=', 1)
        delay = int(delay_text)
        if delay < 0:
            raise ValueError(f'PL output delay must be non-negative for {name}, got {delay}.')
    return name, value, delay


def load_version(spec, cache):
    name, value, delay = parse_version_spec(spec)
    if value == 'official':
        return {'name': name, 'kind': 'official', 'path': None, 'model': None, 'config': None, 'pl_output_delay': delay}
    if value not in cache:
        checkpoint = torch.load(value, map_location=DEVICE)
        model = build_pl_curve_model(checkpoint.get('config', {})).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        cache[value] = {
            'kind': checkpoint.get('model_type', 'pl_curve_v1'),
            'path': value,
            'model': model,
            'config': checkpoint.get('config', {}),
        }
    version = dict(cache[value])
    version.update({'name': name, 'pl_output_delay': delay})
    return version


def angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target, avoid_nan=True)
    cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def finite_diff(x):
    if x.shape[0] < 2:
        return x.new_zeros((0,) + x.shape[1:])
    return x[1:] - x[:-1]


def jitter(x):
    if x.shape[0] < 3:
        return x.new_zeros(())
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]).norm(dim=-1).mean()


def align_future_output(pred, target, delay):
    if delay <= 0:
        return pred, target
    if pred.shape[0] <= delay:
        raise ValueError(f'Cannot apply delay={delay} to sequence with {pred.shape[0]} frames.')
    return pred[delay:], target[:-delay]


def metrics_for_prediction(pred, target):
    pred = pred.to(target.device, target.dtype)
    p_err = pred[..., :15] - target[..., :15]
    p_leaf = p_err.reshape(p_err.shape[:-1] + (5, 3)).norm(dim=-1) * 100.0
    return {
        'pRB_L1_cm': float(p_err.abs().mean() * 100.0),
        'pRB_L2_cm': float(p_leaf.mean()),
        'per_leaf_pRB_L2_cm': [float(v) for v in p_leaf.mean(dim=0)],
        'pRB_temporal_velocity_error_cm_per_frame': float((finite_diff(pred[..., :15]) - finite_diff(target[..., :15])).reshape(-1, 5, 3).norm(dim=-1).mean() * 100.0) if pred.shape[0] > 1 else 0.0,
        'pRB_smooth_jitter_cm': float(jitter(pred[..., :15].reshape(pred.shape[0], 5, 3)).mean() * 100.0) if pred.shape[0] > 2 else 0.0,
        'gR1_angle_deg': float(angle_deg(pred[..., 15:18], target[..., 15:18]).mean()),
        'gR1_temporal_angle_velocity_error_deg_per_frame': float(angle_deg(finite_diff(pred[..., 15:18]), finite_diff(target[..., 15:18])).mean()) if pred.shape[0] > 1 else 0.0,
        'gR1_smooth_jitter': float(jitter(art.math.normalize_tensor(pred[..., 15:18], avoid_nan=True))) if pred.shape[0] > 2 else 0.0,
        'root_vel_status': 'not applicable',
        'root_vel_L1': None,
        'root_vel_L2': None,
        'root_vel_direction_angle_deg': None,
        'root_vel_smooth_jitter': None,
    }


def average_metrics(rows):
    out = {}
    numeric_keys = [k for k, v in rows[0]['metrics'].items() if isinstance(v, (int, float))]
    for key in numeric_keys:
        out[key] = sum(float(row['metrics'][key]) for row in rows) / len(rows)
    if 'per_leaf_pRB_L2_cm' in rows[0]['metrics']:
        leaf = torch.tensor([row['metrics']['per_leaf_pRB_L2_cm'] for row in rows])
        out['per_leaf_pRB_L2_cm'] = [float(v) for v in leaf.mean(dim=0)]
    out['root_vel_status'] = 'not applicable'
    out['root_vel_L1'] = None
    out['root_vel_L2'] = None
    out['root_vel_direction_angle_deg'] = None
    out['root_vel_smooth_jitter'] = None
    return out


def table_value(value):
    if value is None:
        return 'not available'
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


@torch.no_grad()
def evaluate_versions(records, versions):
    rows_by_version = {version['name']: [] for version in versions}
    for record in records:
        features = record['pl_input'].to(DEVICE)
        target = record['pl_target'].to(DEVICE)
        base = record['pl_base'].to(DEVICE)
        init_feature = record.get('pl_init_feature')
        if init_feature is not None:
            init_feature = init_feature.to(DEVICE)
        raw_cache = {}
        for version in versions:
            raw_key = version['path'] if version['kind'] != 'official' else 'official'
            if raw_key not in raw_cache:
                if version['kind'] == 'official':
                    raw_cache[raw_key] = base
                else:
                    raw_cache[raw_key] = version['model'].forward_sequence(
                        features,
                        base,
                        init_feature=init_feature,
                    )['pl']
            pred = raw_cache[raw_key]
            delay = int(version.get('pl_output_delay', 0))
            pred_aligned, target_aligned = align_future_output(pred, target, delay)
            rows_by_version[version['name']].append({
                'name': record['name'],
                'metrics': {
                    **metrics_for_prediction(pred_aligned.detach().cpu(), target_aligned.detach().cpu()),
                    'pl_output_delay_frames': delay,
                    'evaluated_frames': int(pred_aligned.shape[0]),
                    'source_frames': int(pred.shape[0]),
                },
            })
    out = []
    for version in versions:
        version = dict(version)
        rows = rows_by_version[version['name']]
        version.update({'rows': rows, 'aggregate': average_metrics(rows)})
        out.append(version)
    return out


def make_tables(versions, dataset_label):
    pl_output_table = []
    per_leaf_table = []
    temporal_table = []
    for version in versions:
        agg = version['aggregate']
        note = f"{version['kind']}; delay={version.get('pl_output_delay', 0)}"
        pl_output_table.append({
            'Dataset': dataset_label,
            'Version': version['name'],
            'pRB L1 cm ↓': table_value(agg.get('pRB_L1_cm')),
            'pRB L2 cm ↓': table_value(agg.get('pRB_L2_cm')),
            'gR1 angle deg ↓': table_value(agg.get('gR1_angle_deg')),
            'Notes': note,
        })
        leaves = agg['per_leaf_pRB_L2_cm']
        per_leaf_table.append({
            'Dataset': dataset_label,
            'Version': version['name'],
            'leaf_1 cm ↓': table_value(leaves[0]),
            'leaf_2 cm ↓': table_value(leaves[1]),
            'leaf_3 cm ↓': table_value(leaves[2]),
            'leaf_4 cm ↓': table_value(leaves[3]),
            'leaf_5 cm ↓': table_value(leaves[4]),
            'Mean': table_value(sum(leaves) / len(leaves)),
        })
        temporal_table.append({
            'Dataset': dataset_label,
            'Version': version['name'],
            'delay': int(agg.get('pl_output_delay_frames', 0)),
            'frames': int(agg.get('evaluated_frames', 0)),
            'pRB temporal velocity error cm/frame': table_value(agg.get('pRB_temporal_velocity_error_cm_per_frame')),
            'pRB smooth jitter cm': table_value(agg.get('pRB_smooth_jitter_cm')),
            'gR1 temporal angle velocity error deg/frame': table_value(agg.get('gR1_temporal_angle_velocity_error_deg_per_frame')),
            'gR1 smooth jitter': table_value(agg.get('gR1_smooth_jitter')),
        })
    return pl_output_table, per_leaf_table, temporal_table


def main():
    parser = argparse.ArgumentParser(description='Evaluate PL curve output delay on precomputed PL curve caches.')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--dataset-label', required=True)
    parser.add_argument('--version', action='append', required=True, help='NAME=official or NAME=/path/checkpoint.pt,delay=N')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--delay-mode', choices=('future_output',), default='future_output')
    args = parser.parse_args()
    records, manifest = load_pl_records(args.cache, max_sequences=args.max_eval_sequences)
    model_cache = {}
    versions = [load_version(spec, model_cache) for spec in args.version]
    evaluated = evaluate_versions(records, versions)
    pl_output_table, per_leaf_table, temporal_table = make_tables(evaluated, args.dataset_label)
    result = {
        'status': 'ok',
        'cache': args.cache,
        'dataset_label': args.dataset_label,
        'delay_mode': args.delay_mode,
        'manifest': manifest,
        'versions': [{k: v for k, v in version.items() if k != 'model'} for version in evaluated],
        'pl_output_comparison_table': pl_output_table,
        'per_leaf_table': per_leaf_table,
        'temporal_table': temporal_table,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'ok', 'output_json': str(output_path), 'versions': [v['name'] for v in evaluated]}, indent=2))


if __name__ == '__main__':
    main()
