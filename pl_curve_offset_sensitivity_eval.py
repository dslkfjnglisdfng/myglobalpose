import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files
from pl_curve import build_pl_curve_model, normalize_gravity


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    config = checkpoint.get('config', {})
    if 'model_variant' not in config and checkpoint.get('model_variant'):
        config = dict(config)
        config['model_variant'] = checkpoint.get('model_variant')
    model = build_pl_curve_model(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def load_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is None or manifest.get('type') != 'pl_curve_cache_v2':
        raise RuntimeError(f'Expected pl_curve_cache_v2 manifest, got {manifest.get("type") if manifest else None}.')
    records = {}
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for idx, name in enumerate(data['name']):
            records[str(name)] = {
                'name': str(name),
                'pl_input': data['pl_input'][idx].float(),
                'pl_base': data['pl_base'][idx].float(),
                'pl_target': data['pl_target'][idx].float(),
                'pl_init_feature': data['pl_init_feature'][idx].float(),
            }
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def summarize(values):
    if not values:
        return {'mean': 0.0, 'median': 0.0, 'max': 0.0}
    tensor = torch.as_tensor(values).float()
    return {
        'mean': float(tensor.mean()),
        'median': float(tensor.median()),
        'max': float(tensor.max()),
    }


def gravity_angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target, avoid_nan=True)
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


@torch.no_grad()
def infer(model, record):
    out = model.forward_sequence(
        record['pl_input'].to(DEVICE),
        record['pl_base'].to(DEVICE),
        init_feature=record['pl_init_feature'].to(DEVICE),
    )['pl']
    return normalize_gravity(out).detach().cpu()


def metric_row(output, target, base):
    target = normalize_gravity(target)
    base = normalize_gravity(base)
    output = normalize_gravity(output)
    leaf_err = (output[..., :15].reshape(-1, 5, 3) - target[..., :15].reshape(-1, 5, 3)).norm(dim=-1) * 100.0
    base_leaf_err = (base[..., :15].reshape(-1, 5, 3) - target[..., :15].reshape(-1, 5, 3)).norm(dim=-1) * 100.0
    grav_err = gravity_angle_deg(output[..., 15:], target[..., 15:])
    base_grav_err = gravity_angle_deg(base[..., 15:], target[..., 15:])
    return {
        'pRB_new_cm': float(leaf_err.mean()),
        'pRB_base_cm': float(base_leaf_err.mean()),
        'pRB_delta_new_minus_base_cm': float(leaf_err.mean() - base_leaf_err.mean()),
        'gR1_new_deg': float(grav_err.mean()),
        'gR1_base_deg': float(base_grav_err.mean()),
        'gR1_delta_new_minus_base_deg': float(grav_err.mean() - base_grav_err.mean()),
    }


def parse_cache_arg(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('cache entries must be NAME=PATH')
    name, path = value.split('=', 1)
    if not name:
        raise argparse.ArgumentTypeError('cache name is empty')
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(description='Diagnose whether PLCurve output changes under different offset init caches.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', action='append', type=parse_cache_arg, required=True, help='NAME=pl_curve_cache_manifest.json')
    parser.add_argument('--reference', default='', help='Reference cache name for pairwise output differences. Defaults to first cache.')
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--max-sequences', type=int, default=0)
    args = parser.parse_args()

    model, config = load_model(args.checkpoint)
    cache_records = {}
    manifests = {}
    for name, path in args.cache:
        records, manifest = load_records(path, max_sequences=args.max_sequences)
        cache_records[name] = records
        manifests[name] = manifest
    names = [name for name, _ in args.cache]
    reference = args.reference or names[0]
    if reference not in cache_records:
        raise RuntimeError(f'Reference cache {reference} was not provided.')
    common_sequences = sorted(set.intersection(*[set(records) for records in cache_records.values()]))
    if args.max_sequences:
        common_sequences = common_sequences[:args.max_sequences]
    if not common_sequences:
        raise RuntimeError('No common sequences across caches.')

    outputs = {name: {} for name in names}
    rows = []
    for seq_name in common_sequences:
        target = cache_records[reference][seq_name]['pl_target']
        base = cache_records[reference][seq_name]['pl_base']
        row = {'name': seq_name, 'methods': {}}
        for name in names:
            record = cache_records[name][seq_name]
            output = infer(model, record)
            outputs[name][seq_name] = output
            row['methods'][name] = metric_row(output, target, base)
        rows.append(row)

    pairwise = {}
    for name in names:
        if name == reference:
            continue
        abs_cm = []
        grav_deg = []
        init_offset_m = []
        ref_init_offset_m = []
        for seq_name in common_sequences:
            ref = outputs[reference][seq_name]
            cur = outputs[name][seq_name]
            abs_cm.append(float((cur[..., :15] - ref[..., :15]).abs().mean() * 100.0))
            grav_deg.append(float(gravity_angle_deg(cur[..., 15:], ref[..., 15:]).mean()))
            cur_init = cache_records[name][seq_name]['pl_init_feature'][:18].reshape(6, 3)
            ref_init = cache_records[reference][seq_name]['pl_init_feature'][:18].reshape(6, 3)
            init_offset_m.append(float((cur_init - ref_init).norm(dim=-1).mean()))
            ref_init_offset_m.append(float(ref_init.norm(dim=-1).mean()))
        pairwise[f'{name}_minus_{reference}'] = {
            'pl_output_abs_diff_cm': summarize(abs_cm),
            'pl_gravity_angle_diff_deg': summarize(grav_deg),
            'init_offset_diff_m': summarize(init_offset_m),
            'reference_offset_norm_m': summarize(ref_init_offset_m),
        }

    result = {
        'status': 'ok',
        'checkpoint': str(args.checkpoint),
        'checkpoint_config': config,
        'reference': reference,
        'caches': {name: str(path) for name, path in args.cache},
        'manifests': manifests,
        'num_common_sequences': len(common_sequences),
        'common_sequences': common_sequences,
        'pairwise_vs_reference': pairwise,
        'rows': rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': 'ok',
        'output_json': str(args.output_json),
        'num_common_sequences': len(common_sequences),
        'pairwise_vs_reference': pairwise,
    }, indent=2))


if __name__ == '__main__':
    main()
