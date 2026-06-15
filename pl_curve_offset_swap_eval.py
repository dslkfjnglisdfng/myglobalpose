import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files
from pl_curve import PL_OFFSET_AWARE_INPUT_SIZE, build_pl_curve_model, normalize_gravity, replace_offset_aware_feature_offset


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
    if manifest is None or manifest.get('type') not in ('pl_curve_cache_v2', 'pl_curve_cache_v3', 'pl_curve_cache_v4'):
        raise RuntimeError(f'Expected pl_curve_cache_v2/v3/v4 manifest, got {manifest.get("type") if manifest else None}.')
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for idx, name in enumerate(data['name']):
            records.append({
                'name': str(name),
                'pl_input': data['pl_input'][idx].float(),
                'pl_base': data['pl_base'][idx].float(),
                'pl_target': data['pl_target'][idx].float(),
                'pl_init_feature': data['pl_init_feature'][idx].float(),
            })
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def summarize(values):
    values = torch.as_tensor(values).float()
    if values.numel() == 0:
        return {'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'mean': float(values.mean()),
        'median': float(values.median()),
        'min': float(values.min()),
        'max': float(values.max()),
    }


def gravity_angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target, avoid_nan=True)
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def make_variant_init(init_feature, variant, seq_idx, all_inits):
    out = init_feature.clone()
    if variant == 'good':
        return out
    if variant == 'zero':
        out[..., :18] = 0.0
        return out
    offset = out[..., :18].reshape(6, 3)
    if variant == 'roll_sensors':
        out[..., :18] = torch.roll(offset, shifts=1 + seq_idx % 5, dims=0).reshape(18)
        return out
    if variant == 'negate':
        out[..., :18] = -out[..., :18]
        return out
    if variant == 'other_sequence':
        if len(all_inits) <= 1:
            out[..., :18] = torch.roll(offset, shifts=1, dims=0).reshape(18)
            return out
        other = all_inits[(seq_idx + 1) % len(all_inits)]
        out[..., :18] = other[..., :18]
        return out
    raise ValueError(f'Unsupported init variant: {variant}')


@torch.no_grad()
def infer(model, record, init_feature, swap_feature_offset=False):
    pl_input = record['pl_input']
    if swap_feature_offset and pl_input.shape[-1] == PL_OFFSET_AWARE_INPUT_SIZE:
        pl_input = replace_offset_aware_feature_offset(pl_input, init_feature[:18].reshape(6, 3), dt=getattr(model, 'dt', 1.0 / 60.0))
    out = model.forward_sequence(
        pl_input.to(DEVICE),
        record['pl_base'].to(DEVICE),
        init_feature=init_feature.to(DEVICE),
    )['pl']
    return normalize_gravity(out).detach().cpu()


def metrics(output, target, base):
    output = normalize_gravity(output)
    target = normalize_gravity(target)
    base = normalize_gravity(base)
    p = output[..., :15].reshape(-1, 5, 3)
    t = target[..., :15].reshape(-1, 5, 3)
    b = base[..., :15].reshape(-1, 5, 3)
    p_err_cm = (p - t).norm(dim=-1).mean() * 100.0
    b_err_cm = (b - t).norm(dim=-1).mean() * 100.0
    g_err_deg = gravity_angle_deg(output[..., 15:], target[..., 15:]).mean()
    b_g_err_deg = gravity_angle_deg(base[..., 15:], target[..., 15:]).mean()
    p_smooth = torch.nn.functional.smooth_l1_loss(output[..., :15], target[..., :15])
    g_loss = (1.0 - (output[..., 15:] * target[..., 15:]).sum(dim=-1).clamp(-1.0, 1.0)).mean()
    return {
        'pRB_cm': float(p_err_cm),
        'pRB_delta_vs_base_cm': float(p_err_cm - b_err_cm),
        'gR1_deg': float(g_err_deg),
        'gR1_delta_vs_base_deg': float(g_err_deg - b_g_err_deg),
        'pl_gt_loss': float(p_smooth + g_loss),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate whether good/bad offset init is separable by PL GT loss.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pl-cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--variants', default='good,zero,roll_sensors,other_sequence,negate')
    parser.add_argument(
        '--swap-feature-offset',
        action='store_true',
        help='For 156D offset_aware PL caches, recompute offset-dependent feature blocks with the variant r_JS.',
    )
    args = parser.parse_args()

    model, config = load_model(args.checkpoint)
    records, manifest = load_records(args.pl_cache, max_sequences=args.max_sequences)
    variants = [item.strip() for item in args.variants.split(',') if item.strip()]
    all_inits = [record['pl_init_feature'] for record in records]
    rows = []
    for seq_idx, record in enumerate(records):
        row = {'name': record['name'], 'variants': {}}
        for variant in variants:
            init = make_variant_init(record['pl_init_feature'], variant, seq_idx, all_inits)
            output = infer(model, record, init, swap_feature_offset=args.swap_feature_offset)
            row['variants'][variant] = metrics(output, record['pl_target'], record['pl_base'])
            row['variants'][variant]['offset_norm_m'] = float(init[:18].reshape(6, 3).norm(dim=-1).mean())
        good = row['variants'].get('good')
        if good:
            row['delta_vs_good'] = {}
            for variant in variants:
                if variant == 'good':
                    continue
                cur = row['variants'][variant]
                row['delta_vs_good'][variant] = {
                    'pRB_cm': cur['pRB_cm'] - good['pRB_cm'],
                    'gR1_deg': cur['gR1_deg'] - good['gR1_deg'],
                    'pl_gt_loss': cur['pl_gt_loss'] - good['pl_gt_loss'],
                }
        rows.append(row)

    aggregate = {'variants': {}, 'delta_vs_good': {}}
    for variant in variants:
        aggregate['variants'][variant] = {
            key: summarize([row['variants'][variant][key] for row in rows])
            for key in ('pRB_cm', 'pRB_delta_vs_base_cm', 'gR1_deg', 'gR1_delta_vs_base_deg', 'pl_gt_loss', 'offset_norm_m')
        }
        if variant != 'good':
            aggregate['delta_vs_good'][variant] = {
                key: summarize([row['delta_vs_good'][variant][key] for row in rows])
                for key in ('pRB_cm', 'gR1_deg', 'pl_gt_loss')
            }

    result = {
        'status': 'ok',
        'checkpoint': str(args.checkpoint),
        'checkpoint_config': config,
        'pl_cache': str(args.pl_cache),
        'manifest': manifest,
        'variants': variants,
        'swap_feature_offset': bool(args.swap_feature_offset),
        'num_sequences': len(records),
        'aggregate': aggregate,
        'rows': rows,
        'interpretation': {
            'positive_delta_vs_good': 'bad/alternate offset is worse than good offset',
            'real_offset_gt': 'not available; this is a downstream separability diagnostic only',
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': 'ok',
        'output_json': str(args.output_json),
        'num_sequences': len(records),
        'delta_vs_good': aggregate['delta_vs_good'],
    }, indent=2))


if __name__ == '__main__':
    main()
