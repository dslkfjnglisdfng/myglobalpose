import argparse
import json
from pathlib import Path

import torch

from l4_tail_update_qstate import UniformCubicBSpline
from l4_train_diverse_short import load_cache_files
from pl_curve import normalize_gravity


SUPPORTED_PL_CACHE_TYPES = {
    'pl_curve_cache_v1',
    'pl_curve_cache_v2',
    'pl_curve_cache_v3',
    'pl_curve_cache_v4',
    'pl_next_control_cache_v1',
    'pl_next_control_cache_v2',
}


def _shift_next(x):
    if x.shape[0] == 0:
        return x
    if x.shape[0] == 1:
        return x.clone()
    return torch.cat((x[1:], x[-1:]), dim=0)


def _central_velocity(x, dt):
    out = torch.zeros_like(x)
    if x.shape[0] <= 1:
        return out
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * float(dt))
    out[0] = (x[1] - x[0]) / float(dt)
    out[-1] = (x[-1] - x[-2]) / float(dt)
    return out


def _central_acceleration(x, dt):
    out = torch.zeros_like(x)
    if x.shape[0] <= 2:
        return out
    out[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (float(dt) ** 2)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def _tail_history(control, tail_size=4):
    tail = control.new_zeros(control.shape[0], int(tail_size), control.shape[-1])
    mask = torch.zeros(control.shape[0], int(tail_size), dtype=torch.bool)
    for frame_idx in range(control.shape[0]):
        count = min(int(tail_size), frame_idx + 1)
        tail[frame_idx, -count:] = control[frame_idx - count + 1:frame_idx + 1]
        mask[frame_idx, -count:] = True
    return tail, mask


def _decode_control_derivatives(control, dt):
    spline = UniformCubicBSpline(dt)
    decode_control = torch.cat((control, control[-1:].clone()), dim=0)
    curve, dot, ddot = spline(decode_control, return_derivatives=True)
    return normalize_gravity(curve[:-1]), dot[:-1], ddot[:-1]


def _load_controls(control_cache):
    if not control_cache:
        return {}, None
    files, manifest = load_cache_files(control_cache)
    controls = {}
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        if 'pl_pRB_gR1_control' not in data:
            raise KeyError(f'{cache_file} missing pl_pRB_gR1_control.')
        for name, control in zip(data['name'], data['pl_pRB_gR1_control']):
            controls[str(name)] = control.float()
    return controls, manifest


def _append_optional(shard, data, seq_idx, key, default):
    if key in data and len(data[key]) > seq_idx:
        shard[key].append(data[key][seq_idx])
    else:
        shard[key].append(default)


def build_next_control_cache(pl_cache, gt_control_cache, output_dir, shard_size=100, max_sequences=0, dt=1.0 / 60.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(pl_cache)
    source_type = source_manifest.get('type') if source_manifest else None
    if source_type not in SUPPORTED_PL_CACHE_TYPES:
        raise RuntimeError(f'Expected PL cache manifest type in {sorted(SUPPORTED_PL_CACHE_TYPES)}, got {source_type}.')
    controls, control_manifest = _load_controls(gt_control_cache)

    cache_files = []
    shard_idx = 0
    total_sequences = 0
    total_frames = 0
    optional_meta_keys = ('source_name', 'pair_id', 'view_type')

    def new_shard():
        shard = {
            'name': [],
            'pl_input': [],
            'pl_target': [],
            'pl_target_next': [],
            'valid_next_mask': [],
            'pl_base': [],
            'pl_base_next': [],
            'pl_init_feature': [],
            'pl_target_control': [],
            'pl_target_control_next': [],
            'tail_control_target': [],
            'tail_control_valid_mask': [],
            'last_control_target': [],
            'gt_pldot': [],
            'gt_plddot': [],
            'gt_pldot_next': [],
            'gt_plddot_next': [],
            'baseline_fd_vel': [],
            'baseline_fd_acc': [],
            'num_frames': [],
        }
        for key in optional_meta_keys:
            shard[key] = []
        return shard

    shard = new_shard()

    def flush():
        nonlocal shard, shard_idx
        if not shard['name']:
            return
        out = output_dir / f'pl_next_control_cache_shard{shard_idx:05d}.pt'
        torch.save(shard, out)
        cache_files.append({
            'path': str(out),
            'num_sequences': len(shard['name']),
            'num_frames': int(sum(shard['num_frames'])),
        })
        shard_idx += 1
        shard = new_shard()

    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        has_init = 'pl_init_feature' in data
        required = ('name', 'pl_input', 'pl_target', 'pl_base')
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f'{cache_file} missing required fields: {missing}')
        for seq_idx, name in enumerate(data['name']):
            if max_sequences and total_sequences >= max_sequences:
                break
            key = str(name)
            if key not in controls:
                raise KeyError(f'{gt_control_cache} missing GT PL control for {key}.')
            pl_target = normalize_gravity(data['pl_target'][seq_idx].float())
            pl_base = normalize_gravity(data['pl_base'][seq_idx].float())
            control = controls[key].float()
            if tuple(control.shape) != tuple(pl_target.shape):
                raise RuntimeError(f'{key} control shape {tuple(control.shape)} != target shape {tuple(pl_target.shape)}.')
            _, gt_dot, gt_ddot = _decode_control_derivatives(control, dt=dt)
            valid_next = torch.ones(pl_target.shape[0], dtype=torch.bool)
            if valid_next.numel():
                valid_next[-1] = False
            baseline_vel = _central_velocity(pl_base, dt=dt)
            baseline_acc = _central_acceleration(pl_base, dt=dt)
            tail_control_target, tail_control_valid_mask = _tail_history(control, tail_size=4)

            tensors = (
                data['pl_input'][seq_idx],
                pl_target,
                pl_base,
                control,
                tail_control_target,
                gt_dot,
                gt_ddot,
                baseline_vel,
                baseline_acc,
            )
            if not all(torch.isfinite(t.float()).all() for t in tensors):
                raise RuntimeError(f'Non-finite next-control cache tensor at {key}.')

            shard['name'].append(key)
            shard['pl_input'].append(data['pl_input'][seq_idx].float())
            shard['pl_target'].append(pl_target)
            shard['pl_target_next'].append(_shift_next(pl_target))
            shard['valid_next_mask'].append(valid_next)
            shard['pl_base'].append(pl_base)
            shard['pl_base_next'].append(_shift_next(pl_base))
            if has_init:
                shard['pl_init_feature'].append(data['pl_init_feature'][seq_idx].float())
            else:
                raise KeyError(f'{cache_file} missing pl_init_feature; next-control v6 expects init36 cache.')
            shard['pl_target_control'].append(control)
            shard['pl_target_control_next'].append(_shift_next(control))
            shard['tail_control_target'].append(tail_control_target)
            shard['tail_control_valid_mask'].append(tail_control_valid_mask)
            shard['last_control_target'].append(control)
            shard['gt_pldot'].append(gt_dot)
            shard['gt_plddot'].append(gt_ddot)
            shard['gt_pldot_next'].append(_shift_next(gt_dot))
            shard['gt_plddot_next'].append(_shift_next(gt_ddot))
            shard['baseline_fd_vel'].append(baseline_vel)
            shard['baseline_fd_acc'].append(baseline_acc)
            shard['num_frames'].append(int(pl_target.shape[0]))
            for meta_key in optional_meta_keys:
                default = key if meta_key in ('source_name', 'pair_id') else 'single'
                _append_optional(shard, data, seq_idx, meta_key, default)
            total_sequences += 1
            total_frames += int(pl_target.shape[0])
            if len(shard['name']) >= int(shard_size):
                flush()
        if max_sequences and total_sequences >= max_sequences:
            break
    flush()

    manifest = {
        'type': 'pl_next_control_cache_v2',
        'source_pl_cache': str(pl_cache),
        'source_manifest': source_manifest,
        'gt_control_cache': str(gt_control_cache),
        'gt_control_manifest': control_manifest,
        'dt': float(dt),
        'input_contract': 'PL input remains aRB[18] + wRB[18] + RRB[45] + gR0[3] = 84D unless inherited source cache says otherwise.',
        'init_contract': 'init36: offset_r[18] + pRL[15] + gR0[3].',
        'control_time_semantics': (
            'pl_target_control[t] is the fitted tail control point whose clamped uniform cubic decode gives '
            'the frame-t PL state; pl_target_control_next[t] is control[t+1] and valid only when valid_next_mask[t].'
        ),
        'derivative_contract': 'gt_pldot/gt_plddot are decoded from derivative-aware GT controls using UniformCubicBSpline at dt.',
        'baseline_contract': 'baseline_fd_vel/baseline_fd_acc are finite differences of the cached official PL baseline outputs.',
        'fields': {
            'pl_input': '[T,D] cached PL input feature',
            'pl_init_feature': '[36] init36 feature',
            'pl_target': '[T,18] current GT pRB[15]+gR1[3]',
            'pl_target_next': '[T,18] shifted t+1 GT state; last value repeats and is masked out',
            'valid_next_mask': '[T] false for the final frame',
            'pl_target_control': '[T,18] current fitted GT control',
            'pl_target_control_next': '[T,18] shifted t+1 fitted GT control',
            'tail_control_target': '[T,4,18] fitted GT controls for the last up-to-four controls ending at t; left padded and masked at sequence start',
            'tail_control_valid_mask': '[T,4] valid mask for tail_control_target',
            'last_control_target': '[T,18] fitted GT current control, duplicated for explicit last-preview-control loss',
            'gt_pldot/gt_plddot': '[T,18] spline derivatives from GT controls',
            'gt_pldot_next/gt_plddot_next': '[T,18] shifted t+1 derivative targets',
            'pl_base/pl_base_next': '[T,18] official PL baseline at t and shifted t+1',
            'baseline_fd_vel/baseline_fd_acc': '[T,18] finite-difference baseline PL velocity/acceleration',
        },
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'max_sequences': int(max_sequences),
    }
    manifest_path = output_dir / 'pl_next_control_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build NewPL v6 next-control module cache from PL cache + GTControlCache.')
    parser.add_argument('--pl-cache', required=True)
    parser.add_argument('--gt-control-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--shard-size', type=int, default=100)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--dt', type=float, default=1.0 / 60.0)
    args = parser.parse_args()
    manifest = build_next_control_cache(
        args.pl_cache,
        args.gt_control_cache,
        args.output_dir,
        shard_size=args.shard_size,
        max_sequences=args.max_sequences,
        dt=args.dt,
    )
    print(json.dumps({
        'status': 'ok',
        'manifest': str(Path(args.output_dir) / 'pl_next_control_cache_manifest.json'),
        'num_sequences': manifest['num_sequences'],
        'num_frames': manifest['num_frames'],
    }, indent=2))


if __name__ == '__main__':
    main()
