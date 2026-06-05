import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files
from net import GPNet
from pl_curve import normalize_gravity, pl_init_feature_from_pose, pl_input_feature, pl_target_from_pose


def select_imu_fields(data, seq_idx, mode):
    if mode == 'official':
        return data['aM'][seq_idx].float(), data['wM'][seq_idx].float(), data['RMB'][seq_idx].float(), {
            'mode': mode,
            'a_field': 'aM',
            'w_field': 'wM',
            'R_field': 'RMB',
            'source': 'official_fields',
        }
    has_l4 = all(key in data for key in ('l4_aM', 'l4_wM', 'l4_RMB'))
    if mode == 'processed':
        if not has_l4:
            raise KeyError('processed mode requires l4_aM/l4_wM/l4_RMB fields.')
        return data['l4_aM'][seq_idx].float(), data['l4_wM'][seq_idx].float(), data['l4_RMB'][seq_idx].float(), {
            'mode': mode,
            'a_field': 'l4_aM',
            'w_field': 'l4_wM',
            'R_field': 'l4_RMB',
            'source': 'processed_l4_fields',
        }
    if mode == 'auto':
        if has_l4:
            return data['l4_aM'][seq_idx].float(), data['l4_wM'][seq_idx].float(), data['l4_RMB'][seq_idx].float(), {
                'mode': mode,
                'a_field': 'l4_aM',
                'w_field': 'l4_wM',
                'R_field': 'l4_RMB',
                'source': 'processed_l4_fields',
            }
        return data['aM'][seq_idx].float(), data['wM'][seq_idx].float(), data['RMB'][seq_idx].float(), {
            'mode': mode,
            'a_field': 'aM',
            'w_field': 'wM',
            'R_field': 'RMB',
            'source': 'fallback_or_generated_primary_fields',
        }
    raise ValueError(f'Unsupported imu input mode: {mode}')


def sequence_pl_inputs(record):
    return torch.stack([
        pl_input_feature(record['aM'][i], record['wM'][i], record['RMB'][i])
        for i in range(record['aM'].shape[0])
    ]).float()


@torch.no_grad()
def sequence_base_pl(gpnet, pl_input, init_output):
    return gpnet.plnet([(pl_input.to(DEVICE), init_output.to(DEVICE))])[0].detach().cpu()


def source_records(cache_path):
    files, manifest = load_cache_files(cache_path)
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            yield cache_file, seq_idx, {
                'name': name,
                'pose_gt': data['pose_gt'][seq_idx].float(),
                'aM': data['aM'][seq_idx].float(),
                'wM': data['wM'][seq_idx].float(),
                'RMB': data['RMB'][seq_idx].float(),
            }
    return manifest


def build_cache(input_cache, output_dir, shard_size, imu_input_mode):
    output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(input_cache)
    gpnet = GPNet().eval().to(DEVICE)
    for parameter in gpnet.parameters():
        parameter.requires_grad_(False)
    body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    cache_files = []
    shard = {'name': [], 'pl_input': [], 'pl_target': [], 'pl_base': [], 'pl_init_feature': [], 'num_frames': []}
    shard_idx = 0
    total_sequences = 0
    total_frames = 0
    imu_field_contracts = {}

    def flush():
        nonlocal shard, shard_idx
        if not shard['name']:
            return
        out = output_dir / f'pl_curve_cache_shard{shard_idx:05d}.pt'
        torch.save(shard, out)
        cache_files.append({
            'path': str(out),
            'num_sequences': len(shard['name']),
            'num_frames': int(sum(shard['num_frames'])),
        })
        shard_idx += 1
        shard = {'name': [], 'pl_input': [], 'pl_target': [], 'pl_base': [], 'pl_init_feature': [], 'num_frames': []}

    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            pose_gt = data['pose_gt'][seq_idx].float()
            if 'offset_r' not in data:
                raise KeyError(f'{cache_file} has no offset_r field required for PL init feature.')
            offset_r = data['offset_r'][seq_idx].float()
            aM, wM, RMB, imu_contract = select_imu_fields(data, seq_idx, imu_input_mode)
            imu_field_contracts[json.dumps(imu_contract, sort_keys=True)] = imu_contract
            record = {'aM': aM, 'wM': wM, 'RMB': RMB}
            pl_input = sequence_pl_inputs(record)
            pl_target = normalize_gravity(pl_target_from_pose(pose_gt.to(DEVICE), body_model).float()).cpu()
            pl_init = pl_init_feature_from_pose(offset_r, pose_gt[0], body_model)
            pl_base = sequence_base_pl(gpnet, pl_input, pl_target[0])
            if not (torch.isfinite(pl_input).all() and torch.isfinite(pl_target).all() and torch.isfinite(pl_base).all() and torch.isfinite(pl_init).all()):
                raise RuntimeError(f'Non-finite PL cache tensors at {name}.')
            shard['name'].append(name)
            shard['pl_input'].append(pl_input.cpu())
            shard['pl_target'].append(pl_target.cpu())
            shard['pl_base'].append(pl_base.cpu())
            shard['pl_init_feature'].append(pl_init.cpu())
            shard['num_frames'].append(int(pl_input.shape[0]))
            total_sequences += 1
            total_frames += int(pl_input.shape[0])
            if len(shard['name']) >= shard_size:
                flush()
            if total_sequences % 25 == 0:
                print(json.dumps({'processed_sequences': total_sequences, 'processed_frames': total_frames}))
    flush()
    manifest = {
        'type': 'pl_curve_cache_v2',
        'source_cache': str(input_cache),
        'source_manifest': source_manifest,
        'imu_input_mode': imu_input_mode,
        'imu_field_contracts': list(imu_field_contracts.values()),
        'init_size': 36,
        'init_layout': 'offset_r[18] + pRL[15] + gR0[3]',
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'fields': {
            'pl_input': '[T,84] PL input feature built from selected IMU fields',
            'pl_target': '[T,18] derived GT pRB[15]+gR[3]',
            'pl_base': '[T,18] official frozen PL output initialized by pl_target[0]',
            'pl_init_feature': '[36] sequence init feature: offset_r flatten[18] + pRL[15] + gR0[3]',
        },
    }
    manifest_path = output_dir / 'pl_curve_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Precompute PLCurve input/target/base tensors.')
    parser.add_argument('--input-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--shard-size', type=int, default=100)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='official')
    args = parser.parse_args()
    manifest = build_cache(args.input_cache, args.output_dir, args.shard_size, args.imu_input_mode)
    print(json.dumps({
        'status': 'ok',
        'manifest': str(args.output_dir / 'pl_curve_cache_manifest.json'),
        'num_sequences': manifest['num_sequences'],
        'num_frames': manifest['num_frames'],
    }, indent=2))


if __name__ == '__main__':
    main()
