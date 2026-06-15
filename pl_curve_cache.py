import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files
from net import GPNet
from pl_curve import (
    PL_LEGACY_INPUT_SIZE,
    PL_OFFSET_AWARE_INPUT_SIZE,
    PL_SMOOTH_RESIDUAL_INPUT_SIZE,
    normalize_gravity,
    PL_BONE_LEAF_JOINT_IDS,
    control_fit_contract,
    pl_bone6d_target_from_pose,
    pl_init_feature_from_pose,
    pl_input_feature,
    pl_offset_aware_sequence_features,
    pl_smooth_residual_sequence_features,
    pl_target_from_pose,
)
from imu_position_offset import load_offset_cache


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


def sequence_offset_aware_pl_inputs(record, offset_r, dt):
    return pl_offset_aware_sequence_features(
        record['aM'],
        record['wM'],
        record['RMB'],
        offset_r,
        dt=dt,
    ).float()


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


def build_cache(
    input_cache,
    output_dir,
    shard_size,
    imu_input_mode,
    offset_cache_path=None,
    max_sequences=0,
    feature_mode='legacy',
    dt=1.0 / 60.0,
    acc_filter_mode='causal_iir',
    cutoff_hz=20.0,
    filter_fs=60.0,
    filter_order=2,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(input_cache)
    external_offsets = load_offset_cache(offset_cache_path) if offset_cache_path else None
    gpnet = GPNet().eval().to(DEVICE)
    for parameter in gpnet.parameters():
        parameter.requires_grad_(False)
    body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    cache_files = []
    if feature_mode not in ('legacy', 'offset_aware', 'smooth_residual'):
        raise ValueError(f'Unsupported PL feature_mode={feature_mode}.')
    if feature_mode == 'offset_aware':
        feature_dim = PL_OFFSET_AWARE_INPUT_SIZE
    elif feature_mode == 'smooth_residual':
        feature_dim = PL_SMOOTH_RESIDUAL_INPUT_SIZE
    else:
        feature_dim = PL_LEGACY_INPUT_SIZE
    shard = {
        'name': [],
        'pl_input': [],
        'pl_target': [],
        'pl_base': [],
        'pl_init_feature': [],
        'bone6d_target': [],
        'num_frames': [],
    }
    optional_meta_keys = ('source_name', 'pair_id', 'view_type')
    for key in optional_meta_keys:
        shard[key] = []
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
        shard = {
            'name': [],
            'pl_input': [],
            'pl_target': [],
            'pl_base': [],
            'pl_init_feature': [],
            'bone6d_target': [],
            'num_frames': [],
        }
        for key in optional_meta_keys:
            shard[key] = []

    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            if max_sequences and total_sequences >= max_sequences:
                break
            pose_gt = data['pose_gt'][seq_idx].float()
            if external_offsets is not None:
                if str(name) not in external_offsets:
                    raise KeyError(f'{offset_cache_path} has no offset for sequence {name}.')
                offset_r = external_offsets[str(name)].float()
            elif 'offset_r' not in data:
                raise KeyError(f'{cache_file} has no offset_r field required for PL init feature.')
            else:
                offset_r = data['offset_r'][seq_idx].float()
            aM, wM, RMB, imu_contract = select_imu_fields(data, seq_idx, imu_input_mode)
            imu_field_contracts[json.dumps(imu_contract, sort_keys=True)] = imu_contract
            record = {'aM': aM, 'wM': wM, 'RMB': RMB}
            legacy_pl_input = sequence_pl_inputs(record)
            if feature_mode == 'offset_aware':
                pl_input = sequence_offset_aware_pl_inputs(record, offset_r, dt=dt)
            elif feature_mode == 'smooth_residual':
                pl_input = pl_smooth_residual_sequence_features(
                    record['aM'],
                    record['wM'],
                    record['RMB'],
                    cutoff_hz=cutoff_hz,
                    fs=filter_fs,
                    smooth_mode=acc_filter_mode,
                    filter_order=filter_order,
                )
            else:
                pl_input = legacy_pl_input
            pl_target = normalize_gravity(pl_target_from_pose(pose_gt.to(DEVICE), body_model).float()).cpu()
            bone6d_target = pl_bone6d_target_from_pose(pose_gt.to(DEVICE), body_model).float().cpu()
            pl_init = pl_init_feature_from_pose(offset_r, pose_gt[0], body_model)
            pl_base = sequence_base_pl(gpnet, legacy_pl_input, pl_target[0])
            if not (
                torch.isfinite(pl_input).all()
                and torch.isfinite(pl_target).all()
                and torch.isfinite(pl_base).all()
                and torch.isfinite(pl_init).all()
                and torch.isfinite(bone6d_target).all()
            ):
                raise RuntimeError(f'Non-finite PL cache tensors at {name}.')
            shard['name'].append(name)
            shard['pl_input'].append(pl_input.cpu())
            shard['pl_target'].append(pl_target.cpu())
            shard['pl_base'].append(pl_base.cpu())
            shard['pl_init_feature'].append(pl_init.cpu())
            shard['bone6d_target'].append(bone6d_target.cpu())
            shard['num_frames'].append(int(pl_input.shape[0]))
            for key in optional_meta_keys:
                if key in data:
                    shard[key].append(data[key][seq_idx])
                else:
                    shard[key].append(str(name) if key in ('source_name', 'pair_id') else 'single')
            total_sequences += 1
            total_frames += int(pl_input.shape[0])
            if len(shard['name']) >= shard_size:
                flush()
            if total_sequences % 25 == 0:
                print(json.dumps({'processed_sequences': total_sequences, 'processed_frames': total_frames}))
        if max_sequences and total_sequences >= max_sequences:
            break
    flush()
    manifest = {
        'type': 'pl_curve_cache_v4',
        'source_cache': str(input_cache),
        'source_manifest': source_manifest,
        'imu_input_mode': imu_input_mode,
        'feature_mode': feature_mode,
        'feature_dim': feature_dim,
        'feature_layout': (
            'raw_aRB[18] + corrected_aRB[18] + a_tangent[18] + '
            'a_centripetal[18] + a_lever[18] + wRB[18] + RRB[45] + gR0[3]'
            if feature_mode == 'offset_aware'
            else (
                'aRB_smooth[18] + aRB_residual_raw_minus_smooth[18] + '
                'wRB[18] + RRB[45] + gR0[3]'
                if feature_mode == 'smooth_residual'
                else 'aRB[18] + wRB[18] + RRB[45] + gR0[3]'
            )
        ),
        'smooth_residual_contract': {
            'enabled': feature_mode == 'smooth_residual',
            'realtime': feature_mode == 'smooth_residual',
            'lookahead_frames': 0,
            'smooth_mode': acc_filter_mode if feature_mode == 'smooth_residual' else None,
            'cutoff_hz': float(cutoff_hz) if feature_mode == 'smooth_residual' else None,
            'fs_hz': float(filter_fs) if feature_mode == 'smooth_residual' else None,
            'filter_order': int(filter_order) if feature_mode == 'smooth_residual' else None,
            'aRB_residual': 'raw root-frame acceleration minus causal-smoothed root-frame acceleration',
            'base_pl': 'official frozen PL output from raw legacy 84D input; residual feature is only for NewPL',
        },
        'lever_arm_contract': {
            'enabled': feature_mode == 'offset_aware',
            'dt': float(dt),
            'alpha': 'causal first difference of root-frame wRB; first frame alpha is zero',
            'r_frame': 'offset_r[6,3] transformed to the root frame with root-relative sensor rotations available in RMB',
            'a_corr': 'aRB - (alpha x r + omega x (omega x r))',
        },
        'offset_cache': str(offset_cache_path) if offset_cache_path else None,
        'offset_source': 'external_offset_cache' if offset_cache_path else 'source_cache_offset_r',
        'imu_field_contracts': list(imu_field_contracts.values()),
        'init_size': 36,
        'init_layout': 'offset_r[18] + pRL[15] + gR0[3]',
        'target_control_fit_contract': control_fit_contract(),
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'max_sequences': int(max_sequences),
        'fields': {
            'pl_input': f'[T,{feature_dim}] PL input feature built from selected IMU fields',
            'pl_target': '[T,18] derived GT pRB[15]+gR[3]',
            'pl_base': '[T,18] official frozen PL output from legacy 84D PL input initialized by pl_target[0]',
            'pl_init_feature': '[36] sequence init feature: offset_r flatten[18] + pRL[15] + gR0[3]',
            'bone6d_target': '[T,30] aux-only root-relative leaf bone/body rotations as 5x6D R_RB = R_WR^T R_WB',
            'source_name/pair_id/view_type': 'preserved when present for same-motion different-offset consistency training',
        },
        'bone_aux_contract': {
            'enabled': True,
            'leaf_joint_ids': list(PL_BONE_LEAF_JOINT_IDS),
            'leaf_order': ['L_LowArm', 'R_LowArm', 'L_LowLeg', 'R_LowLeg', 'Head'],
            'rotation': 'column-major 6D representation of R_RB, where R_AB maps frame B into frame A',
            'rotation_6d_encoding': 'first 3 values are rotation column 0; last 3 values are rotation column 1',
            'target_source': 'SMPL FK global joint rotations; root-relative by R_WR^T R_WB',
            'not_official_pl_output': True,
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
    parser.add_argument('--feature-mode', choices=('legacy', 'offset_aware', 'smooth_residual'), default='legacy')
    parser.add_argument('--dt', type=float, default=1.0 / 60.0)
    parser.add_argument('--acc-filter-mode', choices=('causal_iir', 'causal_butterworth'), default='causal_iir')
    parser.add_argument('--cutoff-hz', type=float, default=20.0)
    parser.add_argument('--filter-fs', type=float, default=60.0)
    parser.add_argument('--filter-order', type=int, default=2)
    parser.add_argument('--offset-cache', type=Path, default=None, help='Optional sequence-level offset cache with name + offset/r_JS/imu_offset_r.')
    parser.add_argument('--max-sequences', type=int, default=0)
    args = parser.parse_args()
    manifest = build_cache(
        args.input_cache,
        args.output_dir,
        args.shard_size,
        args.imu_input_mode,
        args.offset_cache,
        args.max_sequences,
        feature_mode=args.feature_mode,
        dt=args.dt,
        acc_filter_mode=args.acc_filter_mode,
        cutoff_hz=args.cutoff_hz,
        filter_fs=args.filter_fs,
        filter_order=args.filter_order,
    )
    print(json.dumps({
        'status': 'ok',
        'manifest': str(args.output_dir / 'pl_curve_cache_manifest.json'),
        'num_sequences': manifest['num_sequences'],
        'num_frames': manifest['num_frames'],
    }, indent=2))


if __name__ == '__main__':
    main()
