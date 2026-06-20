#!/usr/bin/env python3
"""Build joint-target NewPL caches with frozen joint-acceleration augmented 102D input."""

import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files
from net import GPNet
from pl_curve import (
    PL_FROZEN_JOINT_ACC_AUG_INPUT_SIZE,
    control_fit_contract,
    fit_uniform_cubic_spline_controls,
    normalize_gravity,
    pl_input_feature,
    pl_joint_leaf_init_feature,
    pl_target_from_pose,
)
from pl_joint_leaf_acc_cache import (
    acceleration_predictor_feature,
    centered_ma,
    finite_check,
    load_acc_model,
    resolve_cache_file,
    sequence_legacy_pl_inputs,
    sequence_official_pl_base,
)
from pl_joint_target import LEAF5_JOINTS, TARGET_CONTRACT, joint_pRB_target_from_pose


EXPERIMENT = 'pl_joint_control_acc_aug102_v1'
FEATURE_MODE = 'frozen_joint_acc_aug102'
DEFAULT_ACC_CHECKPOINT = Path(
    'code/outputs/imu_leaf_acc_predictor_v1/'
    'full_world_leaf5_no_trans_smoothed_gtacc_centered_ma_w9_20260619_155137/'
    'dip_finetune/best.pt'
)
FALLBACK_ACC_CHECKPOINT = Path(
    '/home/lingfeng/projects/imu_acc_explainability/code/outputs/'
    'imu_leaf_acc_predictor_v1/'
    'full_world_leaf5_no_trans_smoothed_gtacc_centered_ma_w9_20260619_155137/'
    'dip_finetune/best.pt'
)


def resolve_acc_checkpoint(path):
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = Path(__file__).resolve().parent / path
        if candidate.exists():
            return candidate
    if FALLBACK_ACC_CHECKPOINT.exists():
        return FALLBACK_ACC_CHECKPOINT
    return path


def source_sequence_count(data, seq_idx):
    if 'num_frames' in data:
        return int(data['num_frames'][seq_idx])
    return int(data['aM'][seq_idx].shape[0])


def get_beta(data, seq_idx, n):
    for key in ('shape', 'beta', 'betas'):
        if key not in data:
            continue
        value = data[key][seq_idx].float()
        if value.dim() == 1:
            return value
        if value.dim() >= 2 and value.shape[0] == n:
            return value
        return value.reshape(-1)
    return None


@torch.no_grad()
def predict_frozen_joint_acc(aM, wM, RMB, model, stats, device, smooth_window):
    feature126, a_smoothed = acceleration_predictor_feature(aM, wM, RMB, smooth_window)
    normed = (feature126.to(device) - stats['input_mean'].view(1, -1)) / stats['input_std'].view(1, -1)
    pred = model(normed.unsqueeze(1), torch.tensor([feature126.shape[0]], device=device))
    return pred[:, 0].detach().cpu().float(), a_smoothed.float(), feature126


def root_frame_joint_acc(pred_world_leaf5, RMB):
    root_R = RMB[:, 5].float()
    return pred_world_leaf5.reshape(pred_world_leaf5.shape[0], 5, 3).bmm(root_R).reshape(pred_world_leaf5.shape[0], 15)


def root_frame_root_acc(a_smoothed, RMB):
    root_R = RMB[:, 5].float()
    return a_smoothed[:, 5:6].bmm(root_R).reshape(a_smoothed.shape[0], 3)


def build_feature102(legacy84, frozen_joint_acc_R, root_acc_smooth_R):
    feature = torch.cat((legacy84, frozen_joint_acc_R, root_acc_smooth_R), dim=-1).float()
    if feature.shape[-1] != PL_FROZEN_JOINT_ACC_AUG_INPUT_SIZE:
        raise RuntimeError(f'frozen_joint_acc_aug102 must be 102D, got {feature.shape[-1]}.')
    return feature


def load_gt_control_records(cache_path):
    if not cache_path:
        return None
    files, manifest = load_cache_files(cache_path)
    manifest_root = Path(manifest.get('root', '')) if manifest else Path()
    records = {}
    for cache_file in files:
        shard_path = Path(cache_file)
        if not shard_path.is_absolute() and not shard_path.exists():
            candidates = []
            if manifest_root:
                candidates.append(manifest_root / shard_path)
            if manifest and manifest.get('cache_files'):
                for entry in manifest['cache_files']:
                    if entry.get('path') == str(cache_file) and entry.get('source_path'):
                        candidates.append(Path(entry['source_path']))
            candidates.extend([Path(cache_path).resolve().parent / shard_path, resolve_cache_file(shard_path)])
            shard_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        data = torch.load(shard_path, map_location='cpu', weights_only=False)
        required = ('joint_pos_R', 'joint_pos_R_control', 'pl_pRB_gR1', 'pl_pRB_gR1_control')
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f'{cache_file} missing GT control fields: {missing}')
        for idx, name in enumerate(data['name']):
            n = int(data['num_frames'][idx]) if 'num_frames' in data else int(data['joint_pos_R'][idx].shape[0])
            records[str(name)] = {
                'joint_pos_R': data['joint_pos_R'][idx][:n].float(),
                'joint_pos_R_control': data['joint_pos_R_control'][idx][:n].float(),
                'pl_pRB_gR1': data['pl_pRB_gR1'][idx][:n].float(),
                'pl_pRB_gR1_control': data['pl_pRB_gR1_control'][idx][:n].float(),
                'num_frames': n,
            }
    return {'path': str(cache_path), 'manifest': manifest, 'records': records}


def joint_target_from_gt_control(record, n):
    leaf = torch.as_tensor(LEAF5_JOINTS, dtype=torch.long)
    joint = record['joint_pos_R'][:n].index_select(1, leaf).reshape(n, 15)
    joint_control = record['joint_pos_R_control'][:n].index_select(1, leaf).reshape(n, 15)
    g = normalize_gravity(record['pl_pRB_gR1'][:n])[..., 15:]
    g_control = normalize_gravity(record['pl_pRB_gR1_control'][:n])[..., 15:]
    return normalize_gravity(torch.cat((joint, g), dim=-1)), torch.cat((joint_control, g_control), dim=-1)


def build_one_record(data, seq_idx, joint_body_model, vertex_body_model, gpnet, acc_ctx, args, gt_controls):
    name = str(data['name'][seq_idx])
    missing = [key for key in ('pose_gt', 'aM', 'wM', 'RMB', 'offset_r') if key not in data]
    if missing:
        raise KeyError(f'{name} missing required fields: {missing}')
    n = min(
        source_sequence_count(data, seq_idx),
        int(data['pose_gt'][seq_idx].shape[0]),
        int(data['aM'][seq_idx].shape[0]),
        int(data['wM'][seq_idx].shape[0]),
        int(data['RMB'][seq_idx].shape[0]),
    )
    if args.max_frames:
        n = min(n, int(args.max_frames))
    pose_gt = data['pose_gt'][seq_idx][:n].float()
    aM = data['aM'][seq_idx][:n].float()
    wM = data['wM'][seq_idx][:n].float()
    RMB = data['RMB'][seq_idx][:n].float()
    offset_r = data['offset_r'][seq_idx].float()
    beta = get_beta(data, seq_idx, n)
    legacy84 = sequence_legacy_pl_inputs(aM, wM, RMB)

    gt_entry = None if gt_controls is None else gt_controls['records'].get(name)
    if gt_entry is not None:
        n = min(n, int(gt_entry['num_frames']))
        pose_gt, aM, wM, RMB, legacy84 = pose_gt[:n], aM[:n], wM[:n], RMB[:n], legacy84[:n]
        pl_target, pl_target_control = joint_target_from_gt_control(gt_entry, n)
    else:
        if gt_controls is not None:
            raise KeyError(f'{name} missing from GT control cache {gt_controls["path"]}.')
        pl_target = normalize_gravity(
            joint_pRB_target_from_pose(pose_gt.to(DEVICE), joint_body_model, beta=beta).float()
        ).cpu()
        target_for_controls = torch.cat((pl_target[:, :15], normalize_gravity(pl_target)[:, 15:]), dim=-1)
        pl_target_control = fit_uniform_cubic_spline_controls(target_for_controls).cpu()

    legacy_init_target = normalize_gravity(
        pl_target_from_pose(pose_gt.to(DEVICE), vertex_body_model).float()
    ).cpu()[0]
    pl_base = normalize_gravity(sequence_official_pl_base(gpnet, legacy84, legacy_init_target))
    acc_world, a_smoothed, feature126 = predict_frozen_joint_acc(
        aM, wM, RMB, acc_ctx['model'], acc_ctx['stats'], acc_ctx['device'], args.acc_smooth_window
    )
    frozen_joint_acc_R = root_frame_joint_acc(acc_world, RMB)
    root_acc_smooth_R = root_frame_root_acc(a_smoothed, RMB)
    pl_input = build_feature102(legacy84, frozen_joint_acc_R, root_acc_smooth_R)
    pl_init = pl_joint_leaf_init_feature(offset_r, pl_target[0], init_size=args.init_size)
    vertex_target = normalize_gravity(pl_target_from_pose(pose_gt.to(DEVICE), vertex_body_model).float()).cpu()

    finite_check(
        name,
        pl_input=pl_input,
        pl_target=pl_target,
        pl_base=pl_base,
        pl_init_feature=pl_init,
        pl_target_control=pl_target_control,
        frozen_joint_acc_R=frozen_joint_acc_R,
        root_acc_smooth_R=root_acc_smooth_R,
    )
    return {
        'name': name,
        'pl_input': pl_input.cpu(),
        'pl_target': pl_target.cpu(),
        'pl_base': pl_base.cpu(),
        'pl_init_feature': pl_init.cpu(),
        'pl_target_control': pl_target_control.cpu(),
        'num_frames': int(n),
        'legacy_pl_input_84D': legacy84.cpu(),
        'frozen_joint_acc_W': acc_world.cpu(),
        'frozen_joint_acc_R': frozen_joint_acc_R.cpu(),
        'root_acc_smooth_W': a_smoothed[:, 5].cpu(),
        'root_acc_smooth_R': root_acc_smooth_R.cpu(),
        'feature126': feature126.cpu(),
        'vertex_pRB_target_legacy': vertex_target[:, :15].cpu(),
        'joint_minus_vertex_l2_mean_m': float((pl_target[:, :15] - vertex_target[:, :15]).reshape(n, 5, 3).norm(dim=-1).mean()),
        'source_name': str(data['source_name'][seq_idx]) if 'source_name' in data else name,
        'pair_id': str(data['pair_id'][seq_idx]) if 'pair_id' in data else name,
        'view_type': str(data['view_type'][seq_idx]) if 'view_type' in data else 'single',
    }


def pad_tensor(values, tail_shape, dtype=torch.float32):
    max_len = max(int(v.shape[0]) for v in values)
    out = torch.zeros((len(values), max_len) + tuple(tail_shape), dtype=dtype)
    for idx, value in enumerate(values):
        out[idx, :value.shape[0]] = value.to(dtype=dtype)
    return out


def write_shard(path, rows):
    payload = {
        'name': [row['name'] for row in rows],
        'pl_input': pad_tensor([row['pl_input'] for row in rows], (102,)),
        'pl_target': pad_tensor([row['pl_target'] for row in rows], (18,)),
        'pl_base': pad_tensor([row['pl_base'] for row in rows], (18,)),
        'pl_init_feature': torch.stack([row['pl_init_feature'] for row in rows]),
        'pl_target_control': pad_tensor([row['pl_target_control'] for row in rows], (18,)),
        'num_frames': torch.tensor([row['num_frames'] for row in rows], dtype=torch.long),
        'source_name': [row['source_name'] for row in rows],
        'pair_id': [row['pair_id'] for row in rows],
        'view_type': [row['view_type'] for row in rows],
        'legacy_pl_input_84D': pad_tensor([row['legacy_pl_input_84D'] for row in rows], (84,)),
        'frozen_joint_acc_W': pad_tensor([row['frozen_joint_acc_W'] for row in rows], (15,)),
        'frozen_joint_acc_R': pad_tensor([row['frozen_joint_acc_R'] for row in rows], (15,)),
        'root_acc_smooth_W': pad_tensor([row['root_acc_smooth_W'] for row in rows], (3,)),
        'root_acc_smooth_R': pad_tensor([row['root_acc_smooth_R'] for row in rows], (3,)),
        'joint_minus_vertex_l2_mean_m': torch.tensor([row['joint_minus_vertex_l2_mean_m'] for row in rows]),
    }
    torch.save(payload, path)
    return {
        'path': str(path),
        'num_sequences': len(rows),
        'num_frames': int(sum(row['num_frames'] for row in rows)),
    }


def summarize_rows(rows):
    if not rows:
        return {}
    stats = {}
    for key in ('frozen_joint_acc_W', 'frozen_joint_acc_R', 'root_acc_smooth_W', 'root_acc_smooth_R'):
        norms = [row[key].reshape(row[key].shape[0], -1, 3).norm(dim=-1).mean() for row in rows]
        stats[f'{key}_l2_norm_mean'] = float(torch.stack(norms).mean())
    stats['joint_minus_vertex_l2_mean_m'] = float(torch.tensor([row['joint_minus_vertex_l2_mean_m'] for row in rows]).mean())
    stats['feature_dim'] = 102
    stats['target_dim'] = 18
    return stats


def build_cache(args):
    args.acc_checkpoint = resolve_acc_checkpoint(args.acc_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(args.input_cache)
    gt_controls = load_gt_control_records(args.gt_control_cache)
    device = torch.device(args.device if args.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    acc_model, acc_stats, acc_checkpoint = load_acc_model(args.acc_checkpoint, device)
    for parameter in acc_model.parameters():
        parameter.requires_grad_(False)
    acc_ctx = {'model': acc_model, 'stats': acc_stats, 'device': device}
    gpnet = GPNet().eval().to(DEVICE)
    for parameter in gpnet.parameters():
        parameter.requires_grad_(False)
    joint_body_model = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    vertex_body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)

    rows, cache_files = [], []
    total_sequences = 0
    total_frames = 0
    sanity_rows = []

    def flush():
        nonlocal rows, total_sequences, total_frames
        if not rows:
            return
        path = args.output_dir / f'pl_joint_control_acc_aug102_shard{len(cache_files):05d}.pt'
        info = write_shard(path, rows)
        cache_files.append(info)
        sanity_rows.extend(rows)
        total_sequences += int(info['num_sequences'])
        total_frames += int(info['num_frames'])
        rows = []

    for cache_file in files:
        data = torch.load(resolve_cache_file(cache_file), map_location='cpu', weights_only=False)
        for seq_idx, _ in enumerate(data['name']):
            if args.max_sequences and total_sequences + len(rows) >= args.max_sequences:
                break
            rows.append(build_one_record(
                data, seq_idx, joint_body_model, vertex_body_model, gpnet, acc_ctx, args, gt_controls
            ))
            if len(rows) >= args.shard_size:
                flush()
        if args.max_sequences and total_sequences + len(rows) >= args.max_sequences:
            break
    flush()
    sanity = summarize_rows(sanity_rows)
    manifest = {
        'type': 'pl_joint_control_acc_aug102_cache_v1',
        'compatible_train_cache_type': 'pl_curve_joint_leaf_acc_cache_v1',
        'experiment': EXPERIMENT,
        'target_mode': 'joint_pRB',
        'target_contract': TARGET_CONTRACT,
        'feature_mode': FEATURE_MODE,
        'feature_dim': 102,
        'feature_layout': 'aRB[18] + wRB[18] + RRB[45] + gR0[3] + frozen_joint_acc_R[15] + root_acc_smooth_R[3]',
        'feature_indices': {
            'aRB': [0, 18],
            'wRB': [18, 36],
            'RRB': [36, 81],
            'gR0': [81, 84],
            'frozen_joint_acc_R': [84, 99],
            'root_acc_smooth_R': [99, 102],
        },
        'source_cache': str(args.input_cache),
        'source_manifest': source_manifest,
        'pl_base_source': 'official_pl_s1',
        'pl_base_source_detail': 'GPNet.plnet official PL-s1 RNN on legacy 84D IMU input; base is diagnostic only.',
        'target_control_fit_contract': control_fit_contract(),
        'init_size': int(args.init_size),
        'init_layout': 'offset_r[18] + initial_joint_pRB[15] + gR0[3]',
        'frozen_acc_source': {
            'network': 'imu_leaf_acc_predictor_v1',
            'checkpoint_path': str(args.acc_checkpoint),
            'model_config': acc_checkpoint.get('model_config'),
            'normalization_keys': sorted(acc_stats),
            'input_contract': acc_checkpoint.get('input_contract'),
            'output_original_frame': 'world/model frame',
            'converted_frame': 'PL/root frame',
            'conversion': 'a_joint_pred_R = a_joint_pred_W @ RMB[:,5]',
            'leaf_joints': LEAF5_JOINTS,
            'eval_only': True,
        },
        'root_acc_smooth_contract': {
            'source': 'selected root/pelvis IMU acceleration aM[:,5]',
            'smoothing': f'centered_ma window={int(args.acc_smooth_window)}',
            'conversion': 'a_root_smooth_R = a_root_smooth_W @ RMB[:,5]',
            'not_smpl_fk_root_acceleration': True,
        },
        'notices': [
            'legacy vertex pRB target is not used as training target.',
            'PL output target changed to SMPL joint pRB.',
            'frozen acceleration predictor is eval-only and not trained.',
            'no acceleration target output; acceleration is an auxiliary input and decoded trajectory evaluation target.',
            '102D frozen_joint_acc_aug102 is not smooth_residual 102D.',
        ],
        'sanity': sanity,
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'command': shlex.join(sys.argv),
    }
    manifest_path = args.output_dir / 'pl_curve_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    summary_path = args.output_dir / 'SUMMARY.md'
    summary_path.write_text(build_summary(manifest) + '\n')
    print(json.dumps({'status': 'ok', 'manifest': str(manifest_path), 'sanity': sanity}, indent=2))
    return manifest


def build_summary(manifest):
    s = manifest.get('sanity', {})
    return '\n'.join([
        '# PL Joint Control Acc-Aug102 v1 Cache Smoke',
        '',
        f"- Experiment: `{manifest['experiment']}`",
        f"- Target: {manifest['target_contract']}",
        f"- Input layout: `{manifest['feature_layout']}`",
        f"- Frozen checkpoint: `{manifest['frozen_acc_source']['checkpoint_path']}`",
        f"- Coordinate transform: `{manifest['frozen_acc_source']['conversion']}`",
        '',
        '| sanity | value |',
        '|---|---:|',
        f"| feature_dim | {s.get('feature_dim', 0)} |",
        f"| target_dim | {s.get('target_dim', 0)} |",
        f"| joint_minus_vertex_l2_mean_m | {s.get('joint_minus_vertex_l2_mean_m', 0.0):.6f} |",
        f"| frozen_joint_acc_W_l2_norm_mean | {s.get('frozen_joint_acc_W_l2_norm_mean', 0.0):.6f} |",
        f"| frozen_joint_acc_R_l2_norm_mean | {s.get('frozen_joint_acc_R_l2_norm_mean', 0.0):.6f} |",
        f"| root_acc_smooth_W_l2_norm_mean | {s.get('root_acc_smooth_W_l2_norm_mean', 0.0):.6f} |",
        f"| root_acc_smooth_R_l2_norm_mean | {s.get('root_acc_smooth_R_l2_norm_mean', 0.0):.6f} |",
    ])


def validate_cache(args):
    files, manifest = load_cache_files(args.cache)
    if manifest.get('type') != 'pl_joint_control_acc_aug102_cache_v1':
        raise ValueError(f'{args.cache} is not pl_joint_control_acc_aug102_cache_v1.')
    if manifest.get('feature_mode') != FEATURE_MODE or int(manifest.get('feature_dim')) != 102:
        raise RuntimeError('Invalid feature mode/dim in manifest.')
    rows = []
    for cache_file in files:
        data = torch.load(resolve_cache_file(cache_file), map_location='cpu', weights_only=False)
        for idx, name in enumerate(data['name']):
            n = int(data['num_frames'][idx])
            feat = data['pl_input'][idx, :n].float()
            finite_check(str(name), pl_input=feat, pl_target=data['pl_target'][idx, :n].float())
            if feat.shape[-1] != 102:
                raise RuntimeError(f'{name} feature dim {feat.shape[-1]} != 102')
            if not torch.allclose(feat[:, 36:81], data['legacy_pl_input_84D'][idx, :n, 36:81], atol=1e-6):
                raise RuntimeError(f'{name} RRB block moved or changed.')
            if not torch.allclose(feat[:, 81:84], data['legacy_pl_input_84D'][idx, :n, 81:84], atol=1e-6):
                raise RuntimeError(f'{name} gR0 block moved or changed.')
            rows.append({
                'name': str(name),
                'feature_84_99_abs_mean': float(feat[:, 84:99].abs().mean()),
                'feature_99_102_abs_mean': float(feat[:, 99:102].abs().mean()),
            })
    payload = {'status': 'ok', 'manifest': str(args.cache), 'rows': rows, 'sanity': manifest.get('sanity', {})}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    build = sub.add_parser('build')
    build.add_argument('--input-cache', type=Path, required=True)
    build.add_argument('--gt-control-cache', type=Path, default=None)
    build.add_argument('--output-dir', type=Path, required=True)
    build.add_argument('--acc-checkpoint', type=Path, default=DEFAULT_ACC_CHECKPOINT)
    build.add_argument('--acc-smooth-window', type=int, default=9)
    build.add_argument('--init-size', type=int, choices=(36,), default=36)
    build.add_argument('--shard-size', type=int, default=100)
    build.add_argument('--max-sequences', type=int, default=0)
    build.add_argument('--max-frames', type=int, default=0)
    build.add_argument('--device', default='cuda:0')
    val = sub.add_parser('validate')
    val.add_argument('--cache', type=Path, required=True)
    val.add_argument('--output-json', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == 'build':
        build_cache(args)
    else:
        validate_cache(args)


if __name__ == '__main__':
    main()
