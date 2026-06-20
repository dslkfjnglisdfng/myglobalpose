#!/usr/bin/env python3
"""Build joint-leaf NewPL caches with optional frozen leaf-acc features."""

import argparse
import json
import shlex
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files
from net import GPNet
from pl_curve import (
    PL_BONE_LEAF_JOINT_IDS,
    PL_JOINT_LEAF_ACC_INPUT_SIZE,
    PL_LEGACY_INPUT_SIZE,
    control_fit_contract,
    fit_uniform_cubic_spline_controls,
    normalize_gravity,
    pl_input_feature,
    pl_joint_leaf_init_feature,
    pl_joint_leaf_target_from_pose,
    pl_target_from_pose,
)


FEATURE_MODES = {
    'baseline_jointtarget_84D': PL_LEGACY_INPUT_SIZE,
    'acc_root_102D': PL_JOINT_LEAF_ACC_INPUT_SIZE,
    'acc_mixed_102D': PL_JOINT_LEAF_ACC_INPUT_SIZE,
}

DEFAULT_ACC_CHECKPOINT = Path(
    '/home/lingfeng/projects/imu_acc_explainability/code/outputs/'
    'imu_leaf_acc_predictor_v1/'
    'full_world_leaf5_no_trans_smoothed_gtacc_centered_ma_w9_20260619_155137/'
    'dip_finetune/best.pt'
)
ACC_CODE_ROOT = Path('/home/lingfeng/projects/imu_acc_explainability/code')


def resolve_cache_file(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, Path(__file__).resolve().parent / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_gt_control_records(cache_path):
    if not cache_path:
        return None
    files, manifest = load_cache_files(cache_path)
    manifest_root = Path(manifest.get('root', '')) if manifest else Path()
    records = {}
    required = ('joint_pos_R', 'joint_pos_R_control', 'pl_pRB_gR1', 'pl_pRB_gR1_control')
    for cache_file in files:
        shard_path = Path(cache_file)
        if not shard_path.is_absolute() and not shard_path.exists():
            candidates = []
            if str(manifest_root):
                candidates.append(manifest_root / shard_path)
            candidates.extend([
                Path(cache_path).resolve().parent / shard_path,
                Path('/home/lingfeng/projects/data/dataset_work') / shard_path,
                Path(__file__).resolve().parent / 'data/dataset_work' / shard_path,
            ])
            shard_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        data = torch.load(resolve_cache_file(shard_path), map_location='cpu', weights_only=False)
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f'{cache_file} missing GT control fields required for joint-leaf PL: {missing}')
        for idx, name in enumerate(data['name']):
            n = int(data['num_frames'][idx]) if 'num_frames' in data else int(data['joint_pos_R'][idx].shape[0])
            records[str(name)] = {
                'joint_pos_R': data['joint_pos_R'][idx][:n].float(),
                'joint_pos_R_control': data['joint_pos_R_control'][idx][:n].float(),
                'pl_pRB_gR1': data['pl_pRB_gR1'][idx][:n].float(),
                'pl_pRB_gR1_control': data['pl_pRB_gR1_control'][idx][:n].float(),
                'num_frames': n,
            }
    return {'manifest': manifest, 'records': records, 'path': str(cache_path)}


def centered_ma(x, window=9):
    window = max(1, int(window))
    if window <= 1 or x.shape[0] <= 1:
        return x.float()
    if window % 2 == 0:
        window += 1
    half = window // 2
    original_shape = x.shape
    flat = x.float().reshape(x.shape[0], -1).transpose(0, 1).unsqueeze(0)
    padded = F.pad(flat, (half, half), mode='replicate')
    smooth = F.avg_pool1d(padded, kernel_size=window, stride=1)
    return smooth.squeeze(0).transpose(0, 1).reshape(original_shape)


def load_acc_model(checkpoint_path, device):
    if str(ACC_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(ACC_CODE_ROOT))
    from model.imu_acc_denoiser import CausalAccelerationDenoiser

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('model_config') or {}
    model = CausalAccelerationDenoiser(
        input_size=int(config.get('input_size', 126)),
        output_size=int(config.get('output_size', 15)),
        hidden_size=int(config.get('hidden_size', 256)),
        num_layers=int(config.get('num_layers', 2)),
        dropout=float(config.get('dropout', 0.2)),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    stats = checkpoint.get('normalization')
    if not stats:
        raise KeyError(f'{checkpoint_path} missing normalization stats.')
    stats = {key: value.to(device).float() for key, value in stats.items()}
    return model, stats, checkpoint


def sequence_legacy_pl_inputs(aM, wM, RMB):
    return torch.stack([
        pl_input_feature(aM[i], wM[i], RMB[i])
        for i in range(aM.shape[0])
    ]).float()


@torch.no_grad()
def sequence_official_pl_base(gpnet, legacy_pl_input, legacy_init_target):
    return gpnet.plnet([(legacy_pl_input.to(DEVICE), legacy_init_target.to(DEVICE))])[0].detach().cpu()


def acceleration_predictor_feature(aM, wM, RMB, smooth_window):
    a_smoothed = centered_ma(aM, window=smooth_window)
    residual = aM.float() - a_smoothed
    feature = torch.cat((
        RMB.float().reshape(RMB.shape[0], 54),
        wM.float().reshape(wM.shape[0], 18),
        a_smoothed.reshape(aM.shape[0], 18),
        aM.float().reshape(aM.shape[0], 18),
        residual.reshape(aM.shape[0], 18),
    ), dim=-1).float()
    if feature.shape[-1] != 126:
        raise RuntimeError(f'Frozen acc predictor feature must be 126D, got {feature.shape[-1]}.')
    return feature, a_smoothed


@torch.no_grad()
def predict_leaf_acc(aM, wM, RMB, model, stats, device, smooth_window):
    feature, a_smoothed = acceleration_predictor_feature(aM, wM, RMB, smooth_window)
    normed = (feature.to(device) - stats['input_mean'].view(1, -1)) / stats['input_std'].view(1, -1)
    pred = model(normed.unsqueeze(1), torch.tensor([feature.shape[0]], device=device))
    return pred[:, 0].detach().cpu().float(), a_smoothed.float()


def root_frame_leaf_acc(pred_world_leaf5, RMB):
    root_R = RMB[:, 5].float()
    return pred_world_leaf5.reshape(pred_world_leaf5.shape[0], 5, 3).bmm(root_R).reshape(pred_world_leaf5.shape[0], 15)


def smoothed_root_imu_acc(a_smoothed, RMB):
    root_R = RMB[:, 5].float()
    return a_smoothed[:, 5:6].bmm(root_R).reshape(a_smoothed.shape[0], 3)


def source_sequence_count(data, seq_idx):
    if 'num_frames' in data:
        return int(data['num_frames'][seq_idx])
    return int(data['aM'][seq_idx].shape[0])


def finite_check(name, **tensors):
    bad = [key for key, value in tensors.items() if not torch.isfinite(value).all()]
    if bad:
        raise RuntimeError(f'Non-finite tensors for {name}: {bad}')


def joint_leaf_from_gt_control(gt_record, n):
    leaf_ids = torch.as_tensor(PL_BONE_LEAF_JOINT_IDS, dtype=torch.long)
    joint = gt_record['joint_pos_R'][:n].index_select(1, leaf_ids).reshape(n, 15)
    joint_control = gt_record['joint_pos_R_control'][:n].index_select(1, leaf_ids).reshape(n, 15)
    g = normalize_gravity(gt_record['pl_pRB_gR1'][:n])[..., 15:]
    g_control = normalize_gravity(gt_record['pl_pRB_gR1_control'][:n])[..., 15:]
    return normalize_gravity(torch.cat((joint, g), dim=-1)), torch.cat((joint_control, g_control), dim=-1)


def build_one_record(
    data,
    seq_idx,
    joint_body_model,
    official_pl_body_model,
    gpnet,
    feature_mode,
    acc_ctx,
    init_size,
    smooth_window,
    gt_controls=None,
    max_frames=0,
):
    name = str(data['name'][seq_idx])
    missing = [key for key in ('pose_gt', 'aM', 'wM', 'RMB', 'offset_r') if key not in data]
    if missing:
        raise KeyError(f'{name} missing required fields for joint-leaf cache: {missing}')
    n = min(
        source_sequence_count(data, seq_idx),
        int(data['pose_gt'][seq_idx].shape[0]),
        int(data['aM'][seq_idx].shape[0]),
        int(data['wM'][seq_idx].shape[0]),
        int(data['RMB'][seq_idx].shape[0]),
    )
    if max_frames:
        n = min(n, int(max_frames))
    pose_gt = data['pose_gt'][seq_idx][:n].float()
    aM = data['aM'][seq_idx][:n].float()
    wM = data['wM'][seq_idx][:n].float()
    RMB = data['RMB'][seq_idx][:n].float()
    offset_r = data['offset_r'][seq_idx].float()
    legacy = sequence_legacy_pl_inputs(aM, wM, RMB)
    gt_entry = None if gt_controls is None else gt_controls['records'].get(name)
    if gt_entry is not None:
        n = min(n, int(gt_entry['num_frames']))
        pose_gt = pose_gt[:n]
        aM = aM[:n]
        wM = wM[:n]
        RMB = RMB[:n]
        legacy = legacy[:n]
        pl_target, pl_target_control = joint_leaf_from_gt_control(gt_entry, n)
    elif gt_controls is not None:
        raise KeyError(f'{name} missing from GT control cache {gt_controls["path"]}.')
    else:
        pl_target = normalize_gravity(pl_joint_leaf_target_from_pose(pose_gt.to(DEVICE), joint_body_model).float()).cpu()
        pl_target_control = fit_uniform_cubic_spline_controls(normalize_gravity(pl_target)).cpu()
    legacy_init_target = normalize_gravity(
        pl_target_from_pose(pose_gt.to(DEVICE), official_pl_body_model).float()
    ).cpu()[0]
    # This pl_base follows official PL-s1 prediction, not pose_pre FK.
    pl_base = normalize_gravity(sequence_official_pl_base(gpnet, legacy, legacy_init_target))
    pl_init = pl_joint_leaf_init_feature(offset_r, pl_target[0], init_size=init_size)
    if feature_mode == 'baseline_jointtarget_84D':
        pl_input = legacy
        acc_debug = {}
    else:
        if acc_ctx is None:
            raise RuntimeError(f'{feature_mode} requires a frozen acceleration predictor.')
        acc_pred, a_smoothed = predict_leaf_acc(
            aM,
            wM,
            RMB,
            acc_ctx['model'],
            acc_ctx['stats'],
            acc_ctx['device'],
            smooth_window,
        )
        pred_root = root_frame_leaf_acc(acc_pred, RMB)
        root_smoothed = smoothed_root_imu_acc(a_smoothed, RMB)
        if feature_mode == 'acc_root_102D':
            acc_extra = torch.cat((root_smoothed, pred_root), dim=-1)
        elif feature_mode == 'acc_mixed_102D':
            acc_extra = torch.cat((a_smoothed[:, 5].float(), acc_pred), dim=-1)
        else:
            raise ValueError(feature_mode)
        pl_input = torch.cat((legacy, acc_extra.float()), dim=-1)
        acc_debug = {
            'a_output_W': acc_pred,
            'a_output_R': pred_root,
            'a_smoothed_W': a_smoothed[:, 5].float(),
            'a_smoothed_R': root_smoothed,
        }
    expected_dim = FEATURE_MODES[feature_mode]
    if pl_input.shape[-1] != expected_dim:
        raise RuntimeError(f'{name} {feature_mode} expected feature dim {expected_dim}, got {pl_input.shape[-1]}.')
    finite_check(
        name,
        pl_input=pl_input,
        pl_target=pl_target,
        pl_base=pl_base,
        pl_init_feature=pl_init,
        pl_target_control=pl_target_control,
    )
    row = {
        'name': name,
        'pl_input': pl_input.cpu(),
        'pl_target': pl_target.cpu(),
        'pl_base': pl_base.cpu(),
        'pl_init_feature': pl_init.cpu(),
        'pl_target_control': pl_target_control.cpu(),
        'num_frames': int(n),
        'legacy_pl_input_84D': legacy.cpu(),
    }
    row.update(acc_debug)
    for key in ('source_name', 'pair_id', 'view_type'):
        row[key] = str(data[key][seq_idx]) if key in data else (name if key != 'view_type' else 'single')
    return row


def pad_tensor(values, tail_shape, dtype=torch.float32):
    max_len = max(int(v.shape[0]) for v in values)
    out = torch.zeros((len(values), max_len) + tuple(tail_shape), dtype=dtype)
    for idx, value in enumerate(values):
        out[idx, :value.shape[0]] = value.to(dtype=dtype)
    return out


def write_shard(path, rows, include_acc_debug):
    payload = {
        'name': [row['name'] for row in rows],
        'pl_input': pad_tensor([row['pl_input'] for row in rows], (rows[0]['pl_input'].shape[-1],)),
        'pl_target': pad_tensor([row['pl_target'] for row in rows], (18,)),
        'pl_base': pad_tensor([row['pl_base'] for row in rows], (18,)),
        'pl_init_feature': torch.stack([row['pl_init_feature'] for row in rows]),
        'pl_target_control': pad_tensor([row['pl_target_control'] for row in rows], (18,)),
        'num_frames': torch.tensor([row['num_frames'] for row in rows], dtype=torch.long),
        'source_name': [row['source_name'] for row in rows],
        'pair_id': [row['pair_id'] for row in rows],
        'view_type': [row['view_type'] for row in rows],
    }
    if include_acc_debug:
        for key, tail in (
            ('legacy_pl_input_84D', (84,)),
            ('a_output_W', (15,)),
            ('a_output_R', (15,)),
            ('a_smoothed_W', (3,)),
            ('a_smoothed_R', (3,)),
        ):
            payload[key] = pad_tensor([row[key] for row in rows], tail)
    torch.save(payload, path)
    return {
        'path': str(path),
        'num_sequences': len(rows),
        'num_frames': int(sum(row['num_frames'] for row in rows)),
    }


def build_cache(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(args.input_cache)
    gt_controls = load_gt_control_records(args.gt_control_cache)
    feature_dim = FEATURE_MODES[args.feature_mode]
    acc_ctx = None
    acc_checkpoint_meta = None
    if args.feature_mode != 'baseline_jointtarget_84D':
        device = torch.device(args.device if args.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
        model, stats, checkpoint = load_acc_model(args.acc_checkpoint, device)
        acc_ctx = {'model': model, 'stats': stats, 'device': device}
        acc_checkpoint_meta = {
            'path': str(args.acc_checkpoint),
            'model_config': checkpoint.get('model_config'),
            'input_contract': checkpoint.get('input_contract'),
            'output_contract': checkpoint.get('output_contract'),
            'target_contract': checkpoint.get('target_contract'),
            'normalization_keys': sorted(stats),
        }
    gpnet = GPNet().eval().to(DEVICE)
    for parameter in gpnet.parameters():
        parameter.requires_grad_(False)
    joint_body_model = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    official_pl_body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    rows = []
    cache_files = []
    total_sequences = 0
    total_frames = 0
    shard_idx = 0

    def flush():
        nonlocal rows, shard_idx, total_sequences, total_frames
        if not rows:
            return
        out = args.output_dir / f'pl_joint_leaf_acc_cache_shard{shard_idx:05d}.pt'
        info = write_shard(out, rows, include_acc_debug=args.feature_mode != 'baseline_jointtarget_84D')
        cache_files.append(info)
        total_sequences += int(info['num_sequences'])
        total_frames += int(info['num_frames'])
        shard_idx += 1
        rows = []

    for cache_file in files:
        data = torch.load(resolve_cache_file(cache_file), map_location='cpu', weights_only=False)
        for seq_idx, _name in enumerate(data['name']):
            if args.max_sequences and total_sequences + len(rows) >= args.max_sequences:
                break
            rows.append(build_one_record(
                data,
                seq_idx,
                joint_body_model,
                official_pl_body_model,
                gpnet,
                args.feature_mode,
                acc_ctx,
                args.init_size,
                args.acc_smooth_window,
                gt_controls=gt_controls,
                max_frames=args.max_frames,
            ))
            if len(rows) >= args.shard_size:
                flush()
        if args.max_sequences and total_sequences + len(rows) >= args.max_sequences:
            break
    flush()
    manifest = {
        'type': 'pl_curve_joint_leaf_acc_cache_v1',
        'source_cache': str(args.input_cache),
        'source_manifest': source_manifest,
        'feature_mode': args.feature_mode,
        'feature_dim': feature_dim,
        'feature_layout': (
            'aRB[18]+wRB[18]+RRB[45]+gR0[3]'
            if args.feature_mode == 'baseline_jointtarget_84D'
            else (
                'aRB[18]+wRB[18]+RRB[45]+gR0[3]+a_smoothed_R[3]+a_output_R[15]'
                if args.feature_mode == 'acc_root_102D'
                else 'aRB[18]+wRB[18]+RRB[45]+gR0[3]+a_smoothed_W[3]+a_output_W[15]'
            )
        ),
        'legacy_first_84_layout': 'aRB[18]+wRB[18]+RRB[45]+gR0[3]',
        'base_mode': 'official_pl_s1_prediction',
        'pl_base_source': 'official_pl_s1',
        'pl_base_source_detail': 'GPNet.plnet official PL-s1 RNN on legacy 84D IMU input, initialized with legacy vertex PL target first frame',
        'target_mode': 'joint_leaf_gravity',
        'pl_target_source': 'joint_leaf_gravity',
        'control_mode': 'fit_uniform_cubic_spline_controls(normalize_gravity(pl_target))',
        'gt_control_cache': None if gt_controls is None else {
            'path': gt_controls['path'],
            'manifest': gt_controls['manifest'],
            'target_fields': 'joint_pos_R leaf ids + pl_pRB_gR1 gravity',
            'control_fields': 'joint_pos_R_control leaf ids + pl_pRB_gR1_control gravity',
        },
        'target_control_fit_contract': control_fit_contract(),
        'init_size': int(args.init_size),
        'init_layout': (
            'offset_r[18]+first_frame_p_leaf_joint_R[15]+first_frame_target_gravity[3]'
            if int(args.init_size) == 36
            else 'first_frame_pl_target[18]'
        ),
        'fields': {
            'pl_input': f'[T,{feature_dim}] NewPL feature',
            'pl_target': '[T,18] p_leaf_joint_R[15]+gR1[3] from SMPL joints',
            'pl_base': '[T,18] official PL-s1 pre-correction prediction from IMU; first 15D follow legacy pRB, last 3D are official PL-s1 gR1',
            'pl_init_feature': f'[{args.init_size}] sequence init feature',
            'pl_target_control': '[T,18] derivative-aware fitted GT controls',
        },
        'ids': {
            'leaf_joint_ids': list(PL_BONE_LEAF_JOINT_IDS),
            'root_joint_id': 0,
            'root_imu_index': 5,
        },
        'flags': {
            'uses_old_vertex_target': False,
            'uses_old_vertex_base_pl': True,
            'requires_pose_prephysics': False,
            'legacy_loss_key_pRB_means': 'p_leaf_joint_R',
            'full_pipeline_eval_supported': False,
            'comparable_to_v5': True,
        },
        'evaluation_protocol_version': 'newpl_joint_leaf_acc_official_base_v2',
        'protocol_check': {
            'pl_base': 'official_pl_s1',
            'pl_target': 'joint_leaf_gravity',
            'comparable_to_v5': True,
        },
        'frozen_acceleration_source': acc_checkpoint_meta,
        'acc_predictor_input_layout': 'RMB[54]+wM[18]+aM_smoothed[18]+aM[18]+(aM-aM_smoothed)[18]',
        'acc_predictor_smoothing': {'mode': 'centered_ma', 'window': int(args.acc_smooth_window)},
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'max_frames': int(args.max_frames),
        'command': shlex.join(sys.argv),
    }
    manifest_path = args.output_dir / 'pl_curve_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    return manifest


def compare_cache_records(mode_to_records):
    names = sorted(set.intersection(*(set(records) for records in mode_to_records.values())))
    if not names:
        raise RuntimeError('No shared sequences across mode caches.')
    max_target = 0.0
    max_base = 0.0
    max_first84 = 0.0
    acc_diff = None
    for name in names:
        base = mode_to_records['baseline_jointtarget_84D'][name]
        for mode in ('acc_root_102D', 'acc_mixed_102D'):
            rec = mode_to_records[mode][name]
            n = min(base['pl_target'].shape[0], rec['pl_target'].shape[0])
            max_target = max(max_target, float((base['pl_target'][:n] - rec['pl_target'][:n]).abs().max().item()))
            max_base = max(max_base, float((base['pl_base'][:n] - rec['pl_base'][:n]).abs().max().item()))
            max_first84 = max(max_first84, float((base['pl_input'][:n, :84] - rec['pl_input'][:n, :84]).abs().max().item()))
        root = mode_to_records['acc_root_102D'][name]
        mixed = mode_to_records['acc_mixed_102D'][name]
        n = min(root['pl_input'].shape[0], mixed['pl_input'].shape[0])
        diff = float((root['pl_input'][:n, 84:] - mixed['pl_input'][:n, 84:]).abs().max().item())
        acc_diff = diff if acc_diff is None else max(acc_diff, diff)
    return {
        'shared_sequences': len(names),
        'max_abs_pl_target_diff': max_target,
        'max_abs_pl_base_diff': max_base,
        'max_abs_first84_input_diff': max_first84,
        'max_abs_acc_root_vs_mixed_last18_diff': acc_diff,
        'passes_same_target_base_first84_lt_1e-6': max(max_target, max_base, max_first84) < 1e-6,
    }


def load_mode_cache(manifest_path):
    files, manifest = load_cache_files(manifest_path)
    if manifest.get('type') != 'pl_curve_joint_leaf_acc_cache_v1':
        raise ValueError(f'{manifest_path} is not pl_curve_joint_leaf_acc_cache_v1.')
    records = {}
    expected = int(manifest['feature_dim'])
    for cache_file in files:
        data = torch.load(resolve_cache_file(cache_file), map_location='cpu', weights_only=False)
        for idx, name in enumerate(data['name']):
            n = int(data['num_frames'][idx])
            rec = {
                'pl_input': data['pl_input'][idx, :n].float(),
                'pl_target': data['pl_target'][idx, :n].float(),
                'pl_base': data['pl_base'][idx, :n].float(),
                'pl_target_control': data['pl_target_control'][idx, :n].float(),
                'pl_init_feature': data['pl_init_feature'][idx].float(),
            }
            if rec['pl_input'].shape[-1] != expected:
                raise RuntimeError(f'{name} feature dim mismatch: expected {expected}, got {rec["pl_input"].shape[-1]}.')
            finite_check(str(name), **rec)
            records[str(name)] = rec
    return manifest, records


def validate_caches(manifest_paths, output_json=None):
    mode_to_records = {}
    manifest_summary = {}
    for path in manifest_paths:
        manifest, records = load_mode_cache(path)
        mode = manifest['feature_mode']
        expected = FEATURE_MODES[mode]
        if int(manifest['feature_dim']) != expected:
            raise RuntimeError(f'{path} mode {mode} expected feature_dim {expected}.')
        flags = manifest.get('flags') or {}
        if flags.get('uses_old_vertex_target'):
            raise RuntimeError(f'{path} has forbidden old vertex target flag.')
        if manifest.get('pl_base_source') != 'official_pl_s1':
            raise RuntimeError(f'{path} must use official_pl_s1 pl_base, got {manifest.get("pl_base_source")}.')
        if 'pose_prephysics' in str(manifest.get('base_mode', '')):
            raise RuntimeError(f'{path} uses forbidden pose_prephysics base mode: {manifest.get("base_mode")}.')
        mode_to_records[mode] = records
        manifest_summary[mode] = {
            'path': str(path),
            'num_sequences': len(records),
            'feature_dim': expected,
            'pl_base_source': manifest.get('pl_base_source'),
            'protocol_check': manifest.get('protocol_check'),
        }
    missing = set(FEATURE_MODES) - set(mode_to_records)
    if missing:
        raise RuntimeError(f'Validation requires all three modes, missing {sorted(missing)}.')
    invariants = compare_cache_records(mode_to_records)
    if not invariants['passes_same_target_base_first84_lt_1e-6']:
        raise RuntimeError(f'Same-sequence invariant failed: {invariants}')
    payload = {'status': 'ok', 'modes': manifest_summary, 'invariants': invariants}
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(payload, indent=2) + '\n')
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    build = sub.add_parser('build')
    build.add_argument('--input-cache', type=Path, required=True)
    build.add_argument('--output-dir', type=Path, required=True)
    build.add_argument('--feature-mode', choices=sorted(FEATURE_MODES), required=True)
    build.add_argument('--shard-size', type=int, default=100)
    build.add_argument('--max-sequences', type=int, default=0)
    build.add_argument('--max-frames', type=int, default=0)
    build.add_argument('--init-size', type=int, choices=(18, 36), default=36)
    build.add_argument('--gt-control-cache', type=Path, default=None)
    build.add_argument('--acc-checkpoint', type=Path, default=DEFAULT_ACC_CHECKPOINT)
    build.add_argument('--acc-smooth-window', type=int, default=9)
    build.add_argument('--device', default='cuda:0')
    val = sub.add_parser('validate')
    val.add_argument('--manifests', type=Path, nargs='+', required=True)
    val.add_argument('--output-json', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == 'build':
        manifest = build_cache(args)
        print(json.dumps({
            'status': 'ok',
            'manifest': str(args.output_dir / 'pl_curve_cache_manifest.json'),
            'feature_mode': manifest['feature_mode'],
            'num_sequences': manifest['num_sequences'],
            'num_frames': manifest['num_frames'],
        }, indent=2))
    else:
        print(json.dumps(validate_caches(args.manifests, args.output_json), indent=2))


if __name__ == '__main__':
    main()
