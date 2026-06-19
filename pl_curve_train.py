import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files, load_records
from pl_curve import (
    PL_LEGACY_INPUT_SIZE,
    PL_OFFSET_AWARE_INPUT_SIZE,
    PL_SMOOTH_RESIDUAL_INPUT_SIZE,
    PL_BONE_AUX_DIM,
    PLCurveModule,
    build_pl_curve_model,
    normalize_gravity,
    pl_bone6d_target_from_pose,
    pl_bone_aux_loss,
    pl_curve_loss,
    pl_init_feature_from_pose,
    pl_input_feature,
    pl_offset_aware_sequence_features,
    pl_target_from_pose,
    replace_offset_aware_feature_offset,
    split_pl_feature,
)


def build_gpnet():
    from net import GPNet
    return GPNet()


def default_weights():
    return {
        'pRB': 1.0,
        'gR1': 1.0,
        'baseline_pRB': 0.2,
        'baseline_gR1': 0.2,
        'gt_control_pRB': 0.0,
        'gt_control_gR1': 0.0,
        'control_point_prior': 0.3,
        'tail_update_prior': 0.005,
        'pRB_dot': 0.03,
        'pRB_ddot': 0.0,
        'pRB_ddot_smooth': 0.0003,
        'gR1_dot': 0.0,
        'gR1_ddot': 0.0,
        'gR_smooth': 0.003,
        'ik1_pRJ': 0.05,
        'ik1_gR2': 0.05,
        'ik2_r6d': 0.05,
        'bone6d': 0.0,
        'bone_geo': 0.0,
        'gt_control_bone6d': 0.0,
        'gt_control_bone_geo': 0.0,
        'bone6d_dot': 0.0,
        'bone6d_ddot': 0.0,
        'bone_control_point_prior': 0.0,
        'bone_tail_update_prior': 0.0,
    }


def slice_record(record, start, length):
    seq_len = record['pl_input'].shape[0] if 'pl_input' in record else record['pose_gt'].shape[0]
    if length <= 0 or seq_len <= length:
        return record
    start = min(max(0, start), seq_len - length)
    end = start + length
    sliced = {}
    for key, value in record.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == seq_len:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    sliced['name'] = f"{record['name']}[{start}:{end}]"
    return sliced


def slice_or_pad_time(value, start, length):
    chunk = value[start:start + length]
    if length <= 0 or chunk.shape[0] >= length:
        return chunk
    if chunk.shape[0] == 0:
        raise RuntimeError('Cannot pad an empty time slice.')
    pad = chunk[-1:].expand((length - chunk.shape[0],) + chunk.shape[1:])
    return torch.cat((chunk, pad), dim=0)


def make_batch(records, starts, length):
    pl_input, pl_target, pl_base, pl_init, bone6d_target, pl_target_control = [], [], [], [], [], []
    names = []
    meta = {'source_name': [], 'pair_id': [], 'view_type': []}
    for record, start in zip(records, starts):
        seq_len = record['pl_input'].shape[0]
        start = min(max(0, int(start)), max(0, seq_len - length))
        end = start + length
        pl_input.append(slice_or_pad_time(record['pl_input'], start, length))
        pl_target.append(slice_or_pad_time(record['pl_target'], start, length))
        pl_base.append(slice_or_pad_time(record['pl_base'], start, length))
        if 'pl_target_control' in record:
            pl_target_control.append(slice_or_pad_time(record['pl_target_control'], start, length))
        if 'pl_init_feature' in record:
            pl_init.append(record['pl_init_feature'])
        if 'bone6d_target' in record:
            bone6d_target.append(slice_or_pad_time(record['bone6d_target'], start, length))
        names.append(f"{record['name']}[{start}:{end}]")
        for key in meta:
            meta[key].append(record.get(key, record.get('name')))
    out = {
        'name': '|'.join(names),
        'pl_input': torch.stack(pl_input, dim=1),
        'pl_target': torch.stack(pl_target, dim=1),
        'pl_base': torch.stack(pl_base, dim=1),
        'records': list(records),
    }
    if pl_init:
        out['pl_init_feature'] = torch.stack(pl_init, dim=0)
    if bone6d_target and len(bone6d_target) == len(records):
        out['bone6d_target'] = torch.stack(bone6d_target, dim=1)
    if pl_target_control and len(pl_target_control) == len(records):
        out['pl_target_control'] = torch.stack(pl_target_control, dim=1)
    out.update(meta)
    return out


def record_init_feature(record):
    init_feature = record.get('pl_init_feature')
    if init_feature is None:
        return None
    if init_feature.dim() == 2:
        return init_feature
    return init_feature.float()


def augment_offset_init(init_feature, dropout_prob=0.0, noise_std=0.0):
    if init_feature is None or init_feature.shape[-1] < 36:
        return init_feature
    if dropout_prob <= 0.0 and noise_std <= 0.0:
        return init_feature
    out = init_feature.clone()
    offset = out[..., :18].reshape(out.shape[:-1] + (6, 3))
    if dropout_prob > 0.0:
        keep = (torch.rand(offset.shape[:-1] + (1,), device=offset.device) >= float(dropout_prob)).to(offset.dtype)
        offset = offset * keep
    if noise_std > 0.0:
        offset = offset + torch.randn_like(offset) * float(noise_std)
    out[..., :18] = offset.reshape(out.shape[:-1] + (18,))
    return out


def build_raw_features(record, input_size, dt=1.0 / 60.0):
    if input_size == PL_OFFSET_AWARE_INPUT_SIZE:
        offset_r = record.get('offset_r', record.get('imu_offset_r'))
        if offset_r is None:
            raise KeyError(f'{record.get("name")} missing offset_r required for offset-aware PL features.')
        return pl_offset_aware_sequence_features(record['aM'], record['wM'], record['RMB'], offset_r.float(), dt=dt)
    return torch.stack([
        pl_input_feature(record['aM'][i], record['wM'][i], record['RMB'][i])
        for i in range(record['pose_gt'].shape[0])
    ]).float()


def build_features_targets(record, body_model, input_size=PL_LEGACY_INPUT_SIZE, dt=1.0 / 60.0):
    if 'pl_input' in record and 'pl_target' in record:
        return record['pl_input'].float(), normalize_gravity(record['pl_target'].float())
    features = build_raw_features(record, input_size, dt=dt)
    target = pl_target_from_pose(record['pose_gt'].float().to(DEVICE), body_model).float().cpu()
    return features, normalize_gravity(target)


def bone_target_for_record(record, body_model):
    if 'bone6d_target' in record:
        return record['bone6d_target'].float()
    if body_model is not None and 'pose_gt' in record:
        return pl_bone6d_target_from_pose(record['pose_gt'].float().to(DEVICE), body_model).float().cpu()
    return None


@torch.no_grad()
def base_pl_outputs(gpnet, features, init_target):
    gpnet.plnet.eval()
    return gpnet.plnet([(features.to(DEVICE), init_target.to(DEVICE))])[0].detach()


def load_pl_curve_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    pl_cache_types = (
        'pl_curve_cache_v1',
        'pl_curve_cache_v2',
        'pl_curve_cache_v3',
        'pl_curve_cache_v4',
        'pl_curve_joint_leaf_acc_cache_v1',
    )
    if manifest is not None and manifest.get('type') in pl_cache_types:
        records = []
        has_init = manifest.get('type') in (
            'pl_curve_cache_v2',
            'pl_curve_cache_v3',
            'pl_curve_cache_v4',
            'pl_curve_joint_leaf_acc_cache_v1',
        )
        for cache_file in files:
            data = torch.load(cache_file, map_location='cpu')
            for seq_idx, name in enumerate(data['name']):
                record = {
                    'name': name,
                    'pl_input': data['pl_input'][seq_idx].float(),
                    'pl_target': data['pl_target'][seq_idx].float(),
                    'pl_base': data['pl_base'][seq_idx].float(),
                }
                if has_init:
                    record['pl_init_feature'] = data['pl_init_feature'][seq_idx].float()
                if 'bone6d_target' in data:
                    record['bone6d_target'] = data['bone6d_target'][seq_idx].float()
                if 'pl_target_control' in data:
                    record['pl_target_control'] = data['pl_target_control'][seq_idx].float()
                for key in ('source_name', 'pair_id', 'view_type'):
                    if key in data and len(data[key]) > seq_idx:
                        record[key] = data[key][seq_idx]
                records.append(record)
                if max_sequences and len(records) >= max_sequences:
                    return records, manifest
        return records, manifest
    return load_records(cache_path, max_sequences=max_sequences)


def attach_pl_target_controls(records, control_cache_path):
    if not control_cache_path:
        return {'enabled': False, 'attached': 0, 'path': ''}
    files, manifest = load_cache_files(control_cache_path)
    controls = {}
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        if 'pl_pRB_gR1_control' not in data:
            raise KeyError(f'{cache_file} missing pl_pRB_gR1_control')
        for name, control in zip(data['name'], data['pl_pRB_gR1_control']):
            controls[str(name)] = control.float()
    missing = []
    attached = 0
    for record in records:
        key = str(record['name'])
        control = controls.get(key)
        if control is None:
            missing.append(key)
            continue
        if 'pl_target' in record and tuple(control.shape) != tuple(record['pl_target'].shape):
            raise RuntimeError(
                f'GT control shape mismatch for {key}: '
                f'control={tuple(control.shape)} target={tuple(record["pl_target"].shape)}'
            )
        record['pl_target_control'] = control
        attached += 1
    if missing:
        raise KeyError(
            f'{control_cache_path} missing {len(missing)} sequence controls; '
            f'first missing={missing[:5]}'
        )
    return {
        'enabled': True,
        'attached': attached,
        'path': str(control_cache_path),
        'manifest_type': manifest.get('type') if manifest else None,
    }


def downstream_ik_outputs(gpnet, features, pl_outputs):
    RRB0, gR0 = split_pl_feature(features.to(DEVICE))
    pRB = pl_outputs[:, :15]
    gR1 = normalize_gravity(pl_outputs)[:, 15:]
    RRB_after_pl = art.math.from_to_rotation_matrix(gR0, gR1).unsqueeze(1).matmul(RRB0)
    ik1_input = torch.cat((RRB_after_pl.flatten(1), gR1, pRB), dim=-1)
    ik1 = gpnet.iknet.net1([ik1_input])[0]
    pRJ = ik1[:, :69]
    gR2 = normalize_gravity(ik1)[:, 69:]
    RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).unsqueeze(1).matmul(RRB_after_pl)
    ik2_input = torch.cat((RRB_after_ik1.flatten(1), gR2, pRJ), dim=-1)
    ik2 = gpnet.iknet.net2([ik2_input])[0]
    return {
        'ik1_pRJ': pRJ,
        'ik1_gR2': gR2,
        'ik2_r6d': ik2,
    }


def init_feature_for_record(record, target, body_model):
    if 'pl_init_feature' in record:
        return record['pl_init_feature'].float()
    if 'offset_r' in record and 'pose_gt' in record:
        return pl_init_feature_from_pose(record['offset_r'].float(), record['pose_gt'][0].float(), body_model)
    return target[0]


def run_sequence(model, gpnet, record, body_model, weights, train_ik_distill=True):
    features, target = build_features_targets(record, body_model, input_size=model.input_size, dt=model.dt)
    features = features.to(DEVICE)
    target = target.to(DEVICE)
    if 'pl_base' in record:
        base = normalize_gravity(record['pl_base'].float()).to(DEVICE)
    else:
        base = base_pl_outputs(gpnet, features, target[0]).to(DEVICE)
    init_feature = init_feature_for_record(record, target, body_model).to(DEVICE)
    if init_feature.shape[-1] != model.init_size:
        raise RuntimeError(f'PL init dim mismatch for {record["name"]}: model expects {model.init_size}, got {init_feature.shape[-1]}.')
    out = model.forward_sequence(features, base, init_feature=init_feature)
    target_control = record.get('pl_target_control')
    target_control = None if target_control is None else target_control.to(DEVICE)
    loss, components = pl_curve_loss(out, target, {k: weights[k] for k in (
        'pRB', 'gR1', 'baseline_pRB', 'baseline_gR1', 'control_point_prior',
        'tail_update_prior', 'pRB_dot', 'pRB_ddot', 'pRB_ddot_smooth', 'gR1_dot',
        'gR1_ddot', 'gR_smooth', 'gt_control_pRB', 'gt_control_gR1'
    )}, dt=model.dt, target_control=target_control)
    bone_target = bone_target_for_record(record, body_model)
    bone_loss, bone_components = pl_bone_aux_loss(out, bone_target, {k: weights[k] for k in (
        'bone6d', 'bone_geo', 'gt_control_bone6d', 'gt_control_bone_geo',
        'bone6d_dot', 'bone6d_ddot', 'bone_control_point_prior', 'bone_tail_update_prior',
    )}, dt=model.dt)
    loss = loss + bone_loss
    components.update(bone_components)
    if train_ik_distill:
        if gpnet is None:
            raise ValueError('IK distillation requires gpnet; disable it for cache-only fast training.')
        with torch.no_grad():
            base_ik = downstream_ik_outputs(gpnet, features, base)
        with torch.backends.cudnn.flags(enabled=False):
            pred_ik = downstream_ik_outputs(gpnet, features, out['pl'])
        ik_losses = {
            'ik1_pRJ': torch.nn.functional.smooth_l1_loss(pred_ik['ik1_pRJ'], base_ik['ik1_pRJ']),
            'ik1_gR2': (1.0 - (pred_ik['ik1_gR2'] * base_ik['ik1_gR2']).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
            'ik2_r6d': torch.nn.functional.smooth_l1_loss(pred_ik['ik2_r6d'], base_ik['ik2_r6d']),
        }
    else:
        zero = loss.new_zeros(())
        ik_losses = {'ik1_pRJ': zero, 'ik1_gR2': zero, 'ik2_r6d': zero}
    for key, value in ik_losses.items():
        loss = loss + value * weights[key]
        components[key] = value
    components.update({
        'new_delta_norm': out['new_delta_norm'],
        'pl_residual_norm_mean': (out['pl'] - out['base']).norm(dim=-1).mean(),
        'gR_norm_mean': out['pl'][..., 15:].norm(dim=-1).mean(),
    })
    if 'condition_norm' in out:
        components['condition_norm'] = out['condition_norm']
    if 'bone_new_delta_norm' in out:
        components['bone_new_delta_norm'] = out['bone_new_delta_norm']
    return loss, components


def run_sequence_with_init(
    model,
    gpnet,
    record,
    body_model,
    weights,
    init_feature,
    train_ik_distill=True,
    imu_proxy_offset_acc_weight=0.0,
    imu_proxy_acc_scale=30.0,
):
    features, target = build_features_targets(record, body_model, input_size=model.input_size, dt=model.dt)
    features = features.to(DEVICE)
    target = target.to(DEVICE)
    if 'pl_base' in record:
        base = normalize_gravity(record['pl_base'].float()).to(DEVICE)
    else:
        base = base_pl_outputs(gpnet, features, target[0]).to(DEVICE)
    if init_feature is None:
        init_feature = init_feature_for_record(record, target, body_model)
    init_feature = init_feature.to(DEVICE)
    if init_feature.shape[-1] != model.init_size:
        raise RuntimeError(f'PL init dim mismatch for {record["name"]}: model expects {model.init_size}, got {init_feature.shape[-1]}.')
    out = model.forward_sequence(features, base, init_feature=init_feature)
    target_control = record.get('pl_target_control')
    target_control = None if target_control is None else target_control.to(DEVICE)
    loss, components = pl_curve_loss(out, target, {k: weights[k] for k in (
        'pRB', 'gR1', 'baseline_pRB', 'baseline_gR1', 'control_point_prior',
        'tail_update_prior', 'pRB_dot', 'pRB_ddot', 'pRB_ddot_smooth', 'gR1_dot',
        'gR1_ddot', 'gR_smooth', 'gt_control_pRB', 'gt_control_gR1'
    )}, dt=model.dt, target_control=target_control)
    bone_target = bone_target_for_record(record, body_model)
    bone_loss, bone_components = pl_bone_aux_loss(out, bone_target, {k: weights[k] for k in (
        'bone6d', 'bone_geo', 'gt_control_bone6d', 'gt_control_bone_geo',
        'bone6d_dot', 'bone6d_ddot', 'bone_control_point_prior', 'bone_tail_update_prior',
    )}, dt=model.dt)
    loss = loss + bone_loss
    components.update(bone_components)
    if train_ik_distill:
        if gpnet is None:
            raise ValueError('IK distillation requires gpnet; disable it for cache-only fast training.')
        with torch.no_grad():
            base_ik = downstream_ik_outputs(gpnet, features, base)
        with torch.backends.cudnn.flags(enabled=False):
            pred_ik = downstream_ik_outputs(gpnet, features, out['pl'])
        ik_losses = {
            'ik1_pRJ': torch.nn.functional.smooth_l1_loss(pred_ik['ik1_pRJ'], base_ik['ik1_pRJ']),
            'ik1_gR2': (1.0 - (pred_ik['ik1_gR2'] * base_ik['ik1_gR2']).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
            'ik2_r6d': torch.nn.functional.smooth_l1_loss(pred_ik['ik2_r6d'], base_ik['ik2_r6d']),
        }
    else:
        zero = loss.new_zeros(())
        ik_losses = {'ik1_pRJ': zero, 'ik1_gR2': zero, 'ik2_r6d': zero}
    for key, value in ik_losses.items():
        loss = loss + value * weights[key]
        components[key] = value
    components.update({
        'new_delta_norm': out['new_delta_norm'],
        'pl_residual_norm_mean': (out['pl'] - out['base']).norm(dim=-1).mean(),
        'gR_norm_mean': out['pl'][..., 15:].norm(dim=-1).mean(),
    })
    if 'condition_norm' in out:
        components['condition_norm'] = out['condition_norm']
    if 'bone_new_delta_norm' in out:
        components['bone_new_delta_norm'] = out['bone_new_delta_norm']
    if imu_proxy_offset_acc_weight > 0.0:
        imu_proxy, imu_proxy_components = imu_proxy_offset_acc_loss(
            out,
            features,
            acc_scale=imu_proxy_acc_scale,
        )
        loss = loss + imu_proxy * float(imu_proxy_offset_acc_weight)
        components.update(imu_proxy_components)
        components['imu_proxy_offset_acc_weighted'] = imu_proxy.detach() * float(imu_proxy_offset_acc_weight)
    return loss, components


def make_offset_contrast_init(init_feature, step=0, mode='roll_random'):
    """Create an intentionally wrong 36D init feature while preserving pRL/gR0."""
    if init_feature is None:
        return None
    if init_feature.shape[-1] < 36:
        return None
    bad = init_feature.clone()
    offset = bad[..., :18].reshape(bad.shape[:-1] + (6, 3))
    if mode == 'zero':
        offset = torch.zeros_like(offset)
    elif mode == 'negate':
        offset = -offset
    elif mode == 'roll_random':
        flat = offset.reshape(-1, 6, 3)
        if flat.shape[0] > 1:
            flat = torch.roll(flat, shifts=1 + int(step % max(1, flat.shape[0] - 1)), dims=0)
        else:
            sensor_shift = 1 + int(step % 5)
            flat = torch.roll(flat, shifts=sensor_shift, dims=1)
        offset = flat.reshape_as(offset)
    else:
        raise ValueError(f'Unsupported offset contrast mode: {mode}')
    bad[..., :18] = offset.reshape(bad.shape[:-1] + (18,))
    return bad


def pl_supervision_metric(output, target, target_mode='full_pl'):
    pred = output['pl']
    target = target.to(pred.device, pred.dtype)
    if target_mode == 'pRB':
        return torch.nn.functional.smooth_l1_loss(pred[..., :15], target[..., :15])
    if target_mode != 'full_pl':
        raise ValueError(f'Unsupported PL supervision target: {target_mode}')
    pred_gR = art.math.normalize_tensor(pred[..., 15:], avoid_nan=True)
    target_gR = art.math.normalize_tensor(target[..., 15:], avoid_nan=True)
    p_loss = torch.nn.functional.smooth_l1_loss(pred[..., :15], target[..., :15])
    g_loss = (1.0 - (pred_gR * target_gR).sum(dim=-1).clamp(-1.0, 1.0)).mean()
    return p_loss + g_loss


def imu_proxy_offset_acc_loss(output, features, acc_scale=30.0):
    """Fit a lightweight offset-aware IMU acceleration proxy from PL motion.

    Contract: features are 156D offset-aware root-frame PL inputs.  The loss
    uses the first five non-root sensors because NewPL pRB is 5x3.  It compares
    pRB_ddot + lever_rJS against raw aRB, all in the root frame.  This avoids
    using global translation, so DIP can still be treated as pseudo-rJS only;
    experiment scripts should decide which stages enable it.
    """
    if features.shape[-1] != PL_OFFSET_AWARE_INPUT_SIZE:
        zero = output['pl'].new_zeros(())
        return zero, {
            'imu_proxy_offset_acc': zero,
            'imu_proxy_offset_acc_rms': zero,
            'imu_proxy_offset_acc_available': zero,
        }
    pred_ddot = output['plddot'][..., :15]
    raw_a = features[..., :15].to(device=pred_ddot.device, dtype=pred_ddot.dtype)
    lever = features[..., 72:87].to(device=pred_ddot.device, dtype=pred_ddot.dtype)
    proxy = pred_ddot + lever
    residual = proxy - raw_a
    scale = max(float(acc_scale), 1e-6)
    loss = torch.nn.functional.smooth_l1_loss(proxy / scale, raw_a / scale)
    rms = residual.reshape(-1, 5, 3).norm(dim=-1).square().mean().sqrt()
    return loss, {
        'imu_proxy_offset_acc': loss.detach(),
        'imu_proxy_offset_acc_rms': rms.detach(),
        'imu_proxy_offset_acc_available': loss.new_tensor(1.0),
    }


def offset_contrast_loss(model, features, base, target, init_feature, step, mode, margin, target_mode='full_pl'):
    bad_init = make_offset_contrast_init(init_feature, step=step, mode=mode)
    if bad_init is None:
        zero = target.new_zeros(())
        return zero, {'offset_contrast': zero, 'offset_bad_metric': zero, 'offset_good_metric': zero, 'offset_bad_minus_good_metric': zero}
    bad_features = features
    if features.shape[-1] == PL_OFFSET_AWARE_INPUT_SIZE:
        bad_features = replace_offset_aware_feature_offset(
            features,
            bad_init[..., :18].reshape(bad_init.shape[:-1] + (6, 3)),
            dt=getattr(model, 'dt', 1.0 / 60.0),
        )
    good_out = model.forward_sequence(features, base, init_feature=init_feature)
    good_metric = pl_supervision_metric(good_out, target, target_mode=target_mode)
    bad_out = model.forward_sequence(bad_features, base, init_feature=bad_init)
    bad_metric = pl_supervision_metric(bad_out, target, target_mode=target_mode)
    contrast = good_metric + torch.relu(good_metric + float(margin) - bad_metric)
    return contrast, {
        'offset_contrast': contrast,
        'offset_bad_metric': bad_metric.detach(),
        'offset_good_metric': good_metric.detach(),
        'offset_bad_minus_good_metric': (bad_metric - good_metric).detach(),
    }


def build_pair_lookup(records):
    groups = {}
    for record in records:
        pair_id = record.get('pair_id')
        if pair_id is None:
            continue
        groups.setdefault(str(pair_id), []).append(record)
    return {
        pair_id: items
        for pair_id, items in groups.items()
        if len(items) >= 2
    }


def paired_record_for(record, pair_lookup, step=0):
    pair_id = record.get('pair_id')
    if pair_id is None:
        return None
    items = pair_lookup.get(str(pair_id), [])
    view_type = record.get('view_type')
    source_name = record.get('source_name')
    candidates = []
    for item in items:
        if item is record or item.get('name') == record.get('name'):
            continue
        if view_type is not None and item.get('view_type') == view_type:
            continue
        if source_name is not None and item.get('source_name') != source_name:
            continue
        candidates.append(item)
    if not candidates:
        return None
    return candidates[int(step) % len(candidates)]


def pl_output_consistency_metric(out_a, out_b, target_mode='full_pl'):
    if target_mode == 'pRB':
        return torch.nn.functional.smooth_l1_loss(out_a[..., :15], out_b[..., :15])
    if target_mode != 'full_pl':
        raise ValueError(f'Unsupported PL consistency target: {target_mode}')
    g_a = art.math.normalize_tensor(out_a[..., 15:], avoid_nan=True)
    g_b = art.math.normalize_tensor(out_b[..., 15:], avoid_nan=True)
    p = torch.nn.functional.smooth_l1_loss(out_a[..., :15], out_b[..., :15])
    g = (1.0 - (g_a * g_b).sum(dim=-1).clamp(-1.0, 1.0)).mean()
    return p + g


def paired_offset_consistency_loss(model, record, pair_lookup, start, length, step, target_mode='full_pl'):
    ref = next(model.parameters())
    source = record
    if 'records' in record:
        source = record['records'][0]
    paired = paired_record_for(source, pair_lookup, step=step)
    if paired is None:
        zero = ref.new_zeros(())
        return zero, {'offset_consistency': zero, 'offset_consistency_available': zero}
    seq_len = min(source['pl_input'].shape[0], paired['pl_input'].shape[0])
    if length <= 0:
        length = seq_len
    length = min(int(length), seq_len)
    if length <= 0:
        zero = ref.new_zeros(())
        return zero, {'offset_consistency': zero, 'offset_consistency_available': zero}
    start = min(max(0, int(start)), max(0, seq_len - length))
    batch = make_batch([source, paired], [start, start], length)
    init_feature = record_init_feature(batch)
    if init_feature is None:
        zero = ref.new_zeros(())
        return zero, {'offset_consistency': zero, 'offset_consistency_available': zero}
    features = batch['pl_input'].to(device=ref.device, dtype=ref.dtype)
    base = normalize_gravity(batch['pl_base'].float()).to(device=ref.device, dtype=ref.dtype)
    out = model.forward_sequence(features, base, init_feature=init_feature.to(device=ref.device, dtype=ref.dtype))
    consistency = pl_output_consistency_metric(out['pl'][:, 0], out['pl'][:, 1], target_mode=target_mode)
    return consistency, {
        'offset_consistency': consistency,
        'offset_consistency_available': consistency.new_tensor(1.0),
        'offset_consistency_pRB_l1': torch.nn.functional.smooth_l1_loss(out['pl'][:, 0, :15], out['pl'][:, 1, :15]).detach(),
    }


def average(rows):
    totals = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / max(1, len(values)) for key, values in totals.items()}


def eval_loss(
    model,
    gpnet,
    records,
    body_model,
    weights,
    max_sequences=0,
    train_ik_distill=True,
    val_window_length=0,
    val_window_seed=0,
):
    model.eval()
    rows = []
    with torch.no_grad():
        selected = records[:max_sequences] if max_sequences else records
        for record_idx, record in enumerate(selected):
            if val_window_length and val_window_length > 0:
                seq_len = record['pl_input'].shape[0] if 'pl_input' in record else record['pose_gt'].shape[0]
                max_start = max(0, seq_len - int(val_window_length))
                start = ((int(val_window_seed) + record_idx * 997) % (max_start + 1)) if max_start > 0 else 0
                record = slice_record(record, start, int(val_window_length))
            loss, components = run_sequence(
                model,
                gpnet,
                record,
                body_model,
                weights,
                train_ik_distill=train_ik_distill,
            )
            row = {'name': record['name'], 'loss': float(loss.detach())}
            row.update({key: float(value.detach()) for key, value in components.items()})
            rows.append(row)
    return {'num_sequences': len(rows), 'loss': average(rows), 'rows': rows}


def checkpoint_selection_value(validation, args):
    losses = validation['loss']
    if args.selection_metric == 'weighted_loss':
        return losses.get('loss', float('inf'))
    if args.selection_metric == 'pl_physical':
        return losses.get('pRB', 0.0) + losses.get('gR1', 0.0)
    if args.selection_metric == 'control_physical':
        return losses.get('gt_control_pRB', 0.0) + losses.get('gt_control_gR1', 0.0)
    if args.selection_metric == 'pl_and_control_physical':
        return (
            losses.get('pRB', 0.0)
            + losses.get('gR1', 0.0)
            + losses.get('gt_control_pRB', 0.0)
            + losses.get('gt_control_gR1', 0.0)
        )
    if args.selection_metric == 'bone_physical':
        return (
            losses.get('bone_geo', 0.0)
            + losses.get('gt_control_bone_geo', 0.0)
            + losses.get('gt_control_bone6d', 0.0)
        )
    if args.selection_metric == 'pl_control_bone_physical':
        return (
            losses.get('pRB', 0.0)
            + losses.get('gR1', 0.0)
            + losses.get('gt_control_pRB', 0.0)
            + losses.get('gt_control_gR1', 0.0)
            + losses.get('bone_geo', 0.0)
            + losses.get('gt_control_bone_geo', 0.0)
            + losses.get('gt_control_bone6d', 0.0)
        )
    raise ValueError(f'Unsupported selection metric: {args.selection_metric}')


def load_partial_checkpoint(model, checkpoint_state):
    model_state = model.state_dict()
    loaded = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            loaded[key] = value
        elif (
            key == 'input.weight'
            and key in model_state
            and value.shape[0] == model_state[key].shape[0]
            and getattr(model, 'input_size', None) == PL_SMOOTH_RESIDUAL_INPUT_SIZE
            and value.shape[1] == PL_LEGACY_INPUT_SIZE + model.state_dim
            and model_state[key].shape[1] == PL_SMOOTH_RESIDUAL_INPUT_SIZE + model.state_dim
        ):
            merged = model_state[key].clone()
            # Legacy layout: raw aRB[18], wRB[18], RRB[45], gR0[3], base PL[18].
            # New layout: smooth aRB[18], raw-minus-smooth residual[18],
            # wRB[18], RRB[45], gR0[3], base PL[18].  Copy the old aRB
            # weights into both smooth and residual blocks so the initialized
            # linear term can represent W*(smooth+residual) ~= W*raw.
            merged[:, 0:18] = value[:, 0:18]
            merged[:, 18:36] = value[:, 0:18]
            merged[:, 36:54] = value[:, 18:36]
            merged[:, 54:99] = value[:, 36:81]
            merged[:, 99:102] = value[:, 81:84]
            merged[:, 102:120] = value[:, 84:102]
            loaded[key] = merged
        elif key == 'init_encoder.0.weight' and key in model_state and value.shape[0] == model_state[key].shape[0]:
            merged = model_state[key].clone()
            copy_width = min(value.shape[1], merged.shape[1])
            merged[:, -copy_width:] = value[:, -copy_width:]
            loaded[key] = merged
        else:
            skipped.append(key)
    model_state.update(loaded)
    model.load_state_dict(model_state)
    return {'loaded': sorted(loaded), 'skipped': sorted(skipped)}


def save_checkpoint(path, model, optimizer, args, epoch, step, val_loss, weights):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'epoch': epoch,
        'step': step,
        'validation_loss': val_loss,
        'weights': weights,
        'model_type': 'pl_curve_v1',
        'model_variant': args.model_variant,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train PLCurve_v1 on existing GlobalPose prephysics caches.')
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--model-variant', choices=('base', 'offset_conditioned', 'offset_aware'), default='base')
    parser.add_argument('--input-size', type=int, default=0, help='PL frame feature dim. 0 infers from cache manifest/model variant.')
    parser.add_argument('--condition-scale', type=float, default=1.0)
    parser.add_argument('--offset-embed-size', type=int, default=128)
    parser.add_argument('--film-scale', type=float, default=0.1)
    parser.add_argument('--bone-aux-dim', type=int, default=0, help='Optional aux-only leaf bone orientation head. Use 30 for five 6D R_RB controls.')
    parser.add_argument('--bone-residual-scale', type=float, default=0.05)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--init-checkpoint', default='', help='Optional PLCurve checkpoint used to initialize model weights for finetuning. Optimizer state is not restored.')
    parser.add_argument('--init-size', type=int, default=36, help='PL curve init feature dim. Use 36 for offset_r[18]+pRL[15]+gR0[3], 18 for legacy init_output.')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0)
    parser.add_argument('--early-stop-patience', type=int, default=0)
    parser.add_argument(
        '--selection-metric',
        choices=('weighted_loss', 'pl_physical', 'control_physical', 'pl_and_control_physical', 'bone_physical', 'pl_control_bone_physical'),
        default='weighted_loss',
        help='Metric used for best_loss.pt. control_physical selects by fitted GT control pRB+gR1; decoded PL metrics are still reported.',
    )
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--val-window-length', type=int, default=0, help='If >0, validate on one deterministic rotating window per val sequence instead of full sequences.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch windows for precomputed PLCurve caches. Falls back to sequence-wise training for raw caches.')
    parser.add_argument('--disable-ik-distill', action='store_true')
    parser.add_argument('--pRB-weight', type=float, default=None)
    parser.add_argument('--gR1-weight', type=float, default=None)
    parser.add_argument('--baseline-pRB-weight', type=float, default=None)
    parser.add_argument('--baseline-gR1-weight', type=float, default=None)
    parser.add_argument('--gt-control-pRB-weight', type=float, default=None)
    parser.add_argument('--gt-control-gR1-weight', type=float, default=None)
    parser.add_argument('--control-point-prior-weight', type=float, default=None)
    parser.add_argument('--tail-update-prior-weight', type=float, default=None)
    parser.add_argument('--pRB-dot-weight', type=float, default=None)
    parser.add_argument('--pRB-ddot-weight', type=float, default=None)
    parser.add_argument('--pRB-ddot-smooth-weight', type=float, default=None)
    parser.add_argument('--gR1-dot-weight', type=float, default=None)
    parser.add_argument('--gR1-ddot-weight', type=float, default=None)
    parser.add_argument('--gR-smooth-weight', type=float, default=None)
    parser.add_argument('--ik-distill-weight', type=float, default=None)
    parser.add_argument('--bone6d-weight', type=float, default=None)
    parser.add_argument('--bone-geo-weight', type=float, default=None)
    parser.add_argument('--gt-control-bone6d-weight', type=float, default=None)
    parser.add_argument('--gt-control-bone-geo-weight', type=float, default=None)
    parser.add_argument('--bone6d-dot-weight', type=float, default=None)
    parser.add_argument('--bone6d-ddot-weight', type=float, default=None)
    parser.add_argument('--bone-control-point-prior-weight', type=float, default=None)
    parser.add_argument('--bone-tail-update-prior-weight', type=float, default=None)
    parser.add_argument('--offset-contrast-weight', type=float, default=0.0, help='Diagnostic offset-sensitive training term. Keeps PL input/output contract unchanged.')
    parser.add_argument('--offset-contrast-margin', type=float, default=0.001)
    parser.add_argument('--offset-contrast-mode', choices=('roll_random', 'zero', 'negate'), default='roll_random')
    parser.add_argument('--offset-contrast-target', choices=('full_pl', 'pRB'), default='full_pl', help='Target used by the diagnostic offset contrast metric.')
    parser.add_argument('--offset-consistency-weight', type=float, default=0.0, help='AMASS same-motion different-offset PL output consistency loss for paired offset caches.')
    parser.add_argument('--offset-consistency-target', choices=('full_pl', 'pRB'), default='full_pl')
    parser.add_argument('--offset-init-dropout-prob', type=float, default=0.0, help='Diagnostic augmentation on offset_r[18] in init_feature only.')
    parser.add_argument('--offset-init-noise-std', type=float, default=0.0, help='Gaussian noise std in meters applied to offset_r[18] in init_feature only.')
    parser.add_argument('--imu-proxy-offset-acc-weight', type=float, default=0.0, help='Weight for pRB_ddot + rJS lever term fitting raw root-frame IMU acceleration.')
    parser.add_argument('--imu-proxy-acc-scale', type=float, default=30.0, help='Acceleration scale used before SmoothL1 in the IMU proxy loss.')
    parser.add_argument('--train-gt-control-cache', default='', help='Optional canonical GTControlCache manifest for train pl_pRB_gR1_control.')
    parser.add_argument('--val-gt-control-cache', default='', help='Optional canonical GTControlCache manifest for val pl_pRB_gR1_control.')
    args = parser.parse_args()

    weights = default_weights()
    overrides = {
        'pRB': args.pRB_weight,
        'gR1': args.gR1_weight,
        'baseline_pRB': args.baseline_pRB_weight,
        'baseline_gR1': args.baseline_gR1_weight,
        'gt_control_pRB': args.gt_control_pRB_weight,
        'gt_control_gR1': args.gt_control_gR1_weight,
        'control_point_prior': args.control_point_prior_weight,
        'tail_update_prior': args.tail_update_prior_weight,
        'pRB_dot': args.pRB_dot_weight,
        'pRB_ddot': args.pRB_ddot_weight,
        'pRB_ddot_smooth': args.pRB_ddot_smooth_weight,
        'gR1_dot': args.gR1_dot_weight,
        'gR1_ddot': args.gR1_ddot_weight,
        'gR_smooth': args.gR_smooth_weight,
        'bone6d': args.bone6d_weight,
        'bone_geo': args.bone_geo_weight,
        'gt_control_bone6d': args.gt_control_bone6d_weight,
        'gt_control_bone_geo': args.gt_control_bone_geo_weight,
        'bone6d_dot': args.bone6d_dot_weight,
        'bone6d_ddot': args.bone6d_ddot_weight,
        'bone_control_point_prior': args.bone_control_point_prior_weight,
        'bone_tail_update_prior': args.bone_tail_update_prior_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            weights[key] = value
    if args.ik_distill_weight is not None:
        weights['ik1_pRJ'] = args.ik_distill_weight
        weights['ik1_gR2'] = args.ik_distill_weight
        weights['ik2_r6d'] = args.ik_distill_weight

    train_records, train_manifest = load_pl_curve_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_pl_curve_records(args.val_cache, max_sequences=args.max_val_sequences)
    train_control_attach = attach_pl_target_controls(train_records, args.train_gt_control_cache)
    val_control_attach = attach_pl_target_controls(val_records, args.val_gt_control_cache)
    pl_cache_types = ('pl_curve_cache_v1', 'pl_curve_cache_v2', 'pl_curve_cache_v3', 'pl_curve_cache_v4', 'pl_curve_joint_leaf_acc_cache_v1')
    using_pl_cache = bool(train_manifest and train_manifest.get('type') in pl_cache_types)
    if args.input_size <= 0:
        if train_manifest and train_manifest.get('feature_dim'):
            args.input_size = int(train_manifest['feature_dim'])
        elif args.model_variant == 'offset_aware':
            args.input_size = PL_OFFSET_AWARE_INPUT_SIZE
        else:
            args.input_size = PL_LEGACY_INPUT_SIZE
    if args.model_variant == 'offset_aware' and args.input_size != PL_OFFSET_AWARE_INPUT_SIZE:
        raise RuntimeError(f'offset_aware model requires input_size={PL_OFFSET_AWARE_INPUT_SIZE}, got {args.input_size}.')
    if args.bone_aux_dim not in (0, PL_BONE_AUX_DIM):
        raise RuntimeError(f'Unsupported bone_aux_dim={args.bone_aux_dim}; expected 0 or {PL_BONE_AUX_DIM}.')
    if args.init_size != 18 and train_manifest and train_manifest.get('type') not in ('pl_curve_cache_v2', 'pl_curve_cache_v3', 'pl_curve_cache_v4', 'pl_curve_joint_leaf_acc_cache_v1'):
        raise RuntimeError(f'init_size={args.init_size} requires pl_curve_cache_v2/v3/v4 or pl_curve_joint_leaf_acc_cache_v1 with pl_init_feature.')
    if using_pl_cache and args.batch_size > 1:
        train_records = [record for record in train_records if record['pl_input'].shape[0] >= args.window]
        if not train_records:
            raise RuntimeError(f'No PL cache training sequence has at least window={args.window} frames.')
    needs_bone_body_model = args.bone_aux_dim > 0 and any('bone6d_target' not in record for record in train_records + val_records)
    gpnet = None
    body_model = None
    if (not using_pl_cache) or (not args.disable_ik_distill) or needs_bone_body_model:
        gpnet = build_gpnet().eval().to(DEVICE)
        for parameter in gpnet.parameters():
            parameter.requires_grad_(False)
    if not using_pl_cache or needs_bone_body_model:
        body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    model = build_pl_curve_model(vars(args)).to(DEVICE)
    pair_lookup = build_pair_lookup(train_records)
    init_checkpoint_load = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        if checkpoint.get('model_type') != 'pl_curve_v1':
            raise RuntimeError(f'Unsupported init checkpoint model_type={checkpoint.get("model_type")}')
        init_checkpoint_load = load_partial_checkpoint(model, checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    log_path = output_dir / 'train_log.jsonl'
    best_loss = float('inf')
    best_epoch = 0
    step = 0
    stale_epochs = 0
    stopped_early = False
    stop_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_rows = []
        if using_pl_cache and args.batch_size > 1:
            iterable = range(0, len(train_records), args.batch_size)
        else:
            iterable = range(len(train_records))
        for seq_idx, batch_start in enumerate(iterable, start=1):
            step += 1
            if using_pl_cache and args.batch_size > 1:
                batch_records = train_records[batch_start:batch_start + args.batch_size]
                starts = []
                for offset, source_record in enumerate(batch_records):
                    seq_len = source_record['pl_input'].shape[0]
                    max_start = max(0, seq_len - args.window)
                    starts.append((step + offset) % (max_start + 1) if max_start > 0 else 0)
                record = make_batch(batch_records, starts, args.window)
            else:
                source_record = train_records[batch_start]
                seq_len = source_record['pl_input'].shape[0] if 'pl_input' in source_record else source_record['pose_gt'].shape[0]
                max_start = max(0, seq_len - args.window)
                start = step % (max_start + 1) if max_start > 0 else 0
                record = slice_record(source_record, start, args.window)
            init_feature = record_init_feature(record)
            if init_feature is not None:
                init_feature = augment_offset_init(
                    init_feature.to(DEVICE),
                    dropout_prob=args.offset_init_dropout_prob,
                    noise_std=args.offset_init_noise_std,
                )
            loss, components = run_sequence_with_init(
                model,
                gpnet,
                record,
                body_model,
                weights,
                init_feature,
                train_ik_distill=not args.disable_ik_distill,
                imu_proxy_offset_acc_weight=args.imu_proxy_offset_acc_weight,
                imu_proxy_acc_scale=args.imu_proxy_acc_scale,
            )
            if args.offset_contrast_weight > 0.0:
                contrast, contrast_components = offset_contrast_loss(
                    model,
                    record['pl_input'].to(DEVICE),
                    record['pl_base'].to(DEVICE),
                    normalize_gravity(record['pl_target'].float()).to(DEVICE),
                    init_feature.to(DEVICE) if init_feature is not None else None,
                    step=step,
                    mode=args.offset_contrast_mode,
                    margin=args.offset_contrast_margin,
                    target_mode=args.offset_contrast_target,
                )
                loss = loss + contrast * float(args.offset_contrast_weight)
                components.update(contrast_components)
            if args.offset_consistency_weight > 0.0:
                consistency, consistency_components = paired_offset_consistency_loss(
                    model,
                    record,
                    pair_lookup,
                    start=starts[0] if using_pl_cache and args.batch_size > 1 else start,
                    length=args.window,
                    step=step,
                    target_mode=args.offset_consistency_target,
                )
                loss = loss + consistency * float(args.offset_consistency_weight)
                components.update(consistency_components)
            if not torch.isfinite(loss):
                component_status = {
                    key: bool(torch.isfinite(value).all())
                    for key, value in components.items()
                }
                raise RuntimeError(f'Non-finite loss at {record["name"]}; components_finite={component_status}.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            row = {
                'epoch': epoch,
                'step': step,
                'seq_idx': seq_idx,
                'seq_name': record['name'],
                'loss': float(loss.detach()),
            }
            row.update({key: float(value.detach()) for key, value in components.items()})
            train_rows.append(row)
        train_loss = average(train_rows)
        validation = eval_loss(
            model,
            gpnet,
            val_records,
            body_model,
            weights,
            max_sequences=args.max_val_sequences,
            train_ik_distill=not args.disable_ik_distill,
            val_window_length=args.val_window_length,
            val_window_seed=epoch,
        )
        weighted_val_loss = validation['loss'].get('loss', float('inf'))
        val_loss = checkpoint_selection_value(validation, args)
        improved = (val_loss < best_loss) if best_loss == float('inf') else ((best_loss - val_loss) > args.early_stop_min_delta)
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(output_dir / 'best_loss.pt', model, optimizer, args, epoch, step, val_loss, weights)
        else:
            stale_epochs += 1
        save_checkpoint(output_dir / 'last.pt', model, optimizer, args, epoch, step, val_loss, weights)
        epoch_row = {
            'epoch': epoch,
            'step': step,
            'train_loss': train_loss,
            'validation': validation,
            'weighted_val_loss': weighted_val_loss,
            'selection_metric': args.selection_metric,
            'selection_value': val_loss,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'improved': improved,
            'stale_epochs': stale_epochs,
        }
        history.append(epoch_row)
        with log_path.open('a') as f:
            f.write(json.dumps(epoch_row) + '\n')
        print(json.dumps({
            'epoch': epoch,
            'train_loss': train_loss.get('loss'),
            'val_loss': val_loss,
            'weighted_val_loss': weighted_val_loss,
            'best_loss': best_loss,
            'stale_epochs': stale_epochs,
        }, indent=2))
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            stopped_early = True
            stop_epoch = epoch
            break
    result = {
        'experiment_name': args.experiment_name,
        'status': 'early_stopped' if stopped_early else 'ok',
        'config': vars(args),
        'weights': weights,
        'train_cache_manifest': train_manifest,
        'val_cache_manifest': val_manifest,
        'train_gt_control_cache': train_control_attach,
        'val_gt_control_cache': val_control_attach,
        'num_train_sequences': len(train_records),
        'num_val_sequences': len(val_records),
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'selection_metric': args.selection_metric,
        'stopped_early': stopped_early,
        'stop_epoch': stop_epoch,
        'early_stop_min_delta': args.early_stop_min_delta,
        'early_stop_patience': args.early_stop_patience,
        'init_checkpoint_load': init_checkpoint_load,
        'history': history,
    }
    (output_dir / 'train_result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
