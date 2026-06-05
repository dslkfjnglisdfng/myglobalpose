import argparse
import json
import random
import time
from pathlib import Path

import torch

import articulate as art
from l4_q75_utils import prephysics_feature, prephysics_feature_dim, q75_to_pose_tran
from l4_tail_update_qstate import StreamingTailUpdateQState
from l4_velocity_losses import finite_difference_translation_velocity, velocity_residual_losses


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
METRIC_NAMES = (
    'L SIP Err (deg)',
    'L Angle Err (deg)',
    'L Joint Err (cm)',
    'L Vertex Err (cm)',
    'G SIP Err (deg)',
    'G Angle Err (deg)',
    'G Joint Err (cm)',
    'G Vertex Err (cm)',
    'Root Jitter (km/s^3)',
    'Joint Jitter (km/s^3)',
)
_BODY_MODEL = None


def _load_runtime_eval_modules():
    from net import GPNet
    from test import MotionEvaluator
    return GPNet, MotionEvaluator


def load_cache_files(cache_path):
    path = Path(cache_path)
    if path.suffix == '.json':
        manifest = json.loads(path.read_text())
        return [Path(item['path']) for item in manifest['cache_files']], manifest
    return [path], None


def load_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    records = []
    required = ('q75_prephysics', 'v_root_vr', 'stationary_prob', 'q75_gt', 'aM', 'wM', 'RMB')
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        missing = [key for key in required if key not in data or not data[key]]
        if missing:
            raise KeyError(f'{cache_file} missing required fields: {missing}')
        for seq_idx, name in enumerate(data['name']):
            if 'pose_gt' in data and data['pose_gt']:
                pose_gt = data['pose_gt'][seq_idx].float()
                tran_gt = data['tran_gt'][seq_idx].float()
            else:
                pose_gt, tran_gt = q75_to_pose_tran(data['q75_gt'][seq_idx].float())
            record = {
                'name': name,
                'q75_prephysics': data['q75_prephysics'][seq_idx].float(),
                'v_root_vr': data['v_root_vr'][seq_idx].float(),
                'stationary_prob': data['stationary_prob'][seq_idx].float(),
                'q75_gt': data['q75_gt'][seq_idx].float(),
                'pose_gt': pose_gt.float(),
                'tran_gt': tran_gt.float(),
                'aM': data['aM'][seq_idx].float(),
                'wM': data['wM'][seq_idx].float(),
                'RMB': data['RMB'][seq_idx].float(),
            }
            if 'pose_prephysics' in data and data['pose_prephysics']:
                record['pose_prephysics'] = data['pose_prephysics'][seq_idx].float()
            if 'offset_r' in data and data['offset_r']:
                record['offset_r'] = data['offset_r'][seq_idx].float()
            for key in ('l4_aM', 'l4_wM', 'l4_RMB'):
                if key in data and data[key]:
                    record[key] = data[key][seq_idx].float()
            if 'pose_baseline' in data and data['pose_baseline']:
                record['pose_baseline'] = data['pose_baseline'][seq_idx].float()
                record['tran_baseline'] = data['tran_baseline'][seq_idx].float()
            records.append(record)
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def split_records(records, val_ratio, seed, max_val_sequences):
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 1
    if max_val_sequences:
        n_val = min(n_val, max_val_sequences)
    val_ids = set(indices[:n_val])
    train = [record for idx, record in enumerate(records) if idx not in val_ids]
    val = [record for idx, record in enumerate(records) if idx in val_ids]
    return train or val, val


def rotation_geodesic(R_pred, R_target, eps=1e-6):
    rel = R_pred.transpose(-1, -2).matmul(R_target)
    trace = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


def q75_residual(q_pred, q_base):
    rot = torch.atan2(torch.sin(q_pred[..., 3:] - q_base[..., 3:]), torch.cos(q_pred[..., 3:] - q_base[..., 3:]))
    return torch.cat((q_pred[..., :3] - q_base[..., :3], rot), dim=-1)


def finite_diff(x, order):
    if order == 1:
        return x[1:] - x[:-1]
    if order == 2:
        return x[2:] - 2.0 * x[1:-1] + x[:-2]
    if order == 3:
        return x[3:] - 3.0 * x[2:-1] + 3.0 * x[1:-2] - x[:-3]
    raise ValueError(order)


def root_relative_joints(pose):
    global _BODY_MODEL
    if _BODY_MODEL is None:
        _BODY_MODEL = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    joints = _BODY_MODEL.forward_kinematics(pose.to(DEVICE))[1]
    return joints - joints[:, :1]


def selected_imu_fields(record, model):
    prefix = getattr(model, 'l4_imu_field_prefix', 'auto')
    if prefix == 'original':
        return record['aM'], record['wM'], record['RMB']
    if prefix == 'l4':
        missing = [key for key in ('l4_aM', 'l4_wM', 'l4_RMB') if key not in record]
        if missing:
            raise KeyError(f'--l4-imu-field-prefix l4 requires fields missing from record {record.get("name")}: {missing}')
        return record['l4_aM'], record['l4_wM'], record['l4_RMB']
    if prefix == 'auto':
        return (
            record.get('l4_aM', record['aM']),
            record.get('l4_wM', record['wM']),
            record.get('l4_RMB', record['RMB']),
        )
    raise ValueError(f'Unsupported l4_imu_field_prefix: {prefix}')


def firstframe_init_feature(model, record):
    if getattr(model, 'rnn_init_mode', 'none') != 'offset_firstframe':
        return None
    pose_base = record.get('pose_prephysics')
    pose0 = None if pose_base is None else pose_base[0]
    a_seq, w_seq, R_seq = selected_imu_fields(record, model)
    a0 = a_seq[0]
    w0 = w_seq[0]
    R0 = R_seq[0]
    return prephysics_feature(
        record['q75_prephysics'][0].detach().cpu(),
        a0,
        w0,
        R0,
        pose=pose0,
        pose_input_mode=getattr(model, 'pose_input_mode', 'euler_q75'),
        euler_seq=getattr(model, 'euler_seq', 'XYZ'),
    )


def run_cached_sequence(model, record):
    model.reset_stream(record.get('offset_r'), firstframe_init_feature(model, record))
    qs = []
    q_residuals = []
    new_norms = []
    tail_norms = []
    v_refined = []
    delta_vs = []
    for frame_idx in range(record['q75_prephysics'].shape[0]):
        q_base = record['q75_prephysics'][frame_idx].to(DEVICE)
        pose_base = record.get('pose_prephysics')
        pose_base = None if pose_base is None else pose_base[frame_idx]
        a_seq, w_seq, R_seq = selected_imu_fields(record, model)
        a_t = a_seq[frame_idx]
        w_t = w_seq[frame_idx]
        R_t = R_seq[frame_idx]
        feature = prephysics_feature(
            q_base.detach().cpu(),
            a_t,
            w_t,
            R_t,
            pose=pose_base,
            pose_input_mode=getattr(model, 'pose_input_mode', 'euler_q75'),
            euler_seq=getattr(model, 'euler_seq', 'XYZ'),
        ).to(DEVICE)
        q_result = model.step(feature, q_base)
        v_result = model.refine_velocity(
            record['v_root_vr'][frame_idx].to(DEVICE),
            record['stationary_prob'][frame_idx].to(DEVICE),
        )
        qs.append(q_result['q_t'][0])
        q_residuals.append(q_result['residual_t'][0])
        new_norms.append(q_result['new_delta_norm'])
        tail_norms.append(q_result['tail_delta_norm'])
        v_refined.append(v_result['v_root_refined'][0])
        delta_vs.append(v_result['delta_v_root'][0])
    return {
        'q_pred': torch.stack(qs),
        'q_residual': torch.stack(q_residuals),
        'new_delta_norm': torch.stack(new_norms).mean(),
        'tail_delta_norm': torch.stack(tail_norms).mean(),
        'v_refined': torch.stack(v_refined),
        'delta_v': torch.stack(delta_vs),
    }


def pose_velocity_loss(model_output, record, weights):
    q_pred = model_output['q_pred']
    q_base = record['q75_prephysics'].to(DEVICE)
    q_gt = record['q75_gt'].to(DEVICE)
    pose_pred, _ = q75_to_pose_tran(q_pred)
    pose_gt = record['pose_gt'].to(DEVICE)
    pose_base, _ = q75_to_pose_tran(q_base)
    pose_pred = pose_pred.to(DEVICE)
    pose_base = pose_base.to(DEVICE)
    geo = rotation_geodesic(pose_pred, pose_gt)
    base_geo = rotation_geodesic(pose_pred, pose_base)
    q_residual = q75_residual(q_pred, q_base)
    velocity_total, velocity_components, _ = velocity_residual_losses(
        model_output['v_refined'],
        finite_difference_translation_velocity(record['tran_gt'].to(DEVICE)),
        record['v_root_vr'].to(DEVICE),
        model_output['delta_v'],
    )
    losses = {
        'pose_geodesic': geo.mean(),
        'pose_geodesic_root': geo[:, 0].mean().detach(),
        'pose_geodesic_body': geo[:, 1:].mean().detach(),
        'q_body': torch.nn.functional.smooth_l1_loss(q_pred[:, 6:], q_gt[:, 6:]),
        'q_root_ori': torch.nn.functional.smooth_l1_loss(q_pred[:, 3:6], q_gt[:, 3:6]),
        'baseline_body': base_geo[:, 1:].mean(),
        'baseline_root_ori': base_geo[:, 0].mean(),
        'residual_prior': q_residual.square().mean(),
        'tail_update_prior': model_output['tail_delta_norm'],
        'root_velocity': velocity_components['root_velocity'],
        'baseline_velocity': velocity_components['baseline_velocity'],
        'velocity_smooth': velocity_components['velocity_smooth'],
    }
    q_res = q75_residual(q_pred, q_base)
    if q_pred.shape[0] >= 2:
        losses['qdot'] = torch.nn.functional.smooth_l1_loss(finite_diff(q_pred, 1), finite_diff(q_gt, 1))
        losses['edge_q'] = q_res[0].square().mean()
        losses['edge_qdot'] = finite_diff(q_res, 1)[0].square().mean()
    else:
        losses['qdot'] = q_pred.new_zeros(())
        losses['edge_q'] = q_pred.new_zeros(())
        losses['edge_qdot'] = q_pred.new_zeros(())
    if q_pred.shape[0] >= 3:
        losses['qddot'] = torch.nn.functional.smooth_l1_loss(finite_diff(q_pred, 2), finite_diff(q_gt, 2))
        losses['edge_qddot'] = finite_diff(q_res, 2)[0].square().mean()
    else:
        losses['qddot'] = q_pred.new_zeros(())
        losses['edge_qddot'] = q_pred.new_zeros(())
    if q_pred.shape[0] >= 4:
        losses['jerk_smooth'] = finite_diff(q_pred, 3).square().mean()
    else:
        losses['jerk_smooth'] = q_pred.new_zeros(())
    joint_pred = root_relative_joints(pose_pred)
    joint_gt = root_relative_joints(pose_gt)
    losses['fk_joint_rootrel'] = torch.nn.functional.smooth_l1_loss(joint_pred, joint_gt)
    total = q_pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * weight
    return total, losses


def train_epoch(model, records, optimizer, weights, grad_clip):
    model.train()
    totals = {'total': 0.0}
    for record in records:
        output = run_cached_sequence(model, record)
        loss, components = pose_velocity_loss(output, record, weights)
        optimizer.zero_grad()
        loss.backward()
        grad_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not grad_ok:
            raise RuntimeError('Non-finite gradient detected.')
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        totals['total'] += float(loss.detach())
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        totals['q_residual_norm_mean'] = totals.get('q_residual_norm_mean', 0.0) + float(output['q_residual'].norm(dim=-1).mean().detach())
        totals['q_residual_norm_max'] = totals.get('q_residual_norm_max', 0.0) + float(output['q_residual'].norm(dim=-1).max().detach())
        totals['delta_v_root_norm_mean'] = totals.get('delta_v_root_norm_mean', 0.0) + float(output['delta_v'].norm(dim=-1).mean().detach())
        totals['delta_v_root_norm_max'] = totals.get('delta_v_root_norm_max', 0.0) + float(output['delta_v'].norm(dim=-1).max().detach())
        totals['new_delta_norm_mean'] = totals.get('new_delta_norm_mean', 0.0) + float(output['new_delta_norm'].detach())
        totals['tail_update_norm_mean'] = totals.get('tail_update_norm_mean', 0.0) + float(output['tail_delta_norm'].detach())
    return {key: value / max(1, len(records)) for key, value in totals.items()}


@torch.no_grad()
def evaluate_cache_loss(model, records, weights, max_eval_sequences=0):
    model.eval()
    totals = {'total': 0.0}
    selected = records[:max_eval_sequences] if max_eval_sequences else records
    for record in selected:
        output = run_cached_sequence(model, record)
        loss, components = pose_velocity_loss(output, record, weights)
        totals['total'] += float(loss.detach())
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        q_norm = output['q_residual'].norm(dim=-1)
        dv_norm = output['delta_v'].norm(dim=-1)
        totals['q_residual_norm_mean'] = totals.get('q_residual_norm_mean', 0.0) + float(q_norm.mean().detach())
        totals['q_residual_norm_max'] = max(totals.get('q_residual_norm_max', 0.0), float(q_norm.max().detach()))
        totals['delta_v_root_norm_mean'] = totals.get('delta_v_root_norm_mean', 0.0) + float(dv_norm.mean().detach())
        totals['delta_v_root_norm_max'] = max(totals.get('delta_v_root_norm_max', 0.0), float(dv_norm.max().detach()))
        totals['new_delta_norm_mean'] = totals.get('new_delta_norm_mean', 0.0) + float(output['new_delta_norm'].detach())
        totals['tail_update_norm_mean'] = totals.get('tail_update_norm_mean', 0.0) + float(output['tail_delta_norm'].detach())
    n = max(1, len(selected))
    averaged = {
        key: (value / n if key not in ('q_residual_norm_max', 'delta_v_root_norm_max') else value)
        for key, value in totals.items()
    }
    return {
        'num_sequences': len(selected),
        'loss': averaged,
        'q_residual_norm_mean': averaged.get('q_residual_norm_mean', 0.0),
        'q_residual_norm_max': averaged.get('q_residual_norm_max', 0.0),
        'delta_v_root_norm_mean': averaged.get('delta_v_root_norm_mean', 0.0),
        'delta_v_root_norm_max': averaged.get('delta_v_root_norm_max', 0.0),
        'tail_update_norm_mean': averaged.get('tail_update_norm_mean', 0.0),
        'new_delta_norm_mean': averaged.get('new_delta_norm_mean', 0.0),
    }


def metric_to_dict(metric):
    return {METRIC_NAMES[i]: {'mean': float(metric[i, 0]), 'std': float(metric[i, 1])} for i in range(len(METRIC_NAMES))}


def delta_metrics(model_metrics, baseline_metrics):
    return {
        name: {
            'mean': model_metrics[name]['mean'] - baseline_metrics[name]['mean'],
            'std': model_metrics[name]['std'] - baseline_metrics[name]['std'],
        }
        for name in METRIC_NAMES
    }


@torch.no_grad()
def get_or_run_baseline(record):
    if 'pose_baseline' in record and 'tran_baseline' in record:
        return record['pose_baseline'], record['tran_baseline']
    GPNet, _ = _load_runtime_eval_modules()
    net = GPNet(enable_l4_prephysics=False).eval().to(DEVICE)
    net.rnn_initialize(record['pose_gt'][0])
    pose_baseline = torch.zeros_like(record['pose_gt'])
    tran_baseline = torch.zeros_like(record['tran_gt'])
    for frame_idx in range(record['pose_gt'].shape[0]):
        pose_baseline[frame_idx], tran_baseline[frame_idx] = net.forward_frame(
            record['aM'][frame_idx].to(DEVICE),
            record['wM'][frame_idx].to(DEVICE),
            record['RMB'][frame_idx].to(DEVICE),
        )
    record['pose_baseline'] = pose_baseline.cpu()
    record['tran_baseline'] = tran_baseline.cpu()
    return record['pose_baseline'], record['tran_baseline']


@torch.no_grad()
def evaluate_physics(model, records, max_eval_sequences=0):
    GPNet, MotionEvaluator = _load_runtime_eval_modules()
    model.eval()
    evaluator = MotionEvaluator()
    rows = []
    selected = records[:max_eval_sequences] if max_eval_sequences else records
    for record in selected:
        net = GPNet(enable_l4_prephysics=True, l4_prephysics_module=model).eval().to(DEVICE)
        net.rnn_initialize(record['pose_gt'][0])
        if 'offset_r' in record and getattr(model, 'offset_conditioning', 'none') == 'hidden_init':
            model.reset_stream(record['offset_r'], firstframe_init_feature(model, record))
        pose_model = torch.zeros_like(record['pose_gt'])
        tran_model = torch.zeros_like(record['tran_gt'])
        delta_v_norms = []
        q_residual_norms = []
        tail_update_norms = []
        l4_a_seq, l4_w_seq, l4_R_seq = selected_imu_fields(record, model)
        for frame_idx in range(record['pose_gt'].shape[0]):
            pose_model[frame_idx], tran_model[frame_idx] = net.forward_frame(
                record['aM'][frame_idx].to(DEVICE),
                record['wM'][frame_idx].to(DEVICE),
                record['RMB'][frame_idx].to(DEVICE),
                l4_a=l4_a_seq[frame_idx].to(DEVICE),
                l4_w=l4_w_seq[frame_idx].to(DEVICE),
                l4_R=l4_R_seq[frame_idx].to(DEVICE),
            )
            debug = getattr(net, 'last_l4_prephysics_debug', {})
            delta_v_norms.append(float(debug.get('delta_v_root_norm', 0.0)))
            residual = debug.get('residual')
            q_residual_norms.append(float(residual.norm()) if residual is not None else 0.0)
            tail_update_norms.append(float(debug.get('tail_delta_norm', 0.0)))
        pose_baseline, tran_baseline = get_or_run_baseline(record)
        baseline_metric = evaluator(
            pose_baseline.to(DEVICE),
            record['pose_gt'].to(DEVICE),
            tran_baseline.to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        model_metric = evaluator(
            pose_model.to(DEVICE),
            record['pose_gt'].to(DEVICE),
            tran_model.to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        baseline = metric_to_dict(baseline_metric)
        model_dict = metric_to_dict(model_metric)
        rows.append({
            'name': record['name'],
            'baseline_metrics': baseline,
            'model_metrics': model_dict,
            'delta_metrics': delta_metrics(model_dict, baseline),
            'delta_v_root_norm_mean': sum(delta_v_norms) / max(1, len(delta_v_norms)),
            'delta_v_root_norm_max': max(delta_v_norms) if delta_v_norms else 0.0,
            'q_residual_norm_mean': sum(q_residual_norms) / max(1, len(q_residual_norms)),
            'q_residual_norm_max': max(q_residual_norms) if q_residual_norms else 0.0,
            'tail_update_norm_mean': sum(tail_update_norms) / max(1, len(tail_update_norms)),
            'tail_update_norm_max': max(tail_update_norms) if tail_update_norms else 0.0,
        })
    return rows


def aggregate_rows(rows, key):
    out = {}
    for name in METRIC_NAMES:
        means = [row[key][name]['mean'] for row in rows]
        stds = [row[key][name]['std'] for row in rows]
        out[name] = {
            'mean': sum(means) / max(1, len(means)),
            'std': sum(stds) / max(1, len(stds)),
        }
    return out


def aggregate_eval(rows):
    baseline = aggregate_rows(rows, 'baseline_metrics')
    model = aggregate_rows(rows, 'model_metrics')
    delta = delta_metrics(model, baseline)
    return {
        'num_sequences': len(rows),
        'baseline_metrics': baseline,
        'model_metrics': model,
        'delta_metrics': delta,
        'delta_v_root_norm_mean': sum(row['delta_v_root_norm_mean'] for row in rows) / max(1, len(rows)),
        'delta_v_root_norm_max': max((row['delta_v_root_norm_max'] for row in rows), default=0.0),
        'q_residual_norm_mean': sum(row['q_residual_norm_mean'] for row in rows) / max(1, len(rows)),
        'q_residual_norm_max': max((row['q_residual_norm_max'] for row in rows), default=0.0),
        'tail_update_norm_mean': sum(row['tail_update_norm_mean'] for row in rows) / max(1, len(rows)),
        'tail_update_norm_max': max((row['tail_update_norm_max'] for row in rows), default=0.0),
    }


def score_for_checkpoint(agg):
    # Lower is better. This is based only on final MotionEvaluator metrics.
    return (
        agg['model_metrics']['L SIP Err (deg)']['mean']
        + agg['model_metrics']['L Angle Err (deg)']['mean']
        + agg['model_metrics']['G SIP Err (deg)']['mean']
        + agg['model_metrics']['G Angle Err (deg)']['mean']
        + 0.1 * agg['model_metrics']['L Joint Err (cm)']['mean']
        + 0.1 * agg['model_metrics']['G Joint Err (cm)']['mean']
        + 0.01 * agg['model_metrics']['Joint Jitter (km/s^3)']['mean']
    )


def score_for_cache_checkpoint(cache_eval, metric):
    loss = cache_eval['loss']
    if metric not in loss:
        raise KeyError(f'Cache validation metric {metric!r} not found. Available: {sorted(loss)}')
    return loss[metric]


def load_compatible_state(model, checkpoint_path, allow_partial=False):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state = checkpoint['model_state_dict']
    if not allow_partial:
        model.load_state_dict(state)
        return {
            'path': checkpoint_path,
            'mode': 'strict',
            'loaded': sorted(state),
            'skipped': [],
        }
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append({
                'key': key,
                'checkpoint_shape': list(value.shape),
                'model_shape': list(current[key].shape) if key in current else None,
            })
    current.update(compatible)
    model.load_state_dict(current)
    return {
        'path': checkpoint_path,
        'mode': 'partial_shape_compatible',
        'loaded': sorted(compatible),
        'skipped': skipped,
    }


def catastrophic_failures(agg, delta_v_root_max_threshold=1.0):
    failures = {}
    baseline = agg['baseline_metrics']
    model = agg['model_metrics']
    delta = agg['delta_metrics']
    if model['Joint Jitter (km/s^3)']['mean'] > baseline['Joint Jitter (km/s^3)']['mean'] * 5.0:
        failures['Joint Jitter (km/s^3)'] = {
            'baseline': baseline['Joint Jitter (km/s^3)']['mean'],
            'model': model['Joint Jitter (km/s^3)']['mean'],
            'threshold': baseline['Joint Jitter (km/s^3)']['mean'] * 5.0,
        }
    if model['Root Jitter (km/s^3)']['mean'] > baseline['Root Jitter (km/s^3)']['mean'] * 5.0:
        failures['Root Jitter (km/s^3)'] = {
            'baseline': baseline['Root Jitter (km/s^3)']['mean'],
            'model': model['Root Jitter (km/s^3)']['mean'],
            'threshold': baseline['Root Jitter (km/s^3)']['mean'] * 5.0,
        }
    for name in ('L Angle Err (deg)', 'G Angle Err (deg)'):
        if delta[name]['mean'] > 5.0:
            failures[name] = {
                'delta': delta[name]['mean'],
                'threshold_delta': 5.0,
            }
    if agg['delta_v_root_norm_max'] > delta_v_root_max_threshold:
        failures['delta_v_root_norm_max'] = {
            'value': agg['delta_v_root_norm_max'],
            'threshold': delta_v_root_max_threshold,
        }
    return failures


def cache_catastrophic_failures(cache_eval, delta_v_root_max_threshold=1.0, q_residual_max_threshold=10.0):
    failures = {}
    for key, value in cache_eval['loss'].items():
        if not torch.isfinite(torch.tensor(value)):
            failures[f'loss.{key}'] = {'value': value}
    if cache_eval['delta_v_root_norm_max'] > delta_v_root_max_threshold:
        failures['delta_v_root_norm_max'] = {
            'value': cache_eval['delta_v_root_norm_max'],
            'threshold': delta_v_root_max_threshold,
        }
    if cache_eval['q_residual_norm_max'] > q_residual_max_threshold:
        failures['q_residual_norm_max'] = {
            'value': cache_eval['q_residual_norm_max'],
            'threshold': q_residual_max_threshold,
        }
    return failures


def unsafe_metric_deltas(agg, sip_angle_tol=0.0, joint_mesh_tol=0.0):
    d = agg['delta_metrics']
    checks = {
        'L SIP Err (deg)': sip_angle_tol,
        'G SIP Err (deg)': sip_angle_tol,
        'L Angle Err (deg)': sip_angle_tol,
        'G Angle Err (deg)': sip_angle_tol,
        'L Joint Err (cm)': joint_mesh_tol,
        'G Joint Err (cm)': joint_mesh_tol,
        'L Vertex Err (cm)': joint_mesh_tol,
        'G Vertex Err (cm)': joint_mesh_tol,
    }
    return {
        name: d[name]['mean']
        for name, tol in checks.items()
        if d[name]['mean'] > tol
    }


def significant_regression(agg, sip_angle_tol=0.0, joint_mesh_tol=0.0):
    return bool(unsafe_metric_deltas(agg, sip_angle_tol, joint_mesh_tol))


def main():
    parser = argparse.ArgumentParser(description='Bounded diverse7 short run for L4 pose+velocity pre-physics module.')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--val-cache', default='', help='Optional explicit validation cache. When set, --val-ratio is ignored.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=512, help='Accepted for run config compatibility; this trainer processes full sequences.')
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--max-val-sequences', type=int, default=7)
    parser.add_argument('--max-eval-sequences', type=int, default=7)
    parser.add_argument('--validation-mode', choices=('physics', 'cache_loss', 'train_loss'), default='physics')
    parser.add_argument('--cache-checkpoint-metric', default='total')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1, help='Accepted for run config compatibility; current streaming head uses one GRUCell.')
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.02)
    parser.add_argument('--velocity-residual-scale', type=float, default=0.02)
    parser.add_argument('--pose-input-mode', choices=('euler_q75', 'rot6d'), default='euler_q75')
    parser.add_argument('--offset-conditioning', choices=('none', 'hidden_init'), default='none')
    parser.add_argument('--rnn-init-mode', choices=('none', 'offset_only', 'offset_firstframe'), default='')
    parser.add_argument('--offset-init-scale', type=float, default=0.1)
    parser.add_argument('--paired-offset-training', action='store_true', help='Reserved K2 flag; first implementation keeps mixed-view records as ordinary samples.')
    parser.add_argument('--pair-consistency-weight', type=float, default=0.0, help='Reserved K2 consistency loss weight. Must stay 0.0 until paired loss is implemented.')
    parser.add_argument('--allow-partial-init', action='store_true', help='Load only shape-compatible tensors from --init-checkpoint.')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init-checkpoint', default='', help='Load model weights before training without resuming optimizer or epoch.')
    parser.add_argument('--resume', default='')
    parser.add_argument('--sip-angle-regression-tol', type=float, default=0.0)
    parser.add_argument('--joint-mesh-regression-tol', type=float, default=0.0)
    parser.add_argument('--early-stop-patience', type=int, default=15)
    parser.add_argument('--delta-v-root-max-catastrophic', type=float, default=1.0)
    parser.add_argument('--q-residual-max-catastrophic', type=float, default=10.0)
    parser.add_argument('--strict-safety-gate', action='store_true')
    parser.add_argument('--no-strict-safety-gate', action='store_true')
    parser.add_argument('--allow-unsafe-best', action='store_true')
    parser.add_argument('--disable-root-velocity-loss', action='store_true', help='For DIP fine-tuning: do not supervise root velocity from unreliable GT translation.')
    parser.add_argument('--disable-root-translation-loss', action='store_true', help='For DIP fine-tuning: keep root translation-related q losses disabled.')
    args = parser.parse_args()
    if args.tail_length != 4:
        raise ValueError('Only tail_length=4 is approved for this migration.')
    if args.early_stop_patience < 1:
        raise ValueError('--early-stop-patience must be >= 1.')
    if args.pair_consistency_weight != 0.0:
        raise ValueError('Pair consistency loss is reserved but not implemented; keep --pair-consistency-weight 0.0.')
    strict_safety_gate = args.strict_safety_gate or not args.no_strict_safety_gate

    records, manifest = load_records(args.cache, max_sequences=args.max_sequences)
    if args.validation_mode == 'train_loss':
        val_manifest = None
        train_records, val_records = records, []
    elif args.val_cache:
        val_records, val_manifest = load_records(args.val_cache)
        if args.max_val_sequences:
            val_records = val_records[:args.max_val_sequences]
        train_records = records
    else:
        val_manifest = None
        train_records, val_records = split_records(records, args.val_ratio, args.seed, args.max_val_sequences)
    weights = {
        'pose_geodesic': 1.0,
        'q_body': 1.0,
        'q_root_ori': 0.5,
        'baseline_body': 2.0,
        'baseline_root_ori': 5.0,
        'qdot': 0.03,
        'qddot': 0.003,
        'fk_joint_rootrel': 0.1,
        'residual_prior': 0.001,
        'tail_update_prior': 0.005,
        'edge_q': 0.01,
        'edge_qdot': 0.03,
        'edge_qddot': 0.003,
        'jerk_smooth': 1e-5,
        'root_velocity': 1.0,
        'baseline_velocity': 2.0,
        'velocity_smooth': 0.03,
    }
    if args.disable_root_velocity_loss:
        weights['root_velocity'] = 0.0
    if args.disable_root_translation_loss:
        # q75 root translation residual is frozen by the model, but zero these
        # terms for split/protocol clarity in DIP fine-tuning configs.
        weights['root_velocity'] = 0.0
    effective_rnn_init_mode = args.rnn_init_mode or ('offset_only' if args.offset_conditioning == 'hidden_init' else 'none')
    if args.rnn_init_mode and args.rnn_init_mode != 'none':
        args.offset_conditioning = 'hidden_init'
    args.effective_rnn_init_mode = effective_rnn_init_mode
    model = StreamingTailUpdateQState(
        hidden_size=args.hidden_size,
        residual_scale=args.residual_scale,
        velocity_residual_scale=args.velocity_residual_scale,
        pose_input_mode=args.pose_input_mode,
        offset_conditioning=args.offset_conditioning,
        rnn_init_mode=effective_rnn_init_mode,
        offset_init_scale=args.offset_init_scale,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'train_log.jsonl'
    result_path = output_dir / 'train_result.json'

    result = {
        'config': vars(args),
        'cache_manifest': manifest,
        'val_cache_manifest': val_manifest,
        'train_sequences': [record['name'] for record in train_records],
        'val_sequences': [record['name'] for record in val_records],
        'weights': weights,
        'pose_input_contract': {
            'pose_input_mode': args.pose_input_mode,
            'pose_input_dim': prephysics_feature_dim(args.pose_input_mode) - 90,
            'imu_feature_dim': 90,
            'model_n_input': model.n_input,
            'sensor_offset_input': args.offset_conditioning != 'none',
            'offset_conditioning': args.offset_conditioning,
            'rnn_init_mode': effective_rnn_init_mode,
            'rnn_init_input_dim': (
                18 if effective_rnn_init_mode == 'offset_only'
                else (18 + model.n_input if effective_rnn_init_mode == 'offset_firstframe' else 0)
            ),
            'rnn_init_uses_first_frame': effective_rnn_init_mode == 'offset_firstframe',
            'imu_position_offset_augmented_datasets': 'excluded',
            'imu_proxy_loss': False,
        },
        'validation_mode': args.validation_mode,
        'checkpoint_selection': (
            'validation final MotionEvaluator score'
            if args.validation_mode == 'physics'
            else (
                f'cache validation loss component: {args.cache_checkpoint_metric}'
                if args.validation_mode == 'cache_loss'
                else f'full training loss component: {args.cache_checkpoint_metric}'
            )
        ),
        'motion_evaluator_disabled_during_training': args.validation_mode in ('cache_loss', 'train_loss'),
        'requires_post_training_motion_evaluator': args.validation_mode in ('cache_loss', 'train_loss'),
        'epochs': [],
        'best': None,
        'status': 'running',
    }
    best_score = None
    start_epoch = 1
    no_improve_epochs = 0
    saw_regression = False
    if args.init_checkpoint:
        result['init_checkpoint_load'] = load_compatible_state(
            model,
            args.init_checkpoint,
            allow_partial=args.allow_partial_init,
        )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
    with log_path.open('w') as log_file:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            train_loss = train_epoch(model, train_records, optimizer, weights, args.grad_clip)
            if args.validation_mode == 'physics':
                rows = evaluate_physics(model, val_records, max_eval_sequences=args.max_eval_sequences)
                validation = aggregate_eval(rows)
                score = score_for_checkpoint(validation)
                unsafe_deltas = unsafe_metric_deltas(
                    validation,
                    sip_angle_tol=args.sip_angle_regression_tol,
                    joint_mesh_tol=args.joint_mesh_regression_tol,
                ) if strict_safety_gate else {}
                saw_regression = saw_regression or bool(unsafe_deltas)
                catastrophic = catastrophic_failures(
                    validation,
                    delta_v_root_max_threshold=args.delta_v_root_max_catastrophic,
                ) if strict_safety_gate else {}
                validation_rows = rows
            elif args.validation_mode == 'cache_loss':
                validation = evaluate_cache_loss(
                    model,
                    val_records,
                    weights,
                    max_eval_sequences=args.max_eval_sequences,
                )
                score = score_for_cache_checkpoint(validation, args.cache_checkpoint_metric)
                unsafe_deltas = {}
                catastrophic = cache_catastrophic_failures(
                    validation,
                    delta_v_root_max_threshold=args.delta_v_root_max_catastrophic,
                    q_residual_max_threshold=args.q_residual_max_catastrophic,
                ) if strict_safety_gate else {}
                validation_rows = []
            else:
                validation = {
                    'num_sequences': 0,
                    'loss': {},
                    'note': 'Validation disabled. Candidate checkpoint is selected by full training loss only.',
                }
                score = score_for_cache_checkpoint({'loss': train_loss}, args.cache_checkpoint_metric)
                unsafe_deltas = {}
                catastrophic = cache_catastrophic_failures(
                    {
                        'loss': train_loss,
                        'q_residual_norm_max': train_loss.get('q_residual_norm_max', 0.0),
                        'delta_v_root_norm_max': train_loss.get('delta_v_root_norm_max', 0.0),
                    },
                    delta_v_root_max_threshold=args.delta_v_root_max_catastrophic,
                    q_residual_max_threshold=args.q_residual_max_catastrophic,
                ) if strict_safety_gate else {}
                validation_rows = []
            improved_best = best_score is None or score < best_score
            if improved_best and not catastrophic:
                best_score = score
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            patience_stop = no_improve_epochs >= args.early_stop_patience
            early_stop = bool(catastrophic) or patience_stop
            epoch_record = {
                'epoch': epoch,
                'epoch_wall_seconds': time.time() - epoch_start,
                'train_loss': train_loss,
                'validation_mode': args.validation_mode,
                'checkpoint_metric': (
                    'motion_evaluator_score'
                    if args.validation_mode == 'physics'
                    else (
                        args.cache_checkpoint_metric
                        if args.validation_mode == 'cache_loss'
                        else f'train_loss.{args.cache_checkpoint_metric}'
                    )
                ),
                'validation': validation,
                'validation_rows': validation_rows,
                'checkpoint_score': score,
                'best_score': best_score,
                'improved_best': improved_best and not catastrophic,
                'unsafe_metric_deltas': unsafe_deltas,
                'saw_regression_so_far': saw_regression,
                'catastrophic_failures': catastrophic,
                'no_improve_epochs': no_improve_epochs,
                'patience_stop': patience_stop,
                'early_stop': early_stop,
            }
            result['epochs'].append(epoch_record)
            log_file.write(json.dumps(epoch_record) + '\n')
            log_file.flush()
            if epoch_record['improved_best']:
                result['best'] = epoch_record
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'config': vars(args), 'epoch': epoch}, output_dir / 'best.pt')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'config': vars(args), 'epoch': epoch}, output_dir / 'last.pt')
            result['status'] = 'catastrophic_stopped' if catastrophic else ('early_stopped_patience' if patience_stop else 'running')
            result_path.write_text(json.dumps(result, indent=2))
            if early_stop:
                break
    if result['status'] == 'running':
        result['status'] = 'completed'
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({'result_path': str(result_path), 'log_path': str(log_path), 'status': result['status'], 'best_epoch': result['best']['epoch'] if result['best'] else None}, indent=2))


if __name__ == '__main__':
    main()
