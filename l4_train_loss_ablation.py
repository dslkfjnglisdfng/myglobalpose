import argparse
import json
import time
from pathlib import Path

import torch

import articulate as art
from l4_q75_utils import prephysics_feature, prephysics_feature_dim, q75_to_pose_tran
from l4_tail_update_qstate import StreamingTailUpdateQState
from l4_train_diverse_short import (
    aggregate_eval,
    evaluate_physics,
    load_records,
    metric_to_dict,
    rotation_geodesic,
    score_for_checkpoint,
)
from l4_velocity_losses import finite_difference_translation_velocity, velocity_residual_losses


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = 1.0 / 60.0
GRAVITY_M = torch.tensor([0.0, -9.8, 0.0], device=DEVICE)
IMU_VERTICES = (1961, 5424, 1176, 4662, 411, 3021)
IMU_JOINTS = (18, 19, 4, 5, 15, 0)
FOOT_JOINTS = (10, 11)
FOOT_STATIONARY_PROB_COLUMNS = (1, 2)
_BODY_MODEL = None
_IMU_BODY_MODEL = None


def body_model(calc_imu=False):
    global _BODY_MODEL, _IMU_BODY_MODEL
    if calc_imu:
        if _IMU_BODY_MODEL is None:
            _IMU_BODY_MODEL = art.ParametricModel('models/SMPL_male.pkl', vert_mask=IMU_VERTICES, device=DEVICE)
        return _IMU_BODY_MODEL
    if _BODY_MODEL is None:
        _BODY_MODEL = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    return _BODY_MODEL


def default_weights(disable_root_velocity_loss=False):
    weights = {
        'pose_geodesic': 1.0,
        'q_body': 1.0,
        'q_root_ori': 0.5,
        'baseline_body': 2.0,
        'baseline_root_ori': 5.0,
        'qdot': 0.03,
        'qddot': 0.003,
        'fk_joint_rootrel': 0.1,
        'fk_joint_baseline_rootrel': 0.0,
        'residual_prior': 0.001,
        'tail_update_prior': 0.005,
        'edge_q': 0.01,
        'edge_qdot': 0.03,
        'edge_qddot': 0.003,
        'jerk_smooth': 1e-5,
        'root_velocity': 1.0,
        'baseline_velocity': 2.0,
        'velocity_smooth': 0.03,
        'contact_foot_velocity': 0.0,
        'contact_foot_height': 0.0,
        'imu_orientation_proxy': 0.0,
        'imu_acc_proxy': 0.0,
        'imu_gyro_proxy': 0.0,
    }
    if disable_root_velocity_loss:
        weights['root_velocity'] = 0.0
    return weights


def finite_diff(x, order):
    if order == 1:
        return x[1:] - x[:-1]
    if order == 2:
        return x[2:] - 2.0 * x[1:-1] + x[:-2]
    if order == 3:
        return x[3:] - 3.0 * x[2:-1] + 3.0 * x[1:-2] - x[:-3]
    raise ValueError(order)


def q75_residual(q_pred, q_base):
    rot = torch.atan2(torch.sin(q_pred[..., 3:] - q_base[..., 3:]), torch.cos(q_pred[..., 3:] - q_base[..., 3:]))
    return torch.cat((q_pred[..., :3] - q_base[..., :3], rot), dim=-1)


def root_relative_joints(pose):
    joints = body_model().forward_kinematics(pose.to(DEVICE))[1]
    return joints - joints[:, :1]


def slice_record(record, start, length):
    seq_len = record['q75_prephysics'].shape[0]
    if length <= 0 or seq_len <= length:
        return record
    start = min(max(0, start), seq_len - length)
    end = start + length
    sliced = {}
    for key, value in record.items():
        if torch.is_tensor(value) and value.shape[0] == seq_len:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    sliced['name'] = f"{record['name']}[{start}:{end}]"
    return sliced


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


def contact_proxy_losses(pose_pred, v_refined, stationary_prob, foot_height_weight, tran_gt):
    joints = root_relative_joints(pose_pred)
    foot_rel = joints[:, FOOT_JOINTS]
    if foot_rel.shape[0] >= 2:
        foot_rel_vel = (foot_rel[1:] - foot_rel[:-1]) / DT
        root_vel = v_refined[1:].unsqueeze(1)
        foot_world_vel = foot_rel_vel + root_vel
        contact_w = stationary_prob[1:, FOOT_STATIONARY_PROB_COLUMNS].to(DEVICE).clamp(0.0, 1.0)
        contact_foot_velocity = (contact_w * foot_world_vel.square().sum(dim=-1)).sum() / contact_w.sum().clamp_min(1.0)
    else:
        contact_foot_velocity = pose_pred.new_zeros(())
    if foot_height_weight > 0.0:
        foot_world_y = foot_rel[..., 1] + tran_gt.to(DEVICE).view(-1, 1, 3)[:, :, 1]
        ground_y = foot_world_y.detach().amin()
        contact_w = stationary_prob[:, FOOT_STATIONARY_PROB_COLUMNS].to(DEVICE).clamp(0.0, 1.0)
        height_error = (foot_world_y - ground_y).square()
        contact_foot_height = (contact_w * height_error).sum() / contact_w.sum().clamp_min(1.0)
    else:
        contact_foot_height = pose_pred.new_zeros(())
    return {
        'contact_foot_velocity': contact_foot_velocity,
        'contact_foot_height': contact_foot_height,
    }


def rotation_to_angular_velocity(R):
    if R.shape[0] < 2:
        return R.new_zeros((0,) + R.shape[1:-2] + (3,))
    dR = R[1:].matmul(R[:-1].transpose(-1, -2))
    return art.math.rotation_matrix_to_axis_angle(dR.reshape(-1, 3, 3)).reshape(dR.shape[:-2] + (3,)) / DT


def imu_proxy_losses(pose_pred, tran_for_proxy, record):
    imu_model = body_model(calc_imu=True)
    grot, _, verts = imu_model.forward_kinematics(
        pose_pred.to(DEVICE),
        None,
        tran_for_proxy.to(DEVICE),
        calc_mesh=True,
    )
    imu_R = grot[:, IMU_JOINTS]
    target_R = record['RMB'].to(DEVICE)
    ori_geo = rotation_geodesic(imu_R, target_R).mean()
    if verts.shape[0] >= 3:
        acc_pred = (verts[2:] - 2.0 * verts[1:-1] + verts[:-2]) / (DT * DT) + GRAVITY_M
        acc_target = record['aM'][1:-1].to(DEVICE)
        acc_proxy = torch.nn.functional.smooth_l1_loss(acc_pred, acc_target)
    else:
        acc_proxy = pose_pred.new_zeros(())
    gyro_pred = rotation_to_angular_velocity(imu_R)
    if gyro_pred.numel() > 0:
        gyro_proxy = torch.nn.functional.smooth_l1_loss(gyro_pred, record['wM'][1:].to(DEVICE))
    else:
        gyro_proxy = pose_pred.new_zeros(())
    return {
        'imu_orientation_proxy': ori_geo,
        'imu_acc_proxy': acc_proxy,
        'imu_gyro_proxy': gyro_proxy,
    }


def pose_velocity_loss(model_output, record, weights, compute_imu_proxy=False):
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
    q_res = q75_residual(q_pred, q_base)
    _, velocity_components, _ = velocity_residual_losses(
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
        'residual_prior': q_res.square().mean(),
        'tail_update_prior': model_output['tail_delta_norm'],
        'root_velocity': velocity_components['root_velocity'],
        'baseline_velocity': velocity_components['baseline_velocity'],
        'velocity_smooth': velocity_components['velocity_smooth'],
    }
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
    losses['fk_joint_rootrel'] = torch.nn.functional.smooth_l1_loss(root_relative_joints(pose_pred), root_relative_joints(pose_gt))
    losses['fk_joint_baseline_rootrel'] = torch.nn.functional.smooth_l1_loss(
        root_relative_joints(pose_pred),
        root_relative_joints(pose_base),
    )
    losses.update(contact_proxy_losses(
        pose_pred,
        model_output['v_refined'],
        record['stationary_prob'].to(DEVICE),
        weights.get('contact_foot_height', 0.0),
        record['tran_gt'],
    ))
    if compute_imu_proxy:
        losses.update(imu_proxy_losses(pose_pred, record['tran_gt'].to(DEVICE), record))
    else:
        zero = q_pred.new_zeros(())
        losses['imu_orientation_proxy'] = zero
        losses['imu_acc_proxy'] = zero
        losses['imu_gyro_proxy'] = zero
    total = q_pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * weight
    return total, losses


def average(items):
    return sum(items) / max(1, len(items))


def train_epoch(model, records, optimizer, weights, args, step):
    model.train()
    totals = {}
    rows = []
    for seq_idx, source_record in enumerate(records, start=1):
        step += 1
        seq_len = source_record['q75_prephysics'].shape[0]
        max_start = max(0, seq_len - args.window)
        start = step % (max_start + 1) if max_start > 0 else 0
        record = slice_record(source_record, start, args.window)
        output = run_cached_sequence(model, record)
        loss, components = pose_velocity_loss(output, record, weights, compute_imu_proxy=args.compute_imu_proxy)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at seq={record["name"]}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not grad_ok:
            raise RuntimeError(f'Non-finite gradient at seq={record["name"]}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        q_norm = output['q_residual'].norm(dim=-1)
        dv_norm = output['delta_v'].norm(dim=-1)
        row = {
            'step': step,
            'seq_idx': seq_idx,
            'seq_name': record['name'],
            'window': int(record['q75_prephysics'].shape[0]),
            'loss': float(loss.detach()),
            'q_residual_norm_mean': float(q_norm.mean().detach()),
            'q_residual_norm_max': float(q_norm.max().detach()),
            'delta_v_root_norm_mean': float(dv_norm.mean().detach()),
            'delta_v_root_norm_max': float(dv_norm.max().detach()),
            'tail_update_norm_mean': float(output['tail_delta_norm'].detach()),
        }
        for key, value in components.items():
            row[key] = float(value.detach())
        rows.append(row)
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
    return {key: average(value) for key, value in totals.items()}, rows, step


def cache_eval(model, records, weights, compute_imu_proxy, max_sequences=0):
    selected = records[:max_sequences] if max_sequences else records
    totals = {}
    rows = []
    model.eval()
    with torch.no_grad():
        for record in selected:
            output = run_cached_sequence(model, record)
            loss, components = pose_velocity_loss(output, record, weights, compute_imu_proxy=compute_imu_proxy)
            q_norm = output['q_residual'].norm(dim=-1)
            dv_norm = output['delta_v'].norm(dim=-1)
            row = {
                'name': record['name'],
                'loss': float(loss.detach()),
                'q_residual_norm_mean': float(q_norm.mean().detach()),
                'q_residual_norm_max': float(q_norm.max().detach()),
                'delta_v_root_norm_mean': float(dv_norm.mean().detach()),
                'delta_v_root_norm_max': float(dv_norm.max().detach()),
                'tail_update_norm_mean': float(output['tail_delta_norm'].detach()),
            }
            for key, value in components.items():
                row[key] = float(value.detach())
            rows.append(row)
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    totals.setdefault(key, []).append(float(value))
    return {
        'num_sequences': len(selected),
        'loss': {key: average(value) for key, value in totals.items()},
        'rows': rows,
    }


def save_checkpoint(path, model, optimizer, args, epoch, step, score, weights, selection):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'epoch': epoch,
        'step': step,
        'validation_score': score,
        'weights': weights,
        'selection': selection,
    }, path)


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
    newly_initialized = sorted(key for key in current if key not in state)
    current.update(compatible)
    model.load_state_dict(current)
    return {
        'path': checkpoint_path,
        'mode': 'partial_shape_compatible',
        'loaded': sorted(compatible),
        'skipped': skipped,
        'newly_initialized': newly_initialized,
    }


def summarize_metrics(aggregate):
    return {
        'baseline_metrics': aggregate['baseline_metrics'],
        'model_metrics': aggregate['model_metrics'],
        'delta_metrics': aggregate['delta_metrics'],
        'delta_v_root_norm_mean': aggregate['delta_v_root_norm_mean'],
        'delta_v_root_norm_max': aggregate['delta_v_root_norm_max'],
        'q_residual_norm_mean': aggregate['q_residual_norm_mean'],
        'q_residual_norm_max': aggregate['q_residual_norm_max'],
        'tail_update_norm_mean': aggregate['tail_update_norm_mean'],
        'tail_update_norm_max': aggregate['tail_update_norm_max'],
    }


def l4_imu_field_contract(records, prefix='auto'):
    sample = records[0] if records else {}
    if prefix == 'original':
        a_field, w_field, r_field = 'aM', 'wM', 'RMB'
    elif prefix == 'l4':
        missing = [key for key in ('l4_aM', 'l4_wM', 'l4_RMB') if key not in sample]
        if missing:
            raise KeyError(f'--l4-imu-field-prefix l4 requested but train cache is missing {missing}.')
        a_field, w_field, r_field = 'l4_aM', 'l4_wM', 'l4_RMB'
    elif prefix == 'auto':
        a_field = 'l4_aM' if 'l4_aM' in sample else 'aM'
        w_field = 'l4_wM' if 'l4_wM' in sample else 'wM'
        r_field = 'l4_RMB' if 'l4_RMB' in sample else 'RMB'
    else:
        raise ValueError(f'Unsupported l4_imu_field_prefix: {prefix}')
    return {
        'prefix': prefix,
        'a_field': a_field,
        'w_field': w_field,
        'R_field': r_field,
        'fallback': 'auto uses l4_* when present; original/l4 are explicit',
    }


def main():
    parser = argparse.ArgumentParser(description='Short validation-selected L4 loss ablation trainer.')
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--loss-mode', choices=('base', 'contact', 'imu_proxy', 'contact_imu'), required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--velocity-residual-scale', type=float, default=0.005)
    parser.add_argument('--pose-input-mode', choices=('euler_q75', 'rot6d'), default='euler_q75')
    parser.add_argument('--offset-conditioning', choices=('none', 'hidden_init'), default='none')
    parser.add_argument('--rnn-init-mode', choices=('none', 'offset_only', 'offset_firstframe'), default='', help='Preferred replacement for --offset-conditioning. Empty keeps the legacy offset-conditioning mapping.')
    parser.add_argument('--offset-init-scale', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0, help='Training-only dropout on the full L4 frame feature before the input MLP.')
    parser.add_argument('--imu-feature-dropout', type=float, default=0.0, help='Training-only dropout on the 90D aM/wM/RMB IMU feature slice.')
    parser.add_argument('--acc-dropout', type=float, default=0.0, help='Training-only dropout on the 18D aM feature slice.')
    parser.add_argument('--gyro-dropout', type=float, default=0.0, help='Training-only dropout on the 18D wM feature slice.')
    parser.add_argument('--orientation-dropout', type=float, default=0.0, help='Training-only dropout on the 54D RMB feature slice.')
    parser.add_argument('--l4-imu-field-prefix', choices=('auto', 'original', 'l4'), default='auto', help='Which IMU fields feed L4. auto uses l4_* when present, original forces aM/wM/RMB, l4 requires l4_aM/l4_wM/l4_RMB.')
    parser.add_argument('--paired-offset-training', action='store_true', help='Reserved K2 flag; first implementation keeps mixed-view records as ordinary samples.')
    parser.add_argument('--pair-consistency-weight', type=float, default=0.0, help='Reserved K2 consistency loss weight. Must stay 0.0 until paired loss is implemented.')
    parser.add_argument('--allow-partial-init', action='store_true', help='Load only shape-compatible tensors from --init-checkpoint.')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--disable-root-velocity-loss', action='store_true')
    parser.add_argument('--pose-geodesic-weight', type=float, default=None)
    parser.add_argument('--q-body-weight', type=float, default=None)
    parser.add_argument('--q-root-ori-weight', type=float, default=None)
    parser.add_argument('--baseline-body-weight', type=float, default=None)
    parser.add_argument('--baseline-root-ori-weight', type=float, default=None)
    parser.add_argument('--baseline-velocity-weight', type=float, default=None)
    parser.add_argument('--residual-prior-weight', type=float, default=None)
    parser.add_argument('--tail-update-prior-weight', type=float, default=None)
    parser.add_argument('--fk-joint-rootrel-weight', type=float, default=None)
    parser.add_argument('--fk-joint-baseline-rootrel-weight', type=float, default=None)
    parser.add_argument('--contact-foot-velocity-weight', type=float, default=0.0)
    parser.add_argument('--contact-foot-height-weight', type=float, default=0.0)
    parser.add_argument('--imu-orientation-weight', type=float, default=0.0)
    parser.add_argument('--imu-acc-weight', type=float, default=0.0)
    parser.add_argument('--imu-gyro-weight', type=float, default=0.0)
    parser.add_argument('--enable-imu-proxy-training', action='store_true')
    parser.add_argument('--validate-every', type=int, default=1)
    args = parser.parse_args()
    if args.tail_length != 4:
        raise ValueError('Only tail_length=4 is approved for the current L4 method.')
    if args.loss_mode in ('imu_proxy', 'contact_imu') and not args.enable_imu_proxy_training:
        if args.imu_orientation_weight or args.imu_acc_weight or args.imu_gyro_weight:
            raise ValueError('IMU proxy weights require --enable-imu-proxy-training after coordinate audit approval.')
    if args.pair_consistency_weight != 0.0:
        raise ValueError('Pair consistency loss is reserved but not implemented; keep --pair-consistency-weight 0.0.')
    train_records, train_manifest = load_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, max_sequences=args.max_val_sequences)
    weights = default_weights(args.disable_root_velocity_loss)
    optional_weight_args = {
        'pose_geodesic': args.pose_geodesic_weight,
        'q_body': args.q_body_weight,
        'q_root_ori': args.q_root_ori_weight,
        'baseline_body': args.baseline_body_weight,
        'baseline_root_ori': args.baseline_root_ori_weight,
        'baseline_velocity': args.baseline_velocity_weight,
        'residual_prior': args.residual_prior_weight,
        'tail_update_prior': args.tail_update_prior_weight,
        'fk_joint_rootrel': args.fk_joint_rootrel_weight,
        'fk_joint_baseline_rootrel': args.fk_joint_baseline_rootrel_weight,
    }
    for key, value in optional_weight_args.items():
        if value is not None:
            weights[key] = value
    if args.loss_mode in ('contact', 'contact_imu'):
        weights['contact_foot_velocity'] = args.contact_foot_velocity_weight
        weights['contact_foot_height'] = args.contact_foot_height_weight
    if args.loss_mode in ('imu_proxy', 'contact_imu') and args.enable_imu_proxy_training:
        weights['imu_orientation_proxy'] = args.imu_orientation_weight
        weights['imu_acc_proxy'] = args.imu_acc_weight
        weights['imu_gyro_proxy'] = args.imu_gyro_weight
    compute_imu_proxy = args.loss_mode in ('imu_proxy', 'contact_imu')
    args.compute_imu_proxy = compute_imu_proxy
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
        dropout=args.dropout,
        imu_feature_dropout=args.imu_feature_dropout,
        acc_dropout=args.acc_dropout,
        gyro_dropout=args.gyro_dropout,
        orientation_dropout=args.orientation_dropout,
    ).to(DEVICE)
    model.l4_imu_field_prefix = args.l4_imu_field_prefix
    checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
    init_checkpoint_load = load_compatible_state(
        model,
        args.init_checkpoint,
        allow_partial=args.allow_partial_init,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    l4_imu_contract = l4_imu_field_contract(train_records, args.l4_imu_field_prefix)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / 'train_result.json'
    log_path = output_dir / 'train_log.jsonl'
    result = {
        'experiment_name': args.experiment_name,
        'config': vars(args),
        'base_checkpoint': args.init_checkpoint,
        'base_checkpoint_info': {
            'epoch': checkpoint.get('epoch'),
            'step': checkpoint.get('step'),
            'train_loss': checkpoint.get('train_loss'),
            'validation_score': checkpoint.get('validation_score'),
            'selection': checkpoint.get('selection'),
        },
        'train_cache_manifest': train_manifest,
        'val_cache_manifest': val_manifest,
        'num_train_sequences': len(train_records),
        'num_val_sequences': len(val_records),
        'l4_imu_field_contract': l4_imu_contract,
        'weights': weights,
        'pose_input_contract': {
            'pose_input_mode': args.pose_input_mode,
            'pose_input_dim': prephysics_feature_dim(args.pose_input_mode) - 90,
            'imu_feature_dim': 90,
            'model_n_input': model.n_input,
            'sensor_offset_input': args.offset_conditioning != 'none',
            'offset_conditioning': args.offset_conditioning,
            'rnn_init_mode': effective_rnn_init_mode,
            'offset_init_scale': args.offset_init_scale,
            'offset_encoder': (
                '18 -> hidden_size -> hidden_size, ReLU after first layer, final layer zero-initialized'
                if effective_rnn_init_mode == 'offset_only'
                else (
                    '252 -> hidden_size -> hidden_size, offset_r 18D + first-frame frame feature 234D, ReLU after first layer, final layer zero-initialized'
                    if effective_rnn_init_mode == 'offset_firstframe'
                    else 'none'
                )
            ),
            'rnn_init_input_dim': (
                18 if effective_rnn_init_mode == 'offset_only'
                else (18 + model.n_input if effective_rnn_init_mode == 'offset_firstframe' else 0)
            ),
            'rnn_init_uses_first_frame': effective_rnn_init_mode == 'offset_firstframe',
            'offset_encoder_legacy': (
                '18 -> hidden_size -> hidden_size, ReLU after first layer, final layer zero-initialized'
                if args.offset_conditioning == 'hidden_init'
                else 'none'
            ),
            'imu_position_offset_augmented_datasets': 'excluded',
            'imu_proxy_loss': bool(args.enable_imu_proxy_training),
            'dropout': {
                'dropout': args.dropout,
                'imu_feature_dropout': args.imu_feature_dropout,
                'acc_dropout': args.acc_dropout,
                'gyro_dropout': args.gyro_dropout,
                'orientation_dropout': args.orientation_dropout,
                'training_only': True,
                'applied_to_firstframe_init_feature': effective_rnn_init_mode == 'offset_firstframe',
                'applied_to_frame_feature': True,
            },
        },
        'init_checkpoint_load': init_checkpoint_load,
        'loss_mode': args.loss_mode,
        'checkpoint_selection': 'lowest full-validation MotionEvaluator score',
        'test_set_used': False,
        'foot_contact_contract': {
            'contact_joints': [0, 10, 11, 22, 23],
            'foot_joints': list(FOOT_JOINTS),
            'stationary_prob_columns_for_feet': list(FOOT_STATIONARY_PROB_COLUMNS),
            'velocity_formula': 'foot_world_velocity ~= d(root_relative_foot_position)/dt + refined_root_velocity',
            'height_loss_ground_assumption': 'disabled unless contact_foot_height_weight > 0; then per-window min GT foot y is used as diagnostic training ground_y',
        },
        'imu_proxy_contract': {
            'enabled_as_training_loss': bool(args.enable_imu_proxy_training),
            'dt': DT,
            'gravity_model_frame': [0.0, -9.8, 0.0],
            'imu_vertices': list(IMU_VERTICES),
            'imu_orientation_joints': list(IMU_JOINTS),
            'note': 'Without --enable-imu-proxy-training, IMU proxy terms are diagnostics only and all IMU proxy weights remain zero.',
        },
        'epochs': [],
        'best': None,
        'status': 'running',
    }
    print(json.dumps({'l4_imu_field_contract': l4_imu_contract}, indent=2), flush=True)
    best_score = None
    step = 0
    with log_path.open('w') as log_file:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_loss, train_rows, step = train_epoch(model, train_records, optimizer, weights, args, step)
            cache_validation = cache_eval(model, val_records, weights, compute_imu_proxy)
            physics_validation = None
            validation_rows = []
            validation_score = None
            if epoch % args.validate_every == 0:
                validation_rows = evaluate_physics(model, val_records)
                physics_validation = aggregate_eval(validation_rows)
                validation_score = score_for_checkpoint(physics_validation)
            improved = validation_score is not None and (best_score is None or validation_score < best_score)
            if improved:
                best_score = validation_score
                save_checkpoint(
                    output_dir / 'best.pt',
                    model,
                    optimizer,
                    args,
                    epoch,
                    step,
                    validation_score,
                    weights,
                    'lowest full-validation MotionEvaluator score',
                )
            save_checkpoint(
                output_dir / 'last.pt',
                model,
                optimizer,
                args,
                epoch,
                step,
                validation_score,
                weights,
                'last epoch',
            )
            epoch_record = {
                'epoch': epoch,
                'step': step,
                'epoch_wall_seconds': time.time() - epoch_start,
                'train_loss': train_loss,
                'cache_validation': cache_validation,
                'physics_validation': physics_validation,
                'physics_validation_rows': validation_rows,
                'validation_score': validation_score,
                'best_score': best_score,
                'improved_best': improved,
            }
            if physics_validation is not None:
                epoch_record['physics_validation_summary'] = summarize_metrics(physics_validation)
            result['epochs'].append(epoch_record)
            if improved:
                result['best'] = epoch_record
            log_file.write(json.dumps(epoch_record) + '\n')
            log_file.flush()
            result_path.write_text(json.dumps(result, indent=2))
            print(
                f"epoch={epoch} train_loss={train_loss.get('loss', train_loss.get('total', 0.0)):.6g} "
                f"val_score={validation_score if validation_score is not None else 'none'} "
                f"best={best_score if best_score is not None else 'none'} "
                f"seconds={epoch_record['epoch_wall_seconds']:.2f}",
                flush=True,
            )
    result['status'] = 'completed'
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'result_path': str(result_path),
        'log_path': str(log_path),
        'best_checkpoint': str(output_dir / 'best.pt') if result['best'] else None,
        'best_epoch': result['best']['epoch'] if result['best'] else None,
        'status': result['status'],
    }, indent=2))


if __name__ == '__main__':
    main()
