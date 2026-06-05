import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import torch

import articulate as art
from l4_q75_utils import prephysics_feature, prephysics_feature_dim, q75_to_pose_tran
from l4_tail_update_qstate import StreamingTailUpdateQState
from k2_so3_curve import StreamingTailUpdateSO3State, pose_tran_to_so3_state, q75_to_so3_state
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
        'fk_joint_global': 0.0,
        'fk_joint_baseline_rootrel': 0.0,
        'local_pose_preserve_body': 0.0,
        'local_pose_preserve_root_ori': 0.0,
        'rootrel_joint_velocity': 0.0,
        'qdot_consistency_body': 0.0,
        'qdot_consistency_all': 0.0,
        'qdot_body_target_fd': 0.0,
        'qdot_consistency_root': 0.0,
        'control_point_prior': 0.0,
        'trajectory_q_body_prior': 0.0,
        'qddot_body_smooth': 0.0,
        'root_translation': 0.0,
        'root_acc_smooth': 0.0,
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
        'imu_proxy_offset_acc': 0.0,
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


def temporal_slice(length, mode='full_window', recent_frames=4, weighting='uniform', device=None):
    if length <= 0:
        raise ValueError('temporal loss selection requires at least one frame.')
    if mode == 'full_window':
        start = 0
    elif mode == 'last':
        start = length - 1
    elif mode == 'recent_l4':
        start = max(0, length - int(recent_frames))
    else:
        raise ValueError(f'Unsupported loss_temporal_mode: {mode}')
    indices = torch.arange(start, length, device=device)
    if weighting == 'uniform' or indices.numel() == 1:
        weights = torch.ones(indices.numel(), device=device)
    elif weighting == 'ramp':
        weights = torch.linspace(0.25, 1.0, indices.numel(), device=device)
    else:
        raise ValueError(f'Unsupported recent_loss_weighting: {weighting}')
    weights = weights / weights.sum().clamp_min(1e-12)
    return indices, weights


def select_time(x, indices):
    return x.index_select(0, indices.to(x.device))


def weighted_frame_mean(values, frame_weights):
    weights = frame_weights.to(values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    per_frame = values.reshape(values.shape[0], -1).mean(dim=1)
    return (per_frame * frame_weights.to(values.device, dtype=values.dtype)).sum()


def weighted_smooth_l1(input_tensor, target_tensor, frame_weights):
    loss = torch.nn.functional.smooth_l1_loss(input_tensor, target_tensor, reduction='none')
    return weighted_frame_mean(loss, frame_weights)


def root_relative_joints(pose):
    joints = body_model().forward_kinematics(pose.to(DEVICE))[1]
    return joints - joints[:, :1]


def root_relative_joint_velocity(pose):
    joints = root_relative_joints(pose)
    if joints.shape[0] < 2:
        return joints.new_zeros((0,) + joints.shape[1:])
    return (joints[1:] - joints[:-1]) / DT


def qdot_consistency_loss(q_pred, qdot_pred, body_only=True):
    if q_pred.shape[0] < 2:
        return q_pred.new_zeros(())
    start = 6 if body_only else 0
    q_step = q_pred[1:, start:] - q_pred[:-1, start:]
    if start >= 3:
        q_step = torch.atan2(torch.sin(q_step), torch.cos(q_step))
    else:
        q_step = torch.cat((
            q_step[..., :3],
            torch.atan2(torch.sin(q_step[..., 3:]), torch.cos(q_step[..., 3:])),
        ), dim=-1)
    return torch.nn.functional.smooth_l1_loss(q_step, DT * qdot_pred[1:, start:])


def qdot_body_target_fd_loss(q_gt, qdot_pred):
    if q_gt.shape[0] < 2:
        return q_gt.new_zeros(())
    q_step = q_gt[1:, 6:] - q_gt[:-1, 6:]
    q_step = torch.atan2(torch.sin(q_step), torch.cos(q_step))
    return torch.nn.functional.smooth_l1_loss(DT * qdot_pred[1:, 6:], q_step)


def qdot_root_consistency_loss(q_pred, qdot_pred):
    if q_pred.shape[0] < 2:
        return q_pred.new_zeros(())
    tran_step = q_pred[1:, :3] - q_pred[:-1, :3]
    root_ori_step = q_pred[1:, 3:6] - q_pred[:-1, 3:6]
    root_ori_step = torch.atan2(torch.sin(root_ori_step), torch.cos(root_ori_step))
    q_step = torch.cat((tran_step, root_ori_step), dim=-1)
    return torch.nn.functional.smooth_l1_loss(q_step, DT * qdot_pred[1:, :6])


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


def imu_input_mode_to_field_prefix(mode):
    if mode == 'official':
        return 'original'
    if mode == 'processed':
        return 'l4'
    if mode == 'auto':
        return 'auto'
    raise ValueError(f'Unsupported imu_input_mode: {mode}')


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
    base_so3_seq = None
    if getattr(model, 'model_type', '') == 'k2_so3curve_v1':
        base_so3_seq = q75_to_so3_state(
            record['q75_prephysics'].to(DEVICE),
            euler_seq=getattr(model, 'euler_seq', 'XYZ'),
        )
    qs = []
    qdots = []
    qddots = []
    pose_preds = []
    q_residuals = []
    new_norms = []
    tail_norms = []
    control_point_priors = []
    generated_controls = []
    interpolated_frames = []
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
        q_result = model.step(
            feature,
            q_base,
            None if base_so3_seq is None else base_so3_seq[frame_idx],
            return_euler=not bool(getattr(model, 'fast_training_so3', False)),
        )
        v_result = model.refine_velocity(
            record['v_root_vr'][frame_idx].to(DEVICE),
            record['stationary_prob'][frame_idx].to(DEVICE),
        )
        qs.append(q_result['q_t'][0])
        qdots.append(q_result['qdot_t'][0])
        qddots.append(q_result['qddot_t'][0])
        if torch.is_tensor(q_result.get('pose_R_t')):
            pose_preds.append(q_result['pose_R_t'][0])
        q_residuals.append(q_result['residual_t'][0])
        new_norms.append(q_result['new_delta_norm'])
        tail_norms.append(q_result['tail_delta_norm'])
        control_point_priors.append(q_result['control_point_prior_t'])
        generated_controls.append(q_result['q_t'].new_tensor(float(bool(q_result.get('generated_control', True)))))
        interpolated_frames.append(q_result['q_t'].new_tensor(float(q_result.get('decode_u', 0.0) > 0.0)))
        v_refined.append(v_result['v_root_refined'][0])
        delta_vs.append(v_result['delta_v_root'][0])
    result = {
        'q_pred': torch.stack(qs),
        'qdot_pred': torch.stack(qdots),
        'qddot_pred': torch.stack(qddots),
        'q_residual': torch.stack(q_residuals),
        'new_delta_norm': torch.stack(new_norms).mean(),
        'tail_delta_norm': torch.stack(tail_norms).mean(),
        'control_point_prior': torch.stack(control_point_priors).mean(),
        'generated_control_fraction': torch.stack(generated_controls).mean(),
        'interpolated_frame_fraction': torch.stack(interpolated_frames).mean(),
        'v_refined': torch.stack(v_refined),
        'delta_v': torch.stack(delta_vs),
    }
    if getattr(model, 'model_type', '') == 'k2_so3curve_v1':
        result['model_type'] = 'k2_so3curve_v1'
        if pose_preds:
            result['pose_pred'] = torch.stack(pose_preds)
    return result


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


def should_apply_offset_imu_proxy(record, views):
    if views == 'all':
        return True
    if views == 'offset_aug_only':
        return 'offset_aug' in str(record.get('name', ''))
    raise ValueError(f'Unsupported imu_proxy_views: {views}')


def adjust_acc_gravity(acc_proxy, gravity_mode):
    if gravity_mode == 'none':
        return acc_proxy
    if gravity_mode == 'plus_g':
        return acc_proxy + GRAVITY_M.view(1, 1, 3)
    if gravity_mode == 'minus_g':
        return acc_proxy - GRAVITY_M.view(1, 1, 3)
    raise ValueError(f'Unsupported imu_proxy_gravity_mode: {gravity_mode}')


def offset_sensor_positions_from_pose(pose, tran, offset_r):
    if offset_r is None:
        raise KeyError('offset-aware IMU proxy requires record["offset_r"] / imu_offset_r / r_JS.')
    offset_r = offset_r.to(DEVICE).float()
    if tuple(offset_r.shape) != (6, 3):
        raise ValueError(f'Expected offset_r shape [6,3], got {tuple(offset_r.shape)}.')
    grot, joints = body_model().forward_kinematics(
        pose.to(DEVICE),
        None,
        tran.to(DEVICE),
    )
    joint_pos = joints[:, IMU_JOINTS]
    joint_rot = grot[:, IMU_JOINTS]
    rotated_offset = joint_rot.matmul(offset_r.view(1, 6, 3, 1)).squeeze(-1)
    return joint_pos + rotated_offset


def acceleration_residual_stats(residual):
    if residual.numel() == 0:
        zero = residual.new_zeros(())
        stats = {
            'imu_proxy_offset_acc_rms': zero,
            'imu_proxy_offset_acc_p50': zero,
            'imu_proxy_offset_acc_p90': zero,
            'imu_proxy_offset_acc_max': zero,
            'imu_proxy_offset_acc_finite': residual.new_tensor(1.0),
        }
        for sensor_idx in range(6):
            stats[f'imu_proxy_offset_acc_rms_s{sensor_idx}'] = zero
        return stats
    norm = residual.norm(dim=-1)
    stats = {
        'imu_proxy_offset_acc_rms': residual.square().mean().sqrt(),
        'imu_proxy_offset_acc_p50': torch.quantile(norm.reshape(-1), 0.50),
        'imu_proxy_offset_acc_p90': torch.quantile(norm.reshape(-1), 0.90),
        'imu_proxy_offset_acc_max': norm.max(),
        'imu_proxy_offset_acc_finite': residual.new_tensor(float(torch.isfinite(residual).all().item())),
    }
    per_sensor_rms = residual.square().mean(dim=(0, 2)).sqrt()
    for sensor_idx in range(6):
        stats[f'imu_proxy_offset_acc_rms_s{sensor_idx}'] = per_sensor_rms[sensor_idx]
    return stats


def offset_acc_fd_proxy_losses(pose_pred, tran_pred, record, views='offset_aug_only', gravity_mode='none'):
    zero = pose_pred.new_zeros(())
    losses = {
        'imu_proxy_offset_acc': zero,
        'imu_proxy_offset_acc_weighted': zero,
        'imu_proxy_offset_acc_applied': zero,
        'imu_proxy_offset_acc_frames': zero,
    }
    losses.update(acceleration_residual_stats(pose_pred.new_zeros((0, 6, 3))))
    if not should_apply_offset_imu_proxy(record, views):
        return losses
    if pose_pred.shape[0] < 3:
        return losses
    p_sensor = offset_sensor_positions_from_pose(pose_pred, tran_pred, record.get('offset_r'))
    acc_proxy = (p_sensor[2:] - 2.0 * p_sensor[1:-1] + p_sensor[:-2]) / (DT * DT)
    acc_proxy = adjust_acc_gravity(acc_proxy, gravity_mode)
    acc_target = record['aM'][1:-1].to(DEVICE)
    residual = acc_proxy - acc_target
    raw_loss = torch.nn.functional.smooth_l1_loss(acc_proxy, acc_target)
    losses.update({
        'imu_proxy_offset_acc': raw_loss,
        'imu_proxy_offset_acc_applied': pose_pred.new_tensor(1.0),
        'imu_proxy_offset_acc_frames': pose_pred.new_tensor(float(acc_proxy.shape[0])),
    })
    losses.update(acceleration_residual_stats(residual))
    return losses


def zero_offset_acc_fd_proxy_losses(reference):
    zero = reference.new_zeros(())
    losses = {
        'imu_proxy_offset_acc': zero,
        'imu_proxy_offset_acc_weighted': zero,
        'imu_proxy_offset_acc_applied': zero,
        'imu_proxy_offset_acc_frames': zero,
    }
    losses.update(acceleration_residual_stats(reference.new_zeros((0, 6, 3))))
    return losses


def pose_velocity_loss(
    model_output,
    record,
    weights,
    compute_imu_proxy=False,
    imu_proxy_mode='none',
    imu_proxy_views='offset_aug_only',
    imu_proxy_gravity_mode='none',
    loss_temporal_mode='full_window',
    recent_loss_frames=4,
    recent_loss_weighting='uniform',
):
    q_pred_full = model_output['q_pred']
    qdot_pred_full = model_output['qdot_pred']
    is_so3_fast = model_output.get('model_type') == 'k2_so3curve_v1' and 'pose_pred' in model_output
    if is_so3_fast:
        q_base_full = q75_to_so3_state(record['q75_prephysics'].to(DEVICE))
        q_gt_full = pose_tran_to_so3_state(record['pose_gt'].to(DEVICE), record['tran_gt'].to(DEVICE))
    else:
        q_base_full = record['q75_prephysics'].to(DEVICE)
        q_gt_full = record['q75_gt'].to(DEVICE)
    time_indices, time_weights = temporal_slice(
        q_pred_full.shape[0],
        mode=loss_temporal_mode,
        recent_frames=recent_loss_frames,
        weighting=recent_loss_weighting,
        device=q_pred_full.device,
    )
    q_pred = select_time(q_pred_full, time_indices)
    qdot_pred = select_time(qdot_pred_full, time_indices)
    q_base = select_time(q_base_full, time_indices)
    q_gt = select_time(q_gt_full, time_indices)
    selected_record = dict(record)
    for key in ('aM', 'wM', 'RMB', 'tran_gt', 'stationary_prob'):
        if key in selected_record and torch.is_tensor(selected_record[key]) and selected_record[key].shape[0] == q_pred_full.shape[0]:
            selected_record[key] = select_time(selected_record[key].to(DEVICE), time_indices)
    if is_so3_fast:
        pose_pred = select_time(model_output['pose_pred'].to(DEVICE), time_indices)
        tran_pred = q_pred[:, :3]
    else:
        pose_pred, tran_pred = q75_to_pose_tran(q_pred)
    pose_gt = select_time(record['pose_gt'].to(DEVICE), time_indices)
    pose_base = select_time(record['pose_prephysics'].to(DEVICE), time_indices) if is_so3_fast and 'pose_prephysics' in record else q75_to_pose_tran(q_base)[0]
    pose_pred = pose_pred.to(DEVICE)
    pose_base = pose_base.to(DEVICE)
    geo = rotation_geodesic(pose_pred, pose_gt)
    base_geo = rotation_geodesic(pose_pred, pose_base)
    q_res = q_pred - q_base if is_so3_fast else q75_residual(q_pred, q_base)
    _, velocity_components, _ = velocity_residual_losses(
        select_time(model_output['v_refined'], time_indices),
        finite_difference_translation_velocity(select_time(record['tran_gt'].to(DEVICE), time_indices)),
        select_time(record['v_root_vr'].to(DEVICE), time_indices),
        select_time(model_output['delta_v'], time_indices),
    )
    losses = {
        'pose_geodesic': weighted_frame_mean(geo, time_weights),
        'pose_geodesic_root': weighted_frame_mean(geo[:, 0:1], time_weights).detach(),
        'pose_geodesic_body': weighted_frame_mean(geo[:, 1:], time_weights).detach(),
        'q_body': weighted_smooth_l1(q_pred[:, 6:], q_gt[:, 6:], time_weights),
        'q_root_ori': weighted_smooth_l1(q_pred[:, 3:6], q_gt[:, 3:6], time_weights),
        'baseline_body': weighted_frame_mean(base_geo[:, 1:], time_weights),
        'baseline_root_ori': weighted_frame_mean(base_geo[:, 0:1], time_weights),
        'local_pose_preserve_body': weighted_frame_mean(base_geo[:, 1:], time_weights),
        'local_pose_preserve_root_ori': weighted_frame_mean(base_geo[:, 0:1], time_weights),
        'root_translation': weighted_smooth_l1(q_pred[:, :3], q_gt[:, :3], time_weights),
        'control_point_prior': model_output['control_point_prior'],
        'trajectory_q_body_prior': weighted_frame_mean(q_res[:, 6:].square(), time_weights),
        'qddot_body_smooth': select_time(model_output['qddot_pred'], time_indices)[:, 6:].square().mean(),
        'residual_prior': weighted_frame_mean(q_res.square(), time_weights),
        'tail_update_prior': model_output['tail_delta_norm'],
        'root_velocity': velocity_components['root_velocity'],
        'baseline_velocity': velocity_components['baseline_velocity'],
        'velocity_smooth': velocity_components['velocity_smooth'],
        'loss_temporal_num_frames': q_pred.new_tensor(float(time_indices.numel())),
        'loss_temporal_start_index': q_pred.new_tensor(float(time_indices[0].item())),
        'loss_temporal_end_index': q_pred.new_tensor(float(time_indices[-1].item())),
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
    losses['fk_joint_rootrel'] = weighted_smooth_l1(root_relative_joints(pose_pred), root_relative_joints(pose_gt), time_weights)
    joints_pred = body_model().forward_kinematics(pose_pred.to(DEVICE), tran=tran_pred.to(DEVICE))[1]
    joints_gt = body_model().forward_kinematics(
        pose_gt.to(DEVICE),
        tran=select_time(record['tran_gt'].to(DEVICE), time_indices),
    )[1]
    losses['fk_joint_global'] = weighted_smooth_l1(joints_pred, joints_gt, time_weights)
    losses['fk_joint_baseline_rootrel'] = weighted_smooth_l1(
        root_relative_joints(pose_pred),
        root_relative_joints(pose_base),
        time_weights,
    )
    if q_pred.shape[0] >= 2:
        losses['rootrel_joint_velocity'] = torch.nn.functional.smooth_l1_loss(
            root_relative_joint_velocity(pose_pred),
            root_relative_joint_velocity(pose_gt),
        )
        if is_so3_fast:
            losses['qdot_consistency_body'] = torch.nn.functional.smooth_l1_loss(
                q_pred[1:, 6:] - q_pred[:-1, 6:],
                DT * qdot_pred[1:, 6:],
            )
            losses['qdot_consistency_all'] = torch.nn.functional.smooth_l1_loss(
                q_pred[1:] - q_pred[:-1],
                DT * qdot_pred[1:],
            )
            losses['qdot_body_target_fd'] = torch.nn.functional.smooth_l1_loss(
                DT * qdot_pred[1:, 6:],
                q_gt[1:, 6:] - q_gt[:-1, 6:],
            )
            losses['qdot_consistency_root'] = torch.nn.functional.smooth_l1_loss(
                q_pred[1:, :6] - q_pred[:-1, :6],
                DT * qdot_pred[1:, :6],
            )
        else:
            losses['qdot_consistency_body'] = qdot_consistency_loss(q_pred, qdot_pred, body_only=True)
            losses['qdot_consistency_all'] = qdot_consistency_loss(q_pred, qdot_pred, body_only=False)
            losses['qdot_body_target_fd'] = qdot_body_target_fd_loss(q_gt, qdot_pred)
            losses['qdot_consistency_root'] = qdot_root_consistency_loss(q_pred, qdot_pred)
    else:
        losses['rootrel_joint_velocity'] = q_pred.new_zeros(())
        losses['qdot_consistency_body'] = q_pred.new_zeros(())
        losses['qdot_consistency_all'] = q_pred.new_zeros(())
        losses['qdot_body_target_fd'] = q_pred.new_zeros(())
        losses['qdot_consistency_root'] = q_pred.new_zeros(())
    if q_pred.shape[0] >= 3:
        losses['root_acc_smooth'] = finite_diff(q_pred[:, :3], 2).square().mean()
    else:
        losses['root_acc_smooth'] = q_pred.new_zeros(())
    losses.update(contact_proxy_losses(
        pose_pred,
        select_time(model_output['v_refined'], time_indices),
        select_time(record['stationary_prob'].to(DEVICE), time_indices),
        weights.get('contact_foot_height', 0.0),
        select_time(record['tran_gt'].to(DEVICE), time_indices),
    ))
    offset_proxy_needed = (
        imu_proxy_mode == 'offset_acc_fd'
        and weights.get('imu_proxy_offset_acc', 0.0) != 0.0
        and should_apply_offset_imu_proxy(record, imu_proxy_views)
    )
    if compute_imu_proxy:
        losses.update(imu_proxy_losses(pose_pred, selected_record['tran_gt'].to(DEVICE), selected_record))
    else:
        zero = q_pred.new_zeros(())
        losses['imu_orientation_proxy'] = zero
        losses['imu_acc_proxy'] = zero
        losses['imu_gyro_proxy'] = zero
    if imu_proxy_mode == 'offset_acc_fd' and offset_proxy_needed:
        offset_proxy = offset_acc_fd_proxy_losses(
            pose_pred,
            tran_pred,
            selected_record,
            views=imu_proxy_views,
            gravity_mode=imu_proxy_gravity_mode,
        )
        offset_proxy['imu_proxy_offset_acc_weighted'] = (
            offset_proxy['imu_proxy_offset_acc'] * weights.get('imu_proxy_offset_acc', 0.0)
        )
        losses.update(offset_proxy)
    elif imu_proxy_mode == 'offset_acc_fd':
        losses.update(zero_offset_acc_fd_proxy_losses(q_pred))
    elif imu_proxy_mode == 'none':
        offset_proxy = zero_offset_acc_fd_proxy_losses(q_pred)
        losses.update(offset_proxy)
    else:
        raise ValueError(f'Unsupported imu_proxy_mode: {imu_proxy_mode}')
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
        loss, components = pose_velocity_loss(
            output,
            record,
            weights,
            compute_imu_proxy=args.compute_imu_proxy,
            imu_proxy_mode=args.imu_proxy_mode,
            imu_proxy_views=args.imu_proxy_views,
            imu_proxy_gravity_mode=args.imu_proxy_gravity_mode,
            loss_temporal_mode=args.loss_temporal_mode,
            recent_loss_frames=args.recent_loss_frames,
            recent_loss_weighting=args.recent_loss_weighting,
        )
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
            'generated_control_fraction': float(output['generated_control_fraction'].detach()),
            'interpolated_frame_fraction': float(output['interpolated_frame_fraction'].detach()),
        }
        for key, value in components.items():
            row[key] = float(value.detach())
        rows.append(row)
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
    return {key: average(value) for key, value in totals.items()}, rows, step


def cache_eval(
    model,
    records,
    weights,
    compute_imu_proxy,
    max_sequences=0,
    imu_proxy_mode='none',
    imu_proxy_views='offset_aug_only',
    imu_proxy_gravity_mode='none',
):
    selected = records[:max_sequences] if max_sequences else records
    totals = {}
    rows = []
    model.eval()
    with torch.no_grad():
        for record in selected:
            output = run_cached_sequence(model, record)
            loss, components = pose_velocity_loss(
                output,
                record,
                weights,
                compute_imu_proxy=compute_imu_proxy,
                imu_proxy_mode=imu_proxy_mode,
                imu_proxy_views=imu_proxy_views,
                imu_proxy_gravity_mode=imu_proxy_gravity_mode,
                loss_temporal_mode=getattr(model, 'loss_temporal_mode', 'full_window'),
                recent_loss_frames=getattr(model, 'recent_loss_frames', 4),
                recent_loss_weighting=getattr(model, 'recent_loss_weighting', 'uniform'),
            )
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
                'generated_control_fraction': float(output['generated_control_fraction'].detach()),
                'interpolated_frame_fraction': float(output['interpolated_frame_fraction'].detach()),
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


def load_compatible_state(model, checkpoint_path, allow_partial=False, skip_prefixes=()):
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
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped.append({
                'key': key,
                'checkpoint_shape': list(value.shape),
                'model_shape': list(current[key].shape) if key in current else None,
                'reason': 'skip_prefix',
            })
        elif key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append({
                'key': key,
                'checkpoint_shape': list(value.shape),
                'model_shape': list(current[key].shape) if key in current else None,
                'reason': 'missing_or_shape_mismatch',
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
    parser.add_argument('--model-type', choices=('euler_q75_l4', 'k2_so3curve_v1'), default='euler_q75_l4')
    parser.add_argument('--loss-mode', choices=('base', 'contact', 'imu_proxy', 'contact_imu'), required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--control-stride', type=int, default=1, help='Generate one control point every N frames. 1 preserves the historical per-frame control-point contract; 2 enables K2 stride-2 controls.')
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
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='official', help='K2 IMU feature source. official uses aM/wM/RMB; processed uses l4_aM/l4_wM/l4_RMB and must exist in the cache; auto uses l4_* when present.')
    parser.add_argument('--l4-imu-field-prefix', choices=('auto', 'original', 'l4'), default=None, help='Legacy alias for --imu-input-mode. If set, overrides --imu-input-mode.')
    parser.add_argument('--paired-offset-training', action='store_true', help='Reserved K2 flag; first implementation keeps mixed-view records as ordinary samples.')
    parser.add_argument('--pair-consistency-weight', type=float, default=0.0, help='Reserved K2 consistency loss weight. Must stay 0.0 until paired loss is implemented.')
    parser.add_argument('--allow-partial-init', action='store_true', help='Load only shape-compatible tensors from --init-checkpoint.')
    parser.add_argument('--skip-output-head-init', action='store_true', help='When partial-loading a representation-changing model, keep new_control/tail_delta zero-initialized instead of loading same-shaped old heads.')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--disable-root-velocity-loss', action='store_true')
    parser.add_argument('--pose-geodesic-weight', type=float, default=None)
    parser.add_argument('--q-body-weight', type=float, default=None)
    parser.add_argument('--q-root-ori-weight', type=float, default=None)
    parser.add_argument('--baseline-body-weight', type=float, default=None)
    parser.add_argument('--baseline-root-ori-weight', type=float, default=None)
    parser.add_argument('--root-translation-weight', type=float, default=None)
    parser.add_argument('--root-velocity-weight', type=float, default=None)
    parser.add_argument('--root-acc-smooth-weight', type=float, default=None)
    parser.add_argument('--baseline-velocity-weight', type=float, default=None)
    parser.add_argument('--residual-prior-weight', type=float, default=None)
    parser.add_argument('--tail-update-prior-weight', type=float, default=None)
    parser.add_argument('--fk-joint-rootrel-weight', type=float, default=None)
    parser.add_argument('--fk-joint-global-weight', type=float, default=None)
    parser.add_argument('--fk-joint-baseline-rootrel-weight', type=float, default=None)
    parser.add_argument('--local-pose-preserve-body-weight', type=float, default=None)
    parser.add_argument('--local-pose-preserve-root-ori-weight', type=float, default=None)
    parser.add_argument('--rootrel-joint-velocity-weight', type=float, default=None)
    parser.add_argument('--qdot-consistency-body-weight', type=float, default=None)
    parser.add_argument('--qdot-consistency-all-weight', type=float, default=None)
    parser.add_argument('--qdot-body-target-fd-weight', type=float, default=None)
    parser.add_argument('--qdot-consistency-root-weight', type=float, default=None)
    parser.add_argument('--control-point-prior-weight', type=float, default=None)
    parser.add_argument('--trajectory-q-body-prior-weight', type=float, default=None)
    parser.add_argument('--qddot-body-smooth-weight', type=float, default=None)
    parser.add_argument('--contact-foot-velocity-weight', type=float, default=0.0)
    parser.add_argument('--contact-foot-height-weight', type=float, default=0.0)
    parser.add_argument('--imu-orientation-weight', type=float, default=0.0)
    parser.add_argument('--imu-acc-weight', type=float, default=0.0)
    parser.add_argument('--imu-gyro-weight', type=float, default=0.0)
    parser.add_argument('--enable-imu-proxy-training', action='store_true')
    parser.add_argument('--imu-proxy-weight', type=float, default=0.0, help='Weight for the offset-aware IMU acceleration proxy loss.')
    parser.add_argument('--imu-proxy-mode', choices=('none', 'offset_acc_fd'), default='none', help='Offset-aware IMU proxy loss mode. offset_acc_fd uses p_S=p_J+R_J@r_JS and finite-difference acceleration.')
    parser.add_argument('--imu-proxy-views', choices=('offset_aug_only', 'all'), default='offset_aug_only', help='Which training records receive the offset-aware IMU proxy loss.')
    parser.add_argument('--imu-proxy-gravity-mode', choices=('none', 'plus_g', 'minus_g'), default='none', help='Gravity convention applied before comparing finite-difference acceleration to aM.')
    parser.add_argument('--loss-temporal-mode', choices=('full_window', 'last', 'recent_l4'), default='full_window', help='Which predicted frames receive the main L4 loss. full_window is the historical behavior; last uses only the last output frame; recent_l4 uses the last --recent-loss-frames frames.')
    parser.add_argument('--recent-loss-frames', type=int, default=4, help='Number of trailing frames used by --loss-temporal-mode recent_l4. Automatically clipped to available window length.')
    parser.add_argument('--recent-loss-weighting', choices=('uniform', 'ramp'), default='uniform', help='Temporal weights for selected recent frames.')
    parser.add_argument('--save-best-by', choices=('motion_eval', 'loss'), default='motion_eval', help='Checkpoint selection: full MotionEvaluator score or lightweight loss.')
    parser.add_argument('--no-epoch-motion-eval', action='store_true', help='Disable epoch-level MotionEvaluator even when validate_every would trigger it.')
    parser.add_argument('--validate-every', type=int, default=1)
    args = parser.parse_args()
    if args.tail_length != 4:
        raise ValueError('Only tail_length=4 is approved for the current L4 method.')
    if args.control_stride < 1:
        raise ValueError('--control-stride must be >= 1.')
    if args.loss_mode in ('imu_proxy', 'contact_imu') and not args.enable_imu_proxy_training:
        if args.imu_orientation_weight or args.imu_acc_weight or args.imu_gyro_weight:
            raise ValueError('IMU proxy weights require --enable-imu-proxy-training after coordinate audit approval.')
    if args.pair_consistency_weight != 0.0:
        raise ValueError('Pair consistency loss is reserved but not implemented; keep --pair-consistency-weight 0.0.')
    if args.imu_proxy_mode == 'none' and args.imu_proxy_weight != 0.0:
        raise ValueError('--imu-proxy-weight requires --imu-proxy-mode offset_acc_fd.')
    if args.imu_proxy_mode == 'offset_acc_fd' and args.imu_proxy_weight < 0.0:
        raise ValueError('--imu-proxy-weight must be non-negative.')
    if args.recent_loss_frames <= 0:
        raise ValueError('--recent-loss-frames must be positive.')
    if args.l4_imu_field_prefix is None:
        args.l4_imu_field_prefix = imu_input_mode_to_field_prefix(args.imu_input_mode)
    else:
        legacy_to_mode = {'original': 'official', 'l4': 'processed', 'auto': 'auto'}
        args.imu_input_mode = legacy_to_mode[args.l4_imu_field_prefix]
    train_records, train_manifest = load_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, max_sequences=args.max_val_sequences)
    weights = default_weights(args.disable_root_velocity_loss)
    optional_weight_args = {
        'pose_geodesic': args.pose_geodesic_weight,
        'q_body': args.q_body_weight,
        'q_root_ori': args.q_root_ori_weight,
        'baseline_body': args.baseline_body_weight,
        'baseline_root_ori': args.baseline_root_ori_weight,
        'root_translation': args.root_translation_weight,
        'root_velocity': args.root_velocity_weight,
        'root_acc_smooth': args.root_acc_smooth_weight,
        'baseline_velocity': args.baseline_velocity_weight,
        'residual_prior': args.residual_prior_weight,
        'tail_update_prior': args.tail_update_prior_weight,
        'fk_joint_rootrel': args.fk_joint_rootrel_weight,
        'fk_joint_global': args.fk_joint_global_weight,
        'fk_joint_baseline_rootrel': args.fk_joint_baseline_rootrel_weight,
        'local_pose_preserve_body': args.local_pose_preserve_body_weight,
        'local_pose_preserve_root_ori': args.local_pose_preserve_root_ori_weight,
        'rootrel_joint_velocity': args.rootrel_joint_velocity_weight,
        'qdot_consistency_body': args.qdot_consistency_body_weight,
        'qdot_consistency_all': args.qdot_consistency_all_weight,
        'qdot_body_target_fd': args.qdot_body_target_fd_weight,
        'qdot_consistency_root': args.qdot_consistency_root_weight,
        'control_point_prior': args.control_point_prior_weight,
        'trajectory_q_body_prior': args.trajectory_q_body_prior_weight,
        'qddot_body_smooth': args.qddot_body_smooth_weight,
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
    if args.imu_proxy_mode == 'offset_acc_fd':
        weights['imu_proxy_offset_acc'] = args.imu_proxy_weight
    compute_imu_proxy = args.loss_mode in ('imu_proxy', 'contact_imu')
    args.compute_imu_proxy = compute_imu_proxy
    effective_rnn_init_mode = args.rnn_init_mode or ('offset_only' if args.offset_conditioning == 'hidden_init' else 'none')
    if args.rnn_init_mode and args.rnn_init_mode != 'none':
        args.offset_conditioning = 'hidden_init'
    args.effective_rnn_init_mode = effective_rnn_init_mode

    model_cls = StreamingTailUpdateSO3State if args.model_type == 'k2_so3curve_v1' else StreamingTailUpdateQState
    model_kwargs = dict(
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
    )
    if model_cls is StreamingTailUpdateQState:
        model_kwargs['control_stride'] = args.control_stride
    elif args.control_stride != 1:
        raise ValueError('--control-stride is only implemented for euler_q75_l4 / K2 q75 models.')
    model = model_cls(**model_kwargs).to(DEVICE)
    model.imu_input_mode = args.imu_input_mode
    model.l4_imu_field_prefix = args.l4_imu_field_prefix
    model.fast_training_so3 = args.model_type == 'k2_so3curve_v1'
    model.loss_temporal_mode = args.loss_temporal_mode
    model.recent_loss_frames = args.recent_loss_frames
    model.recent_loss_weighting = args.recent_loss_weighting
    checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
    init_checkpoint_load = load_compatible_state(
        model,
        args.init_checkpoint,
        allow_partial=args.allow_partial_init,
        skip_prefixes=('new_control.', 'tail_delta.') if args.skip_output_head_init else (),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    l4_imu_contract = l4_imu_field_contract(train_records, args.l4_imu_field_prefix)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
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
        'imu_input_mode_contract': {
            'imu_input_mode': args.imu_input_mode,
            'field_prefix': args.l4_imu_field_prefix,
            'official': 'frame input and offset_firstframe init use record["aM"], record["wM"], record["RMB"].',
            'processed': 'frame input and offset_firstframe init use record["l4_aM"], record["l4_wM"], record["l4_RMB"]; all three fields must exist.',
            'auto': 'frame input and offset_firstframe init prefer l4_* if present, otherwise fall back to official fields.',
            'synchronizes_first_frame_init': True,
        },
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
            'offset_aware_imu_proxy_loss': {
                'mode': args.imu_proxy_mode,
                'weight': args.imu_proxy_weight,
                'views': args.imu_proxy_views,
                'gravity_mode': args.imu_proxy_gravity_mode,
                'formula': 'p_S(t)=p_J(t)+R_J(t)@r_JS; a_proxy=(p_S[t+1]-2*p_S[t]+p_S[t-1]) / dt^2',
                'interior_frames_only': True,
                'imu_joints': list(IMU_JOINTS),
            },
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
            'loss_temporal_contract': {
                'mode': args.loss_temporal_mode,
                'recent_loss_frames': args.recent_loss_frames,
                'recent_loss_weighting': args.recent_loss_weighting,
                'historical_behavior': 'full_window: existing l4_train_loss_ablation.py supervised all frames in the sliced training window, not only the final frame.',
                'recent_l4_behavior': 'selects the last min(recent_loss_frames, available_window_frames) predicted frames and averages the main pose/q/FK/residual losses over them.',
                'last_behavior': 'selects only the final predicted frame for main pose/q/FK/residual losses; qdot/qddot become zero when the selected length is too short.',
                'control_stride': args.control_stride,
                'control_stride_behavior': '1 is historical per-frame control points. 2 updates the recurrent state every frame but generates a new spline control point only every two frames; intermediate frames are causal half-step spline evaluations using ghost extrapolation.',
            },
            'pose_position_velocity_consistency_losses': {
                'local_pose_preserve_body': 'geodesic(pred refined pose body joints, baseline/K2 pose body joints), root translation excluded',
                'local_pose_preserve_root_ori': 'geodesic(pred refined root orientation, baseline/K2 root orientation), small-weight optional',
                'rootrel_joint_velocity': 'SmoothL1((FK_rootrel(pred)[t]-FK_rootrel(pred)[t-1])/dt, (FK_rootrel(gt)[t]-FK_rootrel(gt)[t-1])/dt)',
                'fk_joint_global': 'SmoothL1(FK(pred pose, pred root translation), FK(gt pose, gt root translation)); default-off for first sweep unless explicitly weighted',
                'qdot_consistency_body': 'SmoothL1(wrapped(q_body[t]-q_body[t-1]), dt*qdot_body[t])',
                'qdot_consistency_all': 'SmoothL1(root translation/orientation/body q step, dt*qdot[t]); intended only with small weights',
                'qdot_consistency_root': 'SmoothL1(concat(root_translation_step, wrapped(root_orientation_step)), dt*qdot_root[t])',
                'qdot_body_target_fd': 'SmoothL1(dt*qdot_curve_body[t], wrapped(q_gt_body[t]-q_gt_body[t-1])); default-off spline derivative target consistency',
                'control_point_prior': 'mean((C_pred-C_base)^2) over the current causal control buffer; default-off nominal-trajectory anchoring',
                'trajectory_q_body_prior': 'mean(wrapped(q_curve_body-q_base_body)^2) on selected frames; default-off root-free waypoint anchoring',
                'qddot_body_smooth': 'mean(qddot_curve_body^2) on selected frames; default-off small-weight acceleration smoothness',
                'root_translation': 'SmoothL1(q_curve[:3], q_gt[:3]) on selected frames; default-off root branch position term',
                'root_acc_smooth': 'mean((q_root[t+1]-2*q_root[t]+q_root[t-1])^2); default-off small root smoothness term',
            },
        },
        'init_checkpoint_load': init_checkpoint_load,
        'loss_mode': args.loss_mode,
        'checkpoint_selection': (
            'lowest lightweight cache_validation total loss; no epoch-level MotionEvaluator'
            if args.save_best_by == 'loss'
            else 'lowest full-validation MotionEvaluator score'
        ),
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
            'offset_acc_fd_enabled': args.imu_proxy_mode == 'offset_acc_fd',
            'offset_acc_fd_weight': args.imu_proxy_weight,
            'offset_acc_fd_views': args.imu_proxy_views,
            'offset_acc_fd_gravity_mode': args.imu_proxy_gravity_mode,
            'dt': DT,
            'gravity_model_frame': [0.0, -9.8, 0.0],
            'imu_vertices': list(IMU_VERTICES),
            'imu_orientation_joints': list(IMU_JOINTS),
            'offset_acc_fd_note': 'Formal new proxy loss uses joint-local offset_r/r_JS, not SMPL vertices and not generic accelerometer bias.',
            'legacy_note': 'Without --enable-imu-proxy-training, legacy IMU proxy terms are diagnostics only and all legacy IMU proxy weights remain zero.',
        },
        'epochs': [],
        'best': None,
        'best_loss': None,
        'status': 'running',
    }
    print(json.dumps({'l4_imu_field_contract': l4_imu_contract}, indent=2), flush=True)
    best_score = None
    best_loss_value = None
    step = 0
    with log_path.open('w') as log_file:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_loss, train_rows, step = train_epoch(model, train_records, optimizer, weights, args, step)
            cache_validation = cache_eval(
                model,
                val_records,
                weights,
                compute_imu_proxy,
                imu_proxy_mode=args.imu_proxy_mode,
                imu_proxy_views=args.imu_proxy_views,
                imu_proxy_gravity_mode=args.imu_proxy_gravity_mode,
            )
            physics_validation = None
            validation_rows = []
            validation_score = None
            if (not args.no_epoch_motion_eval) and epoch % args.validate_every == 0:
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
            loss_selection = cache_validation['loss']['loss'] if cache_validation['num_sequences'] else train_loss['loss']
            improved_loss = best_loss_value is None or loss_selection < best_loss_value
            if args.save_best_by == 'loss' and improved_loss:
                best_loss_value = loss_selection
                save_checkpoint(
                    output_dir / 'best_loss.pt',
                    model,
                    optimizer,
                    args,
                    epoch,
                    step,
                    validation_score,
                    weights,
                    'lowest lightweight cache_validation total loss' if cache_validation['num_sequences'] else 'lowest train total loss',
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
                'loss_selection': loss_selection,
                'best_loss_value': best_loss_value,
                'improved_best_loss': args.save_best_by == 'loss' and improved_loss,
            }
            if physics_validation is not None:
                epoch_record['physics_validation_summary'] = summarize_metrics(physics_validation)
            result['epochs'].append(epoch_record)
            if improved:
                result['best'] = epoch_record
            if args.save_best_by == 'loss' and improved_loss:
                result['best_loss'] = epoch_record
            log_file.write(json.dumps(epoch_record) + '\n')
            log_file.flush()
            result_path.write_text(json.dumps(result, indent=2))
            print(
                f"epoch={epoch} train_loss={train_loss.get('loss', train_loss.get('total', 0.0)):.6g} "
                f"loss_select={loss_selection:.6g} "
                f"val_score={validation_score if validation_score is not None else 'none'} "
                f"best={best_score if best_score is not None else 'none'} "
                f"best_loss={best_loss_value if best_loss_value is not None else 'none'} "
                f"seconds={epoch_record['epoch_wall_seconds']:.2f}",
                flush=True,
            )
    result['status'] = 'completed'
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'result_path': str(result_path),
        'log_path': str(log_path),
        'best_checkpoint': str(output_dir / 'best.pt') if result['best'] else None,
        'best_loss_checkpoint': str(output_dir / 'best_loss.pt') if result['best_loss'] else None,
        'best_epoch': result['best']['epoch'] if result['best'] else None,
        'best_loss_epoch': result['best_loss']['epoch'] if result['best_loss'] else None,
        'status': result['status'],
    }, indent=2))


if __name__ == '__main__':
    main()
