import argparse
import json
import math
from pathlib import Path

import torch

from k2_so3_curve import (
    SO3CurveStateDecoder,
    StreamingTailUpdateSO3State,
    pose_tran_to_so3_state,
    so3_state_to_euler_q75,
    so3_state_to_pose_tran,
)
from l4_train_diverse_short import DEVICE, load_records, rotation_geodesic
from l4_train_loss_ablation import (
    default_weights,
    load_compatible_state,
    pose_velocity_loss,
    run_cached_sequence,
    slice_record,
)


def tensor_stats(x):
    x = x.detach().float().reshape(-1)
    return {
        'mean': float(x.mean()) if x.numel() else 0.0,
        'max': float(x.max()) if x.numel() else 0.0,
        'min': float(x.min()) if x.numel() else 0.0,
        'rms': float(x.square().mean().sqrt()) if x.numel() else 0.0,
        'finite': bool(torch.isfinite(x).all()),
    }


def conversion_smoke(record, frames):
    record = slice_record(record, 0, frames)
    pose = record['pose_prephysics'] if 'pose_prephysics' in record else record['pose_gt']
    tran = record['q75_prephysics'][:, :3]
    q_so3 = pose_tran_to_so3_state(pose, tran)
    pose_rt, tran_rt = so3_state_to_pose_tran(q_so3)
    q_euler_rt = so3_state_to_euler_q75(q_so3)
    decoder = SO3CurveStateDecoder()
    decoded = decoder(q_so3.unsqueeze(0), return_derivatives=True)
    geo = rotation_geodesic(pose_rt.to(DEVICE), pose.to(DEVICE)).detach().cpu()
    euler_pose_rt, _ = so3_state_to_pose_tran(pose_tran_to_so3_state(pose_rt, tran_rt))
    euler_geo = rotation_geodesic(euler_pose_rt.to(DEVICE), pose.to(DEVICE)).detach().cpu()
    rotvec_norm = q_so3[:, 3:].reshape(-1, 3).norm(dim=-1)
    return {
        'sequence': record['name'],
        'frames': int(q_so3.shape[0]),
        'q_so3_shape': list(q_so3.shape),
        'decoder_q_so3_shape': list(decoded['q_so3'].shape),
        'decoder_euler_q75_shape': list(decoded['euler_q75'].shape),
        'decoder_qdot_so3_shape': list(decoded['qdot_so3'].shape),
        'decoder_qddot_so3_shape': list(decoded['qddot_so3'].shape),
        'decoder_angular_velocity_shape': list(decoded['angular_velocity'].shape),
        'rotvec_norm': tensor_stats(rotvec_norm),
        'pose_roundtrip_geodesic_rad': tensor_stats(geo),
        'euler_compat_geodesic_rad': tensor_stats(euler_geo),
        'tran_roundtrip_error': tensor_stats((tran_rt - tran).norm(dim=-1)),
        'euler_q75_finite': bool(torch.isfinite(q_euler_rt).all()),
        'qdot_so3_norm': tensor_stats(decoded['qdot_so3'].norm(dim=-1)),
        'qddot_so3_norm': tensor_stats(decoded['qddot_so3'].norm(dim=-1)),
        'angular_velocity_norm': tensor_stats(decoded['angular_velocity'].norm(dim=-1)),
        'angular_acceleration_norm': tensor_stats(decoded['angular_acceleration'].norm(dim=-1)),
        'all_finite': bool(
            torch.isfinite(q_so3).all()
            and torch.isfinite(decoded['euler_q75']).all()
            and torch.isfinite(decoded['qdot_so3']).all()
            and torch.isfinite(decoded['qddot_so3']).all()
            and torch.isfinite(decoded['angular_velocity']).all()
            and torch.isfinite(decoded['angular_acceleration']).all()
        ),
    }


def one_batch_train_smoke(record, frames, init_checkpoint=None):
    record = slice_record(record, 0, frames)
    model = StreamingTailUpdateSO3State(
        hidden_size=256,
        residual_scale=0.005,
        velocity_residual_scale=0.0,
        pose_input_mode='rot6d',
        offset_conditioning='hidden_init',
        rnn_init_mode='offset_firstframe',
        offset_init_scale=0.1,
    ).to(DEVICE)
    model.fast_training_so3 = True
    load_info = None
    if init_checkpoint:
        load_info = load_compatible_state(model, init_checkpoint, allow_partial=True)
        if load_info is not None:
            load_info['path'] = str(load_info.get('path', ''))
    model.l4_imu_field_prefix = 'original'
    model.loss_temporal_mode = 'recent_l4'
    model.recent_loss_frames = 4
    model.recent_loss_weighting = 'uniform'
    weights = default_weights(disable_root_velocity_loss=False)
    output = run_cached_sequence(model, record)
    loss, components = pose_velocity_loss(
        output,
        record,
        weights,
        compute_imu_proxy=False,
        imu_proxy_mode='none',
        loss_temporal_mode='recent_l4',
        recent_loss_frames=4,
        recent_loss_weighting='uniform',
    )
    loss.backward()
    grad_sq = torch.zeros((), device=DEVICE)
    grad_finite = True
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_finite = grad_finite and bool(torch.isfinite(parameter.grad).all())
            grad_sq = grad_sq + parameter.grad.detach().square().sum()
    grad_norm = float(torch.sqrt(grad_sq).detach().cpu())
    return {
        'sequence': record['name'],
        'frames': int(record['q75_prephysics'].shape[0]),
        'loss': float(loss.detach().cpu()),
        'loss_finite': bool(torch.isfinite(loss).item()),
        'grad_finite': grad_finite,
        'grad_norm': grad_norm if math.isfinite(grad_norm) else None,
        'q_pred_shape': list(output['q_pred'].shape),
        'qdot_pred_shape': list(output['qdot_pred'].shape),
        'qddot_pred_shape': list(output['qddot_pred'].shape),
        'q_pred_finite': bool(torch.isfinite(output['q_pred']).all()),
        'qdot_pred_finite': bool(torch.isfinite(output['qdot_pred']).all()),
        'qddot_pred_finite': bool(torch.isfinite(output['qddot_pred']).all()),
        'q_residual_norm_mean': float(output['q_residual'].norm(dim=-1).mean().detach().cpu()),
        'init_checkpoint_load': load_info,
        'component_sample': {key: float(value.detach().cpu()) for key, value in components.items() if key in (
            'loss_temporal_num_frames',
            'pose_geodesic',
            'q_body',
            'baseline_body',
            'qdot',
            'qddot',
            'fk_joint_rootrel',
            'residual_prior',
            'tail_update_prior',
        )},
        'all_finite': bool(torch.isfinite(loss).item() and grad_finite and torch.isfinite(output['q_pred']).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--sequence', default='')
    parser.add_argument('--init-checkpoint', type=Path, default=None)
    args = parser.parse_args()
    records, manifest = load_records(args.cache)
    if args.sequence:
        records = [record for record in records if record['name'] == args.sequence]
        if not records:
            raise KeyError(f'No sequence {args.sequence!r} in {args.cache}.')
    record = records[0]
    result = {
        'cache': str(args.cache),
        'manifest': manifest,
        'conversion': conversion_smoke(record, args.frames),
        'one_batch_train': one_batch_train_smoke(record, args.frames, args.init_checkpoint),
    }
    result['status'] = 'ok' if result['conversion']['all_finite'] and result['one_batch_train']['all_finite'] else 'failed'
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'output_json': str(args.output_json),
        'status': result['status'],
        'conversion_all_finite': result['conversion']['all_finite'],
        'one_batch_all_finite': result['one_batch_train']['all_finite'],
        'loss': result['one_batch_train']['loss'],
    }, indent=2))


if __name__ == '__main__':
    main()
