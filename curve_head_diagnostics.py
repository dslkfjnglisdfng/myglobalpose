import argparse
import json
from pathlib import Path

import torch

import articulate as art
from curve_control_pose_head import CurveControlPoseHead, build_curve_frame_features
from curve_head_train import curve_loss
from curve_state_decoder import CurveStateDecoder
from l4_q75_utils import pose_tran_to_q75, q75_to_pose_tran
from l4_train_diverse_short import load_records
from net import GPNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_BODY_MODEL = None


class LossArgs:
    recent_loss_frames = 4
    q_body_weight = 1.0
    q_root_ori_weight = 0.5
    pose_geodesic_weight = 1.0
    fk_joint_rootrel_weight = 0.1
    qdot_smooth_weight = 1e-5
    qddot_smooth_weight = 1e-7
    control_update_reg_weight = 1e-3


def body_model():
    global _BODY_MODEL
    if _BODY_MODEL is None:
        _BODY_MODEL = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    return _BODY_MODEL


def rotation_geodesic(R_pred, R_target, eps=1e-6):
    rel = R_pred.transpose(-1, -2).matmul(R_target)
    trace = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


def tensor_stats(x):
    x = x.detach().float().cpu()
    return {
        'mean': float(x.mean()),
        'std': float(x.std(unbiased=False)),
        'min': float(x.min()),
        'max': float(x.max()),
        'rms': float(x.square().mean().sqrt()),
        'finite': bool(torch.isfinite(x).all().item()),
    }


def region_stats(q, prefix=''):
    return {
        f'{prefix}full': tensor_stats(q),
        f'{prefix}root_translation_0_3': tensor_stats(q[..., :3]),
        f'{prefix}root_orientation_3_6': tensor_stats(q[..., 3:6]),
        f'{prefix}body_6_75': tensor_stats(q[..., 6:]),
    }


def error_stats(pred, target):
    diff = pred.detach().to(target.device) - target.detach()
    return {
        'full_abs': tensor_stats(diff.abs()),
        'root_translation_abs': tensor_stats(diff[..., :3].abs()),
        'root_orientation_abs': tensor_stats(diff[..., 3:6].abs()),
        'body_abs': tensor_stats(diff[..., 6:].abs()),
    }


def delta_stats(q):
    if q.shape[0] < 2:
        return tensor_stats(q.new_zeros(1))
    return tensor_stats((q[1:] - q[:-1]).abs())


def qdot_qddot_stats(decoded):
    return {
        'qdot': tensor_stats(decoded['qdot']),
        'qddot': tensor_stats(decoded['qddot']),
    }


def load_head(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = checkpoint['model_config']
    head = CurveControlPoseHead(
        input_dim=cfg['input_dim'],
        hidden_size=cfg['hidden_size'],
        state_dim=cfg['state_dim'],
        residual_scale=cfg['residual_scale'],
        use_imu=cfg['use_imu'],
        use_feature_velocity=cfg['use_feature_velocity'],
        rnn_init_mode=cfg['rnn_init_mode'],
        freeze_root_translation=cfg['freeze_root_translation'],
        predict_root_orientation=cfg['predict_root_orientation'],
        offset_init_scale=cfg['offset_init_scale'],
    ).to(DEVICE)
    head.load_state_dict(checkpoint['model_state_dict'])
    return head, checkpoint


@torch.no_grad()
def teacher_ik2_from_ik1(model, ik1_features, R):
    x = ik1_features['ik2_teacher_input'].to(next(model.parameters()).device)
    x, model.ik2hc = model.iknet.net2.rnn(x.view(1, 1, -1), model.ik2hc)
    x = model.iknet.net2.linear2(x.squeeze())
    RRJ = art.math.r6d_to_rotation_matrix(x).cpu()
    glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
    glb_pose[:, model.j_reduce] = RRJ.view(1, 15, 3, 3)
    pose = model.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
    pose[model.j_ignore, ...] = torch.eye(3)
    gR2 = ik1_features['gR2'].to(R.device)
    gR0 = ik1_features['gR0'].to(R.device)
    pose[0] = R[5].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze()).cpu()
    return pose.detach().cpu()


def extract_features_and_teacher(record, start, length, use_imu=True, use_feature_velocity=True, include_teacher=True):
    end = min(record['aM'].shape[0], start + length)
    net = GPNet().to(DEVICE).eval()
    net.rnn_initialize(record['pose_gt'][start])
    features, feature_parts, gR2_rows, teacher_pose = [], [], [], []
    prev_pRJ = None
    for frame_idx in range(start, end):
        a = record['aM'][frame_idx].to(DEVICE)
        w = record['wM'][frame_idx].to(DEVICE)
        R = record['RMB'][frame_idx].to(DEVICE)
        ik1 = net.forward_until_ik1(a, w, R)
        feature = build_curve_frame_features(
            ik1['ik2_teacher_input'],
            ik1['pRJ_ik1'],
            record['aM'][frame_idx],
            record['wM'][frame_idx],
            record['RMB'][frame_idx],
            prev_pRJ_ik1=prev_pRJ,
            use_imu=use_imu,
            use_feature_velocity=use_feature_velocity,
        )
        velocity = torch.zeros_like(ik1['pRJ_ik1']) if prev_pRJ is None else ik1['pRJ_ik1'] - prev_pRJ
        feature_parts.append({
            'ik2_input': ik1['ik2_teacher_input'],
            'feature_velocity': velocity,
            'imu': torch.cat((record['aM'][frame_idx].reshape(-1), record['wM'][frame_idx].reshape(-1), record['RMB'][frame_idx].reshape(-1))),
        })
        features.append(feature)
        gR2_rows.append(ik1['gR2'])
        if include_teacher:
            teacher_pose.append(teacher_ik2_from_ik1(net, ik1, R))
        prev_pRJ = ik1['pRJ_ik1']
    return {
        'features': torch.stack(features).to(DEVICE),
        'feature_parts': feature_parts,
        'gR2': torch.stack(gR2_rows),
        'teacher_pose': torch.stack(teacher_pose) if include_teacher else None,
        'start': start,
        'end': end,
    }


def fk_errors(pose_pred, tran_pred, pose_gt, tran_gt):
    bm = body_model()
    pose_pred = pose_pred.to(DEVICE)
    pose_gt = pose_gt.to(DEVICE)
    tran_pred = tran_pred.to(DEVICE)
    tran_gt = tran_gt.to(DEVICE)
    grot_pred, joint_pred, verts_pred = bm.forward_kinematics(pose_pred, None, tran_pred, calc_mesh=True)
    grot_gt, joint_gt, verts_gt = bm.forward_kinematics(pose_gt, None, tran_gt, calc_mesh=True)
    rootrel_pred = joint_pred - joint_pred[:, :1]
    rootrel_gt = joint_gt - joint_gt[:, :1]
    per_joint = (rootrel_pred - rootrel_gt).norm(dim=-1).mean(dim=0)
    ranking = sorted(
        [{'joint': int(i), 'rootrel_error_m': float(v)} for i, v in enumerate(per_joint.detach().cpu())],
        key=lambda row: row['rootrel_error_m'],
        reverse=True,
    )[:10]
    return {
        'pose_geodesic_mean_rad': float(rotation_geodesic(pose_pred, pose_gt).mean().detach().cpu()),
        'root_geodesic_mean_rad': float(rotation_geodesic(pose_pred[:, :1], pose_gt[:, :1]).mean().detach().cpu()),
        'body_geodesic_mean_rad': float(rotation_geodesic(pose_pred[:, 1:], pose_gt[:, 1:]).mean().detach().cpu()),
        'fk_rootrel_joint_error_m': float((rootrel_pred - rootrel_gt).norm(dim=-1).mean().detach().cpu()),
        'global_joint_error_m': float((joint_pred - joint_gt).norm(dim=-1).mean().detach().cpu()),
        'mesh_error_m': float((verts_pred - verts_gt).norm(dim=-1).mean().detach().cpu()),
        'per_joint_rootrel_top10': ranking,
    }


def roundtrip_check(pose, tran):
    q = pose_tran_to_q75(pose.detach().cpu(), tran.detach().cpu())
    pose_recon, tran_recon = q75_to_pose_tran(q)
    geo = rotation_geodesic(pose_recon.to(DEVICE), pose.to(DEVICE))
    return {
        'q_stats': region_stats(q),
        'pose_recon_geodesic_mean_rad': float(geo.mean().detach().cpu()),
        'pose_recon_geodesic_max_rad': float(geo.max().detach().cpu()),
        'tran_recon_abs_max': float((tran_recon - tran.detach().cpu()).abs().max()),
    }


def feature_audit(features, feature_parts):
    parts = {
        'full_input': tensor_stats(features),
        'ik2_original_input': tensor_stats(torch.stack([p['ik2_input'] for p in feature_parts])),
        'feature_velocity': tensor_stats(torch.stack([p['feature_velocity'] for p in feature_parts])),
        'imu': tensor_stats(torch.stack([p['imu'] for p in feature_parts])),
    }
    return parts


def zero_feature_slices(features, use_velocity=True, use_imu=True):
    out = {}
    out['original'] = features
    if use_velocity:
        z = features.clone()
        z[:, 117:117 + 69] = 0.0
        out['feature_velocity_zeroed'] = z
    if use_imu:
        z = features.clone()
        imu_start = 117 + (69 if use_velocity else 0)
        z[:, imu_start:imu_start + 90] = 0.0
        out['imu_zeroed'] = z
    return out


@torch.no_grad()
def forward_head(head, decoder, features, offset_r):
    head.eval()
    out = head(features, offset_r=offset_r)
    decoded = decoder(out['control'], return_pose=True)
    return out, decoded


def loss_gradient_audit(head, decoder, record, window, start):
    head.train()
    data = extract_features_and_teacher(record, start, window, use_imu=head.use_imu, use_feature_velocity=head.use_feature_velocity, include_teacher=False)
    offset = record.get('offset_r')
    offset = None if offset is None else offset.view(1, 6, 3).to(DEVICE)
    out = head(data['features'], offset_r=offset)
    decoded = decoder(out['control'], return_pose=True)
    decoded['q75'].retain_grad()
    loss, losses = curve_loss(out, decoded, record, data['start'], data['end'], LossArgs())
    head.zero_grad(set_to_none=True)
    loss.backward()
    param_grad_sq = 0.0
    nonzero = 0
    for p in head.parameters():
        if p.grad is not None:
            g = float(p.grad.detach().square().sum().cpu())
            param_grad_sq += g
            if g > 0:
                nonzero += 1
    return {
        'loss_components': {k: float(v.detach().cpu()) for k, v in losses.items()},
        'q75_grad_norm': float(decoded['q75'].grad.detach().norm().cpu()),
        'head_param_grad_norm': param_grad_sq ** 0.5,
        'head_params_with_nonzero_grad': nonzero,
    }


def init_feature_ablation(checkpoint, features, offset_r):
    cfg = checkpoint['model_config']
    decoder = CurveStateDecoder().to(DEVICE).eval()
    heads = {}
    for mode in ('r_js_firstframe', 'none'):
        head = CurveControlPoseHead(
            input_dim=cfg['input_dim'],
            hidden_size=cfg['hidden_size'],
            state_dim=cfg['state_dim'],
            residual_scale=cfg['residual_scale'],
            use_imu=cfg['use_imu'],
            use_feature_velocity=cfg['use_feature_velocity'],
            rnn_init_mode=mode,
            freeze_root_translation=cfg['freeze_root_translation'],
            predict_root_orientation=cfg['predict_root_orientation'],
            offset_init_scale=cfg['offset_init_scale'],
        ).to(DEVICE)
        current = head.state_dict()
        compatible = {k: v for k, v in checkpoint['model_state_dict'].items() if k in current and current[k].shape == v.shape}
        current.update(compatible)
        head.load_state_dict(current)
        heads[mode] = head.eval()
    out_rjs, dec_rjs = forward_head(heads['r_js_firstframe'], decoder, features, offset_r)
    out_none, dec_none = forward_head(heads['none'], decoder, features, None)
    return {
        'h0_rjs_norm': float(heads['r_js_firstframe'].initial_hidden(features[:1], offset_r=offset_r).norm().detach().cpu()),
        'h0_none_norm': float(heads['none'].initial_hidden(features[:1], offset_r=None).norm().detach().cpu()),
        'q75_rjs_vs_none_abs': tensor_stats((dec_rjs['q75'] - dec_none['q75']).abs()),
    }


def diagnose_checkpoint(checkpoint_path, records, sequence_indices, windows):
    head, checkpoint = load_head(checkpoint_path)
    head.eval()
    decoder = CurveStateDecoder().to(DEVICE).eval()
    result = {
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_selection': checkpoint.get('selection'),
        'model_config': checkpoint.get('model_config'),
        'windows': [],
    }
    for seq_idx in sequence_indices:
        record = records[seq_idx]
        for window in windows:
            start = 0
            data = extract_features_and_teacher(record, start, window, use_imu=head.use_imu, use_feature_velocity=head.use_feature_velocity, include_teacher=True)
            offset = record.get('offset_r')
            offset = None if offset is None else offset.view(1, 6, 3).to(DEVICE)
            out, decoded = forward_head(head, decoder, data['features'], offset)
            q_pred = decoded['q75'].detach().cpu()
            q_gt = record['q75_gt'][data['start']:data['end']].detach().cpu()
            pose_gt = record['pose_gt'][data['start']:data['end']]
            tran_gt = record['tran_gt'][data['start']:data['end']]
            q_teacher = pose_tran_to_q75(data['teacher_pose'], record['q75_prephysics'][data['start']:data['end'], :3])
            pose_teacher_recon, tran_teacher_recon = q75_to_pose_tran(q_teacher)
            window_row = {
                'sequence': record['name'],
                'sequence_index': seq_idx,
                'window': int(data['end'] - data['start']),
                'pred_stats': region_stats(q_pred),
                'gt_stats': region_stats(q_gt),
                'teacher_stats': region_stats(q_teacher),
                'pred_vs_gt_error': error_stats(q_pred, q_gt),
                'teacher_vs_gt_error': error_stats(q_teacher, q_gt),
                'pred_delta': delta_stats(q_pred),
                'gt_delta': delta_stats(q_gt),
                'teacher_delta': delta_stats(q_teacher),
                'qdot_qddot': qdot_qddot_stats(decoded),
                'feature_audit': feature_audit(data['features'].detach().cpu(), data['feature_parts']),
                'roundtrip_gt': roundtrip_check(pose_gt, tran_gt),
                'roundtrip_teacher': roundtrip_check(data['teacher_pose'], record['q75_prephysics'][data['start']:data['end'], :3]),
                'teacher_recompute_errors': fk_errors(pose_teacher_recon, tran_teacher_recon, pose_gt, tran_gt),
                'pred_pose_errors': fk_errors(decoded['pose'].detach().cpu(), decoded['tran'].detach().cpu(), pose_gt, tran_gt),
            }
            if seq_idx == sequence_indices[0] and window == windows[-1]:
                window_row['loss_gradient_audit'] = loss_gradient_audit(head, decoder, record, window, start)
                window_row['init_feature_ablation'] = init_feature_ablation(checkpoint, data['features'], offset)
                ablations = {}
                for name, feat in zero_feature_slices(data['features'], head.use_feature_velocity, head.use_imu).items():
                    _, dec = forward_head(head, decoder, feat, offset)
                    ablations[name] = {
                        'q75_stats': region_stats(dec['q75'].detach().cpu()),
                        'abs_vs_original': tensor_stats((dec['q75'] - decoded['q75']).abs()) if name != 'original' else tensor_stats(torch.zeros_like(dec['q75'])),
                    }
                window_row['feature_ablation'] = ablations
            result['windows'].append(window_row)
    return result


def compact_summary(result):
    rows = result['windows']
    main = rows[0]
    return {
        'checkpoint': result['checkpoint'],
        'checkpoint_epoch': result['checkpoint_epoch'],
        'first_window': {
            'sequence': main['sequence'],
            'window': main['window'],
            'pred_full_std': main['pred_stats']['full']['std'],
            'gt_full_std': main['gt_stats']['full']['std'],
            'pred_body_std': main['pred_stats']['body_6_75']['std'],
            'gt_body_std': main['gt_stats']['body_6_75']['std'],
            'pred_delta_rms': main['pred_delta']['rms'],
            'gt_delta_rms': main['gt_delta']['rms'],
            'qdot_rms': main['qdot_qddot']['qdot']['rms'],
            'qddot_rms': main['qdot_qddot']['qddot']['rms'],
            'pose_geodesic_rad': main['pred_pose_errors']['pose_geodesic_mean_rad'],
            'teacher_pose_geodesic_rad': main['teacher_recompute_errors']['pose_geodesic_mean_rad'],
        },
    }


def main():
    parser = argparse.ArgumentParser(description='CurveHead P1 failure diagnostics; no training.')
    parser.add_argument('--best-checkpoint', default='data/experiments/CurveHead_P1_TC_smoke_train_v1/best_loss.pt')
    parser.add_argument('--last-checkpoint', default='data/experiments/CurveHead_P1_TC_smoke_train_v1/last.pt')
    parser.add_argument('--val-cache', default='data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json')
    parser.add_argument('--sequence-indices', default='0,1,2')
    parser.add_argument('--windows', default='8,16,61')
    parser.add_argument('--output', default='data/experiments/CurveHead_P1_TC_smoke_train_v1/curve_head_diagnostics.json')
    args = parser.parse_args()
    records, manifest = load_records(args.val_cache)
    seq_ids = [int(x) for x in args.sequence_indices.split(',') if x.strip()]
    windows = [int(x) for x in args.windows.split(',') if x.strip()]
    output = {
        'val_cache': args.val_cache,
        'sequence_indices': seq_ids,
        'windows': windows,
        's5_used': False,
        'training_started': False,
        'diagnostics': [],
    }
    for ckpt in (args.best_checkpoint, args.last_checkpoint):
        diag = diagnose_checkpoint(ckpt, records, seq_ids, windows)
        output['diagnostics'].append(diag)
    output['summary'] = [compact_summary(diag) for diag in output['diagnostics']]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output['summary'], indent=2))


if __name__ == '__main__':
    main()
