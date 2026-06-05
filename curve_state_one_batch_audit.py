import argparse
import json
from pathlib import Path

import torch

import articulate as art
from curve_state_decoder import CurveStateDecoder
from l4_q75_utils import pose_tran_to_q75
from net import GPNet


def load_first_record(manifest_path, sequence_index=0):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    shard_path = Path(manifest['cache_files'][0]['path'])
    if not shard_path.is_absolute():
        shard_path = Path.cwd() / shard_path
    shard = torch.load(shard_path, map_location='cpu')
    record = {}
    for key, value in shard.items():
        if isinstance(value, list):
            if len(value) == 0:
                record[key] = value
                continue
            if len(value) <= sequence_index:
                raise IndexError(f'{key} has {len(value)} records, requested {sequence_index}.')
            record[key] = value[sequence_index]
        else:
            record[key] = value
    return manifest, record


def finite(name, tensor):
    ok = bool(torch.isfinite(tensor).all().item())
    if not ok:
        raise RuntimeError(f'{name} contains non-finite values.')
    return ok


def teacher_ik2_from_ik1(model, ik1_features, R):
    x = ik1_features['ik2_teacher_input'].to(next(model.parameters()).device)
    x, model.ik2hc = model.iknet.net2.rnn(x.view(1, 1, -1), model.ik2hc)
    x = model.iknet.net2.linear2(x.squeeze())

    RRJ = art.math.r6d_to_rotation_matrix(x).cpu()
    glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
    glb_pose[:, model.j_reduce] = RRJ.view(1, 15, 3, 3)
    pose = model.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
    pose[model.j_ignore, ...] = torch.eye(3)
    pRJ = model.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
    gR2 = ik1_features['gR2'].to(R.device)
    gR0 = ik1_features['gR0'].to(R.device)
    pose[0] = R[5].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze()).cpu()
    return pose.detach().cpu(), RRJ.detach().cpu(), pRJ.detach().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest',
        default='data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json',
    )
    parser.add_argument('--sequence-index', type=int, default=0)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--output', default='data/experiments/curve_state_decoder_ik1_audit_v1/audit.json')
    args = parser.parse_args()

    manifest, record = load_first_record(args.manifest, args.sequence_index)
    frames = min(int(args.frames), int(record['num_frames']) if 'num_frames' in record else record['aM'].shape[0])
    if frames < 3:
        raise ValueError('Need at least 3 frames for qdot/qddot diagnostic.')

    model = GPNet()
    model.eval()
    model.rnn_initialize()

    teacher_pose, teacher_RRJ, teacher_pRJ, ik1_rows = [], [], [], []
    for t in range(frames):
        a = record['aM'][t].float()
        w = record['wM'][t].float()
        R = record['RMB'][t].float()
        ik1 = model.forward_until_ik1(a, w, R)
        pose_t, RRJ_t, pRJ_t = teacher_ik2_from_ik1(model, ik1, R)
        teacher_pose.append(pose_t)
        teacher_RRJ.append(RRJ_t)
        teacher_pRJ.append(pRJ_t)
        ik1_rows.append(ik1)

    teacher_pose = torch.stack(teacher_pose)
    if 'q75_prephysics' in record:
        teacher_tran = record['q75_prephysics'][:frames, :3].float()
    elif 'tran_gt' in record:
        teacher_tran = record['tran_gt'][:frames].float()
    else:
        teacher_tran = torch.zeros(frames, 3)
    teacher_q75 = pose_tran_to_q75(teacher_pose, teacher_tran, euler_seq=manifest.get('euler_seq', 'XYZ'))

    decoder = CurveStateDecoder(dt=1.0 / 60.0, euler_seq=manifest.get('euler_seq', 'XYZ'))
    decoded = decoder(teacher_q75.clone(), return_pose=True)
    pose_decoded = decoded['pose']
    pose_body = pose_decoded.clone()
    pose_body[:, 0] = torch.eye(3)
    glb_pose_body, joint_body = model.body_model.forward_kinematics(pose_body)
    RRJ_decoded = glb_pose_body[:, model.j_reduce].contiguous()
    pRJ_decoded = joint_body[:, 1:]
    aRB_decoded = torch.stack([record['aM'][t].float().mm(pose_decoded[t, 0]) for t in range(frames)])
    wRB_decoded = torch.stack([record['wM'][t].float().mm(pose_decoded[t, 0]) for t in range(frames)])

    checks = {
        'frames': frames,
        'sequence_name': record['name'],
        'manifest': args.manifest,
        'used_s5': False,
        'training_started': False,
        'official_forward_changed': False,
        'teacher_ik2_called_only_in_script': True,
        'shapes': {
            'pRB': list(ik1_rows[0]['pRB'].shape),
            'pRJ_ik1': list(ik1_rows[0]['pRJ_ik1'].shape),
            'ik2_teacher_input': list(ik1_rows[0]['ik2_teacher_input'].shape),
            'teacher_pose': list(teacher_pose.shape),
            'teacher_q75': list(teacher_q75.shape),
            'decoded_q75': list(decoded['q75'].shape),
            'decoded_qdot': list(decoded['qdot'].shape),
            'decoded_qddot': list(decoded['qddot'].shape),
            'decoded_pose': list(pose_decoded.shape),
            'RRJ_decoded': list(RRJ_decoded.shape),
            'pRJ_decoded': list(pRJ_decoded.shape),
            'aRB_decoded': list(aRB_decoded.shape),
            'wRB_decoded': list(wRB_decoded.shape),
        },
        'finite': {
            'teacher_q75': finite('teacher_q75', teacher_q75),
            'decoded_q75': finite('decoded_q75', decoded['q75']),
            'decoded_qdot': finite('decoded_qdot', decoded['qdot']),
            'decoded_qddot': finite('decoded_qddot', decoded['qddot']),
            'decoded_pose': finite('decoded_pose', pose_decoded),
            'RRJ_decoded': finite('RRJ_decoded', RRJ_decoded),
            'pRJ_decoded': finite('pRJ_decoded', pRJ_decoded),
            'aRB_decoded': finite('aRB_decoded', aRB_decoded),
            'wRB_decoded': finite('wRB_decoded', wRB_decoded),
        },
        'dependency_variables': [
            'PL-s1: aRB0/wRB0/RRB0/gR0 -> pRB/gR1/RRB_after_pl',
            'IK-s1: RRB_after_pl/gR1/pRB -> pRJ_ik1/gR2/RRB_after_ik1',
            'Teacher-only IK-s2: RRB_after_ik1/gR2/pRJ_ik1 -> teacher pose',
            'Curve decoder: copied teacher q75 controls -> q75/qdot/qddot',
            'Kinematic recompute: decoded pose -> RRJ/pRJ/aRB/wRB',
        ],
        'norms': {
            'teacher_q75_mean_abs': float(teacher_q75.abs().mean().item()),
            'decoded_qdot_mean_abs': float(decoded['qdot'].abs().mean().item()),
            'decoded_qddot_mean_abs': float(decoded['qddot'].abs().mean().item()),
            'teacher_vs_decoded_q75_max_abs': float((teacher_q75 - decoded['q75']).abs().max().item()),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as f:
        json.dump(checks, f, indent=2)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
