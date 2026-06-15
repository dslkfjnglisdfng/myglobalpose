import argparse
import json
import os
from pathlib import Path

import torch

import articulate as art
from l4_q75_utils import pose_tran_to_q75, q75_to_pose_tran
from l4_train_diverse_short import DEVICE, load_cache_files
from net import GPNet
from pl_curve import causal_iir_lowpass_sequence


def selected_imu_fields(data, seq_idx):
    return data['aM'][seq_idx].float(), data['wM'][seq_idx].float(), data['RMB'][seq_idx].float()


def offset_for_record(data, seq_idx):
    for key in ('offset_r', 'imu_offset_r', 'r_JS'):
        if key in data and data[key]:
            return data[key][seq_idx].float()
    return torch.zeros(6, 3)


def pose_tran_q75(data, seq_idx):
    if 'pose_gt' in data and data['pose_gt']:
        pose = data['pose_gt'][seq_idx].float()
        tran = data['tran_gt'][seq_idx].float() if 'tran_gt' in data and data['tran_gt'] else torch.zeros(pose.shape[0], 3)
        q75 = pose_tran_to_q75(pose, tran)
        return pose, tran, q75
    if 'q75_gt' in data and data['q75_gt']:
        q75 = data['q75_gt'][seq_idx].float()
        pose, tran = q75_to_pose_tran(q75)
        return pose.float(), tran.float(), q75.float()
    raise KeyError('cache sequence has neither pose_gt nor q75_gt.')


def baseline_ik2_pose(net, ik2_input, R_t, gR0, gR2):
    x, net.ik2hc = net.iknet.net2.rnn(ik2_input.to(DEVICE).view(1, 1, -1), net.ik2hc)
    rrj = net.iknet.net2.linear2(x.squeeze())
    RRJ = art.math.r6d_to_rotation_matrix(rrj).cpu()
    glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
    glb_pose[:, net.j_reduce] = RRJ.view(1, 15, 3, 3)
    pose = net.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
    pose[list(net.j_ignore)] = torch.eye(3)
    root_pose = R_t[5].cpu().mm(art.math.from_to_rotation_matrix(gR2.cpu(), gR0.cpu()).squeeze())
    pose[0] = root_pose
    return pose, root_pose


@torch.no_grad()
def build_sequence(net, name, pose_gt, tran_gt, q75_gt, a, w, R, offset_r, cutoff_hz, filter_fs):
    net.rnn_initialize(pose_gt[0].to(DEVICE), offset_r=offset_r.to(DEVICE))
    smooth_a = causal_iir_lowpass_sequence(a, cutoff_hz=cutoff_hz, fs=filter_fs)
    features, roots, gR2_rows, teacher_pose = [], [], [], []
    for idx in range(pose_gt.shape[0]):
        a_t = a[idx].to(DEVICE)
        w_t = w[idx].to(DEVICE)
        R_t = R[idx].to(DEVICE)
        ik1 = net.forward_until_ik1(a_t, w_t, R_t)
        aRB_raw = a[idx].mm(R[idx, 5])
        aRB_smooth = smooth_a[idx].mm(R[idx, 5])
        aRB_residual = aRB_raw - aRB_smooth
        feature = torch.cat((
            ik1['ik2_teacher_input'].float(),
            aRB_smooth.reshape(-1).float(),
            aRB_residual.reshape(-1).float(),
        ))
        pose_teacher, root_pose = baseline_ik2_pose(
            net,
            ik1['ik2_teacher_input'],
            R_t,
            ik1['gR0'].to(DEVICE),
            ik1['gR2'].to(DEVICE),
        )
        features.append(feature.cpu())
        roots.append(root_pose.cpu())
        gR2_rows.append(ik1['gR2'].float())
        teacher_pose.append(pose_teacher.cpu())
    return {
        'name': name,
        'num_frames': int(pose_gt.shape[0]),
        'ik2_q75_input': torch.stack(features).float(),
        'q75_gt': q75_gt.float(),
        'pose_gt': pose_gt.float(),
        'tran_gt': tran_gt.float(),
        'baseline_root_pose': torch.stack(roots).float(),
        'baseline_gR2': torch.stack(gR2_rows).float(),
        'teacher_pose': torch.stack(teacher_pose).float(),
        'offset_r': offset_r.float(),
        'aM': a.float(),
        'wM': w.float(),
        'RMB': R.float(),
    }


def build_cache(input_cache, output_dir, shard_size=25, max_sequences=0, cutoff_hz=20.0, filter_fs=60.0):
    output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(input_cache)
    net = GPNet().eval().to(DEVICE)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    cache_files = []
    shard_idx = 0
    total_sequences = 0
    total_frames = 0

    def new_shard():
        return {
            'name': [],
            'num_frames': [],
            'ik2_q75_input': [],
            'q75_gt': [],
            'pose_gt': [],
            'tran_gt': [],
            'baseline_root_pose': [],
            'baseline_gR2': [],
            'teacher_pose': [],
            'offset_r': [],
            'aM': [],
            'wM': [],
            'RMB': [],
        }

    shard = new_shard()

    def flush():
        nonlocal shard, shard_idx
        if not shard['name']:
            return
        out = output_dir / f'ik2_q75_ctrl_cache_shard{shard_idx:05d}.pt'
        tmp_out = out.with_suffix(out.suffix + '.tmp')
        if tmp_out.exists():
            tmp_out.unlink()
        torch.save(shard, tmp_out)
        os.replace(tmp_out, out)
        cache_files.append({
            'path': str(out),
            'num_sequences': len(shard['name']),
            'num_frames': int(sum(shard['num_frames'])),
        })
        shard_idx += 1
        shard = new_shard()

    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            if max_sequences and total_sequences >= max_sequences:
                break
            pose_gt, tran_gt, q75_gt = pose_tran_q75(data, seq_idx)
            a, w, R = selected_imu_fields(data, seq_idx)
            offset_r = offset_for_record(data, seq_idx)
            record = build_sequence(net, name, pose_gt, tran_gt, q75_gt, a, w, R, offset_r, cutoff_hz, filter_fs)
            for key, value in record.items():
                shard[key].append(value)
            total_sequences += 1
            total_frames += int(record['num_frames'])
            if len(shard['name']) >= shard_size:
                flush()
            if total_sequences % 10 == 0:
                print(json.dumps({'processed_sequences': total_sequences, 'processed_frames': total_frames}), flush=True)
        if max_sequences and total_sequences >= max_sequences:
            break
    flush()
    manifest = {
        'type': 'ik2_q75_ctrl_cache_v1',
        'source_cache': str(input_cache),
        'source_manifest': source_manifest,
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'input_size': 153,
        'input_contract': 'baseline IK2-slot 117D from baseline PL+IK1 plus root-frame causal_iir20 acceleration smooth/residual 36D.',
        'output_contract': 'q75 control target; root translation/root orientation are baseline-overwritten during loss/eval.',
        'acc_smoothing': {
            'mode': 'causal_iir',
            'cutoff_hz': float(cutoff_hz),
            'filter_fs': float(filter_fs),
            'lookahead': 0,
        },
    }
    manifest_path = output_dir / 'ik2_q75_ctrl_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build IK2 q75 control module cache.')
    parser.add_argument('--input-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--shard-size', type=int, default=25)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--cutoff-hz', type=float, default=20.0)
    parser.add_argument('--filter-fs', type=float, default=60.0)
    args = parser.parse_args()
    manifest = build_cache(
        args.input_cache,
        args.output_dir,
        shard_size=args.shard_size,
        max_sequences=args.max_sequences,
        cutoff_hz=args.cutoff_hz,
        filter_fs=args.filter_fs,
    )
    print(json.dumps({
        'status': 'ok',
        'manifest': str(args.output_dir / 'ik2_q75_ctrl_cache_manifest.json'),
        'num_sequences': manifest['num_sequences'],
        'num_frames': manifest['num_frames'],
    }, indent=2))


if __name__ == '__main__':
    main()
