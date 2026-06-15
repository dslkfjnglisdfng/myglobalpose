import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_q75_utils import q75_to_pose_tran
from l4_train_diverse_short import DEVICE, load_cache_files
from net import GPNet
from newpose_ctrl import (
    STATE_DIM,
    fit_pose_controls,
    normalize_pose_state,
    padded_control_tail,
    pose_state_target_from_pose,
)
from pl_curve import build_pl_curve_model, control_fit_contract, normalize_gravity, pl_init_feature_from_pose, pl_input_feature, pl_target_from_pose, split_pl_feature


INPUT_SIZE = 174


def selected_imu_fields(data, seq_idx, mode):
    if mode == 'official':
        return data['aM'][seq_idx].float(), data['wM'][seq_idx].float(), data['RMB'][seq_idx].float()
    if mode == 'processed':
        return data['l4_aM'][seq_idx].float(), data['l4_wM'][seq_idx].float(), data['l4_RMB'][seq_idx].float()
    if mode == 'auto' and all(key in data for key in ('l4_aM', 'l4_wM', 'l4_RMB')):
        return data['l4_aM'][seq_idx].float(), data['l4_wM'][seq_idx].float(), data['l4_RMB'][seq_idx].float()
    return data['aM'][seq_idx].float(), data['wM'][seq_idx].float(), data['RMB'][seq_idx].float()


def sequence_pl_inputs(a, w, R):
    return torch.stack([pl_input_feature(a[i], w[i], R[i]) for i in range(a.shape[0])]).float()


def sequence_imu_feature(a, w, R):
    return torch.cat((a.reshape(a.shape[0], -1), w.reshape(w.shape[0], -1), R.reshape(R.shape[0], -1)), dim=-1).float()


def load_pl_curve(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get('config', {})
    model = build_pl_curve_model(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def offset_for_record(data, seq_idx):
    if 'offset_r' in data and data['offset_r']:
        return data['offset_r'][seq_idx].float()
    if 'imu_offset_r' in data and data['imu_offset_r']:
        return data['imu_offset_r'][seq_idx].float()
    if 'r_JS' in data and data['r_JS']:
        return data['r_JS'][seq_idx].float()
    return torch.zeros(6, 3)


@torch.no_grad()
def run_pl_stream(gpnet, pl_curve, pl_input, pose_gt, offset_r, body_model_pl, include_official_distill=False):
    gpnet.rnn_initialize(pose_gt[0].to(DEVICE), offset_r=offset_r.to(DEVICE))
    if getattr(pl_curve, 'init_size', 18) == 18:
        pl_target = normalize_gravity(pl_target_from_pose(pose_gt.to(DEVICE), body_model_pl).float()).cpu()
        pl_curve.reset_stream(init_output=pl_target[0].to(DEVICE))
    else:
        pl_init = pl_init_feature_from_pose(offset_r.float(), pose_gt[0].float(), body_model_pl)
        pl_curve.reset_stream(init_feature=pl_init.to(DEVICE))
    RRB0, gR0 = split_pl_feature(pl_input.to(DEVICE))
    pRB_list, gR1_list, tail_list, rrb_after_list, official_state_list = [], [], [], [], []
    for idx in range(pl_input.shape[0]):
        pl_in = pl_input[idx].to(DEVICE)
        base_pl, _ = gpnet._run_pl_stage(pl_in)
        curve = pl_curve.step(pl_in, base_pl)
        pl_out = normalize_gravity(curve['pl_t'][0])
        pRB = pl_out[:15]
        gR1 = pl_out[15:]
        RRB_after_pl = art.math.from_to_rotation_matrix(gR0[idx], gR1).matmul(RRB0[idx])
        tail = pl_curve.control_buffer[:, -4:, :].detach()
        if tail.shape[1] < 4:
            pad = tail[:, :1].expand(-1, 4 - tail.shape[1], -1)
            tail = torch.cat((pad, tail), dim=1)
        pRB_list.append(pRB.detach().cpu())
        gR1_list.append(gR1.detach().cpu())
        tail_list.append(tail[0].detach().cpu())
        rrb_after_list.append(RRB_after_pl.detach().cpu())
        if include_official_distill:
            ik1_base, gR2 = gpnet._run_ik1_stage(RRB_after_pl, gR1, pRB)
            RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB_after_pl)
            ik2_input = torch.cat((RRB_after_ik1.ravel(), gR2, ik1_base[:69]))
            x_ik2, gpnet.ik2hc = gpnet.iknet.net2.rnn(ik2_input.view(1, 1, -1), gpnet.ik2hc)
            ik2_base = gpnet.iknet.net2.linear2(x_ik2.squeeze())
            official_state_list.append(normalize_pose_state(torch.cat((ik2_base.detach().cpu(), gR2.detach().cpu()), dim=-1)))
    return {
        'pl_pRB_dec': torch.stack(pRB_list),
        'pl_gR1_dec': torch.stack(gR1_list),
        'pl_control_tail': torch.stack(tail_list),
        'RRB_after_pl': torch.stack(rrb_after_list),
        'official_ik2_state': torch.stack(official_state_list) if include_official_distill else None,
        'gR0': gR0.detach().cpu(),
    }


def build_newpose_feature(imu_feature, RRB_after_pl, pl_pRB, pl_gR1, pl_control_tail, gR0):
    last_control = normalize_gravity(pl_control_tail[:, -1])
    return torch.cat((
        imu_feature,
        RRB_after_pl.reshape(RRB_after_pl.shape[0], -1),
        pl_pRB,
        pl_gR1,
        last_control[:, :15],
        last_control[:, 15:18],
        gR0,
    ), dim=-1).float()


def build_cache(input_cache, output_dir, imu_input_mode, pl_checkpoint, shard_size=100, max_sequences=0, include_official_distill=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(input_cache)
    gpnet = GPNet().eval().to(DEVICE)
    for parameter in gpnet.parameters():
        parameter.requires_grad_(False)
    pl_curve, pl_config = load_pl_curve(pl_checkpoint)
    body_model_pl = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    body_model_pose = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    cache_files = []
    shard_idx = 0
    total_sequences = 0
    total_frames = 0

    def new_shard():
        return {
            'name': [], 'num_frames': [],
            'newpose_input': [], 'newpose_target': [], 'newpose_target_control_tail': [],
            'official_ik2_state': [], 'offset_r': [],
            'pose_gt': [], 'tran_gt': [], 'aM': [], 'wM': [], 'RMB': [],
            'pl_pRB_dec': [], 'pl_gR1_dec': [], 'pl_control_tail': [], 'RRB_after_pl': [], 'gR0': [],
        }

    shard = new_shard()

    def flush():
        nonlocal shard, shard_idx
        if not shard['name']:
            return
        out = output_dir / f'newpose_ctrl_cache_shard{shard_idx:05d}.pt'
        torch.save(shard, out)
        cache_files.append({'path': str(out), 'num_sequences': len(shard['name']), 'num_frames': int(sum(shard['num_frames']))})
        shard_idx += 1
        shard = new_shard()

    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            if 'pose_gt' in data:
                pose_gt = data['pose_gt'][seq_idx].float()
                tran_gt = data.get('tran_gt', [torch.zeros(pose_gt.shape[0], 3)])[seq_idx].float() if 'tran_gt' in data else torch.zeros(pose_gt.shape[0], 3)
            elif 'q75_gt' in data:
                pose_gt, tran_gt = q75_to_pose_tran(data['q75_gt'][seq_idx].float())
            else:
                raise KeyError(f'{cache_file} has no pose_gt or q75_gt fields.')
            a, w, R = selected_imu_fields(data, seq_idx, imu_input_mode)
            offset_r = offset_for_record(data, seq_idx)
            pl_input = sequence_pl_inputs(a, w, R)
            imu_feature = sequence_imu_feature(a, w, R)
            pl_stream = run_pl_stream(gpnet, pl_curve, pl_input, pose_gt, offset_r, body_model_pl, include_official_distill=include_official_distill)
            newpose_input = build_newpose_feature(
                imu_feature,
                pl_stream['RRB_after_pl'],
                pl_stream['pl_pRB_dec'],
                pl_stream['pl_gR1_dec'],
                pl_stream['pl_control_tail'],
                pl_stream['gR0'],
            )
            target = pose_state_target_from_pose(
                pose_gt.to(DEVICE),
                R.to(DEVICE),
                pl_stream['gR0'].to(DEVICE),
                body_model_pose,
                gpnet.j_reduce,
            ).detach().cpu()
            controls = fit_pose_controls(target.to(DEVICE)).detach().cpu()
            target_tail = torch.stack([padded_control_tail(controls, idx, 4) for idx in range(controls.shape[0])])
            official_ik2_state = pl_stream['official_ik2_state'] if pl_stream['official_ik2_state'] is not None else target.clone()
            tensors = [newpose_input, target, target_tail, official_ik2_state]
            if not all(torch.isfinite(t).all() for t in tensors):
                raise RuntimeError(f'Non-finite NewPose cache tensors at {name}.')
            shard['name'].append(name)
            shard['num_frames'].append(int(newpose_input.shape[0]))
            shard['newpose_input'].append(newpose_input.float())
            shard['newpose_target'].append(target.float())
            shard['newpose_target_control_tail'].append(target_tail.float())
            shard['official_ik2_state'].append(official_ik2_state.float())
            shard['offset_r'].append(offset_r.float())
            shard['pose_gt'].append(pose_gt.float())
            shard['tran_gt'].append(tran_gt.float())
            shard['aM'].append(a.float())
            shard['wM'].append(w.float())
            shard['RMB'].append(R.float())
            for key in ('pl_pRB_dec', 'pl_gR1_dec', 'pl_control_tail', 'RRB_after_pl', 'gR0'):
                shard[key].append(pl_stream[key].float())
            total_sequences += 1
            total_frames += int(newpose_input.shape[0])
            if len(shard['name']) >= shard_size:
                flush()
            if total_sequences % 25 == 0:
                print(json.dumps({'processed_sequences': total_sequences, 'processed_frames': total_frames}), flush=True)
            if max_sequences and total_sequences >= max_sequences:
                break
        if max_sequences and total_sequences >= max_sequences:
            break
    flush()
    manifest = {
        'type': 'newpose_ctrl_cache_v1',
        'source_cache': str(input_cache),
        'source_manifest': source_manifest,
        'imu_input_mode': imu_input_mode,
        'pl_checkpoint': str(pl_checkpoint),
        'pl_checkpoint_config': pl_config,
        'input_size': INPUT_SIZE,
        'state_dim': STATE_DIM,
        'tail_len': 4,
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'fields': {
            'newpose_input': '[T,174] official IMU[90]+RRB_after_pl[45]+pRB/gR1[18]+last PL control[18]+gR0[3]. offset_r is intentionally excluded from frame input.',
            'offset_r': '[6,3] r_JS used only for sequence initialization.',
            'newpose_target': '[T,93] RRJ_reduced_6d[90]+gR_pose[3].',
            'newpose_target_control_tail': '[T,4,93] fitted GT pose-control tail.',
            'official_ik2_state': '[T,93] frozen official IK1/IK2 state under same NewPL stream when include_official_distill=true; otherwise copied from GT target and distill weight should be 0. Not frame input.',
        },
        'target_control_fit_contract': control_fit_contract(),
        'include_official_distill': bool(include_official_distill),
        'coordinate_contract': {
            'offset_r': 'r_JS: IMU origin position relative to mapped joint J, expressed in joint-local coordinates. It is not used as frame-wise input.',
            'gR_pose': 'body/root-frame gravity direction satisfying pose_root = RMB_pelvis @ from_to_rotation(gR_pose, gR0).',
        },
    }
    manifest_path = output_dir / 'newpose_ctrl_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'ok', 'manifest': str(manifest_path), 'num_sequences': total_sequences, 'num_frames': total_frames}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build newpose_ctrl_v1 cache from raw GlobalPose cache and frozen NewPL stream.')
    parser.add_argument('--input-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pl-checkpoint', type=Path, required=True)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='official')
    parser.add_argument('--shard-size', type=int, default=100)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--include-official-distill', action='store_true')
    args = parser.parse_args()
    build_cache(
        args.input_cache,
        args.output_dir,
        args.imu_input_mode,
        args.pl_checkpoint,
        shard_size=args.shard_size,
        max_sequences=args.max_sequences,
        include_official_distill=args.include_official_distill,
    )


if __name__ == '__main__':
    main()
