import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files
from net import GPNet
from pl_curve import PLCurveModule, normalize_gravity, pl_init_feature_from_pose, pl_input_feature, pl_target_from_pose, split_pl_feature
from ik1_curve import IK1_NONLEAF_PRJ_INDICES


@torch.no_grad()
def build_pl_curve(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = checkpoint.get('config', {})
    model = PLCurveModule(
        init_size=int(cfg.get('init_size', 18)),
        hidden_size=int(cfg.get('hidden_size', 512)),
        tail_update=int(cfg.get('tail_length', 4)),
        residual_scale=float(cfg.get('residual_scale', 0.005)),
        dropout=float(cfg.get('dropout', 0.4)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, cfg


def sequence_pl_inputs(record):
    return torch.stack([
        pl_input_feature(record['aM'][i], record['wM'][i], record['RMB'][i])
        for i in range(record['aM'].shape[0])
    ]).float()


@torch.no_grad()
def sequence_base_pl(gpnet, pl_input, init_output):
    return gpnet.plnet([(pl_input.to(DEVICE), init_output.to(DEVICE))])[0].detach().cpu()


@torch.no_grad()
def run_pl_curve(pl_curve, pl_input, pl_base, init_output=None, init_feature=None):
    if init_feature is not None:
        pl_curve.reset_stream(init_feature=init_feature.to(DEVICE))
    else:
        pl_curve.reset_stream(init_output=init_output.to(DEVICE))
    pl_outputs, pl_dots, pl_ddots, control_tails = [], [], [], []
    for frame_idx in range(pl_input.shape[0]):
        out = pl_curve.step(pl_input[frame_idx].to(DEVICE), pl_base[frame_idx].to(DEVICE))
        control = pl_curve.control_buffer[:, -4:, :].detach()
        if control.shape[1] < 4:
            pad = control[:, :1].expand(-1, 4 - control.shape[1], -1)
            control = torch.cat((pad, control), dim=1)
        control_tails.append(control[0].cpu())
        pl_outputs.append(out['pl_t'][0].detach().cpu())
        pl_dots.append(out['pldot_t'][0].detach().cpu())
        pl_ddots.append(out['plddot_t'][0].detach().cpu())
    return {
        'pl_curve': torch.stack(pl_outputs),
        'pldot': torch.stack(pl_dots),
        'plddot': torch.stack(pl_ddots),
        'pl_control_tail': torch.stack(control_tails),
    }


@torch.no_grad()
def ik1_base_and_ik2(gpnet, pl_input, pl_curve_output):
    RRB0, gR0 = split_pl_feature(pl_input.to(DEVICE))
    pRB = pl_curve_output['pl_curve'].to(DEVICE)[:, :15]
    gR1 = normalize_gravity(pl_curve_output['pl_curve'].to(DEVICE))[:, 15:]
    RRB_after_pl = art.math.from_to_rotation_matrix(gR0, gR1).unsqueeze(1).matmul(RRB0)
    ik1_feature = torch.cat((
        RRB_after_pl.flatten(1),
        gR1,
        pl_curve_output['pl_control_tail'].to(DEVICE).flatten(1),
    ), dim=-1)
    official_ik1_input = torch.cat((RRB_after_pl.flatten(1), gR1, pRB), dim=-1)
    ik1_base = gpnet.iknet.net1([official_ik1_input])[0]
    gR2 = normalize_gravity(ik1_base)[:, 69:]
    RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).unsqueeze(1).matmul(RRB_after_pl)
    ik2_input = torch.cat((RRB_after_ik1.flatten(1), gR2, ik1_base[:, :69]), dim=-1)
    ik2_base = gpnet.iknet.net2([ik2_input])[0]
    return {
        'ik1_input': ik1_feature.detach().cpu(),
        'ik1_base': normalize_gravity(ik1_base.detach().cpu()),
        'ik2_base': ik2_base.detach().cpu(),
    }


@torch.no_grad()
def ik1_targets_from_pose(pose, body_model):
    pose = pose.to(DEVICE)
    pose_body = pose.clone()
    pose_body[:, 0] = torch.eye(3, device=DEVICE)
    _, joints = body_model.forward_kinematics(pose_body)[:2]
    pRJ = joints[:, 1:].detach()
    pRJ_nonleaf = pRJ[:, IK1_NONLEAF_PRJ_INDICES].reshape(pose.shape[0], -1)
    gR2 = -pose[:, 0, :, 1]
    return pRJ.reshape(pose.shape[0], 69).cpu(), pRJ_nonleaf.cpu(), gR2.cpu()


def build_cache(input_cache, pl_checkpoint, output_dir, shard_size, max_sequences=None, start_sequence=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(input_cache)
    gpnet = GPNet().eval().to(DEVICE)
    for parameter in gpnet.parameters():
        parameter.requires_grad_(False)
    pl_curve, pl_config = build_pl_curve(pl_checkpoint)
    body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    joint_body_model = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    cache_files = []
    shard = {
        'name': [], 'ik1_input': [], 'ik1_target_nonleaf': [], 'ik1_target_full': [], 'ik1_target_gR2': [],
        'ik1_base': [], 'ik2_base': [], 'leaf_pRB': [], 'imu_acc_target': [], 'num_frames': [],
    }
    shard_idx = 0
    total_sequences = 0
    total_frames = 0
    seen_sequences = 0

    def flush():
        nonlocal shard, shard_idx
        if not shard['name']:
            return
        out = output_dir / f'ik1_curve_cache_shard{shard_idx:05d}.pt'
        torch.save(shard, out)
        cache_files.append({
            'path': str(out),
            'num_sequences': len(shard['name']),
            'num_frames': int(sum(shard['num_frames'])),
        })
        shard_idx += 1
        shard = {
            'name': [], 'ik1_input': [], 'ik1_target_nonleaf': [], 'ik1_target_full': [], 'ik1_target_gR2': [],
            'ik1_base': [], 'ik2_base': [], 'leaf_pRB': [], 'imu_acc_target': [], 'num_frames': [],
        }

    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            if seen_sequences < int(start_sequence):
                seen_sequences += 1
                continue
            seen_sequences += 1
            pose_gt = data['pose_gt'][seq_idx].float()
            record = {
                'aM': data['aM'][seq_idx].float(),
                'wM': data['wM'][seq_idx].float(),
                'RMB': data['RMB'][seq_idx].float(),
            }
            pl_input = sequence_pl_inputs(record)
            pl_target = normalize_gravity(pl_target_from_pose(pose_gt.to(DEVICE), body_model).float()).cpu()
            pl_base = sequence_base_pl(gpnet, pl_input, pl_target[0])
            pl_init = None
            if getattr(pl_curve, 'init_size', 18) != 18:
                if 'pl_init_feature' in data:
                    pl_init = data['pl_init_feature'][seq_idx].float()
                else:
                    if 'offset_r' not in data:
                        raise KeyError(f'{cache_file} has no offset_r field required for PL init feature.')
                    pl_init = pl_init_feature_from_pose(data['offset_r'][seq_idx].float(), pose_gt[0].float(), body_model)
            pl_curve_output = run_pl_curve(pl_curve, pl_input, pl_base, init_output=pl_target[0], init_feature=pl_init)
            base_outputs = ik1_base_and_ik2(gpnet, pl_input, pl_curve_output)
            target_full, target_nonleaf, target_gR2 = ik1_targets_from_pose(pose_gt, joint_body_model)
            imu_acc_target = pl_input[:, :15].reshape(pl_input.shape[0], 5, 3)
            tensors = [
                base_outputs['ik1_input'], target_nonleaf, target_full, target_gR2,
                base_outputs['ik1_base'], base_outputs['ik2_base'], pl_curve_output['pl_curve'][:, :15],
                imu_acc_target,
            ]
            if not all(torch.isfinite(t).all() for t in tensors):
                raise RuntimeError(f'Non-finite IK1 cache tensors at {name}.')
            shard['name'].append(name)
            shard['ik1_input'].append(base_outputs['ik1_input'])
            shard['ik1_target_nonleaf'].append(target_nonleaf)
            shard['ik1_target_full'].append(target_full)
            shard['ik1_target_gR2'].append(target_gR2)
            shard['ik1_base'].append(base_outputs['ik1_base'])
            shard['ik2_base'].append(base_outputs['ik2_base'])
            shard['leaf_pRB'].append(pl_curve_output['pl_curve'][:, :15])
            shard['imu_acc_target'].append(imu_acc_target)
            shard['num_frames'].append(int(pl_input.shape[0]))
            total_sequences += 1
            total_frames += int(pl_input.shape[0])
            if len(shard['name']) >= shard_size:
                flush()
            if total_sequences % 25 == 0:
                print(json.dumps({'processed_sequences': total_sequences, 'processed_frames': total_frames}), flush=True)
            if max_sequences is not None and total_sequences >= int(max_sequences):
                flush()
                manifest = {
                    'type': 'ik1_curve_cache_v1',
                    'source_cache': str(input_cache),
                    'source_manifest': source_manifest,
                    'pl_checkpoint': str(pl_checkpoint),
                    'pl_checkpoint_config': pl_config,
                    'cache_files': cache_files,
                    'num_sequences': total_sequences,
                    'num_frames': total_frames,
                    'leaf_prj_indices': [17, 18, 3, 4, 14],
                    'nonleaf_prj_indices': list(IK1_NONLEAF_PRJ_INDICES),
                    'fields': {
                        'ik1_input': '[T,120] RRB_after_pl[45]+gR1[3]+PL control tail[4,18]',
                        'ik1_target_nonleaf': '[T,54] GT root-relative non-leaf pRJ joints',
                        'ik1_target_full': '[T,69] GT root-relative pRJ joints',
                        'ik1_target_gR2': '[T,3] GT root gravity direction',
                        'ik1_base': '[T,72] frozen official IK1 output using PLCurve PL input',
                        'ik2_base': '[T,90] frozen official IK2 output using ik1_base',
                        'leaf_pRB': '[T,15] PLCurve decoded pRB inserted into leaf pRJ slots',
                        'imu_acc_target': '[T,5,3] official root-frame acceleration for first five IMUs from PL input',
                    },
                }
                manifest_path = output_dir / 'ik1_curve_cache_manifest.json'
                manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
                return manifest
    flush()
    manifest = {
        'type': 'ik1_curve_cache_v1',
        'source_cache': str(input_cache),
        'source_manifest': source_manifest,
        'pl_checkpoint': str(pl_checkpoint),
        'pl_checkpoint_config': pl_config,
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'leaf_prj_indices': [17, 18, 3, 4, 14],
        'nonleaf_prj_indices': list(IK1_NONLEAF_PRJ_INDICES),
        'fields': {
            'ik1_input': '[T,120] RRB_after_pl[45]+gR1[3]+PL control tail[4,18]',
            'ik1_target_nonleaf': '[T,54] GT root-relative non-leaf pRJ joints',
            'ik1_target_full': '[T,69] GT root-relative pRJ joints',
            'ik1_target_gR2': '[T,3] GT root gravity direction',
            'ik1_base': '[T,72] frozen official IK1 output using PLCurve PL input',
            'ik2_base': '[T,90] frozen official IK2 output using ik1_base',
            'leaf_pRB': '[T,15] PLCurve decoded pRB inserted into leaf pRJ slots',
            'imu_acc_target': '[T,5,3] official root-frame acceleration for first five IMUs from PL input',
        },
    }
    manifest_path = output_dir / 'ik1_curve_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Precompute IK1Curve input/target/base tensors.')
    parser.add_argument('--input-cache', type=Path, required=True)
    parser.add_argument('--pl-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--shard-size', type=int, default=100)
    parser.add_argument('--max-sequences', type=int, default=None)
    parser.add_argument('--start-sequence', type=int, default=0)
    args = parser.parse_args()
    manifest = build_cache(
        args.input_cache,
        args.pl_checkpoint,
        args.output_dir,
        args.shard_size,
        args.max_sequences,
        args.start_sequence,
    )
    print(json.dumps({
        'status': 'ok',
        'manifest': str(args.output_dir / 'ik1_curve_cache_manifest.json'),
        'num_sequences': manifest['num_sequences'],
        'num_frames': manifest['num_frames'],
    }, indent=2))


if __name__ == '__main__':
    main()
