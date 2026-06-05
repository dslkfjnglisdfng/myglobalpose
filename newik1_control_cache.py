import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_q75_utils import q75_to_pose_tran
from l4_train_diverse_short import DEVICE, load_cache_files
from net import GPNet
from newik1_control_point import fit_ik1_controls, ik1_target_from_pose, normalize_ik1, padded_control_tail
from pl_curve import PLCurveModule, fit_uniform_cubic_spline_controls, normalize_gravity, pl_init_feature_from_pose, pl_input_feature, pl_target_from_pose, split_pl_feature


def selected_imu_fields(data, seq_idx, mode):
    if mode == 'official':
        return data['aM'][seq_idx].float(), data['wM'][seq_idx].float(), data['RMB'][seq_idx].float()
    if mode == 'processed':
        return data['l4_aM'][seq_idx].float(), data['l4_wM'][seq_idx].float(), data['l4_RMB'][seq_idx].float()
    if mode == 'auto' and all(key in data for key in ('l4_aM', 'l4_wM', 'l4_RMB')):
        return data['l4_aM'][seq_idx].float(), data['l4_wM'][seq_idx].float(), data['l4_RMB'][seq_idx].float()
    return data['aM'][seq_idx].float(), data['wM'][seq_idx].float(), data['RMB'][seq_idx].float()


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


def sequence_pl_inputs(a, w, R):
    return torch.stack([pl_input_feature(a[i], w[i], R[i]) for i in range(a.shape[0])]).float()


def build_gt_pl_controls(pl_target, tail_len):
    controls = fit_uniform_cubic_spline_controls(normalize_gravity(pl_target))
    return torch.stack([padded_control_tail(controls, idx, tail_len) for idx in range(controls.shape[0])])


@torch.no_grad()
def teacher_forced_sequence(gpnet, pl_input, pl_target, ik1_target, tail_len, init_pose):
    gpnet.rnn_initialize(init_pose)
    RRB0, gR0 = split_pl_feature(pl_input.to(DEVICE))
    pRB = pl_target.to(DEVICE)[:, :15]
    gR1 = normalize_gravity(pl_target.to(DEVICE))[:, 15:]
    RRB_after_pl = art.math.from_to_rotation_matrix(gR0, gR1).unsqueeze(1).matmul(RRB0)
    pl_control_tail = build_gt_pl_controls(pl_target, tail_len).to(DEVICE)
    feature = torch.cat((RRB_after_pl.flatten(1), pl_control_tail.flatten(1), gR1), dim=-1)
    base = []
    for idx in range(pl_input.shape[0]):
        out, _ = gpnet._run_ik1_stage(RRB_after_pl[idx], gR1[idx], pRB[idx])
        base.append(normalize_ik1(out.detach().cpu()))
    return feature.cpu(), torch.stack(base)


@torch.no_grad()
def pl1_streaming_sequence(gpnet, pl_curve, pl_input, pl_target, tail_len, init_pose, pl_init_feature=None):
    gpnet.rnn_initialize(init_pose)
    if getattr(pl_curve, 'init_size', 18) == 18:
        pl_curve.reset_stream(init_output=normalize_gravity(pl_target[0].to(DEVICE)))
    else:
        if pl_init_feature is None:
            raise ValueError(f'PL curve init_size={pl_curve.init_size} requires pl_init_feature.')
        pl_curve.reset_stream(init_feature=pl_init_feature.to(DEVICE))
    RRB0, gR0 = split_pl_feature(pl_input.to(DEVICE))
    features, bases, pldec, gr1dec, control_tails, rrb_after = [], [], [], [], [], []
    for idx in range(pl_input.shape[0]):
        pl_in = pl_input[idx].to(DEVICE)
        base_pl, _ = gpnet._run_pl_stage(pl_in)
        # If gpnet has no pl backend, explicitly run the supplied PL1 curve.
        if getattr(gpnet, 'pl_backend', 'original') != 'curve_v1':
            curve = pl_curve.step(pl_in, base_pl)
            pl_out = curve['pl_t'][0]
        else:
            pl_out = base_pl
        pl_out = normalize_gravity(pl_out)
        pRB = pl_out[:15]
        gR1 = pl_out[15:]
        RRB_after_pl = art.math.from_to_rotation_matrix(gR0[idx], gR1).matmul(RRB0[idx])
        tail = pl_curve.control_buffer[:, -tail_len:, :].detach()
        if tail.shape[1] < tail_len:
            pad = tail[:, :1].expand(-1, tail_len - tail.shape[1], -1)
            tail = torch.cat((pad, tail), dim=1)
        feature = torch.cat((RRB_after_pl.ravel(), tail[0].reshape(-1), gR1), dim=-1)
        base_ik1, _ = gpnet._run_ik1_stage(RRB_after_pl, gR1, pRB)
        features.append(feature.detach().cpu())
        bases.append(normalize_ik1(base_ik1.detach().cpu()))
        pldec.append(pRB.detach().cpu())
        gr1dec.append(gR1.detach().cpu())
        control_tails.append(tail[0].detach().cpu())
        rrb_after.append(RRB_after_pl.detach().cpu())
    return {
        'ik1_input': torch.stack(features),
        'ik1_base': torch.stack(bases),
        'pl_pRB_dec': torch.stack(pldec),
        'pl_gR1_dec': torch.stack(gr1dec),
        'pl_control_tail': torch.stack(control_tails),
        'RRB_after_pl': torch.stack(rrb_after),
    }


def build_cache(input_cache, output_dir, mode, imu_input_mode, pl_checkpoint, shard_size, tail_len, max_sequences=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(input_cache)
    gpnet = GPNet().eval().to(DEVICE)
    for p in gpnet.parameters():
        p.requires_grad_(False)
    body_model_pl = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    body_model_ik1 = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    pl_curve, pl_cfg = (None, None)
    if mode == 'pl1_streaming':
        if not pl_checkpoint:
            raise ValueError('--pl-checkpoint is required for pl1_streaming mode.')
        pl_curve, pl_cfg = build_pl_curve(pl_checkpoint)
    cache_files = []
    shard_idx = 0
    total_sequences = 0
    total_frames = 0
    shard = None

    def new_shard():
        return {
            'name': [], 'ik1_input': [], 'ik1_target': [], 'ik1_target_control_tail': [],
            'ik1_base': [], 'pl_target': [], 'pl_control_tail_gt': [], 'num_frames': [],
            'pl_pRB_dec': [], 'pl_gR1_dec': [], 'RRB_after_pl': [],
        }

    shard = new_shard()

    def flush():
        nonlocal shard_idx, shard
        if not shard['name']:
            return
        out = output_dir / f'newik1_control_cache_shard{shard_idx:05d}.pt'
        torch.save(shard, out)
        cache_files.append({'path': str(out), 'num_sequences': len(shard['name']), 'num_frames': int(sum(shard['num_frames']))})
        shard_idx += 1
        shard = new_shard()

    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            if 'pose_gt' in data:
                pose_gt = data['pose_gt'][seq_idx].float()
            elif 'q75_gt' in data:
                pose_gt, _ = q75_to_pose_tran(data['q75_gt'][seq_idx].float())
            else:
                raise KeyError(f'{cache_file} has no pose_gt or q75_gt fields')
            a, w, R = selected_imu_fields(data, seq_idx, imu_input_mode)
            pl_input = sequence_pl_inputs(a, w, R)
            pl_target = normalize_gravity(pl_target_from_pose(pose_gt.to(DEVICE), body_model_pl).float()).cpu()
            ik1_target = normalize_ik1(ik1_target_from_pose(pose_gt.to(DEVICE), body_model_ik1).float()).cpu()
            ik1_controls = fit_ik1_controls(ik1_target.to(DEVICE)).cpu()
            ik1_tail = torch.stack([padded_control_tail(ik1_controls, idx, tail_len) for idx in range(ik1_controls.shape[0])])
            pl_tail_gt = build_gt_pl_controls(pl_target, tail_len).cpu()
            if mode == 'teacher_forced':
                ik1_input, ik1_base = teacher_forced_sequence(gpnet, pl_input, pl_target, ik1_target, tail_len, pose_gt[0].to(DEVICE))
                pl_pRB_dec, pl_gR1_dec = pl_target[:, :15], pl_target[:, 15:]
                RRB_after_pl = ik1_input[:, :45].reshape(ik1_input.shape[0], 5, 3, 3)
            elif mode == 'pl1_streaming':
                pl_init = None
                if getattr(pl_curve, 'init_size', 18) != 18:
                    if 'pl_init_feature' in data:
                        pl_init = data['pl_init_feature'][seq_idx].float()
                    else:
                        if 'offset_r' not in data:
                            raise KeyError(f'{cache_file} has no offset_r field required for PL init feature.')
                        pl_init = pl_init_feature_from_pose(data['offset_r'][seq_idx].float(), pose_gt[0].float(), body_model_pl)
                out = pl1_streaming_sequence(gpnet, pl_curve, pl_input, pl_target, tail_len, pose_gt[0].to(DEVICE), pl_init_feature=pl_init)
                ik1_input, ik1_base = out['ik1_input'], out['ik1_base']
                pl_pRB_dec, pl_gR1_dec = out['pl_pRB_dec'], out['pl_gR1_dec']
                RRB_after_pl = out['RRB_after_pl']
            else:
                raise ValueError(mode)
            tensors = [ik1_input, ik1_target, ik1_tail, ik1_base, pl_target, pl_tail_gt, pl_pRB_dec, pl_gR1_dec, RRB_after_pl]
            if not all(torch.isfinite(t).all() for t in tensors):
                raise RuntimeError(f'Non-finite NewIK1 cache tensors at {name}.')
            shard['name'].append(name)
            shard['ik1_input'].append(ik1_input.float())
            shard['ik1_target'].append(ik1_target.float())
            shard['ik1_target_control_tail'].append(ik1_tail.float())
            shard['ik1_base'].append(ik1_base.float())
            shard['pl_target'].append(pl_target.float())
            shard['pl_control_tail_gt'].append(pl_tail_gt.float())
            shard['num_frames'].append(int(ik1_input.shape[0]))
            shard['pl_pRB_dec'].append(pl_pRB_dec.float())
            shard['pl_gR1_dec'].append(pl_gR1_dec.float())
            shard['RRB_after_pl'].append(RRB_after_pl.float())
            total_sequences += 1
            total_frames += int(ik1_input.shape[0])
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
        'type': 'newik1_control_cache_v1',
        'mode': mode,
        'source_cache': str(input_cache),
        'source_manifest': source_manifest,
        'imu_input_mode': imu_input_mode,
        'pl_checkpoint': str(pl_checkpoint) if pl_checkpoint else None,
        'pl_checkpoint_config': pl_cfg,
        'cache_type': 'streaming-compatible' if mode == 'pl1_streaming' else 'teacher-forced-gt-like',
        'not_batch_cache': mode == 'pl1_streaming',
        'tail_len': tail_len,
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'target_definition': {
            'pRJ_GT': 'SMPL male FK with root pose set to identity; joints[:,1:] flattened to 69D, meters, root-relative body frame.',
            'gR2_GT': '-pose_gt[:,0,:,1], normalized.',
            'control_fit': 'fit_uniform_cubic_spline_controls(concat(pRJ_GT, normalize(gR2_GT))).',
        },
        'fields': {
            'ik1_input': '[T,120] RRB_after_pl[45] + PL control tail[4,18] + gR1[3]',
            'ik1_target': '[T,72] pRJ_GT[69]+gR2_GT[3]',
            'ik1_target_control_tail': '[T,4,72] fitted IK1 GT control tail',
            'ik1_base': '[T,72] frozen official IK1 output under this input contract',
        },
    }
    manifest_path = output_dir / 'newik1_control_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build NewIK1_ControlPoint_v1 teacher-forced or PL1-streaming caches.')
    parser.add_argument('--input-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('teacher_forced', 'pl1_streaming'), required=True)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='processed')
    parser.add_argument('--pl-checkpoint', type=Path)
    parser.add_argument('--shard-size', type=int, default=50)
    parser.add_argument('--tail-len', type=int, default=4)
    parser.add_argument('--max-sequences', type=int, default=0)
    args = parser.parse_args()
    manifest = build_cache(
        args.input_cache, args.output_dir, args.mode, args.imu_input_mode,
        args.pl_checkpoint, args.shard_size, args.tail_len, args.max_sequences,
    )
    print(json.dumps({
        'status': 'ok',
        'manifest': str(args.output_dir / 'newik1_control_cache_manifest.json'),
        'num_sequences': manifest['num_sequences'],
        'num_frames': manifest['num_frames'],
        'mode': manifest['mode'],
    }, indent=2))


if __name__ == '__main__':
    main()
