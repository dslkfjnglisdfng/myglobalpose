import argparse
import json
from pathlib import Path

import torch

import articulate as art
from articulate.utils.torch import RNN, RNNWithInit
from l4_q75_utils import q75_to_pose_tran
from l4_train_diverse_short import DEVICE, load_cache_files
from newik1_control_point import ik1_target_from_pose, normalize_ik1
from pl_curve import PLCurveModule, normalize_gravity, pl_init_feature_from_pose, pl_input_feature, pl_target_from_pose, split_pl_feature


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


class OfficialStageRunner(torch.nn.Module):
    v_imu = (1961, 5424, 1176, 4662, 411, 3021)

    def __init__(self):
        super().__init__()
        self.plnet = RNNWithInit(
            input_linear=False,
            input_size=84,
            output_size=18,
            hidden_size=512,
            num_rnn_layer=3,
            dropout=0.4,
        )
        self.ik1net = RNN(
            input_linear=False,
            input_size=63,
            output_size=72,
            hidden_size=512,
            num_rnn_layer=3,
            dropout=0.4,
        )
        weights = torch.load('data/weights.pt', map_location='cpu')
        pl_state = {key[len('plnet.'):]: value for key, value in weights.items() if key.startswith('plnet.')}
        ik1_state = {key[len('iknet.net1.'):]: value for key, value in weights.items() if key.startswith('iknet.net1.')}
        self.plnet.load_state_dict(pl_state)
        self.ik1net.load_state_dict(ik1_state)
        self.pl1hc = None
        self.ik1hc = None

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None):
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        pRL = torch.zeros(15)
        gR = -init_pose[0, 0, :, 1]
        x1 = torch.cat((pRL, gR)).to(self.plnet.init_net[0].weight.device)
        self.pl1hc = [_.contiguous() for _ in self.plnet.init_net(x1).view(1, 2, self.plnet.num_layers, self.plnet.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None

    def _run_pl_stage(self, x_pl_in):
        x, self.pl1hc = self.plnet.rnn(x_pl_in.view(1, 1, -1), self.pl1hc)
        base = self.plnet.linear2(x.squeeze())
        return base, art.math.normalize_tensor(base[15:])

    def _run_ik1_stage(self, RRB_after_pl, gR1, pRB):
        x_ik1 = torch.cat((RRB_after_pl.ravel(), gR1, pRB))
        x, self.ik1hc = self.ik1net.rnn(x_ik1.view(1, 1, -1), self.ik1hc)
        base = self.ik1net.linear2(x.squeeze())
        return base, art.math.normalize_tensor(base[69:])


@torch.no_grad()
def teacher_forced_sequence(gpnet, pl_input, pl_target, init_pose):
    gpnet.rnn_initialize(init_pose)
    RRB0, gR0 = split_pl_feature(pl_input.to(DEVICE))
    pRB = pl_target.to(DEVICE)[:, :15]
    gR1 = normalize_gravity(pl_target.to(DEVICE))[:, 15:]
    RRB_after_pl = art.math.from_to_rotation_matrix(gR0, gR1).unsqueeze(1).matmul(RRB0)
    inputs, bases = [], []
    for idx in range(pl_input.shape[0]):
        feature = torch.cat((RRB_after_pl[idx].ravel(), gR1[idx], pRB[idx]), dim=-1)
        base, _ = gpnet._run_ik1_stage(RRB_after_pl[idx], gR1[idx], pRB[idx])
        inputs.append(feature.detach().cpu())
        bases.append(normalize_ik1(base.detach().cpu()))
    return {
        'ik1_input': torch.stack(inputs),
        'ik1_base': torch.stack(bases),
        'pl_pRB_dec': pRB.detach().cpu(),
        'pl_gR1_dec': gR1.detach().cpu(),
        'RRB_after_pl': RRB_after_pl.detach().cpu(),
    }


@torch.no_grad()
def pl1_streaming_sequence(gpnet, pl_curve, pl_input, pl_target, init_pose, pl_init_feature=None):
    gpnet.rnn_initialize(init_pose)
    if getattr(pl_curve, 'init_size', 18) == 18:
        pl_curve.reset_stream(init_output=normalize_gravity(pl_target[0].to(DEVICE)))
    else:
        if pl_init_feature is None:
            raise ValueError(f'PL curve init_size={pl_curve.init_size} requires pl_init_feature.')
        pl_curve.reset_stream(init_feature=pl_init_feature.to(DEVICE))
    RRB0, gR0 = split_pl_feature(pl_input.to(DEVICE))
    inputs, bases, pldec, gr1dec, rrb_after = [], [], [], [], []
    for idx in range(pl_input.shape[0]):
        pl_in = pl_input[idx].to(DEVICE)
        base_pl, _ = gpnet._run_pl_stage(pl_in)
        curve = pl_curve.step(pl_in, base_pl)
        pl_out = normalize_gravity(curve['pl_t'][0])
        pRB = pl_out[:15]
        gR1 = pl_out[15:]
        RRB_after_pl = art.math.from_to_rotation_matrix(gR0[idx], gR1).matmul(RRB0[idx])
        feature = torch.cat((RRB_after_pl.ravel(), gR1, pRB), dim=-1)
        base_ik1, _ = gpnet._run_ik1_stage(RRB_after_pl, gR1, pRB)
        inputs.append(feature.detach().cpu())
        bases.append(normalize_ik1(base_ik1.detach().cpu()))
        pldec.append(pRB.detach().cpu())
        gr1dec.append(gR1.detach().cpu())
        rrb_after.append(RRB_after_pl.detach().cpu())
    return {
        'ik1_input': torch.stack(inputs),
        'ik1_base': torch.stack(bases),
        'pl_pRB_dec': torch.stack(pldec),
        'pl_gR1_dec': torch.stack(gr1dec),
        'RRB_after_pl': torch.stack(rrb_after),
    }


def pose_gt_from_data(data, seq_idx, cache_file):
    if 'pose_gt' in data:
        return data['pose_gt'][seq_idx].float()
    if 'q75_gt' in data:
        pose_gt, _ = q75_to_pose_tran(data['q75_gt'][seq_idx].float())
        return pose_gt
    raise KeyError(f'{cache_file} has no pose_gt or q75_gt fields')


def build_cache(input_cache, output_dir, mode, imu_input_mode, pl_checkpoint, shard_size, max_sequences=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(input_cache)
    gpnet = OfficialStageRunner().eval().to(DEVICE)
    for p in gpnet.parameters():
        p.requires_grad_(False)
    body_model_pl = art.ParametricModel('models/SMPL_male.pkl', vert_mask=OfficialStageRunner.v_imu, device=DEVICE)
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

    def new_shard():
        return {
            'name': [],
            'ik1_input': [],
            'ik1_target': [],
            'ik1_base': [],
            'pl_target': [],
            'pl_pRB_dec': [],
            'pl_gR1_dec': [],
            'RRB_after_pl': [],
            'num_frames': [],
        }

    shard = new_shard()

    def flush():
        nonlocal shard_idx, shard
        if not shard['name']:
            return
        out = output_dir / f'newik1_official_input_cache_shard{shard_idx:05d}.pt'
        torch.save(shard, out)
        cache_files.append({'path': str(out), 'num_sequences': len(shard['name']), 'num_frames': int(sum(shard['num_frames']))})
        shard_idx += 1
        shard = new_shard()

    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            pose_gt = None
            if 'pose_gt' in data or 'q75_gt' in data:
                pose_gt = pose_gt_from_data(data, seq_idx, cache_file)
            if 'pl_input' in data:
                pl_input = data['pl_input'][seq_idx].float()
            else:
                a, w, R = selected_imu_fields(data, seq_idx, imu_input_mode)
                pl_input = sequence_pl_inputs(a, w, R)
            if 'pl_target' in data:
                pl_target = normalize_gravity(data['pl_target'][seq_idx].float())
            else:
                pl_target = normalize_gravity(pl_target_from_pose(pose_gt.to(DEVICE), body_model_pl).float()).cpu()
            if 'ik1_target' in data:
                ik1_target = normalize_ik1(data['ik1_target'][seq_idx].float())
            else:
                ik1_target = normalize_ik1(ik1_target_from_pose(pose_gt.to(DEVICE), body_model_ik1).float()).cpu()
            init_pose = None if pose_gt is None else pose_gt[0].to(DEVICE)
            pl_init = None
            if mode == 'pl1_streaming' and getattr(pl_curve, 'init_size', 18) != 18:
                if 'pl_init_feature' in data:
                    pl_init = data['pl_init_feature'][seq_idx].float()
                else:
                    if 'offset_r' not in data:
                        raise KeyError(f'{cache_file} has no offset_r field required for PL init feature.')
                    pl_init = pl_init_feature_from_pose(data['offset_r'][seq_idx].float(), pose_gt[0].float(), body_model_pl)
            if mode == 'teacher_forced':
                out = teacher_forced_sequence(gpnet, pl_input, pl_target, init_pose)
            elif mode == 'pl1_streaming':
                out = pl1_streaming_sequence(gpnet, pl_curve, pl_input, pl_target, init_pose, pl_init_feature=pl_init)
            else:
                raise ValueError(mode)
            tensors = [out['ik1_input'], ik1_target, out['ik1_base'], pl_target, out['pl_pRB_dec'], out['pl_gR1_dec'], out['RRB_after_pl']]
            if out['ik1_input'].shape[-1] != 63:
                raise RuntimeError(f'Expected official IK1 input dim 63, got {out["ik1_input"].shape[-1]} at {name}.')
            if not all(torch.isfinite(t).all() for t in tensors):
                raise RuntimeError(f'Non-finite official-input IK1 cache tensors at {name}.')
            shard['name'].append(name)
            shard['ik1_input'].append(out['ik1_input'].float())
            shard['ik1_target'].append(ik1_target.float())
            shard['ik1_base'].append(out['ik1_base'].float())
            shard['pl_target'].append(pl_target.float())
            shard['pl_pRB_dec'].append(out['pl_pRB_dec'].float())
            shard['pl_gR1_dec'].append(out['pl_gR1_dec'].float())
            shard['RRB_after_pl'].append(out['RRB_after_pl'].float())
            shard['num_frames'].append(int(out['ik1_input'].shape[0]))
            total_sequences += 1
            total_frames += int(out['ik1_input'].shape[0])
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
        'type': 'newik1_official_input_cache_v1',
        'mode': mode,
        'source_cache': str(input_cache),
        'source_manifest': source_manifest,
        'imu_input_mode': imu_input_mode,
        'pl_checkpoint': str(pl_checkpoint) if pl_checkpoint else None,
        'pl_checkpoint_config': pl_cfg,
        'cache_type': 'streaming-compatible' if mode == 'pl1_streaming' else 'teacher-forced-gt-like',
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'target_definition': {
            'pRJ_GT': 'SMPL male FK with root pose set to identity; joints[:,1:] flattened to 69D, meters, root-relative body frame.',
            'gR2_GT': '-pose_gt[:,0,:,1], normalized.',
        },
        'fields': {
            'ik1_input': '[T,63] official IK1 order: RRB_after_pl[45] + decoded gR1[3] + decoded pRB[15]',
            'ik1_target': '[T,72] pRJ_GT[69]+gR2_GT[3]',
            'ik1_base': '[T,72] frozen official IK1 output under the same 63D input contract',
            'pl_pRB_dec': '[T,15] decoded PL leaf positions used in ik1_input',
            'pl_gR1_dec': '[T,3] decoded PL root gravity direction used in ik1_input',
        },
    }
    manifest_path = output_dir / 'newik1_official_input_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build official-shape NewIK1 caches with decoded PL pRB/gR1 inputs.')
    parser.add_argument('--input-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('teacher_forced', 'pl1_streaming'), required=True)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='processed')
    parser.add_argument('--pl-checkpoint', type=Path)
    parser.add_argument('--shard-size', type=int, default=50)
    parser.add_argument('--max-sequences', type=int, default=0)
    args = parser.parse_args()
    manifest = build_cache(
        args.input_cache, args.output_dir, args.mode, args.imu_input_mode,
        args.pl_checkpoint, args.shard_size, args.max_sequences,
    )
    print(json.dumps({
        'status': 'ok',
        'manifest': str(args.output_dir / 'newik1_official_input_cache_manifest.json'),
        'num_sequences': manifest['num_sequences'],
        'num_frames': manifest['num_frames'],
        'mode': manifest['mode'],
    }, indent=2))


if __name__ == '__main__':
    main()
