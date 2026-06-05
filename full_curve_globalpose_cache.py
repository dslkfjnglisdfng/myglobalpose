import argparse
import inspect
import json
from pathlib import Path

import numpy as np

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _value in {
    'bool': bool,
    'int': int,
    'float': float,
    'complex': complex,
    'object': object,
    'unicode': str,
    'str': str,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

import torch

import articulate as art
from articulate.utils.torch import RNN, RNNWithInit
from full_curve_globalpose import FullCurveGlobalPoseV1, rotation_matrix_to_6d
from l4_train_diverse_short import DEVICE, load_cache_files
from pl_curve import normalize_gravity, pl_input_feature, pl_target_from_pose, split_pl_feature
from ik1_curve import normalize_ik1


FULL_CURVE_FIELDS = {
    'pl_input': '[T,84] PL input feature built from selected processed IMU fields',
    'pl_target': '[T,18] GT pRB[15]+gR1[3]',
    'pl_base': '[T,18] frozen official PL output on selected IMU input',
    'ik1_target': '[T,72] GT pRJ[69] with official-base gR2[3]',
    'ik1_base': '[T,72] frozen official IK1 output from PL base geometry',
    'ik2_target': '[T,90] GT reduced global joint rotations as 15x6D',
    'ik2_base': '[T,90] frozen official IK2 output from IK1 base geometry',
    'vr_target': '[T,9] source v_root_vr/stationary target when available, otherwise vr_base',
    'vr_base': '[T,9] frozen official VR-s1 output from official pose intermediates',
    'processed_imu': '[T,90] processed IMU a/w/R flattened from selected fields',
}


class OfficialNeuralBases(torch.nn.Module):
    v_imu = (1961, 5424, 1176, 4662, 411, 3021)
    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)

    def __init__(self):
        super().__init__()
        self.plnet = RNNWithInit(input_linear=False, input_size=84, output_size=18, hidden_size=512, num_rnn_layer=3, dropout=0.4)
        self.iknet = torch.nn.ModuleDict({
            'net1': RNN(input_linear=False, input_size=63, output_size=72, hidden_size=512, num_rnn_layer=3, dropout=0.4),
            'net2': RNN(input_linear=False, input_size=117, output_size=90, hidden_size=512, num_rnn_layer=3, dropout=0.4),
        })
        self.vrnet = RNNWithInit(input_linear=False, input_size=243, output_size=9, hidden_size=512, num_rnn_layer=3, dropout=0.4)
        weights = torch.load('data/weights.pt', map_location='cpu')
        self.load_state_dict({key: value for key, value in weights.items() if key in self.state_dict()}, strict=True)
        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.v_imu)
        self.pl1hc = None
        self.ik1hc = None
        self.ik2hc = None
        self.vr1hc = None

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None, init_vel=None):
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        init_vel = torch.zeros(3) if init_vel is None else init_vel.cpu().view(3)
        vRR_V = init_vel[1].view(1).clone()
        init_vel[1] = 0
        _, j, v = self.body_model.forward_kinematics(init_pose, calc_mesh=True)
        pRL, gR = (v[0, :5] - v[0, 5:]).mm(init_pose[0, 0]).ravel(), -init_pose[0, 0, 1]
        x1 = torch.cat((pRL, gR)).to(self.plnet.init_net[0].weight.device)
        vRR_H, c = init_pose[0, 0].t().mm(init_vel.unsqueeze(-1)).squeeze(-1), torch.zeros(5)
        x2 = torch.cat((vRR_V, vRR_H, c)).to(self.vrnet.init_net[0].weight.device)
        self.pl1hc = [_.contiguous() for _ in self.plnet.init_net(x1).view(1, 2, self.plnet.num_layers, self.plnet.hidden_size).permute(1, 2, 0, 3)]
        self.vr1hc = [_.contiguous() for _ in self.vrnet.init_net(x2).view(1, 2, self.vrnet.num_layers, self.vrnet.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None

    def _run_pl_stage(self, x_pl_in):
        x, self.pl1hc = self.plnet.rnn(x_pl_in.view(1, 1, -1), self.pl1hc)
        base = self.plnet.linear2(x.squeeze())
        return base, art.math.normalize_tensor(base[15:])

    def _run_ik1_stage(self, RRB_after_pl, gR1, pRB):
        x_ik1 = torch.cat((RRB_after_pl.ravel(), gR1, pRB))
        x, self.ik1hc = self.iknet.net1.rnn(x_ik1.view(1, 1, -1), self.ik1hc)
        base = self.iknet.net1.linear2(x.squeeze())
        return base, art.math.normalize_tensor(base[69:])


def select_imu_fields(data, seq_idx, mode):
    has_l4 = all(key in data for key in ('l4_aM', 'l4_wM', 'l4_RMB'))
    if mode == 'processed':
        if not has_l4:
            raise KeyError('processed mode requires l4_aM/l4_wM/l4_RMB fields.')
        return data['l4_aM'][seq_idx].float(), data['l4_wM'][seq_idx].float(), data['l4_RMB'][seq_idx].float(), {
            'mode': mode,
            'a_field': 'l4_aM',
            'w_field': 'l4_wM',
            'R_field': 'l4_RMB',
            'source': 'processed_l4_fields',
        }
    if mode == 'official':
        return data['aM'][seq_idx].float(), data['wM'][seq_idx].float(), data['RMB'][seq_idx].float(), {
            'mode': mode,
            'a_field': 'aM',
            'w_field': 'wM',
            'R_field': 'RMB',
            'source': 'official_fields',
        }
    if mode == 'auto':
        if has_l4:
            return data['l4_aM'][seq_idx].float(), data['l4_wM'][seq_idx].float(), data['l4_RMB'][seq_idx].float(), {
                'mode': mode,
                'a_field': 'l4_aM',
                'w_field': 'l4_wM',
                'R_field': 'l4_RMB',
                'source': 'processed_l4_fields',
            }
        return data['aM'][seq_idx].float(), data['wM'][seq_idx].float(), data['RMB'][seq_idx].float(), {
            'mode': mode,
            'a_field': 'aM',
            'w_field': 'wM',
            'R_field': 'RMB',
            'source': 'fallback_official_fields',
        }
    raise ValueError(f'Unsupported imu input mode: {mode}')


def sequence_pl_inputs(a_seq, w_seq, R_seq):
    return torch.stack([
        pl_input_feature(a_seq[i], w_seq[i], R_seq[i])
        for i in range(a_seq.shape[0])
    ]).float()


def sequence_processed_imu(a_seq, w_seq, R_seq):
    return torch.stack([
        FullCurveGlobalPoseV1.processed_imu_feature(a_seq[i], w_seq[i], R_seq[i])
        for i in range(a_seq.shape[0])
    ]).float()


@torch.no_grad()
def ik1_target_from_pose(pose, body_model, ik1_base):
    pose_body = pose.to(DEVICE).clone()
    pose_body[:, 0] = torch.eye(3, device=DEVICE)
    _, joints = body_model.forward_kinematics(pose_body)[:2]
    pRJ = joints[:, 1:].reshape(pose.shape[0], 69).detach().cpu()
    gR2 = normalize_ik1(ik1_base.detach().cpu())[:, 69:]
    return torch.cat((pRJ, gR2), dim=-1)


@torch.no_grad()
def ik2_target_from_pose(pose, body_model, j_reduce):
    pose_body = pose.to(DEVICE).clone()
    pose_body[:, 0] = torch.eye(3, device=DEVICE)
    global_pose = body_model.forward_kinematics(pose_body)[0]
    return rotation_matrix_to_6d(global_pose[:, j_reduce]).reshape(pose.shape[0], 90).detach().cpu()


def vr_target_from_source(data, seq_idx, vr_base):
    if 'v_root_vr' not in data or 'stationary_prob' not in data or not data['v_root_vr'] or not data['stationary_prob']:
        return vr_base.detach().cpu()
    v_root = data['v_root_vr'][seq_idx].float()
    stationary = data['stationary_prob'][seq_idx].float()
    if v_root.shape[-1] == 3:
        vertical = v_root[..., 1:2]
        horizontal = v_root.clone()
        horizontal[..., 1] = 0.0
        vr = torch.cat((vertical, horizontal, stationary), dim=-1)
    elif v_root.shape[-1] == 4:
        vr = torch.cat((v_root, stationary), dim=-1)
    else:
        raise ValueError(f'Unsupported v_root_vr shape {tuple(v_root.shape)}.')
    if vr.shape[-1] != 9:
        raise ValueError(f'Expected vr_target last dim 9, got {vr.shape[-1]}.')
    return vr


@torch.no_grad()
def official_bases(gpnet, pl_input, a_seq, w_seq, R_seq, pose_gt):
    gpnet.rnn_initialize(init_pose=pose_gt[0])
    pl_base, ik1_base, ik2_base, vr_base = [], [], [], []
    body_model = gpnet.body_model
    for frame_idx in range(pl_input.shape[0]):
        x_pl = pl_input[frame_idx].to(DEVICE)
        base_pl, gR1 = gpnet._run_pl_stage(x_pl)
        RRB0, gR0 = split_pl_feature(x_pl)
        RRB_after_pl = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB0)
        base_ik1, gR2 = gpnet._run_ik1_stage(RRB_after_pl, gR1, base_pl[:15])
        RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB_after_pl)
        ik2_input = torch.cat((RRB_after_ik1.ravel(), gR2, base_ik1[:69]))
        x_ik2, gpnet.ik2hc = gpnet.iknet.net2.rnn(ik2_input.view(1, 1, -1), gpnet.ik2hc)
        base_ik2 = gpnet.iknet.net2.linear2(x_ik2.squeeze())
        RRJ = art.math.r6d_to_rotation_matrix(base_ik2).cpu()
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, gpnet.j_reduce] = RRJ.view(1, 15, 3, 3)
        pose = body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose[gpnet.j_ignore, ...] = torch.eye(3)
        pRJ = body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
        pose[0] = R_seq[frame_idx, 5].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze().cpu()).cpu()
        aRB = a_seq[frame_idx].cpu().mm(pose[0])
        wRB = w_seq[frame_idx].cpu().mm(pose[0])
        vr_input = torch.cat((RRJ.ravel(), pRJ.ravel(), aRB.ravel(), wRB.ravel(), gR2.detach().cpu())).to(DEVICE)
        x_vr, gpnet.vr1hc = gpnet.vrnet.rnn(vr_input.view(1, 1, -1), gpnet.vr1hc)
        base_vr = gpnet.vrnet.linear2(x_vr.squeeze())
        pl_base.append(normalize_gravity(base_pl.detach().cpu()))
        ik1_base.append(normalize_ik1(base_ik1.detach().cpu()))
        ik2_base.append(base_ik2.detach().cpu())
        vr_base.append(base_vr.detach().cpu())
    return {
        'pl_base': torch.stack(pl_base),
        'ik1_base': torch.stack(ik1_base),
        'ik2_base': torch.stack(ik2_base),
        'vr_base': torch.stack(vr_base),
    }


def build_cache(input_cache, output_dir, shard_size, imu_input_mode='processed', max_sequences=None, start_sequence=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    files, source_manifest = load_cache_files(input_cache)
    gpnet = OfficialNeuralBases().eval().to(DEVICE)
    for parameter in gpnet.parameters():
        parameter.requires_grad_(False)
    pl_body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    joint_body_model = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    cache_files = []
    shard = {'name': [], 'num_frames': []}
    for field in FULL_CURVE_FIELDS:
        shard[field] = []
    shard['offset_r'] = []
    shard_idx = 0
    total_sequences = 0
    total_frames = 0
    seen_sequences = 0
    imu_field_contracts = {}

    def flush():
        nonlocal shard, shard_idx
        if not shard['name']:
            return
        out = output_dir / f'full_curve_globalpose_cache_shard{shard_idx:05d}.pt'
        torch.save(shard, out)
        cache_files.append({
            'path': str(out),
            'num_sequences': len(shard['name']),
            'num_frames': int(sum(shard['num_frames'])),
        })
        shard_idx += 1
        shard = {'name': [], 'num_frames': []}
        for field in FULL_CURVE_FIELDS:
            shard[field] = []
        shard['offset_r'] = []

    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            if seen_sequences < int(start_sequence):
                seen_sequences += 1
                continue
            seen_sequences += 1
            pose_gt = data['pose_gt'][seq_idx].float()
            a_seq, w_seq, R_seq, imu_contract = select_imu_fields(data, seq_idx, imu_input_mode)
            imu_field_contracts[json.dumps(imu_contract, sort_keys=True)] = imu_contract
            pl_input = sequence_pl_inputs(a_seq, w_seq, R_seq)
            processed_imu = sequence_processed_imu(a_seq, w_seq, R_seq)
            pl_target = normalize_gravity(pl_target_from_pose(pose_gt.to(DEVICE), pl_body_model).float()).cpu()
            bases = official_bases(gpnet, pl_input, a_seq, w_seq, R_seq, pose_gt)
            ik1_target = ik1_target_from_pose(pose_gt, joint_body_model, bases['ik1_base'])
            ik2_target = ik2_target_from_pose(pose_gt, joint_body_model, gpnet.j_reduce)
            vr_target = vr_target_from_source(data, seq_idx, bases['vr_base'])
            tensors = [pl_input, processed_imu, pl_target, ik1_target, ik2_target, vr_target] + list(bases.values())
            if not all(torch.isfinite(t).all() for t in tensors):
                raise RuntimeError(f'Non-finite FullCurve cache tensors at {name}.')
            shard['name'].append(name)
            shard['num_frames'].append(int(pl_input.shape[0]))
            for field in ('pl_input', 'processed_imu', 'pl_target', 'ik1_target', 'ik2_target', 'vr_target'):
                shard[field].append(locals()[field].cpu())
            for field, value in bases.items():
                shard[field].append(value.cpu())
            if 'offset_r' in data and data['offset_r']:
                shard['offset_r'].append(data['offset_r'][seq_idx].float())
            else:
                shard['offset_r'].append(torch.empty(0))
            total_sequences += 1
            total_frames += int(pl_input.shape[0])
            if len(shard['name']) >= shard_size:
                flush()
            if total_sequences % 25 == 0:
                print(json.dumps({'processed_sequences': total_sequences, 'processed_frames': total_frames}), flush=True)
            if max_sequences is not None and total_sequences >= int(max_sequences):
                flush()
                manifest = {
                    'type': 'full_curve_globalpose_cache_v1',
                    'source_cache': str(input_cache),
                    'source_manifest': source_manifest,
                    'imu_input_mode': imu_input_mode,
                    'imu_field_contracts': list(imu_field_contracts.values()),
                    'cache_files': cache_files,
                    'num_sequences': total_sequences,
                    'num_frames': total_frames,
                    'fields': FULL_CURVE_FIELDS,
                    'target_notes': {
                        'ik1_gR2': 'uses official ik1_base gR2 to avoid treating GT root gravity as IK1 gR2',
                        'vr_target': 'uses source v_root_vr/stationary_prob when available; contact values are probabilities, not logits',
                    },
                }
                (output_dir / 'full_curve_globalpose_cache_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                return manifest
    flush()
    manifest = {
        'type': 'full_curve_globalpose_cache_v1',
        'source_cache': str(input_cache),
        'source_manifest': source_manifest,
        'imu_input_mode': imu_input_mode,
        'imu_field_contracts': list(imu_field_contracts.values()),
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'fields': FULL_CURVE_FIELDS,
        'target_notes': {
            'ik1_gR2': 'uses official ik1_base gR2 to avoid treating GT root gravity as IK1 gR2',
            'vr_target': 'uses source v_root_vr/stationary_prob when available; contact values are probabilities, not logits',
        },
    }
    (output_dir / 'full_curve_globalpose_cache_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Precompute FullCurveGlobalPose_v1 full-chain tensors.')
    parser.add_argument('--input-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--shard-size', type=int, default=50)
    parser.add_argument('--imu-input-mode', choices=('processed', 'official', 'auto'), default='processed')
    parser.add_argument('--max-sequences', type=int, default=None)
    parser.add_argument('--start-sequence', type=int, default=0)
    args = parser.parse_args()
    manifest = build_cache(
        args.input_cache,
        args.output_dir,
        args.shard_size,
        args.imu_input_mode,
        args.max_sequences,
        args.start_sequence,
    )
    print(json.dumps({
        'status': 'ok',
        'manifest': str(args.output_dir / 'full_curve_globalpose_cache_manifest.json'),
        'num_sequences': manifest['num_sequences'],
        'num_frames': manifest['num_frames'],
    }, indent=2))


if __name__ == '__main__':
    main()
