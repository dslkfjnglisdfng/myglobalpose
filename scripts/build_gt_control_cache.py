import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

import articulate as art
from l4_q75_utils import q75_to_pose_tran
from l4_train_diverse_short import load_cache_files
from pl_curve import (
    CONTROL_FIT_DEFAULT_MODE,
    CONTROL_FIT_DERIVATIVE_AWARE_WEIGHTS,
    CONTROL_FIT_DT,
    control_fit_contract,
    normalize_gravity,
    rotation_matrix_to_6d,
)


IMU_VERTICES = (1961, 5424, 1176, 4662, 411, 3021)
DEFAULT_PRESETS = {
    'amass_train': {
        'input_cache': 'data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json',
        'root_trans_policy': 'available',
        'dataset': 'AMASS',
        'split': 'train',
    },
    'dip_train': {
        'input_cache': 'data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json',
        'root_trans_policy': 'unavailable',
        'dataset': 'DIP-IMU',
        'split': 'train',
    },
    'dip_val': {
        'input_cache': 'data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json',
        'root_trans_policy': 'unavailable',
        'dataset': 'DIP-IMU',
        'split': 'val',
    },
    'dip_test': {
        'input_cache': 'data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json',
        'root_trans_policy': 'unavailable',
        'dataset': 'DIP-IMU',
        'split': 'test',
    },
    'totalcapture_train': {
        'input_cache': 'data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json',
        'root_trans_policy': 'available',
        'dataset': 'TotalCapture',
        'split': 'train',
    },
    'totalcapture_val': {
        'input_cache': 'data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json',
        'root_trans_policy': 'available',
        'dataset': 'TotalCapture',
        'split': 'val',
    },
    'totalcapture_test': {
        'input_cache': 'data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json',
        'root_trans_policy': 'available',
        'dataset': 'TotalCapture',
        'split': 'test',
    },
}
_CONTROL_SOLVER_CACHE = {}


def _central_difference_first_np(samples, dt):
    out = np.zeros_like(samples)
    if samples.shape[0] <= 1:
        return out
    out[1:-1] = (samples[2:] - samples[:-2]) / (2.0 * dt)
    out[0] = (samples[1] - samples[0]) / dt
    out[-1] = (samples[-1] - samples[-2]) / dt
    return out


def _central_difference_second_np(samples, dt):
    out = np.zeros_like(samples)
    if samples.shape[0] <= 2:
        return out
    out[1:-1] = (samples[2:] - 2.0 * samples[1:-1] + samples[:-2]) / (dt ** 2)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def _clamped_uniform_ops_sparse(t, dt):
    rows = []
    cols = []
    s_vals = []
    d1_vals = []
    d2_vals = []
    inv_2dt = 1.0 / (2.0 * dt)
    inv_dt2 = 1.0 / (dt ** 2)
    for i in range(t):
        left = max(i - 1, 0)
        right = min(i + 1, t - 1)
        for col, s_val, d1_val, d2_val in (
            (left, 1.0 / 6.0, -inv_2dt, inv_dt2),
            (i, 4.0 / 6.0, 0.0, -2.0 * inv_dt2),
            (right, 1.0 / 6.0, inv_2dt, inv_dt2),
        ):
            rows.append(i)
            cols.append(col)
            s_vals.append(s_val)
            d1_vals.append(d1_val)
            d2_vals.append(d2_val)
    shape = (t, t)
    s = scipy.sparse.coo_matrix((s_vals, (rows, cols)), shape=shape).tocsr()
    d1 = scipy.sparse.coo_matrix((d1_vals, (rows, cols)), shape=shape).tocsr()
    d2 = scipy.sparse.coo_matrix((d2_vals, (rows, cols)), shape=shape).tocsr()
    return s, d1, d2


def fit_controls_sparse(samples, dt=CONTROL_FIT_DT):
    """Solve the project derivative-aware control objective with sparse bands.

    This is mathematically equivalent to pl_curve.fit_uniform_cubic_spline_controls
    for mode=derivative_aware_v1, but avoids materializing dense TxT matrices for
    long DIP/AMASS sequences.
    """
    if samples.shape[0] <= 1:
        return samples.clone()
    shape = samples.shape
    flat = samples.detach().cpu().reshape(shape[0], -1).numpy().astype(np.float64, copy=False)
    vel = _central_difference_first_np(flat, float(dt))
    acc = _central_difference_second_np(flat, float(dt))
    weights = CONTROL_FIT_DERIVATIVE_AWARE_WEIGHTS
    wp = float(weights['position'])
    wv = float(weights['velocity'])
    wa = float(weights['acceleration'])
    wr = float(weights['ridge'])
    cache_key = (shape[0], float(dt), wp, wv, wa, wr)
    cached = _CONTROL_SOLVER_CACHE.get(cache_key)
    if cached is None:
        s, d1, d2 = _clamped_uniform_ops_sparse(shape[0], float(dt))
        lhs = wp * (s.T @ s) + wv * (d1.T @ d1) + wa * (d2.T @ d2)
        if wr > 0.0:
            lhs = lhs + wr * scipy.sparse.eye(shape[0], format='csr')
        cached = (s, d1, d2, scipy.sparse.linalg.splu(lhs.tocsc()))
        _CONTROL_SOLVER_CACHE.clear()
        _CONTROL_SOLVER_CACHE[cache_key] = cached
    s, d1, d2, solver = cached
    rhs = wp * (s.T @ flat) + wv * (d1.T @ vel) + wa * (d2.T @ acc)
    solved = solver.solve(rhs)
    solved = np.asarray(solved, dtype=np.float32).reshape(shape)
    return torch.from_numpy(solved)


def unwrap_angles(angles):
    if angles.shape[0] <= 1:
        return angles.clone()
    out = torch.empty_like(angles)
    out[0] = angles[0]
    for i in range(1, angles.shape[0]):
        delta = torch.atan2(torch.sin(angles[i] - angles[i - 1]), torch.cos(angles[i] - angles[i - 1]))
        out[i] = out[i - 1] + delta
    return out


def finite_difference_velocity(x, dt):
    if x.shape[0] <= 1:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * float(dt))
    out[0] = (x[1] - x[0]) / float(dt)
    out[-1] = (x[-1] - x[-2]) / float(dt)
    return out


def load_pose_tran(data, seq_idx, euler_seq):
    if data.get('pose_gt') and data['pose_gt']:
        pose = data['pose_gt'][seq_idx].float()
        tran = data['tran_gt'][seq_idx].float() if data.get('tran_gt') and data['tran_gt'] else None
        return pose, tran
    if data.get('q75_gt') and data['q75_gt']:
        return q75_to_pose_tran(data['q75_gt'][seq_idx].float(), euler_seq=euler_seq)
    raise KeyError('source sequence has neither pose_gt nor q75_gt')


def root_trans_available(policy, preset_name, tran):
    if policy == 'available':
        return tran is not None
    if policy == 'unavailable':
        return False
    if preset_name and preset_name.startswith('dip_'):
        return False
    return tran is not None


@torch.no_grad()
def fk_control_targets(pose, body_model, device, batch_size):
    joint_pos = []
    pl_states = []
    pose = pose.to(device)
    for start in range(0, pose.shape[0], batch_size):
        p = pose[start:start + batch_size]
        _, joints, verts = body_model.forward_kinematics(p, calc_mesh=True)
        root_rot = p[:, 0]
        p_rj = torch.matmul(joints - joints[:, :1], root_rot)
        p_rb = torch.matmul(verts[:, :5] - verts[:, 5:], root_rot).reshape(p.shape[0], 15)
        gr = -root_rot[:, :, 1]
        joint_pos.append(p_rj.detach().cpu())
        pl_states.append(normalize_gravity(torch.cat((p_rb, gr), dim=-1)).detach().cpu())
    return torch.cat(joint_pos, dim=0).float(), torch.cat(pl_states, dim=0).float()


def rotation6d_sequence(rotation):
    return rotation_matrix_to_6d(rotation.reshape(-1, 3, 3)).reshape(rotation.shape[:-2] + (6,)).float()


def build_sequence_targets(data, seq_idx, name, args, body_model, device, preset_name, source_file):
    pose, tran = load_pose_tran(data, seq_idx, args.euler_seq)
    if pose.shape[0] < 1:
        raise ValueError(f'{name} has no frames')
    pose_rot6d = rotation6d_sequence(pose)
    euler = art.math.rotation_matrix_to_euler_angle(pose.reshape(-1, 3, 3), seq=args.euler_seq).reshape(pose.shape[0], 72).float()
    joint_angle_euler = unwrap_angles(euler)
    joint_pos_r, pl_state = fk_control_targets(pose, body_model, device, args.fk_batch_size)
    imu_rmb = data['RMB'][seq_idx].float()
    imu_rmb_6d = rotation6d_sequence(imu_rmb)
    record = {
        'name': str(name),
        'num_frames': int(pose.shape[0]),
        'source_cache_file': str(source_file),
        'source_seq_idx': int(seq_idx),
        'pose_rot6d': pose_rot6d,
        'pose_rot6d_control': fit_controls_sparse(pose_rot6d, dt=args.dt),
        'joint_angle_euler': joint_angle_euler,
        'joint_angle_euler_control': fit_controls_sparse(joint_angle_euler, dt=args.dt),
        'joint_pos_R': joint_pos_r,
        'joint_pos_R_control': fit_controls_sparse(joint_pos_r, dt=args.dt),
        'imu_RMB_6d': imu_rmb_6d,
        'imu_RMB_6d_control': fit_controls_sparse(imu_rmb_6d, dt=args.dt),
        'pl_pRB_gR1': pl_state,
        'pl_pRB_gR1_control': fit_controls_sparse(pl_state, dt=args.dt),
        'root_trans_available': bool(root_trans_available(args.root_trans_policy, preset_name, tran)),
    }
    if record['root_trans_available']:
        root_trans = tran.float()
        record['root_trans_W'] = root_trans
        record['root_trans_W_control'] = fit_controls_sparse(root_trans, dt=args.dt)
        record['root_vel_W_fd'] = finite_difference_velocity(root_trans, dt=args.dt)
    if data.get('offset_r') and data['offset_r']:
        record['offset_r'] = data['offset_r'][seq_idx].float()
    for key in ('source_name', 'source_cache_name', 'source_aug_name', 'view_type', 'pair_id'):
        if key in data and data[key]:
            record[key] = data[key][seq_idx]
    finite_keys = [
        'pose_rot6d',
        'pose_rot6d_control',
        'joint_angle_euler',
        'joint_angle_euler_control',
        'joint_pos_R',
        'joint_pos_R_control',
        'imu_RMB_6d',
        'imu_RMB_6d_control',
        'pl_pRB_gR1',
        'pl_pRB_gR1_control',
    ]
    if record['root_trans_available']:
        finite_keys.extend(('root_trans_W', 'root_trans_W_control', 'root_vel_W_fd'))
    bad = [key for key in finite_keys if not torch.isfinite(record[key]).all()]
    if bad:
        raise RuntimeError(f'non-finite target tensors for {name}: {bad}')
    return record


def empty_shard():
    return {
        'name': [],
        'num_frames': [],
        'source_cache_file': [],
        'source_seq_idx': [],
        'pose_rot6d': [],
        'pose_rot6d_control': [],
        'joint_angle_euler': [],
        'joint_angle_euler_control': [],
        'joint_pos_R': [],
        'joint_pos_R_control': [],
        'imu_RMB_6d': [],
        'imu_RMB_6d_control': [],
        'pl_pRB_gR1': [],
        'pl_pRB_gR1_control': [],
        'root_trans_available': [],
        'root_trans_W': [],
        'root_trans_W_control': [],
        'root_vel_W_fd': [],
        'offset_r': [],
        'source_name': [],
        'source_cache_name': [],
        'source_aug_name': [],
        'view_type': [],
        'pair_id': [],
    }


def append_record(shard, record):
    for key in shard:
        if key in record:
            shard[key].append(record[key])
        elif key in ('root_trans_W', 'root_trans_W_control', 'root_vel_W_fd', 'offset_r'):
            shard[key].append(None)
        elif key in ('source_name', 'source_cache_name', 'source_aug_name', 'view_type', 'pair_id'):
            shard[key].append('')
        else:
            raise KeyError(f'missing required record key: {key}')


def prune_optional_empty_lists(shard):
    pruned = {}
    for key, value in shard.items():
        if key in ('root_trans_W', 'root_trans_W_control', 'root_vel_W_fd', 'offset_r') and all(v is None for v in value):
            continue
        pruned[key] = value
    return pruned


def prepare_output_dir(output_dir, overwrite):
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(f'{output_dir} is not empty; pass --overwrite only for intentional replacement.')
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_cache(args, preset_name=None):
    input_cache = Path(args.input_cache)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    files, source_manifest = load_cache_files(input_cache)
    device = torch.device(args.device)
    body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=IMU_VERTICES, device=device)
    shard = empty_shard()
    cache_files = []
    total_sequences = 0
    total_frames = 0
    started = time.time()

    def flush():
        nonlocal shard
        if not shard['name']:
            return
        shard_idx = len(cache_files)
        out = output_dir / f'gt_control_cache_shard{shard_idx:05d}.pt'
        saved = prune_optional_empty_lists(shard)
        torch.save(saved, out)
        cache_files.append({
            'path': str(out),
            'num_sequences': len(saved['name']),
            'num_frames': int(sum(saved['num_frames'])),
        })
        shard = empty_shard()

    for source_file in files:
        data = torch.load(source_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            if args.max_sequences and total_sequences >= args.max_sequences:
                break
            record = build_sequence_targets(data, seq_idx, name, args, body_model, device, preset_name, source_file)
            append_record(shard, record)
            total_sequences += 1
            total_frames += record['num_frames']
            if len(shard['name']) >= args.shard_size:
                flush()
            if args.progress_every and total_sequences % args.progress_every == 0:
                print(json.dumps({
                    'processed_sequences': total_sequences,
                    'processed_frames': total_frames,
                    'elapsed_sec': round(time.time() - started, 3),
                }))
        if args.max_sequences and total_sequences >= args.max_sequences:
            break
    flush()
    manifest = {
        'type': 'canonical_gt_control_cache_v1',
        'preset': preset_name,
        'dataset': args.dataset,
        'split': args.split,
        'source_cache': str(input_cache),
        'source_manifest': source_manifest,
        'created_at_unix': time.time(),
        'dt': float(args.dt),
        'euler_seq': args.euler_seq,
        'control_fit_contract': control_fit_contract(),
        'control_fit_solver': {
            'mode': CONTROL_FIT_DEFAULT_MODE,
            'implementation': 'scipy sparse solve of the same derivative-aware normal equations',
            'reason': 'dense TxT solve is too slow and memory-heavy for long DIP/AMASS sequences',
        },
        'coordinate_contract': {
            'pose_rot6d': 'SMPL local joint rotations encoded as first two rotation-matrix columns, shape [T,24,6].',
            'joint_angle_euler': 'SMPL local joint Euler angles, unwrapped over time, shape [T,72]. No translation is included.',
            'joint_pos_R': 'Root/body-frame SMPL joint positions in meters: p_RJ = (p_WJ - p_WR) @ R_WR under the existing GlobalPose row-vector convention.',
            'imu_RMB_6d': 'Selected source-cache RMB rotations encoded as 6D. The source RMB matrices are not regenerated here.',
            'pl_pRB_gR1': 'PL control state pRB[15]+gR1[3] from SMPL male FK with IMU vertex mask (1961,5424,1176,4662,411,3021); pRB meters, gR1 unit vector.',
            'root_trans_W': 'World/model-frame root translation from source tran_gt only when root_trans_policy marks it available.',
            'root_vel_W_fd': 'Central finite difference of root_trans_W in m/s when root translation is available.',
        },
        'root_translation_policy': {
            'requested': args.root_trans_policy,
            'effective': 'unavailable for DIP presets; available only when source tran_gt is treated as reliable',
            'dip_note': 'DIP-IMU root translation is not considered reliable GT here; no DIP root velocity GT is synthesized.',
        },
        'fields': {
            'pose_rot6d_control': '[T,24,6] derivative-aware controls for local pose rotations in 6D representation',
            'joint_angle_euler_control': '[T,72] derivative-aware controls for unwrapped local Euler joint angles',
            'joint_pos_R_control': '[T,24,3] derivative-aware controls for root/body-frame joint positions',
            'imu_RMB_6d_control': '[T,6,6] derivative-aware controls for source IMU orientation matrices encoded as 6D',
            'pl_pRB_gR1_control': '[T,18] derivative-aware controls for PL pRB/gR1 targets',
            'root_trans_W_control': '[T,3] derivative-aware controls for root translation, omitted when unavailable',
        },
        'cache_files': cache_files,
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'max_sequences': int(args.max_sequences),
        'shard_size': int(args.shard_size),
    }
    manifest_path = output_dir / 'gt_control_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({
        'output_manifest': str(manifest_path),
        'num_sequences': total_sequences,
        'num_frames': total_frames,
        'elapsed_sec': round(time.time() - started, 3),
    }, indent=2))
    return manifest_path


def resolved_args_for_preset(base_args, preset_name):
    preset = DEFAULT_PRESETS[preset_name]
    args = argparse.Namespace(**vars(base_args))
    args.input_cache = preset['input_cache']
    args.output_dir = str(Path(base_args.output_root) / preset_name)
    args.root_trans_policy = preset['root_trans_policy']
    args.dataset = preset['dataset']
    args.split = preset['split']
    return args


def parse_args():
    parser = argparse.ArgumentParser(description='Build canonical derivative-aware GT-control caches.')
    parser.add_argument('--preset', choices=sorted(DEFAULT_PRESETS), help='Build one known dataset/split preset.')
    parser.add_argument('--all-defaults', action='store_true', help='Build all default AMASS/DIP/TotalCapture presets.')
    parser.add_argument('--input-cache', help='Source baseline cache manifest or shard for custom builds.')
    parser.add_argument('--output-dir', help='Output directory for a custom build.')
    parser.add_argument('--output-root', default='data/dataset_work/GTControlCache')
    parser.add_argument('--dataset', default='custom')
    parser.add_argument('--split', default='custom')
    parser.add_argument('--root-trans-policy', choices=('auto', 'available', 'unavailable'), default='auto')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dt', type=float, default=CONTROL_FIT_DT)
    parser.add_argument('--euler-seq', default='XYZ')
    parser.add_argument('--shard-size', type=int, default=32)
    parser.add_argument('--fk-batch-size', type=int, default=2048)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--progress-every', type=int, default=10)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all_defaults:
        for preset_name in DEFAULT_PRESETS:
            build_cache(resolved_args_for_preset(args, preset_name), preset_name=preset_name)
        return
    if args.preset:
        build_cache(resolved_args_for_preset(args, args.preset), preset_name=args.preset)
        return
    if not args.input_cache or not args.output_dir:
        raise SystemExit('custom builds require --input-cache and --output-dir, or use --preset/--all-defaults')
    build_cache(args, preset_name=None)


if __name__ == '__main__':
    main()
