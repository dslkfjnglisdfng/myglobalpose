import argparse
import json
import time
from pathlib import Path

import torch

import articulate as art
from l4_generate_baseline_cache import load_input_specs, make_official_inputs, trim_sequence
from net import GPNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pick_spec(input_path, shard_index):
    specs, _ = load_input_specs(input_path)
    if shard_index is None:
        return specs[0]
    for spec in specs:
        if spec['index'] == shard_index:
            return spec
    raise ValueError(f'No shard with index {shard_index}.')


def load_sequence(input_path, shard_index, sequence_index, max_frames):
    spec = pick_spec(input_path, shard_index)
    data = torch.load(spec['path'], map_location='cpu')
    pose_gt = art.math.axis_angle_to_rotation_matrix(data['pose'][sequence_index]).view(-1, 24, 3, 3)
    tran_gt = data['tran'][sequence_index]
    aM, wM, RMB = make_official_inputs(data, sequence_index)
    pose_gt, tran_gt, aM, wM, RMB = trim_sequence(pose_gt, tran_gt, aM, wM, RMB, max_frames)
    name = data['name'][sequence_index] if 'name' in data else f"{Path(spec['path']).stem}:{sequence_index}"
    return {
        'spec': spec,
        'name': name,
        'pose_gt': pose_gt,
        'tran_gt': tran_gt,
        'aM': aM,
        'wM': wM,
        'RMB': RMB,
    }


@torch.no_grad()
def run_full_trace(sequence, euler_seq):
    net = GPNet(enable_l4_prephysics=True).eval().to(DEVICE)
    net.rnn_initialize(sequence['pose_gt'][0])
    q75 = []
    v_root = []
    stationary = []
    started = time.perf_counter()
    for frame_idx in range(sequence['pose_gt'].shape[0]):
        net.forward_frame(
            sequence['aM'][frame_idx],
            sequence['wM'][frame_idx],
            sequence['RMB'][frame_idx],
        )
        debug = net.last_l4_prephysics_debug
        q75.append(debug['q75_before'].cpu())
        v_root.append(debug['v_root_vr'].cpu())
        stationary.append(debug['stationary_prob'].cpu())
    elapsed = time.perf_counter() - started
    return {
        'q75': torch.stack(q75),
        'v_root': torch.stack(v_root),
        'stationary': torch.stack(stationary),
        'elapsed_sec': elapsed,
        'frames_per_sec': float(sequence['pose_gt'].shape[0] / elapsed) if elapsed > 0 else 0.0,
    }


@torch.no_grad()
def run_neural_only(sequence, euler_seq):
    net = GPNet(enable_l4_prephysics=False).eval().to(DEVICE)
    net.rnn_initialize(sequence['pose_gt'][0])
    q75 = []
    v_root = []
    stationary = []
    started = time.perf_counter()
    for frame_idx in range(sequence['pose_gt'].shape[0]):
        features = net.forward_prephysics_features(
            sequence['aM'][frame_idx],
            sequence['wM'][frame_idx],
            sequence['RMB'][frame_idx],
            prephysics_tran=None,
            euler_seq=euler_seq,
        )
        q75.append(features['q75_prephysics'].cpu())
        v_root.append(features['v_root_vr'].cpu())
        stationary.append(features['stationary_prob'].cpu())
    elapsed = time.perf_counter() - started
    return {
        'q75': torch.stack(q75),
        'v_root': torch.stack(v_root),
        'stationary': torch.stack(stationary),
        'elapsed_sec': elapsed,
        'frames_per_sec': float(sequence['pose_gt'].shape[0] / elapsed) if elapsed > 0 else 0.0,
    }


def diff_stats(x, y):
    diff = (x - y).abs()
    return {
        'mean': float(diff.mean()),
        'max': float(diff.max()),
    }


def main():
    parser = argparse.ArgumentParser(description='Check and benchmark neural-only prephysics extraction.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--shard-index', type=int, default=None)
    parser.add_argument('--sequence-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=120)
    parser.add_argument('--euler-seq', default='XYZ')
    parser.add_argument('--output-json', required=True)
    args = parser.parse_args()

    sequence = load_sequence(args.input, args.shard_index, args.sequence_index, args.max_frames)
    full = run_full_trace(sequence, args.euler_seq)
    neural = run_neural_only(sequence, args.euler_seq)
    result = {
        'input': args.input,
        'source_path': str(sequence['spec']['path']),
        'shard_index': sequence['spec']['index'],
        'sequence_index': args.sequence_index,
        'sequence_name': sequence['name'],
        'num_frames': int(sequence['pose_gt'].shape[0]),
        'full_physics_frames_per_sec': full['frames_per_sec'],
        'neural_only_frames_per_sec': neural['frames_per_sec'],
        'speedup': neural['frames_per_sec'] / full['frames_per_sec'] if full['frames_per_sec'] > 0 else 0.0,
        'q75_all_abs_diff': diff_stats(full['q75'], neural['q75']),
        'q75_translation_abs_diff': diff_stats(full['q75'][:, :3], neural['q75'][:, :3]),
        'q75_rotation_abs_diff': diff_stats(full['q75'][:, 3:], neural['q75'][:, 3:]),
        'v_root_vr_abs_diff': diff_stats(full['v_root'], neural['v_root']),
        'stationary_prob_abs_diff': diff_stats(full['stationary'], neural['stationary']),
        'note': 'Neural-only q75 uses zero translation because carticulate physics state is intentionally not read; rotation, v_root_vr, and stationary_prob are the required equality checks.',
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
