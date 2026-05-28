import argparse
import json
from pathlib import Path

import torch
import tqdm
import articulate as art

from l4_generate_baseline_cache import make_official_inputs
from net import GPNet
from test import MotionEvaluator


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pose_tran(data, seq_idx, max_frames):
    pose = art.math.axis_angle_to_rotation_matrix(data['pose'][seq_idx]).view(-1, 24, 3, 3)
    tran = data['tran'][seq_idx]
    if max_frames:
        pose = pose[:max_frames]
        tran = tran[:max_frames]
    return pose, tran


@torch.no_grad()
def run_net(net, data, seq_idx, max_frames):
    pose_gt, tran_gt = load_pose_tran(data, seq_idx, max_frames)
    aM, wM, RMB = make_official_inputs(data, seq_idx)
    if max_frames:
        aM, wM, RMB = aM[:max_frames], wM[:max_frames], RMB[:max_frames]
    net.rnn_initialize(pose_gt[0])
    pose_out = torch.zeros_like(pose_gt)
    tran_out = torch.zeros_like(tran_gt)
    prephysics_changed = []
    velocity_changed = []
    delta_v_norms = []
    for frame_idx in tqdm.trange(pose_gt.shape[0], leave=False):
        pose_out[frame_idx], tran_out[frame_idx] = net.forward_frame(aM[frame_idx], wM[frame_idx], RMB[frame_idx])
        debug = getattr(net, 'last_l4_prephysics_debug', {})
        prephysics_changed.append(bool(debug.get('changed', False)))
        velocity_changed.append(bool(debug.get('velocity_changed', False)))
        delta_v_norms.append(float(debug.get('delta_v_root_norm', 0.0)))
    return pose_out, tran_out, pose_gt, tran_gt, prephysics_changed, velocity_changed, delta_v_norms


def metric_dict(evaluator, pose_p, pose_t, tran_p, tran_t):
    metric = evaluator(pose_p.to(DEVICE), pose_t.to(DEVICE), tran_p.to(DEVICE), tran_t.to(DEVICE)).cpu()
    return {name: {'mean': float(metric[i, 0]), 'std': float(metric[i, 1])} for i, name in enumerate(MotionEvaluator.names)}


def compare_dataset(path, max_sequences, max_frames):
    data = torch.load(path, map_location='cpu')
    evaluator = MotionEvaluator()
    results = []
    for seq_idx in range(min(max_sequences, len(data['pose']))):
        base = GPNet(enable_l4_prephysics=False).eval().to(DEVICE)
        zero = GPNet(enable_l4_prephysics=True).eval().to(DEVICE)
        pose_base, tran_base, pose_gt, tran_gt, _, _, _ = run_net(base, data, seq_idx, max_frames)
        pose_zero, tran_zero, _, _, changed, v_changed, delta_v_norms = run_net(zero, data, seq_idx, max_frames)
        base_metric = metric_dict(evaluator, pose_base, pose_gt, tran_base, tran_gt)
        zero_metric = metric_dict(evaluator, pose_zero, pose_gt, tran_zero, tran_gt)
        delta_metric = {
            name: {
                'mean_delta': zero_metric[name]['mean'] - base_metric[name]['mean'],
                'std_delta': zero_metric[name]['std'] - base_metric[name]['std'],
            }
            for name in MotionEvaluator.names
        }
        results.append({
            'sequence_index': seq_idx,
            'num_frames': int(pose_gt.shape[0]),
            'max_pose_element_error': float((pose_zero - pose_base).abs().max()),
            'max_translation_error': float((tran_zero - tran_base).abs().max()),
            'any_prephysics_changed': any(changed),
            'any_velocity_changed': any(v_changed),
            'max_delta_v_root_norm': max(delta_v_norms) if delta_v_norms else 0.0,
            'baseline_metrics': base_metric,
            'zero_residual_metrics': zero_metric,
            'metric_delta_vs_baseline': delta_metric,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description='L4 pre-physics zero-residual identity diagnostics.')
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--max-sequences', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=60)
    parser.add_argument('--output', type=Path, default=Path('data/experiments/l4_prephysics_diagnostics/zero_identity.json'))
    args = parser.parse_args()

    out = {
        'max_sequences': args.max_sequences,
        'max_frames': args.max_frames,
        'datasets': {},
        'gate': 'zero-residual pre-physics identity',
    }
    for dataset in args.datasets:
        out['datasets'][dataset] = compare_dataset(Path(dataset), args.max_sequences, args.max_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
