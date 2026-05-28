import argparse
import json
from pathlib import Path

import torch
import tqdm

import articulate as art
from l4_q75_utils import pose_tran_to_q75
from net import GPNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = torch.tensor([0, -9.8, 0])


def load_input_specs(input_path):
    path = Path(input_path)
    if path.suffix == '.json':
        manifest = json.loads(path.read_text())
        specs = []
        for idx, shard in enumerate(manifest['shards']):
            shard_path = Path(shard['path'])
            if not shard_path.is_absolute() and not shard_path.exists():
                shard_path = path.parent / shard_path
            specs.append({'path': shard_path, 'index': idx, 'manifest': shard})
        return specs, manifest
    return [{'path': path, 'index': 0, 'manifest': None}], None


def assert_not_real_test_cache(input_path, allow_real_test_cache):
    normalized = str(Path(input_path))
    if 'data/test_datasets' in normalized and not allow_real_test_cache:
        raise ValueError(
            'Refusing to generate a training cache from official test datasets. '
            'Pass --allow-real-test-cache only for bounded diagnostics, never for training.'
        )


def make_official_inputs(data, seq_idx):
    if all(key in data for key in ('RIM', 'RIS', 'RSB', 'aS', 'wS')):
        aS = data['aS'][seq_idx]
        wS = data['wS'][seq_idx]
        RIS = data['RIS'][seq_idx]
        RIM = data['RIM'][seq_idx]
        RSB = data['RSB'][seq_idx]
        RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB)
        aM = RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + G
        wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1)
        return aM.to(DEVICE), wM.to(DEVICE), RMB.to(DEVICE)
    if all(key in data for key in ('aM', 'wM', 'RMB')):
        return data['aM'][seq_idx].to(DEVICE), data['wM'][seq_idx].to(DEVICE), data['RMB'][seq_idx].to(DEVICE)
    raise KeyError('Input shard must contain raw-style RIM/RIS/RSB/aS/wS or saved aM/wM/RMB fields.')


def trim_sequence(pose_gt, tran_gt, aM, wM, RMB, max_frames):
    if max_frames:
        pose_gt = pose_gt[:max_frames]
        tran_gt = tran_gt[:max_frames]
        aM = aM[:max_frames]
        wM = wM[:max_frames]
        RMB = RMB[:max_frames]
    return pose_gt, tran_gt, aM, wM, RMB


@torch.no_grad()
def run_baseline_sequence(net, data, seq_idx, max_frames, euler_seq, save_prephysics=False, neural_only_prephysics=False):
    pose_gt = art.math.axis_angle_to_rotation_matrix(data['pose'][seq_idx]).view(-1, 24, 3, 3)
    tran_gt = data['tran'][seq_idx]
    aM, wM, RMB = make_official_inputs(data, seq_idx)
    pose_gt, tran_gt, aM, wM, RMB = trim_sequence(pose_gt, tran_gt, aM, wM, RMB, max_frames)

    net.rnn_initialize(pose_gt[0])
    gt_q75 = pose_tran_to_q75(pose_gt, tran_gt, euler_seq=euler_seq)
    if neural_only_prephysics:
        q75_prephysics = []
        pose_prephysics = []
        v_root_vr = []
        stationary_prob = []
        for frame_idx in range(pose_gt.shape[0]):
            features = net.forward_prephysics_features(
                aM[frame_idx],
                wM[frame_idx],
                RMB[frame_idx],
                prephysics_tran=None,
                euler_seq=euler_seq,
            )
            q75_prephysics.append(features['q75_prephysics'])
            pose_prephysics.append(features['pose_prephysics'])
            v_root_vr.append(features['v_root_vr'])
            stationary_prob.append(features['stationary_prob'])
        return {
            'pose_gt': pose_gt.cpu(),
            'tran_gt': tran_gt.cpu(),
            'q75_gt': gt_q75.cpu(),
            'q75_prephysics': torch.stack(q75_prephysics).cpu(),
            'pose_prephysics': torch.stack(pose_prephysics).cpu(),
            'v_root_vr': torch.stack(v_root_vr).cpu(),
            'stationary_prob': torch.stack(stationary_prob).cpu(),
            'aM': aM.cpu(),
            'wM': wM.cpu(),
            'RMB': RMB.cpu(),
            'num_frames': int(pose_gt.shape[0]),
        }

    pose_base = torch.zeros_like(pose_gt)
    tran_base = torch.zeros_like(tran_gt)
    q75_prephysics = []
    pose_prephysics = []
    v_root_vr = []
    stationary_prob = []
    for frame_idx in range(pose_gt.shape[0]):
        pose_base[frame_idx], tran_base[frame_idx] = net.forward_frame(aM[frame_idx], wM[frame_idx], RMB[frame_idx])
        if save_prephysics:
            debug = getattr(net, 'last_l4_prephysics_debug', {})
            if 'q75_before' not in debug:
                raise RuntimeError('Pre-physics trace was requested, but GPNet did not expose q75_before.')
            q75_prephysics.append(debug['q75_before'].cpu())
            pose_prephysics.append(debug.get('pose_before', torch.zeros(24, 3, 3)).cpu())
            if 'v_root_vr' not in debug:
                raise RuntimeError('Pre-physics trace was requested, but GPNet did not expose v_root_vr.')
            v_root_vr.append(debug['v_root_vr'].cpu())
            stationary_prob.append(debug.get('stationary_prob', torch.zeros(5)).cpu())

    baseline_q75 = pose_tran_to_q75(pose_base, tran_base, euler_seq=euler_seq)
    record = {
        'pose_gt': pose_gt.cpu(),
        'tran_gt': tran_gt.cpu(),
        'pose_baseline': pose_base.cpu(),
        'tran_baseline': tran_base.cpu(),
        'q75_baseline': baseline_q75.cpu(),
        'q75_gt': gt_q75.cpu(),
        'aM': aM.cpu(),
        'wM': wM.cpu(),
        'RMB': RMB.cpu(),
        'num_frames': int(pose_gt.shape[0]),
    }
    if save_prephysics:
        record['q75_prephysics'] = torch.stack(q75_prephysics).cpu()
        record['pose_prephysics'] = torch.stack(pose_prephysics).cpu()
        record['v_root_vr'] = torch.stack(v_root_vr).cpu()
        record['stationary_prob'] = torch.stack(stationary_prob).cpu()
    return record


def empty_cache():
    return {
        'name': [],
        'pose_gt': [],
        'tran_gt': [],
        'pose_baseline': [],
        'tran_baseline': [],
        'q75_baseline': [],
        'q75_gt': [],
        'pose_prephysics': [],
        'aM': [],
        'wM': [],
        'RMB': [],
        'num_frames': [],
        'q75_prephysics': [],
        'v_root_vr': [],
        'stationary_prob': [],
    }


def append_cache(cache, name, record):
    cache['name'].append(name)
    for key in ('pose_gt', 'tran_gt', 'pose_baseline', 'tran_baseline', 'q75_baseline', 'q75_gt', 'pose_prephysics', 'aM', 'wM', 'RMB', 'num_frames'):
        if key in record:
            cache[key].append(record[key])
    if 'q75_prephysics' in record:
        cache['q75_prephysics'].append(record['q75_prephysics'])
    if 'v_root_vr' in record:
        cache['v_root_vr'].append(record['v_root_vr'])
    if 'stationary_prob' in record:
        cache['stationary_prob'].append(record['stationary_prob'])


def process_cache(args):
    assert_not_real_test_cache(args.input, args.allow_real_test_cache)
    specs, input_manifest = load_input_specs(args.input)
    if args.shard_indices:
        wanted = {int(item) for item in args.shard_indices.split(',') if item.strip()}
        specs = [spec for spec in specs if spec['index'] in wanted]
        if not specs:
            raise ValueError(f'No input shards matched --shard-indices={args.shard_indices}')
    elif args.max_shards:
        specs = specs[:args.max_shards]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.neural_only_prephysics and not args.save_prephysics:
        args.save_prephysics = True
    net = GPNet(enable_l4_prephysics=(args.save_prephysics and not args.neural_only_prephysics)).eval().to(DEVICE)
    output_manifest = {
        'source_input': args.input,
        'source_manifest': input_manifest,
        'device': str(DEVICE),
        'euler_seq': args.euler_seq,
        'max_frames': args.max_frames,
        'max_sequences': args.max_sequences,
        'save_prephysics': args.save_prephysics,
        'neural_only_prephysics': args.neural_only_prephysics,
        'cache_files': [],
        'num_sequences': 0,
        'num_frames': 0,
    }

    processed_total = 0
    for spec in specs:
        data = torch.load(spec['path'], map_location='cpu')
        cache = empty_cache()
        failures = []
        nseq = len(data['pose'])
        processed_in_shard = 0
        for seq_idx in tqdm.trange(nseq, desc=f"cache shard {spec['index']}"):
            if args.max_sequences and processed_total >= args.max_sequences:
                break
            if args.max_sequences_per_shard and processed_in_shard >= args.max_sequences_per_shard:
                break
            try:
                record = run_baseline_sequence(
                    net,
                    data,
                    seq_idx,
                    args.max_frames,
                    args.euler_seq,
                    save_prephysics=args.save_prephysics,
                    neural_only_prephysics=args.neural_only_prephysics,
                )
                name = data['name'][seq_idx] if 'name' in data else f"{spec['path'].stem}:{seq_idx}"
                append_cache(cache, name, record)
                processed_total += 1
                processed_in_shard += 1
            except Exception as exc:
                failures.append({
                    'sequence_index': seq_idx,
                    'error': f'{type(exc).__name__}: {exc}',
                })

        if cache['name']:
            cache_path = output_dir / f"baseline_cache_shard{spec['index']:05d}.pt"
            torch.save(cache, cache_path)
            num_frames = int(sum(cache['num_frames']))
            output_manifest['cache_files'].append({
                'path': str(cache_path),
                'source_path': str(spec['path']),
                'num_sequences': len(cache['name']),
                'num_frames': num_frames,
                'failures': failures,
            })
            output_manifest['num_sequences'] += len(cache['name'])
            output_manifest['num_frames'] += num_frames
            print(f"Saved {cache_path}: {len(cache['name'])} sequences / {num_frames} frames")
        if args.max_sequences and processed_total >= args.max_sequences:
            break

    if output_manifest['num_sequences'] == 0:
        raise RuntimeError('No baseline cache sequence was generated.')
    manifest_path = output_dir / 'baseline_cache_manifest.json'
    manifest_path.write_text(json.dumps(output_manifest, indent=2))
    print(f"Saved baseline-cache manifest to {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate frozen-GlobalPose baseline q75 cache for L4 training.')
    parser.add_argument('--input', required=True, help='AMASS shard .pt or sharded manifest .json.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-shards', type=int, default=0)
    parser.add_argument('--shard-indices', default='', help='Comma-separated source shard indices to cache, e.g. 0,35,70,105.')
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--max-sequences-per-shard', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--euler-seq', default='XYZ')
    parser.add_argument('--allow-real-test-cache', action='store_true')
    parser.add_argument('--save-prephysics', action='store_true', help='Enable zero-residual pre-physics hook and save q75 before VR/physics.')
    parser.add_argument('--neural-only-prephysics', action='store_true', help='Extract PL/IK/VR pre-physics features without carticulate physics.')
    return parser.parse_args()


if __name__ == '__main__':
    process_cache(parse_args())
