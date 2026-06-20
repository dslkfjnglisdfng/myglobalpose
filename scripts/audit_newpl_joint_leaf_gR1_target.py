#!/usr/bin/env python3
"""Audit whether joint-leaf NewPL cache gravity targets match legacy PL gravity.

The audit compares only the 3D gravity block gR1 = -R_root[:, 1].  Positions are
intentionally ignored because the new cache uses joint-leaf positions while the
legacy PL target uses vertex positions.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import articulate as art
from l4_train_diverse_short import load_cache_files
from pl_curve import normalize_gravity, pl_joint_leaf_target_from_pose, pl_target_from_pose


DEFAULT_JOINT_CACHE = (
    'data/experiments/newpl_joint_leaf_acc_20260619/full/caches/'
    'baseline_jointtarget_84D/dip_test/pl_curve_cache_manifest.json'
)
DEFAULT_RAW_CACHE = (
    'data/experiments/newpl_v5_official_protocol_20260607/caches/'
    'dip_test_with_offset_r/baseline_cache_manifest.json'
)
DEFAULT_GT_CONTROL_CACHE = '/home/lingfeng/projects/data/dataset_work/dip/gt_control/test/manifest.json'
LEGACY_IMU_VERTICES = (1961, 5424, 1176, 4662, 411, 3021)


def resolve_cache_file(path_text, manifest_path=None, manifest=None):
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = []
    if manifest and manifest.get('root'):
        candidates.append(Path(manifest['root']) / path)
    if manifest_path is not None:
        candidates.append(Path(manifest_path).resolve().parent / path)
    candidates.extend([Path.cwd() / path, ROOT / path, path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def angle_deg(a, b):
    a = torch.nn.functional.normalize(a.float(), dim=-1, eps=1e-8)
    b = torch.nn.functional.normalize(b.float(), dim=-1, eps=1e-8)
    dot = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(dot) * (180.0 / math.pi)


def summarize_angles(values):
    if not values:
        return None
    x = torch.cat([value.reshape(-1).float().cpu() for value in values])
    if x.numel() == 0:
        return None
    return {
        'mean_deg': float(x.mean().item()),
        'max_deg': float(x.max().item()),
        'p95_deg': float(torch.quantile(x, 0.95).item()),
        'num_frames': int(x.numel()),
    }


def seq_item(value, idx, n):
    item = value[idx]
    return item[:n]


def load_joint_cache_gR(manifest_path):
    files, manifest = load_cache_files(manifest_path)
    if manifest is None or manifest.get('type') != 'pl_curve_joint_leaf_acc_cache_v1':
        raise ValueError(f'{manifest_path} must be pl_curve_joint_leaf_acc_cache_v1.')
    records = {}
    for cache_file in files:
        data = torch.load(
            resolve_cache_file(cache_file, manifest_path=manifest_path, manifest=manifest),
            map_location='cpu',
            weights_only=False,
        )
        for idx, name in enumerate(data['name']):
            n = int(data['num_frames'][idx]) if 'num_frames' in data else int(data['pl_target'][idx].shape[0])
            target = normalize_gravity(seq_item(data['pl_target'], idx, n).float())
            records[str(name)] = {
                'gR': target[:, 15:18].cpu(),
                'num_frames': n,
            }
    return manifest, records


def load_raw_pose_gt(manifest_path):
    files, manifest = load_cache_files(manifest_path)
    records = {}
    for cache_file in files:
        data = torch.load(
            resolve_cache_file(cache_file, manifest_path=manifest_path, manifest=manifest),
            map_location='cpu',
            weights_only=False,
        )
        if 'pose_gt' not in data:
            raise KeyError(f'{cache_file} missing pose_gt.')
        for idx, name in enumerate(data['name']):
            n = int(data['num_frames'][idx]) if 'num_frames' in data else int(data['pose_gt'][idx].shape[0])
            records[str(name)] = {
                'pose_gt': seq_item(data['pose_gt'], idx, n).float().cpu(),
                'num_frames': n,
            }
    return manifest, records


def load_gt_control_gR(manifest_path):
    if not manifest_path:
        return None, {}
    files, manifest = load_cache_files(manifest_path)
    records = {}
    for cache_file in files:
        data = torch.load(
            resolve_cache_file(cache_file, manifest_path=manifest_path, manifest=manifest),
            map_location='cpu',
            weights_only=False,
        )
        if 'pl_pRB_gR1' not in data:
            raise KeyError(f'{cache_file} missing pl_pRB_gR1.')
        for idx, name in enumerate(data['name']):
            n = int(data['num_frames'][idx]) if 'num_frames' in data else int(data['pl_pRB_gR1'][idx].shape[0])
            target = normalize_gravity(seq_item(data['pl_pRB_gR1'], idx, n).float())
            records[str(name)] = {
                'gR': target[:, 15:18].cpu(),
                'num_frames': n,
            }
    return manifest, records


@torch.no_grad()
def fk_gravity_targets(pose_gt, legacy_body_model, joint_body_model, device, chunk_size):
    old_chunks = []
    joint_chunks = []
    for start in range(0, pose_gt.shape[0], chunk_size):
        pose = pose_gt[start:start + chunk_size].to(device=device, dtype=legacy_body_model._J.dtype)
        old_chunks.append(normalize_gravity(pl_target_from_pose(pose, legacy_body_model))[:, 15:18].detach().cpu())
        joint_chunks.append(
            normalize_gravity(pl_joint_leaf_target_from_pose(pose, joint_body_model))[:, 15:18].detach().cpu()
        )
    return torch.cat(old_chunks, dim=0), torch.cat(joint_chunks, dim=0)


def print_summary(summary):
    print('metric,mean_deg,max_deg,p95_deg,num_frames')
    for key, value in summary['metrics'].items():
        if value is None:
            print(f'{key},NA,NA,NA,0')
        else:
            print(
                f'{key},{value["mean_deg"]:.10f},{value["max_deg"]:.10f},'
                f'{value["p95_deg"]:.10f},{value["num_frames"]}'
            )


def run_audit(args):
    joint_manifest, joint_records = load_joint_cache_gR(args.joint_cache)
    _raw_manifest, raw_records = load_raw_pose_gt(args.raw_cache)
    gt_manifest, gt_records = load_gt_control_gR(args.gt_control_cache)

    shared_names = sorted(set(joint_records) & set(raw_records))
    if args.max_sequences:
        shared_names = shared_names[:args.max_sequences]
    if not shared_names:
        raise RuntimeError('No shared sequences between joint cache and raw cache.')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'Requested {args.device}, but CUDA is not available.')
    legacy_body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=LEGACY_IMU_VERTICES, device=device)
    joint_body_model = art.ParametricModel('models/SMPL_male.pkl', device=device)

    angle_buckets = {
        'angle_cache_gR_old_gR': [],
        'angle_cache_gR_joint_leaf_gR': [],
        'angle_old_gR_joint_leaf_gR': [],
        'angle_gt_control_gR_old_gR': [],
    }
    rows = []
    total_frames = 0
    gt_shared = 0

    for name in shared_names:
        joint = joint_records[name]
        raw = raw_records[name]
        n = min(joint['num_frames'], raw['num_frames'])
        if n <= 0:
            continue
        cache_gR = joint['gR'][:n]
        old_gR, joint_leaf_gR = fk_gravity_targets(
            raw['pose_gt'][:n],
            legacy_body_model,
            joint_body_model,
            device,
            args.chunk_size,
        )

        row = {
            'name': name,
            'num_frames': int(n),
            'has_gt_control': name in gt_records,
        }
        for metric, values in (
            ('angle_cache_gR_old_gR', angle_deg(cache_gR, old_gR)),
            ('angle_cache_gR_joint_leaf_gR', angle_deg(cache_gR, joint_leaf_gR)),
            ('angle_old_gR_joint_leaf_gR', angle_deg(old_gR, joint_leaf_gR)),
        ):
            angle_buckets[metric].append(values)
            row[metric] = summarize_angles([values])

        if name in gt_records:
            gt_shared += 1
            gt = gt_records[name]
            m = min(n, gt['num_frames'])
            values = angle_deg(gt['gR'][:m], old_gR[:m])
            angle_buckets['angle_gt_control_gR_old_gR'].append(values)
            row['angle_gt_control_gR_old_gR'] = summarize_angles([values])
        else:
            row['angle_gt_control_gR_old_gR'] = None

        rows.append(row)
        total_frames += int(n)

    summary = {
        'status': 'ok',
        'joint_cache': str(args.joint_cache),
        'raw_cache': str(args.raw_cache),
        'gt_control_cache': str(args.gt_control_cache) if args.gt_control_cache else None,
        'joint_cache_type': joint_manifest.get('type'),
        'joint_feature_mode': joint_manifest.get('feature_mode'),
        'legacy_pl_vertex_mask': list(LEGACY_IMU_VERTICES),
        'gt_control_manifest_type': None if gt_manifest is None else gt_manifest.get('type'),
        'num_shared_sequences': len(rows),
        'num_shared_frames': total_frames,
        'num_gt_control_shared_sequences': gt_shared,
        'metrics': {key: summarize_angles(values) for key, values in angle_buckets.items()},
        'rows': rows,
        'interpretation': (
            'All four angle groups should be close to 0 degrees. Non-zero cache-vs-old or '
            'cache-vs-joint_leaf angles mean the joint-leaf cache gravity target is not '
            'directly comparable with legacy PL gR1.'
        ),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-cache', type=Path, default=Path(DEFAULT_JOINT_CACHE))
    parser.add_argument('--raw-cache', type=Path, default=Path(DEFAULT_RAW_CACHE))
    parser.add_argument('--gt-control-cache', type=Path, default=Path(DEFAULT_GT_CONTROL_CACHE))
    parser.add_argument('--output-json', type=Path, default=None)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_audit(args)
    print_summary(summary)
    if args.output_json:
        print(f'wrote {args.output_json}')


if __name__ == '__main__':
    main()
