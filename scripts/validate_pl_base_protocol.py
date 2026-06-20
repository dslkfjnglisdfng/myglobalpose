#!/usr/bin/env python3
"""Validate that a joint-leaf NewPL cache uses the official PL-s1 base protocol."""

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
from pl_curve import normalize_gravity, pl_joint_leaf_target_from_pose


DEFAULT_CACHE = (
    'data/experiments/newpl_joint_leaf_acc_20260619/full/caches/'
    'baseline_jointtarget_84D/dip_test/pl_curve_cache_manifest.json'
)
DEFAULT_RAW_CACHE = (
    'data/experiments/newpl_v5_official_protocol_20260607/caches/'
    'dip_test_with_offset_r/baseline_cache_manifest.json'
)


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


def seq_item(value, idx, n):
    return value[idx][:n]


def angle_deg(a, b):
    a = torch.nn.functional.normalize(a.float(), dim=-1, eps=1e-8)
    b = torch.nn.functional.normalize(b.float(), dim=-1, eps=1e-8)
    dot = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(dot) * (180.0 / math.pi)


def summarize(values):
    if not values:
        return None
    x = torch.cat([v.reshape(-1).cpu().float() for v in values])
    return {
        'mean_deg': float(x.mean().item()),
        'max_deg': float(x.max().item()),
        'p95_deg': float(torch.quantile(x, 0.95).item()),
        'num_frames': int(x.numel()),
    }


def load_joint_cache(path, max_sequences=0):
    files, manifest = load_cache_files(path)
    if manifest is None or manifest.get('type') != 'pl_curve_joint_leaf_acc_cache_v1':
        raise ValueError(f'{path} must be pl_curve_joint_leaf_acc_cache_v1.')
    records = {}
    for cache_file in files:
        data = torch.load(resolve_cache_file(cache_file, path, manifest), map_location='cpu', weights_only=False)
        for idx, name in enumerate(data['name']):
            n = int(data['num_frames'][idx])
            records[str(name)] = {
                'pl_base': normalize_gravity(seq_item(data['pl_base'], idx, n).float()).cpu(),
                'pl_target': normalize_gravity(seq_item(data['pl_target'], idx, n).float()).cpu(),
                'num_frames': n,
            }
            if max_sequences and len(records) >= max_sequences:
                return manifest, records
    return manifest, records


def load_pose_prephysics(path, names, max_sequences=0):
    files, manifest = load_cache_files(path)
    records = {}
    wanted = set(names)
    for cache_file in files:
        data = torch.load(resolve_cache_file(cache_file, path, manifest), map_location='cpu', weights_only=False)
        if 'pose_prephysics' not in data:
            continue
        for idx, name in enumerate(data['name']):
            key = str(name)
            if key not in wanted:
                continue
            n = int(data['num_frames'][idx]) if 'num_frames' in data else int(data['pose_prephysics'][idx].shape[0])
            records[key] = {
                'pose_prephysics': seq_item(data['pose_prephysics'], idx, n).float().cpu(),
                'num_frames': n,
            }
            if max_sequences and len(records) >= max_sequences:
                return records
    return records


@torch.no_grad()
def pose_pre_gR(pose_pre, body_model, device, chunk_size):
    chunks = []
    for start in range(0, pose_pre.shape[0], chunk_size):
        pose = pose_pre[start:start + chunk_size].to(device=device, dtype=body_model._J.dtype)
        chunks.append(normalize_gravity(pl_joint_leaf_target_from_pose(pose, body_model))[:, 15:18].detach().cpu())
    return torch.cat(chunks, dim=0)


def declared_source_type(manifest):
    text = ' '.join(str(manifest.get(key, '')) for key in ('pl_base_source', 'base_mode', 'pl_base_source_detail'))
    if manifest.get('pl_base_source') == 'official_pl_s1' or manifest.get('base_mode') == 'official_pl_s1_prediction':
        return 'official_pl_s1'
    if 'pose_prephysics' in text or 'pose_pre' in text:
        return 'pose_prephysics_fk'
    return 'unknown'


def validate(args):
    manifest, records = load_joint_cache(args.cache, max_sequences=args.max_sequences)
    base_vs_target = [
        angle_deg(record['pl_base'][:, 15:18], record['pl_target'][:, 15:18])
        for record in records.values()
    ]
    base_gR1 = summarize(base_vs_target)
    source_type = declared_source_type(manifest)
    pose_pre_summary = None

    if args.raw_cache:
        raw_records = load_pose_prephysics(args.raw_cache, records.keys(), max_sequences=args.max_sequences)
        if raw_records:
            device = torch.device(args.device)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError(f'Requested {args.device}, but CUDA is not available.')
            body_model = art.ParametricModel('models/SMPL_male.pkl', device=device)
            pose_pre_angles = []
            for name, record in records.items():
                if name not in raw_records:
                    continue
                raw = raw_records[name]
                n = min(record['num_frames'], raw['num_frames'])
                gR = pose_pre_gR(raw['pose_prephysics'][:n], body_model, device, args.chunk_size)
                pose_pre_angles.append(angle_deg(record['pl_base'][:n, 15:18], gR))
            pose_pre_summary = summarize(pose_pre_angles)
            if pose_pre_summary and pose_pre_summary['mean_deg'] < args.pose_pre_match_threshold_deg:
                source_type = 'pose_prephysics_fk'

    manifest_text = json.dumps({
        'base_mode': manifest.get('base_mode'),
        'pl_base_source': manifest.get('pl_base_source'),
        'pl_base_source_detail': manifest.get('pl_base_source_detail'),
        'pl_base_field': (manifest.get('fields') or {}).get('pl_base'),
        'protocol_check': manifest.get('protocol_check'),
    }, default=str)
    pose_pre_in_pipeline = 'pose_prephysics' in manifest_text or 'pose_pre FK' in manifest_text
    base_too_good = base_gR1 is not None and base_gR1['mean_deg'] < args.min_base_gR1_deg
    is_valid = source_type == 'official_pl_s1' and not pose_pre_in_pipeline and not base_too_good
    payload = {
        'status': 'ok' if is_valid else 'invalid',
        'cache': str(args.cache),
        'raw_cache': str(args.raw_cache) if args.raw_cache else None,
        'base_source_type': source_type,
        'base_gR1_angle_deg': None if base_gR1 is None else base_gR1['mean_deg'],
        'base_gR1_angle_summary': base_gR1,
        'base_vs_pose_prephysics_gR1_angle_summary': pose_pre_summary,
        'pose_prephysics_in_pl_base_pipeline': bool(pose_pre_in_pipeline),
        'base_gR1_below_min_threshold': bool(base_too_good),
        'min_base_gR1_deg': float(args.min_base_gR1_deg),
        'is_protocol_valid': bool(is_valid),
        'protocol_check': {
            'pl_base': source_type,
            'comparable_to_v5': bool(is_valid),
        },
        'rule': 'If base_source_type != official_pl_s1, cache is INVALID for NewPL-v5 comparison.',
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + '\n')
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument('--raw-cache', type=Path, default=Path(DEFAULT_RAW_CACHE))
    parser.add_argument('--output-json', type=Path, default=None)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--min-base-gR1-deg', type=float, default=5.0)
    parser.add_argument('--pose-pre-match-threshold-deg', type=float, default=0.1)
    return parser.parse_args()


def main():
    payload = validate(parse_args())
    print(json.dumps({
        'base_source_type': payload['base_source_type'],
        'base_gR1_angle_deg': payload['base_gR1_angle_deg'],
        'is_protocol_valid': payload['is_protocol_valid'],
        'status': payload['status'],
    }, indent=2))


if __name__ == '__main__':
    main()
