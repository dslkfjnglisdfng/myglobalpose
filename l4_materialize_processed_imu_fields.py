import argparse
import json
from pathlib import Path

import torch


def resolve_path(path, base):
    out = Path(path)
    if not out.is_absolute() and not out.exists():
        out = Path(base) / out
    return out


def load_cache_specs(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    specs = []
    for item in manifest['cache_files']:
        specs.append((resolve_path(item['path'], path.parent), item))
    return manifest, specs


def materialize(input_manifest, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest, specs = load_cache_specs(input_manifest)
    manifest_files = []
    view_counts = {}
    for shard_idx, (cache_path, item) in enumerate(specs):
        data = torch.load(cache_path, map_location='cpu')
        missing = [key for key in ('aM', 'wM', 'RMB') if key not in data]
        if missing:
            raise KeyError(f'{cache_path} missing required fields: {missing}')
        nseq = len(data['name'])
        for key in ('aM', 'wM', 'RMB'):
            if not isinstance(data[key], list) or len(data[key]) != nseq:
                raise RuntimeError(f'{cache_path} field {key} is not a per-sequence list matching name.')
        out = {key: list(value) if isinstance(value, list) else value for key, value in data.items()}
        out['l4_aM'] = [value.float().clone() for value in data['aM']]
        out['l4_wM'] = [value.float().clone() for value in data['wM']]
        out['l4_RMB'] = [value.float().clone() for value in data['RMB']]
        if 'view_type' in data:
            for view_type in data['view_type']:
                view_counts[str(view_type)] = view_counts.get(str(view_type), 0) + 1
        dest_path = output_dir / f'baseline_cache_shard{shard_idx:05d}.pt'
        torch.save(out, dest_path)
        manifest_files.append({
            'path': str(dest_path),
            'source_cache_path': str(cache_path),
            'num_sequences': nseq,
            'num_frames': int(sum(data['num_frames'])) if 'num_frames' in data else int(item.get('num_frames', 0)),
        })
    manifest = dict(source_manifest)
    manifest.update({
        'cache_type': 'processed_imu_materialized',
        'source_cache_manifest': str(input_manifest),
        'source_manifest': source_manifest,
        'cache_files': manifest_files,
        'num_sequences': int(sum(item['num_sequences'] for item in manifest_files)),
        'num_frames': int(sum(item['num_frames'] for item in manifest_files)),
        'processed_imu_fields': {
            'l4_aM': 'materialized copy of aM from the source cache',
            'l4_wM': 'materialized copy of wM from the source cache',
            'l4_RMB': 'materialized copy of RMB from the source cache',
        },
        'source_view_counts': view_counts,
    })
    manifest_path = output_dir / 'baseline_cache_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description='Materialize processed IMU fields l4_aM/l4_wM/l4_RMB from an already processed cache.')
    parser.add_argument('--input-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    manifest_path, manifest = materialize(args.input_manifest, args.output_dir)
    print(json.dumps({
        'manifest': str(manifest_path),
        'num_sequences': manifest['num_sequences'],
        'num_frames': manifest['num_frames'],
        'source_view_counts': manifest['source_view_counts'],
    }, indent=2))


if __name__ == '__main__':
    main()
