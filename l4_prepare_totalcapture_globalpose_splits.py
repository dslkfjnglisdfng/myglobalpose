import argparse
import json
from pathlib import Path

import torch


SPLIT_PREFIXES = {
    'train': ('s1_', 's2_', 's3_'),
    'val': ('s4_',),
    'test': ('s5_',),
}


def subset(data, indices):
    out = {}
    for key, value in data.items():
        if isinstance(value, list):
            out[key] = [value[idx] for idx in indices]
        else:
            out[key] = value
    return out


def summarize(data):
    return {
        'num_sequences': len(data['name']),
        'num_frames': int(sum(seq.shape[0] for seq in data['pose'])),
        'names': [str(name) for name in data['name']],
        'frames_per_sequence': {
            str(name): int(seq.shape[0])
            for name, seq in zip(data['name'], data['pose'])
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Split GlobalPose-format TotalCapture data into S1-S3/S4/S5.')
    parser.add_argument('--input', default='data/test_datasets/totalcapture_officalib.pt')
    parser.add_argument('--output-dir', default='data/dataset_work/TotalCapture_globalpose_official')
    args = parser.parse_args()

    data = torch.load(args.input, map_location='cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'source_input': args.input,
        'split_rule': {
            'train': 'subject prefix s1_, s2_, s3_',
            'val': 'subject prefix s4_',
            'test': 'subject prefix s5_',
        },
        'splits': {},
    }
    for split, prefixes in SPLIT_PREFIXES.items():
        indices = [idx for idx, name in enumerate(data['name']) if str(name).startswith(prefixes)]
        split_data = subset(data, indices)
        out_path = output_dir / f'{split}.pt'
        torch.save(split_data, out_path)
        manifest['splits'][split] = {
            'path': str(out_path),
            'subjects': list(prefixes),
            **summarize(split_data),
        }
    manifest_path = output_dir / 'split_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'manifest': str(manifest_path), 'splits': manifest['splits']}, indent=2))


if __name__ == '__main__':
    main()
