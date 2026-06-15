import argparse
import json
from pathlib import Path

import torch


def make_T(R, r):
    eye = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(R.shape[0], 1, 1)
    eye[:, :3, :3] = R.float()
    eye[:, :3, 3] = r.float()
    return eye


def load_offset_map(path):
    cache = torch.load(path, map_location='cpu')
    if 'name' not in cache or 'offset' not in cache:
        raise KeyError(f'{path} must contain name and offset fields.')
    names = [str(name) for name in cache['name']]
    offsets = cache['offset'].float()
    if offsets.shape != (len(names), 6, 3):
        raise ValueError(f'Expected offset shape [N,6,3], got {tuple(offsets.shape)}.')
    # This cache stores only r_JS. For orientation correction we use the same
    # official sensor-to-body proxy already present in TotalCapture as RSB^T,
    # matching the historical append script fallback when no separate R cache
    # exists for a split.
    return {name: offsets[idx] for idx, name in enumerate(names)}


def append_offsets(input_path, offset_cache, output_path):
    data = torch.load(input_path, map_location='cpu')
    offset_map = load_offset_map(offset_cache)
    missing = [str(name) for name in data['name'] if str(name) not in offset_map]
    if missing:
        raise RuntimeError(f'Missing S5 offsets for sequences: {missing}')
    out = dict(data)
    r_list = []
    R_list = []
    T_list = []
    for idx, name in enumerate(data['name']):
        r = offset_map[str(name)].float()
        if 'RSB' not in data:
            raise KeyError('Need RSB to derive R_JS fallback for S5 orientation offset.')
        RSB = data['RSB'][idx].float()
        if RSB.shape == (6, 3, 3):
            R = RSB.transpose(-1, -2).contiguous()
        elif RSB.ndim == 4 and RSB.shape[1:] == (6, 3, 3):
            R = RSB[0].transpose(-1, -2).contiguous()
        else:
            raise ValueError(f'Expected RSB shape [6,3,3] or [T,6,3,3], got {tuple(RSB.shape)} for {name}.')
        r_list.append(r)
        R_list.append(R)
        T_list.append(make_T(R, r))
    out['imu_offset_r'] = r_list
    out['r_JS'] = r_list
    out['imu_offset_R'] = R_list
    out['R_JS'] = R_list
    out['imu_offset_T'] = T_list
    out['T_JS'] = T_list
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    norms = torch.stack(r_list).norm(dim=-1)
    return {
        'source_test': str(input_path),
        'offset_cache': str(offset_cache),
        'output': str(output_path),
        'num_sequences': len(out['name']),
        'names': [str(name) for name in out['name']],
        'fields_added': ['imu_offset_r', 'r_JS', 'imu_offset_R', 'R_JS', 'imu_offset_T', 'T_JS'],
        'R_JS_source': 'first-frame RSB^T from TotalCapture official calibration because S5 offset estimator output contains r_JS only',
        'offset_norm_min': float(norms.min()),
        'offset_norm_median': float(norms.median()),
        'offset_norm_max': float(norms.max()),
    }


def main():
    parser = argparse.ArgumentParser(description='Create TotalCapture S5 test split with sequence-level offset fields.')
    parser.add_argument('--input-test', type=Path, default=Path('data/dataset_work/TotalCapture_globalpose_official/test.pt'))
    parser.add_argument('--offset-cache', type=Path, required=True)
    parser.add_argument('--output-test', type=Path, required=True)
    parser.add_argument('--summary-json', type=Path, required=True)
    args = parser.parse_args()
    summary = append_offsets(args.input_test, args.offset_cache, args.output_test)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
