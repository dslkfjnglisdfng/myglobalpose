import argparse
import json
import os
import pickle
from pathlib import Path

import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

import articulate as art


IMU_MASK = [7, 8, 11, 12, 0, 2]
GRAVITY = torch.tensor([0, -9.798, 0.0])
DEFAULT_SPLITS = {
    'train': [f's_{idx:02d}' for idx in range(1, 8)],
    'val': ['s_08'],
    'test': ['s_09', 's_10'],
}


def fill_nan_like_globalpose(acc, ori):
    for frame_idx in range(ori.shape[0]):
        for sensor_idx in range(6):
            if torch.isnan(ori[frame_idx, sensor_idx]).sum() > 0:
                k1, k2 = frame_idx - 1, frame_idx + 1
                while k1 >= 0 and torch.isnan(ori[k1, sensor_idx]).sum() > 0:
                    k1 -= 1
                while k2 < ori.shape[0] and torch.isnan(ori[k2, sensor_idx]).sum() > 0:
                    k2 += 1
                if k1 >= 0 and k2 < ori.shape[0]:
                    slerp = Slerp([k1, k2], Rotation.from_matrix(ori[[k1, k2], sensor_idx].numpy()))
                    ori[k1 + 1:k2, sensor_idx] = torch.from_numpy(
                        slerp(list(range(k1 + 1, k2))).as_matrix()
                    ).float()
                elif k1 < 0:
                    ori[:k2, sensor_idx] = ori[k2, sensor_idx]
                elif k2 >= ori.shape[0]:
                    ori[k1 + 1:, sensor_idx] = ori[k1, sensor_idx]
            if torch.isnan(acc[frame_idx, sensor_idx]).sum() > 0:
                k1, k2 = frame_idx - 1, frame_idx + 1
                while k1 >= 0 and torch.isnan(acc[k1, sensor_idx]).sum() > 0:
                    k1 -= 1
                while k2 < ori.shape[0] and torch.isnan(acc[k2, sensor_idx]).sum() > 0:
                    k2 += 1
                if k1 >= 0 and k2 < ori.shape[0]:
                    lerp = interp1d([k1, k2], acc[[k1, k2], sensor_idx].numpy(), axis=0)
                    acc[k1 + 1:k2, sensor_idx] = torch.from_numpy(
                        lerp(list(range(k1 + 1, k2)))
                    ).float()
                elif k1 < 0:
                    acc[:k2, sensor_idx] = acc[k2, sensor_idx]
                elif k2 >= ori.shape[0]:
                    acc[k1 + 1:, sensor_idx] = acc[k1, sensor_idx]
    return acc, ori


def process_subjects(raw_dir, subjects):
    data = {'name': [], 'RIM': [], 'RSB': [], 'RIS': [], 'aS': [], 'wS': [], 'mS': [], 'tran': [], 'pose': []}
    failures = []
    for subject_name in subjects:
        subject_dir = Path(raw_dir) / subject_name
        for motion_path in sorted(subject_dir.glob('*.pkl')):
            try:
                payload = pickle.load(open(motion_path, 'rb'), encoding='latin1')
                acc = torch.from_numpy(payload['imu_acc'][:, IMU_MASK]).float()
                ori = torch.from_numpy(payload['imu_ori'][:, IMU_MASK]).float()
                pose = torch.from_numpy(payload['gt']).float()
                acc, ori = fill_nan_like_globalpose(acc, ori)
                if torch.isnan(acc).sum() > 0 or torch.isnan(ori).sum() > 0 or torch.isnan(pose).sum() > 0:
                    failures.append({'path': str(motion_path), 'error': 'remaining NaN after interpolation'})
                    continue

                w = art.math.rotation_matrix_to_axis_angle(
                    ori[:-1].transpose(2, 3).matmul(ori[1:])
                ).view(-1, ori.shape[1], 3) * 60
                w = torch.cat((w, torch.zeros_like(w[:1])))
                m = ori.transpose(2, 3).matmul(torch.tensor([1, 0, 0.]).unsqueeze(-1)).squeeze(-1)
                a = ori.transpose(2, 3).matmul((acc - GRAVITY).unsqueeze(-1)).squeeze(-1)

                name = subject_name.replace('_', '') + '_' + motion_path.stem
                data['name'].append(name)
                data['RIM'].append(torch.eye(3).repeat(6, 1, 1))
                data['RSB'].append(torch.eye(3).repeat(6, 1, 1))
                data['RIS'].append(ori)
                data['aS'].append(a)
                data['wS'].append(w)
                data['mS'].append(m)
                data['tran'].append(torch.zeros(pose.shape[0], 3))
                data['pose'].append(pose)
                print(f'Finish Processing {name}: {pose.shape[0]} frames', flush=True)
            except Exception as exc:
                failures.append({'path': str(motion_path), 'error': f'{type(exc).__name__}: {exc}'})
    return data, failures


def summarize(data):
    return {
        'num_sequences': len(data['name']),
        'num_frames': int(sum(seq.shape[0] for seq in data['pose'])),
        'names': list(data['name']),
        'frames_per_sequence': {name: int(seq.shape[0]) for name, seq in zip(data['name'], data['pose'])},
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare GlobalPose-style DIP-IMU train/val/test splits from raw DIP .pkl files.')
    parser.add_argument('--raw-dir', default='/home/lingfeng/projects/data/data_raw/DIP_IMU')
    parser.add_argument('--output-dir', default='data/dataset_work/DIP_IMU_globalpose')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'raw_dir': args.raw_dir,
        'split_subjects': DEFAULT_SPLITS,
        'splits': {},
        'failures': {},
        'processing_contract': 'GlobalPose process.py DIP-IMU semantics: IMU mask [7,8,11,12,0,2], Slerp/linear NaN interpolation, RIM/RSB identity, RIS=ori, aS=ori^T(acc-g), wS=finite rotation delta*60, zero tran.',
    }
    for split, subjects in DEFAULT_SPLITS.items():
        data, failures = process_subjects(args.raw_dir, subjects)
        out_path = output_dir / f'{split}.pt'
        torch.save(data, out_path)
        manifest['splits'][split] = {'path': str(out_path), 'subjects': subjects, **summarize(data)}
        manifest['failures'][split] = failures
    manifest_path = output_dir / 'split_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'manifest': str(manifest_path), 'splits': manifest['splits']}, indent=2))


if __name__ == '__main__':
    main()
