import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


REQUIRED_CACHE_FIELDS = (
    'q75_prephysics',
    'v_root_vr',
    'stationary_prob',
    'q75_gt',
    'aM',
    'wM',
    'RMB',
)


def process_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for_pid(pid_file, poll_seconds):
    pid = int(Path(pid_file).read_text().strip())
    print(f'waiting for cache pid {pid}', flush=True)
    while process_exists(pid):
        time.sleep(poll_seconds)
    print(f'cache pid {pid} exited', flush=True)


def verify_cache(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f'Cache manifest not found: {path}')
    manifest = json.loads(path.read_text())
    missing = []
    for item in manifest.get('cache_files', []):
        cache_path = Path(item['path'])
        data = torch.load(cache_path, map_location='cpu')
        item_missing = [key for key in REQUIRED_CACHE_FIELDS if key not in data or not data[key]]
        if item_missing:
            missing.append({'path': str(cache_path), 'missing': item_missing})
    if missing:
        raise RuntimeError(f'Cache files missing required fields: {missing}')
    print(
        'cache verified: '
        f"{manifest.get('num_sequences')} sequences, "
        f"{manifest.get('num_frames')} frames, "
        f"{len(manifest.get('cache_files', []))} files",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description='Wait for L4 cache generation, verify it, then launch AMASS training.')
    parser.add_argument('--cache-pid-file', required=True)
    parser.add_argument('--cache-manifest', required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--train-log', required=True)
    parser.add_argument('train_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.train_args:
        raise ValueError('Training command arguments are required after --.')
    if args.train_args[0] == '--':
        args.train_args = args.train_args[1:]

    wait_for_pid(args.cache_pid_file, args.poll_seconds)
    verify_cache(args.cache_manifest)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '1')
    train_log = Path(args.train_log)
    train_log.parent.mkdir(parents=True, exist_ok=True)
    command = [args.python, '-u'] + args.train_args
    print('starting training command:', ' '.join(command), flush=True)
    with train_log.open('w') as log_file:
        return subprocess.call(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
