import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RBDL_PYTHON = Path('/home/lingfeng/rbdl/build/python')
OUTPUT = ROOT / 'data/experiments/pip_physics_backend_v1/smoke_pip_physics_v1_8f.json'
VAL_CACHE = ROOT / 'data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json'


def main():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(RBDL_PYTHON) + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [
        sys.executable,
        'pip_physics_eval.py',
        '--val-cache', str(VAL_CACHE),
        '--physics-backend', 'pip_physics_v1',
        '--smoke-sequence', 's4_acting3',
        '--max-smoke-frames', '8',
        '--output-json', str(OUTPUT),
    ]
    print(' '.join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
