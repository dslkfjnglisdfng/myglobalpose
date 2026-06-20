#!/usr/bin/env python3
"""Train the joint-target NewPL control module for frozen_joint_acc_aug102 caches."""

import argparse
import os
import subprocess
import sys


LOSS_ARGS = [
    '--input-size', '102',
    '--init-size', '36',
    '--disable-ik-distill',
    '--selection-metric', 'pl_and_control_physical',
    '--pRB-weight', '1.0',
    '--gR1-weight', '0.3',
    '--baseline-pRB-weight', '0.0',
    '--baseline-gR1-weight', '0.0',
    '--gt-control-pRB-weight', '0.5',
    '--gt-control-gR1-weight', '0.1',
    '--pRB-dot-weight', '0.5',
    '--pRB-ddot-weight', '0.1',
    '--pRB-ddot-smooth-weight', '0.001',
    '--gR-smooth-weight', '0.001',
    '--control-point-prior-weight', '0.001',
    '--tail-update-prior-weight', '0.001',
    '--gR1-dot-weight', '0.0',
    '--gR1-ddot-weight', '0.0',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment-name', default='pl_joint_control_acc_aug102_v1')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--val-window-length', type=int, default=0)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--extra-arg', action='append', default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        'pl_curve_train.py',
        '--train-cache', args.train_cache,
        '--val-cache', args.val_cache,
        '--output-dir', args.output_dir,
        '--experiment-name', args.experiment_name,
        '--epochs', str(args.epochs),
        '--window', str(args.window),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--hidden-size', str(args.hidden_size),
        '--tail-length', str(args.tail_length),
        '--residual-scale', str(args.residual_scale),
        '--dropout', str(args.dropout),
        '--grad-clip', str(args.grad_clip),
        '--max-train-sequences', str(args.max_train_sequences),
        '--max-val-sequences', str(args.max_val_sequences),
        '--val-window-length', str(args.val_window_length),
    ]
    if args.init_checkpoint:
        cmd += ['--init-checkpoint', args.init_checkpoint]
    cmd += LOSS_ARGS
    for extra in args.extra_arg:
        cmd.append(extra)
    env = dict(os.environ)
    env.setdefault('PYTHONUNBUFFERED', '1')
    print(' '.join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == '__main__':
    main()
