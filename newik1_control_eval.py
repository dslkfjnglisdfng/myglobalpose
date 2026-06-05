import argparse
import json
import traceback
from pathlib import Path

import torch

from l4_train_diverse_short import DEVICE, aggregate_eval, load_records, metric_to_dict, score_for_checkpoint
from net import GPNet
from newik1_control_point import NewIK1ControlPointModule
from pl_curve_eval import build_pl_curve, selected_imu_fields
from test import MotionEvaluator


def build_newik1_control(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if checkpoint.get('model_type') != 'newik1_control_point_v1':
        raise RuntimeError(f"Unsupported NewIK1 checkpoint model_type={checkpoint.get('model_type')}")
    config = checkpoint.get('config', {})
    model = NewIK1ControlPointModule(
        hidden_size=int(config.get('hidden_size', 512)),
        tail_update=int(config.get('tail_length', 4)),
        residual_scale=float(config.get('residual_scale', 0.005)),
        dropout=float(config.get('dropout', 0.4)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


@torch.no_grad()
def run_sequence(record, pl_curve, ik1_curve=None, imu_input_mode='processed'):
    net = GPNet(
        pl_backend='curve_v1',
        pl_curve_module=pl_curve,
        ik1_backend='control_point_v1' if ik1_curve is not None else 'original',
        ik1_curve_module=ik1_curve,
    ).eval().to(DEVICE)
    net.rnn_initialize(record['pose_gt'][0])
    pose = torch.zeros_like(record['pose_gt'])
    tran = torch.zeros_like(record['tran_gt'])
    a_seq, w_seq, R_seq = selected_imu_fields(record, imu_input_mode)
    ik1_residual_norms = []
    tail_update_norms = []
    for i in range(record['pose_gt'].shape[0]):
        pose[i], tran[i] = net.forward_frame(
            a_seq[i].to(DEVICE),
            w_seq[i].to(DEVICE),
            R_seq[i].to(DEVICE),
        )
        debug = getattr(net, 'last_ik1_curve_debug', {})
        ik1 = debug.get('ik1_t')
        base = debug.get('base_t')
        if torch.is_tensor(ik1) and torch.is_tensor(base):
            ik1_residual_norms.append(float((ik1 - base).norm(dim=-1).mean()))
        tail_update_norms.append(float(debug.get('tail_delta_norm', 0.0)))
    return {
        'pose': pose.cpu(),
        'tran': tran.cpu(),
        'ik1_residual_norm_mean': sum(ik1_residual_norms) / max(1, len(ik1_residual_norms)),
        'ik1_residual_norm_max': max(ik1_residual_norms) if ik1_residual_norms else 0.0,
        'tail_update_norm_mean': sum(tail_update_norms) / max(1, len(tail_update_norms)),
        'tail_update_norm_max': max(tail_update_norms) if tail_update_norms else 0.0,
        'finite': bool(torch.isfinite(pose).all() and torch.isfinite(tran).all()),
        'root_step_norm_max': float((tran[1:] - tran[:-1]).norm(dim=-1).max()) if tran.shape[0] > 1 else 0.0,
    }


@torch.no_grad()
def evaluate(records, pl_curve, ik1_curve=None, max_eval_sequences=0, imu_input_mode='processed'):
    evaluator = MotionEvaluator()
    rows = []
    selected = records[:max_eval_sequences] if max_eval_sequences else records
    for record in selected:
        output = run_sequence(record, pl_curve=pl_curve, ik1_curve=ik1_curve, imu_input_mode=imu_input_mode)
        baseline = run_sequence(record, pl_curve=pl_curve, ik1_curve=None, imu_input_mode=imu_input_mode)
        baseline_metric = evaluator(
            baseline['pose'].to(DEVICE),
            record['pose_gt'].to(DEVICE),
            baseline['tran'].to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        model_metric = evaluator(
            output['pose'].to(DEVICE),
            record['pose_gt'].to(DEVICE),
            output['tran'].to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        rows.append({
            'name': record['name'],
            'baseline_metrics': metric_to_dict(baseline_metric),
            'model_metrics': metric_to_dict(model_metric),
            'delta_v_root_norm_mean': 0.0,
            'delta_v_root_norm_max': 0.0,
            'q_residual_norm_mean': 0.0,
            'q_residual_norm_max': 0.0,
            'ik1_residual_norm_mean': output['ik1_residual_norm_mean'],
            'ik1_residual_norm_max': output['ik1_residual_norm_max'],
            'tail_update_norm_mean': output['tail_update_norm_mean'],
            'tail_update_norm_max': output['tail_update_norm_max'],
            'finite': output['finite'],
            'root_step_norm_max': output['root_step_norm_max'],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Evaluate NewIK1_ControlPoint_v1 inside streaming GPNet.')
    parser.add_argument('--val-cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--pl-checkpoint', type=Path, required=True)
    parser.add_argument('--ik1-checkpoint', type=Path)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='processed')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--smoke-sequence', default='')
    parser.add_argument('--max-smoke-frames', type=int, default=0)
    args = parser.parse_args()
    result = {
        'pl_checkpoint': str(args.pl_checkpoint),
        'ik1_checkpoint': str(args.ik1_checkpoint) if args.ik1_checkpoint else None,
        'pl_backend': 'curve_v1',
        'ik1_backend': 'control_point_v1' if args.ik1_checkpoint else 'original',
        'imu_input_mode': args.imu_input_mode,
        'val_cache': str(args.val_cache),
        'status': 'started',
        'streaming_contract': 'GPNet.rnn_initialize per sequence, then GPNet.forward_frame frame by frame with PL curve and NewIK1 control-point backend.',
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        records, manifest = load_records(args.val_cache)
        if args.smoke_sequence:
            records = [record for record in records if record['name'] == args.smoke_sequence]
            if not records:
                raise KeyError(f'No sequence named {args.smoke_sequence!r}.')
            args.max_eval_sequences = 1
        if args.max_smoke_frames:
            for record in records:
                original_frames = record['pose_gt'].shape[0]
                for key, value in list(record.items()):
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == original_frames:
                        record[key] = value[:args.max_smoke_frames]
        pl_curve, pl_config = build_pl_curve(args.pl_checkpoint)
        ik1_curve, ik1_config = (None, None)
        if args.ik1_checkpoint:
            ik1_curve, ik1_config = build_newik1_control(args.ik1_checkpoint)
        rows = evaluate(
            records,
            pl_curve=pl_curve,
            ik1_curve=ik1_curve,
            max_eval_sequences=args.max_eval_sequences,
            imu_input_mode=args.imu_input_mode,
        )
        aggregate = aggregate_eval(rows)
        result.update({
            'status': 'ok',
            'pl_checkpoint_config': pl_config,
            'ik1_checkpoint_config': ik1_config,
            'val_manifest': manifest,
            'rows': rows,
            'aggregate': aggregate,
            'score': score_for_checkpoint(aggregate),
            'all_finite': all(row['finite'] for row in rows),
        })
    except Exception as exc:
        result.update({
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ('status', 'pl_backend', 'ik1_backend', 'score', 'all_finite', 'error_type', 'error')}, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
