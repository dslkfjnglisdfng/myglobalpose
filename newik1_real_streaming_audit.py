import argparse
import json
import traceback
from pathlib import Path

import torch
import articulate as art

from l4_train_diverse_short import (
    DEVICE,
    aggregate_eval,
    load_records,
    metric_to_dict,
    score_for_checkpoint,
)
from net import GPNet
from newik1_control_eval import build_newik1_control
from newik1_official_input_eval import build_official_ik1
from official_processed_module_audit import build_targets, gravity_angle_deg, metric_stats
from pl_curve_eval import build_pl_curve, selected_imu_fields
from test import MotionEvaluator


def mean_l1_cm(pred, target):
    pred = pred.float().reshape(pred.shape[0], 23, 3)
    target = target.float().reshape(target.shape[0], 23, 3)
    return (pred - target).abs().mean(dim=-1).mean(dim=-1) * 100.0


def mean_l2_cm(pred, target):
    pred = pred.float().reshape(pred.shape[0], 23, 3)
    target = target.float().reshape(target.shape[0], 23, 3)
    return (pred - target).norm(dim=-1).mean(dim=-1) * 100.0


def diff_l2(pred, target, order, start, end):
    if pred.shape[0] <= order:
        return pred.new_empty(0)
    pred = pred.float()
    target = target.float()
    if order == 1:
        d_pred = pred[1:, start:end] - pred[:-1, start:end]
        d_target = target[1:, start:end] - target[:-1, start:end]
    elif order == 2:
        d_pred = pred[2:, start:end] - 2.0 * pred[1:-1, start:end] + pred[:-2, start:end]
        d_target = target[2:, start:end] - 2.0 * target[1:-1, start:end] + target[:-2, start:end]
    else:
        raise ValueError(order)
    return (d_pred - d_target).norm(dim=-1)


def ik1_metric_dict(ik1_out, targets):
    target = targets['ik1_target'].float()
    pred = ik1_out.float()
    pred_g = art.math.normalize_tensor(pred[:, 69:], avoid_nan=True)
    target_g = art.math.normalize_tensor(target[:, 69:], avoid_nan=True)
    metrics = {
        'pRJ_l1_cm': metric_stats(mean_l1_cm(pred[:, :69], target[:, :69])),
        'pRJ_l2_cm': metric_stats(mean_l2_cm(pred[:, :69], target[:, :69])),
        'gR2_angle_deg': metric_stats(gravity_angle_deg(pred_g, target_g)),
        'pRJ_dot_l2': metric_stats(diff_l2(pred, target, 1, 0, 69)),
        'pRJ_ddot_l2': metric_stats(diff_l2(pred, target, 2, 0, 69)),
        'gR2_dot_l2': metric_stats(diff_l2(torch.cat((pred[:, :69], pred_g), dim=-1), torch.cat((target[:, :69], target_g), dim=-1), 1, 69, 72)),
        'gR2_ddot_l2': metric_stats(diff_l2(torch.cat((pred[:, :69], pred_g), dim=-1), torch.cat((target[:, :69], target_g), dim=-1), 2, 69, 72)),
    }
    return metrics


def build_net(pl_curve, ik1_model, ik1_backend, ik1_config=None):
    ik1_config = ik1_config or {}
    kwargs = {
        'pl_backend': 'curve_v1' if pl_curve is not None else 'original',
        'pl_curve_module': pl_curve,
    }
    if ik1_backend == 'official_input_v1':
        kwargs.update({
            'ik1_backend': ik1_backend,
            'ik1_official_module': ik1_model,
            'ik1_official_output_mode': ik1_config.get('output_mode', 'full'),
            'ik1_official_residual_alpha': float(ik1_config.get('residual_alpha', 1.0)),
        })
    elif ik1_backend in ('control_point_v1', 'control_point_last_v1'):
        kwargs.update({'ik1_backend': ik1_backend, 'ik1_curve_module': ik1_model})
    elif ik1_backend != 'original':
        raise ValueError(f'Unsupported IK1 backend {ik1_backend}.')
    return GPNet(**kwargs).eval().to(DEVICE)


@torch.no_grad()
def run_sequence(record, pl_curve=None, ik1_model=None, ik1_backend='original', ik1_config=None, imu_input_mode='processed'):
    if pl_curve is not None and getattr(pl_curve, 'init_size', 18) == 36:
        if 'offset_r' not in record:
            raise KeyError(f'Record {record["name"]} lacks offset_r required by PL init36.')
        offset_r = record['offset_r'].float()
    else:
        offset_r = record.get('offset_r')
    a_seq, w_seq, R_seq = selected_imu_fields(record, imu_input_mode)
    pose = torch.zeros_like(record['pose_gt'])
    tran = torch.zeros_like(record['tran_gt'])

    # Pass 1: real streaming PL/IK1 state capture. This intentionally stops at
    # IK1 so the metric is the raw IK1 contract output, not IK2/FK output.
    net = build_net(pl_curve, ik1_model, ik1_backend, ik1_config=ik1_config)
    net.rnn_initialize(record['pose_gt'][0], offset_r=offset_r)
    ik1_outputs = []
    for idx in range(record['pose_gt'].shape[0]):
        ik1_debug = net.forward_until_ik1(
            a_seq[idx].to(DEVICE),
            w_seq[idx].to(DEVICE),
            R_seq[idx].to(DEVICE),
        )
        ik1_outputs.append(torch.cat((ik1_debug['pRJ_ik1'].reshape(-1), ik1_debug['gR2'].reshape(-1))))

    # Pass 2: canonical full-pipeline pose/translation. This is separate
    # because forward_until_ik1 consumed PL/IK1 recurrent state in pass 1.
    net = build_net(pl_curve, ik1_model, ik1_backend, ik1_config=ik1_config)
    net.rnn_initialize(record['pose_gt'][0], offset_r=offset_r)
    for idx in range(record['pose_gt'].shape[0]):
        pose[idx], tran[idx] = net.forward_frame(
            a_seq[idx].to(DEVICE),
            w_seq[idx].to(DEVICE),
            R_seq[idx].to(DEVICE),
        )
    return {
        'pose': pose.cpu(),
        'tran': tran.cpu(),
        'ik1_output': torch.stack(ik1_outputs).cpu(),
        'finite': bool(torch.isfinite(pose).all() and torch.isfinite(tran).all()),
        'root_step_norm_max': float((tran[1:] - tran[:-1]).norm(dim=-1).max()) if tran.shape[0] > 1 else 0.0,
    }


@torch.no_grad()
def evaluate(records, pl_curve=None, ik1_model=None, ik1_backend='original', ik1_config=None, imu_input_mode='processed', max_eval_sequences=0):
    evaluator = MotionEvaluator()
    rows = []
    selected = records[:max_eval_sequences] if max_eval_sequences else records
    for record in selected:
        output = run_sequence(record, pl_curve=pl_curve, ik1_model=ik1_model, ik1_backend=ik1_backend, ik1_config=ik1_config, imu_input_mode=imu_input_mode)
        targets = build_targets(record, GPNet())
        model_metric = evaluator(
            output['pose'].to(DEVICE),
            record['pose_gt'].to(DEVICE),
            output['tran'].to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        ik1_metrics = ik1_metric_dict(output['ik1_output'], targets)
        rows.append({
            'name': record['name'],
            'model_metrics': metric_to_dict(model_metric),
            'baseline_metrics': metric_to_dict(model_metric),
            'ik1_module_metrics': ik1_metrics,
            'delta_v_root_norm_mean': 0.0,
            'delta_v_root_norm_max': 0.0,
            'q_residual_norm_mean': 0.0,
            'q_residual_norm_max': 0.0,
            'tail_update_norm_mean': 0.0,
            'tail_update_norm_max': 0.0,
            'finite': output['finite'],
            'root_step_norm_max': output['root_step_norm_max'],
            'num_frames': int(record['pose_gt'].shape[0]),
        })
    aggregate = aggregate_eval(rows)
    return rows, aggregate


def aggregate_module(rows):
    out = {}
    keys = rows[0]['ik1_module_metrics'].keys() if rows else []
    for key in keys:
        vals = []
        for row in rows:
            mean = row['ik1_module_metrics'][key]['mean']
            if mean is not None:
                vals.extend([float(mean)] * int(row.get('num_frames', 1)))
        out[key] = metric_stats(torch.tensor(vals)) if vals else metric_stats(torch.empty(0))
    return out


def load_ik1(args):
    if args.ik1_backend == 'original':
        return None, None
    if not args.ik1_checkpoint:
        raise ValueError(f'--ik1-checkpoint is required for {args.ik1_backend}.')
    if args.ik1_backend == 'official_input_v1':
        return build_official_ik1(args.ik1_checkpoint)
    if args.ik1_backend in ('control_point_v1', 'control_point_last_v1', 'auto_control_point'):
        model, config = build_newik1_control(args.ik1_checkpoint)
        backend = config.get('ik1_backend', 'control_point_last_v1' if int(config.get('input_size', 120)) == 63 else 'control_point_v1')
        if args.ik1_backend != 'auto_control_point' and backend != args.ik1_backend:
            raise ValueError(f'Checkpoint backend {backend} does not match requested {args.ik1_backend}.')
        args.ik1_backend = backend
        return model, config
    raise ValueError(args.ik1_backend)


def main():
    parser = argparse.ArgumentParser(description='Real S4/S5 streaming full-pipeline IK1 output vs GT audit.')
    parser.add_argument('--val-cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--split-label', required=True)
    parser.add_argument('--version-name', required=True)
    parser.add_argument('--pl-checkpoint', type=Path)
    parser.add_argument('--ik1-checkpoint', type=Path)
    parser.add_argument('--ik1-backend', choices=('original', 'official_input_v1', 'control_point_v1', 'control_point_last_v1', 'auto_control_point'), default='original')
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='processed')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--smoke-sequence', default='')
    parser.add_argument('--max-smoke-frames', type=int, default=0)
    args = parser.parse_args()
    result = {
        'status': 'started',
        'version_name': args.version_name,
        'split_label': args.split_label,
        'val_cache': str(args.val_cache),
        'pl_checkpoint': str(args.pl_checkpoint) if args.pl_checkpoint else None,
        'ik1_checkpoint': str(args.ik1_checkpoint) if args.ik1_checkpoint else None,
        'ik1_backend': args.ik1_backend,
        'imu_input_mode': args.imu_input_mode,
        'metric_contract': 'Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence.',
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
                frames = record['pose_gt'].shape[0]
                for key, value in list(record.items()):
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == frames:
                        record[key] = value[:args.max_smoke_frames]
        pl_curve, pl_config = (None, None)
        if args.pl_checkpoint:
            pl_curve, pl_config = build_pl_curve(args.pl_checkpoint)
        ik1_model, ik1_config = load_ik1(args)
        rows, aggregate = evaluate(
            records,
            pl_curve=pl_curve,
            ik1_model=ik1_model,
            ik1_backend=args.ik1_backend,
            ik1_config=ik1_config,
            imu_input_mode=args.imu_input_mode,
            max_eval_sequences=args.max_eval_sequences,
        )
        result.update({
            'status': 'ok',
            'split_manifest': manifest,
            'sequence_names': [row['name'] for row in rows],
            'pl_checkpoint_config': pl_config,
            'ik1_checkpoint_config': ik1_config,
            'ik1_backend': args.ik1_backend,
            'rows': rows,
            'aggregate': aggregate,
            'ik1_module_aggregate': aggregate_module(rows),
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
    print(json.dumps({k: result.get(k) for k in ('status', 'version_name', 'split_label', 'score', 'all_finite', 'error_type', 'error')}, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
