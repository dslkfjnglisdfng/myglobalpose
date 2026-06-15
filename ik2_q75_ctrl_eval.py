import argparse
import json
import traceback
from pathlib import Path

import torch

import articulate as art
from ik2_q75_ctrl import (
    DEFAULT_FK_VERTEX_MASK,
    IK2Q75ControlModule,
    fk_pva_from_pose,
    fk_pva_from_q,
    q75_to_pose_with_baseline_root,
    rotation_geodesic,
)
from ik2_q75_ctrl_train import load_records
from l4_train_diverse_short import DEVICE, aggregate_eval, metric_to_dict, score_for_checkpoint
from net import GPNet
from test import MotionEvaluator


def metric_stats(values):
    values = values.detach().cpu().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {'mean': None, 'median': None, 'std': None, 'min': None, 'max': None, 'count': 0}
    return {
        'mean': float(values.mean()),
        'median': float(values.median()),
        'std': float(values.std(unbiased=False)),
        'min': float(values.min()),
        'max': float(values.max()),
        'count': int(values.numel()),
    }


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = checkpoint.get('config', {})
    model = IK2Q75ControlModule(
        input_size=int(cfg.get('input_size', 153)),
        hidden_size=int(cfg.get('hidden_size', 512)),
        residual_scale=float(cfg.get('residual_scale', 0.05)),
        dropout=float(cfg.get('dropout', 0.2)),
        offset_init_scale=float(cfg.get('offset_init_scale', 0.1)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def run_sequence(model, record, body_model, max_frames=0):
    source = record
    if max_frames and record['ik2_q75_input'].shape[0] > max_frames:
        source = {}
        seq_len = record['ik2_q75_input'].shape[0]
        for key, value in record.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == seq_len:
                source[key] = value[:max_frames]
            else:
                source[key] = value
    device_record = {key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in source.items()}
    output = model.forward_sequence(device_record['ik2_q75_input'], offset_r=device_record.get('offset_r'))
    pred_fk = fk_pva_from_q(
        output['q75'],
        output['qdot'],
        output['qddot'],
        device_record['baseline_root_pose'],
        body_model,
        dt=model.dt,
        euler_seq=model.euler_seq,
    )
    gt_fk = fk_pva_from_pose(device_record['pose_gt'], body_model, dt=model.dt)
    teacher_fk = fk_pva_from_pose(device_record['teacher_pose'], body_model, dt=model.dt)
    pred_pose = q75_to_pose_with_baseline_root(output['q75'], device_record['baseline_root_pose'], euler_seq=model.euler_seq)
    return source, output, pred_pose.detach().cpu(), pred_fk, gt_fk, teacher_fk


def pva_metrics(pred_fk, gt_fk, teacher_fk, pred_pose, pose_gt):
    out = {}
    for prefix, ref in (('gt', gt_fk), ('baseline', teacher_fk)):
        for point in ('joint', 'leaf'):
            for quantity, suffix in (('pos', 'L2_cm'), ('vel', 'vel_L2_cm_s'), ('acc', 'acc_L2_cm_s2')):
                key = f'{point}_{quantity}'
                err = (pred_fk[key] - ref[key]).norm(dim=-1)
                out[f'FK_{point}_{suffix}_vs_{prefix}'] = metric_stats(err.mean(dim=-1) * 100.0)
    out['pose_body_geodesic_deg_vs_gt'] = metric_stats(
        torch.rad2deg(rotation_geodesic(pred_pose[..., 1:, :, :], pose_gt[..., 1:, :, :]))
    )
    return out


def aggregate_module(rows):
    out = {}
    if not rows:
        return out
    keys = rows[0]['module_metrics'].keys()
    for key in keys:
        vals = []
        for row in rows:
            item = row['module_metrics'][key]
            if isinstance(item, dict) and item.get('mean') is not None:
                vals.extend([float(item['mean'])] * int(row.get('num_frames', 1)))
            elif isinstance(item, (int, float)):
                vals.append(float(item))
        out[key] = metric_stats(torch.tensor(vals)) if vals else {'mean': None, 'count': 0}
    return out


@torch.no_grad()
def run_full_pipeline(record, pred_pose, max_frames=0):
    net = GPNet().eval().to(DEVICE)
    offset_r = record.get('offset_r')
    net.rnn_initialize(record['pose_gt'][0], offset_r=offset_r)
    n = pred_pose.shape[0] if not max_frames else min(pred_pose.shape[0], max_frames)
    pose_out = torch.zeros_like(record['pose_gt'][:n])
    tran_out = torch.zeros_like(record['tran_gt'][:n])
    for idx in range(n):
        pose, tran, _debug = net.forward_frame_from_curve_pose(
            record['aM'][idx].to(DEVICE),
            record['wM'][idx].to(DEVICE),
            record['RMB'][idx].to(DEVICE),
            pred_pose[idx],
            record['baseline_gR2'][idx],
        )
        pose_out[idx] = pose.cpu()
        tran_out[idx] = tran.cpu()
    return pose_out, tran_out


def main():
    parser = argparse.ArgumentParser(description='Evaluate IK2 q75 control module.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--split-label', required=True)
    parser.add_argument('--version-name', required=True)
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--max-smoke-frames', type=int, default=0)
    parser.add_argument('--module-only', action='store_true')
    args = parser.parse_args()
    result = {
        'status': 'started',
        'checkpoint': str(args.checkpoint),
        'cache': str(args.cache),
        'split_label': args.split_label,
        'version_name': args.version_name,
        'module_only': args.module_only,
        'metric_contract': 'IK2 q75 control module, baseline PL+IK1 input, root baseline-overwritten, FK p/v/a module metrics.',
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        records, manifest = load_records(args.cache, max_sequences=args.max_eval_sequences)
        model, checkpoint = load_model(args.checkpoint)
        body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=DEFAULT_FK_VERTEX_MASK, device=DEVICE)
        evaluator = MotionEvaluator()
        rows = []
        for record in records:
            source, output, pred_pose, pred_fk, gt_fk, teacher_fk = run_sequence(model, record, body_model, max_frames=args.max_smoke_frames)
            metrics = pva_metrics(
                pred_fk,
                gt_fk,
                teacher_fk,
                pred_pose.to(DEVICE),
                source['pose_gt'].to(DEVICE),
            )
            row = {
                'name': source['name'],
                'num_frames': int(source['ik2_q75_input'].shape[0]),
                'module_metrics': metrics,
                'control_shape': list(output['control'].shape),
                'q75_shape': list(output['q75'].shape),
                'finite': bool(torch.isfinite(output['q75']).all() and torch.isfinite(pred_pose).all()),
            }
            if not args.module_only:
                pose_out, tran_out = run_full_pipeline(source, pred_pose, max_frames=args.max_smoke_frames)
                model_metric = evaluator(
                    pose_out.to(DEVICE),
                    source['pose_gt'][:pose_out.shape[0]].to(DEVICE),
                    tran_out.to(DEVICE),
                    source['tran_gt'][:tran_out.shape[0]].to(DEVICE),
                ).cpu()
                metric_dict = metric_to_dict(model_metric)
                row.update({
                    'baseline_metrics': metric_dict,
                    'model_metrics': metric_dict,
                    'delta_v_root_norm_mean': 0.0,
                    'delta_v_root_norm_max': 0.0,
                    'q_residual_norm_mean': 0.0,
                    'q_residual_norm_max': 0.0,
                    'tail_update_norm_mean': 0.0,
                    'tail_update_norm_max': 0.0,
                    'finite': row['finite'] and bool(torch.isfinite(pose_out).all() and torch.isfinite(tran_out).all()),
                })
            rows.append(row)
        aggregate = aggregate_eval(rows) if not args.module_only else {}
        result.update({
            'status': 'ok',
            'checkpoint_config': checkpoint.get('config', {}),
            'cache_manifest': manifest,
            'rows': rows,
            'aggregate': aggregate,
            'score': score_for_checkpoint(aggregate) if aggregate else None,
            'module_aggregate': aggregate_module(rows),
            'all_finite': all(row['finite'] for row in rows),
        })
    except Exception as exc:
        result.update({
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'version_name', 'split_label', 'score', 'all_finite', 'error_type', 'error')}, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
