import argparse
import json
import traceback
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, aggregate_eval, metric_to_dict, score_for_checkpoint
from net import GPNet
from newpose_ctrl import (
    DEFAULT_FK_VERTEX_MASK,
    NewPoseControlModule,
    decode_pose_state,
    direction_cosine_loss,
    finite_diff,
    normalize_pose_state,
    root_relative_fk_targets,
    rrj_geodesic_deg,
)
from newpose_ctrl_train import load_newpose_records
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


def load_eval_records(cache_path, max_sequences=0):
    records, manifest = load_newpose_records(cache_path, max_sequences=max_sequences)
    files = [Path(item['path']) for item in manifest['cache_files']]
    full_records = []
    seen = 0
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            record = records[seen]
            for key in ('pose_gt', 'tran_gt', 'aM', 'wM', 'RMB', 'gR0'):
                record[key] = data[key][seq_idx].float()
            full_records.append(record)
            seen += 1
            if max_sequences and seen >= max_sequences:
                return full_records, manifest
    return full_records, manifest


def load_newpose(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = checkpoint.get('config', {})
    model = NewPoseControlModule(
        input_size=int(cfg.get('input_size', cfg.get('input_dim', 174))) if 'input_size' in cfg or 'input_dim' in cfg else 174,
        hidden_size=int(cfg.get('hidden_size', 512)),
        tail_update=int(cfg.get('tail_length', 4)),
        residual_scale=float(cfg.get('residual_scale', 0.1)),
        dropout=float(cfg.get('dropout', 0.2)),
        offset_init_scale=float(cfg.get('offset_init_scale', 0.1)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def body_joints_from_pose(body_model, pose):
    pose_body = pose.clone()
    pose_body[:, 0] = torch.eye(3)
    return body_model.forward_kinematics(pose_body)[1][:, 1:]


def fk_leaf_pose_metrics(body_model, decoded_pose, gt_pose):
    pred_leaf, _pred_joint = root_relative_fk_targets(decoded_pose, body_model)
    target_leaf, _target_joint = root_relative_fk_targets(gt_pose, body_model)
    err = pred_leaf - target_leaf
    l2 = err.norm(dim=-1) * 100.0
    metrics = {
        'FK_leaf_L1_cm': metric_stats(err.abs().mean(dim=(-1, -2)) * 100.0),
        'FK_leaf_L2_cm': metric_stats(l2.mean(dim=-1)),
        'FK_leaf_per_leaf_L2_cm': [float(v) for v in l2.mean(dim=0)],
        'FK_leaf_smooth_jitter_cm_frame2': float(finite_diff(pred_leaf, 2).norm(dim=-1).mean() * 100.0) if pred_leaf.shape[0] > 2 else 0.0,
    }
    if pred_leaf.shape[0] > 1:
        vel_err = (finite_diff(pred_leaf, 1) - finite_diff(target_leaf, 1)).norm(dim=-1) * 100.0
        metrics['FK_leaf_vel_L2_cm_per_frame'] = metric_stats(vel_err.mean(dim=-1))
    else:
        metrics['FK_leaf_vel_L2_cm_per_frame'] = metric_stats(torch.empty(0))
    if pred_leaf.shape[0] > 2:
        acc_err = (finite_diff(pred_leaf, 2) - finite_diff(target_leaf, 2)).norm(dim=-1) * 100.0
        metrics['FK_leaf_acc_L2_cm_per_frame2'] = metric_stats(acc_err.mean(dim=-1))
    else:
        metrics['FK_leaf_acc_L2_cm_per_frame2'] = metric_stats(torch.empty(0))
    return metrics


def module_metrics(output, record, decoded_pose, body_model):
    pred = normalize_pose_state(output['state'].detach().cpu())
    target = normalize_pose_state(record['newpose_target'])
    pred_control = normalize_pose_state(output['new_control'].detach().cpu())
    target_control = normalize_pose_state(record['newpose_target_control_tail'][:, -1])
    pred_tail = normalize_pose_state(output['control_tail'].detach().cpu())
    target_tail = normalize_pose_state(record['newpose_target_control_tail'])
    gt_pose = record['pose_gt']
    pred_j = body_joints_from_pose(body_model, decoded_pose).reshape(decoded_pose.shape[0], 23, 3)
    target_j = body_joints_from_pose(body_model, gt_pose).reshape(gt_pose.shape[0], 23, 3)
    joint_err = (pred_j - target_j).norm(dim=-1) * 100.0
    metrics = {
        'state_RRJ_geodesic_deg': metric_stats(rrj_geodesic_deg(pred, target)),
        'control_RRJ_geodesic_deg': metric_stats(rrj_geodesic_deg(pred_control, target_control)),
        'control_tail_RRJ_l1': metric_stats((pred_tail[..., :90] - target_tail[..., :90]).abs().mean(dim=-1)),
        'control_gR_pose_loss': float(direction_cosine_loss(pred_control[..., 90:], target_control[..., 90:])),
        'state_gR_pose_loss': float(direction_cosine_loss(pred[..., 90:], target[..., 90:])),
        'FK_joint_L2_cm': metric_stats(joint_err.mean(dim=-1)),
        'FK_joint_per_joint_L2_cm': [float(v) for v in joint_err.mean(dim=0)],
        'state_RRJ_smooth_jitter': float((pred[2:, :90] - 2.0 * pred[1:-1, :90] + pred[:-2, :90]).norm(dim=-1).mean()) if pred.shape[0] > 2 else 0.0,
    }
    metrics.update(fk_leaf_pose_metrics(body_model, decoded_pose, gt_pose))
    return metrics


@torch.no_grad()
def run_newpose_sequence(model, record, body_model, max_frames=0):
    source = record
    if max_frames and record['newpose_input'].shape[0] > max_frames:
        source = {}
        seq_len = record['newpose_input'].shape[0]
        for key, value in record.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == seq_len:
                source[key] = value[:max_frames]
            else:
                source[key] = value
    offset_r = source.get('offset_r')
    output = model.forward_sequence(source['newpose_input'].to(DEVICE), offset_r=offset_r.to(DEVICE) if torch.is_tensor(offset_r) else None)
    decoded_pose = decode_pose_state(output['state'].detach().cpu(), source['RMB'][:, 5], source['gR0'], body_model)
    return output, decoded_pose, source


@torch.no_grad()
def run_full_pipeline_from_pose(net, decoded_pose, pred_state, record):
    offset_r = record.get('offset_r')
    net.rnn_initialize(record['pose_gt'][0], offset_r=offset_r)
    pose_out = torch.zeros_like(record['pose_gt'])
    tran_out = torch.zeros_like(record['tran_gt'])
    for idx in range(decoded_pose.shape[0]):
        g_pose = pred_state[idx, 90:].detach().cpu()
        pose, tran, _debug = net.forward_frame_from_curve_pose(
            record['aM'][idx].to(DEVICE),
            record['wM'][idx].to(DEVICE),
            record['RMB'][idx].to(DEVICE),
            decoded_pose[idx],
            g_pose,
        )
        pose_out[idx] = pose.cpu()
        tran_out[idx] = tran.cpu()
    return pose_out, tran_out


def aggregate_module(rows):
    out = {}
    if not rows:
        return out
    keys = rows[0]['module_metrics'].keys()
    for key in keys:
        vals = []
        if isinstance(rows[0]['module_metrics'][key], list):
            count = len(rows[0]['module_metrics'][key])
            denom = max(1, sum(row['num_frames'] for row in rows))
            out[key] = [
                sum(float(row['module_metrics'][key][idx]) * row['num_frames'] for row in rows) / denom
                for idx in range(count)
            ]
            continue
        for row in rows:
            item = row['module_metrics'][key]
            if isinstance(item, dict) and item.get('mean') is not None:
                vals.extend([float(item['mean'])] * int(row.get('num_frames', 1)))
            elif isinstance(item, (int, float)):
                vals.append(float(item))
        out[key] = metric_stats(torch.tensor(vals)) if vals else {'mean': None, 'count': 0}
    return out


def main():
    parser = argparse.ArgumentParser(description='Evaluate newpose_ctrl_v1 module and optional full pipeline.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--newpose-cache', type=Path, required=True)
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
        'newpose_cache': str(args.newpose_cache),
        'split_label': args.split_label,
        'version_name': args.version_name,
        'module_only': args.module_only,
        'metric_contract': 'newpose_ctrl_v1 control/decoded-pose metrics plus optional GPNet VR/physics full-pipeline MotionEvaluator score.',
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        records, manifest = load_eval_records(args.newpose_cache, max_sequences=args.max_eval_sequences)
        model, checkpoint = load_newpose(args.checkpoint)
        evaluator = MotionEvaluator()
        full_net = GPNet().eval().to(DEVICE) if not args.module_only else None
        body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=DEFAULT_FK_VERTEX_MASK)
        rows = []
        for record in records:
            output, decoded_pose, source = run_newpose_sequence(model, record, body_model, max_frames=args.max_smoke_frames)
            metrics = module_metrics(output, source, decoded_pose, body_model)
            row = {
                'name': source['name'],
                'num_frames': int(source['newpose_input'].shape[0]),
                'module_metrics': metrics,
                'finite': bool(torch.isfinite(decoded_pose).all()),
            }
            if not args.module_only:
                pose_out, tran_out = run_full_pipeline_from_pose(full_net, decoded_pose, output['state'].detach().cpu(), source)
                model_metric = evaluator(
                    pose_out.to(DEVICE),
                    source['pose_gt'].to(DEVICE),
                    tran_out.to(DEVICE),
                    source['tran_gt'].to(DEVICE),
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
                    'root_step_norm_max': float((tran_out[1:] - tran_out[:-1]).norm(dim=-1).max()) if tran_out.shape[0] > 1 else 0.0,
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
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ('status', 'version_name', 'split_label', 'score', 'all_finite', 'error_type', 'error')}, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
