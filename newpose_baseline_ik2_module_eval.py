import argparse
import json
import traceback
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_records
from net import GPNet
from newpose_ctrl import (
    DEFAULT_FK_VERTEX_MASK,
    decode_pose_state,
    direction_cosine_loss,
    normalize_pose_state,
    pose_state_target_from_pose,
    rrj_geodesic_deg,
)
from newpose_ctrl_eval import body_joints_from_pose, fk_leaf_pose_metrics, metric_stats
from pl_curve import build_pl_curve_model


def selected_imu_fields(record, mode):
    if mode == 'official':
        return record['aM'], record['wM'], record['RMB']
    has_l4 = all(key in record for key in ('l4_aM', 'l4_wM', 'l4_RMB'))
    if mode == 'processed':
        if not has_l4:
            raise KeyError(f'processed mode requires l4_aM/l4_wM/l4_RMB in record {record.get("name")}.')
        return record['l4_aM'], record['l4_wM'], record['l4_RMB']
    if mode == 'auto' and has_l4:
        return record['l4_aM'], record['l4_wM'], record['l4_RMB']
    return record['aM'], record['wM'], record['RMB']


def load_pl_curve(checkpoint_path):
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get('config', {})
    if 'model_variant' not in config and checkpoint.get('model_variant'):
        config = dict(config)
        config['model_variant'] = checkpoint.get('model_variant')
    model = build_pl_curve_model(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def run_ik2_state_sequence(net, record, pl_curve=None, imu_input_mode='official'):
    net.rnn_initialize(record['pose_gt'][0], offset_r=record.get('offset_r'))
    a_seq, w_seq, r_seq = selected_imu_fields(record, imu_input_mode)
    states = []
    gR0_seq = []
    for idx in range(record['pose_gt'].shape[0]):
        a = a_seq[idx].to(DEVICE)
        w = w_seq[idx].to(DEVICE)
        r = r_seq[idx].to(DEVICE)
        aRB = a.mm(r[5])
        wRB = w.mm(r[5])
        RRB = r[5].t().matmul(r[:5])
        gR0 = -r[5, 1]
        x_pl = torch.cat((aRB.ravel(), wRB.ravel(), RRB.ravel(), gR0))
        x_curve = net._pl_curve_frame_feature(a, w, r, x_pl)
        pl_out, gR1 = net._run_pl_stage(x_pl, x_curve)
        RRB_after_pl = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)
        ik1_out, gR2 = net._run_ik1_stage(RRB_after_pl, gR1, pl_out[:15])
        RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB_after_pl)
        ik2_input = torch.cat((RRB_after_ik1.ravel(), gR2, ik1_out[:69]))
        x_ik2, net.ik2hc = net.iknet.net2.rnn(ik2_input.view(1, 1, -1), net.ik2hc)
        rrj = net.iknet.net2.linear2(x_ik2.squeeze())
        states.append(normalize_pose_state(torch.cat((rrj.detach().cpu(), gR2.detach().cpu()), dim=-1)))
        gR0_seq.append(gR0.detach().cpu())
    return torch.stack(states), torch.stack(gR0_seq), r_seq.float()


def module_metrics(pred_state, target_state, decoded_pose, gt_pose, body_model):
    pred_state = normalize_pose_state(pred_state)
    target_state = normalize_pose_state(target_state)
    pred_j = body_joints_from_pose(body_model, decoded_pose).reshape(decoded_pose.shape[0], 23, 3)
    target_j = body_joints_from_pose(body_model, gt_pose).reshape(gt_pose.shape[0], 23, 3)
    joint_err = (pred_j - target_j).norm(dim=-1) * 100.0
    metrics = {
        'state_RRJ_geodesic_deg': metric_stats(rrj_geodesic_deg(pred_state, target_state)),
        'state_gR_pose_loss': float(direction_cosine_loss(pred_state[..., 90:], target_state[..., 90:])),
        'FK_joint_L2_cm': metric_stats(joint_err.mean(dim=-1)),
        'FK_joint_per_joint_L2_cm': [float(v) for v in joint_err.mean(dim=0)],
        'state_RRJ_smooth_jitter': float((pred_state[2:, :90] - 2.0 * pred_state[1:-1, :90] + pred_state[:-2, :90]).norm(dim=-1).mean()) if pred_state.shape[0] > 2 else 0.0,
    }
    metrics.update(fk_leaf_pose_metrics(body_model, decoded_pose, gt_pose))
    return metrics


def aggregate_module(rows):
    out = {}
    if not rows:
        return out
    for key in rows[0]['module_metrics']:
        if isinstance(rows[0]['module_metrics'][key], list):
            count = len(rows[0]['module_metrics'][key])
            out[key] = [
                sum(float(row['module_metrics'][key][idx]) * row['num_frames'] for row in rows) / max(1, sum(row['num_frames'] for row in rows))
                for idx in range(count)
            ]
            continue
        vals = []
        for row in rows:
            item = row['module_metrics'][key]
            if isinstance(item, dict) and item.get('mean') is not None:
                vals.extend([float(item['mean'])] * int(row['num_frames']))
            elif isinstance(item, (int, float)):
                vals.append(float(item))
        out[key] = metric_stats(torch.tensor(vals)) if vals else {'mean': None, 'count': 0}
    return out


def main():
    parser = argparse.ArgumentParser(description='Module-only IK2-slot metrics for official GPNet/NewPL+official IK2 baselines.')
    parser.add_argument('--val-cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--version-name', required=True)
    parser.add_argument('--split-label', required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='official')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    args = parser.parse_args()

    result = {
        'status': 'started',
        'version_name': args.version_name,
        'split_label': args.split_label,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'imu_input_mode': args.imu_input_mode,
        'metric_contract': 'IK2-slot module-only metrics: RRJ[90]+gR2[3] decoded pose/FK versus GT pose-state. No VR, translation fusion, or carticulate physics.',
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        records, manifest = load_records(args.val_cache)
        selected = records[:args.max_eval_sequences] if args.max_eval_sequences else records
        pl_curve = load_pl_curve(args.checkpoint)
        net = GPNet(
            pl_backend='curve_v1' if pl_curve is not None else 'original',
            pl_curve_module=pl_curve,
        ).eval().to(DEVICE)
        body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=DEFAULT_FK_VERTEX_MASK)
        rows = []
        for record in selected:
            pred_state, gR0, r_seq = run_ik2_state_sequence(net, record, pl_curve=pl_curve, imu_input_mode=args.imu_input_mode)
            target_state = pose_state_target_from_pose(
                record['pose_gt'].float(),
                r_seq.float(),
                gR0.float(),
                body_model,
            ).detach().cpu()
            decoded_pose = decode_pose_state(pred_state, r_seq[:, 5], gR0, body_model)
            metrics = module_metrics(pred_state, target_state, decoded_pose, record['pose_gt'], body_model)
            rows.append({
                'name': record['name'],
                'num_frames': int(record['pose_gt'].shape[0]),
                'module_metrics': metrics,
                'finite': bool(torch.isfinite(pred_state).all() and torch.isfinite(decoded_pose).all()),
            })
        result.update({
            'status': 'ok',
            'val_cache': str(args.val_cache),
            'val_manifest': manifest,
            'rows': rows,
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
    print(json.dumps({k: result.get(k) for k in ('status', 'version_name', 'split_label', 'all_finite', 'error_type', 'error')}, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
