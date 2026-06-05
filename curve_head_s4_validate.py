import argparse
import json
from pathlib import Path

import torch

from curve_control_pose_head import CurveControlPoseHead, build_curve_frame_features
from curve_state_decoder import CurveStateDecoder
from l4_train_diverse_short import aggregate_eval, load_records, metric_to_dict, score_for_checkpoint
from net import GPNet
from test import MotionEvaluator


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def extract_full_features(net, record, use_imu=True, use_feature_velocity=True):
    net.rnn_initialize(record['pose_gt'][0])
    features = []
    gR2_rows = []
    prev_pRJ = None
    for frame_idx in range(record['aM'].shape[0]):
        ik1 = net.forward_until_ik1(
            record['aM'][frame_idx].to(DEVICE),
            record['wM'][frame_idx].to(DEVICE),
            record['RMB'][frame_idx].to(DEVICE),
        )
        features.append(build_curve_frame_features(
            ik1['ik2_teacher_input'],
            ik1['pRJ_ik1'],
            record['aM'][frame_idx],
            record['wM'][frame_idx],
            record['RMB'][frame_idx],
            prev_pRJ_ik1=prev_pRJ,
            use_imu=use_imu,
            use_feature_velocity=use_feature_velocity,
        ))
        gR2_rows.append(ik1['gR2'])
        prev_pRJ = ik1['pRJ_ik1']
    if net.ik2hc is not None:
        raise RuntimeError('CurveHead validation feature extraction touched IK-s2 hidden state.')
    return torch.stack(features).to(DEVICE), torch.stack(gR2_rows)


@torch.no_grad()
def run_curve_sequence(head, decoder, record):
    feature_net = GPNet().to(DEVICE).eval()
    features, gR2_rows = extract_full_features(
        feature_net,
        record,
        use_imu=head.use_imu,
        use_feature_velocity=head.use_feature_velocity,
    )
    offset = record.get('offset_r')
    offset = None if offset is None else offset.view(1, 6, 3).to(DEVICE)
    head_out = head(features, offset_r=offset)
    decoded = decoder(head_out['control'], return_pose=True)

    down_net = GPNet().to(DEVICE).eval()
    down_net.rnn_initialize(record['pose_gt'][0])
    pose_model = torch.zeros_like(record['pose_gt'])
    tran_model = torch.zeros_like(record['tran_gt'])
    downstream_shapes = None
    for frame_idx in range(record['aM'].shape[0]):
        pose_t, tran_t, debug = down_net.forward_frame_from_curve_pose(
            record['aM'][frame_idx].to(DEVICE),
            record['wM'][frame_idx].to(DEVICE),
            record['RMB'][frame_idx].to(DEVICE),
            decoded['pose'][frame_idx].detach().cpu(),
            gR2_rows[frame_idx],
        )
        pose_model[frame_idx] = pose_t
        tran_model[frame_idx] = tran_t
        if downstream_shapes is None:
            downstream_shapes = {key: list(value.shape) for key, value in debug.items() if torch.is_tensor(value)}
    if down_net.ik2hc is not None:
        raise RuntimeError('CurveHead validation downstream path touched IK-s2 hidden state.')
    return pose_model, tran_model, {
        'control_shape': list(head_out['control'].shape),
        'q75_shape': list(decoded['q75'].shape),
        'qdot_shape': list(decoded['qdot'].shape),
        'qddot_shape': list(decoded['qddot'].shape),
        'pose_shape': list(decoded['pose'].shape),
        'downstream_shapes': downstream_shapes,
    }


@torch.no_grad()
def run_baseline(record):
    if 'pose_baseline' in record and 'tran_baseline' in record:
        return record['pose_baseline'], record['tran_baseline']
    net = GPNet().to(DEVICE).eval()
    net.rnn_initialize(record['pose_gt'][0])
    pose = torch.zeros_like(record['pose_gt'])
    tran = torch.zeros_like(record['tran_gt'])
    for frame_idx in range(record['aM'].shape[0]):
        pose[frame_idx], tran[frame_idx] = net.forward_frame(
            record['aM'][frame_idx].to(DEVICE),
            record['wM'][frame_idx].to(DEVICE),
            record['RMB'][frame_idx].to(DEVICE),
        )
    return pose, tran


def load_head(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = checkpoint['model_config']
    head = CurveControlPoseHead(
        input_dim=cfg['input_dim'],
        hidden_size=cfg['hidden_size'],
        state_dim=cfg['state_dim'],
        residual_scale=cfg['residual_scale'],
        use_imu=cfg['use_imu'],
        use_feature_velocity=cfg['use_feature_velocity'],
        rnn_init_mode=cfg['rnn_init_mode'],
        freeze_root_translation=cfg['freeze_root_translation'],
        predict_root_orientation=cfg['predict_root_orientation'],
        offset_init_scale=cfg['offset_init_scale'],
    ).to(DEVICE)
    head.load_state_dict(checkpoint['model_state_dict'])
    head.eval()
    return head, checkpoint


def main():
    parser = argparse.ArgumentParser(description='TotalCapture S4 validation for Curve-Control Pose Head.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--val-cache', default='data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    args = parser.parse_args()

    records, manifest = load_records(args.val_cache, max_sequences=args.max_eval_sequences)
    head, checkpoint = load_head(args.checkpoint)
    decoder = CurveStateDecoder().to(DEVICE).eval()
    evaluator = MotionEvaluator()
    rows = []
    for record in records:
        pose_model, tran_model, shape_info = run_curve_sequence(head, decoder, record)
        pose_baseline, tran_baseline = run_baseline(record)
        baseline_metric = evaluator(
            pose_baseline.to(DEVICE),
            record['pose_gt'].to(DEVICE),
            tran_baseline.to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        model_metric = evaluator(
            pose_model.to(DEVICE),
            record['pose_gt'].to(DEVICE),
            tran_model.to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        rows.append({
            'name': record['name'],
            'baseline_metrics': metric_to_dict(baseline_metric),
            'model_metrics': metric_to_dict(model_metric),
            'shape_info': shape_info,
        })
        print(json.dumps({'validated': record['name'], 'shape_info': shape_info}, indent=2), flush=True)
    aggregate = aggregate_eval([
        {
            **row,
            'delta_metrics': {},
            'delta_v_root_norm_mean': 0.0,
            'delta_v_root_norm_max': 0.0,
            'q_residual_norm_mean': 0.0,
            'q_residual_norm_max': 0.0,
            'tail_update_norm_mean': 0.0,
            'tail_update_norm_max': 0.0,
        }
        for row in rows
    ])
    score = score_for_checkpoint(aggregate)
    result = {
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_selection': checkpoint.get('selection'),
        'val_cache': args.val_cache,
        's5_used': False,
        'test_py_modified': False,
        'motion_evaluator_modified': False,
        'official_weights_modified': False,
        'inference_uses_ik2': False,
        'num_sequences': len(records),
        'score': score,
        'aggregate': aggregate,
        'rows': rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'output': str(output), 'score': score, 'num_sequences': len(records)}, indent=2))


if __name__ == '__main__':
    main()
