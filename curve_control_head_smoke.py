import argparse
import json
from pathlib import Path

import torch

from curve_control_pose_head import CurveControlPoseHead, build_curve_frame_features
from curve_state_decoder import CurveStateDecoder
from l4_train_diverse_short import load_records
from net import GPNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_curve_features(net, record, frames, use_imu=True, use_feature_velocity=True):
    rows = []
    prev_pRJ = None
    for frame_idx in range(frames):
        a = record['aM'][frame_idx].to(DEVICE)
        w = record['wM'][frame_idx].to(DEVICE)
        R = record['RMB'][frame_idx].to(DEVICE)
        ik1 = net.forward_until_ik1(a, w, R)
        feature = build_curve_frame_features(
            ik1['ik2_teacher_input'],
            ik1['pRJ_ik1'],
            record['aM'][frame_idx],
            record['wM'][frame_idx],
            record['RMB'][frame_idx],
            prev_pRJ_ik1=prev_pRJ,
            use_imu=use_imu,
            use_feature_velocity=use_feature_velocity,
        )
        rows.append({
            'feature': feature,
            'pRJ_ik1': ik1['pRJ_ik1'],
            'gR2': ik1['gR2'],
        })
        prev_pRJ = ik1['pRJ_ik1']
    return rows


def finite(name, tensor):
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f'{name} contains non-finite values.')
    return True


def main():
    parser = argparse.ArgumentParser(description='No-training smoke for Curve-Control Pose Head no-IK-s2 branch.')
    parser.add_argument('--val-cache', default='data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json')
    parser.add_argument('--sequence-index', type=int, default=0)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--residual-scale', type=float, default=0.05)
    parser.add_argument('--curve-rnn-init-mode', choices=('none', 'r_js_firstframe'), default='r_js_firstframe')
    parser.add_argument('--curve-head-use-imu', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--curve-head-use-feature-velocity', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--curve-freeze-root-translation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--curve-predict-root-orientation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', default='data/experiments/CurveHead_P1_no_ik2_smoke_v1/smoke.json')
    args = parser.parse_args()

    records, manifest = load_records(args.val_cache)
    record = records[args.sequence_index]
    frames = min(args.frames, record['aM'].shape[0])
    if frames < 4:
        raise ValueError('Need at least 4 frames for smoke.')

    feature_net = GPNet().to(DEVICE).eval()
    feature_net.rnn_initialize(record['pose_gt'][0])
    rows = extract_curve_features(
        feature_net,
        record,
        frames,
        use_imu=args.curve_head_use_imu,
        use_feature_velocity=args.curve_head_use_feature_velocity,
    )
    if feature_net.ik2hc is not None:
        raise RuntimeError('no-IK-s2 feature path touched ik2hc.')

    frame_features = torch.stack([row['feature'] for row in rows]).to(DEVICE)
    offset_r = record.get('offset_r')
    offset_batch = None if offset_r is None else offset_r.view(1, 6, 3).to(DEVICE)
    head = CurveControlPoseHead(
        hidden_size=args.hidden_size,
        residual_scale=args.residual_scale,
        use_imu=args.curve_head_use_imu,
        use_feature_velocity=args.curve_head_use_feature_velocity,
        rnn_init_mode=args.curve_rnn_init_mode,
        freeze_root_translation=args.curve_freeze_root_translation,
        predict_root_orientation=args.curve_predict_root_orientation,
    ).to(DEVICE).eval()
    decoder = CurveStateDecoder().to(DEVICE).eval()
    with torch.no_grad():
        head_out = head(frame_features, offset_r=offset_batch)
        decoded = decoder(head_out['control'], return_pose=True)

    down_net = GPNet().to(DEVICE).eval()
    down_net.rnn_initialize(record['pose_gt'][0])
    pose_model = []
    tran_model = []
    downstream_debug = []
    for frame_idx in range(frames):
        pose_t, tran_t, debug = down_net.forward_frame_from_curve_pose(
            record['aM'][frame_idx].to(DEVICE),
            record['wM'][frame_idx].to(DEVICE),
            record['RMB'][frame_idx].to(DEVICE),
            decoded['pose'][frame_idx].detach().cpu(),
            rows[frame_idx]['gR2'],
        )
        pose_model.append(pose_t)
        tran_model.append(tran_t)
        downstream_debug.append(debug)
    if down_net.ik2hc is not None:
        raise RuntimeError('no-IK-s2 downstream path touched ik2hc.')

    baseline_net = GPNet().to(DEVICE).eval()
    baseline_net.rnn_initialize(record['pose_gt'][0])
    baseline_pose, baseline_tran = baseline_net.forward_frame(
        record['aM'][0].to(DEVICE),
        record['wM'][0].to(DEVICE),
        record['RMB'][0].to(DEVICE),
    )

    checks = {
        'sequence_name': record['name'],
        'frames': frames,
        'manifest': args.val_cache,
        'used_s5': False,
        'training_started': False,
        'no_ik2_branch_called_ik2': False,
        'official_forward_frame_checked': True,
        'config': vars(args),
        'input_dim': int(frame_features.shape[-1]),
        'input_components': {
            'ik2_original_input': 117,
            'feature_velocity': 69 if args.curve_head_use_feature_velocity else 0,
            'imu': 90 if args.curve_head_use_imu else 0,
        },
        'rnn_init_input_dim': 18 + int(frame_features.shape[-1]) if args.curve_rnn_init_mode == 'r_js_firstframe' else 0,
        'shapes': {
            'frame_features': list(frame_features.shape),
            'control': list(head_out['control'].shape),
            'q75': list(decoded['q75'].shape),
            'qdot': list(decoded['qdot'].shape),
            'qddot': list(decoded['qddot'].shape),
            'pose': list(decoded['pose'].shape),
            'tran': list(decoded['tran'].shape),
            'RRJ': list(downstream_debug[0]['RRJ'].shape),
            'pRJ': list(downstream_debug[0]['pRJ'].shape),
            'aRB': list(downstream_debug[0]['aRB'].shape),
            'wRB': list(downstream_debug[0]['wRB'].shape),
            'baseline_pose_first_frame': list(baseline_pose.shape),
            'baseline_tran_first_frame': list(baseline_tran.shape),
        },
        'finite': {
            'frame_features': finite('frame_features', frame_features),
            'control': finite('control', head_out['control']),
            'q75': finite('q75', decoded['q75']),
            'qdot': finite('qdot', decoded['qdot']),
            'qddot': finite('qddot', decoded['qddot']),
            'pose': finite('pose', decoded['pose']),
            'tran': finite('tran', decoded['tran']),
            'pose_model': finite('pose_model', torch.stack(pose_model)),
            'tran_model': finite('tran_model', torch.stack(tran_model)),
            'baseline_pose_first_frame': finite('baseline_pose_first_frame', baseline_pose),
            'baseline_tran_first_frame': finite('baseline_tran_first_frame', baseline_tran),
        },
        'downstream_recomputed': ['RRJ', 'pRJ', 'aRB', 'wRB', 'VR-s1 input', 'cjoint', 'velocity', 'physics pose target'],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
