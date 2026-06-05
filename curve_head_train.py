import argparse
import json
import time
from pathlib import Path

import torch

import articulate as art
from curve_control_pose_head import CurveControlPoseHead, build_curve_frame_features
from curve_state_decoder import CurveStateDecoder
from l4_q75_utils import q75_to_pose_tran
from l4_train_diverse_short import load_records
from net import GPNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = 1.0 / 60.0
_BODY_MODEL = None


def body_model():
    global _BODY_MODEL
    if _BODY_MODEL is None:
        _BODY_MODEL = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    return _BODY_MODEL


def finite_diff(x, order):
    if order == 1:
        return x[1:] - x[:-1]
    if order == 2:
        return x[2:] - 2.0 * x[1:-1] + x[:-2]
    raise ValueError(order)


def rotation_geodesic(R_pred, R_target, eps=1e-6):
    rel = R_pred.transpose(-1, -2).matmul(R_target)
    trace = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


def root_relative_joints(pose):
    joints = body_model().forward_kinematics(pose.to(DEVICE))[1]
    return joints - joints[:, :1]


def temporal_indices(length, recent_frames):
    start = max(0, length - int(recent_frames))
    return torch.arange(start, length, device=DEVICE)


@torch.no_grad()
def extract_window_features(net, record, start, length, use_imu=True, use_feature_velocity=True):
    end = min(start + length, record['aM'].shape[0])
    if end - start < 4:
        start = max(0, end - 4)
    net.rnn_initialize(record['pose_gt'][start])
    features = []
    gR2_rows = []
    prev_pRJ = None
    for frame_idx in range(start, end):
        ik1 = net.forward_until_ik1(
            record['aM'][frame_idx].to(DEVICE),
            record['wM'][frame_idx].to(DEVICE),
            record['RMB'][frame_idx].to(DEVICE),
        )
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
        features.append(feature)
        gR2_rows.append(ik1['gR2'])
        prev_pRJ = ik1['pRJ_ik1']
    if net.ik2hc is not None:
        raise RuntimeError('CurveHead training feature extraction touched IK-s2 hidden state.')
    return torch.stack(features).to(DEVICE), torch.stack(gR2_rows), start, end


def curve_loss(head_out, decoded, record, start, end, args):
    q_pred_full = decoded['q75']
    pose_pred_full = decoded['pose'].to(DEVICE)
    q_gt_full = record['q75_gt'][start:end].to(DEVICE)
    pose_gt_full = record['pose_gt'][start:end].to(DEVICE)
    idx = temporal_indices(q_pred_full.shape[0], args.recent_loss_frames)
    q_pred = q_pred_full.index_select(0, idx)
    pose_pred = pose_pred_full.index_select(0, idx)
    q_gt = q_gt_full.index_select(0, idx)
    pose_gt = pose_gt_full.index_select(0, idx)

    losses = {
        'q_body': torch.nn.functional.smooth_l1_loss(q_pred[:, 6:], q_gt[:, 6:]),
        'q_root_ori': torch.nn.functional.smooth_l1_loss(q_pred[:, 3:6], q_gt[:, 3:6]),
        'pose_geodesic': rotation_geodesic(pose_pred, pose_gt).mean(),
        'fk_joint_rootrel': torch.nn.functional.smooth_l1_loss(root_relative_joints(pose_pred), root_relative_joints(pose_gt)),
        'control_update_reg': head_out['delta_control'].index_select(0, idx).square().mean(),
        'loss_temporal_num_frames': q_pred.new_tensor(float(idx.numel())),
        'loss_temporal_start_index': q_pred.new_tensor(float(idx[0].item())),
        'loss_temporal_end_index': q_pred.new_tensor(float(idx[-1].item())),
    }
    if q_pred_full.shape[0] >= 2:
        losses['qdot_smooth'] = decoded['qdot'].square().mean()
    else:
        losses['qdot_smooth'] = q_pred.new_zeros(())
    if q_pred_full.shape[0] >= 3:
        losses['qddot_smooth'] = decoded['qddot'].square().mean()
    else:
        losses['qddot_smooth'] = q_pred.new_zeros(())
    total = (
        args.q_body_weight * losses['q_body']
        + args.q_root_ori_weight * losses['q_root_ori']
        + args.pose_geodesic_weight * losses['pose_geodesic']
        + args.fk_joint_rootrel_weight * losses['fk_joint_rootrel']
        + args.qdot_smooth_weight * losses['qdot_smooth']
        + args.qddot_smooth_weight * losses['qddot_smooth']
        + args.control_update_reg_weight * losses['control_update_reg']
    )
    losses['loss'] = total
    return total, losses


def run_sequence(head, decoder, feature_net, record, start, length, args):
    features, gR2_rows, start, end = extract_window_features(
        feature_net,
        record,
        start,
        length,
        use_imu=args.curve_head_use_imu,
        use_feature_velocity=args.curve_head_use_feature_velocity,
    )
    offset = record.get('offset_r')
    offset = None if offset is None else offset.view(1, 6, 3).to(DEVICE)
    head_out = head(features, offset_r=offset)
    decoded = decoder(head_out['control'], return_pose=True)
    return head_out, decoded, start, end


def row_from_losses(losses):
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def average(rows):
    out = {}
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, (int, float)):
                continue
            out.setdefault(key, []).append(value)
    return {key: sum(values) / max(1, len(values)) for key, values in out.items()}


def train_epoch(head, decoder, feature_net, records, optimizer, args, step):
    head.train()
    rows = []
    for record in records:
        seq_len = record['q75_gt'].shape[0]
        max_start = max(0, seq_len - args.window)
        start = step % (max_start + 1) if max_start > 0 else 0
        step += 1
        head_out, decoded, start, end = run_sequence(head, decoder, feature_net, record, start, args.window, args)
        loss, losses = curve_loss(head_out, decoded, record, start, end, args)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at {record["name"]}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        optimizer.step()
        row = row_from_losses(losses)
        row['seq_name'] = record['name']
        row['window'] = end - start
        rows.append(row)
    return average(rows), rows, step


@torch.no_grad()
def cache_eval(head, decoder, feature_net, records, args):
    head.eval()
    rows = []
    selected = records[:args.max_val_sequences] if args.max_val_sequences else records
    for seq_idx, record in enumerate(selected):
        seq_len = record['q75_gt'].shape[0]
        if args.val_window > 0 and seq_len > args.val_window:
            start = (seq_idx * args.val_window) % max(1, seq_len - args.val_window + 1)
            length = args.val_window
        else:
            start = 0
            length = seq_len
        head_out, decoded, start, end = run_sequence(head, decoder, feature_net, record, start, length, args)
        loss, losses = curve_loss(head_out, decoded, record, start, end, args)
        row = row_from_losses(losses)
        row['seq_name'] = record['name']
        row['window'] = end - start
        rows.append(row)
    return average(rows), rows


def save_checkpoint(path, head, optimizer, args, epoch, step, val_loss, selection):
    torch.save({
        'model_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss,
        'selection': selection,
        'config': vars(args),
        'model_config': {
            'input_dim': head.input_dim,
            'hidden_size': head.hidden_size,
            'state_dim': head.state_dim,
            'residual_scale': head.residual_scale,
            'use_imu': head.use_imu,
            'use_feature_velocity': head.use_feature_velocity,
            'rnn_init_mode': head.rnn_init_mode,
            'freeze_root_translation': head.freeze_root_translation,
            'predict_root_orientation': head.predict_root_orientation,
            'offset_init_scale': head.offset_init_scale,
        },
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Phase-1 Curve-Control Pose Head TotalCapture trainer.')
    parser.add_argument('--train-cache', default='data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json')
    parser.add_argument('--val-cache', default='data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json')
    parser.add_argument('--output-dir', default='data/experiments/CurveHead_P1_TC_smoke_train_v1')
    parser.add_argument('--experiment-name', default='CurveHead_P1_TC_smoke_train_v1')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--val-window', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--residual-scale', type=float, default=0.05)
    parser.add_argument('--curve-rnn-init-mode', choices=('none', 'r_js_firstframe'), default='r_js_firstframe')
    parser.add_argument('--curve-head-use-imu', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--curve-head-use-feature-velocity', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--curve-freeze-root-translation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--curve-predict-root-orientation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--offset-init-scale', type=float, default=0.2)
    parser.add_argument('--recent-loss-frames', type=int, default=4)
    parser.add_argument('--recent-loss-weighting', choices=('uniform',), default='uniform')
    parser.add_argument('--q-body-weight', type=float, default=1.0)
    parser.add_argument('--q-root-ori-weight', type=float, default=0.5)
    parser.add_argument('--pose-geodesic-weight', type=float, default=1.0)
    parser.add_argument('--fk-joint-rootrel-weight', type=float, default=0.1)
    parser.add_argument('--qdot-smooth-weight', type=float, default=1e-5)
    parser.add_argument('--qddot-smooth-weight', type=float, default=1e-7)
    parser.add_argument('--control-update-reg-weight', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--validate-every', type=int, default=9999)
    parser.add_argument('--save-best-by', choices=('loss',), default='loss')
    parser.add_argument('--no-epoch-motion-eval', action='store_true', default=True)
    args = parser.parse_args()

    train_records, train_manifest = load_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, max_sequences=args.max_val_sequences)
    feature_net = GPNet().to(DEVICE).eval()
    for param in feature_net.parameters():
        param.requires_grad_(False)
    head = CurveControlPoseHead(
        hidden_size=args.hidden_size,
        residual_scale=args.residual_scale,
        use_imu=args.curve_head_use_imu,
        use_feature_velocity=args.curve_head_use_feature_velocity,
        rnn_init_mode=args.curve_rnn_init_mode,
        freeze_root_translation=args.curve_freeze_root_translation,
        predict_root_orientation=args.curve_predict_root_orientation,
        offset_init_scale=args.offset_init_scale,
    ).to(DEVICE)
    decoder = CurveStateDecoder().to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'train_log.jsonl'
    result_path = output_dir / 'train_result.json'
    result = {
        'experiment_name': args.experiment_name,
        'config': vars(args),
        'model_config': {
            'input_dim': head.input_dim,
            'input_components': {
                'ik2_original_input': 117,
                'feature_velocity': 69 if args.curve_head_use_feature_velocity else 0,
                'imu': 90 if args.curve_head_use_imu else 0,
            },
            'rnn_init_input_dim': 18 + head.input_dim if args.curve_rnn_init_mode == 'r_js_firstframe' else 0,
            'rnn_init_components': 'offset_r 18D + first-frame curve head feature' if args.curve_rnn_init_mode == 'r_js_firstframe' else 'none',
            'state_dim': 75,
            'output': 'curve control points C_t in R75; CurveStateDecoder returns q75/qdot/qddot',
            'inference_uses_ik2': False,
            'teacher_ik2_distillation': False,
            'distillation_note': 'Not implemented in v1; training uses GT q75/pose/FK losses.',
        },
        'train_cache_manifest': train_manifest,
        'val_cache_manifest': val_manifest,
        'num_train_sequences': len(train_records),
        'num_val_sequences': len(val_records),
        'official_weights_modified': False,
        'test_py_modified': False,
        'motion_evaluator_modified': False,
        's5_used': False,
        'epochs': [],
        'best_loss': None,
        'status': 'running',
    }
    best_loss = None
    step = 0
    with log_path.open('w') as log_file:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_loss, train_rows, step = train_epoch(head, decoder, feature_net, train_records, optimizer, args, step)
            val_loss, val_rows = cache_eval(head, decoder, feature_net, val_records, args)
            selection_loss = val_loss['loss']
            improved = best_loss is None or selection_loss < best_loss
            if improved:
                best_loss = selection_loss
                save_checkpoint(output_dir / 'best_loss.pt', head, optimizer, args, epoch, step, selection_loss, 'lowest lightweight S4 cache loss')
            save_checkpoint(output_dir / 'last.pt', head, optimizer, args, epoch, step, selection_loss, 'last epoch')
            epoch_record = {
                'epoch': epoch,
                'step': step,
                'epoch_wall_seconds': time.time() - start_time,
                'train_loss': train_loss,
                'cache_validation': val_loss,
                'validation_score': None,
                'physics_validation': None,
                'loss_selection': selection_loss,
                'best_loss_value': best_loss,
                'improved_best_loss': improved,
            }
            result['epochs'].append(epoch_record)
            if improved:
                result['best_loss'] = epoch_record
            log_file.write(json.dumps(epoch_record) + '\n')
            log_file.flush()
            result_path.write_text(json.dumps(result, indent=2))
            print(
                f"epoch={epoch} train_loss={train_loss['loss']:.6g} "
                f"val_loss={selection_loss:.6g} best_loss={best_loss:.6g} "
                f"seconds={epoch_record['epoch_wall_seconds']:.2f}",
                flush=True,
            )
    result['status'] = 'completed'
    result['best_loss_checkpoint'] = str(output_dir / 'best_loss.pt') if result['best_loss'] else None
    result['last_checkpoint'] = str(output_dir / 'last.pt')
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'result_path': str(result_path),
        'log_path': str(log_path),
        'best_loss_checkpoint': result['best_loss_checkpoint'],
        'last_checkpoint': result['last_checkpoint'],
        'status': result['status'],
    }, indent=2))


if __name__ == '__main__':
    main()
