import argparse
import json
import math
import random
import shlex
import sys
import time
from pathlib import Path

import torch

import articulate as art
from k2_so3_curve import StreamingTailUpdateSO3State, pose_tran_to_so3_state, q75_to_so3_state
from l4_q75_utils import rotation_matrix_to_6d
from l4_train_diverse_short import DEVICE, load_records, rotation_geodesic
from l4_train_loss_ablation import default_weights, load_compatible_state, root_relative_joints, temporal_slice


DT = 1.0 / 60.0


def mean(values):
    return sum(values) / max(1, len(values))


def batch_feature(q75, pose, a, w, R, pose_input_mode='rot6d'):
    if pose_input_mode != 'rot6d':
        raise ValueError('K2_SO3Curve_v1 batched trainer currently uses rot6d pose input.')
    pose_feat = pose[..., :, :2].reshape(pose.shape[0], -1).detach()
    imu_feat = torch.cat((a.reshape(a.shape[0], -1), w.reshape(w.shape[0], -1), R.reshape(R.shape[0], -1)), dim=-1).detach()
    return torch.cat((pose_feat, imu_feat), dim=-1)


def slice_window(record, start, length):
    end = start + length
    item = {
        'name': record['name'],
        'q75_prephysics': record['q75_prephysics'][start:end],
        'pose_prephysics': record.get('pose_prephysics', record['pose_gt'])[start:end],
        'pose_gt': record['pose_gt'][start:end],
        'tran_gt': record['tran_gt'][start:end],
        'q75_gt': record['q75_gt'][start:end],
        'aM': record['aM'][start:end],
        'wM': record['wM'][start:end],
        'RMB': record['RMB'][start:end],
        'v_root_vr': record['v_root_vr'][start:end],
        'stationary_prob': record['stationary_prob'][start:end],
        'offset_r': record.get('offset_r', torch.zeros(6, 3)),
    }
    actual = item['q75_prephysics'].shape[0]
    if actual < length:
        for key, value in list(item.items()):
            if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == actual:
                pad = value[-1:].expand((length - actual,) + value.shape[1:])
                item[key] = torch.cat((value, pad), dim=0)
    return item


def collate_windows(records, window):
    batch = {}
    for key in ('q75_prephysics', 'pose_prephysics', 'pose_gt', 'tran_gt', 'q75_gt', 'aM', 'wM', 'RMB', 'v_root_vr', 'stationary_prob', 'offset_r'):
        batch[key] = torch.stack([record[key] for record in records]).to(DEVICE)
    batch['name'] = [record['name'] for record in records]
    batch['window'] = window
    return batch


def run_batch_sequence(model, batch):
    B, T = batch['q75_prephysics'].shape[:2]
    init_feature = batch_feature(
        batch['q75_prephysics'][:, 0],
        batch['pose_prephysics'][:, 0],
        batch['aM'][:, 0],
        batch['wM'][:, 0],
        batch['RMB'][:, 0],
        pose_input_mode=model.pose_input_mode,
    )
    model.reset_stream(batch['offset_r'], init_feature)
    base_so3 = q75_to_so3_state(batch['q75_prephysics'].reshape(-1, 75), euler_seq=model.euler_seq).reshape(B, T, 75)
    qs, qdots, qddots, poses, residuals, new_norms, tail_norms, cp_priors = [], [], [], [], [], [], [], []
    v_refined, delta_vs = [], []
    for t in range(T):
        feature = batch_feature(
            batch['q75_prephysics'][:, t],
            batch['pose_prephysics'][:, t],
            batch['aM'][:, t],
            batch['wM'][:, t],
            batch['RMB'][:, t],
            pose_input_mode=model.pose_input_mode,
        )
        out = model.step(feature, batch['q75_prephysics'][:, t], base_so3[:, t], return_euler=False)
        vout = model.refine_velocity(batch['v_root_vr'][:, t], batch['stationary_prob'][:, t])
        qs.append(out['q_t'])
        qdots.append(out['qdot_t'])
        qddots.append(out['qddot_t'])
        poses.append(out['pose_R_t'])
        residuals.append(out['residual_t'])
        new_norms.append(out['new_delta_norm'])
        tail_norms.append(out['tail_delta_norm'])
        cp_priors.append(out['control_point_prior_t'])
        v_refined.append(vout['v_root_refined'])
        delta_vs.append(vout['delta_v_root'])
    return {
        'q_pred': torch.stack(qs, dim=1),
        'qdot_pred': torch.stack(qdots, dim=1),
        'qddot_pred': torch.stack(qddots, dim=1),
        'pose_pred': torch.stack(poses, dim=1),
        'q_residual': torch.stack(residuals, dim=1),
        'new_delta_norm': torch.stack(new_norms).mean(),
        'tail_delta_norm': torch.stack(tail_norms).mean(),
        'control_point_prior': torch.stack(cp_priors).mean(),
        'v_refined': torch.stack(v_refined, dim=1),
        'delta_v': torch.stack(delta_vs, dim=1),
    }


def finite_diff(x, order):
    if order == 1:
        return x[:, 1:] - x[:, :-1]
    if order == 2:
        return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
    raise ValueError(order)


def weighted_recent(x, indices):
    return x.index_select(1, indices.to(x.device))


def so3_batch_loss(output, batch, weights, recent_frames=4):
    q_pred_full = output['q_pred']
    B, T = q_pred_full.shape[:2]
    idx, _ = temporal_slice(T, mode='recent_l4', recent_frames=recent_frames, weighting='uniform', device=q_pred_full.device)
    q_pred = weighted_recent(q_pred_full, idx)
    qdot_pred = weighted_recent(output['qdot_pred'], idx)
    pose_pred = weighted_recent(output['pose_pred'], idx)
    q_base = weighted_recent(q75_to_so3_state(batch['q75_prephysics'].reshape(-1, 75)).reshape(B, T, 75), idx)
    q_gt = weighted_recent(pose_tran_to_so3_state(batch['pose_gt'].reshape(-1, 24, 3, 3), batch['tran_gt'].reshape(-1, 3)).reshape(B, T, 75), idx)
    pose_gt = weighted_recent(batch['pose_gt'], idx)
    pose_base = weighted_recent(batch['pose_prephysics'], idx)
    q_res = q_pred - q_base
    geo = rotation_geodesic(pose_pred, pose_gt)
    base_geo = rotation_geodesic(pose_pred, pose_base)
    losses = {
        'pose_geodesic': geo.mean(),
        'pose_geodesic_root': geo[:, :, 0].mean().detach(),
        'pose_geodesic_body': geo[:, :, 1:].mean().detach(),
        'q_body': torch.nn.functional.smooth_l1_loss(q_pred[..., 6:], q_gt[..., 6:]),
        'q_root_ori': torch.nn.functional.smooth_l1_loss(q_pred[..., 3:6], q_gt[..., 3:6]),
        'baseline_body': base_geo[:, :, 1:].mean(),
        'baseline_root_ori': base_geo[:, :, 0].mean(),
        'root_translation': torch.nn.functional.smooth_l1_loss(q_pred[..., :3], q_gt[..., :3]),
        'residual_prior': q_res.square().mean(),
        'tail_update_prior': output['tail_delta_norm'],
        'control_point_prior': output['control_point_prior'],
        'qddot_body_smooth': weighted_recent(output['qddot_pred'], idx)[..., 6:].square().mean(),
        'trajectory_q_body_prior': q_res[..., 6:].square().mean(),
        'root_velocity': torch.nn.functional.smooth_l1_loss(
            output['v_refined'][:, 1:],
            (batch['tran_gt'][:, 1:] - batch['tran_gt'][:, :-1]) / DT,
        ) if T > 1 else q_pred.new_zeros(()),
        'baseline_velocity': torch.nn.functional.smooth_l1_loss(
            batch['v_root_vr'][:, 1:],
            (batch['tran_gt'][:, 1:] - batch['tran_gt'][:, :-1]) / DT,
        ) if T > 1 else q_pred.new_zeros(()),
        'velocity_smooth': output['delta_v'].square().mean(),
    }
    if q_pred.shape[1] >= 2:
        losses['qdot'] = torch.nn.functional.smooth_l1_loss(finite_diff(q_pred, 1), finite_diff(q_gt, 1))
        losses['edge_q'] = q_res[:, 0].square().mean()
        losses['edge_qdot'] = finite_diff(q_res, 1)[:, 0].square().mean()
        losses['qdot_consistency_body'] = torch.nn.functional.smooth_l1_loss(q_pred[:, 1:, 6:] - q_pred[:, :-1, 6:], DT * qdot_pred[:, 1:, 6:])
    else:
        losses['qdot'] = q_pred.new_zeros(())
        losses['edge_q'] = q_pred.new_zeros(())
        losses['edge_qdot'] = q_pred.new_zeros(())
        losses['qdot_consistency_body'] = q_pred.new_zeros(())
    if q_pred.shape[1] >= 3:
        losses['qddot'] = torch.nn.functional.smooth_l1_loss(finite_diff(q_pred, 2), finite_diff(q_gt, 2))
        losses['edge_qddot'] = finite_diff(q_res, 2)[:, 0].square().mean()
    else:
        losses['qddot'] = q_pred.new_zeros(())
        losses['edge_qddot'] = q_pred.new_zeros(())
    flat_pred = pose_pred.reshape(-1, 24, 3, 3)
    flat_gt = pose_gt.reshape(-1, 24, 3, 3)
    losses['fk_joint_rootrel'] = torch.nn.functional.smooth_l1_loss(root_relative_joints(flat_pred), root_relative_joints(flat_gt))
    total = q_pred.new_zeros(())
    for key, weight in weights.items():
        if key in losses:
            total = total + losses[key] * weight
    return total, losses


def make_model(args):
    model = StreamingTailUpdateSO3State(
        hidden_size=args.hidden_size,
        residual_scale=args.residual_scale,
        velocity_residual_scale=args.velocity_residual_scale,
        pose_input_mode='rot6d',
        offset_conditioning='hidden_init',
        rnn_init_mode='offset_firstframe',
        offset_init_scale=args.offset_init_scale,
        dropout=args.dropout,
        imu_feature_dropout=args.imu_feature_dropout,
        acc_dropout=args.acc_dropout,
        gyro_dropout=args.gyro_dropout,
        orientation_dropout=args.orientation_dropout,
    ).to(DEVICE)
    model.l4_imu_field_prefix = 'original'
    model.fast_training_so3 = True
    return model


def save_checkpoint(path, model, optimizer, args, epoch, step, loss_value, weights, selection):
    cfg = vars(args).copy()
    cfg.update({
        'model_type': 'k2_so3curve_v1',
        'pose_input_mode': 'rot6d',
        'offset_conditioning': 'hidden_init',
        'effective_rnn_init_mode': 'offset_firstframe',
        'rnn_init_mode': 'offset_firstframe',
        'l4_imu_field_prefix': 'original',
        'loss_temporal_mode': 'recent_l4',
        'recent_loss_frames': args.recent_loss_frames,
        'recent_loss_weighting': 'uniform',
    })
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg,
        'epoch': epoch,
        'step': step,
        'validation_score': None,
        'best_loss_value': loss_value,
        'weights': weights,
        'selection': selection,
    }, path)


def run_epoch(model, records, optimizer, weights, args, epoch):
    model.train()
    order = list(range(len(records)))
    random.Random(args.seed + epoch).shuffle(order)
    rows = []
    totals = {}
    step = 0
    for batch_start in range(0, len(order), args.batch_size):
        ids = order[batch_start:batch_start + args.batch_size]
        windows = []
        for idx in ids:
            record = records[idx]
            max_start = max(0, record['q75_prephysics'].shape[0] - args.window)
            start = (epoch + idx + batch_start) % (max_start + 1) if max_start > 0 else 0
            windows.append(slice_window(record, start, args.window))
        batch = collate_windows(windows, args.window)
        output = run_batch_sequence(model, batch)
        loss, components = so3_batch_loss(output, batch, weights, recent_frames=args.recent_loss_frames)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at epoch={epoch} batch_start={batch_start}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f'Non-finite gradient at epoch={epoch} batch_start={batch_start}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        step += 1
        row = {
            'loss': float(loss.detach()),
            'q_residual_norm_mean': float(output['q_residual'].norm(dim=-1).mean().detach()),
            'tail_update_norm_mean': float(output['tail_delta_norm'].detach()),
        }
        row.update({key: float(value.detach()) for key, value in components.items()})
        rows.append(row)
        for key, value in row.items():
            totals.setdefault(key, []).append(float(value))
    return {key: mean(value) for key, value in totals.items()}, rows, step


@torch.no_grad()
def cache_eval(model, records, weights, args):
    model.eval()
    selected = records[:args.max_val_sequences] if args.max_val_sequences else records
    rows, totals = [], {}
    for start in range(0, len(selected), args.batch_size):
        windows = [slice_window(record, 0, min(args.window, record['q75_prephysics'].shape[0])) for record in selected[start:start + args.batch_size]]
        batch = collate_windows(windows, args.window)
        output = run_batch_sequence(model, batch)
        loss, components = so3_batch_loss(output, batch, weights, recent_frames=args.recent_loss_frames)
        row = {'loss': float(loss.detach())}
        row.update({key: float(value.detach()) for key, value in components.items()})
        rows.append(row)
        for key, value in row.items():
            totals.setdefault(key, []).append(float(value))
    return {'num_sequences': len(selected), 'loss': {key: mean(value) for key, value in totals.items()}, 'rows': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--velocity-residual-scale', type=float, default=0.0)
    parser.add_argument('--offset-init-scale', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--imu-feature-dropout', type=float, default=0.0)
    parser.add_argument('--acc-dropout', type=float, default=0.0)
    parser.add_argument('--gyro-dropout', type=float, default=0.0)
    parser.add_argument('--orientation-dropout', type=float, default=0.0)
    parser.add_argument('--recent-loss-frames', type=int, default=4)
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=100)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_records, train_manifest = load_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, max_sequences=args.max_val_sequences)
    weights = default_weights(disable_root_velocity_loss=False)
    model = make_model(args)
    init_info = load_compatible_state(
        model,
        Path(args.init_checkpoint),
        allow_partial=True,
        skip_prefixes=('new_control.', 'tail_delta.'),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    result = {
        'experiment_name': args.experiment_name,
        'config': vars(args),
        'model_type': 'k2_so3curve_v1',
        'train_cache_manifest': train_manifest,
        'val_cache_manifest': val_manifest,
        'num_train_sequences': len(train_records),
        'num_val_sequences': len(val_records),
        'init_checkpoint_load': init_info | {'path': str(init_info.get('path', ''))},
        'weights': weights,
        'epochs': [],
        'best_loss': None,
        'status': 'running',
    }
    best_loss = None
    global_step = 0
    with (out / 'train_log.jsonl').open('w') as log:
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_rows, steps = run_epoch(model, train_records, optimizer, weights, args, epoch)
            global_step += steps
            val = cache_eval(model, val_records, weights, args)
            loss_select = val['loss']['loss']
            improved = best_loss is None or loss_select < best_loss
            if improved:
                best_loss = loss_select
                save_checkpoint(out / 'best_loss.pt', model, optimizer, args, epoch, global_step, best_loss, weights, 'lowest lightweight SO3 cache validation loss')
            save_checkpoint(out / 'last.pt', model, optimizer, args, epoch, global_step, loss_select, weights, 'last epoch')
            rec = {
                'epoch': epoch,
                'step': global_step,
                'epoch_wall_seconds': time.time() - start,
                'train_loss': train_loss,
                'cache_validation': val,
                'loss_selection': loss_select,
                'best_loss_value': best_loss,
                'improved_best_loss': improved,
            }
            result['epochs'].append(rec)
            if improved:
                result['best_loss'] = rec
            log.write(json.dumps(rec) + '\n')
            log.flush()
            (out / 'train_result.json').write_text(json.dumps(result, indent=2))
            print(
                f"epoch={epoch} train_loss={train_loss['loss']:.6g} "
                f"loss_select={loss_select:.6g} best_loss={best_loss:.6g} "
                f"seconds={rec['epoch_wall_seconds']:.2f}",
                flush=True,
            )
    result['status'] = 'completed'
    (out / 'train_result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'result_path': str(out / 'train_result.json'),
        'best_loss_checkpoint': str(out / 'best_loss.pt'),
        'last_checkpoint': str(out / 'last.pt'),
        'status': result['status'],
    }, indent=2))


if __name__ == '__main__':
    main()
