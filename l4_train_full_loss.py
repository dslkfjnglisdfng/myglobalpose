import argparse
import json
import time
from pathlib import Path

import torch

import articulate as art
from l4_q75_utils import prephysics_feature, q75_to_pose_tran
from l4_tail_update_qstate import StreamingTailUpdateQState
from l4_velocity_losses import finite_difference_translation_velocity, velocity_residual_losses


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_BODY_MODEL = None


def load_cache_files(cache_path):
    path = Path(cache_path)
    if path.suffix == '.json':
        manifest = json.loads(path.read_text())
        return [Path(item['path']) for item in manifest['cache_files']], manifest
    return [path], None


def load_records(cache_path):
    files, manifest = load_cache_files(cache_path)
    records = []
    required = ('q75_prephysics', 'v_root_vr', 'stationary_prob', 'q75_gt', 'aM', 'wM', 'RMB')
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        missing = [key for key in required if key not in data or not data[key]]
        if missing:
            raise KeyError(f'{cache_file} missing required fields: {missing}')
        for seq_idx, name in enumerate(data['name']):
            if 'pose_gt' in data and data['pose_gt']:
                pose_gt = data['pose_gt'][seq_idx].float()
                tran_gt = data['tran_gt'][seq_idx].float()
            else:
                pose_gt, tran_gt = q75_to_pose_tran(data['q75_gt'][seq_idx].float())
            records.append({
                'name': name,
                'q75_prephysics': data['q75_prephysics'][seq_idx].float(),
                'v_root_vr': data['v_root_vr'][seq_idx].float(),
                'stationary_prob': data['stationary_prob'][seq_idx].float(),
                'q75_gt': data['q75_gt'][seq_idx].float(),
                'pose_gt': pose_gt.float(),
                'tran_gt': tran_gt.float(),
                'aM': data['aM'][seq_idx].float(),
                'wM': data['wM'][seq_idx].float(),
                'RMB': data['RMB'][seq_idx].float(),
            })
    return records, manifest


def default_weights(disable_root_velocity_loss=False):
    weights = {
        'pose_geodesic': 1.0,
        'q_body': 1.0,
        'q_root_ori': 0.5,
        'baseline_body': 2.0,
        'baseline_root_ori': 5.0,
        'qdot': 0.03,
        'qddot': 0.003,
        'fk_joint_rootrel': 0.1,
        'residual_prior': 0.001,
        'tail_update_prior': 0.005,
        'edge_q': 0.01,
        'edge_qdot': 0.03,
        'edge_qddot': 0.003,
        'jerk_smooth': 1e-5,
        'root_velocity': 1.0,
        'baseline_velocity': 2.0,
        'velocity_smooth': 0.03,
    }
    if disable_root_velocity_loss:
        weights['root_velocity'] = 0.0
    return weights


def average(items):
    return sum(items) / max(1, len(items))


def finite_diff(x, order):
    if order == 1:
        return x[1:] - x[:-1]
    if order == 2:
        return x[2:] - 2.0 * x[1:-1] + x[:-2]
    if order == 3:
        return x[3:] - 3.0 * x[2:-1] + 3.0 * x[1:-2] - x[:-3]
    raise ValueError(order)


def q75_residual(q_pred, q_base):
    rot = torch.atan2(torch.sin(q_pred[..., 3:] - q_base[..., 3:]), torch.cos(q_pred[..., 3:] - q_base[..., 3:]))
    return torch.cat((q_pred[..., :3] - q_base[..., :3], rot), dim=-1)


def rotation_geodesic(R_pred, R_target, eps=1e-6):
    rel = R_pred.transpose(-1, -2).matmul(R_target)
    trace = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


def root_relative_joints(pose):
    global _BODY_MODEL
    if _BODY_MODEL is None:
        _BODY_MODEL = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
    joints = _BODY_MODEL.forward_kinematics(pose.to(DEVICE))[1]
    return joints - joints[:, :1]


def slice_record(record, start, length):
    seq_len = record['q75_prephysics'].shape[0]
    if length <= 0 or seq_len <= length:
        return record
    start = min(max(0, start), seq_len - length)
    end = start + length
    sliced = {}
    for key, value in record.items():
        if torch.is_tensor(value) and value.shape[0] == seq_len:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    sliced['name'] = f"{record['name']}[{start}:{end}]"
    return sliced


def run_cached_sequence(model, record):
    model.reset_stream()
    qs = []
    q_residuals = []
    new_norms = []
    tail_norms = []
    v_refined = []
    delta_vs = []
    for frame_idx in range(record['q75_prephysics'].shape[0]):
        q_base = record['q75_prephysics'][frame_idx].to(DEVICE)
        feature = prephysics_feature(
            q_base.detach().cpu(),
            record['aM'][frame_idx],
            record['wM'][frame_idx],
            record['RMB'][frame_idx],
        ).to(DEVICE)
        q_result = model.step(feature, q_base)
        v_result = model.refine_velocity(
            record['v_root_vr'][frame_idx].to(DEVICE),
            record['stationary_prob'][frame_idx].to(DEVICE),
        )
        qs.append(q_result['q_t'][0])
        q_residuals.append(q_result['residual_t'][0])
        new_norms.append(q_result['new_delta_norm'])
        tail_norms.append(q_result['tail_delta_norm'])
        v_refined.append(v_result['v_root_refined'][0])
        delta_vs.append(v_result['delta_v_root'][0])
    return {
        'q_pred': torch.stack(qs),
        'q_residual': torch.stack(q_residuals),
        'new_delta_norm': torch.stack(new_norms).mean(),
        'tail_delta_norm': torch.stack(tail_norms).mean(),
        'v_refined': torch.stack(v_refined),
        'delta_v': torch.stack(delta_vs),
    }


def pose_velocity_loss(model_output, record, weights):
    q_pred = model_output['q_pred']
    q_base = record['q75_prephysics'].to(DEVICE)
    q_gt = record['q75_gt'].to(DEVICE)
    pose_pred, _ = q75_to_pose_tran(q_pred)
    pose_gt = record['pose_gt'].to(DEVICE)
    pose_base, _ = q75_to_pose_tran(q_base)
    pose_pred = pose_pred.to(DEVICE)
    pose_base = pose_base.to(DEVICE)
    geo = rotation_geodesic(pose_pred, pose_gt)
    base_geo = rotation_geodesic(pose_pred, pose_base)
    q_res = q75_residual(q_pred, q_base)
    _, velocity_components, _ = velocity_residual_losses(
        model_output['v_refined'],
        finite_difference_translation_velocity(record['tran_gt'].to(DEVICE)),
        record['v_root_vr'].to(DEVICE),
        model_output['delta_v'],
    )
    losses = {
        'pose_geodesic': geo.mean(),
        'pose_geodesic_root': geo[:, 0].mean().detach(),
        'pose_geodesic_body': geo[:, 1:].mean().detach(),
        'q_body': torch.nn.functional.smooth_l1_loss(q_pred[:, 6:], q_gt[:, 6:]),
        'q_root_ori': torch.nn.functional.smooth_l1_loss(q_pred[:, 3:6], q_gt[:, 3:6]),
        'baseline_body': base_geo[:, 1:].mean(),
        'baseline_root_ori': base_geo[:, 0].mean(),
        'residual_prior': q_res.square().mean(),
        'tail_update_prior': model_output['tail_delta_norm'],
        'root_velocity': velocity_components['root_velocity'],
        'baseline_velocity': velocity_components['baseline_velocity'],
        'velocity_smooth': velocity_components['velocity_smooth'],
    }
    if q_pred.shape[0] >= 2:
        losses['qdot'] = torch.nn.functional.smooth_l1_loss(finite_diff(q_pred, 1), finite_diff(q_gt, 1))
        losses['edge_q'] = q_res[0].square().mean()
        losses['edge_qdot'] = finite_diff(q_res, 1)[0].square().mean()
    else:
        losses['qdot'] = q_pred.new_zeros(())
        losses['edge_q'] = q_pred.new_zeros(())
        losses['edge_qdot'] = q_pred.new_zeros(())
    if q_pred.shape[0] >= 3:
        losses['qddot'] = torch.nn.functional.smooth_l1_loss(finite_diff(q_pred, 2), finite_diff(q_gt, 2))
        losses['edge_qddot'] = finite_diff(q_res, 2)[0].square().mean()
    else:
        losses['qddot'] = q_pred.new_zeros(())
        losses['edge_qddot'] = q_pred.new_zeros(())
    if q_pred.shape[0] >= 4:
        losses['jerk_smooth'] = finite_diff(q_pred, 3).square().mean()
    else:
        losses['jerk_smooth'] = q_pred.new_zeros(())
    losses['fk_joint_rootrel'] = torch.nn.functional.smooth_l1_loss(root_relative_joints(pose_pred), root_relative_joints(pose_gt))
    total = q_pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * weight
    return total, losses


def save_checkpoint(path, model, optimizer, args, epoch, step, loss_value, weights):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'epoch': epoch,
        'step': step,
        'train_loss': loss_value,
        'weights': weights,
        'selection': 'full_sequence_set_training_loss',
    }, path)


def main():
    parser = argparse.ArgumentParser(description='TransPose-style full-cache loss-only L4 pose+velocity trainer.')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.01)
    parser.add_argument('--velocity-residual-scale', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--disable-root-velocity-loss', action='store_true')
    args = parser.parse_args()
    if args.tail_length != 4:
        raise ValueError('Only tail_length=4 is approved for the current L4 method.')

    records, manifest = load_records(args.cache)
    weights = default_weights(args.disable_root_velocity_loss)
    model = StreamingTailUpdateQState(
        hidden_size=args.hidden_size,
        residual_scale=args.residual_scale,
        velocity_residual_scale=args.velocity_residual_scale,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 1
    step = 0
    best_loss = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        step = int(checkpoint.get('step', 0))
        best_loss = checkpoint.get('train_loss')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / 'train_result.json'
    log_path = output_dir / 'train_log.jsonl'
    result = {
        'config': vars(args),
        'cache_manifest': manifest,
        'num_sequences': len(records),
        'weights': weights,
        'training_mode': 'transpose_style_full_cache_window_loss_only',
        'validation': 'disabled',
        'motion_evaluator': 'disabled_during_training',
        'checkpoint_selection': 'lowest_epoch_mean_training_loss',
        'epochs': [],
        'best': None,
        'status': 'running',
    }
    print(
        f"training_mode=transpose_style_full_cache_window_loss_only num_sequences={len(records)} "
        f"window={args.window} epochs={args.epochs} steps_per_epoch={len(records)} "
        f"lr={args.lr} residual_scale={args.residual_scale} "
        f"velocity_residual_scale={args.velocity_residual_scale}",
        flush=True,
    )
    print(f"loss_weights={json.dumps(weights, sort_keys=True)}", flush=True)

    with log_path.open('a') as log_file:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            totals = {}
            losses_for_epoch = []
            model.train()
            for seq_idx, source_record in enumerate(records, start=1):
                step += 1
                seq_len = source_record['q75_prephysics'].shape[0]
                max_start = max(0, seq_len - args.window)
                start = step % (max_start + 1) if max_start > 0 else 0
                record = slice_record(source_record, start, args.window)
                output = run_cached_sequence(model, record)
                loss, components = pose_velocity_loss(output, record, weights)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch={epoch} seq={seq_idx} name={record["name"]}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                if not grad_ok:
                    raise RuntimeError(f'Non-finite gradient at epoch={epoch} seq={seq_idx} name={record["name"]}')
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                q_norm = output['q_residual'].norm(dim=-1)
                dv_norm = output['delta_v'].norm(dim=-1)
                row = {
                    'epoch': epoch,
                    'step': step,
                    'seq_idx': seq_idx,
                    'seq_name': record['name'],
                    'window': int(record['q75_prephysics'].shape[0]),
                    'loss': float(loss.detach()),
                    'q_residual_norm_mean': float(q_norm.mean().detach()),
                    'q_residual_norm_max': float(q_norm.max().detach()),
                    'delta_v_root_norm_mean': float(dv_norm.mean().detach()),
                    'delta_v_root_norm_max': float(dv_norm.max().detach()),
                    'tail_update_norm_mean': float(output['tail_delta_norm'].detach()),
                }
                for key, value in components.items():
                    row[key] = float(value.detach())
                log_file.write(json.dumps(row) + '\n')
                log_file.flush()
                print(
                    f"epoch={epoch} step={step} seq={seq_idx}/{len(records)} "
                    f"name={record['name']} window={row['window']} loss={row['loss']:.6g} "
                    f"q_res={row['q_residual_norm_mean']:.6g} "
                    f"dv={row['delta_v_root_norm_mean']:.6g} "
                    f"tail={row['tail_update_norm_mean']:.6g}",
                    flush=True,
                )
                losses_for_epoch.append(row['loss'])
                for key, value in row.items():
                    if key in ('epoch', 'step', 'seq_idx', 'seq_name'):
                        continue
                    totals.setdefault(key, []).append(value)

            epoch_loss = average(losses_for_epoch)
            epoch_record = {
                'epoch': epoch,
                'step': step,
                'epoch_wall_seconds': time.time() - epoch_start,
                'train_loss': {key: average(value) for key, value in totals.items()},
            }
            result['epochs'].append(epoch_record)
            save_checkpoint(output_dir / 'last.pt', model, optimizer, args, epoch, step, epoch_loss, weights)
            if best_loss is None or epoch_loss < best_loss:
                best_loss = epoch_loss
                result['best'] = epoch_record
                save_checkpoint(output_dir / 'best.pt', model, optimizer, args, epoch, step, epoch_loss, weights)
                print(f"Saved best checkpoint: {output_dir / 'best.pt'} train_loss={best_loss:.6g}", flush=True)
            print(
                f"epoch_summary epoch={epoch} loss={epoch_loss:.6g} "
                f"seconds={epoch_record['epoch_wall_seconds']:.2f} best_loss={best_loss:.6g}",
                flush=True,
            )
            result_path.write_text(json.dumps(result, indent=2))

    result['status'] = 'completed'
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({'result_path': str(result_path), 'status': result['status'], 'best_epoch': result['best']['epoch'] if result['best'] else None}, indent=2))


if __name__ == '__main__':
    main()
