import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from l4_q75_utils import q75_to_pose_tran
from l4_train_diverse_short import DEVICE, load_records
from net import GPNet
from newpl_root import (
    NewPLRootModule,
    extend_base_pl,
    freeze_root_head_gradients,
    load_partial_pl_checkpoint,
    newpl_root_loss,
    newpl_root_weights,
    pl_root_target_from_pose_tran,
)
from pl_curve import normalize_gravity, pl_init_feature_from_pose, pl_input_feature, pl_target_from_pose


def selected_imu_fields(record, mode):
    if mode == 'official':
        return record['aM'], record['wM'], record['RMB']
    has_l4 = all(key in record for key in ('l4_aM', 'l4_wM', 'l4_RMB'))
    if mode == 'processed':
        if not has_l4:
            raise KeyError(f'processed mode requires l4_aM/l4_wM/l4_RMB in record {record.get("name")}.')
        return record['l4_aM'], record['l4_wM'], record['l4_RMB']
    if mode == 'auto':
        if has_l4:
            return record['l4_aM'], record['l4_wM'], record['l4_RMB']
        return record['aM'], record['wM'], record['RMB']
    raise ValueError(f'Unsupported imu input mode: {mode}')


def slice_record(record, start, length):
    seq_len = record['pose_gt'].shape[0]
    if length <= 0 or seq_len <= length:
        return record
    start = min(max(0, start), seq_len - length)
    end = start + length
    sliced = {}
    for key, value in record.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == seq_len:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    sliced['name'] = f"{record['name']}[{start}:{end}]"
    return sliced


def make_batch(records, starts, length):
    sliced = [slice_record(record, int(start), length) for record, start in zip(records, starts)]
    return {
        'name': '|'.join(record['name'] for record in sliced),
        'records': sliced,
    }


def ensure_pose_tran(record):
    if 'pose_gt' in record and 'tran_gt' in record:
        return record['pose_gt'].float(), record['tran_gt'].float()
    if 'q75_gt' not in record:
        raise KeyError(f'{record.get("name")} has neither pose/tran GT nor q75_gt.')
    return q75_to_pose_tran(record['q75_gt'].float())


def build_features(record, imu_input_mode):
    a_seq, w_seq, R_seq = selected_imu_fields(record, imu_input_mode)
    return torch.stack([
        pl_input_feature(a_seq[i], w_seq[i], R_seq[i])
        for i in range(a_seq.shape[0])
    ]).float()


def init_feature_for_record(record, pose, body_model, allow_zero_offset_init=False):
    if 'offset_r' in record:
        offset_r = record['offset_r'].float()
    elif 'imu_offset_r' in record:
        offset_r = record['imu_offset_r'].float()
    elif allow_zero_offset_init:
        offset_r = torch.zeros(6, 3)
    else:
        raise KeyError(f'{record.get("name")} missing offset_r required by init36.')
    return pl_init_feature_from_pose(offset_r, pose[0].float(), body_model)


@torch.no_grad()
def base_pl_outputs(gpnet, features, init_target):
    gpnet.plnet.eval()
    return gpnet.plnet([(features.to(DEVICE), init_target.to(DEVICE))])[0].detach()


@torch.no_grad()
def base_pl_outputs_batch(gpnet, features, init_target):
    gpnet.plnet.eval()
    if features.dim() == 2:
        return base_pl_outputs(gpnet, features, init_target)
    batch_items = [
        (features[:, i].to(DEVICE), init_target[i].to(DEVICE))
        for i in range(features.shape[1])
    ]
    return torch.stack(gpnet.plnet(batch_items), dim=1).detach()


def build_target(record, body_model, dataset, root_vel_mode, dt):
    pose, tran = ensure_pose_tran(record)
    pose_device = pose.float().to(DEVICE)
    root_vel_available = dataset in ('amass', 'totalcapture') and root_vel_mode == 'gt'
    if root_vel_available:
        target = pl_root_target_from_pose_tran(pose_device, tran.float().to(DEVICE), body_model, dt=dt).cpu()
    else:
        pl_target = normalize_gravity(pl_target_from_pose(pose_device, body_model).float()).cpu()
        target = torch.cat((pl_target, torch.zeros(pl_target.shape[:-1] + (3,))), dim=-1)
    return pose, target.float(), root_vel_available


def run_sequence(model, gpnet, record, body_model, weights, args):
    if 'records' in record:
        return run_batch(model, gpnet, record['records'], body_model, weights, args)
    features = build_features(record, args.imu_input_mode).to(DEVICE)
    pose, target, root_vel_available = build_target(record, body_model, args.dataset, args.root_vel_mode, args.dt)
    target = target.to(DEVICE)
    base18 = base_pl_outputs(gpnet, features, normalize_gravity(target[..., :18])[0])
    base = extend_base_pl(base18)
    init_feature = init_feature_for_record(
        record,
        pose,
        body_model,
        allow_zero_offset_init=args.allow_zero_offset_init,
    ).to(DEVICE)
    out = model.forward_sequence(features, base, init_feature=init_feature)
    loss, components = newpl_root_loss(
        out,
        target,
        weights,
        root_vel_available=root_vel_available,
        teacher=None,
        dt=args.dt,
    )
    components.update({
        'new_delta_norm': out['new_delta_norm'],
        'pl_residual_norm_mean': (out['pl'][..., :18] - out['base'][..., :18]).norm(dim=-1).mean(),
        'root_vel_available': loss.new_tensor(1.0 if root_vel_available else 0.0),
        'root_vel_norm_mean': out['pl'][..., 18:21].norm(dim=-1).mean(),
    })
    return loss, components


def run_batch(model, gpnet, records, body_model, weights, args):
    features, targets, bases, init_features = [], [], [], []
    root_vel_flags = []
    for record in records:
        feat = build_features(record, args.imu_input_mode)
        pose, target, root_vel_available = build_target(record, body_model, args.dataset, args.root_vel_mode, args.dt)
        init_feature = init_feature_for_record(
            record,
            pose,
            body_model,
            allow_zero_offset_init=args.allow_zero_offset_init,
        )
        features.append(feat)
        targets.append(target.float())
        init_features.append(init_feature)
        root_vel_flags.append(root_vel_available)
    features = torch.stack(features, dim=1).to(DEVICE)
    target = torch.stack(targets, dim=1).to(DEVICE)
    init_feature = torch.stack(init_features, dim=0).to(DEVICE)
    init_target = normalize_gravity(target[..., :18])[0]
    base18 = base_pl_outputs_batch(gpnet, features, init_target)
    base = extend_base_pl(base18)
    out = model.forward_sequence(features, base, init_feature=init_feature)
    root_vel_available = all(root_vel_flags)
    loss, components = newpl_root_loss(
        out,
        target,
        weights,
        root_vel_available=root_vel_available,
        teacher=None,
        dt=args.dt,
    )
    components.update({
        'new_delta_norm': out['new_delta_norm'],
        'pl_residual_norm_mean': (out['pl'][..., :18] - out['base'][..., :18]).norm(dim=-1).mean(),
        'root_vel_available': loss.new_tensor(1.0 if root_vel_available else 0.0),
        'root_vel_norm_mean': out['pl'][..., 18:21].norm(dim=-1).mean(),
    })
    return loss, components


def average(rows):
    totals = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / max(1, len(values)) for key, values in totals.items()}


def eval_loss(model, gpnet, records, body_model, weights, args):
    model.eval()
    rows = []
    selected = records[:args.max_val_sequences] if args.max_val_sequences else records
    with torch.no_grad():
        for record in selected:
            loss, components = run_sequence(model, gpnet, record, body_model, weights, args)
            row = {'name': record['name'], 'loss': float(loss.detach())}
            row.update({key: float(value.detach()) for key, value in components.items()})
            rows.append(row)
    return {'num_sequences': len(rows), 'loss': average(rows), 'rows': rows}


def checkpoint_selection_value(validation, args):
    losses = validation['loss']
    if args.selection_metric == 'weighted_loss':
        return losses.get('loss', float('inf'))
    if args.selection_metric == 'pl_physical':
        return losses.get('pRB', 0.0) + losses.get('gR1', 0.0)
    if args.selection_metric == 'pl_root_physical':
        return losses.get('pRB', 0.0) + losses.get('gR1', 0.0) + losses.get('root_vel', 0.0)
    if args.selection_metric == 'control_physical':
        return losses.get('gt_control_pRB', 0.0) + losses.get('gt_control_gR1', 0.0)
    if args.selection_metric == 'control_root_physical':
        return (
            losses.get('gt_control_pRB', 0.0)
            + losses.get('gt_control_gR1', 0.0)
            + losses.get('gt_control_root_vel', 0.0)
        )
    if args.selection_metric == 'pl_and_control_physical':
        return (
            losses.get('pRB', 0.0)
            + losses.get('gR1', 0.0)
            + losses.get('gt_control_pRB', 0.0)
            + losses.get('gt_control_gR1', 0.0)
        )
    raise ValueError(f'Unsupported selection metric: {args.selection_metric}')


def save_checkpoint(path, model, optimizer, args, epoch, step, val_loss, weights):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'epoch': epoch,
        'step': step,
        'validation_loss': val_loss,
        'weights': weights,
        'model_type': 'newpl_root_v1',
        'output_contract': 'pRB[15]+gR1[3]+root_vel[3]',
        'root_vel_frame': 'root/body frame; finite-difference world translation right-multiplied by pose[:,0]',
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train newpl_root_v1 module-level PL-root output.')
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--dataset', choices=('amass', 'totalcapture', 'dip'), required=True)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='official')
    parser.add_argument('--root-vel-mode', choices=('gt', 'none', 'smooth_only'), default='gt')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--condition-scale', type=float, default=1.0)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--freeze-root-head', action='store_true')
    parser.add_argument('--allow-zero-offset-init', action='store_true')
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1, help='Number of fixed-length windows per optimizer step.')
    parser.add_argument(
        '--selection-metric',
        choices=(
            'control_physical',
            'control_root_physical',
            'pl_and_control_physical',
            'pl_physical',
            'pl_root_physical',
            'weighted_loss',
        ),
        default='control_physical',
        help='Metric used for best_loss.pt. control_physical=GT fitted control pRB+gR1; decoded PL metrics are still reported.',
    )
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0)
    parser.add_argument('--early-stop-patience', type=int, default=0)
    parser.add_argument('--dt', type=float, default=1.0 / 60.0)
    parser.add_argument('--pRB-weight', type=float, default=None)
    parser.add_argument('--gR1-weight', type=float, default=None)
    parser.add_argument('--root-vel-weight', type=float, default=None)
    parser.add_argument('--pRB-dot-weight', type=float, default=None)
    parser.add_argument('--gR1-dot-weight', type=float, default=None)
    parser.add_argument('--root-vel-smooth-weight', type=float, default=None)
    parser.add_argument('--gt-control-pRB-weight', type=float, default=None)
    parser.add_argument('--gt-control-gR1-weight', type=float, default=None)
    parser.add_argument('--gt-control-root-vel-weight', type=float, default=None)
    args = parser.parse_args()

    if args.dataset == 'dip' and args.root_vel_mode == 'gt':
        raise RuntimeError('DIP root_vel GT is not allowed; use --root-vel-mode none or smooth_only.')

    weights = newpl_root_weights()
    overrides = {
        'pRB': args.pRB_weight,
        'gR1': args.gR1_weight,
        'root_vel': args.root_vel_weight,
        'pRB_dot': args.pRB_dot_weight,
        'gR1_dot': args.gR1_dot_weight,
        'root_vel_smooth': args.root_vel_smooth_weight,
        'gt_control_pRB': args.gt_control_pRB_weight,
        'gt_control_gR1': args.gt_control_gR1_weight,
        'gt_control_root_vel': args.gt_control_root_vel_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            weights[key] = value
    if args.root_vel_mode in ('none', 'smooth_only'):
        weights['root_vel'] = 0.0
    if args.root_vel_mode == 'none':
        weights['root_vel_smooth'] = 0.0

    train_records, train_manifest = load_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, max_sequences=args.max_val_sequences)
    if args.batch_size > 1:
        train_records = [
            record for record in train_records
            if (record['pose_gt'].shape[0] if 'pose_gt' in record else record['q75_gt'].shape[0]) >= args.window
        ]
        if not train_records:
            raise RuntimeError(f'No training sequence has at least window={args.window} frames.')
    gpnet = GPNet().eval().to(DEVICE)
    for parameter in gpnet.parameters():
        parameter.requires_grad_(False)
    body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    model = NewPLRootModule(
        init_size=36,
        hidden_size=args.hidden_size,
        tail_update=args.tail_length,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
        condition_scale=args.condition_scale,
    ).to(DEVICE)
    init_checkpoint_load = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        init_checkpoint_load = load_partial_pl_checkpoint(model, checkpoint['model_state_dict'])
    freeze_handles = freeze_root_head_gradients(model) if args.freeze_root_head else []
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    log_path = output_dir / 'train_log.jsonl'
    best_loss = float('inf')
    best_epoch = 0
    step = 0
    stale_epochs = 0
    stopped_early = False
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_rows = []
        if args.batch_size > 1:
            iterable = range(0, len(train_records), args.batch_size)
        else:
            iterable = range(len(train_records))
        for seq_idx, batch_start in enumerate(iterable, start=1):
            step += 1
            if args.batch_size > 1:
                batch_records = train_records[batch_start:batch_start + args.batch_size]
                starts = []
                for offset, source_record in enumerate(batch_records):
                    seq_len = source_record['pose_gt'].shape[0] if 'pose_gt' in source_record else source_record['q75_gt'].shape[0]
                    max_start = max(0, seq_len - args.window)
                    starts.append((step + offset) % (max_start + 1) if max_start > 0 else 0)
                record = make_batch(batch_records, starts, args.window)
            else:
                source_record = train_records[batch_start]
                seq_len = source_record['pose_gt'].shape[0] if 'pose_gt' in source_record else source_record['q75_gt'].shape[0]
                max_start = max(0, seq_len - args.window)
                start = step % (max_start + 1) if max_start > 0 else 0
                record = slice_record(source_record, start, args.window)
            loss, components = run_sequence(model, gpnet, record, body_model, weights, args)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at {record["name"]}.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            row = {'epoch': epoch, 'step': step, 'seq_idx': seq_idx, 'seq_name': record['name'], 'loss': float(loss.detach())}
            row.update({key: float(value.detach()) for key, value in components.items()})
            train_rows.append(row)
        train_loss = average(train_rows)
        validation = eval_loss(model, gpnet, val_records, body_model, weights, args)
        weighted_val_loss = validation['loss'].get('loss', float('inf'))
        val_loss = checkpoint_selection_value(validation, args)
        improved = (val_loss < best_loss) if best_loss == float('inf') else ((best_loss - val_loss) > args.early_stop_min_delta)
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(output_dir / 'best_loss.pt', model, optimizer, args, epoch, step, val_loss, weights)
        else:
            stale_epochs += 1
        save_checkpoint(output_dir / 'last.pt', model, optimizer, args, epoch, step, val_loss, weights)
        epoch_row = {
            'epoch': epoch,
            'step': step,
            'train_loss': train_loss,
            'validation': validation,
            'weighted_val_loss': weighted_val_loss,
            'selection_metric': args.selection_metric,
            'selection_value': val_loss,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'improved': improved,
            'stale_epochs': stale_epochs,
        }
        history.append(epoch_row)
        with log_path.open('a') as f:
            f.write(json.dumps(epoch_row) + '\n')
        print(json.dumps({'epoch': epoch, 'train_loss': train_loss.get('loss'), 'val_loss': val_loss, 'best_loss': best_loss}, indent=2))
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            stopped_early = True
            break
    for handle in freeze_handles:
        handle.remove()
    result = {
        'experiment_name': args.experiment_name,
        'status': 'early_stopped' if stopped_early else 'ok',
        'config': vars(args),
        'weights': weights,
        'train_cache_manifest': train_manifest,
        'val_cache_manifest': val_manifest,
        'num_train_sequences': len(train_records),
        'num_val_sequences': len(val_records),
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'init_checkpoint_load': init_checkpoint_load,
        'history': history,
    }
    (output_dir / 'train_result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
