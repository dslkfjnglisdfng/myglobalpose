import argparse
import json
import random
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files
from pl_curve import build_pl_curve_model, normalize_gravity, pl_curve_loss
from pl_curve_train import load_partial_checkpoint
from pl_next_control_cache import _tail_history


CURRENT_LOSS_KEYS = (
    'pRB',
    'gR1',
    'baseline_pRB',
    'baseline_gR1',
    'control_point_prior',
    'tail_update_prior',
    'pRB_dot',
    'pRB_ddot',
    'pRB_ddot_smooth',
    'gR1_dot',
    'gR1_ddot',
    'gR_smooth',
    'gt_control_pRB',
    'gt_control_gR1',
)


def default_weights():
    return {
        'pRB': 1.0,
        'gR1': 1.0,
        'baseline_pRB': 0.0,
        'baseline_gR1': 0.0,
        'gt_control_pRB': 0.3,
        'gt_control_gR1': 0.1,
        'control_point_prior': 0.0,
        'tail_update_prior': 0.005,
        'pRB_dot': 0.03,
        'pRB_ddot': 0.0,
        'pRB_ddot_smooth': 1e-6,
        'gR1_dot': 0.03,
        'gR1_ddot': 0.001,
        'gR_smooth': 0.0,
        'next_pRB': 1.0,
        'next_gR1': 1.0,
        'next_gt_control_pRB': 0.3,
        'next_gt_control_gR1': 0.1,
        'next_pRB_vel': 0.03,
        'next_pRB_acc': 0.0003,
        'next_gR1_vel': 0.03,
        'next_gR1_acc': 0.001,
        'next_control_delta_prior': 0.01,
        'next_tail4_control_pRB': 0.15,
        'next_tail4_control_gR1': 0.05,
        'last_control_pRB': 0.3,
        'last_control_gR1': 0.1,
    }


def average(rows):
    totals = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / max(1, len(values)) for key, values in totals.items()}


def load_next_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    supported = {'pl_next_control_cache_v1', 'pl_next_control_cache_v2'}
    if manifest is None or manifest.get('type') not in supported:
        raise RuntimeError(f'Expected manifest type in {sorted(supported)}, got {manifest.get("type") if manifest else None}.')
    records = []
    required = (
        'name',
        'pl_input',
        'pl_target',
        'pl_target_next',
        'valid_next_mask',
        'pl_base',
        'pl_init_feature',
        'pl_target_control',
        'pl_target_control_next',
        'gt_pldot_next',
        'gt_plddot_next',
    )
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f'{cache_file} missing fields: {missing}')
        for seq_idx, name in enumerate(data['name']):
            record = {key: data[key][seq_idx].float() for key in required if key != 'name' and key != 'valid_next_mask'}
            record['name'] = str(name)
            record['valid_next_mask'] = data['valid_next_mask'][seq_idx].bool()
            if 'tail_control_target' in data and 'tail_control_valid_mask' in data:
                record['tail_control_target'] = data['tail_control_target'][seq_idx].float()
                record['tail_control_valid_mask'] = data['tail_control_valid_mask'][seq_idx].bool()
            else:
                tail_target, tail_mask = _tail_history(record['pl_target_control'], tail_size=4)
                record['tail_control_target'] = tail_target
                record['tail_control_valid_mask'] = tail_mask
            if 'last_control_target' in data:
                record['last_control_target'] = data['last_control_target'][seq_idx].float()
            else:
                record['last_control_target'] = record['pl_target_control']
            for key in ('gt_pldot', 'gt_plddot', 'baseline_fd_vel', 'baseline_fd_acc', 'pl_base_next'):
                if key in data:
                    record[key] = data[key][seq_idx].float()
            for key in ('source_name', 'pair_id', 'view_type'):
                if key in data and len(data[key]) > seq_idx:
                    record[key] = data[key][seq_idx]
            records.append(record)
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def slice_or_pad_time(value, start, length):
    chunk = value[start:start + length]
    if length <= 0 or chunk.shape[0] >= length:
        return chunk
    if chunk.shape[0] == 0:
        raise RuntimeError('Cannot pad an empty time slice.')
    pad = chunk[-1:].expand((length - chunk.shape[0],) + chunk.shape[1:])
    return torch.cat((chunk, pad), dim=0)


def make_batch(records, starts, length):
    tensor_keys = (
        'pl_input',
        'pl_target',
        'pl_target_next',
        'valid_next_mask',
        'pl_base',
        'pl_target_control',
        'pl_target_control_next',
        'tail_control_target',
        'tail_control_valid_mask',
        'last_control_target',
        'gt_pldot_next',
        'gt_plddot_next',
    )
    out = {'name': [], 'records': list(records)}
    for key in tensor_keys:
        chunks = []
        for record, start in zip(records, starts):
            seq_len = record['pl_input'].shape[0]
            start = min(max(0, int(start)), max(0, seq_len - length))
            chunks.append(slice_or_pad_time(record[key], start, length))
        out[key] = torch.stack(chunks, dim=1)
    out['pl_init_feature'] = torch.stack([record['pl_init_feature'] for record in records], dim=0)
    for record, start in zip(records, starts):
        out['name'].append(f'{record["name"]}[{start}:{start + length}]')
    out['name'] = '|'.join(out['name'])
    return out


def angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target, avoid_nan=True)
    cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def _masked_values(x, mask):
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return x.masked_select(mask.expand_as(x))


def masked_smooth_l1(pred, target, mask):
    values = _masked_values(pred - target.to(pred.device, pred.dtype), mask.to(pred.device))
    if values.numel() == 0:
        return pred.new_zeros(())
    return torch.nn.functional.smooth_l1_loss(values, torch.zeros_like(values))


def masked_direction_loss(pred, target, mask):
    pred_n = art.math.normalize_tensor(pred, avoid_nan=True)
    target_n = art.math.normalize_tensor(target.to(pred.device, pred.dtype), avoid_nan=True)
    cos = (pred_n * target_n).sum(dim=-1).clamp(-1.0, 1.0)
    values = (1.0 - cos).masked_select(mask.to(pred.device))
    if values.numel() == 0:
        return pred.new_zeros(())
    return values.mean()


def next_control_loss(output, batch, weights):
    if 'next_pl' not in output:
        raise RuntimeError('Model output missing next_pl; use --model-variant newpl_v6_next_control.')
    mask = batch['valid_next_mask'].to(output['next_pl'].device)
    target_next = batch['pl_target_next'].to(output['next_pl'].device, output['next_pl'].dtype)
    control_next = batch['pl_target_control_next'].to(output['next_pl'].device, output['next_pl'].dtype)
    tail_target = batch['tail_control_target'].to(output['next_pl'].device, output['next_pl'].dtype)
    tail_mask = batch['tail_control_valid_mask'].to(output['next_pl'].device)
    last_target = batch['last_control_target'].to(output['next_pl'].device, output['next_pl'].dtype)
    full_mask = torch.ones_like(mask, dtype=torch.bool)
    gt_dot_next = batch['gt_pldot_next'].to(output['next_pl'].device, output['next_pl'].dtype)
    gt_ddot_next = batch['gt_plddot_next'].to(output['next_pl'].device, output['next_pl'].dtype)
    pred_next_g = art.math.normalize_tensor(output['next_pl'][..., 15:], avoid_nan=True)
    control_next_g = art.math.normalize_tensor(output['next_control'][..., 15:], avoid_nan=True)
    tail_control_g = art.math.normalize_tensor(output['next_tail_control'][..., 15:], avoid_nan=True)
    last_control_g = art.math.normalize_tensor(output['last_preview_control'][..., 15:], avoid_nan=True)
    target_next_g = art.math.normalize_tensor(target_next[..., 15:], avoid_nan=True)
    target_control_next_g = art.math.normalize_tensor(control_next[..., 15:], avoid_nan=True)
    tail_target_g = art.math.normalize_tensor(tail_target[..., 15:], avoid_nan=True)
    last_target_g = art.math.normalize_tensor(last_target[..., 15:], avoid_nan=True)
    losses = {
        'next_pRB': masked_smooth_l1(output['next_pl'][..., :15], target_next[..., :15], mask),
        'next_gR1': masked_direction_loss(pred_next_g, target_next_g, mask),
        'next_gt_control_pRB': masked_smooth_l1(output['next_control'][..., :15], control_next[..., :15], mask),
        'next_gt_control_gR1': masked_smooth_l1(control_next_g, target_control_next_g, mask),
        'next_tail4_control_pRB': masked_smooth_l1(output['next_tail_control'][..., :15], tail_target[..., :15], tail_mask),
        'next_tail4_control_gR1': masked_smooth_l1(tail_control_g, tail_target_g, tail_mask),
        'last_control_pRB': masked_smooth_l1(output['last_preview_control'][..., :15], last_target[..., :15], full_mask),
        'last_control_gR1': masked_smooth_l1(last_control_g, last_target_g, full_mask),
        'next_pRB_vel': masked_smooth_l1(output['next_pldot'][..., :15], gt_dot_next[..., :15], mask),
        'next_pRB_acc': masked_smooth_l1(output['next_plddot'][..., :15], gt_ddot_next[..., :15], mask),
        'next_gR1_vel': masked_smooth_l1(output['next_pldot'][..., 15:], gt_dot_next[..., 15:], mask),
        'next_gR1_acc': masked_smooth_l1(output['next_plddot'][..., 15:], gt_ddot_next[..., 15:], mask),
        'next_control_delta_prior': output.get('next_control_delta_norm', output['next_pl'].new_zeros(())),
    }
    total = output['next_pl'].new_zeros(())
    for key in losses:
        total = total + losses[key] * float(weights.get(key, 0.0))
    return total, losses


def _masked_mean_tensor(values, mask=None):
    if mask is None:
        return values.mean() if values.numel() else values.new_zeros(())
    selected = values.masked_select(mask.to(values.device).expand_as(values))
    if selected.numel() == 0:
        return values.new_zeros(())
    return selected.mean()


def _control_l2_cm(pred, target, mask=None):
    leaf = (pred[..., :15] - target[..., :15]).reshape(pred.shape[:-1] + (5, 3)).norm(dim=-1)
    if mask is None:
        return leaf.mean() * 100.0
    while mask.dim() < leaf.dim():
        mask = mask.unsqueeze(-1)
    return _masked_mean_tensor(leaf, mask) * 100.0


def _control_angle_mean(pred, target, mask=None):
    values = angle_deg(pred[..., 15:], target[..., 15:])
    if mask is None:
        return values.mean()
    return _masked_mean_tensor(values, mask)


def control_metric_summary(output, batch):
    device, dtype = output['pl'].device, output['pl'].dtype
    current_target = batch['pl_target_control'].to(device, dtype)
    next_target = batch['pl_target_control_next'].to(device, dtype)
    tail_target = batch['tail_control_target'].to(device, dtype)
    last_target = batch['last_control_target'].to(device, dtype)
    next_mask = batch['valid_next_mask'].to(device)
    tail_mask = batch['tail_control_valid_mask'].to(device)

    current_l2 = _control_l2_cm(output['new_control'], current_target)
    current_g = _control_angle_mean(output['new_control'], current_target)
    next_l2 = _control_l2_cm(output['next_control'], next_target, next_mask.unsqueeze(-1))
    next_g = _control_angle_mean(output['next_control'], next_target, next_mask)
    last_l2 = _control_l2_cm(output['last_preview_control'], last_target)
    last_g = _control_angle_mean(output['last_preview_control'], last_target)
    tail_l2 = _control_l2_cm(output['next_tail_control'], tail_target, tail_mask.unsqueeze(-1))
    tail_g = _control_angle_mean(output['next_tail_control'], tail_target, tail_mask)
    score = last_l2 + next_l2 + tail_l2 + 0.1 * (last_g + next_g + tail_g)
    gravity_score = current_g + next_g + last_g + tail_g
    return {
        'current_control_pRB_L2_cm': float(current_l2.detach()),
        'current_control_gR1_angle_deg': float(current_g.detach()),
        'next_control_pRB_L2_cm': float(next_l2.detach()),
        'next_control_gR1_angle_deg': float(next_g.detach()),
        'last_control_pRB_L2_cm': float(last_l2.detach()),
        'last_control_gR1_angle_deg': float(last_g.detach()),
        'tail4_control_pRB_L2_cm': float(tail_l2.detach()),
        'tail4_control_gR1_angle_deg': float(tail_g.detach()),
        'control_module_score': float(score.detach()),
        'gravity_control_score': float(gravity_score.detach()),
    }


def metric_summary(output, batch):
    target = batch['pl_target'].to(output['pl'].device, output['pl'].dtype)
    target_next = batch['pl_target_next'].to(output['pl'].device, output['pl'].dtype)
    mask = batch['valid_next_mask'].to(output['pl'].device)
    current_leaf = (output['pl'][..., :15] - target[..., :15]).reshape(output['pl'].shape[:2] + (5, 3)).norm(dim=-1)
    next_leaf = (output['next_pl'][..., :15] - target_next[..., :15]).reshape(output['pl'].shape[:2] + (5, 3)).norm(dim=-1)
    next_leaf_values = next_leaf.masked_select(mask.unsqueeze(-1).expand_as(next_leaf))
    current_g = angle_deg(output['pl'][..., 15:], target[..., 15:])
    next_g = angle_deg(output['next_pl'][..., 15:], target_next[..., 15:])
    next_g_values = next_g.masked_select(mask)
    gt_dot_next = batch['gt_pldot_next'].to(output['pl'].device, output['pl'].dtype)
    gt_ddot_next = batch['gt_plddot_next'].to(output['pl'].device, output['pl'].dtype)
    vel_leaf = (output['next_pldot'][..., :15] - gt_dot_next[..., :15]).reshape(output['pl'].shape[:2] + (5, 3)).norm(dim=-1)
    acc_leaf = (output['next_plddot'][..., :15] - gt_ddot_next[..., :15]).reshape(output['pl'].shape[:2] + (5, 3)).norm(dim=-1)
    vel_values = vel_leaf.masked_select(mask.unsqueeze(-1).expand_as(vel_leaf))
    acc_values = acc_leaf.masked_select(mask.unsqueeze(-1).expand_as(acc_leaf))
    return {
        'current_pRB_L2_cm': float(current_leaf.mean().detach() * 100.0),
        'current_gR1_angle_deg': float(current_g.mean().detach()),
        'current_module_score': float((current_leaf.mean() * 100.0 + 0.1 * current_g.mean()).detach()),
        'next_pRB_L2_cm': float(next_leaf_values.mean().detach() * 100.0) if next_leaf_values.numel() else 0.0,
        'next_gR1_angle_deg': float(next_g_values.mean().detach()) if next_g_values.numel() else 0.0,
        'next_module_score': float((next_leaf_values.mean() * 100.0 + 0.1 * next_g_values.mean()).detach()) if next_leaf_values.numel() else 0.0,
        'pRB_vel_L2_cm_s': float(vel_values.mean().detach() * 100.0) if vel_values.numel() else 0.0,
        'pRB_acc_L2_cm_s2': float(acc_values.mean().detach() * 100.0) if acc_values.numel() else 0.0,
        'dynamics_score': float((vel_values.mean() * 100.0 + 0.01 * acc_values.mean() * 100.0).detach()) if vel_values.numel() else 0.0,
    }


def run_batch(model, batch, weights):
    features = batch['pl_input'].to(DEVICE)
    base = normalize_gravity(batch['pl_base'].float()).to(DEVICE)
    target = normalize_gravity(batch['pl_target'].float()).to(DEVICE)
    init_feature = batch['pl_init_feature'].to(DEVICE)
    output = model.forward_sequence(features, base, init_feature=init_feature)
    current_loss, current_components = pl_curve_loss(
        output,
        target,
        {key: weights[key] for key in CURRENT_LOSS_KEYS},
        dt=model.dt,
        target_control=batch['pl_target_control'].to(DEVICE),
    )
    next_loss, next_components = next_control_loss(output, batch, weights)
    loss = current_loss + next_loss
    components = {'loss': loss.detach(), 'current_loss': current_loss.detach(), 'next_loss': next_loss.detach()}
    components.update({key: value.detach() for key, value in current_components.items()})
    components.update({key: value.detach() for key, value in next_components.items()})
    components.update({
        'new_delta_norm': output.get('new_delta_norm', loss.new_zeros(())).detach(),
        'next_control_delta_norm': output.get('next_control_delta_norm', loss.new_zeros(())).detach(),
        'next_tail_delta_norm': output.get('next_tail_delta_norm', loss.new_zeros(())).detach(),
    })
    metrics = metric_summary(output, batch)
    metrics.update(control_metric_summary(output, batch))
    return loss, components, metrics


@torch.no_grad()
def validate(model, records, weights, max_sequences=0, window=0, seed=0, keep_rows=False, batch_size=1):
    model.eval()
    selected = records[:max_sequences] if max_sequences else records
    if window and window > 0:
        selected = [record for record in selected if record['pl_input'].shape[0] >= int(window)]
    rows = []
    batch_size = max(1, int(batch_size))
    for batch_start in range(0, len(selected), batch_size):
        batch_records = selected[batch_start:batch_start + batch_size]
        if not batch_records:
            continue
        if window and window > 0:
            length = int(window)
        else:
            length = min(record['pl_input'].shape[0] for record in batch_records)
        starts = []
        for offset, record in enumerate(batch_records):
            idx = batch_start + offset
            max_start = max(0, record['pl_input'].shape[0] - length)
            starts.append(((int(seed) + idx * 997) % (max_start + 1)) if max_start > 0 else 0)
        batch = make_batch(batch_records, starts, length)
        loss, components, metrics = run_batch(model, batch, weights)
        row = {'name': batch['name'], **{key: float(value) for key, value in metrics.items()}}
        row.update({key: float(value.detach()) for key, value in components.items()})
        rows.append(row)
    result = {'num_sequences': len(rows), 'loss': average(rows)}
    if keep_rows:
        result['rows'] = rows
    return result


def save_checkpoint(path, model, optimizer, args, epoch, step, selection_values, weights, validation):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'epoch': epoch,
        'step': step,
        'selection_values': selection_values,
        'validation': validation,
        'weights': weights,
        'model_type': 'pl_curve_v1',
        'model_variant': 'newpl_v6_next_control',
        'output_contract': {
            'current_pl': 'pRB_t[15] + gR1_t[3] = 18D; compatible with official IK1',
            'next_pl': 'pRB_{t+1}[15] + gR1_{t+1}[3] = 18D auxiliary output',
            'next_derivatives': 'next_pldot/next_plddot decoded from predicted next control via UniformCubicBSpline',
        },
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train NewPL v6 next-control module on precomputed module cache.')
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--window', type=int, default=81)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--next-residual-scale', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--init-size', type=int, default=36)
    parser.add_argument('--input-size', type=int, default=84)
    parser.add_argument('--model-variant', default='newpl_v6_next_control')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--val-window-length', type=int, default=512)
    parser.add_argument('--val-batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0)
    parser.add_argument('--early-stop-patience', type=int, default=0)
    for key in default_weights():
        parser.add_argument(f'--{key.replace("_", "-")}-weight', type=float, default=None)
    args = parser.parse_args()
    if args.model_variant != 'newpl_v6_next_control':
        raise RuntimeError('pl_next_control_train.py only trains --model-variant newpl_v6_next_control.')
    if args.window <= 0:
        raise RuntimeError(f'--window must be positive; got {args.window}.')
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    weights = default_weights()
    for key in list(weights):
        value = getattr(args, f'{key}_weight')
        if value is not None:
            weights[key] = value

    train_records, train_manifest = load_next_records(args.train_cache, args.max_train_sequences)
    val_records, val_manifest = load_next_records(args.val_cache, args.max_val_sequences)
    if not train_records:
        raise RuntimeError('No train records loaded.')
    if not val_records:
        raise RuntimeError('No val records loaded.')
    train_records = [record for record in train_records if record['pl_input'].shape[0] >= args.window]
    if not train_records:
        raise RuntimeError(f'No training sequence has at least window={args.window} frames.')

    model = build_pl_curve_model(vars(args)).to(DEVICE)
    init_checkpoint_load = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        if checkpoint.get('model_type') != 'pl_curve_v1':
            raise RuntimeError(f'Unsupported init checkpoint model_type={checkpoint.get("model_type")}.')
        init_checkpoint_load = load_partial_checkpoint(model, checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')

    best = {
        'best_total_loss.pt': float('inf'),
        'best_current_module_metric.pt': float('inf'),
        'best_next_module_metric.pt': float('inf'),
        'best_dynamics_metric.pt': float('inf'),
        'best_control_metric.pt': float('inf'),
        'best_current_gR1.pt': float('inf'),
        'best_next_gR1.pt': float('inf'),
        'best_gravity_control.pt': float('inf'),
    }
    best_epochs = {key: 0 for key in best}
    stale_epochs = 0
    history = []
    step = 0
    log_path = output_dir / 'train_log.jsonl'
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_records)))
        random.shuffle(order)
        rows = []
        for batch_start in range(0, len(order), args.batch_size):
            ids = order[batch_start:batch_start + args.batch_size]
            batch_records = [train_records[i] for i in ids]
            starts = []
            for offset, record in enumerate(batch_records):
                max_start = max(0, record['pl_input'].shape[0] - args.window)
                starts.append(random.randint(0, max_start) if max_start > 0 else 0)
            batch = make_batch(batch_records, starts, args.window)
            loss, components, metrics = run_batch(model, batch, weights)
            if not torch.isfinite(loss):
                finite = {key: bool(torch.isfinite(value).all()) for key, value in components.items()}
                raise RuntimeError(f'Non-finite loss at epoch={epoch} step={step}; components_finite={finite}.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            row = {'epoch': epoch, 'step': step, 'loss': float(loss.detach()), **metrics}
            row.update({key: float(value.detach()) for key, value in components.items()})
            rows.append(row)

        train_loss = average(rows)
        validation = validate(
            model,
            val_records,
            weights,
            max_sequences=args.max_val_sequences,
            window=args.val_window_length,
            seed=epoch,
            batch_size=args.val_batch_size,
        )
        val = validation['loss']
        selection_values = {
            'total_loss': val.get('loss', float('inf')),
            'current_module_metric': val.get('current_module_score', float('inf')),
            'next_module_metric': val.get('next_module_score', float('inf')),
            'dynamics_metric': val.get('dynamics_score', float('inf')),
            'control_metric': val.get('control_module_score', float('inf')),
            'current_gR1_metric': val.get('current_gR1_angle_deg', float('inf')),
            'next_gR1_metric': val.get('next_gR1_angle_deg', float('inf')),
            'gravity_control_metric': val.get('gravity_control_score', float('inf')),
        }
        improved_any = False
        mapping = {
            'best_total_loss.pt': 'total_loss',
            'best_current_module_metric.pt': 'current_module_metric',
            'best_next_module_metric.pt': 'next_module_metric',
            'best_dynamics_metric.pt': 'dynamics_metric',
            'best_control_metric.pt': 'control_metric',
            'best_current_gR1.pt': 'current_gR1_metric',
            'best_next_gR1.pt': 'next_gR1_metric',
            'best_gravity_control.pt': 'gravity_control_metric',
        }
        for ckpt_name, metric_key in mapping.items():
            value = selection_values[metric_key]
            if value < best[ckpt_name] - float(args.early_stop_min_delta):
                best[ckpt_name] = value
                best_epochs[ckpt_name] = epoch
                improved_any = True
                save_checkpoint(output_dir / ckpt_name, model, optimizer, args, epoch, step, selection_values, weights, validation)
        save_checkpoint(output_dir / 'last.pt', model, optimizer, args, epoch, step, selection_values, weights, validation)
        stale_epochs = 0 if improved_any else stale_epochs + 1
        epoch_row = {
            'epoch': epoch,
            'step': step,
            'train_loss': train_loss,
            'validation': validation,
            'selection_values': selection_values,
            'best': best,
            'best_epochs': best_epochs,
            'improved_any': improved_any,
            'stale_epochs': stale_epochs,
        }
        history.append(epoch_row)
        with log_path.open('a') as f:
            f.write(json.dumps(epoch_row) + '\n')
        print(json.dumps({
            'epoch': epoch,
            'train_loss': train_loss.get('loss'),
            'total_loss': selection_values['total_loss'],
            'current_module_metric': selection_values['current_module_metric'],
            'next_module_metric': selection_values['next_module_metric'],
            'dynamics_metric': selection_values['dynamics_metric'],
            'control_metric': selection_values['control_metric'],
            'current_gR1_metric': selection_values['current_gR1_metric'],
            'next_gR1_metric': selection_values['next_gR1_metric'],
            'gravity_control_metric': selection_values['gravity_control_metric'],
            'stale_epochs': stale_epochs,
        }, indent=2))
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            break

    result = {
        'experiment_name': args.experiment_name,
        'status': 'ok',
        'config': vars(args),
        'weights': weights,
        'train_cache_manifest': train_manifest,
        'val_cache_manifest': val_manifest,
        'num_train_sequences': len(train_records),
        'num_val_sequences': len(val_records),
        'best': best,
        'best_epochs': best_epochs,
        'init_checkpoint_load': init_checkpoint_load,
        'history': history,
    }
    (output_dir / 'train_result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
