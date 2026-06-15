import argparse
import json
import math
import random
import shlex
import sys
from pathlib import Path

import torch

from l4_train_diverse_short import DEVICE, load_cache_files
from newik1_control_point import NewIK1ControlPointModule, newik1_loss


def default_weights():
    return {
        'pRJ': 1.0,
        'leaf_pRJ': 0.0,
        'gR2': 1.0,
        'pRJ_dot': 0.03,
        'pRJ_ddot': 0.001,
        'gR2_dot': 0.03,
        'gR2_ddot': 0.001,
        'control_pRJ': 0.1,
        'control_gR2': 0.1,
        'control_pRJ_dot': 0.0,
        'control_gR2_dot': 0.0,
        'control_pRJ_ddot': 0.0,
        'control_gR2_ddot': 0.0,
        'gt_control_pRJ': 0.0,
        'gt_control_gR2': 0.0,
        'gt_control_leaf_pRJ': 0.0,
        'bone_length': 0.5,
        'control_point_prior': 0.3,
        'tail_update_prior': 0.005,
    }


def load_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is None or manifest.get('type') != 'newik1_control_cache_v1':
        raise RuntimeError(f'Expected newik1_control_cache_v1 manifest, got {manifest.get("type") if manifest else None}.')
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            records.append({
                'name': name,
                'ik1_input': data['ik1_input'][seq_idx].float(),
                'ik1_target': data['ik1_target'][seq_idx].float(),
                'ik1_target_control_tail': data['ik1_target_control_tail'][seq_idx].float(),
                'ik1_base': data['ik1_base'][seq_idx].float(),
            })
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def average(rows):
    totals = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / max(1, len(values)) for key, values in totals.items()}


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def scheduled_lr(epoch, total_epochs, base_lr, min_lr, warmup_epochs):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if total_epochs <= warmup_epochs:
        return base_lr
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + (base_lr - min_lr) * cosine


def make_batch(records, starts, length):
    out = {'name': '|'.join(f"{record['name']}[{int(start)}:{int(start) + length}]" for record, start in zip(records, starts))}
    for key in ('ik1_input', 'ik1_target', 'ik1_target_control_tail', 'ik1_base'):
        vals = []
        for record, start in zip(records, starts):
            seq_len = record['ik1_input'].shape[0]
            start = min(max(0, int(start)), max(0, seq_len - length))
            vals.append(record[key][start:start + length])
        out[key] = torch.stack(vals, dim=1)
    return out


def slice_record(record, start, length):
    seq_len = record['ik1_input'].shape[0]
    if length <= 0 or seq_len <= length:
        return record
    start = min(max(0, int(start)), seq_len - int(length))
    end = start + int(length)
    out = {'name': f"{record['name']}[{start}:{end}]"}
    for key, value in record.items():
        if key == 'name':
            continue
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == seq_len:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def run_sequence(model, record, weights):
    features = record['ik1_input'].to(DEVICE)
    target = record['ik1_target'].to(DEVICE)
    target_tail = record['ik1_target_control_tail'].to(DEVICE)
    base = record['ik1_base'].to(DEVICE)
    output = model.forward_sequence(features, base, init_output=base[0])
    loss, losses = newik1_loss(output, target, target_tail, weights)
    components = {key: value.detach() for key, value in losses.items()}
    components.update({
        'loss': loss.detach(),
        'new_delta_norm': output['new_delta_norm'].detach(),
        'tail_delta_norm': output['tail_delta_norm'].detach(),
        'ik1_residual_norm_mean': (output['ik1'] - output['base']).norm(dim=-1).mean().detach(),
    })
    return loss, components


@torch.no_grad()
def eval_loss(model, records, weights, max_sequences=0, val_window_length=0, val_window_seed=0):
    model.eval()
    rows = []
    selected = records[:max_sequences] if max_sequences else records
    for record_idx, record in enumerate(selected):
        if val_window_length and val_window_length > 0:
            seq_len = record['ik1_input'].shape[0]
            max_start = max(0, seq_len - int(val_window_length))
            start = ((int(val_window_seed) + record_idx * 997) % (max_start + 1)) if max_start > 0 else 0
            record = slice_record(record, start, int(val_window_length))
        loss, components = run_sequence(model, record, weights)
        row = {'name': record['name'], 'loss': float(loss)}
        row.update({key: float(value) for key, value in components.items()})
        rows.append(row)
    return {'num_sequences': len(rows), 'loss': average(rows), 'rows': rows}


def checkpoint_selection_value(validation, args):
    losses = validation.get('loss', {})
    if args.selection_metric == 'weighted_loss':
        return losses.get('loss', float('inf'))
    if args.selection_metric == 'ik1_physical':
        return losses.get('pRJ', 0.0) + losses.get('leaf_pRJ', 0.0) + losses.get('gR2', 0.0)
    if args.selection_metric == 'control_physical':
        return (
            losses.get('control_pRJ', 0.0)
            + losses.get('control_gR2', 0.0)
            + losses.get('gt_control_pRJ', 0.0)
            + losses.get('gt_control_gR2', 0.0)
            + losses.get('gt_control_leaf_pRJ', 0.0)
        )
    if args.selection_metric == 'ik1_control_physical':
        return (
            losses.get('pRJ', 0.0)
            + losses.get('leaf_pRJ', 0.0)
            + losses.get('gR2', 0.0)
            + losses.get('control_pRJ', 0.0)
            + losses.get('control_gR2', 0.0)
            + losses.get('gt_control_pRJ', 0.0)
            + losses.get('gt_control_gR2', 0.0)
            + losses.get('gt_control_leaf_pRJ', 0.0)
            + losses.get('bone_length', 0.0)
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
        'model_type': 'newik1_control_point_v1',
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train NewIK1_ControlPoint_v1 on precomputed teacher-forced or PL1 streaming caches.')
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--min-lr', type=float, default=1e-7)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--early-stop-patience', type=int, default=0)
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--input-size', type=int, default=0)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--output-mode', choices=('full', 'pRJ_only'), default='full')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--val-window-length', type=int, default=0, help='If >0, validate one deterministic rotating window per val sequence.')
    parser.add_argument(
        '--selection-metric',
        choices=('weighted_loss', 'ik1_physical', 'control_physical', 'ik1_control_physical'),
        default='weighted_loss',
        help='Metric used for best_loss.pt. Use ik1_control_physical for control-only IK1 experiments.',
    )
    for key, value in default_weights().items():
        parser.add_argument(f'--{key.replace("_", "-")}-weight', type=float, default=None)
    args = parser.parse_args()

    weights = default_weights()
    for key in list(weights):
        value = getattr(args, f'{key}_weight')
        if value is not None:
            weights[key] = value

    train_records, train_manifest = load_records(args.train_cache, args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, args.max_val_sequences)
    input_size = int(args.input_size or train_manifest.get('input_size', train_records[0]['ik1_input'].shape[-1]))
    if any(record['ik1_input'].shape[-1] != input_size for record in train_records + val_records):
        raise RuntimeError(f'Cache input size mismatch; expected {input_size}.')
    if args.batch_size > 1:
        train_records = [record for record in train_records if record['ik1_input'].shape[0] >= args.window]
        if not train_records:
            raise RuntimeError(f'No training sequence has at least window={args.window} frames.')

    args.input_size = input_size
    args.feature_mode = train_manifest.get('feature_mode', '')
    args.ik1_backend = train_manifest.get('ik1_backend', 'control_point_last_v1' if input_size == 63 else 'control_point_v1')
    model = NewIK1ControlPointModule(
        input_size=input_size,
        hidden_size=args.hidden_size,
        tail_update=args.tail_length,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
        output_mode=args.output_mode,
    ).to(DEVICE)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        if checkpoint.get('model_type') != 'newik1_control_point_v1':
            raise RuntimeError(f'Unsupported init checkpoint model_type={checkpoint.get("model_type")}')
        cfg = checkpoint.get('config', {})
        ckpt_input_size = int(cfg.get('input_size', 120))
        if ckpt_input_size != input_size:
            raise RuntimeError(f'Init checkpoint input_size={ckpt_input_size} does not match cache input_size={input_size}.')
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    best_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    step = 0
    history = []
    log_path = output_dir / 'train_log.jsonl'
    for epoch in range(1, args.epochs + 1):
        current_lr = scheduled_lr(epoch, args.epochs, args.lr, args.min_lr, args.warmup_epochs)
        set_lr(optimizer, current_lr)
        model.train()
        rows = []
        if args.batch_size > 1:
            order = list(range(max(1, len(train_records))))
            random.shuffle(order)
            for batch_start in range(0, len(order), args.batch_size):
                ids = order[batch_start:batch_start + args.batch_size]
                recs = [train_records[i] for i in ids]
                starts = [
                    random.randint(0, max(0, rec['ik1_input'].shape[0] - args.window))
                    for rec in recs
                ]
                batch = make_batch(recs, starts, args.window)
                optimizer.zero_grad(set_to_none=True)
                loss, comps = run_sequence(model, batch, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                rows.append({key: float(value) for key, value in comps.items()})
        else:
            order = list(range(len(train_records)))
            random.shuffle(order)
            for idx in order:
                optimizer.zero_grad(set_to_none=True)
                loss, comps = run_sequence(model, train_records[idx], weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                rows.append({key: float(value) for key, value in comps.items()})
        train_loss = average(rows)
        val = eval_loss(
            model,
            val_records,
            weights,
            val_window_length=args.val_window_length,
            val_window_seed=epoch,
        )
        weighted_val_loss = float(val['loss'].get('loss', float('inf')))
        val_scalar = float(checkpoint_selection_value(val, args))
        improved = val_scalar < best_loss - args.early_stop_min_delta
        if improved:
            best_loss = val_scalar
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / 'best_loss.pt', model, optimizer, args, epoch, step, val['loss'], weights)
        else:
            epochs_without_improvement += 1
        save_checkpoint(output_dir / 'last.pt', model, optimizer, args, epoch, step, val['loss'], weights)
        row = {
            'epoch': epoch,
            'lr': current_lr,
            'train': train_loss,
            'validation': val['loss'],
            'weighted_val_loss': weighted_val_loss,
            'selection_metric': args.selection_metric,
            'selection_value': val_scalar,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'epochs_without_improvement': epochs_without_improvement,
        }
        history.append(row)
        with log_path.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            break
    result = {
        'status': 'ok',
        'experiment_name': args.experiment_name,
        'config': vars(args),
        'weights': weights,
        'train_manifest': train_manifest,
        'val_manifest': val_manifest,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'selection_metric': args.selection_metric,
        'stopped_early': args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience,
        'history': history,
        'checkpoints': {
            'best_loss': str(output_dir / 'best_loss.pt'),
            'last': str(output_dir / 'last.pt'),
        },
    }
    (output_dir / 'train_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'ok', 'best_epoch': best_epoch, 'best_loss': best_loss}, indent=2))


if __name__ == '__main__':
    main()
