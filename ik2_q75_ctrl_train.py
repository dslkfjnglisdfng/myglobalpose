import argparse
import json
import random
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from ik2_q75_ctrl import (
    DEFAULT_FK_VERTEX_MASK,
    IK2Q75ControlModule,
    ik2_q75_loss,
    ik2_q75_weights_for_preset,
    selection_value,
)
from l4_train_diverse_short import DEVICE, load_cache_files


REQUIRED_KEYS = (
    'ik2_q75_input',
    'q75_gt',
    'pose_gt',
    'baseline_root_pose',
    'teacher_pose',
)
OPTIONAL_KEYS = (
    'tran_gt',
    'baseline_gR2',
    'offset_r',
    'aM',
    'wM',
    'RMB',
)


def load_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is not None and manifest.get('type') != 'ik2_q75_ctrl_cache_v1':
        raise RuntimeError(f'Expected ik2_q75_ctrl_cache_v1 manifest, got {manifest.get("type")}.')
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f'{cache_file} missing fields: {missing}')
        for seq_idx, name in enumerate(data['name']):
            record = {'name': name}
            for key in REQUIRED_KEYS:
                record[key] = data[key][seq_idx].float()
            for key in OPTIONAL_KEYS:
                if key in data and data[key]:
                    record[key] = data[key][seq_idx].float()
            records.append(record)
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def make_batch(records, starts, length):
    out = {'name': '|'.join(f"{record['name']}[{start}:{start + length}]" for record, start in zip(records, starts))}
    seq_lens = [record['ik2_q75_input'].shape[0] for record in records]
    for key in REQUIRED_KEYS:
        vals = []
        for record, start, seq_len in zip(records, starts, seq_lens):
            start = min(max(0, int(start)), max(0, seq_len - length))
            vals.append(record[key][start:start + length])
        out[key] = torch.stack(vals, dim=1)
    for key in OPTIONAL_KEYS:
        if key == 'offset_r':
            if all(key in record and torch.is_tensor(record[key]) and record[key].numel() == 18 for record in records):
                out[key] = torch.stack([record[key] for record in records], dim=0)
            continue
        if all(key in record for record in records):
            vals = []
            for record, start, seq_len in zip(records, starts, seq_lens):
                start = min(max(0, int(start)), max(0, seq_len - length))
                vals.append(record[key][start:start + length])
            out[key] = torch.stack(vals, dim=1)
    return out


def slice_record(record, start, length):
    seq_len = record['ik2_q75_input'].shape[0]
    start = min(max(0, int(start)), max(0, seq_len - length))
    end = start + length
    out = {'name': f"{record['name']}[{start}:{end}]"}
    for key, value in record.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == seq_len:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def to_device(record):
    return {key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in record.items()}


def run_sequence(model, record, weights, body_model):
    record = to_device(record)
    output = model.forward_sequence(record['ik2_q75_input'], offset_r=record.get('offset_r'))
    loss, components = ik2_q75_loss(output, record, weights, body_model, dt=model.dt, euler_seq=model.euler_seq)
    return loss, components, output


def average(rows):
    out = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                out.setdefault(key, []).append(float(value))
    return {key: sum(values) / max(1, len(values)) for key, values in out.items()}


@torch.no_grad()
def eval_loss(model, records, weights, body_model, max_sequences=0, window=0):
    model.eval()
    rows = []
    selected = records[:max_sequences] if max_sequences else records
    for record in selected:
        source = slice_record(record, 0, window) if window else record
        loss, components, output = run_sequence(model, source, weights, body_model)
        row = {'name': source['name'], 'loss': float(loss.detach())}
        row.update({key: float(value.detach()) for key, value in components.items()})
        row['control_shape'] = list(output['control'].shape)
        rows.append(row)
    return {'num_sequences': len(rows), 'loss': average(rows), 'rows': rows}


def scheduled_lr(epoch, total_epochs, base_lr, min_lr, warmup_epochs):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if total_epochs <= warmup_epochs:
        return base_lr
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
    return min_lr + (base_lr - min_lr) * cosine


def save_checkpoint(path, model, optimizer, args, epoch, step, validation, weights):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'epoch': epoch,
        'step': step,
        'validation_loss': validation,
        'weights': weights,
        'model_type': 'ik2_q75_ctrl_v1',
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train IK2 q75 control module.')
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--residual-scale', type=float, default=0.05)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--offset-init-scale', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--val-window', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--early-stop-patience', type=int, default=0)
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0)
    parser.add_argument('--loss-preset', choices=('pos_dominant', 'balanced', 'acc_stronger'), default='pos_dominant')
    parser.add_argument('--selection-metric', choices=('fk_pva', 'weighted_loss'), default='fk_pva')
    for key in ik2_q75_weights_for_preset('pos_dominant'):
        parser.add_argument(f'--{key.replace("_", "-")}-weight', type=float, default=None)
    args = parser.parse_args()

    weights = ik2_q75_weights_for_preset(args.loss_preset)
    for key in list(weights):
        override = getattr(args, f'{key}_weight')
        if override is not None:
            weights[key] = override

    train_records, train_manifest = load_records(args.train_cache, args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, args.max_val_sequences)
    input_size = int(train_manifest.get('input_size', train_records[0]['ik2_q75_input'].shape[-1]) if train_manifest else train_records[0]['ik2_q75_input'].shape[-1])
    if any(record['ik2_q75_input'].shape[-1] != input_size for record in train_records + val_records):
        raise RuntimeError(f'IK2 q75 input size mismatch; expected {input_size}.')
    args.input_size = input_size
    if args.batch_size > 1:
        train_records = [record for record in train_records if record['ik2_q75_input'].shape[0] >= args.window]
        if not train_records:
            raise RuntimeError(f'No training sequence has at least window={args.window} frames.')

    model = IK2Q75ControlModule(
        input_size=input_size,
        hidden_size=args.hidden_size,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
        offset_init_scale=args.offset_init_scale,
    ).to(DEVICE)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        if checkpoint.get('model_type') != 'ik2_q75_ctrl_v1':
            raise RuntimeError(f'Unsupported init checkpoint model_type={checkpoint.get("model_type")}')
        model.load_state_dict(checkpoint['model_state_dict'])
    body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=DEFAULT_FK_VERTEX_MASK, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    (output_dir / 'weights.json').write_text(json.dumps(weights, indent=2) + '\n')
    best_value = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    step = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        lr = scheduled_lr(epoch, args.epochs, args.lr, args.min_lr, args.warmup_epochs)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        rows = []
        if args.batch_size > 1:
            order = list(range(len(train_records)))
            random.shuffle(order)
            for batch_start in range(0, len(order), args.batch_size):
                ids = order[batch_start:batch_start + args.batch_size]
                recs = [train_records[i] for i in ids]
                starts = [random.randint(0, max(0, rec['ik2_q75_input'].shape[0] - args.window)) for rec in recs]
                record = make_batch(recs, starts, args.window)
                optimizer.zero_grad(set_to_none=True)
                loss, components, _output = run_sequence(model, record, weights, body_model)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at {record["name"]}.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                rows.append({key: float(value.detach()) for key, value in components.items()})
        else:
            order = list(range(len(train_records)))
            random.shuffle(order)
            for idx in order:
                source = train_records[idx]
                start = random.randint(0, max(0, source['ik2_q75_input'].shape[0] - args.window))
                record = slice_record(source, start, args.window)
                optimizer.zero_grad(set_to_none=True)
                loss, components, _output = run_sequence(model, record, weights, body_model)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at {record["name"]}.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                rows.append({key: float(value.detach()) for key, value in components.items()})
        train_loss = average(rows)
        validation = eval_loss(model, val_records, weights, body_model, max_sequences=args.max_val_sequences, window=args.val_window)
        current_value = selection_value(validation, args.selection_metric)
        improved = current_value < best_value - args.early_stop_min_delta
        if improved:
            best_value = current_value
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / 'best_fk_pva.pt', model, optimizer, args, epoch, step, validation['loss'], weights)
        else:
            epochs_without_improvement += 1
        save_checkpoint(output_dir / 'last.pt', model, optimizer, args, epoch, step, validation['loss'], weights)
        row = {
            'epoch': epoch,
            'lr': lr,
            'train': train_loss,
            'validation': validation['loss'],
            'selection_metric': args.selection_metric,
            'selection_value': current_value,
            'best_value': best_value,
            'best_epoch': best_epoch,
            'epochs_without_improvement': epochs_without_improvement,
        }
        history.append(row)
        with (output_dir / 'train_log.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            break

    result = {
        'experiment_name': args.experiment_name,
        'status': 'ok',
        'config': vars(args),
        'weights': weights,
        'train_manifest': train_manifest,
        'val_manifest': val_manifest,
        'num_train_sequences': len(train_records),
        'num_val_sequences': len(val_records),
        'best_value': best_value,
        'best_epoch': best_epoch,
        'selection_metric': args.selection_metric,
        'stopped_early': args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience,
        'history': history,
        'checkpoints': {
            'best_fk_pva': str(output_dir / 'best_fk_pva.pt'),
            'last': str(output_dir / 'last.pt'),
        },
    }
    (output_dir / 'train_result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
