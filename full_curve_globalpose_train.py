import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

from full_curve_globalpose import FullCurveGlobalPoseV1, full_curve_default_weights, full_curve_loss
from l4_train_diverse_short import DEVICE, load_cache_files


FULL_CURVE_KEYS = (
    'pl_input', 'pl_target', 'pl_base',
    'ik1_target', 'ik1_base',
    'ik2_target', 'ik2_base',
    'vr_target', 'vr_base', 'processed_imu',
)


def load_full_curve_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is not None and manifest.get('type') != 'full_curve_globalpose_cache_v1':
        raise RuntimeError(f'Expected full_curve_globalpose_cache_v1 manifest, got {manifest.get("type")}.')
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        missing = [key for key in FULL_CURVE_KEYS if key not in data]
        if missing:
            raise KeyError(f'{cache_file} missing fields: {missing}')
        for seq_idx, name in enumerate(data['name']):
            record = {'name': name}
            for key in FULL_CURVE_KEYS:
                record[key] = data[key][seq_idx].float()
            if 'offset_r' in data and data['offset_r']:
                record['offset_r'] = data['offset_r'][seq_idx].float()
            records.append(record)
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def slice_record(record, start, length):
    seq_len = record['pl_input'].shape[0]
    if length <= 0 or seq_len <= length:
        return record
    start = min(max(0, int(start)), seq_len - length)
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
    out = {'name': '|'.join(f"{record['name']}[{int(start)}:{int(start) + length}]" for record, start in zip(records, starts))}
    for key in FULL_CURVE_KEYS:
        values = []
        for record, start in zip(records, starts):
            seq_len = record['pl_input'].shape[0]
            start = min(max(0, int(start)), max(0, seq_len - length))
            values.append(record[key][start:start + length])
        out[key] = torch.stack(values, dim=1)
    if all('offset_r' in record and torch.is_tensor(record['offset_r']) and record['offset_r'].numel() > 0 for record in records):
        out['offset_r'] = torch.stack([record['offset_r'] for record in records], dim=0)
    return out


def to_device_record(record):
    return {key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in record.items()}


def run_sequence(model, record, weights):
    record = to_device_record(record)
    output = model.forward_sequence(record)
    loss, components = full_curve_loss(output, record, weights)
    return loss, components, output


def average(rows):
    totals = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / max(1, len(values)) for key, values in totals.items()}


@torch.no_grad()
def eval_loss(model, records, weights, max_sequences=0, window=0):
    model.eval()
    rows = []
    selected = records[:max_sequences] if max_sequences else records
    for record in selected:
        eval_record = slice_record(record, 0, window) if window else record
        loss, components, output = run_sequence(model, eval_record, weights)
        row = {'name': eval_record['name'], 'loss': float(loss.detach())}
        row.update({key: float(value.detach()) for key, value in components.items()})
        row['control_shapes'] = output['control_shapes']
        rows.append(row)
    return {'num_sequences': len(rows), 'loss': average(rows), 'rows': rows}


def save_checkpoint(path, model, optimizer, args, epoch, step, val_loss, weights):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'epoch': epoch,
        'step': step,
        'validation_loss': val_loss,
        'weights': weights,
        'model_type': 'full_curve_globalpose_v1',
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train FullCurveGlobalPose_v1 on precomputed full-chain caches.')
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--vr-residual-scale', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--offset-init-scale', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--val-window', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    for key in full_curve_default_weights():
        parser.add_argument(f'--{key.replace("_", "-")}-weight', type=float, default=None)
    args = parser.parse_args()

    weights = full_curve_default_weights()
    for key in list(weights):
        override = getattr(args, f'{key}_weight')
        if override is not None:
            weights[key] = override
    train_records, train_manifest = load_full_curve_records(args.train_cache, args.max_train_sequences)
    val_records, val_manifest = load_full_curve_records(args.val_cache, args.max_val_sequences)
    if args.batch_size > 1:
        train_records = [record for record in train_records if record['pl_input'].shape[0] >= args.window]
        if not train_records:
            raise RuntimeError(f'No FullCurve training sequence has at least window={args.window} frames.')

    model = FullCurveGlobalPoseV1(
        hidden_size=args.hidden_size,
        tail_update=args.tail_length,
        residual_scale=args.residual_scale,
        vr_residual_scale=args.vr_residual_scale,
        dropout=args.dropout,
        offset_init_scale=args.offset_init_scale,
    ).to(DEVICE)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        if checkpoint.get('model_type') != 'full_curve_globalpose_v1':
            raise RuntimeError(f'Unsupported init checkpoint model_type={checkpoint.get("model_type")}')
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    (output_dir / 'weights.json').write_text(json.dumps(weights, indent=2) + '\n')
    log_path = output_dir / 'train_log.jsonl'
    best_loss = float('inf')
    best_epoch = 0
    step = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_rows = []
        iterable = range(0, len(train_records), args.batch_size) if args.batch_size > 1 else range(len(train_records))
        for seq_idx, batch_start in enumerate(iterable, start=1):
            step += 1
            if args.batch_size > 1:
                batch_records = train_records[batch_start:batch_start + args.batch_size]
                starts = []
                for offset, source_record in enumerate(batch_records):
                    seq_len = source_record['pl_input'].shape[0]
                    max_start = max(0, seq_len - args.window)
                    starts.append((step + offset) % (max_start + 1) if max_start > 0 else 0)
                record = make_batch(batch_records, starts, args.window)
            else:
                source_record = train_records[batch_start]
                seq_len = source_record['pl_input'].shape[0]
                max_start = max(0, seq_len - args.window)
                start = step % (max_start + 1) if max_start > 0 else 0
                record = slice_record(source_record, start, args.window)
            loss, components, output = run_sequence(model, record, weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at {record["name"]}.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            row = {'epoch': epoch, 'step': step, 'seq_idx': seq_idx, 'seq_name': record['name'], 'loss': float(loss.detach())}
            row.update({key: float(value.detach()) for key, value in components.items()})
            row['control_shapes'] = output['control_shapes']
            train_rows.append(row)
        train_loss = average(train_rows)
        validation = eval_loss(model, val_records, weights, max_sequences=args.max_val_sequences, window=args.val_window)
        val_loss = validation['loss'].get('loss', float('inf'))
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            save_checkpoint(output_dir / 'best_loss.pt', model, optimizer, args, epoch, step, val_loss, weights)
        save_checkpoint(output_dir / 'last.pt', model, optimizer, args, epoch, step, val_loss, weights)
        epoch_row = {
            'epoch': epoch,
            'step': step,
            'train_loss': train_loss,
            'validation': validation,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'train_manifest': train_manifest,
            'val_manifest': val_manifest,
        }
        history.append(epoch_row)
        with log_path.open('a') as f:
            f.write(json.dumps(epoch_row) + '\n')
        print(json.dumps({'epoch': epoch, 'train_loss': train_loss.get('loss'), 'val_loss': val_loss, 'best_loss': best_loss}, indent=2))
    result = {
        'experiment_name': args.experiment_name,
        'status': 'ok',
        'config': vars(args),
        'weights': weights,
        'num_train_sequences': len(train_records),
        'num_val_sequences': len(val_records),
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'history': history,
    }
    (output_dir / 'train_result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
