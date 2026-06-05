import argparse
import json
import random
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from articulate.utils.torch import RNN
from l4_train_diverse_short import DEVICE, load_cache_files
from newik1_control_point import finite_diff, normalize_ik1, pRJ_bone_lengths


MODEL_TYPE = 'newik1_official_input_v1'


def default_weights():
    return {
        'pRJ': 2.0,
        'gR2': 1.0,
        'bone_length': 0.5,
        'pRJ_dot': 0.05,
        'gR2_dot': 0.03,
        'pRJ_ddot': 0.002,
        'gR2_ddot': 0.001,
        'ik1_distill_pRJ': 0.2,
        'ik1_distill_gR2': 0.0,
    }


def build_model(dropout=0.4):
    return RNN(
        input_linear=False,
        input_size=63,
        output_size=72,
        hidden_size=512,
        num_rnn_layer=3,
        dropout=dropout,
    )


def load_official_weights(model, weights_path):
    weights = torch.load(weights_path, map_location='cpu')
    prefix = 'iknet.net1.'
    state = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Official IK1 weight mismatch: missing={missing}, unexpected={unexpected}')


def load_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is None or manifest.get('type') != 'newik1_official_input_cache_v1':
        raise RuntimeError(f'Expected newik1_official_input_cache_v1 manifest, got {manifest.get("type") if manifest else None}.')
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            records.append({
                'name': name,
                'ik1_input': data['ik1_input'][seq_idx].float(),
                'ik1_target': data['ik1_target'][seq_idx].float(),
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


def make_batch(records, starts, length):
    out = {'name': '|'.join(f"{record['name']}[{int(start)}:{int(start) + length}]" for record, start in zip(records, starts))}
    for key in ('ik1_input', 'ik1_target', 'ik1_base'):
        vals = []
        for record, start in zip(records, starts):
            seq_len = record['ik1_input'].shape[0]
            start = min(max(0, int(start)), max(0, seq_len - length))
            vals.append(record[key][start:start + length])
        out[key] = torch.stack(vals, dim=1)
    return out


def forward_sequence(model, features):
    squeeze_batch = features.dim() == 2
    if squeeze_batch:
        features = features.unsqueeze(1)
    x, _ = model.rnn(features, None)
    pred = model.linear2(x)
    pred = normalize_ik1(pred)
    return pred[:, 0] if squeeze_batch else pred


def official_ik1_loss(pred, target, base, weights):
    pred = normalize_ik1(pred)
    target = normalize_ik1(target.to(pred.device, pred.dtype))
    base = normalize_ik1(base.to(pred.device, pred.dtype))
    pred_g = art.math.normalize_tensor(pred[..., 69:], avoid_nan=True)
    target_g = art.math.normalize_tensor(target[..., 69:], avoid_nan=True)
    base_g = art.math.normalize_tensor(base[..., 69:], avoid_nan=True)
    losses = {
        'pRJ': torch.nn.functional.smooth_l1_loss(pred[..., :69], target[..., :69]),
        'gR2': (1.0 - (pred_g * target_g).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
        'bone_length': torch.nn.functional.smooth_l1_loss(pRJ_bone_lengths(pred), pRJ_bone_lengths(target)),
        'ik1_distill_pRJ': torch.nn.functional.smooth_l1_loss(pred[..., :69], base[..., :69]),
        'ik1_distill_gR2': (1.0 - (pred_g * base_g).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
    }
    if pred.shape[0] >= 2:
        losses['pRJ_dot'] = torch.nn.functional.smooth_l1_loss(
            pred[1:, ..., :69] - pred[:-1, ..., :69],
            target[1:, ..., :69] - target[:-1, ..., :69],
        )
        losses['gR2_dot'] = torch.nn.functional.smooth_l1_loss(pred_g[1:] - pred_g[:-1], target_g[1:] - target_g[:-1])
    else:
        losses['pRJ_dot'] = pred.new_zeros(())
        losses['gR2_dot'] = pred.new_zeros(())
    if pred.shape[0] >= 3:
        losses['pRJ_ddot'] = torch.nn.functional.smooth_l1_loss(finite_diff(pred[..., :69], 2), finite_diff(target[..., :69], 2))
        losses['gR2_ddot'] = torch.nn.functional.smooth_l1_loss(finite_diff(pred_g, 2), finite_diff(target_g, 2))
    else:
        losses['pRJ_ddot'] = pred.new_zeros(())
        losses['gR2_ddot'] = pred.new_zeros(())
    total = pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * weight
    return total, losses


def run_sequence(model, record, weights):
    features = record['ik1_input'].to(DEVICE)
    target = record['ik1_target'].to(DEVICE)
    base = record['ik1_base'].to(DEVICE)
    pred = forward_sequence(model, features)
    loss, losses = official_ik1_loss(pred, target, base, weights)
    components = {key: value.detach() for key, value in losses.items()}
    components.update({
        'loss': loss.detach(),
        'ik1_residual_norm_mean': (pred - base).norm(dim=-1).mean().detach(),
    })
    return loss, components


@torch.no_grad()
def eval_loss(model, records, weights, max_sequences=0):
    model.eval()
    rows = []
    selected = records[:max_sequences] if max_sequences else records
    for record in selected:
        loss, components = run_sequence(model, record, weights)
        row = {'name': record['name'], 'loss': float(loss)}
        row.update({key: float(value) for key, value in components.items()})
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
        'model_type': MODEL_TYPE,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train official-shape IK1 net1 on decoded PL pRB/gR1 caches.')
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--official-weights', default='data/weights.pt')
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    for key in default_weights():
        parser.add_argument(f'--{key.replace("_", "-")}-weight', type=float, default=None)
    args = parser.parse_args()

    weights = default_weights()
    for key in list(weights):
        value = getattr(args, f'{key}_weight')
        if value is not None:
            weights[key] = value

    train_records, train_manifest = load_records(args.train_cache, args.max_train_sequences)
    val_records, val_manifest = load_records(args.val_cache, args.max_val_sequences)
    if args.batch_size > 1:
        train_records = [record for record in train_records if record['ik1_input'].shape[0] >= args.window]
        if not train_records:
            raise RuntimeError(f'No training sequence has at least window={args.window} frames.')

    model = build_model(dropout=args.dropout).to(DEVICE)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        if checkpoint.get('model_type') != MODEL_TYPE:
            raise RuntimeError(f'Unsupported init checkpoint model_type={checkpoint.get("model_type")}')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        load_official_weights(model, args.official_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    best_loss = float('inf')
    best_epoch = 0
    step = 0
    history = []
    log_path = output_dir / 'train_log.jsonl'
    for epoch in range(1, args.epochs + 1):
        model.train()
        rows = []
        if args.batch_size > 1:
            order = list(range(len(train_records)))
            random.shuffle(order)
            for batch_start in range(0, len(order), args.batch_size):
                ids = order[batch_start:batch_start + args.batch_size]
                recs = [train_records[i] for i in ids]
                starts = [random.randint(0, max(0, rec['ik1_input'].shape[0] - args.window)) for rec in recs]
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
        val = eval_loss(model, val_records, weights)
        val_scalar = float(val['loss'].get('loss', float('inf')))
        if val_scalar < best_loss:
            best_loss = val_scalar
            best_epoch = epoch
            save_checkpoint(output_dir / 'best_loss.pt', model, optimizer, args, epoch, step, val['loss'], weights)
        save_checkpoint(output_dir / 'last.pt', model, optimizer, args, epoch, step, val['loss'], weights)
        row = {'epoch': epoch, 'train': train_loss, 'validation': val['loss'], 'best_loss': best_loss, 'best_epoch': best_epoch}
        history.append(row)
        with log_path.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    result = {
        'status': 'ok',
        'experiment_name': args.experiment_name,
        'config': vars(args),
        'weights': weights,
        'train_manifest': train_manifest,
        'val_manifest': val_manifest,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
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
