import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from ik1_curve import IK1CurveModule, assemble_pRJ, full_ik1_from_state
from l4_train_diverse_short import DEVICE, load_cache_files
from pl_curve import normalize_gravity


DT = 1.0 / 60.0


def default_weights():
    return {
        'pRJ_nonleaf': 1.0,
        'gR2': 1.0,
        'baseline_pRJ_nonleaf': 0.2,
        'baseline_gR2': 0.2,
        'control_point_prior': 0.3,
        'tail_update_prior': 0.005,
        'pRJ_dot': 0.03,
        'pRJ_ddot_smooth': 1e-6,
        'bone_length': 0.05,
        'imu_acc_proxy': 0.0,
    }


def load_ik1_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is None or manifest.get('type') != 'ik1_curve_cache_v1':
        raise RuntimeError(f'Expected ik1_curve_cache_v1 manifest, got {manifest.get("type") if manifest else None}.')
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            records.append({
                'name': name,
                'ik1_input': data['ik1_input'][seq_idx].float(),
                'ik1_target_nonleaf': data['ik1_target_nonleaf'][seq_idx].float(),
                'ik1_target_full': data['ik1_target_full'][seq_idx].float(),
                'ik1_target_gR2': data['ik1_target_gR2'][seq_idx].float(),
                'ik1_base': data['ik1_base'][seq_idx].float(),
                'ik2_base': data['ik2_base'][seq_idx].float(),
                'leaf_pRB': data['leaf_pRB'][seq_idx].float(),
                'imu_acc_target': data['imu_acc_target'][seq_idx].float(),
            })
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def slice_record(record, start, length):
    seq_len = record['ik1_input'].shape[0]
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
    out = {'name': '|'.join(f"{record['name']}[{int(start)}:{int(start) + length}]" for record, start in zip(records, starts))}
    keys = ('ik1_input', 'ik1_target_nonleaf', 'ik1_target_full', 'ik1_target_gR2', 'ik1_base', 'ik2_base', 'leaf_pRB', 'imu_acc_target')
    for key in keys:
        values = []
        for record, start in zip(records, starts):
            seq_len = record['ik1_input'].shape[0]
            start = min(max(0, int(start)), max(0, seq_len - length))
            values.append(record[key][start:start + length])
        out[key] = torch.stack(values, dim=1)
    return out


def average(rows):
    totals = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / max(1, len(values)) for key, values in totals.items()}


def bone_lengths(pRJ_full, parent):
    pRJ = pRJ_full.reshape(pRJ_full.shape[:-1] + (23, 3))
    root = pRJ.new_zeros(pRJ.shape[:-2] + (1, 3))
    joints = torch.cat((root, pRJ), dim=-2)
    lengths = []
    for joint_idx in range(1, 24):
        parent_idx = parent[joint_idx]
        lengths.append((joints[..., joint_idx, :] - joints[..., parent_idx, :]).norm(dim=-1))
    return torch.stack(lengths, dim=-1)


def imu_acc_proxy_loss(full_pRJ, imu_acc_target):
    if full_pRJ.shape[0] < 3:
        return full_pRJ.new_zeros(())
    leaf = full_pRJ[..., :69].reshape(full_pRJ.shape[:-1] + (23, 3))[..., [17, 18, 3, 4, 14], :]
    acc = (leaf[2:] - 2.0 * leaf[1:-1] + leaf[:-2]) / (DT * DT)
    target = imu_acc_target[1:-1].to(acc.device, acc.dtype)
    return torch.nn.functional.smooth_l1_loss(acc, target)


def run_sequence(model, record, weights, parent):
    features = record['ik1_input'].float().to(DEVICE)
    target_nonleaf = record['ik1_target_nonleaf'].float().to(DEVICE)
    target_full = record['ik1_target_full'].float().to(DEVICE)
    target_gR2 = record['ik1_target_gR2'].float().to(DEVICE)
    base = normalize_gravity(record['ik1_base'].float()).to(DEVICE)
    leaf_pRB = record['leaf_pRB'].float().to(DEVICE)
    imu_acc_target = record['imu_acc_target'].float().to(DEVICE)
    output = model.forward_sequence(features, base, leaf_pRB, init_output=base[0])
    pred = output['ik1']
    pred_nonleaf = output['state'][..., :54]
    pred_gR2 = pred[..., 69:]
    base_nonleaf = output['base'][..., :69].reshape(output['base'].shape[:-1] + (23, 3))[..., [i for i in range(23) if i not in (17, 18, 3, 4, 14)], :].reshape_as(pred_nonleaf)
    base_gR2 = output['base'][..., 69:]
    losses = {
        'pRJ_nonleaf': torch.nn.functional.smooth_l1_loss(pred_nonleaf, target_nonleaf),
        'gR2': (1.0 - (pred_gR2 * target_gR2).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
        'baseline_pRJ_nonleaf': torch.nn.functional.smooth_l1_loss(pred_nonleaf, base_nonleaf.detach()),
        'baseline_gR2': (1.0 - (pred_gR2 * base_gR2.detach()).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
        'control_point_prior': output['control_point_prior'],
        'tail_update_prior': output['tail_delta_norm'],
        'pRJ_ddot_smooth': output['stateddot'][..., :54].square().mean(),
        'bone_length': torch.nn.functional.smooth_l1_loss(bone_lengths(pred[..., :69], parent), bone_lengths(target_full, parent)),
        'imu_acc_proxy': imu_acc_proxy_loss(pred, imu_acc_target),
    }
    if pred.shape[0] >= 2:
        target_step = target_nonleaf[1:] - target_nonleaf[:-1]
        losses['pRJ_dot'] = torch.nn.functional.smooth_l1_loss(DT * output['statedot'][1:, ..., :54], target_step)
    else:
        losses['pRJ_dot'] = pred.new_zeros(())
    total = pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * weight
    components = {key: value.detach() for key, value in losses.items()}
    components.update({
        'loss': total.detach(),
        'new_delta_norm': output['new_delta_norm'].detach(),
        'tail_delta_norm': output['tail_delta_norm'].detach(),
        'ik1_residual_norm_mean': (pred - output['base']).norm(dim=-1).mean().detach(),
    })
    return total, components


@torch.no_grad()
def eval_loss(model, records, weights, parent, max_sequences=0):
    model.eval()
    rows = []
    selected = records[:max_sequences] if max_sequences else records
    for record in selected:
        loss, components = run_sequence(model, record, weights, parent)
        row = {'name': record['name'], 'loss': float(loss.detach())}
        row.update({key: float(value.detach()) for key, value in components.items()})
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
        'model_type': 'ik1_curve_v1',
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train IK1Curve_v1 on precomputed IK1 curve caches.')
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
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--pRJ-nonleaf-weight', type=float, default=None)
    parser.add_argument('--gR2-weight', type=float, default=None)
    parser.add_argument('--baseline-pRJ-nonleaf-weight', type=float, default=None)
    parser.add_argument('--baseline-gR2-weight', type=float, default=None)
    parser.add_argument('--control-point-prior-weight', type=float, default=None)
    parser.add_argument('--pRJ-dot-weight', type=float, default=None)
    parser.add_argument('--pRJ-ddot-smooth-weight', type=float, default=None)
    parser.add_argument('--bone-length-weight', type=float, default=None)
    parser.add_argument('--imu-acc-proxy-weight', type=float, default=None)
    args = parser.parse_args()

    weights = default_weights()
    overrides = {
        'pRJ_nonleaf': args.pRJ_nonleaf_weight,
        'gR2': args.gR2_weight,
        'baseline_pRJ_nonleaf': args.baseline_pRJ_nonleaf_weight,
        'baseline_gR2': args.baseline_gR2_weight,
        'control_point_prior': args.control_point_prior_weight,
        'pRJ_dot': args.pRJ_dot_weight,
        'pRJ_ddot_smooth': args.pRJ_ddot_smooth_weight,
        'bone_length': args.bone_length_weight,
        'imu_acc_proxy': args.imu_acc_proxy_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            weights[key] = value

    train_records, train_manifest = load_ik1_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_ik1_records(args.val_cache, max_sequences=args.max_val_sequences)
    if args.batch_size > 1:
        train_records = [record for record in train_records if record['ik1_input'].shape[0] >= args.window]
        if not train_records:
            raise RuntimeError(f'No IK1 cache training sequence has at least window={args.window} frames.')

    model = IK1CurveModule(
        hidden_size=args.hidden_size,
        tail_update=args.tail_length,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
    ).to(DEVICE)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        if checkpoint.get('model_type') != 'ik1_curve_v1':
            raise RuntimeError(f'Unsupported init checkpoint model_type={checkpoint.get("model_type")}')
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    parent = art.ParametricModel('models/SMPL_male.pkl').parent

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
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
                    seq_len = source_record['ik1_input'].shape[0]
                    max_start = max(0, seq_len - args.window)
                    starts.append((step + offset) % (max_start + 1) if max_start > 0 else 0)
                record = make_batch(batch_records, starts, args.window)
            else:
                source_record = train_records[batch_start]
                seq_len = source_record['ik1_input'].shape[0]
                max_start = max(0, seq_len - args.window)
                start = step % (max_start + 1) if max_start > 0 else 0
                record = slice_record(source_record, start, args.window)
            loss, components = run_sequence(model, record, weights, parent)
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
        validation = eval_loss(model, val_records, weights, parent, max_sequences=args.max_val_sequences)
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
        'train_cache_manifest': train_manifest,
        'val_cache_manifest': val_manifest,
        'num_train_sequences': len(train_records),
        'num_val_sequences': len(val_records),
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'history': history,
    }
    (output_dir / 'train_result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
