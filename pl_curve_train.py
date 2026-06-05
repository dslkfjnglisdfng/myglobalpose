import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_cache_files, load_records
from net import GPNet
from pl_curve import (
    PLCurveModule,
    normalize_gravity,
    pl_curve_loss,
    pl_init_feature_from_pose,
    pl_input_feature,
    pl_target_from_pose,
    split_pl_feature,
)


def default_weights():
    return {
        'pRB': 1.0,
        'gR1': 1.0,
        'baseline_pRB': 0.2,
        'baseline_gR1': 0.2,
        'gt_control_pRB': 0.0,
        'gt_control_gR1': 0.0,
        'control_point_prior': 0.3,
        'tail_update_prior': 0.005,
        'pRB_dot': 0.03,
        'pRB_ddot_smooth': 0.0003,
        'gR1_dot': 0.0,
        'gR1_ddot': 0.0,
        'gR_smooth': 0.003,
        'ik1_pRJ': 0.05,
        'ik1_gR2': 0.05,
        'ik2_r6d': 0.05,
    }


def slice_record(record, start, length):
    seq_len = record['pl_input'].shape[0] if 'pl_input' in record else record['pose_gt'].shape[0]
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
    pl_input, pl_target, pl_base, pl_init = [], [], [], []
    names = []
    for record, start in zip(records, starts):
        seq_len = record['pl_input'].shape[0]
        start = min(max(0, int(start)), max(0, seq_len - length))
        end = start + length
        pl_input.append(record['pl_input'][start:end])
        pl_target.append(record['pl_target'][start:end])
        pl_base.append(record['pl_base'][start:end])
        if 'pl_init_feature' in record:
            pl_init.append(record['pl_init_feature'])
        names.append(f"{record['name']}[{start}:{end}]")
    out = {
        'name': '|'.join(names),
        'pl_input': torch.stack(pl_input, dim=1),
        'pl_target': torch.stack(pl_target, dim=1),
        'pl_base': torch.stack(pl_base, dim=1),
    }
    if pl_init:
        out['pl_init_feature'] = torch.stack(pl_init, dim=0)
    return out


def build_features_targets(record, body_model):
    if 'pl_input' in record and 'pl_target' in record:
        return record['pl_input'].float(), normalize_gravity(record['pl_target'].float())
    features = torch.stack([
        pl_input_feature(record['aM'][i], record['wM'][i], record['RMB'][i])
        for i in range(record['pose_gt'].shape[0])
    ]).float()
    target = pl_target_from_pose(record['pose_gt'].float().to(DEVICE), body_model).float().cpu()
    return features, normalize_gravity(target)


@torch.no_grad()
def base_pl_outputs(gpnet, features, init_target):
    gpnet.plnet.eval()
    return gpnet.plnet([(features.to(DEVICE), init_target.to(DEVICE))])[0].detach()


def load_pl_curve_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    if manifest is not None and manifest.get('type') in ('pl_curve_cache_v1', 'pl_curve_cache_v2'):
        records = []
        has_init = manifest.get('type') == 'pl_curve_cache_v2'
        for cache_file in files:
            data = torch.load(cache_file, map_location='cpu')
            for seq_idx, name in enumerate(data['name']):
                record = {
                    'name': name,
                    'pl_input': data['pl_input'][seq_idx].float(),
                    'pl_target': data['pl_target'][seq_idx].float(),
                    'pl_base': data['pl_base'][seq_idx].float(),
                }
                if has_init:
                    record['pl_init_feature'] = data['pl_init_feature'][seq_idx].float()
                records.append(record)
                if max_sequences and len(records) >= max_sequences:
                    return records, manifest
        return records, manifest
    return load_records(cache_path, max_sequences=max_sequences)


def downstream_ik_outputs(gpnet, features, pl_outputs):
    RRB0, gR0 = split_pl_feature(features.to(DEVICE))
    pRB = pl_outputs[:, :15]
    gR1 = normalize_gravity(pl_outputs)[:, 15:]
    RRB_after_pl = art.math.from_to_rotation_matrix(gR0, gR1).unsqueeze(1).matmul(RRB0)
    ik1_input = torch.cat((RRB_after_pl.flatten(1), gR1, pRB), dim=-1)
    ik1 = gpnet.iknet.net1([ik1_input])[0]
    pRJ = ik1[:, :69]
    gR2 = normalize_gravity(ik1)[:, 69:]
    RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).unsqueeze(1).matmul(RRB_after_pl)
    ik2_input = torch.cat((RRB_after_ik1.flatten(1), gR2, pRJ), dim=-1)
    ik2 = gpnet.iknet.net2([ik2_input])[0]
    return {
        'ik1_pRJ': pRJ,
        'ik1_gR2': gR2,
        'ik2_r6d': ik2,
    }


def init_feature_for_record(record, target, body_model):
    if 'pl_init_feature' in record:
        return record['pl_init_feature'].float()
    if 'offset_r' in record and 'pose_gt' in record:
        return pl_init_feature_from_pose(record['offset_r'].float(), record['pose_gt'][0].float(), body_model)
    return target[0]


def run_sequence(model, gpnet, record, body_model, weights, train_ik_distill=True):
    features, target = build_features_targets(record, body_model)
    features = features.to(DEVICE)
    target = target.to(DEVICE)
    if 'pl_base' in record:
        base = normalize_gravity(record['pl_base'].float()).to(DEVICE)
    else:
        base = base_pl_outputs(gpnet, features, target[0]).to(DEVICE)
    init_feature = init_feature_for_record(record, target, body_model).to(DEVICE)
    if init_feature.shape[-1] != model.init_size:
        raise RuntimeError(f'PL init dim mismatch for {record["name"]}: model expects {model.init_size}, got {init_feature.shape[-1]}.')
    out = model.forward_sequence(features, base, init_feature=init_feature)
    loss, components = pl_curve_loss(out, target, {k: weights[k] for k in (
        'pRB', 'gR1', 'baseline_pRB', 'baseline_gR1', 'control_point_prior',
        'tail_update_prior', 'pRB_dot', 'pRB_ddot_smooth', 'gR1_dot',
        'gR1_ddot', 'gR_smooth', 'gt_control_pRB', 'gt_control_gR1'
    )})
    if train_ik_distill:
        if gpnet is None:
            raise ValueError('IK distillation requires gpnet; disable it for cache-only fast training.')
        with torch.no_grad():
            base_ik = downstream_ik_outputs(gpnet, features, base)
        with torch.backends.cudnn.flags(enabled=False):
            pred_ik = downstream_ik_outputs(gpnet, features, out['pl'])
        ik_losses = {
            'ik1_pRJ': torch.nn.functional.smooth_l1_loss(pred_ik['ik1_pRJ'], base_ik['ik1_pRJ']),
            'ik1_gR2': (1.0 - (pred_ik['ik1_gR2'] * base_ik['ik1_gR2']).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
            'ik2_r6d': torch.nn.functional.smooth_l1_loss(pred_ik['ik2_r6d'], base_ik['ik2_r6d']),
        }
    else:
        zero = loss.new_zeros(())
        ik_losses = {'ik1_pRJ': zero, 'ik1_gR2': zero, 'ik2_r6d': zero}
    for key, value in ik_losses.items():
        loss = loss + value * weights[key]
        components[key] = value
    components.update({
        'new_delta_norm': out['new_delta_norm'],
        'pl_residual_norm_mean': (out['pl'] - out['base']).norm(dim=-1).mean(),
        'gR_norm_mean': out['pl'][:, 15:].norm(dim=-1).mean(),
    })
    return loss, components


def average(rows):
    totals = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / max(1, len(values)) for key, values in totals.items()}


def eval_loss(model, gpnet, records, body_model, weights, max_sequences=0, train_ik_distill=True):
    model.eval()
    rows = []
    with torch.no_grad():
        selected = records[:max_sequences] if max_sequences else records
        for record in selected:
            loss, components = run_sequence(
                model,
                gpnet,
                record,
                body_model,
                weights,
                train_ik_distill=train_ik_distill,
            )
            row = {'name': record['name'], 'loss': float(loss.detach())}
            row.update({key: float(value.detach()) for key, value in components.items()})
            rows.append(row)
    return {'num_sequences': len(rows), 'loss': average(rows), 'rows': rows}


def load_partial_checkpoint(model, checkpoint_state):
    model_state = model.state_dict()
    loaded = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            loaded[key] = value
        elif key == 'init_encoder.0.weight' and key in model_state and value.shape[0] == model_state[key].shape[0]:
            merged = model_state[key].clone()
            copy_width = min(value.shape[1], merged.shape[1])
            merged[:, -copy_width:] = value[:, -copy_width:]
            loaded[key] = merged
        else:
            skipped.append(key)
    model_state.update(loaded)
    model.load_state_dict(model_state)
    return {'loaded': sorted(loaded), 'skipped': sorted(skipped)}


def save_checkpoint(path, model, optimizer, args, epoch, step, val_loss, weights):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'epoch': epoch,
        'step': step,
        'validation_loss': val_loss,
        'weights': weights,
        'model_type': 'pl_curve_v1',
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train PLCurve_v1 on existing GlobalPose prephysics caches.')
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--init-checkpoint', default='', help='Optional PLCurve checkpoint used to initialize model weights for finetuning. Optimizer state is not restored.')
    parser.add_argument('--init-size', type=int, default=36, help='PL curve init feature dim. Use 36 for offset_r[18]+pRL[15]+gR0[3], 18 for legacy init_output.')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0)
    parser.add_argument('--early-stop-patience', type=int, default=0)
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1, help='Batch windows for precomputed PLCurve caches. Falls back to sequence-wise training for raw caches.')
    parser.add_argument('--disable-ik-distill', action='store_true')
    parser.add_argument('--pRB-weight', type=float, default=None)
    parser.add_argument('--gR1-weight', type=float, default=None)
    parser.add_argument('--baseline-pRB-weight', type=float, default=None)
    parser.add_argument('--baseline-gR1-weight', type=float, default=None)
    parser.add_argument('--gt-control-pRB-weight', type=float, default=None)
    parser.add_argument('--gt-control-gR1-weight', type=float, default=None)
    parser.add_argument('--control-point-prior-weight', type=float, default=None)
    parser.add_argument('--tail-update-prior-weight', type=float, default=None)
    parser.add_argument('--pRB-dot-weight', type=float, default=None)
    parser.add_argument('--pRB-ddot-smooth-weight', type=float, default=None)
    parser.add_argument('--gR1-dot-weight', type=float, default=None)
    parser.add_argument('--gR1-ddot-weight', type=float, default=None)
    parser.add_argument('--ik-distill-weight', type=float, default=None)
    args = parser.parse_args()

    weights = default_weights()
    overrides = {
        'pRB': args.pRB_weight,
        'gR1': args.gR1_weight,
        'baseline_pRB': args.baseline_pRB_weight,
        'baseline_gR1': args.baseline_gR1_weight,
        'gt_control_pRB': args.gt_control_pRB_weight,
        'gt_control_gR1': args.gt_control_gR1_weight,
        'control_point_prior': args.control_point_prior_weight,
        'tail_update_prior': args.tail_update_prior_weight,
        'pRB_dot': args.pRB_dot_weight,
        'pRB_ddot_smooth': args.pRB_ddot_smooth_weight,
        'gR1_dot': args.gR1_dot_weight,
        'gR1_ddot': args.gR1_ddot_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            weights[key] = value
    if args.ik_distill_weight is not None:
        weights['ik1_pRJ'] = args.ik_distill_weight
        weights['ik1_gR2'] = args.ik_distill_weight
        weights['ik2_r6d'] = args.ik_distill_weight

    train_records, train_manifest = load_pl_curve_records(args.train_cache, max_sequences=args.max_train_sequences)
    val_records, val_manifest = load_pl_curve_records(args.val_cache, max_sequences=args.max_val_sequences)
    using_pl_cache = bool(train_manifest and train_manifest.get('type') in ('pl_curve_cache_v1', 'pl_curve_cache_v2'))
    if args.init_size != 18 and train_manifest and train_manifest.get('type') != 'pl_curve_cache_v2':
        raise RuntimeError(f'init_size={args.init_size} requires pl_curve_cache_v2 with pl_init_feature.')
    if using_pl_cache and args.batch_size > 1:
        train_records = [record for record in train_records if record['pl_input'].shape[0] >= args.window]
        if not train_records:
            raise RuntimeError(f'No PL cache training sequence has at least window={args.window} frames.')
    gpnet = None
    body_model = None
    if (not using_pl_cache) or (not args.disable_ik_distill):
        gpnet = GPNet().eval().to(DEVICE)
        for parameter in gpnet.parameters():
            parameter.requires_grad_(False)
    if not using_pl_cache:
        body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    model = PLCurveModule(
        init_size=args.init_size,
        hidden_size=args.hidden_size,
        tail_update=args.tail_length,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
    ).to(DEVICE)
    init_checkpoint_load = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        if checkpoint.get('model_type') != 'pl_curve_v1':
            raise RuntimeError(f'Unsupported init checkpoint model_type={checkpoint.get("model_type")}')
        init_checkpoint_load = load_partial_checkpoint(model, checkpoint['model_state_dict'])
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
    stop_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_rows = []
        if using_pl_cache and args.batch_size > 1:
            iterable = range(0, len(train_records), args.batch_size)
        else:
            iterable = range(len(train_records))
        for seq_idx, batch_start in enumerate(iterable, start=1):
            step += 1
            if using_pl_cache and args.batch_size > 1:
                batch_records = train_records[batch_start:batch_start + args.batch_size]
                starts = []
                for offset, source_record in enumerate(batch_records):
                    seq_len = source_record['pl_input'].shape[0]
                    max_start = max(0, seq_len - args.window)
                    starts.append((step + offset) % (max_start + 1) if max_start > 0 else 0)
                record = make_batch(batch_records, starts, args.window)
            else:
                source_record = train_records[batch_start]
                seq_len = source_record['pl_input'].shape[0] if 'pl_input' in source_record else source_record['pose_gt'].shape[0]
                max_start = max(0, seq_len - args.window)
                start = step % (max_start + 1) if max_start > 0 else 0
                record = slice_record(source_record, start, args.window)
            loss, components = run_sequence(
                model,
                gpnet,
                record,
                body_model,
                weights,
                train_ik_distill=not args.disable_ik_distill,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at {record["name"]}.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            row = {
                'epoch': epoch,
                'step': step,
                'seq_idx': seq_idx,
                'seq_name': record['name'],
                'loss': float(loss.detach()),
            }
            row.update({key: float(value.detach()) for key, value in components.items()})
            train_rows.append(row)
        train_loss = average(train_rows)
        validation = eval_loss(
            model,
            gpnet,
            val_records,
            body_model,
            weights,
            max_sequences=args.max_val_sequences,
            train_ik_distill=not args.disable_ik_distill,
        )
        val_loss = validation['loss'].get('loss', float('inf'))
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
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'improved': improved,
            'stale_epochs': stale_epochs,
        }
        history.append(epoch_row)
        with log_path.open('a') as f:
            f.write(json.dumps(epoch_row) + '\n')
        print(json.dumps({
            'epoch': epoch,
            'train_loss': train_loss.get('loss'),
            'val_loss': val_loss,
            'best_loss': best_loss,
            'stale_epochs': stale_epochs,
        }, indent=2))
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            stopped_early = True
            stop_epoch = epoch
            break
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
        'stopped_early': stopped_early,
        'stop_epoch': stop_epoch,
        'early_stop_min_delta': args.early_stop_min_delta,
        'early_stop_patience': args.early_stop_patience,
        'init_checkpoint_load': init_checkpoint_load,
        'history': history,
    }
    (output_dir / 'train_result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
