import argparse
import json
import math
import random
import shlex
import sys
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import load_cache_files
from pl_curve import (
    PL_LEARNED_OFFSET_ACC_CONTRACT,
    build_pl_curve_model,
    bounded_offset_from_raw,
    learned_offset_imu_acc_loss,
    learned_offset_imu_acc_terms,
    normalize_gravity,
    pl_curve_loss,
    raw_offset_from_bounded,
)
from pl_curve_train import attach_pl_target_controls, load_partial_checkpoint


LEAF_NAMES = ('L_LowArm', 'R_LowArm', 'L_LowLeg', 'R_LowLeg', 'Head')


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def stats_tensor(values):
    values = torch.as_tensor(values).detach().float().reshape(-1).cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {'mean': None, 'median': None, 'p95': None, 'count': 0}
    return {
        'mean': float(values.mean()),
        'median': float(values.median()),
        'p95': float(torch.quantile(values, 0.95)),
        'count': int(values.numel()),
    }


def gravity_angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target, avoid_nan=True)
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def finite_second(x, dt):
    out = torch.zeros_like(x)
    if x.shape[0] <= 2:
        return out
    out[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (float(dt) ** 2)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def slice_or_pad_time(value, start, length):
    chunk = value[start:start + length]
    if length <= 0 or chunk.shape[0] >= length:
        return chunk
    if chunk.shape[0] == 0:
        raise RuntimeError('Cannot pad an empty time slice.')
    pad = chunk[-1:].expand((length - chunk.shape[0],) + chunk.shape[1:])
    return torch.cat((chunk, pad), dim=0)


def load_pl_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    supported = {'pl_curve_cache_v1', 'pl_curve_cache_v2', 'pl_curve_cache_v3', 'pl_curve_cache_v4'}
    if manifest is None or manifest.get('type') not in supported:
        raise RuntimeError(f'Expected PL cache manifest type in {sorted(supported)}, got {manifest.get("type") if manifest else None}.')
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            record = {
                'name': str(name),
                'pl_input': data['pl_input'][seq_idx].float(),
                'pl_target': normalize_gravity(data['pl_target'][seq_idx].float()),
                'pl_base': normalize_gravity(data['pl_base'][seq_idx].float()),
            }
            if 'pl_init_feature' in data:
                record['pl_init_feature'] = data['pl_init_feature'][seq_idx].float()
            if 'pl_target_control' in data:
                record['pl_target_control'] = data['pl_target_control'][seq_idx].float()
            records.append(record)
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def init_feature_for_model(record, model):
    init = record.get('pl_init_feature')
    if init is None:
        if model.init_size == 18:
            return record['pl_target'][0]
        raise KeyError(f'{record["name"]} missing pl_init_feature for init_size={model.init_size}.')
    init = init.float()
    if init.shape[-1] == model.init_size:
        return init
    if model.init_size == 18 and init.shape[-1] == 36:
        return init[..., 18:36].contiguous()
    raise RuntimeError(f'Init dim mismatch for {record["name"]}: model expects {model.init_size}, cache has {init.shape[-1]}.')


def make_batch(records, starts, length):
    tensor_keys = ('pl_input', 'pl_target', 'pl_base', 'pl_target_control')
    out = {'name': [], 'records': list(records)}
    for key in tensor_keys:
        chunks = []
        available = all(key in record for record in records)
        if not available:
            continue
        for record, start in zip(records, starts):
            seq_len = record['pl_input'].shape[0]
            start = min(max(0, int(start)), max(0, seq_len - length))
            chunks.append(slice_or_pad_time(record[key], start, length))
        out[key] = torch.stack(chunks, dim=1)
    out['init18'] = torch.stack([init_feature_for_model(record, _DummyInit(18)) for record in records], dim=0)
    out['init36_or_18'] = [record.get('pl_init_feature', record['pl_target'][0]) for record in records]
    for record, start in zip(records, starts):
        out['name'].append(f'{record["name"]}[{start}:{start + length}]')
    out['name'] = '|'.join(out['name'])
    return out


class _DummyInit:
    def __init__(self, init_size):
        self.init_size = init_size


def batch_for_model_init(batch, model, device):
    items = []
    for record in batch['records']:
        items.append(init_feature_for_model(record, model))
    return torch.stack(items, dim=0).to(device)


def model_output(model, batch, device):
    features = batch['pl_input'].to(device)
    base = normalize_gravity(batch['pl_base'].float()).to(device)
    init = batch_for_model_init(batch, model, device)
    return model.forward_sequence(features, base, init_feature=init)


def official_output(batch, dt, device):
    pl = normalize_gravity(batch['pl_base'].float()).to(device)
    return {
        'pl': pl,
        'pldot': torch.zeros_like(pl),
        'plddot': finite_second(pl, dt=dt),
        'base': pl,
        'new_control': pl,
        'control_point_prior': pl.new_zeros(()),
        'tail_delta_norm': pl.new_zeros(()),
        'new_delta_norm': pl.new_zeros(()),
    }


def pl_metrics_from_output(output, target):
    pred = normalize_gravity(output['pl']).detach().cpu()
    target = normalize_gravity(target.detach().cpu())
    err = pred[..., :15] - target[..., :15]
    leaf = err.reshape(err.shape[:-1] + (5, 3)).norm(dim=-1)
    g = gravity_angle_deg(pred[..., 15:], target[..., 15:])
    per_leaf = [float(leaf[..., idx].mean() * 100.0) for idx in range(5)]
    return {
        'pRB_L1_cm': float(err.abs().mean() * 100.0),
        'pRB_L2_cm': float(leaf.mean() * 100.0),
        'per_leaf_pRB_L2_cm': per_leaf,
        'gR1_angle_deg': float(g.mean()),
        'all_finite': bool(torch.isfinite(pred).all() and torch.isfinite(target).all()),
    }


def acc_residual_for_offset(output, features, offset, dt, device):
    terms = learned_offset_imu_acc_terms(
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in output.items()},
        features.to(device),
        offset.to(device),
        dt=dt,
    )
    residual = terms['residual'].detach().cpu()
    return {
        'imu_acc_residual_l2_mps2': float(residual.norm(dim=-1).mean()) if residual.numel() else 0.0,
        'imu_acc_residual_rms_mps2': float(residual.square().mean().sqrt()) if residual.numel() else 0.0,
        'all_finite': bool(torch.isfinite(residual).all()),
    }


def offset_summary(offset):
    norms = offset.detach().float().norm(dim=-1).cpu()
    return {
        'offset_norm_mean_m': float(norms.mean()),
        'offset_norm_median_m': float(norms.median()),
        'offset_norm_p95_m': float(torch.quantile(norms, 0.95)),
    }


def fixed_official_batch(records, window, batch_size, seed):
    selected = [record for record in records if record['pl_input'].shape[0] >= int(window)]
    if not selected:
        raise RuntimeError(f'No sequence has at least window={window} frames.')
    selected = selected[:max(1, int(batch_size))]
    starts = []
    rng = random.Random(seed)
    for record in selected:
        max_start = max(0, record['pl_input'].shape[0] - int(window))
        starts.append(rng.randint(0, max_start) if max_start else 0)
    return make_batch(selected, starts, int(window))


def build_v7_model(args, learned_offset_init=None):
    config = {
        'model_variant': 'newpl_v7_learned_offset_accaux',
        'input_size': 84,
        'init_size': 18,
        'hidden_size': args.hidden_size,
        'tail_length': 4,
        'residual_scale': args.residual_scale,
        'dropout': args.dropout,
        'offset_max': args.offset_max,
        'learned_offset_init': learned_offset_init,
    }
    return build_pl_curve_model(config), config


def maybe_load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return {'loaded': [], 'skipped': [], 'path': ''}
    path = Path(checkpoint_path)
    if not path.exists():
        return {'loaded': [], 'skipped': [], 'path': str(path), 'status': 'not found'}
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('model_type') != 'pl_curve_v1':
        return {'loaded': [], 'skipped': [], 'path': str(path), 'status': f'unsupported model_type={checkpoint.get("model_type")}'}
    result = load_partial_checkpoint(model, checkpoint['model_state_dict'])
    result['path'] = str(path)
    result['status'] = 'ok'
    return result


def stage0_identifiability(args, records, device, output_dir):
    batch = fixed_official_batch(records, args.window, args.batch_size, args.seed)
    features = batch['pl_input'].to(device)
    out = official_output(batch, args.dt, device)
    zero = torch.zeros(6, 3, device=device)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 7)
    random_offset = torch.empty(6, 3, device=device).uniform_(
        -args.offset_max,
        args.offset_max,
        generator=rng,
    )
    zero_res = acc_residual_for_offset(out, features, zero, args.dt, device)
    random_res = acc_residual_for_offset(out, features, random_offset, args.dt, device)
    raw = torch.nn.Parameter(torch.zeros(6, 3, device=device))
    opt = torch.optim.AdamW([raw], lr=args.offset_lr)
    history = []
    for step in range(1, args.stage0_steps + 1):
        opt.zero_grad(set_to_none=True)
        offset = bounded_offset_from_raw(raw, args.offset_max)
        loss, comps = learned_offset_imu_acc_loss(out, features, offset, dt=args.dt, acc_scale=args.acc_scale)
        prior = offset.square().mean()
        total = loss + args.offset_prior_weight * prior
        if not torch.isfinite(total):
            raise RuntimeError(f'Stage 0 non-finite loss at step={step}.')
        total.backward()
        torch.nn.utils.clip_grad_norm_([raw], args.grad_clip)
        opt.step()
        if step == 1 or step == args.stage0_steps:
            history.append({
                'step': step,
                'loss': float(loss.detach()),
                'imu_acc_l2_mps2': float(comps['imu_acc_l2_mps2']),
                'offset_norm_mean_m': float(offset.detach().norm(dim=-1).mean()),
            })
    learned = bounded_offset_from_raw(raw.detach(), args.offset_max)
    learned_res = acc_residual_for_offset(out, features, learned, args.dt, device)
    payload = {
        'stage': 'stage0_identifiability',
        'dataset_split': args.dataset_label,
        'batch_name': batch['name'],
        'steps': args.stage0_steps,
        'zero_offset_residual': zero_res,
        'random_offset_residual': random_res,
        'learned_offset_residual': learned_res,
        'learned_vs_zero_improvement': zero_res['imu_acc_residual_l2_mps2'] - learned_res['imu_acc_residual_l2_mps2'],
        'random_vs_zero_degradation': random_res['imu_acc_residual_l2_mps2'] - zero_res['imu_acc_residual_l2_mps2'],
        'learned_offset': learned.detach().cpu(),
        'learned_offset_norm': offset_summary(learned.cpu()),
        'all_finite': bool(zero_res['all_finite'] and random_res['all_finite'] and learned_res['all_finite']),
        'history': history,
    }
    torch.save({'raw_offset': raw.detach().cpu(), 'learned_offset': learned.detach().cpu(), 'payload': json_safe(payload)}, output_dir / 'checkpoints' / 'stage0_learned_offset.pt')
    return payload


def stage1_freeze_offset(args, records, device, output_dir, init_offset):
    batch = fixed_official_batch(records, args.window, args.batch_size, args.seed + 1)
    model, config = build_v7_model(args, learned_offset_init=init_offset.cpu().tolist())
    model = model.to(device)
    load_info = maybe_load_checkpoint(model, args.init_checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.raw_offset.requires_grad_(True)
    opt = torch.optim.AdamW([model.raw_offset], lr=args.offset_lr)
    with torch.no_grad():
        out_before = model_output(model, batch, device)
        metrics_before = pl_metrics_from_output(out_before, batch['pl_target'])
        acc_before = acc_residual_for_offset(out_before, batch['pl_input'], model.learned_offset(), args.dt, device)
    history = []
    for step in range(1, args.stage1_steps + 1):
        with torch.no_grad():
            frozen_out = {
                key: value.detach()
                for key, value in model_output(model, batch, device).items()
                if torch.is_tensor(value)
            }
        opt.zero_grad(set_to_none=True)
        offset = model.learned_offset()
        loss, comps = learned_offset_imu_acc_loss(
            frozen_out,
            batch['pl_input'].to(device),
            offset,
            dt=args.dt,
            acc_scale=args.acc_scale,
        )
        prior = offset.square().mean()
        total = loss + args.offset_prior_weight * prior
        if not torch.isfinite(total):
            raise RuntimeError(f'Stage 1 non-finite loss at step={step}.')
        total.backward()
        torch.nn.utils.clip_grad_norm_([model.raw_offset], args.grad_clip)
        opt.step()
        if step == 1 or step == args.stage1_steps:
            history.append({
                'step': step,
                'loss': float(loss.detach()),
                'imu_acc_l2_mps2': float(comps['imu_acc_l2_mps2']),
                'offset_norm_mean_m': float(model.learned_offset().detach().norm(dim=-1).mean()),
            })
    with torch.no_grad():
        out_after = model_output(model, batch, device)
        metrics_after = pl_metrics_from_output(out_after, batch['pl_target'])
        acc_after = acc_residual_for_offset(out_after, batch['pl_input'], model.learned_offset(), args.dt, device)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'stage': 'stage1_freeze_offset_only',
        'coordinate_contract': PL_LEARNED_OFFSET_ACC_CONTRACT,
    }, output_dir / 'checkpoints' / 'stage1_freeze_offset.pt')
    return {
        'stage': 'stage1_freeze_offset_only',
        'dataset_split': args.dataset_label,
        'batch_name': batch['name'],
        'init_checkpoint_load': load_info,
        'pl_before': metrics_before,
        'pl_after': metrics_after,
        'pl_delta_after_minus_before': {
            'pRB_L2_cm': metrics_after['pRB_L2_cm'] - metrics_before['pRB_L2_cm'],
            'gR1_angle_deg': metrics_after['gR1_angle_deg'] - metrics_before['gR1_angle_deg'],
        },
        'acc_before': acc_before,
        'acc_after': acc_after,
        'acc_improvement_l2_mps2': acc_before['imu_acc_residual_l2_mps2'] - acc_after['imu_acc_residual_l2_mps2'],
        'learned_offset': model.learned_offset().detach().cpu(),
        'learned_offset_norm': offset_summary(model.learned_offset().detach().cpu()),
        'all_finite': bool(acc_before['all_finite'] and acc_after['all_finite'] and metrics_after['all_finite']),
        'history': history,
    }, model


def current_weights():
    return {
        'pRB': 1.0,
        'gR1': 1.0,
        'baseline_pRB': 0.05,
        'baseline_gR1': 0.0,
        'gt_control_pRB': 0.3,
        'gt_control_gR1': 0.1,
        'control_point_prior': 0.0,
        'tail_update_prior': 0.005,
        'pRB_dot': 0.03,
        'pRB_ddot': 0.0,
        'pRB_ddot_smooth': 0.0003,
        'gR1_dot': 0.0,
        'gR1_ddot': 0.0,
        'gR_smooth': 0.003,
    }


def stage2_tiny_joint(args, records, device, output_dir, init_model):
    model, config = build_v7_model(args, learned_offset_init=init_model.learned_offset().detach().cpu().tolist())
    model = model.to(device)
    model.load_state_dict(init_model.state_dict(), strict=False)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.joint_lr)
    weights = current_weights()
    train_records = [record for record in records if record['pl_input'].shape[0] >= args.window]
    train_records = train_records[:max(args.batch_size, args.max_train_sequences)]
    if not train_records:
        raise RuntimeError('Stage 2 has no trainable records.')
    history = []
    rng = random.Random(args.seed + 2)
    for epoch in range(1, args.stage2_epochs + 1):
        rng.shuffle(train_records)
        rows = []
        for batch_start in range(0, len(train_records), args.batch_size):
            batch_records = train_records[batch_start:batch_start + args.batch_size]
            starts = []
            for record in batch_records:
                max_start = max(0, record['pl_input'].shape[0] - args.window)
                starts.append(rng.randint(0, max_start) if max_start else 0)
            batch = make_batch(batch_records, starts, args.window)
            out = model_output(model, batch, device)
            target = batch['pl_target'].to(device)
            target_control = batch.get('pl_target_control')
            target_control = None if target_control is None else target_control.to(device)
            pl_loss, comps = pl_curve_loss(out, target, weights, dt=args.dt, target_control=target_control)
            acc_loss, acc_comps = learned_offset_imu_acc_loss(
                out,
                batch['pl_input'].to(device),
                model.learned_offset(),
                dt=args.dt,
                acc_scale=args.acc_scale,
            )
            offset_prior = model.learned_offset().square().mean()
            loss = pl_loss + args.imu_acc_weight * acc_loss + args.offset_prior_weight * offset_prior
            if not torch.isfinite(loss):
                raise RuntimeError(f'Stage 2 non-finite loss at epoch={epoch}.')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            row = {
                'loss': float(loss.detach()),
                'pl_loss': float(pl_loss.detach()),
                'imu_acc_loss': float(acc_loss.detach()),
                'offset_prior': float(offset_prior.detach()),
            }
            row.update({key: float(value.detach()) for key, value in comps.items()})
            row.update({key: float(value.detach()) for key, value in acc_comps.items()})
            rows.append(row)
        history.append({'epoch': epoch, 'train': average(rows)})
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'stage': 'stage2_tiny_joint_training',
        'weights': weights,
        'imu_acc_weight': args.imu_acc_weight,
        'offset_prior_weight': args.offset_prior_weight,
        'coordinate_contract': PL_LEARNED_OFFSET_ACC_CONTRACT,
    }, output_dir / 'checkpoints' / 'stage2_tiny_joint.pt')
    table = evaluate_versions(args, records, device, output_dir, v7_model=model)
    return {
        'stage': 'stage2_tiny_joint_training',
        'history': history,
        'learned_offset': model.learned_offset().detach().cpu(),
        'learned_offset_norm': offset_summary(model.learned_offset().detach().cpu()),
        'pl_module_table': table,
        'all_finite': all(row.get('all_finite', True) for row in table),
    }


def average(rows):
    totals = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / max(1, len(values)) for key, values in totals.items()}


def load_checkpoint_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get('config', {})
    if 'model_variant' not in config and checkpoint.get('model_variant'):
        config = dict(config)
        config['model_variant'] = checkpoint.get('model_variant')
    model = build_pl_curve_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


@torch.no_grad()
def evaluate_versions(args, records, device, output_dir, v7_model):
    batch = fixed_official_batch(records, args.window, args.batch_size, args.seed + 20)
    target = batch['pl_target']
    versions = [
        ('official PL baseline', 'official', None),
        ('newpl_v4_init36 baseline', 'checkpoint', args.newpl_v4_checkpoint),
        ('newpl_v5_dip_best baseline', 'checkpoint', args.init_checkpoint),
        ('newpl_v7_learned_offset_accaux', 'v7', None),
    ]
    rows = []
    for name, kind, path in versions:
        row = {
            'Dataset/split': args.dataset_label,
            'Version': name,
            'notes': '',
            'all_finite': False,
        }
        try:
            if kind == 'official':
                out = official_output(batch, args.dt, device)
                offset = torch.zeros(6, 3, device=device)
                row['notes'] = 'cached official PL baseline; zero-offset acceleration residual'
            elif kind == 'v7':
                v7_model.eval()
                out = model_output(v7_model, batch, device)
                offset = v7_model.learned_offset()
                row['notes'] = 'v7 learned offset acceleration residual; full-pipeline not measured'
            else:
                if not path or not Path(path).exists():
                    row.update({
                        'pRB L1 cm': 'not available',
                        'pRB L2 cm': 'not available',
                        'gR1 angle deg': 'not available',
                        'IMU acc residual': 'not available',
                        'offset norm mean/median/p95': 'not applicable',
                    })
                    row['notes'] = 'checkpoint not available'
                    rows.append(row)
                    continue
                model, _ = load_checkpoint_model(path, device)
                out = model_output(model, batch, device)
                offset = torch.zeros(6, 3, device=device)
                row['notes'] = 'checkpoint PL output; zero-offset acceleration residual'
            metrics = pl_metrics_from_output(out, target)
            acc = acc_residual_for_offset(out, batch['pl_input'], offset, args.dt, device)
            norms = offset_summary(offset.detach().cpu())
            row.update({
                'pRB L1 cm': metrics['pRB_L1_cm'],
                'pRB L2 cm': metrics['pRB_L2_cm'],
                'per_leaf pRB L2 cm': metrics['per_leaf_pRB_L2_cm'],
                'gR1 angle deg': metrics['gR1_angle_deg'],
                'IMU acc residual': acc['imu_acc_residual_l2_mps2'],
                'offset norm mean/median/p95': (
                    f"{norms['offset_norm_mean_m']:.6f}/"
                    f"{norms['offset_norm_median_m']:.6f}/"
                    f"{norms['offset_norm_p95_m']:.6f}"
                ),
                'all_finite': bool(metrics['all_finite'] and acc['all_finite']),
            })
        except Exception as exc:
            row.update({
                'pRB L1 cm': 'failed',
                'pRB L2 cm': 'failed',
                'gR1 angle deg': 'failed',
                'IMU acc residual': 'failed',
                'offset norm mean/median/p95': 'failed',
                'notes': f'failed: {type(exc).__name__}: {exc}',
            })
        rows.append(row)
    return rows


def offset_diagnostic_table(stage0):
    zero = stage0['zero_offset_residual']['imu_acc_residual_l2_mps2']
    random_res = stage0['random_offset_residual']['imu_acc_residual_l2_mps2']
    learned = stage0['learned_offset_residual']['imu_acc_residual_l2_mps2']
    return [{
        'zero offset residual': zero,
        'random offset residual': random_res,
        'learned offset residual': learned,
        'learned vs zero improvement': zero - learned,
        'random vs zero degradation': random_res - zero,
        'all finite?': stage0['all_finite'],
    }]


def markdown_table(rows):
    if not rows:
        return ''
    keys = list(rows[0].keys())
    out = ['| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join(['---'] * len(keys)) + ' |']
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f'{value:.6f}')
            elif isinstance(value, list):
                values.append('/'.join(f'{float(x):.4f}' for x in value))
            else:
                values.append(str(value))
        out.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(out)


def write_summary(output_dir, result):
    lines = [
        '# NewPL v7 learned-offset accaux smoke',
        '',
        '## Contract',
        '',
        PL_LEARNED_OFFSET_ACC_CONTRACT,
        '',
        'DIP trans/root velocity/real offset GT are not used. Full-pipeline S4/S5 not measured.',
        '',
        '## Offset Diagnostic',
        '',
        markdown_table(result['offset_diagnostic_table']),
        '',
        '## PL Module Table',
        '',
        markdown_table(result.get('pl_module_table', [])),
        '',
        '## Decision',
        '',
        result['final_decision'],
        '',
    ]
    (output_dir / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Smoke diagnostics for newpl_v7_learned_offset_accaux.')
    parser.add_argument('--pl-cache', default='data/experiments/newpl_v5_official_protocol_20260607/caches/pl_amass_official_init36/pl_curve_cache_manifest.json')
    parser.add_argument('--gt-control-cache', default='data/dataset_work/GTControlCache/amass_train/gt_control_cache_manifest.json')
    parser.add_argument('--dataset-label', default='AMASS smoke')
    parser.add_argument('--output-dir', default='data/experiments/newpl_v7_learned_offset_accaux_20260612')
    parser.add_argument('--init-checkpoint', default='data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt')
    parser.add_argument('--newpl-v4-checkpoint', default='data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-sequences', type=int, default=4)
    parser.add_argument('--max-train-sequences', type=int, default=4)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--dt', type=float, default=1.0 / 60.0)
    parser.add_argument('--offset-max', type=float, default=0.30)
    parser.add_argument('--acc-scale', type=float, default=30.0)
    parser.add_argument('--imu-acc-weight', type=float, default=0.01)
    parser.add_argument('--offset-prior-weight', type=float, default=0.001)
    parser.add_argument('--offset-lr', type=float, default=1e-3)
    parser.add_argument('--joint-lr', type=float, default=1e-5)
    parser.add_argument('--stage0-steps', type=int, default=50)
    parser.add_argument('--stage1-steps', type=int, default=50)
    parser.add_argument('--stage2-epochs', type=int, default=1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--meaningful-prb-tolerance-cm', type=float, default=0.05)
    parser.add_argument('--meaningful-gravity-tolerance-deg', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=20260612)
    parser.add_argument('--skip-stage2', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(json_safe(vars(args)), indent=2) + '\n')

    records, manifest = load_pl_records(args.pl_cache, max_sequences=args.max_sequences)
    if args.gt_control_cache:
        attach_pl_target_controls(records, args.gt_control_cache)

    stage0 = stage0_identifiability(args, records, device, output_dir)
    (output_dir / 'stage0_identifiability.json').write_text(json.dumps(json_safe(stage0), indent=2) + '\n')

    stage1, stage1_model = stage1_freeze_offset(
        args,
        records,
        device,
        output_dir,
        init_offset=torch.as_tensor(stage0['learned_offset']).float(),
    )
    (output_dir / 'stage1_freeze_offset.json').write_text(json.dumps(json_safe(stage1), indent=2) + '\n')

    stage2 = {'stage': 'stage2_tiny_joint_training', 'status': 'skipped'}
    meaningful = (
        stage0['all_finite']
        and stage1['all_finite']
        and stage0['learned_vs_zero_improvement'] > 0.0
        and stage0['random_vs_zero_degradation'] > 0.0
    )
    if meaningful and not args.skip_stage2:
        stage2 = stage2_tiny_joint(args, records, device, output_dir, stage1_model)
    else:
        stage2['reason'] = 'Stage 0/1 not meaningful or --skip-stage2 set'
        stage2['pl_module_table'] = evaluate_versions(args, records, device, output_dir, v7_model=stage1_model)
    (output_dir / 'stage2_tiny_joint.json').write_text(json.dumps(json_safe(stage2), indent=2) + '\n')

    offset_table = offset_diagnostic_table(stage0)
    pl_table = stage2.get('pl_module_table', [])
    decision = 'reject current formulation'
    if meaningful and stage2.get('all_finite', False):
        v7_rows = [row for row in pl_table if row['Version'] == 'newpl_v7_learned_offset_accaux']
        official_rows = [row for row in pl_table if row['Version'] == 'official PL baseline']
        if v7_rows and official_rows and isinstance(v7_rows[0].get('pRB L2 cm'), float):
            p_ok = v7_rows[0]['pRB L2 cm'] <= official_rows[0]['pRB L2 cm'] + args.meaningful_prb_tolerance_cm
            g_ok = v7_rows[0]['gR1 angle deg'] <= official_rows[0]['gR1 angle deg'] + args.meaningful_gravity_tolerance_deg
            norm_ok = stage2['learned_offset_norm']['offset_norm_p95_m'] < args.offset_max * 0.95
            decision = 'continue to full AMASS->DIP' if (p_ok and g_ok and norm_ok) else 'diagnostic only'
        else:
            decision = 'diagnostic only'
    elif stage0['all_finite'] and stage1['all_finite']:
        decision = 'diagnostic only'
    result = {
        'status': 'ok',
        'model_variant': 'newpl_v7_learned_offset_accaux',
        'coordinate_contract': PL_LEARNED_OFFSET_ACC_CONTRACT,
        'forbidden_supervision': ['DIP trans', 'DIP root velocity', 'real offset GT for DIP/TotalCapture'],
        'full_pipeline_S4_S5': 'not measured',
        'source_manifest': manifest,
        'stage0': stage0,
        'stage1': stage1,
        'stage2': stage2,
        'offset_diagnostic_table': offset_table,
        'pl_module_table': pl_table,
        'final_decision': decision,
        'artifacts': {
            'root': str(output_dir),
            'config': str(output_dir / 'config.json'),
            'stage0_json': str(output_dir / 'stage0_identifiability.json'),
            'stage1_json': str(output_dir / 'stage1_freeze_offset.json'),
            'stage2_json': str(output_dir / 'stage2_tiny_joint.json'),
            'summary_md': str(output_dir / 'summary.md'),
        },
    }
    (output_dir / 'summary.json').write_text(json.dumps(json_safe(result), indent=2) + '\n')
    write_summary(output_dir, result)
    print(json.dumps({
        'status': 'ok',
        'output_dir': str(output_dir),
        'stage0_learned_vs_zero_improvement': stage0['learned_vs_zero_improvement'],
        'stage0_random_vs_zero_degradation': stage0['random_vs_zero_degradation'],
        'stage1_acc_improvement_l2_mps2': stage1['acc_improvement_l2_mps2'],
        'stage2_status': stage2.get('status', 'ok'),
        'final_decision': decision,
    }, indent=2))


if __name__ == '__main__':
    main()
