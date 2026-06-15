import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE
from pl_curve import build_pl_curve_model, normalize_gravity
from pl_next_control_cache import _central_acceleration, _central_velocity, _shift_next
from pl_next_control_train import load_next_records


def angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target, avoid_nan=True)
    cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def masked_mean(x, mask):
    if mask is None:
        return x.mean()
    values = x.masked_select(mask.to(x.device))
    if values.numel() == 0:
        return x.new_zeros(())
    return values.mean()


def jitter(x):
    if x.shape[0] < 3:
        return x.new_zeros(())
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]).reshape(x.shape[0] - 2, -1, 3).norm(dim=-1).mean()


def pl_metrics(pred, target, mask=None):
    pred = pred.to(target.device, target.dtype)
    err = pred[..., :15] - target[..., :15]
    leaf = err.reshape(err.shape[:-1] + (5, 3)).norm(dim=-1)
    if mask is not None:
        leaf_values = leaf.masked_select(mask.unsqueeze(-1).expand_as(leaf))
        p_l1_values = err.abs().masked_select(mask.unsqueeze(-1).expand_as(err))
        g_values = angle_deg(pred[..., 15:], target[..., 15:]).masked_select(mask)
        per_leaf = []
        for leaf_idx in range(5):
            values = leaf[..., leaf_idx].masked_select(mask)
            per_leaf.append(float(values.mean() * 100.0) if values.numel() else 0.0)
    else:
        leaf_values = leaf.reshape(-1)
        p_l1_values = err.abs().reshape(-1)
        g_values = angle_deg(pred[..., 15:], target[..., 15:]).reshape(-1)
        per_leaf = [float(leaf[..., leaf_idx].mean() * 100.0) for leaf_idx in range(5)]
    return {
        'pRB_L1_cm': float(p_l1_values.mean() * 100.0) if p_l1_values.numel() else 0.0,
        'pRB_L2_cm': float(leaf_values.mean() * 100.0) if leaf_values.numel() else 0.0,
        'per_leaf_pRB_L2_cm': per_leaf,
        'gR1_angle_deg': float(g_values.mean()) if g_values.numel() else 0.0,
        'pRB_smooth_jitter_cm': float(jitter(pred[..., :15]) * 100.0) if pred.shape[0] > 2 else 0.0,
        'gR1_smooth_jitter': float(jitter(art.math.normalize_tensor(pred[..., 15:], avoid_nan=True))) if pred.shape[0] > 2 else 0.0,
    }


def dynamics_metrics(pred_vel, pred_acc, gt_vel, gt_acc, mask):
    vel_leaf = (pred_vel[..., :15] - gt_vel[..., :15]).reshape(pred_vel.shape[:-1] + (5, 3)).norm(dim=-1)
    acc_leaf = (pred_acc[..., :15] - gt_acc[..., :15]).reshape(pred_acc.shape[:-1] + (5, 3)).norm(dim=-1)
    vel_l1 = (pred_vel[..., :15] - gt_vel[..., :15]).abs()
    acc_l1 = (pred_acc[..., :15] - gt_acc[..., :15]).abs()
    g_vel = (pred_vel[..., 15:] - gt_vel[..., 15:]).norm(dim=-1)
    g_acc = (pred_acc[..., 15:] - gt_acc[..., 15:]).norm(dim=-1)
    mask_leaf = mask.unsqueeze(-1).expand_as(vel_leaf)
    mask_vec = mask.unsqueeze(-1).expand_as(vel_l1)
    return {
        'pRB_vel_L1_cm_s': float(vel_l1.masked_select(mask_vec).mean() * 100.0) if mask.any() else 0.0,
        'pRB_vel_L2_cm_s': float(vel_leaf.masked_select(mask_leaf).mean() * 100.0) if mask.any() else 0.0,
        'pRB_acc_L1_cm_s2': float(acc_l1.masked_select(mask_vec).mean() * 100.0) if mask.any() else 0.0,
        'pRB_acc_L2_cm_s2': float(acc_leaf.masked_select(mask_leaf).mean() * 100.0) if mask.any() else 0.0,
        'gR1_vel_vector_L2': float(g_vel.masked_select(mask).mean()) if mask.any() else 0.0,
        'gR1_acc_vector_L2': float(g_acc.masked_select(mask).mean()) if mask.any() else 0.0,
        'pRB_vel_smooth_jitter_cm_s': float(jitter(pred_vel[..., :15]) * 100.0) if pred_vel.shape[0] > 2 else 0.0,
        'pRB_acc_smooth_jitter_cm_s2': float(jitter(pred_acc[..., :15]) * 100.0) if pred_acc.shape[0] > 2 else 0.0,
    }


def control_state_metrics(pred, target, mask=None):
    pred = pred.to(target.device, target.dtype)
    err = pred[..., :15] - target[..., :15]
    leaf = err.reshape(err.shape[:-1] + (5, 3)).norm(dim=-1)
    g = angle_deg(pred[..., 15:], target[..., 15:])
    if mask is not None:
        leaf_mask = mask.to(target.device)
        while leaf_mask.dim() < leaf.dim():
            leaf_mask = leaf_mask.unsqueeze(-1)
        leaf_values = leaf.masked_select(leaf_mask.expand_as(leaf))
        g_values = g.masked_select(mask.to(target.device).expand_as(g))
    else:
        leaf_values = leaf.reshape(-1)
        g_values = g.reshape(-1)
    return {
        'pRB_L2_cm': float(leaf_values.mean() * 100.0) if leaf_values.numel() else None,
        'gR1_angle_deg': float(g_values.mean()) if g_values.numel() else None,
    }


def control_metrics(output, record):
    if 'new_control' not in output:
        return {
            'current_control_pRB_L2_cm': None,
            'current_control_gR1_angle_deg': None,
            'next_control_pRB_L2_cm': None,
            'next_control_gR1_angle_deg': None,
            'last_control_pRB_L2_cm': None,
            'last_control_gR1_angle_deg': None,
            'tail4_control_pRB_L2_cm': None,
            'tail4_control_gR1_angle_deg': None,
            'source': 'not applicable: cached official PL baseline has no learned controls',
        }
    current = control_state_metrics(output['new_control'].detach().cpu(), record['pl_target_control'])
    if 'next_control' not in output:
        return {
            'current_control_pRB_L2_cm': current['pRB_L2_cm'],
            'current_control_gR1_angle_deg': current['gR1_angle_deg'],
            'next_control_pRB_L2_cm': None,
            'next_control_gR1_angle_deg': None,
            'last_control_pRB_L2_cm': None,
            'last_control_gR1_angle_deg': None,
            'tail4_control_pRB_L2_cm': None,
            'tail4_control_gR1_angle_deg': None,
            'source': 'current control only: baseline NewPL has no one-step preview branch',
        }
    valid_next = record['valid_next_mask'].bool()
    next_control = control_state_metrics(
        output['next_control'].detach().cpu(),
        record['pl_target_control_next'],
        valid_next,
    )
    last_control = control_state_metrics(
        output['last_preview_control'].detach().cpu(),
        record['last_control_target'],
    )
    tail4 = control_state_metrics(
        output['next_tail_control'].detach().cpu(),
        record['tail_control_target'],
        record['tail_control_valid_mask'].bool(),
    )
    return {
        'current_control_pRB_L2_cm': current['pRB_L2_cm'],
        'current_control_gR1_angle_deg': current['gR1_angle_deg'],
        'next_control_pRB_L2_cm': next_control['pRB_L2_cm'],
        'next_control_gR1_angle_deg': next_control['gR1_angle_deg'],
        'last_control_pRB_L2_cm': last_control['pRB_L2_cm'],
        'last_control_gR1_angle_deg': last_control['gR1_angle_deg'],
        'tail4_control_pRB_L2_cm': tail4['pRB_L2_cm'],
        'tail4_control_gR1_angle_deg': tail4['gR1_angle_deg'],
        'source': 'learned current, tail4 preview, and one-step next control',
    }


def average_metric_dict(rows, section):
    keys = rows[0][section].keys()
    out = {}
    for key in keys:
        first = rows[0][section][key]
        if isinstance(first, list):
            vals = torch.tensor([row[section][key] for row in rows], dtype=torch.float32)
            out[key] = [float(v) for v in vals.mean(dim=0)]
        elif isinstance(first, (int, float)):
            out[key] = sum(float(row[section][key]) for row in rows) / len(rows)
        else:
            out[key] = first
    return out


def load_version(spec):
    name, value = spec.split('=', 1)
    if value == 'official':
        return {'name': name, 'kind': 'official', 'path': None, 'model': None, 'notes': 'cached official PL baseline'}
    checkpoint = torch.load(value, map_location=DEVICE)
    if checkpoint.get('model_type') != 'pl_curve_v1':
        raise RuntimeError(f'{value} has unsupported model_type={checkpoint.get("model_type")}.')
    config = checkpoint.get('config', {})
    model = build_pl_curve_model(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return {
        'name': name,
        'kind': checkpoint.get('model_variant', config.get('model_variant', 'pl_curve_v1')),
        'path': value,
        'model': model,
        'config': config,
        'notes': 'checkpoint module output',
    }


def truncate_record(record, max_frames):
    if not max_frames or max_frames <= 0:
        return record
    seq_len = int(record['pl_input'].shape[0])
    if seq_len <= max_frames:
        return record
    out = {}
    for key, value in record.items():
        if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == seq_len:
            out[key] = value[:max_frames].clone()
        else:
            out[key] = value
    out['name'] = f'{record["name"]}[:{max_frames}]'
    return out


def run_model(version, record):
    if version['kind'] == 'official':
        pl = normalize_gravity(record['pl_base'].to(DEVICE))
        return {'pl': pl}
    model = version['model']
    features = record['pl_input'].to(DEVICE)
    base = normalize_gravity(record['pl_base'].to(DEVICE))
    init = record['pl_init_feature'].to(DEVICE)
    return model.forward_sequence(features, base, init_feature=init)


@torch.no_grad()
def evaluate_version(version, records, dt):
    rows = []
    for record in records:
        out = run_model(version, record)
        pred_current = normalize_gravity(out['pl']).detach().cpu()
        target_current = record['pl_target']
        valid_next = record['valid_next_mask'].bool()
        target_next = record['pl_target_next']
        if 'next_pl' in out:
            pred_next = normalize_gravity(out['next_pl']).detach().cpu()
            pred_vel = out['next_pldot'].detach().cpu()
            pred_acc = out['next_plddot'].detach().cpu()
            next_source = 'direct predicted next control decoded by spline'
            dyn_source = 'spline derivatives at predicted next control'
        else:
            pred_next = pred_current
            fd_vel = _central_velocity(pred_current, dt=dt)
            fd_acc = _central_acceleration(pred_current, dt=dt)
            pred_vel = _shift_next(fd_vel)
            pred_acc = _shift_next(fd_acc)
            next_source = 'causal persistence baseline: module output at t used for t+1'
            dyn_source = 'finite differences of module current PL output'
        current = pl_metrics(pred_current, target_current)
        next_metrics = pl_metrics(pred_next, target_next, valid_next)
        dynamics = dynamics_metrics(
            pred_vel,
            pred_acc,
            record['gt_pldot_next'],
            record['gt_plddot_next'],
            valid_next,
        )
        controls = control_metrics(out, record)
        rows.append({
            'name': record['name'],
            'current': current,
            'next': next_metrics,
            'dynamics': dynamics,
            'controls': controls,
            'next_source': next_source,
            'dynamics_source': dyn_source,
            'evaluated_next_frames': int(valid_next.sum()),
            'source_frames': int(valid_next.shape[0]),
        })
    return {
        'rows': rows,
        'aggregate': {
            'current': average_metric_dict(rows, 'current'),
            'next': average_metric_dict(rows, 'next'),
            'dynamics': average_metric_dict(rows, 'dynamics'),
            'controls': average_metric_dict(rows, 'controls'),
        },
        'next_source': rows[0]['next_source'] if rows else '',
        'dynamics_source': rows[0]['dynamics_source'] if rows else '',
    }


def table_value(value):
    if value is None:
        return 'not available'
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


def build_tables(result):
    current_table = []
    next_table = []
    dynamics_table = []
    per_leaf_table = []
    control_table = []
    dataset = result['dataset_label']
    for version in result['versions']:
        cur = version['aggregate']['current']
        nxt = version['aggregate']['next']
        dyn = version['aggregate']['dynamics']
        ctl = version['aggregate']['controls']
        current_table.append({
            'Dataset': dataset,
            'Version': version['name'],
            'pRB_t L1 cm ↓': table_value(cur['pRB_L1_cm']),
            'pRB_t L2 cm ↓': table_value(cur['pRB_L2_cm']),
            'gR1_t angle deg ↓': table_value(cur['gR1_angle_deg']),
            'Notes': version['notes'],
        })
        next_table.append({
            'Dataset': dataset,
            'Version': version['name'],
            'pRB_t+1 L1 cm ↓': table_value(nxt['pRB_L1_cm']),
            'pRB_t+1 L2 cm ↓': table_value(nxt['pRB_L2_cm']),
            'gR1_t+1 angle deg ↓': table_value(nxt['gR1_angle_deg']),
            'Notes': version['next_source'],
        })
        dynamics_table.append({
            'Dataset': dataset,
            'Version': version['name'],
            'pRB_vel L1 cm/s ↓': table_value(dyn['pRB_vel_L1_cm_s']),
            'pRB_vel L2 cm/s ↓': table_value(dyn['pRB_vel_L2_cm_s']),
            'pRB_acc L1 cm/s^2 ↓': table_value(dyn['pRB_acc_L1_cm_s2']),
            'pRB_acc L2 cm/s^2 ↓': table_value(dyn['pRB_acc_L2_cm_s2']),
            'gR1_vel vector L2 ↓': table_value(dyn['gR1_vel_vector_L2']),
            'Notes': version['dynamics_source'],
        })
        leaves = nxt['per_leaf_pRB_L2_cm']
        per_leaf_table.append({
            'Dataset': dataset,
            'Version': version['name'],
            'leaf_1 cm ↓': table_value(leaves[0]),
            'leaf_2 cm ↓': table_value(leaves[1]),
            'leaf_3 cm ↓': table_value(leaves[2]),
            'leaf_4 cm ↓': table_value(leaves[3]),
            'leaf_5 cm ↓': table_value(leaves[4]),
            'Mean': table_value(sum(leaves) / len(leaves)),
        })
        control_table.append({
            'Dataset': dataset,
            'Version': version['name'],
            'current control pRB L2 cm ↓': table_value(ctl['current_control_pRB_L2_cm']),
            'current control gR1 angle deg ↓': table_value(ctl['current_control_gR1_angle_deg']),
            'next control pRB L2 cm ↓': table_value(ctl['next_control_pRB_L2_cm']),
            'next control gR1 angle deg ↓': table_value(ctl['next_control_gR1_angle_deg']),
            'last preview control pRB L2 cm ↓': table_value(ctl['last_control_pRB_L2_cm']),
            'tail4 control pRB L2 cm ↓': table_value(ctl['tail4_control_pRB_L2_cm']),
            'Notes': ctl['source'],
        })
    return current_table, next_table, dynamics_table, per_leaf_table, control_table


def main():
    parser = argparse.ArgumentParser(description='Module-level evaluation for NewPL v6 next-control and PL baselines.')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--dataset-label', default='')
    parser.add_argument('--version', action='append', required=True, help='NAME=official or NAME=/path/to/checkpoint.pt')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--max-frames-per-sequence', type=int, default=0)
    parser.add_argument('--dt', type=float, default=1.0 / 60.0)
    args = parser.parse_args()
    records, manifest = load_next_records(args.cache, max_sequences=args.max_eval_sequences)
    records = [truncate_record(record, args.max_frames_per_sequence) for record in records]
    versions = []
    for spec in args.version:
        version = load_version(spec)
        evaluated = evaluate_version(version, records, dt=args.dt)
        version = {key: value for key, value in version.items() if key not in ('model',)}
        version.update(evaluated)
        versions.append(version)
    result = {
        'status': 'ok',
        'cache': args.cache,
        'dataset_label': args.dataset_label or Path(args.cache).parent.name,
        'manifest': manifest,
        'evaluation_contract': {
            'current': 'Compare module current PL output against GT pRB_t/gR1_t.',
            'next': 'Fair causal comparison: baselines use output_t as persistence prediction for t+1; v6 uses next_pl_t.',
            'dynamics': 'Baselines use finite differences of current PL outputs; v6 uses spline derivatives decoded at predicted next control.',
            'full_pipeline_11_metrics': False,
            'max_eval_sequences': args.max_eval_sequences,
            'max_frames_per_sequence': args.max_frames_per_sequence,
        },
        'versions': versions,
    }
    current_table, next_table, dynamics_table, per_leaf_table, control_table = build_tables(result)
    result['current_frame_table'] = current_table
    result['next_frame_table'] = next_table
    result['dynamics_table'] = dynamics_table
    result['per_leaf_next_table'] = per_leaf_table
    result['control_table'] = control_table
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': 'ok',
        'output_json': str(out_path),
        'versions': [version['name'] for version in versions],
    }, indent=2))


if __name__ == '__main__':
    main()
