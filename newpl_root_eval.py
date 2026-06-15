import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, load_records
from net import GPNet
from newpl_root import NewPLRootModule, extend_base_pl, pl_root_target_from_pose_tran, root_velocity_target_from_pose_tran
from pl_curve import build_pl_curve_model, normalize_gravity, pl_init_feature_from_pose, pl_input_feature, pl_target_from_pose


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


def parse_version_spec(spec):
    name, value = spec.split('=', 1)
    delay = 0
    if ',delay=' in value:
        value, delay_text = value.rsplit(',delay=', 1)
        delay = int(delay_text)
        if delay < 0:
            raise ValueError(f'PL output delay must be non-negative for {name}, got {delay}.')
    return name, value, delay


def load_version(spec):
    name, value, delay = parse_version_spec(spec)
    if value == 'official':
        return {'name': name, 'kind': 'official', 'path': None, 'model': None, 'config': None, 'pl_output_delay': delay}
    checkpoint = torch.load(value, map_location=DEVICE)
    model_type = checkpoint.get('model_type', '')
    if model_type == 'newpl_root_v1':
        config = checkpoint.get('config', {})
        model = NewPLRootModule(
            hidden_size=int(config.get('hidden_size', 512)),
            tail_update=int(config.get('tail_length', 4)),
            residual_scale=float(config.get('residual_scale', 0.005)),
            dropout=float(config.get('dropout', 0.4)),
            condition_scale=float(config.get('condition_scale', 1.0)),
        ).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return {'name': name, 'kind': 'newpl_root_v1', 'path': value, 'model': model, 'config': config, 'pl_output_delay': delay}
    model = build_pl_curve_model(checkpoint.get('config', {})).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return {'name': name, 'kind': 'pl_curve_v1', 'path': value, 'model': model, 'config': checkpoint.get('config', {}), 'pl_output_delay': delay}


def load_versions(specs):
    loaded_by_value = {}
    versions = []
    for spec in specs:
        name, value, delay = parse_version_spec(spec)
        if value in loaded_by_value:
            base = loaded_by_value[value]
            version = dict(base)
            version.update({'name': name, 'pl_output_delay': delay})
        else:
            version = load_version(spec)
            loaded_by_value[value] = {k: v for k, v in version.items() if k != 'name'}
        versions.append(version)
    return versions


def angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target, avoid_nan=True)
    cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def finite_diff(x):
    if x.shape[0] < 2:
        return x.new_zeros((0,) + x.shape[1:])
    return x[1:] - x[:-1]


def jitter(x):
    if x.shape[0] < 3:
        return x.new_zeros(())
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]).norm(dim=-1).mean()


def metrics_for_prediction(pred, target, root_vel_gt_available):
    pred = pred.to(target.device, target.dtype)
    p_err = pred[..., :15] - target[..., :15]
    p_leaf = p_err.reshape(p_err.shape[:-1] + (5, 3)).norm(dim=-1) * 100.0
    metrics = {
        'pRB_L1_cm': float(p_err.abs().mean() * 100.0),
        'pRB_L2_cm': float(p_leaf.mean()),
        'per_leaf_pRB_L2_cm': [float(v) for v in p_leaf.mean(dim=0)],
        'pRB_temporal_velocity_error_cm_per_frame': float((finite_diff(pred[..., :15]) - finite_diff(target[..., :15])).reshape(-1, 5, 3).norm(dim=-1).mean() * 100.0) if pred.shape[0] > 1 else 0.0,
        'pRB_smooth_jitter_cm': float(jitter(pred[..., :15].reshape(pred.shape[0], 5, 3)).mean() * 100.0) if pred.shape[0] > 2 else 0.0,
        'gR1_angle_deg': float(angle_deg(pred[..., 15:18], target[..., 15:18]).mean()),
        'gR1_temporal_angle_velocity_error_deg_per_frame': float(angle_deg(finite_diff(pred[..., 15:18]), finite_diff(target[..., 15:18])).mean()) if pred.shape[0] > 1 else 0.0,
        'gR1_smooth_jitter': float(jitter(art.math.normalize_tensor(pred[..., 15:18], avoid_nan=True))) if pred.shape[0] > 2 else 0.0,
    }
    if pred.shape[-1] < 21:
        metrics.update({
            'root_vel_status': 'not applicable',
            'root_vel_L1': None,
            'root_vel_L2': None,
            'root_vel_direction_angle_deg': None,
            'root_vel_smooth_jitter': None,
        })
    elif not root_vel_gt_available:
        metrics.update({
            'root_vel_status': 'root_vel GT not available',
            'root_vel_L1': None,
            'root_vel_L2': None,
            'root_vel_direction_angle_deg': None,
            'root_vel_smooth_jitter': float(jitter(pred[..., 18:21])) if pred.shape[0] > 2 else 0.0,
        })
    else:
        rv_pred = pred[..., 18:21]
        rv_target = target[..., 18:21]
        rv_norm = rv_pred.norm(dim=-1) * rv_target.norm(dim=-1)
        valid_dir = rv_norm > 1e-6
        if valid_dir.any():
            rv_angle = angle_deg(rv_pred[valid_dir], rv_target[valid_dir]).mean()
        else:
            rv_angle = rv_pred.new_zeros(())
        metrics.update({
            'root_vel_status': 'ok',
            'root_vel_L1': float((rv_pred - rv_target).abs().mean()),
            'root_vel_L2': float((rv_pred - rv_target).norm(dim=-1).mean()),
            'root_vel_direction_angle_deg': float(rv_angle),
            'root_vel_smooth_jitter': float(jitter(rv_pred)) if pred.shape[0] > 2 else 0.0,
        })
    return metrics


def align_future_output(pred, target, delay):
    """Evaluate output at t+delay against target at t."""
    if delay <= 0:
        return pred, target
    if pred.shape[0] <= delay:
        raise ValueError(f'Cannot apply delay={delay} to sequence with {pred.shape[0]} frames.')
    return pred[delay:], target[:-delay]


@torch.no_grad()
def run_pipeline_translation(record, version, imu_input_mode):
    pl_curve = version['model'] if version['kind'] in ('pl_curve_v1', 'newpl_root_v1') else None
    net = GPNet(
        pl_backend='curve_v1' if pl_curve is not None else 'original',
        pl_curve_module=pl_curve,
    ).eval().to(DEVICE)
    init_pose = record['pose_gt'][0]
    net.rnn_initialize(init_pose, offset_r=record.get('offset_r'))
    pose = torch.zeros_like(record['pose_gt'])
    tran = torch.zeros_like(record['tran_gt'])
    a_seq, w_seq, R_seq = selected_imu_fields(record, imu_input_mode)
    for i in range(record['pose_gt'].shape[0]):
        pose[i], tran[i] = net.forward_frame(
            a_seq[i].to(DEVICE),
            w_seq[i].to(DEVICE),
            R_seq[i].to(DEVICE),
        )
    return pose.cpu(), tran.cpu()


def root_velocity_metrics(pred_root_vel, target_root_vel, root_vel_gt_available):
    if not root_vel_gt_available:
        return {
            'root_vel_status': 'root_vel GT not available',
            'root_vel_L1': None,
            'root_vel_L2': None,
            'root_vel_direction_angle_deg': None,
            'root_vel_smooth_jitter': float(jitter(pred_root_vel)) if pred_root_vel is not None and pred_root_vel.shape[0] > 2 else None,
        }
    diff = pred_root_vel - target_root_vel
    norm_product = pred_root_vel.norm(dim=-1) * target_root_vel.norm(dim=-1)
    valid_dir = norm_product > 1e-6
    if valid_dir.any():
        direction = angle_deg(pred_root_vel[valid_dir], target_root_vel[valid_dir]).mean()
    else:
        direction = pred_root_vel.new_zeros(())
    return {
        'root_vel_status': 'ok',
        'root_vel_L1': float(diff.abs().mean()),
        'root_vel_L2': float(diff.norm(dim=-1).mean()),
        'root_vel_direction_angle_deg': float(direction),
        'root_vel_smooth_jitter': float(jitter(pred_root_vel)) if pred_root_vel.shape[0] > 2 else 0.0,
    }


def average_metrics(rows):
    out = {}
    numeric_keys = [k for k, v in rows[0]['metrics'].items() if isinstance(v, (int, float))]
    nullable_keys = [k for k, v in rows[0]['metrics'].items() if v is None]
    for key in numeric_keys:
        out[key] = sum(float(row['metrics'][key]) for row in rows) / len(rows)
    for key in nullable_keys:
        vals = [row['metrics'][key] for row in rows if row['metrics'][key] is not None]
        out[key] = sum(float(v) for v in vals) / len(vals) if vals else None
    if 'per_leaf_pRB_L2_cm' in rows[0]['metrics']:
        leaf = torch.tensor([row['metrics']['per_leaf_pRB_L2_cm'] for row in rows])
        out['per_leaf_pRB_L2_cm'] = [float(v) for v in leaf.mean(dim=0)]
    statuses = sorted({row['metrics']['root_vel_status'] for row in rows})
    out['root_vel_status'] = statuses[0] if len(statuses) == 1 else ', '.join(statuses)
    return out


def table_value(value):
    if value is None:
        return 'not available'
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


def make_tables(result, dataset_label):
    metric_table = []
    pl_output_table = []
    root_velocity_table = []
    per_leaf_table = []
    gt_root = result.get('gt_root_velocity')
    root_velocity_table.append({
        'Dataset': dataset_label,
        'Version': 'GT',
        'root_vel source': gt_root.get('source', 'not available') if gt_root else 'not available',
        'root_vel L1 ↓': '0.000000' if gt_root and gt_root.get('available') else 'not available',
        'root_vel L2 ↓': '0.000000' if gt_root and gt_root.get('available') else 'not available',
        'root_vel angle ↓': '0.000000' if gt_root and gt_root.get('available') else 'not available',
        'Notes': gt_root.get('notes', '') if gt_root else '',
    })
    for version in result['versions']:
        agg = version['aggregate']
        metric_table.append({
            'Version': version['name'],
            'pRB L1 cm ↓': table_value(agg.get('pRB_L1_cm')),
            'pRB L2 cm ↓': table_value(agg.get('pRB_L2_cm')),
            'gR1 angle deg ↓': table_value(agg.get('gR1_angle_deg')),
            'root_vel L1 ↓': table_value(agg.get('root_vel_L1')) if agg.get('root_vel_status') == 'ok' else agg.get('root_vel_status'),
            'root_vel L2 ↓': table_value(agg.get('root_vel_L2')) if agg.get('root_vel_status') == 'ok' else agg.get('root_vel_status'),
            'Notes': f"{version['kind']}; delay={version.get('pl_output_delay', 0)}",
        })
        pl_output_table.append({
            'Dataset': dataset_label,
            'Version': version['name'],
            'pRB L1 cm ↓': table_value(agg.get('pRB_L1_cm')),
            'pRB L2 cm ↓': table_value(agg.get('pRB_L2_cm')),
            'gR1 angle deg ↓': table_value(agg.get('gR1_angle_deg')),
            'Notes': f"{version['kind']}; delay={version.get('pl_output_delay', 0)}",
        })
        root_agg = version.get('root_velocity_aggregate', {})
        root_velocity_table.append({
            'Dataset': dataset_label,
            'Version': version['name'],
            'root_vel source': version.get('root_velocity_source', 'not available'),
            'root_vel L1 ↓': table_value(root_agg.get('root_vel_L1')) if root_agg.get('root_vel_status') == 'ok' else root_agg.get('root_vel_status', 'not available'),
            'root_vel L2 ↓': table_value(root_agg.get('root_vel_L2')) if root_agg.get('root_vel_status') == 'ok' else root_agg.get('root_vel_status', 'not available'),
            'root_vel angle ↓': table_value(root_agg.get('root_vel_direction_angle_deg')) if root_agg.get('root_vel_status') == 'ok' else root_agg.get('root_vel_status', 'not available'),
            'Notes': version.get('root_velocity_notes', ''),
        })
        leaves = agg['per_leaf_pRB_L2_cm']
        per_leaf_table.append({
            'Dataset': dataset_label,
            'Version': version['name'],
            'leaf_1 cm ↓': table_value(leaves[0]),
            'leaf_2 cm ↓': table_value(leaves[1]),
            'leaf_3 cm ↓': table_value(leaves[2]),
            'leaf_4 cm ↓': table_value(leaves[3]),
            'leaf_5 cm ↓': table_value(leaves[4]),
            'Mean': table_value(sum(leaves) / len(leaves)),
        })
    return metric_table, pl_output_table, root_velocity_table, per_leaf_table


@torch.no_grad()
def evaluate_version(version, records, gpnet, body_model, args):
    rows = []
    root_rows = []
    for record in records:
        features = build_features(record, args.imu_input_mode).to(DEVICE)
        pose = record['pose_gt'].float()
        root_vel_gt_available = args.dataset in ('amass', 'totalcapture') and args.root_vel_gt
        target_root_vel = None
        if root_vel_gt_available:
            target = pl_root_target_from_pose_tran(
                pose.to(DEVICE),
                record['tran_gt'].float().to(DEVICE),
                body_model,
                dt=args.dt,
            )
            target_root_vel = target[..., 18:21].detach().cpu()
        else:
            target18 = normalize_gravity(pl_target_from_pose(pose.to(DEVICE), body_model).float())
            target = torch.cat((target18, torch.zeros(target18.shape[:-1] + (3,), device=target18.device)), dim=-1)
        base18 = base_pl_outputs(gpnet, features, normalize_gravity(target[..., :18])[0])
        if version['kind'] == 'official':
            pred = base18
        else:
            init_feature = init_feature_for_record(
                record,
                pose,
                body_model,
                allow_zero_offset_init=args.allow_zero_offset_init,
            ).to(DEVICE)
            if version['kind'] == 'newpl_root_v1':
                pred = version['model'].forward_sequence(features, extend_base_pl(base18), init_feature=init_feature)['pl']
            else:
                pred = version['model'].forward_sequence(features, base18, init_feature=init_feature)['pl']
        delay = int(version.get('pl_output_delay', 0))
        pred_aligned, target_aligned = align_future_output(pred, target, delay)
        root_vel_gt_available_aligned = root_vel_gt_available
        rows.append({
            'name': record['name'],
            'metrics': {
                **metrics_for_prediction(
                    pred_aligned.detach().cpu(),
                    target_aligned.detach().cpu(),
                    root_vel_gt_available_aligned,
                ),
                'pl_output_delay_frames': delay,
                'evaluated_frames': int(pred_aligned.shape[0]),
                'source_frames': int(pred.shape[0]),
            },
        })
        if version['kind'] == 'newpl_root_v1':
            root_pred = pred.detach().cpu()[..., 18:21]
            root_source = 'direct newpl_root_v1 root_vel head'
            root_notes = 'module output'
        elif not root_vel_gt_available:
            root_pred = None
            root_source = 'baseline root_vel not comparable because GT root_vel is not available'
            root_notes = 'no baseline velocity metric computed'
        else:
            _, pipeline_tran = run_pipeline_translation(record, version, args.imu_input_mode)
            root_pred = root_velocity_target_from_pose_tran(
                pose,
                pipeline_tran,
                dt=args.dt,
            ).detach().cpu()
            root_source = 'finite difference of final pipeline translation, projected to GT root frame'
            root_notes = 'baseline velocity from official pipeline trajectory'
        root_rows.append({
            'name': record['name'],
            'root_velocity_source': root_source,
            'root_velocity_notes': root_notes,
            'metrics': root_velocity_metrics(root_pred, target_root_vel, root_vel_gt_available),
        })
    return rows, root_rows


@torch.no_grad()
def evaluate_versions(versions, records, gpnet, body_model, args):
    rows_by_version = {version['name']: [] for version in versions}
    root_rows_by_version = {version['name']: [] for version in versions}
    for record in records:
        features = build_features(record, args.imu_input_mode).to(DEVICE)
        pose = record['pose_gt'].float()
        root_vel_gt_available = args.dataset in ('amass', 'totalcapture') and args.root_vel_gt
        target_root_vel = None
        if root_vel_gt_available:
            target = pl_root_target_from_pose_tran(
                pose.to(DEVICE),
                record['tran_gt'].float().to(DEVICE),
                body_model,
                dt=args.dt,
            )
            target_root_vel = target[..., 18:21].detach().cpu()
        else:
            target18 = normalize_gravity(pl_target_from_pose(pose.to(DEVICE), body_model).float())
            target = torch.cat((target18, torch.zeros(target18.shape[:-1] + (3,), device=target18.device)), dim=-1)
        base18 = base_pl_outputs(gpnet, features, normalize_gravity(target[..., :18])[0])
        init_feature = None
        raw_pred_cache = {}
        for version in versions:
            raw_key = version['path'] if version['kind'] != 'official' else 'official'
            if raw_key not in raw_pred_cache:
                if version['kind'] == 'official':
                    raw_pred_cache[raw_key] = base18
                else:
                    if init_feature is None:
                        init_feature = init_feature_for_record(
                            record,
                            pose,
                            body_model,
                            allow_zero_offset_init=args.allow_zero_offset_init,
                        ).to(DEVICE)
                    if version['kind'] == 'newpl_root_v1':
                        raw_pred_cache[raw_key] = version['model'].forward_sequence(
                            features,
                            extend_base_pl(base18),
                            init_feature=init_feature,
                        )['pl']
                    else:
                        raw_pred_cache[raw_key] = version['model'].forward_sequence(
                            features,
                            base18,
                            init_feature=init_feature,
                        )['pl']
            pred = raw_pred_cache[raw_key]
            delay = int(version.get('pl_output_delay', 0))
            pred_aligned, target_aligned = align_future_output(pred, target, delay)
            rows_by_version[version['name']].append({
                'name': record['name'],
                'metrics': {
                    **metrics_for_prediction(
                        pred_aligned.detach().cpu(),
                        target_aligned.detach().cpu(),
                        root_vel_gt_available,
                    ),
                    'pl_output_delay_frames': delay,
                    'evaluated_frames': int(pred_aligned.shape[0]),
                    'source_frames': int(pred.shape[0]),
                },
            })
            if version['kind'] == 'newpl_root_v1':
                root_pred = pred.detach().cpu()[..., 18:21]
                if delay and root_vel_gt_available:
                    root_pred, root_target = align_future_output(root_pred, target_root_vel, delay)
                else:
                    root_target = target_root_vel
                root_source = 'direct newpl_root_v1 root_vel head'
                root_notes = f'module output; delay={delay}'
            elif not root_vel_gt_available:
                root_pred = None
                root_target = target_root_vel
                root_source = 'baseline root_vel not comparable because GT root_vel is not available'
                root_notes = 'no baseline velocity metric computed'
            else:
                _, pipeline_tran = run_pipeline_translation(record, version, args.imu_input_mode)
                root_pred = root_velocity_target_from_pose_tran(
                    pose,
                    pipeline_tran,
                    dt=args.dt,
                ).detach().cpu()
                root_target = target_root_vel
                root_source = 'finite difference of final pipeline translation, projected to GT root frame'
                root_notes = 'baseline velocity from official pipeline trajectory'
            root_rows_by_version[version['name']].append({
                'name': record['name'],
                'root_velocity_source': root_source,
                'root_velocity_notes': root_notes,
                'metrics': root_velocity_metrics(root_pred, root_target, root_vel_gt_available),
            })
    evaluated = []
    for version in versions:
        version = dict(version)
        rows = rows_by_version[version['name']]
        root_rows = root_rows_by_version[version['name']]
        version.update({'rows': rows, 'aggregate': average_metrics(rows)})
        root_aggregate = average_metrics(root_rows)
        version.update({
            'root_velocity_rows': root_rows,
            'root_velocity_aggregate': root_aggregate,
            'root_velocity_source': root_rows[0]['root_velocity_source'] if root_rows else 'not available',
            'root_velocity_notes': root_rows[0]['root_velocity_notes'] if root_rows else '',
        })
        evaluated.append(version)
    return evaluated


def main():
    parser = argparse.ArgumentParser(description='Module-level evaluation for NewPL-root and PL baselines.')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--dataset', choices=('amass', 'totalcapture', 'dip'), required=True)
    parser.add_argument('--dataset-label', default='')
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='official')
    parser.add_argument('--version', action='append', required=True, help='NAME=official or NAME=/path/checkpoint.pt')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--delay-mode', choices=('future_output',), default='future_output')
    parser.add_argument('--root-vel-gt', action='store_true')
    parser.add_argument('--allow-zero-offset-init', action='store_true')
    parser.add_argument('--dt', type=float, default=1.0 / 60.0)
    args = parser.parse_args()
    if args.dataset == 'dip' and args.root_vel_gt:
        raise RuntimeError('DIP root velocity GT evaluation is not allowed.')
    records, manifest = load_records(args.cache, max_sequences=args.max_eval_sequences)
    gpnet = GPNet().eval().to(DEVICE)
    for parameter in gpnet.parameters():
        parameter.requires_grad_(False)
    body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
    versions = evaluate_versions(load_versions(args.version), records, gpnet, body_model, args)
    gt_root_velocity = {
        'available': bool(args.root_vel_gt and args.dataset != 'dip'),
        'source': 'translation GT finite difference projected to GT root frame' if args.root_vel_gt and args.dataset != 'dip' else 'not available',
        'notes': '' if args.root_vel_gt and args.dataset != 'dip' else 'root_vel GT not available',
    }
    result = {
        'status': 'ok',
        'cache': args.cache,
        'dataset': args.dataset,
        'dataset_label': args.dataset_label or args.dataset,
        'imu_input_mode': args.imu_input_mode,
        'delay_mode': args.delay_mode,
        'root_vel_gt_requested': args.root_vel_gt,
        'root_vel_gt_used': bool(args.root_vel_gt and args.dataset != 'dip'),
        'gt_root_velocity': gt_root_velocity,
        'manifest': manifest,
        'versions': [
            {k: v for k, v in version.items() if k not in ('model',)}
            for version in versions
        ],
    }
    metric_table, pl_output_table, root_velocity_table, per_leaf_table = make_tables(result, args.dataset_label or args.dataset)
    result['metric_table'] = metric_table
    result['pl_output_comparison_table'] = pl_output_table
    result['root_velocity_comparison_table'] = root_velocity_table
    result['per_leaf_table'] = per_leaf_table
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'ok', 'output_json': str(output_path), 'versions': [v['name'] for v in versions]}, indent=2))


if __name__ == '__main__':
    main()
