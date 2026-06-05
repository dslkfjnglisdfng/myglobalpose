import argparse
import json
from pathlib import Path

import torch

import articulate as art
from l4_q75_utils import pose_tran_to_q75, q75_to_pose_tran
from l4_tail_update_qstate import UniformCubicBSpline
from l4_train_diverse_short import DEVICE, load_cache_files
from net import GPNet
from newik1_control_point import ik1_target_from_pose, normalize_ik1
from pl_curve import fit_uniform_cubic_spline_controls, normalize_gravity, pl_target_from_pose


DT = 1.0 / 60.0
FPS = 60.0
QUANTILES = (0.5, 0.9, 0.95)


def load_source_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            if 'pose_gt' in data and 'tran_gt' in data:
                pose_gt = data['pose_gt'][seq_idx].float()
                tran_gt = data['tran_gt'][seq_idx].float()
            elif 'q75_gt' in data:
                pose_gt, tran_gt = q75_to_pose_tran(data['q75_gt'][seq_idx].float())
            else:
                raise KeyError(f'{cache_file} has no pose_gt/tran_gt or q75_gt fields')
            records.append({
                'name': str(name),
                'pose_gt': pose_gt,
                'tran_gt': tran_gt,
                'cache_file': str(cache_file),
            })
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def finite_difference_centered(x, order, scale):
    if order == 1:
        if x.shape[0] < 3:
            return torch.empty((0,) + x.shape[1:], dtype=x.dtype)
        return (x[2:] - x[:-2]) * (0.5 * scale)
    if order == 2:
        if x.shape[0] < 3:
            return torch.empty((0,) + x.shape[1:], dtype=x.dtype)
        return (x[2:] - 2.0 * x[1:-1] + x[:-2]) * (scale ** 2)
    raise ValueError(order)


def finite_difference_edge_handled(x, order, scale):
    if order == 1:
        out = torch.zeros_like(x)
        if x.shape[0] < 2:
            return out
        out[1:-1] = (x[2:] - x[:-2]) * (0.5 * scale)
        out[0] = (x[1] - x[0]) * scale
        out[-1] = (x[-1] - x[-2]) * scale
        return out
    if order == 2:
        out = torch.zeros_like(x)
        if x.shape[0] < 3:
            return out
        out[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) * (scale ** 2)
        out[0] = out[1]
        out[-1] = out[-2]
        return out
    raise ValueError(order)


def stats(values):
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'p90': None, 'p95': None, 'max': None}
    x = torch.cat([v.detach().reshape(-1).float().cpu() for v in values if v.numel()])
    if x.numel() == 0:
        return {'count': 0, 'mean': None, 'median': None, 'p90': None, 'p95': None, 'max': None}
    return {
        'count': int(x.numel()),
        'mean': float(x.mean()),
        'median': float(torch.quantile(x, 0.5)),
        'p90': float(torch.quantile(x, 0.9)),
        'p95': float(torch.quantile(x, 0.95)),
        'max': float(x.max()),
    }


def dim_rms(values):
    if not values:
        return []
    flat = [v.detach().reshape(-1, v.shape[-1]).float().cpu() for v in values if v.numel()]
    if not flat:
        return []
    x = torch.cat(flat, dim=0)
    return [float(v) for v in x.square().mean(dim=0).sqrt()]


def cosine_stats(a_values, b_values):
    rows = []
    for a, b in zip(a_values, b_values):
        if a.numel() == 0 or b.numel() == 0:
            continue
        ar = a.reshape(-1, a.shape[-1]).float()
        br = b.reshape(-1, b.shape[-1]).float()
        denom = ar.norm(dim=-1) * br.norm(dim=-1)
        mask = denom > 1e-12
        if mask.any():
            rows.append((ar[mask] * br[mask]).sum(dim=-1) / denom[mask])
    return stats(rows)


def integrate_drift(dot, target):
    if dot.shape[0] < 2 or target.shape[0] < 2:
        return torch.empty(0, dtype=target.dtype)
    increments = torch.cumsum(dot[:-1] * DT, dim=0)
    recon = torch.cat((target[:1], target[:1] + increments), dim=0)
    return (recon - target).reshape(target.shape[0], -1).norm(dim=-1)


def curve_decode(target):
    control = fit_uniform_cubic_spline_controls(target.to(DEVICE)).cpu()
    spline = UniformCubicBSpline(DT)
    curve, dot, ddot = spline(control, return_derivatives=True)
    return control, curve.cpu(), dot.cpu(), ddot.cpu()


def sequence_metrics(target):
    control, curve, dot_curve, ddot_curve = curve_decode(target)
    dot_fd_center = finite_difference_centered(target, 1, FPS)
    ddot_fd_center = finite_difference_centered(target, 2, FPS)
    dot_fd_edge = finite_difference_edge_handled(target, 1, FPS)
    ddot_fd_edge = finite_difference_edge_handled(target, 2, FPS)
    return {
        'control_shape': list(control.shape),
        'reconstruction_l2': (curve - target).reshape(target.shape[0], -1).norm(dim=-1),
        'dot_curve_center': dot_curve[1:-1],
        'ddot_curve_center': ddot_curve[1:-1],
        'dot_fd_center': dot_fd_center,
        'ddot_fd_center': ddot_fd_center,
        'dot_curve_edge': dot_curve,
        'ddot_curve_edge': ddot_curve,
        'dot_fd_edge': dot_fd_edge,
        'ddot_fd_edge': ddot_fd_edge,
        'dot_agreement_l2': (dot_curve[1:-1] - dot_fd_center).reshape(max(0, target.shape[0] - 2), -1).norm(dim=-1),
        'ddot_agreement_l2': (ddot_curve[1:-1] - ddot_fd_center).reshape(max(0, target.shape[0] - 2), -1).norm(dim=-1),
        'dot_dim_error': dot_curve[1:-1] - dot_fd_center,
        'ddot_dim_error': ddot_curve[1:-1] - ddot_fd_center,
        'dot_curve_jitter': (dot_curve[1:] - dot_curve[:-1]).reshape(max(0, target.shape[0] - 1), -1).norm(dim=-1),
        'dot_fd_jitter': (dot_fd_edge[1:] - dot_fd_edge[:-1]).reshape(max(0, target.shape[0] - 1), -1).norm(dim=-1),
        'ddot_curve_jitter': (ddot_curve[1:] - ddot_curve[:-1]).reshape(max(0, target.shape[0] - 1), -1).norm(dim=-1),
        'ddot_fd_jitter': (ddot_fd_edge[1:] - ddot_fd_edge[:-1]).reshape(max(0, target.shape[0] - 1), -1).norm(dim=-1),
        'dot_curve_l2': dot_curve.reshape(target.shape[0], -1).norm(dim=-1),
        'dot_fd_l2': dot_fd_edge.reshape(target.shape[0], -1).norm(dim=-1),
        'ddot_curve_l2': ddot_curve.reshape(target.shape[0], -1).norm(dim=-1),
        'ddot_fd_l2': ddot_fd_edge.reshape(target.shape[0], -1).norm(dim=-1),
        'curve_integrated_drift': integrate_drift(dot_curve, target),
        'fd_integrated_drift': integrate_drift(dot_fd_edge, target),
    }


def summarize_metric_rows(rows):
    dot_curve_center = [row['dot_curve_center'] for row in rows]
    dot_fd_center = [row['dot_fd_center'] for row in rows]
    ddot_curve_center = [row['ddot_curve_center'] for row in rows]
    ddot_fd_center = [row['ddot_fd_center'] for row in rows]
    return {
        'reconstruction_l2': stats([row['reconstruction_l2'] for row in rows]),
        'dot_agreement_l2': stats([row['dot_agreement_l2'] for row in rows]),
        'ddot_agreement_l2': stats([row['ddot_agreement_l2'] for row in rows]),
        'dot_agreement_cosine': cosine_stats(dot_curve_center, dot_fd_center),
        'ddot_agreement_cosine': cosine_stats(ddot_curve_center, ddot_fd_center),
        'dot_error_per_dimension_rms': dim_rms([row['dot_dim_error'] for row in rows]),
        'ddot_error_per_dimension_rms': dim_rms([row['ddot_dim_error'] for row in rows]),
        'per_sequence_dot_agreement_mean_l2': stats([row['dot_agreement_l2'].mean().reshape(1) for row in rows if row['dot_agreement_l2'].numel()]),
        'per_sequence_ddot_agreement_mean_l2': stats([row['ddot_agreement_l2'].mean().reshape(1) for row in rows if row['ddot_agreement_l2'].numel()]),
        'curve_dot_jitter_l2': stats([row['dot_curve_jitter'] for row in rows]),
        'fd_dot_jitter_l2': stats([row['dot_fd_jitter'] for row in rows]),
        'curve_ddot_jitter_l2': stats([row['ddot_curve_jitter'] for row in rows]),
        'fd_ddot_jitter_l2': stats([row['ddot_fd_jitter'] for row in rows]),
        'mean_abs_dot_curve_l2': stats([row['dot_curve_l2'] for row in rows]),
        'mean_abs_dot_fd_l2': stats([row['dot_fd_l2'] for row in rows]),
        'mean_abs_ddot_curve_l2': stats([row['ddot_curve_l2'] for row in rows]),
        'mean_abs_ddot_fd_l2': stats([row['ddot_fd_l2'] for row in rows]),
        'curve_integrated_drift_l2': stats([row['curve_integrated_drift'] for row in rows]),
        'fd_integrated_drift_l2': stats([row['fd_integrated_drift'] for row in rows]),
    }


def audit_split(records, body_model_pl, body_model_ik1, max_frames=0):
    buckets = {
        'PL': {'pRB': [], 'gR1': []},
        'IK1': {'pRJ': [], 'gR2': []},
        'q': {'q75': []},
    }
    samples = []
    for record in records:
        pose = record['pose_gt'][:max_frames] if max_frames else record['pose_gt']
        tran = record['tran_gt'][:max_frames] if max_frames else record['tran_gt']
        if pose.shape[0] < 3:
            continue
        pl_target = normalize_gravity(pl_target_from_pose(pose.to(DEVICE), body_model_pl).float()).cpu()
        ik1_target = normalize_ik1(ik1_target_from_pose(pose.to(DEVICE), body_model_ik1).float()).cpu()
        q75 = pose_tran_to_q75(pose, tran)
        buckets['PL']['pRB'].append(sequence_metrics(pl_target[:, :15]))
        buckets['PL']['gR1'].append(sequence_metrics(pl_target[:, 15:]))
        buckets['IK1']['pRJ'].append(sequence_metrics(ik1_target[:, :69]))
        buckets['IK1']['gR2'].append(sequence_metrics(ik1_target[:, 69:]))
        buckets['q']['q75'].append(sequence_metrics(q75))
        samples.append({'name': record['name'], 'frames': int(pose.shape[0]), 'cache_file': record['cache_file']})
    return {
        'samples': samples,
        'PL': {key: summarize_metric_rows(rows) for key, rows in buckets['PL'].items()},
        'IK1': {key: summarize_metric_rows(rows) for key, rows in buckets['IK1'].items()},
        'q': {key: summarize_metric_rows(rows) for key, rows in buckets['q'].items()},
    }


def smoothness_ratio(module_rows):
    curve = module_rows['curve_dot_jitter_l2']['mean']
    fd = module_rows['fd_dot_jitter_l2']['mean']
    if curve is None or fd in (None, 0.0):
        return None
    return curve / fd


def main():
    parser = argparse.ArgumentParser(description='Audit finite-difference versus spline-derivative targets for PL, IK1, and q75.')
    parser.add_argument('--tc-train-cache', type=Path, default=Path('data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json'))
    parser.add_argument('--tc-val-cache', type=Path, default=Path('data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json'))
    parser.add_argument('--amass-cache', type=Path, default=Path('data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged/baseline_cache_manifest.json'))
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--max-amass-sequences', type=int, default=20)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--output-json', type=Path, default=Path('data/experiments/derivative_source_audit/derivative_source_audit.json'))
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'status': 'started',
        'data_splits': [],
        'dt_used': True,
        'dt': DT,
        'fps': FPS,
        'finite_difference': {
            'fd_1_center': 'dot_fd[t]=(x[t+1]-x[t-1])/(2*dt), interior frames only',
            'fd_2_center': 'ddot_fd[t]=(x[t+1]-2*x[t]+x[t-1])/(dt^2), interior frames only',
            'edge_handled': 'dot endpoints use one-sided differences; ddot endpoints copy nearest interior second difference',
            'rotation_semantics': 'No rotation-matrix elementwise angular velocity is computed. gR1/gR2 are normalized direction-vector derivatives; q75 uses Euler-coordinate differences only as a diagnostic coordinate derivative.',
        },
        'spline': {
            'type': 'uniform cubic B-spline with repeated boundary controls',
            'tail_len': 4,
            'control_fit': 'fit_uniform_cubic_spline_controls solves q[i]=(C[i-1]+4*C[i]+C[i+1])/6 for GT samples.',
            'decode_formula': 'q=(left+4*C+right)/6 at frame knots',
            'derivative': 'analytic',
            'dot_formula': 'qdot=(right-left)/(2*dt) at frame knots',
            'ddot_formula': 'qddot=(left-2*C+right)/(dt^2) at frame knots',
        },
        'semantics': {
            'PL': {'pRB': 'position-like root/body vertex feature', 'gR1': 'normalized direction vector'},
            'IK1': {'pRJ': 'position-like SMPL joint feature with root pose identity', 'gR2': 'normalized direction vector'},
            'q': {'q75': 'root translation + 24x Euler angles; no official qdot/qddot supervision target found in current curve loss'},
            'official_qdot_qddot_target_exists': False,
            'official_qdot_qddot_note': 'CurveStateDecoder/UniformCubicBSpline produce qdot/qddot from control points for smoothness/diagnostics; curve_head_train uses qdot_smooth and qddot_smooth, not GT qdot/qddot target loss.',
        },
    }

    try:
        gpnet = GPNet().eval().to(DEVICE)
        body_model_pl = art.ParametricModel('models/SMPL_male.pkl', vert_mask=gpnet.v_imu, device=DEVICE)
        body_model_ik1 = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)
        splits = [
            ('TotalCapture train S1-S3', args.tc_train_cache, args.max_train_sequences),
            ('TotalCapture val S4', args.tc_val_cache, args.max_val_sequences),
        ]
        if args.max_amass_sequences:
            splits.append(('AMASS sample', args.amass_cache, args.max_amass_sequences))
        for split_name, cache_path, max_sequences in splits:
            records, manifest = load_source_records(cache_path, max_sequences=max_sequences)
            split_result = audit_split(records, body_model_pl, body_model_ik1, max_frames=args.max_frames)
            split_result.update({
                'name': split_name,
                'cache': str(cache_path),
                'manifest_type': None if manifest is None else manifest.get('type'),
                'num_sequences': len(split_result['samples']),
                'max_sequences': int(max_sequences),
                'max_frames': int(args.max_frames),
            })
            result['data_splits'].append(split_result)
        tc_val = next((s for s in result['data_splits'] if s['name'] == 'TotalCapture val S4'), result['data_splits'][0])
        pl_prb = tc_val['PL']['pRB']
        ik_prj = tc_val['IK1']['pRJ']
        pl_ratio = smoothness_ratio(pl_prb)
        ik_ratio = smoothness_ratio(ik_prj)
        result['PL'] = tc_val['PL']
        result['IK1'] = tc_val['IK1']
        result['q'] = tc_val['q']
        result['conclusion'] = {
            'basis_split': tc_val['name'],
            'recommended_dot_source': 'curve derivative for position-like PL/IK1 targets if agreement p95 is acceptable; retain finite-difference direction-vector diagnostics for gR1/gR2 unless explicitly redefining them as angular quantities.',
            'recommended_ddot_source': 'curve derivative for smoothness/regularization; use caution for acceleration supervision because spline ddot can attenuate high-frequency motion.',
            'recommended_loss_weight_adjustment': 'If replacing FD dot/ddot targets with curve derivatives, re-tune derivative weights from the observed magnitude ratios rather than reusing old weights blindly.',
            'pl_pRB_curve_dot_jitter_to_fd_ratio': pl_ratio,
            'ik1_pRJ_curve_dot_jitter_to_fd_ratio': ik_ratio,
            'cache_dot_ddot_targets_in_gt_control_cache': 'recommended for reproducibility if derivative targets become supervised losses; store source=analytic_uniform_cubic_spline_derivative, dt, fps, and edge policy.',
        }
        result['status'] = 'ok'
    except Exception as exc:
        result.update({
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
        })
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': result['status'],
        'output_json': str(args.output_json),
        'num_splits': len(result.get('data_splits', [])),
        'error_type': result.get('error_type'),
        'error': result.get('error'),
    }, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
