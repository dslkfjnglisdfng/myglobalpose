import argparse
import json
import math
from pathlib import Path

import torch
import articulate as art

from l4_train_diverse_short import DEVICE, load_records, rotation_geodesic
from net import GPNet
from full_curve_globalpose_s4_eval import load_full_curve_records
from pl_curve import normalize_gravity, pl_target_from_pose


LEAF_NAMES = ('L_LowArm', 'R_LowArm', 'L_LowLeg', 'R_LowLeg', 'Head')
DEFAULT_S4_CACHE = Path('data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json')
DEFAULT_PROCESSED_CACHE = Path('data/dataset_work/FullCurveGlobalPoseCache/tc_val_Roffset_A_processed/full_curve_globalpose_cache_manifest.json')
DEFAULT_OUTPUT_JSON = Path('data/experiments/official_processed_module_audit/module_output_gt_audit_s4.json')


@torch.no_grad()
def mean_std_min_max(tensor):
    tensor = tensor.detach().float().reshape(-1)
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return {'mean': None, 'std': None, 'min': None, 'max': None}
    return {
        'mean': float(finite.mean()),
        'std': float(finite.std(unbiased=False)) if finite.numel() > 1 else 0.0,
        'min': float(finite.min()),
        'max': float(finite.max()),
    }


@torch.no_grad()
def metric_stats(values):
    values = torch.as_tensor(values).float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {'mean': None, 'median': None, 'std': None, 'min': None, 'max': None, 'count': 0}
    return {
        'mean': float(finite.mean()),
        'median': float(finite.median()),
        'std': float(finite.std(unbiased=False)) if finite.numel() > 1 else 0.0,
        'min': float(finite.min()),
        'max': float(finite.max()),
        'count': int(finite.numel()),
    }


@torch.no_grad()
def l2_cm(pred, target):
    return (pred.float() - target.float()).norm(dim=-1) * 100.0


@torch.no_grad()
def gravity_angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred.float(), avoid_nan=True)
    target = art.math.normalize_tensor(target.float(), avoid_nan=True)
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


@torch.no_grad()
def rotation_angle_deg(pred, target):
    return torch.rad2deg(rotation_geodesic(pred.float(), target.float()))


@torch.no_grad()
def rotation_quality(rot):
    rot = rot.float().reshape(-1, 3, 3)
    eye = torch.eye(3, device=rot.device).expand_as(rot)
    ortho = rot.transpose(-1, -2).matmul(rot)
    return {
        'det': mean_std_min_max(torch.det(rot)),
        'orthogonality_fro': mean_std_min_max((ortho - eye).norm(dim=(-2, -1))),
    }


@torch.no_grad()
def load_processed_imu_by_name(processed_cache):
    if processed_cache is None:
        return {}, None
    full_records, manifest = load_full_curve_records(processed_cache)
    processed_by_name = {}
    for record in full_records:
        imu = record['processed_imu'].float()
        processed_by_name[record['name']] = {
            'l4_aM': imu[..., :18].reshape(-1, 6, 3),
            'l4_wM': imu[..., 18:36].reshape(-1, 6, 3),
            'l4_RMB': imu[..., 36:90].reshape(-1, 6, 3, 3),
        }
    return processed_by_name, manifest


@torch.no_grad()
def attach_processed_fields(record, processed_by_name):
    if all(key in record for key in ('l4_aM', 'l4_wM', 'l4_RMB')):
        return record, 'record_l4_fields'
    if record['name'] not in processed_by_name:
        raise KeyError(f'Record {record["name"]} lacks l4 fields and is absent from processed cache.')
    out = dict(record)
    processed = processed_by_name[record['name']]
    seq_len = record['pose_gt'].shape[0]
    for key, value in processed.items():
        if value.shape[0] < seq_len:
            raise ValueError(f'Processed cache field {key} for {record["name"]} is shorter than baseline record.')
        out[key] = value[:seq_len]
    return out, 'processed_cache_processed_imu'


@torch.no_grad()
def selected_imu_fields(record, mode):
    if mode == 'official':
        return record['aM'], record['wM'], record['RMB']
    if mode == 'processed':
        missing = [key for key in ('l4_aM', 'l4_wM', 'l4_RMB') if key not in record]
        if missing:
            raise KeyError(f'Record {record["name"]} missing processed IMU fields: {missing}')
        return record['l4_aM'], record['l4_wM'], record['l4_RMB']
    raise ValueError(f'Unsupported mode: {mode}')


@torch.no_grad()
def vr_target_from_record(record):
    v_root = record['v_root_vr'].float()
    stationary = record['stationary_prob'].float()
    if v_root.shape[-1] == 3:
        vertical = v_root[..., 1:2]
        horizontal = v_root.clone()
        horizontal[..., 1] = 0.0
        out = torch.cat((vertical, horizontal, stationary), dim=-1)
    elif v_root.shape[-1] == 4:
        out = torch.cat((v_root, stationary), dim=-1)
    else:
        raise ValueError(f'Unsupported v_root_vr shape {tuple(v_root.shape)} for {record["name"]}')
    if out.shape[-1] != 9:
        raise ValueError(f'Expected VR target width 9, got {out.shape[-1]} for {record["name"]}')
    return out


@torch.no_grad()
def build_targets(record, net):
    pose_gt = record['pose_gt'].float()
    pl_body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=net.v_imu, device=DEVICE)
    joint_body_model = art.ParametricModel('models/SMPL_male.pkl', device=DEVICE)

    pl_target = normalize_gravity(pl_target_from_pose(pose_gt.to(DEVICE), pl_body_model).float()).cpu()

    pose_body = pose_gt.to(DEVICE).clone()
    pose_body[:, 0] = torch.eye(3, device=DEVICE)
    global_pose_body, joints_body = joint_body_model.forward_kinematics(pose_body)[:2]
    pRJ_target = joints_body[:, 1:].reshape(pose_gt.shape[0], 69).detach().cpu()
    gR2_target = (-pose_gt[:, 0, :, 1]).detach().cpu()
    ik1_target = torch.cat((pRJ_target, gR2_target), dim=-1)
    ik2_rot_target = global_pose_body[:, net.j_reduce].detach().cpu()

    joints_global = joint_body_model.forward_kinematics(pose_gt.to(DEVICE))[1].detach().cpu()
    joints_root_relative = joints_global - joints_global[:, :1]

    return {
        'pl_target': pl_target,
        'ik1_target': ik1_target,
        'pRJ_target': pRJ_target,
        'gR2_target': gR2_target,
        'ik2_rot_target': ik2_rot_target,
        'body_rot_target': global_pose_body.detach().cpu(),
        'pose_gt': pose_gt,
        'joints_root_relative': joints_root_relative,
        'vr_target': vr_target_from_record(record),
    }


@torch.no_grad()
def run_official_stack(record, mode):
    a_seq, w_seq, R_seq = selected_imu_fields(record, mode)
    net = GPNet().eval().to(DEVICE)
    net.rnn_initialize(record['pose_gt'][0])
    body_model = net.body_model

    rows = []
    for frame_idx in range(a_seq.shape[0]):
        a = a_seq[frame_idx].to(DEVICE)
        w = w_seq[frame_idx].to(DEVICE)
        R = R_seq[frame_idx].to(DEVICE)

        aRB_pl = a.mm(R[5])
        wRB_pl = w.mm(R[5])
        RRB0 = R[5].t().matmul(R[:5])
        gR0 = -R[5, 1]
        pl_input = torch.cat((aRB_pl.ravel(), wRB_pl.ravel(), RRB0.ravel(), gR0))

        pl_raw, gR1 = net._run_pl_stage(pl_input)
        pl_out = normalize_gravity(pl_raw.detach().cpu())
        gR1 = art.math.normalize_tensor(gR1.detach().cpu(), avoid_nan=True)
        RRB_after_pl = art.math.from_to_rotation_matrix(gR0, gR1.to(DEVICE)).matmul(RRB0)

        ik1_out, gR2 = net._run_ik1_stage(RRB_after_pl, gR1.to(DEVICE), pl_raw[:15])
        ik1_out = ik1_out.detach().cpu()
        gR2 = art.math.normalize_tensor(gR2.detach().cpu(), avoid_nan=True)
        RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1.to(DEVICE), gR2.to(DEVICE)).matmul(RRB_after_pl)

        ik2_input = torch.cat((RRB_after_ik1.ravel(), gR2.to(DEVICE), ik1_out[:69].to(DEVICE)))
        ik2_hidden, net.ik2hc = net.iknet.net2.rnn(ik2_input.view(1, 1, -1), net.ik2hc)
        ik2_out = net.iknet.net2.linear2(ik2_hidden.squeeze()).detach().cpu()
        RRJ = art.math.r6d_to_rotation_matrix(ik2_out).cpu().view(15, 3, 3)

        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, net.j_reduce] = RRJ.view(1, 15, 3, 3)
        pose_body = body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose_body[net.j_ignore, ...] = torch.eye(3)
        pRJ_fk = body_model.forward_kinematics(pose_body.unsqueeze(0))[1][0, 1:].detach().cpu()
        root_pose = R[5].mm(art.math.from_to_rotation_matrix(gR2.to(DEVICE), gR0).squeeze()).detach().cpu()
        pose_full = pose_body.detach().cpu().clone()
        pose_full[0] = root_pose

        aRB_vr = a.detach().cpu().mm(root_pose)
        wRB_vr = w.detach().cpu().mm(root_pose)
        vr_input = torch.cat((RRJ.ravel(), pRJ_fk.ravel(), aRB_vr.ravel(), wRB_vr.ravel(), gR2.cpu())).to(DEVICE)
        vr_hidden, net.vr1hc = net.vrnet.rnn(vr_input.view(1, 1, -1), net.vr1hc)
        vr_out = net.vrnet.linear2(vr_hidden.squeeze()).detach().cpu()

        rows.append({
            'pl_input': pl_input.detach().cpu(),
            'aRB_pl': aRB_pl.detach().cpu(),
            'wRB_pl': wRB_pl.detach().cpu(),
            'RRB0': RRB0.detach().cpu(),
            'gR0': gR0.detach().cpu(),
            'pl_out': pl_out,
            'gR1': gR1,
            'RRB_after_pl': RRB_after_pl.detach().cpu(),
            'ik1_out': ik1_out,
            'gR2': gR2,
            'RRB_after_ik1': RRB_after_ik1.detach().cpu(),
            'ik2_out': ik2_out,
            'ik2_rot': RRJ.detach().cpu(),
            'pose_body': pose_body.detach().cpu(),
            'pose_full': pose_full.detach().cpu(),
            'root_pose': root_pose,
            'pRJ_fk': pRJ_fk,
            'vr_input': vr_input.detach().cpu(),
            'vr_out': vr_out,
        })
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


@torch.no_grad()
def input_stats(record):
    official = {key: record[key].float() for key in ('aM', 'wM', 'RMB')}
    processed = {key: record[key].float() for key in ('l4_aM', 'l4_wM', 'l4_RMB')}
    return {
        'official': {
            'aM': mean_std_min_max(official['aM']),
            'wM': mean_std_min_max(official['wM']),
            'RMB_quality': rotation_quality(official['RMB']),
            'all_finite': bool(all(torch.isfinite(value).all() for value in official.values())),
        },
        'processed': {
            'l4_aM': mean_std_min_max(processed['l4_aM']),
            'l4_wM': mean_std_min_max(processed['l4_wM']),
            'l4_RMB_quality': rotation_quality(processed['l4_RMB']),
            'all_finite': bool(all(torch.isfinite(value).all() for value in processed.values())),
        },
        'processed_minus_official': {
            'a_norm': mean_std_min_max((processed['l4_aM'] - official['aM']).norm(dim=-1)),
            'w_norm': mean_std_min_max((processed['l4_wM'] - official['wM']).norm(dim=-1)),
            'R_geodesic_deg': metric_stats(rotation_angle_deg(processed['l4_RMB'], official['RMB'])),
            'a_allclose': bool(torch.allclose(processed['l4_aM'], official['aM'])),
            'w_allclose': bool(torch.allclose(processed['l4_wM'], official['wM'])),
            'R_allclose': bool(torch.allclose(processed['l4_RMB'], official['RMB'])),
        },
    }


@torch.no_grad()
def evaluate_outputs(outputs, targets):
    pl = outputs['pl_out']
    pl_target = targets['pl_target']
    ik1 = outputs['ik1_out']
    ik1_target = targets['ik1_target']
    ik2_rot = outputs['ik2_rot']
    ik2_target = targets['ik2_rot_target']
    pose_body = outputs['pose_body']
    pose_full = outputs['pose_full']
    pose_gt = targets['pose_gt']
    pRJ_fk = outputs['pRJ_fk']
    joints_root_relative = targets['joints_root_relative'][:, 1:]
    vr = outputs['vr_out']
    vr_target = targets['vr_target']

    body_target = targets['body_rot_target']
    root_target = pose_gt[:, 0]
    reduced_indices = list(range(len(GPNet.j_reduce)))

    metrics = {
        'pl_leaf_cm': metric_stats(l2_cm(pl[:, :15].reshape(-1, 5, 3), pl_target[:, :15].reshape(-1, 5, 3))),
        'pl_gR1_angle_deg': metric_stats(gravity_angle_deg(pl[:, 15:], pl_target[:, 15:])),
        'RRB_after_pl_vs_input_angle_deg': metric_stats(rotation_angle_deg(outputs['RRB_after_pl'], outputs['RRB0'])),
        'ik1_pRJ_cm': metric_stats(l2_cm(ik1[:, :69].reshape(-1, 23, 3), ik1_target[:, :69].reshape(-1, 23, 3))),
        'ik1_gR2_angle_deg': metric_stats(gravity_angle_deg(ik1[:, 69:], ik1_target[:, 69:])),
        'RRB_after_ik1_vs_input_angle_deg': metric_stats(rotation_angle_deg(outputs['RRB_after_ik1'], outputs['RRB0'])),
        'ik2_reduced_rotation_deg': metric_stats(rotation_angle_deg(ik2_rot, ik2_target)),
        'post_ik2_body_rotation_deg': metric_stats(rotation_angle_deg(pose_body[None, :, GPNet.j_reduce].squeeze(0), body_target[:, GPNet.j_reduce])),
        'root_orientation_deg': metric_stats(rotation_angle_deg(outputs['root_pose'], root_target)),
        'postprocessed_full_pose_rotation_deg': metric_stats(rotation_angle_deg(pose_full, pose_gt)),
        'fk_joint_root_relative_cm': metric_stats(l2_cm(pRJ_fk, joints_root_relative)),
        'vr_velocity_l2': metric_stats((vr[:, :4] - vr_target[:, :4]).norm(dim=-1)),
        'vr_contact_prob_abs': metric_stats((vr[:, 4:].sigmoid() - vr_target[:, 4:]).abs()),
        'vr_raw_l2': metric_stats((vr - vr_target).norm(dim=-1)),
        'vr_input_l2': metric_stats(outputs['vr_input'].norm(dim=-1)),
        'finite_outputs': bool(all(torch.isfinite(value).all() for value in outputs.values())),
    }
    return metrics


def flatten_metric_rows(sequence_rows, aggregate):
    names = [
        ('Input IMU', 'processed-official acceleration norm', 'input_processed_minus_official.a_norm.mean'),
        ('Input IMU', 'processed-official gyro norm', 'input_processed_minus_official.w_norm.mean'),
        ('Input IMU', 'processed-official RMB geodesic deg', 'input_processed_minus_official.R_geodesic_deg.mean'),
        ('PL output', 'leaf pRB cm', 'pl_leaf_cm.mean'),
        ('PL output', 'gR1 gravity deg', 'pl_gR1_angle_deg.mean'),
        ('RRB after PL', 'RRB_after_pl vs input deg', 'RRB_after_pl_vs_input_angle_deg.mean'),
        ('IK1 output', 'pRJ cm', 'ik1_pRJ_cm.mean'),
        ('IK1 output', 'gR2 gravity deg', 'ik1_gR2_angle_deg.mean'),
        ('RRB after IK1', 'RRB_after_ik1 vs input deg', 'RRB_after_ik1_vs_input_angle_deg.mean'),
        ('IK2 output', 'reduced rotation deg', 'ik2_reduced_rotation_deg.mean'),
        ('IK2 postprocess', 'body reduced rotation deg', 'post_ik2_body_rotation_deg.mean'),
        ('Root output', 'root orientation deg', 'root_orientation_deg.mean'),
        ('FK joints', 'root-relative joint cm', 'fk_joint_root_relative_cm.mean'),
        ('VR output', 'velocity l2', 'vr_velocity_l2.mean'),
        ('VR output', 'contact probability abs', 'vr_contact_prob_abs.mean'),
        ('VR output', 'raw 9D l2', 'vr_raw_l2.mean'),
    ]
    rows = []
    for module, metric, path in names:
        if path.startswith('input_processed_minus_official'):
            value = aggregate['input_stats']['processed_minus_official'][path.split('.')[1]][path.split('.')[2]]
            rows.append({
                'module_output': module,
                'metric': metric,
                'official_imu_error': 0.0,
                'processed_imu_error': value,
                'delta_processed_minus_official': value,
                'better_input': 'diagnostic_only',
                'is_gt_error': False,
            })
            continue
        metric_key = path.split('.')[0]
        stat_key = path.split('.')[1]
        off = aggregate['official']['metrics'][metric_key][stat_key]
        proc = aggregate['processed']['metrics'][metric_key][stat_key]
        if off is None or proc is None:
            better = 'n/a'
            delta = None
        else:
            delta = proc - off
            if math.isclose(delta, 0.0, abs_tol=1e-12):
                better = 'tie'
            else:
                better = 'processed' if delta < 0 else 'official'
        rows.append({
            'module_output': module,
            'metric': metric,
            'official_imu_error': off,
            'processed_imu_error': proc,
            'delta_processed_minus_official': delta,
            'better_input': better,
            'is_gt_error': module not in ('Input IMU', 'RRB after PL', 'RRB after IK1') and metric != 'VR input l2',
        })
    return rows


def aggregate_sequence_metrics(sequence_rows):
    metric_keys = sorted(
        key for key, value in sequence_rows[0]['official']['metrics'].items()
        if isinstance(value, dict)
    )
    out = {'official': {'metrics': {}}, 'processed': {'metrics': {}}, 'input_stats': {'processed_minus_official': {}}}
    for mode in ('official', 'processed'):
        for key in metric_keys:
            out[mode]['metrics'][key] = {}
            for stat in ('mean', 'median', 'std', 'min', 'max'):
                vals = [row[mode]['metrics'][key].get(stat) for row in sequence_rows]
                vals = [value for value in vals if value is not None]
                out[mode]['metrics'][key][stat] = float(torch.tensor(vals).mean()) if vals else None
            out[mode]['metrics'][key]['count'] = int(sum(row[mode]['metrics'][key].get('count', 0) for row in sequence_rows))
    for key in ('a_norm', 'w_norm', 'R_geodesic_deg'):
        out['input_stats']['processed_minus_official'][key] = {}
        for stat in ('mean', 'median', 'std', 'min', 'max'):
            vals = [row['input_stats']['processed_minus_official'][key].get(stat) for row in sequence_rows]
            vals = [value for value in vals if value is not None]
            out['input_stats']['processed_minus_official'][key][stat] = float(torch.tensor(vals).mean()) if vals else None
        out['input_stats']['processed_minus_official'][key]['count'] = int(sum(row['input_stats']['processed_minus_official'][key].get('count', 0) for row in sequence_rows))
    out['input_stats']['processed_minus_official']['a_allclose_all_sequences'] = all(row['input_stats']['processed_minus_official']['a_allclose'] for row in sequence_rows)
    out['input_stats']['processed_minus_official']['w_allclose_all_sequences'] = all(row['input_stats']['processed_minus_official']['w_allclose'] for row in sequence_rows)
    out['input_stats']['processed_minus_official']['R_allclose_all_sequences'] = all(row['input_stats']['processed_minus_official']['R_allclose'] for row in sequence_rows)
    return out


def main():
    parser = argparse.ArgumentParser(description='Forward-only official GPNet per-module GT audit for official vs processed IMU.')
    parser.add_argument('--cache', type=Path, default=DEFAULT_S4_CACHE)
    parser.add_argument('--processed-cache', type=Path, default=DEFAULT_PROCESSED_CACHE)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    args = parser.parse_args()

    result = {
        'experiment': 'EXP-OfficialGPNet-ProcessedIMU-PerModule-GT-Audit',
        'cache': str(args.cache),
        'processed_cache': str(args.processed_cache) if args.processed_cache else None,
        'output_json': str(args.output_json),
        'split': 'TotalCapture S4 validation',
        'status': 'started',
        'notes': [
            'Forward-only audit; no training.',
            'Official weights are loaded by GPNet from data/weights.pt and are not modified.',
            'Official and processed inputs use separate GPNet instances per sequence to avoid hidden-state mixing.',
            'RRB-after metrics are diagnostic geometry deltas, not direct GT errors.',
            'VR contact comparison applies sigmoid to predicted contact logits before comparing to stationary_prob.',
        ],
        'input_contract': {
            'official': {'a': 'aM [T,6,3]', 'w': 'wM [T,6,3]', 'R': 'RMB [T,6,3,3]'},
            'processed': {'a': 'l4_aM [T,6,3]', 'w': 'l4_wM [T,6,3]', 'R': 'l4_RMB [T,6,3,3]'},
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        records, manifest = load_records(args.cache, args.max_sequences)
        processed_by_name, processed_manifest = load_processed_imu_by_name(args.processed_cache)
        sequence_rows = []
        processed_sources = {}
        for record in records:
            record, processed_source = attach_processed_fields(record, processed_by_name)
            processed_sources[processed_source] = processed_sources.get(processed_source, 0) + 1
            if args.max_frames and record['pose_gt'].shape[0] > args.max_frames:
                record = dict(record)
                seq_len = record['pose_gt'].shape[0]
                for key, value in list(record.items()):
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == seq_len:
                        record[key] = value[:args.max_frames]
            probe = GPNet().eval().to(DEVICE)
            targets = build_targets(record, probe)
            del probe
            off_outputs = run_official_stack(record, 'official')
            proc_outputs = run_official_stack(record, 'processed')
            row = {
                'name': record['name'],
                'num_frames': int(record['pose_gt'].shape[0]),
                'input_stats': input_stats(record),
                'processed_imu_source': processed_source,
                'official': {'metrics': evaluate_outputs(off_outputs, targets)},
                'processed': {'metrics': evaluate_outputs(proc_outputs, targets)},
            }
            sequence_rows.append(row)
            print(json.dumps({
                'audited': record['name'],
                'frames': row['num_frames'],
                'official_pl_leaf_cm': row['official']['metrics']['pl_leaf_cm']['mean'],
                'processed_pl_leaf_cm': row['processed']['metrics']['pl_leaf_cm']['mean'],
                'official_vr_velocity_l2': row['official']['metrics']['vr_velocity_l2']['mean'],
                'processed_vr_velocity_l2': row['processed']['metrics']['vr_velocity_l2']['mean'],
            }), flush=True)
        aggregate = aggregate_sequence_metrics(sequence_rows)
        table_rows = flatten_metric_rows(sequence_rows, aggregate)
        result.update({
            'status': 'ok',
            'manifest': manifest,
            'processed_manifest': processed_manifest,
            'processed_sources': processed_sources,
            'num_sequences': len(sequence_rows),
            'num_frames': int(sum(row['num_frames'] for row in sequence_rows)),
            'rows': sequence_rows,
            'aggregate': aggregate,
            'summary_table': table_rows,
            'official_weights_modified': False,
            'motion_evaluator_modified': False,
            'test_py_modified': False,
            'carticulate_physics_modified': False,
            'training_run': False,
            's5_run': False,
            'all_finite': all(
                row['official']['metrics']['finite_outputs'] and row['processed']['metrics']['finite_outputs']
                for row in sequence_rows
            ),
        })
    except Exception as exc:
        import traceback
        result.update({
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'num_sequences', 'num_frames', 'all_finite', 'error_type', 'error')}, indent=2), flush=True)
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
