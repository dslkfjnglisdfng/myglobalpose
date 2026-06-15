import argparse
import json
import traceback
from pathlib import Path

import torch
import articulate as art

from l4_train_diverse_short import DEVICE, load_records
from net import GPNet
from newik1_real_streaming_audit import build_net, load_ik1
from official_processed_module_audit import build_targets, metric_stats
from pl_curve_eval import build_pl_curve, selected_imu_fields


FPS = 60.0
EPS = 1e-8


def pRJ_to_full_joints(pRJ):
    joints = pRJ.float().reshape(pRJ.shape[0], 23, 3)
    root = joints.new_zeros(joints.shape[0], 1, 3)
    return torch.cat((root, joints), dim=1)


def bone_tensors(ik1_state, parent):
    joints = pRJ_to_full_joints(ik1_state[..., :69])
    parents = []
    children = []
    for child in range(1, 24):
        p = parent[child]
        if p is None:
            continue
        parents.append(int(p))
        children.append(child)
    parent_idx = torch.tensor(parents, device=joints.device)
    child_idx = torch.tensor(children, device=joints.device)
    return joints[:, child_idx] - joints[:, parent_idx]


def safe_unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def angle_deg(a, b):
    ua = safe_unit(a)
    ub = safe_unit(b)
    cos = (ua * ub).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def omega_proxy(bone, fps=FPS):
    if bone.shape[0] < 2:
        return bone.new_empty(0, bone.shape[1], 3)
    b = bone[:-1]
    v = (bone[1:] - bone[:-1]) * fps
    return torch.cross(b, v, dim=-1) / b.square().sum(dim=-1, keepdim=True).clamp_min(EPS)


def bone_metrics(pred_ik1, target_ik1, parent):
    pred_b = bone_tensors(pred_ik1, parent)
    target_b = bone_tensors(target_ik1, parent)
    pred_len = pred_b.norm(dim=-1)
    target_len = target_b.norm(dim=-1)
    out = {
        'bone_len_abs_cm': (pred_len - target_len).abs().reshape(-1) * 100.0,
        'bone_vec_l2_cm': (pred_b - target_b).norm(dim=-1).reshape(-1) * 100.0,
        'bone_dir_angle_deg': angle_deg(pred_b, target_b).reshape(-1),
    }
    if pred_b.shape[0] >= 2:
        pred_vrel = (pred_b[1:] - pred_b[:-1]) * FPS
        target_vrel = (target_b[1:] - target_b[:-1]) * FPS
        pred_u = safe_unit(pred_b[:-1])
        target_u = safe_unit(target_b[:-1])
        pred_radial = (pred_u * pred_vrel).sum(dim=-1)
        target_radial = (target_u * target_vrel).sum(dim=-1)
        pred_omega = omega_proxy(pred_b)
        target_omega = omega_proxy(target_b)
        out.update({
            'bone_radial_vel_abs_mps': pred_radial.abs().reshape(-1),
            'bone_radial_vel_error_mps': (pred_radial - target_radial).abs().reshape(-1),
            'bone_omega_l2_radps': (pred_omega - target_omega).norm(dim=-1).reshape(-1),
            'bone_vrel_l2_mps': (pred_vrel - target_vrel).norm(dim=-1).reshape(-1),
        })
    else:
        empty = pred_b.new_empty(0)
        out.update({
            'bone_radial_vel_abs_mps': empty,
            'bone_radial_vel_error_mps': empty,
            'bone_omega_l2_radps': empty,
            'bone_vrel_l2_mps': empty,
        })
    if pred_b.shape[0] >= 3:
        pred_vrel = (pred_b[1:] - pred_b[:-1]) * FPS
        pred_arel = (pred_vrel[1:] - pred_vrel[:-1]) * FPS
        b_mid = pred_b[1:-1]
        pred_stretch_acc = (b_mid * pred_arel).sum(dim=-1) + pred_vrel[:-1].square().sum(dim=-1)
        out['bone_stretch_acc_abs'] = pred_stretch_acc.abs().reshape(-1)
    else:
        out['bone_stretch_acc_abs'] = pred_b.new_empty(0)
    return out


@torch.no_grad()
def run_ik1_only(record, pl_curve=None, ik1_model=None, ik1_backend='original', ik1_config=None, imu_input_mode='official'):
    if pl_curve is not None and getattr(pl_curve, 'init_size', 18) == 36:
        if 'offset_r' not in record:
            raise KeyError(f'Record {record["name"]} lacks offset_r required by PL init36.')
        offset_r = record['offset_r'].float()
    else:
        offset_r = record.get('offset_r')
    a_seq, w_seq, R_seq = selected_imu_fields(record, imu_input_mode)
    net = build_net(pl_curve, ik1_model, ik1_backend, ik1_config=ik1_config)
    net.rnn_initialize(record['pose_gt'][0], offset_r=offset_r)
    outputs = []
    for idx in range(record['pose_gt'].shape[0]):
        ik1_debug = net.forward_until_ik1(
            a_seq[idx].to(DEVICE),
            w_seq[idx].to(DEVICE),
            R_seq[idx].to(DEVICE),
        )
        outputs.append(torch.cat((ik1_debug['pRJ_ik1'].reshape(-1), ik1_debug['gR2'].reshape(-1))).cpu())
    return torch.stack(outputs)


def aggregate_metric_tensors(rows):
    out = {}
    keys = rows[0]['metric_tensors'].keys() if rows else []
    for key in keys:
        vals = [row['metric_tensors'][key] for row in rows if row['metric_tensors'][key].numel() > 0]
        flat = torch.cat(vals) if vals else torch.empty(0)
        out[key] = metric_stats(flat)
    return out


def evaluate(args):
    records, manifest = load_records(args.val_cache)
    if args.max_eval_sequences:
        records = records[:args.max_eval_sequences]
    pl_curve, pl_config = (None, None)
    if args.pl_checkpoint:
        pl_curve, pl_config = build_pl_curve(args.pl_checkpoint)
    ik1_model, ik1_config = load_ik1(args)
    parent = art.ParametricModel('models/SMPL_male.pkl').parent
    rows = []
    for record in records:
        pred = run_ik1_only(
            record,
            pl_curve=pl_curve,
            ik1_model=ik1_model,
            ik1_backend=args.ik1_backend,
            ik1_config=ik1_config,
            imu_input_mode=args.imu_input_mode,
        )
        targets = build_targets(record, GPNet())
        target = targets['ik1_target'].cpu()
        metrics = bone_metrics(pred, target, parent)
        rows.append({
            'name': record['name'],
            'num_frames': int(pred.shape[0]),
            'metrics': {k: metric_stats(v) for k, v in metrics.items()},
            'metric_tensors': metrics,
        })
    aggregate = aggregate_metric_tensors(rows)
    serial_rows = []
    for row in rows:
        serial_rows.append({
            'name': row['name'],
            'num_frames': row['num_frames'],
            'metrics': row['metrics'],
        })
    return {
        'status': 'ok',
        'version_name': args.version_name,
        'split_label': args.split_label,
        'val_cache': str(args.val_cache),
        'pl_checkpoint': str(args.pl_checkpoint) if args.pl_checkpoint else None,
        'ik1_checkpoint': str(args.ik1_checkpoint) if args.ik1_checkpoint else None,
        'ik1_backend': args.ik1_backend,
        'imu_input_mode': args.imu_input_mode,
        'metric_contract': 'IK1 raw pRJ bone kinematic metrics. pRJ[69] is expanded with zero root to 24 SMPL joints; metrics use SMPL parent edges.',
        'split_manifest': manifest,
        'pl_checkpoint_config': pl_config,
        'ik1_checkpoint_config': ik1_config,
        'rows': serial_rows,
        'aggregate': aggregate,
    }


def main():
    parser = argparse.ArgumentParser(description='Audit IK1 pRJ bone kinematic consistency.')
    parser.add_argument('--val-cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--split-label', required=True)
    parser.add_argument('--version-name', required=True)
    parser.add_argument('--pl-checkpoint', type=Path)
    parser.add_argument('--ik1-checkpoint', type=Path)
    parser.add_argument('--ik1-backend', choices=('original', 'official_input_v1', 'control_point_v1', 'control_point_last_v1', 'auto_control_point'), default='original')
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='official')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    args = parser.parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate(args)
    except Exception as exc:
        result = {
            'status': 'failed',
            'version_name': args.version_name,
            'split_label': args.split_label,
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        }
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    preview = {k: result.get(k) for k in ('status', 'version_name', 'split_label', 'error_type', 'error')}
    if result.get('status') == 'ok':
        preview['aggregate'] = {
            k: v.get('mean') for k, v in result.get('aggregate', {}).items()
        }
    print(json.dumps(preview, indent=2))
    if result.get('status') != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
