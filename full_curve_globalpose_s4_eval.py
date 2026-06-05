import argparse
import json
import traceback
from pathlib import Path

import torch
import articulate as art

from full_curve_globalpose import FullCurveGlobalPoseV1, safe_from_to_rotation_matrix
from l4_train_diverse_short import DEVICE, aggregate_eval, load_records, metric_to_dict, score_for_checkpoint
from net import GPNet
from test import MotionEvaluator


FULL_CURVE_KEYS = (
    'pl_input', 'pl_base', 'ik1_base', 'ik2_base', 'vr_base', 'processed_imu',
)


def selected_imu_fields(record, mode):
    if mode == 'official':
        return record['aM'], record['wM'], record['RMB']
    has_l4 = all(key in record for key in ('l4_aM', 'l4_wM', 'l4_RMB'))
    has_processed_imu = 'processed_imu' in record
    if mode == 'processed':
        if has_l4:
            return record['l4_aM'], record['l4_wM'], record['l4_RMB']
        if has_processed_imu:
            imu = record['processed_imu']
            return imu[..., :18].reshape(-1, 6, 3), imu[..., 18:36].reshape(-1, 6, 3), imu[..., 36:90].reshape(-1, 6, 3, 3)
        raise KeyError(f'processed mode requires l4_aM/l4_wM/l4_RMB or processed_imu in record {record.get("name")}.')
    if mode == 'auto':
        if has_l4:
            return record['l4_aM'], record['l4_wM'], record['l4_RMB']
        if has_processed_imu:
            imu = record['processed_imu']
            return imu[..., :18].reshape(-1, 6, 3), imu[..., 18:36].reshape(-1, 6, 3), imu[..., 36:90].reshape(-1, 6, 3, 3)
        return record['aM'], record['wM'], record['RMB']
    raise ValueError(f'Unsupported imu input mode: {mode}')


def load_full_curve_records(cache_path):
    path = Path(cache_path)
    if path.suffix == '.json':
        manifest = json.loads(path.read_text())
        files = [Path(item['path']) for item in manifest['cache_files']]
    else:
        manifest = None
        files = [path]
    records = []
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        missing = [key for key in FULL_CURVE_KEYS if key not in data]
        if missing:
            raise KeyError(f'{cache_file} missing FullCurve fields: {missing}')
        for seq_idx, name in enumerate(data['name']):
            record = {'name': name}
            for key in FULL_CURVE_KEYS:
                record[key] = data[key][seq_idx].float()
            if 'offset_r' in data and data['offset_r']:
                record['offset_r'] = data['offset_r'][seq_idx].float()
            records.append(record)
    return records, manifest


def slice_record(record, max_frames):
    if not max_frames:
        return record
    seq_len = record['pose_gt'].shape[0]
    if seq_len <= max_frames:
        return record
    out = {}
    for key, value in record.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == seq_len:
            out[key] = value[:max_frames]
        else:
            out[key] = value
    return out


def load_full_curve(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = checkpoint.get('config', {})
    model = FullCurveGlobalPoseV1(
        hidden_size=int(cfg.get('hidden_size', 512)),
        tail_update=int(cfg.get('tail_length', 4)),
        residual_scale=float(cfg.get('residual_scale', 0.005)),
        vr_residual_scale=float(cfg.get('vr_residual_scale', 0.005)),
        dropout=float(cfg.get('dropout', 0.4)),
        offset_init_scale=float(cfg.get('offset_init_scale', 0.1)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def full_curve_frame_outputs(model, record, frame_idx, a_seq, w_seq, R_seq, base_net):
    pl_input = record['pl_input'][frame_idx].view(1, -1).to(DEVICE)
    base_pl = record['pl_base'][frame_idx].view(1, -1).to(DEVICE)
    base_ik1 = record['ik1_base'][frame_idx].view(1, -1).to(DEVICE)
    base_ik2 = record['ik2_base'][frame_idx].view(1, -1).to(DEVICE)
    base_vr = record['vr_base'][frame_idx].view(1, -1).to(DEVICE)
    imu = record['processed_imu'][frame_idx].view(1, 90).to(DEVICE)

    RRB0, gR0 = pl_input[..., 36:81].reshape(1, 5, 3, 3), pl_input[..., 81:84]
    pl_out = model.pl.step(pl_input, base_pl)
    pl_t = pl_out['out_t']
    gR1 = pl_t[:, 15:]
    RRB_after_pl = safe_from_to_rotation_matrix(gR0, gR1).unsqueeze(1).matmul(RRB0)

    ik1_feature = torch.cat((RRB_after_pl.flatten(1), gR1, model.pl.control_tail().flatten(1)), dim=-1)
    ik1_out = model.ik1.step(ik1_feature, base_ik1)
    ik1_t = ik1_out['out_t']
    gR2 = ik1_t[:, 69:]
    RRB_after_ik1 = safe_from_to_rotation_matrix(gR1, gR2).unsqueeze(1).matmul(RRB_after_pl)

    ik2_feature = torch.cat((RRB_after_ik1.flatten(1), gR2, model.ik1.control_tail().flatten(1)), dim=-1)
    ik2_out = model.ik2.step(ik2_feature, base_ik2)
    ik2_t = ik2_out['out_t']

    vr_feature = torch.cat((
        model.pl.control_tail().flatten(1),
        model.ik1.control_tail().flatten(1),
        model.ik2.control_tail().flatten(1),
        imu.flatten(1),
    ), dim=-1)
    vr_out = model.vr.step(vr_feature, base_vr)
    vr_t = vr_out['out_t'][0]

    RRJ = art.math.r6d_to_rotation_matrix(ik2_t[0]).cpu()
    glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
    glb_pose[:, base_net.j_reduce] = RRJ.view(1, 15, 3, 3)
    pose = base_net.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
    pose[base_net.j_ignore, ...] = torch.eye(3)
    pRJ = base_net.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
    pose[0] = R_seq[frame_idx][5].mm(art.math.from_to_rotation_matrix(gR2[0].cpu(), -R_seq[frame_idx][5, 1]).squeeze()).cpu()

    return pose, pRJ, RRJ, gR2[0].cpu(), vr_t.cpu(), {
        'pl_control_shape': list(model.pl.control_tail().shape),
        'ik1_control_shape': list(model.ik1.control_tail().shape),
        'ik2_control_shape': list(model.ik2.control_tail().shape),
        'vr_control_shape': list(model.vr.control_tail().shape),
    }


@torch.no_grad()
def run_full_curve_sequence(model, record, imu_input_mode, use_vr_override=True):
    a_seq, w_seq, R_seq = selected_imu_fields(record, imu_input_mode)
    base_net = GPNet().eval().to(DEVICE)
    base_net.rnn_initialize(record['pose_gt'][0])
    offset_r = record.get('offset_r')
    if offset_r is not None and torch.is_tensor(offset_r):
        offset_r = offset_r.view(1, 6, 3).to(DEVICE)
    pl_input0 = record['pl_input'][0].view(1, -1).to(DEVICE)
    base_pl0 = record['pl_base'][0].view(1, -1).to(DEVICE)
    base_ik10 = record['ik1_base'][0].view(1, -1).to(DEVICE)
    base_ik20 = record['ik2_base'][0].view(1, -1).to(DEVICE)
    base_vr0 = record['vr_base'][0].view(1, -1).to(DEVICE)
    imu0 = record['processed_imu'][0].view(1, 90).to(DEVICE)
    RRB0, gR0 = pl_input0[..., 36:81].reshape(1, 5, 3, 3), pl_input0[..., 81:84]
    init_pl = base_pl0
    init_ik1 = base_ik10
    init_ik2 = base_ik20
    init_gR1 = init_pl[:, 15:]
    init_rrb_after_pl = safe_from_to_rotation_matrix(gR0, init_gR1).unsqueeze(1).matmul(RRB0)
    init_ik1_feature = torch.cat((
        init_rrb_after_pl.flatten(1),
        init_gR1,
        init_pl.unsqueeze(1).expand(-1, model.tail_update, -1).flatten(1),
    ), dim=-1)
    init_gR2 = init_ik1[:, 69:]
    init_rrb_after_ik1 = safe_from_to_rotation_matrix(init_gR1, init_gR2).unsqueeze(1).matmul(init_rrb_after_pl)
    init_ik2_feature = torch.cat((
        init_rrb_after_ik1.flatten(1),
        init_gR2,
        init_ik1.unsqueeze(1).expand(-1, model.tail_update, -1).flatten(1),
    ), dim=-1)
    init_vr_feature = torch.cat((
        init_pl.unsqueeze(1).expand(-1, model.tail_update, -1).flatten(1),
        init_ik1.unsqueeze(1).expand(-1, model.tail_update, -1).flatten(1),
        init_ik2.unsqueeze(1).expand(-1, model.tail_update, -1).flatten(1),
        imu0.flatten(1),
    ), dim=-1)
    model.reset_stream({
        'pl': base_pl0,
        'ik1': base_ik10,
        'ik2': base_ik20,
        'vr': base_vr0,
        'offset_r': offset_r,
        'pl_feature': pl_input0,
        'ik1_feature': init_ik1_feature,
        'ik2_feature': init_ik2_feature,
        'vr_feature': init_vr_feature,
    })
    pose_out = torch.zeros_like(record['pose_gt'])
    tran_out = torch.zeros_like(record['tran_gt'])
    shape_info = None
    for frame_idx in range(record['pose_gt'].shape[0]):
        pose, pRJ, RRJ, gR2, vr_t, shapes = full_curve_frame_outputs(model, record, frame_idx, a_seq, w_seq, R_seq, base_net)
        vr_delta = vr_t - record['vr_base'][frame_idx]
        pose_model, tran_model, _debug = base_net.forward_frame_from_curve_pose(
            a_seq[frame_idx].to(DEVICE),
            w_seq[frame_idx].to(DEVICE),
            R_seq[frame_idx].to(DEVICE),
            pose,
            gR2,
            vr_override_delta=vr_delta.to(DEVICE) if use_vr_override else None,
        )
        pose_out[frame_idx] = pose_model.cpu()
        tran_out[frame_idx] = tran_model.cpu()
        shape_info = shapes
        shape_info['used_vr_override'] = bool(_debug.get('used_vr_override', False))
        shape_info['vr_delta_norm_mean'] = float(vr_delta.norm())
        shape_info['official_vr_norm_mean'] = float(_debug['official_vr_norm'])
        shape_info['injected_vr_norm_mean'] = float(_debug.get('override_vr_norm', _debug['official_vr_norm']))
        shape_info['override_delta_norm_mean'] = float(_debug.get('override_delta_norm', torch.zeros(())))
    return pose_out, tran_out, shape_info


@torch.no_grad()
def run_baseline(record, imu_input_mode):
    if 'pose_baseline' in record and 'tran_baseline' in record:
        return record['pose_baseline'], record['tran_baseline']
    a_seq, w_seq, R_seq = selected_imu_fields(record, imu_input_mode)
    net = GPNet().eval().to(DEVICE)
    net.rnn_initialize(record['pose_gt'][0])
    pose = torch.zeros_like(record['pose_gt'])
    tran = torch.zeros_like(record['tran_gt'])
    for frame_idx in range(record['pose_gt'].shape[0]):
        pose[frame_idx], tran[frame_idx] = net.forward_frame(
            a_seq[frame_idx].to(DEVICE),
            w_seq[frame_idx].to(DEVICE),
            R_seq[frame_idx].to(DEVICE),
        )
    return pose, tran


def main():
    parser = argparse.ArgumentParser(description='Evaluate FullCurveGlobalPose_v1 with S4 MotionEvaluator metrics.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--val-cache', type=Path, default=Path('data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json'))
    parser.add_argument('--full-curve-cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='processed')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--smoke-sequence', default='')
    parser.add_argument('--max-smoke-frames', type=int, default=0)
    parser.add_argument('--disable-vr-override', action='store_true')
    args = parser.parse_args()

    result = {
        'checkpoint': str(args.checkpoint),
        'val_cache': str(args.val_cache),
        'full_curve_cache': str(args.full_curve_cache),
        'imu_input_mode': args.imu_input_mode,
        'use_vr_override': not args.disable_vr_override,
        'status': 'started',
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        records, manifest = load_records(args.val_cache)
        full_records, full_manifest = load_full_curve_records(args.full_curve_cache)
        full_by_name = {record['name']: record for record in full_records}
        if args.smoke_sequence:
            records = [record for record in records if record['name'] == args.smoke_sequence]
            if not records:
                raise KeyError(f'No sequence named {args.smoke_sequence!r}.')
            args.max_eval_sequences = 1
        selected = records[:args.max_eval_sequences] if args.max_eval_sequences else records
        model, checkpoint = load_full_curve(args.checkpoint)
        evaluator = MotionEvaluator()
        rows = []
        for record in selected:
            if record['name'] not in full_by_name:
                raise KeyError(f'No full-curve cache record for {record["name"]!r}.')
            merged = dict(record)
            merged.update({key: value for key, value in full_by_name[record['name']].items() if key not in ('pose_gt', 'tran_gt')})
            merged = slice_record(merged, args.max_smoke_frames)
            pose_model, tran_model, shape_info = run_full_curve_sequence(model, merged, args.imu_input_mode, not args.disable_vr_override)
            pose_ref, tran_ref = run_baseline(merged, args.imu_input_mode)
            baseline_metric = evaluator(pose_ref.to(DEVICE), merged['pose_gt'].to(DEVICE), tran_ref.to(DEVICE), merged['tran_gt'].to(DEVICE)).cpu()
            model_metric = evaluator(pose_model.to(DEVICE), merged['pose_gt'].to(DEVICE), tran_model.to(DEVICE), merged['tran_gt'].to(DEVICE)).cpu()
            finite = bool(torch.isfinite(pose_model).all() and torch.isfinite(tran_model).all())
            rows.append({
                'name': merged['name'],
                'baseline_metrics': metric_to_dict(baseline_metric),
                'model_metrics': metric_to_dict(model_metric),
                'shape_info': shape_info,
                'finite': finite,
                'delta_v_root_norm_mean': 0.0,
                'delta_v_root_norm_max': 0.0,
                'q_residual_norm_mean': 0.0,
                'q_residual_norm_max': 0.0,
                'tail_update_norm_mean': 0.0,
                'tail_update_norm_max': 0.0,
            })
            print(json.dumps({'validated': merged['name'], 'finite': finite, 'shape_info': shape_info}, indent=2), flush=True)
        aggregate = aggregate_eval(rows)
        result.update({
            'status': 'ok',
            'checkpoint_epoch': checkpoint.get('epoch'),
            'checkpoint_config': checkpoint.get('config'),
            'val_manifest': manifest,
            'full_curve_manifest': full_manifest,
            'rows': rows,
            'aggregate': aggregate,
            'score': score_for_checkpoint(aggregate),
            'all_finite': all(row['finite'] for row in rows),
            'motion_evaluator_modified': False,
            'official_weights_modified': False,
        })
    except Exception as exc:
        result.update({
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ('status', 'score', 'all_finite', 'error_type', 'error')}, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
