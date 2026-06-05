import argparse
import json
from pathlib import Path

import torch

from l4_tail_update_qstate import StreamingTailUpdateQState
from k2_so3_curve import StreamingTailUpdateSO3State, q75_to_so3_state
from l4_train_diverse_short import (
    DEVICE,
    aggregate_eval,
    get_or_run_baseline,
    load_records,
    metric_to_dict,
    score_for_checkpoint,
)
from l4_train_loss_ablation import firstframe_init_feature, selected_imu_fields, slice_record
from net import GPNet
from pl_curve import PLCurveModule
from test import MotionEvaluator


def build_pl_curve(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get('config', {})
    model = PLCurveModule(
        init_size=int(config.get('init_size', 18)),
        hidden_size=int(config.get('hidden_size', 512)),
        tail_update=int(config.get('tail_length', 4)),
        residual_scale=float(config.get('residual_scale', 0.005)),
        dropout=float(config.get('dropout', 0.4)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def build_l4_model(checkpoint_path, control_point_refine_config=None):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = checkpoint.get('config', {})
    model_type = cfg.get('model_type', 'euler_q75_l4')
    model_cls = StreamingTailUpdateSO3State if model_type == 'k2_so3curve_v1' else StreamingTailUpdateQState
    model_kwargs = dict(
        hidden_size=cfg.get('hidden_size', 256),
        residual_scale=cfg.get('residual_scale', 0.005),
        velocity_residual_scale=cfg.get('velocity_residual_scale', 0.0),
        pose_input_mode=cfg.get('pose_input_mode', 'rot6d'),
        offset_conditioning=cfg.get('offset_conditioning', 'hidden_init'),
        rnn_init_mode=cfg.get('effective_rnn_init_mode', cfg.get('rnn_init_mode', 'offset_firstframe')),
        offset_init_scale=cfg.get('offset_init_scale', 0.2),
        dropout=cfg.get('dropout', 0.0),
        imu_feature_dropout=cfg.get('imu_feature_dropout', 0.0),
        acc_dropout=cfg.get('acc_dropout', 0.0),
        gyro_dropout=cfg.get('gyro_dropout', 0.0),
        orientation_dropout=cfg.get('orientation_dropout', 0.0),
    )
    if model_cls is StreamingTailUpdateQState:
        model_kwargs['control_stride'] = cfg.get('control_stride', 1)
    model = model_cls(**model_kwargs).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    imu_input_mode = cfg.get('imu_input_mode')
    if imu_input_mode == 'official':
        field_prefix = 'original'
    elif imu_input_mode == 'processed':
        field_prefix = 'l4'
    elif imu_input_mode == 'auto':
        field_prefix = 'auto'
    else:
        field_prefix = cfg.get('l4_imu_field_prefix', 'original')
        imu_input_mode = {'original': 'official', 'l4': 'processed', 'auto': 'auto'}.get(field_prefix, 'official')
    model.imu_input_mode = imu_input_mode
    model.l4_imu_field_prefix = field_prefix
    if control_point_refine_config is not None and hasattr(model, 'set_control_point_refine'):
        model.set_control_point_refine(control_point_refine_config)
    model.eval()
    return model, cfg


def delta_metrics(model_metrics, baseline_metrics):
    return {
        name: {
            'mean': model_metrics[name]['mean'] - baseline_metrics[name]['mean'],
            'std': model_metrics[name]['std'] - baseline_metrics[name]['std'],
        }
        for name in model_metrics
    }


@torch.no_grad()
def mean_or_zero(values):
    return sum(values) / max(1, len(values))


@torch.no_grad()
def run_sequence(
    model,
    record,
    physics_mode='original',
    qdot_velocity_blend=0.5,
    qstate_alpha=0.1,
    qstate_max_delta=0.1,
    qstate_gate='no_gate',
    pl_curve=None,
    cp_refine_debug_tensors=False,
):
    net = GPNet(
        enable_l4_prephysics=True,
        l4_prephysics_module=model,
        physics_mode=physics_mode,
        pl_backend='curve_v1' if pl_curve is not None else 'original',
        pl_curve_module=pl_curve,
        l4_qdot_velocity_blend=qdot_velocity_blend,
        l4_qstate_alpha=qstate_alpha,
        l4_qstate_max_delta=qstate_max_delta,
        l4_qstate_gate=qstate_gate,
    ).eval().to(DEVICE)
    net.rnn_initialize(record['pose_gt'][0], offset_r=record.get('offset_r'))
    if 'offset_r' in record and getattr(model, 'offset_conditioning', 'none') == 'hidden_init':
        model.reset_stream(record['offset_r'], firstframe_init_feature(model, record))

    pose_model = torch.zeros_like(record['pose_gt'])
    tran_model = torch.zeros_like(record['tran_gt'])
    adapter_velocity_delta_norms = []
    raw_velocity_delta_norms = []
    clamped_velocity_delta_norms = []
    qstate_gate_values = []
    vr_velocity_norms = []
    qdot_root_norms = []
    adapted_velocity_norms = []
    q75_target_norms = []
    qdot_target_norms = []
    qddot_target_norms = []
    cp_refine_loss_initial = []
    cp_refine_loss_final = []
    cp_refine_delta_mean = []
    cp_refine_delta_max = []
    cp_refine_q_drift = []
    cp_refine_qdot_norm = []
    cp_refine_qddot_norm = []
    cp_refine_contact_loss = []
    cp_refine_dyn_loss = []
    cp_refine_gate_mean = []
    cp_refine_gate_max = []
    cp_refine_gate_active_fraction = []
    cp_refine_all_finite = []
    cp_refine_control_shape = None
    cp_refine_control_decode_shape = None
    cp_refine_q_shape = None
    cp_refine_qdot_shape = None
    cp_refine_qddot_shape = None
    q_residual_norms = []
    tail_update_norms = []
    generated_control_values = []
    interpolated_frame_values = []
    qdot_shapes = []
    qddot_shapes = []
    cp_refine_tensor_records = []
    l4_a_seq, l4_w_seq, l4_R_seq = selected_imu_fields(record, model)
    base_so3_seq = None
    if getattr(model, 'model_type', '') == 'k2_so3curve_v1':
        base_so3_seq = q75_to_so3_state(
            record['q75_prephysics'].to(DEVICE),
            euler_seq=getattr(model, 'euler_seq', 'XYZ'),
        )
    for frame_idx in range(record['pose_gt'].shape[0]):
        if base_so3_seq is not None:
            model.current_base_so3_t = base_so3_seq[frame_idx]
        pose_model[frame_idx], tran_model[frame_idx] = net.forward_frame(
            record['aM'][frame_idx].to(DEVICE),
            record['wM'][frame_idx].to(DEVICE),
            record['RMB'][frame_idx].to(DEVICE),
            l4_a=l4_a_seq[frame_idx].to(DEVICE),
            l4_w=l4_w_seq[frame_idx].to(DEVICE),
            l4_R=l4_R_seq[frame_idx].to(DEVICE),
        )
        debug = getattr(net, 'last_l4_prephysics_debug', {})
        adapter = debug.get('physics_adapter', {})
        adapter_velocity_delta_norms.append(float(adapter.get('velocity_delta_norm', 0.0)))
        raw_velocity_delta_norms.append(float(adapter.get('raw_velocity_delta_norm', 0.0)))
        clamped_velocity_delta_norms.append(float(adapter.get('clamped_velocity_delta_norm', 0.0)))
        qstate_gate_values.append(float(adapter.get('qstate_gate_value', 0.0)))
        vr_velocity_norms.append(float(adapter.get('vr_velocity_norm', 0.0)))
        qdot_root_norms.append(float(adapter.get('qdot_root_norm', 0.0)))
        adapted_velocity_norms.append(float(adapter.get('adapted_velocity_norm', 0.0)))
        q75_target_norms.append(float(adapter.get('q75_target_norm', 0.0)))
        qdot_target_norms.append(float(adapter.get('qdot_target_norm', 0.0)))
        qddot_target_norms.append(float(adapter.get('qddot_target_norm', 0.0)))
        cp_refine = debug.get('control_point_refine', {})
        if cp_refine.get('enabled', False):
            cp_refine_loss_initial.append(float(cp_refine.get('loss_initial', 0.0)))
            cp_refine_loss_final.append(float(cp_refine.get('loss_final', 0.0)))
            cp_refine_delta_mean.append(float(cp_refine.get('delta_control_norm_mean', 0.0)))
            cp_refine_delta_max.append(float(cp_refine.get('delta_control_norm_max', 0.0)))
            cp_refine_q_drift.append(float(cp_refine.get('q_body_drift_rms', 0.0)))
            cp_refine_qdot_norm.append(float(cp_refine.get('qdot_body_norm_mean', 0.0)))
            cp_refine_qddot_norm.append(float(cp_refine.get('qddot_body_norm_mean', 0.0)))
            final_components = cp_refine.get('component_final', {})
            cp_refine_contact_loss.append(float(final_components.get('contact', 0.0)))
            cp_refine_dyn_loss.append(float(final_components.get('dyn_proxy', 0.0)))
            cp_refine_gate_mean.append(float(cp_refine.get('contact_gate_mean', 0.0)))
            cp_refine_gate_max.append(float(cp_refine.get('contact_gate_max', 0.0)))
            cp_refine_gate_active_fraction.append(float(cp_refine.get('contact_gate_active_fraction', 0.0)))
            cp_refine_all_finite.append(bool(cp_refine.get('all_finite', False)))
            cp_refine_control_shape = cp_refine.get('control_shape', cp_refine_control_shape)
            cp_refine_control_decode_shape = cp_refine.get('control_decode_shape', cp_refine_control_decode_shape)
            cp_refine_q_shape = cp_refine.get('q_shape', cp_refine_q_shape)
            cp_refine_qdot_shape = cp_refine.get('qdot_shape', cp_refine_qdot_shape)
            cp_refine_qddot_shape = cp_refine.get('qddot_shape', cp_refine_qddot_shape)
            if cp_refine_debug_tensors and torch.is_tensor(cp_refine.get('C_refined')):
                cp_refine_tensor_records.append({
                    'frame_idx': frame_idx,
                    'C_refined': cp_refine['C_refined'],
                    'q_refined': cp_refine['q_refined'],
                    'qdot_refined': cp_refine['qdot_refined'],
                    'qddot_refined': cp_refine['qddot_refined'],
                })
        residual = debug.get('residual')
        q_residual_norms.append(float(residual.norm()) if residual is not None else 0.0)
        tail_update_norms.append(float(debug.get('tail_delta_norm', 0.0)))
        generated_control_values.append(1.0 if debug.get('generated_control', True) else 0.0)
        interpolated_frame_values.append(1.0 if float(debug.get('decode_u', 0.0)) > 0.0 else 0.0)
        if torch.is_tensor(debug.get('qdot_after')):
            qdot_shapes.append(list(debug['qdot_after'].shape))
        if torch.is_tensor(debug.get('qddot_after')):
            qddot_shapes.append(list(debug['qddot_after'].shape))

    return {
        'pose': pose_model.cpu(),
        'tran': tran_model.cpu(),
        'adapter_velocity_delta_norm_mean': sum(adapter_velocity_delta_norms) / max(1, len(adapter_velocity_delta_norms)),
        'adapter_velocity_delta_norm_max': max(adapter_velocity_delta_norms) if adapter_velocity_delta_norms else 0.0,
        'raw_velocity_delta_norm_mean': mean_or_zero(raw_velocity_delta_norms),
        'raw_velocity_delta_norm_max': max(raw_velocity_delta_norms) if raw_velocity_delta_norms else 0.0,
        'clamped_velocity_delta_norm_mean': mean_or_zero(clamped_velocity_delta_norms),
        'clamped_velocity_delta_norm_max': max(clamped_velocity_delta_norms) if clamped_velocity_delta_norms else 0.0,
        'qstate_gate_mean': mean_or_zero(qstate_gate_values),
        'qstate_gate_max': max(qstate_gate_values) if qstate_gate_values else 0.0,
        'vr_velocity_norm_mean': mean_or_zero(vr_velocity_norms),
        'qdot_root_norm_mean': mean_or_zero(qdot_root_norms),
        'adapted_velocity_norm_mean': mean_or_zero(adapted_velocity_norms),
        'q75_target_norm_mean': mean_or_zero(q75_target_norms),
        'qdot_target_norm_mean': mean_or_zero(qdot_target_norms),
        'qddot_target_norm_mean': mean_or_zero(qddot_target_norms),
        'cp_refine_applied_frames': len(cp_refine_loss_final),
        'cp_refine_loss_initial_mean': mean_or_zero(cp_refine_loss_initial),
        'cp_refine_loss_final_mean': mean_or_zero(cp_refine_loss_final),
        'cp_refine_delta_control_norm_mean': mean_or_zero(cp_refine_delta_mean),
        'cp_refine_delta_control_norm_max': max(cp_refine_delta_max) if cp_refine_delta_max else 0.0,
        'cp_refine_q_body_drift_rms_mean': mean_or_zero(cp_refine_q_drift),
        'cp_refine_qdot_body_norm_mean': mean_or_zero(cp_refine_qdot_norm),
        'cp_refine_qddot_body_norm_mean': mean_or_zero(cp_refine_qddot_norm),
        'cp_refine_contact_loss_mean': mean_or_zero(cp_refine_contact_loss),
        'cp_refine_dyn_proxy_loss_mean': mean_or_zero(cp_refine_dyn_loss),
        'cp_refine_contact_gate_mean': mean_or_zero(cp_refine_gate_mean),
        'cp_refine_contact_gate_max': max(cp_refine_gate_max) if cp_refine_gate_max else 0.0,
        'cp_refine_contact_gate_active_fraction_mean': mean_or_zero(cp_refine_gate_active_fraction),
        'cp_refine_control_shape': cp_refine_control_shape,
        'cp_refine_control_decode_shape': cp_refine_control_decode_shape,
        'cp_refine_q_shape': cp_refine_q_shape,
        'cp_refine_qdot_shape': cp_refine_qdot_shape,
        'cp_refine_qddot_shape': cp_refine_qddot_shape,
        'cp_refine_tensor_records': cp_refine_tensor_records,
        'cp_refine_all_finite': all(cp_refine_all_finite) if cp_refine_all_finite else True,
        'q_residual_norm_mean': sum(q_residual_norms) / max(1, len(q_residual_norms)),
        'q_residual_norm_max': max(q_residual_norms) if q_residual_norms else 0.0,
        'tail_update_norm_mean': sum(tail_update_norms) / max(1, len(tail_update_norms)),
        'tail_update_norm_max': max(tail_update_norms) if tail_update_norms else 0.0,
        'generated_control_fraction': mean_or_zero(generated_control_values),
        'interpolated_frame_fraction': mean_or_zero(interpolated_frame_values),
        'control_stride': int(getattr(model, 'control_stride', 1)),
        'qdot_shape': qdot_shapes[-1] if qdot_shapes else None,
        'qddot_shape': qddot_shapes[-1] if qddot_shapes else None,
        'finite': bool(torch.isfinite(pose_model).all() and torch.isfinite(tran_model).all()),
        'root_step_norm_max': float((tran_model[1:] - tran_model[:-1]).norm(dim=-1).max()) if tran_model.shape[0] > 1 else 0.0,
    }


@torch.no_grad()
def evaluate(
    model,
    records,
    physics_mode='original',
    qdot_velocity_blend=0.5,
    qstate_alpha=0.1,
    qstate_max_delta=0.1,
    qstate_gate='no_gate',
    pl_curve=None,
    max_eval_sequences=0,
    cp_refine_debug_tensors=False,
):
    evaluator = MotionEvaluator()
    rows = []
    selected = records[:max_eval_sequences] if max_eval_sequences else records
    for record in selected:
        output = run_sequence(
            model,
            record,
            physics_mode=physics_mode,
            qdot_velocity_blend=qdot_velocity_blend,
            qstate_alpha=qstate_alpha,
            qstate_max_delta=qstate_max_delta,
            qstate_gate=qstate_gate,
            pl_curve=pl_curve,
            cp_refine_debug_tensors=cp_refine_debug_tensors,
        )
        pose_baseline, tran_baseline = get_or_run_baseline(record)
        baseline_metric = evaluator(
            pose_baseline.to(DEVICE),
            record['pose_gt'].to(DEVICE),
            tran_baseline.to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        model_metric = evaluator(
            output['pose'].to(DEVICE),
            record['pose_gt'].to(DEVICE),
            output['tran'].to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        baseline = metric_to_dict(baseline_metric)
        model_dict = metric_to_dict(model_metric)
        rows.append({
            'name': record['name'],
            'baseline_metrics': baseline,
            'model_metrics': model_dict,
            'delta_metrics': delta_metrics(model_dict, baseline),
            'delta_v_root_norm_mean': output['adapter_velocity_delta_norm_mean'],
            'delta_v_root_norm_max': output['adapter_velocity_delta_norm_max'],
            'adapter_velocity_delta_norm_mean': output['adapter_velocity_delta_norm_mean'],
            'adapter_velocity_delta_norm_max': output['adapter_velocity_delta_norm_max'],
            'raw_velocity_delta_norm_mean': output['raw_velocity_delta_norm_mean'],
            'raw_velocity_delta_norm_max': output['raw_velocity_delta_norm_max'],
            'clamped_velocity_delta_norm_mean': output['clamped_velocity_delta_norm_mean'],
            'clamped_velocity_delta_norm_max': output['clamped_velocity_delta_norm_max'],
            'qstate_gate_mean': output['qstate_gate_mean'],
            'qstate_gate_max': output['qstate_gate_max'],
            'vr_velocity_norm_mean': output['vr_velocity_norm_mean'],
            'qdot_root_norm_mean': output['qdot_root_norm_mean'],
            'adapted_velocity_norm_mean': output['adapted_velocity_norm_mean'],
            'q75_target_norm_mean': output['q75_target_norm_mean'],
            'qdot_target_norm_mean': output['qdot_target_norm_mean'],
            'qddot_target_norm_mean': output['qddot_target_norm_mean'],
            'cp_refine_applied_frames': output['cp_refine_applied_frames'],
            'cp_refine_loss_initial_mean': output['cp_refine_loss_initial_mean'],
            'cp_refine_loss_final_mean': output['cp_refine_loss_final_mean'],
            'cp_refine_delta_control_norm_mean': output['cp_refine_delta_control_norm_mean'],
            'cp_refine_delta_control_norm_max': output['cp_refine_delta_control_norm_max'],
            'cp_refine_q_body_drift_rms_mean': output['cp_refine_q_body_drift_rms_mean'],
            'cp_refine_qdot_body_norm_mean': output['cp_refine_qdot_body_norm_mean'],
            'cp_refine_qddot_body_norm_mean': output['cp_refine_qddot_body_norm_mean'],
            'cp_refine_contact_loss_mean': output['cp_refine_contact_loss_mean'],
            'cp_refine_dyn_proxy_loss_mean': output['cp_refine_dyn_proxy_loss_mean'],
            'cp_refine_contact_gate_mean': output['cp_refine_contact_gate_mean'],
            'cp_refine_contact_gate_max': output['cp_refine_contact_gate_max'],
            'cp_refine_contact_gate_active_fraction_mean': output['cp_refine_contact_gate_active_fraction_mean'],
            'cp_refine_control_shape': output['cp_refine_control_shape'],
            'cp_refine_control_decode_shape': output['cp_refine_control_decode_shape'],
            'cp_refine_q_shape': output['cp_refine_q_shape'],
            'cp_refine_qdot_shape': output['cp_refine_qdot_shape'],
            'cp_refine_qddot_shape': output['cp_refine_qddot_shape'],
            'cp_refine_all_finite': output['cp_refine_all_finite'],
            'q_residual_norm_mean': output['q_residual_norm_mean'],
            'q_residual_norm_max': output['q_residual_norm_max'],
            'tail_update_norm_mean': output['tail_update_norm_mean'],
            'tail_update_norm_max': output['tail_update_norm_max'],
            'generated_control_fraction': output['generated_control_fraction'],
            'interpolated_frame_fraction': output['interpolated_frame_fraction'],
            'control_stride': output['control_stride'],
            'qdot_shape': output['qdot_shape'],
            'qddot_shape': output['qddot_shape'],
            'finite': output['finite'],
            'root_step_norm_max': output['root_step_norm_max'],
        })
        if cp_refine_debug_tensors:
            rows[-1]['cp_refine_tensor_records'] = output['cp_refine_tensor_records']
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pl-curve-checkpoint', type=Path, default=None)
    parser.add_argument('--val-cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--physics-mode', choices=('original', 'l4_pip_v1', 'l4_pip_v2', 'l4_qstate_p1', 'l4_qstate_p2b'), default='original')
    parser.add_argument('--qdot-velocity-blend', type=float, default=0.5)
    parser.add_argument('--qstate-alpha', type=float, default=0.1)
    parser.add_argument('--qstate-max-delta', type=float, default=0.1)
    parser.add_argument('--qstate-gate', choices=('no_gate', 'stationary_gate'), default='no_gate')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--smoke-sequence', default='')
    parser.add_argument('--control-point-refine', action='store_true')
    parser.add_argument('--cp-refine-steps', type=int, default=10)
    parser.add_argument('--cp-refine-lr', type=float, default=3e-3)
    parser.add_argument('--cp-refine-lambda-prior', type=float, default=1.0)
    parser.add_argument('--cp-refine-lambda-q', type=float, default=1.0)
    parser.add_argument('--cp-refine-lambda-v', type=float, default=0.03)
    parser.add_argument('--cp-refine-lambda-a', type=float, default=0.0003)
    parser.add_argument('--cp-refine-lambda-contact', type=float, default=0.0)
    parser.add_argument('--cp-refine-lambda-dyn', type=float, default=0.0)
    parser.add_argument('--cp-refine-contact-gate-mode', choices=('heuristic',), default='heuristic')
    parser.add_argument('--cp-refine-contact-height-threshold', type=float, default=0.08)
    parser.add_argument('--cp-refine-contact-velocity-threshold', type=float, default=0.20)
    parser.add_argument('--cp-refine-window', type=int, default=0, help='Optimize only the most recent N control points. 0 keeps the historical full-buffer behavior.')
    parser.add_argument('--cp-refine-include-root', action='store_true')
    parser.add_argument('--cp-refine-persist-buffer', action='store_true')
    parser.add_argument('--cp-refine-debug-tensor-path', type=Path, default=None)
    args = parser.parse_args()

    cp_refine_config = None
    if args.control_point_refine:
        cp_refine_config = {
            'enabled': True,
            'steps': args.cp_refine_steps,
            'lr': args.cp_refine_lr,
            'lambda_prior': args.cp_refine_lambda_prior,
            'lambda_q': args.cp_refine_lambda_q,
            'lambda_v': args.cp_refine_lambda_v,
            'lambda_a': args.cp_refine_lambda_a,
            'lambda_contact': args.cp_refine_lambda_contact,
            'lambda_dyn': args.cp_refine_lambda_dyn,
            'contact_gate_mode': args.cp_refine_contact_gate_mode,
            'contact_height_threshold': args.cp_refine_contact_height_threshold,
            'contact_velocity_threshold': args.cp_refine_contact_velocity_threshold,
            'refine_window': args.cp_refine_window,
            'optimize_body_only': not args.cp_refine_include_root,
            'persist_refined_buffer': args.cp_refine_persist_buffer,
            'save_debug_tensors': args.cp_refine_debug_tensor_path is not None,
        }
    model, cfg = build_l4_model(args.checkpoint, cp_refine_config)
    pl_curve = None
    pl_curve_config = None
    if args.pl_curve_checkpoint is not None:
        pl_curve, pl_curve_config = build_pl_curve(args.pl_curve_checkpoint)
    records, manifest = load_records(args.val_cache)
    if args.smoke_sequence:
        records = [record for record in records if record['name'] == args.smoke_sequence]
        if not records:
            raise KeyError(f'No sequence named {args.smoke_sequence!r} in {args.val_cache}.')
        args.max_eval_sequences = 1
    if args.max_frames > 0:
        records = [slice_record(record, 0, args.max_frames) for record in records]

    rows = evaluate(
        model,
        records,
        physics_mode=args.physics_mode,
        qdot_velocity_blend=args.qdot_velocity_blend,
        qstate_alpha=args.qstate_alpha,
        qstate_max_delta=args.qstate_max_delta,
        qstate_gate=args.qstate_gate,
        pl_curve=pl_curve,
        max_eval_sequences=args.max_eval_sequences,
        cp_refine_debug_tensors=args.cp_refine_debug_tensor_path is not None,
    )
    if args.cp_refine_debug_tensor_path is not None:
        tensor_payload = []
        for row in rows:
            tensor_payload.append({
                'name': row['name'],
                'records': row.pop('cp_refine_tensor_records', []),
            })
        args.cp_refine_debug_tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_payload, args.cp_refine_debug_tensor_path)
    agg = aggregate_eval(rows)
    result = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_config': cfg,
        'pl_curve_checkpoint': str(args.pl_curve_checkpoint) if args.pl_curve_checkpoint is not None else None,
        'pl_curve_config': pl_curve_config,
        'pl_backend': 'curve_v1' if pl_curve is not None else 'original',
        'val_cache': str(args.val_cache),
        'val_manifest': manifest,
        'physics_mode': args.physics_mode,
        'qdot_velocity_blend': args.qdot_velocity_blend,
        'qstate_alpha': args.qstate_alpha,
        'qstate_max_delta': args.qstate_max_delta,
        'qstate_gate': args.qstate_gate,
        'control_point_refine': cp_refine_config,
        'max_eval_sequences': args.max_eval_sequences,
        'smoke_sequence': args.smoke_sequence,
        'rows': rows,
        'aggregate': agg,
        'score': score_for_checkpoint(agg),
        'all_finite': all(row['finite'] for row in rows),
        'cp_refine_all_finite': all(row.get('cp_refine_all_finite', True) for row in rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'output_json': str(args.output_json),
        'physics_mode': args.physics_mode,
        'pl_backend': result['pl_backend'],
        'score': result['score'],
        'all_finite': result['all_finite'],
        'num_sequences': len(rows),
    }, indent=2))


if __name__ == '__main__':
    main()
