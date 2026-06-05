import argparse
import json
import traceback
from pathlib import Path

import torch

from l4_physics_adapter_eval import build_l4_model
from l4_train_diverse_short import (
    DEVICE,
    aggregate_eval,
    get_or_run_baseline,
    load_records,
    metric_to_dict,
    score_for_checkpoint,
)
from l4_train_loss_ablation import firstframe_init_feature, selected_imu_fields
from net import GPNet
from pip_physics_backend import q_conversion_diagnostics
from test import MotionEvaluator


def direct_imu_fields(record, imu_input_mode):
    if imu_input_mode == 'official':
        return record['aM'], record['wM'], record['RMB']
    if imu_input_mode == 'processed':
        missing = [key for key in ('l4_aM', 'l4_wM', 'l4_RMB') if key not in record]
        if missing:
            raise KeyError(f'--imu-input-mode processed requires fields missing from record {record.get("name")}: {missing}')
        return record['l4_aM'], record['l4_wM'], record['l4_RMB']
    raise ValueError(f'Unsupported imu_input_mode: {imu_input_mode}')


@torch.no_grad()
def run_gpnet_sequence(record, physics_backend='original_carticulate', l4_model=None, imu_input_mode='official'):
    net = GPNet(
        enable_l4_prephysics=l4_model is not None,
        l4_prephysics_module=l4_model,
        physics_mode='original',
        physics_backend=physics_backend,
    ).eval().to(DEVICE)
    net.rnn_initialize(record['pose_gt'][0])
    if l4_model is not None and 'offset_r' in record and getattr(l4_model, 'offset_conditioning', 'none') == 'hidden_init':
        l4_model.reset_stream(record['offset_r'], firstframe_init_feature(l4_model, record))
        l4_a_seq, l4_w_seq, l4_R_seq = selected_imu_fields(record, l4_model)
    else:
        l4_a_seq = l4_w_seq = l4_R_seq = None

    pose_model = torch.zeros_like(record['pose_gt'])
    tran_model = torch.zeros_like(record['tran_gt'])
    pip_debug = []
    a_seq, w_seq, R_seq = direct_imu_fields(record, imu_input_mode)
    for frame_idx in range(record['pose_gt'].shape[0]):
        kwargs = {}
        if l4_model is not None:
            kwargs = {
                'l4_a': l4_a_seq[frame_idx].to(DEVICE),
                'l4_w': l4_w_seq[frame_idx].to(DEVICE),
                'l4_R': l4_R_seq[frame_idx].to(DEVICE),
            }
        pose_model[frame_idx], tran_model[frame_idx] = net.forward_frame(
            a_seq[frame_idx].to(DEVICE),
            w_seq[frame_idx].to(DEVICE),
            R_seq[frame_idx].to(DEVICE),
            **kwargs,
        )
        if physics_backend == 'pip_physics_v1':
            pip_debug.append(dict(getattr(net, 'last_pip_physics_debug', {})))
    return {
        'pose': pose_model.cpu(),
        'tran': tran_model.cpu(),
        'finite': bool(torch.isfinite(pose_model).all() and torch.isfinite(tran_model).all()),
        'root_step_norm_max': float((tran_model[1:] - tran_model[:-1]).norm(dim=-1).max()) if pose_model.shape[0] > 1 else 0.0,
        'pip_debug_tail': pip_debug[-5:],
    }


@torch.no_grad()
def evaluate(records, physics_backend='original_carticulate', l4_model=None, max_eval_sequences=0, imu_input_mode='official'):
    evaluator = MotionEvaluator()
    rows = []
    selected = records[:max_eval_sequences] if max_eval_sequences else records
    for record in selected:
        output = run_gpnet_sequence(
            record,
            physics_backend=physics_backend,
            l4_model=l4_model,
            imu_input_mode=imu_input_mode,
        )
        if physics_backend == 'original_carticulate' and l4_model is None and 'pose_baseline' in record:
            pose_ref, tran_ref = record['pose_baseline'], record['tran_baseline']
        else:
            pose_ref, tran_ref = get_or_run_baseline(record)
        baseline_metric = evaluator(
            pose_ref.to(DEVICE),
            record['pose_gt'].to(DEVICE),
            tran_ref.to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        model_metric = evaluator(
            output['pose'].to(DEVICE),
            record['pose_gt'].to(DEVICE),
            output['tran'].to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        rows.append({
            'name': record['name'],
            'baseline_metrics': metric_to_dict(baseline_metric),
            'model_metrics': metric_to_dict(model_metric),
            'delta_v_root_norm_mean': 0.0,
            'delta_v_root_norm_max': 0.0,
            'q_residual_norm_mean': 0.0,
            'q_residual_norm_max': 0.0,
            'tail_update_norm_mean': 0.0,
            'tail_update_norm_max': 0.0,
            'finite': output['finite'],
            'root_step_norm_max': output['root_step_norm_max'],
            'pip_debug_tail': output['pip_debug_tail'],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--physics-backend', choices=('original_carticulate', 'pip_physics_v1'), default='original_carticulate')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed'), default='official',
                        help='Direct GPNet input source. official uses aM/wM/RMB; processed uses l4_aM/l4_wM/l4_RMB.')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--smoke-sequence', default='')
    parser.add_argument('--max-smoke-frames', type=int, default=0)
    args = parser.parse_args()

    result = {
        'physics_backend': args.physics_backend,
        'imu_input_mode': args.imu_input_mode,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'val_cache': str(args.val_cache),
        'status': 'started',
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        records, manifest = load_records(args.val_cache)
        if args.smoke_sequence:
            records = [record for record in records if record['name'] == args.smoke_sequence]
            if not records:
                raise KeyError(f'No sequence named {args.smoke_sequence!r} in {args.val_cache}.')
            args.max_eval_sequences = 1
        if args.max_smoke_frames:
            for record in records:
                original_frames = record['pose_gt'].shape[0]
                for key, value in list(record.items()):
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == original_frames:
                        record[key] = value[:args.max_smoke_frames]

        conv_tran = records[0].get('tran_baseline', records[0]['tran_gt'])[0]
        conv_pose = records[0].get('pose_baseline', records[0]['pose_gt'])[0]
        result['q_conversion_diagnostics'] = q_conversion_diagnostics(conv_pose, conv_tran)
        l4_model, checkpoint_config = (None, None)
        if args.checkpoint:
            l4_model, checkpoint_config = build_l4_model(args.checkpoint)
        rows = evaluate(
            records,
            physics_backend=args.physics_backend,
            l4_model=l4_model,
            max_eval_sequences=args.max_eval_sequences,
            imu_input_mode=args.imu_input_mode,
        )
        aggregate = aggregate_eval(rows)
        result.update({
            'status': 'ok',
            'val_manifest': manifest,
            'checkpoint_config': checkpoint_config,
            'rows': rows,
            'aggregate': aggregate,
            'score': score_for_checkpoint(aggregate),
            'all_finite': all(row['finite'] for row in rows),
        })
    except Exception as exc:
        result.update({
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ('status', 'physics_backend', 'imu_input_mode', 'score', 'all_finite', 'error_type', 'error')}, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
