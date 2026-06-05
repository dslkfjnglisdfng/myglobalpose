import argparse
import json
import traceback
from pathlib import Path

import torch

from l4_train_diverse_short import DEVICE, aggregate_eval, load_records, metric_to_dict, score_for_checkpoint
from net import GPNet
from pl_curve import PLCurveModule
from test import MotionEvaluator


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


@torch.no_grad()
def run_sequence(record, pl_curve=None, imu_input_mode='official'):
    net = GPNet(
        pl_backend='curve_v1' if pl_curve is not None else 'original',
        pl_curve_module=pl_curve,
    ).eval().to(DEVICE)
    net.rnn_initialize(record['pose_gt'][0], offset_r=record.get('offset_r'))
    pose = torch.zeros_like(record['pose_gt'])
    tran = torch.zeros_like(record['tran_gt'])
    a_seq, w_seq, R_seq = selected_imu_fields(record, imu_input_mode)
    for i in range(record['pose_gt'].shape[0]):
        pose[i], tran[i] = net.forward_frame(
            a_seq[i].to(DEVICE),
            w_seq[i].to(DEVICE),
            R_seq[i].to(DEVICE),
        )
    return {
        'pose': pose.cpu(),
        'tran': tran.cpu(),
        'finite': bool(torch.isfinite(pose).all() and torch.isfinite(tran).all()),
        'root_step_norm_max': float((tran[1:] - tran[:-1]).norm(dim=-1).max()) if tran.shape[0] > 1 else 0.0,
    }


@torch.no_grad()
def evaluate(records, pl_curve=None, max_eval_sequences=0, imu_input_mode='official'):
    evaluator = MotionEvaluator()
    rows = []
    selected = records[:max_eval_sequences] if max_eval_sequences else records
    for record in selected:
        output = run_sequence(record, pl_curve=pl_curve, imu_input_mode=imu_input_mode)
        if 'pose_baseline' in record and pl_curve is None:
            pose_ref, tran_ref = record['pose_baseline'], record['tran_baseline']
        elif 'pose_baseline' in record:
            pose_ref, tran_ref = record['pose_baseline'], record['tran_baseline']
        else:
            baseline = run_sequence(record, pl_curve=None, imu_input_mode=imu_input_mode)
            pose_ref, tran_ref = baseline['pose'], baseline['tran']
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
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Evaluate PLCurve_v1 inside official GPNet.')
    parser.add_argument('--val-cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='official')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--smoke-sequence', default='')
    parser.add_argument('--max-smoke-frames', type=int, default=0)
    args = parser.parse_args()
    result = {
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'pl_backend': 'curve_v1' if args.checkpoint else 'original',
        'imu_input_mode': args.imu_input_mode,
        'val_cache': str(args.val_cache),
        'status': 'started',
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        records, manifest = load_records(args.val_cache)
        if args.smoke_sequence:
            records = [record for record in records if record['name'] == args.smoke_sequence]
            if not records:
                raise KeyError(f'No sequence named {args.smoke_sequence!r}.')
            args.max_eval_sequences = 1
        if args.max_smoke_frames:
            for record in records:
                original_frames = record['pose_gt'].shape[0]
                for key, value in list(record.items()):
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == original_frames:
                        record[key] = value[:args.max_smoke_frames]
        pl_curve, config = (None, None)
        if args.checkpoint:
            pl_curve, config = build_pl_curve(args.checkpoint)
        rows = evaluate(
            records,
            pl_curve=pl_curve,
            max_eval_sequences=args.max_eval_sequences,
            imu_input_mode=args.imu_input_mode,
        )
        aggregate = aggregate_eval(rows)
        result.update({
            'status': 'ok',
            'checkpoint_config': config,
            'val_manifest': manifest,
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
    print(json.dumps({k: result.get(k) for k in ('status', 'pl_backend', 'score', 'all_finite', 'error_type', 'error')}, indent=2))
    if result['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
