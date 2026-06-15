import argparse
import json
import traceback
from pathlib import Path

import torch

from l4_train_diverse_short import DEVICE, aggregate_eval, load_records, metric_to_dict, score_for_checkpoint
from net import GPNet
from pl_curve import build_pl_curve_model
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
    if 'model_variant' not in config and checkpoint.get('model_variant'):
        config = dict(config)
        config['model_variant'] = checkpoint.get('model_variant')
    model = build_pl_curve_model(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


class HybridBaselinePRBNewGR1(torch.nn.Module):
    """Evaluation-only PL wrapper: official PL pRB plus checkpoint gR1.

    GPNet first computes the official PL output `base_pl_t`.  This wrapper runs
    a NewPL checkpoint normally, keeps only its gravity direction, and returns
    `base_pl_t[:15] + newpl_gR1[3]` as the PL output consumed by IK1.
    """

    def __init__(self, gR1_model):
        super().__init__()
        self.gR1_model = gR1_model
        self.input_size = getattr(gR1_model, 'input_size', 84)
        self.init_size = getattr(gR1_model, 'init_size', 18)
        self.control_buffer = None
        self.last_debug = {}

    def reset_stream(self, init_output=None, init_feature=None):
        self.gR1_model.reset_stream(init_output=init_output, init_feature=init_feature)
        self.control_buffer = getattr(self.gR1_model, 'control_buffer', None)
        self.last_debug = {}

    def step(self, feature_t, base_pl_t):
        out = self.gR1_model.step(feature_t, base_pl_t)
        base = base_pl_t if base_pl_t.dim() > 1 else base_pl_t.unsqueeze(0)
        new_g = out['pl_t'][..., 15:]
        hybrid = torch.cat((base[..., :15], new_g), dim=-1)
        result = dict(out)
        result['candidate_pl_t'] = out['pl_t']
        result['pl_t'] = hybrid
        result['hybrid_source'] = 'baseline_pRB_plus_checkpoint_gR1'
        self.control_buffer = getattr(self.gR1_model, 'control_buffer', None)
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return result


def build_hybrid_baseline_pRB_new_gR1(checkpoint_path):
    model, config = build_pl_curve(checkpoint_path)
    hybrid = HybridBaselinePRBNewGR1(model).to(DEVICE)
    hybrid.eval()
    hybrid_config = dict(config or {})
    hybrid_config['hybrid_pl_mode'] = 'baseline_pRB_plus_checkpoint_gR1'
    hybrid_config['gR1_checkpoint'] = str(checkpoint_path)
    return hybrid, hybrid_config


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
def evaluate(
    records,
    pl_curve=None,
    max_eval_sequences=0,
    imu_input_mode='official',
    skip_baseline_rerun=False,
    force_baseline_rerun=False,
):
    evaluator = MotionEvaluator()
    rows = []
    selected = records[:max_eval_sequences] if max_eval_sequences else records
    for record in selected:
        output = run_sequence(record, pl_curve=pl_curve, imu_input_mode=imu_input_mode)
        if pl_curve is None:
            pose_ref, tran_ref = output['pose'], output['tran']
        elif skip_baseline_rerun:
            pose_ref, tran_ref = output['pose'], output['tran']
        elif 'pose_baseline' in record and not force_baseline_rerun:
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
    parser.add_argument('--hybrid-gR1-checkpoint', type=Path)
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='official')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--smoke-sequence', default='')
    parser.add_argument('--max-smoke-frames', type=int, default=0)
    parser.add_argument('--skip-baseline-rerun', action='store_true')
    parser.add_argument('--force-baseline-rerun', action='store_true')
    args = parser.parse_args()
    if args.checkpoint and args.hybrid_gR1_checkpoint:
        raise SystemExit('--checkpoint and --hybrid-gR1-checkpoint are mutually exclusive.')
    result = {
        'checkpoint': str(args.checkpoint or args.hybrid_gR1_checkpoint) if (args.checkpoint or args.hybrid_gR1_checkpoint) else None,
        'pl_backend': 'hybrid_baseline_pRB_new_gR1' if args.hybrid_gR1_checkpoint else ('curve_v1' if args.checkpoint else 'original'),
        'imu_input_mode': args.imu_input_mode,
        'val_cache': str(args.val_cache),
        'force_baseline_rerun': bool(args.force_baseline_rerun),
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
        elif args.hybrid_gR1_checkpoint:
            pl_curve, config = build_hybrid_baseline_pRB_new_gR1(args.hybrid_gR1_checkpoint)
        rows = evaluate(
            records,
            pl_curve=pl_curve,
            max_eval_sequences=args.max_eval_sequences,
            imu_input_mode=args.imu_input_mode,
            skip_baseline_rerun=args.skip_baseline_rerun,
            force_baseline_rerun=args.force_baseline_rerun,
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
