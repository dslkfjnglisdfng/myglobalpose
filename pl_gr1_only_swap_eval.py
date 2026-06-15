import argparse
import json
import traceback
from pathlib import Path

import torch

import articulate as art
from l4_train_diverse_short import DEVICE, aggregate_eval, load_records, metric_to_dict, score_for_checkpoint
from net import GPNet
from pl_curve import build_pl_curve_model, normalize_gravity, pl_target_from_pose
from pl_curve_eval import selected_imu_fields
from pl_curve_train import load_cache_files
from test import MotionEvaluator


DT = 1.0 / 60.0


def angle_deg(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target, avoid_nan=True)
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def finite_diff(x, dt=DT):
    if x.shape[0] <= 1:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    if x.shape[0] == 2:
        d = (x[1] - x[0]) / float(dt)
        out[0] = d
        out[1] = d
        return out
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * float(dt))
    out[0] = (x[1] - x[0]) / float(dt)
    out[-1] = (x[-1] - x[-2]) / float(dt)
    return out


def jitter(x):
    if x.shape[0] < 3:
        return x.new_zeros(())
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]).reshape(x.shape[0] - 2, -1).norm(dim=-1).mean()


def vector_metrics(pred, target, label):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target, avoid_nan=True)
    vel_pred = finite_diff(pred)
    vel_target = finite_diff(target)
    return {
        f'{label}_angle_deg': float(angle_deg(pred, target).mean()),
        f'{label}_vel_vector_l2': float((vel_pred - vel_target).norm(dim=-1).mean()),
        f'{label}_jitter': float(jitter(pred)),
    }


def load_pl_curve(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get('config', {})
    if 'model_variant' not in config and checkpoint.get('model_variant'):
        config = dict(config)
        config['model_variant'] = checkpoint.get('model_variant')
    model = build_pl_curve_model(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def load_pl_records(cache_path, max_sequences=0):
    files, manifest = load_cache_files(cache_path)
    records = []
    required = ('name', 'pl_input', 'pl_base', 'pl_init_feature')
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f'{cache_file} missing fields: {missing}')
        for seq_idx, name in enumerate(data['name']):
            record = {
                'name': str(name),
                'pl_input': data['pl_input'][seq_idx].float(),
                'pl_base': normalize_gravity(data['pl_base'][seq_idx].float()),
                'pl_init_feature': data['pl_init_feature'][seq_idx].float(),
            }
            records.append(record)
            if max_sequences and len(records) >= max_sequences:
                return records, manifest
    return records, manifest


def slice_record(record, max_frames):
    if not max_frames:
        return record
    seq_len = None
    for value in record.values():
        if torch.is_tensor(value) and value.ndim > 0:
            seq_len = int(value.shape[0])
            break
    if seq_len is None or seq_len <= max_frames:
        return record
    out = {}
    for key, value in record.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == seq_len:
            out[key] = value[:max_frames].clone()
        else:
            out[key] = value
    out['name'] = record['name']
    return out


def align_records(raw_records, gr1_records):
    by_name = {record['name']: record for record in gr1_records}
    aligned = []
    missing = []
    for record in raw_records:
        match = by_name.get(record['name'])
        if match is None:
            missing.append(record['name'])
        else:
            aligned.append((record, match))
    if missing:
        raise KeyError(f'gR1 cache missing {len(missing)} sequences, first missing={missing[0]!r}.')
    return aligned


@torch.no_grad()
def run_v6_gr1_sequence(model, record):
    init = record['pl_init_feature'].to(DEVICE)
    if init.shape[-1] != getattr(model, 'init_size', init.shape[-1]):
        raise RuntimeError(
            f'PL init dim mismatch for {record["name"]}: model expects {model.init_size}, got {init.shape[-1]}.'
        )
    out = model.forward_sequence(
        record['pl_input'].to(DEVICE),
        record['pl_base'].to(DEVICE),
        init_feature=init,
    )
    return normalize_gravity(out['pl']).detach().cpu()[..., 15:]


def reset_net(net, record):
    net.rnn_initialize(record['pose_gt'][0], offset_r=record.get('offset_r'))


def run_ik1_trace(raw_record, gr1_record, v6_gr1, imu_input_mode):
    baseline = GPNet().eval().to(DEVICE)
    swap = GPNet().eval().to(DEVICE)
    reset_net(baseline, raw_record)
    reset_net(swap, raw_record)
    a_seq, w_seq, r_seq = selected_imu_fields(raw_record, imu_input_mode)
    body_ref = baseline.body_model._J
    target = pl_target_from_pose(
        raw_record['pose_gt'].to(device=body_ref.device, dtype=body_ref.dtype),
        baseline.body_model,
    ).detach().cpu()
    target = normalize_gravity(target)
    seq_len = raw_record['pose_gt'].shape[0]
    if gr1_record['pl_base'].shape[0] != seq_len or v6_gr1.shape[0] != seq_len:
        raise RuntimeError(
            f'sequence length mismatch for {raw_record["name"]}: raw={seq_len}, '
            f'gR1_base={gr1_record["pl_base"].shape[0]}, v6={v6_gr1.shape[0]}.'
        )
    traces = {
        'official_gR1': [],
        'official_gR2': [],
        'swap_gR1': [],
        'swap_gR2': [],
        'target_gR1': target[:, 15:],
        'official_pRB': [],
        'swap_pRB': [],
    }
    for frame_idx in range(seq_len):
        a = a_seq[frame_idx].to(DEVICE)
        w = w_seq[frame_idx].to(DEVICE)
        R = r_seq[frame_idx].to(DEVICE)

        base_dbg = baseline.forward_until_ik1(a, w, R)
        traces['official_gR1'].append(base_dbg['gR1'])
        traces['official_gR2'].append(base_dbg['gR2'])
        traces['official_pRB'].append(base_dbg['pRB'])

        aRB0 = a.mm(R[5])
        wRB0 = w.mm(R[5])
        RRB0 = R[5].t().matmul(R[:5])
        gR0 = -R[5, 1]
        pRB = base_dbg['pRB'].to(DEVICE)
        gR1 = art.math.normalize_tensor(v6_gr1[frame_idx].to(DEVICE), avoid_nan=True)
        RRB_after_pl = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB0)
        x_ik1, gR2 = swap._run_ik1_stage(RRB_after_pl, gR1, pRB)
        traces['swap_gR1'].append(gR1.detach().cpu())
        traces['swap_gR2'].append(gR2.detach().cpu())
        traces['swap_pRB'].append(pRB.detach().cpu())
    _ = aRB0, wRB0, x_ik1
    return {key: torch.stack(value) if isinstance(value, list) else value for key, value in traces.items()}


def run_baseline_full(raw_record, imu_input_mode):
    net = GPNet().eval().to(DEVICE)
    reset_net(net, raw_record)
    a_seq, w_seq, r_seq = selected_imu_fields(raw_record, imu_input_mode)
    pose = torch.zeros_like(raw_record['pose_gt'])
    tran = torch.zeros_like(raw_record['tran_gt'])
    for frame_idx in range(raw_record['pose_gt'].shape[0]):
        pose[frame_idx], tran[frame_idx] = net.forward_frame(
            a_seq[frame_idx].to(DEVICE),
            w_seq[frame_idx].to(DEVICE),
            r_seq[frame_idx].to(DEVICE),
        )
    return pose.cpu(), tran.cpu()


class GR1SwapPL(torch.nn.Module):
    input_size = 84
    init_size = 18

    def __init__(self, gr1_seq):
        super().__init__()
        self.gr1_seq = art.math.normalize_tensor(gr1_seq.detach().cpu(), avoid_nan=True)
        self.idx = 0
        self.control_buffer = None
        self.last_debug = {}

    def reset_stream(self, init_output=None, init_feature=None):
        self.idx = 0
        self.control_buffer = None
        self.last_debug = {}

    def step(self, feature_t, base_pl_t):
        base = base_pl_t if base_pl_t.dim() > 1 else base_pl_t.unsqueeze(0)
        if self.idx >= self.gr1_seq.shape[0]:
            raise IndexError(f'gR1 sequence exhausted at frame {self.idx}.')
        gr1 = self.gr1_seq[self.idx].to(base.device, base.dtype).view(1, 3)
        self.idx += 1
        pl_t = torch.cat((base[..., :15], gr1), dim=-1)
        result = {
            'pl_t': pl_t,
            'base_t': normalize_gravity(base),
            'candidate_gR1_t': gr1,
            'hybrid_source': 'official_pRB_plus_v6_gR1',
        }
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return result


def run_swap_full(raw_record, v6_gr1, imu_input_mode):
    pl = GR1SwapPL(v6_gr1)
    net = GPNet(pl_backend='curve_v1', pl_curve_module=pl).eval().to(DEVICE)
    reset_net(net, raw_record)
    a_seq, w_seq, r_seq = selected_imu_fields(raw_record, imu_input_mode)
    pose = torch.zeros_like(raw_record['pose_gt'])
    tran = torch.zeros_like(raw_record['tran_gt'])
    for frame_idx in range(raw_record['pose_gt'].shape[0]):
        pose[frame_idx], tran[frame_idx] = net.forward_frame(
            a_seq[frame_idx].to(DEVICE),
            w_seq[frame_idx].to(DEVICE),
            r_seq[frame_idx].to(DEVICE),
        )
    return pose.cpu(), tran.cpu()


def summarize_trace(trace):
    official_pRB = trace['official_pRB']
    swap_pRB = trace['swap_pRB']
    return {
        'official': {
            **vector_metrics(trace['official_gR1'], trace['target_gR1'], 'gR1'),
            **vector_metrics(trace['official_gR2'], trace['target_gR1'], 'gR2'),
        },
        'swap': {
            **vector_metrics(trace['swap_gR1'], trace['target_gR1'], 'gR1'),
            **vector_metrics(trace['swap_gR2'], trace['target_gR1'], 'gR2'),
        },
        'pRB_fixed_max_abs_delta': float((official_pRB - swap_pRB).abs().max()),
    }


def mean_trace_metrics(rows):
    out = {}
    for section in ('official', 'swap'):
        out[section] = {}
        keys = rows[0]['trace_metrics'][section].keys()
        for key in keys:
            out[section][key] = sum(row['trace_metrics'][section][key] for row in rows) / len(rows)
    out['delta_swap_minus_official'] = {
        key: out['swap'][key] - out['official'][key]
        for key in out['official']
        if key in out['swap']
    }
    out['pRB_fixed_max_abs_delta'] = max(row['trace_metrics']['pRB_fixed_max_abs_delta'] for row in rows)
    return out


def score_for_metric_section(metrics):
    # Same project score as score_for_checkpoint(), but applicable to either
    # official baseline metrics or swap/model metrics.
    return (
        metrics['L SIP Err (deg)']['mean']
        + metrics['L Angle Err (deg)']['mean']
        + metrics['G SIP Err (deg)']['mean']
        + metrics['G Angle Err (deg)']['mean']
        + 0.1 * metrics['L Joint Err (cm)']['mean']
        + 0.1 * metrics['G Joint Err (cm)']['mean']
        + 0.01 * metrics['Joint Jitter (km/s^3)']['mean']
    )


@torch.no_grad()
def evaluate(args):
    raw_records, raw_manifest = load_records(args.raw_cache, max_sequences=args.max_eval_sequences)
    gr1_records, gr1_manifest = load_pl_records(args.gr1_cache, max_sequences=args.max_eval_sequences)
    raw_records = [slice_record(record, args.max_frames_per_sequence) for record in raw_records]
    gr1_records = [slice_record(record, args.max_frames_per_sequence) for record in gr1_records]
    pairs = align_records(raw_records, gr1_records)
    model, model_config = load_pl_curve(args.v6_gR1_checkpoint)
    evaluator = MotionEvaluator()
    rows = []
    for raw_record, gr1_record in pairs:
        v6_gr1 = run_v6_gr1_sequence(model, gr1_record)
        trace = run_ik1_trace(raw_record, gr1_record, v6_gr1, args.imu_input_mode)
        trace_metrics = summarize_trace(trace)
        baseline_pose, baseline_tran = run_baseline_full(raw_record, args.imu_input_mode)
        swap_pose, swap_tran = run_swap_full(raw_record, v6_gr1, args.imu_input_mode)
        baseline_metric = evaluator(
            baseline_pose.to(DEVICE),
            raw_record['pose_gt'].to(DEVICE),
            baseline_tran.to(DEVICE),
            raw_record['tran_gt'].to(DEVICE),
        ).cpu()
        swap_metric = evaluator(
            swap_pose.to(DEVICE),
            raw_record['pose_gt'].to(DEVICE),
            swap_tran.to(DEVICE),
            raw_record['tran_gt'].to(DEVICE),
        ).cpu()
        rows.append({
            'name': raw_record['name'],
            'num_frames': int(raw_record['pose_gt'].shape[0]),
            'trace_metrics': trace_metrics,
            'baseline_metrics': metric_to_dict(baseline_metric),
            'model_metrics': metric_to_dict(swap_metric),
            'delta_v_root_norm_mean': 0.0,
            'delta_v_root_norm_max': 0.0,
            'q_residual_norm_mean': 0.0,
            'q_residual_norm_max': 0.0,
            'tail_update_norm_mean': 0.0,
            'tail_update_norm_max': 0.0,
            'finite': bool(
                torch.isfinite(baseline_pose).all()
                and torch.isfinite(baseline_tran).all()
                and torch.isfinite(swap_pose).all()
                and torch.isfinite(swap_tran).all()
            ),
            'root_step_norm_max': float((swap_tran[1:] - swap_tran[:-1]).norm(dim=-1).max()) if swap_tran.shape[0] > 1 else 0.0,
        })
    aggregate = aggregate_eval(rows)
    aggregate['trace_metrics'] = mean_trace_metrics(rows) if rows else {}
    official_score = score_for_metric_section(aggregate['baseline_metrics'])
    swap_score = score_for_metric_section(aggregate['model_metrics'])
    return {
        'status': 'ok',
        'dataset_label': args.dataset_label,
        'raw_cache': str(args.raw_cache),
        'gr1_cache': str(args.gr1_cache),
        'v6_gR1_checkpoint': str(args.v6_gR1_checkpoint),
        'imu_input_mode': args.imu_input_mode,
        'evaluation_contract': {
            'purpose': 'gR1-only downstream swap: keep official pRB[15] fixed and replace only gR1[3] with v6 prediction.',
            'baseline': 'official PL pRB + official PL gR1, original IK1/IK2/VR/physics.',
            'swap': 'official PL pRB + v6 gR1, original IK1/IK2/VR/physics.',
            'pRB_claim': 'pRB is a fixed control variable; pRB metrics are not used for success.',
            'gR2_target': 'gR2 is compared to the same root gravity target used for PL gR1 diagnostics.',
            'dip_translation_policy': 'DIP translation is used only by the existing MotionEvaluator when present in the cache; no root velocity or translation GT is fabricated.',
        },
        'raw_manifest': raw_manifest,
        'gr1_manifest': gr1_manifest,
        'v6_gR1_checkpoint_config': model_config,
        'rows': rows,
        'aggregate': aggregate,
        'official_score': official_score,
        'swap_score': swap_score,
        'score': score_for_checkpoint(aggregate),
        'score_delta_swap_minus_official': swap_score - official_score,
        'all_finite': all(row['finite'] for row in rows),
        'pRB_fixed_check_passed': bool(aggregate.get('trace_metrics', {}).get('pRB_fixed_max_abs_delta', 1.0) <= args.prb_fixed_tolerance),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate gR1-only swap: official pRB + v6 gR1.')
    parser.add_argument('--raw-cache', type=Path, required=True)
    parser.add_argument('--gr1-cache', type=Path, required=True)
    parser.add_argument('--v6-gR1-checkpoint', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--dataset-label', default='')
    parser.add_argument('--imu-input-mode', choices=('official', 'processed', 'auto'), default='official')
    parser.add_argument('--max-eval-sequences', type=int, default=0)
    parser.add_argument('--max-frames-per-sequence', type=int, default=0)
    parser.add_argument('--prb-fixed-tolerance', type=float, default=1e-7)
    args = parser.parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate(args)
    except Exception as exc:
        result = {
            'status': 'failed',
            'dataset_label': args.dataset_label,
            'raw_cache': str(args.raw_cache),
            'gr1_cache': str(args.gr1_cache),
            'v6_gR1_checkpoint': str(args.v6_gR1_checkpoint),
            'imu_input_mode': args.imu_input_mode,
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        }
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': result.get('status'),
        'dataset_label': result.get('dataset_label'),
        'official_score': result.get('official_score'),
        'swap_score': result.get('swap_score'),
        'score_delta_swap_minus_official': result.get('score_delta_swap_minus_official'),
        'all_finite': result.get('all_finite'),
        'pRB_fixed_check_passed': result.get('pRB_fixed_check_passed'),
        'error_type': result.get('error_type'),
        'error': result.get('error'),
    }, indent=2))
    if result.get('status') != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
