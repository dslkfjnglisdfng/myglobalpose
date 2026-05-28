import argparse
import json
import random
from pathlib import Path

import torch

from l4_q75_utils import prephysics_feature, q75_to_pose_tran
from l4_tail_update_qstate import StreamingTailUpdateQState
from l4_velocity_losses import finite_difference_translation_velocity, velocity_residual_losses
from test import MotionEvaluator


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_cache_files(cache_path):
    path = Path(cache_path)
    if path.suffix == '.json':
        manifest = json.loads(path.read_text())
        files = [Path(item['path']) for item in manifest['cache_files']]
        return files
    return [path]


def load_records(cache_path, max_sequences=0):
    records = []
    for cache_file in load_cache_files(cache_path):
        data = torch.load(cache_file, map_location='cpu')
        required = ('q75_prephysics', 'v_root_vr', 'stationary_prob', 'q75_gt', 'tran_baseline', 'pose_baseline', 'aM', 'wM', 'RMB')
        missing = [key for key in required if key not in data or not data[key]]
        if missing:
            raise KeyError(f'{cache_file} missing required smoke fields: {missing}')
        for seq_idx, name in enumerate(data['name']):
            pose_gt, tran_gt = q75_to_pose_tran(data['q75_gt'][seq_idx].float())
            records.append({
                'name': name,
                'q75_prephysics': data['q75_prephysics'][seq_idx].float(),
                'v_root_vr': data['v_root_vr'][seq_idx].float(),
                'stationary_prob': data['stationary_prob'][seq_idx].float(),
                'q75_gt': data['q75_gt'][seq_idx].float(),
                'pose_gt': pose_gt.float(),
                'tran_gt': tran_gt.float(),
                'pose_baseline': data['pose_baseline'][seq_idx].float(),
                'tran_baseline': data['tran_baseline'][seq_idx].float(),
                'aM': data['aM'][seq_idx].float(),
                'wM': data['wM'][seq_idx].float(),
                'RMB': data['RMB'][seq_idx].float(),
            })
            if max_sequences and len(records) >= max_sequences:
                return records
    return records


def split_records(records, val_ratio=0.5, seed=42):
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 1
    val_ids = set(indices[:n_val])
    train = [record for idx, record in enumerate(records) if idx not in val_ids]
    val = [record for idx, record in enumerate(records) if idx in val_ids]
    return train or val, val


def run_sequence(model, record, train=False):
    model.reset_stream()
    refined = []
    deltas = []
    for frame_idx in range(record['q75_prephysics'].shape[0]):
        q = record['q75_prephysics'][frame_idx].to(DEVICE)
        feature = prephysics_feature(
            q.detach().cpu(),
            record['aM'][frame_idx],
            record['wM'][frame_idx],
            record['RMB'][frame_idx],
        ).to(DEVICE)
        model.step(feature, q)
        result = model.refine_velocity(
            record['v_root_vr'][frame_idx].to(DEVICE),
            record['stationary_prob'][frame_idx].to(DEVICE),
        )
        refined.append(result['v_root_refined'][0])
        deltas.append(result['delta_v_root'][0])
    return torch.stack(refined), torch.stack(deltas)


def train_epoch(model, records, optimizer):
    model.train()
    totals = {'total': 0.0}
    for record in records:
        v_refined, delta_v = run_sequence(model, record, train=True)
        v_gt = finite_difference_translation_velocity(record['tran_gt'].to(DEVICE))
        v_base = record['v_root_vr'].to(DEVICE)
        loss, components, _ = velocity_residual_losses(v_refined, v_gt, v_base, delta_v)
        optimizer.zero_grad()
        loss.backward()
        grad_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not grad_ok:
            raise RuntimeError('Non-finite gradient detected.')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals['total'] += float(loss.detach())
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / max(1, len(records)) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, records):
    model.eval()
    evaluator = MotionEvaluator()
    metric_names = MotionEvaluator.names
    rows = []
    for record in records:
        v_refined, delta_v = run_sequence(model, record)
        tran_refined = torch.zeros_like(record['tran_baseline'])
        tran_refined[0] = record['tran_baseline'][0]
        if tran_refined.shape[0] > 1:
            tran_refined[1:] = tran_refined[0] + torch.cumsum(v_refined[:-1].cpu() / 60.0, dim=0)
        baseline_metric = evaluator(
            record['pose_baseline'].to(DEVICE),
            record['pose_gt'].to(DEVICE),
            record['tran_baseline'].to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        refined_metric = evaluator(
            record['pose_baseline'].to(DEVICE),
            record['pose_gt'].to(DEVICE),
            tran_refined.to(DEVICE),
            record['tran_gt'].to(DEVICE),
        ).cpu()
        rows.append({
            'name': record['name'],
            'delta_v_norm_mean': float(delta_v.norm(dim=-1).mean().cpu()),
            'baseline_metrics': {metric_names[i]: {'mean': float(baseline_metric[i, 0]), 'std': float(baseline_metric[i, 1])} for i in range(len(metric_names))},
            'refined_metrics': {metric_names[i]: {'mean': float(refined_metric[i, 0]), 'std': float(refined_metric[i, 1])} for i in range(len(metric_names))},
            'delta_metrics': {metric_names[i]: {'mean': float(refined_metric[i, 0] - baseline_metric[i, 0]), 'std': float(refined_metric[i, 1] - baseline_metric[i, 1])} for i in range(len(metric_names))},
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Tiny smoke for L4 pre-physics root velocity residual.')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--max-sequences', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--velocity-residual-scale', type=float, default=0.02)
    args = parser.parse_args()

    records = load_records(args.cache, max_sequences=args.max_sequences)
    if not records:
        raise RuntimeError('No cache records loaded.')
    train_records, val_records = split_records(records)
    model = StreamingTailUpdateQState(
        hidden_size=args.hidden_size,
        velocity_residual_scale=args.velocity_residual_scale,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log = {
        'cache': args.cache,
        'epochs': args.epochs,
        'train_sequences': [record['name'] for record in train_records],
        'val_sequences': [record['name'] for record in val_records],
        'epochs_log': [],
    }
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_records, optimizer)
        val_metrics = evaluate(model, val_records)
        log['epochs_log'].append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_metrics': val_metrics,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'config': vars(args)}, output_dir / 'last.pt')
    (output_dir / 'smoke_result.json').write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
