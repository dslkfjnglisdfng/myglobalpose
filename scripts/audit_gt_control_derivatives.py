import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l4_tail_update_qstate import UniformCubicBSpline
from l4_train_diverse_short import load_cache_files


DT = 1.0 / 60.0


def stats(values):
    if not values:
        return {'mean': None, 'median': None, 'p95': None, 'max': None, 'count': 0}
    x = torch.cat([v.detach().cpu().float().reshape(-1) for v in values])
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {'mean': None, 'median': None, 'p95': None, 'max': None, 'count': 0}
    return {
        'mean': float(x.mean()),
        'median': float(x.median()),
        'p95': float(torch.quantile(x, 0.95)),
        'max': float(x.max()),
        'count': int(x.numel()),
    }


def finite_velocity(x, mode, dt=DT):
    out = torch.zeros_like(x)
    if x.shape[0] <= 1:
        return out
    if mode == 'backward':
        out[1:] = (x[1:] - x[:-1]) / dt
        out[0] = out[1]
        return out
    if mode == 'forward':
        out[:-1] = (x[1:] - x[:-1]) / dt
        out[-1] = out[-2]
        return out
    if mode == 'central':
        out[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        out[0] = (x[1] - x[0]) / dt
        out[-1] = (x[-1] - x[-2]) / dt
        return out
    raise ValueError(mode)


def finite_acceleration(x, dt=DT):
    out = torch.zeros_like(x)
    if x.shape[0] <= 2:
        return out
    out[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt ** 2)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def l2_per_frame_cm(x):
    return x.reshape(x.shape[0], -1, 3).norm(dim=-1).mean(dim=-1) * 100.0


def audit_sequence(target, controls, spline, trim_edges):
    q, qdot, qddot = spline(controls, return_derivatives=True)
    p = target[:, :69]
    c_q = q[:, :69]
    c_v = qdot[:, :69]
    c_a = qddot[:, :69]
    sl = slice(trim_edges, max(trim_edges, p.shape[0] - trim_edges)) if p.shape[0] > 2 * trim_edges else slice(0, p.shape[0])
    rows = {
        'reconstruct_l2_cm': l2_per_frame_cm(c_q[sl] - p[sl]),
        'control_v_l2_cm_s': l2_per_frame_cm(c_v[sl]),
        'diff_backward_v_l2_cm_s': l2_per_frame_cm(finite_velocity(p, 'backward')[sl]),
        'diff_central_v_l2_cm_s': l2_per_frame_cm(finite_velocity(p, 'central')[sl]),
        'diff_forward_v_l2_cm_s': l2_per_frame_cm(finite_velocity(p, 'forward')[sl]),
        'control_vs_backward_v_l2_cm_s': l2_per_frame_cm(c_v[sl] - finite_velocity(p, 'backward')[sl]),
        'control_vs_central_v_l2_cm_s': l2_per_frame_cm(c_v[sl] - finite_velocity(p, 'central')[sl]),
        'control_vs_forward_v_l2_cm_s': l2_per_frame_cm(c_v[sl] - finite_velocity(p, 'forward')[sl]),
        'control_a_l2_cm_s2': l2_per_frame_cm(c_a[sl]),
        'diff_a_l2_cm_s2': l2_per_frame_cm(finite_acceleration(p)[sl]),
        'control_vs_diff_a_l2_cm_s2': l2_per_frame_cm(c_a[sl] - finite_acceleration(p)[sl]),
    }
    return rows


def main():
    parser = argparse.ArgumentParser(description='Audit GT IK1 control-point derivatives against finite-difference GT joint derivatives.')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--trim-edges', type=int, default=2)
    args = parser.parse_args()

    files, manifest = load_cache_files(args.cache)
    if manifest is not None and manifest.get('type') != 'newik1_control_cache_v1':
        raise RuntimeError(f'Expected newik1_control_cache_v1, got {manifest.get("type")}.')

    spline = UniformCubicBSpline(DT)
    totals = {}
    rows = []
    seen = 0
    for cache_file in files:
        data = torch.load(cache_file, map_location='cpu')
        for seq_idx, name in enumerate(data['name']):
            target = data['ik1_target'][seq_idx].float()
            # The last entry of each frame's padded tail is exactly C[i].
            controls = data['ik1_target_control_tail'][seq_idx].float()[:, -1, :]
            seq_rows = audit_sequence(target, controls, spline, args.trim_edges)
            row = {'name': name, 'num_frames': int(target.shape[0])}
            for key, value in seq_rows.items():
                totals.setdefault(key, []).append(value)
                row[key] = stats([value])
            rows.append(row)
            seen += 1
            if args.max_sequences and seen >= args.max_sequences:
                break
        if args.max_sequences and seen >= args.max_sequences:
            break

    result = {
        'status': 'ok',
        'cache': str(args.cache),
        'num_sequences': seen,
        'trim_edges': args.trim_edges,
        'units': {
            'reconstruct_l2_cm': 'cm',
            'velocity': 'cm/s',
            'acceleration': 'cm/s^2',
        },
        'aggregate': {key: stats(value) for key, value in totals.items()},
        'rows': rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'ok', 'output_json': str(args.output_json), 'num_sequences': seen, 'aggregate': result['aggregate']}, indent=2))


if __name__ == '__main__':
    main()
