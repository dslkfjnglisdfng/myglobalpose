import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import articulate as art
from l4_tail_update_qstate import UniformCubicBSpline
from l4_train_diverse_short import DEVICE, load_records
from pl_curve import fit_uniform_cubic_spline_controls


DT = 1.0 / 60.0
GRAVITY_M = torch.tensor([0.0, -9.8, 0.0], dtype=torch.float32)
IMU_VERTICES = (1961, 5424, 1176, 4662, 411, 3021)


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


def per_sensor_l2(x):
    return x.norm(dim=-1)


def velocity_fd(x, mode):
    out = torch.zeros_like(x)
    if x.shape[0] <= 1:
        return out
    if mode == 'backward':
        out[1:] = (x[1:] - x[:-1]) / DT
        out[0] = out[1]
    elif mode == 'forward':
        out[:-1] = (x[1:] - x[:-1]) / DT
        out[-1] = out[-2]
    elif mode == 'central':
        out[1:-1] = (x[2:] - x[:-2]) / (2.0 * DT)
        out[0] = (x[1] - x[0]) / DT
        out[-1] = (x[-1] - x[-2]) / DT
    elif mode == 'five_point':
        if x.shape[0] < 5:
            return velocity_fd(x, 'central')
        out[2:-2] = (-x[4:] + 8.0 * x[3:-1] - 8.0 * x[1:-3] + x[:-4]) / (12.0 * DT)
        out[:2] = velocity_fd(x, 'central')[:2]
        out[-2:] = velocity_fd(x, 'central')[-2:]
    else:
        raise ValueError(mode)
    return out


def acceleration_fd(x, mode):
    out = torch.zeros_like(x)
    if x.shape[0] <= 2:
        return out
    if mode == 'three_point':
        out[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (DT ** 2)
        out[0] = out[1]
        out[-1] = out[-2]
    elif mode == 'five_point':
        if x.shape[0] < 5:
            return acceleration_fd(x, 'three_point')
        out[2:-2] = (
            -x[4:] + 16.0 * x[3:-1] - 30.0 * x[2:-2] + 16.0 * x[1:-3] - x[:-4]
        ) / (12.0 * DT ** 2)
        out[:2] = acceleration_fd(x, 'three_point')[:2]
        out[-2:] = acceleration_fd(x, 'three_point')[-2:]
    else:
        raise ValueError(mode)
    return out


@torch.no_grad()
def imu_vertices_from_pose(record, body_model):
    pose = record['pose_gt'].to(DEVICE).float()
    tran = record['tran_gt'].to(DEVICE).float()
    _grot, _joints, verts = body_model.forward_kinematics(pose, None, tran, calc_mesh=True)
    return verts.detach().cpu().float()


def audit_record(record, body_model, spline, trim):
    p = imu_vertices_from_pose(record, body_model)
    controls = fit_uniform_cubic_spline_controls(p.reshape(p.shape[0], -1)).reshape_as(p)
    q, v_cp, a_cp = spline(controls.reshape(controls.shape[0], -1), return_derivatives=True)
    q = q.reshape_as(p)
    v_cp = v_cp.reshape_as(p)
    a_cp = a_cp.reshape_as(p)

    v_ref = velocity_fd(p, 'five_point')
    a_ref = acceleration_fd(p, 'five_point')
    sl = slice(trim, p.shape[0] - trim) if p.shape[0] > 2 * trim else slice(0, p.shape[0])
    aM = record.get('aM')
    out = {
        'control_reconstruct_pos_m': per_sensor_l2(q[sl] - p[sl]),
        'velocity_control_vs_5pt_m_s': per_sensor_l2(v_cp[sl] - v_ref[sl]),
        'velocity_central_vs_5pt_m_s': per_sensor_l2(velocity_fd(p, 'central')[sl] - v_ref[sl]),
        'velocity_backward_vs_5pt_m_s': per_sensor_l2(velocity_fd(p, 'backward')[sl] - v_ref[sl]),
        'velocity_forward_vs_5pt_m_s': per_sensor_l2(velocity_fd(p, 'forward')[sl] - v_ref[sl]),
        'acc_control_vs_5pt_m_s2': per_sensor_l2(a_cp[sl] - a_ref[sl]),
        'acc_3pt_vs_5pt_m_s2': per_sensor_l2(acceleration_fd(p, 'three_point')[sl] - a_ref[sl]),
        'velocity_ref_5pt_m_s': per_sensor_l2(v_ref[sl]),
        'acc_ref_5pt_m_s2': per_sensor_l2(a_ref[sl]),
    }
    if aM is not None:
        target = aM.float()[sl]
        gravity_modes = {
            'none': torch.zeros(1, 1, 3),
            'plus_g': GRAVITY_M.view(1, 1, 3),
            'minus_g': -GRAVITY_M.view(1, 1, 3),
        }
        for mode, gravity in gravity_modes.items():
            out[f'acc_control_{mode}_vs_aM_m_s2'] = per_sensor_l2(a_cp[sl] + gravity - target)
            out[f'acc_3pt_{mode}_vs_aM_m_s2'] = per_sensor_l2(acceleration_fd(p, 'three_point')[sl] + gravity - target)
            out[f'acc_5pt_{mode}_vs_aM_m_s2'] = per_sensor_l2(a_ref[sl] + gravity - target)
        out['aM_norm_m_s2'] = per_sensor_l2(target)
    return out


def main():
    parser = argparse.ArgumentParser(description='Compare AMASS exact IMU-proxy derivatives from cubic controls vs finite differences.')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--max-records', type=int, default=20)
    parser.add_argument('--trim', type=int, default=4)
    parser.add_argument('--view-filter', default='')
    args = parser.parse_args()

    records, manifest = load_records(args.cache)
    if args.view_filter:
        records = [r for r in records if args.view_filter in str(r.get('name', ''))]
    records = records[:args.max_records] if args.max_records else records
    body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=IMU_VERTICES, device=DEVICE)
    spline = UniformCubicBSpline(DT)
    totals = {}
    rows = []
    for record in records:
        values = audit_record(record, body_model, spline, args.trim)
        row = {'name': record.get('name'), 'num_frames': int(record['pose_gt'].shape[0])}
        for key, value in values.items():
            totals.setdefault(key, []).append(value)
            row[key] = stats([value])
        rows.append(row)

    result = {
        'status': 'ok',
        'cache': str(args.cache),
        'cache_manifest': manifest,
        'num_records': len(rows),
        'trim': args.trim,
        'reference': {
            'position': 'AMASS GT pose/tran -> SMPL FK at IMU vertices',
            'velocity_truth_proxy': '5-point central derivative of exact FK IMU-vertex trajectory',
            'acceleration_truth_proxy': '5-point central second derivative of exact FK IMU-vertex trajectory; aM comparison uses + gravity',
            'gravity_added_for_aM': [0.0, -9.8, 0.0],
        },
        'aggregate': {key: stats(value) for key, value in totals.items()},
        'rows': rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'ok', 'output_json': str(args.output_json), 'num_records': len(rows), 'aggregate': result['aggregate']}, indent=2))


if __name__ == '__main__':
    main()
