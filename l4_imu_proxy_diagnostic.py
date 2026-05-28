import argparse
import json
from pathlib import Path

import torch

from l4_train_diverse_short import load_records
from l4_train_loss_ablation import (
    DEVICE,
    default_weights,
    imu_proxy_losses,
    pose_velocity_loss,
    run_cached_sequence,
)
from l4_tail_update_qstate import StreamingTailUpdateQState
from l4_q75_utils import q75_to_pose_tran


def average(items):
    return sum(items) / max(1, len(items))


def main():
    parser = argparse.ArgumentParser(description='Diagnostic-only IMU proxy audit for L4 checkpoints.')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--velocity-residual-scale', type=float, default=0.005)
    parser.add_argument('--max-sequences', type=int, default=0)
    args = parser.parse_args()

    records, manifest = load_records(args.cache, max_sequences=args.max_sequences)
    model = StreamingTailUpdateQState(
        hidden_size=args.hidden_size,
        residual_scale=args.residual_scale,
        velocity_residual_scale=args.velocity_residual_scale,
    ).to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    weights = default_weights()
    rows = []
    with torch.no_grad():
        for record in records:
            output = run_cached_sequence(model, record)
            total, losses = pose_velocity_loss(output, record, weights, compute_imu_proxy=True)
            q_pred = output['q_pred']
            pose_pred, _ = q75_to_pose_tran(q_pred)
            proxy = imu_proxy_losses(pose_pred.to(DEVICE), record['tran_gt'].to(DEVICE), record)
            q_norm = output['q_residual'].norm(dim=-1)
            dv_norm = output['delta_v'].norm(dim=-1)
            rows.append({
                'name': record['name'],
                'total_loss_with_zero_imu_weights': float(total.detach()),
                'imu_orientation_proxy': float(proxy['imu_orientation_proxy'].detach()),
                'imu_acc_proxy': float(proxy['imu_acc_proxy'].detach()),
                'imu_gyro_proxy': float(proxy['imu_gyro_proxy'].detach()),
                'q_residual_norm_mean': float(q_norm.mean().detach()),
                'q_residual_norm_max': float(q_norm.max().detach()),
                'delta_v_root_norm_mean': float(dv_norm.mean().detach()),
                'delta_v_root_norm_max': float(dv_norm.max().detach()),
                'tail_update_norm_mean': float(output['tail_delta_norm'].detach()),
                'loss_components': {key: float(value.detach()) for key, value in losses.items()},
            })

    aggregate = {}
    for key in (
        'total_loss_with_zero_imu_weights',
        'imu_orientation_proxy',
        'imu_acc_proxy',
        'imu_gyro_proxy',
        'q_residual_norm_mean',
        'q_residual_norm_max',
        'delta_v_root_norm_mean',
        'delta_v_root_norm_max',
        'tail_update_norm_mean',
    ):
        values = [row[key] for row in rows]
        aggregate[key] = max(values) if key.endswith('_max') else average(values)

    result = {
        'experiment_name': args.experiment_name,
        'mode': 'imu_proxy_diagnostic_only',
        'checkpoint': args.checkpoint,
        'checkpoint_info': {
            'epoch': checkpoint.get('epoch'),
            'step': checkpoint.get('step'),
            'train_loss': checkpoint.get('train_loss'),
            'validation_score': checkpoint.get('validation_score'),
            'selection': checkpoint.get('selection'),
        },
        'cache': args.cache,
        'cache_manifest': manifest,
        'num_sequences': len(records),
        'test_set_used': False,
        'coordinate_audit': {
            'dt': 1.0 / 60.0,
            'gravity_model_frame': [0.0, -9.8, 0.0],
            'aM_meaning_from_process_amass_globalpose': 'model-frame acceleration generated as R_sim @ aS + gravity',
            'wM_meaning_from_process_amass_globalpose': 'model-frame angular velocity generated as R_sim @ wS',
            'RMB_meaning_from_process_amass_globalpose': 'model/body orientation after synthetic sensor calibration perturbation',
            'proxy_limitation': 'Diagnostic predicted IMU signals use SMPL IMU vertices and IMU-joint global rotations, but do not model synthetic random sensor-to-body RBS, dynamic dR, ESKF drift, or sensor sliding offsets.',
            'training_loss_approval': 'not approved; leave IMU proxy weights at zero unless a forward IMU model including attachment/noise frames is implemented and validated',
        },
        'weights': {
            'imu_orientation_proxy': 0.0,
            'imu_acc_proxy': 0.0,
            'imu_gyro_proxy': 0.0,
        },
        'aggregate': aggregate,
        'rows': rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / 'imu_proxy_diagnostic_result.json'
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'result_path': str(result_path),
        'num_sequences': len(records),
        'aggregate': aggregate,
        'training_loss_approval': result['coordinate_audit']['training_loss_approval'],
    }, indent=2))


if __name__ == '__main__':
    main()
