import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def nested_get(obj, path, default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(value):
    if value is None:
        return 'not measured'
    if isinstance(value, str):
        return value
    return f'{float(value):.6g}'


def fmt_md(value):
    return f'`{fmt(value)}`' if value is not None and not isinstance(value, str) else fmt(value)


def load_offset_norm(path):
    if not path:
        return None
    data = read_json(path)
    return nested_get(data, ['summary', 'offset_norm_m', 'median'])


def read_json_optional(path):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return read_json(path)


def method_row(method, offset_json, pl_eval_json, sensitivity_json=None):
    pl_eval = read_json(pl_eval_json)
    agg = pl_eval.get('aggregate', {})
    sensitivity = read_json(sensitivity_json) if sensitivity_json and Path(sensitivity_json).exists() else {}
    pair_key = f'{method}_minus_zero'
    pair = sensitivity.get('pairwise_vs_reference', {}).get(pair_key, {}) if method != 'zero' else {}
    return {
        'method': method,
        'offset_median_norm_m': load_offset_norm(offset_json),
        'pl_pRB_original_cm': nested_get(agg, ['leaf_position_error_cm', 'original', 'mean']),
        'pl_pRB_new_cm': nested_get(agg, ['leaf_position_error_cm', 'new', 'mean']),
        'pl_pRB_delta_cm': nested_get(agg, ['leaf_position_error_cm', 'delta_new_minus_original', 'mean']),
        'pl_gR1_original_deg': nested_get(agg, ['gravity_angle_deg', 'original', 'mean']),
        'pl_gR1_new_deg': nested_get(agg, ['gravity_angle_deg', 'new', 'mean']),
        'pl_gR1_delta_deg': nested_get(agg, ['gravity_angle_deg', 'delta_new_minus_original', 'mean']),
        'output_diff_vs_zero_cm': nested_get(pair, ['pl_output_abs_diff_cm', 'mean'], 0.0 if method == 'zero' else None),
        'gravity_diff_vs_zero_deg': nested_get(pair, ['pl_gravity_angle_diff_deg', 'mean'], 0.0 if method == 'zero' else None),
        'ik1_metrics': 'not measured',
        'full_pipeline_11_metrics': 'not measured',
    }


def markdown_table(rows):
    headers = [
        'Method', 'Offset median m', 'PL pRB orig cm', 'PL pRB NewPL cm',
        'Delta cm', 'gR1 orig deg', 'gR1 NewPL deg', 'Delta deg',
        'Output diff vs zero cm', 'IK1', 'Full 11 metrics'
    ]
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] + ['---:'] * 8 + ['---', '---']) + ' |',
    ]
    for row in rows:
        lines.append(
            '| {method} | {offset_median_norm_m} | {pl_pRB_original_cm} | {pl_pRB_new_cm} | '
            '{pl_pRB_delta_cm} | {pl_gR1_original_deg} | {pl_gR1_new_deg} | {pl_gR1_delta_deg} | '
            '{output_diff_vs_zero_cm} | {ik1_metrics} | {full_pipeline_11_metrics} |'.format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    return '\n'.join(lines)


def load_pl_table(path):
    data = read_json_optional(path)
    if data is None:
        return {}
    if isinstance(data, list):
        return {row.get('method'): row for row in data}
    if isinstance(data, dict) and 'rows' in data:
        return {row.get('method'): row for row in data['rows']}
    return {}


def decision_for_row(row):
    if row['method'] == 'random':
        return 'negative_control'
    if row.get('pl_pRB_delta_vs_zero_cm') is not None and row['pl_pRB_delta_vs_zero_cm'] < -0.01:
        if row.get('forward_improvement_mps2') is not None and row['forward_improvement_mps2'] > 0:
            return 'physical_signal_but_newpl_gain_too_small'
    if row.get('forward_improvement_mps2') is not None and row['forward_improvement_mps2'] > 0:
        return 'physical_signal_downstream_not_selected'
    return 'not_selected'


def build_decision_matrix(args):
    methods = ['zero', 'random', 'solver_v1', 'net_v2', 'hybrid_v3']
    conditioned = load_pl_table(args.conditioned_pl_table)
    consistency = read_json_optional(args.consistency_json) or {}
    swap = read_json_optional(args.swap_json) or {}
    dip = read_json_optional(args.dip_stageb_json) or {}
    stageb_v4_compare = read_json_optional(args.stageb_v4_consistency_json) or {}
    rows = []
    zero_prb = conditioned.get('zero', {}).get('pRB_new')
    zero_gr = conditioned.get('zero', {}).get('gR_new')
    consistency_agg = consistency.get('aggregate', {})
    for method in methods:
        pl = conditioned.get(method, {})
        cons = consistency_agg.get(method, {})
        row = {
            'method': method,
            'algorithm': {
                'zero': 'zero offset baseline',
                'random': 'random offset negative control',
                'solver_v1': 'lever-arm kinematic optimization',
                'net_v2': 'OffsetNet synthetic + DIP self-supervised artifact',
                'hybrid_v3': 'solver init plus OffsetNet residual/blend',
            }[method],
            'offset_coord_frame': 'r_JS joint-local',
            'offset_gt_real_data': 'not available',
            'tc_forward_residual_mps2': nested_get(cons, ['sequence_residual_mps2', 'mean']),
            'forward_improvement_mps2': nested_get(cons, ['sequence_improvement_mps2', 'mean']),
            'offset_median_norm_m': nested_get(cons, ['offset_norm_m', 'median']),
            'conditioned_pl_pRB_cm': pl.get('pRB_new'),
            'conditioned_pl_pRB_delta_vs_zero_cm': (pl.get('pRB_new') - zero_prb) if pl.get('pRB_new') is not None and zero_prb is not None else None,
            'conditioned_pl_gR1_deg': pl.get('gR_new'),
            'conditioned_pl_gR1_delta_vs_zero_deg': (pl.get('gR_new') - zero_gr) if pl.get('gR_new') is not None and zero_gr is not None else None,
            'ik1_metrics': 'not measured',
            'full_pipeline_11_metrics': 'not measured',
        }
        row['decision'] = decision_for_row(row)
        rows.append(row)

    swap_delta = nested_get(swap, ['aggregate', 'delta_vs_good'], {})
    dip_initial = nested_get(dip, ['initial_val', 'pose_acc_proxy'])
    dip_last = nested_get(dip, ['last_val', 'pose_acc_proxy'])
    net_v2_improvement = nested_get(stageb_v4_compare, ['aggregate', 'net_v2', 'sequence_improvement_mps2', 'mean'])
    stageb_v4_improvement = nested_get(stageb_v4_compare, ['aggregate', 'net_v2_stageB_v4', 'sequence_improvement_mps2', 'mean'])
    if isinstance(net_v2_improvement, (int, float)) and isinstance(stageb_v4_improvement, (int, float)):
        stageb_v4_delta = stageb_v4_improvement - net_v2_improvement
    else:
        stageb_v4_delta = 'not available'
    payload = {
        'status': 'ok',
        'root': str(args.root),
        'coordinate_contract': 'r_JS is the IMU origin position relative to mapped joint J, expressed in joint-local coordinates.',
        'real_offset_gt': 'not available for DIP/TotalCapture',
        'dip_trans_usage': 'not used',
        'totalcapture_usage': 'diagnostic/adaptation only; not official protocol',
        'rows': rows,
        'supporting_diagnostics': {
            'conditioned_pl_table': str(args.conditioned_pl_table),
            'consistency_json': str(args.consistency_json),
            'swap_json': str(args.swap_json),
            'dip_stageb_json': str(args.dip_stageb_json),
            'stageb_v4_consistency_json': str(args.stageb_v4_consistency_json) if args.stageb_v4_consistency_json else None,
            'swap_delta_vs_good': swap_delta,
            'dip_stageb_pose_acc_proxy_initial': dip_initial,
            'dip_stageb_pose_acc_proxy_last': dip_last,
            'dip_stageb_pose_acc_proxy_delta': (dip_last - dip_initial) if isinstance(dip_initial, (int, float)) and isinstance(dip_last, (int, float)) else 'not available',
            'net_v2_stageB_v4_tc_forward_improvement_delta_mps2': stageb_v4_delta,
        },
        'selection': {
            'best_offset_method_by_forward_consistency': 'net_v2',
            'best_offset_method_for_newpl': 'not selected',
            'run_ik1_or_full_pipeline': False,
            'reason': 'Forward consistency improves for solver/net/hybrid, but NewPL pRB/gR1 gains are tiny or conflicting and full PL loss can prefer bad offsets.',
        },
    }
    return payload


def decision_markdown(payload):
    headers = [
        'Method', 'Algorithm', 'Forward residual m/s^2', 'Forward improvement',
        'Offset median m', 'Cond. PL pRB cm', 'pRB vs zero cm',
        'Cond. PL gR1 deg', 'gR1 vs zero deg', 'IK1', 'Full 11', 'Decision'
    ]
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---', '---'] + ['---:'] * 7 + ['---', '---', '---']) + ' |',
    ]
    for row in payload['rows']:
        lines.append(
            '| {method} | {algorithm} | {tc_forward_residual_mps2} | {forward_improvement_mps2} | '
            '{offset_median_norm_m} | {conditioned_pl_pRB_cm} | {conditioned_pl_pRB_delta_vs_zero_cm} | '
            '{conditioned_pl_gR1_deg} | {conditioned_pl_gR1_delta_vs_zero_deg} | {ik1_metrics} | '
            '{full_pipeline_11_metrics} | {decision} |'.format(
                **{key: fmt_md(value) for key, value in row.items()}
            )
        )
    lines.extend([
        '',
        'Notes:',
        f"- Coordinate contract: `{payload['coordinate_contract']}`",
        f"- Real offset GT: `{payload['real_offset_gt']}`",
        f"- DIP trans usage: `{payload['dip_trans_usage']}`",
        f"- TotalCapture usage: `{payload['totalcapture_usage']}`",
        f"- Selection: `{payload['selection']['best_offset_method_for_newpl']}`",
        f"- Downstream IK1/full pipeline: `{payload['selection']['run_ik1_or_full_pipeline']}` because {payload['selection']['reason']}",
    ])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Summarize IMU offset -> NewPL diagnostic JSONs.')
    parser.add_argument('--root', type=Path, default=Path('data/experiments/imu_position_offset_newpl/tc_val_2seq'))
    parser.add_argument('--sensitivity-json', type=Path, default=None)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    parser.add_argument('--decision-json', type=Path, default=None)
    parser.add_argument('--decision-md', type=Path, default=None)
    parser.add_argument('--conditioned-pl-table', type=Path, default=Path('data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_conditioned_pl_eval_table.json'))
    parser.add_argument('--consistency-json', type=Path, default=Path('data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_consistency_eval_v1.json'))
    parser.add_argument('--swap-json', type=Path, default=Path('data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_swap_eval_prb_contrast_v1_hybrid_cache.json'))
    parser.add_argument('--dip-stageb-json', type=Path, default=Path('data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/train_result.json'))
    parser.add_argument('--stageb-v4-consistency-json', type=Path, default=None)
    args = parser.parse_args()

    methods = ['zero', 'random', 'solver_v1', 'net_v2', 'hybrid_v3']
    rows = []
    for method in methods:
        rows.append(method_row(
            method,
            args.root / f'{method}_offsets.json',
            args.root / f'pl_eval_{method}.json',
            args.sensitivity_json,
        ))
    payload = {
        'status': 'ok',
        'root': str(args.root),
        'sensitivity_json': str(args.sensitivity_json) if args.sensitivity_json else None,
        'rows': rows,
        'notes': {
            'coordinate_contract': 'r_JS is the IMU origin position relative to mapped joint J, expressed in joint-local coordinates.',
            'real_offset_gt': 'not available',
            'dip_trans_usage': 'not used',
            'ik1_and_full_metrics': 'not measured in this smoke summary',
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + '\n')
    args.output_md.write_text(markdown_table(rows) + '\n')
    result = {'status': 'ok', 'output_json': str(args.output_json), 'output_md': str(args.output_md)}
    if args.decision_json or args.decision_md:
        decision = build_decision_matrix(args)
        if args.decision_json:
            args.decision_json.parent.mkdir(parents=True, exist_ok=True)
            args.decision_json.write_text(json.dumps(decision, indent=2) + '\n')
            result['decision_json'] = str(args.decision_json)
        if args.decision_md:
            args.decision_md.parent.mkdir(parents=True, exist_ok=True)
            args.decision_md.write_text(decision_markdown(decision) + '\n')
            result['decision_md'] = str(args.decision_md)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
