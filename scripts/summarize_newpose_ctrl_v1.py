import argparse
import json
from pathlib import Path


FULL_EVALS = {
    'DIP-IMU test': {
        'official_gpnet': 'dip_official_gpnet.json',
        'newpl_v5_amass_official_ik2': 'dip_newpl_v5_amass_official_ik2.json',
        'newpl_v5_dip_official_ik2': 'dip_newpl_v5_dip_official_ik2.json',
        'newpose_ctrl_v1_stage_a_best': 'dip_stage_a_best.json',
        'newpose_ctrl_v1_stage_b_best': 'dip_stage_b_best.json',
        'newpose_ctrl_v1_stage_b_last': 'dip_stage_b_last.json',
    },
    'TotalCapture test': {
        'official_gpnet': 'tc_official_gpnet.json',
        'newpl_v5_amass_official_ik2': 'tc_newpl_v5_amass_official_ik2.json',
        'newpl_v5_dip_official_ik2': 'tc_newpl_v5_dip_official_ik2.json',
        'newpose_ctrl_v1_stage_a_best': 'tc_stage_a_best.json',
        'newpose_ctrl_v1_stage_b_best': 'tc_stage_b_best.json',
        'newpose_ctrl_v1_stage_b_last': 'tc_stage_b_last.json',
    },
}

MODULE_EVALS = {
    'DIP-IMU test': {
        'official_gpnet': ('eval_module', 'dip_official_gpnet_module.json'),
        'newpl_v5_amass_official_ik2': ('eval_module', 'dip_newpl_v5_amass_official_ik2_module.json'),
        'newpl_v5_dip_official_ik2': ('eval_module', 'dip_newpl_v5_dip_official_ik2_module.json'),
        'newpose_ctrl_v1_stage_a_best': ('eval_module', 'dip_stage_a_best_module.json'),
        'newpose_ctrl_v1_stage_b_best': ('eval_module', 'dip_stage_b_best_module.json'),
        'newpose_ctrl_v1_stage_b_last': ('eval_module', 'dip_stage_b_last_module.json'),
    },
    'TotalCapture test': {
        'official_gpnet': ('eval_module', 'tc_official_gpnet_module.json'),
        'newpl_v5_amass_official_ik2': ('eval_module', 'tc_newpl_v5_amass_official_ik2_module.json'),
        'newpl_v5_dip_official_ik2': ('eval_module', 'tc_newpl_v5_dip_official_ik2_module.json'),
        'newpose_ctrl_v1_stage_a_best': ('eval_module', 'tc_stage_a_best_module.json'),
        'newpose_ctrl_v1_stage_b_best': ('eval_module', 'tc_stage_b_best_module.json'),
        'newpose_ctrl_v1_stage_b_last': ('eval_module', 'tc_stage_b_last_module.json'),
    },
}

METRIC_COLUMNS = [
    'L SIP Err (deg)',
    'L Angle Err (deg)',
    'L Joint Err (cm)',
    'L Mesh Err (cm)',
    'G SIP Err (deg)',
    'G Angle Err (deg)',
    'G Joint Err (cm)',
    'G Mesh Err (cm)',
    'Root Jitter (km/s^3)',
    'Joint Jitter (km/s^3)',
]


def load_json(path):
    if not path.exists():
        return None, 'missing'
    try:
        return json.loads(path.read_text()), None
    except Exception as exc:
        return None, f'failed_to_read: {exc}'


def mean_metric(stats, key):
    try:
        return stats[key]['mean']
    except Exception:
        return None


def module_mean(module_aggregate, key):
    item = (module_aggregate or {}).get(key)
    if isinstance(item, dict):
        return item.get('mean')
    if isinstance(item, (int, float)):
        return float(item)
    return None


def fmt(value):
    if value is None:
        return 'not available'
    if isinstance(value, str):
        return value
    return f'{float(value):.6f}'


def markdown_table(headers, rows):
    out = []
    out.append('| ' + ' | '.join(headers) + ' |')
    out.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        out.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
    return '\n'.join(out)


def train_summary(root):
    out = {}
    for stage, rel in (
        ('stage_a_amass_pretrain', 'stage_a_amass_pretrain/train_result.json'),
        ('stage_b_dip_finetune', 'stage_b_dip_finetune/train_result.json'),
    ):
        data, err = load_json(root / rel)
        if data is None:
            out[stage] = {'status': err}
        else:
            out[stage] = {
                'status': data.get('status'),
                'best_epoch': data.get('best_epoch'),
                'best_loss': data.get('best_loss'),
                'selection_metric': data.get('selection_metric'),
                'stopped_early': data.get('stopped_early'),
                'num_train_sequences': data.get('num_train_sequences'),
                'num_val_sequences': data.get('num_val_sequences'),
                'history_len': len(data.get('history', [])),
            }
    return out


def full_tables(root):
    rows = []
    details = {}
    missing = []
    failed = []
    for dataset, versions in FULL_EVALS.items():
        for version, filename in versions.items():
            path = root / 'eval' / filename
            data, err = load_json(path)
            if data is None:
                missing.append(str(path))
                rows.append([dataset, version, 'missing', 'not available', 'not available', 'not available', 'not available'])
                continue
            status = data.get('status')
            if status != 'ok':
                failed.append(str(path))
            agg = data.get('aggregate') or {}
            model = agg.get('model_metrics') or {}
            row = [
                dataset,
                version,
                status,
                fmt(data.get('score')),
                fmt(mean_metric(model, 'L Angle Err (deg)')),
                fmt(mean_metric(model, 'G Angle Err (deg)')),
                fmt(data.get('all_finite')),
            ]
            rows.append(row)
            details[f'{dataset}/{version}'] = {
                'path': str(path),
                'status': status,
                'score': data.get('score'),
                'all_finite': data.get('all_finite'),
                'model_metrics': {key: mean_metric(model, key) for key in METRIC_COLUMNS},
            }
    headers = ['Dataset', 'Version', 'Status', 'S4 score ↓', 'Local angle ↓', 'Global angle ↓', 'all_finite']
    return rows, details, missing, failed, markdown_table(headers, rows)


def module_tables(root):
    rows = []
    details = {}
    missing = []
    failed = []
    for dataset, versions in MODULE_EVALS.items():
        for version, (subdir, filename) in versions.items():
            path = root / subdir / filename
            data, err = load_json(path)
            if data is None:
                missing.append(str(path))
                rows.append([dataset, version, 'missing', 'not available', 'not available', 'not available', 'not available', 'not available'])
                continue
            module = data.get('module_aggregate') or {}
            status = data.get('status')
            if status != 'ok':
                failed.append(str(path))
            row = [
                dataset,
                version,
                status,
                fmt(module_mean(module, 'control_RRJ_geodesic_deg')),
                fmt(module_mean(module, 'state_RRJ_geodesic_deg')),
                fmt(module_mean(module, 'FK_joint_L2_cm')),
                fmt(module_mean(module, 'state_gR_pose_loss')),
                fmt(data.get('all_finite')),
            ]
            rows.append(row)
            details[f'{dataset}/{version}'] = {
                'path': str(path),
                'status': data.get('status'),
                'all_finite': data.get('all_finite'),
                'module_aggregate': module,
            }
    headers = ['Dataset', 'Version', 'Status', 'Control RRJ deg ↓', 'State RRJ deg ↓', 'FK joint L2 cm ↓', 'gR loss ↓', 'all_finite']
    return rows, details, missing, failed, markdown_table(headers, rows)


def main():
    parser = argparse.ArgumentParser(description='Summarize newpose_ctrl_v1 official-protocol experiment artifacts.')
    parser.add_argument('--root', type=Path, default=Path('data/experiments/newpose_ctrl_v1_20260608'))
    args = parser.parse_args()
    root = args.root
    full_rows, full_details, full_missing, full_failed, full_md = full_tables(root)
    module_rows, module_details, module_missing, module_failed, module_md = module_tables(root)
    summary = {
        'status': 'complete' if not full_missing and not module_missing and not full_failed and not module_failed else 'incomplete',
        'root': str(root),
        'train': train_summary(root),
        'full_pipeline': full_details,
        'module': module_details,
        'missing': {
            'full_pipeline': full_missing,
            'module': module_missing,
        },
        'failed': {
            'full_pipeline': full_failed,
            'module': module_failed,
        },
        'tables_markdown': {
            'full_pipeline': full_md,
            'module': module_md,
        },
    }
    out_json = root / 'summary.json'
    out_md = root / 'summary_tables.md'
    out_json.write_text(json.dumps(summary, indent=2) + '\n')
    out_md.write_text(
        '# newpose_ctrl_v1 summary tables\n\n'
        '## Full Pipeline\n\n'
        + full_md
        + '\n\n## Module IK2-slot / Pose-control\n\n'
        + module_md
        + '\n'
    )
    print(json.dumps({
        'status': summary['status'],
        'summary': str(out_json),
        'tables': str(out_md),
        'missing': summary['missing'],
        'failed': summary['failed'],
    }, indent=2))


if __name__ == '__main__':
    main()
