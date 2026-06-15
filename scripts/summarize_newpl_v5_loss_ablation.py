import argparse
import csv
import json
from pathlib import Path


VARIANTS = (
    'q_only',
    'q_control',
    'q_qdot',
    'q_qddot',
    'q_qdot_qddot',
    'q_control_qdot',
    'q_control_qddot',
    'q_control_qdot_qddot',
)

DATASETS = (
    ('dip', 'dip_test'),
    ('tc', 'tc_test'),
)


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def number(value):
    if value in (None, 'not available'):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def train_summary(root, variant, stage):
    path = root / variant / stage / 'train_result.json'
    data = read_json(path)
    if data is None:
        return {'status': 'missing', 'path': str(path)}
    history = data.get('history') or []
    last = history[-1] if history else {}
    validation = (last.get('validation') or {}).get('loss') or {}
    train_loss = last.get('train_loss') or {}
    return {
        'status': data.get('status'),
        'path': str(path),
        'best_epoch': data.get('best_epoch'),
        'best_loss': data.get('best_loss'),
        'selection_metric': data.get('selection_metric'),
        'num_train_sequences': data.get('num_train_sequences'),
        'num_val_sequences': data.get('num_val_sequences'),
        'weights': data.get('weights'),
        'final_epoch': last.get('epoch'),
        'final_weighted_val_loss': last.get('weighted_val_loss'),
        'final_train_loss': train_loss.get('loss'),
        'final_val_terms': validation,
    }


def find_eval_version(data, variant, checkpoint_name):
    if data is None:
        return None
    wanted = f'{variant}_{checkpoint_name}'
    for row in data.get('pl_output_comparison_table', []):
        if row.get('Version') == wanted:
            return row
    return None


def eval_summary(root, variant, stage, dataset_key, dataset_suffix, checkpoint_name):
    path = root / 'eval' / variant / f'after_{stage}_{dataset_suffix}.json'
    data = read_json(path)
    row = find_eval_version(data, variant, checkpoint_name)
    if row is None:
        return {'status': 'missing', 'path': str(path)}
    version = None
    if data is not None:
        for item in data.get('versions', []):
            if item.get('name') == row.get('Version'):
                version = item
                break
    aggregate = (version or {}).get('aggregate', {})
    return {
        'status': 'ok',
        'path': str(path),
        'dataset': dataset_key,
        'stage': stage,
        'checkpoint': checkpoint_name,
        'version': row.get('Version'),
        'pRB_L1_cm': number(row.get('pRB L1 cm ↓')),
        'pRB_L2_cm': number(row.get('pRB L2 cm ↓')),
        'gR1_angle_deg': number(row.get('gR1 angle deg ↓')),
        'pRB_temporal_velocity_error_cm_per_frame': aggregate.get('pRB_temporal_velocity_error_cm_per_frame'),
        'pRB_smooth_jitter_cm': aggregate.get('pRB_smooth_jitter_cm'),
        'gR1_temporal_angle_velocity_error_deg_per_frame': aggregate.get('gR1_temporal_angle_velocity_error_deg_per_frame'),
        'gR1_smooth_jitter': aggregate.get('gR1_smooth_jitter'),
    }


def gradient_audits(root):
    audit_root = root / 'gradient_audit'
    out = {}
    for path in sorted(audit_root.glob('*/result.json')):
        data = read_json(path)
        if data is None:
            continue
        out[path.parent.name] = {
            'path': str(path),
            'groups': data.get('groups', []),
            'gradient_cosine': data.get('gradient_cosine', []),
            'loss_family_contract': data.get('loss_family_contract', {}),
        }
    return out


def variant_summary(root, variant):
    stages = {
        'amass_pretrain': train_summary(root, variant, 'amass_pretrain'),
        'dip_finetune': train_summary(root, variant, 'dip_finetune'),
    }
    evals = {}
    for dataset_key, suffix in DATASETS:
        evals[f'{dataset_key}_after_amass_best'] = eval_summary(root, variant, 'amass', dataset_key, suffix, 'amass_best')
        evals[f'{dataset_key}_after_amass_last'] = eval_summary(root, variant, 'amass', dataset_key, suffix, 'amass_last')
        evals[f'{dataset_key}_after_dip_best'] = eval_summary(root, variant, 'dip', dataset_key, suffix, 'dip_best')
        evals[f'{dataset_key}_after_dip_last'] = eval_summary(root, variant, 'dip', dataset_key, suffix, 'dip_last')
    return {'train': stages, 'eval': evals}


def write_csv(path, summary):
    fields = [
        'variant',
        'dataset',
        'stage',
        'checkpoint',
        'pRB_L2_cm',
        'gR1_angle_deg',
        'pRB_temporal_velocity_error_cm_per_frame',
        'pRB_smooth_jitter_cm',
        'gR1_temporal_angle_velocity_error_deg_per_frame',
        'gR1_smooth_jitter',
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant, payload in summary['variants'].items():
            for row in payload['eval'].values():
                if row.get('status') != 'ok':
                    continue
                csv_row = {field: row.get(field) for field in fields}
                csv_row['variant'] = variant
                writer.writerow(csv_row)


def best_rows(summary):
    rows = []
    for variant, payload in summary['variants'].items():
        for key, row in payload['eval'].items():
            if row.get('status') == 'ok' and key.endswith('_after_dip_best'):
                rows.append({
                    'variant': variant,
                    'dataset': row['dataset'],
                    'pRB_L2_cm': row['pRB_L2_cm'],
                    'gR1_angle_deg': row['gR1_angle_deg'],
                    'pRB_smooth_jitter_cm': row['pRB_smooth_jitter_cm'],
                    'gR1_smooth_jitter': row['gR1_smooth_jitter'],
                })
    return sorted(rows, key=lambda item: (item['dataset'], item['pRB_L2_cm'] or 1e9, item['gR1_angle_deg'] or 1e9))


def main():
    parser = argparse.ArgumentParser(description='Summarize NewPL v5 loss-family ablation.')
    parser.add_argument('--root', type=Path, default=Path('data/experiments/newpl_v5_loss_family_ablation_20260611'))
    parser.add_argument('--output-json', type=Path, default=None)
    parser.add_argument('--output-csv', type=Path, default=None)
    args = parser.parse_args()

    output_json = args.output_json or args.root / 'summary.json'
    output_csv = args.output_csv or args.root / 'summary_eval_rows.csv'
    summary = {
        'status': 'ok',
        'root': str(args.root),
        'question': 'Effect of q/control/qdot/qddot loss families on NewPL v5 training and generalization.',
        'contract': {
            'q': 'NewPL v5 decoded PL state pRB[15] + gR1[3], not full-body RBDL q75.',
            'protocol': 'AMASS pretrain -> DIP-IMU train fine-tune -> DIP test and TotalCapture official-input test.',
            'selection_metric': 'pl_physical for every ablation variant.',
        },
        'variants': {variant: variant_summary(args.root, variant) for variant in VARIANTS},
        'gradient_audits': gradient_audits(args.root),
    }
    summary['best_dip_finetune_rows'] = best_rows(summary)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + '\n')
    write_csv(output_csv, summary)
    print(json.dumps({
        'status': 'ok',
        'output_json': str(output_json),
        'output_csv': str(output_csv),
        'variants': len(summary['variants']),
        'best_rows': len(summary['best_dip_finetune_rows']),
    }, indent=2))


if __name__ == '__main__':
    main()
