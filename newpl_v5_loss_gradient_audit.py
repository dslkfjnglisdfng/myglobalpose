import argparse
import json
import random
from pathlib import Path

import torch

from l4_train_diverse_short import DEVICE
from pl_curve import build_pl_curve_model, normalize_gravity, pl_curve_loss
from pl_curve_train import (
    attach_pl_target_controls,
    load_partial_checkpoint,
    load_pl_curve_records,
    make_batch,
)


LOSS_KEYS = (
    'pRB',
    'gR1',
    'baseline_pRB',
    'baseline_gR1',
    'control_point_prior',
    'tail_update_prior',
    'pRB_dot',
    'pRB_ddot',
    'pRB_ddot_smooth',
    'gR1_dot',
    'gR1_ddot',
    'gR_smooth',
    'gt_control_pRB',
    'gt_control_gR1',
)


GROUPS = {
    'q': {
        'pRB': 1.0,
        'gR1': 1.0,
    },
    'control': {
        'gt_control_pRB': 0.3,
        'gt_control_gR1': 0.1,
        'control_point_prior': 0.3,
        'tail_update_prior': 0.005,
    },
    'qdot': {
        'pRB_dot': 0.03,
        'gR1_dot': 0.03,
    },
    'qddot': {
        'pRB_ddot': 0.0003,
        'gR1_ddot': 0.001,
        'pRB_ddot_smooth': 0.000001,
    },
}


def make_model_config(args):
    return {
        'model_variant': 'base',
        'input_size': args.input_size,
        'init_size': args.init_size,
        'hidden_size': args.hidden_size,
        'tail_length': args.tail_length,
        'residual_scale': args.residual_scale,
        'dropout': args.dropout,
    }


def slice_records(records, length, batch_size):
    selected = [record for record in records if record['pl_input'].shape[0] >= length]
    if not selected:
        raise RuntimeError(f'No records have at least {length} frames.')
    selected = selected[:batch_size]
    starts = []
    for idx, record in enumerate(selected):
        max_start = max(0, record['pl_input'].shape[0] - length)
        starts.append((idx * 997) % (max_start + 1) if max_start else 0)
    return make_batch(selected, starts, length)


def build_model(args):
    model = build_pl_curve_model(make_model_config(args)).to(DEVICE)
    checkpoint_info = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        if checkpoint.get('model_type') != 'pl_curve_v1':
            raise RuntimeError(f'Unsupported checkpoint model_type={checkpoint.get("model_type")}')
        checkpoint_info = load_partial_checkpoint(model, checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint_info


def compute_components(model, batch):
    features = batch['pl_input'].to(DEVICE)
    base = normalize_gravity(batch['pl_base'].float()).to(DEVICE)
    target = normalize_gravity(batch['pl_target'].float()).to(DEVICE)
    init_feature = batch.get('pl_init_feature')
    if init_feature is not None:
        init_feature = init_feature.to(DEVICE)
    target_control = batch.get('pl_target_control')
    if target_control is not None:
        target_control = target_control.to(DEVICE)
    output = model.forward_sequence(features, base, init_feature=init_feature)
    _, components = pl_curve_loss(
        output,
        target,
        {key: 0.0 for key in LOSS_KEYS},
        dt=model.dt,
        target_control=target_control,
    )
    return components


def flatten_grads(model):
    flats = []
    nonzero = 0
    total = 0
    for parameter in model.parameters():
        total += 1
        grad = parameter.grad
        if grad is None:
            flats.append(torch.zeros(parameter.numel(), device=DEVICE, dtype=parameter.dtype))
            continue
        grad = grad.detach()
        if grad.abs().sum() > 0:
            nonzero += 1
        flats.append(grad.reshape(-1))
    flat = torch.cat(flats) if flats else torch.zeros(0, device=DEVICE)
    return flat, nonzero, total


def audit_group(args, batch, group_name, group_weights):
    model, checkpoint_info = build_model(args)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed)
    components = compute_components(model, batch)
    loss = None
    weighted_components = {}
    raw_components = {}
    for key, weight in group_weights.items():
        value = components[key]
        raw_components[key] = float(value.detach().cpu())
        weighted = value * float(weight)
        weighted_components[key] = float(weighted.detach().cpu())
        loss = weighted if loss is None else loss + weighted
    if loss is None:
        loss = next(model.parameters()).new_zeros(())
    loss.backward()
    grad, nonzero, total = flatten_grads(model)
    norm = float(grad.norm().detach().cpu())
    return {
        'group': group_name,
        'loss': float(loss.detach().cpu()),
        'raw_components': raw_components,
        'weighted_components': weighted_components,
        'grad_norm': norm,
        'grad_finite': bool(torch.isfinite(grad).all()),
        'params_with_nonzero_grad': nonzero,
        'params_total': total,
        'params_with_nonzero_grad_ratio': float(nonzero / max(1, total)),
        'checkpoint_load': checkpoint_info,
    }, grad.cpu()


def cosine_matrix(grads):
    names = list(grads)
    matrix = []
    for left in names:
        row = {'group': left}
        g_left = grads[left].float()
        left_norm = g_left.norm()
        for right in names:
            g_right = grads[right].float()
            denom = left_norm * g_right.norm()
            if denom <= 0:
                value = None
            else:
                value = float(torch.dot(g_left, g_right) / denom)
            row[right] = value
        matrix.append(row)
    return matrix


def main():
    parser = argparse.ArgumentParser(description='Gradient audit for NewPL v5 loss families.')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--gt-control-cache', default='')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--window', type=int, default=61)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input-size', type=int, default=84)
    parser.add_argument('--init-size', type=int, default=36)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--tail-length', type=int, default=4)
    parser.add_argument('--residual-scale', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.4)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records, manifest = load_pl_curve_records(args.cache)
    control_attach = attach_pl_target_controls(records, args.gt_control_cache)
    batch = slice_records(records, args.window, args.batch_size)
    rows = []
    grads = {}
    for group_name, weights in GROUPS.items():
        row, grad = audit_group(args, batch, group_name, weights)
        rows.append(row)
        grads[group_name] = grad
    result = {
        'status': 'ok',
        'cache': args.cache,
        'gt_control_cache': args.gt_control_cache,
        'checkpoint': args.checkpoint,
        'batch_name': batch['name'],
        'batch_size': args.batch_size,
        'window': args.window,
        'seed': args.seed,
        'manifest_type': manifest.get('type') if manifest else None,
        'control_attach': control_attach,
        'groups': rows,
        'gradient_cosine': cosine_matrix(grads),
        'loss_family_contract': {
            'q': 'decoded PL state pRB[15] + gR1[3]',
            'control': 'direct GT spline-control supervision plus control-buffer priors',
            'qdot': 'first-difference state supervision against decoded spline pldot',
            'qddot': 'second-difference state supervision and pRB plddot smoothness',
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': 'ok',
        'output_json': str(output_path),
        'groups': [row['group'] for row in rows],
    }, indent=2))


if __name__ == '__main__':
    main()
