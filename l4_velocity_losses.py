import torch


VELOCITY_LOSS_WEIGHTS = {
    'root_velocity': 1.0,
    'baseline_velocity': 2.0,
    'velocity_smooth': 0.05,
}


def finite_difference_translation_velocity(tran, dt=1.0 / 60.0):
    velocity = torch.zeros_like(tran)
    if tran.shape[-2] > 1:
        velocity[..., 1:, :] = (tran[..., 1:, :] - tran[..., :-1, :]) / dt
        velocity[..., 0, :] = velocity[..., 1, :]
    return velocity


def velocity_residual_losses(v_refined, v_gt, v_baseline, delta_v, weights=None):
    weights = VELOCITY_LOSS_WEIGHTS if weights is None else weights
    losses = {
        'root_velocity': torch.nn.functional.smooth_l1_loss(v_refined, v_gt),
        'baseline_velocity': torch.nn.functional.smooth_l1_loss(v_refined, v_baseline),
    }
    if delta_v.shape[-2] > 1:
        losses['velocity_smooth'] = (delta_v[..., 1:, :] - delta_v[..., :-1, :]).square().mean()
    else:
        losses['velocity_smooth'] = delta_v.new_zeros(())

    total = v_refined.new_zeros(())
    weighted = {}
    for name, weight in weights.items():
        weighted[name] = losses[name] * weight
        total = total + weighted[name]
    return total, losses, weighted
