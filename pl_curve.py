import torch

import articulate as art
from l4_tail_update_qstate import UniformCubicBSpline


def pl_input_feature(a, w, R):
    aRB = a.mm(R[5])
    wRB = w.mm(R[5])
    RRB = R[5].t().matmul(R[:5])
    gR0 = -R[5, 1]
    return torch.cat((aRB.ravel(), wRB.ravel(), RRB.ravel(), gR0))


def split_pl_feature(feature):
    RRB = feature[..., 36:81].reshape(feature.shape[:-1] + (5, 3, 3))
    gR0 = feature[..., 81:84]
    return RRB, gR0


def pl_init_feature(offset_r, pRL, gR0):
    offset = offset_r.reshape(-1)
    if offset.shape[-1] != 18:
        raise ValueError(f'Expected offset_r flatten dim 18, got {offset.shape[-1]}.')
    return torch.cat((offset, pRL.reshape(-1), gR0.reshape(-1))).float()


def pl_init_feature_from_pose(offset_r, pose, body_model):
    ref = body_model._J
    pose = pose.to(device=ref.device, dtype=ref.dtype).view(1, 24, 3, 3)
    _, _, verts = body_model.forward_kinematics(pose, calc_mesh=True)
    pRL = (verts[0, :5] - verts[0, 5:]).mm(pose[0, 0]).ravel()
    gR0 = -pose[0, 0, 1]
    return pl_init_feature(offset_r.to(device=ref.device, dtype=ref.dtype), pRL, gR0).cpu()


def normalize_gravity(pl_output):
    return torch.cat((
        pl_output[..., :15],
        art.math.normalize_tensor(pl_output[..., 15:], avoid_nan=True),
    ), dim=-1)


def fit_uniform_cubic_spline_controls(samples):
    """Fit control points C so UniformCubicBSpline(C) reconstructs samples.

    The spline decoder used in PLCurve evaluates
      q[i] = (C[i-1] + 4*C[i] + C[i+1]) / 6
    with repeated boundary controls at the ends. This solves the resulting
    tridiagonal linear system over the time dimension.
    """
    if samples.shape[0] <= 1:
        return samples.clone()
    t = samples.shape[0]
    flat = samples.reshape(t, -1)
    mat = samples.new_zeros((t, t))
    idx = torch.arange(t, device=samples.device)
    mat[idx, idx] = 4.0
    mat[0, 0] = 5.0
    mat[-1, -1] = 5.0
    mat[idx[1:], idx[:-1]] = 1.0
    mat[idx[:-1], idx[1:]] = 1.0
    fitted = torch.linalg.solve(mat, 6.0 * flat)
    return fitted.reshape_as(samples)


def pl_target_from_pose(pose, body_model):
    if pose.dim() == 3:
        pose = pose.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    _, _, verts = body_model.forward_kinematics(pose, calc_mesh=True)
    pRB = (verts[:, :5] - verts[:, 5:]).bmm(pose[:, 0]).reshape(pose.shape[0], 15)
    gR = -pose[:, 0, :, 1]
    target = torch.cat((pRB, gR), dim=-1)
    return target[0] if squeeze else target


def pl_curve_loss(output, target, weights, dt=1.0 / 60.0):
    pred = output['pl']
    base = output['base']
    target = target.to(pred.device, pred.dtype)
    pred_gR = art.math.normalize_tensor(pred[..., 15:], avoid_nan=True)
    target_gR = art.math.normalize_tensor(target[..., 15:], avoid_nan=True)
    target_for_controls = torch.cat((target[..., :15], target_gR), dim=-1)
    target_control = fit_uniform_cubic_spline_controls(target_for_controls)
    pred_control = output.get('new_control')
    if pred_control is None:
        pred_control = pred
    pred_control_gR = art.math.normalize_tensor(pred_control[..., 15:], avoid_nan=True)
    target_control_gR = art.math.normalize_tensor(target_control[..., 15:], avoid_nan=True)
    losses = {
        'pRB': torch.nn.functional.smooth_l1_loss(pred[..., :15], target[..., :15]),
        'gR1': (1.0 - (pred_gR * target_gR).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
        'baseline_pRB': torch.nn.functional.smooth_l1_loss(pred[..., :15], base[..., :15].detach()),
        'baseline_gR1': (1.0 - (pred[..., 15:] * base[..., 15:].detach()).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
        'gt_control_pRB': torch.nn.functional.smooth_l1_loss(pred_control[..., :15], target_control[..., :15]),
        'gt_control_gR1': torch.nn.functional.smooth_l1_loss(pred_control_gR, target_control_gR),
        'control_point_prior': output['control_point_prior'],
        'tail_update_prior': output['tail_delta_norm'],
    }
    if pred.shape[0] >= 2:
        target_step = target[1:, ..., :15] - target[:-1, ..., :15]
        losses['pRB_dot'] = torch.nn.functional.smooth_l1_loss(dt * output['pldot'][1:, ..., :15], target_step)
        pred_gR_dot = pred_gR[1:] - pred_gR[:-1]
        target_gR_dot = target_gR[1:] - target_gR[:-1]
        losses['gR1_dot'] = torch.nn.functional.smooth_l1_loss(pred_gR_dot, target_gR_dot)
        losses['gR_smooth'] = pred_gR_dot.square().mean()
    else:
        losses['pRB_dot'] = pred.new_zeros(())
        losses['gR1_dot'] = pred.new_zeros(())
        losses['gR_smooth'] = pred.new_zeros(())
    if pred.shape[0] >= 3:
        pred_gR_ddot = pred_gR[2:] - 2.0 * pred_gR[1:-1] + pred_gR[:-2]
        target_gR_ddot = target_gR[2:] - 2.0 * target_gR[1:-1] + target_gR[:-2]
        losses['gR1_ddot'] = torch.nn.functional.smooth_l1_loss(pred_gR_ddot, target_gR_ddot)
    else:
        losses['gR1_ddot'] = pred.new_zeros(())
    losses['pRB_ddot_smooth'] = output['plddot'][..., :15].square().mean()
    total = pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * weight
    return total, losses


class PLCurveModule(torch.nn.Module):
    def __init__(
        self,
        input_size=84,
        state_dim=18,
        init_size=18,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.005,
        dt=1.0 / 60.0,
        dropout=0.4,
    ):
        super().__init__()
        if state_dim != 18:
            raise ValueError('PLCurveModule v1 uses the official PL 18D state.')
        if tail_update != 4:
            raise ValueError('PLCurveModule v1 keeps the K2 L=4 tail-update contract.')
        self.input_size = int(input_size)
        self.state_dim = int(state_dim)
        self.init_size = int(init_size)
        self.hidden_size = int(hidden_size)
        self.tail_update = int(tail_update)
        self.residual_scale = float(residual_scale)
        self.dt = float(dt)
        self.input = torch.nn.Linear(input_size + state_dim, hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.cell = torch.nn.GRUCell(hidden_size, hidden_size)
        self.init_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.init_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.new_control = torch.nn.Linear(hidden_size, state_dim)
        self.tail_delta = torch.nn.Linear(hidden_size, tail_update * state_dim)
        self.spline = UniformCubicBSpline(dt)
        self.reset_stream()
        torch.nn.init.zeros_(self.init_encoder[-1].weight)
        torch.nn.init.zeros_(self.init_encoder[-1].bias)
        torch.nn.init.zeros_(self.new_control.weight)
        torch.nn.init.zeros_(self.new_control.bias)
        torch.nn.init.zeros_(self.tail_delta.weight)
        torch.nn.init.zeros_(self.tail_delta.bias)

    def reset_stream(self, init_output=None, init_feature=None):
        self.hidden = None
        init = init_feature if init_feature is not None else init_output
        if init is not None:
            if init.dim() == 1:
                init = init.unsqueeze(0)
            if init.shape[-1] != self.init_size:
                raise ValueError(f'Expected PL init dim {self.init_size}, got {init.shape[-1]}.')
            self.hidden = self.init_encoder(init.detach())
        self.control_buffer = None
        self.base_buffer = None
        self.last_debug = {}

    def _initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _ghost(self, buffer, count=1):
        return buffer[:, -1:].expand(-1, int(count), -1).clone()

    def step(self, feature_t, base_pl_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_pl_t.dim() == 1:
            base_pl_t = base_pl_t.unsqueeze(0)
        if feature_t.shape[-1] != self.input_size:
            raise ValueError(f'Expected PL feature dim {self.input_size}, got {feature_t.shape[-1]}.')
        base_pl_t = normalize_gravity(base_pl_t)
        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)
        z = torch.relu(self.input(torch.cat((feature_t, base_pl_t.detach()), dim=-1)))
        z = self.dropout(z)
        self.hidden = self.cell(z, self.hidden)
        new_delta = self.new_control(self.hidden) * self.residual_scale
        new_control = base_pl_t + new_delta
        if self.control_buffer is None:
            self.control_buffer = new_control.unsqueeze(1)
            self.base_buffer = base_pl_t.unsqueeze(1)
            tail_delta_norm = new_delta.norm(dim=-1).mean()
        else:
            frozen_control = self.control_buffer.detach()
            frozen_base = self.base_buffer.detach()
            update_count = min(self.tail_update, frozen_control.shape[1])
            old_control = frozen_control[:, :-update_count]
            old_base = frozen_base[:, :-update_count]
            tail_control = frozen_control[:, -update_count:]
            tail_base = frozen_base[:, -update_count:]
            tail_delta = self.tail_delta(self.hidden).reshape(
                self.hidden.shape[0], self.tail_update, self.state_dim
            )[:, -update_count:] * self.residual_scale
            tail_control = tail_control + tail_delta
            self.control_buffer = torch.cat((old_control, tail_control, new_control.unsqueeze(1)), dim=1)
            self.base_buffer = torch.cat((old_base, tail_base, base_pl_t.unsqueeze(1)), dim=1)
            tail_delta_norm = tail_delta.norm(dim=-1).mean()
        decode_control = torch.cat((self.control_buffer, self._ghost(self.control_buffer, 1)), dim=1)
        decode_base = torch.cat((self.base_buffer, self._ghost(self.base_buffer, 1)), dim=1)
        pl_curve, pldot_curve, plddot_curve = self.spline(decode_control, return_derivatives=True)
        pl_base = self.spline(decode_base)
        pl_t = normalize_gravity(pl_curve[:, -2])
        base_t = normalize_gravity(pl_base[:, -2])
        result = {
            'pl_t': pl_t,
            'pldot_t': pldot_curve[:, -2],
            'plddot_t': plddot_curve[:, -2],
            'base_t': base_t,
            'new_control_t': new_control,
            'residual_t': pl_t - base_t,
            'control_point_prior_t': (self.control_buffer - self.base_buffer).square().mean(),
            'new_delta_norm': new_delta.norm(dim=-1).mean(),
            'tail_delta_norm': tail_delta_norm,
            'buffer_length': self.control_buffer.shape[1],
        }
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return result

    def forward_sequence(self, features, base_outputs, init_output=None, init_feature=None):
        squeeze_batch = features.dim() == 2
        if squeeze_batch:
            features = features.unsqueeze(1)
            base_outputs = base_outputs.unsqueeze(1)
            if init_output is not None and init_output.dim() == 1:
                init_output = init_output.unsqueeze(0)
            if init_feature is not None and init_feature.dim() == 1:
                init_feature = init_feature.unsqueeze(0)
        self.reset_stream(init_output=init_output, init_feature=init_feature)
        outputs, dots, ddots, bases, new_controls = [], [], [], [], []
        priors, tails, deltas = [], [], []
        for i in range(features.shape[0]):
            out = self.step(features[i], base_outputs[i])
            outputs.append(out['pl_t'])
            dots.append(out['pldot_t'])
            ddots.append(out['plddot_t'])
            bases.append(out['base_t'])
            new_controls.append(out['new_control_t'])
            priors.append(out['control_point_prior_t'])
            tails.append(out['tail_delta_norm'])
            deltas.append(out['new_delta_norm'])
        result = {
            'pl': torch.stack(outputs),
            'pldot': torch.stack(dots),
            'plddot': torch.stack(ddots),
            'base': torch.stack(bases),
            'new_control': torch.stack(new_controls),
            'control_point_prior': torch.stack(priors).mean(),
            'tail_delta_norm': torch.stack(tails).mean(),
            'new_delta_norm': torch.stack(deltas).mean(),
        }
        if squeeze_batch:
            for key in ('pl', 'pldot', 'plddot', 'base', 'new_control'):
                result[key] = result[key][:, 0]
        return result
