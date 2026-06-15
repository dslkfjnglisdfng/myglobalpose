import torch

import articulate as art
from l4_tail_update_qstate import UniformCubicBSpline
from l4_velocity_losses import finite_difference_translation_velocity
from pl_curve import fit_uniform_cubic_spline_controls, normalize_gravity, pl_target_from_pose


ROOT_STATE_DIM = 21
PL_STATE_DIM = 18


def normalize_pl_root(output):
    return torch.cat((
        output[..., :15],
        art.math.normalize_tensor(output[..., 15:18], avoid_nan=True),
        output[..., 18:21],
    ), dim=-1)


def root_velocity_target_from_pose_tran(pose, tran, dt=1.0 / 60.0):
    """Root velocity in the same root/body frame convention as pRB.

    `tran` is root translation in the mocap/world frame.  The finite-difference
    world velocity is right-multiplied by `pose[:, 0]`, matching
    `pl_target_from_pose`'s row-vector root-frame projection for pRB.
    """
    v_world = finite_difference_translation_velocity(tran, dt=dt)
    return v_world.to(pose.device, pose.dtype).unsqueeze(1).bmm(pose[:, 0]).squeeze(1)


def pl_root_target_from_pose_tran(pose, tran, body_model, dt=1.0 / 60.0):
    pl_target = pl_target_from_pose(pose, body_model)
    root_vel = root_velocity_target_from_pose_tran(pose, tran.to(pose.device, pose.dtype), dt=dt)
    return normalize_pl_root(torch.cat((pl_target, root_vel), dim=-1))


def extend_base_pl(base_pl, root_base=None):
    base_pl = normalize_gravity(base_pl)
    if root_base is None:
        root_base = base_pl.new_zeros(base_pl.shape[:-1] + (3,))
    return torch.cat((base_pl[..., :18], root_base.to(base_pl.device, base_pl.dtype)), dim=-1)


class NewPLRootModule(torch.nn.Module):
    """PLCurve-style PL module with `pRB[15]+gR1[3]+root_vel[3]` output."""

    def __init__(
        self,
        input_size=84,
        state_dim=ROOT_STATE_DIM,
        init_size=36,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.005,
        dt=1.0 / 60.0,
        dropout=0.4,
        condition_scale=1.0,
    ):
        super().__init__()
        if state_dim != ROOT_STATE_DIM:
            raise ValueError('NewPLRootModule uses the 21D pRB+gR1+root_vel state.')
        if tail_update != 4:
            raise ValueError('NewPLRootModule keeps the K2 L=4 tail-update contract.')
        if init_size < 36:
            raise ValueError('NewPLRootModule expects init36: offset_r[18]+pRL[15]+gR0[3].')
        self.input_size = int(input_size)
        self.state_dim = int(state_dim)
        self.init_size = int(init_size)
        self.hidden_size = int(hidden_size)
        self.tail_update = int(tail_update)
        self.residual_scale = float(residual_scale)
        self.dt = float(dt)
        self.condition_scale = float(condition_scale)
        self.input = torch.nn.Linear(input_size + state_dim, hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.cell = torch.nn.GRUCell(hidden_size, hidden_size)
        self.init_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.init_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.init_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.condition_input = torch.nn.Linear(hidden_size, hidden_size)
        self.condition_hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.new_control = torch.nn.Linear(hidden_size, state_dim)
        self.tail_delta = torch.nn.Linear(hidden_size, tail_update * state_dim)
        self.spline = UniformCubicBSpline(dt)
        self.reset_stream()
        torch.nn.init.zeros_(self.init_encoder[-1].weight)
        torch.nn.init.zeros_(self.init_encoder[-1].bias)
        torch.nn.init.xavier_uniform_(self.condition_encoder[-1].weight, gain=0.1)
        torch.nn.init.zeros_(self.condition_encoder[-1].bias)
        torch.nn.init.xavier_uniform_(self.condition_input.weight, gain=0.1)
        torch.nn.init.zeros_(self.condition_input.bias)
        torch.nn.init.xavier_uniform_(self.condition_hidden.weight, gain=0.1)
        torch.nn.init.zeros_(self.condition_hidden.bias)
        torch.nn.init.zeros_(self.new_control.weight)
        torch.nn.init.zeros_(self.new_control.bias)
        torch.nn.init.zeros_(self.tail_delta.weight)
        torch.nn.init.zeros_(self.tail_delta.bias)

    def reset_stream(self, init_output=None, init_feature=None):
        self.hidden = None
        self.condition = None
        init = init_feature if init_feature is not None else init_output
        if init is not None:
            if init.dim() == 1:
                init = init.unsqueeze(0)
            if init.shape[-1] != self.init_size:
                raise ValueError(f'Expected PL init dim {self.init_size}, got {init.shape[-1]}.')
            init = init.detach()
            self.hidden = self.init_encoder(init)
            self.condition = self.condition_encoder(init)
        self.control_buffer = None
        self.base_buffer = None
        self.last_debug = {}

    def _initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _ghost(self, buffer, count=1):
        return buffer[:, -1:].expand(-1, int(count), -1).clone()

    def _initial_condition(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def step(self, feature_t, base_pl_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_pl_t.dim() == 1:
            base_pl_t = base_pl_t.unsqueeze(0)
        if feature_t.shape[-1] != self.input_size:
            raise ValueError(f'Expected PL feature dim {self.input_size}, got {feature_t.shape[-1]}.')
        if base_pl_t.shape[-1] == PL_STATE_DIM:
            base_pl_t = extend_base_pl(base_pl_t)
        if base_pl_t.shape[-1] != self.state_dim:
            raise ValueError(f'Expected base PL-root dim {self.state_dim}, got {base_pl_t.shape[-1]}.')
        base_pl_t = normalize_pl_root(base_pl_t)
        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)
        if self.condition is None or self.condition.shape[0] != feature_t.shape[0]:
            self.condition = self._initial_condition(feature_t.shape[0], feature_t.device, feature_t.dtype)
        z = torch.relu(self.input(torch.cat((feature_t, base_pl_t.detach()), dim=-1)))
        z = z + self.condition_input(self.condition) * self.condition_scale
        z = self.dropout(z)
        hidden0 = self.hidden + self.condition_hidden(self.condition) * self.condition_scale
        self.hidden = self.cell(z, hidden0)
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
        pl_t = normalize_pl_root(pl_curve[:, -2])
        base_t = normalize_pl_root(pl_base[:, -2])
        result = {
            'pl': pl_t,
            'pldot': pldot_curve[:, -2],
            'plddot': plddot_curve[:, -2],
            'base': base_t,
            'new_control': new_control,
            'control_point_prior': (self.control_buffer - self.base_buffer).square().mean(),
            'tail_delta_norm': tail_delta_norm,
            'new_delta_norm': new_delta.norm(dim=-1).mean(),
            'condition_norm': self.condition.norm(dim=-1).mean(),
            'buffer_length': self.control_buffer.shape[1],
        }
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return {
            'pl_t': result['pl'],
            'pldot_t': result['pldot'],
            'plddot_t': result['plddot'],
            'base_t': result['base'],
            'new_control_t': result['new_control'],
            **{k: v for k, v in result.items() if k not in ('pl', 'pldot', 'plddot', 'base', 'new_control')},
        }

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
        outputs, dots, ddots, bases, controls = [], [], [], [], []
        priors, tails, deltas, conditions = [], [], [], []
        for i in range(features.shape[0]):
            out = self.step(features[i], base_outputs[i])
            outputs.append(out['pl_t'])
            dots.append(out['pldot_t'])
            ddots.append(out['plddot_t'])
            bases.append(out['base_t'])
            controls.append(out['new_control_t'])
            priors.append(out['control_point_prior'])
            tails.append(out['tail_delta_norm'])
            deltas.append(out['new_delta_norm'])
            conditions.append(out['condition_norm'])
        result = {
            'pl': torch.stack(outputs),
            'pldot': torch.stack(dots),
            'plddot': torch.stack(ddots),
            'base': torch.stack(bases),
            'new_control': torch.stack(controls),
            'control_point_prior': torch.stack(priors).mean(),
            'tail_delta_norm': torch.stack(tails).mean(),
            'new_delta_norm': torch.stack(deltas).mean(),
            'condition_norm': torch.stack(conditions).mean(),
        }
        if squeeze_batch:
            for key in ('pl', 'pldot', 'plddot', 'base', 'new_control'):
                result[key] = result[key][:, 0]
        return result


def newpl_root_weights():
    return {
        'pRB': 1.0,
        'gR1': 1.0,
        'root_vel': 0.25,
        'pRB_dot': 0.03,
        'gR1_dot': 0.03,
        'root_vel_smooth': 0.01,
        'gt_control_pRB': 0.3,
        'gt_control_gR1': 0.1,
        'gt_control_root_vel': 0.0,
        'root_vel_distill': 0.0,
        'control_point_prior': 0.3,
        'tail_update_prior': 0.005,
        'pRB_ddot_smooth': 0.0003,
        'gR1_ddot': 0.001,
    }


def newpl_root_loss(output, target, weights, root_vel_available=True, teacher=None, dt=1.0 / 60.0):
    pred = output['pl']
    target = target.to(pred.device, pred.dtype)
    pred_gR = art.math.normalize_tensor(pred[..., 15:18], avoid_nan=True)
    target_gR = art.math.normalize_tensor(target[..., 15:18], avoid_nan=True)
    target_pl = torch.cat((target[..., :15], target_gR, target[..., 18:21]), dim=-1)
    target_control = fit_uniform_cubic_spline_controls(target_pl)
    pred_control = output.get('new_control', pred)
    pred_control_gR = art.math.normalize_tensor(pred_control[..., 15:18], avoid_nan=True)
    target_control_gR = art.math.normalize_tensor(target_control[..., 15:18], avoid_nan=True)
    losses = {
        'pRB': torch.nn.functional.smooth_l1_loss(pred[..., :15], target[..., :15]),
        'gR1': (1.0 - (pred_gR * target_gR).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
        'gt_control_pRB': torch.nn.functional.smooth_l1_loss(pred_control[..., :15], target_control[..., :15]),
        'gt_control_gR1': torch.nn.functional.smooth_l1_loss(pred_control_gR, target_control_gR),
        'control_point_prior': output['control_point_prior'],
        'tail_update_prior': output['tail_delta_norm'],
    }
    if root_vel_available:
        losses['root_vel'] = torch.nn.functional.smooth_l1_loss(pred[..., 18:21], target[..., 18:21])
        losses['gt_control_root_vel'] = torch.nn.functional.smooth_l1_loss(pred_control[..., 18:21], target_control[..., 18:21])
    else:
        losses['root_vel'] = pred.new_zeros(())
        losses['gt_control_root_vel'] = pred.new_zeros(())
    if teacher is not None:
        losses['root_vel_distill'] = torch.nn.functional.smooth_l1_loss(pred[..., 18:21], teacher.to(pred.device, pred.dtype)[..., 18:21])
    else:
        losses['root_vel_distill'] = pred.new_zeros(())
    if pred.shape[0] >= 2:
        target_step = target[1:, ..., :15] - target[:-1, ..., :15]
        losses['pRB_dot'] = torch.nn.functional.smooth_l1_loss(dt * output['pldot'][1:, ..., :15], target_step)
        pred_gR_dot = pred_gR[1:] - pred_gR[:-1]
        target_gR_dot = target_gR[1:] - target_gR[:-1]
        losses['gR1_dot'] = torch.nn.functional.smooth_l1_loss(pred_gR_dot, target_gR_dot)
        root_vel_step = pred[1:, ..., 18:21] - pred[:-1, ..., 18:21]
        losses['root_vel_smooth'] = root_vel_step.square().mean()
    else:
        losses['pRB_dot'] = pred.new_zeros(())
        losses['gR1_dot'] = pred.new_zeros(())
        losses['root_vel_smooth'] = pred.new_zeros(())
    if pred.shape[0] >= 3:
        pred_gR_ddot = pred_gR[2:] - 2.0 * pred_gR[1:-1] + pred_gR[:-2]
        target_gR_ddot = target_gR[2:] - 2.0 * target_gR[1:-1] + target_gR[:-2]
        losses['gR1_ddot'] = torch.nn.functional.smooth_l1_loss(pred_gR_ddot, target_gR_ddot)
    else:
        losses['gR1_ddot'] = pred.new_zeros(())
    losses['pRB_ddot_smooth'] = output['plddot'][..., :15].square().mean()
    total = pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * float(weight)
    return total, losses


def load_partial_pl_checkpoint(model, checkpoint_state):
    model_state = model.state_dict()
    loaded, skipped = {}, []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped.append(key)
            continue
        target = model_state[key]
        if target.shape == value.shape:
            loaded[key] = value
        elif key == 'input.weight' and target.ndim == 2 and value.ndim == 2 and target.shape[0] == value.shape[0]:
            merged = target.clone()
            width = min(target.shape[1], value.shape[1])
            merged[:, :width] = value[:, :width]
            loaded[key] = merged
        elif key in ('new_control.weight', 'new_control.bias') and target.shape[0] >= value.shape[0]:
            merged = target.clone()
            merged[:value.shape[0], ...] = value
            loaded[key] = merged
        elif key == 'tail_delta.weight' and target.shape[1] == value.shape[1]:
            merged = target.clone()
            old = value.reshape(4, 18, value.shape[1])
            new = merged.reshape(4, 21, merged.shape[1])
            new[:, :18] = old
            loaded[key] = new.reshape_as(merged)
        elif key == 'tail_delta.bias':
            merged = target.clone()
            old = value.reshape(4, 18)
            new = merged.reshape(4, 21)
            new[:, :18] = old
            loaded[key] = new.reshape_as(merged)
        elif key == 'init_encoder.0.weight' and target.shape[0] == value.shape[0]:
            merged = target.clone()
            width = min(target.shape[1], value.shape[1])
            merged[:, -width:] = value[:, -width:]
            loaded[key] = merged
        else:
            skipped.append(key)
    model_state.update(loaded)
    model.load_state_dict(model_state)
    return {'loaded': sorted(loaded), 'skipped': sorted(skipped)}


def freeze_root_head_gradients(model):
    def zero_root_rows(grad):
        out = grad.clone()
        out[18:21] = 0
        return out

    def zero_tail_root_rows(grad):
        out = grad.clone()
        shaped = out.reshape(4, 21, *out.shape[1:])
        shaped[:, 18:21] = 0
        return shaped.reshape_as(out)

    handles = [
        model.new_control.weight.register_hook(zero_root_rows),
        model.new_control.bias.register_hook(zero_root_rows),
        model.tail_delta.weight.register_hook(zero_tail_root_rows),
        model.tail_delta.bias.register_hook(zero_tail_root_rows),
    ]
    return handles
