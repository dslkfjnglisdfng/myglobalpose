import torch

import articulate as art
from l4_tail_update_qstate import UniformCubicBSpline
from pl_curve import fit_uniform_cubic_spline_controls


def normalize_ik1(output):
    return torch.cat((
        output[..., :69],
        art.math.normalize_tensor(output[..., 69:], avoid_nan=True),
    ), dim=-1)


def split_newik1_feature(feature, tail_len=4):
    rrb_after_pl = feature[..., :45].reshape(feature.shape[:-1] + (5, 3, 3))
    control_tail = feature[..., 45:45 + tail_len * 18].reshape(feature.shape[:-1] + (tail_len, 18))
    gR1 = feature[..., 45 + tail_len * 18:45 + tail_len * 18 + 3]
    return rrb_after_pl, control_tail, gR1


def ik1_target_from_pose(pose, body_model):
    if pose.dim() == 3:
        pose = pose.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    pose_body = pose.clone()
    pose_body[:, 0] = torch.eye(3, device=pose.device, dtype=pose.dtype)
    _, joints = body_model.forward_kinematics(pose_body)[:2]
    pRJ = joints[:, 1:].reshape(pose.shape[0], 69)
    gR2 = -pose[:, 0, :, 1]
    target = normalize_ik1(torch.cat((pRJ, gR2), dim=-1))
    return target[0] if squeeze else target


def padded_control_tail(controls, frame_idx, tail_len=4):
    tail = controls[max(0, frame_idx - tail_len + 1):frame_idx + 1]
    if tail.shape[0] < tail_len:
        pad = tail[:1].expand(tail_len - tail.shape[0], -1)
        tail = torch.cat((pad, tail), dim=0)
    return tail


def fit_ik1_controls(target):
    target = normalize_ik1(target)
    target_for_control = torch.cat((target[..., :69], target[..., 69:]), dim=-1)
    return fit_uniform_cubic_spline_controls(target_for_control)


def finite_diff(x, order):
    if order == 1:
        return x[1:] - x[:-1]
    if order == 2:
        return x[2:] - 2.0 * x[1:-1] + x[:-2]
    raise ValueError(order)


SMPL_BODY_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 6), (6, 9),
    (0, 4), (4, 5), (5, 7), (7, 10),
    (0, 8), (8, 11), (11, 14), (14, 17),
    (11, 12), (12, 15), (15, 18), (18, 20), (20, 22),
    (11, 13), (13, 16), (16, 19), (19, 21),
)


def pRJ_bone_lengths(state):
    joints = state[..., :69].reshape(state.shape[:-1] + (23, 3))
    parents = torch.tensor([p for p, _ in SMPL_BODY_BONES], device=joints.device)
    children = torch.tensor([c for _, c in SMPL_BODY_BONES], device=joints.device)
    return (joints[..., children, :] - joints[..., parents, :]).norm(dim=-1)


class NewIK1ControlPointModule(torch.nn.Module):
    def __init__(
        self,
        input_size=120,
        state_dim=72,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.005,
        dt=1.0 / 60.0,
        dropout=0.4,
    ):
        super().__init__()
        if state_dim != 72:
            raise ValueError('NewIK1_ControlPoint_v1 predicts full IK1 state pRJ[69]+gR2[3].')
        if tail_update != 4:
            raise ValueError('NewIK1_ControlPoint_v1 currently uses tail_len=4.')
        self.input_size = int(input_size)
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.tail_update = int(tail_update)
        self.residual_scale = float(residual_scale)
        self.dt = float(dt)
        self.input = torch.nn.Linear(input_size + state_dim, hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.cell = torch.nn.GRUCell(hidden_size, hidden_size)
        self.init_encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
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

    def reset_stream(self, init_output=None):
        self.hidden = None
        if init_output is not None:
            if init_output.dim() == 1:
                init_output = init_output.unsqueeze(0)
            self.hidden = self.init_encoder(normalize_ik1(init_output).detach())
        self.control_buffer = None
        self.base_buffer = None
        self.last_debug = {}

    def _initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _ghost(self, buffer, count=1):
        return buffer[:, -1:].expand(-1, int(count), -1).clone()

    def control_tail(self):
        if self.control_buffer is None:
            return None
        tail = self.control_buffer[:, -self.tail_update:, :]
        if tail.shape[1] < self.tail_update:
            pad = tail[:, :1].expand(-1, self.tail_update - tail.shape[1], -1)
            tail = torch.cat((pad, tail), dim=1)
        return tail

    def step(self, feature_t, base_ik1_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_ik1_t.dim() == 1:
            base_ik1_t = base_ik1_t.unsqueeze(0)
        if feature_t.shape[-1] != self.input_size:
            raise ValueError(f'Expected NewIK1 feature dim {self.input_size}, got {feature_t.shape[-1]}.')
        base_ik1_t = normalize_ik1(base_ik1_t)
        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)
        z = torch.relu(self.input(torch.cat((feature_t, base_ik1_t.detach()), dim=-1)))
        z = self.dropout(z)
        self.hidden = self.cell(z, self.hidden)
        new_delta = self.new_control(self.hidden) * self.residual_scale
        new_control = normalize_ik1(base_ik1_t + new_delta)
        if self.control_buffer is None:
            self.control_buffer = new_control.unsqueeze(1)
            self.base_buffer = base_ik1_t.unsqueeze(1)
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
            tail_control = normalize_ik1(tail_control + tail_delta)
            self.control_buffer = torch.cat((old_control, tail_control, new_control.unsqueeze(1)), dim=1)
            self.base_buffer = torch.cat((old_base, tail_base, base_ik1_t.unsqueeze(1)), dim=1)
            tail_delta_norm = tail_delta.norm(dim=-1).mean()
        decode_control = torch.cat((self.control_buffer, self._ghost(self.control_buffer, 1)), dim=1)
        decode_base = torch.cat((self.base_buffer, self._ghost(self.base_buffer, 1)), dim=1)
        curve, dot_curve, ddot_curve = self.spline(decode_control, return_derivatives=True)
        base_curve = self.spline(decode_base)
        ik1_t = normalize_ik1(curve[:, -2])
        base_t = normalize_ik1(base_curve[:, -2])
        result = {
            'ik1_t': ik1_t,
            'ik1dot_t': dot_curve[:, -2],
            'ik1ddot_t': ddot_curve[:, -2],
            'base_t': base_t,
            'new_control_t': new_control,
            'control_tail_t': self.control_tail(),
            'control_point_prior_t': (self.control_buffer - self.base_buffer).square().mean(),
            'new_delta_norm': new_delta.norm(dim=-1).mean(),
            'tail_delta_norm': tail_delta_norm,
            'buffer_length': self.control_buffer.shape[1],
        }
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return result

    def forward_sequence(self, features, base_outputs, init_output=None):
        squeeze_batch = features.dim() == 2
        if squeeze_batch:
            features = features.unsqueeze(1)
            base_outputs = base_outputs.unsqueeze(1)
            if init_output is not None and init_output.dim() == 1:
                init_output = init_output.unsqueeze(0)
        self.reset_stream(init_output)
        outputs, dots, ddots, bases, controls, tails = [], [], [], [], [], []
        priors, tail_norms, delta_norms = [], [], []
        for i in range(features.shape[0]):
            out = self.step(features[i], base_outputs[i])
            outputs.append(out['ik1_t'])
            dots.append(out['ik1dot_t'])
            ddots.append(out['ik1ddot_t'])
            bases.append(out['base_t'])
            controls.append(out['new_control_t'])
            tails.append(out['control_tail_t'])
            priors.append(out['control_point_prior_t'])
            tail_norms.append(out['tail_delta_norm'])
            delta_norms.append(out['new_delta_norm'])
        result = {
            'ik1': torch.stack(outputs),
            'ik1dot': torch.stack(dots),
            'ik1ddot': torch.stack(ddots),
            'base': torch.stack(bases),
            'new_control': torch.stack(controls),
            'control_tail': torch.stack(tails),
            'control_point_prior': torch.stack(priors).mean(),
            'tail_delta_norm': torch.stack(tail_norms).mean(),
            'new_delta_norm': torch.stack(delta_norms).mean(),
        }
        if squeeze_batch:
            for key in ('ik1', 'ik1dot', 'ik1ddot', 'base', 'new_control', 'control_tail'):
                result[key] = result[key][:, 0]
        return result


def newik1_loss(output, target, target_control_tail, weights):
    pred = normalize_ik1(output['ik1'])
    target = normalize_ik1(target.to(pred.device, pred.dtype))
    target_control_tail = target_control_tail.to(pred.device, pred.dtype)
    pred_g = art.math.normalize_tensor(pred[..., 69:], avoid_nan=True)
    target_g = art.math.normalize_tensor(target[..., 69:], avoid_nan=True)
    pred_tail = output['control_tail']
    pred_tail_g = art.math.normalize_tensor(pred_tail[..., 69:], avoid_nan=True)
    target_tail_g = art.math.normalize_tensor(target_control_tail[..., 69:], avoid_nan=True)
    control_pRJ = torch.nn.functional.smooth_l1_loss(pred_tail[..., :69], target_control_tail[..., :69])
    control_gR2 = (1.0 - (pred_tail_g * target_tail_g).sum(dim=-1).clamp(-1.0, 1.0)).mean()
    losses = {
        'control': torch.nn.functional.smooth_l1_loss(
            torch.cat((pred_tail[..., :69], pred_tail_g), dim=-1),
            torch.cat((target_control_tail[..., :69], target_tail_g), dim=-1),
        ),
        'control_pRJ': control_pRJ,
        'control_gR2': control_gR2,
        'pRJ': torch.nn.functional.smooth_l1_loss(pred[..., :69], target[..., :69]),
        'gR2': (1.0 - (pred_g * target_g).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
        'bone_length': torch.nn.functional.smooth_l1_loss(pRJ_bone_lengths(pred), pRJ_bone_lengths(target)),
        'control_point_prior': output['control_point_prior'],
        'tail_update_prior': output['tail_delta_norm'],
    }
    if pred.shape[0] >= 2:
        losses['pRJ_dot'] = torch.nn.functional.smooth_l1_loss(
            pred[1:, ..., :69] - pred[:-1, ..., :69],
            target[1:, ..., :69] - target[:-1, ..., :69],
        )
        losses['gR2_dot'] = torch.nn.functional.smooth_l1_loss(
            pred_g[1:] - pred_g[:-1],
            target_g[1:] - target_g[:-1],
        )
    else:
        losses['pRJ_dot'] = pred.new_zeros(())
        losses['gR2_dot'] = pred.new_zeros(())
    if pred.shape[0] >= 3:
        losses['pRJ_ddot'] = torch.nn.functional.smooth_l1_loss(
            finite_diff(pred[..., :69], 2),
            finite_diff(target[..., :69], 2),
        )
        losses['gR2_ddot'] = torch.nn.functional.smooth_l1_loss(
            finite_diff(pred_g, 2),
            finite_diff(target_g, 2),
        )
    else:
        losses['pRJ_ddot'] = pred.new_zeros(())
        losses['gR2_ddot'] = pred.new_zeros(())
    total = pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * weight
    return total, losses
