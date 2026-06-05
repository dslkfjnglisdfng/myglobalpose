import torch

import articulate as art
from l4_tail_update_qstate import UniformCubicBSpline


IK1_LEAF_JOINT_IDS = (18, 19, 4, 5, 15)
IK1_LEAF_PRJ_INDICES = tuple(j - 1 for j in IK1_LEAF_JOINT_IDS)
IK1_NONLEAF_PRJ_INDICES = tuple(i for i in range(23) if i not in IK1_LEAF_PRJ_INDICES)


def normalize_ik1(output):
    return torch.cat((
        output[..., :69],
        art.math.normalize_tensor(output[..., 69:], avoid_nan=True),
    ), dim=-1)


def split_ik1_input(feature):
    rrb_after_pl = feature[..., :45].reshape(feature.shape[:-1] + (5, 3, 3))
    gR1 = feature[..., 45:48]
    pl_control_tail = feature[..., 48:].reshape(feature.shape[:-1] + (4, 18))
    return rrb_after_pl, gR1, pl_control_tail


def assemble_pRJ(nonleaf_pRJ, leaf_pRB):
    out_shape = nonleaf_pRJ.shape[:-1] + (23, 3)
    pRJ = nonleaf_pRJ.new_zeros(out_shape)
    pRJ[..., IK1_NONLEAF_PRJ_INDICES, :] = nonleaf_pRJ.reshape(nonleaf_pRJ.shape[:-1] + (len(IK1_NONLEAF_PRJ_INDICES), 3))
    pRJ[..., IK1_LEAF_PRJ_INDICES, :] = leaf_pRB.reshape(leaf_pRB.shape[:-1] + (len(IK1_LEAF_PRJ_INDICES), 3))
    return pRJ.reshape(nonleaf_pRJ.shape[:-1] + (69,))


def extract_nonleaf_pRJ(full_pRJ):
    pRJ = full_pRJ.reshape(full_pRJ.shape[:-1] + (23, 3))
    return pRJ[..., IK1_NONLEAF_PRJ_INDICES, :].reshape(full_pRJ.shape[:-1] + (len(IK1_NONLEAF_PRJ_INDICES) * 3,))


def ik1_state_from_full(full_ik1):
    full_ik1 = normalize_ik1(full_ik1)
    return torch.cat((extract_nonleaf_pRJ(full_ik1[..., :69]), full_ik1[..., 69:]), dim=-1)


def full_ik1_from_state(state, leaf_pRB):
    pRJ = assemble_pRJ(state[..., :54], leaf_pRB)
    return normalize_ik1(torch.cat((pRJ, state[..., 54:]), dim=-1))


class IK1CurveModule(torch.nn.Module):
    def __init__(
        self,
        input_size=120,
        state_dim=57,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.005,
        dt=1.0 / 60.0,
        dropout=0.4,
    ):
        super().__init__()
        if state_dim != 57:
            raise ValueError('IK1CurveModule v1 uses state [18 non-leaf pRJ * 3 + gR2 3].')
        if tail_update != 4:
            raise ValueError('IK1CurveModule v1 keeps the K2 L=4 tail-update contract.')
        self.input_size = int(input_size)
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.tail_update = int(tail_update)
        self.residual_scale = float(residual_scale)
        self.dt = float(dt)
        self.input = torch.nn.Linear(input_size + state_dim, hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
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

    def reset_stream(self, init_state=None):
        self.hidden = None
        if init_state is not None:
            if init_state.dim() == 1:
                init_state = init_state.unsqueeze(0)
            self.hidden = self.init_encoder(init_state.detach())
        self.control_buffer = None
        self.base_buffer = None
        self.last_debug = {}

    def _initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _ghost(self, buffer, count=1):
        return buffer[:, -1:].expand(-1, int(count), -1).clone()

    def step(self, feature_t, base_ik1_t, leaf_pRB_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_ik1_t.dim() == 1:
            base_ik1_t = base_ik1_t.unsqueeze(0)
        if leaf_pRB_t.dim() == 1:
            leaf_pRB_t = leaf_pRB_t.unsqueeze(0)
        if feature_t.shape[-1] != self.input_size:
            raise ValueError(f'Expected IK1 feature dim {self.input_size}, got {feature_t.shape[-1]}.')
        base_state_t = ik1_state_from_full(base_ik1_t)
        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)
        z = torch.relu(self.input(torch.cat((feature_t, base_state_t.detach()), dim=-1)))
        z = self.dropout(z)
        self.hidden = self.cell(z, self.hidden)
        new_delta = self.new_control(self.hidden) * self.residual_scale
        new_control = base_state_t + new_delta
        if self.control_buffer is None:
            self.control_buffer = new_control.unsqueeze(1)
            self.base_buffer = base_state_t.unsqueeze(1)
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
            self.base_buffer = torch.cat((old_base, tail_base, base_state_t.unsqueeze(1)), dim=1)
            tail_delta_norm = tail_delta.norm(dim=-1).mean()
        decode_control = torch.cat((self.control_buffer, self._ghost(self.control_buffer, 1)), dim=1)
        decode_base = torch.cat((self.base_buffer, self._ghost(self.base_buffer, 1)), dim=1)
        state_curve, statedot_curve, stateddot_curve = self.spline(decode_control, return_derivatives=True)
        base_curve = self.spline(decode_base)
        state_t = state_curve[:, -2]
        base_state = base_curve[:, -2]
        statedot_t = statedot_curve[:, -2]
        stateddot_t = stateddot_curve[:, -2]
        ik1_t = full_ik1_from_state(state_t, leaf_pRB_t)
        base_t = full_ik1_from_state(base_state, leaf_pRB_t)
        result = {
            'ik1_t': ik1_t,
            'state_t': torch.cat((state_t[:, :54], art.math.normalize_tensor(state_t[:, 54:], avoid_nan=True)), dim=-1),
            'statedot_t': statedot_t,
            'stateddot_t': stateddot_t,
            'base_t': base_t,
            'control_point_prior_t': (self.control_buffer - self.base_buffer).square().mean(),
            'new_delta_norm': new_delta.norm(dim=-1).mean(),
            'tail_delta_norm': tail_delta_norm,
            'buffer_length': self.control_buffer.shape[1],
        }
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return result

    def forward_sequence(self, features, base_outputs, leaf_pRB, init_output=None):
        squeeze_batch = features.dim() == 2
        if squeeze_batch:
            features = features.unsqueeze(1)
            base_outputs = base_outputs.unsqueeze(1)
            leaf_pRB = leaf_pRB.unsqueeze(1)
            if init_output is not None and init_output.dim() == 1:
                init_output = init_output.unsqueeze(0)
        init_state = ik1_state_from_full(init_output) if init_output is not None else None
        self.reset_stream(init_state)
        outputs, states, dots, ddots, bases = [], [], [], [], []
        priors, tails, deltas = [], [], []
        for i in range(features.shape[0]):
            out = self.step(features[i], base_outputs[i], leaf_pRB[i])
            outputs.append(out['ik1_t'])
            states.append(out['state_t'])
            dots.append(out['statedot_t'])
            ddots.append(out['stateddot_t'])
            bases.append(out['base_t'])
            priors.append(out['control_point_prior_t'])
            tails.append(out['tail_delta_norm'])
            deltas.append(out['new_delta_norm'])
        result = {
            'ik1': torch.stack(outputs),
            'state': torch.stack(states),
            'statedot': torch.stack(dots),
            'stateddot': torch.stack(ddots),
            'base': torch.stack(bases),
            'control_point_prior': torch.stack(priors).mean(),
            'tail_delta_norm': torch.stack(tails).mean(),
            'new_delta_norm': torch.stack(deltas).mean(),
        }
        if squeeze_batch:
            for key in ('ik1', 'state', 'statedot', 'stateddot', 'base'):
                result[key] = result[key][:, 0]
        return result
