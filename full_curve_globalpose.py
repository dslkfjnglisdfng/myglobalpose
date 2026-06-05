import torch

import articulate as art
from l4_tail_update_qstate import UniformCubicBSpline
from pl_curve import normalize_gravity, split_pl_feature
from ik1_curve import normalize_ik1


DT = 1.0 / 60.0


IK2_REDUCED_JOINTS = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
IK2_REDUCED_PARENTS = (0, 1, 2, 3, 4, 5, 2, 0, 12, 13, 14, 12, 16, 17, 18)
IK2_REDUCED_INDEX = {joint: idx for idx, joint in enumerate(IK2_REDUCED_JOINTS)}
IK2_REDUCED_PARENT_PAIRS = tuple(
    (idx, IK2_REDUCED_INDEX[parent])
    for idx, parent in enumerate(IK2_REDUCED_PARENTS)
    if parent in IK2_REDUCED_INDEX
)


def rotation_matrix_to_6d(rotation):
    return rotation[..., :, :2].reshape(rotation.shape[:-2] + (6,))


def cosine_direction_loss(pred, target):
    target = target.to(pred.device, pred.dtype)
    return (1.0 - (pred * target).sum(dim=-1).clamp(-1.0, 1.0)).mean()


def sequence_delta_loss(pred_dot, target, start_dim, end_dim, dt=DT):
    if target.shape[0] < 2:
        return pred_dot.new_zeros(())
    target_step = target[1:, ..., start_dim:end_dim] - target[:-1, ..., start_dim:end_dim]
    return torch.nn.functional.smooth_l1_loss(dt * pred_dot[1:, ..., start_dim:end_dim], target_step.to(pred_dot.device, pred_dot.dtype))


def smooth_ddot_loss(ddot, start_dim=0, end_dim=None):
    end_dim = ddot.shape[-1] if end_dim is None else end_dim
    return ddot[..., start_dim:end_dim].square().mean()


def direction_delta_loss(pred_dot, target, start_dim, end_dim, dt=DT):
    if target.shape[0] < 2:
        return pred_dot.new_zeros(())
    target = target.to(pred_dot.device, pred_dot.dtype)
    target_step = target[1:, ..., start_dim:end_dim] - target[:-1, ..., start_dim:end_dim]
    return torch.nn.functional.smooth_l1_loss(dt * pred_dot[1:, ..., start_dim:end_dim], target_step)


def ik2_parent_relative_loss(pred, target):
    pred_rot = art.math.r6d_to_rotation_matrix(pred.reshape(pred.shape[:-1] + (15, 6)))
    target_rot = art.math.r6d_to_rotation_matrix(target.to(pred.device, pred.dtype).reshape(target.shape[:-1] + (15, 6)))
    losses = []
    for child_idx, parent_idx in IK2_REDUCED_PARENT_PAIRS:
        pred_rel = pred_rot[..., parent_idx, :, :].transpose(-1, -2).matmul(pred_rot[..., child_idx, :, :])
        target_rel = target_rot[..., parent_idx, :, :].transpose(-1, -2).matmul(target_rot[..., child_idx, :, :])
        losses.append(torch.nn.functional.smooth_l1_loss(pred_rel, target_rel))
    return torch.stack(losses).mean() if losses else pred.new_zeros(())


def joint_two_node_distance_loss(pred_ik1, target_ik1):
    pred_pRJ = pred_ik1[..., :69].reshape(pred_ik1.shape[:-1] + (23, 3))
    target_pRJ = target_ik1.to(pred_ik1.device, pred_ik1.dtype)[..., :69].reshape(target_ik1.shape[:-1] + (23, 3))
    losses = []
    for child_idx, parent_idx in IK2_REDUCED_PARENT_PAIRS:
        child_joint = IK2_REDUCED_JOINTS[child_idx]
        parent_joint = IK2_REDUCED_JOINTS[parent_idx]
        pred_vec = pred_pRJ[..., child_joint - 1, :] - pred_pRJ[..., parent_joint - 1, :]
        target_vec = target_pRJ[..., child_joint - 1, :] - target_pRJ[..., parent_joint - 1, :]
        losses.append(torch.nn.functional.smooth_l1_loss(pred_vec.norm(dim=-1), target_vec.norm(dim=-1)))
    return torch.stack(losses).mean() if losses else pred_ik1.new_zeros(())


def safe_from_to_rotation_matrix(from_vector, to_vector, eps=1e-6):
    shape = from_vector.shape[:-1]
    source = torch.nn.functional.normalize(from_vector.reshape(-1, 3), dim=-1, eps=eps)
    target = torch.nn.functional.normalize(to_vector.reshape(-1, 3), dim=-1, eps=eps)
    cross = torch.cross(source, target, dim=-1)
    dot = (source * target).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    skew = source.new_zeros(source.shape[0], 3, 3)
    skew[:, 0, 1] = -cross[:, 2]
    skew[:, 0, 2] = cross[:, 1]
    skew[:, 1, 0] = cross[:, 2]
    skew[:, 1, 2] = -cross[:, 0]
    skew[:, 2, 0] = -cross[:, 1]
    skew[:, 2, 1] = cross[:, 0]
    eye = torch.eye(3, device=source.device, dtype=source.dtype).unsqueeze(0)
    rotation = eye + skew + skew.matmul(skew) / (1.0 + dot).clamp_min(eps).unsqueeze(-1)
    return rotation.reshape(shape + (3, 3))


def normalize_stage_output(x, state_dim):
    if state_dim == 18:
        return normalize_gravity(x)
    if state_dim == 72:
        return normalize_ik1(x)
    return x


class FullCurveStage(torch.nn.Module):
    def __init__(self, input_size, state_dim, hidden_size=512, tail_update=4, residual_scale=0.005, dt=DT, dropout=0.4, offset_init_scale=0.1):
        super().__init__()
        if tail_update != 4:
            raise ValueError('FullCurveGlobalPose_v1 uses L=4 control tails.')
        self.input_size = int(input_size)
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.tail_update = int(tail_update)
        self.residual_scale = float(residual_scale)
        self.offset_init_scale = float(offset_init_scale)
        self.input = torch.nn.Linear(self.input_size + self.state_dim, self.hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.cell = torch.nn.GRUCell(self.hidden_size, self.hidden_size)
        self.init_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.rnn_init_encoder = torch.nn.Sequential(
            torch.nn.Linear(18 + self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.new_control = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.tail_delta = torch.nn.Linear(self.hidden_size, self.tail_update * self.state_dim)
        self.spline = UniformCubicBSpline(dt)
        self.reset_stream()
        torch.nn.init.zeros_(self.init_encoder[-1].weight)
        torch.nn.init.zeros_(self.init_encoder[-1].bias)
        torch.nn.init.zeros_(self.rnn_init_encoder[-1].weight)
        torch.nn.init.zeros_(self.rnn_init_encoder[-1].bias)
        torch.nn.init.zeros_(self.new_control.weight)
        torch.nn.init.zeros_(self.new_control.bias)
        torch.nn.init.zeros_(self.tail_delta.weight)
        torch.nn.init.zeros_(self.tail_delta.bias)

    def reset_stream(self, init_output=None, offset_r=None, init_feature=None):
        self.hidden = None
        if offset_r is not None and init_feature is not None:
            self.hidden = self._firstframe_hidden(offset_r, init_feature)
        elif init_output is not None:
            if init_output.dim() == 1:
                init_output = init_output.unsqueeze(0)
            self.hidden = self.init_encoder(init_output.detach())
        self.control_buffer = None
        self.base_buffer = None

    def _firstframe_hidden(self, offset_r, init_feature):
        ref = next(self.rnn_init_encoder.parameters())
        offset = offset_r.detach().to(device=ref.device, dtype=ref.dtype)
        feature = init_feature.detach().to(device=ref.device, dtype=ref.dtype)
        if offset.dim() == 2:
            offset = offset.unsqueeze(0)
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)
        offset = offset.reshape(feature.shape[0], -1)
        feature = feature.reshape(feature.shape[0], -1)
        if offset.shape[-1] != 18:
            raise ValueError(f'Expected offset_r flatten dim 18, got {offset.shape[-1]}.')
        if feature.shape[-1] != self.input_size:
            raise ValueError(f'Expected first-frame feature dim {self.input_size}, got {feature.shape[-1]}.')
        return self.rnn_init_encoder(torch.cat((offset, feature), dim=-1)) * self.offset_init_scale

    def _initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _ghost(self, buffer, count=1):
        return buffer[:, -1:].expand(-1, int(count), -1).clone()

    def control_tail(self):
        if self.control_buffer is None:
            raise RuntimeError('control_tail requested before stage has run.')
        control = self.control_buffer[:, -self.tail_update:, :]
        if control.shape[1] < self.tail_update:
            pad = control[:, :1].expand(-1, self.tail_update - control.shape[1], -1)
            control = torch.cat((pad, control), dim=1)
        return control

    def control_diagnostics(self):
        if self.control_buffer is None or self.base_buffer is None:
            zero = next(self.parameters()).new_zeros(())
            return {
                'residual_norm_mean': zero,
                'residual_norm_max': zero,
                'residual_norm_std': zero,
                'temporal_step_norm_mean': zero,
                'temporal_step_norm_min': zero,
                'temporal_step_norm_std': zero,
                'value_std': zero,
                'buffer_length': 0,
            }
        residual_norm = (self.control_buffer - self.base_buffer).norm(dim=-1)
        if self.control_buffer.shape[1] > 1:
            step_norm = (self.control_buffer[:, 1:] - self.control_buffer[:, :-1]).norm(dim=-1)
            step_mean = step_norm.mean()
            step_min = step_norm.min()
            step_std = step_norm.std(unbiased=False)
        else:
            step_mean = residual_norm.new_zeros(())
            step_min = residual_norm.new_zeros(())
            step_std = residual_norm.new_zeros(())
        return {
            'residual_norm_mean': residual_norm.mean(),
            'residual_norm_max': residual_norm.max(),
            'residual_norm_std': residual_norm.std(unbiased=False),
            'temporal_step_norm_mean': step_mean,
            'temporal_step_norm_min': step_min,
            'temporal_step_norm_std': step_std,
            'value_std': self.control_buffer.std(unbiased=False),
            'buffer_length': int(self.control_buffer.shape[1]),
        }

    def step(self, feature_t, base_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_t.dim() == 1:
            base_t = base_t.unsqueeze(0)
        if feature_t.shape[-1] != self.input_size:
            raise ValueError(f'Expected feature dim {self.input_size}, got {feature_t.shape[-1]}.')
        base_t = normalize_stage_output(base_t, self.state_dim)
        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)
        z = torch.relu(self.input(torch.cat((feature_t, base_t.detach()), dim=-1)))
        z = self.dropout(z)
        self.hidden = self.cell(z, self.hidden)
        new_delta = self.new_control(self.hidden) * self.residual_scale
        new_control = base_t + new_delta
        if self.control_buffer is None:
            self.control_buffer = new_control.unsqueeze(1)
            self.base_buffer = base_t.unsqueeze(1)
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
            self.base_buffer = torch.cat((old_base, tail_base, base_t.unsqueeze(1)), dim=1)
            tail_delta_norm = tail_delta.norm(dim=-1).mean()
        decode_control = torch.cat((self.control_buffer, self._ghost(self.control_buffer, 1)), dim=1)
        decode_base = torch.cat((self.base_buffer, self._ghost(self.base_buffer, 1)), dim=1)
        curve, dot, ddot = self.spline(decode_control, return_derivatives=True)
        base_curve = self.spline(decode_base)
        out_t = normalize_stage_output(curve[:, -2], self.state_dim)
        base_out_t = normalize_stage_output(base_curve[:, -2], self.state_dim)
        return {
            'out_t': out_t,
            'dot_t': dot[:, -2],
            'ddot_t': ddot[:, -2],
            'base_t': base_out_t,
            'control_point_prior_t': (self.control_buffer - self.base_buffer).square().mean(),
            'new_delta_norm': new_delta.norm(dim=-1).mean(),
            'tail_delta_norm': tail_delta_norm,
        }


class FullCurveGlobalPoseV1(torch.nn.Module):
    def __init__(self, hidden_size=512, tail_update=4, residual_scale=0.005, vr_residual_scale=0.005, dropout=0.4, offset_init_scale=0.1):
        super().__init__()
        self.tail_update = int(tail_update)
        self.pl = FullCurveStage(84, 18, hidden_size, tail_update, residual_scale, dropout=dropout, offset_init_scale=offset_init_scale)
        self.ik1 = FullCurveStage(45 + 3 + tail_update * 18, 72, hidden_size, tail_update, residual_scale, dropout=dropout, offset_init_scale=offset_init_scale)
        self.ik2 = FullCurveStage(45 + 3 + tail_update * 72, 90, hidden_size, tail_update, residual_scale, dropout=dropout, offset_init_scale=offset_init_scale)
        self.vr = FullCurveStage(tail_update * (18 + 72 + 90) + 90, 9, hidden_size, tail_update, vr_residual_scale, dropout=dropout, offset_init_scale=offset_init_scale)

    @staticmethod
    def processed_imu_feature(a, w, R):
        return torch.cat((a.reshape(-1), w.reshape(-1), R.reshape(-1)), dim=-1)

    def reset_stream(self, init):
        offset_r = init.get('offset_r')
        self.pl.reset_stream(init.get('pl'), offset_r, init.get('pl_feature'))
        self.ik1.reset_stream(init.get('ik1'), offset_r, init.get('ik1_feature'))
        self.ik2.reset_stream(init.get('ik2'), offset_r, init.get('ik2_feature'))
        self.vr.reset_stream(init.get('vr'), offset_r, init.get('vr_feature'))

    def forward_sequence(self, record):
        pl_input = record['pl_input']
        base_pl = record['pl_base']
        base_ik1 = record['ik1_base']
        base_ik2 = record['ik2_base']
        base_vr = record['vr_base']
        imu = record['processed_imu']
        squeeze_batch = pl_input.dim() == 2
        if squeeze_batch:
            pl_input = pl_input.unsqueeze(1)
            base_pl = base_pl.unsqueeze(1)
            base_ik1 = base_ik1.unsqueeze(1)
            base_ik2 = base_ik2.unsqueeze(1)
            base_vr = base_vr.unsqueeze(1)
            imu = imu.unsqueeze(1)
        RRB0, gR0 = split_pl_feature(pl_input)
        init_pl = normalize_stage_output(base_pl[0], 18)
        init_ik1 = normalize_stage_output(base_ik1[0], 72)
        init_ik2 = normalize_stage_output(base_ik2[0], 90)
        init_vr = base_vr[0]
        init_gR1 = init_pl[:, 15:]
        init_rrb_after_pl = safe_from_to_rotation_matrix(gR0[0], init_gR1).unsqueeze(1).matmul(RRB0[0])
        init_ik1_feature = torch.cat((
            init_rrb_after_pl.flatten(1),
            init_gR1,
            init_pl.unsqueeze(1).expand(-1, self.tail_update, -1).flatten(1),
        ), dim=-1)
        init_gR2 = init_ik1[:, 69:]
        init_rrb_after_ik1 = safe_from_to_rotation_matrix(init_gR1, init_gR2).unsqueeze(1).matmul(init_rrb_after_pl)
        init_ik2_feature = torch.cat((
            init_rrb_after_ik1.flatten(1),
            init_gR2,
            init_ik1.unsqueeze(1).expand(-1, self.tail_update, -1).flatten(1),
        ), dim=-1)
        init_vr_feature = torch.cat((
            init_pl.unsqueeze(1).expand(-1, self.tail_update, -1).flatten(1),
            init_ik1.unsqueeze(1).expand(-1, self.tail_update, -1).flatten(1),
            init_ik2.unsqueeze(1).expand(-1, self.tail_update, -1).flatten(1),
            imu[0].flatten(1),
        ), dim=-1)
        self.reset_stream({
            'pl': init_pl,
            'ik1': init_ik1,
            'ik2': init_ik2,
            'vr': init_vr,
            'offset_r': record.get('offset_r'),
            'pl_feature': pl_input[0],
            'ik1_feature': init_ik1_feature,
            'ik2_feature': init_ik2_feature,
            'vr_feature': init_vr_feature,
        })
        outputs = {key: [] for key in (
            'pl', 'pldot', 'plddot', 'pl_base',
            'ik1', 'ik1dot', 'ik1ddot', 'ik1_base',
            'ik2', 'ik2dot', 'ik2ddot', 'ik2_base',
            'vr', 'vrdot', 'vrddot', 'vr_base',
        )}
        priors, tails, deltas = [], [], []
        for t in range(pl_input.shape[0]):
            pl_out = self.pl.step(pl_input[t], base_pl[t])
            pl_t = pl_out['out_t']
            gR1 = pl_t[:, 15:]
            RRB_after_pl = safe_from_to_rotation_matrix(gR0[t], gR1).unsqueeze(1).matmul(RRB0[t])
            ik1_feature = torch.cat((RRB_after_pl.flatten(1), gR1, self.pl.control_tail().flatten(1)), dim=-1)
            ik1_out = self.ik1.step(ik1_feature, base_ik1[t])
            ik1_t = ik1_out['out_t']
            gR2 = ik1_t[:, 69:]
            RRB_after_ik1 = safe_from_to_rotation_matrix(gR1, gR2).unsqueeze(1).matmul(RRB_after_pl)
            ik2_feature = torch.cat((RRB_after_ik1.flatten(1), gR2, self.ik1.control_tail().flatten(1)), dim=-1)
            ik2_out = self.ik2.step(ik2_feature, base_ik2[t])
            vr_feature = torch.cat((
                self.pl.control_tail().flatten(1),
                self.ik1.control_tail().flatten(1),
                self.ik2.control_tail().flatten(1),
                imu[t].flatten(1),
            ), dim=-1)
            vr_out = self.vr.step(vr_feature, base_vr[t])
            for prefix, out in (('pl', pl_out), ('ik1', ik1_out), ('ik2', ik2_out), ('vr', vr_out)):
                outputs[prefix].append(out['out_t'])
                outputs[f'{prefix}dot'].append(out['dot_t'])
                outputs[f'{prefix}ddot'].append(out['ddot_t'])
                outputs[f'{prefix}_base'].append(out['base_t'])
                priors.append(out['control_point_prior_t'])
                tails.append(out['tail_delta_norm'])
                deltas.append(out['new_delta_norm'])
        result = {key: torch.stack(value) for key, value in outputs.items()}
        result.update({
            'control_point_prior': torch.stack(priors).mean(),
            'tail_delta_norm': torch.stack(tails).mean(),
            'new_delta_norm': torch.stack(deltas).mean(),
            'control_shapes': {
                'pl': list(self.pl.control_buffer.shape),
                'ik1': list(self.ik1.control_buffer.shape),
                'ik2': list(self.ik2.control_buffer.shape),
                'vr': list(self.vr.control_buffer.shape),
            },
            'control_diagnostics': {
                'pl': self.pl.control_diagnostics(),
                'ik1': self.ik1.control_diagnostics(),
                'ik2': self.ik2.control_diagnostics(),
                'vr': self.vr.control_diagnostics(),
            },
        })
        if squeeze_batch:
            for key, value in list(result.items()):
                if torch.is_tensor(value) and value.dim() >= 2:
                    result[key] = value[:, 0]
        return result


def full_curve_default_weights():
    return {
        'pl_pRB': 1.0,
        'pl_gR1': 1.0,
        'pl_dot': 0.03,
        'pl_ddot': 0.0003,
        'pl_gR1_dot': 0.01,
        'pl_gR1_ddot': 1e-6,
        'ik1_pRJ': 1.0,
        'ik1_gR2_base': 0.2,
        'ik1_gR2_dot': 0.01,
        'ik1_gR2_ddot': 1e-6,
        'ik1_dot': 0.03,
        'ik1_ddot': 1e-6,
        'ik2_r6d': 1.0,
        'ik2_parent_relative': 0.05,
        'ik_joint_two_node_distance': 0.05,
        'ik2_dot': 0.01,
        'ik2_ddot': 1e-6,
        'vr': 1.0,
        'vr_velocity': 1.0,
        'vr_contact': 1.0,
        'vr_dot': 0.01,
        'control_point_prior': 0.3,
        'tail_update_prior': 0.005,
    }


def full_curve_loss(output, record, weights, dt=DT):
    losses = {
        'pl_pRB': torch.nn.functional.smooth_l1_loss(
            output['pl'][..., :15], record['pl_target'].to(output['pl'].device, output['pl'].dtype)[..., :15]
        ),
        'pl_gR1': cosine_direction_loss(output['pl'][..., 15:], record['pl_target'][..., 15:]),
        'pl_dot': sequence_delta_loss(output['pldot'], record['pl_target'], 0, 15, dt),
        'pl_ddot': smooth_ddot_loss(output['plddot'], 0, 15),
        'pl_gR1_dot': direction_delta_loss(output['pldot'], record['pl_target'], 15, 18, dt),
        'pl_gR1_ddot': smooth_ddot_loss(output['plddot'], 15, 18),
        'ik1_pRJ': torch.nn.functional.smooth_l1_loss(
            output['ik1'][..., :69], record['ik1_target'].to(output['ik1'].device, output['ik1'].dtype)[..., :69]
        ),
        'ik1_gR2_base': cosine_direction_loss(output['ik1'][..., 69:], output['ik1_base'][..., 69:].detach()),
        'ik1_gR2_dot': direction_delta_loss(output['ik1dot'], record['ik1_target'], 69, 72, dt),
        'ik1_gR2_ddot': smooth_ddot_loss(output['ik1ddot'], 69, 72),
        'ik1_dot': sequence_delta_loss(output['ik1dot'], record['ik1_target'], 0, 69, dt),
        'ik1_ddot': smooth_ddot_loss(output['ik1ddot'], 0, 69),
        'ik2_r6d': torch.nn.functional.smooth_l1_loss(
            output['ik2'], record['ik2_target'].to(output['ik2'].device, output['ik2'].dtype)
        ),
        'ik2_parent_relative': ik2_parent_relative_loss(output['ik2'], record['ik2_target']),
        'ik_joint_two_node_distance': joint_two_node_distance_loss(output['ik1'], record['ik1_target']),
        'ik2_dot': sequence_delta_loss(output['ik2dot'], record['ik2_target'], 0, 90, dt),
        'ik2_ddot': smooth_ddot_loss(output['ik2ddot'], 0, 90),
        'vr': torch.nn.functional.smooth_l1_loss(
            output['vr'], record['vr_target'].to(output['vr'].device, output['vr'].dtype)
        ),
        'vr_velocity': torch.nn.functional.smooth_l1_loss(
            output['vr'][..., :4], record['vr_target'].to(output['vr'].device, output['vr'].dtype)[..., :4]
        ),
        'vr_contact': torch.nn.functional.binary_cross_entropy_with_logits(
            output['vr'][..., 4:], record['vr_target'].to(output['vr'].device, output['vr'].dtype)[..., 4:].clamp(0.0, 1.0)
        ),
        'vr_dot': sequence_delta_loss(output['vrdot'], record['vr_target'], 0, 9, dt),
        'control_point_prior': output['control_point_prior'],
        'tail_update_prior': output['tail_delta_norm'],
    }
    total = output['pl'].new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * weight
    components = {key: value.detach() for key, value in losses.items()}
    for stage, diagnostics in output.get('control_diagnostics', {}).items():
        for name, value in diagnostics.items():
            if torch.is_tensor(value):
                components[f'{stage}_control_{name}'] = value.detach()
    components.update({
        'loss': total.detach(),
        'new_delta_norm': output['new_delta_norm'].detach(),
        'tail_delta_norm': output['tail_delta_norm'].detach(),
        'pl_residual_norm_mean': (output['pl'] - output['pl_base']).norm(dim=-1).mean().detach(),
        'ik1_residual_norm_mean': (output['ik1'] - output['ik1_base']).norm(dim=-1).mean().detach(),
        'ik2_residual_norm_mean': (output['ik2'] - output['ik2_base']).norm(dim=-1).mean().detach(),
        'vr_residual_norm_mean': (output['vr'] - output['vr_base']).norm(dim=-1).mean().detach(),
    })
    return total, components
