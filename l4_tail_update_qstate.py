import torch

from l4_q75_utils import pose_tran_to_q75, prephysics_feature, prephysics_feature_dim, q75_to_pose_tran


class UniformCubicBSpline(torch.nn.Module):
    def __init__(self, dt=1.0 / 60.0):
        super().__init__()
        self.dt = float(dt)

    def forward(self, control, return_derivatives=False):
        squeeze_batch = control.dim() == 2
        if squeeze_batch:
            control = control.unsqueeze(0)

        left = torch.cat((control[:, :1], control[:, :-1]), dim=1)
        right = torch.cat((control[:, 1:], control[:, -1:]), dim=1)
        q = (left + 4.0 * control + right) / 6.0
        if not return_derivatives:
            return q.squeeze(0) if squeeze_batch else q

        qdot = (right - left) / (2.0 * self.dt)
        qddot = (left - 2.0 * control + right) / (self.dt ** 2)
        if squeeze_batch:
            return q.squeeze(0), qdot.squeeze(0), qddot.squeeze(0)
        return q, qdot, qddot


class StreamingTailUpdateQState(torch.nn.Module):
    def __init__(
        self,
        n_input=None,
        state_dim=75,
        hidden_size=256,
        tail_update=4,
        residual_scale=0.02,
        velocity_residual_scale=0.02,
        freeze_root_translation=True,
        boundary_strategy='repeat',
        dt=1.0 / 60.0,
        pose_input_mode='euler_q75',
        euler_seq='XYZ',
        offset_conditioning='none',
        rnn_init_mode=None,
        offset_init_scale=0.1,
        dropout=0.0,
        imu_feature_dropout=0.0,
        acc_dropout=0.0,
        gyro_dropout=0.0,
        orientation_dropout=0.0,
    ):
        super().__init__()
        if tail_update != 4:
            raise ValueError('This migration keeps the approved L=4 tail-update contract.')
        if boundary_strategy not in ('repeat', 'linear_extrap'):
            raise ValueError(f'Unsupported boundary strategy: {boundary_strategy}')
        if pose_input_mode not in ('euler_q75', 'rot6d'):
            raise ValueError(f'Unsupported pose_input_mode: {pose_input_mode}')
        if offset_conditioning not in ('none', 'hidden_init'):
            raise ValueError(f'Unsupported offset_conditioning: {offset_conditioning}')
        if rnn_init_mode is None:
            rnn_init_mode = 'offset_only' if offset_conditioning == 'hidden_init' else 'none'
        if rnn_init_mode not in ('none', 'offset_only', 'offset_firstframe'):
            raise ValueError(f'Unsupported rnn_init_mode: {rnn_init_mode}')
        for name, value in (
            ('dropout', dropout),
            ('imu_feature_dropout', imu_feature_dropout),
            ('acc_dropout', acc_dropout),
            ('gyro_dropout', gyro_dropout),
            ('orientation_dropout', orientation_dropout),
        ):
            if value < 0.0 or value >= 1.0:
                raise ValueError(f'{name} must be in [0, 1), got {value}.')
        self.pose_input_mode = pose_input_mode
        self.euler_seq = euler_seq
        self.offset_conditioning = offset_conditioning
        self.rnn_init_mode = rnn_init_mode
        self.offset_init_scale = float(offset_init_scale)
        self.n_input = int(prephysics_feature_dim(pose_input_mode) if n_input is None else n_input)
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.tail_update = int(tail_update)
        self.residual_scale = float(residual_scale)
        self.velocity_residual_scale = float(velocity_residual_scale)
        self.freeze_root_translation = bool(freeze_root_translation)
        self.boundary_strategy = boundary_strategy
        self.dropout = float(dropout)
        self.imu_feature_dropout = float(imu_feature_dropout)
        self.acc_dropout = float(acc_dropout)
        self.gyro_dropout = float(gyro_dropout)
        self.orientation_dropout = float(orientation_dropout)
        self.pose_feature_dim = self.n_input - 90
        self.input = torch.nn.Linear(self.n_input, hidden_size)
        self.cell = torch.nn.GRUCell(hidden_size, hidden_size)
        self.new_control = torch.nn.Linear(hidden_size, state_dim)
        self.tail_delta = torch.nn.Linear(hidden_size, tail_update * state_dim)
        self.velocity_delta = torch.nn.Linear(hidden_size + tail_update * 3 + 5, 3)
        self.offset_encoder = None
        self.rnn_init_encoder = None
        if self.rnn_init_mode == 'offset_only':
            self.offset_encoder = torch.nn.Sequential(
                torch.nn.Linear(18, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
            )
        elif self.rnn_init_mode == 'offset_firstframe':
            self.rnn_init_encoder = torch.nn.Sequential(
                torch.nn.Linear(18 + self.n_input, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
            )
        self.spline = UniformCubicBSpline(dt)
        self.reset_stream()
        torch.nn.init.zeros_(self.new_control.weight)
        torch.nn.init.zeros_(self.new_control.bias)
        torch.nn.init.zeros_(self.tail_delta.weight)
        torch.nn.init.zeros_(self.tail_delta.bias)
        torch.nn.init.zeros_(self.velocity_delta.weight)
        torch.nn.init.zeros_(self.velocity_delta.bias)
        if self.offset_encoder is not None:
            torch.nn.init.zeros_(self.offset_encoder[-1].weight)
            torch.nn.init.zeros_(self.offset_encoder[-1].bias)
        if self.rnn_init_encoder is not None:
            torch.nn.init.zeros_(self.rnn_init_encoder[-1].weight)
            torch.nn.init.zeros_(self.rnn_init_encoder[-1].bias)

    def reset_stream(self, offset_r=None, init_feature=None):
        if self.rnn_init_mode == 'offset_firstframe' and offset_r is not None and init_feature is not None:
            self.hidden = self._firstframe_hidden(offset_r, init_feature)
        elif self.rnn_init_mode == 'offset_only' and offset_r is not None:
            self.hidden = self._offset_hidden(offset_r)
        else:
            self.hidden = None
        self.control_buffer = None
        self.base_buffer = None
        self.velocity_buffer = None

    def _initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _offset_hidden(self, offset_r):
        if self.offset_encoder is None:
            return None
        try:
            ref = next(self.offset_encoder.parameters())
            device, dtype = ref.device, ref.dtype
        except StopIteration:
            device, dtype = offset_r.device, offset_r.dtype
        x = offset_r.detach().to(device=device, dtype=dtype).reshape(1, -1)
        if x.shape[-1] != 18:
            raise ValueError(f'Expected offset_r flatten dim 18, got {x.shape[-1]}.')
        return self.offset_encoder(x) * self.offset_init_scale

    def _firstframe_hidden(self, offset_r, init_feature):
        if self.rnn_init_encoder is None:
            return None
        try:
            ref = next(self.rnn_init_encoder.parameters())
            device, dtype = ref.device, ref.dtype
        except StopIteration:
            device, dtype = init_feature.device, init_feature.dtype
        offset = offset_r.detach().to(device=device, dtype=dtype).reshape(1, -1)
        feature = init_feature.detach().to(device=device, dtype=dtype).reshape(1, -1)
        if offset.shape[-1] != 18:
            raise ValueError(f'Expected offset_r flatten dim 18, got {offset.shape[-1]}.')
        if feature.shape[-1] != self.n_input:
            raise ValueError(f'Expected first-frame feature dim {self.n_input}, got {feature.shape[-1]}.')
        feature = self._apply_feature_dropout(feature)
        return self.rnn_init_encoder(torch.cat((offset, feature), dim=-1)) * self.offset_init_scale

    def _dropout_slice(self, feature, start, end, p):
        if p <= 0.0:
            return feature
        feature = feature.clone()
        feature[..., start:end] = torch.nn.functional.dropout(feature[..., start:end], p=p, training=True)
        return feature

    def _apply_feature_dropout(self, feature):
        if not self.training:
            return feature
        if self.dropout > 0.0:
            feature = torch.nn.functional.dropout(feature, p=self.dropout, training=True)
        imu_start = self.pose_feature_dim
        if self.imu_feature_dropout > 0.0:
            feature = self._dropout_slice(feature, imu_start, self.n_input, self.imu_feature_dropout)
        if self.acc_dropout > 0.0:
            feature = self._dropout_slice(feature, imu_start, imu_start + 18, self.acc_dropout)
        if self.gyro_dropout > 0.0:
            feature = self._dropout_slice(feature, imu_start + 18, imu_start + 36, self.gyro_dropout)
        if self.orientation_dropout > 0.0:
            feature = self._dropout_slice(feature, imu_start + 36, imu_start + 90, self.orientation_dropout)
        return feature

    def _ghost(self, buffer):
        last = buffer[:, -1:]
        if self.boundary_strategy == 'repeat' or buffer.shape[1] < 2:
            return last
        return last + (last - buffer[:, -2:-1])

    def _freeze_root(self, delta):
        if self.freeze_root_translation and self.state_dim >= 3:
            delta = delta.clone()
            delta[..., :3] = 0.0
        return delta

    def step(self, feature_t, base_q_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_q_t.dim() == 1:
            base_q_t = base_q_t.unsqueeze(0)
        if feature_t.shape[-1] != self.n_input:
            raise ValueError(f'Expected feature dim {self.n_input}, got {feature_t.shape[-1]}.')

        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)

        feature_t = self._apply_feature_dropout(feature_t)
        z = torch.relu(self.input(feature_t))
        self.hidden = self.cell(z, self.hidden)
        new_delta = self._freeze_root(self.new_control(self.hidden) * self.residual_scale)
        new_control = base_q_t + new_delta

        if self.control_buffer is None:
            self.control_buffer = new_control.unsqueeze(1)
            self.base_buffer = base_q_t.unsqueeze(1)
            tail_delta_norm = new_delta.new_tensor(0.0)
            updated_count = 1
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
            )
            tail_delta = self._freeze_root(tail_delta[:, -update_count:] * self.residual_scale)
            tail_control = tail_control + tail_delta
            self.control_buffer = torch.cat((old_control, tail_control, new_control.unsqueeze(1)), dim=1)
            self.base_buffer = torch.cat((old_base, tail_base, base_q_t.unsqueeze(1)), dim=1)
            tail_delta_norm = tail_delta.norm(dim=-1).mean()
            updated_count = update_count + 1

        control_decode = torch.cat((self.control_buffer, self._ghost(self.control_buffer)), dim=1)
        base_decode = torch.cat((self.base_buffer, self._ghost(self.base_buffer)), dim=1)
        q_control, qdot_control, qddot_control = self.spline(control_decode, return_derivatives=True)
        q_base, qdot_base, qddot_base = self.spline(base_decode, return_derivatives=True)
        current_index = self.control_buffer.shape[1] - 1
        residual_t = q_control[:, current_index] - q_base[:, current_index]
        residual_t = self._freeze_root(residual_t)
        q_t = base_q_t + residual_t

        return {
            'q_t': q_t,
            'qdot_t': qdot_control[:, current_index],
            'qddot_t': qddot_control[:, current_index],
            'residual_t': residual_t,
            'new_delta_norm': new_delta.norm(dim=-1).mean(),
            'tail_delta_norm': tail_delta_norm,
            'buffer_length': self.control_buffer.shape[1],
            'updated_control_count': updated_count,
            'frozen_control_count': max(0, self.control_buffer.shape[1] - updated_count),
            'uses_future_frames': False,
        }

    def refine_velocity(self, v_root_vr, stationary_prob=None):
        if v_root_vr.dim() == 1:
            v_root_vr = v_root_vr.unsqueeze(0)
        if self.hidden is None or self.hidden.shape[0] != v_root_vr.shape[0]:
            self.hidden = self._initial_hidden(v_root_vr.shape[0], v_root_vr.device, v_root_vr.dtype)
        if stationary_prob is None:
            stationary_prob = torch.zeros(v_root_vr.shape[0], 5, device=v_root_vr.device, dtype=v_root_vr.dtype)
        elif stationary_prob.dim() == 1:
            stationary_prob = stationary_prob.unsqueeze(0)

        if self.velocity_buffer is None:
            self.velocity_buffer = v_root_vr.unsqueeze(1)
        else:
            self.velocity_buffer = torch.cat((self.velocity_buffer.detach(), v_root_vr.unsqueeze(1)), dim=1)
            if self.velocity_buffer.shape[1] > self.tail_update:
                self.velocity_buffer = self.velocity_buffer[:, -self.tail_update:]

        if self.velocity_buffer.shape[1] < self.tail_update:
            pad = self.velocity_buffer[:, :1].expand(-1, self.tail_update - self.velocity_buffer.shape[1], -1)
            velocity_tail = torch.cat((pad, self.velocity_buffer), dim=1)
        else:
            velocity_tail = self.velocity_buffer

        velocity_feature = torch.cat((self.hidden, velocity_tail.reshape(v_root_vr.shape[0], -1), stationary_prob), dim=-1)
        delta_v = self.velocity_delta(velocity_feature) * self.velocity_residual_scale
        return {
            'v_root_refined': v_root_vr + delta_v,
            'delta_v_root': delta_v,
            'delta_v_root_norm': delta_v.norm(dim=-1).mean(),
            'velocity_history_length': self.velocity_buffer.shape[1],
        }


class L4PrePhysicsRefiner(torch.nn.Module):
    def __init__(self, qstate_module=None, euler_seq='XYZ', residual_epsilon=0.0):
        super().__init__()
        self.qstate_module = qstate_module
        self.euler_seq = euler_seq
        self.residual_epsilon = float(residual_epsilon)
        self.last_debug = {}

    def reset(self):
        self.last_debug = {}
        if hasattr(self.qstate_module, 'reset_stream'):
            self.qstate_module.reset_stream()

    @torch.no_grad()
    def refine(self, pose, prephysics_tran, a, w, R):
        q75 = pose_tran_to_q75(
            pose.detach().cpu().view(1, 24, 3, 3),
            prephysics_tran.detach().cpu().view(1, 3),
            euler_seq=self.euler_seq,
        )[0]
        if self.qstate_module is None:
            self.last_debug = {
                'q75_before': q75.detach().clone(),
                'q75_after': q75.detach().clone(),
                'pose_before': pose.detach().cpu().clone(),
                'residual': torch.zeros_like(q75),
                'residual_norm': 0.0,
                'changed': False,
                'tail_update': 4,
                'uses_future_frames': False,
                'delta_v_root': torch.zeros(3),
                'delta_v_root_norm': 0.0,
                'velocity_changed': False,
            }
            return pose, prephysics_tran, False

        try:
            module_device = next(self.qstate_module.parameters()).device
        except StopIteration:
            module_device = q75.device
        feature = prephysics_feature(
            q75,
            a.detach().cpu(),
            w.detach().cpu(),
            R.detach().cpu(),
            pose=pose.detach().cpu(),
            pose_input_mode=getattr(self.qstate_module, 'pose_input_mode', 'euler_q75'),
            euler_seq=self.euler_seq,
        ).to(module_device)
        result = self.qstate_module.step(feature, q75.to(module_device))
        q_refined = result['q_t'][0].detach().cpu()
        residual = q_refined - q75
        changed = float(residual.norm()) > self.residual_epsilon
        self.last_debug = {
            'q75_before': q75.detach().clone(),
            'q75_after': q_refined.detach().clone(),
            'pose_before': pose.detach().cpu().clone(),
            'residual': residual.detach().clone(),
            'residual_norm': float(residual.norm()),
            'changed': changed,
            'tail_update': 4,
            'uses_future_frames': bool(result.get('uses_future_frames', False)),
            'updated_control_count': int(result.get('updated_control_count', 0)),
            'frozen_control_count': int(result.get('frozen_control_count', 0)),
            'new_delta_norm': float(result.get('new_delta_norm', 0.0)),
            'tail_delta_norm': float(result.get('tail_delta_norm', 0.0)),
        }
        if not changed:
            return pose, prephysics_tran, False

        pose_refined, tran_refined = q75_to_pose_tran(q_refined.view(1, 75), euler_seq=self.euler_seq)
        return pose_refined[0].to(dtype=pose.dtype), tran_refined[0].to(dtype=prephysics_tran.dtype), True

    @torch.no_grad()
    def refine_velocity(self, v_root_vr, stationary_prob=None):
        if self.qstate_module is None:
            self.last_debug.update({
                'v_root_vr': v_root_vr.detach().cpu().clone(),
                'v_root_refined': v_root_vr.detach().cpu().clone(),
                'delta_v_root': torch.zeros_like(v_root_vr.detach().cpu()),
                'delta_v_root_norm': 0.0,
                'velocity_changed': False,
            })
            return v_root_vr, False

        try:
            module_device = next(self.qstate_module.parameters()).device
        except StopIteration:
            module_device = v_root_vr.device
        result = self.qstate_module.refine_velocity(
            v_root_vr.detach().to(module_device),
            None if stationary_prob is None else stationary_prob.detach().to(module_device),
        )
        v_refined = result['v_root_refined'][0].detach().cpu()
        delta_v = v_refined - v_root_vr.detach().cpu()
        changed = float(delta_v.norm()) > self.residual_epsilon
        self.last_debug.update({
            'v_root_vr': v_root_vr.detach().cpu().clone(),
            'v_root_refined': v_refined.detach().clone(),
            'delta_v_root': delta_v.detach().clone(),
            'delta_v_root_norm': float(delta_v.norm()),
            'velocity_changed': changed,
            'velocity_history_length': int(result.get('velocity_history_length', 0)),
        })
        return v_refined.to(dtype=v_root_vr.dtype), changed
