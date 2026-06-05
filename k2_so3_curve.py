import torch

import articulate as art
from l4_q75_utils import prephysics_feature_dim, q75_to_pose_tran
from l4_tail_update_qstate import UniformCubicBSpline


def _wrap_angle_delta(delta):
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _clamp_rotvec(rotvec, max_norm=3.10):
    norm = rotvec.norm(dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / norm.clamp_min(1e-8), max=1.0)
    return rotvec * scale


def rotvec_to_rotation_matrix(rotvec):
    orig_shape = rotvec.shape[:-1]
    rv = rotvec.reshape(-1, 3)
    theta2 = (rv * rv).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-12))
    x, y, z = rv.unbind(dim=-1)
    zero = torch.zeros_like(x)
    K = torch.stack((
        zero, -z, y,
        z, zero, -x,
        -y, x, zero,
    ), dim=-1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=rv.device, dtype=rv.dtype).expand(rv.shape[0], 3, 3)
    small = theta2 < 1e-8
    A = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, torch.sin(theta) / theta)
    B = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12))
    R = eye + A.reshape(-1, 1, 1) * K + B.reshape(-1, 1, 1) * K.matmul(K)
    return R.reshape(orig_shape + (3, 3))


def pose_tran_to_so3_state(pose, tran, max_rotvec_norm=3.10):
    rotvec = art.math.rotation_matrix_to_axis_angle(pose.reshape(-1, 3, 3))
    rotvec = _clamp_rotvec(rotvec, max_norm=max_rotvec_norm)
    return torch.cat((tran.reshape(-1, 3), rotvec.reshape(pose.shape[0], 72)), dim=-1)


def so3_state_to_pose_tran(q_so3, max_rotvec_norm=3.10):
    tran = q_so3[..., :3].reshape(-1, 3)
    rotvec = _clamp_rotvec(q_so3[..., 3:].reshape(-1, 3), max_norm=max_rotvec_norm)
    pose = rotvec_to_rotation_matrix(rotvec).reshape(-1, 24, 3, 3)
    return pose, tran


def q75_to_so3_state(q75, euler_seq='XYZ', max_rotvec_norm=3.10):
    pose, tran = q75_to_pose_tran(q75.reshape(-1, 75), euler_seq=euler_seq)
    return pose_tran_to_so3_state(pose, tran, max_rotvec_norm=max_rotvec_norm)


def so3_state_to_euler_q75(q_so3, euler_seq='XYZ', max_rotvec_norm=3.10):
    pose, tran = so3_state_to_pose_tran(q_so3.reshape(-1, 75), max_rotvec_norm=max_rotvec_norm)
    euler = art.math.rotation_matrix_to_euler_angle(pose.reshape(-1, 3, 3), seq=euler_seq)
    return torch.cat((tran.reshape(-1, 3), euler.reshape(-1, 72)), dim=-1)


def central_euler_derivatives(q_euler, dt):
    left = torch.cat((q_euler[:, :1], q_euler[:, :-1]), dim=1)
    right = torch.cat((q_euler[:, 1:], q_euler[:, -1:]), dim=1)
    step = right - left
    step_rot = _wrap_angle_delta(step[..., 3:])
    qdot = torch.cat((step[..., :3], step_rot), dim=-1) / (2.0 * dt)
    second = left - 2.0 * q_euler + right
    second_rot = _wrap_angle_delta(left[..., 3:] - q_euler[..., 3:]) + _wrap_angle_delta(right[..., 3:] - q_euler[..., 3:])
    qddot = torch.cat((second[..., :3], second_rot), dim=-1) / (dt ** 2)
    return qdot, qddot


class SO3CurveStateDecoder(torch.nn.Module):
    def __init__(self, dt=1.0 / 60.0, euler_seq='XYZ', max_rotvec_norm=3.10):
        super().__init__()
        self.dt = float(dt)
        self.euler_seq = euler_seq
        self.max_rotvec_norm = float(max_rotvec_norm)
        self.spline = UniformCubicBSpline(dt)

    def forward(self, control, return_derivatives=True):
        q_so3, qdot_so3, qddot_so3 = self.spline(control, return_derivatives=True)
        q_so3 = q_so3.clone()
        q_so3[..., 3:] = _clamp_rotvec(q_so3[..., 3:].reshape(-1, 3), self.max_rotvec_norm).reshape_as(q_so3[..., 3:])
        pose, tran = so3_state_to_pose_tran(q_so3.reshape(-1, 75), self.max_rotvec_norm)
        pose = pose.reshape(q_so3.shape[:-1] + (24, 3, 3))
        tran = tran.reshape(q_so3.shape[:-1] + (3,))
        euler_q75 = so3_state_to_euler_q75(q_so3.reshape(-1, 75), self.euler_seq, self.max_rotvec_norm).reshape_as(q_so3)
        qdot_euler, qddot_euler = central_euler_derivatives(euler_q75, self.dt)
        angular_velocity, angular_acceleration = self._angular_motion(pose)
        result = {
            'q_so3': q_so3,
            'qdot_so3': qdot_so3,
            'qddot_so3': qddot_so3,
            'pose_R': pose,
            'tran': tran,
            'euler_q75': euler_q75,
            'euler_qdot': qdot_euler,
            'euler_qddot': qddot_euler,
            'angular_velocity': angular_velocity,
            'angular_acceleration': angular_acceleration,
        }
        if not return_derivatives:
            return euler_q75
        return result

    def _angular_motion(self, pose):
        if pose.shape[1] < 2:
            omega = pose.new_zeros(pose.shape[:2] + (24, 3))
            alpha = pose.new_zeros(pose.shape[:2] + (24, 3))
            return omega, alpha
        rel = pose[:, 1:].transpose(-1, -2).matmul(pose[:, :-1])
        step = art.math.rotation_matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(rel.shape[:-2] + (3,))
        omega_step = -step / self.dt
        first = omega_step[:, :1]
        omega = torch.cat((first, omega_step), dim=1)
        if omega.shape[1] < 2:
            alpha = torch.zeros_like(omega)
        else:
            alpha_step = (omega[:, 1:] - omega[:, :-1]) / self.dt
            alpha = torch.cat((alpha_step[:, :1], alpha_step), dim=1)
        return omega, alpha


class StreamingTailUpdateSO3State(torch.nn.Module):
    def __init__(
        self,
        n_input=None,
        state_dim=75,
        hidden_size=256,
        tail_update=4,
        residual_scale=0.005,
        velocity_residual_scale=0.0,
        freeze_root_translation=True,
        boundary_strategy='repeat',
        dt=1.0 / 60.0,
        pose_input_mode='rot6d',
        euler_seq='XYZ',
        offset_conditioning='hidden_init',
        rnn_init_mode='offset_firstframe',
        offset_init_scale=0.1,
        dropout=0.0,
        imu_feature_dropout=0.0,
        acc_dropout=0.0,
        gyro_dropout=0.0,
        orientation_dropout=0.0,
        max_rotvec_norm=3.10,
    ):
        super().__init__()
        if tail_update != 4:
            raise ValueError('K2_SO3Curve_v1 keeps the approved L=4 tail-update contract.')
        if state_dim != 75:
            raise ValueError('K2_SO3Curve_v1 state is [root translation 3D, 24 rotvecs 72D].')
        if pose_input_mode not in ('euler_q75', 'rot6d'):
            raise ValueError(f'Unsupported pose_input_mode: {pose_input_mode}')
        if offset_conditioning not in ('none', 'hidden_init'):
            raise ValueError(f'Unsupported offset_conditioning: {offset_conditioning}')
        if rnn_init_mode not in ('none', 'offset_only', 'offset_firstframe'):
            raise ValueError(f'Unsupported rnn_init_mode: {rnn_init_mode}')
        self.model_type = 'k2_so3curve_v1'
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
        self.decoder = SO3CurveStateDecoder(dt=dt, euler_seq=euler_seq, max_rotvec_norm=max_rotvec_norm)
        self.spline = self.decoder.spline
        self.reset_stream()
        self._zero_residual_heads()

    def _zero_residual_heads(self):
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
        ref = next(self.offset_encoder.parameters())
        x = offset_r.detach().to(device=ref.device, dtype=ref.dtype)
        x = x.reshape(x.shape[0], -1) if x.dim() >= 3 else x.reshape(1, -1)
        if x.shape[-1] != 18:
            raise ValueError(f'Expected offset_r flatten dim 18, got {x.shape[-1]}.')
        return self.offset_encoder(x) * self.offset_init_scale

    def _firstframe_hidden(self, offset_r, init_feature):
        if self.rnn_init_encoder is None:
            return None
        ref = next(self.rnn_init_encoder.parameters())
        offset = offset_r.detach().to(device=ref.device, dtype=ref.dtype)
        feature = init_feature.detach().to(device=ref.device, dtype=ref.dtype)
        offset = offset.reshape(offset.shape[0], -1) if offset.dim() >= 3 else offset.reshape(1, -1)
        feature = feature.reshape(feature.shape[0], -1) if feature.dim() >= 2 else feature.reshape(1, -1)
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

    def _decode_current(self, buffer, return_euler=True):
        current = buffer[:, -1]
        if buffer.shape[1] > 1:
            prev = buffer[:, -2]
        else:
            prev = current
        right = current
        q_so3 = (prev + 4.0 * current + right) / 6.0
        qdot_so3 = (right - prev) / (2.0 * self.decoder.dt)
        qddot_so3 = (prev - 2.0 * current + right) / (self.decoder.dt ** 2)
        q_rot = _clamp_rotvec(
            q_so3[..., 3:].reshape(-1, 3),
            self.decoder.max_rotvec_norm,
        ).reshape_as(q_so3[..., 3:])
        q_so3 = torch.cat((q_so3[..., :3], q_rot), dim=-1)
        pose, _ = so3_state_to_pose_tran(q_so3, max_rotvec_norm=self.decoder.max_rotvec_norm)
        if buffer.shape[1] > 1:
            pose_prev, _ = so3_state_to_pose_tran(prev, max_rotvec_norm=self.decoder.max_rotvec_norm)
            rel = pose.transpose(-1, -2).matmul(pose_prev)
            angular_velocity = -art.math.rotation_matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(buffer.shape[0], 24, 3) / self.decoder.dt
        else:
            angular_velocity = torch.zeros(buffer.shape[0], 24, 3, device=buffer.device, dtype=buffer.dtype)
        angular_acceleration = torch.zeros_like(angular_velocity)
        result = {
            'q_so3': q_so3,
            'qdot_so3': qdot_so3,
            'qddot_so3': qddot_so3,
            'pose_R': pose.reshape(buffer.shape[0], 24, 3, 3),
            'angular_velocity': angular_velocity,
            'angular_acceleration': angular_acceleration,
        }
        if return_euler:
            euler_q75 = so3_state_to_euler_q75(
                q_so3,
                euler_seq=self.euler_seq,
                max_rotvec_norm=self.decoder.max_rotvec_norm,
            )
            if buffer.shape[1] > 1:
                prev_q_so3 = (buffer[:, -3] + 4.0 * prev + current) / 6.0 if buffer.shape[1] > 2 else prev
                euler_prev = so3_state_to_euler_q75(
                    prev_q_so3,
                    euler_seq=self.euler_seq,
                    max_rotvec_norm=self.decoder.max_rotvec_norm,
                )
                euler_step = euler_q75 - euler_prev
                euler_qdot = torch.cat((
                    euler_step[..., :3],
                    _wrap_angle_delta(euler_step[..., 3:]),
                ), dim=-1) / self.decoder.dt
            else:
                euler_qdot = torch.zeros_like(euler_q75)
            result.update({
                'euler_q75': euler_q75,
                'euler_qdot': euler_qdot,
                'euler_qddot': torch.zeros_like(euler_q75),
            })
        return result

    def step(self, feature_t, base_q_t, base_so3_t=None, return_euler=True):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_q_t.dim() == 1:
            base_q_t = base_q_t.unsqueeze(0)
        if feature_t.shape[-1] != self.n_input:
            raise ValueError(f'Expected feature dim {self.n_input}, got {feature_t.shape[-1]}.')
        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)

        if base_so3_t is None and hasattr(self, 'current_base_so3_t'):
            base_so3_t = self.current_base_so3_t
        if base_so3_t is None:
            base_so3_t = q75_to_so3_state(base_q_t.detach().to(feature_t.device), euler_seq=self.euler_seq)
        else:
            if base_so3_t.dim() == 1:
                base_so3_t = base_so3_t.unsqueeze(0)
            base_so3_t = base_so3_t.detach().to(feature_t.device)
        feature_t = self._apply_feature_dropout(feature_t)
        z = torch.relu(self.input(feature_t))
        self.hidden = self.cell(z, self.hidden)
        new_delta = self._freeze_root(self.new_control(self.hidden) * self.residual_scale)
        new_control = base_so3_t + new_delta

        if self.control_buffer is None:
            self.control_buffer = new_control.unsqueeze(1)
            self.base_buffer = base_so3_t.unsqueeze(1)
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
            self.base_buffer = torch.cat((old_base, tail_base, base_so3_t.unsqueeze(1)), dim=1)
            tail_delta_norm = tail_delta.norm(dim=-1).mean()
            updated_count = update_count + 1

        decoded = self._decode_current(self.control_buffer, return_euler=return_euler)
        decoded_base = self._decode_current(self.base_buffer, return_euler=return_euler)
        if return_euler:
            q_t = decoded['euler_q75']
            residual_t = q_t - base_q_t
            residual_t = torch.cat((residual_t[..., :3], _wrap_angle_delta(residual_t[..., 3:])), dim=-1)
            qdot_t = decoded['euler_qdot']
            qddot_t = decoded['euler_qddot']
        else:
            q_t = decoded['q_so3']
            residual_t = q_t - base_so3_t
            qdot_t = decoded['qdot_so3']
            qddot_t = decoded['qddot_so3']
        residual_t = self._freeze_root(residual_t)
        control_residual = self.control_buffer - self.base_buffer
        rotvec_norm = decoded['q_so3'][..., 3:].reshape(-1, 3).norm(dim=-1)

        return {
            'q_t': q_t,
            'qdot_t': qdot_t,
            'qddot_t': qddot_t,
            'q_so3_t': decoded['q_so3'],
            'qdot_so3_t': decoded['qdot_so3'],
            'qddot_so3_t': decoded['qddot_so3'],
            'pose_R_t': decoded['pose_R'],
            'angular_velocity_t': decoded['angular_velocity'],
            'angular_acceleration_t': decoded['angular_acceleration'],
            'residual_t': residual_t,
            'control_point_prior_t': control_residual.square().mean(),
            'so3_control_point_prior_t': control_residual.square().mean(),
            'rotvec_norm_mean': rotvec_norm.mean(),
            'rotvec_norm_max': rotvec_norm.max(),
            'angular_velocity_norm_mean': decoded['angular_velocity'].norm(dim=-1).mean(),
            'angular_acceleration_norm_mean': decoded['angular_acceleration'].norm(dim=-1).mean(),
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
