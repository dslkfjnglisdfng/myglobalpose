import torch


class L4PhysicsAdapter:
    """
    Default-off adapter for routing L4 curve-state outputs into the existing
    GlobalPose physics interface.

    P1 is intentionally a no-op on numerics: the L4-refined pose is already the
    pose target used by GlobalPose physics. The older P2 mode directly blends
    the L4 generalized root translation derivative into the root velocity
    target. The q-state P2b mode keeps that signal bounded with a small blend,
    norm clamp, and optional stationary/contact gate.
    """

    VALID_MODES = ('original', 'l4_pip_v1', 'l4_pip_v2', 'l4_qstate_p1', 'l4_qstate_p2b')
    VALID_GATES = ('no_gate', 'stationary_gate')

    def __init__(
        self,
        mode='original',
        qdot_velocity_blend=0.5,
        qstate_alpha=0.1,
        qstate_max_delta=0.1,
        qstate_gate='no_gate',
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f'Unsupported physics mode {mode!r}; expected one of {self.VALID_MODES}.')
        if qdot_velocity_blend < 0.0 or qdot_velocity_blend > 1.0:
            raise ValueError(f'qdot_velocity_blend must be in [0, 1], got {qdot_velocity_blend}.')
        if qstate_alpha < 0.0 or qstate_alpha > 1.0:
            raise ValueError(f'qstate_alpha must be in [0, 1], got {qstate_alpha}.')
        if qstate_max_delta <= 0.0:
            raise ValueError(f'qstate_max_delta must be positive, got {qstate_max_delta}.')
        if qstate_gate not in self.VALID_GATES:
            raise ValueError(f'Unsupported qstate gate {qstate_gate!r}; expected one of {self.VALID_GATES}.')
        self.mode = mode
        self.qdot_velocity_blend = float(qdot_velocity_blend)
        self.qstate_alpha = float(qstate_alpha)
        self.qstate_max_delta = float(qstate_max_delta)
        self.qstate_gate = qstate_gate
        self.last_debug = {}

    @property
    def enabled(self):
        return self.mode != 'original'

    def _stationary_gate(self, l4_debug, device, dtype):
        if self.qstate_gate == 'no_gate':
            return torch.ones((), device=device, dtype=dtype)
        stationary_prob = l4_debug.get('stationary_prob')
        if not torch.is_tensor(stationary_prob):
            return torch.ones((), device=device, dtype=dtype)
        stationary_prob = stationary_prob.to(device=device, dtype=dtype)
        stationary_weight = (stationary_prob * 5.0 - 3.0).clamp(0.0, 1.0)
        return stationary_weight.max().clamp(0.0, 1.0)

    @staticmethod
    def _clamp_delta(delta, max_norm):
        norm = delta.norm()
        if float(norm.detach().cpu()) <= max_norm:
            return delta, norm
        return delta * (max_norm / norm.clamp_min(1e-8)), norm

    def adapt_root_velocity(self, velocity, l4_debug):
        if self.mode == 'original':
            self.last_debug = {
                'physics_mode': self.mode,
                'used_l4_pose_target': False,
                'used_l4_qdot_target': False,
            }
            return velocity

        q75_after = l4_debug.get('q75_after')
        qdot_after = l4_debug.get('qdot_after')
        qddot_after = l4_debug.get('qddot_after')
        self.last_debug = {
            'physics_mode': self.mode,
            'used_l4_pose_target': q75_after is not None,
            'used_l4_qdot_target': False,
            'used_l4_qddot_target': qddot_after is not None,
            'q75_target_shape': list(q75_after.shape) if torch.is_tensor(q75_after) else None,
            'qdot_target_shape': list(qdot_after.shape) if torch.is_tensor(qdot_after) else None,
            'qddot_target_shape': list(qddot_after.shape) if torch.is_tensor(qddot_after) else None,
            'q75_target_norm': float(q75_after.detach().norm().cpu()) if torch.is_tensor(q75_after) else 0.0,
            'qdot_target_norm': float(qdot_after.detach().norm().cpu()) if torch.is_tensor(qdot_after) else 0.0,
            'qddot_target_norm': float(qddot_after.detach().norm().cpu()) if torch.is_tensor(qddot_after) else 0.0,
            'vr_velocity_norm': float(velocity.detach().norm().cpu()),
        }

        if self.mode in ('l4_pip_v1', 'l4_qstate_p1') or qdot_after is None:
            return velocity

        qdot_root = qdot_after[:3].to(device=velocity.device, dtype=velocity.dtype)
        if self.mode == 'l4_pip_v2':
            adapted = (1.0 - self.qdot_velocity_blend) * velocity + self.qdot_velocity_blend * qdot_root
            velocity_delta = adapted - velocity
            self.last_debug.update({
                'used_l4_qdot_target': True,
                'qdot_velocity_blend': self.qdot_velocity_blend,
                'vr_velocity': velocity.detach().cpu().clone(),
                'l4_qdot_root': qdot_root.detach().cpu().clone(),
                'adapted_velocity': adapted.detach().cpu().clone(),
                'velocity_delta_norm': float(velocity_delta.detach().norm().cpu()),
                'qdot_root_norm': float(qdot_root.detach().norm().cpu()),
                'adapted_velocity_norm': float(adapted.detach().norm().cpu()),
            })
            return adapted

        delta = qdot_root - velocity
        delta_clamped, raw_delta_norm = self._clamp_delta(delta, self.qstate_max_delta)
        gate = self._stationary_gate(l4_debug, velocity.device, velocity.dtype)
        adapted = velocity + self.qstate_alpha * gate * delta_clamped
        velocity_delta = adapted - velocity
        self.last_debug.update({
            'used_l4_qdot_target': True,
            'qstate_alpha': self.qstate_alpha,
            'qstate_max_delta': self.qstate_max_delta,
            'qstate_gate': self.qstate_gate,
            'qstate_gate_value': float(gate.detach().cpu()),
            'vr_velocity': velocity.detach().cpu().clone(),
            'l4_qdot_root': qdot_root.detach().cpu().clone(),
            'adapted_velocity': adapted.detach().cpu().clone(),
            'raw_velocity_delta_norm': float(raw_delta_norm.detach().cpu()),
            'clamped_velocity_delta_norm': float(delta_clamped.detach().norm().cpu()),
            'velocity_delta_norm': float(velocity_delta.detach().norm().cpu()),
            'qdot_root_norm': float(qdot_root.detach().norm().cpu()),
            'adapted_velocity_norm': float(adapted.detach().norm().cpu()),
        })
        return adapted
