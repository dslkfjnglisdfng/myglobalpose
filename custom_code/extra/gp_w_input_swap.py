"""Validated causal RMB-derived world angular velocity for GP input swaps."""

from __future__ import annotations

import torch
import articulate as art


FPS = 60.0
DT = 1.0 / FPS
LAG = 2
EMA_BETA = 0.3


class CausalRMBWorldAngularVelocity:
    """RMB is R_M_B: body coordinates to model/world coordinates."""

    def __init__(self, lag=LAG, beta=EMA_BETA, dt=DT):
        self.lag, self.beta, self.dt = int(lag), float(beta), float(dt)
        if self.lag != 2 or self.beta != 0.3 or self.dt != 1.0 / 60.0:
            raise ValueError("This experiment is fixed to lag=2, beta=0.3, fps=60")
        self.reset()

    def reset(self):
        self.rmb_history = []
        self.previous_ema_w = None
        self.frame_count = 0

    def step(self, rmb_t):
        if self.frame_count < self.lag:
            w_hat = rmb_t.new_zeros(rmb_t.shape[:-2] + (3,))
        else:
            delta_r = rmb_t @ self.rmb_history[-self.lag].transpose(-1, -2)
            w_raw = art.math.rotation_matrix_to_axis_angle(delta_r.reshape(-1, 3, 3)).reshape(
                delta_r.shape[:-2] + (3,)
            ) / (self.lag * self.dt)
            w_hat = w_raw if self.frame_count == self.lag else (
                (1.0 - self.beta) * self.previous_ema_w + self.beta * w_raw
            )
            self.previous_ema_w = w_hat
        self.rmb_history.append(rmb_t)
        if len(self.rmb_history) > self.lag:
            self.rmb_history.pop(0)
        self.frame_count += 1
        return w_hat


def causal_w_sequence(rmb):
    state = CausalRMBWorldAngularVelocity()
    return torch.stack([state.step(frame) for frame in rmb]) if len(rmb) else rmb.new_zeros(rmb.shape[:-2] + (3,))
