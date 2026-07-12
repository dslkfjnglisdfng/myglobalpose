"""PL-VA-State-V1: causal 102D PL features and explicit p/v/a state rollout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import articulate as art


FPS = 60.0
DT = 1.0 / FPS
INPUT_SIZE = 102
RAW_OUTPUT_SIZE = 33
LEGACY_OUTPUT_SIZE = 18
LEAF_NAMES = ("left_forearm", "right_forearm", "left_lower_leg", "right_lower_leg", "head")
ANGULAR_VELOCITY_METHOD = "causal_world_so3_backward_lag2_ema03"
ANGULAR_VELOCITY_FRAME = "world_then_root"
ANGULAR_VELOCITY_LAG = 2
ANGULAR_VELOCITY_EMA_BETA = 0.3


class CausalRMBWorldAngularVelocityEMA:
    """Strictly causal world-frame SO(3) lag difference followed by EMA.

    RMB is R_M_B (body/sensor coordinates to model/world coordinates).
    """

    def __init__(self, lag=ANGULAR_VELOCITY_LAG, beta=ANGULAR_VELOCITY_EMA_BETA, dt=DT):
        if int(lag) < 1 or not 0.0 < float(beta) <= 1.0 or float(dt) <= 0.0:
            raise ValueError("lag >= 1, 0 < beta <= 1, and dt > 0 are required")
        self.lag = int(lag)
        self.beta = float(beta)
        self.dt = float(dt)
        self.reset()

    def reset(self):
        self.rmb_history = []
        self.previous_filtered_w = None
        self.num_frames_seen = 0

    def step(self, rmb_t):
        if len(self.rmb_history) < self.lag:
            filtered = rmb_t.new_zeros(rmb_t.shape[:-2] + (3,))
        else:
            delta_r_m = rmb_t.matmul(self.rmb_history[-self.lag].transpose(-1, -2))
            raw = art.math.rotation_matrix_to_axis_angle(delta_r_m.reshape(-1, 3, 3)).reshape(
                delta_r_m.shape[:-2] + (3,)
            ) / (self.lag * self.dt)
            filtered = raw if self.previous_filtered_w is None else (
                (1.0 - self.beta) * self.previous_filtered_w + self.beta * raw
            )
            self.previous_filtered_w = filtered
        self.rmb_history.append(rmb_t)
        if len(self.rmb_history) > self.lag:
            self.rmb_history.pop(0)
        self.num_frames_seen += 1
        return filtered


def causal_world_angular_velocity_from_rmb_sequence(
        rmb, lag=ANGULAR_VELOCITY_LAG, beta=ANGULAR_VELOCITY_EMA_BETA, dt=DT):
    state = CausalRMBWorldAngularVelocityEMA(lag=lag, beta=beta, dt=dt)
    if rmb.shape[0] == 0:
        return rmb.new_zeros(rmb.shape[:-2] + (3,))
    return torch.stack([state.step(frame) for frame in rmb])


def world_angular_velocity_to_root_frame(w_m, rmb_root):
    """Apply original GlobalPose row-vector convention: wRB = wM @ RMB_root."""
    return w_m.unsqueeze(-2).matmul(rmb_root).squeeze(-2)


def legacy_lag1_body_angular_velocity_from_rmb_sequence(rmb, dt=DT):
    """Previous PL-VA implementation retained only for diagnostics."""
    omega = torch.zeros(rmb.shape[:-2] + (3,), device=rmb.device, dtype=rmb.dtype)
    if rmb.shape[0] > 1:
        rel = rmb[1:].transpose(-1, -2).matmul(rmb[:-1])
        rv = art.math.rotation_matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(rel.shape[:-2] + (3,))
        omega[1:] = -rv / float(dt)
    return omega


def legacy_body_omega_to_world_frame(omega_body, rmb):
    return rmb.matmul(omega_body.unsqueeze(-1)).squeeze(-1)


class CausalButterworthLowpass:
    """Torch streaming SOS filter with scipy-compatible first-sample state."""

    def __init__(self, shape, fs=FPS, cutoff_hz=4.0, order=2, device=None, dtype=torch.float32):
        from scipy.signal import butter, sosfilt_zi

        sos = torch.as_tensor(butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos"), device=device, dtype=dtype)
        zi = torch.as_tensor(sosfilt_zi(sos.cpu().double().numpy()), device=device, dtype=dtype)
        self.sos = sos
        self.zi_base = zi
        self.shape = tuple(shape)
        self.state = None

    def reset(self):
        self.state = None

    def step(self, x):
        flat = x.reshape(-1)
        if self.state is None:
            self.state = self.zi_base[:, :, None] * flat[None, None, :]
        y = flat
        next_state = []
        for section, z in zip(self.sos, self.state):
            b0, b1, b2, a0, a1, a2 = section
            out = b0 / a0 * y + z[0]
            z0 = b1 / a0 * y - a1 / a0 * out + z[1]
            z1 = b2 / a0 * y - a2 / a0 * out
            next_state.append(torch.stack((z0, z1)))
            y = out
        self.state = torch.stack(next_state)
        return y.reshape(self.shape)


def causal_butterworth_lowpass_sequence(x, fs=FPS, cutoff_hz=4.0, order=2):
    if x.shape[0] == 0:
        return x
    filt = CausalButterworthLowpass(x.shape[1:], fs, cutoff_hz, order, x.device, x.dtype)
    return torch.stack([filt.step(frame) for frame in x])


def pl_va_feature_step(a_m_raw, rmb_t, angular_velocity_state, acc_filter):
    w_m = angular_velocity_state.step(rmb_t)
    root = rmb_t[-1]
    a_smooth = acc_filter.step(a_m_raw)
    a_rb_raw = a_m_raw.matmul(root)
    a_rb_smooth = a_smooth.matmul(root)
    w_rb = world_angular_velocity_to_root_frame(w_m, root)
    r_rb = root.transpose(-1, -2).matmul(rmb_t[:-1])
    g_r0 = -root[:, 1]
    feature = torch.cat((a_rb_raw.reshape(-1), w_rb.reshape(-1), r_rb.reshape(-1), g_r0, a_rb_smooth.reshape(-1)))
    if feature.numel() != INPUT_SIZE:
        raise RuntimeError(f"PL-VA feature must be {INPUT_SIZE}D, got {feature.numel()}")
    return feature


def pl_va_feature_sequence(a_m_raw, rmb, dt=DT, fs=FPS, cutoff_hz=4.0, order=2):
    smooth = causal_butterworth_lowpass_sequence(a_m_raw, fs, cutoff_hz, order)
    w_m = causal_world_angular_velocity_from_rmb_sequence(rmb, dt=dt)
    root = rmb[:, -1]
    a_raw = a_m_raw.matmul(root)
    a_smooth = smooth.matmul(root)
    w_rb = world_angular_velocity_to_root_frame(w_m, root[:, None])
    r_rb = root.transpose(-1, -2)[:, None].matmul(rmb[:, :-1])
    g_r0 = -root[:, :, 1]
    return torch.cat((a_raw.flatten(1), w_rb.flatten(1), r_rb.flatten(1), g_r0, a_smooth.flatten(1)), dim=-1)


@dataclass
class PLVAState:
    p: torch.Tensor
    v: torch.Tensor
    a_prev: torch.Tensor
    hc: tuple[torch.Tensor, torch.Tensor] | None = None

    def detach(self):
        hc = None if self.hc is None else tuple(x.detach() for x in self.hc)
        return PLVAState(self.p.detach(), self.v.detach(), self.a_prev.detach(), hc)


class PLVAStateV1(torch.nn.Module):
    def __init__(self, beta=0.7, dt=DT, cutoff_hz=4.0, filter_order=2):
        super().__init__()
        from articulate.utils.torch import RNNWithInit

        self.net = RNNWithInit(input_linear=False, input_size=INPUT_SIZE, output_size=RAW_OUTPUT_SIZE,
                               hidden_size=512, num_rnn_layer=3, dropout=0.4, init_size=18)
        self.beta = float(beta)
        self.dt = float(dt)
        self.cutoff_hz = float(cutoff_hz)
        self.filter_order = int(filter_order)
        self.stream_state = None
        self.prev_rmb = None
        self.acc_filter = None
        self.last_debug = {}

    def initial_state(self, p_rl, g_r0=None):
        p = p_rl.reshape(-1, 15)
        return PLVAState(p, torch.zeros_like(p), torch.zeros_like(p))

    def _init_hc(self, init_legacy):
        n, h = self.net.num_layers, self.net.hidden_size
        hc = self.net.init_net(init_legacy).view(-1, 2, n, h).permute(1, 2, 0, 3)
        return hc[0].contiguous(), hc[1].contiguous()

    def _update(self, raw, state, valid=None):
        v_direct, a_t, g_raw = raw.split((15, 15, 3), dim=-1)
        g = art.math.normalize_tensor(g_raw, avoid_nan=True)
        v_from_acc = state.v + 0.5 * self.dt * (state.a_prev + a_t)
        v = self.beta * v_direct + (1.0 - self.beta) * v_from_acc
        p = state.p + 0.5 * self.dt * (state.v + v)
        if valid is not None:
            m = valid.reshape(-1, 1)
            p, v, a_t = torch.where(m, p, state.p), torch.where(m, v, state.v), torch.where(m, a_t, state.a_prev)
        new_state = PLVAState(p, v, a_t, state.hc)
        debug = {"vRB_direct": v_direct, "vRB_from_acc": v_from_acc, "vRB_state": v,
                 "aRB_leaf": a_t, "pRB_state": p, "gR1": g,
                 "integration_residual": v - v_from_acc, "beta": self.beta}
        return torch.cat((p, g), dim=-1), new_state, debug

    def forward_sequence(self, features, init_legacy, lengths=None, state=None, detach_chunks=False, chunk_size=0):
        if features.dim() == 2:
            features, init_legacy, squeeze = features.unsqueeze(0), init_legacy.unsqueeze(0), True
        else:
            squeeze = False
        b, t, _ = features.shape
        if state is None:
            state = self.initial_state(init_legacy[:, :15])
            state.hc = self._init_hc(init_legacy)
        outputs, debug_rows = [], []
        lengths = torch.full((b,), t, device=features.device, dtype=torch.long) if lengths is None else lengths
        for i in range(t):
            y, hc = self.net.rnn(features[:, i:i + 1].transpose(0, 1), state.hc)
            state.hc = hc
            raw = self.net.linear2(y[0])
            legacy, state, debug = self._update(raw, state, i < lengths)
            outputs.append(legacy)
            debug_rows.append(debug)
            if detach_chunks and chunk_size and (i + 1) % chunk_size == 0:
                state = state.detach()
        result = {k: torch.stack([row[k] for row in debug_rows], dim=1) if k != "beta" else self.beta
                  for k in debug_rows[0]}
        result["raw_output"] = torch.cat((result["vRB_direct"], result["aRB_leaf"],
                                          torch.stack([row["gR1"] for row in debug_rows], dim=1)), dim=-1)
        result["pl"] = torch.stack(outputs, dim=1)
        result["state"] = state
        if squeeze:
            for key, value in list(result.items()):
                if torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == 1:
                    result[key] = value[0]
        return result

    def reset_stream(self, p_rl, g_r0, rmb0=None):
        init = torch.cat((p_rl.reshape(1, 15), g_r0.reshape(1, 3)), dim=-1)
        self.stream_state = self.initial_state(p_rl)
        self.stream_state.hc = self._init_hc(init)
        self.prev_rmb = rmb0
        self.acc_filter = None

    def step(self, feature_t):
        if self.stream_state is None:
            raise RuntimeError("reset_stream must be called before step")
        x = feature_t.reshape(1, 1, INPUT_SIZE)
        y, hc = self.net.rnn(x, self.stream_state.hc)
        self.stream_state.hc = hc
        raw = self.net.linear2(y[0])
        legacy, self.stream_state, debug = self._update(raw, self.stream_state)
        self.last_debug = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in debug.items()}
        return {"pl_t": legacy, **debug}


def partial_initialize_from_official(model, weights_path, report_path=None, new_input_init="zero"):
    official = torch.load(weights_path, map_location="cpu")
    old = {k.removeprefix("plnet."): v for k, v in official.items() if k.startswith("plnet.")}
    target = model.net.state_dict()
    report = {"source": str(weights_path), "copied_keys": [], "partially_copied_keys": [],
              "newly_initialized_keys": [], "skipped_keys": [], "new_input_init": new_input_init}
    with torch.no_grad():
        for key, dst in target.items():
            src = old.get(key)
            entry = {"key": key, "shape_before": list(src.shape) if src is not None else None, "shape_after": list(dst.shape)}
            if src is not None and src.shape == dst.shape:
                dst.copy_(src); report["copied_keys"].append(entry)
            elif key == "rnn.weight_ih_l0" and src is not None:
                dst.zero_(); dst[:, :84].copy_(src); report["partially_copied_keys"].append({**entry, "copied": "columns 0:84", "new": "columns 84:102 zero"})
            elif key in ("linear2.weight", "linear2.bias") and src is not None:
                dst.zero_()
                if key.endswith("weight"): dst[30:33].copy_(src[15:18])
                else: dst[30:33].copy_(src[15:18])
                report["partially_copied_keys"].append({**entry, "copied": "old gravity rows 15:18 to new 30:33", "new": "v/a rows zero"})
            else:
                report["newly_initialized_keys"].append(entry)
        model.net.load_state_dict(target)
        for key, src in old.items():
            if key not in target:
                report["skipped_keys"].append({"key": key, "shape_before": list(src.shape), "shape_after": None})
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


def centered_derivative_targets(p, dt=DT):
    v = torch.zeros_like(p); a = torch.zeros_like(p)
    if p.shape[0] < 3:
        return v, a, torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
    v[1:-1] = (p[2:] - p[:-2]) / (2 * dt)
    v[0], v[-1] = (p[1] - p[0]) / dt, (p[-1] - p[-2]) / dt
    a[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / (dt * dt)
    a[0], a[-1] = (v[1] - v[0]) / dt, (v[-1] - v[-2]) / dt
    return v, a, torch.ones(p.shape[0], dtype=torch.bool, device=p.device)
