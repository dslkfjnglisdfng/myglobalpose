import torch

from l4_tail_update_qstate import UniformCubicBSpline


ACC_CURVE_INPUT_SIZE = 108
ACC_CURVE_STATE_DIM = 18


def rotation_matrix_to_6d(rotation):
    return rotation[..., :, :2].transpose(-1, -2).reshape(rotation.shape[:-2] + (6,))


def acc_curve_features(aM_raw, aM_smooth, wM, RMB):
    """Build model/world-frame AccCurve features.

    Layout:
      aM_raw[18] + aM_smooth[18] + residual[18] + wM[18] + RMB_6d[36].
    """
    rmb_6d = rotation_matrix_to_6d(RMB).reshape(RMB.shape[0], 36)
    return torch.cat((
        aM_raw.reshape(aM_raw.shape[0], 18),
        aM_smooth.reshape(aM_smooth.shape[0], 18),
        (aM_raw - aM_smooth).reshape(aM_raw.shape[0], 18),
        wM.reshape(wM.shape[0], 18),
        rmb_6d,
    ), dim=-1).float()


class PLStyleAccCurveModule(torch.nn.Module):
    """PLCurve-style residual curve over absolute 18D sensor-site acceleration."""

    def __init__(
        self,
        input_size=ACC_CURVE_INPUT_SIZE,
        state_dim=ACC_CURVE_STATE_DIM,
        hidden_size=512,
        tail_update=4,
        residual_scale=1.0,
        dt=1.0 / 60.0,
        dropout=0.1,
    ):
        super().__init__()
        if input_size != ACC_CURVE_INPUT_SIZE:
            raise ValueError(f"AccCurve expects 108D input, got {input_size}.")
        if state_dim != ACC_CURVE_STATE_DIM:
            raise ValueError(f"AccCurve predicts 18D acceleration, got {state_dim}.")
        if tail_update != 4:
            raise ValueError("AccCurve keeps the PL-style L=4 tail-update contract.")
        self.input_size = int(input_size)
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.tail_update = int(tail_update)
        self.residual_scale = float(residual_scale)
        self.dt = float(dt)
        self.input = torch.nn.Linear(input_size + state_dim, hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.cell = torch.nn.GRUCell(hidden_size, hidden_size)
        self.new_control = torch.nn.Linear(hidden_size, state_dim)
        self.tail_delta = torch.nn.Linear(hidden_size, tail_update * state_dim)
        self.spline = UniformCubicBSpline(dt)
        self.reset_stream()
        torch.nn.init.zeros_(self.new_control.weight)
        torch.nn.init.zeros_(self.new_control.bias)
        torch.nn.init.zeros_(self.tail_delta.weight)
        torch.nn.init.zeros_(self.tail_delta.bias)

    def reset_stream(self):
        self.hidden = None
        self.control_buffer = None
        self.base_buffer = None

    def _initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _ghost(self, buffer, count=1):
        return buffer[:, -1:].expand(-1, int(count), -1).clone()

    def step(self, feature_t, base_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_t.dim() == 1:
            base_t = base_t.unsqueeze(0)
        if feature_t.shape[-1] != self.input_size:
            raise ValueError(f"Expected feature dim {self.input_size}, got {feature_t.shape[-1]}.")
        if base_t.shape[-1] != self.state_dim:
            raise ValueError(f"Expected base dim {self.state_dim}, got {base_t.shape[-1]}.")
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
        pred_curve = self.spline(decode_control)
        base_curve = self.spline(decode_base)
        pred_t = pred_curve[:, -2]
        base_decoded_t = base_curve[:, -2]
        return {
            "pred_aM_t": pred_t,
            "base_t": base_decoded_t,
            "residual_t": pred_t - base_decoded_t,
            "new_control_t": new_control,
            "control_point_prior_t": (self.control_buffer - self.base_buffer).square().mean(),
            "new_delta_norm": new_delta.norm(dim=-1).mean(),
            "tail_delta_norm": tail_delta_norm,
        }

    def forward_sequence(self, features, base):
        squeeze_batch = features.dim() == 2
        if squeeze_batch:
            features = features.unsqueeze(1)
            base = base.unsqueeze(1)
        self.reset_stream()
        preds, bases, residuals, controls = [], [], [], []
        priors, new_norms, tail_norms = [], [], []
        for i in range(features.shape[0]):
            out = self.step(features[i], base[i])
            preds.append(out["pred_aM_t"])
            bases.append(out["base_t"])
            residuals.append(out["residual_t"])
            controls.append(out["new_control_t"])
            priors.append(out["control_point_prior_t"])
            new_norms.append(out["new_delta_norm"])
            tail_norms.append(out["tail_delta_norm"])
        result = {
            "pred_aM": torch.stack(preds),
            "base": torch.stack(bases),
            "residual": torch.stack(residuals),
            "new_control": torch.stack(controls),
            "control_point_prior_t": torch.stack(priors).mean(),
            "new_delta_norm": torch.stack(new_norms).mean(),
            "tail_delta_norm": torch.stack(tail_norms).mean(),
        }
        if squeeze_batch:
            for key in ("pred_aM", "base", "residual", "new_control"):
                result[key] = result[key][:, 0]
        return result
