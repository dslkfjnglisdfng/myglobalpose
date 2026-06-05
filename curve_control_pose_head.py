import torch


def curve_head_input_dim(use_imu=True, use_feature_velocity=True):
    dim = 117
    if use_feature_velocity:
        dim += 69
    if use_imu:
        dim += 90
    return dim


def build_curve_frame_features(
    ik2_input,
    pRJ_ik1,
    aM,
    wM,
    RMB,
    prev_pRJ_ik1=None,
    use_imu=True,
    use_feature_velocity=True,
):
    parts = [ik2_input.reshape(-1)]
    if use_feature_velocity:
        if prev_pRJ_ik1 is None:
            velocity = torch.zeros_like(pRJ_ik1.reshape(-1))
        else:
            velocity = pRJ_ik1.reshape(-1) - prev_pRJ_ik1.reshape(-1)
        parts.append(velocity)
    if use_imu:
        parts.extend((aM.reshape(-1), wM.reshape(-1), RMB.reshape(-1)))
    return torch.cat(parts)


class CurveControlPoseHead(torch.nn.Module):
    r"""
    Phase-1 Curve-Control Pose Head.

    The module replaces IK-s2 inference in the experimental branch. It consumes
    IK-s2-slot features from PL-s1/IK-s1, optionally adds feature velocity and
    raw IMU, and outputs q75 curve control points. It never calls IK-s2.
    """

    def __init__(
        self,
        input_dim=None,
        hidden_size=256,
        state_dim=75,
        residual_scale=0.05,
        use_imu=True,
        use_feature_velocity=True,
        rnn_init_mode='r_js_firstframe',
        freeze_root_translation=True,
        predict_root_orientation=True,
        offset_init_scale=0.2,
    ):
        super().__init__()
        if rnn_init_mode not in ('none', 'r_js_firstframe'):
            raise ValueError(f'Unsupported curve rnn init mode: {rnn_init_mode}')
        self.use_imu = bool(use_imu)
        self.use_feature_velocity = bool(use_feature_velocity)
        self.input_dim = int(curve_head_input_dim(self.use_imu, self.use_feature_velocity) if input_dim is None else input_dim)
        self.hidden_size = int(hidden_size)
        self.state_dim = int(state_dim)
        self.residual_scale = float(residual_scale)
        self.rnn_init_mode = rnn_init_mode
        self.freeze_root_translation = bool(freeze_root_translation)
        self.predict_root_orientation = bool(predict_root_orientation)
        self.offset_init_scale = float(offset_init_scale)

        self.input_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.control_head = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.base_control = torch.nn.Parameter(torch.zeros(self.state_dim))
        if self.rnn_init_mode == 'r_js_firstframe':
            self.init_encoder = torch.nn.Sequential(
                torch.nn.Linear(18 + self.input_dim, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size),
            )
        else:
            self.init_encoder = None

        torch.nn.init.zeros_(self.control_head.weight)
        torch.nn.init.zeros_(self.control_head.bias)
        if self.init_encoder is not None:
            torch.nn.init.zeros_(self.init_encoder[-1].weight)
            torch.nn.init.zeros_(self.init_encoder[-1].bias)

    def initial_hidden(self, first_frame_feature, offset_r=None):
        batch = first_frame_feature.shape[0]
        device = first_frame_feature.device
        dtype = first_frame_feature.dtype
        if self.rnn_init_mode == 'none' or self.init_encoder is None:
            return torch.zeros(1, batch, self.hidden_size, device=device, dtype=dtype)
        if offset_r is None:
            offset = torch.zeros(batch, 18, device=device, dtype=dtype)
        else:
            offset = offset_r.to(device=device, dtype=dtype).reshape(batch, -1)
            if offset.shape[-1] != 18:
                raise ValueError(f'Expected offset_r flatten dim 18, got {offset.shape[-1]}.')
        init = self.init_encoder(torch.cat((offset, first_frame_feature), dim=-1))
        return (init * self.offset_init_scale).unsqueeze(0)

    def _apply_output_mask(self, control):
        if self.freeze_root_translation:
            control = control.clone()
            control[..., :3] = 0.0
        if not self.predict_root_orientation:
            control = control.clone()
            control[..., 3:6] = 0.0
        return control

    def forward(self, frame_features, offset_r=None):
        squeeze_batch = frame_features.dim() == 2
        if squeeze_batch:
            frame_features = frame_features.unsqueeze(0)
        if frame_features.dim() != 3 or frame_features.shape[-1] != self.input_dim:
            raise ValueError(f'Expected frame_features [T,{self.input_dim}] or [B,T,{self.input_dim}], got {tuple(frame_features.shape)}.')
        h0 = self.initial_hidden(frame_features[:, 0], offset_r=offset_r)
        z = self.input_encoder(frame_features)
        out, hidden = self.gru(z, h0)
        delta_control = self.control_head(out)
        control = self.base_control.view(1, 1, -1) + self.residual_scale * delta_control
        control = self._apply_output_mask(control)
        result = {
            'control': control.squeeze(0) if squeeze_batch else control,
            'delta_control': delta_control.squeeze(0) if squeeze_batch else delta_control,
            'hidden': hidden,
        }
        return result
