import torch

from l4_sensor_offset_utils import (
    FPS,
    GRAVITY_WORLD,
    finite_difference_second,
    fk_imu_joints_and_vertices,
)
from pl_curve import normalize_gravity


OFFSET_COORDINATE_CONTRACT = (
    "offset_r is r_JS: IMU origin position relative to mapped joint J, "
    "expressed in joint-local coordinates. World position is "
    "p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS."
)


def flatten_imu_features(aM, wM, RMB):
    return torch.cat((aM.reshape(aM.shape[0], -1), wM.reshape(wM.shape[0], -1), RMB.reshape(RMB.shape[0], -1)), dim=-1)


def offset_input_feature(pl_output, aM, wM, RMB, joint_rel=None):
    parts = [normalize_gravity(pl_output).reshape(pl_output.shape[0], -1), flatten_imu_features(aM, wM, RMB)]
    if joint_rel is not None:
        parts.append(joint_rel.reshape(joint_rel.shape[0], -1))
    return torch.cat(parts, dim=-1).float()


def finite_diff(x, order):
    if order == 1:
        return x[1:] - x[:-1]
    if order == 2:
        return x[2:] - 2.0 * x[1:-1] + x[:-2]
    raise ValueError(order)


def smooth_l2(x):
    return x.square().mean().sqrt()


class IMUOffsetNet(torch.nn.Module):
    """Estimate sequence-level or frame-level joint-local IMU position offsets."""

    def __init__(
        self,
        version="offset_v1_mlp_frame",
        input_size=90,
        hidden_size=256,
        num_sensors=6,
        prior_offset=None,
        residual_scale=0.05,
        dropout=0.1,
    ):
        super().__init__()
        if version not in {"offset_v1_mlp_frame", "offset_v2_temporal_rnn", "offset_v3_residual_prior"}:
            raise ValueError(f"Unsupported IMUOffsetNet version={version}")
        self.version = version
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_sensors = int(num_sensors)
        self.residual_scale = float(residual_scale)
        prior = torch.zeros(num_sensors, 3) if prior_offset is None else prior_offset.float().view(num_sensors, 3)
        self.register_buffer("prior_offset", prior)
        self.input = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )
        if version == "offset_v2_temporal_rnn":
            self.rnn = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = None
        self.output = torch.nn.Linear(hidden_size, num_sensors * 3)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, feature):
        single = feature.dim() == 2
        if single:
            feature = feature.unsqueeze(0)
        b, t, d = feature.shape
        if d != self.input_size:
            raise ValueError(f"Expected feature dim {self.input_size}, got {d}")
        h = self.input(feature.reshape(b * t, d)).reshape(b, t, self.hidden_size)
        if self.rnn is not None:
            h, _ = self.rnn(h)
        raw = self.output(h).reshape(b, t, self.num_sensors, 3)
        if self.version == "offset_v3_residual_prior":
            offset = self.prior_offset.view(1, 1, self.num_sensors, 3) + raw * self.residual_scale
        else:
            offset = raw
        return offset[0] if single else offset


def offset_supervised_loss(pred_offset, target_offset):
    target = target_offset.to(pred_offset.device, pred_offset.dtype)
    if target.dim() == 2:
        target = target.view(1, 1, *target.shape).expand_as(pred_offset)
    elif target.dim() == 3:
        target = target.unsqueeze(1).expand_as(pred_offset)
    return torch.nn.functional.smooth_l1_loss(pred_offset, target)


def offset_regularizers(pred_offset, prior_offset=None):
    losses = {
        "magnitude": pred_offset.norm(dim=-1).mean(),
        "temporal_smooth": pred_offset.new_zeros(()),
    }
    if pred_offset.shape[1] > 1:
        losses["temporal_smooth"] = (pred_offset[:, 1:] - pred_offset[:, :-1]).square().mean()
    if prior_offset is not None:
        prior = prior_offset.to(pred_offset.device, pred_offset.dtype).view(1, 1, pred_offset.shape[-2], 3)
        losses["prior_l2"] = (pred_offset - prior).square().mean().clamp_min(1e-12).sqrt()
    else:
        losses["prior_l2"] = pred_offset.new_zeros(())
    return losses


@torch.no_grad()
def pose_derived_sensor_acceleration(pose, tran, offset_r, device="cpu", fps=FPS):
    p_wj, R_wj, _ = fk_imu_joints_and_vertices(pose, tran, device=device)
    offset = offset_r.float().view(1, 6, 3, 1)
    p_ws = p_wj + R_wj.matmul(offset).squeeze(-1)
    return finite_difference_second(p_ws, fps=fps)


def acceleration_consistency_loss(pose, tran, pred_offset, aM, device="cpu"):
    if pred_offset.dim() == 4:
        offset = pred_offset.mean(dim=1)[0]
    elif pred_offset.dim() == 3:
        offset = pred_offset.mean(dim=0)
    else:
        offset = pred_offset
    acc_proxy = pose_derived_sensor_acceleration(pose, tran, offset.detach().cpu(), device=device).to(aM.device, aM.dtype)
    return torch.nn.functional.smooth_l1_loss(acc_proxy, aM)


def acceleration_consistency_error(pose, tran, offset_r, aM, device="cpu"):
    proxy = pose_derived_sensor_acceleration(pose, tran, offset_r, device=device).to(aM.device, aM.dtype)
    err = (proxy - aM).norm(dim=-1)
    return {
        "acc_consistency_mean": float(err.mean().detach().cpu()),
        "acc_consistency_median": float(err.median().detach().cpu()),
        "acc_consistency_p95": float(torch.quantile(err.reshape(-1), 0.95).detach().cpu()),
    }


def make_checkpoint(model, config, epoch, step, val_loss, optimizer=None):
    out = {
        "model_type": "imu_offset_net_v1",
        "version": model.version,
        "coordinate_contract": OFFSET_COORDINATE_CONTRACT,
        "config": config,
        "epoch": int(epoch),
        "step": int(step),
        "val_loss": float(val_loss),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        out["optimizer_state_dict"] = optimizer.state_dict()
    return out
