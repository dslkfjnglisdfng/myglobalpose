import inspect
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import articulate as art
from l4_tail_update_qstate import UniformCubicBSpline
from pl_curve import fit_uniform_cubic_spline_controls, rotation_matrix_to_6d


FPS = 60.0
DT = 1.0 / FPS
INPUT_DIM = 90
OUTPUT_DIM = 33
SENSOR_NAMES = (
    "left_forearm",
    "right_forearm",
    "left_lowerleg",
    "right_lowerleg",
    "head",
    "pelvis_root",
)
IMU_JOINTS = (18, 19, 4, 5, 15, 0)
NEIGHBOR_NODE_GROUPS = (
    (18, 20),
    (19, 21),
    (4, 7),
    (5, 8),
    (12, 15),
    (0,),
)
OUTPUT_GROUP_DIMS = tuple(len(group) * 3 for group in NEIGHBOR_NODE_GROUPS)
ROOT_SLICE = slice(30, 33)
COORDINATE_CONTRACT = (
    "imu_neighbor_vel_ctrl_v1 predicts world-frame velocity controls. "
    "r_JS is the IMU origin relative to mapped joint J, expressed in joint-local "
    "coordinates; p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS. Targets are v_WJ=d p_WJ/dt "
    "and a_WJ=d v_WJ/dt in world/model frame W. DIP trans/root/world velocity "
    "GT is not trusted and must not be fabricated."
)


def finite_difference_first(x: torch.Tensor, dt: float = DT) -> torch.Tensor:
    x = x.float()
    out = torch.zeros_like(x)
    if x.shape[0] <= 1:
        return out
    if x.shape[0] == 2:
        diff = (x[1] - x[0]) / float(dt)
        out[0] = diff
        out[1] = diff
        return out
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * float(dt))
    out[0] = (x[1] - x[0]) / float(dt)
    out[-1] = (x[-1] - x[-2]) / float(dt)
    return out


def finite_difference_second(x: torch.Tensor, dt: float = DT) -> torch.Tensor:
    x = x.float()
    out = torch.zeros_like(x)
    if x.shape[0] <= 2:
        return out
    out[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (float(dt) ** 2)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def pose_to_rotation_matrices(pose: torch.Tensor) -> torch.Tensor:
    pose = pose.float()
    if pose.dim() == 4 and pose.shape[-3:] == (24, 3, 3):
        return pose
    if pose.dim() == 3 and pose.shape[-2:] == (24, 3):
        return art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
    if pose.dim() == 2 and pose.shape[-1] == 72:
        return art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
    raise ValueError(f"Unsupported pose shape for SMPL FK: {tuple(pose.shape)}")


def normalize_offset_r(offset_r: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    offset = offset_r.float()
    if offset.numel() != 18:
        raise ValueError(f"Expected r_JS/offset_r with 18 values, got shape {tuple(offset_r.shape)}")
    offset = offset.reshape(6, 3)
    if device is not None or dtype is not None:
        offset = offset.to(device=device, dtype=dtype)
    return offset


def imu_neighbor_features(
    aM: torch.Tensor,
    wM: torch.Tensor,
    RMB: torch.Tensor,
    offset_r: torch.Tensor,
) -> torch.Tensor:
    """Build 90D frame features: aM[18]+wM[18]+RMB_6d[36]+r_JS[18]."""
    if aM.shape[-2:] != (6, 3) or wM.shape[-2:] != (6, 3):
        raise ValueError(f"Expected aM/wM shape [T,6,3], got {tuple(aM.shape)} / {tuple(wM.shape)}")
    if RMB.shape[-3:] != (6, 3, 3):
        raise ValueError(f"Expected RMB shape [T,6,3,3], got {tuple(RMB.shape)}")
    offset = normalize_offset_r(offset_r, device=aM.device, dtype=aM.dtype)
    rmb6d = rotation_matrix_to_6d(RMB).reshape(RMB.shape[0], 36)
    offset_seq = offset.reshape(1, 18).expand(aM.shape[0], -1)
    return torch.cat((aM.reshape(aM.shape[0], 18), wM.reshape(wM.shape[0], 18), rmb6d, offset_seq), dim=-1).float()


def world_gt_available(dataset: str, world_gt_mode: str = "auto") -> bool:
    if world_gt_mode == "available":
        return dataset != "dip"
    if world_gt_mode == "unavailable":
        return False
    if world_gt_mode != "auto":
        raise ValueError(f"Unsupported world_gt_mode={world_gt_mode!r}")
    return dataset in {"amass", "totalcapture"}


def neighbor_world_positions_from_pose_tran(
    pose: torch.Tensor,
    tran: torch.Tensor,
    body_model,
    device: torch.device,
) -> torch.Tensor:
    pose_R = pose_to_rotation_matrices(pose).to(device)
    tran = tran.float().to(device)
    _, joints = body_model.forward_kinematics(pose_R, None, tran, calc_mesh=False)
    pieces = []
    for group in NEIGHBOR_NODE_GROUPS:
        node_pos = joints[:, list(group)]
        pieces.append(node_pos.reshape(joints.shape[0], -1))
    return torch.cat(pieces, dim=-1).float()


def neighbor_velocity_targets_from_pose_tran(
    pose: torch.Tensor,
    tran: torch.Tensor,
    body_model,
    device: torch.device,
    dt: float = DT,
) -> Dict[str, torch.Tensor]:
    pos_W = neighbor_world_positions_from_pose_tran(pose, tran, body_model, device)
    vel_W = finite_difference_first(pos_W, dt=dt)
    acc_W = finite_difference_first(vel_W, dt=dt)
    acc_W_from_pos = finite_difference_second(pos_W, dt=dt)
    control_vel_W = fit_uniform_cubic_spline_controls(vel_W, dt=dt)
    return {
        "pos_W": pos_W.detach().cpu(),
        "vel_W": vel_W.detach().cpu(),
        "acc_W": acc_W.detach().cpu(),
        "acc_W_from_pos": acc_W_from_pos.detach().cpu(),
        "control_vel_W": control_vel_W.detach().cpu(),
    }


def output_as_nodes(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[:-1] + (11, 3))


def flatten_nodes(x: torch.Tensor) -> torch.Tensor:
    return output_as_nodes(x).reshape(-1, 11, 3)


def paired_segment_tensor(x: torch.Tensor) -> torch.Tensor:
    return x[..., :30].reshape(x.shape[:-1] + (5, 2, 3))


def direction_angle_deg(pred: torch.Tensor, target: torch.Tensor, speed_threshold: float = 1e-4) -> torch.Tensor:
    pred_norm = pred.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    valid = (pred_norm > float(speed_threshold)) & (target_norm > float(speed_threshold))
    if not bool(valid.any()):
        return pred.new_zeros(())
    pred_n = pred[valid] / pred_norm[valid].unsqueeze(-1).clamp_min(1e-8)
    target_n = target[valid] / target_norm[valid].unsqueeze(-1).clamp_min(1e-8)
    cos = (pred_n * target_n).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).mean()


def temporal_jitter(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 3:
        return x.new_zeros(())
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]).norm(dim=-1).mean()


class IMUNeighborVelocityControlModule(torch.nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_DIM,
        output_size: int = OUTPUT_DIM,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        dt: float = DT,
    ):
        super().__init__()
        if input_size != INPUT_DIM:
            raise ValueError(f"imu_neighbor_vel_ctrl_v1 input is fixed at {INPUT_DIM}D, got {input_size}.")
        if output_size != OUTPUT_DIM:
            raise ValueError(f"imu_neighbor_vel_ctrl_v1 output is fixed at {OUTPUT_DIM}D, got {output_size}.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dt = float(dt)
        self.input = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)) if dropout > 0.0 else torch.nn.Identity(),
        )
        self.rnn = torch.nn.GRU(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.control_head = torch.nn.Linear(self.hidden_size, self.output_size)
        self.spline = UniformCubicBSpline(dt)
        torch.nn.init.xavier_uniform_(self.control_head.weight, gain=0.05)
        torch.nn.init.zeros_(self.control_head.bias)

    def forward_sequence(self, features: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        squeeze_batch = features.dim() == 2
        if squeeze_batch:
            features = features.unsqueeze(1)
        if features.shape[-1] != self.input_size:
            raise ValueError(f"Expected feature dim {self.input_size}, got {features.shape[-1]}.")
        z = self.input(features)
        h, hidden_out = self.rnn(z, hidden)
        control_tbd = self.control_head(h)
        control_btd = control_tbd.transpose(0, 1)
        vel_btd, acc_btd, jerk_btd = self.spline(control_btd, return_derivatives=True)
        result = {
            "control_vel_W": control_tbd,
            "vel_W": vel_btd.transpose(0, 1),
            "acc_W": acc_btd.transpose(0, 1),
            "jerk_W": jerk_btd.transpose(0, 1),
            "hidden": hidden_out,
            "control_prior": control_tbd.square().mean(),
        }
        if squeeze_batch:
            for key in ("control_vel_W", "vel_W", "acc_W", "jerk_W"):
                result[key] = result[key][:, 0]
        return result


def default_loss_weights(world_gt: bool = True) -> Dict[str, float]:
    if world_gt:
        return {
            "ctrl_vel": 1.0,
            "decoded_vel": 0.5,
            "acc_W": 0.5,
            "root_vel": 0.5,
            "root_acc_W": 0.1,
            "segment_consistency": 0.05,
            "vel_smooth": 0.01,
            "jerk_smooth": 0.005,
            "control_prior": 0.001,
            "distill": 0.0,
        }
    return {
        "ctrl_vel": 0.0,
        "decoded_vel": 0.0,
        "acc_W": 0.0,
        "root_vel": 0.0,
        "root_acc_W": 0.0,
        "segment_consistency": 0.0,
        "vel_smooth": 0.01,
        "jerk_smooth": 0.005,
        "control_prior": 0.001,
        "distill": 0.2,
    }


def _zero_like_loss(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    return output["vel_W"].new_zeros(())


def neighbor_velocity_loss(
    output: Dict[str, torch.Tensor],
    target: Optional[Dict[str, torch.Tensor]],
    weights: Dict[str, float],
    world_gt: bool = True,
    teacher_output: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pred = output["vel_W"]
    losses = {
        "ctrl_vel": _zero_like_loss(output),
        "decoded_vel": _zero_like_loss(output),
        "acc_W": _zero_like_loss(output),
        "root_vel": _zero_like_loss(output),
        "root_acc_W": _zero_like_loss(output),
        "segment_consistency": _zero_like_loss(output),
        "vel_smooth": _zero_like_loss(output),
        "jerk_smooth": _zero_like_loss(output),
        "control_prior": output["control_prior"],
        "distill": _zero_like_loss(output),
    }
    if world_gt:
        if target is None:
            raise ValueError("world_gt=True requires target tensors.")
        target_control = target["control_vel_W"].to(pred.device, pred.dtype)
        target_vel = target["vel_W"].to(pred.device, pred.dtype)
        target_acc = target["acc_W"].to(pred.device, pred.dtype)
        losses["ctrl_vel"] = torch.nn.functional.smooth_l1_loss(output["control_vel_W"], target_control)
        losses["decoded_vel"] = torch.nn.functional.smooth_l1_loss(output["vel_W"], target_vel)
        losses["acc_W"] = torch.nn.functional.smooth_l1_loss(output["acc_W"], target_acc)
        losses["root_vel"] = torch.nn.functional.smooth_l1_loss(output["vel_W"][..., ROOT_SLICE], target_vel[..., ROOT_SLICE])
        losses["root_acc_W"] = torch.nn.functional.smooth_l1_loss(output["acc_W"][..., ROOT_SLICE], target_acc[..., ROOT_SLICE])
        pred_pair_v = paired_segment_tensor(output["vel_W"])
        target_pair_v = paired_segment_tensor(target_vel)
        pred_pair_a = paired_segment_tensor(output["acc_W"])
        target_pair_a = paired_segment_tensor(target_acc)
        pred_rel_v = pred_pair_v[..., :, 1, :] - pred_pair_v[..., :, 0, :]
        target_rel_v = target_pair_v[..., :, 1, :] - target_pair_v[..., :, 0, :]
        pred_rel_a = pred_pair_a[..., :, 1, :] - pred_pair_a[..., :, 0, :]
        target_rel_a = target_pair_a[..., :, 1, :] - target_pair_a[..., :, 0, :]
        losses["segment_consistency"] = (
            torch.nn.functional.smooth_l1_loss(pred_rel_v, target_rel_v)
            + 0.5 * torch.nn.functional.smooth_l1_loss(pred_rel_a, target_rel_a)
        )
    if pred.shape[0] >= 2:
        losses["vel_smooth"] = (pred[1:] - pred[:-1]).square().mean()
    if output["jerk_W"].numel() > 0:
        losses["jerk_smooth"] = output["jerk_W"].square().mean()
    if teacher_output is not None:
        losses["distill"] = (
            torch.nn.functional.smooth_l1_loss(output["vel_W"], teacher_output["vel_W"].detach().to(pred.device, pred.dtype))
            + 0.5
            * torch.nn.functional.smooth_l1_loss(
                output["control_vel_W"],
                teacher_output["control_vel_W"].detach().to(pred.device, pred.dtype),
            )
        )
    total = pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * float(weight)
    return total, losses


def selection_value(losses: Dict[str, float], world_gt: bool = True) -> float:
    if world_gt:
        return float(losses.get("ctrl_vel", 0.0)) + float(losses.get("acc_W", 0.0))
    return float(losses.get("distill", 0.0)) + float(losses.get("vel_smooth", 0.0))


def metric_dict(
    output: Dict[str, torch.Tensor],
    target: Optional[Dict[str, torch.Tensor]],
    world_gt: bool,
    baseline_output: Optional[Dict[str, torch.Tensor]] = None,
    baseline_root_velocity: Optional[torch.Tensor] = None,
    baseline_root_source: Optional[str] = None,
) -> Dict[str, object]:
    vel = output["vel_W"].detach().cpu()
    acc = output["acc_W"].detach().cpu()
    baseline_source = "baseline velocity not available"
    if baseline_output is not None:
        baseline_source = "pose_baseline/tran_baseline finite difference"
    elif baseline_root_velocity is not None:
        baseline_source = baseline_root_source or "v_root_vr root velocity"
    metrics: Dict[str, object] = {
        "world_gt_status": "ok" if world_gt else "world velocity GT not available",
        "baseline_velocity_source": baseline_source,
        "vel_smooth_jitter": float(temporal_jitter(output_as_nodes(vel))) if vel.shape[0] > 2 else 0.0,
        "acc_jitter": float(temporal_jitter(output_as_nodes(acc))) if acc.shape[0] > 2 else 0.0,
    }
    if not world_gt or target is None:
        metrics.update({
            "velocity_L1_mps": None,
            "velocity_L2_mps": None,
            "acceleration_L1_mps2": None,
            "acceleration_L2_mps2": None,
            "root_velocity_L1_mps": None,
            "root_velocity_L2_mps": None,
            "root_acceleration_L1_mps2": None,
            "root_acceleration_L2_mps2": None,
            "velocity_direction_angle_deg": None,
            "per_node_velocity_L2_mps": None,
            "per_node_acceleration_L2_mps2": None,
            "baseline_velocity_L2_mps": None,
            "baseline_acceleration_L2_mps2": None,
            "baseline_root_velocity_L1_mps": None,
            "baseline_root_velocity_L2_mps": None,
            "baseline_root_acceleration_L1_mps2": None,
            "baseline_root_acceleration_L2_mps2": None,
        })
        return metrics
    target_vel = target["vel_W"].detach().cpu()
    target_acc = target["acc_W"].detach().cpu()
    vel_diff = flatten_nodes(vel - target_vel)
    acc_diff = flatten_nodes(acc - target_acc)
    per_node_vel = vel_diff.norm(dim=-1).mean(dim=0)
    per_node_acc = acc_diff.norm(dim=-1).mean(dim=0)
    metrics.update({
        "velocity_L1_mps": float((vel - target_vel).abs().mean()),
        "velocity_L2_mps": float(vel_diff.norm(dim=-1).mean()),
        "acceleration_L1_mps2": float((acc - target_acc).abs().mean()),
        "acceleration_L2_mps2": float(acc_diff.norm(dim=-1).mean()),
        "root_velocity_L1_mps": float((vel[..., ROOT_SLICE] - target_vel[..., ROOT_SLICE]).abs().mean()),
        "root_velocity_L2_mps": float((vel[..., ROOT_SLICE] - target_vel[..., ROOT_SLICE]).norm(dim=-1).mean()),
        "root_acceleration_L1_mps2": float((acc[..., ROOT_SLICE] - target_acc[..., ROOT_SLICE]).abs().mean()),
        "root_acceleration_L2_mps2": float((acc[..., ROOT_SLICE] - target_acc[..., ROOT_SLICE]).norm(dim=-1).mean()),
        "velocity_direction_angle_deg": float(direction_angle_deg(output_as_nodes(vel), output_as_nodes(target_vel))),
        "per_node_velocity_L2_mps": [float(v) for v in per_node_vel],
        "per_node_acceleration_L2_mps2": [float(v) for v in per_node_acc],
    })
    if baseline_output is not None:
        b_vel = baseline_output["vel_W"].detach().cpu()
        b_acc = baseline_output["acc_W"].detach().cpu()
        metrics["baseline_velocity_L2_mps"] = float(output_as_nodes(b_vel - target_vel).norm(dim=-1).mean())
        metrics["baseline_acceleration_L2_mps2"] = float(output_as_nodes(b_acc - target_acc).norm(dim=-1).mean())
        b_root_vel = b_vel[..., ROOT_SLICE]
        b_root_acc = b_acc[..., ROOT_SLICE]
        metrics["baseline_root_velocity_L1_mps"] = float((b_root_vel - target_vel[..., ROOT_SLICE]).abs().mean())
        metrics["baseline_root_velocity_L2_mps"] = float((b_root_vel - target_vel[..., ROOT_SLICE]).norm(dim=-1).mean())
        metrics["baseline_root_acceleration_L1_mps2"] = float((b_root_acc - target_acc[..., ROOT_SLICE]).abs().mean())
        metrics["baseline_root_acceleration_L2_mps2"] = float((b_root_acc - target_acc[..., ROOT_SLICE]).norm(dim=-1).mean())
    else:
        metrics["baseline_velocity_L2_mps"] = None
        metrics["baseline_acceleration_L2_mps2"] = None
        if baseline_root_velocity is not None:
            b_root_vel = baseline_root_velocity.detach().cpu().to(target_vel.dtype)
            b_root_acc = finite_difference_first(b_root_vel)
            metrics["baseline_root_velocity_L1_mps"] = float((b_root_vel - target_vel[..., ROOT_SLICE]).abs().mean())
            metrics["baseline_root_velocity_L2_mps"] = float((b_root_vel - target_vel[..., ROOT_SLICE]).norm(dim=-1).mean())
            metrics["baseline_root_acceleration_L1_mps2"] = float((b_root_acc - target_acc[..., ROOT_SLICE]).abs().mean())
            metrics["baseline_root_acceleration_L2_mps2"] = float((b_root_acc - target_acc[..., ROOT_SLICE]).norm(dim=-1).mean())
        else:
            metrics["baseline_root_velocity_L1_mps"] = None
            metrics["baseline_root_velocity_L2_mps"] = None
            metrics["baseline_root_acceleration_L1_mps2"] = None
            metrics["baseline_root_acceleration_L2_mps2"] = None
    return metrics


def average_metric_dicts(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    out: Dict[str, object] = {}
    keys = rows[0].keys()
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            out[key] = None
        elif isinstance(values[0], (int, float)):
            out[key] = float(sum(float(v) for v in values) / len(values))
        elif isinstance(values[0], list):
            tensor = torch.tensor(values, dtype=torch.float32)
            out[key] = [float(v) for v in tensor.mean(dim=0)]
        else:
            unique = sorted({str(v) for v in values})
            out[key] = unique[0] if len(unique) == 1 else ", ".join(unique)
    return out


def model_contract() -> Dict[str, object]:
    return {
        "name": "imu_neighbor_vel_ctrl_v1",
        "input": "aM[18]+wM[18]+RMB_6d[36]+r_JS[18]=90D",
        "output": "neighbor_vel_W_control[33]",
        "coordinate_contract": COORDINATE_CONTRACT,
        "sensor_names": list(SENSOR_NAMES),
        "imu_joints": list(IMU_JOINTS),
        "neighbor_node_groups": [list(group) for group in NEIGHBOR_NODE_GROUPS],
        "root_slice": [ROOT_SLICE.start, ROOT_SLICE.stop],
    }
