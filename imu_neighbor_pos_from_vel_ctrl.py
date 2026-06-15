from typing import Dict, Optional, Sequence, Tuple

import torch

from imu_neighbor_vel_ctrl import (
    DT,
    IMU_JOINTS,
    NEIGHBOR_NODE_GROUPS,
    OUTPUT_DIM as VELOCITY_OUTPUT_DIM,
    OUTPUT_GROUP_DIMS,
    ROOT_SLICE,
    SENSOR_NAMES,
    finite_difference_first,
    flatten_nodes,
    imu_neighbor_features,
    output_as_nodes,
    paired_segment_tensor,
    pose_to_rotation_matrices,
    temporal_jitter,
)
from l4_tail_update_qstate import UniformCubicBSpline
from pl_curve import fit_uniform_cubic_spline_controls


IMU_FEATURE_DIM = 90
VELOCITY_CONTROL_DIM = VELOCITY_OUTPUT_DIM
VELOCITY_DECODED_DIM = VELOCITY_OUTPUT_DIM
VELOCITY_ACC_DIM = VELOCITY_OUTPUT_DIM
INPUT_DIM = IMU_FEATURE_DIM + VELOCITY_CONTROL_DIM + VELOCITY_DECODED_DIM + VELOCITY_ACC_DIM
OUTPUT_DIM = 33
NODE_NAMES = (
    "left_forearm_parent_18",
    "left_forearm_child_20",
    "right_forearm_parent_19",
    "right_forearm_child_21",
    "left_lowerleg_parent_4",
    "left_lowerleg_child_7",
    "right_lowerleg_parent_5",
    "right_lowerleg_child_8",
    "head_segment_parent_12",
    "head_segment_child_15",
    "pelvis_root_0",
)
COORDINATE_CONTRACT = (
    "imu_neighbor_pos_from_vel_ctrl_v1 predicts root-relative position controls "
    "for the same 11 IMU-neighbor nodes used by imu_neighbor_vel_ctrl_v1. "
    "Input is imu_feature[90] + neighbor_vel_W_control[33] + decoded "
    "neighbor_vel_W[33] + decoded neighbor_acc_W[33] = 189D. Output is "
    "neighbor_pos_R_control[33]. Decoded pos_R, vel_R, acc_R are in the root "
    "frame R. Root-relative position uses the existing GlobalPose row-vector "
    "contract p_RJ=(p_WJ-p_WR) @ R_WR. The root channel is retained for layout "
    "alignment and is zero by construction. DIP trans/root/world velocity GT is "
    "not used or fabricated; DIP supervision is pose-derived root-relative "
    "position only."
)


def velocity_pack_keys() -> Tuple[str, str, str]:
    return "control_vel_W", "vel_W", "acc_W"


def position_input_features(
    imu_feature: torch.Tensor,
    velocity_input: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build fixed 189D input: IMU feature plus velocity control/vel/acc."""
    keys = velocity_pack_keys()
    if imu_feature.shape[-1] != IMU_FEATURE_DIM:
        raise ValueError(f"Expected imu_feature dim {IMU_FEATURE_DIM}, got {imu_feature.shape[-1]}.")
    missing = [key for key in keys if key not in velocity_input]
    if missing:
        raise KeyError(f"velocity_input missing required keys: {missing}")
    parts = [imu_feature.float()]
    for key in keys:
        value = velocity_input[key].float()
        if value.shape[-1] != OUTPUT_DIM:
            raise ValueError(f"{key} expected dim {OUTPUT_DIM}, got {value.shape[-1]}.")
        parts.append(value)
    return torch.cat(parts, dim=-1).float()


def mix_velocity_inputs(
    pred_velocity: Dict[str, torch.Tensor],
    gt_velocity: Optional[Dict[str, torch.Tensor]],
    gt_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Mix frozen velocity-model predictions with GT velocity controls when allowed."""
    ratio = max(0.0, min(1.0, float(gt_ratio)))
    if gt_velocity is None or ratio <= 0.0:
        return {key: pred_velocity[key].float() for key in velocity_pack_keys()}
    mixed = {}
    for key in velocity_pack_keys():
        pred = pred_velocity[key].float()
        gt = gt_velocity[key].float().to(pred.device, pred.dtype)
        mixed[key] = (1.0 - ratio) * pred + ratio * gt
    return mixed


def neighbor_root_relative_positions_from_pose(
    pose: torch.Tensor,
    body_model,
    device: torch.device,
) -> torch.Tensor:
    """Return [T,33] node positions in root frame R from SMPL pose only.

    The implementation follows the existing PL convention:
    row-vector world offsets are right-multiplied by R_WR.
    """
    pose_R = pose_to_rotation_matrices(pose).to(device)
    global_pose, joints = body_model.forward_kinematics(pose_R, None, None, calc_mesh=False)
    root_pos = joints[:, :1]
    root_rot = global_pose[:, 0]
    pieces = []
    for group in NEIGHBOR_NODE_GROUPS:
        node_pos_W = joints[:, list(group)]
        node_pos_R = torch.matmul(node_pos_W - root_pos, root_rot)
        pieces.append(node_pos_R.reshape(joints.shape[0], -1))
    return torch.cat(pieces, dim=-1).float()


def neighbor_position_targets_from_pose(
    pose: torch.Tensor,
    body_model,
    device: torch.device,
    dt: float = DT,
) -> Dict[str, torch.Tensor]:
    pos_R = neighbor_root_relative_positions_from_pose(pose, body_model, device)
    vel_R = finite_difference_first(pos_R, dt=dt)
    acc_R = finite_difference_first(vel_R, dt=dt)
    control_pos_R = fit_uniform_cubic_spline_controls(pos_R, dt=dt)
    return {
        "pos_R": pos_R.detach().cpu(),
        "vel_R": vel_R.detach().cpu(),
        "acc_R": acc_R.detach().cpu(),
        "control_pos_R": control_pos_R.detach().cpu(),
    }


def segment_lengths(x: torch.Tensor) -> torch.Tensor:
    pair = paired_segment_tensor(x)
    return (pair[..., :, 1, :] - pair[..., :, 0, :]).norm(dim=-1)


def segment_length_error_cm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (segment_lengths(pred) - segment_lengths(target)).abs().mean() * 100.0


def default_loss_weights(dataset: str) -> Dict[str, float]:
    if dataset == "dip":
        return {
            "ctrl_pos": 1.0,
            "decoded_pos": 1.0,
            "vel_R": 0.0,
            "acc_R": 0.0,
            "vel_input_consistency": 0.0,
            "segment_length": 0.05,
            "smooth": 0.01,
            "jerk": 0.005,
            "control_prior": 0.001,
        }
    return {
        "ctrl_pos": 1.0,
        "decoded_pos": 1.0,
        "vel_R": 0.2,
        "acc_R": 0.05,
        "vel_input_consistency": 0.05,
        "segment_length": 0.05,
        "smooth": 0.01,
        "jerk": 0.005,
        "control_prior": 0.001,
    }


def _zero_like_loss(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    return output["pos_R"].new_zeros(())


def neighbor_position_loss(
    output: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pred = output["pos_R"]
    target_control = target["control_pos_R"].to(pred.device, pred.dtype)
    target_pos = target["pos_R"].to(pred.device, pred.dtype)
    target_vel = target["vel_R"].to(pred.device, pred.dtype)
    target_acc = target["acc_R"].to(pred.device, pred.dtype)
    losses = {
        "ctrl_pos": torch.nn.functional.smooth_l1_loss(output["control_pos_R"], target_control),
        "decoded_pos": torch.nn.functional.smooth_l1_loss(output["pos_R"], target_pos),
        "vel_R": torch.nn.functional.smooth_l1_loss(output["vel_R"], target_vel),
        "acc_R": torch.nn.functional.smooth_l1_loss(output["acc_R"], target_acc),
        "vel_input_consistency": torch.nn.functional.smooth_l1_loss(output["vel_R"], target_vel),
        "segment_length": torch.nn.functional.smooth_l1_loss(segment_lengths(output["pos_R"]), segment_lengths(target_pos)),
        "smooth": _zero_like_loss(output),
        "jerk": _zero_like_loss(output),
        "control_prior": output["control_prior"],
    }
    if pred.shape[0] >= 2:
        losses["smooth"] = (output["vel_R"][1:] - output["vel_R"][:-1]).square().mean()
    if pred.shape[0] >= 2:
        jerk = finite_difference_first(output["acc_R"], dt=output.get("dt", DT))
        losses["jerk"] = jerk.square().mean()
    total = pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * float(weight)
    return total, losses


def selection_value(losses: Dict[str, float]) -> float:
    return float(losses.get("ctrl_pos", 0.0)) + float(losses.get("decoded_pos", 0.0)) + 0.1 * float(losses.get("vel_R", 0.0))


class IMUNeighborPositionFromVelocityModule(torch.nn.Module):
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
            raise ValueError(f"imu_neighbor_pos_from_vel_ctrl_v1 input is fixed at {INPUT_DIM}D, got {input_size}.")
        if output_size != OUTPUT_DIM:
            raise ValueError(f"imu_neighbor_pos_from_vel_ctrl_v1 output is fixed at {OUTPUT_DIM}D, got {output_size}.")
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
        pos_btd, vel_btd, acc_btd = self.spline(control_btd, return_derivatives=True)
        result = {
            "control_pos_R": control_tbd,
            "pos_R": pos_btd.transpose(0, 1),
            "vel_R": vel_btd.transpose(0, 1),
            "acc_R": acc_btd.transpose(0, 1),
            "hidden": hidden_out,
            "control_prior": control_tbd.square().mean(),
            "dt": self.dt,
        }
        if squeeze_batch:
            for key in ("control_pos_R", "pos_R", "vel_R", "acc_R"):
                result[key] = result[key][:, 0]
        return result


def position_metric_dict(
    output: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    baseline_target: Optional[Dict[str, torch.Tensor]] = None,
    baseline_source: Optional[str] = None,
) -> Dict[str, object]:
    pos = output["pos_R"].detach().cpu()
    vel = output["vel_R"].detach().cpu()
    acc = output["acc_R"].detach().cpu()
    target_pos = target["pos_R"].detach().cpu()
    target_vel = target["vel_R"].detach().cpu()
    target_acc = target["acc_R"].detach().cpu()
    pos_diff_nodes = flatten_nodes(pos - target_pos)
    vel_diff_nodes = flatten_nodes(vel - target_vel)
    acc_diff_nodes = flatten_nodes(acc - target_acc)
    per_node_pos = pos_diff_nodes.norm(dim=-1).mean(dim=0) * 100.0
    per_node_vel = vel_diff_nodes.norm(dim=-1).mean(dim=0) * 100.0
    per_node_acc = acc_diff_nodes.norm(dim=-1).mean(dim=0) * 100.0
    metrics: Dict[str, object] = {
        "pos_R_L1_cm": float((pos - target_pos).abs().mean() * 100.0),
        "pos_R_L2_cm": float(pos_diff_nodes.norm(dim=-1).mean() * 100.0),
        "vel_R_L2_cm_s": float(vel_diff_nodes.norm(dim=-1).mean() * 100.0),
        "acc_R_L2_cm_s2": float(acc_diff_nodes.norm(dim=-1).mean() * 100.0),
        "segment_length_error_cm": float(segment_length_error_cm(pos, target_pos)),
        "pos_smooth_jitter_cm": float(temporal_jitter(output_as_nodes(pos)) * 100.0) if pos.shape[0] > 2 else 0.0,
        "acc_jitter_cm_s2": float(temporal_jitter(output_as_nodes(acc)) * 100.0) if acc.shape[0] > 2 else 0.0,
        "per_node_pos_R_L2_cm": [float(v) for v in per_node_pos],
        "per_node_vel_R_L2_cm_s": [float(v) for v in per_node_vel],
        "per_node_acc_R_L2_cm_s2": [float(v) for v in per_node_acc],
        "node_names": list(NODE_NAMES),
        "baseline_source": baseline_source or "pose baseline not available",
    }
    if baseline_target is None:
        metrics.update({
            "baseline_pos_R_L1_cm": None,
            "baseline_pos_R_L2_cm": None,
            "baseline_vel_R_L2_cm_s": None,
            "baseline_acc_R_L2_cm_s2": None,
            "baseline_segment_length_error_cm": None,
            "baseline_per_node_pos_R_L2_cm": None,
        })
        return metrics
    base_pos = baseline_target["pos_R"].detach().cpu()
    base_vel = baseline_target["vel_R"].detach().cpu()
    base_acc = baseline_target["acc_R"].detach().cpu()
    base_pos_diff_nodes = flatten_nodes(base_pos - target_pos)
    metrics.update({
        "baseline_pos_R_L1_cm": float((base_pos - target_pos).abs().mean() * 100.0),
        "baseline_pos_R_L2_cm": float(base_pos_diff_nodes.norm(dim=-1).mean() * 100.0),
        "baseline_vel_R_L2_cm_s": float(flatten_nodes(base_vel - target_vel).norm(dim=-1).mean() * 100.0),
        "baseline_acc_R_L2_cm_s2": float(flatten_nodes(base_acc - target_acc).norm(dim=-1).mean() * 100.0),
        "baseline_segment_length_error_cm": float(segment_length_error_cm(base_pos, target_pos)),
        "baseline_per_node_pos_R_L2_cm": [float(v) for v in base_pos_diff_nodes.norm(dim=-1).mean(dim=0) * 100.0],
    })
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
            if values and all(isinstance(item, (int, float)) for item in values[0]):
                tensor = torch.tensor(values, dtype=torch.float32)
                out[key] = [float(v) for v in tensor.mean(dim=0)]
            else:
                out[key] = values[0]
        else:
            unique = sorted({str(v) for v in values})
            out[key] = unique[0] if len(unique) == 1 else ", ".join(unique)
    return out


def model_contract() -> Dict[str, object]:
    return {
        "name": "imu_neighbor_pos_from_vel_ctrl_v1",
        "input": "imu_feature[90]+neighbor_vel_W_control[33]+neighbor_vel_W[33]+neighbor_acc_W[33]=189D",
        "output": "neighbor_pos_R_control[33]",
        "decoded_outputs": "pos_R[33]+vel_R[33]+acc_R[33]",
        "coordinate_contract": COORDINATE_CONTRACT,
        "sensor_names": list(SENSOR_NAMES),
        "imu_joints": list(IMU_JOINTS),
        "neighbor_node_groups": [list(group) for group in NEIGHBOR_NODE_GROUPS],
        "output_group_dims": list(OUTPUT_GROUP_DIMS),
        "node_names": list(NODE_NAMES),
        "root_slice": [ROOT_SLICE.start, ROOT_SLICE.stop],
    }
