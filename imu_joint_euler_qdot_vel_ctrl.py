import inspect
from typing import Dict, Optional, Sequence, Tuple

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
from pl_curve import fit_uniform_cubic_spline_controls


FPS = 60.0
DT = 1.0 / FPS
INPUT_DIM = 90
INIT_DIM = 54
HEAD_DIM = 18
EULER_SEQ = "XYZ"
INPUT_ROTATION_FRAME = "root_imu_relative_rmb_v1"
IMU_JOINTS = (18, 19, 4, 5, 15, 0)
JOINT_NAMES = (
    "left_forearm_18",
    "right_forearm_19",
    "left_lowerleg_4",
    "right_lowerleg_5",
    "head_15",
    "pelvis_root_0",
)
COORDINATE_CONTRACT = (
    "imu_joint_euler_qdot_vel_ctrl_v1 predicts root-relative controls for "
    "IMU-mapped joints [18,19,4,5,15,0]. Input uses official "
    "world/model-frame aM[18]+wM[18] plus root-frame "
    "R_rootIMU_sensorIMU[54]=90D, where R_rootIMU_sensorIMU = "
    "RMB[root_imu=5]^T @ RMB[sensor]. Rotation target is "
    "R_RJ=R_WR^T R_WJ converted to unwrapped XYZ Euler. Position target is "
    "p_RJ=(p_WJ-p_WR) @ R_WR and velocity/acceleration are finite differences "
    "in root frame R. DIP translation is not used."
)


def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def unwrap_euler_sequence(euler: torch.Tensor) -> torch.Tensor:
    if euler.shape[0] <= 1:
        return euler.clone()
    diff = wrap_pi(euler[1:] - euler[:-1])
    return torch.cat((euler[:1], euler[:1] + torch.cumsum(diff, dim=0)), dim=0)


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


def pose_to_rotation_matrices(pose: torch.Tensor) -> torch.Tensor:
    pose = pose.float()
    if pose.dim() == 4 and pose.shape[-3:] == (24, 3, 3):
        return pose
    if pose.dim() == 3 and pose.shape[-2:] == (24, 3):
        return art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
    if pose.dim() == 2 and pose.shape[-1] == 72:
        return art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
    raise ValueError(f"Unsupported pose shape: {tuple(pose.shape)}")


def imu_world_features(aM: torch.Tensor, wM: torch.Tensor, RMB: torch.Tensor) -> torch.Tensor:
    """Build 90D official/world-frame features: aM[18]+wM[18]+RMB_flat[54]."""
    if aM.shape[-2:] != (6, 3) or wM.shape[-2:] != (6, 3):
        raise ValueError(f"Expected aM/wM [T,6,3], got {tuple(aM.shape)} / {tuple(wM.shape)}")
    if RMB.shape[-3:] != (6, 3, 3):
        raise ValueError(f"Expected RMB [T,6,3,3], got {tuple(RMB.shape)}")
    return torch.cat(
        (aM.reshape(aM.shape[0], 18), wM.reshape(wM.shape[0], 18), RMB.reshape(RMB.shape[0], 54)),
        dim=-1,
    ).float()


def imu_rootframe_features(aM: torch.Tensor, wM: torch.Tensor, RMB: torch.Tensor, root_imu_index: int = 5) -> torch.Tensor:
    """Build 90D features: aM[18]+wM[18]+root-frame RMB[54].

    `RMB` is the project IMU body-to-model/world orientation. The rotation input is
    converted to the pelvis/root IMU frame as R_rootIMU_sensorIMU = RMB[root]^T @ RMB[sensor].
    Acceleration and gyro are intentionally left in their existing project frame.
    """
    if aM.shape[-2:] != (6, 3) or wM.shape[-2:] != (6, 3):
        raise ValueError(f"Expected aM/wM [T,6,3], got {tuple(aM.shape)} / {tuple(wM.shape)}")
    if RMB.shape[-3:] != (6, 3, 3):
        raise ValueError(f"Expected RMB [T,6,3,3], got {tuple(RMB.shape)}")
    if not 0 <= int(root_imu_index) < RMB.shape[-3]:
        raise ValueError(f"root_imu_index={root_imu_index} is outside RMB sensor dimension {RMB.shape[-3]}.")
    root_R = RMB[:, int(root_imu_index)]
    RMB_root = root_R.transpose(-1, -2).unsqueeze(1).matmul(RMB)
    return torch.cat(
        (aM.reshape(aM.shape[0], 18), wM.reshape(wM.shape[0], 18), RMB_root.reshape(RMB.shape[0], 54)),
        dim=-1,
    ).float()


def output_as_joints(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[:-1] + (6, 3))


def flatten_joint_error(x: torch.Tensor) -> torch.Tensor:
    return output_as_joints(x).reshape(-1, 6, 3)


def root_relative_targets_from_pose(
    pose: torch.Tensor,
    body_model,
    device: torch.device,
    dt: float = DT,
    euler_seq: str = EULER_SEQ,
) -> Dict[str, torch.Tensor]:
    pose_R = pose_to_rotation_matrices(pose).to(device)
    glb_R = body_model.forward_kinematics_R(pose_R)
    root_R = glb_R[:, 0]
    selected_R = glb_R[:, list(IMU_JOINTS)]
    R_RJ = root_R.transpose(-1, -2).unsqueeze(1).matmul(selected_R)
    euler = art.math.rotation_matrix_to_euler_angle(R_RJ.reshape(-1, 3, 3), seq=euler_seq).view(-1, 6, 3)
    q = unwrap_euler_sequence(euler).reshape(euler.shape[0], HEAD_DIM).float()

    _, joints = body_model.forward_kinematics(pose_R, None, None, calc_mesh=False)
    selected_p = joints[:, list(IMU_JOINTS)]
    p_RJ = (selected_p - joints[:, :1]).bmm(root_R).reshape(joints.shape[0], HEAD_DIM).float()
    vel = finite_difference_first(p_RJ, dt=dt)
    acc = finite_difference_first(vel, dt=dt)
    qdot = finite_difference_first(q, dt=dt)
    qddot = finite_difference_first(qdot, dt=dt)
    control_q = fit_uniform_cubic_spline_controls(q, dt=dt)
    control_qdot = fit_uniform_cubic_spline_controls(qdot, dt=dt)
    control_vel = fit_uniform_cubic_spline_controls(vel, dt=dt)
    init = torch.cat((q[0], qdot[0], vel[0]), dim=-1)
    return {
        "q_euler": q.detach().cpu(),
        "qdot_euler": qdot.detach().cpu(),
        "qddot_euler": qddot.detach().cpu(),
        "vel_RJ": vel.detach().cpu(),
        "acc_RJ": acc.detach().cpu(),
        "control_q_euler": control_q.detach().cpu(),
        "control_qdot_euler": control_qdot.detach().cpu(),
        "control_vel_RJ": control_vel.detach().cpu(),
        "R_RJ": R_RJ.detach().cpu(),
        "p_RJ": p_RJ.detach().cpu(),
        "init_state": init.detach().cpu(),
    }


class IMUJointEulerQdotVelControlModule(torch.nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_DIM,
        init_size: int = INIT_DIM,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.4,
        dt: float = DT,
    ):
        super().__init__()
        if input_size != INPUT_DIM:
            raise ValueError(f"input is fixed at {INPUT_DIM}D, got {input_size}.")
        if init_size != INIT_DIM:
            raise ValueError(f"init is fixed at {INIT_DIM}D, got {init_size}.")
        self.input_size = int(input_size)
        self.init_size = int(init_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dt = float(dt)
        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(self.init_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size * self.num_layers),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size * self.num_layers, 2 * self.num_layers * self.hidden_size),
        )
        self.rnn = torch.nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.q_head = torch.nn.Linear(self.hidden_size, HEAD_DIM)
        self.qdot_head = torch.nn.Linear(self.hidden_size, HEAD_DIM)
        self.vel_head = torch.nn.Linear(self.hidden_size, HEAD_DIM)
        self.spline = UniformCubicBSpline(dt)
        for head in (self.q_head, self.qdot_head, self.vel_head):
            torch.nn.init.xavier_uniform_(head.weight, gain=0.05)
            torch.nn.init.zeros_(head.bias)

    def _initial_state(self, init_state: Optional[torch.Tensor], batch_size: int, device, dtype):
        if init_state is None:
            init_state = torch.zeros(batch_size, self.init_size, device=device, dtype=dtype)
        elif init_state.dim() == 1:
            init_state = init_state.view(1, -1).expand(batch_size, -1)
        if init_state.shape != (batch_size, self.init_size):
            raise ValueError(f"Expected init_state [{batch_size},{self.init_size}], got {tuple(init_state.shape)}")
        raw = self.init_net(init_state.to(device=device, dtype=dtype))
        h, c = raw.view(batch_size, 2, self.num_layers, self.hidden_size).permute(1, 2, 0, 3)
        return h.contiguous(), c.contiguous()

    def forward_sequence(self, features: torch.Tensor, init_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        squeeze_batch = features.dim() == 2
        if squeeze_batch:
            features = features.unsqueeze(1)
        if features.shape[-1] != self.input_size:
            raise ValueError(f"Expected feature dim {self.input_size}, got {features.shape[-1]}.")
        batch_size = features.shape[1]
        hc = self._initial_state(init_state, batch_size, features.device, features.dtype)
        h, hidden = self.rnn(features, hc)
        control_q = self.q_head(h)
        control_qdot = self.qdot_head(h)
        control_vel = self.vel_head(h)

        q_btd, qdot_from_q_btd, qddot_from_q_btd = self.spline(control_q.transpose(0, 1), return_derivatives=True)
        qdot_btd, qddot_from_qdot_btd, qjerk_btd = self.spline(control_qdot.transpose(0, 1), return_derivatives=True)
        vel_btd, acc_btd, jerk_btd = self.spline(control_vel.transpose(0, 1), return_derivatives=True)
        result = {
            "control_q_euler": control_q,
            "control_qdot_euler": control_qdot,
            "control_vel_RJ": control_vel,
            "q_euler": q_btd.transpose(0, 1),
            "qdot_from_q": qdot_from_q_btd.transpose(0, 1),
            "qddot_from_q": qddot_from_q_btd.transpose(0, 1),
            "qdot_euler": qdot_btd.transpose(0, 1),
            "qddot_from_qdot": qddot_from_qdot_btd.transpose(0, 1),
            "qdot_jerk": qjerk_btd.transpose(0, 1),
            "vel_RJ": vel_btd.transpose(0, 1),
            "acc_RJ": acc_btd.transpose(0, 1),
            "jerk_RJ": jerk_btd.transpose(0, 1),
            "hidden": hidden,
            "control_prior": (
                control_q.square().mean()
                + control_qdot.square().mean()
                + control_vel.square().mean()
            ) / 3.0,
        }
        if squeeze_batch:
            for key, value in list(result.items()):
                if torch.is_tensor(value) and value.dim() >= 3 and value.shape[1] == 1:
                    result[key] = value[:, 0]
        return result


VARIANT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "A_qctrl_main": {
        "q_control": 1.0,
        "q": 0.7,
        "qdot": 0.3,
        "qdot_decoded": 0.3,
        "qdot_control": 0.3,
        "qddot_from_q": 0.2,
        "qddot_from_qdot": 0.2,
        "vel_control": 0.0,
        "vel": 0.2,
        "acc": 0.2,
        "consistency": 0.05,
        "smooth": 0.01,
        "jerk": 0.0,
        "control_prior": 0.001,
    },
    "B_qdot_qddot_strong": {
        "q_control": 0.8,
        "q": 0.6,
        "qdot": 0.8,
        "qdot_decoded": 0.8,
        "qdot_control": 0.8,
        "qddot_from_q": 0.6,
        "qddot_from_qdot": 0.6,
        "vel_control": 0.0,
        "vel": 0.2,
        "acc": 0.2,
        "consistency": 0.1,
        "smooth": 0.01,
        "jerk": 0.0,
        "control_prior": 0.001,
    },
    "C_vel_acc_strong": {
        "q_control": 0.5,
        "q": 0.5,
        "qdot": 0.3,
        "qdot_decoded": 0.3,
        "qdot_control": 0.3,
        "qddot_from_q": 0.2,
        "qddot_from_qdot": 0.2,
        "vel_control": 1.0,
        "vel": 0.8,
        "acc": 0.8,
        "consistency": 0.05,
        "smooth": 0.01,
        "jerk": 0.005,
        "control_prior": 0.001,
    },
    "D_all_balanced": {
        "q_control": 1.0,
        "q": 0.7,
        "qdot": 0.7,
        "qdot_decoded": 0.7,
        "qdot_control": 1.0,
        "qddot_from_q": 0.5,
        "qddot_from_qdot": 0.5,
        "vel_control": 0.6,
        "vel": 0.6,
        "acc": 0.6,
        "consistency": 0.15,
        "smooth": 0.01,
        "jerk": 0.005,
        "control_prior": 0.001,
    },
}


def zero_loss(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    return output["q_euler"].new_zeros(())


def target_to_device(target: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    ref = output["q_euler"]
    return {key: value.to(ref.device, ref.dtype) for key, value in target.items() if torch.is_tensor(value)}


def imu_joint_loss(
    output: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    tgt = target_to_device(target, output)
    losses = {
        "q_control": torch.nn.functional.smooth_l1_loss(output["control_q_euler"], tgt["control_q_euler"]),
        "q": torch.nn.functional.smooth_l1_loss(output["q_euler"], tgt["q_euler"]),
        "qdot": torch.nn.functional.smooth_l1_loss(output["qdot_from_q"], tgt["qdot_euler"]),
        "qdot_decoded": torch.nn.functional.smooth_l1_loss(output["qdot_euler"], tgt["qdot_euler"]),
        "qdot_control": torch.nn.functional.smooth_l1_loss(output["control_qdot_euler"], tgt["control_qdot_euler"]),
        "qddot_from_q": torch.nn.functional.smooth_l1_loss(output["qddot_from_q"], tgt["qddot_euler"]),
        "qddot_from_qdot": torch.nn.functional.smooth_l1_loss(output["qddot_from_qdot"], tgt["qddot_euler"]),
        "vel_control": torch.nn.functional.smooth_l1_loss(output["control_vel_RJ"], tgt["control_vel_RJ"]),
        "vel": torch.nn.functional.smooth_l1_loss(output["vel_RJ"], tgt["vel_RJ"]),
        "acc": torch.nn.functional.smooth_l1_loss(output["acc_RJ"], tgt["acc_RJ"]),
        "smooth": zero_loss(output),
        "jerk": zero_loss(output),
        "control_prior": output["control_prior"],
        "consistency": zero_loss(output),
    }
    if output["q_euler"].shape[0] >= 2:
        q_step = output["q_euler"][1:] - output["q_euler"][:-1]
        v_step = output["vel_RJ"][1:] - output["vel_RJ"][:-1]
        losses["consistency"] = (
            torch.nn.functional.smooth_l1_loss(output["qdot_from_q"], output["qdot_euler"].detach())
            + 0.5 * torch.nn.functional.smooth_l1_loss(output["qddot_from_q"], output["qddot_from_qdot"].detach())
            + 0.5 * torch.nn.functional.smooth_l1_loss(q_step, output["qdot_euler"][1:] * DT)
            + 0.5 * torch.nn.functional.smooth_l1_loss(v_step, output["acc_RJ"][1:] * DT)
        )
        losses["smooth"] = (
            output["qdot_from_q"].square().mean()
            + 0.1 * output["qddot_from_q"].square().mean()
            + 0.1 * output["acc_RJ"].square().mean()
        )
    if output["jerk_RJ"].numel() > 0:
        losses["jerk"] = (output["jerk_RJ"].square().mean() + output["qdot_jerk"].square().mean()) * 0.5
    total = zero_loss(output)
    for key, weight in weights.items():
        total = total + losses[key] * float(weight)
    return total, losses


def selection_value(losses: Dict[str, float]) -> float:
    return (
        float(losses.get("q_control", 0.0))
        + 0.5 * float(losses.get("q", 0.0))
        + 0.4 * float(losses.get("qdot", 0.0))
        + 0.4 * float(losses.get("qdot_control", 0.0))
        + 0.3 * float(losses.get("qddot_from_q", 0.0))
        + 0.3 * float(losses.get("qddot_from_qdot", 0.0))
        + 0.3 * float(losses.get("vel", 0.0))
        + 0.3 * float(losses.get("acc", 0.0))
    )


def rotation_geodesic_deg(pred_euler: torch.Tensor, target_euler: torch.Tensor, seq: str = EULER_SEQ) -> torch.Tensor:
    pred_R = art.math.euler_angle_to_rotation_matrix(pred_euler.reshape(-1, 3), seq=seq).view(-1, 3, 3)
    target_R = art.math.euler_angle_to_rotation_matrix(target_euler.reshape(-1, 3), seq=seq).view(-1, 3, 3)
    rel = pred_R.transpose(-1, -2).matmul(target_R)
    trace = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def temporal_jitter(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 3:
        return x.new_zeros(())
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]).norm(dim=-1).mean()


def direction_angle_deg(pred: torch.Tensor, target: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    pred_norm = pred.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    valid = (pred_norm > threshold) & (target_norm > threshold)
    if not bool(valid.any()):
        return pred.new_zeros(())
    pred_n = pred[valid] / pred_norm[valid].unsqueeze(-1).clamp_min(1e-8)
    target_n = target[valid] / target_norm[valid].unsqueeze(-1).clamp_min(1e-8)
    cos = (pred_n * target_n).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).mean()


def metric_dict(
    output: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    baseline_target: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, object]:
    q = output["q_euler"].detach().cpu()
    qdot_q = output["qdot_from_q"].detach().cpu()
    qdot_head = output["qdot_euler"].detach().cpu()
    qddot_q = output["qddot_from_q"].detach().cpu()
    qddot_head = output["qddot_from_qdot"].detach().cpu()
    vel = output["vel_RJ"].detach().cpu()
    acc = output["acc_RJ"].detach().cpu()
    tq = target["q_euler"].detach().cpu()
    tqdot = target["qdot_euler"].detach().cpu()
    tqddot = target["qddot_euler"].detach().cpu()
    tvel = target["vel_RJ"].detach().cpu()
    tacc = target["acc_RJ"].detach().cpu()
    q_diff = flatten_joint_error(q - tq)
    qdot_diff = flatten_joint_error(qdot_q - tqdot)
    qdot_head_diff = flatten_joint_error(qdot_head - tqdot)
    qddot_diff = flatten_joint_error(qddot_q - tqddot)
    qddot_head_diff = flatten_joint_error(qddot_head - tqddot)
    vel_diff = flatten_joint_error(vel - tvel)
    acc_diff = flatten_joint_error(acc - tacc)
    geodesic = rotation_geodesic_deg(output_as_joints(q), output_as_joints(tq)).view(-1, 6)
    metrics: Dict[str, object] = {
        "q_euler_L1_deg": float(torch.rad2deg((q - tq).abs()).mean()),
        "q_euler_L2_deg": float(torch.rad2deg(q_diff.norm(dim=-1)).mean()),
        "rotation_geodesic_deg": float(geodesic.mean()),
        "qdot_from_q_L2_deg_s": float(torch.rad2deg(qdot_diff.norm(dim=-1)).mean()),
        "qdot_head_L2_deg_s": float(torch.rad2deg(qdot_head_diff.norm(dim=-1)).mean()),
        "qddot_from_q_L2_deg_s2": float(torch.rad2deg(qddot_diff.norm(dim=-1)).mean()),
        "qddot_head_L2_deg_s2": float(torch.rad2deg(qddot_head_diff.norm(dim=-1)).mean()),
        "vel_RJ_L1_cm_s": float((vel - tvel).abs().mean() * 100.0),
        "vel_RJ_L2_cm_s": float(vel_diff.norm(dim=-1).mean() * 100.0),
        "acc_RJ_L2_cm_s2": float(acc_diff.norm(dim=-1).mean() * 100.0),
        "velocity_direction_angle_deg": float(direction_angle_deg(output_as_joints(vel), output_as_joints(tvel))),
        "q_smooth_jitter_deg": float(torch.rad2deg(temporal_jitter(output_as_joints(q)))),
        "vel_smooth_jitter_cm_s": float(temporal_jitter(output_as_joints(vel)) * 100.0),
        "per_joint_rotation_geodesic_deg": [float(v) for v in geodesic.mean(dim=0)],
        "per_joint_vel_RJ_L2_cm_s": [float(v * 100.0) for v in vel_diff.norm(dim=-1).mean(dim=0)],
        "joint_names": list(JOINT_NAMES),
    }
    if baseline_target is None:
        metrics.update({
            "baseline_source": "not available",
            "baseline_rotation_geodesic_deg": None,
            "baseline_vel_RJ_L2_cm_s": None,
            "baseline_acc_RJ_L2_cm_s2": None,
        })
    else:
        bq = baseline_target["q_euler"].detach().cpu()
        bvel = baseline_target["vel_RJ"].detach().cpu()
        bacc = baseline_target["acc_RJ"].detach().cpu()
        b_geo = rotation_geodesic_deg(output_as_joints(bq), output_as_joints(tq)).view(-1, 6)
        metrics.update({
            "baseline_source": "pose_prephysics FK root-relative",
            "baseline_rotation_geodesic_deg": float(b_geo.mean()),
            "baseline_vel_RJ_L2_cm_s": float(flatten_joint_error(bvel - tvel).norm(dim=-1).mean() * 100.0),
            "baseline_acc_RJ_L2_cm_s2": float(flatten_joint_error(bacc - tacc).norm(dim=-1).mean() * 100.0),
            "baseline_per_joint_rotation_geodesic_deg": [float(v) for v in b_geo.mean(dim=0)],
            "baseline_per_joint_vel_RJ_L2_cm_s": [
                float(v * 100.0) for v in flatten_joint_error(bvel - tvel).norm(dim=-1).mean(dim=0)
            ],
        })
    return metrics


def average_metric_dicts(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    out: Dict[str, object] = {}
    for key in rows[0].keys():
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            out[key] = None
        elif isinstance(values[0], (int, float)):
            out[key] = float(sum(float(v) for v in values) / len(values))
        elif isinstance(values[0], list) and values[0] and isinstance(values[0][0], (int, float)):
            tensor = torch.tensor(values, dtype=torch.float32)
            out[key] = [float(v) for v in tensor.mean(dim=0)]
        else:
            unique = sorted({str(v) for v in values})
            out[key] = unique[0] if len(unique) == 1 else ", ".join(unique)
    return out


def model_contract() -> Dict[str, object]:
    return {
        "name": "imu_joint_euler_qdot_vel_ctrl_v1",
        "input": "aM[18]+wM[18]+R_rootIMU_sensorIMU_flat[54]=90D",
        "input_rotation_frame_id": INPUT_ROTATION_FRAME,
        "input_rotation_frame": "root IMU frame, R_rootIMU_sensorIMU = RMB[root_imu=5]^T @ RMB[sensor]",
        "output": "q_RJ_euler_control[18]+qdot_RJ_euler_control[18]+vel_RJ_control[18]=54D",
        "decoded_outputs": "q_RJ_euler/qdot_from_q/qddot_from_q/qdot_RJ/qddot_from_qdot/vel_RJ/acc_RJ",
        "coordinate_contract": COORDINATE_CONTRACT,
        "euler_seq": EULER_SEQ,
        "imu_joints": list(IMU_JOINTS),
        "joint_names": list(JOINT_NAMES),
    }
