import torch

import articulate as art
from l4_q75_utils import q75_to_pose_tran
from newpose_ctrl import DEFAULT_FK_VERTEX_MASK, root_relative_fk_targets


DT = 1.0 / 60.0
INPUT_SIZE = 153
Q75_DIM = 75


def finite_diff_same_length(x, order, dt=DT):
    if x.shape[0] == 0:
        return x
    if order == 1:
        if x.shape[0] == 1:
            return torch.zeros_like(x)
        out = torch.empty_like(x)
        out[0] = (x[1] - x[0]) / dt
        out[-1] = (x[-1] - x[-2]) / dt
        if x.shape[0] > 2:
            out[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        return out
    if order == 2:
        if x.shape[0] <= 2:
            return torch.zeros_like(x)
        out = torch.empty_like(x)
        mid = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt ** 2)
        out[1:-1] = mid
        out[0] = mid[0]
        out[-1] = mid[-1]
        return out
    raise ValueError(order)


def rotation_geodesic(R_pred, R_target, eps=1e-6):
    rel = R_pred.transpose(-1, -2).matmul(R_target)
    trace = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


def normalize_record_tensor(x, device, dtype):
    return x.to(device=device, dtype=dtype) if torch.is_tensor(x) else x


def ensure_time_batch(x):
    if x.dim() == 2:
        return x.unsqueeze(1), True
    return x, False


def streaming_decode_controls(control, dt=DT):
    squeeze = control.dim() == 2
    if squeeze:
        control = control.unsqueeze(1)
    prev = torch.cat((control[:1], control[:-1]), dim=0)
    q = (prev + 5.0 * control) / 6.0
    qdot = (control - prev) / (2.0 * dt)
    qddot = (prev - control) / (dt ** 2)
    if squeeze:
        return q[:, 0], qdot[:, 0], qddot[:, 0]
    return q, qdot, qddot


def q75_to_pose_with_baseline_root(q75, baseline_root_pose, euler_seq='XYZ'):
    squeeze = q75.dim() == 2
    if squeeze:
        q75 = q75.unsqueeze(1)
        baseline_root_pose = baseline_root_pose.unsqueeze(1)
    time, batch = q75.shape[:2]
    pose, _tran = q75_to_pose_tran(q75.reshape(-1, Q75_DIM), euler_seq=euler_seq)
    pose = pose.reshape(time, batch, 24, 3, 3)
    pose[..., 0, :, :] = baseline_root_pose.to(pose.device, pose.dtype)
    return pose[:, 0] if squeeze else pose


def flatten_pose(pose):
    return pose.reshape(-1, 24, 3, 3)


def unflatten_fk(value, time, batch):
    return value.reshape(time, batch, *value.shape[1:])


def fk_position_from_pose(pose, body_model):
    squeeze = pose.dim() == 4
    if squeeze:
        pose = pose.unsqueeze(1)
    time, batch = pose.shape[:2]
    leaf, joint = root_relative_fk_targets(flatten_pose(pose), body_model)
    leaf = unflatten_fk(leaf, time, batch)
    joint = unflatten_fk(joint, time, batch)
    if squeeze:
        return leaf[:, 0], joint[:, 0]
    return leaf, joint


def fk_pva_from_q(q, qdot, qddot, baseline_root_pose, body_model, dt=DT, euler_seq='XYZ'):
    squeeze = q.dim() == 2
    if squeeze:
        q = q.unsqueeze(1)
        qdot = qdot.unsqueeze(1)
        qddot = qddot.unsqueeze(1)
        baseline_root_pose = baseline_root_pose.unsqueeze(1)
    q_plus = q.clone()
    q_minus = q.clone()
    q_plus[..., 6:] = q[..., 6:] + qdot[..., 6:] * dt + 0.5 * qddot[..., 6:] * (dt ** 2)
    q_minus[..., 6:] = q[..., 6:] - qdot[..., 6:] * dt + 0.5 * qddot[..., 6:] * (dt ** 2)
    pose = q75_to_pose_with_baseline_root(q, baseline_root_pose, euler_seq=euler_seq)
    pose_plus = q75_to_pose_with_baseline_root(q_plus, baseline_root_pose, euler_seq=euler_seq)
    pose_minus = q75_to_pose_with_baseline_root(q_minus, baseline_root_pose, euler_seq=euler_seq)
    leaf, joint = fk_position_from_pose(pose, body_model)
    leaf_plus, joint_plus = fk_position_from_pose(pose_plus, body_model)
    leaf_minus, joint_minus = fk_position_from_pose(pose_minus, body_model)
    leaf_vel = (leaf_plus - leaf_minus) / (2.0 * dt)
    joint_vel = (joint_plus - joint_minus) / (2.0 * dt)
    leaf_acc = (leaf_plus - 2.0 * leaf + leaf_minus) / (dt ** 2)
    joint_acc = (joint_plus - 2.0 * joint + joint_minus) / (dt ** 2)
    if squeeze:
        return {
            'leaf_pos': leaf[:, 0],
            'joint_pos': joint[:, 0],
            'leaf_vel': leaf_vel[:, 0],
            'joint_vel': joint_vel[:, 0],
            'leaf_acc': leaf_acc[:, 0],
            'joint_acc': joint_acc[:, 0],
            'pose': pose[:, 0],
        }
    return {
        'leaf_pos': leaf,
        'joint_pos': joint,
        'leaf_vel': leaf_vel,
        'joint_vel': joint_vel,
        'leaf_acc': leaf_acc,
        'joint_acc': joint_acc,
        'pose': pose,
    }


def fk_pva_from_pose(pose, body_model, dt=DT):
    leaf, joint = fk_position_from_pose(pose, body_model)
    return {
        'leaf_pos': leaf,
        'joint_pos': joint,
        'leaf_vel': finite_diff_same_length(leaf, 1, dt=dt),
        'joint_vel': finite_diff_same_length(joint, 1, dt=dt),
        'leaf_acc': finite_diff_same_length(leaf, 2, dt=dt),
        'joint_acc': finite_diff_same_length(joint, 2, dt=dt),
    }


class IK2Q75ControlModule(torch.nn.Module):
    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=512,
        state_dim=Q75_DIM,
        residual_scale=0.05,
        dropout=0.2,
        offset_init_scale=0.1,
        dt=DT,
        euler_seq='XYZ',
    ):
        super().__init__()
        if state_dim != Q75_DIM:
            raise ValueError('IK2Q75ControlModule state_dim must be q75 = 75.')
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.state_dim = int(state_dim)
        self.residual_scale = float(residual_scale)
        self.offset_init_scale = float(offset_init_scale)
        self.dt = float(dt)
        self.euler_seq = euler_seq
        self.input = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.init_encoder = torch.nn.Sequential(
            torch.nn.Linear(18 + self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.base_control = torch.nn.Parameter(torch.zeros(self.state_dim))
        self.control_head = torch.nn.Linear(self.hidden_size, self.state_dim)
        torch.nn.init.zeros_(self.control_head.weight)
        torch.nn.init.zeros_(self.control_head.bias)
        torch.nn.init.zeros_(self.init_encoder[-1].weight)
        torch.nn.init.zeros_(self.init_encoder[-1].bias)

    def initial_hidden(self, first_feature, offset_r=None):
        batch = first_feature.shape[0]
        if offset_r is None:
            offset = torch.zeros(batch, 18, device=first_feature.device, dtype=first_feature.dtype)
        else:
            offset = offset_r.to(device=first_feature.device, dtype=first_feature.dtype).reshape(batch, -1)
            if offset.shape[-1] != 18:
                raise ValueError(f'Expected offset_r flatten dim 18, got {offset.shape[-1]}.')
        init = self.init_encoder(torch.cat((offset, first_feature), dim=-1))
        return (init * self.offset_init_scale).unsqueeze(0)

    def _apply_output_mask(self, control):
        control = control.clone()
        control[..., :6] = 0.0
        return control

    def forward_sequence(self, features, offset_r=None):
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(1)
            if offset_r is not None and offset_r.dim() == 2:
                offset_r = offset_r.unsqueeze(0)
        if features.dim() != 3 or features.shape[-1] != self.input_size:
            raise ValueError(f'Expected features [T,B,{self.input_size}], got {tuple(features.shape)}.')
        batch_first = features.transpose(0, 1).contiguous()
        h0 = self.initial_hidden(batch_first[:, 0], offset_r=offset_r)
        encoded = self.input(batch_first)
        out, hidden = self.gru(encoded, h0)
        delta = self.control_head(out)
        control = self.base_control.view(1, 1, -1) + self.residual_scale * delta
        control = self._apply_output_mask(control).transpose(0, 1).contiguous()
        q, qdot, qddot = streaming_decode_controls(control, dt=self.dt)
        result = {
            'control': control,
            'q75': q,
            'qdot': qdot,
            'qddot': qddot,
            'delta_control': delta.transpose(0, 1).contiguous(),
            'delta_control_norm': delta.norm(dim=-1).mean(),
            'hidden': hidden,
        }
        if squeeze:
            for key, value in list(result.items()):
                if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] == 1:
                    result[key] = value[:, 0]
        return result


def ik2_q75_weights_for_preset(preset):
    common = {
        'q_body': 0.5,
        'qdot_body': 0.02,
        'qddot_body': 0.001,
        'pose_body_geodesic': 1.0,
        'fk_joint_pos': 10.0,
        'fk_leaf_pos': 5.0,
        'fk_joint_vel': 0.5,
        'fk_leaf_vel': 0.25,
        'fk_joint_acc': 0.02,
        'fk_leaf_acc': 0.01,
        'distill_pose_body': 0.03,
        'distill_fk_joint': 0.3,
        'control_delta_prior': 0.001,
    }
    if preset == 'pos_dominant':
        return common
    if preset == 'balanced':
        weights = dict(common)
        weights.update({
            'fk_joint_vel': 1.0,
            'fk_leaf_vel': 0.5,
            'fk_joint_acc': 0.05,
            'fk_leaf_acc': 0.025,
            'qdot_body': 0.05,
            'qddot_body': 0.003,
        })
        return weights
    if preset == 'acc_stronger':
        weights = dict(common)
        weights.update({
            'fk_joint_pos': 8.0,
            'fk_leaf_pos': 4.0,
            'fk_joint_vel': 1.5,
            'fk_leaf_vel': 0.75,
            'fk_joint_acc': 0.12,
            'fk_leaf_acc': 0.06,
            'qdot_body': 0.08,
            'qddot_body': 0.008,
        })
        return weights
    raise ValueError(f'Unsupported IK2 q75 loss preset: {preset}')


def ik2_q75_loss(output, record, weights, body_model, dt=DT, euler_seq='XYZ'):
    pred = output['q75']
    pred_tb, squeezed = ensure_time_batch(pred)
    device, dtype = pred_tb.device, pred_tb.dtype
    qdot = output['qdot'].unsqueeze(1) if squeezed else output['qdot']
    qddot = output['qddot'].unsqueeze(1) if squeezed else output['qddot']
    q_gt = normalize_record_tensor(record['q75_gt'], device, dtype)
    pose_gt = normalize_record_tensor(record['pose_gt'], device, dtype)
    baseline_root = normalize_record_tensor(record['baseline_root_pose'], device, dtype)
    if q_gt.dim() == 2:
        q_gt = q_gt.unsqueeze(1)
        pose_gt = pose_gt.unsqueeze(1)
        baseline_root = baseline_root.unsqueeze(1)
    qdot_gt = finite_diff_same_length(q_gt, 1, dt=dt)
    qddot_gt = finite_diff_same_length(q_gt, 2, dt=dt)
    pred_pose = q75_to_pose_with_baseline_root(pred_tb, baseline_root, euler_seq=euler_seq)
    pred_fk = fk_pva_from_q(pred_tb, qdot, qddot, baseline_root, body_model, dt=dt, euler_seq=euler_seq)
    target_fk = fk_pva_from_pose(pose_gt, body_model, dt=dt)
    losses = {
        'q_body': torch.nn.functional.smooth_l1_loss(pred_tb[..., 6:], q_gt[..., 6:]),
        'qdot_body': torch.nn.functional.smooth_l1_loss(qdot[..., 6:], qdot_gt[..., 6:]),
        'qddot_body': torch.nn.functional.smooth_l1_loss(qddot[..., 6:], qddot_gt[..., 6:]),
        'pose_body_geodesic': rotation_geodesic(pred_pose[..., 1:, :, :], pose_gt[..., 1:, :, :]).mean(),
        'fk_joint_pos': torch.nn.functional.smooth_l1_loss(pred_fk['joint_pos'], target_fk['joint_pos']),
        'fk_leaf_pos': torch.nn.functional.smooth_l1_loss(pred_fk['leaf_pos'], target_fk['leaf_pos']),
        'fk_joint_vel': torch.nn.functional.smooth_l1_loss(pred_fk['joint_vel'], target_fk['joint_vel']),
        'fk_leaf_vel': torch.nn.functional.smooth_l1_loss(pred_fk['leaf_vel'], target_fk['leaf_vel']),
        'fk_joint_acc': torch.nn.functional.smooth_l1_loss(pred_fk['joint_acc'], target_fk['joint_acc']),
        'fk_leaf_acc': torch.nn.functional.smooth_l1_loss(pred_fk['leaf_acc'], target_fk['leaf_acc']),
        'control_delta_prior': output['delta_control'].square().mean(),
    }
    if 'teacher_pose' in record:
        teacher_pose = normalize_record_tensor(record['teacher_pose'], device, dtype)
        if teacher_pose.dim() == 4:
            teacher_pose = teacher_pose.unsqueeze(1)
        teacher_fk = fk_pva_from_pose(teacher_pose, body_model, dt=dt)
        losses['distill_pose_body'] = rotation_geodesic(pred_pose[..., 1:, :, :], teacher_pose[..., 1:, :, :]).mean()
        losses['distill_fk_joint'] = torch.nn.functional.smooth_l1_loss(pred_fk['joint_pos'], teacher_fk['joint_pos'])
    else:
        zero = pred_tb.new_zeros(())
        losses['distill_pose_body'] = zero
        losses['distill_fk_joint'] = zero

    total = pred_tb.new_zeros(())
    for key, weight in weights.items():
        if key not in losses:
            raise KeyError(f'Loss weight {key} has no computed component.')
        total = total + float(weight) * losses[key]

    with torch.no_grad():
        joint_err = (pred_fk['joint_pos'] - target_fk['joint_pos']).norm(dim=-1)
        leaf_err = (pred_fk['leaf_pos'] - target_fk['leaf_pos']).norm(dim=-1)
        joint_vel_err = (pred_fk['joint_vel'] - target_fk['joint_vel']).norm(dim=-1)
        leaf_vel_err = (pred_fk['leaf_vel'] - target_fk['leaf_vel']).norm(dim=-1)
        joint_acc_err = (pred_fk['joint_acc'] - target_fk['joint_acc']).norm(dim=-1)
        leaf_acc_err = (pred_fk['leaf_acc'] - target_fk['leaf_acc']).norm(dim=-1)
        components = {key: value.detach() for key, value in losses.items()}
        components.update({
            'loss': total.detach(),
            'fk_joint_L2_cm': joint_err.mean().detach() * 100.0,
            'fk_leaf_L2_cm': leaf_err.mean().detach() * 100.0,
            'fk_joint_vel_L2_cm_s': joint_vel_err.mean().detach() * 100.0,
            'fk_leaf_vel_L2_cm_s': leaf_vel_err.mean().detach() * 100.0,
            'fk_joint_acc_L2_cm_s2': joint_acc_err.mean().detach() * 100.0,
            'fk_leaf_acc_L2_cm_s2': leaf_acc_err.mean().detach() * 100.0,
            'pose_body_geodesic_deg': torch.rad2deg(
                rotation_geodesic(pred_pose[..., 1:, :, :], pose_gt[..., 1:, :, :])
            ).mean().detach(),
            'qdot_body_rms': qdot[..., 6:].square().mean().sqrt().detach(),
            'qddot_body_rms': qddot[..., 6:].square().mean().sqrt().detach(),
            'delta_control_norm': output['delta_control_norm'].detach(),
        })
    return total, components


def selection_value(validation, metric='fk_pva'):
    losses = validation['loss']
    if metric == 'weighted_loss':
        return losses.get('loss', float('inf'))
    if metric == 'fk_pva':
        return (
            losses.get('fk_joint_L2_cm', float('inf'))
            + 0.5 * losses.get('fk_leaf_L2_cm', 0.0)
            + 0.05 * losses.get('fk_joint_vel_L2_cm_s', 0.0)
            + 0.025 * losses.get('fk_leaf_vel_L2_cm_s', 0.0)
            + 0.002 * losses.get('fk_joint_acc_L2_cm_s2', 0.0)
            + 0.001 * losses.get('fk_leaf_acc_L2_cm_s2', 0.0)
        )
    raise ValueError(f'Unsupported selection metric: {metric}')
