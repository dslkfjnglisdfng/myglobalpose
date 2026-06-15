import torch

import articulate as art
from full_curve_globalpose import IK2_REDUCED_JOINTS, IK2_REDUCED_PARENT_PAIRS, safe_from_to_rotation_matrix
from l4_tail_update_qstate import UniformCubicBSpline
from pl_curve import fit_uniform_cubic_spline_controls


DT = 1.0 / 60.0
STATE_DIM = 93
RRJ_DIM = 90
GRAVITY_DIM = 3
DEFAULT_LEAF_VERTEX_IDS = (1961, 5424, 1176, 4662, 411)
DEFAULT_ROOT_VERTEX_ID = 3021
DEFAULT_FK_VERTEX_MASK = DEFAULT_LEAF_VERTEX_IDS + (DEFAULT_ROOT_VERTEX_ID,)


def normalize_pose_state(state):
    return torch.cat((
        state[..., :RRJ_DIM],
        art.math.normalize_tensor(state[..., RRJ_DIM:STATE_DIM], avoid_nan=True),
    ), dim=-1)


def finite_diff(x, order):
    if order == 1:
        return x[1:] - x[:-1]
    if order == 2:
        return x[2:] - 2.0 * x[1:-1] + x[:-2]
    raise ValueError(order)


def rotation_matrix_to_6d(rotation):
    return rotation[..., :, :2].transpose(-1, -2).reshape(rotation.shape[:-2] + (6,))


def project_to_rotation_matrix(rotation):
    original_shape = rotation.shape
    flat = rotation.reshape(-1, 3, 3)
    u, _, vh = torch.linalg.svd(flat)
    projected = u.matmul(vh)
    det = torch.linalg.det(projected)
    if (det < 0).any():
        u = u.clone()
        u[det < 0, :, -1] *= -1.0
        projected = u.matmul(vh)
    return projected.reshape(original_shape)


def pose_state_target_from_pose(pose, R_seq, gR0, body_model, j_reduce=IK2_REDUCED_JOINTS):
    pose = pose.to(gR0.device, gR0.dtype)
    R_seq = R_seq.to(gR0.device, gR0.dtype)
    pose_body = pose.clone()
    pose_body[:, 0] = torch.eye(3, device=pose.device, dtype=pose.dtype)
    global_pose = body_model.forward_kinematics(pose_body)[0]
    rrj = rotation_matrix_to_6d(global_pose[:, j_reduce]).reshape(pose.shape[0], RRJ_DIM)
    root_from_imu = R_seq[:, 5].transpose(1, 2).matmul(pose[:, 0])
    g_pose = root_from_imu.transpose(1, 2).matmul(gR0.view(-1, 3, 1)).view(-1, 3)
    return normalize_pose_state(torch.cat((rrj, g_pose), dim=-1))


def fit_pose_controls(target):
    return fit_uniform_cubic_spline_controls(normalize_pose_state(target))


def padded_control_tail(controls, frame_idx, tail_len=4):
    tail = controls[max(0, frame_idx - tail_len + 1):frame_idx + 1]
    if tail.shape[0] < tail_len:
        pad = tail[:1].expand(tail_len - tail.shape[0], -1)
        tail = torch.cat((pad, tail), dim=0)
    return tail


def pose_control_tail_from_target(target, tail_len=4):
    controls = fit_pose_controls(target)
    return torch.stack([padded_control_tail(controls, idx, tail_len) for idx in range(controls.shape[0])])


def rrj_geodesic_deg(pred, target):
    pred_rot = art.math.r6d_to_rotation_matrix(pred[..., :RRJ_DIM].reshape(pred.shape[:-1] + (15, 6)).contiguous())
    target_rot = art.math.r6d_to_rotation_matrix(target[..., :RRJ_DIM].reshape(target.shape[:-1] + (15, 6)).contiguous())
    rel = pred_rot.transpose(-1, -2).matmul(target_rot)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def rrj_rotation_matrix_loss(pred, target):
    pred_rot = art.math.r6d_to_rotation_matrix(pred[..., :RRJ_DIM].reshape(pred.shape[:-1] + (15, 6)).contiguous())
    target_rot = art.math.r6d_to_rotation_matrix(target[..., :RRJ_DIM].reshape(target.shape[:-1] + (15, 6)).contiguous().to(pred.device, pred.dtype))
    return torch.nn.functional.smooth_l1_loss(pred_rot, target_rot)


def direction_cosine_loss(pred, target):
    pred = art.math.normalize_tensor(pred, avoid_nan=True)
    target = art.math.normalize_tensor(target.to(pred.device, pred.dtype), avoid_nan=True)
    return (1.0 - (pred * target).sum(dim=-1).clamp(-1.0, 1.0)).mean()


def ik2_parent_relative_loss(pred, target):
    pred_rot = art.math.r6d_to_rotation_matrix(pred[..., :RRJ_DIM].reshape(pred.shape[:-1] + (15, 6)).contiguous())
    target_rot = art.math.r6d_to_rotation_matrix(target[..., :RRJ_DIM].reshape(target.shape[:-1] + (15, 6)).to(pred.device, pred.dtype).contiguous())
    losses = []
    for child_idx, parent_idx in IK2_REDUCED_PARENT_PAIRS:
        pred_rel = pred_rot[..., parent_idx, :, :].transpose(-1, -2).matmul(pred_rot[..., child_idx, :, :])
        target_rel = target_rot[..., parent_idx, :, :].transpose(-1, -2).matmul(target_rot[..., child_idx, :, :])
        losses.append(torch.nn.functional.smooth_l1_loss(pred_rel, target_rel))
    return torch.stack(losses).mean() if losses else pred.new_zeros(())


def decode_pose_state(state, R_pelvis, gR0, body_model, j_reduce=IK2_REDUCED_JOINTS, j_ignore=(0, 7, 8, 10, 11, 20, 21, 22, 23)):
    state = normalize_pose_state(state)
    squeeze = state.dim() == 1
    if squeeze:
        state = state.unsqueeze(0)
        R_pelvis = R_pelvis.unsqueeze(0)
        gR0 = gR0.unsqueeze(0)
    rrj = art.math.r6d_to_rotation_matrix(state[..., :RRJ_DIM].contiguous()).reshape(state.shape[0], 15, 3, 3).cpu()
    glb_pose = torch.eye(3).repeat(state.shape[0], 24, 1, 1)
    glb_pose[:, j_reduce] = rrj
    pose = body_model.inverse_kinematics_R(glb_pose).view(state.shape[0], 24, 3, 3)
    pose[:, j_ignore] = torch.eye(3)
    root = []
    for idx in range(state.shape[0]):
        root.append(R_pelvis[idx].cpu().mm(safe_from_to_rotation_matrix(state[idx, RRJ_DIM:].cpu(), gR0[idx].cpu()).squeeze()))
    pose[:, 0] = torch.stack(root)
    pose = project_to_rotation_matrix(pose)
    return pose[0] if squeeze else pose


def pose_state_to_pose(state, R_pelvis, gR0, body_model, j_reduce=IK2_REDUCED_JOINTS, j_ignore=(0, 7, 8, 10, 11, 20, 21, 22, 23)):
    state = normalize_pose_state(state)
    squeeze = state.dim() == 1
    if squeeze:
        state = state.unsqueeze(0)
        R_pelvis = R_pelvis.unsqueeze(0)
        gR0 = gR0.unsqueeze(0)
    device, dtype = state.device, state.dtype
    R_pelvis = R_pelvis.to(device=device, dtype=dtype)
    gR0 = gR0.to(device=device, dtype=dtype)
    rrj = art.math.r6d_to_rotation_matrix(state[..., :RRJ_DIM].contiguous()).reshape(state.shape[0], 15, 3, 3)
    glb_pose = torch.eye(3, device=device, dtype=dtype).repeat(state.shape[0], 24, 1, 1)
    glb_pose[:, j_reduce] = rrj
    pose = body_model.inverse_kinematics_R(glb_pose).view(state.shape[0], 24, 3, 3)
    ignore = torch.as_tensor(j_ignore, device=device)
    pose[:, ignore] = torch.eye(3, device=device, dtype=dtype)
    root_align = safe_from_to_rotation_matrix(state[..., RRJ_DIM:], gR0).squeeze()
    if root_align.dim() == 2:
        root_align = root_align.unsqueeze(0)
    pose[:, 0] = R_pelvis.matmul(root_align)
    return pose[0] if squeeze else pose


def root_relative_fk_targets(pose, body_model, leaf_vertex_ids=DEFAULT_LEAF_VERTEX_IDS, root_vertex_id=DEFAULT_ROOT_VERTEX_ID):
    pose = pose.view(pose.shape[0], 24, 3, 3)
    _global_rot, joints, verts = body_model.forward_kinematics(pose, calc_mesh=True)
    root_rot = pose[:, 0]
    if verts.shape[1] == len(leaf_vertex_ids) + 1:
        leaf_world = verts[:, :len(leaf_vertex_ids)]
        root_world = verts[:, len(leaf_vertex_ids):len(leaf_vertex_ids) + 1]
    else:
        leaf_world = verts[:, list(leaf_vertex_ids)]
        root_world = verts[:, int(root_vertex_id):int(root_vertex_id) + 1]
    leaf_root = (leaf_world - root_world).matmul(root_rot)
    joint_root = (joints[:, 1:] - joints[:, :1]).matmul(root_rot)
    return leaf_root, joint_root


def fk_targets_from_state(state, R_pelvis, gR0, body_model):
    pose = pose_state_to_pose(state, R_pelvis, gR0, body_model)
    return root_relative_fk_targets(pose, body_model)


def temporal_l2(pred, target, order):
    if pred.shape[0] <= order:
        return pred.new_zeros(())
    return torch.nn.functional.smooth_l1_loss(finite_diff(pred, order), finite_diff(target.to(pred.device, pred.dtype), order))


def fk_leaf_joint_losses(output, record, body_model):
    pred = normalize_pose_state(output['state'])
    pose_gt = record.get('pose_gt')
    R_seq = record.get('RMB')
    gR0 = record.get('gR0')
    if pose_gt is None or R_seq is None or gR0 is None:
        zero = pred.new_zeros(())
        return {
            'fk_leaf_pos': zero,
            'fk_leaf_vel': zero,
            'fk_leaf_acc': zero,
            'fk_joint_pos': zero,
            'fk_leaf_L2_cm': zero,
            'fk_joint_L2_cm': zero,
            'fk_leaf_vel_L2_cm_per_frame': zero,
            'fk_leaf_acc_L2_cm_per_frame2': zero,
        }
    pose_gt = pose_gt.to(pred.device, pred.dtype)
    R_seq = R_seq.to(pred.device, pred.dtype)
    gR0 = gR0.to(pred.device, pred.dtype)
    pred_leaf, pred_joint = fk_targets_from_state(pred.reshape(-1, STATE_DIM), R_seq.reshape(-1, 6, 3, 3)[:, 5], gR0.reshape(-1, 3), body_model)
    target_leaf, target_joint = root_relative_fk_targets(pose_gt.reshape(-1, 24, 3, 3), body_model)
    pred_leaf = pred_leaf.reshape(pred.shape[:-1] + pred_leaf.shape[1:])
    pred_joint = pred_joint.reshape(pred.shape[:-1] + pred_joint.shape[1:])
    target_leaf = target_leaf.reshape(pred.shape[:-1] + target_leaf.shape[1:]).to(pred.device, pred.dtype)
    target_joint = target_joint.reshape(pred.shape[:-1] + target_joint.shape[1:]).to(pred.device, pred.dtype)
    leaf_err = (pred_leaf - target_leaf).norm(dim=-1)
    joint_err = (pred_joint - target_joint).norm(dim=-1)
    if pred_leaf.shape[0] > 1:
        leaf_vel_err = (finite_diff(pred_leaf, 1) - finite_diff(target_leaf, 1)).norm(dim=-1)
    else:
        leaf_vel_err = leaf_err.new_zeros(())
    if pred_leaf.shape[0] > 2:
        leaf_acc_err = (finite_diff(pred_leaf, 2) - finite_diff(target_leaf, 2)).norm(dim=-1)
    else:
        leaf_acc_err = leaf_err.new_zeros(())
    return {
        'fk_leaf_pos': torch.nn.functional.smooth_l1_loss(pred_leaf, target_leaf),
        'fk_leaf_vel': temporal_l2(pred_leaf, target_leaf, 1),
        'fk_leaf_acc': temporal_l2(pred_leaf, target_leaf, 2),
        'fk_joint_pos': torch.nn.functional.smooth_l1_loss(pred_joint, target_joint),
        'fk_leaf_L2_cm': leaf_err.mean() * 100.0,
        'fk_joint_L2_cm': joint_err.mean() * 100.0,
        'fk_leaf_vel_L2_cm_per_frame': leaf_vel_err.mean() * 100.0,
        'fk_leaf_acc_L2_cm_per_frame2': leaf_acc_err.mean() * 100.0,
    }


class NewPoseControlModule(torch.nn.Module):
    def __init__(
        self,
        input_size=174,
        state_dim=STATE_DIM,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.1,
        dt=DT,
        dropout=0.2,
        offset_init_scale=0.1,
    ):
        super().__init__()
        if state_dim != STATE_DIM:
            raise ValueError('newpose_ctrl_v1 state must be RRJ_control[90]+gR_pose_control[3] = 93D.')
        if tail_update != 4:
            raise ValueError('newpose_ctrl_v1 currently uses tail_len=4.')
        self.input_size = int(input_size)
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.tail_update = int(tail_update)
        self.residual_scale = float(residual_scale)
        self.dt = float(dt)
        self.offset_init_scale = float(offset_init_scale)
        self.input = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.cell = torch.nn.GRUCell(self.hidden_size, self.hidden_size)
        self.init_encoder = torch.nn.Sequential(
            torch.nn.Linear(18 + self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.base_control = torch.nn.Parameter(torch.zeros(self.state_dim))
        self.new_control = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.tail_delta = torch.nn.Linear(self.hidden_size, self.tail_update * self.state_dim)
        self.spline = UniformCubicBSpline(dt)
        self.reset_stream()
        torch.nn.init.zeros_(self.init_encoder[-1].weight)
        torch.nn.init.zeros_(self.init_encoder[-1].bias)
        torch.nn.init.zeros_(self.new_control.weight)
        torch.nn.init.zeros_(self.new_control.bias)
        torch.nn.init.zeros_(self.tail_delta.weight)
        torch.nn.init.zeros_(self.tail_delta.bias)
        with torch.no_grad():
            identity_r6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            self.base_control[:RRJ_DIM].copy_(identity_r6d.repeat(15))
            self.base_control[RRJ_DIM:STATE_DIM].copy_(torch.tensor([0.0, -1.0, 0.0]))

    def reset_stream(self, offset_r=None, init_feature=None):
        self.hidden = None
        if offset_r is not None and init_feature is not None:
            self.hidden = self._firstframe_hidden(offset_r, init_feature)
        self.control_buffer = None
        self.last_debug = {}

    def _firstframe_hidden(self, offset_r, init_feature):
        ref = next(self.init_encoder.parameters())
        feature = init_feature.detach().to(ref.device, ref.dtype)
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)
        offset = offset_r.detach().to(ref.device, ref.dtype)
        if offset.dim() == 2:
            offset = offset.unsqueeze(0)
        offset = offset.reshape(feature.shape[0], -1)
        if offset.shape[-1] != 18:
            raise ValueError(f'Expected offset_r flatten dim 18, got {offset.shape[-1]}.')
        if feature.shape[-1] != self.input_size:
            raise ValueError(f'Expected init feature dim {self.input_size}, got {feature.shape[-1]}.')
        return self.init_encoder(torch.cat((offset, feature), dim=-1)) * self.offset_init_scale

    def _initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _ghost(self, buffer, count=1):
        return buffer[:, -1:].expand(-1, int(count), -1).clone()

    def control_tail(self):
        if self.control_buffer is None:
            raise RuntimeError('control_tail requested before first step.')
        tail = self.control_buffer[:, -self.tail_update:, :]
        if tail.shape[1] < self.tail_update:
            pad = tail[:, :1].expand(-1, self.tail_update - tail.shape[1], -1)
            tail = torch.cat((pad, tail), dim=1)
        return tail

    def step(self, feature_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if feature_t.shape[-1] != self.input_size:
            raise ValueError(f'Expected NewPose feature dim {self.input_size}, got {feature_t.shape[-1]}.')
        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)
        z = torch.relu(self.input(feature_t))
        z = self.dropout(z)
        self.hidden = self.cell(z, self.hidden)
        base = self.base_control.view(1, -1).expand(feature_t.shape[0], -1)
        new_delta = self.new_control(self.hidden) * self.residual_scale
        new_control = normalize_pose_state(base + new_delta)
        if self.control_buffer is None:
            self.control_buffer = new_control.unsqueeze(1)
            tail_delta_norm = new_delta.norm(dim=-1).mean()
        else:
            frozen = self.control_buffer.detach()
            update_count = min(self.tail_update, frozen.shape[1])
            old_control = frozen[:, :-update_count]
            tail_control = frozen[:, -update_count:]
            tail_delta = self.tail_delta(self.hidden).reshape(feature_t.shape[0], self.tail_update, self.state_dim)[:, -update_count:] * self.residual_scale
            tail_control = normalize_pose_state(tail_control + tail_delta)
            self.control_buffer = torch.cat((old_control, tail_control, new_control.unsqueeze(1)), dim=1)
            tail_delta_norm = tail_delta.norm(dim=-1).mean()
        # The downstream state only uses spline output at the newest control
        # point with a repeated right boundary. Computing the whole prefix every
        # frame is O(T^2) on long evaluation sequences, so evaluate the local
        # stencil directly.
        current = self.control_buffer[:, -1]
        previous = self.control_buffer[:, -2] if self.control_buffer.shape[1] >= 2 else current
        state_t = normalize_pose_state((previous + 5.0 * current) / 6.0)
        dot_t = (current - previous) / (2.0 * self.dt)
        ddot_t = (previous - current) / (self.dt ** 2)
        result = {
            'state_t': state_t,
            'dot_t': dot_t,
            'ddot_t': ddot_t,
            'new_control_t': new_control,
            'control_tail_t': self.control_tail(),
            'control_point_prior_t': self.control_buffer.square().mean(),
            'new_delta_norm': new_delta.norm(dim=-1).mean(),
            'tail_delta_norm': tail_delta_norm,
            'buffer_length': int(self.control_buffer.shape[1]),
        }
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return result

    def forward_sequence(self, features, offset_r=None):
        squeeze_batch = features.dim() == 2
        if squeeze_batch:
            features = features.unsqueeze(1)
            if offset_r is not None and offset_r.dim() == 2:
                offset_r = offset_r.unsqueeze(0)
        self.reset_stream(offset_r=offset_r, init_feature=features[0])
        outputs = {key: [] for key in ('state', 'dot', 'ddot', 'new_control', 'control_tail')}
        priors, tails, deltas = [], [], []
        for idx in range(features.shape[0]):
            out = self.step(features[idx])
            outputs['state'].append(out['state_t'])
            outputs['dot'].append(out['dot_t'])
            outputs['ddot'].append(out['ddot_t'])
            outputs['new_control'].append(out['new_control_t'])
            outputs['control_tail'].append(out['control_tail_t'])
            priors.append(out['control_point_prior_t'])
            tails.append(out['tail_delta_norm'])
            deltas.append(out['new_delta_norm'])
        result = {key: torch.stack(value) for key, value in outputs.items()}
        result.update({
            'control_point_prior': torch.stack(priors).mean(),
            'tail_delta_norm': torch.stack(tails).mean(),
            'new_delta_norm': torch.stack(deltas).mean(),
            'control_shape': list(self.control_buffer.shape),
        })
        if squeeze_batch:
            for key, value in list(result.items()):
                if torch.is_tensor(value) and value.dim() >= 2:
                    result[key] = value[:, 0]
        return result


def newpose_default_weights():
    return {
        'control_RRJ': 1.0,
        'control_gR_pose': 0.5,
        'gt_control_RRJ': 1.0,
        'gt_control_gR_pose': 0.5,
        'control_RRJ_dot': 0.03,
        'control_gR_dot': 0.01,
        'control_RRJ_ddot': 0.003,
        'control_gR_ddot': 0.001,
        'state_RRJ': 0.5,
        'state_geodesic': 1.0,
        'state_gR_pose': 0.5,
        'parent_relative': 0.0,
        'state_dot': 0.01,
        'state_ddot': 0.001,
        'distill_ik2': 0.03,
        'control_point_prior': 0.0,
        'tail_update_prior': 0.0,
    }


def newpose_v2_fk_leaf_weights():
    weights = newpose_default_weights()
    weights.update({
        'control_RRJ': 0.5,
        'control_gR_pose': 0.25,
        'gt_control_RRJ': 0.5,
        'gt_control_gR_pose': 0.25,
        'state_RRJ': 0.2,
        'state_geodesic': 0.2,
        'state_gR_pose': 0.25,
        'distill_ik2': 0.0,
        'fk_leaf_pos': 10.0,
        'fk_leaf_vel': 2.0,
        'fk_leaf_acc': 0.5,
        'fk_joint_pos': 1.0,
    })
    return weights


def newpose_weights_for_preset(preset):
    if preset == 'v1':
        return newpose_default_weights()
    if preset == 'v2_fk_leaf':
        return newpose_v2_fk_leaf_weights()
    raise ValueError(f'Unsupported newpose loss preset: {preset}')


def newpose_loss(output, record, weights, body_model=None):
    pred = normalize_pose_state(output['state'])
    target = normalize_pose_state(record['newpose_target'].to(pred.device, pred.dtype))
    pred_tail = normalize_pose_state(output['control_tail'])
    target_tail = normalize_pose_state(record['newpose_target_control_tail'].to(pred.device, pred.dtype))
    pred_control = normalize_pose_state(output['new_control'])
    target_control = normalize_pose_state(target_tail[..., -1, :])
    losses = {
        'control_RRJ': torch.nn.functional.smooth_l1_loss(pred_tail[..., :RRJ_DIM], target_tail[..., :RRJ_DIM]),
        'control_gR_pose': direction_cosine_loss(pred_tail[..., RRJ_DIM:], target_tail[..., RRJ_DIM:]),
        'gt_control_RRJ': torch.nn.functional.smooth_l1_loss(pred_control[..., :RRJ_DIM], target_control[..., :RRJ_DIM]),
        'gt_control_gR_pose': direction_cosine_loss(pred_control[..., RRJ_DIM:], target_control[..., RRJ_DIM:]),
        'state_RRJ': torch.nn.functional.smooth_l1_loss(pred[..., :RRJ_DIM], target[..., :RRJ_DIM]),
        'state_geodesic': rrj_rotation_matrix_loss(pred, target),
        'state_gR_pose': direction_cosine_loss(pred[..., RRJ_DIM:], target[..., RRJ_DIM:]),
        'parent_relative': ik2_parent_relative_loss(pred, target),
        'control_point_prior': output['control_point_prior'],
        'tail_update_prior': output['tail_delta_norm'],
    }
    if pred_tail.shape[-2] >= 2:
        losses['control_RRJ_dot'] = torch.nn.functional.smooth_l1_loss(
            pred_tail[..., 1:, :RRJ_DIM] - pred_tail[..., :-1, :RRJ_DIM],
            target_tail[..., 1:, :RRJ_DIM] - target_tail[..., :-1, :RRJ_DIM],
        )
        losses['control_gR_dot'] = torch.nn.functional.smooth_l1_loss(
            pred_tail[..., 1:, RRJ_DIM:] - pred_tail[..., :-1, RRJ_DIM:],
            target_tail[..., 1:, RRJ_DIM:] - target_tail[..., :-1, RRJ_DIM:],
        )
    else:
        losses['control_RRJ_dot'] = pred.new_zeros(())
        losses['control_gR_dot'] = pred.new_zeros(())
    if pred_tail.shape[-2] >= 3:
        losses['control_RRJ_ddot'] = torch.nn.functional.smooth_l1_loss(
            finite_diff(pred_tail[..., :RRJ_DIM], 2),
            finite_diff(target_tail[..., :RRJ_DIM], 2),
        )
        losses['control_gR_ddot'] = torch.nn.functional.smooth_l1_loss(
            finite_diff(pred_tail[..., RRJ_DIM:], 2),
            finite_diff(target_tail[..., RRJ_DIM:], 2),
        )
    else:
        losses['control_RRJ_ddot'] = pred.new_zeros(())
        losses['control_gR_ddot'] = pred.new_zeros(())
    if pred.shape[0] >= 2:
        losses['state_dot'] = torch.nn.functional.smooth_l1_loss(
            pred[1:, ..., :RRJ_DIM] - pred[:-1, ..., :RRJ_DIM],
            target[1:, ..., :RRJ_DIM] - target[:-1, ..., :RRJ_DIM],
        )
    else:
        losses['state_dot'] = pred.new_zeros(())
    if pred.shape[0] >= 3:
        losses['state_ddot'] = torch.nn.functional.smooth_l1_loss(
            finite_diff(pred[..., :RRJ_DIM], 2),
            finite_diff(target[..., :RRJ_DIM], 2),
        )
    else:
        losses['state_ddot'] = pred.new_zeros(())
    if 'official_ik2_state' in record:
        base = normalize_pose_state(record['official_ik2_state'].to(pred.device, pred.dtype))
        losses['distill_ik2'] = torch.nn.functional.smooth_l1_loss(pred[..., :RRJ_DIM], base[..., :RRJ_DIM])
    else:
        losses['distill_ik2'] = pred.new_zeros(())
    if body_model is not None:
        losses.update(fk_leaf_joint_losses(output, record, body_model))
    else:
        zero = pred.new_zeros(())
        losses.update({
            'fk_leaf_pos': zero,
            'fk_leaf_vel': zero,
            'fk_leaf_acc': zero,
            'fk_joint_pos': zero,
            'fk_leaf_L2_cm': zero,
            'fk_joint_L2_cm': zero,
            'fk_leaf_vel_L2_cm_per_frame': zero,
            'fk_leaf_acc_L2_cm_per_frame2': zero,
        })
    total = pred.new_zeros(())
    for key, weight in weights.items():
        if key not in losses:
            raise KeyError(f'Loss weight {key} has no computed loss component.')
        total = total + losses[key] * weight
    components = {key: value.detach() for key, value in losses.items()}
    components.update({
        'loss': total.detach(),
        'new_delta_norm': output['new_delta_norm'].detach(),
        'tail_delta_norm': output['tail_delta_norm'].detach(),
        'control_RRJ_geodesic_deg': rrj_geodesic_deg(pred_control, target_control).mean().detach(),
        'state_RRJ_geodesic_deg': rrj_geodesic_deg(pred, target).mean().detach(),
    })
    return total, components


def checkpoint_selection_value(validation, metric='control_pose_physical'):
    losses = validation['loss']
    if metric == 'weighted_loss':
        return losses.get('loss', float('inf'))
    if metric == 'control_pose_physical':
        return (
            losses.get('gt_control_RRJ', 0.0)
            + 0.5 * losses.get('gt_control_gR_pose', 0.0)
            + 0.5 * losses.get('control_RRJ', 0.0)
            + 0.25 * losses.get('control_gR_pose', 0.0)
        )
    if metric == 'decoded_pose_physical':
        return losses.get('state_RRJ', 0.0) + losses.get('state_geodesic', 0.0) + 0.5 * losses.get('state_gR_pose', 0.0)
    if metric == 'fk_leaf_physical':
        leaf = losses.get('fk_leaf_L2_cm', float('inf'))
        return (
            leaf
            + 0.25 * losses.get('fk_leaf_vel_L2_cm_per_frame', 0.0)
            + 0.10 * losses.get('fk_leaf_acc_L2_cm_per_frame2', 0.0)
            + 0.10 * losses.get('fk_joint_L2_cm', 0.0)
            + 0.05 * losses.get('state_gR_pose', 0.0)
        )
    raise ValueError(f'Unsupported selection metric: {metric}')
