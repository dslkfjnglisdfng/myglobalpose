import math

import torch

import articulate as art
from l4_tail_update_qstate import UniformCubicBSpline


PL_LEGACY_INPUT_SIZE = 84
PL_SMOOTH_RESIDUAL_INPUT_SIZE = 102
PL_OFFSET_AWARE_INPUT_SIZE = 156
PL_BONE_AUX_DIM = 30
PL_BONE_LEAF_JOINT_IDS = (18, 19, 4, 5, 15)
PL_LEARNED_OFFSET_ACC_CONTRACT = (
    'newpl_v7_learned_offset_accaux keeps the official 84D PL frame input and '
    '18D pRB[15]+gR1[3] output. learned_offset is r_BS[6,3]: sensor origin '
    'position relative to the mapped body/sensor frame B, expressed in B. '
    'Root-frame lever acceleration uses R_RB from the PL input and '
    'alpha_RB x r_RB + omega_RB x (omega_RB x r_RB). The acceleration loss '
    'compares non-root minus root accelerations in the root frame and does not '
    'use global translation or real offset GT.'
)
PL_LEARNED_LEAF_OFFSET_LOCAL_ACC_CONTRACT = (
    'newpl_v7b_local_accaux keeps the official 84D PL frame input and 18D '
    'pRB[15]+gR1[3] output with init18 pRL[15]+gR0[3]. It learns leaf-only '
    'r_BS[5,3] offsets, bounded by offset_max*tanh(raw_leaf_offset). The '
    'acceleration auxiliary loss stays in the PL root frame but no longer '
    'subtracts root acceleration: root gyro wRB[5] and alpha_root=dwRB[5]/dt '
    'correct pRB/pRBdot/pRBddot for the rotating root frame, then each leaf '
    'gyro wRB[i] and alpha_i explain lever acceleration '
    'alpha_i x r_i + w_i x (w_i x r_i). DIP trans/root velocity and real '
    'offset GT are not used.'
)
CONTROL_FIT_DEFAULT_MODE = 'derivative_aware_v1'
CONTROL_FIT_DERIVATIVE_AWARE_WEIGHTS = {
    'position': 1.0,
    'velocity': 0.03,
    'acceleration': 0.0003,
    'ridge': 1e-6,
}
CONTROL_FIT_DT = 1.0 / 60.0


def _first_difference(x, dt=1.0 / 60.0):
    if x.shape[0] <= 1:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    out[1:] = (x[1:] - x[:-1]) / float(dt)
    return out


def _central_difference_time(x, dt=1.0 / 60.0):
    if x.shape[0] <= 1:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    if x.shape[0] == 2:
        diff = (x[1:] - x[:-1]) / float(dt)
        out[0] = diff[0]
        out[1] = diff[0]
        return out
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * float(dt))
    out[0] = (x[1] - x[0]) / float(dt)
    out[-1] = (x[-1] - x[-2]) / float(dt)
    return out


def _root_relative_rotations(R):
    RRB = R[5].t().matmul(R[:5])
    full = torch.eye(3, device=R.device, dtype=R.dtype).expand(6, 3, 3).clone()
    full[:5] = RRB
    return RRB, full


def _root_relative_offset(offset_r, full_RRB):
    offset = offset_r.to(device=full_RRB.device, dtype=full_RRB.dtype).view(6, 3)
    return offset.unsqueeze(-2).matmul(full_RRB).squeeze(-2)


def _lever_terms(wRB, alphaRB, offset_r, full_RRB):
    rRB = _root_relative_offset(offset_r, full_RRB)
    tangent = torch.cross(alphaRB, rRB, dim=-1)
    centripetal = torch.cross(wRB, torch.cross(wRB, rRB, dim=-1), dim=-1)
    lever = tangent + centripetal
    return tangent, centripetal, lever


def pl_input_feature(a, w, R):
    aRB = a.mm(R[5])
    wRB = w.mm(R[5])
    RRB = R[5].t().matmul(R[:5])
    gR0 = -R[5, 1]
    return torch.cat((aRB.ravel(), wRB.ravel(), RRB.ravel(), gR0))


def causal_iir_lowpass_sequence(x, cutoff_hz=20.0, fs=60.0):
    """Zero-lookahead first-order low-pass filter along time.

    The filter is intended as a lightweight realtime IMU smoother.  It starts
    from the first sample, so cache generation matches streaming inference:
    output[t] depends only on samples <= t.
    """
    x = x.float()
    if x.shape[0] <= 1:
        return x.clone()
    cutoff_hz = float(cutoff_hz)
    fs = float(fs)
    if cutoff_hz <= 0.0:
        raise ValueError(f'cutoff_hz must be positive, got {cutoff_hz}.')
    dt = 1.0 / fs
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (tau + dt)
    out = torch.empty_like(x)
    out[0] = x[0]
    for idx in range(1, x.shape[0]):
        out[idx] = out[idx - 1] + alpha * (x[idx] - out[idx - 1])
    return out


def pl_smooth_residual_sequence_features(
    a,
    w,
    R,
    cutoff_hz=20.0,
    fs=60.0,
    smooth_mode='causal_iir',
    filter_order=2,
):
    """Return 102D realtime smooth+residual NewPL features.

    Layout:
      aRB_smooth[18] + aRB_residual[18] + wRB[18] + RRB[45] + gR0[3].

    `aRB_residual = aRB_raw - aRB_smooth`, so the high-frequency acceleration
    removed by the realtime smoother is still available to the model instead
    of being discarded.  Both terms are expressed in the pelvis/root frame used
    by the legacy PL input.
    """
    a = a.float()
    if smooth_mode == 'causal_iir':
        smooth_a = causal_iir_lowpass_sequence(a, cutoff_hz=cutoff_hz, fs=fs)
    elif smooth_mode == 'causal_butterworth':
        from l4_sensor_offset_utils import causal_butterworth_lowpass_sequence
        smooth_a = causal_butterworth_lowpass_sequence(
            a,
            fs=fs,
            cutoff_hz=cutoff_hz,
            order=filter_order,
        )
    else:
        raise ValueError(f'Unsupported smooth_residual smooth_mode={smooth_mode}.')
    features = []
    for i in range(a.shape[0]):
        aRB_raw = a[i].mm(R[i, 5])
        aRB_smooth = smooth_a[i].mm(R[i, 5])
        aRB_residual = aRB_raw - aRB_smooth
        wRB = w[i].mm(R[i, 5])
        RRB = R[i, 5].t().matmul(R[i, :5])
        gR0 = -R[i, 5, 1]
        features.append(torch.cat((
            aRB_smooth.ravel(),
            aRB_residual.ravel(),
            wRB.ravel(),
            RRB.ravel(),
            gR0,
        )))
    return torch.stack(features).float()


def pl_offset_aware_frame_feature(a, w, R, offset_r, prev_wRB=None, dt=1.0 / 60.0):
    aRB = a.mm(R[5])
    wRB = w.mm(R[5])
    RRB, full_RRB = _root_relative_rotations(R)
    gR0 = -R[5, 1]
    if prev_wRB is None:
        alphaRB = torch.zeros_like(wRB)
    else:
        alphaRB = (wRB - prev_wRB.to(device=wRB.device, dtype=wRB.dtype)) / float(dt)
    a_tangent, a_centripetal, a_lever = _lever_terms(wRB, alphaRB, offset_r, full_RRB)
    a_corr = aRB - a_lever
    feature = torch.cat((
        aRB.ravel(),
        a_corr.ravel(),
        a_tangent.ravel(),
        a_centripetal.ravel(),
        a_lever.ravel(),
        wRB.ravel(),
        RRB.ravel(),
        gR0,
    ))
    return feature, wRB.detach()


def pl_offset_aware_sequence_features(a, w, R, offset_r, dt=1.0 / 60.0):
    raw_features = []
    wRB_seq = torch.stack([w[i].mm(R[i, 5]) for i in range(w.shape[0])])
    alpha_seq = _first_difference(wRB_seq, dt=dt)
    for i in range(a.shape[0]):
        aRB = a[i].mm(R[i, 5])
        RRB, full_RRB = _root_relative_rotations(R[i])
        gR0 = -R[i, 5, 1]
        a_tangent, a_centripetal, a_lever = _lever_terms(wRB_seq[i], alpha_seq[i], offset_r, full_RRB)
        a_corr = aRB - a_lever
        raw_features.append(torch.cat((
            aRB.ravel(),
            a_corr.ravel(),
            a_tangent.ravel(),
            a_centripetal.ravel(),
            a_lever.ravel(),
            wRB_seq[i].ravel(),
            RRB.ravel(),
            gR0,
        )))
    return torch.stack(raw_features).float()


def replace_offset_aware_feature_offset(feature, offset_r, dt=1.0 / 60.0):
    """Rebuild offset-dependent 156D PL feature blocks with a different r_JS.

    The offset-aware layout stores offset-independent raw terms:
    aRB[18], wRB[18], RRB[45], and gR0[3].  This helper preserves those
    terms and recomputes a_corr/tangent/centripetal/lever for a replacement
    sequence-level r_JS.  It supports [T,156] and [T,B,156] features.
    """
    if feature.shape[-1] != PL_OFFSET_AWARE_INPUT_SIZE:
        raise ValueError(
            f'replace_offset_aware_feature_offset expects last dim '
            f'{PL_OFFSET_AWARE_INPUT_SIZE}, got {feature.shape[-1]}.'
        )
    if feature.dim() not in (2, 3):
        raise ValueError(f'Expected feature shape [T,D] or [T,B,D], got {tuple(feature.shape)}.')
    squeeze_batch = feature.dim() == 2
    feat = feature.unsqueeze(1) if squeeze_batch else feature
    t, b = feat.shape[:2]
    device, dtype = feat.device, feat.dtype

    offset = offset_r.to(device=device, dtype=dtype)
    if offset.dim() == 1:
        offset = offset.view(1, 6, 3).expand(b, -1, -1)
    elif offset.dim() == 2:
        offset = offset.view(1, 6, 3).expand(b, -1, -1)
    elif offset.dim() == 3:
        if offset.shape[0] != b:
            if offset.shape[0] == 1:
                offset = offset.expand(b, -1, -1)
            else:
                raise ValueError(f'Offset batch={offset.shape[0]} does not match feature batch={b}.')
    else:
        raise ValueError(f'Expected offset_r shape [18], [6,3], or [B,6,3], got {tuple(offset.shape)}.')
    if offset.shape[-2:] != (6, 3):
        raise ValueError(f'Expected offset_r shape ending [6,3], got {tuple(offset.shape)}.')

    aRB = feat[..., :18].reshape(t, b, 6, 3)
    wRB = feat[..., 90:108].reshape(t, b, 6, 3)
    RRB = feat[..., 108:153].reshape(t, b, 5, 3, 3)
    gR0 = feat[..., 153:156]

    alpha = _first_difference(wRB, dt=dt)
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3).expand(t, b, 1, 3, 3)
    full_RRB = torch.cat((RRB, eye), dim=2)
    rRB = offset.view(1, b, 6, 1, 3).matmul(full_RRB).squeeze(-2)
    tangent = torch.cross(alpha, rRB, dim=-1)
    centripetal = torch.cross(wRB, torch.cross(wRB, rRB, dim=-1), dim=-1)
    lever = tangent + centripetal
    a_corr = aRB - lever
    out = torch.cat((
        aRB.reshape(t, b, 18),
        a_corr.reshape(t, b, 18),
        tangent.reshape(t, b, 18),
        centripetal.reshape(t, b, 18),
        lever.reshape(t, b, 18),
        wRB.reshape(t, b, 18),
        RRB.reshape(t, b, 45),
        gR0,
    ), dim=-1)
    return out[:, 0] if squeeze_batch else out


def split_pl_feature(feature):
    if feature.shape[-1] == PL_OFFSET_AWARE_INPUT_SIZE:
        RRB = feature[..., 108:153].reshape(feature.shape[:-1] + (5, 3, 3))
        gR0 = feature[..., 153:156]
    elif feature.shape[-1] == PL_SMOOTH_RESIDUAL_INPUT_SIZE:
        RRB = feature[..., 54:99].reshape(feature.shape[:-1] + (5, 3, 3))
        gR0 = feature[..., 99:102]
    else:
        RRB = feature[..., 36:81].reshape(feature.shape[:-1] + (5, 3, 3))
        gR0 = feature[..., 81:84]
    return RRB, gR0


def split_legacy_pl_imu_feature(feature):
    if feature.shape[-1] != PL_LEGACY_INPUT_SIZE:
        raise ValueError(f'Expected legacy PL feature dim {PL_LEGACY_INPUT_SIZE}, got {feature.shape[-1]}.')
    aRB = feature[..., :18].reshape(feature.shape[:-1] + (6, 3))
    wRB = feature[..., 18:36].reshape(feature.shape[:-1] + (6, 3))
    RRB = feature[..., 36:81].reshape(feature.shape[:-1] + (5, 3, 3))
    gR0 = feature[..., 81:84]
    return aRB, wRB, RRB, gR0


def _atanh_clamped(x, eps=1e-6):
    x = x.clamp(-1.0 + float(eps), 1.0 - float(eps))
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def bounded_offset_from_raw(raw_offset, offset_max=0.30):
    return float(offset_max) * torch.tanh(raw_offset)


def raw_offset_from_bounded(offset, offset_max=0.30):
    scale = max(float(offset_max), 1e-8)
    return _atanh_clamped(offset / scale)


def root_frame_offsets_from_rrb(offset_b, RRB):
    """Map learned r_BS into the PL root frame with the existing row-vector contract.

    `RRB` is R_RB for the five non-root sensor/body frames.  The root sensor
    uses identity.  The returned r_RB has shape prefix+[6,3].
    """
    if offset_b.shape[-2:] != (6, 3):
        raise ValueError(f'Expected offset_b shape [6,3], got {tuple(offset_b.shape)}.')
    prefix = RRB.shape[:-3]
    eye = torch.eye(3, device=RRB.device, dtype=RRB.dtype).expand(prefix + (1, 3, 3))
    full_RRB = torch.cat((RRB, eye), dim=-3)
    view_shape = (1,) * len(prefix) + (6, 1, 3)
    return offset_b.to(RRB.device, RRB.dtype).view(view_shape).matmul(full_RRB).squeeze(-2)


def leaf_root_frame_offsets_from_rrb(leaf_offset_b, RRB):
    """Map leaf-only r_BS into the PL root frame.

    Supports a global [5,3] offset and diagnostic per-batch [B,5,3] offsets.
    """
    if leaf_offset_b.shape[-2:] != (5, 3):
        raise ValueError(f'Expected leaf_offset_b shape [5,3], got {tuple(leaf_offset_b.shape)}.')
    prefix = RRB.shape[:-3]
    offset = leaf_offset_b.to(RRB.device, RRB.dtype)
    if offset.dim() == 2:
        view_shape = (1,) * len(prefix) + (5, 1, 3)
    elif offset.dim() == 3:
        if not prefix or offset.shape[0] != prefix[-1]:
            raise ValueError(
                f'Batched leaf offsets require shape [B,5,3] matching RRB batch prefix; '
                f'got offset={tuple(offset.shape)}, RRB prefix={tuple(prefix)}.'
            )
        view_shape = (1,) * (len(prefix) - 1) + (offset.shape[0], 5, 1, 3)
    else:
        raise ValueError(f'Expected leaf_offset_b shape [5,3] or [B,5,3], got {tuple(offset.shape)}.')
    return offset.view(view_shape).matmul(RRB).squeeze(-2)


def _interior_time_slice(x):
    if x.shape[0] >= 3:
        return slice(1, -1)
    if x.shape[0] == 2:
        return slice(1, None)
    return slice(None)


def learned_offset_imu_acc_terms(output, features, learned_offset, dt=1.0 / 60.0):
    """Return root-relative IMU acceleration residual terms for v7 diagnostics.

    The loss intentionally stays in the official PL root frame.  This frame is
    non-inertial, so the residual is a diagnostic consistency proxy rather than
    a global acceleration target.  No global translation or real offset label is
    used.
    """
    aRB, wRB, RRB, _ = split_legacy_pl_imu_feature(features)
    pred_ddot = output['plddot'][..., :15].reshape(output['plddot'].shape[:-1] + (5, 3))
    aRB = aRB.to(pred_ddot.device, pred_ddot.dtype)
    wRB = wRB.to(pred_ddot.device, pred_ddot.dtype)
    RRB = RRB.to(pred_ddot.device, pred_ddot.dtype)
    rRB = root_frame_offsets_from_rrb(learned_offset, RRB)
    alpha = _first_difference(wRB, dt=dt)
    tangent = torch.cross(alpha, rRB, dim=-1)
    centripetal = torch.cross(wRB, torch.cross(wRB, rRB, dim=-1), dim=-1)
    lever = tangent + centripetal
    pred_rel = pred_ddot + lever[..., :5, :] - lever[..., 5:6, :]
    obs_rel = aRB[..., :5, :] - aRB[..., 5:6, :]
    if pred_rel.shape[0] >= 3:
        valid = slice(1, -1)
    elif pred_rel.shape[0] == 2:
        valid = slice(1, None)
    else:
        valid = slice(None)
    residual = pred_rel[valid] - obs_rel[valid].to(pred_rel.device, pred_rel.dtype)
    return {
        'pred_rel_acc': pred_rel[valid],
        'obs_rel_acc': obs_rel[valid].to(pred_rel.device, pred_rel.dtype),
        'residual': residual,
        'offset_root_frame': rRB,
        'lever': lever,
    }


def learned_offset_imu_acc_loss(output, features, learned_offset, dt=1.0 / 60.0, acc_scale=30.0):
    terms = learned_offset_imu_acc_terms(output, features, learned_offset, dt=dt)
    pred = terms['pred_rel_acc']
    obs = terms['obs_rel_acc']
    scale = max(float(acc_scale), 1e-6)
    if pred.numel() == 0:
        loss = output['pl'].new_zeros(())
        rms = output['pl'].new_zeros(())
        l2 = output['pl'].new_zeros(())
    else:
        loss = torch.nn.functional.smooth_l1_loss(pred / scale, obs / scale)
        residual = pred - obs
        l2 = residual.norm(dim=-1).mean()
        rms = residual.square().mean().sqrt()
    return loss, {
        'imu_acc': loss.detach(),
        'imu_acc_l2_mps2': l2.detach(),
        'imu_acc_rms_mps2': rms.detach(),
        'imu_acc_available': loss.new_tensor(1.0),
    }


def learned_leaf_offset_local_imu_acc_terms(
    output,
    features,
    learned_leaf_offset,
    dt=1.0 / 60.0,
    gravity_mode='none',
    gravity_magnitude=9.81,
):
    """Return v7b leaf-local acceleration residual terms in the PL root frame.

    This helper uses root gyro to correct derivatives of pRB, which are decoded
    in a rotating root frame. It intentionally does not subtract root
    acceleration and does not require global translation.
    """
    aRB, wRB, RRB, gR0 = split_legacy_pl_imu_feature(features)
    pl = output['pl']
    pdot = output['pldot']
    pddot = output['plddot']
    p = pl[..., :15].reshape(pl.shape[:-1] + (5, 3))
    pdot = pdot[..., :15].reshape(pdot.shape[:-1] + (5, 3))
    pddot = pddot[..., :15].reshape(pddot.shape[:-1] + (5, 3))
    aRB = aRB.to(p.device, p.dtype)
    wRB = wRB.to(p.device, p.dtype)
    RRB = RRB.to(p.device, p.dtype)
    gR0 = gR0.to(p.device, p.dtype)

    alpha = _central_difference_time(wRB, dt=dt)
    omega_root = wRB[..., 5:6, :]
    alpha_root = alpha[..., 5:6, :]
    coriolis = 2.0 * torch.cross(omega_root, pdot, dim=-1)
    tangential_root = torch.cross(alpha_root, p, dim=-1)
    centripetal_root = torch.cross(omega_root, torch.cross(omega_root, p, dim=-1), dim=-1)
    anchor_acc = pddot + coriolis + tangential_root + centripetal_root

    rRB = leaf_root_frame_offsets_from_rrb(learned_leaf_offset, RRB)
    omega_leaf = wRB[..., :5, :]
    alpha_leaf = alpha[..., :5, :]
    leaf_tangent = torch.cross(alpha_leaf, rRB, dim=-1)
    leaf_centripetal = torch.cross(omega_leaf, torch.cross(omega_leaf, rRB, dim=-1), dim=-1)
    offset_acc = leaf_tangent + leaf_centripetal
    pred = anchor_acc + offset_acc
    obs = aRB[..., :5, :]

    if gravity_mode not in ('none', 'minus_gR0', 'plus_gR0'):
        raise ValueError(f'Unsupported gravity_mode={gravity_mode}.')
    if gravity_mode == 'minus_gR0':
        pred = pred - float(gravity_magnitude) * gR0.unsqueeze(-2)
    elif gravity_mode == 'plus_gR0':
        pred = pred + float(gravity_magnitude) * gR0.unsqueeze(-2)

    valid = _interior_time_slice(pred)
    residual = pred[valid] - obs[valid].to(pred.device, pred.dtype)
    return {
        'pred_leaf_acc': pred[valid],
        'obs_leaf_acc': obs[valid].to(pred.device, pred.dtype),
        'residual': residual,
        'anchor_acc': anchor_acc[valid],
        'offset_acc': offset_acc[valid],
        'root_coriolis': coriolis[valid],
        'root_tangential': tangential_root[valid],
        'root_centripetal': centripetal_root[valid],
        'offset_root_frame': rRB,
        'gravity_mode': gravity_mode,
    }


def learned_leaf_offset_local_imu_acc_loss(
    output,
    features,
    learned_leaf_offset,
    dt=1.0 / 60.0,
    acc_scale=30.0,
    gravity_mode='none',
    gravity_magnitude=9.81,
):
    terms = learned_leaf_offset_local_imu_acc_terms(
        output,
        features,
        learned_leaf_offset,
        dt=dt,
        gravity_mode=gravity_mode,
        gravity_magnitude=gravity_magnitude,
    )
    pred = terms['pred_leaf_acc']
    obs = terms['obs_leaf_acc']
    scale = max(float(acc_scale), 1e-6)
    if pred.numel() == 0:
        loss = output['pl'].new_zeros(())
        rms = output['pl'].new_zeros(())
        l2 = output['pl'].new_zeros(())
    else:
        loss = torch.nn.functional.smooth_l1_loss(pred / scale, obs / scale)
        residual = pred - obs
        l2 = residual.norm(dim=-1).mean()
        rms = residual.square().mean().sqrt()
    return loss, {
        'local_imu_acc': loss.detach(),
        'local_imu_acc_l2_mps2': l2.detach(),
        'local_imu_acc_rms_mps2': rms.detach(),
        'local_imu_acc_available': loss.new_tensor(1.0),
    }


def rotation_matrix_to_6d(rotation):
    return rotation[..., :, :2].transpose(-1, -2).reshape(rotation.shape[:-2] + (6,))


def pl_bone6d_base_from_feature(feature):
    RRB, _ = split_pl_feature(feature)
    return rotation_matrix_to_6d(RRB).reshape(feature.shape[:-1] + (PL_BONE_AUX_DIM,))


def pl_bone6d_target_from_pose(pose, body_model, leaf_joint_ids=PL_BONE_LEAF_JOINT_IDS):
    """Return 5 root-relative bone orientations as 6D rotations.

    Contract: rotations use column-vector convention, `R_AB` maps frame B into
    frame A.  The target is `R_RB = R_WR^T R_WB`: each leaf bone/body frame B
    expressed in the root frame R.  This is an aux target only; it is not the
    official 18D PL output.
    """
    if pose.dim() == 3:
        pose = pose.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    pose = pose.to(device=body_model._J.device, dtype=body_model._J.dtype).view(-1, 24, 3, 3)
    global_pose = body_model.forward_kinematics(pose)[0]
    root = global_pose[:, 0]
    leaf = global_pose[:, list(leaf_joint_ids)]
    root_relative = root.transpose(1, 2).unsqueeze(1).matmul(leaf)
    target = rotation_matrix_to_6d(root_relative).reshape(pose.shape[0], PL_BONE_AUX_DIM)
    return target[0] if squeeze else target


def pl_init_feature(offset_r, pRL, gR0):
    offset = offset_r.reshape(-1)
    if offset.shape[-1] != 18:
        raise ValueError(f'Expected offset_r flatten dim 18, got {offset.shape[-1]}.')
    return torch.cat((offset, pRL.reshape(-1), gR0.reshape(-1))).float()


def pl_init_feature_from_pose(offset_r, pose, body_model):
    ref = body_model._J
    pose = pose.to(device=ref.device, dtype=ref.dtype).view(1, 24, 3, 3)
    _, _, verts = body_model.forward_kinematics(pose, calc_mesh=True)
    pRL = (verts[0, :5] - verts[0, 5:]).mm(pose[0, 0]).ravel()
    gR0 = -pose[0, 0, 1]
    return pl_init_feature(offset_r.to(device=ref.device, dtype=ref.dtype), pRL, gR0).cpu()


def normalize_gravity(pl_output):
    return torch.cat((
        pl_output[..., :15],
        art.math.normalize_tensor(pl_output[..., 15:], avoid_nan=True),
    ), dim=-1)


def control_fit_contract():
    return {
        'mode': CONTROL_FIT_DEFAULT_MODE,
        'dt': CONTROL_FIT_DT,
        'weights': dict(CONTROL_FIT_DERIVATIVE_AWARE_WEIGHTS),
        'objective': (
            'min_C wp||S C - x||^2 + wv||D1 C - fd_dot(x)||^2 + '
            'wa||D2 C - fd_ddot(x)||^2 + wr||C||^2'
        ),
        'finite_difference': {
            'velocity': 'central difference with one-sided endpoints, divided by dt',
            'acceleration': 'three-point second difference with endpoint copy, divided by dt^2',
        },
        'note': 'Use fit_uniform_cubic_spline_controls_position_only for historical exact sample reconstruction.',
    }


def _central_difference_first(samples, dt=CONTROL_FIT_DT):
    if samples.shape[0] <= 1:
        return torch.zeros_like(samples)
    out = torch.zeros_like(samples)
    out[1:-1] = (samples[2:] - samples[:-2]) / (2.0 * float(dt))
    out[0] = (samples[1] - samples[0]) / float(dt)
    out[-1] = (samples[-1] - samples[-2]) / float(dt)
    return out


def _central_difference_second(samples, dt=CONTROL_FIT_DT):
    if samples.shape[0] <= 2:
        return torch.zeros_like(samples)
    out = torch.zeros_like(samples)
    out[1:-1] = (samples[2:] - 2.0 * samples[1:-1] + samples[:-2]) / (float(dt) ** 2)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def _uniform_cubic_spline_operators(t, device, dtype, dt=CONTROL_FIT_DT):
    mat = torch.zeros((t, t), device=device, dtype=dtype)
    d1 = torch.zeros_like(mat)
    d2 = torch.zeros_like(mat)
    idx = torch.arange(t, device=device)
    mat[idx, idx] = 4.0 / 6.0
    left = torch.clamp(idx - 1, min=0)
    right = torch.clamp(idx + 1, max=t - 1)
    mat[idx, left] += 1.0 / 6.0
    mat[idx, right] += 1.0 / 6.0
    d1[idx, right] += 1.0 / (2.0 * float(dt))
    d1[idx, left] -= 1.0 / (2.0 * float(dt))
    d2[idx, left] += 1.0 / (float(dt) ** 2)
    d2[idx, idx] -= 2.0 / (float(dt) ** 2)
    d2[idx, right] += 1.0 / (float(dt) ** 2)
    return mat, d1, d2


def fit_uniform_cubic_spline_controls_position_only(samples):
    """Fit control points C so UniformCubicBSpline(C) reconstructs samples.

    The spline decoder used in PLCurve evaluates
      q[i] = (C[i-1] + 4*C[i] + C[i+1]) / 6
    with repeated boundary controls at the ends. This solves the resulting
    tridiagonal linear system over the time dimension.
    """
    if samples.shape[0] <= 1:
        return samples.clone()
    t = samples.shape[0]
    flat = samples.reshape(t, -1)
    mat = samples.new_zeros((t, t))
    idx = torch.arange(t, device=samples.device)
    mat[idx, idx] = 4.0
    mat[0, 0] = 5.0
    mat[-1, -1] = 5.0
    mat[idx[1:], idx[:-1]] = 1.0
    mat[idx[:-1], idx[1:]] = 1.0
    fitted = torch.linalg.solve(mat, 6.0 * flat)
    return fitted.reshape_as(samples)


def fit_uniform_cubic_spline_controls(
    samples,
    dt=CONTROL_FIT_DT,
    position_weight=None,
    velocity_weight=None,
    acceleration_weight=None,
    ridge_weight=None,
    mode=CONTROL_FIT_DEFAULT_MODE,
):
    """Fit cubic controls using derivative-aware targets by default.

    This is the project-default GT control synthesis contract. It trades exact
    sample reconstruction for controls whose decoded velocity and acceleration
    also match finite-difference derivatives of the physical target.
    """
    if mode in ('position_only', 'legacy_position_only', 'exact'):
        return fit_uniform_cubic_spline_controls_position_only(samples)
    if mode != CONTROL_FIT_DEFAULT_MODE:
        raise ValueError(f'Unsupported control fit mode: {mode}')
    if samples.shape[0] <= 1:
        return samples.clone()
    weights = CONTROL_FIT_DERIVATIVE_AWARE_WEIGHTS
    wp = float(weights['position'] if position_weight is None else position_weight)
    wv = float(weights['velocity'] if velocity_weight is None else velocity_weight)
    wa = float(weights['acceleration'] if acceleration_weight is None else acceleration_weight)
    wr = float(weights['ridge'] if ridge_weight is None else ridge_weight)
    t = samples.shape[0]
    flat = samples.reshape(t, -1)
    vel = _central_difference_first(samples, dt=dt).reshape(t, -1)
    acc = _central_difference_second(samples, dt=dt).reshape(t, -1)
    s, d1, d2 = _uniform_cubic_spline_operators(t, samples.device, samples.dtype, dt=dt)
    lhs = wp * s.t().matmul(s) + wv * d1.t().matmul(d1) + wa * d2.t().matmul(d2)
    if wr > 0.0:
        lhs = lhs + wr * torch.eye(t, device=samples.device, dtype=samples.dtype)
    rhs = wp * s.t().matmul(flat) + wv * d1.t().matmul(vel) + wa * d2.t().matmul(acc)
    fitted = torch.linalg.solve(lhs, rhs)
    return fitted.reshape_as(samples)


def pl_target_from_pose(pose, body_model):
    if pose.dim() == 3:
        pose = pose.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    _, _, verts = body_model.forward_kinematics(pose, calc_mesh=True)
    pRB = (verts[:, :5] - verts[:, 5:]).bmm(pose[:, 0]).reshape(pose.shape[0], 15)
    gR = -pose[:, 0, :, 1]
    target = torch.cat((pRB, gR), dim=-1)
    return target[0] if squeeze else target


def pl_curve_loss(output, target, weights, dt=1.0 / 60.0, target_control=None):
    pred = output['pl']
    base = output['base']
    target = target.to(pred.device, pred.dtype)
    pred_gR = art.math.normalize_tensor(pred[..., 15:], avoid_nan=True)
    target_gR = art.math.normalize_tensor(target[..., 15:], avoid_nan=True)
    target_for_controls = torch.cat((target[..., :15], target_gR), dim=-1)
    if target_control is None:
        target_control = fit_uniform_cubic_spline_controls(target_for_controls)
    else:
        target_control = target_control.to(pred.device, pred.dtype)
    pred_control = output.get('new_control')
    if pred_control is None:
        pred_control = pred
    pred_control_gR = art.math.normalize_tensor(pred_control[..., 15:], avoid_nan=True)
    target_control_gR = art.math.normalize_tensor(target_control[..., 15:], avoid_nan=True)
    losses = {
        'pRB': torch.nn.functional.smooth_l1_loss(pred[..., :15], target[..., :15]),
        'gR1': (1.0 - (pred_gR * target_gR).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
        'baseline_pRB': torch.nn.functional.smooth_l1_loss(pred[..., :15], base[..., :15].detach()),
        'baseline_gR1': (1.0 - (pred[..., 15:] * base[..., 15:].detach()).sum(dim=-1).clamp(-1.0, 1.0)).mean(),
        'gt_control_pRB': torch.nn.functional.smooth_l1_loss(pred_control[..., :15], target_control[..., :15]),
        'gt_control_gR1': torch.nn.functional.smooth_l1_loss(pred_control_gR, target_control_gR),
        'control_point_prior': output['control_point_prior'],
        'tail_update_prior': output['tail_delta_norm'],
    }
    if pred.shape[0] >= 2:
        target_step = target[1:, ..., :15] - target[:-1, ..., :15]
        losses['pRB_dot'] = torch.nn.functional.smooth_l1_loss(dt * output['pldot'][1:, ..., :15], target_step)
        pred_gR_dot = pred_gR[1:] - pred_gR[:-1]
        target_gR_dot = target_gR[1:] - target_gR[:-1]
        losses['gR1_dot'] = torch.nn.functional.smooth_l1_loss(pred_gR_dot, target_gR_dot)
        losses['gR_smooth'] = pred_gR_dot.square().mean()
    else:
        losses['pRB_dot'] = pred.new_zeros(())
        losses['gR1_dot'] = pred.new_zeros(())
        losses['gR_smooth'] = pred.new_zeros(())
    if pred.shape[0] >= 3:
        target_pRB_ddot = target[2:, ..., :15] - 2.0 * target[1:-1, ..., :15] + target[:-2, ..., :15]
        losses['pRB_ddot'] = torch.nn.functional.smooth_l1_loss(
            (float(dt) ** 2) * output['plddot'][1:-1, ..., :15],
            target_pRB_ddot,
        )
        pred_gR_ddot = pred_gR[2:] - 2.0 * pred_gR[1:-1] + pred_gR[:-2]
        target_gR_ddot = target_gR[2:] - 2.0 * target_gR[1:-1] + target_gR[:-2]
        losses['gR1_ddot'] = torch.nn.functional.smooth_l1_loss(pred_gR_ddot, target_gR_ddot)
    else:
        losses['pRB_ddot'] = pred.new_zeros(())
        losses['gR1_ddot'] = pred.new_zeros(())
    losses['pRB_ddot_smooth'] = output['plddot'][..., :15].square().mean()
    total = pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * weight
    return total, losses


def _r6d_to_rot6x(rotation6d, n_rot):
    r6d = rotation6d.contiguous().reshape(rotation6d.shape[:-1] + (n_rot, 6))
    a1 = r6d[..., 0:3]
    a2 = r6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-6)
    b2_raw = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2_norm = b2_raw.norm(dim=-1, keepdim=True)
    basis = torch.zeros_like(b1)
    min_axis = b1.abs().argmin(dim=-1, keepdim=True)
    basis.scatter_(-1, min_axis, 1.0)
    fallback = basis - (basis * b1).sum(dim=-1, keepdim=True) * b1
    fallback = torch.nn.functional.normalize(fallback, dim=-1, eps=1e-6)
    b2_raw = torch.where(b2_norm > 1e-6, b2_raw, fallback)
    b2 = torch.nn.functional.normalize(b2_raw, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotation_geodesic_loss_6d(pred, target, n_rot):
    target = target.to(pred.device, pred.dtype)
    pred_rot = _r6d_to_rot6x(pred, n_rot)
    target_rot = _r6d_to_rot6x(target, n_rot)
    rel = pred_rot.transpose(-1, -2).matmul(target_rot)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


def pl_bone_aux_loss(output, target_bone6d, weights, dt=1.0 / 60.0):
    pred = output.get('bone6d')
    if pred is None or target_bone6d is None:
        ref = output['pl']
        zero = ref.new_zeros(())
        losses = {
            'bone6d': zero,
            'bone_geo': zero,
            'gt_control_bone6d': zero,
            'gt_control_bone_geo': zero,
            'bone6d_dot': zero,
            'bone6d_ddot': zero,
            'bone_control_point_prior': zero,
            'bone_tail_update_prior': zero,
            'bone_aux_available': zero,
        }
        return zero, losses
    target = target_bone6d.to(pred.device, pred.dtype)
    target_control = fit_uniform_cubic_spline_controls(target)
    pred_control = output.get('bone_new_control', pred)
    losses = {
        'bone6d': torch.nn.functional.smooth_l1_loss(pred, target),
        'bone_geo': rotation_geodesic_loss_6d(pred, target, 5),
        'gt_control_bone6d': torch.nn.functional.smooth_l1_loss(pred_control, target_control),
        'gt_control_bone_geo': rotation_geodesic_loss_6d(pred_control, target_control, 5),
        'bone_control_point_prior': output.get('bone_control_point_prior', pred.new_zeros(())),
        'bone_tail_update_prior': output.get('bone_tail_delta_norm', pred.new_zeros(())),
        'bone_aux_available': pred.new_tensor(1.0),
    }
    if pred.shape[0] >= 2:
        pred_step = pred[1:] - pred[:-1]
        target_step = target[1:] - target[:-1]
        losses['bone6d_dot'] = torch.nn.functional.smooth_l1_loss(pred_step, target_step)
    else:
        losses['bone6d_dot'] = pred.new_zeros(())
    if pred.shape[0] >= 3:
        pred_ddot = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
        target_ddot = target[2:] - 2.0 * target[1:-1] + target[:-2]
        losses['bone6d_ddot'] = torch.nn.functional.smooth_l1_loss(pred_ddot, target_ddot)
    else:
        losses['bone6d_ddot'] = pred.new_zeros(())
    total = pred.new_zeros(())
    for key, weight in weights.items():
        total = total + losses[key] * weight
    return total, losses


class PLCurveModule(torch.nn.Module):
    def __init__(
        self,
        input_size=84,
        state_dim=18,
        init_size=18,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.005,
        dt=1.0 / 60.0,
        dropout=0.4,
        bone_aux_dim=0,
        bone_residual_scale=0.05,
    ):
        super().__init__()
        if state_dim != 18:
            raise ValueError('PLCurveModule v1 uses the official PL 18D state.')
        if tail_update != 4:
            raise ValueError('PLCurveModule v1 keeps the K2 L=4 tail-update contract.')
        self.input_size = int(input_size)
        self.state_dim = int(state_dim)
        self.init_size = int(init_size)
        self.hidden_size = int(hidden_size)
        self.tail_update = int(tail_update)
        self.residual_scale = float(residual_scale)
        self.dt = float(dt)
        self.bone_aux_dim = int(bone_aux_dim)
        self.bone_residual_scale = float(bone_residual_scale)
        self.input = torch.nn.Linear(input_size + state_dim, hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.cell = torch.nn.GRUCell(hidden_size, hidden_size)
        self.init_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.init_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.new_control = torch.nn.Linear(hidden_size, state_dim)
        self.tail_delta = torch.nn.Linear(hidden_size, tail_update * state_dim)
        if self.bone_aux_dim:
            if self.bone_aux_dim != PL_BONE_AUX_DIM:
                raise ValueError(f'PL bone aux currently expects {PL_BONE_AUX_DIM}D, got {self.bone_aux_dim}.')
            self.bone_new_control = torch.nn.Linear(hidden_size, self.bone_aux_dim)
            self.bone_tail_delta = torch.nn.Linear(hidden_size, tail_update * self.bone_aux_dim)
        self.spline = UniformCubicBSpline(dt)
        self.reset_stream()
        torch.nn.init.zeros_(self.init_encoder[-1].weight)
        torch.nn.init.zeros_(self.init_encoder[-1].bias)
        torch.nn.init.zeros_(self.new_control.weight)
        torch.nn.init.zeros_(self.new_control.bias)
        torch.nn.init.zeros_(self.tail_delta.weight)
        torch.nn.init.zeros_(self.tail_delta.bias)
        if self.bone_aux_dim:
            torch.nn.init.zeros_(self.bone_new_control.weight)
            torch.nn.init.zeros_(self.bone_new_control.bias)
            torch.nn.init.zeros_(self.bone_tail_delta.weight)
            torch.nn.init.zeros_(self.bone_tail_delta.bias)

    def reset_stream(self, init_output=None, init_feature=None):
        self.hidden = None
        init = init_feature if init_feature is not None else init_output
        if init is not None:
            if init.dim() == 1:
                init = init.unsqueeze(0)
            if init.shape[-1] != self.init_size:
                raise ValueError(f'Expected PL init dim {self.init_size}, got {init.shape[-1]}.')
            self.hidden = self.init_encoder(init.detach())
        self.control_buffer = None
        self.base_buffer = None
        self.bone_control_buffer = None
        self.bone_base_buffer = None
        self.last_debug = {}

    def _initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _ghost(self, buffer, count=1):
        return buffer[:, -1:].expand(-1, int(count), -1).clone()

    def _bone_aux_step(self, feature_t):
        if not self.bone_aux_dim:
            return {}
        base_bone_t = pl_bone6d_base_from_feature(feature_t).to(device=self.hidden.device, dtype=self.hidden.dtype)
        if base_bone_t.dim() == 1:
            base_bone_t = base_bone_t.unsqueeze(0)
        bone_delta = self.bone_new_control(self.hidden) * self.bone_residual_scale
        bone_new_control = base_bone_t.detach() + bone_delta
        if self.bone_control_buffer is None:
            self.bone_control_buffer = bone_new_control.unsqueeze(1)
            self.bone_base_buffer = base_bone_t.detach().unsqueeze(1)
            bone_tail_delta_norm = bone_delta.norm(dim=-1).mean()
        else:
            frozen_control = self.bone_control_buffer.detach()
            frozen_base = self.bone_base_buffer.detach()
            update_count = min(self.tail_update, frozen_control.shape[1])
            old_control = frozen_control[:, :-update_count]
            old_base = frozen_base[:, :-update_count]
            tail_control = frozen_control[:, -update_count:]
            tail_base = frozen_base[:, -update_count:]
            tail_delta = self.bone_tail_delta(self.hidden).reshape(
                self.hidden.shape[0], self.tail_update, self.bone_aux_dim
            )[:, -update_count:] * self.bone_residual_scale
            tail_control = tail_control + tail_delta
            self.bone_control_buffer = torch.cat((old_control, tail_control, bone_new_control.unsqueeze(1)), dim=1)
            self.bone_base_buffer = torch.cat((old_base, tail_base, base_bone_t.detach().unsqueeze(1)), dim=1)
            bone_tail_delta_norm = tail_delta.norm(dim=-1).mean()
        decode_control = torch.cat((self.bone_control_buffer, self._ghost(self.bone_control_buffer, 1)), dim=1)
        decode_base = torch.cat((self.bone_base_buffer, self._ghost(self.bone_base_buffer, 1)), dim=1)
        bone_curve, bone_dot_curve, bone_ddot_curve = self.spline(decode_control, return_derivatives=True)
        bone_base_curve = self.spline(decode_base)
        return {
            'bone6d_t': bone_curve[:, -2],
            'bone6d_dot_t': bone_dot_curve[:, -2],
            'bone6d_ddot_t': bone_ddot_curve[:, -2],
            'bone_base_t': bone_base_curve[:, -2],
            'bone_new_control_t': bone_new_control,
            'bone_control_point_prior_t': (self.bone_control_buffer - self.bone_base_buffer).square().mean(),
            'bone_new_delta_norm': bone_delta.norm(dim=-1).mean(),
            'bone_tail_delta_norm': bone_tail_delta_norm,
        }

    def _append_bone_aux_result(self, result, feature_t):
        result.update(self._bone_aux_step(feature_t))
        return result

    def step(self, feature_t, base_pl_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_pl_t.dim() == 1:
            base_pl_t = base_pl_t.unsqueeze(0)
        if feature_t.shape[-1] != self.input_size:
            raise ValueError(f'Expected PL feature dim {self.input_size}, got {feature_t.shape[-1]}.')
        base_pl_t = normalize_gravity(base_pl_t)
        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)
        z = torch.relu(self.input(torch.cat((feature_t, base_pl_t.detach()), dim=-1)))
        z = self.dropout(z)
        self.hidden = self.cell(z, self.hidden)
        new_delta = self.new_control(self.hidden) * self.residual_scale
        new_control = base_pl_t + new_delta
        if self.control_buffer is None:
            self.control_buffer = new_control.unsqueeze(1)
            self.base_buffer = base_pl_t.unsqueeze(1)
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
            self.base_buffer = torch.cat((old_base, tail_base, base_pl_t.unsqueeze(1)), dim=1)
            tail_delta_norm = tail_delta.norm(dim=-1).mean()
        decode_control = torch.cat((self.control_buffer, self._ghost(self.control_buffer, 1)), dim=1)
        decode_base = torch.cat((self.base_buffer, self._ghost(self.base_buffer, 1)), dim=1)
        pl_curve, pldot_curve, plddot_curve = self.spline(decode_control, return_derivatives=True)
        pl_base = self.spline(decode_base)
        pl_t = normalize_gravity(pl_curve[:, -2])
        base_t = normalize_gravity(pl_base[:, -2])
        result = {
            'pl_t': pl_t,
            'pldot_t': pldot_curve[:, -2],
            'plddot_t': plddot_curve[:, -2],
            'base_t': base_t,
            'new_control_t': new_control,
            'residual_t': pl_t - base_t,
            'control_point_prior_t': (self.control_buffer - self.base_buffer).square().mean(),
            'new_delta_norm': new_delta.norm(dim=-1).mean(),
            'tail_delta_norm': tail_delta_norm,
            'buffer_length': self.control_buffer.shape[1],
        }
        self._append_bone_aux_result(result, feature_t)
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return result


class PLCurveOffsetConditionedModule(PLCurveModule):
    """PLCurve diagnostic variant with per-frame init36 conditioning.

    The external PL frame input remains 84D and the output remains the official
    18D pRB[15]+gR1[3]. The difference from PLCurveModule is that the encoded
    init feature, including offset_r[18], is injected at every recurrent step
    instead of only initializing the hidden state.
    """

    def __init__(
        self,
        input_size=84,
        state_dim=18,
        init_size=36,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.005,
        dt=1.0 / 60.0,
        dropout=0.4,
        condition_scale=1.0,
        bone_aux_dim=0,
        bone_residual_scale=0.05,
    ):
        super().__init__(
            input_size=input_size,
            state_dim=state_dim,
            init_size=init_size,
            hidden_size=hidden_size,
            tail_update=tail_update,
            residual_scale=residual_scale,
            dt=dt,
            dropout=dropout,
            bone_aux_dim=bone_aux_dim,
            bone_residual_scale=bone_residual_scale,
        )
        if init_size < 36:
            raise ValueError('PLCurveOffsetConditionedModule expects init36 with offset_r[18]+pRL[15]+gR0[3].')
        self.condition_scale = float(condition_scale)
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.init_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.condition_input = torch.nn.Linear(hidden_size, hidden_size)
        self.condition_hidden = torch.nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.condition_encoder[-1].weight, gain=0.1)
        torch.nn.init.zeros_(self.condition_encoder[-1].bias)
        torch.nn.init.xavier_uniform_(self.condition_input.weight, gain=0.1)
        torch.nn.init.zeros_(self.condition_input.bias)
        torch.nn.init.xavier_uniform_(self.condition_hidden.weight, gain=0.1)
        torch.nn.init.zeros_(self.condition_hidden.bias)

    def reset_stream(self, init_output=None, init_feature=None):
        super().reset_stream(init_output=init_output, init_feature=init_feature)
        self.condition = None
        init = init_feature if init_feature is not None else init_output
        if init is not None:
            if init.dim() == 1:
                init = init.unsqueeze(0)
            if init.shape[-1] != self.init_size:
                raise ValueError(f'Expected PL init dim {self.init_size}, got {init.shape[-1]}.')
            self.condition = self.condition_encoder(init.detach())

    def _initial_condition(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def step(self, feature_t, base_pl_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_pl_t.dim() == 1:
            base_pl_t = base_pl_t.unsqueeze(0)
        if feature_t.shape[-1] != self.input_size:
            raise ValueError(f'Expected PL feature dim {self.input_size}, got {feature_t.shape[-1]}.')
        base_pl_t = normalize_gravity(base_pl_t)
        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)
        if self.condition is None or self.condition.shape[0] != feature_t.shape[0]:
            self.condition = self._initial_condition(feature_t.shape[0], feature_t.device, feature_t.dtype)
        z = torch.relu(self.input(torch.cat((feature_t, base_pl_t.detach()), dim=-1)))
        z = z + self.condition_input(self.condition) * self.condition_scale
        z = self.dropout(z)
        hidden0 = self.hidden + self.condition_hidden(self.condition) * self.condition_scale
        self.hidden = self.cell(z, hidden0)
        new_delta = self.new_control(self.hidden) * self.residual_scale
        new_control = base_pl_t + new_delta
        if self.control_buffer is None:
            self.control_buffer = new_control.unsqueeze(1)
            self.base_buffer = base_pl_t.unsqueeze(1)
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
            self.base_buffer = torch.cat((old_base, tail_base, base_pl_t.unsqueeze(1)), dim=1)
            tail_delta_norm = tail_delta.norm(dim=-1).mean()
        decode_control = torch.cat((self.control_buffer, self._ghost(self.control_buffer, 1)), dim=1)
        decode_base = torch.cat((self.base_buffer, self._ghost(self.base_buffer, 1)), dim=1)
        pl_curve, pldot_curve, plddot_curve = self.spline(decode_control, return_derivatives=True)
        pl_base = self.spline(decode_base)
        pl_t = normalize_gravity(pl_curve[:, -2])
        base_t = normalize_gravity(pl_base[:, -2])
        result = {
            'pl_t': pl_t,
            'pldot_t': pldot_curve[:, -2],
            'plddot_t': plddot_curve[:, -2],
            'base_t': base_t,
            'new_control_t': new_control,
            'residual_t': pl_t - base_t,
            'control_point_prior_t': (self.control_buffer - self.base_buffer).square().mean(),
            'new_delta_norm': new_delta.norm(dim=-1).mean(),
            'tail_delta_norm': tail_delta_norm,
            'condition_norm': self.condition.norm(dim=-1).mean(),
            'buffer_length': self.control_buffer.shape[1],
        }
        self._append_bone_aux_result(result, feature_t)
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return result


class PLCurveOffsetAwareModule(PLCurveModule):
    """Offset-aware NewPL variant with 18D PL output contract.

    The frame input is 156D offset-aware physics features. The sequence-level
    `offset_r[6,3]` is read from init36 and modulates the recurrent hidden state
    through a small FiLM/gate block, while the parent class still preserves the
    official pRB[15]+gR1[3] output shape.
    """

    def __init__(
        self,
        input_size=PL_OFFSET_AWARE_INPUT_SIZE,
        state_dim=18,
        init_size=36,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.005,
        dt=1.0 / 60.0,
        dropout=0.4,
        offset_embed_size=128,
        film_scale=0.1,
        bone_aux_dim=0,
        bone_residual_scale=0.05,
    ):
        super().__init__(
            input_size=input_size,
            state_dim=state_dim,
            init_size=init_size,
            hidden_size=hidden_size,
            tail_update=tail_update,
            residual_scale=residual_scale,
            dt=dt,
            dropout=dropout,
            bone_aux_dim=bone_aux_dim,
            bone_residual_scale=bone_residual_scale,
        )
        if init_size < 36:
            raise ValueError('PLCurveOffsetAwareModule expects init36 with offset_r[18]+pRL[15]+gR0[3].')
        if input_size != PL_OFFSET_AWARE_INPUT_SIZE:
            raise ValueError(f'PLCurveOffsetAwareModule expects input_size={PL_OFFSET_AWARE_INPUT_SIZE}, got {input_size}.')
        self.offset_embed_size = int(offset_embed_size)
        self.film_scale = float(film_scale)
        self.offset_encoder = torch.nn.Sequential(
            torch.nn.Linear(18, self.offset_embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.offset_embed_size, self.offset_embed_size),
            torch.nn.ReLU(),
        )
        self.offset_to_input = torch.nn.Linear(self.offset_embed_size, hidden_size)
        self.offset_to_film = torch.nn.Linear(self.offset_embed_size, hidden_size * 2)
        self.offset_to_gate = torch.nn.Linear(self.offset_embed_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.offset_to_input.weight, gain=0.1)
        torch.nn.init.zeros_(self.offset_to_input.bias)
        torch.nn.init.xavier_uniform_(self.offset_to_film.weight, gain=0.1)
        torch.nn.init.zeros_(self.offset_to_film.bias)
        torch.nn.init.xavier_uniform_(self.offset_to_gate.weight, gain=0.1)
        torch.nn.init.zeros_(self.offset_to_gate.bias)

    def reset_stream(self, init_output=None, init_feature=None):
        super().reset_stream(init_output=init_output, init_feature=init_feature)
        self.offset_embedding = None
        init = init_feature if init_feature is not None else init_output
        if init is not None:
            if init.dim() == 1:
                init = init.unsqueeze(0)
            if init.shape[-1] != self.init_size:
                raise ValueError(f'Expected PL init dim {self.init_size}, got {init.shape[-1]}.')
            offset = init[..., :18].detach()
            self.offset_embedding = self.offset_encoder(offset)

    def _initial_offset_embedding(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.offset_embed_size, device=device, dtype=dtype)

    def step(self, feature_t, base_pl_t):
        if feature_t.dim() == 1:
            feature_t = feature_t.unsqueeze(0)
        if base_pl_t.dim() == 1:
            base_pl_t = base_pl_t.unsqueeze(0)
        if feature_t.shape[-1] != self.input_size:
            raise ValueError(f'Expected PL feature dim {self.input_size}, got {feature_t.shape[-1]}.')
        base_pl_t = normalize_gravity(base_pl_t)
        if self.hidden is None or self.hidden.shape[0] != feature_t.shape[0]:
            self.hidden = self._initial_hidden(feature_t.shape[0], feature_t.device, feature_t.dtype)
        if self.offset_embedding is None or self.offset_embedding.shape[0] != feature_t.shape[0]:
            self.offset_embedding = self._initial_offset_embedding(feature_t.shape[0], feature_t.device, feature_t.dtype)
        z = torch.relu(self.input(torch.cat((feature_t, base_pl_t.detach()), dim=-1)))
        z = z + self.offset_to_input(self.offset_embedding)
        z = self.dropout(z)
        gamma, beta = self.offset_to_film(self.offset_embedding).chunk(2, dim=-1)
        hidden_mod = self.hidden * (1.0 + torch.tanh(gamma) * self.film_scale) + beta * self.film_scale
        gate = torch.sigmoid(self.offset_to_gate(self.offset_embedding))
        hidden_next = self.cell(z, hidden_mod)
        self.hidden = gate * hidden_next + (1.0 - gate) * self.hidden
        new_delta = self.new_control(self.hidden) * self.residual_scale
        new_control = base_pl_t + new_delta
        if self.control_buffer is None:
            self.control_buffer = new_control.unsqueeze(1)
            self.base_buffer = base_pl_t.unsqueeze(1)
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
            self.base_buffer = torch.cat((old_base, tail_base, base_pl_t.unsqueeze(1)), dim=1)
            tail_delta_norm = tail_delta.norm(dim=-1).mean()
        decode_control = torch.cat((self.control_buffer, self._ghost(self.control_buffer, 1)), dim=1)
        decode_base = torch.cat((self.base_buffer, self._ghost(self.base_buffer, 1)), dim=1)
        pl_curve, pldot_curve, plddot_curve = self.spline(decode_control, return_derivatives=True)
        pl_base = self.spline(decode_base)
        pl_t = normalize_gravity(pl_curve[:, -2])
        base_t = normalize_gravity(pl_base[:, -2])
        result = {
            'pl_t': pl_t,
            'pldot_t': pldot_curve[:, -2],
            'plddot_t': plddot_curve[:, -2],
            'base_t': base_t,
            'new_control_t': new_control,
            'residual_t': pl_t - base_t,
            'control_point_prior_t': (self.control_buffer - self.base_buffer).square().mean(),
            'new_delta_norm': new_delta.norm(dim=-1).mean(),
            'tail_delta_norm': tail_delta_norm,
            'offset_embedding_norm': self.offset_embedding.norm(dim=-1).mean(),
            'hidden_gate_mean': gate.mean(),
            'buffer_length': self.control_buffer.shape[1],
        }
        self._append_bone_aux_result(result, feature_t)
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return result


class PLCurveLearnedOffsetAccAuxModule(PLCurveModule):
    """NewPL v7 diagnostic variant with internal learned IMU position offsets.

    The external contract stays official: 84D frame input, 18D PL output, and
    init18 pRL[15]+gR0[3].  `raw_offset` is only consumed by the auxiliary IMU
    acceleration consistency loss; downstream IK1 never sees it.
    """

    coordinate_contract = PL_LEARNED_OFFSET_ACC_CONTRACT

    def __init__(
        self,
        input_size=PL_LEGACY_INPUT_SIZE,
        state_dim=18,
        init_size=18,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.005,
        dt=1.0 / 60.0,
        dropout=0.4,
        offset_max=0.30,
        learned_offset_init=None,
        bone_aux_dim=0,
        bone_residual_scale=0.05,
    ):
        if input_size != PL_LEGACY_INPUT_SIZE:
            raise ValueError(
                f'PLCurveLearnedOffsetAccAuxModule preserves the legacy 84D PL input; got input_size={input_size}.'
            )
        if init_size != 18:
            raise ValueError('newpl_v7_learned_offset_accaux expects init18: pRL[15]+gR0[3].')
        super().__init__(
            input_size=input_size,
            state_dim=state_dim,
            init_size=init_size,
            hidden_size=hidden_size,
            tail_update=tail_update,
            residual_scale=residual_scale,
            dt=dt,
            dropout=dropout,
            bone_aux_dim=bone_aux_dim,
            bone_residual_scale=bone_residual_scale,
        )
        self.offset_max = float(offset_max)
        raw = torch.zeros(6, 3)
        if learned_offset_init is not None:
            init = torch.as_tensor(learned_offset_init, dtype=torch.float32).view(6, 3)
            raw = raw_offset_from_bounded(init, offset_max=self.offset_max)
        self.raw_offset = torch.nn.Parameter(raw)

    def learned_offset(self):
        return bounded_offset_from_raw(self.raw_offset, offset_max=self.offset_max)

    def offset_norm_summary(self):
        norms = self.learned_offset().detach().norm(dim=-1)
        return {
            'offset_norm_mean_m': float(norms.mean().cpu()),
            'offset_norm_median_m': float(norms.median().cpu()),
            'offset_norm_p95_m': float(torch.quantile(norms.cpu(), 0.95)),
            'offset_max_m': self.offset_max,
        }


class PLCurveLearnedLeafOffsetLocalAccAuxModule(PLCurveModule):
    """NewPL v7b diagnostic variant with leaf-only local acceleration offsets."""

    coordinate_contract = PL_LEARNED_LEAF_OFFSET_LOCAL_ACC_CONTRACT

    def __init__(
        self,
        input_size=PL_LEGACY_INPUT_SIZE,
        state_dim=18,
        init_size=18,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.005,
        dt=1.0 / 60.0,
        dropout=0.4,
        offset_max=0.30,
        learned_leaf_offset_init=None,
        learned_offset_init=None,
        bone_aux_dim=0,
        bone_residual_scale=0.05,
    ):
        if input_size != PL_LEGACY_INPUT_SIZE:
            raise ValueError(
                f'PLCurveLearnedLeafOffsetLocalAccAuxModule preserves the legacy 84D PL input; got input_size={input_size}.'
            )
        if init_size != 18:
            raise ValueError('newpl_v7b_local_accaux expects init18: pRL[15]+gR0[3].')
        super().__init__(
            input_size=input_size,
            state_dim=state_dim,
            init_size=init_size,
            hidden_size=hidden_size,
            tail_update=tail_update,
            residual_scale=residual_scale,
            dt=dt,
            dropout=dropout,
            bone_aux_dim=bone_aux_dim,
            bone_residual_scale=bone_residual_scale,
        )
        self.offset_max = float(offset_max)
        raw = torch.zeros(5, 3)
        init = learned_leaf_offset_init
        if init is None:
            init = learned_offset_init
        if init is not None:
            init = torch.as_tensor(init, dtype=torch.float32)
            if init.numel() == 18:
                init = init.view(6, 3)[:5]
            else:
                init = init.view(5, 3)
            raw = raw_offset_from_bounded(init, offset_max=self.offset_max)
        self.raw_leaf_offset = torch.nn.Parameter(raw)

    def learned_leaf_offset(self):
        return bounded_offset_from_raw(self.raw_leaf_offset, offset_max=self.offset_max)

    def learned_offset(self):
        leaf = self.learned_leaf_offset()
        root = leaf.new_zeros(1, 3)
        return torch.cat((leaf, root), dim=0)

    def offset_norm_summary(self):
        norms = self.learned_leaf_offset().detach().norm(dim=-1)
        return {
            'offset_norm_mean_m': float(norms.mean().cpu()),
            'offset_norm_median_m': float(norms.median().cpu()),
            'offset_norm_p95_m': float(torch.quantile(norms.cpu(), 0.95)),
            'offset_max_m': self.offset_max,
            'offset_count': 5,
        }


class PLCurveNextControlModule(PLCurveModule):
    """One-step predictive NewPL variant with the official current PL contract.

    The recurrent stream still emits the normal current-frame 18D
    `pRB[15]+gR1[3]` output.  This variant adds a preview branch that follows the
    same tail-update idea as the current-frame PLCurve path: it adjusts the last
    up-to-four existing controls and appends exactly one predicted next control.
    The preview is decoded by the same uniform cubic spline, but it is not
    written back to the live streaming buffer, so downstream IK1 remains
    unchanged.
    """

    def __init__(
        self,
        input_size=PL_LEGACY_INPUT_SIZE,
        state_dim=18,
        init_size=36,
        hidden_size=512,
        tail_update=4,
        residual_scale=0.005,
        dt=1.0 / 60.0,
        dropout=0.4,
        next_residual_scale=None,
        bone_aux_dim=0,
        bone_residual_scale=0.05,
    ):
        super().__init__(
            input_size=input_size,
            state_dim=state_dim,
            init_size=init_size,
            hidden_size=hidden_size,
            tail_update=tail_update,
            residual_scale=residual_scale,
            dt=dt,
            dropout=dropout,
            bone_aux_dim=bone_aux_dim,
            bone_residual_scale=bone_residual_scale,
        )
        if input_size != PL_LEGACY_INPUT_SIZE:
            raise ValueError(
                f'PLCurveNextControlModule preserves the legacy 84D PL input; got input_size={input_size}.'
            )
        self.next_residual_scale = float(residual_scale if next_residual_scale is None else next_residual_scale)
        self.next_control_delta = torch.nn.Linear(hidden_size, state_dim)
        self.next_tail_delta = torch.nn.Linear(hidden_size, tail_update * state_dim)
        torch.nn.init.zeros_(self.next_control_delta.weight)
        torch.nn.init.zeros_(self.next_control_delta.bias)
        torch.nn.init.zeros_(self.next_tail_delta.weight)
        torch.nn.init.zeros_(self.next_tail_delta.bias)

    def step(self, feature_t, base_pl_t):
        result = super().step(feature_t, base_pl_t)
        next_delta = self.next_control_delta(self.hidden) * self.next_residual_scale
        next_control = result['new_control_t'].detach() + next_delta
        frozen_control = self.control_buffer.detach()
        update_count = min(self.tail_update, frozen_control.shape[1])
        old_control = frozen_control[:, :-update_count]
        tail_control = frozen_control[:, -update_count:]
        next_tail_delta = self.next_tail_delta(self.hidden).reshape(
            self.hidden.shape[0], self.tail_update, self.state_dim
        )[:, -update_count:] * self.next_residual_scale
        adjusted_tail = tail_control + next_tail_delta
        preview_control = torch.cat(
            (old_control, adjusted_tail, next_control.unsqueeze(1), next_control.unsqueeze(1)),
            dim=1,
        )
        next_curve, next_dot_curve, next_ddot_curve = self.spline(preview_control, return_derivatives=True)
        next_pl = normalize_gravity(next_curve[:, -2])
        tail_full = frozen_control.new_zeros(self.hidden.shape[0], self.tail_update, self.state_dim)
        tail_mask = torch.zeros(self.hidden.shape[0], self.tail_update, device=frozen_control.device, dtype=torch.bool)
        tail_full[:, -update_count:] = adjusted_tail
        tail_mask[:, -update_count:] = True
        result.update({
            'next_control_t': next_control,
            'next_tail_control_t': tail_full,
            'next_tail_control_mask_t': tail_mask,
            'last_preview_control_t': adjusted_tail[:, -1],
            'next_pl_t': next_pl,
            'next_pldot_t': next_dot_curve[:, -2],
            'next_plddot_t': next_ddot_curve[:, -2],
            'next_control_delta_norm': next_delta.norm(dim=-1).mean(),
            'next_tail_delta_norm': next_tail_delta.norm(dim=-1).mean(),
            'next_control_step_norm': (next_control - result['new_control_t'].detach()).norm(dim=-1).mean(),
        })
        self.last_debug = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()}
        return result


def _pl_curve_forward_sequence(self, features, base_outputs, init_output=None, init_feature=None):
    squeeze_batch = features.dim() == 2
    if squeeze_batch:
        features = features.unsqueeze(1)
        base_outputs = base_outputs.unsqueeze(1)
        if init_output is not None and init_output.dim() == 1:
            init_output = init_output.unsqueeze(0)
        if init_feature is not None and init_feature.dim() == 1:
            init_feature = init_feature.unsqueeze(0)
    self.reset_stream(init_output=init_output, init_feature=init_feature)
    outputs, dots, ddots, bases, new_controls = [], [], [], [], []
    bone_outputs, bone_dots, bone_ddots, bone_bases, bone_controls = [], [], [], [], []
    bone_priors, bone_tails, bone_deltas = [], [], []
    next_outputs, next_dots, next_ddots, next_controls = [], [], [], []
    next_tail_controls, next_tail_masks, last_preview_controls = [], [], []
    next_deltas, next_tail_deltas, next_steps = [], [], []
    priors, tails, deltas, conditions, offset_embeddings, hidden_gates = [], [], [], [], [], []
    for i in range(features.shape[0]):
        out = self.step(features[i], base_outputs[i])
        outputs.append(out['pl_t'])
        dots.append(out['pldot_t'])
        ddots.append(out['plddot_t'])
        bases.append(out['base_t'])
        new_controls.append(out['new_control_t'])
        priors.append(out['control_point_prior_t'])
        tails.append(out['tail_delta_norm'])
        deltas.append(out['new_delta_norm'])
        if 'condition_norm' in out:
            conditions.append(out['condition_norm'])
        if 'offset_embedding_norm' in out:
            offset_embeddings.append(out['offset_embedding_norm'])
        if 'hidden_gate_mean' in out:
            hidden_gates.append(out['hidden_gate_mean'])
        if 'bone6d_t' in out:
            bone_outputs.append(out['bone6d_t'])
            bone_dots.append(out['bone6d_dot_t'])
            bone_ddots.append(out['bone6d_ddot_t'])
            bone_bases.append(out['bone_base_t'])
            bone_controls.append(out['bone_new_control_t'])
            bone_priors.append(out['bone_control_point_prior_t'])
            bone_tails.append(out['bone_tail_delta_norm'])
            bone_deltas.append(out['bone_new_delta_norm'])
        if 'next_pl_t' in out:
            next_outputs.append(out['next_pl_t'])
            next_dots.append(out['next_pldot_t'])
            next_ddots.append(out['next_plddot_t'])
            next_controls.append(out['next_control_t'])
            next_tail_controls.append(out['next_tail_control_t'])
            next_tail_masks.append(out['next_tail_control_mask_t'])
            last_preview_controls.append(out['last_preview_control_t'])
            next_deltas.append(out['next_control_delta_norm'])
            next_tail_deltas.append(out['next_tail_delta_norm'])
            next_steps.append(out['next_control_step_norm'])
    result = {
        'pl': torch.stack(outputs),
        'pldot': torch.stack(dots),
        'plddot': torch.stack(ddots),
        'base': torch.stack(bases),
        'new_control': torch.stack(new_controls),
        'control_point_prior': torch.stack(priors).mean(),
        'tail_delta_norm': torch.stack(tails).mean(),
        'new_delta_norm': torch.stack(deltas).mean(),
    }
    if conditions:
        result['condition_norm'] = torch.stack(conditions).mean()
    if offset_embeddings:
        result['offset_embedding_norm'] = torch.stack(offset_embeddings).mean()
    if hidden_gates:
        result['hidden_gate_mean'] = torch.stack(hidden_gates).mean()
    if bone_outputs:
        result.update({
            'bone6d': torch.stack(bone_outputs),
            'bone6d_dot': torch.stack(bone_dots),
            'bone6d_ddot': torch.stack(bone_ddots),
            'bone_base': torch.stack(bone_bases),
            'bone_new_control': torch.stack(bone_controls),
            'bone_control_point_prior': torch.stack(bone_priors).mean(),
            'bone_tail_delta_norm': torch.stack(bone_tails).mean(),
            'bone_new_delta_norm': torch.stack(bone_deltas).mean(),
        })
    if next_outputs:
        result.update({
            'next_pl': torch.stack(next_outputs),
            'next_pldot': torch.stack(next_dots),
            'next_plddot': torch.stack(next_ddots),
            'next_control': torch.stack(next_controls),
            'next_tail_control': torch.stack(next_tail_controls),
            'next_tail_control_mask': torch.stack(next_tail_masks),
            'last_preview_control': torch.stack(last_preview_controls),
            'next_control_delta_norm': torch.stack(next_deltas).mean(),
            'next_tail_delta_norm': torch.stack(next_tail_deltas).mean(),
            'next_control_step_norm': torch.stack(next_steps).mean(),
        })
    if squeeze_batch:
        for key in (
            'pl', 'pldot', 'plddot', 'base', 'new_control',
            'bone6d', 'bone6d_dot', 'bone6d_ddot', 'bone_base', 'bone_new_control',
            'next_pl', 'next_pldot', 'next_plddot', 'next_control',
            'next_tail_control', 'next_tail_control_mask', 'last_preview_control',
        ):
            if key in result:
                result[key] = result[key][:, 0]
    return result


PLCurveModule.forward_sequence = _pl_curve_forward_sequence
PLCurveOffsetConditionedModule.forward_sequence = _pl_curve_forward_sequence
PLCurveOffsetAwareModule.forward_sequence = _pl_curve_forward_sequence
PLCurveLearnedOffsetAccAuxModule.forward_sequence = _pl_curve_forward_sequence
PLCurveLearnedLeafOffsetLocalAccAuxModule.forward_sequence = _pl_curve_forward_sequence
PLCurveNextControlModule.forward_sequence = _pl_curve_forward_sequence


def build_pl_curve_model(config):
    variant = config.get('model_variant', config.get('pl_curve_variant', 'base'))
    if variant == 'offset_conditioned':
        cls = PLCurveOffsetConditionedModule
    elif variant == 'offset_aware':
        cls = PLCurveOffsetAwareModule
    elif variant == 'newpl_v6_next_control':
        cls = PLCurveNextControlModule
    elif variant == 'newpl_v7_learned_offset_accaux':
        cls = PLCurveLearnedOffsetAccAuxModule
    elif variant == 'newpl_v7b_local_accaux':
        cls = PLCurveLearnedLeafOffsetLocalAccAuxModule
    else:
        cls = PLCurveModule
    kwargs = {
        'input_size': int(config.get('input_size', PL_OFFSET_AWARE_INPUT_SIZE if variant == 'offset_aware' else PL_LEGACY_INPUT_SIZE)),
        'init_size': int(config.get('init_size', 18)),
        'hidden_size': int(config.get('hidden_size', 512)),
        'tail_update': int(config.get('tail_length', config.get('tail_update', 4))),
        'residual_scale': float(config.get('residual_scale', 0.005)),
        'dropout': float(config.get('dropout', 0.4)),
        'bone_aux_dim': int(config.get('bone_aux_dim', 0)),
        'bone_residual_scale': float(config.get('bone_residual_scale', 0.05)),
    }
    if cls is PLCurveOffsetConditionedModule:
        kwargs['condition_scale'] = float(config.get('condition_scale', 1.0))
    if cls is PLCurveOffsetAwareModule:
        kwargs['offset_embed_size'] = int(config.get('offset_embed_size', 128))
        kwargs['film_scale'] = float(config.get('film_scale', 0.1))
    if cls is PLCurveNextControlModule:
        kwargs['next_residual_scale'] = float(config.get('next_residual_scale', kwargs['residual_scale']))
    if cls is PLCurveLearnedOffsetAccAuxModule:
        kwargs['offset_max'] = float(config.get('offset_max', 0.30))
        if config.get('learned_offset_init') is not None:
            kwargs['learned_offset_init'] = config.get('learned_offset_init')
    if cls is PLCurveLearnedLeafOffsetLocalAccAuxModule:
        kwargs['offset_max'] = float(config.get('offset_max', 0.30))
        if config.get('learned_leaf_offset_init') is not None:
            kwargs['learned_leaf_offset_init'] = config.get('learned_leaf_offset_init')
        elif config.get('learned_offset_init') is not None:
            kwargs['learned_offset_init'] = config.get('learned_offset_init')
    return cls(**kwargs)
