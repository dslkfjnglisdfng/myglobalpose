import torch

import articulate as art
from l4_q75_utils import q75_to_pose_tran


DT = 1.0 / 60.0
FOOT_JOINTS = (10, 11)
_BODY_MODEL = None


def body_model():
    global _BODY_MODEL
    if _BODY_MODEL is None:
        _BODY_MODEL = art.ParametricModel('models/SMPL_male.pkl', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return _BODY_MODEL


def wrapped_angle_delta(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def body_qdot_fd_target(q_body, dt=DT):
    if q_body.shape[1] < 2:
        return q_body.new_zeros(q_body.shape)
    step = wrapped_angle_delta(q_body[:, 1:], q_body[:, :-1]) / dt
    first = step[:, :1]
    return torch.cat((first, step), dim=1)


def q75_root_relative_foot_positions(q):
    batch, length, dim = q.shape
    pose, _ = q75_to_pose_tran(q.reshape(batch * length, dim))
    joints = body_model().forward_kinematics(pose.to(q.device))[1]
    joints = joints.to(device=q.device, dtype=q.dtype).reshape(batch, length, 24, 3)
    rootrel = joints - joints[:, :, :1]
    return rootrel[:, :, FOOT_JOINTS]


def foot_velocity_from_q(q, dt=DT):
    foot = q75_root_relative_foot_positions(q)
    if foot.shape[1] < 2:
        return foot.new_zeros(foot.shape)
    vel = (foot[:, 1:] - foot[:, :-1]) / dt
    return torch.cat((vel[:, :1], vel), dim=1)


def heuristic_contact_gate(q_net, height_threshold=0.08, velocity_threshold=0.20, sharpness=40.0):
    with torch.no_grad():
        foot = q75_root_relative_foot_positions(q_net)
        foot_vel = foot_velocity_from_q(q_net)
        height = foot[..., 1]
        ground = height.amin(dim=(1, 2), keepdim=True)
        rel_height = height - ground
        speed = foot_vel.norm(dim=-1)
        height_gate = torch.sigmoid((float(height_threshold) - rel_height) * float(sharpness))
        vel_gate = torch.sigmoid((float(velocity_threshold) - speed) * float(sharpness))
        gate = (height_gate * vel_gate).clamp(0.0, 1.0)
    return gate


def curve_dynamics_proxy_loss(qdot, qddot, dt=DT):
    if qdot.shape[1] < 2:
        return qdot.new_zeros(())
    qdot_fd = (qdot[:, 1:, 6:] - qdot[:, :-1, 6:]) / dt
    return torch.nn.functional.smooth_l1_loss(qddot[:, 1:, 6:], qdot_fd)


def control_point_refinement_loss(
    spline,
    control,
    control_net,
    q_net,
    qdot_fd_ref,
    lambda_prior=1.0,
    lambda_q=1.0,
    lambda_v=0.03,
    lambda_a=0.0003,
    lambda_contact=0.0,
    lambda_dyn=0.0,
    contact_gate=None,
):
    q, qdot, qddot = spline(control, return_derivatives=True)
    prior = (control[..., 6:] - control_net[..., 6:]).square().mean()
    q_body = wrapped_angle_delta(q[..., 6:], q_net[..., 6:]).square().mean()
    v_body = torch.nn.functional.smooth_l1_loss(qdot[..., 6:], qdot_fd_ref)
    a_body = qddot[..., 6:].square().mean()
    if float(lambda_contact) > 0.0:
        if contact_gate is None:
            contact_gate = heuristic_contact_gate(q_net)
        foot_vel = foot_velocity_from_q(q)
        contact = (contact_gate.to(q.device, q.dtype).unsqueeze(-1) * foot_vel.square()).mean()
    else:
        contact = q.new_zeros(())
    if float(lambda_dyn) > 0.0:
        dyn_proxy = curve_dynamics_proxy_loss(qdot, qddot, dt=spline.dt)
    else:
        dyn_proxy = q.new_zeros(())
    total = (
        float(lambda_prior) * prior
        + float(lambda_q) * q_body
        + float(lambda_v) * v_body
        + float(lambda_a) * a_body
        + float(lambda_contact) * contact
        + float(lambda_dyn) * dyn_proxy
    )
    return total, {
        'prior': prior,
        'q_body': q_body,
        'v_body': v_body,
        'a_body': a_body,
        'contact': contact,
        'dyn_proxy': dyn_proxy,
    }, (q, qdot, qddot)


def refine_control_points(
    spline,
    control_buffer,
    base_buffer,
    steps=10,
    lr=3e-3,
    lambda_prior=1.0,
    lambda_q=1.0,
    lambda_v=0.03,
    lambda_a=0.0003,
    lambda_contact=0.0,
    lambda_dyn=0.0,
    contact_gate_mode='heuristic',
    contact_height_threshold=0.08,
    contact_velocity_threshold=0.20,
    refine_window=0,
    optimize_body_only=True,
):
    if steps <= 0:
        control_decode = torch.cat((control_buffer, control_buffer[:, -1:]), dim=1)
        q, qdot, qddot = spline(control_decode, return_derivatives=True)
        return control_buffer, q, qdot, qddot, {
            'enabled': False,
            'steps': int(steps),
        }

    full_control_net = control_buffer.detach()
    base = base_buffer.detach()
    if int(refine_window) > 0 and full_control_net.shape[1] > int(refine_window):
        prefix = full_control_net[:, :-int(refine_window)]
        control_net = full_control_net[:, -int(refine_window):]
        base_for_diag = base[:, -int(refine_window):]
    else:
        prefix = None
        control_net = full_control_net
        base_for_diag = base
    ghost = control_net[:, -1:]
    control_decode_net = torch.cat((control_net, ghost), dim=1)
    q_net, qdot_net, qddot_net = spline(control_decode_net, return_derivatives=True)
    qdot_fd_ref = body_qdot_fd_target(q_net[..., 6:], dt=spline.dt).detach()
    contact_gate = None
    if float(lambda_contact) > 0.0:
        if contact_gate_mode != 'heuristic':
            raise ValueError(f'Unsupported contact_gate_mode for control point refinement: {contact_gate_mode}')
        contact_gate = heuristic_contact_gate(
            q_net,
            height_threshold=contact_height_threshold,
            velocity_threshold=contact_velocity_threshold,
        ).detach()

    delta = torch.zeros_like(control_net, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(lr))
    loss_values = []
    component_values = []
    with torch.enable_grad():
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            delta_eff = delta
            if optimize_body_only:
                delta_eff = delta_eff.clone()
                delta_eff[..., :6] = 0.0
            control = control_net + delta_eff
            control_decode = torch.cat((control, control[:, -1:]), dim=1)
            loss, components, _ = control_point_refinement_loss(
                spline,
                control_decode,
                control_decode_net,
                q_net,
                qdot_fd_ref,
                lambda_prior=lambda_prior,
                lambda_q=lambda_q,
                lambda_v=lambda_v,
                lambda_a=lambda_a,
                lambda_contact=lambda_contact,
                lambda_dyn=lambda_dyn,
                contact_gate=contact_gate,
            )
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite control point refinement loss.')
            loss.backward()
            optimizer.step()
            loss_values.append(float(loss.detach()))
            component_values.append({key: float(value.detach()) for key, value in components.items()})

    with torch.no_grad():
        delta_eff = delta
        if optimize_body_only:
            delta_eff = delta_eff.clone()
            delta_eff[..., :6] = 0.0
        control_refined = control_net + delta_eff
        control_decode = torch.cat((control_refined, control_refined[:, -1:]), dim=1)
        q_refined, qdot_refined, qddot_refined = spline(control_decode, return_derivatives=True)
        q_drift = wrapped_angle_delta(q_refined[..., 6:], q_net[..., 6:]).square().mean().sqrt()
        diagnostics = {
        'enabled': True,
        'steps': int(steps),
        'lr': float(lr),
        'refine_window': int(refine_window),
            'lambda_prior': float(lambda_prior),
            'lambda_q': float(lambda_q),
            'lambda_v': float(lambda_v),
            'lambda_a': float(lambda_a),
            'lambda_contact': float(lambda_contact),
            'lambda_dyn': float(lambda_dyn),
            'contact_gate_mode': contact_gate_mode,
            'contact_height_threshold': float(contact_height_threshold),
            'contact_velocity_threshold': float(contact_velocity_threshold),
            'optimize_body_only': bool(optimize_body_only),
            'control_shape': list(control_net.shape),
            'control_decode_shape': list(control_decode.shape),
            'q_shape': list(q_refined.shape),
            'qdot_shape': list(qdot_refined.shape),
            'qddot_shape': list(qddot_refined.shape),
            'loss_initial': loss_values[0] if loss_values else 0.0,
            'loss_final': loss_values[-1] if loss_values else 0.0,
            'component_initial': component_values[0] if component_values else {},
            'component_final': component_values[-1] if component_values else {},
            'delta_control_norm_mean': float((control_refined - control_net).norm(dim=-1).mean()),
            'delta_control_norm_max': float((control_refined - control_net).norm(dim=-1).max()),
            'q_body_drift_rms': float(q_drift),
            'qdot_body_norm_mean': float(qdot_refined[..., 6:].norm(dim=-1).mean()),
            'qddot_body_norm_mean': float(qddot_refined[..., 6:].norm(dim=-1).mean()),
            'contact_gate_mean': float(contact_gate.mean()) if contact_gate is not None else 0.0,
            'contact_gate_max': float(contact_gate.max()) if contact_gate is not None else 0.0,
            'contact_gate_active_fraction': float((contact_gate > 0.5).float().mean()) if contact_gate is not None else 0.0,
            'base_control_delta_norm_mean': float((control_refined - base_for_diag).norm(dim=-1).mean()),
            'all_finite': bool(
                torch.isfinite(control_refined).all()
                and torch.isfinite(q_refined).all()
                and torch.isfinite(qdot_refined).all()
                and torch.isfinite(qddot_refined).all()
            ),
        }
    if prefix is not None:
        full_control_refined = torch.cat((prefix, control_refined.detach()), dim=1)
    else:
        full_control_refined = control_refined.detach()
    return full_control_refined, q_refined, qdot_refined, qddot_refined, diagnostics
