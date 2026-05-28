import torch

import articulate as art


def pose_tran_to_q75(pose, tran, euler_seq='XYZ'):
    euler = art.math.rotation_matrix_to_euler_angle(pose.reshape(-1, 3, 3), seq=euler_seq)
    return torch.cat((tran.reshape(-1, 3), euler.reshape(pose.shape[0], 72)), dim=1)


def q75_to_pose_tran(q75, euler_seq='XYZ'):
    tran = q75[:, :3].clone()
    euler = q75[:, 3:].reshape(-1, 3)
    pose = art.math.euler_angle_to_rotation_matrix(euler, seq=euler_seq).reshape(-1, 24, 3, 3)
    return pose, tran


def globalpose_input_feature(a, w, R):
    return torch.cat((
        a.detach().reshape(-1),
        w.detach().reshape(-1),
        R.detach().reshape(-1),
    ))


def rotation_matrix_to_6d(pose):
    """Return the 6D rotation representation from the first two matrix columns."""
    return pose.detach()[..., :, :2].reshape(-1)


def pose_input_feature(q75, pose=None, pose_input_mode='euler_q75', euler_seq='XYZ'):
    if pose_input_mode == 'euler_q75':
        return q75.detach().reshape(-1)
    if pose_input_mode == 'rot6d':
        if pose is None:
            pose, _ = q75_to_pose_tran(q75.detach().view(1, 75), euler_seq=euler_seq)
            pose = pose[0]
        return rotation_matrix_to_6d(pose)
    raise ValueError(f'Unsupported pose_input_mode: {pose_input_mode}')


def pose_input_dim(pose_input_mode):
    if pose_input_mode == 'euler_q75':
        return 75
    if pose_input_mode == 'rot6d':
        return 24 * 6
    raise ValueError(f'Unsupported pose_input_mode: {pose_input_mode}')


def prephysics_feature(q75, a, w, R, pose=None, pose_input_mode='euler_q75', euler_seq='XYZ'):
    return torch.cat((
        pose_input_feature(q75, pose=pose, pose_input_mode=pose_input_mode, euler_seq=euler_seq),
        globalpose_input_feature(a, w, R),
    ))


def prephysics_feature_dim(pose_input_mode):
    return pose_input_dim(pose_input_mode) + 6 * 3 + 6 * 3 + 6 * 3 * 3
