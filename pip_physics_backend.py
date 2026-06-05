import enum
import json
from pathlib import Path

import numpy as np
import torch
from qpsolvers import solve_qp

import articulate as art
from articulate.math import (
    euler_angle_to_rotation_matrix_np,
    euler_convert_np,
    normalize_angle,
    rotation_matrix_to_euler_angle_np,
)


PIP_PROJECT_ROOT = Path('/home/lingfeng/projects/PIP')
PIP_PHYSICS_MODEL_FILE = Path('/home/lingfeng/projects/data/models/rbdl_pip/physics.urdf')
PIP_PHYSICS_PARAMETER_FILE = Path('/home/lingfeng/projects/data/models/physics_parameters.json')

_SMPL_TO_RBDL = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32, 6, 7, 8,
                 15, 16, 17, 24, 25, 26, 36, 37, 38, 45, 46, 47, 51, 52, 53, 57, 58, 59, 63, 64, 65, 39, 40, 41,
                 48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68, 33, 34, 35, 42, 43, 44]
_RBDL_TO_SMPL = [0, 1, 2, 12, 13, 14, 24, 25, 26, 3, 4, 5, 15, 16, 17, 27, 28, 29, 6, 7, 8, 18, 19, 20, 30, 31, 32,
                 9, 10, 11, 21, 22, 23, 63, 64, 65, 33, 34, 35, 48, 49, 50, 66, 67, 68, 36, 37, 38, 51, 52, 53, 39,
                 40, 41, 54, 55, 56, 42, 43, 44, 57, 58, 59, 45, 46, 47, 60, 61, 62]


class Body(enum.Enum):
    ROOT = 2
    PELVIS = 2
    SPINE = 2
    LHIP = 5
    RHIP = 17
    SPINE1 = 29
    LKNEE = 8
    RKNEE = 20
    SPINE2 = 32
    LANKLE = 11
    RANKLE = 23
    SPINE3 = 35
    LFOOT = 14
    RFOOT = 26
    NECK = 68
    LCLAVICLE = 38
    RCLAVICLE = 53
    HEAD = 71
    LSHOULDER = 41
    RSHOULDER = 56
    LELBOW = 44
    RELBOW = 59
    LWRIST = 47
    RWRIST = 62
    LHAND = 50
    RHAND = 65


def read_param_values_from_json(file_path):
    with open(file_path, 'r') as f:
        return {param['name']: param['value'] for param in json.load(f)}


def smpl_to_pip_rbdl(poses, trans):
    poses = np.array(poses).reshape(-1, 24, 3, 3)
    trans = np.array(trans).reshape(-1, 3)
    euler_poses = rotation_matrix_to_euler_angle_np(poses[:, 1:], 'XYZ').reshape(-1, 69)
    euler_glbrots = rotation_matrix_to_euler_angle_np(poses[:, :1], 'xyz').reshape(-1, 3)
    euler_glbrots = euler_convert_np(euler_glbrots[:, [2, 1, 0]], 'xyz', 'zyx')
    qs = np.concatenate((trans, euler_glbrots, euler_poses[:, _SMPL_TO_RBDL]), axis=1)
    qs[:, 3:] = normalize_angle(qs[:, 3:])
    return qs


def pip_rbdl_to_smpl(qs):
    qs = np.array(qs).reshape(-1, 75)
    trans, euler_glbrots, euler_poses = qs[:, :3], qs[:, 3:6], qs[:, 6:][:, _RBDL_TO_SMPL]
    euler_glbrots = euler_convert_np(euler_glbrots, 'zyx', 'xyz')[:, [2, 1, 0]]
    glbrots = euler_angle_to_rotation_matrix_np(euler_glbrots, 'xyz').reshape(-1, 1, 3, 3)
    poses = euler_angle_to_rotation_matrix_np(euler_poses, 'XYZ').reshape(-1, 23, 3, 3)
    return np.concatenate((glbrots, poses), axis=1), trans


def rotation_geodesic_np(a, b):
    rel = np.matmul(np.swapaxes(a, -1, -2), b)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cos)


def q_conversion_diagnostics(pose, tran):
    pose_np = np.asarray(pose.detach().cpu() if torch.is_tensor(pose) else pose, dtype=np.float64).reshape(24, 3, 3)
    tran_np = np.asarray(tran.detach().cpu() if torch.is_tensor(tran) else tran, dtype=np.float64).reshape(3)
    q = smpl_to_pip_rbdl(pose_np[None], tran_np[None])
    pose_rt, tran_rt = pip_rbdl_to_smpl(q)
    geo = rotation_geodesic_np(pose_np[None], pose_rt)[0]
    return {
        'q_shape': list(q.shape),
        'q_norm': float(np.linalg.norm(q)),
        'pose_geodesic_mean_rad': float(np.mean(geo)),
        'pose_geodesic_max_rad': float(np.max(geo)),
        'tran_l2': float(np.linalg.norm(tran_rt[0] - tran_np)),
        'root_geodesic_rad': float(geo[0]),
        'finite': bool(np.isfinite(q).all() and np.isfinite(pose_rt).all() and np.isfinite(tran_rt).all()),
    }


class PIPStylePhysicsOptimizer:
    test_contact_joints = [
        'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2',
        'SPINE3', 'LSHOULDER', 'RSHOULDER', 'HEAD',
        'LELBOW', 'RELBOW', 'LHAND', 'RHAND', 'LFOOT', 'RFOOT',
    ]

    joint_velocity_targets = [
        'ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
        'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
        'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST',
    ]

    def __init__(self, model_file=PIP_PHYSICS_MODEL_FILE, parameter_file=PIP_PHYSICS_PARAMETER_FILE, use_imu_acc=False):
        from articulate.utils.rbdl import RBDLModel

        self.model = RBDLModel(str(model_file), update_kinematics_by_hand=True)
        self.params = read_param_values_from_json(parameter_file)
        self.use_imu_acc = bool(use_imu_acc)
        self.friction_constraint_matrix = np.array([
            [np.sqrt(2), -0.6, 0],
            [-np.sqrt(2), -0.6, 0],
            [0, -0.6, np.sqrt(2)],
            [0, -0.6, -np.sqrt(2)],
        ])
        supp_poly_size = 0.2
        self.support_polygon = np.array([
            [-supp_poly_size / 2, 0, -supp_poly_size / 2],
            [supp_poly_size / 2, 0, -supp_poly_size / 2],
            [-supp_poly_size / 2, 0, supp_poly_size / 2],
            [supp_poly_size / 2, 0, supp_poly_size / 2],
        ])
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.last_x = []
        self.last_debug = {}

    def reset_states(self):
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.last_x = []
        self.last_debug = {}

    def optimize_frame(self, pose, tran_target, jvel, contact_prob, acc=None):
        pose_np = np.asarray(pose.detach().cpu() if torch.is_tensor(pose) else pose, dtype=np.float64).reshape(24, 3, 3)
        tran_np = np.asarray(tran_target.detach().cpu() if torch.is_tensor(tran_target) else tran_target, dtype=np.float64).reshape(3)
        q_ref = smpl_to_pip_rbdl(pose_np[None], tran_np[None])[0]
        v_ref = np.asarray(jvel.detach().cpu() if torch.is_tensor(jvel) else jvel, dtype=np.float64).reshape(24, 3)
        c_ref = np.asarray(contact_prob.detach().cpu() if torch.is_tensor(contact_prob) else contact_prob, dtype=np.float64).reshape(2)
        if acc is None or not self.use_imu_acc:
            a_ref = np.zeros((6, 3), dtype=np.float64)
        else:
            a_ref = np.asarray(acc.detach().cpu() if torch.is_tensor(acc) else acc, dtype=np.float64).reshape(6, 3)

        if self.q is None:
            self.q = q_ref.copy()
            self.last_debug = {'initialized': True, 'num_contacts': 0, 'q_norm': float(np.linalg.norm(self.q))}
            return torch.from_numpy(pose_np).float(), torch.from_numpy(tran_np).float()

        q = self.q
        qdot = self.qdot
        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))

        Js = [np.empty((0, self.model.qdot_size))]
        collision_points = []
        for joint_name in self.test_contact_joints:
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)
            is_left_foot = joint_id == Body.LFOOT and c_ref[0] > 0.5 and pos[1] <= self.params['floor_y'] + 0.03
            is_right_foot = joint_id == Body.RFOOT and c_ref[1] > 0.5 and pos[1] <= self.params['floor_y'] + 0.03
            if is_left_foot or is_right_foot or pos[1] <= self.params['floor_y']:
                for ps in self.support_polygon + pos:
                    collision_points.append(ps)
                    pb = self.model.calc_base_to_body_coordinates(q, joint_id, ps)
                    Js.append(self.model.calc_point_Jacobian(q, joint_id, pb))
        Js = np.vstack(Js)
        nc = len(collision_points)

        dof = self.model.qdot_size
        As1, bs1 = [np.zeros((0, dof))], [np.empty(0)]
        As2, bs2 = [np.empty((0, nc * 3))], [np.empty(0)]
        As3, bs3 = [np.zeros((0, dof))], [np.empty(0)]
        Gs1, hs1 = [np.zeros((0, dof))], [np.empty(0)]
        Gs2, hs2 = [np.empty((0, nc * 3))], [np.empty(0)]
        Gs3, hs3 = [np.zeros((0, dof))], [np.empty(0)]

        A = np.hstack((np.zeros((dof - 3, 3)), np.eye(dof - 3)))
        b = self.params['kp_angular'] * art.math.angle_difference(q_ref[3:], q[3:]) - self.params['kd_angular'] * qdot[3:]
        As1.append(A)
        bs1.append(b)

        for joint_name, v in zip(self.joint_velocity_targets, v_ref[:22]):
            joint_id = vars(Body)[joint_name]
            if joint_id == Body.LFOOT or joint_id == Body.RFOOT:
                continue
            cur_vel = self.model.calc_point_velocity(q, qdot, joint_id)
            a_des = self.params['kp_linear'] * v * self.params['delta_t'] - self.params['kd_linear'] * cur_vel
            A = self.model.calc_point_Jacobian(q, joint_id)
            b = -self.model.calc_point_acceleration(q, qdot, np.zeros(dof), joint_id) + a_des
            As1.append(A * self.params['coeff_jvel'])
            bs1.append(b * self.params['coeff_jvel'])

        if self.use_imu_acc:
            for imu_link, acc_target in zip(
                ['imu_left_forearm', 'imu_right_forearm', 'imu_left_knee', 'imu_right_knee', 'imu_head', 'imu_root'],
                a_ref,
            ):
                body_id = int(self.model.model.GetBodyId(imu_link))
                if body_id == 2**32 - 1:
                    raise KeyError(f'IMU site link `{imu_link}` not found in {PIP_PHYSICS_MODEL_FILE}.')
                joint_id = type('BodyRef', (), {'value': body_id})()
                A = self.model.calc_point_Jacobian(q, joint_id, np.zeros(3))
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(dof), joint_id, np.zeros(3)) + acc_target
                As1.append(A * self.params['coeff_acc'])
                bs1.append(b * self.params['coeff_acc'])

        if nc != 0:
            A = [np.eye(3) * max(cp[1] - self.params['floor_y'], 0.005) for cp in collision_points]
            As2.append(A and art.math.block_diagonal_matrix_np(A) * self.params['coeff_lambda'])
            bs2.append(np.zeros(nc * 3))

        As3.append(art.math.block_diagonal_matrix_np([
            np.eye(6) * self.params['coeff_virtual'],
            np.eye(dof - 6) * self.params['coeff_tau'],
        ]))
        bs3.append(np.zeros(dof))

        for joint_name, stable in zip(['LFOOT', 'RFOOT'], c_ref):
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)
            J = self.model.calc_point_Jacobian(q, joint_id)
            v = self.model.calc_point_velocity(q, qdot, joint_id)
            th = -np.log(min(float(stable), 0.84999) / 0.85)
            th_y = (self.params['floor_y'] - pos[1]) / self.params['delta_t']
            Gs1.append(-self.params['delta_t'] * J)
            hs1.append(v - [-th, th_y, -th])
            Gs1.append(self.params['delta_t'] * J)
            hs1.append(-v + [th, max(th, th_y) + 1e-6, th])

        if nc > 0:
            Gs2.append(art.math.block_diagonal_matrix_np([self.friction_constraint_matrix] * nc))
            hs2.append(np.zeros(nc * 4))

        M = self.model.calc_M(q)
        h = self.model.calc_h(q, qdot)
        A_eq = np.hstack((-M, Js.T, np.eye(dof)))
        b_eq = h

        As1, bs1 = np.vstack(As1), np.concatenate(bs1)
        As2, bs2 = np.vstack(As2), np.concatenate(bs2)
        As3, bs3 = np.vstack(As3), np.concatenate(bs3)
        Gs1, hs1 = np.vstack(Gs1), np.concatenate(hs1)
        Gs2, hs2 = np.vstack(Gs2), np.concatenate(hs2)
        Gs3, hs3 = np.vstack(Gs3), np.concatenate(hs3)
        G = art.math.block_diagonal_matrix_np([Gs1, Gs2, Gs3])
        h_ineq = np.concatenate((hs1, hs2, hs3))
        P = art.math.block_diagonal_matrix_np([As1.T @ As1, As2.T @ As2, As3.T @ As3])
        q_vec = np.concatenate((-As1.T @ bs1, -As2.T @ bs2, -As3.T @ bs3))

        init = self.last_x if len(self.last_x) == len(q_vec) else None
        x = solve_qp(P, q_vec, G, h_ineq, A_eq, b_eq, solver='quadprog', initvals=init)
        solver = 'quadprog'
        if x is None or np.linalg.norm(x) > 10000:
            x = solve_qp(P, q_vec, G, h_ineq, A_eq, b_eq, solver='cvxopt', initvals=init)
            solver = 'cvxopt'
        if x is None:
            raise RuntimeError('PIP physics QP failed with both quadprog and cvxopt.')

        qddot = x[:dof]
        qdot = qdot + qddot * self.params['delta_t']
        q = q + qdot * self.params['delta_t']
        self.q = q
        self.qdot = qdot
        self.last_x = x
        pose_opt, tran_opt = pip_rbdl_to_smpl(q)
        self.last_debug = {
            'initialized': False,
            'solver': solver,
            'num_contacts': int(nc),
            'q_norm': float(np.linalg.norm(q)),
            'qdot_norm': float(np.linalg.norm(qdot)),
            'qddot_norm': float(np.linalg.norm(qddot)),
            'contact_prob': c_ref.tolist(),
            'finite': bool(np.isfinite(q).all() and np.isfinite(qdot).all() and np.isfinite(qddot).all()),
        }
        return torch.from_numpy(pose_opt[0]).float(), torch.from_numpy(tran_opt[0]).float()


class PIPPhysicsBackendV1:
    def __init__(self, dt=1 / 60, use_imu_acc=False):
        self.dt = float(dt)
        self.optimizer = PIPStylePhysicsOptimizer(use_imu_acc=use_imu_acc)
        self.body_model = art.ParametricModel('models/SMPL_male.pkl')
        self.prev_pose = None
        self.prev_tran = None
        self.prev_joints = None
        self.last_debug = {}

    def reset(self):
        self.optimizer.reset_states()
        self.prev_pose = None
        self.prev_tran = None
        self.prev_joints = None
        self.last_debug = {}

    def _joints_world(self, pose, tran):
        joints = self.body_model.forward_kinematics(pose.view(1, 24, 3, 3))[1][0]
        return joints + tran.view(1, 3)

    @staticmethod
    def _globalpose_contact_to_pip(stationary_prob):
        prob = stationary_prob.detach().cpu().float().view(-1)
        if prob.numel() >= 3:
            return prob[1:3].clamp(1e-4, 1 - 1e-4)
        return torch.full((2,), 0.5)

    def step(self, pose_target, velocity_target, stationary_prob, acc=None, tran_hint=None):
        pose = pose_target.detach().cpu().float().view(24, 3, 3)
        if tran_hint is None:
            if self.prev_tran is None:
                tran = torch.zeros(3)
            else:
                tran = self.prev_tran + velocity_target.detach().cpu().float().view(3) * self.dt
        else:
            tran = tran_hint.detach().cpu().float().view(3)

        joints = self._joints_world(pose, tran)
        if self.prev_joints is None:
            jvel = torch.zeros(24, 3)
        else:
            jvel = (joints - self.prev_joints) / self.dt
        contact_prob = self._globalpose_contact_to_pip(stationary_prob)
        pose_out, tran_out = self.optimizer.optimize_frame(pose, tran, jvel, contact_prob, acc=acc)
        self.prev_pose = pose.clone()
        self.prev_tran = tran_out.detach().cpu().float()
        self.prev_joints = joints.detach().cpu().float()
        self.last_debug = {
            **self.optimizer.last_debug,
            'contact_mapping': 'globalpose_stationary_prob[1:3] -> PIP [LFOOT,RFOOT]',
            'tran_hint_norm': float(tran.norm()),
            'jvel_norm': float(jvel.norm()),
            'velocity_target_norm': float(velocity_target.detach().cpu().float().norm()),
        }
        return pose_out, tran_out
