__all__ = ['GPNet']


import time
import torch
import numpy as np
import articulate as art
import carticulate as cart
from scipy.sparse import csc_array
from scipy.sparse.linalg import lsqr
from qpsolvers import solve_qp


class Scene:
    SupportBlock = 0
    GraspBlock = 1

    def __init__(self, viewer):
        self.viewer = viewer
        self.blocks = []

    def add_block(self, type, position, lifetime=120, render=True):
        exist = False
        for block in self.blocks:
            if block['type'] == type and np.abs(block['position'][1] - position[1]) < 0.05:
                position[1] = block['position'][1]
            if block['type'] == type and np.linalg.norm(block['position'] - position) < 0.1:
                block['lifetime'] = lifetime
                exist = True
        if not exist:
            name = str(time.time_ns()) + '_' + str(np.random.randint(0, 1000))
            self.blocks.append({'name': name, 'type': type, 'position': position, 'lifetime': lifetime})
            self.viewer.instantiate(type, name, position, render=render)

    def update(self, render=True):
        for block in self.blocks:
            block['lifetime'] -= 1
            if block['lifetime'] <= 0:
                self.viewer.destroy(block['name'], render=render)
        self.blocks = [block for block in self.blocks if block['lifetime'] > 0]


class GPNet(torch.nn.Module):
    dt = 1 / 60
    mu = 0.7           # environment fiction coefficient
    kp_pose = 3600     # kp in pose PD controller
    kd_pose = 60       # kd in pose PD controller
    kp_tran = 3600     # kp in tran PD controller
    kd_tran = 60       # kd in tran PD controller
    alpha_pd = 0.0     # relaxation in stable PD controller
    floor_y = -0.97    # floor height
    beta_velocity = 1
    beta_cjoint = 1
    beta_extforce = 0.4
    beta_torque = 1e-3 / 80
    v_imu = (1961, 5424, 1176, 4662, 411, 3021)
    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)
    j_contact = (0, 10, 11, 22, 23)

    class Visualization:
        enable = False
        show_residual_force = False
        show_contact_force = True
        show_block = True
        show_contact = False
        show_stationary = False
        show_torque = False

    def __init__(
        self,
        enable_l4_prephysics=False,
        l4_prephysics_module=None,
        l4_euler_seq='XYZ',
        physics_mode='original',
        physics_backend='original_carticulate',
        pl_backend='original',
        pl_curve_module=None,
        ik1_backend='original',
        ik1_curve_module=None,
        ik1_official_module=None,
        l4_qdot_velocity_blend=0.5,
        l4_qstate_alpha=0.1,
        l4_qstate_max_delta=0.1,
        l4_qstate_gate='no_gate',
    ):
        from articulate.utils.torch import RNN, RNNWithInit
        super(GPNet, self).__init__()
        self.plnet = RNNWithInit(input_linear=False,
                                 input_size=84,
                                 output_size=18,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)
        self.iknet = torch.nn.ModuleDict({
            'net1': RNN(input_linear=False,
                        input_size=63,
                        output_size=72,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4),
            'net2': RNN(input_linear=False,
                        input_size=117,
                        output_size=90,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4)
        })
        self.vrnet = RNNWithInit(input_linear=False,
                                 input_size=243,
                                 output_size=9,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)

        # for training
        # self.plnet.load_state_dict(torch.load('data/weights/Pose-GR/PL/best_weights.pt'))
        # self.iknet.load_state_dict(torch.load('data/weights/Pose-GR/IK/best_weights.pt'))
        # self.vrnet.load_state_dict(torch.load('data/weights/Tran-OV/VR/best_weights.pt'))

        # for testing
        self.load_state_dict(torch.load('data/weights.pt', map_location='cpu'))

        self.B = np.array([[self.mu, -self.mu, 0,       0       ],
                           [1,       1,        1,       1       ],
                           [0,       0,        self.mu, -self.mu]]) / np.sqrt(1 + self.mu ** 2)  # basis of friction cone
        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.v_imu)
        self.physics_model = cart.get_dynamic_model('models/SMPL_male.pkl')
        self.physics_backend = physics_backend
        self.pl_backend = pl_backend
        self.pl_curve = pl_curve_module
        self.ik1_backend = ik1_backend
        self.ik1_curve = ik1_curve_module
        self.ik1_official = ik1_official_module
        self.last_pl_curve_debug = {}
        self.last_ik1_curve_debug = {}
        if self.pl_backend not in ('original', 'curve_v1'):
            raise ValueError(f'Unsupported pl_backend: {self.pl_backend}')
        if self.pl_backend == 'curve_v1' and self.pl_curve is None:
            raise ValueError('pl_backend=curve_v1 requires pl_curve_module.')
        if self.ik1_backend not in ('original', 'curve_v1', 'control_point_v1', 'official_input_v1'):
            raise ValueError(f'Unsupported ik1_backend: {self.ik1_backend}')
        if self.ik1_backend in ('curve_v1', 'control_point_v1') and self.ik1_curve is None:
            raise ValueError(f'ik1_backend={self.ik1_backend} requires ik1_curve_module.')
        if self.ik1_backend == 'official_input_v1' and self.ik1_official is None:
            raise ValueError('ik1_backend=official_input_v1 requires ik1_official_module.')
        self.pip_physics_backend = None
        self.last_pip_physics_debug = {}
        if self.physics_backend == 'pip_physics_v1':
            from pip_physics_backend import PIPPhysicsBackendV1
            self.pip_physics_backend = PIPPhysicsBackendV1(dt=self.dt, use_imu_acc=False)
        elif self.physics_backend != 'original_carticulate':
            raise ValueError(f'Unsupported physics_backend: {self.physics_backend}')
        # self.rnn_initialize()  # using T-pose
        self.eval()
        self.enable_l4_prephysics = bool(enable_l4_prephysics)
        self.l4_prephysics = None
        self.last_l4_prephysics_debug = {}
        from l4_physics_adapter import L4PhysicsAdapter
        self.l4_physics_adapter = L4PhysicsAdapter(
            mode=physics_mode,
            qdot_velocity_blend=l4_qdot_velocity_blend,
            qstate_alpha=l4_qstate_alpha,
            qstate_max_delta=l4_qstate_max_delta,
            qstate_gate=l4_qstate_gate,
        )
        if self.enable_l4_prephysics:
            from l4_tail_update_qstate import L4PrePhysicsRefiner
            self.l4_prephysics = L4PrePhysicsRefiner(
                l4_prephysics_module,
                euler_seq=l4_euler_seq,
            )

        if self.Visualization.enable:
            from articulate.utils.unity import MotionViewer
            MotionViewer.colors = [(1, 1, 1)]
            self.viewer = MotionViewer(1)
            self.viewer.connect()
            self.scene = Scene(self.viewer)
            self.force_lpf = [art.LowPassFilter(0.3) for _ in range(5)]
            self.torque_lpf = art.LowPassFilter(0.3)
            self.tran_offset = torch.zeros(3)
            if self.Visualization.show_torque:
                self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19])

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None, init_vel=None, offset_r=None):
        r"""
        Initialize the hidden states of the RNNs.

        :param init_pose: Pose in shape [24, 3, 3]. T-pose by default.
        :param init_vel: Root world-space velocity in shape [3]. Zero by default.
        :param offset_r: Optional first-frame IMU position offset in shape [6, 3].
        """
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        init_vel = torch.zeros(3) if init_vel is None else init_vel.cpu().view(3)
        vRR_V = init_vel[1].view(1).clone()
        init_vel[1] = 0
        _, j, v = self.body_model.forward_kinematics(init_pose, calc_mesh=True)
        pRL, gR = (v[0, :5] - v[0, 5:]).mm(init_pose[0, 0]).ravel(), -init_pose[0, 0, 1]
        x1 = torch.cat((pRL, gR)).to(self.plnet.init_net[0].weight.device)
        h, vRR_H, c = -j[:, :, 1].min().view(1), init_pose[0, 0].t().mm(init_vel.unsqueeze(-1)).squeeze(-1), torch.zeros(5)
        x2 = torch.cat((vRR_V, vRR_H, c)).to(self.vrnet.init_net[0].weight.device)
        self.pl1hc = [_.contiguous() for _ in self.plnet.init_net(x1).view(1, 2, self.plnet.num_layers, self.plnet.hidden_size).permute(1, 2, 0, 3)]
        self.vr1hc = [_.contiguous() for _ in self.vrnet.init_net(x2).view(1, 2, self.vrnet.num_layers, self.vrnet.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None
        self.last_cjoint = torch.tensor([0, h + self.floor_y, 0]) + j[0, self.j_contact]
        self.is_init = False
        self.contact = np.zeros(5, dtype=bool)
        self.contact_counter = np.zeros(5, dtype=int)
        self.last_l4_prephysics_debug = {}
        self.l4_physics_adapter.last_debug = {}
        self.last_pip_physics_debug = {}
        if self.pip_physics_backend is not None:
            self.pip_physics_backend.reset()
        if self.l4_prephysics is not None:
            self.l4_prephysics.reset()
        if self.pl_curve is not None:
            if getattr(self.pl_curve, 'init_size', 18) == 18:
                self.pl_curve.reset_stream(init_output=x1)
            elif self.pl_curve.init_size == 36:
                if offset_r is None:
                    raise ValueError('PL curve init_size=36 requires offset_r for rnn_initialize.')
                offset = offset_r.detach().cpu().reshape(-1)
                if offset.shape[-1] != 18:
                    raise ValueError(f'Expected offset_r flatten dim 18, got {offset.shape[-1]}.')
                pl_init = torch.cat((offset, pRL, gR)).to(x1.device, x1.dtype)
                self.pl_curve.reset_stream(init_feature=pl_init)
            else:
                raise ValueError(f'Unsupported PL curve init_size={self.pl_curve.init_size}.')
        if self.ik1_curve is not None:
            self.ik1_curve.reset_stream()

    def _run_pl_stage(self, x_pl_in):
        x, self.pl1hc = self.plnet.rnn(x_pl_in.view(1, 1, -1), self.pl1hc)
        base = self.plnet.linear2(x.squeeze())
        if self.pl_backend == 'curve_v1':
            curve = self.pl_curve.step(x_pl_in.to(base.device), base)
            pl_out = curve['pl_t'][0]
            self.last_pl_curve_debug = dict(self.pl_curve.last_debug)
        else:
            pl_out = base
            self.last_pl_curve_debug = {}
        gR1 = art.math.normalize_tensor(pl_out[15:])
        return pl_out, gR1

    def _pl_control_tail(self, pRB, gR1):
        if self.pl_curve is not None and getattr(self.pl_curve, 'control_buffer', None) is not None:
            control = self.pl_curve.control_buffer[:, -4:, :].detach()
            if control.shape[1] < 4:
                pad = control[:, :1].expand(-1, 4 - control.shape[1], -1)
                control = torch.cat((pad, control), dim=1)
            return control[0].reshape(-1).to(pRB.device)
        current = torch.cat((pRB, gR1)).view(1, 18).expand(4, 18)
        return current.reshape(-1)

    def _run_ik1_stage(self, RRB_after_pl, gR1, pRB):
        x_ik1_official = torch.cat((RRB_after_pl.ravel(), gR1, pRB))
        ik1_net = self.ik1_official if self.ik1_backend == 'official_input_v1' else self.iknet.net1
        x, self.ik1hc = ik1_net.rnn(x_ik1_official.view(1, 1, -1), self.ik1hc)
        base = ik1_net.linear2(x.squeeze())
        if self.ik1_backend == 'curve_v1':
            feature = torch.cat((RRB_after_pl.ravel(), gR1, self._pl_control_tail(pRB, gR1))).to(base.device)
            curve = self.ik1_curve.step(feature, base, pRB.to(base.device))
            ik1_out = curve['ik1_t'][0]
            self.last_ik1_curve_debug = dict(self.ik1_curve.last_debug)
        elif self.ik1_backend == 'control_point_v1':
            feature = torch.cat((RRB_after_pl.ravel(), self._pl_control_tail(pRB, gR1), gR1)).to(base.device)
            curve = self.ik1_curve.step(feature, base)
            ik1_out = curve['ik1_t'][0]
            self.last_ik1_curve_debug = dict(self.ik1_curve.last_debug)
        else:
            ik1_out = base
            self.last_ik1_curve_debug = {}
        gR2 = art.math.normalize_tensor(ik1_out[69:])
        return ik1_out, gR2

    @torch.no_grad()
    def _explain_residual_force(self, contact, contact_Jacobian, contact_position, residual_force):
        if np.any(contact):
            J = contact_Jacobian.reshape(5, 3, 75)[contact, :, :6].reshape(-1, 6)
            B, lb = [], []
            for i in np.where(contact)[0]:
                if i > 3 and contact_position[i, 1] > self.floor_y + 0.15:      # hand grasp
                    B.append(np.eye(3))
                    lb.append(-np.ones(3) * np.inf)
                else:
                    B.append(self.B)
                    lb.append(np.zeros(4))
            JTB = J.T @ art.math.block_diagonal_matrix_np(B)
            P = csc_array(JTB.T @ JTB + self.beta_extforce * np.eye(JTB.shape[1]))
            q = -JTB.T @ residual_force
            force = solve_qp(P, q, lb=np.concatenate(lb), solver='osqp')
            error = np.linalg.norm(JTB @ force - residual_force)
        else:
            B = None
            force = np.zeros(0)
            error = np.linalg.norm(residual_force)
        return force, error, B

    @torch.no_grad()
    def forward_prephysics_features(self, a, w, R, prephysics_tran=None, euler_seq='XYZ'):
        r"""
        Run only the official neural PL/IK/VR stack for one frame.

        This helper intentionally does not touch carticulate physics state: no
        ``get_state_R``, ``update_state``, QP, LSQR, contact tracking, or final
        post-physics pose/translation is used. It is for L4 pre-physics cache
        extraction; ``forward_frame`` remains the official inference path.
        """
        from l4_q75_utils import pose_tran_to_q75

        aRB = a.mm(R[5])
        wRB = w.mm(R[5])
        RRB = R[5].t().matmul(R[:5])
        gR0 = -R[5, 1]

        # PL-s1
        x = torch.cat((aRB.ravel(), wRB.ravel(), RRB.ravel(), gR0))
        x, gR1 = self._run_pl_stage(x)
        RRB = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)

        # IK-s1
        x = torch.cat((RRB.ravel(), gR1, x[:15]))
        x, self.ik1hc = self.iknet.net1.rnn(x.view(1, 1, -1), self.ik1hc)
        x = self.iknet.net1.linear2(x.squeeze())
        gR2 = art.math.normalize_tensor(x[69:])
        RRB = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB)

        # IK-s2
        x = torch.cat((RRB.ravel(), gR2, x[:69]))
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())

        RRJ = art.math.r6d_to_rotation_matrix(x).cpu()
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RRJ.view(1, 15, 3, 3)
        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose[self.j_ignore, ...] = torch.eye(3)
        pRJ = self.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
        pose[0] = R[5].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze()).cpu()

        # VR-s1
        aRB = a.cpu().mm(pose[0])
        wRB = w.cpu().mm(pose[0])
        x = torch.cat((RRJ.ravel(), pRJ.ravel(), aRB.ravel(), wRB.ravel(), gR2.cpu())).to(gR2.device)
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())

        vRR_V, vRR_H, stationary_prob = x[0].item(), x[1:4].cpu(), x[4:].sigmoid().cpu()
        vWR = pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1)
        vWR[1] = vRR_V
        if prephysics_tran is None:
            prephysics_tran = torch.zeros(3)
        q75_prephysics = pose_tran_to_q75(
            pose.detach().cpu().view(1, 24, 3, 3),
            prephysics_tran.detach().cpu().view(1, 3),
            euler_seq=euler_seq,
        )[0]
        return {
            'pose_prephysics': pose.detach().cpu(),
            'q75_prephysics': q75_prephysics.detach().cpu(),
            'v_root_vr': vWR.detach().cpu(),
            'stationary_prob': stationary_prob.detach().cpu(),
            'RRJ': RRJ.detach().cpu(),
            'pRJ': pRJ.detach().cpu(),
            'gR2': gR2.detach().cpu(),
        }

    @torch.no_grad()
    def forward_until_ik1(self, a, w, R):
        r"""
        Diagnostic-only PL-s1/IK-s1 forward path for curve-state redesign.

        This helper updates PL-s1 and IK-s1 recurrent states exactly like
        ``forward_frame`` but stops before IK-s2. It does not touch IK-s2,
        VR-s1, velocity fusion, carticulate physics, or L4 refinement.
        """
        aRB0 = a.mm(R[5])
        wRB0 = w.mm(R[5])
        RRB0 = R[5].t().matmul(R[:5])
        gR0 = -R[5, 1]

        # PL-s1
        x_pl_in = torch.cat((aRB0.ravel(), wRB0.ravel(), RRB0.ravel(), gR0))
        x_pl, gR1 = self._run_pl_stage(x_pl_in)
        pRB = x_pl[:15]
        RRB_after_pl = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB0)

        # IK-s1
        x_ik1_in = torch.cat((RRB_after_pl.ravel(), gR1, pRB))
        x_ik1, gR2 = self._run_ik1_stage(RRB_after_pl, gR1, pRB)
        pRJ_ik1 = x_ik1[:69]
        RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB_after_pl)
        ik2_teacher_input = torch.cat((RRB_after_ik1.ravel(), gR2, pRJ_ik1))

        return {
            'aRB0': aRB0.detach().cpu(),
            'wRB0': wRB0.detach().cpu(),
            'RRB0': RRB0.detach().cpu(),
            'gR0': gR0.detach().cpu(),
            'pl_input': x_pl_in.detach().cpu(),
            'pRB': pRB.detach().cpu(),
            'gR1': gR1.detach().cpu(),
            'RRB_after_pl': RRB_after_pl.detach().cpu(),
            'ik1_input': x_ik1_in.detach().cpu(),
            'pRJ_ik1': pRJ_ik1.detach().cpu(),
            'gR2': gR2.detach().cpu(),
            'RRB_after_ik1': RRB_after_ik1.detach().cpu(),
            'ik2_teacher_input': ik2_teacher_input.detach().cpu(),
        }

    @torch.no_grad()
    def forward_frame_from_curve_pose(self, a, w, R, pose_curve, gR2_curve, vr_override=None, vr_override_delta=None):
        r"""
        Experimental no-IK-s2 downstream path.

        ``pose_curve`` must come from the Curve-Control Pose Head branch. This
        method recomputes all pose-dependent VR/physics inputs from that pose
        and then runs the original VR-s1, velocity fusion, and physics logic.
        It intentionally does not call IK-s2.
        """
        pose = pose_curve.detach().cpu().view(24, 3, 3)
        vr_device = next(self.vrnet.parameters()).device
        gR2 = gR2_curve.detach().to(vr_device).view(3)

        pose_body = pose.clone()
        pose_body[0] = torch.eye(3)
        glb_pose_body, joint_body = self.body_model.forward_kinematics(pose_body.unsqueeze(0))[:2]
        RRJ = glb_pose_body[0, self.j_reduce].contiguous()
        pRJ = joint_body[0, 1:]

        # VR-s1
        aRB = a.cpu().mm(pose[0])
        wRB = w.cpu().mm(pose[0])
        x = torch.cat((RRJ.ravel(), pRJ.ravel(), aRB.ravel(), wRB.ravel(), gR2.detach().cpu())).to(vr_device)
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        official_vr = self.vrnet.linear2(x.squeeze())
        x = official_vr
        used_vr_override = vr_override is not None or vr_override_delta is not None
        if vr_override is not None:
            x = vr_override.detach().to(vr_device, official_vr.dtype).view(9)
        elif vr_override_delta is not None:
            x = official_vr + vr_override_delta.detach().to(vr_device, official_vr.dtype).view(9)

        # get translation estimation
        vRR_V, vRR_H, stationary_prob = x[0].item(), x[1:4].cpu(), x[4:].sigmoid().cpu()
        vWR = pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1)
        vWR[1] = vRR_V
        cjoint = torch.cat((torch.zeros(1, 3), pRJ.mm(pose[0].t())))[self.j_contact, :]
        stationary_weight = (stationary_prob * 5 - 3).clip(0, 1)
        velocity = (stationary_weight.unsqueeze(0).mm(self.last_cjoint - cjoint)[0] / self.dt + self.beta_velocity * vWR) / (self.beta_velocity + stationary_weight.sum())
        velocity = self.l4_physics_adapter.adapt_root_velocity(
            velocity,
            self.last_l4_prephysics_debug if self.l4_prephysics is not None else {},
        )
        self.last_l4_prephysics_debug['physics_adapter'] = dict(self.l4_physics_adapter.last_debug)
        self.last_cjoint = cjoint

        if self.physics_backend == 'pip_physics_v1':
            if not self.is_init:
                self.is_init = True
                tran_hint = torch.tensor([0, self.floor_y - cjoint[:, 1].min().item(), 0], dtype=torch.float32)
            else:
                tran_hint = None
            refined_pose, refined_tran = self.pip_physics_backend.step(
                pose_target=pose,
                velocity_target=velocity,
                stationary_prob=stationary_prob,
                acc=a.cpu(),
                tran_hint=tran_hint,
            )
            self.last_pip_physics_debug = dict(self.pip_physics_backend.last_debug)
            return refined_pose, refined_tran

        # physics optimization
        if not self.is_init:
            self.is_init = True
            self.physics_model.set_state_R(pose.numpy(), np.array([0, self.floor_y - cjoint[:, 1].min().item(), 0]), np.zeros(75))
        else:
            pose_cur, tran_cur, qdot = self.physics_model.get_state_R()
            cjoint_cur = np.vstack([self.physics_model.get_position(j) for j in self.j_contact])
            cvel_cur = np.vstack([self.physics_model.get_linear_velocity(j) for j in self.j_contact])
            cJ_cur = np.vstack([self.physics_model.get_linear_Jacobian(j) for j in self.j_contact])
            cJdot_cur = np.vstack([self.physics_model.get_linear_Jacobian_dot(j) for j in self.j_contact])
            M = self.physics_model.mass_matrix()
            h = self.physics_model.inverse_dynamics(np.zeros(75))
            stationary = stationary_prob.numpy() > 0.7

            R_delta = art.math.axis_angle_to_rotation_matrix(torch.from_numpy(qdot[3:]) * self.alpha_pd * self.dt)
            delta_pose = art.math.rotation_matrix_to_axis_angle(torch.from_numpy(pose_cur).bmm(R_delta).transpose(1, 2).bmm(pose)).ravel().numpy()
            thetaddotdes = (self.kp_pose * delta_pose - self.kd_pose * qdot[3:]) / (1 + self.kd_pose * self.alpha_pd * self.dt)
            cjoint = tran_cur + velocity.numpy() * self.dt + cjoint.numpy()
            cjoint = art.math.lerp(cjoint, cjoint_cur, stationary_weight.view(5, 1).numpy())
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)

            cjoint_cur[0, 1] -= 0.15
            cjoint[0, 1] -= 0.15
            k = np.ones((5, 3)) * self.beta_cjoint
            k[self.contact] *= 10
            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque) * (-h)))
            qddot = lsqr(csc_array(A), b)[0]
            residual_force = M[:6] @ qddot + h[:6]

            vdist = np.abs(cjoint_cur[np.newaxis, :, 1] - cjoint_cur[:, np.newaxis, 1])
            contact = stationary & (self.contact | (cjoint_cur[:, 1] < self.floor_y + 0.05))
            if np.any(contact):
                contact |= stationary & (vdist[contact].min(axis=0) < 0.05)
            potential_contact = stationary & ~contact
            if contact[0] or potential_contact[0]:
                lleg = self.physics_model.get_position(4) - self.physics_model.get_position(1)
                rleg = self.physics_model.get_position(5) - self.physics_model.get_position(2)
                if min(np.arccos(-lleg[1] / np.linalg.norm(lleg)), np.arccos(-rleg[1] / np.linalg.norm(rleg))) < np.pi / 4:
                    contact[0], potential_contact[0] = False, False

            force, err, forceB = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
            for i in np.argsort(cjoint_cur[:, 1]):
                if err > 400 and potential_contact[i]:
                    contact[i] = True
                    force_new, err_new, forceB_new = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
                    self.contact_counter[i] = self.contact_counter[i] + 1 if err_new / err < 0.6 else 0
                    if self.contact_counter[i] >= 5:
                        force, err, forceB = force_new, err_new, forceB_new
                    else:
                        contact[i] = False
                else:
                    self.contact_counter[i] = 0

            near_ground = cjoint[:, 1] < self.floor_y + 0.15
            for i in np.where(contact & near_ground)[0]:
                cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1)
            cjoint[cjoint[:, 1] < self.floor_y, 1] = self.floor_y

            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
            if np.any(contact):
                J = cJ_cur.reshape(5, 3, 75)[contact].reshape(-1, 75)
                B = art.math.block_diagonal_matrix_np(forceB)
                force = B @ force
                torque = J.T @ force
            else:
                torque = np.zeros(75)
            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque * 3) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque * 3) * (-h + torque)))
            qddot = lsqr(csc_array(A), b)[0]
            self.physics_model.update_state(qddot, self.dt)
            self.contact = contact

        refined_pose, refined_tran, qdot = self.physics_model.get_state_R()
        debug = {
            'RRJ': RRJ.detach().cpu(),
            'pRJ': pRJ.detach().cpu(),
            'aRB': aRB.detach().cpu(),
            'wRB': wRB.detach().cpu(),
            'gR2': gR2.detach().cpu(),
            'stationary_prob': stationary_prob.detach().cpu(),
            'used_vr_override': used_vr_override,
            'official_vr_norm': official_vr.detach().norm().cpu(),
        }
        if used_vr_override:
            debug.update({
                'override_vr_norm': x.detach().norm().cpu(),
                'override_delta_norm': (x.detach() - official_vr.detach()).norm().cpu(),
            })
        return torch.from_numpy(refined_pose), torch.from_numpy(refined_tran), debug

    @torch.no_grad()
    def forward_frame(self, a, w, R, l4_a=None, l4_w=None, l4_R=None):
        aRB = a.mm(R[5])
        wRB = w.mm(R[5])
        RRB = R[5].t().matmul(R[:5])
        gR0 = -R[5, 1]

        # PL-s1
        x = torch.cat((aRB.ravel(), wRB.ravel(), RRB.ravel(), gR0))
        x, gR1 = self._run_pl_stage(x)             # pRB, gR
        RRB = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)

        # IK-s1
        x = self._run_ik1_stage(RRB, gR1, x[:15])[0]   # pRJ, gR
        gR2 = art.math.normalize_tensor(x[69:])
        RRB = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB)

        # IK-s2
        x = torch.cat((RRB.ravel(), gR2, x[:69]))
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())   # RRJ

        # get pose estimation
        RRJ = art.math.r6d_to_rotation_matrix(x).cpu()
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RRJ.view(1, 15, 3, 3)
        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose[self.j_ignore, ...] = torch.eye(3)
        pRJ = self.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
        pose[0] = R[5].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze()).cpu()

        if self.l4_prephysics is not None:
            if self.is_init:
                _, prephysics_tran_np, _ = self.physics_model.get_state_R()
                prephysics_tran = torch.from_numpy(prephysics_tran_np).float()
            else:
                prephysics_tran = torch.zeros(3)
            l4_a = a if l4_a is None else l4_a
            l4_w = w if l4_w is None else l4_w
            l4_R = R if l4_R is None else l4_R
            pose, _, l4_changed = self.l4_prephysics.refine(pose, prephysics_tran, l4_a, l4_w, l4_R)
            self.last_l4_prephysics_debug = dict(self.l4_prephysics.last_debug)
            if l4_changed:
                pose_body = pose.clone()
                pose_body[0] = torch.eye(3)
                glb_pose_body, joint_body = self.body_model.forward_kinematics(pose_body.unsqueeze(0))[:2]
                RRJ = glb_pose_body[0, self.j_reduce].contiguous()
                pRJ = joint_body[0, 1:]
                root_from_imu = R[5].t().mm(pose[0].to(R.device))
                gR2 = root_from_imu.t().mm(gR0.view(3, 1)).view(3)

        # VR-s1
        aRB = a.cpu().mm(pose[0])
        wRB = w.cpu().mm(pose[0])
        x = torch.cat((RRJ.ravel(), pRJ.ravel(), aRB.ravel(), wRB.ravel(), gR2.cpu())).to(gR2.device)
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())  # vRR_V, vRR_H, stationary_prob

        # get translation estimation
        vRR_V, vRR_H, stationary_prob = x[0].item(), x[1:4].cpu(), x[4:].sigmoid().cpu()
        vWR = pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1)
        vWR[1] = vRR_V
        if self.l4_prephysics is not None:
            vWR, _ = self.l4_prephysics.refine_velocity(vWR, stationary_prob)
            self.last_l4_prephysics_debug = dict(self.l4_prephysics.last_debug)
            self.last_l4_prephysics_debug['stationary_prob'] = stationary_prob.detach().cpu().clone()
        cjoint = torch.cat((torch.zeros(1, 3), pRJ.mm(pose[0].t())))[self.j_contact, :]
        stationary_weight = (stationary_prob * 5 - 3).clip(0, 1)
        velocity = (stationary_weight.unsqueeze(0).mm(self.last_cjoint - cjoint)[0] / self.dt + self.beta_velocity * vWR) / (self.beta_velocity + stationary_weight.sum())
        velocity = self.l4_physics_adapter.adapt_root_velocity(
            velocity,
            self.last_l4_prephysics_debug if self.l4_prephysics is not None else {},
        )
        self.last_l4_prephysics_debug['physics_adapter'] = dict(self.l4_physics_adapter.last_debug)
        self.last_cjoint = cjoint

        if self.physics_backend == 'pip_physics_v1':
            if not self.is_init:
                self.is_init = True
                tran_hint = torch.tensor([0, self.floor_y - cjoint[:, 1].min().item(), 0], dtype=torch.float32)
            else:
                tran_hint = None
            refined_pose, refined_tran = self.pip_physics_backend.step(
                pose_target=pose,
                velocity_target=velocity,
                stationary_prob=stationary_prob,
                acc=a.cpu(),
                tran_hint=tran_hint,
            )
            self.last_pip_physics_debug = dict(self.pip_physics_backend.last_debug)
            return refined_pose, refined_tran

        # physics optimization
        if not self.is_init:
            self.is_init = True
            self.physics_model.set_state_R(pose.numpy(), np.array([0, self.floor_y - cjoint[:, 1].min().item(), 0]), np.zeros(75))
        else:
            # get current state
            pose_cur, tran_cur, qdot = self.physics_model.get_state_R()
            cjoint_cur = np.vstack([self.physics_model.get_position(j) for j in self.j_contact])
            cvel_cur = np.vstack([self.physics_model.get_linear_velocity(j) for j in self.j_contact])
            cJ_cur = np.vstack([self.physics_model.get_linear_Jacobian(j) for j in self.j_contact])
            cJdot_cur = np.vstack([self.physics_model.get_linear_Jacobian_dot(j) for j in self.j_contact])
            M = self.physics_model.mass_matrix()
            h = self.physics_model.inverse_dynamics(np.zeros(75))
            stationary = stationary_prob.numpy() > 0.7

            # dual RSPD controller
            R = art.math.axis_angle_to_rotation_matrix(torch.from_numpy(qdot[3:]) * self.alpha_pd * self.dt)
            delta_pose = art.math.rotation_matrix_to_axis_angle(torch.from_numpy(pose_cur).bmm(R).transpose(1, 2).bmm(pose)).ravel().numpy()
            thetaddotdes = (self.kp_pose * delta_pose - self.kd_pose * qdot[3:]) / (1 + self.kd_pose * self.alpha_pd * self.dt)
            cjoint = tran_cur + velocity.numpy() * self.dt + cjoint.numpy()
            cjoint = art.math.lerp(cjoint, cjoint_cur, stationary_weight.view(5, 1).numpy())
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)

            # unconstrained tracking
            cjoint_cur[0, 1] -= 0.15
            cjoint[0, 1] -= 0.15
            k = np.ones((5, 3)) * self.beta_cjoint
            k[self.contact] *= 10
            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque) * (-h)))
            qddot = lsqr(csc_array(A), b)[0]
            residual_force = M[:6] @ qddot + h[:6]

            # determine potential contact
            vdist = np.abs(cjoint_cur[np.newaxis, :, 1] - cjoint_cur[:, np.newaxis, 1])
            contact = stationary & (self.contact | (cjoint_cur[:, 1] < self.floor_y + 0.05))
            if np.any(contact):
                contact |= stationary & (vdist[contact].min(axis=0) < 0.05)
            potential_contact = stationary & ~contact
            if contact[0] or potential_contact[0]:  # root joint
                lleg = self.physics_model.get_position(4) - self.physics_model.get_position(1)
                rleg = self.physics_model.get_position(5) - self.physics_model.get_position(2)
                if min(np.arccos(-lleg[1] / np.linalg.norm(lleg)), np.arccos(-rleg[1] / np.linalg.norm(rleg))) < np.pi / 4:
                    contact[0], potential_contact[0] = False, False

            # explain residual force by contacts
            force, err, forceB = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
            for i in np.argsort(cjoint_cur[:, 1]):  # add potential contact from lowest to highest
                if err > 400 and potential_contact[i]:
                    contact[i] = True
                    force_new, err_new, forceB_new = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
                    self.contact_counter[i] = self.contact_counter[i] + 1 if err_new / err < 0.6 else 0
                    if self.contact_counter[i] >= 5:
                        force, err, forceB = force_new, err_new, forceB_new
                    else:
                        contact[i] = False
                else:
                    self.contact_counter[i] = 0

            # update contact position
            near_ground = cjoint[:, 1] < self.floor_y + 0.15
            # object_y = []
            # for i in np.where(contact & ~near_ground)[0]:
            #     is_added = False
            #     for y in object_y:
            #         if not is_added and abs(sum(y) / len(y) - cjoint[i, 1]) < 0.05:
            #             y.append(cjoint[i, 1])
            #             is_added = True
            #     if not is_added:
            #         object_y.append([cjoint[i, 1]])
            # object_y = np.array([sum(y) / len(y) for y in object_y])
            # for i in np.where(contact)[0]:
            #     cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1) if near_ground[i] else object_y[np.abs(cjoint[i, 1] - object_y).argmin()]
            for i in np.where(contact & near_ground)[0]:
                cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1)
            cjoint[cjoint[:, 1] < self.floor_y, 1] = self.floor_y

            # re-optimization and update state
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
            if np.any(contact):
                J = cJ_cur.reshape(5, 3, 75)[contact].reshape(-1, 75)
                B = art.math.block_diagonal_matrix_np(forceB)
                force = B @ force
                torque = J.T @ force
            else:
                torque = np.zeros(75)
            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque * 3) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque * 3) * (-h + torque)))
            qddot = lsqr(csc_array(A), b)[0]
            self.physics_model.update_state(qddot, self.dt)
            self.contact = contact

            # visualization
            if self.Visualization.enable:
                force = iter(force.reshape(-1, 3))
                self.viewer.clear_line(render=False)
                self.viewer.clear_point(render=False)
                self.viewer.update(torch.from_numpy(pose_cur), torch.from_numpy(tran_cur) - self.tran_offset, render=False)
                cjoint -= self.tran_offset.numpy()
                for i in np.where(contact)[0]:
                    if self.Visualization.show_block and not near_ground[i]:
                        if i < 3 or True:   # always show block
                            self.scene.add_block(Scene.SupportBlock, cjoint[i] - [0, 0.2, 0], lifetime=120, render=False)
                        else:
                            self.scene.add_block(Scene.GraspBlock, cjoint[i], lifetime=30, render=False)
                    if self.Visualization.show_contact_force:
                        self.viewer.draw_line(cjoint[i], cjoint[i] + self.force_lpf[i](next(force)) * 0.002, [1, 0, 0], 0.02, render=False)
                    if self.Visualization.show_contact:
                        self.viewer.draw_point(cjoint[i], [0, 1, 0], 0.15, render=False)
                if self.Visualization.show_stationary:
                    for i in np.where(stationary & ~contact)[0]:
                        self.viewer.draw_point(cjoint[i], [0.4, 0.4, 1], 0.15, render=False)
                if self.Visualization.show_block:
                    self.scene.update(render=False)
                if self.Visualization.show_residual_force:
                    self.viewer.draw_line_from_joint(0, 0, residual_force[:3] * 0.001, (0, 0, 1), 0.02, render=False)
                if self.Visualization.show_torque:
                    self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19], render=False)
                    tau = M @ qddot + h - torque
                    tau = self.torque_lpf(tau)
                    self.viewer.update_torque(tau[3:], render=False)
                else:
                    self.viewer.hide_torque(render=False)
                self.viewer.render()

        refined_pose, refined_tran, qdot = self.physics_model.get_state_R()
        return torch.from_numpy(refined_pose), torch.from_numpy(refined_tran)

    def forward(self, x, fast=True):
        # PL-s1
        RRB = [x_[:, 36:81].view(-1, 5, 3, 3) for x_, y_ in x]
        gR0 = [x_[:, 81:] for x_, y_ in x]
        x1 = [(x_, y_[:18]) for x_, y_ in x]
        x1 = self.plnet(x1)       # pRB, gR

        # IK-s1
        pRB = [x_[:, :15] for x_ in x1]
        gR1 = [art.math.normalize_tensor(x_[:, 15:].clone().detach()) for x_ in x1]
        RRB = [art.math.from_to_rotation_matrix(gR0_, gR1_).unsqueeze(1).matmul(RRB_) for gR0_, gR1_, RRB_ in zip(gR0, gR1, RRB)]
        x2 = [torch.cat((RRB_.flatten(1), gR1_, pRB_), dim=1) for RRB_, gR1_, pRB_ in zip(RRB, gR1, pRB)]
        x2 = self.iknet.net1(x2)   # pRJ, gR

        # IK-s2
        pRJ = [x_[:, :69] for x_ in x2]
        gR2 = [art.math.normalize_tensor(x_[:, 69:].clone().detach()) for x_ in x2]
        RRB = [art.math.from_to_rotation_matrix(gR1_, gR2_).unsqueeze(1).matmul(RRB_) for gR1_, gR2_, RRB_ in zip(gR1, gR2, RRB)]
        x3 = [torch.cat((RRB_.flatten(1), gR2_, pRJ_), dim=1) for RRB_, gR2_, pRJ_ in zip(RRB, gR2, pRJ)]
        x3 = self.iknet.net2(x3)   # RRJ

        # VR-s1
        if fast:   # faster approximation
            RRJ = [art.math.r6d_to_rotation_matrix(x3_.detach()).view(-1, 135) for x3_ in x3]
            awRB = [x_[:, :36].view(-1, 12, 3).bmm(art.math.from_to_rotation_matrix(gR2_, gR0_)).view(-1, 36) for gR2_, gR0_, (x_, y_) in zip(gR2, gR0, x)]
            x4 = [(torch.cat((RRJ_, pRJ_.detach(), awRB_, gR2_), dim=1), torch.zeros_like(y_[-9:]) if torch.isnan(y_[-9]) else y_[-9:]) for RRJ_, pRJ_, awRB_, gR2_, (x_, y_) in zip(RRJ, pRJ, awRB, gR2, x)]
        else:
            RRJ, pRJ, aRB_new, wRB_new, pose = [], [], [], [], []
            aRB = [x_[:, :18].view(-1, 6, 3) for x_, y_ in x]
            wRB = [x_[:, 18:36].view(-1, 6, 3) for x_, y_ in x]
            for i in range(len(x3)):
                x3_ = x3[i].clone().detach().cpu()
                RRJ_ = art.math.r6d_to_rotation_matrix(x3_).view(-1, 15, 3, 3)
                glb_pose_ = torch.eye(3).repeat(RRJ_.shape[0], 24, 1, 1)
                glb_pose_[:, self.j_reduce] = RRJ_
                pose_ = self.body_model.inverse_kinematics_R(glb_pose_).view(-1, 24, 3, 3)
                pose_[:, self.j_ignore] = torch.eye(3)
                pRJ_ = self.body_model.forward_kinematics(pose_)[1][:, 1:]
                aRB_ = aRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                wRB_ = wRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                RRJ.append(RRJ_)
                pRJ.append(pRJ_)
                aRB_new.append(aRB_)
                wRB_new.append(wRB_)
                pose.append(pose_)
            no_translation = [torch.isnan(y_[-9]).item() for x_, y_ in x]
            x4 = [(torch.cat((RRJ_.flatten(1).to(aRB_.device), pRJ_.flatten(1).to(aRB_.device), aRB_.flatten(1), wRB_.flatten(1), gR2_), dim=1), x_[1][-9:] if not nt_ else torch.zeros_like(x_[1][-9:])) for RRJ_, pRJ_, aRB_, wRB_, gR2_, x_, nt_ in zip(RRJ, pRJ, aRB_new, wRB_new, gR2, x, no_translation)]
        x4 = self.vrnet(x4)   # vRR_V, vRR_H, stationary_prob
        result = [torch.cat((x1_, x2_, x3_, x4_), dim=1) for x1_, x2_, x3_, x4_ in zip(x1, x2, x3, x4)]
        return result
