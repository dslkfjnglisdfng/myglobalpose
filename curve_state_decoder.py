import torch

from l4_q75_utils import q75_to_pose_tran


class CurveStateDecoder(torch.nn.Module):
    r"""
    Decode curve control points into q75, qdot, and qddot.

    This is the no-training Phase-1 decoder for the curve-state redesign.  It
    intentionally matches the existing L4 cubic-control formula so diagnostics
    can compare the future main-network path against historical L4 behavior.
    """

    def __init__(self, dt=1.0 / 60.0, euler_seq='XYZ'):
        super().__init__()
        self.dt = float(dt)
        self.euler_seq = euler_seq

    def forward(self, control, return_pose=False):
        squeeze_batch = control.dim() == 2
        if squeeze_batch:
            control = control.unsqueeze(0)
        if control.dim() != 3 or control.shape[-1] != 75:
            raise ValueError(f'Expected control shape [T,75] or [B,T,75], got {tuple(control.shape)}.')

        left = torch.cat((control[:, :1], control[:, :-1]), dim=1)
        right = torch.cat((control[:, 1:], control[:, -1:]), dim=1)
        q75 = (left + 4.0 * control + right) / 6.0
        qdot = (right - left) / (2.0 * self.dt)
        qddot = (left - 2.0 * control + right) / (self.dt ** 2)

        result = {
            'q75': q75.squeeze(0) if squeeze_batch else q75,
            'qdot': qdot.squeeze(0) if squeeze_batch else qdot,
            'qddot': qddot.squeeze(0) if squeeze_batch else qddot,
        }
        if return_pose:
            pose, tran = q75_to_pose_tran(q75.reshape(-1, 75), euler_seq=self.euler_seq)
            pose = pose.reshape(q75.shape[0], q75.shape[1], 24, 3, 3)
            tran = tran.reshape(q75.shape[0], q75.shape[1], 3)
            result['pose'] = pose.squeeze(0) if squeeze_batch else pose
            result['tran'] = tran.squeeze(0) if squeeze_batch else tran
        return result
