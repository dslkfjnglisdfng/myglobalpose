"""Joint-target PL helpers for SMPL leaf-joint pRB experiments."""

from pl_curve import PL_BONE_LEAF_JOINT_IDS, joint_pRB_target_from_pose


LEAF5_JOINTS = list(PL_BONE_LEAF_JOINT_IDS)

TARGET_CONTRACT = (
    'SMPL joint-based root-frame pRB target. '
    'joint_pRB = (joints[:, [18,19,4,5,15]] - joints[:, 0:1]) @ root_R. '
    'This is not the legacy IMU-vertex pRB target.'
)


__all__ = ['LEAF5_JOINTS', 'TARGET_CONTRACT', 'joint_pRB_target_from_pose']
