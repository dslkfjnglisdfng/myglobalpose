# PL Joint Control Acc-Aug102 v1 Full

- Root: `data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805`
- Target: SMPL joint-based root-frame pRB target. joint_pRB = (joints[:, [18,19,4,5,15]] - joints[:, 0:1]) @ root_R. This is not the legacy IMU-vertex pRB target.
- Input layout: `aRB[18] + wRB[18] + RRB[45] + gR0[3] + frozen_joint_acc_R[15] + root_acc_smooth_R[3]`
- AMASS best: epoch `1`, loss `0.8822397489263083`
- DIP best: epoch `37`, loss `0.4478730436509757`

| split | joint_pos_l2_m | joint_vel_l2_mps | joint_acc_l2_mps2 | gravity_angle_deg |
|---|---:|---:|---:|---:|
| dip_test_after_amass | 0.277489 | 0.423087 | 43.323986 | 12.944853 |
| dip_test_after_dip | 0.277398 | 0.423258 | 43.328894 | 12.946709 |

Scope: module-level joint-target PL control. No IK/full-pipeline/S4 evaluation is included.
