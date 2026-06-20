# PL Joint Control Acc-Aug102 v1 Cache Smoke

- Experiment: `pl_joint_control_acc_aug102_v1`
- Target: SMPL joint-based root-frame pRB target. joint_pRB = (joints[:, [18,19,4,5,15]] - joints[:, 0:1]) @ root_R. This is not the legacy IMU-vertex pRB target.
- Input layout: `aRB[18] + wRB[18] + RRB[45] + gR0[3] + frozen_joint_acc_R[15] + root_acc_smooth_R[3]`
- Frozen checkpoint: `/home/lingfeng/projects/imu_acc_explainability/code/outputs/imu_leaf_acc_predictor_v1/full_world_leaf5_no_trans_smoothed_gtacc_centered_ma_w9_20260619_155137/dip_finetune/best.pt`
- Coordinate transform: `a_joint_pred_R = a_joint_pred_W @ RMB[:,5]`

| sanity | value |
|---|---:|
| feature_dim | 102 |
| target_dim | 18 |
| joint_minus_vertex_l2_mean_m | 0.261554 |
| frozen_joint_acc_W_l2_norm_mean | 0.123468 |
| frozen_joint_acc_R_l2_norm_mean | 0.123468 |
| root_acc_smooth_W_l2_norm_mean | 0.063012 |
| root_acc_smooth_R_l2_norm_mean | 0.063012 |
