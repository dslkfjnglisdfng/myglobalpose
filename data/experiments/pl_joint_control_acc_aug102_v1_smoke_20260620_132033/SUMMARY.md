# PL Joint Control Acc-Aug102 v1

## 1. Question
Smoke-test a new joint-target NewPL control module using frozen joint acceleration as an auxiliary input.

## 2. Why joint target instead of vertex target
The target is SMPL joint-based root-frame leaf positions, not the legacy IMU-vertex pRB target.

## 3. Frozen acceleration source
`/home/lingfeng/projects/imu_acc_explainability/code/outputs/imu_leaf_acc_predictor_v1/full_world_leaf5_no_trans_smoothed_gtacc_centered_ma_w9_20260619_155137/dip_finetune/best.pt`

## 4. Input layout
`aRB[18] + wRB[18] + RRB[45] + gR0[3] + frozen_joint_acc_R[15] + root_acc_smooth_R[3]`

## 5. Target definition
SMPL joint-based root-frame pRB target. joint_pRB = (joints[:, [18,19,4,5,15]] - joints[:, 0:1]) @ root_R. This is not the legacy IMU-vertex pRB target.

## 6. Control-point decoder design
The module predicts 18D control points and decodes joint_pRB/gR plus joint_pRB_dot/joint_pRB_ddot with UniformCubicBSpline.

## 7. Loss weights
`pRB=1.0, gR1=0.3, gt_control_pRB=0.5, gt_control_gR1=0.1, pRB_dot=0.5, pRB_ddot=0.1, pRB_ddot_smooth=0.001, gR_smooth=0.001, control_point_prior=0.001, tail_update_prior=0.001`

## 8. Smoke cache results
`feature_dim=102`, `target_dim=18`, `joint_minus_vertex_l2_mean_m=0.261554`

## 9. Smoke training results
`status=ok`, `best_epoch=1`, `best_loss=0.02633477109066007`

## 10. Eval metrics

| split | joint_pos_l2_m | joint_vel_l2_mps | joint_acc_l2_mps2 | gravity_angle_deg |
|---|---:|---:|---:|---:|
| smoke | 0.267529 | 0.022015 | 1.920843 | 0.582246 |

## 11. Interpretation
This is a smoke result only. It verifies cache construction, forward/backward training, spline derivatives, and metric reporting.

## 12. Limitations
No full training or full-pipeline evaluation was run.

## 13. Artifacts
- Cache manifest: `data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033/cache/dip_val/pl_curve_cache_manifest.json`
- Train result: `data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033/train/train_result.json`
- Eval metrics: `data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033/eval/metrics.json`
