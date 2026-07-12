# GP test-time angular velocity swap

## 结论

- **G1 PL-only：不支持总体姿态改善。** DIP 的 local SIP 略好，但 angle/joint 与 PL/IK1 直接误差整体变差；TotalCapture 的全部主要 local pose 指标明显变差。TotalCapture root translation、root jitter、joint jitter 和 foot slip 改善，说明 PL swap 的影响不是单向退化。
- **G2 VR-only：姿态近似不变，TotalCapture root translation/foot slip 改善，但 root velocity RMSE、root jitter 与 max step 变差。** 这是最清楚的“root translation 改善但 jitter 恶化”情形。
- **G3 all-swap：不支持总体姿态改善，但 TotalCapture global angle 略好，root translation、drift、contact velocity、foot slip 和 joint jitter 明显改善。** local SIP/angle/joint/mesh 与 PL/IK1 模块误差变差。
- 改善并非在多数 pose 序列/指标上一致成立；不同模块与数据集存在明显 trade-off。
- 存在 **PL 直接输出变差、VR/root trajectory 变好**，而不是“PL 改善但 VR 恶化”。DIP 上 PL swap 的 local SIP 略好但模块 pRB/IK1 误差变差；TotalCapture 上 PL swap 的姿态变差而 root/jitter 变好。
- 输入审计复现此前方向：causal RMB w 比 cached wM 更接近解析 FK w；但 GP 输出没有获得一致 pose 改善。因此这轮主要证明了 **FK consistency 改善不等价于 zero-shot GP pose 改善**。

明确边界：**test-time swap 变差，不足以证明新 w 无效**。官方 GP 是在 cached measured wM 输入分布上训练的；若要判断新 w 的上限，下一步应进行 matched-input retraining，再用同协议评估。

## DIP test aggregate（19 sequences）

| variant | motion.local_sip_deg | motion.local_angle_deg | motion.local_joint_cm | motion.local_mesh_cm | motion.global_sip_deg | motion.global_angle_deg | motion.global_joint_cm | motion.global_mesh_cm | motion.root_jitter_km_s3 | motion.joint_jitter_km_s3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0_official | 13.649738 | 8.442486 | 4.755371 | 5.491239 | 13.593697 | 8.311586 | 4.726594 | 5.452566 | 0.119346 | 0.196920 |
| G1_pl_swap | 13.578641 | 8.514711 | 4.762918 | 5.454549 | 13.552447 | 8.375044 | 4.751441 | 5.449779 | 0.115192 | 0.187752 |
| G2_vr_swap | 13.650378 | 8.444679 | 4.756473 | 5.491874 | 13.595776 | 8.316170 | 4.729825 | 5.454639 | 0.127813 | 0.207801 |
| G3_all_swap | 13.581084 | 8.517656 | 4.764960 | 5.456108 | 13.556407 | 8.381047 | 4.756212 | 5.453492 | 0.120928 | 0.195567 |

DIP translation GT 不作为可信主结论；root/translation 辅助值保留在逐序列 JSON 和 comparison.csv 中，但不用于判断。

## TotalCapture test aggregate（4 official sequences）

| variant | motion.local_sip_deg | motion.local_angle_deg | motion.local_joint_cm | motion.local_mesh_cm | motion.global_sip_deg | motion.global_angle_deg | motion.global_joint_cm | motion.global_mesh_cm | motion.root_jitter_km_s3 | motion.joint_jitter_km_s3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0_official | 9.674329 | 12.429153 | 4.385540 | 5.176626 | 9.135217 | 11.723856 | 3.732858 | 4.395319 | 0.353319 | 0.758003 |
| G1_pl_swap | 10.623622 | 12.862608 | 4.986260 | 5.631764 | 9.578120 | 11.614327 | 3.943969 | 4.610736 | 0.248281 | 0.485086 |
| G2_vr_swap | 9.674604 | 12.429612 | 4.386625 | 5.177087 | 9.132980 | 11.723860 | 3.734667 | 4.395988 | 0.413965 | 0.771545 |
| G3_all_swap | 10.626745 | 12.865938 | 4.989192 | 5.634674 | 9.579261 | 11.616563 | 3.948004 | 4.614187 | 0.262073 | 0.481362 |

### TotalCapture root/contact

| variant | root.root_translation_rmse_m | root.root_translation_first_frame_aligned_rmse_m | root.root_trajectory_drift_m | root.root_velocity_rmse_m_s | root.max_frame_root_step_m | root.contact_velocity_mean_m_s | root.foot_slip_mean_m_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0_official | 0.765232 | 0.764856 | 1.296737 | 0.273943 | 0.034471 | 0.133829 | 0.109305 |
| G1_pl_swap | 0.628464 | 0.627971 | 1.127830 | 0.249152 | 0.032277 | 0.119066 | 0.092744 |
| G2_vr_swap | 0.562584 | 0.562061 | 0.867930 | 0.288435 | 0.036840 | 0.096970 | 0.072807 |
| G3_all_swap | 0.486076 | 0.485471 | 0.824403 | 0.255546 | 0.034576 | 0.091319 | 0.065665 |

## PL / IK1 module aggregate

### DIP

| variant | module.PL.pRB_l1_cm | module.PL.pRB_l2_cm | module.PL.pRB_rmse_cm | module.PL.gR1_angle_deg | module.IK1.pRJ_l1_cm | module.IK1.pRJ_l2_cm | module.IK1.pRJ_rmse_cm | module.IK1.gR2_angle_deg | module.IK1.pRJ_first_difference_l2_cm_per_frame | module.IK1.pRJ_second_difference_l2_cm_per_frame2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0_official | 3.174660 | 6.529110 | 4.240561 | 15.267150 | 2.451623 | 5.082862 | 3.642666 | 15.268177 | 0.474984 | 0.398217 |
| G1_pl_swap | 3.340300 | 6.853820 | 4.444069 | 15.291069 | 2.521105 | 5.211408 | 3.736820 | 15.264287 | 0.497305 | 0.397803 |
| G2_vr_swap | 3.174660 | 6.529110 | 4.240561 | 15.267150 | 2.451623 | 5.082862 | 3.642666 | 15.268177 | 0.474984 | 0.398217 |
| G3_all_swap | 3.340300 | 6.853820 | 4.444069 | 15.291069 | 2.521105 | 5.211408 | 3.736820 | 15.264287 | 0.497305 | 0.397803 |

### TotalCapture

| variant | module.PL.pRB_l1_cm | module.PL.pRB_l2_cm | module.PL.pRB_rmse_cm | module.PL.gR1_angle_deg | module.IK1.pRJ_l1_cm | module.IK1.pRJ_l2_cm | module.IK1.pRJ_rmse_cm | module.IK1.gR2_angle_deg | module.IK1.pRJ_first_difference_l2_cm_per_frame | module.IK1.pRJ_second_difference_l2_cm_per_frame2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0_official | 3.097614 | 6.413079 | 4.008243 | 15.258340 | 2.330376 | 4.773225 | 3.388589 | 15.323588 | 0.465775 | 0.402958 |
| G1_pl_swap | 3.583735 | 7.448313 | 4.651524 | 15.469272 | 2.656548 | 5.464340 | 3.964673 | 15.537371 | 0.375749 | 0.218095 |
| G2_vr_swap | 3.097614 | 6.413079 | 4.008243 | 15.258340 | 2.330376 | 4.773225 | 3.388589 | 15.323588 | 0.465775 | 0.402958 |
| G3_all_swap | 3.583735 | 7.448313 | 4.651524 | 15.469272 | 2.656548 | 5.464340 | 3.964673 | 15.537371 | 0.375749 | 0.218095 |

逐 leaf/sensor 统计位于每条序列的 `G*.json`，聚合字段位于各 dataset 的 `aggregate.json` 与 `comparison.csv`。

## Per-sequence win counts

计数规则：指标越低越好，严格小于同序列 G0 记为 win。

```json
{
  "dip": {
    "G1_pl_swap": {
      "motion.local_sip_deg": 9,
      "motion.local_angle_deg": 6,
      "motion.local_joint_cm": 10,
      "motion.local_mesh_cm": 9,
      "motion.global_sip_deg": 7,
      "motion.global_angle_deg": 5,
      "motion.global_joint_cm": 10,
      "motion.global_mesh_cm": 9,
      "motion.root_jitter_km_s3": 14,
      "motion.joint_jitter_km_s3": 15
    },
    "G2_vr_swap": {
      "motion.local_sip_deg": 9,
      "motion.local_angle_deg": 3,
      "motion.local_joint_cm": 9,
      "motion.local_mesh_cm": 10,
      "motion.global_sip_deg": 7,
      "motion.global_angle_deg": 3,
      "motion.global_joint_cm": 6,
      "motion.global_mesh_cm": 8,
      "motion.root_jitter_km_s3": 4,
      "motion.joint_jitter_km_s3": 4
    },
    "G3_all_swap": {
      "motion.local_sip_deg": 9,
      "motion.local_angle_deg": 6,
      "motion.local_joint_cm": 10,
      "motion.local_mesh_cm": 9,
      "motion.global_sip_deg": 7,
      "motion.global_angle_deg": 5,
      "motion.global_joint_cm": 10,
      "motion.global_mesh_cm": 9,
      "motion.root_jitter_km_s3": 8,
      "motion.joint_jitter_km_s3": 8
    }
  },
  "totalcapture": {
    "G1_pl_swap": {
      "motion.local_sip_deg": 1,
      "motion.local_angle_deg": 1,
      "motion.local_joint_cm": 1,
      "motion.local_mesh_cm": 1,
      "motion.global_sip_deg": 1,
      "motion.global_angle_deg": 2,
      "motion.global_joint_cm": 1,
      "motion.global_mesh_cm": 1,
      "motion.root_jitter_km_s3": 4,
      "motion.joint_jitter_km_s3": 4,
      "root.root_translation_rmse_m": 4,
      "root.root_translation_first_frame_aligned_rmse_m": 4,
      "root.root_trajectory_drift_m": 3,
      "root.root_velocity_rmse_m_s": 4,
      "root.max_frame_root_step_m": 2,
      "root.contact_velocity_mean_m_s": 2,
      "root.foot_slip_mean_m_s": 3
    },
    "G2_vr_swap": {
      "motion.local_sip_deg": 1,
      "motion.local_angle_deg": 0,
      "motion.local_joint_cm": 1,
      "motion.local_mesh_cm": 1,
      "motion.global_sip_deg": 1,
      "motion.global_angle_deg": 2,
      "motion.global_joint_cm": 2,
      "motion.global_mesh_cm": 2,
      "motion.root_jitter_km_s3": 0,
      "motion.joint_jitter_km_s3": 1,
      "root.root_translation_rmse_m": 3,
      "root.root_translation_first_frame_aligned_rmse_m": 3,
      "root.root_trajectory_drift_m": 3,
      "root.root_velocity_rmse_m_s": 0,
      "root.max_frame_root_step_m": 2,
      "root.contact_velocity_mean_m_s": 4,
      "root.foot_slip_mean_m_s": 4
    },
    "G3_all_swap": {
      "motion.local_sip_deg": 1,
      "motion.local_angle_deg": 1,
      "motion.local_joint_cm": 1,
      "motion.local_mesh_cm": 1,
      "motion.global_sip_deg": 1,
      "motion.global_angle_deg": 2,
      "motion.global_joint_cm": 1,
      "motion.global_mesh_cm": 1,
      "motion.root_jitter_km_s3": 4,
      "motion.joint_jitter_km_s3": 4,
      "root.root_translation_rmse_m": 4,
      "root.root_translation_first_frame_aligned_rmse_m": 4,
      "root.root_trajectory_drift_m": 4,
      "root.root_velocity_rmse_m_s": 4,
      "root.max_frame_root_step_m": 3,
      "root.contact_velocity_mean_m_s": 4,
      "root.foot_slip_mean_m_s": 4
    }
  }
}
```

## Input w / FK audit（TotalCapture）

| comparison | RMSE | Pearson | mean L2 | cosine |
|---|---:|---:|---:|---:|
| cached wM vs FK w | 1.234275 | 0.725116 | 1.084119 | 0.733463 |
| causal RMB w vs FK w | 0.674296 | 0.872634 | 0.715831 | 0.809420 |
| cached wM vs causal RMB w | 1.148147 | 0.767527 | 1.011997 | 0.786364 |

## Protocol

- Baseline code: `90523d6f38c28ee3a1afd27346cd3624c5efe38a`; official `data/weights.pt` unchanged.
- No training, no acceleration/RMB/IK1/IK2/VR/physics/contact logic change.
- Every sequence resets GP and RMB-derived-w streaming states; inference is chronological with no shuffle.
- Frames 0 and 1 use zero replacement w; frame 2 initializes EMA from raw lag-2 SO(3) velocity; later frames use `0.7 prev + 0.3 raw`.
- Official PL row-vector conversion remains `w_pl_M @ RMB_root`; VR remains `w_vr_M @ predicted_pose_root`.
- Smoke failures/retries and the abandoned unchunked evaluator attempt are preserved under `logs/` and `dip_test_incomplete_unbounded_metric/`; no result from them enters the final tables.
