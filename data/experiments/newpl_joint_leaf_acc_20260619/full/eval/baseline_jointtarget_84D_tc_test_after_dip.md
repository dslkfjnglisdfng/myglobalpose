# Joint-Leaf NewPL Module Eval

- Cache: `data/experiments/newpl_joint_leaf_acc_20260619/full/caches/baseline_jointtarget_84D/tc_test/pl_curve_cache_manifest.json`
- Checkpoint: `data/experiments/newpl_joint_leaf_acc_20260619/full/baseline_jointtarget_84D/dip_finetune/best_loss.pt`
- Feature mode: `baseline_jointtarget_84D`
- Scope: module-level decoded joint-leaf PL only; no IK/full-pipeline.

| metric | value |
|---|---:|
| p_leaf_joint_R_l1_cm | 2.530743 |
| p_leaf_joint_R_l2_cm | 5.174598 |
| base_p_leaf_joint_R_l2_cm | 5.187598 |
| gR1_angle_deg | 4.200546 |
| base_gR1_angle_deg | 4.203858 |
| control_p_leaf_joint_R_l2_cm | 5.168195 |
| control_gR1_angle_deg | 4.208381 |

## Per Leaf L2 cm

| leaf | pred | base |
|---|---:|---:|
| left_forearm | 6.408602 | 6.475623 |
| right_forearm | 5.013014 | 5.064811 |
| left_lower_leg | 4.682361 | 4.707471 |
| right_lower_leg | 4.188270 | 4.175006 |
| head | 5.580742 | 5.515078 |
