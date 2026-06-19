# Joint-Leaf NewPL Module Eval

- Cache: `data/experiments/newpl_joint_leaf_acc_20260619/full/caches/acc_root_102D/tc_test/pl_curve_cache_manifest.json`
- Checkpoint: `data/experiments/newpl_joint_leaf_acc_20260619/full/acc_root_102D/amass_pretrain/best_loss.pt`
- Feature mode: `acc_root_102D`
- Scope: module-level decoded joint-leaf PL only; no IK/full-pipeline.

| metric | value |
|---|---:|
| p_leaf_joint_R_l1_cm | 2.531566 |
| p_leaf_joint_R_l2_cm | 5.176007 |
| base_p_leaf_joint_R_l2_cm | 5.187598 |
| gR1_angle_deg | 4.200364 |
| base_gR1_angle_deg | 4.203858 |
| control_p_leaf_joint_R_l2_cm | 5.169640 |
| control_gR1_angle_deg | 4.208212 |

## Per Leaf L2 cm

| leaf | pred | base |
|---|---:|---:|
| left_forearm | 6.408949 | 6.475623 |
| right_forearm | 5.018446 | 5.064811 |
| left_lower_leg | 4.684297 | 4.707471 |
| right_lower_leg | 4.187550 | 4.175006 |
| head | 5.580790 | 5.515078 |
