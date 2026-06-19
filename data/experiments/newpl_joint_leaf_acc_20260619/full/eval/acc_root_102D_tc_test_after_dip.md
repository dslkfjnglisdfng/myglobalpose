# Joint-Leaf NewPL Module Eval

- Cache: `data/experiments/newpl_joint_leaf_acc_20260619/full/caches/acc_root_102D/tc_test/pl_curve_cache_manifest.json`
- Checkpoint: `data/experiments/newpl_joint_leaf_acc_20260619/full/acc_root_102D/dip_finetune/best_loss.pt`
- Feature mode: `acc_root_102D`
- Scope: module-level decoded joint-leaf PL only; no IK/full-pipeline.

| metric | value |
|---|---:|
| p_leaf_joint_R_l1_cm | 2.531604 |
| p_leaf_joint_R_l2_cm | 5.176143 |
| base_p_leaf_joint_R_l2_cm | 5.187598 |
| gR1_angle_deg | 4.200393 |
| base_gR1_angle_deg | 4.203858 |
| control_p_leaf_joint_R_l2_cm | 5.169865 |
| control_gR1_angle_deg | 4.208238 |

## Per Leaf L2 cm

| leaf | pred | base |
|---|---:|---:|
| left_forearm | 6.409912 | 6.475623 |
| right_forearm | 5.019225 | 5.064811 |
| left_lower_leg | 4.684097 | 4.707471 |
| right_lower_leg | 4.187483 | 4.175006 |
| head | 5.579999 | 5.515078 |
