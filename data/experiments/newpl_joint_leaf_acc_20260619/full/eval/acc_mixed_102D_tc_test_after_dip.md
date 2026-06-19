# Joint-Leaf NewPL Module Eval

- Cache: `data/experiments/newpl_joint_leaf_acc_20260619/full/caches/acc_mixed_102D/tc_test/pl_curve_cache_manifest.json`
- Checkpoint: `data/experiments/newpl_joint_leaf_acc_20260619/full/acc_mixed_102D/dip_finetune/best_loss.pt`
- Feature mode: `acc_mixed_102D`
- Scope: module-level decoded joint-leaf PL only; no IK/full-pipeline.

| metric | value |
|---|---:|
| p_leaf_joint_R_l1_cm | 2.531727 |
| p_leaf_joint_R_l2_cm | 5.176165 |
| base_p_leaf_joint_R_l2_cm | 5.187598 |
| gR1_angle_deg | 4.200376 |
| base_gR1_angle_deg | 4.203858 |
| control_p_leaf_joint_R_l2_cm | 5.169837 |
| control_gR1_angle_deg | 4.208225 |

## Per Leaf L2 cm

| leaf | pred | base |
|---|---:|---:|
| left_forearm | 6.410118 | 6.475623 |
| right_forearm | 5.016294 | 5.064811 |
| left_lower_leg | 4.683005 | 4.707471 |
| right_lower_leg | 4.185216 | 4.175006 |
| head | 5.586192 | 5.515078 |
