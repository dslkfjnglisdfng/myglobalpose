# Joint-Leaf NewPL Module Eval

- Cache: `data/experiments/newpl_joint_leaf_acc_20260619/full/caches/acc_mixed_102D/dip_test/pl_curve_cache_manifest.json`
- Checkpoint: `data/experiments/newpl_joint_leaf_acc_20260619/full/acc_mixed_102D/dip_finetune/best_loss.pt`
- Feature mode: `acc_mixed_102D`
- Scope: module-level decoded joint-leaf PL only; no IK/full-pipeline.

| metric | value |
|---|---:|
| p_leaf_joint_R_l1_cm | 2.718843 |
| p_leaf_joint_R_l2_cm | 5.703366 |
| base_p_leaf_joint_R_l2_cm | 5.689492 |
| gR1_angle_deg | 2.343174 |
| base_gR1_angle_deg | 2.341986 |
| control_p_leaf_joint_R_l2_cm | 5.700448 |
| control_gR1_angle_deg | 2.343062 |

## Per Leaf L2 cm

| leaf | pred | base |
|---|---:|---:|
| left_forearm | 8.224453 | 8.191435 |
| right_forearm | 7.963469 | 7.939158 |
| left_lower_leg | 3.589031 | 3.599232 |
| right_lower_leg | 4.443413 | 4.433803 |
| head | 4.296462 | 4.283832 |
