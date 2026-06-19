# Joint-Leaf NewPL Module Eval

- Cache: `data/experiments/newpl_joint_leaf_acc_20260619/full/caches/acc_mixed_102D/dip_test/pl_curve_cache_manifest.json`
- Checkpoint: `data/experiments/newpl_joint_leaf_acc_20260619/full/acc_mixed_102D/amass_pretrain/best_loss.pt`
- Feature mode: `acc_mixed_102D`
- Scope: module-level decoded joint-leaf PL only; no IK/full-pipeline.

| metric | value |
|---|---:|
| p_leaf_joint_R_l1_cm | 2.719005 |
| p_leaf_joint_R_l2_cm | 5.703530 |
| base_p_leaf_joint_R_l2_cm | 5.689492 |
| gR1_angle_deg | 2.343286 |
| base_gR1_angle_deg | 2.341986 |
| control_p_leaf_joint_R_l2_cm | 5.700649 |
| control_gR1_angle_deg | 2.343161 |

## Per Leaf L2 cm

| leaf | pred | base |
|---|---:|---:|
| left_forearm | 8.225081 | 8.191435 |
| right_forearm | 7.963438 | 7.939158 |
| left_lower_leg | 3.589120 | 3.599232 |
| right_lower_leg | 4.443355 | 4.433803 |
| head | 4.296655 | 4.283832 |
