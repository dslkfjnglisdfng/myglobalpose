# Joint-Leaf NewPL Module Eval

- Cache: `data/experiments/newpl_joint_leaf_acc_20260619/full/caches/baseline_jointtarget_84D/dip_test/pl_curve_cache_manifest.json`
- Checkpoint: `data/experiments/newpl_joint_leaf_acc_20260619/full/baseline_jointtarget_84D/amass_pretrain/best_loss.pt`
- Feature mode: `baseline_jointtarget_84D`
- Scope: module-level decoded joint-leaf PL only; no IK/full-pipeline.

| metric | value |
|---|---:|
| p_leaf_joint_R_l1_cm | 2.718547 |
| p_leaf_joint_R_l2_cm | 5.702977 |
| base_p_leaf_joint_R_l2_cm | 5.689492 |
| gR1_angle_deg | 2.343299 |
| base_gR1_angle_deg | 2.341986 |
| control_p_leaf_joint_R_l2_cm | 5.700045 |
| control_gR1_angle_deg | 2.343165 |

## Per Leaf L2 cm

| leaf | pred | base |
|---|---:|---:|
| left_forearm | 8.225951 | 8.191435 |
| right_forearm | 7.961069 | 7.939158 |
| left_lower_leg | 3.587859 | 3.599232 |
| right_lower_leg | 4.443987 | 4.433803 |
| head | 4.296020 | 4.283832 |
