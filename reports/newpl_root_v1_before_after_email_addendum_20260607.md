Subject: NewPL-root v1 before/after fine-tune audit addendum - 2026-06-07

[NEWPL ROOT BEFORE/AFTER CHECKPOINT AUDIT]

Additional JSONs:
- data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/amass_best_last_module_metrics.json
- data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/tc_before_after_pl_only_metrics.json
- data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/tc_before_after_root_head_only_metrics.json
- data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/dip_before_after_pl_metrics.json

PL before/after:
| Dataset | Version | pRB L1 cm | pRB L2 cm | gR1 angle deg |
|---|---|---:|---:|---:|
| AMASS-val20 | newpl_root_amass_best | 1.671904 | 3.460373 | 4.869618 |
| AMASS-val20 | newpl_root_amass_last | 1.654693 | 3.415629 | 4.865114 |
| TotalCapture-test | before_tc_amass_best | 3.290194 | 6.825256 | 13.408251 |
| TotalCapture-test | tc_finetune_best | 3.268334 | 6.779195 | 13.376897 |
| TotalCapture-test | tc_finetune_last | 3.267189 | 6.776795 | 13.375096 |
| DIP-IMU-test | before_dip_amass_best | 3.115633 | 6.428928 | 12.852417 |
| DIP-IMU-test | dip_finetune_best | 3.115812 | 6.429121 | 12.854242 |
| DIP-IMU-test | dip_finetune_last | 3.117006 | 6.430589 | 12.864961 |

Root-head before/after:
| Dataset | Version | root_vel L1 | root_vel L2 | root_vel angle deg |
|---|---|---:|---:|---:|
| AMASS-val20 | newpl_root_amass_best | 0.193051 | 0.427344 | 76.915063 |
| AMASS-val20 | newpl_root_amass_last | 0.193132 | 0.427571 | 74.706794 |
| TotalCapture-test | before_tc_amass_best | 0.269022 | 0.587931 | 76.761979 |
| TotalCapture-test | tc_finetune_best | 0.268907 | 0.587701 | 75.600269 |
| TotalCapture-test | tc_finetune_last | 0.268900 | 0.587688 | 75.533853 |
| DIP-IMU-test | all NewPL-root checkpoints | not available | not available | not available |

Conclusion:
- TC fine-tune has a real but small positive effect on NewPL-root pRB/gR1 and a tiny effect on direct root_vel.
- DIP fine-tune does not help; best and last are both slightly worse than AMASS-pretrained.
- AMASS last is better than AMASS best on decoded PL, but root_vel L1/L2 is slightly worse; control-point selection and decoded PL are not perfectly aligned.
- None of this changes the main decision: newpl_root_v1 remains unselected and should not be connected to IK1/full pipeline.
