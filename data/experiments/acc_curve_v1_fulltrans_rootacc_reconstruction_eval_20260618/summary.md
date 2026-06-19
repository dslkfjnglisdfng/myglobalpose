# acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618

## Main Question

Was the old TotalCapture full-trans failure caused by comparing the zero-trans AccCurve v1 prediction against full-trans ground truth?

## Historical Recap

- DIP trained target was effectively zero-trans: `smooth(diff_acc(p_WJ + R_WJ @ rJS))`.
- Old TC full-trans eval directly compared `pred_zero` against `GT_full` and got pred/base ratio `2.393977`.
- TC zero-trans eval got pred/base ratio `0.772415`.

## Definitions

- `pred_zero`: AccCurve v1 prediction from the zero-trans v1 cache.
- `a_root_trans_smooth`: cache-consistent `GT_full - GT_zero`, checked against `smooth(diff_acc(tran_gt))`.
- `pred_full_reconstructed = pred_zero + a_root_trans_smooth`.
- `GT_full`: full-trans `aFK_smooth` from the TotalCapture v1 cache.
- `aM_smooth baseline`: AccCurve v1 spline-decoded base stream from `aM_smooth`, matching the historical v1 evaluator.

## Decomposition Sanity

| Check | max abs | mean abs | RMSE |
|---|---:|---:|---:|
| `GT_full - (GT_zero + cache_root_acc)` | 0.000366449 | 0.000013115 | 0.000026308 |
| `cache_root_acc - smooth(diff_acc(tran_gt))` | 0.000267982 | 0.000010596 | 0.000021218 |

## Main Table

All 6 sensors are primary for v1 historical compatibility.

| Row | L2 | RMSE | MAE | Corr | Cosine | Mag MAE | Pred/Base ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_full | 0.873843 | 0.693060 | 0.429852 | 0.974734 | 0.774869 | 0.456501 | 1.000000 |
| wrong_pred_zero_vs_full | 2.091960 | 1.539445 | 1.029687 | 0.866428 | 0.442708 | 1.125138 | 2.393977 |
| correct_pred_zero_plus_gt_root_trans | 1.415560 | 0.977232 | 0.709856 | 0.949590 | 0.659106 | 0.798133 | 1.619925 |
| optional_pred_zero_plus_imu_root_est | 1.204596 | 0.886916 | 0.597049 | 0.958057 | 0.699401 | 0.636703 | 1.378504 |
| zero_trans_sanity | 1.415560 | 0.977232 | 0.709856 | 0.945382 | 0.552023 | 0.786097 | 0.772415 |

## Interpretation

- Corrected prediction beats baseline_full: `False`.
- Target mismatch explains part but not all of the old full-trans failure; the corrected prediction still does not beat the aM_smooth full-trans baseline.

## Current Project Implication

AccCurve v1 should be described as a root-translation-free sensor-site acceleration predictor, not a full absolute acceleration predictor. Future target construction should explicitly define whether root translational acceleration is included. For current leaf-relative work, use root-reference/translation-free formulation deliberately.

## Non-Claims

- No PL/NewPL/full-pipeline claim.
- No S4 metrics.
- No retraining.
- This is a historical AccCurve v1 evaluation correction only.
