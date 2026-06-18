# AccCurve v3 Error Distribution and rJS Audit

## 1. Main Question

Is AccCurve v3 failing on TotalCapture because IMU acceleration error distributions shift, rJS/offset_r target construction differs, or the learned correction is DIP-specific?

Contract: root index 5 is reference only and excluded from all residual/correction metrics; leaf indices are 0..4; frame is model/world frame M; IMU base and GT target both use causal Butterworth order=2 cutoff=4Hz.

## 2. AccCurve v3 Recap

| Dataset | Pred L2 | Base L2 | Pred/Base L2 | Pred Corr | Base Corr |
|---|---:|---:|---:|---:|---:|
| DIP test | 0.990334 | 1.196030 | 0.828017 | 0.958276 | 0.943321 |
| TotalCapture test | 1.365116 | 1.052403 | 1.297142 | 0.923813 | 0.946864 |

DIP improves, while TotalCapture worsens.

## 3. Error Distribution Comparison

| Pair | Quantity | Mean diff norm | Std ratio | Diag Gaussian FD | MMD RBF | Mean-vector cosine |
|---|---|---:|---:|---:|---:|---:|
| DIP train vs TC test | e_base | 0.809654 | 0.995797 | 2.173161 | 0.061301 | 0.268100 |
| DIP train vs TC test | c_true | 0.809654 | 0.995797 | 2.173161 | 0.061301 | 0.268100 |

Largest shifted TC test sensors by base residual L2: right_forearm L2=1.354584, left_lower_leg L2=1.212403, right_lower_leg L2=1.204943.

## 4. rJS Comparison

| Pair | Mean diff norm | Std ratio | Diag Gaussian FD | MMD RBF | Mean rJS cosine |
|---|---:|---:|---:|---:|---:|
| DIP train vs TC test | 0.136650 | 0.322934 | 0.034094 | 0.387823 | 0.961810 |

Root/pelvis rJS is audited for offset distribution only; it is not part of AccCurve v3 prediction/loss/metric.

## 5. rJS Acceleration Contribution

| Dataset | Split | Offset contribution ratio | Joint contribution ratio | Offset-site corr | Joint-site corr |
|---|---|---:|---:|---:|---:|
| DIP | test | 0.733277 | 1.139943 | 0.011098 | 0.777662 |
| TotalCapture | test | 0.756321 | 0.859996 | 0.514364 | 0.659245 |

## 6. Correction Transfer

| Dataset | Split | c_pred/c_true ratio | Correction error L2 | Correction cosine | Correction corr | Overcorrection | Harmful rate |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP | test | 0.918560 | 0.990334 | 0.483997 | 0.640673 | 0.237173 | 0.442142 |
| TotalCapture | test | 1.060700 | 1.365116 | 0.321156 | 0.227905 | 0.304232 | 0.564954 |

## 7. Final Diagnosis

`likely_model_overfit`

## 8. Next Recommendation

use base only for TC-like distribution; then test residual_scale smaller or a sensor-specific residual gate

## 9. Non-Claims

- This is diagnostic only.
- No PL/NewPL/full-pipeline claim.
- AMASS is synthetic sanity only.
- Root channel is not predicted, supervised, or evaluated as a residual/correction target.
