# Acc Leaf-Relative Residual v3 20260618

Experiment: `acc_leaf_relative_residual_v3_20260618`

## Contract

- root index = 5
- leaf indices = 0..4
- root is used only as reference acceleration
- root is not included in residual/loss/metric
- frame = model/world frame M
- no sensor-local rotation
- GT FK uses tran=0
- diff method = centered second difference, dt=1/60
- smooth method = centered moving average, window=9

## Main Result Table

| Dataset | Formulation | L2 | RMSE | MAE | Corr | Cosine | Mag MAE | Valid frames |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AMASS | raw_leaf_relative | 0.849336 | 0.893008 | 0.419723 | 0.980479 | 0.847992 | 0.471936 | 1115416 |
| AMASS | smooth_leaf_relative | 0.272780 | 0.389472 | 0.132378 | 0.990993 | 0.900879 | 0.158198 | 1115416 |
| DIP | raw_leaf_relative | 2.360895 | 3.530190 | 1.169078 | 0.610050 | 0.650471 | 1.313753 | 317450 |
| DIP | smooth_leaf_relative | 0.796990 | 0.791042 | 0.395198 | 0.948895 | 0.779124 | 0.392613 | 317450 |
| TotalCapture | raw_leaf_relative | 2.905783 | 4.418546 | 1.438626 | 0.563076 | 0.580539 | 1.685336 | 176159 |
| TotalCapture | smooth_leaf_relative | 1.011330 | 0.897591 | 0.492138 | 0.945537 | 0.722444 | 0.579187 | 176159 |
| ALL | raw_leaf_relative | 1.372700 | 2.269143 | 0.679118 | 0.857988 | 0.779741 | 0.770866 | 1609025 |
| ALL | smooth_leaf_relative | 0.457061 | 0.562864 | 0.223618 | 0.979057 | 0.857322 | 0.250537 | 1609025 |

## Per-Sensor Compact Table

| Sensor | Raw L2 | Raw RMSE | Raw Corr | Smooth L2 | Smooth RMSE | Smooth Corr |
|---|---:|---:|---:|---:|---:|---:|
| head | 1.149957 | 1.738649 | 0.818465 | 0.347073 | 0.386625 | 0.943683 |
| left_forearm | 1.392992 | 2.688526 | 0.863942 | 0.496403 | 0.650060 | 0.979776 |
| left_lower_leg | 1.436470 | 2.330646 | 0.798750 | 0.452963 | 0.523478 | 0.966204 |
| right_forearm | 1.401320 | 2.203322 | 0.908200 | 0.514870 | 0.643452 | 0.982492 |
| right_lower_leg | 1.482761 | 2.281984 | 0.810232 | 0.473998 | 0.569178 | 0.962981 |

## Required Judgment

- Smooth L2 delta vs raw: `0.915639`; smooth RMSE delta vs raw: `1.706279`.
- Smooth corr delta vs raw: `0.121069`.
- smoothing is necessary before comparing IMU/FK acceleration residuals.
- root-relative smoothed acceleration is a cleaner explainability target.
- If residual remains large after smoothing, measured IMU acceleration still contains noise / bias / soft-tissue / convention mismatch not explained by zero-trans FK.

## Explicit Non-Claims

- This is not AccCurve training.
- This is not PL/NewPL training.
- This is not full-pipeline evaluation.
- Root channel is not evaluated as residual.
- Do not claim downstream pose improvement.

## Artifacts

- `cache_manifest.json`: leaf-relative cache file list and contract.
- `metrics.json`: aggregate metrics and checks.
- `per_sequence.csv`: per-sequence leaf-only metrics.
- `debug.json`: root-reference/debug checks.
