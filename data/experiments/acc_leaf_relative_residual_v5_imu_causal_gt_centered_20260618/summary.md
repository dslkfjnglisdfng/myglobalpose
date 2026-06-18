# Acc Leaf-Relative Residual v5 IMU Causal GT Centered 20260618

Experiment: `acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618`

## Contract

- root index 5
- leaf indices 0..4
- root used only as reference acceleration
- root excluded from residual/loss/metric
- GT FK uses tran=0
- frame = model/world frame M
- no sensor-local rotation
- IMU smoothing: causal Butterworth order=2 cutoff=4Hz; realtime / zero-lookahead
- GT smoothing: centered moving average window=9; non-realtime; target-only
- GT centered smoothing is allowed because GT is used only during training/eval; runtime does not use GT

## Primary DIP / TotalCapture Table

| Dataset | Split | Formulation | L2 | RMSE | MAE | Corr | Cosine | Mag MAE | Valid frames |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| DIP | all | raw_leaf_relative | 2.360895 | 3.530190 | 1.169078 | 0.610050 | 0.650471 | 1.313753 | 317450 |
| DIP | all | imu_butter2_4hz_vs_gt_centered_ma9 | 1.828599 | 2.060184 | 0.901031 | 0.687890 | 0.506284 | 0.817715 | 317450 |
| TotalCapture | all | raw_leaf_relative | 2.905783 | 4.418546 | 1.438626 | 0.563076 | 0.580539 | 1.685336 | 176159 |
| TotalCapture | all | imu_butter2_4hz_vs_gt_centered_ma9 | 2.480798 | 2.426839 | 1.220637 | 0.656914 | 0.517376 | 1.285422 | 176159 |

## Reference Comparison

- DIP v4 symmetric butter L2/RMSE/corr = 0.893481 / 0.914904 / 0.943719
- TotalCapture v4 symmetric butter L2/RMSE/corr = 1.092481 / 1.050146 / 0.941474
- DIP v3 centered_ma9 L2/RMSE/corr = 0.796990 / 0.791042 / 0.948895
- TotalCapture v3 centered_ma9 L2/RMSE/corr = 1.011330 / 0.897591 / 0.945537
- asymmetric improves over raw on DIP and TC: `True`
- DIP asym/v4 RMSE ratio: `2.251804`; asym/v3 oracle RMSE ratio: `2.604393`
- TotalCapture asym/v4 RMSE ratio: `2.310953`; asym/v3 oracle RMSE ratio: `2.703725`
- DIP asymmetric is worse than v4 symmetric butter by RMSE.
- TotalCapture asymmetric is worse than v4 symmetric butter by RMSE.

## Decision Rule

Decision: `fail`.

- Pass if DIP and TC improve over raw in L2/RMSE/corr, corr remains >=0.93, RMSE is not worse than v4 symmetric butter by more than 15%, and RMSE is not worse than v3 centered oracle by more than 30%.
- Soft pass if DIP and TC improve strongly over raw, corr remains >=0.90 on both, but RMSE is 15-35% worse than v4 symmetric butter.
- Fail if either DIP or TC does not improve over raw, corr drops below 0.90, or TC degrades sharply relative to v4.

## Interpretation Notes

- GT centered smoothing may reduce target noise.
- IMU causal smoothing has phase lag, while GT centered smoothing is near zero-phase.
- Therefore residual can become slightly worse than v4 symmetric butter even if the target is cleaner.
- This experiment tests training-target cleanliness, not runtime GT availability.

## Secondary Synthetic Sanity

| Dataset | Formulation | L2 | RMSE | Corr | Valid frames |
|---|---|---:|---:|---:|---:|
| AMASS | raw_leaf_relative | 0.849336 | 0.893008 | 0.980479 | 1115416 |
| AMASS | v4_symmetric_butter_reference | 0.288948 | 0.329019 | 0.996262 | 1115416 |
| AMASS | imu_butter2_4hz_vs_gt_centered_ma9 | 1.908422 | 1.962954 | 0.724675 | 1115416 |
| AMASS | centered_ma9_oracle | 0.258037 | 0.291373 | 0.996474 | 1115416 |
| ALL | raw_leaf_relative | 1.372700 | 2.269143 | 0.857988 | 1609025 |
| ALL | v4_symmetric_butter_reference | 0.496191 | 0.600771 | 0.981588 | 1609025 |
| ALL | imu_butter2_4hz_vs_gt_centered_ma9 | 1.955338 | 2.037966 | 0.710342 | 1609025 |
| ALL | centered_ma9_oracle | 0.447000 | 0.521426 | 0.982637 | 1609025 |

## Per-Sensor Compact Table

| Sensor | Raw L2 | Raw RMSE | Raw Corr | Asym L2 | Asym RMSE | Asym Corr |
|---|---:|---:|---:|---:|---:|---:|
| head | 1.149957 | 1.738649 | 0.818465 | 1.153871 | 1.013474 | 0.657195 |
| left_forearm | 1.392992 | 2.688526 | 0.863942 | 2.416456 | 2.464005 | 0.734064 |
| left_lower_leg | 1.436470 | 2.330646 | 0.798750 | 1.826643 | 1.751456 | 0.666943 |
| right_forearm | 1.401320 | 2.203322 | 0.908200 | 2.508109 | 2.705204 | 0.717873 |
| right_lower_leg | 1.482761 | 2.281984 | 0.810232 | 1.871611 | 1.811729 | 0.671815 |

## Non-Claims

- This is not AccCurve training.
- This is not PL/NewPL training.
- This is not full-pipeline evaluation.
- This does not claim downstream pose improvement.
- AMASS is synthetic and not primary evidence.
- GT centered smoothing is not available at runtime; it is target-only.

## Artifacts

- `cache_manifest.json`
- `metrics.json`
- `per_sequence.csv`
- `debug.json`
- `summary.md`
