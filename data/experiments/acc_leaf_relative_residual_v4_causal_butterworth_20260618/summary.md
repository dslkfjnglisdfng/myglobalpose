# Acc Leaf-Relative Residual v4 Causal Butterworth 20260618

Experiment: `acc_leaf_relative_residual_v4_causal_butterworth_20260618`

## Contract

- root index 5
- leaf indices 0..4
- root used only as reference acceleration
- root excluded from residual/loss/metric
- GT FK uses tran=0
- frame = model/world frame M
- no sensor-local rotation
- realtime smoother = causal Butterworth order=2 cutoff=4Hz
- zero-lookahead

## Primary DIP / TotalCapture Table

| Dataset | Split | Formulation | L2 | RMSE | MAE | Corr | Cosine | Mag MAE | Valid frames |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| DIP | all | raw_leaf_relative | 2.360895 | 3.530190 | 1.169078 | 0.610050 | 0.650471 | 1.313753 | 317450 |
| DIP | all | butter2_4hz_leaf_relative | 0.893481 | 0.914904 | 0.443269 | 0.943719 | 0.784781 | 0.442306 | 317450 |
| TotalCapture | all | raw_leaf_relative | 2.905783 | 4.418546 | 1.438626 | 0.563076 | 0.580539 | 1.685336 | 176159 |
| TotalCapture | all | butter2_4hz_leaf_relative | 1.092481 | 1.050146 | 0.533257 | 0.941474 | 0.734535 | 0.613797 | 176159 |

## Compare Against v3 Centered Moving Average Reference

- v3 DIP centered_ma9 L2/RMSE/corr = 0.796990 / 0.791042 / 0.948895
- v3 TotalCapture centered_ma9 L2/RMSE/corr = 1.011330 / 0.897591 / 0.945537
- butter2_4hz improves over raw on DIP and TC: `True`
- DIP butter/oracle RMSE ratio: `1.156581`
- TotalCapture butter/oracle RMSE ratio: `1.169961`

## Decision Rule

Decision: `pass`.

- Pass if DIP and TC improve over raw in L2/RMSE/corr, corr remains >=0.94 on both, and RMSE is no more than 20% worse than centered_ma9 oracle.
- Soft pass if it strongly improves over raw, corr remains >=0.93 on both, but RMSE is 20-35% worse than centered oracle.
- Fail if either real dataset corr drops below 0.90, Butterworth does not improve over raw, or TC degrades sharply.

## Secondary Synthetic Sanity

| Dataset | Formulation | L2 | RMSE | Corr | Valid frames |
|---|---|---:|---:|---:|---:|
| AMASS | raw_leaf_relative | 0.849336 | 0.893008 | 0.980479 | 1115416 |
| AMASS | butter2_4hz_leaf_relative | 0.288948 | 0.329019 | 0.996262 | 1115416 |
| AMASS | centered_ma9_oracle | 0.258037 | 0.291373 | 0.996474 | 1115416 |
| ALL | raw_leaf_relative | 1.372700 | 2.269143 | 0.857988 | 1609025 |
| ALL | butter2_4hz_leaf_relative | 0.496191 | 0.600771 | 0.981588 | 1609025 |
| ALL | centered_ma9_oracle | 0.447000 | 0.521426 | 0.982637 | 1609025 |

## Per-Sensor Compact Table

| Sensor | Raw L2 | Raw RMSE | Raw Corr | Butter L2 | Butter RMSE | Butter Corr |
|---|---:|---:|---:|---:|---:|---:|
| head | 1.149957 | 1.738649 | 0.818465 | 0.380233 | 0.436147 | 0.942804 |
| left_forearm | 1.392992 | 2.688526 | 0.863942 | 0.536701 | 0.680357 | 0.981920 |
| left_lower_leg | 1.436470 | 2.330646 | 0.798750 | 0.494317 | 0.570925 | 0.968591 |
| right_forearm | 1.401320 | 2.203322 | 0.908200 | 0.554534 | 0.647372 | 0.985808 |
| right_lower_leg | 1.482761 | 2.281984 | 0.810232 | 0.515169 | 0.637552 | 0.963731 |

## Non-Claims

- This is not AccCurve training.
- This is not PL/NewPL training.
- This is not full-pipeline evaluation.
- This does not claim downstream pose improvement.
- AMASS is synthetic and not primary evidence.

## Artifacts

- `cache_manifest.json`
- `metrics.json`
- `per_sequence.csv`
- `debug.json`
- `summary.md`
