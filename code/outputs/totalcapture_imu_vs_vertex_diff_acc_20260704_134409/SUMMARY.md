# TotalCapture IMU Acceleration vs Vertex Difference Acceleration

## Setup

- Dataset path: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/data/dataset_work/TotalCapture_globalpose_official/test.pt`
- Split: `test`; sequences: 4
- FPS: 60.0; gravity: [0.0, -9.800000190734863, 0.0]
- Preferred comparison: `sensor_specific_force`
- IMU acceleration contract: TotalCapture raw accelerometer field in sensor frame; preprocessing stores specific force.
- Sensor-like formula: `acc_sensor_like = R_WS^T @ (acc_vertex_world - gravity_world), where R_WS = RIM^T @ RIS.`
- World/model formula: `aM = RIM^T @ RIS @ aS + gravity_world.`
- Vertex position source: SMPL FK with `tran`, using `fk_imu_joints_and_vertices`; this is a full world trajectory, not root-relative positions.

## Vertex IDs

| sensor | body part | vertex id |
|---|---:|---:|
| left_forearm | left_forearm | 1961 |
| right_forearm | right_forearm | 5424 |
| left_lower_leg | left_lower_leg | 1176 |
| right_lower_leg | right_lower_leg | 4662 |
| head | head | 411 |

## Overall Metrics

| method | mean L2 error | RMSE | Pearson corr | cosine | magnitude MAE |
|---|---:|---:|---:|---:|---:|
| raw_fd | 2.8364 | 3.3360 | 0.8960 | 0.8967 | 1.3006 |
| savgol9_p3_fd | 2.2807 | 2.7421 | 0.9269 | 0.9274 | 1.0716 |
| savgol15_p3_fd | 2.3721 | 2.8712 | 0.9195 | 0.9201 | 1.1556 |

## Required Answers

1. Average difference on TotalCapture: best preferred-space method is `savgol9_p3_fd` with mean L2 2.2807 m/s^2 and RMSE 2.7421 m/s^2.
2. Closest method: `savgol9_p3_fd`. See `summary_overall.csv` for raw/SavGol-9/SavGol-15 in both comparison spaces.
3. With `savgol9_p3_fd`, largest sensor error is `right_lower_leg` vertex 4662 RMSE 3.9564; smallest is `head` vertex 411 RMSE 1.1183. Worst sequence is `s5_freestyle3` RMSE 4.6446.
4. Error diagnosis: largest aggregate bias is on `left_lower_leg` with bias (-0.3226, -0.2460, 0.3693); compare this with residual p95 11.5347, magnitude MAE 1.5313, and correlations in the CSVs/plots to separate bias, scale/noise, and axis-specific failure.
5. Conclusion: this diagnostic says vertex finite-difference acceleration is partially supported, but should still be gated by per-sensor calibration checks.
6. If unsupported, most likely causes are coordinate/gravity convention mismatch, non-equivalence between SMPL vertex and real IMU mount, soft-tissue or strap motion, and finite-difference noise. The script explicitly tests both sensor-specific-force and world/model-frame comparisons so a large gap in both spaces is not just a missing gravity-addition issue.

## Outputs

- `summary_overall.csv`
- `summary_per_sensor.csv`
- `summary_per_sequence.csv`
- `frame_level_metrics.csv` locally; GitHub commit stores `frame_level_metrics.csv.gz` to stay under the single-file size limit.
- `config.json`
- `vertex_ids.json`
- `error_bar_rmse.png`, `corr_bar.png`, `timeseries_examples_*.png`, `scatter_*.png`, `residual_hist_*.png`, `boxplot_residuals.png`
