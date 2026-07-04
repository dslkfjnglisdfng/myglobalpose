# TotalCapture IMU Acceleration vs rJS Difference Acceleration

## Setup

- Dataset path: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/data/dataset_work/TotalCapture_globalpose_official/test.pt`
- rJS path: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/data/experiments/footlock_transpose_rjs_smoothacc_20260609/totalcapture_test_footlock_transpose_rjs.pt`
- rJS path selection: `default_auto_search`
- Vertex baseline dir: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409`
- Split: `test`; sequences: 4
- FPS: 60.0; gravity: [0.0, -9.800000190734863, 0.0]
- Preferred comparison: `sensor_specific_force`
- rJS contract: `r_JS is the IMU origin position relative to mapped joint J, expressed in joint-local coordinates. World prediction: p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS. For DIP-IMU, tran is not used as trusted global translation; acceleration objectives are diagnostic/self-supervised only.`
- Position source: `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS`, using FK joint world positions/rotations with `tran`; no root-relative subtraction is applied.

## Overall Metrics

| method | mean L2 error | RMSE | Pearson corr | cosine | magnitude MAE |
|---|---:|---:|---:|---:|---:|
| raw_fd | 4.1887 | 4.8440 | 0.7872 | 0.7874 | 1.8846 |
| savgol9_p3_fd | 3.4257 | 3.8561 | 0.8532 | 0.8533 | 1.5603 |
| savgol15_p3_fd | 3.3107 | 3.6899 | 0.8644 | 0.8645 | 1.5590 |

## Vertex vs rJS SavGol-9

| sensor | vertex SavGol-9 RMSE | rJS SavGol-9 RMSE | delta vertex-rJS |
|---|---:|---:|---:|
| left_forearm | 1.6518 | 4.6840 | -3.0322 |
| right_forearm | 2.2542 | 4.4866 | -2.2324 |
| left_lower_leg | 3.5891 | 3.4906 | 0.0985 |
| right_lower_leg | 3.9564 | 4.1698 | -0.2133 |
| head | 1.1183 | 1.6451 | -0.5268 |

## Required Answers

1. rJS average difference on TotalCapture: best method `savgol15_p3_fd` has mean L2 3.3107 m/s^2 and RMSE 3.6899 m/s^2.
2. Best rJS method: `savgol15_p3_fd`.
3. rJS is `not better` than the five-vertex baseline under sensor-specific SavGol-9 comparison.
4. Sensors that improved: left_lower_leg 0.0985.
5. Lower-leg change: left_lower_leg: delta 0.0985 m/s^2, right_lower_leg: delta -0.2133 m/s^2.
6. If rJS did not improve, likely causes include footlock/pseudo-constraint rJS not being a GT mount, mapped joint plus fixed offset still missing soft-tissue/strap motion, coordinate convention mismatch in R_WJ/R_WS/r_JS, or time/filtering differences.
7. Conclusion: rJS position acceleration is not clearly supported as a stronger target than the vertex baseline.

Best rJS sensor: `head` RMSE 1.4464; worst rJS sensor: `right_lower_leg` RMSE 4.3324.

## Outputs

- `summary_overall.csv`, `summary_per_sensor.csv`, `summary_per_sequence.csv`, local `frame_level_metrics.csv`; GitHub commit stores `frame_level_metrics.csv.gz` to stay under the single-file size limit.
- `config.json`, `rjs_offsets.json`, `SUMMARY.md`
- `error_bar_rmse.png`, `corr_bar.png`, `timeseries_examples_*.png`, `scatter_*.png`, `residual_hist_*.png`, `boxplot_residuals.png`, `vertex_vs_rjs_rmse_bar.png`
