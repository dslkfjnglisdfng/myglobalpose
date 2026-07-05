# TotalCapture FK/IMU Smooth Protocol and Smoother Pipeline

## Scope

This is a diagnostic pipeline only. It does not train GPNet, PL, IK, VR, or any full-pipeline replacement.

## Step 1 - Both-Smooth FK/IMU Audit

Full TotalCapture official test, 4 sequences.

Best matched-processing result:

| source | protocol | frame | RMSE | L2 | corr | cosine | p95 |
|---|---|---|---:|---:|---:|---:|---:|
| accfit_global_rjs | centered_ma21 | sensor_specific_force | 0.492988 | 0.612267 | 0.996950 | 0.996965 | 1.759593 |

Raw accfit-global rJS sensor-specific-force RMSE was 3.053103. Lower-leg-only RMSE improved from raw 3.613702 to centered_ma21 0.430999.

Conclusion: old smooth protocols help mainly because both IMU and FK acceleration are low-passed before comparison. Frame choice matters less than matched smoothing; both sensor-frame and model/world-frame best rows are centered_ma21.

Recommended measurement for later smoothing: centered_ma21 sensor-frame specific force with accfit global rJS.

## Step 2 - Simplified Kalman-Style Smoother

Bounded run: 4 sequences, 180 frames each, 20 iterations, centered_ma21 measurement.

| group | before RMSE | after RMSE | delta |
|---|---:|---:|---:|
| all | 0.339067 | 0.339417 | +0.000350 |
| heldout_lower_leg | 0.281647 | 0.282017 | +0.000370 |

Pose/tran changes were small, but gyro, jerk, and foot sliding also increased slightly. This does not pass the conservative refinement gate.

## Step 3 - rJS/Pose Alternating Diagnostic

Bounded run: 3 rounds, 4 sequences, 120 frames each, 5 smoother iterations per round.

Each round slightly worsened all-sensor and held-out lower-leg smooth acceleration RMSE. Round 3 all-sensor delta was +0.000150; lower-leg delta was +0.000119.

Conclusion: alternating rJS and pose smoothing is implemented and smoke/bounded validated, but this bounded configuration is not successful.

## Decision

Use centered_ma21 smooth/low-frequency acceleration as the measurement target. Do not optimize raw acceleration. Do not claim pose-refinement success from the current bounded smoother or alternating run.

## Artifacts

- Smooth audit: `code/outputs/totalcapture_both_smooth_fk_imu_acc_full/`
- Kalman-style smoother: `code/outputs/totalcapture_kalman_style_smoother_bounded/`
- Alternating rJS/pose: `code/outputs/totalcapture_rjs_pose_alternating_bounded/`
- This bundle: `code/outputs/totalcapture_fk_imu_smooth_kalman_pipeline_20260705/`
