# TotalCapture IMU-aware Pose Refinement

Diagnostic only: no network training and no PL/IK/VR changes.

## Coordinate Contract

- `R_WJ` maps joint-local vectors into world coordinates.
- `r_JS` is the IMU sensor origin relative to mapped joint origin, expressed in joint-local coordinates.
- `R_WS = R_WJ @ R_JS` maps sensor-frame vectors into world coordinates.
- `aS_pred = R_WS_obs^T @ (d2(p_WJ + R_WJ @ r_JS)/dt2 - gravity_world)`.

## Default Sensor Split

- Fit sensors: left_forearm, right_forearm, head
- Held-out sensors: left_lower_leg, right_lower_leg

## Overall Metrics

| mode | acc RMSE orig | acc RMSE refined | held-out delta | pose mean deg | trans mean m | gyro delta | jerk delta | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 1.200145 | 1.245425 | 0.030263 | 0.000000 | 0.000617 | 0.000000 | 0.008850 | diagnostic |

Success requires acc residual improvement, no held-out/gyro/smoothness/contact regression, and small pose/tran deviation.

## Config

- dataset: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/data/dataset_work/TotalCapture_globalpose_official/test.pt`
- rjs: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/code/outputs/totalcapture_accfit_rjs_synthesis_20260704_171103/rjs_accfit_global.pt`
- rjs method/field: `savgol9_p3_fd` / `r_JS_projected`
- robust loss: `huber`
