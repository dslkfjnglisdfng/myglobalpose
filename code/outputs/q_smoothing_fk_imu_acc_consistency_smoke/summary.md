# q Smoothing FK/IMU Acceleration Consistency

Evaluation only: no pose optimization, no FK-acceleration post-smoothing, no model training, no dataset cache generation.

## Coordinate Contract

- `R_WJ` maps joint-local vectors into world coordinates.
- `r_JS` is the sensor-site offset from mapped joint J to sensor S, expressed in the joint frame.
- `p_WS = p_WJ + R_WJ @ r_JS`.
- `a_pred_world = d2(p_WS)/dt2` uses strict centered finite differences.
- `a_pred_sensor = R_WS_obs^T @ (a_pred_world - gravity_world)`.
- Predicted FK acceleration is not smoothed. Only `aS` targets use `centered_ma21` and `lowpass_5hz`.

## Main Result

| target | best method | RMSE | mean L2 | corr | cosine | residual p95 |
|---|---|---:|---:|---:|---:|---:|
| centered_ma21 | bspline_q_knot21 | 0.704636 | 0.909437 | 0.992816 | 0.992540 | 2.604501 |
| lowpass_5hz | savgol_q_w15_p3 | 0.582770 | 0.760485 | 0.995228 | 0.994953 | 2.209566 |

## B-spline Refinement Gate

- Supported methods: `bspline_q_knot21`.
- Gate: B-spline is best or within 5% of best centered_ma21 RMSE, lower-leg RMSE is not worse than raw, and q deviation is acceptable.
- q deviation acceptable means mean pose delta <= 2 deg and mean translation delta <= 2 cm.

## Outputs

- `summary.json`: machine-readable result summary.
- `config.json`: paths, filters, methods, frame contract, and spline-order metadata.
- `overall_summary.csv`: method-level metrics.
- `per_sequence_summary.csv`: per-sequence metrics.
- `per_sensor_summary.csv`: per-sensor metrics.
- `lower_leg_summary.csv`: lower-leg-only method metrics.
- `q_deviation_summary.csv`: q displacement from raw trajectory.
- `refinement_gate.csv`: B-spline control-point-refinement gate.
- `figures/*.png`: overall RMSE and q-deviation plots.
