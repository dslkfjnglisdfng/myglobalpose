# TotalCapture Both-Smooth FK/IMU Acceleration Audit

Evaluation only: no pose/tran refinement, no model training, no dataset cache generation.

## Coordinate Contract

- `R_WJ` maps joint-local vectors into world coordinates.
- `r_JS` is the IMU sensor origin relative to the mapped joint origin, expressed in joint-local coordinates.
- `p_WS = p_WJ + R_WJ @ r_JS`.
- Sensor-frame specific force comparison uses `a_S = R_WS_obs^T @ (ddot(p_WS) - g_W)`.
- World/model-frame comparison uses `ddot(p_WS)` against reconstructed/stored `aM`.

## Best Rows

| scope | source | protocol | space | RMSE | L2 | corr | cosine | p95 |
|---|---|---|---|---:|---:|---:|---:|---:|
| overall best | accfit_global_rjs | centered_ma21 | sensor_specific_force | 0.492988 | 0.612267 | 0.996950 | 0.996965 | 1.759593 |
| accfit raw | accfit_global_rjs | raw | sensor_specific_force | 3.053103 | 2.659551 | 0.912537 | 0.913169 | 8.075048 |
| accfit best smooth | accfit_global_rjs | centered_ma21 | sensor_specific_force | 0.492988 | 0.612267 | 0.996950 | 0.996965 | 1.759593 |
| world/model best | accfit_global_rjs | centered_ma21 | model_world_linear_acc | 0.448956 | 0.580442 | 0.979122 | 0.978972 | 1.615185 |
| lower-leg accfit raw | accfit_global_rjs | raw | sensor_specific_force | 3.613702 | 3.306250 | 0.873858 | 0.875454 | 11.069432 |
| lower-leg accfit best | accfit_global_rjs | centered_ma21 | sensor_specific_force | 0.430999 | 0.600381 | 0.997401 | 0.997448 | 1.323008 |

## Required Answers

1. The old/smooth protocol helps because it applies the same low-pass operation to the noisy IMU signal and to the noisy second-difference FK acceleration. Raw finite differences amplify pose/tran noise; both-smoothing removes high-frequency components neither side can match reliably.
2. The improvement is primarily from smoothing/filtering, not only from frame choice. Both comparison spaces are reported; best sensor-frame row is `centered_ma21` RMSE `0.492988`, while best world/model row is `centered_ma21` RMSE `0.448956`.
3. Under SavGol-9 sensor specific force, accfit global rJS vs old rJS delta RMSE is `-1.939099` and vs vertex delta RMSE is `-0.204214`. Under lowpass-5Hz, the deltas are `-1.797249` and `-0.205957`.
4. Lower-leg accfit global rJS raw RMSE `3.613702` changes to best smooth `centered_ma21` RMSE `0.430999`.
5. Recommended measurement for the next Kalman-style smoother is `centered_ma21` sensor-frame specific force with accfit global rJS, because it is the lowest-RMSE matched-processing protocol for the fixed-rJS source.
6. If raw remains worse than matched smoothing, downstream refinement should pursue smooth/low-frequency acceleration only. This audit should be treated as the gate against optimizing raw acceleration.

## Outputs

- `smooth_protocol_summary.csv`: overall source/protocol/frame metrics.
- `per_sensor_summary.csv`: per-sensor metrics.
- `per_sequence_summary.csv`: per-sequence metrics.
- `lower_leg_summary.csv`: lower-leg-only aggregate metrics.
- `protocol_improvement_vs_raw.csv`: smoothing gains relative to raw for each source/frame.
- `rjs_source_comparison.csv`: accfit global vs old rJS vs vertex deltas.
- `plots/*.png`: protocol, lower-leg, and example time-series plots.
