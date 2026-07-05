# TotalCapture Kalman-Style Smoother

Offline window optimization only; no EKF, no raw-acc objective, no network changes.

- Measurement: `centered_ma21` sensor-frame specific force.
- rJS: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/code/outputs/totalcapture_accfit_rjs_synthesis_20260704_171103/rjs_accfit_global.pt` field `r_JS_projected` method `savgol9_p3_fd`.
- Verdict: `diagnostic`.

| group | before acc RMSE | after acc RMSE | delta acc RMSE |
|---|---:|---:|---:|
| all | 0.339067 | 0.339417 | 0.000350 |
| heldout_lower_leg | 0.281647 | 0.282017 | 0.000370 |

Diagnostics are in `kalman_style_refinement_summary.csv`; refined trajectories are in `refined_sequences.pt`.
