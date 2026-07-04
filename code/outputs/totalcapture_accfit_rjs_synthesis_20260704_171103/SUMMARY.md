# TotalCapture Accfit rJS Synthesis

## Setup

- Dataset: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/data/dataset_work/TotalCapture_globalpose_official/test.pt`
- Sequences: s5_freestyle1, s5_freestyle3, s5_rom3, s5_walking2
- FPS: 60.0; gravity: [0.0, -9.800000190734863, 0.0]
- Main objective: sensor-frame specific force `a_S = R_WS^T @ (ddot(p_WJ) + ddot(R_WJ) @ r_JS - g_W)`.
- rJS contract: `r_JS is the IMU origin position relative to mapped joint J, expressed in joint-local coordinates. World prediction: p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS. For DIP-IMU, tran is not used as trusted global translation; acceleration objectives are diagnostic/self-supervised only.`
- Smoothing variants: raw FD, SavGol-9/poly3, SavGol-15/poly3; fit and evaluation use the same smoothing per variant.
- Ridge: 0.0001; max_norm projection: 0.5 m; no-bias is main, with-bias is diagnostic only.
- Old rJS comparison source: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/data/experiments/footlock_transpose_rjs_smoothacc_20260609/totalcapture_test_footlock_transpose_rjs.pt`

## Overall Sensor-Specific RMSE

| source | SavGol-9 RMSE | SavGol-15 RMSE |
|---|---:|---:|
| vertex_baseline | 2.742093 | 2.871197 |
| old_footlock_rjs | 3.856079 | 3.689875 |
| accfit_per_sequence | 2.636344 | 2.807562 |
| accfit_global | 2.640858 | 2.811195 |
| accfit_loo | 2.642856 | 2.813506 |

## Required Answers

1. Accfit rJS vs old footlock rJS: better overall. Per-sequence accfit SavGol-9/SavGol-15 RMSE = 2.636344/2.807562; old footlock = 3.856079/3.689875.
2. Accfit rJS vs 5-vertex baseline: better under the requested SavGol-9 gate. Vertex SavGol-9 RMSE = 2.742093; per-sequence accfit SavGol-9 RMSE = 2.636344.
3. Per-sequence vs global: per-sequence is better on SavGol-9. Per-sequence/global/LOO RMSE = 2.636344/2.640858/2.642856.
4. Lower-leg: left lower leg old -> accfit SavGol-9 RMSE 3.490567 -> 3.380552; right lower leg 4.169763 -> 3.825016.
5. right_lower_leg large error decreased versus old rJS under SavGol-9.
6. Forearm: left forearm old -> accfit 4.683967 -> 1.622808; right forearm 4.486643 -> 2.216878. If recovered, the old degradation was likely from footlock/pseudo offset not optimizing acceleration; if not, fixed joint-local offset is still insufficient for forearm soft-tissue/mount behavior.
7. rJS norm: maximum projected norm is 0.249660 m. Values at the 0.5 m cap indicate noise/frame-convention absorption; see `rjs_offsets.json` and `rjs_norm_bar.png`.
8. Condition number: median 1.390331, max 1.848262. Large outliers mean weak lever-arm observability for that sensor/sequence/motion.
9. With-bias diagnostic: mean extra fit-improvement over projected no-bias is 0.036158. A large positive gap points to bias/gravity/frame convention rather than rJS position alone.
10. Supervision target decision: supported for follow-up acceleration explainability. This is a diagnostic only; no network, PL, IK, or VR module was trained or changed.

## Outputs

- `config.json`, `rjs_accfit_per_sequence.pt`, `rjs_accfit_global.pt`, `rjs_accfit_summary.json`, `rjs_offsets.json`
- `summary_overall.csv`, `summary_per_sensor.csv`, `summary_per_sequence.csv`, `fit_summary.csv`, `frame_level_metrics.csv.gz`
- requested PNGs including RMSE bars, rJS norm, fit improvement, condition number, timeseries, scatter, residual histograms, and boxplot.
