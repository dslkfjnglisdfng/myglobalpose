# AccCurve v1 TotalCapture Eval 20260618

## Purpose

Evaluate AccCurve v1 on TotalCapture test before using v1 acceleration to retrain NewPL. This is acceleration-level evaluation only: no AccCurve training, no PL training, no IK/VR/full-pipeline evaluation, and no S4 metrics.

## Contract

- Experiment root: `data/experiments/acc_curve_v1_totalcapture_eval_20260618`
- Cache root: `code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test`
- Checkpoint: `data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt`
- Cache manifest: `code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test/acc_curve_cache_manifest.json`
- Target: `smooth(diff_acc(p_WS))`, where `p_WS = p_WJ + R_WJ @ rJS`
- Target key: `aFK_smooth[18]`
- Frame: model/world frame M for input, base, prediction, and target
- This is v1 target namespace, not strict GTFK v2.

## TotalCapture Results

| Dataset | Target | Acc source | L2 error | RMSE | pred/base ratio | corr | valid frames |
|---|---|---|---:|---:|---:|---:|---:|
| TotalCapture test | smooth(diff_acc(p_WS)) | aM_smooth | 0.873843 | 0.693060 | 1.000000 | 0.974734 | 16084 |
| TotalCapture test | smooth(diff_acc(p_WS)) | AccCurve v1 pred | 2.091960 | 1.539445 | 2.393977 | 0.866428 | 16084 |

## DIP v1 Historical Reference

| Dataset | Target | pred L2 | base L2 | pred/base ratio | pred RMSE | base RMSE | corr |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP test | smooth(diff_acc(p_WS)) | 1.202067 | 2.368697 | 0.622049 | 0.930242 | 1.733464 | 0.940837 |
| TotalCapture test | smooth(diff_acc(p_WS)) | 2.091960 | 0.873843 | 2.393977 | 1.539445 | 0.693060 | 0.866428 |

## Conclusion

- TotalCapture pred/base ratio: `2.393977`.
- DIP historical pred/base ratio: `0.622049`.
- Ratio gap TC-DIP: `1.771928`.
- AccCurve v1 is not better than the aM_smooth baseline on TotalCapture v1 acceleration targets.
- The TotalCapture ratio is not suitable for direct cross-dataset acceleration replacement.

Recommendation: do not use v1 acceleration as a cross-dataset NewPL retrain input without revising the acceleration module.
