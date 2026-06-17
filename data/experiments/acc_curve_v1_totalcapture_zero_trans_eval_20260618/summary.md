# acc_curve_v1_totalcapture_zero_trans_eval_20260618

## Purpose

Evaluate AccCurve v1 on TotalCapture test before using v1 acceleration to retrain NewPL. This is acceleration-level evaluation only: no AccCurve training, no PL training, no IK/VR/full-pipeline evaluation, and no S4 metrics.

## Contract

- Experiment root: `data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618`
- Cache root: `code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test`
- Checkpoint: `data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt`
- Cache manifest: `code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test/acc_curve_cache_manifest.json`
- Target translation: `forced zero`
- Target: `smooth(diff_acc(p_WS_zero_trans))`
- Target key: `aFK_smooth[18]`
- Frame: model/world frame M for input, base, prediction, and target
- This is v1 target namespace, not strict GTFK v2.
- Force zero tran: `True`

## TotalCapture Results

| Dataset | Target translation | Target | Acc source | L2 error | RMSE | pred/base ratio | corr | valid frames |
|---|---|---|---|---:|---:|---:|---:|---:|
| TotalCapture test | forced zero | smooth(diff_acc(p_WS_zero_trans)) | aM_smooth | 1.832642 | 1.466451 | 1.000000 | 0.883554 | 16084 |
| TotalCapture test | forced zero | smooth(diff_acc(p_WS_zero_trans)) | AccCurve v1 pred | 1.415560 | 0.977232 | 0.772415 | 0.945382 | 16084 |

## Three-Way Comparison

| Eval | Target translation | Target | Base L2 | Pred L2 | Pred/Base | Base RMSE | Pred RMSE | Pred Corr |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DIP historical | zero/none | smooth(diff_acc(p_WS_zero_trans)) | 2.368697 | 1.202067 | 0.622049 | 1.733464 | 0.930242 | 0.940837 |
| TC previous | source tran | smooth(diff_acc(trans + p_WS)) | 0.873843 | 2.091960 | 2.393977 | 0.693060 | 1.539445 | 0.866428 |
| TC zero-trans | forced zero | smooth(diff_acc(p_WS_zero_trans)) | 1.832642 | 1.415560 | 0.772415 | 1.466451 | 0.977232 | 0.945382 |

## Conclusion

- TotalCapture pred/base ratio: `0.772415`.
- DIP historical pred/base ratio: `0.622049`.
- TC zero-trans pred/base ratio is below 1, so v1's DIP-style zero-trans acceleration target can generalize to TC; the previous full-trans failure is mainly consistent with target translation mismatch.
- Continue considering zero-trans v1 acceleration as a NewPL retrain input candidate, but still gate it with same-cache PL module metrics before any full-pipeline claim.
