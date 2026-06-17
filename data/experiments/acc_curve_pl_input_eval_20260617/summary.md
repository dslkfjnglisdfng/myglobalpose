# EXP-20260617-acc_curve_pl_input_eval

## Purpose

Evaluate whether AccCurve v1/v2 acceleration predictions help the frozen official baseline PL module when used only as the PL acceleration input. No PL weights are trained or modified.

## Protocol

- Experiment root: `data/experiments/acc_curve_pl_input_eval_20260617`
- Evaluator: `scripts/eval_pl_with_acc_curve_input_20260617.py`
- Frozen baseline PL checkpoint: `data/weights.pt` (`GPNet.plnet` weights only)
- DIP test cache/protocol: `data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json`
- AccCurve v1 checkpoint: `data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt`
- AccCurve v2 checkpoint: `data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/dip_finetune/best_loss.pt`
- AccCurve v1 target: `smooth(diff_acc(p_WS))`
- AccCurve v2 target: `smooth(GTFKacc(q,qdot,qddot,rJS))`

Only the first 18D acceleration block of the legacy PL feature is replaced. The other 66D (`wRB[18] + RRB[45] + gR0[3]`) are asserted identical across variants. AccCurve outputs are model/world-frame accelerations and are converted to PL root frame with `aRB = acc_M @ RMB_root` before PL forward.

DIP test is used only for evaluation. It is not used for training, normalization fitting, or checkpoint selection.

## Validation

- Official raw vectorized 84D feature vs `pl_input_feature`: max abs diff `7.6293945e-06`.
- Non-acceleration 66D feature block max abs diff across variants: `0`.
- AccCurve v1/v2 predictions were asserted as `[T,6,3]` for every evaluated sequence.
- Debug acceleration stats: `data/experiments/acc_curve_pl_input_eval_20260617/debug_first_sequence_acceleration_blocks.json`.

## Results

| Variant | Acc source | Target used by AccCurve | PL pRB L2 cm | PL pRB RMSE cm | PL gR1 deg | valid frames |
|---|---|---|---:|---:|---:|---:|
| official_raw_acc | raw aM | none | 6.529110 | 4.638030 | 15.267153 | 57994 |
| smooth_acc | smooth(aM) | none | 6.462386 | 4.589704 | 15.216247 | 57994 |
| acc_curve_v1_pred | AccCurve v1 pred | smooth(diff_acc(p_WS)) | 6.967961 | 4.866400 | 15.036875 | 57994 |
| acc_curve_v2_gtfk_pred | AccCurve v2 pred | smooth(GTFKacc(q,qdot,qddot,rJS)) | 8.347050 | 5.958994 | 15.229429 | 57994 |

## Conclusion

- `smooth_acc`: pRB improves (6.462386 vs 6.529110 cm); gR1 improves (15.216247 vs 15.267153 deg).
- `acc_curve_v1_pred`: pRB does not improve (6.967961 vs 6.529110 cm); gR1 improves (15.036875 vs 15.267153 deg).
- `acc_curve_v2_gtfk_pred`: pRB does not improve (8.347050 vs 6.529110 cm); gR1 improves (15.229429 vs 15.267153 deg).

Acceleration-level improvement did not transfer into a simultaneous PL pRB and gR1 improvement against official raw acceleration.

This is a standalone PL module-input evaluation. It does not claim full-pipeline motion quality improvement and does not mix v1/v2 acceleration-level RMSE target namespaces.
