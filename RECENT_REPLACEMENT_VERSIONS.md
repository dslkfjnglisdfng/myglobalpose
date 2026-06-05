# Recent Replacement Versions

This document tracks the current GlobalPose module-replacement series. Every replacement attempt should get an explicit version name and record the replaced module, design idea, measured effect, comparison, conclusion, and artifacts. Use `PROJECT_STATUS.md` as the first-read summary and `EXPERIMENT_LOG.md` for detailed commands, metrics, and interpretation.

## Current validated baseline/context

- Official GlobalPose with official S4 input score: `42.522402`.
- Official GlobalPose with processed orientation input score: `38.753660`.
- Processed input is orientation-only / RMB-only correction: `l4_aM == aM`, `l4_wM == wM`, `l4_RMB != RMB`.
- The current best selected replacement checkpoint is NewPL init36 RunD-style: `38.625657482802865` on TotalCapture S4.
- The verified IK replacement artifacts found so far are NewIK1 / official-shape IK1 artifacts, not confirmed NewIK2 artifacts. Do not relabel them as NewIK2 unless the exact NewIK2 paths are identified.

## Version index

| Version | Replaced module | Design idea | Best measured effect | Comparison | Conclusion | Detail |
|---|---|---|---:|---|---|---|
| `newpl_init36_v1` / `newpl_init36_第一版` | PL-s1 (`plnet`) | PLCurve with K2-like 36D stream init `offset_r[18] + pRL[15] + gR0[3]`, trained with RunD-style losses. | S4 score `38.625657482802865` | Better than Run D PL-only `38.69484578047692` and Original GPNet + processed `38.753660`. | Select as current PL1 upstream checkpoint. | `EXPERIMENT_LOG.md:297-354` |
| `newpl_v1` / `newpl_第一版` | PL-s1 (`plnet`) | PLCurve-style streaming PL replacement with processed orientation input and GT spline-control supervision. | S4 score `38.694846` | Better than Original GPNet + processed `38.753660`, gR dynamics continuation `38.730901`, and Run A `38.714056`. | Historical baseline superseded by `newpl_init36_v1`. | `EXPERIMENT_LOG.md:107-136` |
| `newik1_v1` / `newik1_第一版` | IK-s1 (`iknet.net1`) | NewIK1 control-point module trained on PL1 streaming TotalCapture outputs. | Local val loss `0.17848628610372544` | Diagnostic only; no final S4 claim from this run alone. | Converged locally but required full-pipeline validation. | `EXPERIMENT_LOG.md:138-174` |
| `newik1_v2` / `newik1_第二版` | IK-s1 (`iknet.net1`) | Continue `newik1_v1` with bone-length consistency (`bone_length=0.5`). | Local val loss `0.1783127911388874` | Slightly better than `newik1_v1` local loss `0.17848628610372544`. | Prefer locally over `newik1_v1`, but local loss does not prove final S4 improvement. | `EXPERIMENT_LOG.md:176-214` |
| `newik1_v3` / `newik1_第三版` | IK-s1 (`iknet.net1`) | Continue `newik1_v2` with stronger pRJ/control supervision (`pRJ=2.0`, `control_pRJ=0.3`). | Local val loss `0.17918791025876998` | Worse than `newik1_v2` local loss `0.1783127911388874`. | Do not select based on local validation loss. | `EXPERIMENT_LOG.md:216-254` |
| `newik1_official_input_v1` / `newik1官方输入_第一版` | IK-s1 (`iknet.net1`) | Full pipeline with Run D PL checkpoint and a finetuned official-shape IK1 replacement. | S4 score `38.70523069866002` | Better than Original GPNet + processed `38.753660`, worse than Run D PL-only `38.694846`. | Do not select over PL-only Run D as current mainline. | `EXPERIMENT_LOG.md:256-295` |

## `newpl_init36_v1` / `newpl_init36_第一版`

Replaced module: official PL-s1 (`plnet`).

Design idea: Keep the official PL frame input and output contracts unchanged, but initialize the PLCurve streaming hidden state with a K2-like 36D feature: `offset_r[18] + pRL[15] + gR0[3]`.

Main change: Train `PLCurveModule(init_size=36)` on `pl_curve_cache_v2` records that store `pl_init_feature`, initialize from the historical Run D checkpoint with partial weight loading, and use the RunD-style processed-input loss recipe.

Measured effect: TotalCapture S4 score `38.625657482802865` for both `best_loss.pt` and `last.pt` streaming evaluation.

Comparison:

- Original GPNet + processed input: `38.753660`.
- Historical NewPL Run D PL-only: `38.69484578047692`.
- NewPL init36 RunD-style: `38.625657482802865`.

Conclusion: K2-like 36D initialization improves the current processed-input PL replacement. Select this version as the current PL1 upstream checkpoint for future downstream replacement experiments.

Artifacts:

```text
data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
data/experiments/pl_curve_init36_processed_rund_style/last.pt
data/experiments/pl_curve_init36_processed_rund_style/train_result.json
data/experiments/pl_curve_init36_processed_rund_style/eval_best_final_streaming_processed.json
data/experiments/pl_curve_init36_processed_rund_style/eval_last_final_streaming_processed.json
data/dataset_work/L4Cache/pl_curve_init36_processed_tc_train_Roffset_A/pl_curve_cache_manifest.json
data/dataset_work/L4Cache/pl_curve_init36_processed_tc_val_Roffset_A/pl_curve_cache_manifest.json
```

Detailed record: `EXPERIMENT_LOG.md:297-354`.

## `newpl_v1` / `newpl_第一版`

Replaced module: official PL-s1 (`plnet`).

Design idea: Replace the official PL stage with a streaming-compatible PLCurve module while preserving the official downstream PL contract: `pRB[15] + gR1[3] = 18D`. Use processed orientation input and add GT spline-control supervision so the control points, not only decoded current-frame outputs, are closer to the target curve.

Main change: Continue from earlier processed-input PLCurve runs with `gt_control_pRB=0.3`, `gt_control_gR1=0.1`, `lr=2e-6`, and 10 TotalCapture fine-tune epochs.

Measured effect: TotalCapture S4 score `38.694846`.

Comparison:

- Original GPNet + processed input: `38.753660`.
- Earlier Run A: `38.714056`.
- gR dynamics continuation: `38.730901`.
- Run D: `38.694846`.

Conclusion: GT control-point supervision is useful for processed-input PL replacement. Run D is the recommended PL1 upstream checkpoint for downstream replacement experiments.

Artifacts:

```text
data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/best_loss.pt
data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/train_result.json
```

Detailed record: `EXPERIMENT_LOG.md:107-136`.

## `newik1_v1` / `newik1_第一版`

Replaced module: official IK-s1 (`iknet.net1`).

Design idea: Train `NewIK1ControlPointModule` on PL1 streaming outputs to reduce teacher-forcing mismatch while preserving the official IK1 output contract: `pRJ[69] + gR2[3] = 72D`.

Main change: Fine-tune from the AMASS-adapted checkpoint on PL1 streaming TotalCapture train/val cache for 10 epochs.

Measured effect: Best local validation loss `0.17848628610372544` at epoch 10.

Comparison: This is a bounded diagnostic only; it does not establish final S4 improvement.

Conclusion: The run converged locally, but final full-pipeline S4 evaluation is required before selecting the module.

Artifacts:

```text
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/train_result.json
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/best_loss.pt
```

Detailed record: `EXPERIMENT_LOG.md:138-174`.

## `newik1_v2` / `newik1_第二版`

Replaced module: official IK-s1 (`iknet.net1`).

Design idea: Add bone-length consistency to stabilize pRJ geometry and potentially improve gravity/root-direction behavior.

Main change: Continue from `newik1_v1` with `bone_length=0.5`.

Measured effect: Best local validation loss `0.1783127911388874` at epoch 10.

Comparison: Slight local improvement over `newik1_v1` local best `0.17848628610372544`.

Conclusion: Prefer this variant locally, but full downstream S4 validation is still required because decoded-state closeness may not track final S4.

Artifacts:

```text
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/train_result.json
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/best_loss.pt
```

Detailed record: `EXPERIMENT_LOG.md:176-214`.

## `newik1_v3` / `newik1_第三版`

Replaced module: official IK-s1 (`iknet.net1`).

Design idea: Test whether stronger Cartesian pRJ and control-tail supervision improves IK1 state quality beyond the bone-length continuation.

Main change: Continue from `newik1_v2` with `pRJ=2.0`, `control_pRJ=0.3`, and `bone_length=0.5`.

Measured effect: Best local validation loss `0.17918791025876998` at epoch 10.

Comparison: Worse than `newik1_v2` local best `0.1783127911388874`.

Conclusion: Stronger pRJ/control supervision hurt local validation loss; do not select this variant based on local metrics.

Artifacts:

```text
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/train_result.json
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/best_loss.pt
```

Detailed record: `EXPERIMENT_LOG.md:216-254`.

## `newik1_official_input_v1` / `newik1官方输入_第一版`

Replaced module: official IK-s1 (`iknet.net1`) in a full official-shape pipeline.

Design idea: Evaluate a finetuned official-shape IK1 replacement under the full streaming `GPNet.forward_frame` contract, using Run D PL as upstream and keeping downstream IK2/VR/physics contracts unchanged.

Main change: Full evaluation with `pl_backend=curve_v1`, `ik1_backend=official_input_v1`, and `imu_input_mode=processed`.

Measured effect: TotalCapture S4 score `38.70523069866002`, `status=ok`, `all_finite=True`.

Comparison:

- Original GPNet + processed: `38.753660`.
- Run D PL-only: `38.694846`.
- NewIK1 official-input full pipeline: `38.70523069866002`.

Conclusion: This is better than Original GPNet + processed, but worse than Run D PL-only. Do not select the IK1 replacement over PL-only Run D as current mainline.

Artifacts:

```text
scripts/run_newik1_official_input_full.sh
data/experiments/newik1_official_input_20260604/eval_pl1_streaming_tc_val.json
data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt
```

Detailed record: `EXPERIMENT_LOG.md:256-295`.

## Pending NewIK2 clarification

The user referred to two good `newik2` versions, but the verified artifacts available in the current documentation pass are NewIK1 / IK replacement artifacts. Before documenting any result as NewIK2, locate or request the exact NewIK2 artifact paths and verify that the replaced module is IK-s2 (`iknet.net2`) with the official output contract `15 reduced joints × 6D = 90D`.
