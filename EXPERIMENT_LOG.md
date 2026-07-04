# Experiment Log

This document stores detailed experiment evidence: orchestrator tasks, commands, log paths, JSON paths, failure traces, stdout/stderr summaries, and timelines. Version-summary decisions belong to `RECENT_REPLACEMENT_VERSIONS.md`; current state belongs to `PROJECT_STATUS.md`.

## Detailed Log Index

This index is grouped by version line. Detailed records below can stay chronological or archival, but every new record should include its line, experiment root, cache/protocol, checkpoint, and whether numbers are same-cache comparisons or historical references.

Metric namespace rule:

```text
Do not compare by version name alone.
`newpl_v5_dip_best` under different experiment roots/caches may have different pRB/gR1 values.
Same-cache comparisons are fair; cross-cache rows are historical references only.
```

| Version / Experiment | Detailed Evidence |
|---|---|
| PL-s1 / historical processed | `newpl_v1` through `newpl_v4_init36`; details in `EXP-20260604-*` and `EXP-20260605-001` |
| PL-s1 / official-route v5 | `EXP-20260607-newpl_v5_official_protocol`; root `data/experiments/newpl_v5_official_protocol_20260607_tuned/` |
| PL-s1 / acceleration input filters | `EXP-20260612-newpl_v5_smoothacc`, `EXP-20260612-newpl_v5_butteracc`, `EXP-20260612-newpl_v5_realtime_smooth_residual` |
| PL-s1 / joint-leaf acceleration | `EXP-20260619-newpl_joint_leaf_acc`; root `data/experiments/newpl_joint_leaf_acc_20260619/`; `EXP-20260620-pl_joint_control_acc_aug102_v1_smoke`; root `data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033/`; `EXP-20260620-pl_joint_control_acc_aug102_v1_full`; root `data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805/` |
| PL-s1 / predictive/root/offset | NewPL-root, next-control, offset-v6, v7/v7b acc-aux records in later detailed sections |
| IK-s1 | `newik1_v1` through v14 search records, orchestrator logs, and S4/module JSONs |
| IK-s2 / NewPose | `newpose_ctrl_v1/v2` records and module/full-pipeline evals |
| IMU offset / r_JS | `footlock_transpose_v1`, smoothed-fit audit, offset-net/solver retired routes |
| AccCurve / acceleration residual | `EXP-20260617-acc_curve_v2_gtfk`; strict GTFK standalone acceleration-level AMASS -> DIP module eval under `data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/`; `EXP-20260617-acc_curve_pl_input_eval` tests v1/v2 acceleration as frozen PL input; `EXP-20260618-acc_curve_v1_totalcapture_eval` is retained as wrong pred_zero-vs-GT_full historical reference; `EXP-20260618-acc_curve_v1_fulltrans_rootacc_reconstruction_eval` corrects v1 full-trans eval by adding root translational acceleration back; `EXP-20260618-acc_curve_v3_leafrel_causal_butter` trains the leaf-relative causal Butterworth AccCurve v3 module; `EXP-20260618-acc_curve_v3_error_distribution_rjs_audit` diagnoses the v3 TC failure |
| AccCurve / acceleration residual datacache | `EXP-20260618-acc_invariance_datacache_v2_rebuild`; root-IMU-relative AMASS/DIP/TotalCapture cache rebuild and validation under `data/experiments/acc_invariance_datacache_v2_20260618/`; `EXP-20260618-acc_leaf_relative_residual_v3`; leaf-only residual audit under `data/experiments/acc_leaf_relative_residual_v3_20260618/`; `EXP-20260618-acc_leaf_relative_residual_v4_causal_butterworth`; realtime causal Butterworth audit under `data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/`; `EXP-20260618-acc_leaf_relative_residual_v5_imu_causal_gt_centered`; asymmetric IMU-causal vs GT-centered target audit under `data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618/` |
| Official GPNet + official/processed IMU | `EXP-20260604-001`, `EXP-20260604-002`; S4 JSONs referenced in `RECENT_REPLACEMENT_VERSIONS.md` |
| newpl_v1_processed_no_baseline | `data/experiments/pl_curve_v2_processed_no_baseline/tc_finetune_10ep/train_log.jsonl`; S4 JSON in artifact index |
| newpl_v2_gRdyn | `data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_finetune_v1/tc_finetune_10ep/train_log.jsonl`; S4 JSON not found |
| newpl_v3_gtcontrol_rund | `EXP-20260604-003`; Run D logs and S4 JSON in artifact index |
| newpl_v4_init36 | `EXP-20260605-001`; init36 logs and S4 JSON in artifact index |
| newpl_v5_official_protocol | `EXP-20260607-newpl_v5_official_protocol`; AMASS pretrain -> DIP fine-tune module-level JSONs under `data/experiments/newpl_v5_official_protocol_20260607_tuned/` |
| newpl_v5_smoothacc | `EXP-20260612-newpl_v5_smoothacc`; smooth-aM caches, AMASS -> DIP training, module-level JSONs under `data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/` |
| newpl_v5_butteracc | `EXP-20260612-newpl_v5_butteracc`; causal Butterworth aM input-only gate under `data/experiments/newpl_v5_butteracc_20260612_full/`, plus forced fc12 longtrain under `data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/` |
| newpl_joint_leaf_acc_20260619 | `EXP-20260619-newpl_joint_leaf_acc`; joint-leaf NewPL cache/training/eval route and smoke artifacts under `data/experiments/newpl_joint_leaf_acc_20260619/smoke/` |
| pl_joint_control_acc_aug102_v1 | `EXP-20260620-pl_joint_control_acc_aug102_v1_smoke`; joint-target NewPL control smoke with frozen joint-acc 102D input under `data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033/` |
| pl_joint_control_acc_aug102_v1_full | `EXP-20260620-pl_joint_control_acc_aug102_v1_full`; AMASS -> DIP full module run under `data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805/` |
| newpl_v5_realtime_smooth_residual | `EXP-20260612-newpl_v5_realtime_smooth_residual`; root `data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/`; raw v5 rows are historical references unless same-cache re-eval is run |
| newpl_v6_next_control_smoothacc_gR1 | `EXP-20260613-newpl_v6_next_control_smoothacc_gR1`; root `/tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full`; smooth-aM cache reuse, AMASS 80 -> DIP 40, module eval only |
| newpl_v6_next_p_pdot_pddot_strong | `EXP-20260616-newpl_v6_next_p_pdot_pddot_strong`; root `data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full`; decoded next p/pd/pdd strong supervision, full AMASS->DIP current/next-frame module eval, diagnostic only |
| derivative-aware control-point target policy | `EXP-20260608-derivative_aware_control_fit`; RBDL-only audit and cache smoke artifacts under `data/experiments/gt_control_derivative_audit_20260608/` |
| newik1_v1_control_tail | `EXP-20260604-004`; logs/S4 JSON in artifact index |
| newik1_v2_bonelength | `EXP-20260604-005`; logs/S4 JSON in artifact index |
| newik1_v3_strong_pRJ_control | `EXP-20260604-006`; logs/S4 JSON in artifact index |
| newik1_v4_official_input | `EXP-20260604-007`; logs/S4 JSON in artifact index |
| newik1_v5_last_pl_control | `EXP-20260605-002`; orchestrator logs under `logs/orchestrator/newik1_last_pl_control_20260605_v2/` |
| newik1_v6_official_input_init36_cascade | `EXP-newik1_v6_official_input_init36_cascade_rerun`; AMASS offset enrichment, task logs, S4 JSONs, Module GT JSONs, and final manual selection JSON |
| DIP footlock TransPose pseudo-rJS | `EXP-20260608-footlock_transpose_rjs`; winner cache under `data/experiments/footlock_transpose_rjs_20260608/` |
| footlock-only smoothed-acc rJS cleanup | `EXP-20260609-footlock_only_smoothacc_rjs`; active route is `footlock_transpose_v1` only; old offset-route artifacts deleted |
| acc_curve_pl_input_eval_20260617 | `EXP-20260617-acc_curve_pl_input_eval`; root `data/experiments/acc_curve_pl_input_eval_20260617`; frozen official `GPNet.plnet` input-only evaluation on DIP test |
| acc_curve_v1_totalcapture_eval_20260618 | `EXP-20260618-acc_curve_v1_totalcapture_eval`; root `data/experiments/acc_curve_v1_totalcapture_eval_20260618`; cache root `code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test`; v1 diff-pos acceleration-level TC test only |
| acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618 | `EXP-20260618-acc_curve_v1_fulltrans_rootacc_reconstruction_eval`; root `data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618`; corrected v1 full-trans reconstruction with GT root translational acceleration; no training or downstream claim |
| acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617 | `EXP-20260617-acc_curve_v2_gtfk`; root `data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617`; cache root `code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617`; module-level only |
| acc_curve_v1_20260617 | `EXP-20260617-acc_curve_v1`; root `data/experiments/acc_curve_v1_20260617`; cache root `code/outputs/smooth_acc_cache_amass_dip_20260617`; historical diff-pos-style target |
| acc_invariance_datacache_v2_rebuild_20260618 | `EXP-20260618-acc_invariance_datacache_v2_rebuild`; root `data/experiments/acc_invariance_datacache_v2_20260618`; cache root `data/experiments/acc_invariance_datacache_v2_20260618`; rootIMU-relative acceleration cache rebuild |
| acc_leaf_relative_residual_v3_20260618 | `EXP-20260618-acc_leaf_relative_residual_v3`; root `data/experiments/acc_leaf_relative_residual_v3_20260618`; cache root `data/experiments/acc_leaf_relative_residual_v3_20260618`; leaf-only residual audit, root reference excluded from metrics |
| acc_leaf_relative_residual_v4_causal_butterworth_20260618 | `EXP-20260618-acc_leaf_relative_residual_v4_causal_butterworth`; root `data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618`; cache root `data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618`; realtime causal Butterworth leaf-only residual audit |
| acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618 | `EXP-20260618-acc_leaf_relative_residual_v5_imu_causal_gt_centered`; root `data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618`; cache root `data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618`; asymmetric IMU-causal vs GT-centered target leaf-only residual audit |
| acc_curve_v3_leafrel_causal_butter_20260618 | `EXP-20260618-acc_curve_v3_leafrel_causal_butter`; root `data/experiments/acc_curve_v3_leafrel_causal_butter_20260618`; reusable cache root `data/dataset_work/AccCurveV3LeafRelCausalButter_20260618`; AccCurve v1-style 99D feature, 5-leaf output, causal Butterworth base/target; fail because DIP improves but TotalCapture worsens |
| acc_curve_v3_error_distribution_rjs_audit_20260618 | `EXP-20260618-acc_curve_v3_error_distribution_rjs_audit`; root `data/experiments/acc_curve_v3_error_distribution_rjs_audit_20260618`; diagnostic-only audit of residual distributions, rJS distributions/contribution, and correction transfer; diagnosis `likely_model_overfit` |

## Detailed Records

## EXP-20260620-pl_joint_control_acc_aug102_v1_full - Joint-Target PL Control Acc-Aug102 Full Module Run

Question: after the smoke pass, does the joint-target PL control module with frozen joint acceleration auxiliary input improve direct joint position, velocity, acceleration, and gravity metrics after AMASS pretrain and DIP finetune?

Scope:

```text
experiment: pl_joint_control_acc_aug102_v1
root: data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805
training: AMASS pretrain 80 epochs -> DIP finetune 40 epochs
target_mode: joint_pRB
feature_mode: frozen_joint_acc_aug102
feature_layout: aRB[18]+wRB[18]+RRB[45]+gR0[3]+frozen_joint_acc_R[15]+root_acc_smooth_R[3]
frozen predictor: imu_leaf_acc_predictor_v1, eval-only
checkpoint: /home/lingfeng/projects/imu_acc_explainability/code/outputs/imu_leaf_acc_predictor_v1/full_world_leaf5_no_trans_smoothed_gtacc_centered_ma_w9_20260619_155137/dip_finetune/best.pt
evaluation: module-level DIP test only
not evaluated: IK, full-pipeline, S4, TotalCapture
```

Command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
CUDA_VISIBLE_DEVICES=0 CACHE_DEVICE=cuda:0 /home/lingfeng/bin/longrun -- \
  bash scripts/run_pl_joint_control_acc_aug102_full.sh
```

Cache sanity:

| Split | feature_dim | target_dim | joint-vs-vertex L2 m | frozen_joint_acc_R norm | root_acc_smooth_R norm |
|---|---:|---:|---:|---:|---:|
| AMASS | 102 | 18 | 0.268507 | 1.466330 | 1.233902 |
| DIP train | 102 | 18 | 0.258752 | 1.756837 | 1.765056 |
| DIP val | 102 | 18 | 0.255726 | 1.866173 | 1.839356 |
| DIP test | 102 | 18 | 0.259370 | 2.385026 | 2.179893 |

Training selection:

| Stage | Epochs | Best epoch | Best selection loss |
|---|---:|---:|---:|
| AMASS pretrain | 80 | 1 | 0.8822397489 |
| DIP finetune | 40 | 37 | 0.4478730437 |

DIP test module metrics:

| split | joint_pos_l2_m | joint_vel_l2_mps | joint_acc_l2_mps2 | gravity_angle_deg |
|---|---:|---:|---:|---:|
| after AMASS | 0.277489 | 0.423087 | 43.323986 | 12.944853 |
| after DIP | 0.277398 | 0.423258 | 43.328894 | 12.946709 |

Interpretation:

```text
DIP finetune makes only a negligible joint-position change and does not improve
decoded velocity, decoded acceleration, or gravity direction. The run should be
treated as a diagnostic module-level result, not a promoted PL replacement.
Before running IK/full-pipeline/S4, diagnose the selection/weighted-loss
divergence and why derivative supervision does not reduce derivative metrics.
```

Artifacts:

```text
summary: data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805/SUMMARY.md
summary json: data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805/summary.json
run log: data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805/logs/run.log
AMASS train result: data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805/train/amass_pretrain/train_result.json
DIP train result: data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805/train/dip_finetune/train_result.json
DIP eval after AMASS: data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805/eval/dip_test_after_amass.json
DIP eval after DIP: data/experiments/pl_joint_control_acc_aug102_v1_full_20260620_134805/eval/dip_test_after_dip.json
```

## EXP-20260620-pl_joint_control_acc_aug102_v1_smoke - Joint-Target PL Control Acc-Aug102 Smoke

Question: can a new PL/NewPL control module use SMPL joint-based pRB targets and frozen joint-acceleration augmented 102D input without falling back to the legacy IMU-vertex pRB target?

Scope:

```text
experiment: pl_joint_control_acc_aug102_v1
root: data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033
target_mode: joint_pRB
target_contract: joint_pRB[15]+gR[3], joint_pRB=(SMPL joints [18,19,4,5,15]-root joint) @ root_R
feature_mode: frozen_joint_acc_aug102
feature_layout: aRB[18]+wRB[18]+RRB[45]+gR0[3]+frozen_joint_acc_R[15]+root_acc_smooth_R[3]
frozen predictor: imu_leaf_acc_predictor_v1, eval-only
frozen checkpoint: /home/lingfeng/projects/imu_acc_explainability/code/outputs/imu_leaf_acc_predictor_v1/full_world_leaf5_no_trans_smoothed_gtacc_centered_ma_w9_20260619_155137/dip_finetune/best.pt
conversion: a_joint_pred_R = a_joint_pred_W @ RMB[:,5]
root_acc_smooth: selected root/pelvis IMU aM[:,5], centered_ma window=9, then @ RMB[:,5]
training: smoke only, DIP val 1 sequence, 120 frames, 1 epoch
not evaluated: full AMASS -> DIP training, IK/full-pipeline/S4
```

Commands:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -m py_compile \
  pl_curve.py pl_joint_target.py pl_joint_control_acc_aug102_cache.py \
  pl_joint_control_acc_aug102_train.py pl_joint_control_acc_aug102_eval.py

/home/lingfeng/bin/longrun -- bash scripts/run_pl_joint_control_acc_aug102_smoke.sh
```

Smoke cache sanity:

| Check | Value |
|---|---:|
| feature_dim | 102 |
| target_dim | 18 |
| joint-vs-legacy-vertex diagnostic L2 m | 0.261554 |
| frozen_joint_acc_world_l2_norm_mean | 0.123468 |
| frozen_joint_acc_root_l2_norm_mean | 0.123468 |
| root_acc_smooth_world_l2_norm_mean | 0.063012 |
| root_acc_smooth_root_l2_norm_mean | 0.063012 |
| feature[84:99] abs mean | 0.056628 |
| feature[99:102] abs mean | 0.033728 |

Smoke training:

| Item | Value |
|---|---:|
| status | ok |
| epoch | 1 |
| train_loss | 0.026675697 |
| selection_metric | pl_and_control_physical |
| best_loss | 0.026334771 |

Smoke module eval:

| split | joint_pos_l2_m | joint_vel_l2_mps | joint_acc_l2_mps2 | gravity_angle_deg |
|---|---:|---:|---:|---:|
| smoke | 0.267529 | 0.022015 | 1.920843 | 0.582246 |

Artifacts:

```text
summary: data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033/SUMMARY.md
cache manifest: data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033/cache/dip_val/pl_curve_cache_manifest.json
cache validation: data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033/cache_validation.json
train result: data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033/train/train_result.json
eval metrics: data/experiments/pl_joint_control_acc_aug102_v1_smoke_20260620_132033/eval/metrics.json
```

Conclusion:

```text
Smoke passed. The implementation now has an explicit joint-target target mode,
an explicit frozen_joint_acc_aug102 layout, frozen predictor normalization from
checkpoint, world-to-root acceleration conversion, spline-decoded derivative
metrics, and four required eval metrics. This is not a full training result and
must not be mixed with old vertex-target PL comparisons.
```

## EXP-20260618-acc_curve_v3_error_distribution_rjs_audit - AccCurve v3 Error Distribution and rJS Audit

Question: why does `acc_curve_v3_leafrel_causal_butter_20260618` improve DIP test but worsen TotalCapture test?

Scope:

```text
experiment: acc_curve_v3_error_distribution_rjs_audit_20260618
v4 cache: data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/cache_manifest.json
v3 checkpoint: data/experiments/acc_curve_v3_leafrel_causal_butter_20260618/dip_finetune/best_loss.pt
output root: data/experiments/acc_curve_v3_error_distribution_rjs_audit_20260618
root index: 5
leaf indices: 0..4
root usage: reference acceleration only
root residual/correction metric: excluded
frame: model/world frame M
smoothing: causal Butterworth order=2 cutoff=4Hz on both IMU base and GT target
training: none
PL/NewPL/IK/VR/full pipeline/S4: not evaluated
AMASS: synthetic sanity only
```

Command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /home/lingfeng/bin/longrun -- /home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/audit_acc_curve_v3_error_distribution_rjs_20260618.py \
  --v4-cache-manifest data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/cache_manifest.json \
  --acc-curve-v3-checkpoint data/experiments/acc_curve_v3_leafrel_causal_butter_20260618/dip_finetune/best_loss.pt \
  --output-root data/experiments/acc_curve_v3_error_distribution_rjs_audit_20260618 \
  --sample-frames-per-group 200000 \
  --overwrite
```

Runtime:

```text
elapsed_sec: 923.440347
checkpoint_epoch: 16
checkpoint_selection: 0.8517200946807861
```

Primary diagnostics:

| Comparison | Metric | Value |
|---|---|---:|
| DIP train vs TC test residual | mean diff norm | 0.809654 |
| DIP train vs TC test residual | diagonal Gaussian FD | 2.173161 |
| DIP train vs TC test residual | MMD RBF | 0.061301 |
| DIP train vs TC test residual | mean-vector cosine | 0.268100 |
| DIP train vs TC test rJS | mean diff norm | 0.136650 |
| DIP train vs TC test rJS | diagonal Gaussian FD | 0.034094 |
| DIP train vs TC test rJS | mean rJS cosine | 0.961810 |
| DIP test rJS acceleration | offset contribution ratio | 0.733277 |
| TC test rJS acceleration | offset contribution ratio | 0.756321 |
| DIP test correction transfer | harmful rate | 0.442142 |
| TC test correction transfer | harmful rate | 0.564954 |
| TC test correction transfer | overcorrection rate | 0.304232 |
| TC test correction transfer | correction corr | 0.227905 |

Largest TC test shifted sensors by base residual L2:

| Sensor | L2 |
|---|---:|
| right_forearm | 1.354584 |
| left_lower_leg | 1.212403 |
| right_lower_leg | 1.204943 |

Conclusion:

```text
final diagnosis: likely_model_overfit

The audit does show a DIP-train to TC-test residual distribution shift, but the
rJS distribution difference is much smaller and rJS acceleration contribution
ratios are similar across DIP and TC.  The strongest failure signal is correction
transfer: TC test correction cosine/corr are low, harmful_rate is 0.564954, and
overcorrection_rate is 0.304232.  The AccCurve v3 residual appears DIP-specific
and often harmful on TC-like distributions.

Recommended next experiment:
  use base only for TC-like distribution; then test residual_scale smaller or a
  sensor-specific residual gate.
```

Artifacts:

```text
script: scripts/audit_acc_curve_v3_error_distribution_rjs_20260618.py
summary: data/experiments/acc_curve_v3_error_distribution_rjs_audit_20260618/summary.md
residual_distribution_by_group.json
residual_distribution_by_sensor.csv
residual_distribution_distance.json
rjs_stats_by_group.json
rjs_stats_by_sensor.csv
rjs_outlier_sequences.csv
rjs_dataset_distance.json
rjs_acc_contribution_by_group.json
rjs_acc_contribution_by_sensor.csv
correction_transfer_by_group.json
correction_transfer_by_sensor.csv
harmful_sequences.csv
```

## EXP-20260618-acc_curve_v3_leafrel_causal_butter - Leaf-Relative Causal Butterworth AccCurve v3

Question: can an AccCurve v1-style residual acceleration module improve the v4 causal Butterworth leaf-relative residual target without including the root sensor in prediction, loss, or metrics?

Scope:

```text
experiment: acc_curve_v3_leafrel_causal_butter_20260618
module: AccCurve v1-style residual acceleration curve, 99D input and 15D state
cache root: data/dataset_work/AccCurveV3LeafRelCausalButter_20260618
source cache: data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/cache_manifest.json
output root: data/experiments/acc_curve_v3_leafrel_causal_butter_20260618
root index: 5
leaf indices: 0..4
root usage: reference acceleration only
root prediction/loss/metric: excluded
input smoothing: causal Butterworth order=2 cutoff=4Hz
target smoothing: causal Butterworth order=2 cutoff=4Hz
frame: model/world frame M; no sensor-local rotation; no root-frame rotation
target/output units: m/s^2; no target/output normalization
normalization: feature z-score from AMASS train split only
training: AMASS pretrain synthetic sanity -> DIP finetune; no TotalCapture train/val in main training
not evaluated: PL, NewPL, IK, VR, full pipeline, S4, downstream pose improvement
```

Reusable project-level cache:

```text
builder: scripts/build_acc_curve_v3_leafrel_feature_cache_20260618.py
manifest: data/dataset_work/AccCurveV3LeafRelCausalButter_20260618/cache_manifest.json
sequences: 1404
valid frames: 1609025
failures: 0
datasets: AMASS train 1298; DIP train/val/test 36/6/19; TotalCapture train/val/test 36/5/4
feature layout: aIMU_leaf_rel_raw[15] + aIMU_leaf_rel_butter2_4hz[15] + raw_minus_butter[15] + wM[18] + RMB_6d[36]
base key: aIMU_leaf_rel_butter2_4hz
target key: aGT_leaf_rel_butter2_4hz
```

Training command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /home/lingfeng/bin/longrun -- /home/lingfeng/.conda/envs/globalpose-gpu/bin/python acc_curve_v3_leafrel_train.py \
  --mode train_full \
  --cache-manifest data/dataset_work/AccCurveV3LeafRelCausalButter_20260618/cache_manifest.json \
  --output-dir data/experiments/acc_curve_v3_leafrel_causal_butter_20260618 \
  --epochs 30 \
  --dip-epochs 20 \
  --window 240 \
  --stride 120 \
  --batch-size 64 \
  --num-workers 8 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --hidden-size 512 \
  --dropout 0.1 \
  --residual-scale 1.0 \
  --control-prior-weight 1e-5 \
  --grad-clip 1.0 \
  --seed 1234 \
  --overwrite
```

Training selection:

| Stage | Best epoch | Best validation pred/base ratio | Train seq | Val seq | Train windows | Val windows |
|---|---:|---:|---:|---:|---:|---:|
| AMASS pretrain | 13 | 0.940240 | 1231 | 67 | 8231 | 407 |
| DIP finetune | 16 | 0.851720 | 36 | 6 | 1887 | 253 |

Primary test results:

| Dataset | Split | Pred L2 | Base L2 | Pred/Base L2 | Pred RMSE | Base RMSE | Corr | Base Corr | Valid frames |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP | test | 0.990334 | 1.196030 | 0.828017 | 0.750235 | 0.924315 | 0.958276 | 0.943321 | 57956 |
| TotalCapture | test | 1.365116 | 1.052403 | 1.297142 | 1.065184 | 0.902071 | 0.923813 | 0.946864 | 16116 |

Decision:

```text
fail

DIP test improves strongly over the causal Butterworth base in L2, RMSE, and corr.
TotalCapture test worsens sharply: pred/base L2 ratio is 1.297142, RMSE ratio is
1.180820, and corr drops from 0.946864 to 0.923813. This fails the primary
cross-dataset gate. AccCurve v3 remains an acceleration-level diagnostic module
only and is not promoted.
```

Artifacts:

```text
module: acc_curve_v3_leafrel.py
trainer: acc_curve_v3_leafrel_train.py
cache builder: scripts/build_acc_curve_v3_leafrel_feature_cache_20260618.py
runner: scripts/run_acc_curve_v3_leafrel_causal_butter_20260618.sh
cache manifest: data/dataset_work/AccCurveV3LeafRelCausalButter_20260618/cache_manifest.json
summary: data/experiments/acc_curve_v3_leafrel_causal_butter_20260618/summary.md
train result: data/experiments/acc_curve_v3_leafrel_causal_butter_20260618/train_result.json
eval JSONs: data/experiments/acc_curve_v3_leafrel_causal_butter_20260618/eval/*_eval.json
```

## EXP-20260618-acc_leaf_relative_residual_v5_imu_causal_gt_centered - Asymmetric IMU-Causal vs GT-Centered Target Audit

Question: can target-only centered smoothing on the GT side improve leaf-relative acceleration residual explainability while preserving realtime causal smoothing on the IMU side?

Scope:

```text
model training: none
AccCurve training: none
PL/NewPL training: none
IK/VR/full pipeline: not evaluated
S4 metrics: not evaluated
root index: 5
leaf indices: 0..4
root usage: reference acceleration only
root residual/loss/metric: excluded
frame: model/world frame M; no sensor-local rotation
GT FK: tran forced to zero
diff method: centered second difference, dt=1/60
IMU smoother: causal Butterworth, order=2, cutoff_hz=4.0, fps=60, zero-lookahead
GT smoother: centered moving average, window=9, non-realtime, target-only
filter boundary: only valid centered-difference segment is smoothed; GT NaN boundary frames are not passed into filters
primary evidence: DIP and TotalCapture only
secondary sanity: AMASS and ALL
```

Main comparison:

```text
raw_leaf_relative:
  aIMU_leaf_rel_raw = aM_leaf - aM_root
  aGT_leaf_rel_raw = diff_acc(p_leaf_zero_trans) - diff_acc(p_root_zero_trans)

imu_butter2_4hz_vs_gt_centered_ma9:
  causal_butterworth(aIMU_leaf_rel_raw[valid_segment], order=2, cutoff_hz=4, fps=60)
  vs centered_ma9(aGT_leaf_rel_raw[valid_segment])
```

Build command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/bin/longrun -- /home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618.py \
  --output-root data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618 \
  --fk-batch-size 2048 \
  --progress-every 50 \
  --overwrite
```

Validate command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/validate_acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618.py \
  --root data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618
```

Primary validation summary:

| Dataset | Formulation | L2 | RMSE | Corr | valid frames |
|---|---|---:|---:|---:|---:|
| DIP | raw_leaf_relative | 2.360895 | 3.530190 | 0.610050 | 317450 |
| DIP | imu_butter2_4hz_vs_gt_centered_ma9 | 1.828599 | 2.060184 | 0.687890 | 317450 |
| TotalCapture | raw_leaf_relative | 2.905783 | 4.418546 | 0.563076 | 176159 |
| TotalCapture | imu_butter2_4hz_vs_gt_centered_ma9 | 2.480798 | 2.426839 | 0.656914 | 176159 |

Reference comparison:

| Dataset | v4 symmetric butter L2/RMSE/corr | v3 centered oracle L2/RMSE/corr | asym/v4 RMSE ratio | asym/v3 RMSE ratio |
|---|---:|---:|---:|---:|
| DIP | 0.893481 / 0.914904 / 0.943719 | 0.796990 / 0.791042 / 0.948895 | 2.251804 | 2.604393 |
| TotalCapture | 1.092481 / 1.050146 / 0.941474 | 1.011330 / 0.897591 / 0.945537 | 2.310953 | 2.703725 |

Checks:

```text
sequences: 1404
valid frames: 1609025
shape consistency: true
root excluded from residual metrics: true
leaf indices: [0, 1, 2, 3, 4]
IMU zero-lookahead: true
GT centered target-only: true
decision: fail
```

Conclusion:

```text
The asymmetric smoothing target fails the primary real-IMU gate.  It improves
DIP and TotalCapture RMSE over raw, but corr remains far below 0.90 and RMSE is
more than 2.25x worse than v4 symmetric Butterworth.  The likely explanation is
phase mismatch: IMU causal smoothing has lag, while GT centered smoothing is
near zero-phase.  GT centered smoothing is target-only and not available at
runtime.  This remains a residual audit only; it is not AccCurve/PL/NewPL
training, not a full-pipeline evaluation, and does not claim downstream pose
improvement.
```

Artifacts:

```text
script: scripts/build_acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618.py
validator: scripts/validate_acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618.py
experiment root: data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618
cache manifest: data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618/cache_manifest.json
metrics: data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618/metrics.json
per-sequence csv: data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618/per_sequence.csv
debug: data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618/debug.json
summary: data/experiments/acc_leaf_relative_residual_v5_imu_causal_gt_centered_20260618/summary.md
```

## EXP-20260618-acc_leaf_relative_residual_v4_causal_butterworth - Realtime Leaf-Only Causal Butterworth Audit

Question: can zero-lookahead causal Butterworth smoothing improve leaf-relative IMU/FK acceleration residual explainability on real-IMU datasets while staying close to the v3 centered moving-average oracle?

Scope:

```text
model training: none
AccCurve training: none
PL/NewPL training: none
IK/VR/full pipeline: not evaluated
S4 metrics: not evaluated
root index: 5
leaf indices: 0..4
root usage: reference acceleration only
root residual/loss/metric: excluded
frame: model/world frame M; no sensor-local rotation
GT FK: tran forced to zero
diff method: centered second difference, dt=1/60
realtime smoother: causal Butterworth, order=2, cutoff_hz=4.0, fps=60, zero-lookahead
filter boundary: only valid centered-difference segment is filtered; GT NaN boundary frames are not passed into the IIR filter
primary evidence: DIP and TotalCapture only
secondary sanity: AMASS and ALL
```

Main comparison:

```text
raw_leaf_relative:
  aIMU_leaf_rel_raw = aM_leaf - aM_root
  aGT_leaf_rel_raw = diff_acc(p_leaf_zero_trans) - diff_acc(p_root_zero_trans)

butter2_4hz_leaf_relative:
  causal_butterworth(aIMU_leaf_rel_raw[valid_segment], order=2, cutoff_hz=4, fps=60)
  vs causal_butterworth(aGT_leaf_rel_raw[valid_segment], order=2, cutoff_hz=4, fps=60)
```

Build command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/bin/longrun -- /home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_acc_leaf_relative_residual_v4_causal_butterworth_20260618.py \
  --output-root data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618 \
  --fk-batch-size 2048 \
  --progress-every 50 \
  --overwrite
```

Validate command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/validate_acc_leaf_relative_residual_v4_causal_butterworth_20260618.py \
  --root data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618
```

Primary validation summary:

| Dataset | Formulation | L2 | RMSE | Corr | valid frames |
|---|---|---:|---:|---:|---:|
| DIP | raw_leaf_relative | 2.360895 | 3.530190 | 0.610050 | 317450 |
| DIP | butter2_4hz_leaf_relative | 0.893481 | 0.914904 | 0.943719 | 317450 |
| TotalCapture | raw_leaf_relative | 2.905783 | 4.418546 | 0.563076 | 176159 |
| TotalCapture | butter2_4hz_leaf_relative | 1.092481 | 1.050146 | 0.941474 | 176159 |

v3 centered moving-average oracle reference:

| Dataset | v3 centered_ma9 L2 | v3 centered_ma9 RMSE | v3 centered_ma9 Corr | v4 butter/oracle RMSE ratio |
|---|---:|---:|---:|---:|
| DIP | 0.796990 | 0.791042 | 0.948895 | 1.156581 |
| TotalCapture | 1.011330 | 0.897591 | 0.945537 | 1.169961 |

Checks:

```text
sequences: 1404
valid frames: 1609025
shape consistency: true
root excluded from residual metrics: true
leaf indices: [0, 1, 2, 3, 4]
zero-lookahead: true
decision: pass
```

Conclusion:

```text
The realtime causal Butterworth smoother passes the primary real-IMU gate.
It improves DIP and TotalCapture over raw in L2/RMSE/corr, keeps corr above
0.94 on both datasets, and remains within 20% RMSE of the v3 centered
moving-average oracle.  AMASS is synthetic sanity only and is not used as the
final selection basis.  This remains a residual audit only; it is not
AccCurve/PL/NewPL training, not a full-pipeline evaluation, and does not claim
downstream pose improvement.
```

Artifacts:

```text
script: scripts/build_acc_leaf_relative_residual_v4_causal_butterworth_20260618.py
validator: scripts/validate_acc_leaf_relative_residual_v4_causal_butterworth_20260618.py
experiment root: data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618
cache manifest: data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/cache_manifest.json
metrics: data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/metrics.json
per-sequence csv: data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/per_sequence.csv
debug: data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/debug.json
summary: data/experiments/acc_leaf_relative_residual_v4_causal_butterworth_20260618/summary.md
```

## EXP-20260618-acc_leaf_relative_residual_v3 - Leaf-Only Acceleration Residual Audit

Question: after fixing the v2 root-channel metric issue, how large is the residual between measured leaf-relative IMU acceleration and zero-translation FK leaf-relative acceleration, and how much does direct smoothing reduce it?

Scope:

```text
model training: none
AccCurve training: none
PL/NewPL training: none
IK/VR/full pipeline: not evaluated
S4 metrics: not evaluated
root index: 5
leaf indices: 0..4
root usage: reference acceleration only
root residual/loss/metric: excluded
frame: model/world frame M; no sensor-local rotation
GT FK: tran forced to zero
diff method: centered second difference, dt=1/60
smooth method: centered moving average, window=9
```

Main comparison:

```text
raw_leaf_relative:
  aIMU_leaf_rel_raw = aM_leaf - aM_root
  aGT_leaf_rel_raw = diff_acc(p_leaf_zero_trans) - diff_acc(p_root_zero_trans)

smooth_leaf_relative:
  smooth_centered(aIMU_leaf_rel_raw, window=9)
  vs smooth_centered(aGT_leaf_rel_raw, window=9)
```

Build command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/bin/longrun -- /home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_acc_leaf_relative_residual_v3_20260618.py \
  --output-root data/experiments/acc_leaf_relative_residual_v3_20260618 \
  --fk-batch-size 2048 \
  --progress-every 50 \
  --overwrite
```

Validate command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/validate_acc_leaf_relative_residual_v3_20260618.py \
  --root data/experiments/acc_leaf_relative_residual_v3_20260618
```

Validation summary:

| Dataset | Formulation | L2 | RMSE | Corr | valid frames |
|---|---|---:|---:|---:|---:|
| AMASS | smooth_leaf_relative | 0.272780 | 0.389472 | 0.990993 | 1115416 |
| DIP | smooth_leaf_relative | 0.796990 | 0.791042 | 0.948895 | 317450 |
| TotalCapture | smooth_leaf_relative | 1.011330 | 0.897591 | 0.945537 | 176159 |
| ALL | raw_leaf_relative | 1.372700 | 2.269143 | 0.857988 | 1609025 |
| ALL | smooth_leaf_relative | 0.457061 | 0.562864 | 0.979057 | 1609025 |

Checks:

```text
sequences: 1404
valid frames: 1609025
shape consistency: true
root excluded from residual metrics: true
leaf indices: [0, 1, 2, 3, 4]
```

Conclusion:

```text
Smoothing strongly reduces the leaf-relative residual: ALL L2 drops by
0.915639 and RMSE drops by 1.706279, while corr improves by 0.121069.
Therefore smoothing is necessary before comparing IMU/FK acceleration
residuals, and root-relative smoothed acceleration is a cleaner explainability
target. This remains a residual audit only; it is not AccCurve/PL/NewPL
training, not a full-pipeline evaluation, and does not claim downstream pose
improvement.
```

Artifacts:

```text
script: scripts/build_acc_leaf_relative_residual_v3_20260618.py
validator: scripts/validate_acc_leaf_relative_residual_v3_20260618.py
experiment root: data/experiments/acc_leaf_relative_residual_v3_20260618
cache manifest: data/experiments/acc_leaf_relative_residual_v3_20260618/cache_manifest.json
metrics: data/experiments/acc_leaf_relative_residual_v3_20260618/metrics.json
per-sequence csv: data/experiments/acc_leaf_relative_residual_v3_20260618/per_sequence.csv
debug: data/experiments/acc_leaf_relative_residual_v3_20260618/debug.json
summary: data/experiments/acc_leaf_relative_residual_v3_20260618/summary.md
```

## EXP-20260618-acc_invariance_datacache_v2_rebuild - Root-IMU-Relative Acceleration Cache Rebuild

Question: can we rebuild a single acceleration cache contract where the IMU and GT targets are both root-IMU-relative and zero-translation aligned across AMASS, DIP, and TotalCapture?

Scope:

```text
model training: none
PL training: none
IK/VR/full pipeline: not evaluated
cache rebuild: yes
validation: root invariance, shape consistency, leakage corr, per-sequence/per-sensor metrics
frame: model/world frame M; no sensor-local rotation
root index: 5 (pelvis)
```

Build command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/bin/longrun -- /home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_acc_invariance_datacache_v2_20260618.py \
  --output-root data/experiments/acc_invariance_datacache_v2_20260618 \
  --fk-batch-size 2048 \
  --progress-every 50 \
  --overwrite
```

Validate command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/validate_acc_invariance_datacache_v2_20260618.py \
  --root data/experiments/acc_invariance_datacache_v2_20260618
```

Cache contract:

```text
manifest: data/experiments/acc_invariance_datacache_v2_20260618/cache_manifest.json
type: rootIMU_relative_acceleration
datasets: AMASS / DIP / TotalCapture
sequences: 1404
root index: 5 (pelvis)
IMU target: aM_rel = aM_smooth - aM_smooth[:, 5]
GT target: aGT_rel = diff_acc(FK_zero_translation(p_WJ + R_WJ @ rJS)) - root
diff method: centered second difference / dt^2 with dt = 1/60
```

Validation summary:

| formulation | dataset | L2 | RMSE | corr |
|---|---|---:|---:|---:|
| raw absolute | ALL | 2.559527 | 2.973042 | 0.693266 |
| zero-trans old | ALL | 2.767145 | 3.342987 | 0.540240 |
| v2 relative (NEW) | ALL | 2.048167 | 3.035057 | 0.639486 |

Root/leakage checks:

```text
shape consistency: True
root invariance max mean |aM_rel[:,5]|: 0.0
leakage pass: 1251/1404 sequences
mean corr(v2 relative): 0.617329
mean corr(raw absolute): 0.623005
```

Conclusion:

```text
The rebuild is physically well-formed: shape checks pass and the root-IMU
relative acceleration is exactly root-invariant by construction.  However, the
new root-relative formulation does not beat raw absolute correlation overall;
mean corr(v2 relative)=0.617329 is slightly below raw absolute=0.623005.
This means the cache is consistent, but raw absolute acceleration still carries
slightly more direct correlation in this validation slice. Keep the new cache
for root-invariant experiments, but do not claim it improves correlation by
itself.
```

Artifacts:

```text
experiment root: data/experiments/acc_invariance_datacache_v2_20260618
cache manifest: data/experiments/acc_invariance_datacache_v2_20260618/cache_manifest.json
metrics: data/experiments/acc_invariance_datacache_v2_20260618/metrics.json
per-sequence csv: data/experiments/acc_invariance_datacache_v2_20260618/per_sequence.csv
debug root leakage: data/experiments/acc_invariance_datacache_v2_20260618/debug_root_leakage.json
summary: data/experiments/acc_invariance_datacache_v2_20260618/summary.md
```

## EXP-20260618-acc_curve_v1_fulltrans_rootacc_reconstruction_eval - v1 Full-Trans Reconstruction with Root Acceleration

Question: was the old TotalCapture full-trans failure caused by comparing the zero-trans AccCurve v1 prediction against full-trans ground truth, and does adding root translational acceleration back beat the full-trans `aM_smooth` baseline?

Scope:

```text
model training: none
PL/NewPL training: none
IK/VR/full pipeline: not evaluated
S4 metrics: not evaluated
target namespace: AccCurve v1 historical target only; not strict GTFK v2 and not leaf-relative causal Butterworth v3
primary corrected eval: pred_zero + a_root_trans_smooth vs GT_full
baseline: v1 historical spline-decoded aM_smooth vs GT_full
frame: model/world frame M for input/base/pred/target
```

Run command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
mkdir -p data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618/logs
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/bin/longrun -- bash -lc 'export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}; /home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/eval_acc_curve_v1_fulltrans_rootacc_reconstruction_20260618.py --zero-trans-cache code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test/acc_curve_cache_manifest.json --full-trans-cache code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test/acc_curve_cache_manifest.json --source-cache data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json --checkpoint data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt --output-root data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618 --device cuda --overwrite 2>&1 | tee data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618/logs/run.log'
```

Input contracts:

```text
checkpoint: data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt
full-trans cache: code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test/acc_curve_cache_manifest.json
zero-trans cache: code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test/acc_curve_cache_manifest.json
source TC cache: data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json
feature_zero/full max abs diff: 0
aM_smooth_zero/full max abs diff: 0
valid mask xor frames: 0
```

Decomposition sanity:

| Check | max abs | mean abs | RMSE |
|---|---:|---:|---:|
| `GT_full - (GT_zero + cache_root_acc)` | `0.000366449` | `0.000013115` | `0.000026308` |
| `cache_root_acc - smooth(diff_acc(tran_gt))` | `0.000267982` | `0.000010596` | `0.000021218` |

All-6-sensor primary results on TotalCapture test:

| Row | L2 | RMSE | MAE | Corr | Cosine | Mag MAE | Pred/Base ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_full: decoded `aM_smooth` vs GT_full | `0.873843` | `0.693060` | `0.429852` | `0.974734` | `0.774869` | `0.456501` | `1.000000` |
| wrong_pred_zero_vs_full | `2.091960` | `1.539445` | `1.029687` | `0.866428` | `0.442708` | `1.125138` | `2.393977` |
| correct_pred_zero_plus_gt_root_trans | `1.415560` | `0.977232` | `0.709856` | `0.949590` | `0.659106` | `0.798133` | `1.619925` |
| optional_pred_zero_plus_imu_root_est | `1.204596` | `0.886916` | `0.597049` | `0.958057` | `0.699401` | `0.636703` | `1.378504` |
| zero_trans_sanity | `1.415560` | `0.977232` | `0.709856` | `0.945382` | `0.552023` | `0.786097` | `0.772415` |

Leaf-only secondary results:

| Row | L2 | RMSE | Corr | Pred/Base ratio |
|---|---:|---:|---:|---:|
| baseline_full | `0.899489` | `0.719889` | `0.976317` | `1.000000` |
| wrong_pred_zero_vs_full | `2.143246` | `1.565493` | `0.881801` | `2.382737` |
| correct_pred_zero_plus_gt_root_trans | `1.492559` | `1.024714` | `0.952284` | `1.659341` |
| optional_pred_zero_plus_imu_root_est | `1.296393` | `0.941162` | `0.959565` | `1.441255` |
| zero_trans_sanity | `1.492558` | `1.024714` | `0.950637` | `0.820339` |

Conclusion:

```text
AccCurve v1 predicts root-translation-free sensor-site acceleration, not full
absolute acceleration. The old TC full-trans result was an unfair
pred_zero-vs-GT_full target-mismatched evaluation and is reproduced here as the
wrong row. However, adding cache-consistent GT root translational acceleration
back gives pred/base ratio 1.619925, still worse than the full-trans aM_smooth
baseline. Therefore target mismatch explains part, but not all, of the old
full-trans failure. This is an acceleration-level historical correction only:
no PL/NewPL/full-pipeline/S4 claim and no retraining.
```

Artifacts:

```text
script: scripts/eval_acc_curve_v1_fulltrans_rootacc_reconstruction_20260618.py
experiment root: data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618
summary: data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618/summary.md
eval JSON: data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618/fulltrans_reconstruction_eval.json
eval CSV: data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618/fulltrans_reconstruction_eval.csv
per-sequence CSV: data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618/per_sequence_metrics.csv
per-sensor CSV: data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618/per_sensor_metrics.csv
sanity JSON: data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618/decomposition_sanity.json
run log: data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618/logs/run.log
exact command: data/experiments/acc_curve_v1_fulltrans_rootacc_reconstruction_eval_20260618/exact_command.txt
```

## EXP-20260618-acc_curve_v1_totalcapture_zero_trans_eval - v1 Zero-Translation Acceleration Target on TotalCapture Test

Question: after forcing TotalCapture translation to zero, does the existing AccCurve v1 checkpoint still fail on the DIP-style acceleration target?

Scope:

```text
model training: none
PL training: none
IK/VR/full pipeline: not evaluated
S4 metrics: not evaluated
target namespace: v1 only, smooth(diff_acc(p_WS_zero_trans)); not strict GTFK v2
p_WS_zero_trans: p_WJ + R_WJ @ rJS with tran forced to zero
frame: model/world frame M for input/base/pred/target
```

Cache build:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/bin/longrun -- /home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_acc_curve_cache.py \
  --input-cache data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json \
  --output-dir code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test \
  --dataset TotalCapture \
  --split test_zero_trans \
  --smooth-window 9 \
  --smoothing-mode centered_moving_average \
  --trim 4 \
  --shard-size 32 \
  --fk-batch-size 2048 \
  --force-zero-tran \
  --overwrite
```

Cache contract:

```text
manifest: code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test/acc_curve_cache_manifest.json
type: acc_curve_cache_v1
force_zero_tran: true
source cache/protocol: data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json
dataset/split: TotalCapture/test_zero_trans
sequences/frames/valid: 4 / 16124 / 16084
target_layout: aFK_smooth[18], six sensor-site accelerations in m/s^2
input_frame: model/world frame M from GlobalPose aM/wM/RMB cache fields
target_frame: model/world frame with root translation removed; p_WS = p_WJ + R_WJ @ rJS
```

Eval command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
mkdir -p data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618/logs
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/bin/longrun -- /home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/eval_acc_curve_v1_totalcapture_20260618.py \
  --experiment-name acc_curve_v1_totalcapture_zero_trans_eval_20260618 \
  --output-root data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618 \
  --cache code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test/acc_curve_cache_manifest.json \
  --checkpoint data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt \
  --device cuda
```

TotalCapture zero-trans results:

| Dataset | Target | Acc source | L2 error | RMSE | pred/base ratio | corr | valid frames |
|---|---|---|---:|---:|---:|---:|---:|
| TotalCapture test | `smooth(diff_acc(p_WS_zero_trans))` | `aM_smooth` | `1.832642` | `1.466451` | `1.000000` | `0.883554` | `16084` |
| TotalCapture test | `smooth(diff_acc(p_WS_zero_trans))` | AccCurve v1 pred | `1.415560` | `0.977232` | `0.772415` | `0.945382` | `16084` |

DIP v1 historical reference:

| Dataset | Target | pred L2 | base L2 | pred/base ratio | pred RMSE | base RMSE | corr |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP test | `smooth(diff_acc(p_WS_zero_trans))` | `1.202067` | `2.368697` | `0.622049` | `0.930242` | `1.733464` | `0.940837` |
| TotalCapture zero-trans test | `smooth(diff_acc(p_WS_zero_trans))` | `1.415560` | `1.832642` | `0.772415` | `0.977232` | `1.466451` | `0.945382` |

TC full-trans historical reference:

| Dataset | Target | pred L2 | base L2 | pred/base ratio | pred RMSE | base RMSE | corr |
|---|---|---:|---:|---:|---:|---:|---:|
| TotalCapture test | `smooth(diff_acc(trans + p_WS))` | `2.091960` | `0.873843` | `2.393977` | `1.539445` | `0.693060` | `0.866428` |

Conclusion:

```text
Forcing TC tran to zero confirms that v1 is a root-translation-free
sensor-site acceleration predictor: it beats the zero-trans target with
pred/base ratio 0.772415 and corr 0.945382. This is not a full-trans success
claim. The corrected full-trans reconstruction experiment shows that adding
root translational acceleration back still does not beat the full-trans
aM_smooth baseline.
```

Artifacts:

```text
script: scripts/build_acc_curve_cache.py
script: scripts/eval_acc_curve_v1_totalcapture_20260618.py
cache root: code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test
experiment root: data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618
cache manifest: code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test/acc_curve_cache_manifest.json
eval JSON: data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618/tc_test_eval.json
per-sequence CSV: data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618/tc_test_per_sequence.csv
summary: data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618/summary.md
cache build log: data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618/logs/cache_build.log
eval log: data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618/logs/run.log
exact command: data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618/exact_command.txt
```

## EXP-20260618-acc_curve_v1_totalcapture_eval - v1 Diff-Pos Acceleration Target on TotalCapture Test

Question: before using AccCurve v1 acceleration to retrain NewPL, does the
existing v1 checkpoint generalize to TotalCapture test on its own acceleration
target?

Scope:

```text
model training: none
PL training: none
IK/VR/full pipeline: not evaluated
S4 metrics: not evaluated
target namespace: v1 only, smooth(diff_acc(p_WS)); not strict GTFK v2
p_WS: p_WJ + R_WJ @ rJS
frame: model/world frame M for input/base/pred/target
```

Cache build:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/bin/longrun -- /home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_acc_curve_cache.py \
  --input-cache data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json \
  --output-dir code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test \
  --dataset TotalCapture \
  --split test \
  --smooth-window 9 \
  --smoothing-mode centered_moving_average \
  --trim 4 \
  --shard-size 32 \
  --fk-batch-size 2048 \
  --overwrite
```

Cache contract:

```text
manifest: code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test/acc_curve_cache_manifest.json
type: acc_curve_cache_v1
source cache/protocol: data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json
dataset/split: TotalCapture/test
sequences/frames/valid: 4 / 16124 / 16084
target_layout: aFK_smooth[18], six sensor-site accelerations in m/s^2
input_frame: model/world frame M from GlobalPose aM/wM/RMB cache fields
target_frame: same model/world frame M; aFK_smooth is ddot(p_WJ + R_WJ @ r_JS)
```

Eval command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
mkdir -p data/experiments/acc_curve_v1_totalcapture_eval_20260618/logs
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/bin/longrun -- /home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/eval_acc_curve_v1_totalcapture_20260618.py \
  --output-root data/experiments/acc_curve_v1_totalcapture_eval_20260618 \
  --cache code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test/acc_curve_cache_manifest.json \
  --checkpoint data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt \
  --device cuda
```

TotalCapture results:

| Dataset | Target | Acc source | L2 error | RMSE | pred/base ratio | corr | valid frames |
|---|---|---|---:|---:|---:|---:|---:|
| TotalCapture test | `smooth(diff_acc(p_WS))` | `aM_smooth` | `0.873843` | `0.693060` | `1.000000` | `0.974734` | `16084` |
| TotalCapture test | `smooth(diff_acc(p_WS))` | AccCurve v1 pred | `2.091960` | `1.539445` | `2.393977` | `0.866428` | `16084` |

DIP v1 historical reference:

| Dataset | Target | pred L2 | base L2 | pred/base ratio | pred RMSE | base RMSE | corr |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP test | `smooth(diff_acc(p_WS))` | `1.202067` | `2.368697` | `0.622049` | `0.930242` | `1.733464` | `0.940837` |
| TotalCapture test | `smooth(diff_acc(p_WS))` | `2.091960` | `0.873843` | `2.393977` | `1.539445` | `0.693060` | `0.866428` |

Conclusion:

```text
TotalCapture pred/base ratio = 2.393977 for this direct pred_zero-vs-GT_full
comparison. This is now retained as the wrong target-mismatched full-trans
reference, not as the corrected v1 full-trans conclusion. Use
EXP-20260618-acc_curve_v1_fulltrans_rootacc_reconstruction_eval for the
corrected full-trans result.
```

Artifacts:

```text
script: scripts/eval_acc_curve_v1_totalcapture_20260618.py
cache root: code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test
experiment root: data/experiments/acc_curve_v1_totalcapture_eval_20260618
eval JSON: data/experiments/acc_curve_v1_totalcapture_eval_20260618/tc_test_eval.json
per-sequence CSV: data/experiments/acc_curve_v1_totalcapture_eval_20260618/tc_test_per_sequence.csv
summary: data/experiments/acc_curve_v1_totalcapture_eval_20260618/summary.md
cache build log: data/experiments/acc_curve_v1_totalcapture_eval_20260618/logs/cache_build.log
eval log: data/experiments/acc_curve_v1_totalcapture_eval_20260618/logs/run.log
exact command: data/experiments/acc_curve_v1_totalcapture_eval_20260618/exact_command.txt
```

## EXP-20260617-acc_curve_pl_input_eval - AccCurve v1/v2 as Frozen Baseline PL Acceleration Input

Question: Do AccCurve v1/v2 acceleration predictions improve the frozen official
baseline PL module output when they replace only the legacy PL acceleration
channel?

Run contract:

```text
experiment: acc_curve_pl_input_eval_20260617
root: data/experiments/acc_curve_pl_input_eval_20260617
evaluator: scripts/eval_pl_with_acc_curve_input_20260617.py
frozen PL checkpoint: data/weights.pt, official GPNet.plnet weights only
DIP test cache/protocol:
  data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
AccCurve v1 checkpoint:
  data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt
AccCurve v2 checkpoint:
  data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/dip_finetune/best_loss.pt
PL input contract:
  aRB[18] + wRB[18] + RRB[45] + gR0[3] = 84D
Replacement rule:
  only replace aRB[18]; keep wRB/RRB/gR0, target, mask, split, and checkpoint fixed
Frame contract:
  AccCurve pred is model/world-frame M acceleration; convert to PL root frame
  with aRB = acc_M @ RMB_root before PL forward
Target:
  pl_target_from_pose(pose_gt): pRB[15]+gR1[3], shared by all variants
DIP test training/norm/checkpoint selection:
  none
```

Exact command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
mkdir -p data/experiments/acc_curve_pl_input_eval_20260617/logs
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/eval_pl_with_acc_curve_input_20260617.py --output-root data/experiments/acc_curve_pl_input_eval_20260617 --device cuda 2>&1 | tee data/experiments/acc_curve_pl_input_eval_20260617/logs/run.log
```

Validation:

| Check | Value |
|---|---:|
| official_raw_acc vectorized 84D feature vs `pl_input_feature` max abs diff | `7.6293945e-06` |
| non-acceleration 66D block max abs diff across variants | `0` |
| AccCurve v1/v2 prediction shape | `[T,6,3]` for every sequence |
| sequences / frames / valid frames | `19 / 57994 / 57994` |

Results:

| Variant | Acc source | Target used by AccCurve | PL pRB L2 cm | PL pRB RMSE cm | PL gR1 deg | valid frames |
|---|---|---|---:|---:|---:|---:|
| official_raw_acc | raw aM | none | `6.529110` | `4.638030` | `15.267153` | `57994` |
| smooth_acc | smooth(aM) | none | `6.462386` | `4.589704` | `15.216247` | `57994` |
| acc_curve_v1_pred | AccCurve v1 pred | smooth(diff_acc(p_WS)) | `6.967961` | `4.866400` | `15.036875` | `57994` |
| acc_curve_v2_gtfk_pred | AccCurve v2 pred | smooth(GTFKacc(q,qdot,qddot,rJS)) | `8.347050` | `5.958994` | `15.229429` | `57994` |

Conclusion:

```text
smooth_acc improves both frozen-PL pRB and gR1 versus official_raw_acc.
acc_curve_v1_pred improves gR1 but regresses pRB.
acc_curve_v2_gtfk_pred slightly improves gR1 but strongly regresses pRB.
Thus AccCurve acceleration-level improvement did not transfer into a simultaneous
PL pRB+gR1 output improvement. Do not connect AccCurve v1/v2 into PL as-is.
This is module-level PL input evaluation only, not a full-pipeline S4 claim.
```

Artifacts:

```text
summary: data/experiments/acc_curve_pl_input_eval_20260617/summary.md
result JSON: data/experiments/acc_curve_pl_input_eval_20260617/result_summary.json
overall CSV: data/experiments/acc_curve_pl_input_eval_20260617/dip_test_pl_input_eval.csv
per-sequence CSV: data/experiments/acc_curve_pl_input_eval_20260617/per_sequence_metrics.csv
debug JSON: data/experiments/acc_curve_pl_input_eval_20260617/debug_first_sequence_acceleration_blocks.json
run log: data/experiments/acc_curve_pl_input_eval_20260617/logs/run.log
exact command: data/experiments/acc_curve_pl_input_eval_20260617/exact_command.txt
```

## EXP-20260617-acc_curve_v2_gtfk - Strict GTFK AccCurve residual module

Question: Can the standalone PL-style AccCurve residual module improve six IMU sensor-site acceleration against a strict `smooth(GTFKacc(q,qdot,qddot,rJS))` target?

Target correction: v1 used `aFK_smooth` from a position finite-difference target cache (`smooth(diff_acc(p_WS))` style). v2 uses an explicit `aFK_gtfk_smooth` cache built from RBDL point acceleration:

```text
GTFKacc(q,qdot,qddot,rJS) -> centered smooth window=9 -> aFK_gtfk_smooth[6,3]
```

Change tested:

```text
module: acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617
input: aM_raw[18] + aM_smooth[18] + (aM_raw-aM_smooth)[18] + wM[18] + RMB_6d[36] = 108D
RMB_6d: rotation[..., :, :2].transpose(-1, -2).reshape(..., 6)
input frame: model/world cache frame for aM_raw/aM_smooth/wM/RMB; no root-frame transform
base: aM_smooth[18]
output: pred_aM_curve[18]
target: aFK_gtfk_smooth[18]
unit: output and target remain m/s^2
normalization: train-set z-score on input features only, fitted from AMASS train split only
```

Cache build:

```text
cache root: code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617
AMASS train: 1298 sequences, 1118012 frames, 1107628 valid frames, failures=0
DIP train: 36 sequences, 228807 frames, 228519 valid frames, failures=0
DIP val: 6 sequences, 30771 frames, 30723 valid frames, failures=0
DIP test: 19 sequences, 57994 frames, 57842 valid frames, failures=0
target source: GTFK(q,qdot,qddot,rJS)
target contract: GTFKacc(q,qdot,qddot,rJS) -> centered smooth -> aFK_gtfk_smooth[6,3]
```

Commands:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
ENV=/home/lingfeng/.conda/envs/globalpose-gpu
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

# Smoke:
for preset in amass_train dip_train dip_val dip_test; do
  "$ENV/bin/python" scripts/build_acc_curve_gtfk_cache.py --preset "$preset" --output-root code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617_smoke --max-sequences 2 --max-frames 180 --shard-size 2 --progress-every 1 --overwrite
done
"$ENV/bin/python" acc_curve_train.py --amass-cache code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617_smoke/amass_train/acc_curve_gtfk_cache_manifest.json --dip-train-cache code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617_smoke/dip_train/acc_curve_gtfk_cache_manifest.json --dip-val-cache code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617_smoke/dip_val/acc_curve_gtfk_cache_manifest.json --dip-test-cache code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617_smoke/dip_test/acc_curve_gtfk_cache_manifest.json --output-dir data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617_smoke --target-key aFK_gtfk_smooth --epochs 1 --dip-epochs 1 --window 120 --stride 60 --batch-size 2 --num-workers 0 --hidden-size 64 --overwrite

# Full cache + training:
CUDA_VISIBLE_DEVICES=0 RUN_SMOKE=0 RUN_FULL_CACHE=1 RUN_TRAIN=1 USE_LONGRUN=1 bash scripts/run_acc_curve_v2_gtfk_20260617.sh
```

Validation:

```text
compile: python -m py_compile acc_curve.py acc_curve_train.py scripts/build_acc_curve_gtfk_cache.py
runner syntax: bash -n scripts/run_acc_curve_v2_gtfk_20260617.sh
smoke: passed; zero_init_max_abs_pred_minus_base=0.0
window check: AMASS smoke train_windows=2, DIP smoke train_windows=4; full AMASS train_windows=8231, DIP train_windows=1887
target safety: acc_curve_train.py only allows target_key=aFK_gtfk_smooth and validates manifest target_source=GTFK(q,qdot,qddot,rJS)
```

Training summary:

| Stage | Train seq | Val seq | Train windows | Val windows | Best epoch | Best selection |
|---|---:|---:|---:|---:|---:|---:|
| AMASS pretrain | `1231` | `67` | `8231` | `407` | `29` | `0.8150924359` |
| DIP finetune | `36` | `6` | `1887` | `253` | `19` | `0.7294550687` |

Module metrics:

| Split | pred L2 | base L2 | pred/base ratio | pred RMSE | base RMSE | corr | cosine | mag MAE | residual std | residual p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP val | `1.990773` | `3.102919` | `0.757471` | `1.689920` | `2.535429` | `0.821207` | `0.488962` | `1.060400` | `1.950429` | `6.949836` |
| DIP test | `2.997944` | `3.958857` | `0.778348` | `2.588015` | `3.341077` | `0.792049` | `0.493421` | `1.639596` | `2.357390` | `8.619020` |

Artifacts:

```text
module: acc_curve.py
train/eval: acc_curve_train.py
strict cache builder: scripts/build_acc_curve_gtfk_cache.py
runner: scripts/run_acc_curve_v2_gtfk_20260617.sh
cache root: code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617
experiment root: data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617
final checkpoint: data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/dip_finetune/best_loss.pt
summary: data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/train_result.json
eval JSONs:
  data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/eval/dip_val_eval.json
  data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/eval/dip_test_eval.json
sanity JSONs:
  code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/*/sanity_gtfk_target.json
  data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/eval/*_sanity_gtfk_prediction.json
```

Interpretation: The strict GTFK v2 module improves acceleration-level regression over the `aM_smooth` base on DIP val/test (`ratio < 1`). It does not satisfy the stronger effectiveness gate (`ratio < 0.7` and `corr > 0.9`), so it should be treated as a useful but not decisive acceleration alignment auxiliary. It is standalone and not connected to PL/IK/full pipeline; no motion-quality improvement is claimed.

## EXP-20260617-acc_curve_v1 - PL-style AccCurve residual module

Question: Can a standalone PLCurve-style residual network improve absolute sensor-site acceleration against the centered-smoothed acceleration base?

Hypothesis: `aM_raw`, `aM_smooth`, acceleration residual, gyro, and IMU orientation can learn the FK sensor-site acceleration residual while preserving a zero-initialized base behavior.

Change tested:

```text
module: acc_curve_v1_20260617
input: aM_raw[18] + aM_smooth[18] + (aM_raw-aM_smooth)[18] + wM[18] + RMB_6d[36] = 108D
base: aM_smooth[18]
output: pred_aM_curve[18]
target: aFK_smooth[18]
frame: model/world frame M
```

Dataset/split:

```text
cache root: code/outputs/smooth_acc_cache_amass_dip_20260617
AMASS train cache: 1298 sequences, 1118012 frames, 1105032 valid frames
DIP train cache: 36 sequences, 228807 frames, 228447 valid frames
DIP val cache: 6 sequences, 30771 frames, 30711 valid frames
DIP test cache: 19 sequences, 57994 frames, 57804 valid frames
AMASS training split: hash 95/5 train/val
DIP checkpoint selection: DIP val only
DIP test: final module-level evaluation only
```

Commands:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
ENV=/home/lingfeng/.conda/envs/globalpose-gpu
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

# Smoke cache and training:
"$ENV/bin/python" scripts/build_acc_curve_cache.py --preset amass_train --output-root code/outputs/acc_curve_smoke_cache_20260617 --max-sequences 2 --max-frames 180 --fk-batch-size 256 --overwrite
"$ENV/bin/python" scripts/build_acc_curve_cache.py --preset dip_train --output-root code/outputs/acc_curve_smoke_cache_20260617 --max-sequences 2 --max-frames 180 --fk-batch-size 256 --overwrite
"$ENV/bin/python" scripts/build_acc_curve_cache.py --preset dip_val --output-root code/outputs/acc_curve_smoke_cache_20260617 --max-sequences 2 --max-frames 180 --fk-batch-size 256 --overwrite
"$ENV/bin/python" scripts/build_acc_curve_cache.py --preset dip_test --output-root code/outputs/acc_curve_smoke_cache_20260617 --max-sequences 2 --max-frames 180 --fk-batch-size 256 --overwrite
"$ENV/bin/python" acc_curve_train.py --amass-cache code/outputs/acc_curve_smoke_cache_20260617/amass_train/acc_curve_cache_manifest.json --dip-train-cache code/outputs/acc_curve_smoke_cache_20260617/dip_train/acc_curve_cache_manifest.json --dip-val-cache code/outputs/acc_curve_smoke_cache_20260617/dip_val/acc_curve_cache_manifest.json --dip-test-cache code/outputs/acc_curve_smoke_cache_20260617/dip_test/acc_curve_cache_manifest.json --output-dir data/experiments/acc_curve_v1_20260617_smoke --epochs 1 --dip-epochs 1 --window 120 --stride 60 --batch-size 2 --num-workers 0 --hidden-size 64 --overwrite

# Full run resume:
CUDA_VISIBLE_DEVICES=0 /home/lingfeng/bin/longrun -- "$ENV/bin/python" acc_curve_train.py --output-dir data/experiments/acc_curve_v1_20260617 --epochs 30 --dip-epochs 20 --window 240 --stride 120 --batch-size 64 --num-workers 8 --hidden-size 512 --resume
```

Validation:

```text
compile: python -m py_compile acc_curve.py acc_curve_train.py scripts/build_acc_curve_cache.py
smoke: passed; zero_init_max_abs_pred_minus_base=0.0
window check: AMASS smoke train_windows=2, DIP smoke train_windows=4; full AMASS train_windows=8231, DIP train_windows=1887
normalization: feature z-score fitted from AMASS train split only and reused for DIP
```

Metrics:

| Stage | Best epoch | Selection |
|---|---:|---:|
| AMASS pretrain | `24` | `0.9526165639` |
| DIP finetune | `20` | `0.5814286023` |

| Split | pred L2 | base L2 | pred/base ratio | pred RMSE | base RMSE | corr | residual std | residual p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP val | `0.846367` | `1.871923` | `0.655075` | `0.628144` | `1.324018` | `0.957387` | `1.204956` | `3.743450` |
| DIP test | `1.202067` | `2.368697` | `0.622049` | `0.930242` | `1.733464` | `0.940837` | `1.450683` | `4.660124` |

Artifacts:

```text
module: acc_curve.py
train/eval: acc_curve_train.py
cache builder: scripts/build_acc_curve_cache.py
runner: scripts/run_acc_curve_v1_20260617.sh
experiment root: data/experiments/acc_curve_v1_20260617
final checkpoint: data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt
summary: data/experiments/acc_curve_v1_20260617/train_result.json
eval JSONs:
  data/experiments/acc_curve_v1_20260617/eval/dip_val_eval.json
  data/experiments/acc_curve_v1_20260617/eval/dip_test_eval.json
longrun log: data/experiments/acc_curve_v1_20260617/logs/resume_longrun_20260617_205531.log
```

Interpretation: The module improves same-cache acceleration target regression over the `aM_smooth` base on DIP val/test. This supports the standalone acceleration residual hypothesis, not a downstream motion-quality claim.

Claim support: validation result

Problems: Full cache and experiment checkpoints are large and ignored by git. Commit only code, scripts, docs, and compact result summaries if needed.

Next action: keep AccCurve separate unless a later experiment explicitly defines how this acceleration output is consumed by PL/IK/physics.

## EXP-20260612-newpl_v5_butteracc — NewPL v5 with realtime causal ButterAcc input

Question: Can NewPL v5 use realtime causal acceleration smoothing instead of raw official `aM` or offline centered smooth-acc, while preserving module-level `pRB/gR1` quality on DIP and TotalCapture?

Implementation:

```text
filter code: l4_sensor_offset_utils.py::causal_butterworth_lowpass_sequence
cache builder: scripts/build_smooth_acc_cache.py --mode causal_butterworth
runner: scripts/run_newpl_v5_butteracc_20260612.sh
summary script: scripts/summarize_newpl_v5_butteracc.py
```

Input/output contract:

```text
PL input remains 84D: aRB[18] + wRB[18] + RRB[45] + gR0[3].
Only source `aM` changes: causal Butterworth low-pass filter, order=2, fs=60 Hz.
Cutoff sweep: 8 Hz, 10 Hz, 12 Hz.
Realtime contract: zero lookahead, output[t] uses samples <= t, latency_ms=0.
`aM_raw` is preserved for audit.
`wM`, `RMB`, targets, offsets, and output contract are unchanged.
PL output remains pRB[15] + gR1[3].
DIP trans/root velocity/global trajectory GT is not used.
Root velocity is not part of this PL v5 input-gate experiment.
Full-pipeline 11 metrics are not run.
```

Commands:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
CUDA_VISIBLE_DEVICES=1 SMOKE=1 bash scripts/run_newpl_v5_butteracc_20260612.sh
CUDA_VISIBLE_DEVICES=1 ROOT=data/experiments/newpl_v5_butteracc_20260612_full CACHE_ROOT=data/experiments/newpl_v5_butteracc_20260612_full/caches RUN_SMOKE=0 /home/lingfeng/bin/longrun -- bash scripts/run_newpl_v5_butteracc_20260612.sh
```

Validation before full run:

```text
python -m py_compile l4_sensor_offset_utils.py scripts/build_smooth_acc_cache.py scripts/summarize_newpl_v5_butteracc.py
bash -n scripts/run_newpl_v5_butteracc_20260612.sh
causal filter smoke: random [120,6,3], finite=true, first output equals first input.
```

Artifacts:

```text
smoke root: data/experiments/newpl_v5_butteracc_20260612_smoke
full root: data/experiments/newpl_v5_butteracc_20260612_full
forced fc12 longtrain root: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain
full log: data/experiments/newpl_v5_butteracc_20260612_full/logs/run.log
selection json: data/experiments/newpl_v5_butteracc_20260612_full/selection.json
summary json: data/experiments/newpl_v5_butteracc_20260612_full/summary.json
summary md: data/experiments/newpl_v5_butteracc_20260612_full/summary.md
eval JSONs: data/experiments/newpl_v5_butteracc_20260612_full/eval/
```

Cache/filter statistics:

```text
DIP test fc8: raw_filtered_rms_delta=2.168163, second_difference_rms_ratio=0.274898.
TC test fc8: raw_filtered_rms_delta=3.223462, second_difference_rms_ratio=0.187012.
DIP test fc10: raw_filtered_rms_delta=1.879955, second_difference_rms_ratio=0.354281.
TC test fc10: raw_filtered_rms_delta=3.046652, second_difference_rms_ratio=0.262172.
DIP test fc12: raw_filtered_rms_delta=1.653283, second_difference_rms_ratio=0.432989.
TC test fc12: raw_filtered_rms_delta=2.873303, second_difference_rms_ratio=0.340913.
```

Selection result:

```text
selection_status: no_candidate
selected_cutoff_hz: none
raw_tc_official_pRB_L2_cm: 6.995536
tc_margin_cm: 0.10
reason: no cutoff passed the TotalCapture pRB guard
AMASS pretrain checkpoint: not available
DIP fine-tune checkpoint: not available
```

Input-only pRB/gR1 comparison:

| Dataset / Cutoff | Version | pRB L1 cm | pRB L2 cm | gR1 deg |
|---|---|---:|---:|---:|
| DIP fc8 | official_PL_butter | `3.351340` | `6.907898` | `12.889588` |
| DIP fc8 | newpl_v4_init36_butter | `3.357099` | `6.934424` | `12.711126` |
| DIP fc8 | newpl_v5_raw_dip_butter | `3.366243` | `6.950735` | `12.503217` |
| TC fc8 | official_PL_butter | `3.603690` | `7.506753` | `13.217498` |
| TC fc8 | newpl_v4_init36_butter | `3.436870` | `7.147257` | `13.118172` |
| TC fc8 | newpl_v5_raw_dip_butter | `3.494429` | `7.287691` | `13.173159` |
| DIP fc10 | official_PL_butter | `3.290294` | `6.780699` | `12.893975` |
| DIP fc10 | newpl_v4_init36_butter | `3.295261` | `6.806182` | `12.714799` |
| DIP fc10 | newpl_v5_raw_dip_butter | `3.303543` | `6.820206` | `12.506282` |
| TC fc10 | official_PL_butter | `3.557797` | `7.404335` | `13.211574` |
| TC fc10 | newpl_v4_init36_butter | `3.386474` | `7.036701` | `13.113819` |
| TC fc10 | newpl_v5_raw_dip_butter | `3.443299` | `7.172809` | `13.173770` |
| DIP fc12 | official_PL_butter | `3.251481` | `6.699946` | `12.900018` |
| DIP fc12 | newpl_v4_init36_butter | `3.255420` | `6.724124` | `12.720327` |
| DIP fc12 | newpl_v5_raw_dip_butter | `3.263422` | `6.736792` | `12.510739` |
| TC fc12 | official_PL_butter | `3.496175` | `7.267198` | `13.254831` |
| TC fc12 | newpl_v4_init36_butter | `3.325723` | `6.903734` | `13.152519` |
| TC fc12 | newpl_v5_raw_dip_butter | `3.381516` | `7.034841` | `13.218887` |

Gate details:

```text
fc8: TC official ButterAcc pRB L2 exceeds raw official by +0.511217 cm.
fc10: TC official ButterAcc pRB L2 exceeds raw official by +0.408799 cm.
fc12: TC official ButterAcc pRB L2 exceeds raw official by +0.271662 cm.
Best ButterAcc cutoff by pRB is fc12, but it still worsens DIP official pRB by +0.280473 cm and TC official pRB by +0.271662 cm versus raw official_PL.
```

Per-leaf pRB L2 at fc12:

| Dataset | Version | leaf_1 | leaf_2 | leaf_3 | leaf_4 | leaf_5 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP fc12 | official_PL_butter | `9.087043` | `9.425798` | `4.402759` | `5.353347` | `5.230786` | `6.699947` |
| DIP fc12 | newpl_v4_init36_butter | `9.153128` | `9.497437` | `4.230015` | `5.426576` | `5.313463` | `6.724124` |
| DIP fc12 | newpl_v5_raw_dip_butter | `9.092113` | `9.579000` | `4.309003` | `5.394506` | `5.309338` | `6.736792` |
| TC fc12 | official_PL_butter | `6.707833` | `6.984410` | `6.754984` | `6.915042` | `8.973717` | `7.267197` |
| TC fc12 | newpl_v4_init36_butter | `6.782565` | `7.052000` | `6.184772` | `6.385552` | `8.113783` | `6.903734` |
| TC fc12 | newpl_v5_raw_dip_butter | `6.756134` | `7.184214` | `6.587662` | `6.444391` | `8.201804` | `7.034841` |

Forced fc12 longtrain command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
CUDA_VISIBLE_DEVICES=1 ROOT=data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain CACHE_ROOT=data/experiments/newpl_v5_butteracc_20260612_full/caches CUTOFFS=12 FORCE_CUTOFF_HZ=12 RUN_SMOKE=0 AMASS_BATCH=512 DIP_BATCH=32 /home/lingfeng/bin/longrun -- bash scripts/run_newpl_v5_butteracc_20260612.sh
```

Forced fc12 training artifacts:

```text
AMASS best/last: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/amass_pretrain/best_loss.pt, last.pt
DIP best/last: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/dip_finetune/best_loss.pt, last.pt
log: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/logs/run.log
summary json: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/summary.json
summary md: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/summary.md
eval JSONs: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/eval/
```

Forced fc12 training result:

| Stage | Batch | LR | Epochs | Best epoch | Selection metric | Best value |
|---|---:|---:|---:|---:|---|---:|
| AMASS pretrain | `512` | `1e-4` | stopped at `15/80` | `3` | `control_physical` | `0.0024848974486531006` |
| DIP fine-tune | `32` | `5e-6` | `40/40` | `38` | `control_physical` | `0.057614331105772486` |

Forced fc12 module eval:

| Dataset / Stage | Version | pRB L1 cm | pRB L2 cm | gR1 deg |
|---|---|---:|---:|---:|
| AMASS after AMASS | official_PL_butter_fc12 | `1.806870` | `3.760248` | `4.859218` |
| AMASS after AMASS | newpl_v5_butteracc_amass_best | `1.816958` | `3.782071` | `4.859811` |
| DIP after AMASS | official_PL_butter_fc12 | `3.251481` | `6.699946` | `12.900018` |
| DIP after AMASS | newpl_v5_butteracc_amass_best | `3.262136` | `6.722528` | `12.894359` |
| TC after AMASS | official_PL_butter_fc12 | `3.496175` | `7.267198` | `13.254831` |
| TC after AMASS | newpl_v5_butteracc_amass_best | `3.498118` | `7.271788` | `13.251097` |
| DIP after DIP FT | official_PL_butter_fc12 | `3.251481` | `6.699946` | `12.900018` |
| DIP after DIP FT | newpl_v5_butteracc_dip_best | `3.261222` | `6.721462` | `12.896323` |
| TC after DIP FT | official_PL_butter_fc12 | `3.496175` | `7.267198` | `13.254831` |
| TC after DIP FT | newpl_v5_butteracc_dip_best | `3.495333` | `7.266006` | `13.257684` |

Final comparison against raw baselines:

```text
DIP raw official_PL: pRB L2=6.419473 cm, gR1=12.947709 deg.
DIP raw newpl_v5_dip_best: pRB L2=6.445578 cm, gR1=12.552613 deg.
DIP forced ButterAcc dip_best: pRB L2=6.721462 cm, gR1=12.896323 deg.

TC raw official_PL: pRB L2=6.995536 cm, gR1=13.450465 deg.
TC raw newpl_v5_dip_best: pRB L2=6.780749 cm, gR1=13.415189 deg.
TC forced ButterAcc dip_best: pRB L2=7.266006 cm, gR1=13.257684 deg.
```

Conclusion: not selected. Causal ButterAcc is a valid realtime filter and substantially reduces acceleration jitter, but it shifts PL outputs enough that all tested cutoffs fail the TC pRB gate and worsen DIP pRB versus raw official input. The requested forced fc12 longtrain confirms the negative result: AMASS pretrain and DIP fine-tune do not recover pRB, and the trained checkpoint remains worse than raw official/raw v5 on both DIP and TotalCapture pRB.

## EXP-20260612-newpl_v5_smoothacc — NewPL v5 with smoothed acceleration input

Question: Does replacing official raw `aM` with centered smoothed acceleration improve NewPL v5 module-level `pRB/gR1` outputs under the same AMASS -> DIP route?

Implementation:

```text
cache builder: scripts/build_smooth_acc_cache.py
runner: scripts/run_newpl_v5_smoothacc_20260612.sh
summary script: scripts/summarize_newpl_v5_smoothacc.py
```

Input/output contract:

```text
PL input remains 84D: aRB[18] + wRB[18] + RRB[45] + gR0[3].
Only source `aM` changes: centered moving-average smoothing, window=9.
`aM_raw` is preserved for audit.
`wM`, `RMB`, targets, offsets, and output contract are unchanged.
PL output remains pRB[15] + gR1[3].
DIP trans/root velocity/global trajectory GT is not used.
Full-pipeline 11 metrics are not run.
```

Commands:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
CUDA_VISIBLE_DEVICES=1 SMOKE=1 bash scripts/run_newpl_v5_smoothacc_20260612.sh
CUDA_VISIBLE_DEVICES=1 ROOT=data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256 CACHE_ROOT=data/experiments/newpl_v5_smoothacc_20260612/caches RUN_SMOKE=0 /home/lingfeng/bin/longrun -- bash scripts/run_newpl_v5_smoothacc_20260612.sh
```

Important run notes:

```text
Initial full run with AMASS_BATCH=512 was interrupted after epoch 1 because full-sequence validation and full AMASS eval were too slow.
The runner was updated to use training validation windows of 61 frames and AMASS module eval max 20 sequences.
Final full run uses AMASS_BATCH=256 and DIP_BATCH=24.
```

Artifacts:

```text
smoke root: data/experiments/newpl_v5_smoothacc_20260612_smoke
partial interrupted root: data/experiments/newpl_v5_smoothacc_20260612
final root: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256
shared smooth cache root: data/experiments/newpl_v5_smoothacc_20260612/caches
AMASS best: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/amass_pretrain/best_loss.pt
AMASS last: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/amass_pretrain/last.pt
DIP best: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/dip_finetune/best_loss.pt
DIP last: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/dip_finetune/last.pt
log: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/logs/run.log
summary json: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/summary.json
summary md: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/summary.md
```

Training result:

| Stage | Batch | LR | Epochs | Best epoch | Selection metric | Best value |
|---|---:|---:|---:|---:|---|---:|
| AMASS pretrain | `256` | `1e-4` | stopped at `15/80` | `3` | `control_physical` | `0.0025220187869422262` |
| DIP fine-tune | `24` | `5e-6` | `40/40` | `39` | `control_physical` | `0.057518671217621886` |

Eval JSONs:

```text
data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/eval/amass_after_amass_pretrain_smoothinput.json
data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/eval/dip_test_after_amass_pretrain_smoothinput.json
data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/eval/tc_test_after_amass_pretrain_smoothinput.json
data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/eval/dip_test_after_dip_finetune_smoothinput.json
data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/eval/tc_test_after_dip_finetune_smoothinput.json
```

Key results:

```text
DIP test after AMASS pretrain:
  smooth official_PL: pRB L2=6.345701 cm, gR1=12.902131 deg.
  newpl_v5_smoothacc_amass_best: pRB L2=6.354993 cm, gR1=12.891370 deg.

TotalCapture test after AMASS pretrain:
  smooth official_PL: pRB L2=7.508985 cm, gR1=13.170892 deg.
  newpl_v5_smoothacc_amass_best: pRB L2=7.484481 cm, gR1=13.174086 deg.

DIP test after DIP fine-tune:
  smooth official_PL: pRB L2=6.345701 cm, gR1=12.902131 deg.
  newpl_v4_init36_smoothacc: pRB L2=6.349507 cm, gR1=12.722395 deg.
  raw newpl_v5_dip_best: pRB L2=6.445578 cm, gR1=12.552613 deg.
  newpl_v5_smoothacc_dip_best: pRB L2=6.350327 cm, gR1=12.894731 deg.

TotalCapture test after DIP fine-tune:
  smooth official_PL: pRB L2=7.508985 cm, gR1=13.170880 deg.
  newpl_v4_init36_smoothacc: pRB L2=7.119541 cm, gR1=13.075063 deg.
  raw newpl_v5_dip_best: pRB L2=6.780749 cm, gR1=13.415189 deg.
  newpl_v5_smoothacc_dip_best: pRB L2=7.473741 cm, gR1=13.185523 deg.
```

Per-leaf pRB L2 after DIP fine-tune:

| Dataset | Version | leaf_1 | leaf_2 | leaf_3 | leaf_4 | leaf_5 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP test | official_PL_smoothacc | `8.350630` | `8.765077` | `4.303051` | `5.137257` | `5.172492` | `6.345701` |
| DIP test | newpl_v5_smoothacc_dip_best | `8.372464` | `8.792264` | `4.272254` | `5.150879` | `5.163776` | `6.350327` |
| TotalCapture test | official_PL_smoothacc | `6.812538` | `7.092184` | `7.168602` | `6.805456` | `9.666146` | `7.508985` |
| TotalCapture test | newpl_v5_smoothacc_dip_best | `6.816790` | `7.088333` | `7.113846` | `6.760619` | `9.589119` | `7.473741` |

Conclusion: not selected. Smoothed acceleration input helps DIP pRB slightly and improves gR1, but it damages TotalCapture pRB. The retrained smooth-acc v5 does not beat the smoothed official PL on DIP pRB and is worse than `newpl_v4_init36_smoothacc` on TotalCapture. Keep this as diagnostic evidence only; do not connect it to IK1/full pipeline.

## EXP-20260609-smoothed_rjs_acc_controlcurve_audit — Does smoothed-footlock rJS explain smoothed aM?

Question: Does the `r_JS` recomputed with smoothed acceleration in `footlock_transpose_v1` better explain smoothed IMU acceleration when the motion state uses control-curve `q/qdot/qddot`?

Implemented script:

```text
scripts/audit_smoothed_rjs_acc_controlcurve.py
```

Frame and validation contract:

```text
r_JS is SMPL joint-local: p_WS = p_WJ + R_WJ @ r_JS.
Primary metric is SMPL-contract root-relative lower-body acceleration residual against aM_smooth_w9.
DIP trans is not used; DIP root translation is zeroed for this audit.
RBDL q/qdot/qddot uses derivative-aware UniformCubicBSpline control decode.
Direct rJS injection into RBDL is not approved because the RBDL IMU link frame is not proven equivalent to the SMPL mapped-joint local frame.
```

Commands:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/audit_smoothed_rjs_acc_controlcurve.py --preset dip_test --max-sequences 1 --max-frames 600 --output-json data/experiments/smoothed_rjs_acc_controlcurve_audit_20260609/smoke_dip_test.json --output-md data/experiments/smoothed_rjs_acc_controlcurve_audit_20260609/smoke_dip_test.md --device cpu
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/audit_smoothed_rjs_acc_controlcurve.py --preset totalcapture_test --max-sequences 1 --max-frames 600 --output-json data/experiments/smoothed_rjs_acc_controlcurve_audit_20260609/smoke_totalcapture_test.json --output-md data/experiments/smoothed_rjs_acc_controlcurve_audit_20260609/smoke_totalcapture_test.md --device cpu
/home/lingfeng/bin/longrun -- bash -lc 'set -euo pipefail; ROOT=data/experiments/smoothed_rjs_acc_controlcurve_audit_20260609; PY=/home/lingfeng/.conda/envs/globalpose-gpu/bin/python; for split in dip_train dip_val dip_test totalcapture_train totalcapture_val totalcapture_test; do "$PY" scripts/audit_smoothed_rjs_acc_controlcurve.py --preset "$split" --output-json "$ROOT/${split}.json" --output-md "$ROOT/${split}.md" --device cuda:1 --skip-rbdl; done'
```

Artifacts:

```text
root: data/experiments/smoothed_rjs_acc_controlcurve_audit_20260609
summary: data/experiments/smoothed_rjs_acc_controlcurve_audit_20260609/summary.md
summary json: data/experiments/smoothed_rjs_acc_controlcurve_audit_20260609/summary.json
```

Primary result, all-frame SMPL-contract root-relative lower-body acceleration L2 in `m/s^2`:

```text
DIP train: zero=3.479233, old_footlock=3.355208, smoothacc=3.524253, smooth-zero=+0.045020, smooth-old=+0.169045.
DIP val: zero=3.450705, old_footlock=3.284543, smoothacc=3.646945, smooth-zero=+0.196240, smooth-old=+0.362402.
DIP test: zero=4.250496, old_footlock=4.170551, smoothacc=4.230922, smooth-zero=-0.019574, smooth-old=+0.060371.
TotalCapture train: zero=5.628481, smoothacc=5.529490, smooth-zero=-0.098991.
TotalCapture val: zero=7.359807, smoothacc=6.957408, smooth-zero=-0.402398.
TotalCapture test: zero=5.659256, smoothacc=5.470786, smooth-zero=-0.188470.
```

RBDL q-control smoke:

```text
DIP test 1seq/600f: rootrel lower L2=0.690369, rootrel nonroot L2=0.904451, DIP trans not used.
TotalCapture test 1seq/600f: rootrel lower L2=3.001982, rootrel nonroot L2=3.141561.
```

Conclusion: smoothed-footlock `r_JS` is useful on TotalCapture by this acceleration-explainability diagnostic, but it is not a stable improvement on DIP. On DIP, it is worse than zero on train/val and only slightly better than zero on test; it is worse than the June 8 old footlock winner on all three DIP splits. Do not claim that smoothed acceleration refit improves DIP pseudo-`r_JS` yet.

## EXP-20260609-footlock_only_smoothacc_rjs — Keep only footlock_transpose_v1 for real-data rJS

Question: When recomputing `r_JS` from smoothed acceleration, should DIP/TotalCapture use the same June 8 foot-lock TransPose winner-foot method and delete all other offset routes?

Decision: yes. The only active real-data `r_JS / offset_r` synthesis route is now `footlock_transpose_v1`. It uses TransPose contact winner-foot stance windows, does not trust DIP `trans`, infers root motion by foot lock, then solves lever-arm offset. Smoothed acceleration is only a fit input inside this same method; it is not a separate generic LS/refit method.

Contract:

```text
r_JS: IMU origin relative to mapped joint J, expressed in joint-local coordinates.
p_WS(t) = p_WJ(t) + R_WJ(t) @ r_JS.
DIP r_JS is pseudo-label only; DIP trans is not trusted and not used as GT.
```

Implemented code changes:

```text
scripts/build_imu_position_offsets.py:
  --method now only accepts footlock_transpose_v1.
  zero/random/solver_v1/net_v2/hybrid_v3 active paths are removed.
  default smooth_window is 9 and derivative_mode is centered.

imu_position_offset.py:
  keeps TransPoseContactEstimator, winner-foot contact windows, and solve_footlock_transpose_offset.
  removes old generic lever-arm solver and OffsetNet inference helpers from active code.
  stores contact_input=raw_official_aM_RMB and fit_input=smoothed_aM_and_zero_translation_FK_window_9.

scripts/run_footlock_transpose_rjs_20260608.sh:
  default OUT_ROOT is data/experiments/footlock_transpose_rjs_smoothacc_20260609.
  builds DIP train/val/test and TotalCapture train/val/test with the same footlock method.

scripts/audit_smoothed_acc_offset_fit.py:
  deleted because it implemented a generic smoothed acceleration LS/refit route, not the selected footlock method.
```

Deleted retired artifacts:

```text
data/dataset_work/SensorOffset/full_diagnostic_v1
data/dataset_work/SensorOffset/rawlike_se3_candidate_a_v1
data/dataset_work/SensorOffset/totalcapture_s5_v2
data/experiments/imu_position_offset_newpl
data/experiments/imu_offset_net_20260607
data/experiments/smoothed_acc_offset_fit_20260609
data/experiments/footlock_transpose_rjs_20260608_independent
```

Smoke commands:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_imu_position_offsets.py --input data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json --output data/experiments/footlock_transpose_rjs_smoothacc_20260609_smoke/dip_test_1seq_300f.pt --summary-json data/experiments/footlock_transpose_rjs_smoothacc_20260609_smoke/dip_test_1seq_300f_summary.json --dataset dip --max-sequences 1 --max-frames 300 --device cpu
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_imu_position_offsets.py --input data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_test_official_neural_only_offset_r/baseline_cache_manifest.json --output data/experiments/footlock_transpose_rjs_smoothacc_20260609_smoke/totalcapture_test_1seq_300f.pt --summary-json data/experiments/footlock_transpose_rjs_smoothacc_20260609_smoke/totalcapture_test_1seq_300f_summary.json --dataset totalcapture --max-sequences 1 --max-frames 300 --device cpu
```

Smoke result:

```text
DIP test 1seq/300f:
  all_finite=true
  median offset norm=0.179266 m
  mean/p95 offset norm=0.168215 / 0.344804 m
  contact_input=raw_official_aM_RMB
  fit_input=smoothed_aM_and_zero_translation_FK_window_9
  contact_selection_mode=transpose_winner
  contact_window_count=3

TotalCapture test 1seq/300f:
  all_finite=true
  median offset norm=0.199703 m
  mean/p95 offset norm=0.203427 / 0.286061 m
  contact_input=raw_official_aM_RMB
  fit_input=smoothed_aM_and_zero_translation_FK_window_9
  contact_selection_mode=transpose_winner
  contact_window_count=5
```

Full cache command started:

```bash
/home/lingfeng/bin/longrun -- bash -lc 'set -euo pipefail; DEVICE=cuda:1 RUN_TOTALCAPTURE=1 OUT_ROOT=data/experiments/footlock_transpose_rjs_smoothacc_20260609 scripts/run_footlock_transpose_rjs_20260608.sh'
```

Expected full outputs:

```text
data/experiments/footlock_transpose_rjs_smoothacc_20260609/dip_train_footlock_transpose_rjs.pt
data/experiments/footlock_transpose_rjs_smoothacc_20260609/dip_val_footlock_transpose_rjs.pt
data/experiments/footlock_transpose_rjs_smoothacc_20260609/dip_test_footlock_transpose_rjs.pt
data/experiments/footlock_transpose_rjs_smoothacc_20260609/totalcapture_train_footlock_transpose_rjs.pt
data/experiments/footlock_transpose_rjs_smoothacc_20260609/totalcapture_val_footlock_transpose_rjs.pt
data/experiments/footlock_transpose_rjs_smoothacc_20260609/totalcapture_test_footlock_transpose_rjs.pt
```

Full cache result:

```text
DIP train: sequences=36, all_finite=true, offset norm mean/median/p95=0.182025/0.175146/0.333083 m, contact windows=3008, fallback sensors=9.
DIP val: sequences=6, all_finite=true, offset norm mean/median/p95=0.197334/0.185467/0.418209 m, contact windows=347, fallback sensors=1.
DIP test: sequences=19, all_finite=true, offset norm mean/median/p95=0.175151/0.169691/0.297217 m, contact windows=738, fallback sensors=2.
TotalCapture train: sequences=36, all_finite=true, offset norm mean/median/p95=0.164509/0.157523/0.307779 m, contact windows=1961, fallback sensors=0.
TotalCapture val: sequences=5, all_finite=true, offset norm mean/median/p95=0.149411/0.139848/0.269620 m, contact windows=202, fallback sensors=0.
TotalCapture test: sequences=4, all_finite=true, offset norm mean/median/p95=0.182435/0.199503/0.287975 m, contact windows=203, fallback sensors=0.
```

All full-cache rows report:

```text
contact_input=raw_official_aM_RMB
fit_input=smoothed_aM_and_zero_translation_FK_window_9
contact_selection_mode=transpose_winner
```

## EXP-20260608-derivative_aware_control_fit — Project-wide GT control-point synthesis update

Question: Can GT control points be synthesized so that the decoded curve fits both the physical state and its finite-difference first/second derivatives, instead of exactly reconstructing the state samples only?

Decision: Adopt `derivative_aware_v1` as the default control-point target synthesis method for new training and cache generation.

Contract:

```text
fit_uniform_cubic_spline_controls(x):
  min_C wp||S C - x||^2
      + wv||D1 C - fd_dot(x)||^2
      + wa||D2 C - fd_ddot(x)||^2
      + wr||C||^2

weights: wp=1.0, wv=0.03, wa=0.0003, wr=1e-6
dt: 1/60
fd_dot: central difference with one-sided endpoints
fd_ddot: three-point second difference with endpoint copy
```

Implemented code paths:

```text
default function: pl_curve.fit_uniform_cubic_spline_controls
legacy exact fit: pl_curve.fit_uniform_cubic_spline_controls_position_only
manifest contract: pl_curve.control_fit_contract
updated cache manifests: pl_curve_cache.py, newik1_control_cache.py, newpose_ctrl_cache.py
updated audits: pl_gt_control_audit.py, derivative_source_audit.py
RBDL-only audit script: scripts/audit_amass_rbdl_only_imu_acc_4way.py
```

RBDL-only evidence:

```text
JSON: data/experiments/gt_control_derivative_audit_20260608/amass_rbdl_only_imu_acc_derivfit.json
Markdown: data/experiments/gt_control_derivative_audit_20260608/amass_rbdl_only_imu_acc_derivfit.md
Dataset: 20 AMASS sequences from data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged/baseline_cache_manifest.json
Constraint: SMPL FK is not used for IMU acceleration synthesis; RBDL IMU links are the only position/velocity/acceleration source; cached aM is not used as GT.
```

Summary metrics:

| q source | position acc target | acc L2 mean ↓ | angle mean ↓ | vel L2 mean ↓ |
|---|---|---:|---:|---:|
| finite-diff qdot/qddot | RBDL position finite diff | `0.453582` | `1.495915` | `0.011516` |
| position-only control qdot/qddot | position-only control position acc | `1.538945` | `4.152926` | `0.011097` |
| derivative-aware qdot/qddot | derivative-aware control position acc | `0.326352` | `1.138201` | `0.009896` |
| derivative-aware qdot/qddot | RBDL position finite diff | `0.356589` | `1.641141` | `0.012599` |

Direct curve-vs-finite-difference diagnostics:

| Curve target | acc vs FD L2 mean ↓ | vel vs FD L2 mean ↓ | decoded position L2 mean |
|---|---:|---:|---:|
| position-only control | `5.647949` | `0.018917` | `~0` |
| derivative-aware control | `0.057588` | `0.004654` | `0.000686 m` |

Validation commands:

```bash
python -m py_compile pl_curve.py newpl_root.py newik1_control_point.py newik1_control_cache.py newpose_ctrl.py newpose_ctrl_cache.py pl_curve_cache.py pl_gt_control_audit.py derivative_source_audit.py scripts/audit_amass_rbdl_only_imu_acc_4way.py

ENV=/home/lingfeng/.conda/envs/globalpose-gpu
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
"$ENV/bin/python" pl_gt_control_audit.py --cache data/dataset_work/L4Cache/pl_curve_v2_processed_no_baseline_tc_val_Roffset_A/pl_curve_cache_manifest.json --max-sequences 1 --output-json data/experiments/gt_control_derivative_audit_20260608/smoke_pl_gt_control_derivative_aware.json
"$ENV/bin/python" pl_curve_cache.py --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/gt_control_derivative_audit_20260608/smoke_pl_cache_derivative_aware --imu-input-mode processed --feature-mode legacy --max-sequences 1 --shard-size 1
"$ENV/bin/python" newik1_control_cache.py --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/gt_control_derivative_audit_20260608/smoke_newik1_cache_derivative_aware --mode teacher_forced --imu-input-mode processed --feature-mode last_control --max-sequences 1 --shard-size 1
"$ENV/bin/python" newpose_ctrl_cache.py --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/gt_control_derivative_audit_20260608/smoke_newpose_cache_derivative_aware --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed --max-sequences 1 --shard-size 1
```

Smoke results: all commands returned `status=ok`. PL GT control audit on one TotalCapture validation sequence decoded derivative-aware controls with pRB error `0.153688 cm`, gR1 angle `0.056454 deg`, all finite.

Cache rebuild requirement: any future cache that stores GT control tails must be regenerated after this policy change. This includes NewIK1 control caches and NewPose control caches. PL caches that store only physical `pl_target` can remain as source data, but their manifest should be regenerated when creating new experiment roots so the control fit contract is explicit.

Canonical dataset-level GT-control cache implementation:

```text
builder: scripts/build_gt_control_cache.py
runner: scripts/run_build_gt_control_caches_20260608.sh
canonical output root: data/dataset_work/GTControlCache/
smoke output root: data/experiments/gt_control_derivative_audit_20260608/smoke_canonical/
full summary JSON: data/experiments/gt_control_derivative_audit_20260608/canonical_gt_control_cache_full_summary.json
full summary Markdown: data/experiments/gt_control_derivative_audit_20260608/canonical_gt_control_cache_full_summary.md
```

Contract:

```text
pose_rot6d_control: [T,24,6] local SMPL joint rotations in 6D.
joint_angle_euler_control: [T,72] unwrapped local Euler joint angles; no translation included.
joint_pos_R_control: [T,24,3] root/body-frame joint positions, p_RJ = (p_WJ - p_WR) @ R_WR.
imu_RMB_6d_control: [T,6,6] source-cache RMB orientation matrices encoded as 6D; not regenerated from SMPL.
pl_pRB_gR1_control: [T,18] PL pRB[15]+gR1[3] control state.
root_trans_W/root_vel_W_fd: emitted only for AMASS and TotalCapture when source tran_gt is reliable.
DIP-IMU root translation / root velocity GT: not available; not synthesized.
```

Implementation note: the canonical builder uses a scipy sparse factorization of the same derivative-aware normal equations. This avoids the dense `T x T` solve cost for long DIP sequences while preserving the objective. Double-precision equivalence check against `pl_curve.fit_uniform_cubic_spline_controls` on a 23-frame random tensor had max absolute difference `1.426e-06`.

Commands:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -m py_compile scripts/build_gt_control_cache.py
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_gt_control_cache.py --preset amass_train --output-root data/experiments/gt_control_derivative_audit_20260608/smoke_canonical --max-sequences 1 --shard-size 1 --overwrite
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_gt_control_cache.py --preset totalcapture_train --output-root data/experiments/gt_control_derivative_audit_20260608/smoke_canonical --max-sequences 1 --shard-size 1 --overwrite
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/build_gt_control_cache.py --preset dip_train --output-root data/experiments/gt_control_derivative_audit_20260608/smoke_canonical --max-sequences 1 --shard-size 1 --overwrite
CUDA_VISIBLE_DEVICES=0 /home/lingfeng/bin/longrun -- bash -lc 'set -o pipefail; scripts/run_build_gt_control_caches_20260608.sh --progress-every 25 2>&1 | tee data/experiments/gt_control_derivative_audit_20260608/logs/build_gt_control_caches_full.log'
```

Full-build note: the underlying builder completed all presets and all manifests validated. The outer `longrun` wrapper returned nonzero because the `tee` log path did not exist at launch, so the formal evidence is the validated manifests plus `canonical_gt_control_cache_full_summary.{json,md}`.

Smoke timing:

| Preset | Frames | Elapsed sec |
|---|---:|---:|
| `amass_train` | `1372` | `1.827` |
| `totalcapture_train` | `4113` | `1.622` |
| `dip_train` | `13778` | `2.821` |

Full canonical cache validation:

| Preset | Dataset | Split | Sequences | Frames | Shards | root trans | Required finite |
|---|---|---|---:|---:|---:|---|---|
| `amass_train` | AMASS | train | `1298` | `1118012` | `41` | available | true |
| `dip_train` | DIP-IMU | train | `36` | `228807` | `2` | not available | true |
| `dip_val` | DIP-IMU | val | `6` | `30771` | `1` | not available | true |
| `dip_test` | DIP-IMU | test | `19` | `57994` | `1` | not available | true |
| `totalcapture_train` | TotalCapture | train | `36` | `142902` | `2` | available | true |
| `totalcapture_val` | TotalCapture | val | `5` | `17223` | `1` | available | true |
| `totalcapture_test` | TotalCapture | test | `4` | `16124` | `1` | available | true |

## EXP-20260608-footlock_transpose_rjs — DIP pseudo-rJS from TransPose winner-foot foot lock

Question: Can DIP pseudo-`r_JS` be estimated without trusting DIP global `trans` by using short foot-lock windows selected from TransPose contact probabilities?

Decision: Use the `transpose_winner` version as the default pseudo-`r_JS` source for subsequent NewPL offset-aware cache generation and evaluation.

Coordinate and data contract:

```text
r_JS: IMU origin relative to mapped joint J, expressed in joint-local coordinates.
p_WS(t) = p_WJ(t) + R_WJ(t) @ r_JS.
DIP has no trusted global translation in this line; all DIP r_JS values are pseudo labels, not GT.
```

Method summary:

- Load frozen TransPose `pose_s1` and `tran_b1` from `/home/lingfeng/projects/TransPose/data/weights.pt`.
- Convert GlobalPose DIP fields to TransPose input as `cat(aM.reshape(T,18)/30, RMB.reshape(T,54))`.
- Select stance windows with `contact_selection_mode=transpose_winner`: each frame uses the foot with max TransPose contact probability, plus a near-lower-foot height sanity check inspired by PIP contact handling.
- For stance foot contact point `C`, use zero-translation SMPL FK and assume `p_WR(t) = -p_WC_zero_tran(t) + constant` over the window.
- Solve sequence-level lever-arm offsets from the resulting joint/root acceleration equations.
- Skip solving the stance-side lower-leg sensor from its own stance window.

Command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/bin/longrun -- bash -lc 'set -euo pipefail; OUT_ROOT=data/experiments/footlock_transpose_rjs_20260608 CONTACT_SELECTION_MODE=transpose_winner scripts/run_footlock_transpose_rjs_20260608.sh; OUT_ROOT=data/experiments/footlock_transpose_rjs_20260608_independent CONTACT_SELECTION_MODE=independent_threshold scripts/run_footlock_transpose_rjs_20260608.sh'
```

Artifacts:

```text
winner root: data/experiments/footlock_transpose_rjs_20260608
independent comparison root: data/experiments/footlock_transpose_rjs_20260608_independent
winner train cache: data/experiments/footlock_transpose_rjs_20260608/dip_train_footlock_transpose_rjs.pt
winner val cache: data/experiments/footlock_transpose_rjs_20260608/dip_val_footlock_transpose_rjs.pt
winner test cache: data/experiments/footlock_transpose_rjs_20260608/dip_test_footlock_transpose_rjs.pt
comparison md: data/experiments/footlock_transpose_rjs_20260608/compare_with_independent.md
comparison json: data/experiments/footlock_transpose_rjs_20260608/compare_with_independent.json
```

Winner split summaries:

| Split | Sequences | Offset norm mean | Offset norm median | Offset norm p95 | All finite |
|---|---:|---:|---:|---:|---|
| DIP train | 36 | `0.1775` | `0.1546` | `0.3302` | yes |
| DIP val | 6 | `0.1660` | `0.1641` | `0.2719` | yes |
| DIP test | 19 | `0.1676` | `0.1553` | `0.2809` | yes |

Winner versus old independent-threshold selection:

| Split | Offset delta mean | Offset delta median | Offset delta p95 | Offset delta max | Winner fallback sensors | Independent fallback sensors |
|---|---:|---:|---:|---:|---:|---:|
| DIP train | `0.0714` | `0.0482` | `0.2379` | `0.3517` | 4 | 6 |
| DIP val | `0.0681` | `0.0449` | `0.2398` | `0.4685` | 1 | 0 |
| DIP test | `0.0587` | `0.0320` | `0.1866` | `0.4265` | 3 | 3 |

Interpretation: The winner version changes the pseudo-offsets materially, with median sensor deltas around `3-5 cm` and high-percentile deltas around `18-24 cm`. This is expected because it avoids the earlier independent-threshold failure mode where both feet could be treated as simultaneously locked when TransPose predicts high contact probability for both. The winner version better matches TransPose translation fusion and PIP-style contact sanity, so it is the selected input for the next NewPL `r_JS` experiments.

## EXP-20260607-mainline-policy-update — official-baseline training route becomes the reference

Question: Should the current replacement mainline keep using the best TotalCapture/processed-input artifact, or switch to the official baseline training route before further module iteration?

Decision: Switch the mainline to the official-like route first, then iterate modules under that route.

Reference route:

```text
AMASS pretrain -> DIP-IMU train fine-tune -> DIP-IMU test + TotalCapture test
```

Reasoning: The prior `newpl_v4_init36` and TotalCapture-oriented runs can beat the official baseline on TotalCapture, but that does not prove the replacement has the same generalization behavior as the official baseline. The official-style baseline is trained on AMASS and adapted on DIP, then evaluated on both DIP and TotalCapture. Future NewPL/IK/VR claims should therefore be made under the same route unless explicitly labeled as TotalCapture-specialized.

Policy:

- Preserve each module's input/output contract unless a new version explicitly documents a compatible extension.
- For PL, select checkpoints by physical module outputs versus GT control points/gravity, not by arbitrary weighted loss alone.
- Do not use DIP translation/root-velocity supervision.
- Do not promote a module from TotalCapture fine-tune results alone.
- Compare against official baseline and the current best replacement under the same input/evaluation contract.

Status documents updated:

```text
PROJECT_STATUS.md
RECENT_REPLACEMENT_VERSIONS.md
```

## EXP-20260604-001 — Processed IMU official GPNet S4 baseline

Question: If frozen official GlobalPose consumes processed TotalCapture IMU orientation directly, does S4 improve versus official `aM/wM/RMB`?

Hypothesis: Corrected orientation `l4_RMB` improves neural features even when `aM/wM` are unchanged.

Change tested: Replace official `RMB` with processed orientation stream while keeping frozen official weights and official evaluation path.

Dataset/split: TotalCapture S4 validation, 5 sequences, 17223 frames.

Command/artifacts:

```text
data/experiments/official_gpnet_processed_imu_v1/s4_official_input_baseline.json
data/experiments/official_gpnet_processed_imu_v1/s4_processed_A_input.json
data/experiments/official_gpnet_processed_imu_v1/s4_comparison.json
data/experiments/official_gpnet_processed_imu_v1/s4_comparison.csv
```

Baseline result: official input score `42.522402`.

New result: processed-A input score `38.753660`.

Metrics: processed input improved all 11 reported S4 aggregate metrics.

Interpretation: The gain comes from corrected orientation, because processed data has `l4_aM == aM`, `l4_wM == wM`, and `l4_RMB != RMB`.

Claim support: validation result.

Problems: None recorded for this comparison.

Next action: Use processed orientation as the input convention for replacement experiments.

## EXP-20260604-002 — Consistent processed IMU v2 audit

Question: After correcting IMU orientation, should stored acceleration and gyro also change?

Hypothesis: Under the GlobalPose stored-field convention, correcting sensor-to-body orientation changes `RMB` but not stored `aM/wM`.

Change tested: Generate/audit a consistent processed IMU v2 where `l4_RMB = RIM^T @ RIS @ RSB_new` and `l4_aM/l4_wM` remain equal to official `aM/wM`.

Dataset/split: TotalCapture train+val cache generation; 41 sequences, 160125 frames.

Command/artifacts:

```text
l4_generate_consistent_processed_imu_v2.py
data/dataset_work/L4Cache/totalcapture_orientation_offset_consistent_v2/train_Roffset_A_consistent/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_consistent_v2/val_Roffset_A_consistent/baseline_cache_manifest.json
data/experiments/consistent_processed_imu_v2/imu_consistency_audit.json
data/experiments/consistent_processed_imu_v2/s4_eval/s4_official_input.json
data/experiments/consistent_processed_imu_v2/s4_eval/s4_processed_v1_rmb_only.json
data/experiments/consistent_processed_imu_v2/s4_eval/s4_processed_v2_consistent.json
```

Baseline formulas:

```text
RMB = RIM^T @ RIS @ RSB
aM  = RIM^T @ RIS @ aS + [0, -9.8, 0]
wM  = RIM^T @ RIS @ wS
```

Final processed v2 formula:

```text
RSB_new = R_JS^T
l4_RMB = RIM^T @ RIS @ RSB_new
l4_aM  = RIM^T @ RIS @ aS + [0, -9.8, 0] = aM
l4_wM  = RIM^T @ RIS @ wS = wM
```

Audit result:

```text
status = ok
num_sequences = 41
num_frames = 160125
official_vs_v2 aM norm mean = 0.0
official_vs_v2 wM norm mean = 0.0
official_vs_v2 RMB geodesic mean = 10.153708 deg
official_vs_v2 aRB norm mean = 1.404047
official_vs_v2 wRB norm mean = 0.520416
official_vs_v2 RRB geodesic mean = 12.865632 deg
official_vs_v2 gR0 norm mean = 0.110008
v1_vs_v2 aM norm mean = 0.0
v1_vs_v2 wM norm mean = 0.0
v1_vs_v2 RMB geodesic mean = 0.008026 deg
official RMB vs GT mapped joint rotation mean = 11.690968 deg
l4_RMB vs GT mapped joint rotation mean = 5.211172 deg
```

S4 result: v2 is exactly equal to v1 in the 11 aggregate metrics; processed score remains `38.753660`.

Interpretation: v1 was already baseline-consistent at stored-field level. The useful signal is corrected orientation and induced neural-feature changes, not reprocessed acceleration/gyro.

Claim support: validation result.

Problems: Naming should avoid implying full IMU reprocessing; use orientation-only / RMB-only correction.

Next action: Treat processed orientation as the stable input contract for downstream replacement experiments.

## EXP-20260604-003 — NewPL Run D processed-input PL replacement

Question: Can PLCurve-style NewPL with ground-truth control supervision improve processed-input S4 beyond original GPNet + processed input?

Hypothesis: A streaming-compatible PL curve module with GT control-point supervision can improve the PL stage while preserving official downstream contracts.

Change tested: Continue from Run A with `gt_control_pRB=0.3`, `gt_control_gR1=0.1`, `lr=2e-6`, 10 TotalCapture fine-tune epochs.

Dataset/split: TotalCapture processed orientation cache, validation on S4.

Command/artifacts:

```text
data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/best_loss.pt
data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/train_result.json
```

Baseline result: Original GPNet + processed score `38.753660`; earlier Run A score `38.714056`; gR dynamics continuation score `38.730901`.

New result: Run D score `38.694846`.

Metrics: Run D improves Original GPNet + processed, gR dynamics continuation, and Run A on S4 score.

Interpretation: GT control-point supervision is useful for processed-input PL replacement. Run D is the recommended PL1 upstream checkpoint for downstream replacement experiments.

Claim support: validation result.

Problems: `tail_update=4` is a control-point revision window, not a declared four-frame output latency.

Next action: Use Run D as PL1 upstream for NewIK1 / other downstream module replacements.

## EXP-20260604-004 — NewIK1 control-point PL1-output TotalCapture fine-tune

Question: Can NewIK1_ControlPoint_v1 improve the official IK1 replacement when trained on PL1 streaming outputs?

Hypothesis: Training the IK1 replacement on upstream PL1 streaming outputs reduces teacher-forcing mismatch.

Change tested: Fine-tune `NewIK1ControlPointModule` on PL1 streaming TotalCapture cache for 10 epochs from AMASS-adapted checkpoint.

Dataset/split: TotalCapture train/val processed orientation cache with PL1 streaming upstream outputs.

Command/artifacts:

```text
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/train_result.json
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/best_loss.pt
```

Config summary:

```text
epochs = 10
window = 61
lr = 3e-06
batch_size = 8
init_checkpoint = data/experiments/newik1_mainline_20260604/pl1_output_amass_adaptation/best_loss.pt
weights: pRJ=1.0, gR2=1.0, pRJ_dot=0.03, pRJ_ddot=0.001, gR2_dot=0.03, gR2_ddot=0.001, control_pRJ=0.1, control_gR2=0.1, control_point_prior=0.3, tail_update_prior=0.005
```

New result: best epoch `10`, best loss `0.17848628610372544`.

Interpretation: PL1-output fine-tuning converged locally, but this local loss alone does not prove final S4 improvement.

Claim support: bounded diagnostic.

Problems: Needs final full-pipeline evaluation before claiming module improvement.

Next action: Compare variants with bone-length and stronger pRJ/control losses, then evaluate full pipeline.

## EXP-20260604-005 — NewIK1 control-point bone-length continuation

Question: Does adding bone-length consistency improve NewIK1 PL1-output fine-tuning?

Hypothesis: Bone-length consistency stabilizes pRJ geometry and may improve gravity/root-direction behavior.

Change tested: Continue from `pl1_output_tc_finetune/best_loss.pt` with `bone_length=0.5`.

Dataset/split: Same PL1 streaming TotalCapture train/val cache as EXP-20260604-004.

Command/artifacts:

```text
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/train_result.json
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/best_loss.pt
```

Config summary:

```text
epochs = 10
window = 61
lr = 3e-06
batch_size = 8
init_checkpoint = data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/best_loss.pt
weights include bone_length=0.5
```

Baseline result: prior local best loss `0.17848628610372544`.

New result: best epoch `10`, best loss `0.1783127911388874`.

Interpretation: Bone-length loss slightly improved local validation loss, but still requires full downstream S4 validation.

Claim support: bounded diagnostic.

Problems: Local decoded-state closeness may not track final S4.

Next action: Test whether stronger pRJ/control supervision helps or hurts.

## EXP-20260604-006 — NewIK1 stronger pRJ/control continuation

Question: Does stronger pRJ and control-tail supervision improve NewIK1 beyond the bone-length continuation?

Hypothesis: Stronger pRJ/control supervision might improve Cartesian IK1 state quality.

Change tested: Continue from bone-length checkpoint with `pRJ=2.0`, `control_pRJ=0.3`, `bone_length=0.5`.

Dataset/split: Same PL1 streaming TotalCapture train/val cache as EXP-20260604-004.

Command/artifacts:

```text
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/train_result.json
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/best_loss.pt
```

Config summary:

```text
epochs = 10
window = 61
lr = 3e-06
batch_size = 8
init_checkpoint = data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/best_loss.pt
weights: pRJ=2.0, control_pRJ=0.3, bone_length=0.5
```

Baseline result: bone-length local best loss `0.1783127911388874`.

New result: best epoch `10`, best loss `0.17918791025876998`.

Interpretation: Stronger pRJ/control supervision worsened local validation loss relative to the bone-length continuation.

Claim support: bounded diagnostic.

Problems: Do not select this variant based on local loss.

Next action: Prefer the bone-length continuation locally unless full S4 says otherwise.

## EXP-20260604-007 — NewIK1 official-input PL1 streaming full-pipeline evaluation

Question: Can a finetuned official-shape IK1 replacement improve the PL1 streaming full GlobalPose pipeline?

Hypothesis: Training IK1 on official-shaped inputs with PL1 streaming upstream outputs can improve the final pipeline without changing downstream IK2/VR/physics contracts.

Change tested: Full evaluation with PL Run D checkpoint and finetuned official-shape IK1 replacement.

Dataset/split: TotalCapture S4 validation, processed orientation input, 5 sequences, 17223 frames.

Command/artifacts:

```text
scripts/run_newik1_official_input_full.sh
data/experiments/newik1_official_input_20260604/eval_pl1_streaming_tc_val.json
data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt
```

Config summary:

```text
pl_checkpoint = data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/best_loss.pt
ik1_checkpoint = data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt
pl_backend = curve_v1
ik1_backend = official_input_v1
imu_input_mode = processed
streaming_contract = GPNet.forward_frame with PL curve and a finetuned official-shape IK1 net1 replacement.
```

Baseline result: Original GPNet + processed score `38.753660`; Run D PL-only score `38.694846`.

New result: final score `38.70523069866002`, `status=ok`, `all_finite=True`.

Interpretation: This is better than Original GPNet + processed, but worse than Run D PL-only. The IK1 replacement should not be selected over PL-only Run D as the current mainline.

Claim support: validation result.

Problems: The file names are NewIK1/IK replacement artifacts, not confirmed `newik2` artifacts. Do not relabel them as NewIK2 without evidence.

Next action: If the user’s two good NewIK2 versions refer to other artifacts, locate or request those exact paths before documenting their metrics.

## EXP-20260605-001 — NewPL init36 RunD-style processed-input PL replacement

Question: Can a K2-like 36D PLCurve hidden-state initialization improve processed-input PL replacement beyond historical Run D?

Hypothesis: Initializing the streaming PL hidden state from first-frame IMU attachment offset, official PL initialization position, and root gravity direction reduces the train/runtime mismatch in the previous NewPL path.

Change tested: Use `PLCurveModule(init_size=36)` with `pl_init_feature = offset_r[18] + pRL[15] + gR0[3]`, preserve official PL frame input `aRB[18] + wRB[18] + RRB[45] + gR0[3] = 84D`, preserve official PL output `pRB[15] + gR1[3] = 18D`, and train from the historical Run D checkpoint with partial loading into the expanded init encoder.

Dataset/split: TotalCapture processed orientation cache, train 36 sequences / 142902 frames, S4 validation 5 sequences / 17223 frames. Cache schema is `pl_curve_cache_v2` with per-sequence `pl_init_feature` and `init_layout = offset_r[18] + pRL[15] + gR0[3]`.

Command/artifacts:

```text
data/dataset_work/L4Cache/pl_curve_init36_processed_tc_train_Roffset_A/pl_curve_cache_manifest.json
data/dataset_work/L4Cache/pl_curve_init36_processed_tc_val_Roffset_A/pl_curve_cache_manifest.json
data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
data/experiments/pl_curve_init36_processed_rund_style/last.pt
data/experiments/pl_curve_init36_processed_rund_style/train_result.json
data/experiments/pl_curve_init36_processed_rund_style/eval_best_final_streaming_processed.json
data/experiments/pl_curve_init36_processed_rund_style/eval_last_final_streaming_processed.json
```

Config summary:

```text
epochs = 60
window = 61
batch_size = 2
lr = 2e-6
init_size = 36
init_checkpoint = data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/best_loss.pt
baseline_pRB_weight = 0
baseline_gR1_weight = 0
disable_ik_distill = true
gR1_dot_weight = 0.03
gR1_ddot_weight = 0.001
pRB_ddot_smooth_weight = 1e-6
gt_control_pRB_weight = 0.3
gt_control_gR1_weight = 0.1
early_stop_min_delta = 1e-5
early_stop_patience = 8
```

Baseline result: Original GPNet + processed score `38.753660`; historical NewPL Run D PL-only score `38.69484578047692`.

New result: `best_loss.pt` streaming score `38.625657482802865`; `last.pt` streaming score `38.625657482802865`; `all_finite=True`; training `best_epoch=60`; training `best_loss=0.18789918906986713`; `stopped_early=False`.

Interpretation: The K2-like 36D init improved PL-only processed-input S4 by `0.069188297674055` over historical Run D and by `0.128002517197135` over Original GPNet + processed input. This becomes the current selected PL1 upstream checkpoint.

Claim support: validation result.

Problems: No blocker for NewPL init36. The result still only replaces PL-s1; downstream IK1/IK2/VR interfaces remain unchanged. NewIK2 artifacts remain unresolved and should not be inferred from NewIK1 paths.

Next action: Use NewPL init36 `best_loss.pt` / `last.pt` as the PL1 upstream for downstream replacement experiments, then locate exact NewIK2 artifacts if they exist.

## EXP-20260605-002 — NewIK1 last PL control-point input variant

Question: Does replacing the NewIK1 control-tail input with only the last PL control point improve the IK1 replacement when NewPL init36 is the upstream PL module?

Hypothesis: The final PL control point may be a cleaner upstream state than the broader control-tail context and may reduce IK1 train/runtime mismatch while preserving the official IK1 interface.

Replaced module: official IK-s1 (`iknet.net1`).

Contract preserved:

```text
IK1 input:  RRB_after_pl[45] + gR1[3] + pRB[15] = 63D
IK1 output: pRJ[69] + gR2[3] = 72D
```

Change tested: Use `control_point_last_v1` / `feature_mode=last_control` so the IK1 replacement consumes `RRB_after_pl[45] + last_control_gR1[3] + last_control_pRB[15] = 63D`. Downstream official IK2, VR, velocity fusion, and physics backend remain unchanged.

Loss design: Keep the existing NewIK1 control-point loss family:

```text
pRJ = 1.0
gR2 = 1.0
control_pRJ = 0.1
control_gR2 = 0.1
bone_length = 0.5
control_point_prior = 0.3
tail_update_prior = 0.005
pRJ_dot = 0.03
pRJ_ddot = 0.001
gR2_dot = 0.03
gR2_ddot = 0.001
```

Training/evaluation recipe:

```text
Stage A: GT/teacher-forced AMASS pretrain.
Stage B: PL streaming AMASS adaptation using NewPL init36 upstream.
Stage C: PL streaming TotalCapture fine-tune using NewPL init36 upstream.
Final eval: TotalCapture S4 full streaming evaluation for both best_loss.pt and last.pt, with real sequence-level offset_r required by NewPL init36.
```

Important orchestration notes: An initial run using `/home/lingfeng/remote-envs/globalpose-gpu-py310` failed because `carticulate.carticulate` was missing. The rerun used `/home/lingfeng/.conda/envs/globalpose-gpu`. Early PL streaming AMASS cache/eval attempts also exposed the NewPL init36 `offset_r` requirement; final S4 evaluation was rerun after passing real sequence-level `offset_r`.

Artifacts:

```text
configs/newik1_last_pl_control_20260605_v2_tasks.json
data/experiments/orchestrator_states/newik1_last_pl_control_20260605_v2.json
logs/orchestrator/newik1_last_pl_control_20260605_v2/
data/experiments/newik1_last_pl_control_20260605_v2/caches/teacher_forced_amass/newik1_control_cache_manifest.json
data/experiments/newik1_last_pl_control_20260605_v2/caches/teacher_forced_tc_train/newik1_control_cache_manifest.json
data/experiments/newik1_last_pl_control_20260605_v2/caches/teacher_forced_tc_val/newik1_control_cache_manifest.json
data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_amass/newik1_control_cache_manifest.json
data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_train/newik1_control_cache_manifest.json
data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json
data/experiments/newik1_last_pl_control_20260605_v2/stage_a_gt_pretrain/best_loss.pt
data/experiments/newik1_last_pl_control_20260605_v2/stage_b_pl_adapt/best_loss.pt
data/experiments/newik1_last_pl_control_20260605_v2/stage_c_tc_finetune/best_loss.pt
data/experiments/newik1_last_pl_control_20260605_v2/stage_c_tc_finetune/last.pt
data/experiments/newik1_last_pl_control_20260605_v2/s4/best_loss/result.json
data/experiments/newik1_last_pl_control_20260605_v2/s4/last/result.json
data/experiments/newik1_last_pl_control_20260605_v2/selection/final_selection.json
```

Cache results:

```text
teacher_forced_amass: 669 sequences / 653106 frames, input_size=63, feature_mode=last_control
teacher_forced_tc_train: 36 sequences / 142902 frames, input_size=63, feature_mode=last_control
teacher_forced_tc_val: 5 sequences / 17223 frames, input_size=63, feature_mode=last_control
pl_streaming_amass: 669 sequences / 653106 frames, streaming-compatible, upstream=data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
pl_streaming_tc_train: 36 sequences / 142902 frames, streaming-compatible, upstream=data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
pl_streaming_tc_val: 5 sequences / 17223 frames, streaming-compatible, upstream=data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
```

Training results:

```text
Stage A best_epoch = 23
Stage A best_loss = 0.003828494460321963
Stage A last_epoch = 30
Stage A last_val_loss = 0.003825079742819071

Stage B best_epoch = 20
Stage B best_loss = 0.18791283518075944
Stage B last_epoch = 20
Stage B last_val_loss = 0.18791283518075944

Stage C best_epoch = 15
Stage C best_loss = 0.18721885234117508
Stage C last_epoch = 15
Stage C last_val_loss = 0.18721885234117508
```

Final S4 result:

```text
best_loss.pt score = 38.84357685862481
last.pt score      = 38.84357685862481
selected checkpoint = best_loss.pt
baseline NewPL init36 S4 = 38.625657482802865
beats baseline = false
delta vs NewPL init36 = +0.217919375821946
```

Detailed S4 metrics for the selected `best_loss.pt` result:

```text
Local SIP = 10.184999465942383
Local Angle = 8.797480010986328
Local Joint = 4.517464208602905
Local Mesh = 5.1623249530792235
Global SIP = 10.369052410125732
Global Angle = 8.599768829345702
Global Joint = 4.358955049514771
Global Mesh = 4.932460117340088
Root Jitter = 0.2770137883722782
Joint Jitter = 0.4634216412901878
```

Comparison:

- NewPL init36 baseline: `38.625657482802865`.
- Original GPNet + processed input: `38.753660`.
- NewIK1 last PL control-point: `38.84357685862481`.

Interpretation: The last-control IK1 input design converged locally, but the full downstream S4 result is worse than both NewPL init36 and the frozen official GPNet processed-input baseline. The module-level Stage C validation loss is not a sufficient selection metric for IK1 replacement.

Claim support: validation result.

Conclusion: Do not select `newik1_last_pl_control_v1`. Keep NewPL init36 as the current mainline PL1 upstream checkpoint. Future IK1 work should change the IK1 design or loss in a way that is explicitly justified, then validate through full S4 before claiming improvement.

## EXP-newik1_v6_official_input_init36_cascade

### Orchestrator Task: v6_teacher_forced_tc_val_cache

Name: Build v6 official-input teacher-forced TotalCapture val cache

Status: completed

Type: cache

Start: 2026-06-05T18:00:07

End: 2026-06-05T18:00:22

GPU: 0

PID: 1065107

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade/caches/teacher_forced_tc_val --mode teacher_forced --imu-input-mode processed --shard-size 50`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade/v6_teacher_forced_tc_val_cache.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json`

Summary:

| metric | value |
|---|---:|
| cache_type | teacher-forced-gt-like |
| imu_input_mode | processed |
| mode | teacher_forced |
| num_frames | 17223 |
| num_sequences | 5 |
| pl_checkpoint | None |
| pl_checkpoint_config | None |
| source_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| type | newik1_official_input_cache_v1 |

### Orchestrator Task: v6_pl_streaming_amass_cache

Name: Build v6 official-input NewPL init36 streaming AMASS cache

Status: failed

Type: cache

Start: 2026-06-05T18:00:22

End: 2026-06-05T18:00:37

GPU: 0

PID: 1065450

Return code: 1

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade/caches/pl_streaming_amass --mode pl1_streaming --imu-input-mode auto --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --shard-size 50`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade/v6_pl_streaming_amass_cache.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade/caches/pl_streaming_amass/newik1_official_input_cache_manifest.json`

Missing outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade/caches/pl_streaming_amass/newik1_official_input_cache_manifest.json`

Blocked downstream tasks:

- `v6_stage_b_pl_amass_adapt`
- `v6_stage_b_continue`
- `v6_stage_b_module_gt`
- `v6_stage_b_s4`
- `v6_stage_c_tc_finetune`
- `v6_stage_c_continue`
- `v6_stage_c_module_gt`
- `v6_stage_c_s4_best`
- `v6_stage_c_s4_last`
- `v6_select_final`

Error: missing outputs: ['data/experiments/newik1_v6_official_input_init36_cascade/caches/pl_streaming_amass/newik1_official_input_cache_manifest.json']

Summary:

Task failed.

Log tail:

```text
# task_id=v6_pl_streaming_amass_cache
# start=2026-06-05T18:00:22
# gpu=0
# command=ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade/caches/pl_streaming_amass --mode pl1_streaming --imu-input-mode auto --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --shard-size 50

Traceback (most recent call last):
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_cache.py", line 316, in <module>
    main()
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_cache.py", line 302, in main
    manifest = build_cache(
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_cache.py", line 230, in build_cache
    raise KeyError(f'{cache_file} has no offset_r field required for PL init feature.')
KeyError: 'data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7/baseline_cache_shard00000.pt has no offset_r field required for PL init feature.'
```

Manual follow-up after failed `v6_select_final`:

- The selection task failed only because the inline `python -c` command contained escaped newlines that produced a `SyntaxError`.
- All training, Module GT diagnostics, and S4 evaluations had already completed successfully.
- Manual selection JSON was generated at:

```text
data/experiments/newik1_v6_official_input_init36_cascade_rerun/selection/final_selection.json
```

Manual selection summary:

| Stage | S4 Score ↓ | Module GT state_l2 delta ↓ | Module GT Better? | Beats NewPL init36? |
|---|---:|---:|---|---|
| stage_a | `38.649136830300094` | `-0.02107336552593999` | yes | no |
| stage_b | `41.534289258092635` | `-0.0059323872736485594` | yes | no |
| stage_c_best | `41.543203821450476` | `-0.006219639595512971` | yes | no |
| stage_c_last | `41.543203821450476` | `-0.006219639595512971` | yes | no |

Best v6 checkpoint by S4:

```text
data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt
```

Conclusion: `newik1_v6_official_input_init36_cascade` improves isolated IK1 output-vs-GT metrics but does not beat the current NewPL init36 full-pipeline S4 baseline.

Additional audit after metric-contract review:

The first v6 summary mixed two input distributions in prose. `stage_a` Module GT was originally measured on `teacher_forced_tc_val`, where official IK1 baseline gR2 error is only `2.522684 deg`. That is not the PL-streaming deployment input. A separate audit was run for `stage_a_continue/best_loss.pt` on `pl_streaming_tc_val`:

```text
data/experiments/newik1_v6_official_input_init36_cascade_rerun/module_gt/stage_a_on_pl_streaming/result.json
```

Correct PL-streaming comparison for Stage A:

| Metric | Official IK1 baseline | NewIK1 v6 Stage A | Delta NewIK1 - Baseline |
|---|---:|---:|---:|
| pRJ L1 | `3.666342 cm` | `3.707329 cm` | `0.040988 cm` |
| pRJ L2 | `6.336144 cm` | `6.347230 cm` | `0.011086 cm` |
| gR2 angle | `24.464683 deg` | `24.420436 deg` | `-0.044246 deg` |
| state L2 | `0.086645` | `0.086617` | `-0.0000278` |

Interpretation: the user's memory that PL-streaming baseline gR2 is about 20+ degrees was correct. Stage A is only essentially tied on PL-streaming Module GT; it does not materially improve pRJ, and the gR2 improvement is tiny.

### Orchestrator Task: v6_teacher_forced_amass_cache

Name: Build v6 official-input teacher-forced AMASS cache

Status: completed

Type: cache

Start: 2026-06-05T18:00:07

End: 2026-06-05T18:06:57

GPU: 1

PID: 1065106

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade/caches/teacher_forced_amass --mode teacher_forced --imu-input-mode auto --shard-size 50`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade/v6_teacher_forced_amass_cache.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json`

Summary:

| metric | value |
|---|---:|
| cache_type | teacher-forced-gt-like |
| imu_input_mode | auto |
| mode | teacher_forced |
| num_frames | 653106 |
| num_sequences | 669 |
| pl_checkpoint | None |
| pl_checkpoint_config | None |
| source_cache | data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged/baseline_cache_manifest.json |
| type | newik1_official_input_cache_v1 |

### Orchestrator Task: v6_pl_streaming_tc_train_cache

Name: Build v6 official-input NewPL init36 streaming TotalCapture train cache

Status: completed

Type: cache

Start: 2026-06-05T18:00:37

End: 2026-06-05T18:07:27

GPU: 0

PID: 1066182

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade/caches/pl_streaming_tc_train --mode pl1_streaming --imu-input-mode processed --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --shard-size 50`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade/v6_pl_streaming_tc_train_cache.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade/caches/pl_streaming_tc_train/newik1_official_input_cache_manifest.json`

Summary:

| metric | value |
|---|---:|
| cache_type | streaming-compatible |
| imu_input_mode | processed |
| mode | pl1_streaming |
| num_frames | 142902 |
| num_sequences | 36 |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| source_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json |
| type | newik1_official_input_cache_v1 |

### Orchestrator Task: v6_stage_a_gt_refresh

Name: Stage A refresh v4 on teacher-forced AMASS

Status: completed

Type: train

Start: 2026-06-05T18:06:57

End: 2026-06-05T18:07:43

GPU: CPU

PID: 1066680

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade/stage_a_gt_refresh --experiment-name newik1_v6_stage_a_gt_refresh --epochs 12 --lr 1e-5 --dropout 0.20 --batch-size 16 --window 61 --max-val-sequences 5 --init-checkpoint data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade/v6_stage_a_gt_refresh.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade/stage_a_gt_refresh/best_loss.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade/stage_a_gt_refresh/last.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade/stage_a_gt_refresh/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 10 |
| best_loss | 0.0011926728824619205 |
| last_epoch | 12 |
| last_val_loss | 0.0012054321356117725 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| ik1_distill_gR2 | 0.0 |
| ik1_distill_pRJ | 0.2 |
| pRJ | 2.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: v6_pl_streaming_tc_val_cache

Name: Build v6 official-input NewPL init36 streaming TotalCapture val cache

Status: completed

Type: cache

Start: 2026-06-05T18:06:57

End: 2026-06-05T18:07:58

GPU: 1

PID: 1066679

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade/caches/pl_streaming_tc_val --mode pl1_streaming --imu-input-mode processed --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --shard-size 50`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade/v6_pl_streaming_tc_val_cache.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade/caches/pl_streaming_tc_val/newik1_official_input_cache_manifest.json`

Summary:

| metric | value |
|---|---:|
| cache_type | streaming-compatible |
| imu_input_mode | processed |
| mode | pl1_streaming |
| num_frames | 17223 |
| num_sequences | 5 |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| source_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| type | newik1_official_input_cache_v1 |

### Orchestrator Task: v6_stage_a_continue

Name: Stage A low-LR continuation

Status: completed

Type: train

Start: 2026-06-05T18:07:43

End: 2026-06-05T18:08:13

GPU: CPU

PID: 1066902

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade/stage_a_continue --experiment-name newik1_v6_stage_a_continue --epochs 6 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --max-val-sequences 5 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade/stage_a_gt_refresh/best_loss.pt`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade/v6_stage_a_continue.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade/stage_a_continue/best_loss.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade/stage_a_continue/last.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade/stage_a_continue/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 6 |
| best_loss | 0.0011761557892896236 |
| last_epoch | 6 |
| last_val_loss | 0.0011761557892896236 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| ik1_distill_gR2 | 0.0 |
| ik1_distill_pRJ | 0.2 |
| pRJ | 2.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: v6_stage_a_module_gt

Name: Stage A module-output-vs-GT diagnostic

Status: completed

Type: audit

Start: 2026-06-05T18:08:13

End: 2026-06-05T18:08:28

GPU: CPU

PID: 1067001

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_diagnostic.py --cache data/experiments/newik1_v6_official_input_init36_cascade/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade/stage_a_continue/best_loss.pt --output-json data/experiments/newik1_v6_official_input_init36_cascade/module_gt/stage_a/result.json`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade/v6_stage_a_module_gt.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade/module_gt/stage_a/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_v6_official_input_init36_cascade/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json |
| ik1_checkpoint | data/experiments/newik1_v6_official_input_init36_cascade/stage_a_continue/best_loss.pt |
| input_contract | RRB_after_pl[45] + gR1[3] + pRB[15] = 63D |
| metric_contract | Compare NewIK1 output and cache ik1_base against ik1_target GT on the same official-shape IK1 cache. |
| module | IK-s1 |
| output_contract | pRJ[69] + gR2[3] = 72D |
| status | ok |

### Orchestrator Task: v6_stage_a_s4

Name: Stage A full S4 11-metric evaluation

Status: completed

Type: eval

Start: 2026-06-05T18:08:13

End: 2026-06-05T18:15:19

GPU: CPU

PID: 1067003

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade/stage_a_continue/best_loss.pt --imu-input-mode processed --output-json data/experiments/newik1_v6_official_input_init36_cascade/s4/stage_a/result.json`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade/v6_stage_a_s4.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade/s4/stage_a/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.649436336755755 |
| Local SIP | 10.067363166809082 |
| Local Angle | 8.797456455230712 |
| Local Joint | 4.523646020889283 |
| Local Mesh | 5.175414228439331 |
| Global SIP | 10.315696525573731 |
| Global Angle | 8.573093509674072 |
| Global Joint | 4.385212564468384 |
| Global Mesh | 4.901029634475708 |
| Root Jitter | 0.3008135363459587 |
| Joint Jitter | 0.49408209323883057 |

## EXP-newik1_v6_official_input_init36_cascade_rerun

### Orchestrator Task: v6_teacher_forced_tc_val_cache

Name: Build v6 official-input teacher-forced TotalCapture val cache

Status: completed

Type: cache

Start: 2026-06-05T18:15:45

End: 2026-06-05T18:16:00

GPU: 0

PID: 1070906

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val --mode teacher_forced --imu-input-mode processed --shard-size 50`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_teacher_forced_tc_val_cache.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json`

Summary:

| metric | value |
|---|---:|
| cache_type | teacher-forced-gt-like |
| imu_input_mode | processed |
| mode | teacher_forced |
| num_frames | 17223 |
| num_sequences | 5 |
| pl_checkpoint | None |
| pl_checkpoint_config | None |
| source_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| type | newik1_official_input_cache_v1 |

### Orchestrator Task: v6_teacher_forced_amass_cache

Name: Build v6 official-input teacher-forced AMASS cache

Status: completed

Type: cache

Start: 2026-06-05T18:15:45

End: 2026-06-05T18:22:36

GPU: 1

PID: 1070905

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged_with_offset_r/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass --mode teacher_forced --imu-input-mode auto --shard-size 50`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_teacher_forced_amass_cache.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json`

Summary:

| metric | value |
|---|---:|
| cache_type | teacher-forced-gt-like |
| imu_input_mode | auto |
| mode | teacher_forced |
| num_frames | 653106 |
| num_sequences | 669 |
| pl_checkpoint | None |
| pl_checkpoint_config | None |
| source_cache | data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged_with_offset_r/baseline_cache_manifest.json |
| type | newik1_official_input_cache_v1 |

### Orchestrator Task: v6_stage_a_gt_refresh

Name: Stage A refresh v4 on teacher-forced AMASS

Status: completed

Type: train

Start: 2026-06-05T18:22:36

End: 2026-06-05T18:23:21

GPU: CPU

PID: 1072994

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_gt_refresh --experiment-name newik1_v6_stage_a_gt_refresh --epochs 12 --lr 1e-5 --dropout 0.20 --batch-size 16 --window 61 --max-val-sequences 5 --init-checkpoint data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_a_gt_refresh.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_gt_refresh/best_loss.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_gt_refresh/last.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_gt_refresh/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 11 |
| best_loss | 0.0011530003452207894 |
| last_epoch | 12 |
| last_val_loss | 0.0011544206412509085 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| ik1_distill_gR2 | 0.0 |
| ik1_distill_pRJ | 0.2 |
| pRJ | 2.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: v6_stage_a_continue

Name: Stage A low-LR continuation

Status: completed

Type: train

Start: 2026-06-05T18:23:22

End: 2026-06-05T18:23:52

GPU: CPU

PID: 1073335

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue --experiment-name newik1_v6_stage_a_continue --epochs 6 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --max-val-sequences 5 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_gt_refresh/best_loss.pt`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_a_continue.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/last.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 6 |
| best_loss | 0.0011462041060440243 |
| last_epoch | 6 |
| last_val_loss | 0.0011462041060440243 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| ik1_distill_gR2 | 0.0 |
| ik1_distill_pRJ | 0.2 |
| pRJ | 2.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: v6_stage_a_module_gt

Name: Stage A module-output-vs-GT diagnostic

Status: completed

Type: audit

Start: 2026-06-05T18:23:52

End: 2026-06-05T18:24:07

GPU: CPU

PID: 1073506

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_diagnostic.py --cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-json data/experiments/newik1_v6_official_input_init36_cascade_rerun/module_gt/stage_a/result.json`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_a_module_gt.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/module_gt/stage_a/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json |
| ik1_checkpoint | data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt |
| input_contract | RRB_after_pl[45] + gR1[3] + pRB[15] = 63D |
| metric_contract | Compare NewIK1 output and cache ik1_base against ik1_target GT on the same official-shape IK1 cache. |
| module | IK-s1 |
| output_contract | pRJ[69] + gR2[3] = 72D |
| status | ok |

### Orchestrator Task: v6_pl_streaming_tc_train_cache

Name: Build v6 official-input NewPL init36 streaming TotalCapture train cache

Status: completed

Type: cache

Start: 2026-06-05T18:22:36

End: 2026-06-05T18:29:27

GPU: 1

PID: 1072993

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_train --mode pl1_streaming --imu-input-mode processed --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --shard-size 50`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_pl_streaming_tc_train_cache.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_train/newik1_official_input_cache_manifest.json`

Summary:

| metric | value |
|---|---:|
| cache_type | streaming-compatible |
| imu_input_mode | processed |
| mode | pl1_streaming |
| num_frames | 142902 |
| num_sequences | 36 |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| source_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json |
| type | newik1_official_input_cache_v1 |

### Orchestrator Task: v6_pl_streaming_tc_val_cache

Name: Build v6 official-input NewPL init36 streaming TotalCapture val cache

Status: completed

Type: cache

Start: 2026-06-05T18:29:28

End: 2026-06-05T18:30:28

GPU: 1

PID: 1075599

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_val --mode pl1_streaming --imu-input-mode processed --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --shard-size 50`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_pl_streaming_tc_val_cache.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_val/newik1_official_input_cache_manifest.json`

Summary:

| metric | value |
|---|---:|
| cache_type | streaming-compatible |
| imu_input_mode | processed |
| mode | pl1_streaming |
| num_frames | 17223 |
| num_sequences | 5 |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| source_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| type | newik1_official_input_cache_v1 |

### Orchestrator Task: v6_stage_a_s4

Name: Stage A full S4 11-metric evaluation

Status: completed

Type: eval

Start: 2026-06-05T18:23:52

End: 2026-06-05T18:31:14

GPU: CPU

PID: 1073507

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --imu-input-mode processed --output-json data/experiments/newik1_v6_official_input_init36_cascade_rerun/s4/stage_a/result.json`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_a_s4.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/s4/stage_a/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.649136830300094 |
| Local SIP | 10.066027450561524 |
| Local Angle | 8.79630355834961 |
| Local Joint | 4.52389087677002 |
| Local Mesh | 5.174596786499023 |
| Global SIP | 10.31864824295044 |
| Global Angle | 8.571873569488526 |
| Global Joint | 4.3895911693573 |
| Global Mesh | 4.904520702362061 |
| Root Jitter | 0.30062844827771185 |
| Joint Jitter | 0.4935804337263107 |

### Orchestrator Task: v6_pl_streaming_amass_cache

Name: Build v6 official-input NewPL init36 streaming AMASS cache

Status: completed

Type: cache

Start: 2026-06-05T18:16:00

End: 2026-06-05T18:48:13

GPU: 0

PID: 1071229

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_cache.py --input-cache data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged_with_offset_r/baseline_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_amass --mode pl1_streaming --imu-input-mode auto --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --shard-size 50`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_pl_streaming_amass_cache.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_amass/newik1_official_input_cache_manifest.json`

Summary:

| metric | value |
|---|---:|
| cache_type | streaming-compatible |
| imu_input_mode | auto |
| mode | pl1_streaming |
| num_frames | 653106 |
| num_sequences | 669 |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| source_cache | data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged_with_offset_r/baseline_cache_manifest.json |
| type | newik1_official_input_cache_v1 |

### Orchestrator Task: v6_stage_b_pl_amass_adapt

Name: Stage B adapt on NewPL init36 streaming AMASS

Status: completed

Type: train

Start: 2026-06-05T18:48:13

End: 2026-06-05T18:48:59

GPU: CPU

PID: 1083779

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_pl_amass_adapt --experiment-name newik1_v6_stage_b_pl_amass_adapt --epochs 16 --lr 5e-6 --dropout 0.20 --batch-size 16 --window 61 --max-val-sequences 5 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_b_pl_amass_adapt.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_pl_amass_adapt/best_loss.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_pl_amass_adapt/last.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_pl_amass_adapt/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 16 |
| best_loss | 0.161585184186697 |
| last_epoch | 16 |
| last_val_loss | 0.161585184186697 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| ik1_distill_gR2 | 0.0 |
| ik1_distill_pRJ | 0.2 |
| pRJ | 2.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: v6_stage_b_continue

Name: Stage B low-LR continuation

Status: completed

Type: train

Start: 2026-06-05T18:48:59

End: 2026-06-05T18:49:29

GPU: CPU

PID: 1084108

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_continue --experiment-name newik1_v6_stage_b_continue --epochs 8 --lr 1e-6 --dropout 0.15 --batch-size 16 --window 61 --max-val-sequences 5 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_pl_amass_adapt/best_loss.pt`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_b_continue.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_continue/best_loss.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_continue/last.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_continue/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 8 |
| best_loss | 0.1600326281040907 |
| last_epoch | 8 |
| last_val_loss | 0.1600326281040907 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| ik1_distill_gR2 | 0.0 |
| ik1_distill_pRJ | 0.2 |
| pRJ | 2.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: v6_stage_b_module_gt

Name: Stage B module-output-vs-GT diagnostic

Status: completed

Type: audit

Start: 2026-06-05T18:49:29

End: 2026-06-05T18:49:44

GPU: CPU

PID: 1084379

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_diagnostic.py --cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_val/newik1_official_input_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_continue/best_loss.pt --output-json data/experiments/newik1_v6_official_input_init36_cascade_rerun/module_gt/stage_b/result.json`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_b_module_gt.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/module_gt/stage_b/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_val/newik1_official_input_cache_manifest.json |
| ik1_checkpoint | data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_continue/best_loss.pt |
| input_contract | RRB_after_pl[45] + gR1[3] + pRB[15] = 63D |
| metric_contract | Compare NewIK1 output and cache ik1_base against ik1_target GT on the same official-shape IK1 cache. |
| module | IK-s1 |
| output_contract | pRJ[69] + gR2[3] = 72D |
| status | ok |

### Orchestrator Task: v6_stage_c_tc_finetune

Name: Stage C fine-tune on NewPL init36 streaming TotalCapture

Status: completed

Type: train

Start: 2026-06-05T18:49:29

End: 2026-06-05T18:50:00

GPU: CPU

PID: 1084382

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_train/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_tc_finetune --experiment-name newik1_v6_stage_c_tc_finetune --epochs 12 --lr 2e-6 --dropout 0.10 --batch-size 8 --window 61 --max-val-sequences 5 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_continue/best_loss.pt`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_c_tc_finetune.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_tc_finetune/best_loss.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_tc_finetune/last.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_tc_finetune/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 12 |
| best_loss | 0.15916615389287472 |
| last_epoch | 12 |
| last_val_loss | 0.15916615389287472 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| ik1_distill_gR2 | 0.0 |
| ik1_distill_pRJ | 0.2 |
| pRJ | 2.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: v6_stage_c_continue

Name: Stage C low-LR continuation

Status: completed

Type: train

Start: 2026-06-05T18:50:00

End: 2026-06-05T18:50:15

GPU: CPU

PID: 1084884

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_train/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_continue --experiment-name newik1_v6_stage_c_continue --epochs 8 --lr 5e-7 --dropout 0.05 --batch-size 8 --window 61 --max-val-sequences 5 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_tc_finetune/best_loss.pt`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_c_continue.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_continue/best_loss.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_continue/last.pt`
- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_continue/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 8 |
| best_loss | 0.15904558263719082 |
| last_epoch | 8 |
| last_val_loss | 0.15904558263719082 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| ik1_distill_gR2 | 0.0 |
| ik1_distill_pRJ | 0.2 |
| pRJ | 2.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: v6_stage_c_module_gt

Name: Stage C module-output-vs-GT diagnostic

Status: completed

Type: audit

Start: 2026-06-05T18:50:15

End: 2026-06-05T18:50:30

GPU: CPU

PID: 1085048

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_diagnostic.py --cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_val/newik1_official_input_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_continue/best_loss.pt --output-json data/experiments/newik1_v6_official_input_init36_cascade_rerun/module_gt/stage_c/result.json`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_c_module_gt.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/module_gt/stage_c/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/pl_streaming_tc_val/newik1_official_input_cache_manifest.json |
| ik1_checkpoint | data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_continue/best_loss.pt |
| input_contract | RRB_after_pl[45] + gR1[3] + pRB[15] = 63D |
| metric_contract | Compare NewIK1 output and cache ik1_base against ik1_target GT on the same official-shape IK1 cache. |
| module | IK-s1 |
| output_contract | pRJ[69] + gR2[3] = 72D |
| status | ok |

### Orchestrator Task: v6_stage_b_s4

Name: Stage B full S4 11-metric evaluation

Status: completed

Type: eval

Start: 2026-06-05T18:49:29

End: 2026-06-05T18:56:22

GPU: CPU

PID: 1084380

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_b_continue/best_loss.pt --imu-input-mode processed --output-json data/experiments/newik1_v6_official_input_init36_cascade_rerun/s4/stage_b/result.json`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_b_s4.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/s4/stage_b/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 41.534289258092635 |
| Local SIP | 10.964134025573731 |
| Local Angle | 9.12691526412964 |
| Local Joint | 4.873451471328735 |
| Local Mesh | 5.439082765579224 |
| Global SIP | 11.36605281829834 |
| Global Angle | 9.083804416656495 |
| Global Joint | 5.0111918449401855 |
| Global Mesh | 5.4620343208312985 |
| Root Jitter | 0.2922183766961098 |
| Joint Jitter | 0.49184018075466157 |

### Orchestrator Task: v6_stage_c_s4_best

Name: Stage C best full S4 11-metric evaluation

Status: completed

Type: eval

Start: 2026-06-05T18:50:15

End: 2026-06-05T18:57:07

GPU: CPU

PID: 1085049

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_continue/best_loss.pt --imu-input-mode processed --output-json data/experiments/newik1_v6_official_input_init36_cascade_rerun/s4/stage_c_best/result.json`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_c_s4_best.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/s4/stage_c_best/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 41.543203821450476 |
| Local SIP | 10.977999496459962 |
| Local Angle | 9.12474546432495 |
| Local Joint | 4.878132009506226 |
| Local Mesh | 5.443132114410401 |
| Global SIP | 11.37718324661255 |
| Global Angle | 9.06924648284912 |
| Global Joint | 5.013017416000366 |
| Global Mesh | 5.463963556289673 |
| Root Jitter | 0.2917546473443508 |
| Joint Jitter | 0.4914188653230667 |

### Orchestrator Task: v6_stage_c_s4_last

Name: Stage C last full S4 11-metric evaluation

Status: completed

Type: eval

Start: 2026-06-05T18:50:15

End: 2026-06-05T18:57:07

GPU: CPU

PID: 1085051

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_c_continue/last.pt --imu-input-mode processed --output-json data/experiments/newik1_v6_official_input_init36_cascade_rerun/s4/stage_c_last/result.json`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_stage_c_s4_last.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/s4/stage_c_last/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 41.543203821450476 |
| Local SIP | 10.977999496459962 |
| Local Angle | 9.12474546432495 |
| Local Joint | 4.878132009506226 |
| Local Mesh | 5.443132114410401 |
| Global SIP | 11.37718324661255 |
| Global Angle | 9.06924648284912 |
| Global Joint | 5.013017416000366 |
| Global Mesh | 5.463963556289673 |
| Root Jitter | 0.2917546473443508 |
| Joint Jitter | 0.4914188653230667 |

### Orchestrator Task: v6_select_final

Name: Select final v6 checkpoint by S4 and module GT delta

Status: failed

Type: parse

Start: 2026-06-05T18:57:07

End: 2026-06-05T18:57:22

GPU: CPU

PID: 1087770

Return code: 1

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" -c "import json; from pathlib import Path; root=Path('data/experiments/newik1_v6_official_input_init36_cascade_rerun'); baseline=38.625657482802865; stages=[('stage_a', root/'s4/stage_a/result.json', root/'module_gt/stage_a/result.json', root/'stage_a_continue/best_loss.pt'), ('stage_b', root/'s4/stage_b/result.json', root/'module_gt/stage_b/result.json', root/'stage_b_continue/best_loss.pt'), ('stage_c_best', root/'s4/stage_c_best/result.json', root/'module_gt/stage_c/result.json', root/'stage_c_continue/best_loss.pt'), ('stage_c_last', root/'s4/stage_c_last/result.json', root/'module_gt/stage_c/result.json', root/'stage_c_continue/last.pt')]; rows=[];\nfor name,s4p,gtp,ckpt in stages:\n    s4=json.loads(s4p.read_text()); gt=json.loads(gtp.read_text()); rows.append({'stage':name,'score':float(s4['score']),'beats_newpl_init36':float(s4['score'])<baseline,'all_finite':bool(s4.get('all_finite')),'module_gt_state_l2_delta':gt['aggregate']['delta_newik1_minus_baseline']['state_l2'],'module_gt_better':bool(gt['aggregate']['newik1_better_by_state_l2']),'s4_json':str(s4p),'module_gt_json':str(gtp),'checkpoint':str(ckpt)});\nselected=min(rows, key=lambda r:r['score']); out={'status':'ok','baseline_newpl_init36_s4':baseline,'selected':selected,'rows':rows}; p=root/'selection/final_selection.json'; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=2)+'\\n'); print(json.dumps(out, indent=2))"`

Log: `logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/v6_select_final.log`

Outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/selection/final_selection.json`

Missing outputs:

- `data/experiments/newik1_v6_official_input_init36_cascade_rerun/selection/final_selection.json`

Error: missing outputs: ['data/experiments/newik1_v6_official_input_init36_cascade_rerun/selection/final_selection.json']

Summary:

Task failed.

Log tail:

```text
# task_id=v6_select_final
# start=2026-06-05T18:57:07
# gpu=CPU
# command=ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" -c "import json; from pathlib import Path; root=Path('data/experiments/newik1_v6_official_input_init36_cascade_rerun'); baseline=38.625657482802865; stages=[('stage_a', root/'s4/stage_a/result.json', root/'module_gt/stage_a/result.json', root/'stage_a_continue/best_loss.pt'), ('stage_b', root/'s4/stage_b/result.json', root/'module_gt/stage_b/result.json', root/'stage_b_continue/best_loss.pt'), ('stage_c_best', root/'s4/stage_c_best/result.json', root/'module_gt/stage_c/result.json', root/'stage_c_continue/best_loss.pt'), ('stage_c_last', root/'s4/stage_c_last/result.json', root/'module_gt/stage_c/result.json', root/'stage_c_continue/last.pt')]; rows=[];\nfor name,s4p,gtp,ckpt in stages:\n    s4=json.loads(s4p.read_text()); gt=json.loads(gtp.read_text()); rows.append({'stage':name,'score':float(s4['score']),'beats_newpl_init36':float(s4['score'])<baseline,'all_finite':bool(s4.get('all_finite')),'module_gt_state_l2_delta':gt['aggregate']['delta_newik1_minus_baseline']['state_l2'],'module_gt_better':bool(gt['aggregate']['newik1_better_by_state_l2']),'s4_json':str(s4p),'module_gt_json':str(gtp),'checkpoint':str(ckpt)});\nselected=min(rows, key=lambda r:r['score']); out={'status':'ok','baseline_newpl_init36_s4':baseline,'selected':selected,'rows':rows}; p=root/'selection/final_selection.json'; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=2)+'\\n'); print(json.dumps(out, indent=2))"

  File "<string>", line 1
    import json; from pathlib import Path; root=Path('data/experiments/newik1_v6_official_input_init36_cascade_rerun'); baseline=38.625657482802865; stages=[('stage_a', root/'s4/stage_a/result.json', root/'module_gt/stage_a/result.json', root/'stage_a_continue/best_loss.pt'), ('stage_b', root/'s4/stage_b/result.json', root/'module_gt/stage_b/result.json', root/'stage_b_continue/best_loss.pt'), ('stage_c_best', root/'s4/stage_c_best/result.json', root/'module_gt/stage_c/result.json', root/'stage_c_continue/best_loss.pt'), ('stage_c_last', root/'s4/stage_c_last/result.json', root/'module_gt/stage_c/result.json', root/'stage_c_continue/last.pt')]; rows=[];\nfor name,s4p,gtp,ckpt in stages:\n    s4=json.loads(s4p.read_text()); gt=json.loads(gtp.read_text()); rows.append({'stage':name,'score':float(s4['score']),'beats_newpl_init36':float(s4['score'])<baseline,'all_finite':bool(s4.get('all_finite')),'module_gt_state_l2_delta':gt['aggregate']['delta_newik1_minus_baseline']['state_l2'],'module_gt_better':bool(gt['aggregate']['newik1_better_by_state_l2']),'s4_json':str(s4p),'module_gt_json':str(gtp),'checkpoint':str(ckpt)});\nselected=min(rows, key=lambda r:r['score']); out={'status':'ok','baseline_newpl_init36_s4':baseline,'selected':selected,'rows':rows}; p=root/'selection/final_selection.json'; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=2)+'\n'); print(json.dumps(out, indent=2))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ^
SyntaxError: unexpected character after line continuation character
```

## EXP-newik1_v7_last_pl_control_lightloss_amass

### Orchestrator Task: v7_stage_a_lightloss_amass

Name: Train v7 last-control lightloss on AMASS

Status: completed

Type: train

Start: 2026-06-05T19:48:27

End: 2026-06-05T19:55:02

GPU: CPU

PID: 1099038

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_train.py --train-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/teacher_forced_amass/newik1_control_cache_manifest.json --val-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --output-dir data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass --experiment-name newik1_v7_last_pl_control_lightloss_amass --epochs 30 --lr 1e-4 --min-lr 1e-6 --warmup-epochs 3 --dropout 0.10 --weight-decay 1e-4 --early-stop-patience 8 --early-stop-min-delta 1e-5 --batch-size 16 --window 61 --pRJ-weight 1.0 --gR2-weight 1.0 --pRJ-dot-weight 0.03 --gR2-dot-weight 0.03 --pRJ-ddot-weight 0.001 --gR2-ddot-weight 0.001 --control-pRJ-weight 0.1 --control-gR2-weight 0.1 --control-pRJ-dot-weight 0.003 --control-gR2-dot-weight 0.003 --control-pRJ-ddot-weight 0.0001 --control-gR2-ddot-weight 0.0001 --bone-length-weight 0 --control-point-prior-weight 0 --tail-update-prior-weight 0 --gt-control-pRJ-weight 0 --gt-control-gR2-weight 0`

Log: `logs/orchestrator/newik1_v7_last_pl_control_lightloss_amass/v7_stage_a_lightloss_amass.log`

Outputs:

- `data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt`
- `data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/last.pt`
- `data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 1 |
| best_loss | 0.19021479934453964 |
| last_epoch | 9 |
| last_val_loss | 0.20256012827157974 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.0 |
| control_gR2 | 0.1 |
| control_gR2_ddot | 0.0001 |
| control_gR2_dot | 0.003 |
| control_pRJ | 0.1 |
| control_pRJ_ddot | 0.0001 |
| control_pRJ_dot | 0.003 |
| control_point_prior | 0.0 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| gt_control_gR2 | 0.0 |
| gt_control_pRJ | 0.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |
| tail_update_prior | 0.0 |

### Orchestrator Task: v7_module_gt_best

Name: Evaluate v7 best Module GT on NewPL streaming TC val

Status: failed

Type: audit

Start: 2026-06-05T19:55:02

End: 2026-06-05T19:55:33

GPU: CPU

PID: 1101440

Return code: 1

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt --output-json data/experiments/newik1_v7_last_pl_control_lightloss_amass/module_gt/best_loss/result.json`

Log: `logs/orchestrator/newik1_v7_last_pl_control_lightloss_amass/v7_module_gt_best.log`

Outputs:

- `data/experiments/newik1_v7_last_pl_control_lightloss_amass/module_gt/best_loss/result.json`

Error: return code 1

Summary:

Task failed.

Log tail:

```text
# task_id=v7_module_gt_best
# start=2026-06-05T19:55:02
# gpu=CPU
# command=ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt --output-json data/experiments/newik1_v7_last_pl_control_lightloss_amass/module_gt/best_loss/result.json

{
  "status": "failed",
  "aggregate": null,
  "error_type": "OutOfMemoryError",
  "error": "CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.35 GiB of which 6.88 MiB is free. Process 1101446 has 5.59 GiB memory in use. Including non-PyTorch memory, this process has 10.62 GiB memory in use. Process 1101443 has 5.59 GiB memory in use. Process 1101441 has 9.19 GiB memory in use. Of the allocated memory 8.17 GiB is allocated by PyTorch, and 1.87 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)"
}
```

### Orchestrator Task: v7_module_gt_last

Name: Evaluate v7 last Module GT on NewPL streaming TC val

Status: failed

Type: audit

Start: 2026-06-05T19:55:02

End: 2026-06-05T19:55:33

GPU: CPU

PID: 1101441

Return code: 1

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/last.pt --output-json data/experiments/newik1_v7_last_pl_control_lightloss_amass/module_gt/last/result.json`

Log: `logs/orchestrator/newik1_v7_last_pl_control_lightloss_amass/v7_module_gt_last.log`

Outputs:

- `data/experiments/newik1_v7_last_pl_control_lightloss_amass/module_gt/last/result.json`

Error: return code 1

Summary:

Task failed.

Log tail:

```text
# task_id=v7_module_gt_last
# start=2026-06-05T19:55:02
# gpu=CPU
# command=ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/last.pt --output-json data/experiments/newik1_v7_last_pl_control_lightloss_amass/module_gt/last/result.json

{
  "status": "failed",
  "aggregate": null,
  "error_type": "OutOfMemoryError",
  "error": "CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 31.35 GiB of which 6.88 MiB is free. Process 1101446 has 5.59 GiB memory in use. Process 1101440 has 10.62 GiB memory in use. Process 1101443 has 5.59 GiB memory in use. Including non-PyTorch memory, this process has 9.19 GiB memory in use. Of the allocated memory 7.17 GiB is allocated by PyTorch, and 1.45 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)"
}
```

### Orchestrator Task: v7_s4_best

Name: Evaluate v7 best_loss full S4

Status: completed

Type: eval

Start: 2026-06-05T19:55:02

End: 2026-06-05T20:02:24

GPU: CPU

PID: 1101443

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt --imu-input-mode processed --output-json data/experiments/newik1_v7_last_pl_control_lightloss_amass/s4/best_loss/result.json`

Log: `logs/orchestrator/newik1_v7_last_pl_control_lightloss_amass/v7_s4_best.log`

Outputs:

- `data/experiments/newik1_v7_last_pl_control_lightloss_amass/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.69478097228706 |
| Local SIP | 10.155638122558594 |
| Local Angle | 8.780602645874023 |
| Local Joint | 4.506718969345092 |
| Local Mesh | 5.149347591400146 |
| Global SIP | 10.316577625274657 |
| Global Angle | 8.55061378479004 |
| Global Joint | 4.360230112075806 |
| Global Mesh | 4.9109173774719235 |
| Root Jitter | 0.2785334773361683 |
| Joint Jitter | 0.4653885647654533 |

### Orchestrator Task: v7_s4_last

Name: Evaluate v7 last full S4

Status: completed

Type: eval

Start: 2026-06-05T19:55:02

End: 2026-06-05T20:02:24

GPU: CPU

PID: 1101446

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/last.pt --imu-input-mode processed --output-json data/experiments/newik1_v7_last_pl_control_lightloss_amass/s4/last/result.json`

Log: `logs/orchestrator/newik1_v7_last_pl_control_lightloss_amass/v7_s4_last.log`

Outputs:

- `data/experiments/newik1_v7_last_pl_control_lightloss_amass/s4/last/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.88527706754208 |
| Local SIP | 10.148862171173096 |
| Local Angle | 8.850164318084717 |
| Local Joint | 4.532132577896118 |
| Local Mesh | 5.171209669113159 |
| Global SIP | 10.334029293060302 |
| Global Angle | 8.633635902404786 |
| Global Joint | 4.606380844116211 |
| Global Mesh | 5.145627117156982 |
| Root Jitter | 0.2841707460582256 |
| Joint Jitter | 0.473404061794281 |

## EXP-newik1_v8_parallel_adaptive_loss_search

### Orchestrator Task: v8_B3_pRJ_half_s4_best_loss

Name: Full S4 v8_B3_pRJ_half best_loss

Status: completed

Type: eval

Start: 2026-06-06T01:31:33

End: 2026-06-06T01:39:35

GPU: 1

PID: 1217135

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B3_pRJ_half/train/best_loss.pt --imu-input-mode processed --output-json data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B3_pRJ_half/s4/best_loss/result.json`

Log: `logs/orchestrator/newik1_v8_parallel_adaptive_loss_search/v8_B3_pRJ_half/s4_best_loss.log`

Outputs:

- `data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B3_pRJ_half/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.694600900232786 |
| Local SIP | 10.155586242675781 |
| Local Angle | 8.780542182922364 |
| Local Joint | 4.506685066223144 |
| Local Mesh | 5.149299621582031 |
| Global SIP | 10.316560840606689 |
| Global Angle | 8.550568962097168 |
| Global Joint | 4.360202789306641 |
| Global Mesh | 4.910870552062988 |
| Root Jitter | 0.27853422090411184 |
| Joint Jitter | 0.4653886377811432 |

### Orchestrator Task: v8_B3_pRJ_half_s4_last

Name: Full S4 v8_B3_pRJ_half last

Status: completed

Type: eval

Start: 2026-06-06T02:22:51

End: 2026-06-06T02:31:54

GPU: 1

PID: 1266021

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B3_pRJ_half/train/last.pt --imu-input-mode processed --output-json data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B3_pRJ_half/s4/last/result.json`

Log: `logs/orchestrator/newik1_v8_parallel_adaptive_loss_search/v8_B3_pRJ_half/s4_last.log`

Outputs:

- `data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B3_pRJ_half/s4/last/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.69430113260448 |
| Local SIP | 10.155500221252442 |
| Local Angle | 8.78043270111084 |
| Local Joint | 4.5066171169281 |
| Local Mesh | 5.149208354949951 |
| Global SIP | 10.316533279418945 |
| Global Angle | 8.550502490997314 |
| Global Joint | 4.360168313980102 |
| Global Mesh | 4.910812902450561 |
| Root Jitter | 0.27853319197893145 |
| Joint Jitter | 0.46538967341184617 |

### Orchestrator Task: v8_B4_pRJ_x2_lowdyn_s4_best_loss

Name: Full S4 v8_B4_pRJ_x2_lowdyn best_loss

Status: completed

Type: eval

Start: 2026-06-06T03:46:18

End: 2026-06-06T03:54:20

GPU: 1

PID: 1333429

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/best_loss.pt --imu-input-mode processed --output-json data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/s4/best_loss/result.json`

Log: `logs/orchestrator/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/s4_best_loss.log`

Outputs:

- `data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.6945438311696 |
| Local SIP | 10.155562114715575 |
| Local Angle | 8.780537319183349 |
| Local Joint | 4.506679582595825 |
| Local Mesh | 5.149294757843018 |
| Global SIP | 10.316538906097412 |
| Global Angle | 8.550564575195313 |
| Global Joint | 4.360190773010254 |
| Global Mesh | 4.910859870910644 |
| Root Jitter | 0.27853341698646544 |
| Joint Jitter | 0.46538804173469545 |

### Orchestrator Task: v8_B4_pRJ_x2_lowdyn_s4_last

Name: Full S4 v8_B4_pRJ_x2_lowdyn last

Status: completed

Type: eval

Start: 2026-06-06T04:59:39

End: 2026-06-06T05:07:41

GPU: 1

PID: 1390380

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --imu-input-mode processed --output-json data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/s4/last/result.json`

Log: `logs/orchestrator/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/s4_last.log`

Outputs:

- `data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/s4/last/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.69415222530066 |
| Local SIP | 10.155435752868652 |
| Local Angle | 8.780413722991943 |
| Local Joint | 4.506601095199585 |
| Local Mesh | 5.149188375473022 |
| Global SIP | 10.31648416519165 |
| Global Angle | 8.550490474700927 |
| Global Joint | 4.360141086578369 |
| Global Mesh | 4.910782766342163 |
| Root Jitter | 0.2785340346395969 |
| Joint Jitter | 0.4653891369700432 |

## EXP-newik1_v9_adaptive_loss_search

### Orchestrator Task: v9_C1_pRJ_x3_lowdyn_train

Name: v9_C1_pRJ_x3_lowdyn_train

Status: completed

Type: train

Start: 2026-06-06T08:57:40

End: 2026-06-06T08:59:40

GPU: 1

PID: 1489671

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_train.py --train-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_train/newik1_control_cache_manifest.json --val-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --output-dir data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/train --experiment-name v9_C1_pRJ_x3_lowdyn --epochs 5 --lr 1e-6 --min-lr 2e-7 --warmup-epochs 1 --dropout 0.02 --weight-decay 5e-5 --early-stop-patience 3 --early-stop-min-delta 1e-5 --batch-size 8 --window 61 --init-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --bone-length-weight 0.0 --control-gR2-weight 0.1 --control-gR2-ddot-weight 0.0001 --control-gR2-dot-weight 0.003 --control-pRJ-weight 0.1 --control-pRJ-ddot-weight 0.0001 --control-pRJ-dot-weight 0.003 --control-point-prior-weight 0.0 --gR2-weight 1.0 --gR2-ddot-weight 0.001 --gR2-dot-weight 0.03 --gt-control-gR2-weight 0.0 --gt-control-pRJ-weight 0.0 --pRJ-weight 3.0 --pRJ-ddot-weight 0.0003 --pRJ-dot-weight 0.01 --tail-update-prior-weight 0.0`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/train.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/train/best_loss.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/train/last.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 1 |
| best_loss | 0.19189082384109496 |
| last_epoch | 4 |
| last_val_loss | 0.19188995510339737 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.0 |
| control_gR2 | 0.1 |
| control_gR2_ddot | 0.0001 |
| control_gR2_dot | 0.003 |
| control_pRJ | 0.1 |
| control_pRJ_ddot | 0.0001 |
| control_pRJ_dot | 0.003 |
| control_point_prior | 0.0 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| gt_control_gR2 | 0.0 |
| gt_control_pRJ | 0.0 |
| pRJ | 3.0 |
| pRJ_ddot | 0.0003 |
| pRJ_dot | 0.01 |
| tail_update_prior | 0.0 |

### Orchestrator Task: v9_C2_pRJ_x2_gR2_x2_train

Name: v9_C2_pRJ_x2_gR2_x2_train

Status: completed

Type: train

Start: 2026-06-06T08:59:40

End: 2026-06-06T09:01:41

GPU: 1

PID: 1491704

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_train.py --train-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_train/newik1_control_cache_manifest.json --val-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --output-dir data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/train --experiment-name v9_C2_pRJ_x2_gR2_x2 --epochs 5 --lr 1e-6 --min-lr 2e-7 --warmup-epochs 1 --dropout 0.02 --weight-decay 5e-5 --early-stop-patience 3 --early-stop-min-delta 1e-5 --batch-size 8 --window 61 --init-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --bone-length-weight 0.0 --control-gR2-weight 0.1 --control-gR2-ddot-weight 0.0001 --control-gR2-dot-weight 0.003 --control-pRJ-weight 0.1 --control-pRJ-ddot-weight 0.0001 --control-pRJ-dot-weight 0.003 --control-point-prior-weight 0.0 --gR2-weight 2.0 --gR2-ddot-weight 0.001 --gR2-dot-weight 0.03 --gt-control-gR2-weight 0.0 --gt-control-pRJ-weight 0.0 --pRJ-weight 2.0 --pRJ-ddot-weight 0.0003 --pRJ-dot-weight 0.01 --tail-update-prior-weight 0.0`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/train.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/train/best_loss.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/train/last.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 1 |
| best_loss | 0.3631203666329384 |
| last_epoch | 4 |
| last_val_loss | 0.3631190299987793 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.0 |
| control_gR2 | 0.1 |
| control_gR2_ddot | 0.0001 |
| control_gR2_dot | 0.003 |
| control_pRJ | 0.1 |
| control_pRJ_ddot | 0.0001 |
| control_pRJ_dot | 0.003 |
| control_point_prior | 0.0 |
| gR2 | 2.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| gt_control_gR2 | 0.0 |
| gt_control_pRJ | 0.0 |
| pRJ | 2.0 |
| pRJ_ddot | 0.0003 |
| pRJ_dot | 0.01 |
| tail_update_prior | 0.0 |

### Orchestrator Task: v9_C3_gR2_x3_train

Name: v9_C3_gR2_x3_train

Status: completed

Type: train

Start: 2026-06-06T09:01:41

End: 2026-06-06T09:03:42

GPU: 1

PID: 1493940

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_train.py --train-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_train/newik1_control_cache_manifest.json --val-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --output-dir data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/train --experiment-name v9_C3_gR2_x3 --epochs 5 --lr 1e-6 --min-lr 2e-7 --warmup-epochs 1 --dropout 0.02 --weight-decay 5e-5 --early-stop-patience 3 --early-stop-min-delta 1e-5 --batch-size 8 --window 61 --init-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --bone-length-weight 0.0 --control-gR2-weight 0.1 --control-gR2-ddot-weight 0.0001 --control-gR2-dot-weight 0.003 --control-pRJ-weight 0.1 --control-pRJ-ddot-weight 0.0001 --control-pRJ-dot-weight 0.003 --control-point-prior-weight 0.0 --gR2-weight 3.0 --gR2-ddot-weight 0.001 --gR2-dot-weight 0.03 --gt-control-gR2-weight 0.0 --gt-control-pRJ-weight 0.0 --pRJ-weight 2.0 --pRJ-ddot-weight 0.0003 --pRJ-dot-weight 0.01 --tail-update-prior-weight 0.0`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/train.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/train/best_loss.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/train/last.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 1 |
| best_loss | 0.5351893112063408 |
| last_epoch | 4 |
| last_val_loss | 0.535187192261219 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.0 |
| control_gR2 | 0.1 |
| control_gR2_ddot | 0.0001 |
| control_gR2_dot | 0.003 |
| control_pRJ | 0.1 |
| control_pRJ_ddot | 0.0001 |
| control_pRJ_dot | 0.003 |
| control_point_prior | 0.0 |
| gR2 | 3.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| gt_control_gR2 | 0.0 |
| gt_control_pRJ | 0.0 |
| pRJ | 2.0 |
| pRJ_ddot | 0.0003 |
| pRJ_dot | 0.01 |
| tail_update_prior | 0.0 |

### Orchestrator Task: v9_C4_dyn_lower_train

Name: v9_C4_dyn_lower_train

Status: completed

Type: train

Start: 2026-06-06T09:01:41

End: 2026-06-06T09:04:42

GPU: 0

PID: 1493941

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_train.py --train-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_train/newik1_control_cache_manifest.json --val-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --output-dir data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/train --experiment-name v9_C4_dyn_lower --epochs 5 --lr 1e-6 --min-lr 2e-7 --warmup-epochs 1 --dropout 0.02 --weight-decay 5e-5 --early-stop-patience 3 --early-stop-min-delta 1e-5 --batch-size 8 --window 61 --init-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --bone-length-weight 0.0 --control-gR2-weight 0.1 --control-gR2-ddot-weight 0.0001 --control-gR2-dot-weight 0.003 --control-pRJ-weight 0.1 --control-pRJ-ddot-weight 0.0001 --control-pRJ-dot-weight 0.003 --control-point-prior-weight 0.0 --gR2-weight 1.0 --gR2-ddot-weight 0.001 --gR2-dot-weight 0.03 --gt-control-gR2-weight 0.0 --gt-control-pRJ-weight 0.0 --pRJ-weight 2.0 --pRJ-ddot-weight 0.0001 --pRJ-dot-weight 0.003 --tail-update-prior-weight 0.0`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/train.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/train/best_loss.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/train/last.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 1 |
| best_loss | 0.19105132967233657 |
| last_epoch | 4 |
| last_val_loss | 0.19105064868927002 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.0 |
| control_gR2 | 0.1 |
| control_gR2_ddot | 0.0001 |
| control_gR2_dot | 0.003 |
| control_pRJ | 0.1 |
| control_pRJ_ddot | 0.0001 |
| control_pRJ_dot | 0.003 |
| control_point_prior | 0.0 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| gt_control_gR2 | 0.0 |
| gt_control_pRJ | 0.0 |
| pRJ | 2.0 |
| pRJ_ddot | 0.0001 |
| pRJ_dot | 0.003 |
| tail_update_prior | 0.0 |

### Orchestrator Task: v9_C5_dyn_higher_train

Name: v9_C5_dyn_higher_train

Status: completed

Type: train

Start: 2026-06-06T09:03:42

End: 2026-06-06T09:06:42

GPU: 1

PID: 1495558

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_train.py --train-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_train/newik1_control_cache_manifest.json --val-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --output-dir data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/train --experiment-name v9_C5_dyn_higher --epochs 5 --lr 1e-6 --min-lr 2e-7 --warmup-epochs 1 --dropout 0.02 --weight-decay 5e-5 --early-stop-patience 3 --early-stop-min-delta 1e-5 --batch-size 8 --window 61 --init-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --bone-length-weight 0.0 --control-gR2-weight 0.1 --control-gR2-ddot-weight 0.0001 --control-gR2-dot-weight 0.003 --control-pRJ-weight 0.1 --control-pRJ-ddot-weight 0.0001 --control-pRJ-dot-weight 0.003 --control-point-prior-weight 0.0 --gR2-weight 1.0 --gR2-ddot-weight 0.001 --gR2-dot-weight 0.03 --gt-control-gR2-weight 0.0 --gt-control-pRJ-weight 0.0 --pRJ-weight 2.0 --pRJ-ddot-weight 0.001 --pRJ-dot-weight 0.03 --tail-update-prior-weight 0.0`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/train.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/train/best_loss.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/train/last.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 1 |
| best_loss | 0.19105171039700508 |
| last_epoch | 4 |
| last_val_loss | 0.19105093777179719 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.0 |
| control_gR2 | 0.1 |
| control_gR2_ddot | 0.0001 |
| control_gR2_dot | 0.003 |
| control_pRJ | 0.1 |
| control_pRJ_ddot | 0.0001 |
| control_pRJ_dot | 0.003 |
| control_point_prior | 0.0 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| gt_control_gR2 | 0.0 |
| gt_control_pRJ | 0.0 |
| pRJ | 2.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |
| tail_update_prior | 0.0 |

### Orchestrator Task: v9_C6_control_x2_train

Name: v9_C6_control_x2_train

Status: completed

Type: train

Start: 2026-06-06T09:06:43

End: 2026-06-06T09:08:43

GPU: 1

PID: 1497500

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_train.py --train-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_train/newik1_control_cache_manifest.json --val-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --output-dir data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/train --experiment-name v9_C6_control_x2 --epochs 5 --lr 1e-6 --min-lr 2e-7 --warmup-epochs 1 --dropout 0.02 --weight-decay 5e-5 --early-stop-patience 3 --early-stop-min-delta 1e-5 --batch-size 8 --window 61 --init-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --bone-length-weight 0.0 --control-gR2-weight 0.2 --control-gR2-ddot-weight 0.0001 --control-gR2-dot-weight 0.003 --control-pRJ-weight 0.2 --control-pRJ-ddot-weight 0.0001 --control-pRJ-dot-weight 0.003 --control-point-prior-weight 0.0 --gR2-weight 1.0 --gR2-ddot-weight 0.001 --gR2-dot-weight 0.03 --gt-control-gR2-weight 0.0 --gt-control-pRJ-weight 0.0 --pRJ-weight 2.0 --pRJ-ddot-weight 0.0003 --pRJ-dot-weight 0.01 --tail-update-prior-weight 0.0`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C6_control_x2/train.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/train/best_loss.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/train/last.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 1 |
| best_loss | 0.20835152715444566 |
| last_epoch | 4 |
| last_val_loss | 0.20835062935948373 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.0 |
| control_gR2 | 0.2 |
| control_gR2_ddot | 0.0001 |
| control_gR2_dot | 0.003 |
| control_pRJ | 0.2 |
| control_pRJ_ddot | 0.0001 |
| control_pRJ_dot | 0.003 |
| control_point_prior | 0.0 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| gt_control_gR2 | 0.0 |
| gt_control_pRJ | 0.0 |
| pRJ | 2.0 |
| pRJ_ddot | 0.0003 |
| pRJ_dot | 0.01 |
| tail_update_prior | 0.0 |

### Orchestrator Task: v9_C7_control_dyn_x2_train

Name: v9_C7_control_dyn_x2_train

Status: completed

Type: train

Start: 2026-06-06T09:08:43

End: 2026-06-06T09:10:44

GPU: 1

PID: 1499268

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_train.py --train-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_train/newik1_control_cache_manifest.json --val-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --output-dir data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/train --experiment-name v9_C7_control_dyn_x2 --epochs 5 --lr 1e-6 --min-lr 2e-7 --warmup-epochs 1 --dropout 0.02 --weight-decay 5e-5 --early-stop-patience 3 --early-stop-min-delta 1e-5 --batch-size 8 --window 61 --init-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --bone-length-weight 0.0 --control-gR2-weight 0.1 --control-gR2-ddot-weight 0.0002 --control-gR2-dot-weight 0.006 --control-pRJ-weight 0.1 --control-pRJ-ddot-weight 0.0002 --control-pRJ-dot-weight 0.006 --control-point-prior-weight 0.0 --gR2-weight 1.0 --gR2-ddot-weight 0.001 --gR2-dot-weight 0.03 --gt-control-gR2-weight 0.0 --gt-control-pRJ-weight 0.0 --pRJ-weight 2.0 --pRJ-ddot-weight 0.0003 --pRJ-dot-weight 0.01 --tail-update-prior-weight 0.0`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/train.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/train/best_loss.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/train/last.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 1 |
| best_loss | 0.19105183556675912 |
| last_epoch | 4 |
| last_val_loss | 0.19105110988020896 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.0 |
| control_gR2 | 0.1 |
| control_gR2_ddot | 0.0002 |
| control_gR2_dot | 0.006 |
| control_pRJ | 0.1 |
| control_pRJ_ddot | 0.0002 |
| control_pRJ_dot | 0.006 |
| control_point_prior | 0.0 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| gt_control_gR2 | 0.0 |
| gt_control_pRJ | 0.0 |
| pRJ | 2.0 |
| pRJ_ddot | 0.0003 |
| pRJ_dot | 0.01 |
| tail_update_prior | 0.0 |

### Orchestrator Task: v9_C8_no_control_dyn_train

Name: v9_C8_no_control_dyn_train

Status: completed

Type: train

Start: 2026-06-06T09:10:44

End: 2026-06-06T09:12:44

GPU: 1

PID: 1500905

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_train.py --train-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_train/newik1_control_cache_manifest.json --val-cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --output-dir data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train --experiment-name v9_C8_no_control_dyn --epochs 5 --lr 1e-6 --min-lr 2e-7 --warmup-epochs 1 --dropout 0.02 --weight-decay 5e-5 --early-stop-patience 3 --early-stop-min-delta 1e-5 --batch-size 8 --window 61 --init-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --bone-length-weight 0.0 --control-gR2-weight 0.1 --control-gR2-ddot-weight 0.0 --control-gR2-dot-weight 0.0 --control-pRJ-weight 0.1 --control-pRJ-ddot-weight 0.0 --control-pRJ-dot-weight 0.0 --control-point-prior-weight 0.0 --gR2-weight 1.0 --gR2-ddot-weight 0.001 --gR2-dot-weight 0.03 --gt-control-gR2-weight 0.0 --gt-control-pRJ-weight 0.0 --pRJ-weight 2.0 --pRJ-ddot-weight 0.0003 --pRJ-dot-weight 0.01 --tail-update-prior-weight 0.0`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/best_loss.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt`
- `data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 1 |
| best_loss | 0.19105083793401717 |
| last_epoch | 4 |
| last_val_loss | 0.19105008095502854 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.0 |
| control_gR2 | 0.1 |
| control_gR2_ddot | 0.0 |
| control_gR2_dot | 0.0 |
| control_pRJ | 0.1 |
| control_pRJ_ddot | 0.0 |
| control_pRJ_dot | 0.0 |
| control_point_prior | 0.0 |
| gR2 | 1.0 |
| gR2_ddot | 0.001 |
| gR2_dot | 0.03 |
| gt_control_gR2 | 0.0 |
| gt_control_pRJ | 0.0 |
| pRJ | 2.0 |
| pRJ_ddot | 0.0003 |
| pRJ_dot | 0.01 |
| tail_update_prior | 0.0 |

### Orchestrator Task: v9_C1_pRJ_x3_lowdyn_module_gt_best_loss

Name: v9_C1_pRJ_x3_lowdyn_module_gt_best_loss

Status: completed

Type: audit

Start: 2026-06-06T09:12:45

End: 2026-06-06T09:13:45

GPU: 1

PID: 1502730

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/train/best_loss.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/module_gt/best_loss/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/module_gt_best_loss.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/module_gt/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/train/best_loss.pt |
| status | ok |

### Orchestrator Task: v9_C1_pRJ_x3_lowdyn_module_gt_last

Name: v9_C1_pRJ_x3_lowdyn_module_gt_last

Status: completed

Type: audit

Start: 2026-06-06T09:13:45

End: 2026-06-06T09:14:45

GPU: 1

PID: 1503544

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/train/last.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/module_gt/last/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/module_gt_last.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/module_gt/last/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/train/last.pt |
| status | ok |

### Orchestrator Task: v9_C2_pRJ_x2_gR2_x2_module_gt_best_loss

Name: v9_C2_pRJ_x2_gR2_x2_module_gt_best_loss

Status: completed

Type: audit

Start: 2026-06-06T09:14:45

End: 2026-06-06T09:15:45

GPU: 1

PID: 1504258

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/train/best_loss.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/module_gt/best_loss/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/module_gt_best_loss.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/module_gt/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/train/best_loss.pt |
| status | ok |

### Orchestrator Task: v9_C2_pRJ_x2_gR2_x2_module_gt_last

Name: v9_C2_pRJ_x2_gR2_x2_module_gt_last

Status: completed

Type: audit

Start: 2026-06-06T09:15:46

End: 2026-06-06T09:16:46

GPU: 1

PID: 1504999

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/train/last.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/module_gt/last/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/module_gt_last.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/module_gt/last/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C2_pRJ_x2_gR2_x2/train/last.pt |
| status | ok |

### Orchestrator Task: v9_C3_gR2_x3_module_gt_best_loss

Name: v9_C3_gR2_x3_module_gt_best_loss

Status: completed

Type: audit

Start: 2026-06-06T09:16:46

End: 2026-06-06T09:17:46

GPU: 1

PID: 1505425

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/train/best_loss.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/module_gt/best_loss/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/module_gt_best_loss.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/module_gt/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/train/best_loss.pt |
| status | ok |

### Orchestrator Task: v9_C3_gR2_x3_module_gt_last

Name: v9_C3_gR2_x3_module_gt_last

Status: completed

Type: audit

Start: 2026-06-06T09:17:46

End: 2026-06-06T09:18:46

GPU: 1

PID: 1505861

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/train/last.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/module_gt/last/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/module_gt_last.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/module_gt/last/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C3_gR2_x3/train/last.pt |
| status | ok |

### Orchestrator Task: v9_C4_dyn_lower_module_gt_best_loss

Name: v9_C4_dyn_lower_module_gt_best_loss

Status: completed

Type: audit

Start: 2026-06-06T09:18:47

End: 2026-06-06T09:19:47

GPU: 1

PID: 1506514

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/train/best_loss.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/module_gt/best_loss/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/module_gt_best_loss.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/module_gt/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/train/best_loss.pt |
| status | ok |

### Orchestrator Task: v9_C4_dyn_lower_module_gt_last

Name: v9_C4_dyn_lower_module_gt_last

Status: completed

Type: audit

Start: 2026-06-06T09:19:47

End: 2026-06-06T09:20:47

GPU: 1

PID: 1507022

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/train/last.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/module_gt/last/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/module_gt_last.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/module_gt/last/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C4_dyn_lower/train/last.pt |
| status | ok |

### Orchestrator Task: v9_C5_dyn_higher_module_gt_best_loss

Name: v9_C5_dyn_higher_module_gt_best_loss

Status: completed

Type: audit

Start: 2026-06-06T09:20:47

End: 2026-06-06T09:21:47

GPU: 1

PID: 1507449

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/train/best_loss.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/module_gt/best_loss/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/module_gt_best_loss.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/module_gt/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/train/best_loss.pt |
| status | ok |

### Orchestrator Task: v9_C5_dyn_higher_module_gt_last

Name: v9_C5_dyn_higher_module_gt_last

Status: completed

Type: audit

Start: 2026-06-06T09:21:47

End: 2026-06-06T09:22:47

GPU: 1

PID: 1507910

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/train/last.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/module_gt/last/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/module_gt_last.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/module_gt/last/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C5_dyn_higher/train/last.pt |
| status | ok |

### Orchestrator Task: v9_C6_control_x2_module_gt_best_loss

Name: v9_C6_control_x2_module_gt_best_loss

Status: completed

Type: audit

Start: 2026-06-06T09:22:48

End: 2026-06-06T09:23:48

GPU: 1

PID: 1508604

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/train/best_loss.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/module_gt/best_loss/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C6_control_x2/module_gt_best_loss.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/module_gt/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/train/best_loss.pt |
| status | ok |

### Orchestrator Task: v9_C6_control_x2_module_gt_last

Name: v9_C6_control_x2_module_gt_last

Status: completed

Type: audit

Start: 2026-06-06T09:23:48

End: 2026-06-06T09:24:48

GPU: 1

PID: 1509031

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/train/last.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/module_gt/last/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C6_control_x2/module_gt_last.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/module_gt/last/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C6_control_x2/train/last.pt |
| status | ok |

### Orchestrator Task: v9_C7_control_dyn_x2_module_gt_best_loss

Name: v9_C7_control_dyn_x2_module_gt_best_loss

Status: completed

Type: audit

Start: 2026-06-06T09:24:48

End: 2026-06-06T09:25:49

GPU: 1

PID: 1509459

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/train/best_loss.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/module_gt/best_loss/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/module_gt_best_loss.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/module_gt/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/train/best_loss.pt |
| status | ok |

### Orchestrator Task: v9_C7_control_dyn_x2_module_gt_last

Name: v9_C7_control_dyn_x2_module_gt_last

Status: completed

Type: audit

Start: 2026-06-06T09:25:49

End: 2026-06-06T09:26:49

GPU: 1

PID: 1510117

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/train/last.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/module_gt/last/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/module_gt_last.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/module_gt/last/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C7_control_dyn_x2/train/last.pt |
| status | ok |

### Orchestrator Task: v9_C8_no_control_dyn_module_gt_best_loss

Name: v9_C8_no_control_dyn_module_gt_best_loss

Status: completed

Type: audit

Start: 2026-06-06T09:25:49

End: 2026-06-06T09:26:49

GPU: 0

PID: 1510119

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/best_loss.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/module_gt/best_loss/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/module_gt_best_loss.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/module_gt/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/best_loss.pt |
| status | ok |

### Orchestrator Task: v9_C8_no_control_dyn_module_gt_last

Name: v9_C8_no_control_dyn_module_gt_last

Status: completed

Type: audit

Start: 2026-06-06T09:26:49

End: 2026-06-06T09:27:49

GPU: 1

PID: 1510743

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_local_diagnostic.py --cache data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/module_gt/last/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/module_gt_last.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/module_gt/last/result.json`

Summary:

| metric | value |
|---|---:|
| cache | data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json |
| contract | Compare cache ik1_base (official baseline IK1 under NewPL decoded input) and NewIK1 checkpoint output against ik1_target GT. |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt |
| status | ok |

### Orchestrator Task: v9_C1_pRJ_x3_lowdyn_s4_best_loss

Name: v9_C1_pRJ_x3_lowdyn_s4_best_loss

Status: completed

Type: eval

Start: 2026-06-06T09:27:49

End: 2026-06-06T09:34:51

GPU: 1

PID: 1511173

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/train/best_loss.pt --imu-input-mode processed --output-json data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/s4/best_loss/result.json`

Log: `logs/orchestrator/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/s4_best_loss.log`

Outputs:

- `data/experiments/newik1_v9_adaptive_loss_search/v9_C1_pRJ_x3_lowdyn/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.69406219679117 |
| Local SIP | 10.155419158935548 |
| Local Angle | 8.78037748336792 |
| Local Joint | 4.506581497192383 |
| Local Mesh | 5.1491601943969725 |
| Global SIP | 10.316485118865966 |
| Global Angle | 8.550456237792968 |
| Global Joint | 4.360121488571167 |
| Global Mesh | 4.910750198364258 |
| Root Jitter | 0.27853571325540544 |
| Joint Jitter | 0.4653899252414703 |

## EXP-newik1_v9_adaptive_loss_search-final-summary

Date: 2026-06-06

Purpose: finalize `newik1_v9_adaptive_loss_search` after all 8 trials completed `best_loss.pt` and `last.pt` evaluation.

Summary:

- Training completed for all 8 v9 trials from `data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt`.
- Official S4 full-pipeline streaming evaluation completed for 16/16 checkpoints.
- Module GT evaluation completed for 16/16 checkpoints on NewPL init36 PL-streaming TotalCapture validation cache:
  `data/experiments/newik1_last_pl_control_20260605_v2/caches/pl_streaming_tc_val/newik1_control_cache_manifest.json`.
- `newik1_local_diagnostic.py` was updated and module audits were recomputed to include `leaf_pRJ` metrics in addition to full `pRJ`, `gR2`, and derivative metrics.
- `scripts/newik1_v9_summarize_trials.py` was updated to include `pRJ_cm_l2_delta`, `leaf_pRJ_cm_l2_delta`, `gR2_angle_delta`, and derivative deltas.
- Remote S4 scripts finished without remaining remote eval processes:
  - `data/experiments/newik1_v9_adaptive_loss_search/remote_s4_scripts/run_remote_s4_gpu0.sh`
  - `data/experiments/newik1_v9_adaptive_loss_search/remote_s4_scripts/run_remote_s4_gpu1.sh`

Final summary JSON:

- `data/experiments/newik1_v9_adaptive_loss_search/summary/phase1_ranking.json`

Best v9 checkpoint:

| Field | Value |
|---|---:|
| Trial | `v9_C8_no_control_dyn` |
| Checkpoint | `last.pt` |
| S4 score | `38.693844566687936` |
| Delta vs v8 best | `-0.000307658612724` |
| Delta vs NewPL init36 | `+0.068187083885071` |
| pRJ L2 cm delta vs official IK1 | `+0.014219195553042852` |
| leaf pRJ L2 cm delta vs official IK1 | `+0.004263760473665279` |
| gR2 angle deg delta vs official IK1 | `-0.005290929126193333` |
| state L2 delta vs official IK1 | `+0.00007313516150815602` |

Best v9 official S4 11 metrics:

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v9_C8_no_control_dyn last | `38.693844566687936` | `10.15534267425537` | `8.780296516418456` | `4.506520128250122` | `5.149078750610352` | `10.316459560394287` | `8.55042896270752` | `4.360109329223633` | `4.910724544525147` | `0.27853641733527185` | `0.46539071649312974` |

Conclusion:

- v9 is not selected as mainline.
- Best v9 slightly improves S4 over v8, but it does not improve standalone IK1 module output versus official IK1 under NewPL streaming input.
- Full pRJ and leaf-pRJ position errors are worse than official IK1 baseline; gR2 angle and some derivative metrics are slightly better.
- Future IK1 work should not continue scaling this same micro-finetune recipe unless a new mechanism directly improves full/leaf pRJ GT error as well as S4.

## EXP-ik1-auto-search-round0-launch

Date: 2026-06-06

Goal: fixed `processed IMU + newpl_v4_init36` IK1 search. Final selection must use S4/S5 full-pipeline 11 metrics; S4/S5 real streaming IK1 output vs GT is diagnostic; AMASS/cache module metrics are diagnostic only.

Baseline:

- PL upstream: `newpl_v4_init36`
- PL checkpoint: `data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt`
- IK1 baseline: official IK1
- S4 Score: `38.625657482802865`

Artifacts created:

- Queue: `experiments/ik1_auto_search_queue.yaml`
- Results CSV: `experiments/ik1_auto_search_results.csv`
- Queue state: `data/experiments/orchestrator_states/ik1_auto_search_queue.json`
- Logs: `logs/orchestrator/ik1_auto_search/round0/*.log`
- New Round 0 JSON root: `data/experiments/ik1_auto_search/round0/`

Seed checkpoint status:

| Seed | Status | Checkpoint |
|---|---|---|
| baseline_official_ik1 | official | official IK1 |
| newik1_v4_official_input | found | `data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt` |
| newik1_v6_stage_a | found | `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt` |
| newik1_v7_best | found | `data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt` |
| newik1_v8_B4_last | found | `data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt` |
| newik1_v9_C8_last | found | `data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt` |

Dry-run result:

- Task count: 14
- Ready tasks: 14
- GPU assignment: explicit local `node01` GPU0/GPU1 via `CUDA_VISIBLE_DEVICES`.
- Conflicts: none.
- Existing output/log conflicts: none.

Known limitation:

- The second server's host/SSH configuration was not verified in this turn, so this launch uses the two local GPUs only. Do not invent remote execution records; add the second server to the queue only after the server name, project path, environment path, and GPU status are verified.

### Orchestrator Task: round1_train_v10_residual_pRJ_only_alpha025_from_v6a

Name: Round1 train v10_residual_pRJ_only_alpha025_from_v6a

Status: failed

Type: train

Start: 2026-06-06T19:44:22

End: 2026-06-06T19:44:38

GPU: 0

PID: 1917332

Return code: 1

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train --experiment-name v10_residual_pRJ_only_alpha025_from_v6a --epochs 3 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.25 --pRJ-weight 2.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --ik1-distill-pRJ-weight 0.5 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 0.1`

Log: `logs/orchestrator/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train/last.pt`
- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train/train_result.json`

Missing outputs:

- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train/last.pt`
- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train/train_result.json`

Blocked downstream tasks:

- `round1_s4_v10_residual_pRJ_only_alpha025_from_v6a`
- `round1_real_s4_v10_residual_pRJ_only_alpha025_from_v6a`

Error: missing outputs: ['data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt', 'data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train/last.pt', 'data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train/train_result.json']

Summary:

Task failed.

Log tail:

```text
# task_id=round1_train_v10_residual_pRJ_only_alpha025_from_v6a
# experiment_id=round1_train_v10_residual_pRJ_only_alpha025_from_v6a
# server=node01
# start=2026-06-06T19:44:22
# gpu=0
# checkpoint_path=data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt
# json_path=data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt
# command=ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha025_from_v6a/train --experiment-name v10_residual_pRJ_only_alpha025_from_v6a --epochs 3 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.25 --pRJ-weight 2.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --ik1-distill-pRJ-weight 0.5 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 0.1

Traceback (most recent call last):
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 343, in <module>
    main()
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 293, in main
    loss, comps = run_sequence(model, batch, weights, output_mode=args.output_mode, residual_alpha=args.residual_alpha)
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 193, in run_sequence
    loss, losses = official_ik1_loss(pred, target, base, weights, base_feature=features, RRB_after_pl=rrb_after_pl)
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 159, in official_ik1_loss
    ik2_input_feature(pred, base_feature.to(pred.device, pred.dtype), RRB_after_pl=RRB_after_pl),
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 139, in ik2_input_feature
    RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).unsqueeze(-3).matmul(RRB_after_pl)
RuntimeError: The size of tensor a (976) must match the size of tensor b (16) at non-singleton dimension 1
```

### Orchestrator Task: round1_train_v10_residual_pRJ_only_alpha05_from_v6a

Name: Round1 train v10_residual_pRJ_only_alpha05_from_v6a

Status: failed

Type: train

Start: 2026-06-06T19:44:22

End: 2026-06-06T19:44:38

GPU: 1

PID: 1917333

Return code: 1

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train --experiment-name v10_residual_pRJ_only_alpha05_from_v6a --epochs 3 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.5 --pRJ-weight 2.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --ik1-distill-pRJ-weight 0.5 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 0.1`

Log: `logs/orchestrator/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train/last.pt`
- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train/train_result.json`

Missing outputs:

- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train/last.pt`
- `data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train/train_result.json`

Blocked downstream tasks:

- `round1_s4_v10_residual_pRJ_only_alpha05_from_v6a`
- `round1_real_s4_v10_residual_pRJ_only_alpha05_from_v6a`

Error: missing outputs: ['data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt', 'data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train/last.pt', 'data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train/train_result.json']

Summary:

Task failed.

Log tail:

```text
# task_id=round1_train_v10_residual_pRJ_only_alpha05_from_v6a
# experiment_id=round1_train_v10_residual_pRJ_only_alpha05_from_v6a
# server=node01
# start=2026-06-06T19:44:22
# gpu=1
# checkpoint_path=data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt
# json_path=data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt
# command=ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1/v10_residual_pRJ_only_alpha05_from_v6a/train --experiment-name v10_residual_pRJ_only_alpha05_from_v6a --epochs 3 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.5 --pRJ-weight 2.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --ik1-distill-pRJ-weight 0.5 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 0.1

Traceback (most recent call last):
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 343, in <module>
    main()
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 293, in main
    loss, comps = run_sequence(model, batch, weights, output_mode=args.output_mode, residual_alpha=args.residual_alpha)
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 193, in run_sequence
    loss, losses = official_ik1_loss(pred, target, base, weights, base_feature=features, RRB_after_pl=rrb_after_pl)
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 159, in official_ik1_loss
    ik2_input_feature(pred, base_feature.to(pred.device, pred.dtype), RRB_after_pl=RRB_after_pl),
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 139, in ik2_input_feature
    RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).unsqueeze(-3).matmul(RRB_after_pl)
RuntimeError: The size of tensor a (976) must match the size of tensor b (16) at non-singleton dimension 1
```

### Orchestrator Task: round1_train_v10_stage_a_low_lr_distill_official

Name: Round1 train v10_stage_a_low_lr_distill_official

Status: failed

Type: train

Start: 2026-06-06T19:44:38

End: 2026-06-06T19:44:53

GPU: 0

PID: 1917677

Return code: 1

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train --experiment-name v10_stage_a_low_lr_distill_official --epochs 3 --lr 1e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode full --residual-alpha 1.0 --pRJ-weight 1.0 --gR2-weight 0.5 --pRJ-dot-weight 0.02 --gR2-dot-weight 0.01 --pRJ-ddot-weight 0.001 --gR2-ddot-weight 0.0005 --ik1-distill-pRJ-weight 1.0 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 0.1`

Log: `logs/orchestrator/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train/last.pt`
- `data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train/train_result.json`

Missing outputs:

- `data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train/last.pt`
- `data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train/train_result.json`

Blocked downstream tasks:

- `round1_s4_v10_stage_a_low_lr_distill_official`
- `round1_real_s4_v10_stage_a_low_lr_distill_official`

Error: missing outputs: ['data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train/best_loss.pt', 'data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train/last.pt', 'data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train/train_result.json']

Summary:

Task failed.

Log tail:

```text
# task_id=round1_train_v10_stage_a_low_lr_distill_official
# experiment_id=round1_train_v10_stage_a_low_lr_distill_official
# server=node01
# start=2026-06-06T19:44:38
# gpu=0
# checkpoint_path=data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt
# json_path=data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train/best_loss.pt
# command=ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1/v10_stage_a_low_lr_distill_official/train --experiment-name v10_stage_a_low_lr_distill_official --epochs 3 --lr 1e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode full --residual-alpha 1.0 --pRJ-weight 1.0 --gR2-weight 0.5 --pRJ-dot-weight 0.02 --gR2-dot-weight 0.01 --pRJ-ddot-weight 0.001 --gR2-ddot-weight 0.0005 --ik1-distill-pRJ-weight 1.0 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 0.1

Traceback (most recent call last):
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 343, in <module>
    main()
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 293, in main
    loss, comps = run_sequence(model, batch, weights, output_mode=args.output_mode, residual_alpha=args.residual_alpha)
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 193, in run_sequence
    loss, losses = official_ik1_loss(pred, target, base, weights, base_feature=features, RRB_after_pl=rrb_after_pl)
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 159, in official_ik1_loss
    ik2_input_feature(pred, base_feature.to(pred.device, pred.dtype), RRB_after_pl=RRB_after_pl),
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 139, in ik2_input_feature
    RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).unsqueeze(-3).matmul(RRB_after_pl)
RuntimeError: The size of tensor a (976) must match the size of tensor b (16) at non-singleton dimension 1
```

### Orchestrator Task: round1_train_v10_ik2_input_distill_from_v6a

Name: Round1 train v10_ik2_input_distill_from_v6a

Status: failed

Type: train

Start: 2026-06-06T19:44:38

End: 2026-06-06T19:44:53

GPU: 1

PID: 1917678

Return code: 1

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train --experiment-name v10_ik2_input_distill_from_v6a --epochs 3 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.25 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --ik1-distill-pRJ-weight 0.5 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train/last.pt`
- `data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train/train_result.json`

Missing outputs:

- `data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train/last.pt`
- `data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train/train_result.json`

Blocked downstream tasks:

- `round1_s4_v10_ik2_input_distill_from_v6a`
- `round1_real_s4_v10_ik2_input_distill_from_v6a`

Error: missing outputs: ['data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train/best_loss.pt', 'data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train/last.pt', 'data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train/train_result.json']

Summary:

Task failed.

Log tail:

```text
# task_id=round1_train_v10_ik2_input_distill_from_v6a
# experiment_id=round1_train_v10_ik2_input_distill_from_v6a
# server=node01
# start=2026-06-06T19:44:38
# gpu=1
# checkpoint_path=data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt
# json_path=data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train/best_loss.pt
# command=ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1/v10_ik2_input_distill_from_v6a/train --experiment-name v10_ik2_input_distill_from_v6a --epochs 3 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.25 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --ik1-distill-pRJ-weight 0.5 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0

Traceback (most recent call last):
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 343, in <module>
    main()
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 293, in main
    loss, comps = run_sequence(model, batch, weights, output_mode=args.output_mode, residual_alpha=args.residual_alpha)
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 193, in run_sequence
    loss, losses = official_ik1_loss(pred, target, base, weights, base_feature=features, RRB_after_pl=rrb_after_pl)
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 159, in official_ik1_loss
    ik2_input_feature(pred, base_feature.to(pred.device, pred.dtype), RRB_after_pl=RRB_after_pl),
  File "/home/lingfeng/projects/GlobalposeMy/GlobalPose/newik1_official_input_train.py", line 139, in ik2_input_feature
    RRB_after_ik1 = art.math.from_to_rotation_matrix(gR1, gR2).unsqueeze(-3).matmul(RRB_after_pl)
RuntimeError: The size of tensor a (976) must match the size of tensor b (16) at non-singleton dimension 1
```

### Orchestrator Task: round1_retry1_train_v10_residual_pRJ_only_alpha025_from_v6a

Name: round1_retry1 train v10_residual_pRJ_only_alpha025_from_v6a

Status: completed

Type: train

Start: 2026-06-06T19:50:01

End: 2026-06-06T19:50:16

GPU: 0

PID: 1922771

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train --experiment-name v10_residual_pRJ_only_alpha025_from_v6a --epochs 3 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.25 --pRJ-weight 2.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --ik1-distill-pRJ-weight 0.5 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 0.1`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train/last.pt`
- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0019857322913594544 |
| last_epoch | 3 |
| last_val_loss | 0.0019857322913594544 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.5 |
| ik2_input_distill | 0.1 |
| pRJ | 2.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: round1_retry1_train_v10_residual_pRJ_only_alpha05_from_v6a

Name: round1_retry1 train v10_residual_pRJ_only_alpha05_from_v6a

Status: completed

Type: train

Start: 2026-06-06T19:50:01

End: 2026-06-06T19:50:16

GPU: 1

PID: 1922772

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train --experiment-name v10_residual_pRJ_only_alpha05_from_v6a --epochs 3 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.5 --pRJ-weight 2.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --ik1-distill-pRJ-weight 0.5 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 0.1`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train/last.pt`
- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0014879830414429307 |
| last_epoch | 3 |
| last_val_loss | 0.0014879830414429307 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.5 |
| ik2_input_distill | 0.1 |
| pRJ | 2.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: round1_retry1_s4_v10_residual_pRJ_only_alpha025_from_v6a

Name: round1_retry1 S4 full-pipeline v10_residual_pRJ_only_alpha025_from_v6a

Status: completed

Type: eval

Start: 2026-06-06T19:50:16

End: 2026-06-06T19:56:36

GPU: 0

PID: 1923124

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.401125624060626 |
| Local SIP | 10.069402694702148 |
| Local Angle | 8.727775764465331 |
| Local Joint | 4.470314264297485 |
| Local Mesh | 5.110921478271484 |
| Global SIP | 10.23533363342285 |
| Global Angle | 8.486337184906006 |
| Global Joint | 4.304667139053345 |
| Global Mesh | 4.840811014175415 |
| Root Jitter | 0.28680822253227234 |
| Joint Jitter | 0.47782062292099 |

### Orchestrator Task: round1_retry1_s4_v10_residual_pRJ_only_alpha05_from_v6a

Name: round1_retry1 S4 full-pipeline v10_residual_pRJ_only_alpha05_from_v6a

Status: completed

Type: eval

Start: 2026-06-06T19:50:16

End: 2026-06-06T19:56:52

GPU: 1

PID: 1923125

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.30681182323395 |
| Local SIP | 10.039692497253418 |
| Local Angle | 8.71478910446167 |
| Local Joint | 4.466692161560059 |
| Local Mesh | 5.10838041305542 |
| Global SIP | 10.208867645263672 |
| Global Angle | 8.463436317443847 |
| Global Joint | 4.285636854171753 |
| Global Mesh | 4.80901026725769 |
| Root Jitter | 0.2881957024335861 |
| Joint Jitter | 0.47933572381734846 |

### Orchestrator Task: round1_retry1_real_s4_v10_residual_pRJ_only_alpha025_from_v6a

Name: round1_retry1 S4 real streaming IK1 audit v10_residual_pRJ_only_alpha025_from_v6a

Status: completed

Type: audit

Start: 2026-06-06T19:56:37

End: 2026-06-06T20:01:26

GPU: 0

PID: 1927985

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v10_residual_pRJ_only_alpha025_from_v6a --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.401125624060626 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v10_residual_pRJ_only_alpha025_from_v6a |

### Orchestrator Task: round1_retry1_real_s4_v10_residual_pRJ_only_alpha05_from_v6a

Name: round1_retry1 S4 real streaming IK1 audit v10_residual_pRJ_only_alpha05_from_v6a

Status: completed

Type: audit

Start: 2026-06-06T19:56:52

End: 2026-06-06T20:01:41

GPU: 1

PID: 1928294

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v10_residual_pRJ_only_alpha05_from_v6a --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.30681182323395 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v10_residual_pRJ_only_alpha05_from_v6a |

### Orchestrator Task: round1_retry1_train_v10_stage_a_low_lr_distill_official

Name: round1_retry1 train v10_stage_a_low_lr_distill_official

Status: failed

Type: train

Start: 2026-06-06T20:01:26

End: 2026-06-06T20:01:41

GPU: 0

PID: 1931829

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1_retry1/v10_stage_a_low_lr_distill_official/train --experiment-name v10_stage_a_low_lr_distill_official --epochs 3 --lr 1e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode full --residual-alpha 1.0 --pRJ-weight 1.0 --gR2-weight 0.5 --pRJ-dot-weight 0.02 --gR2-dot-weight 0.01 --pRJ-ddot-weight 0.001 --gR2-ddot-weight 0.0005 --ik1-distill-pRJ-weight 1.0 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 0.1`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_stage_a_low_lr_distill_official/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_stage_a_low_lr_distill_official/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1_retry1/v10_stage_a_low_lr_distill_official/train/last.pt`
- `data/experiments/ik1_auto_search/round1_retry1/v10_stage_a_low_lr_distill_official/train/train_result.json`

Missing outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_stage_a_low_lr_distill_official/train/best_loss.pt`

Blocked downstream tasks:

- `round1_retry1_s4_v10_stage_a_low_lr_distill_official`
- `round1_retry1_real_s4_v10_stage_a_low_lr_distill_official`

Error: missing outputs: ['data/experiments/ik1_auto_search/round1_retry1/v10_stage_a_low_lr_distill_official/train/best_loss.pt']

Summary:

Task failed.

Log tail:

```text
# task_id=round1_retry1_train_v10_stage_a_low_lr_distill_official
# experiment_id=round1_retry1_train_v10_stage_a_low_lr_distill_official
# server=node01
# start=2026-06-06T20:01:26
# gpu=0
# checkpoint_path=data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt
# json_path=data/experiments/ik1_auto_search/round1_retry1/v10_stage_a_low_lr_distill_official/train/best_loss.pt
# command=ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1_retry1/v10_stage_a_low_lr_distill_official/train --experiment-name v10_stage_a_low_lr_distill_official --epochs 3 --lr 1e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode full --residual-alpha 1.0 --pRJ-weight 1.0 --gR2-weight 0.5 --pRJ-dot-weight 0.02 --gR2-dot-weight 0.01 --pRJ-ddot-weight 0.001 --gR2-ddot-weight 0.0005 --ik1-distill-pRJ-weight 1.0 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 0.1

{"epoch": 1, "train": {"pRJ": NaN, "gR2": 0.9761928475804563, "bone_length": NaN, "ik1_distill_pRJ": NaN, "ik1_distill_gR2": 0.976237806341877, "ik2_input_distill": NaN, "pRJ_dot": NaN, "gR2_dot": 7.676399244852897e-06, "pRJ_ddot": NaN, "gR2_ddot": 3.396820177602508e-06, "loss": NaN, "ik1_residual_norm_mean": NaN}, "validation": {"loss": NaN, "pRJ": NaN, "gR2": 1.0, "bone_length": NaN, "ik1_distill_pRJ": NaN, "ik1_distill_gR2": 1.0, "ik2_input_distill": NaN, "pRJ_dot": NaN, "gR2_dot": 4.5684353790420576e-05, "pRJ_ddot": NaN, "gR2_ddot": 6.375209270572668e-05, "ik1_residual_norm_mean": NaN}, "best_loss": Infinity, "best_epoch": 0}
{"epoch": 2, "train": {"pRJ": NaN, "gR2": 0.9999999418145135, "bone_length": NaN, "ik1_distill_pRJ": NaN, "ik1_distill_gR2": 0.9999999418145135, "ik2_input_distill": NaN, "pRJ_dot": NaN, "gR2_dot": 7.584228076969642e-06, "pRJ_ddot": NaN, "gR2_ddot": 2.9426670448087645e-06, "loss": NaN, "ik1_residual_norm_mean": NaN}, "validation": {"loss": NaN, "pRJ": NaN, "gR2": 1.0, "bone_length": NaN, "ik1_distill_pRJ": NaN, "ik1_distill_gR2": 1.0, "ik2_input_distill": NaN, "pRJ_dot": NaN, "gR2_dot": 4.5684353790420576e-05, "pRJ_ddot": NaN, "gR2_ddot": 6.375209270572668e-05, "ik1_residual_norm_mean": NaN}, "best_loss": Infinity, "best_epoch": 0}
{"epoch": 3, "train": {"pRJ": NaN, "gR2": 0.9999999418145135, "bone_length": NaN, "ik1_distill_pRJ": NaN, "ik1_distill_gR2": 0.9999999418145135, "ik2_input_distill": NaN, "pRJ_dot": NaN, "gR2_dot": 7.706706593948876e-06, "pRJ_ddot": NaN, "gR2_ddot": 2.943503794483604e-06, "loss": NaN, "ik1_residual_norm_mean": NaN}, "validation": {"loss": NaN, "pRJ": NaN, "gR2": 1.0, "bone_length": NaN, "ik1_distill_pRJ": NaN, "ik1_distill_gR2": 1.0, "ik2_input_distill": NaN, "pRJ_dot": NaN, "gR2_dot": 4.5684353790420576e-05, "pRJ_ddot": NaN, "gR2_ddot": 6.375209270572668e-05, "ik1_residual_norm_mean": NaN}, "best_loss": Infinity, "best_epoch": 0}
{
  "status": "ok",
  "best_epoch": 0,
  "best_loss": Infinity
}
```

### Orchestrator Task: round1_retry1_train_v10_ik2_input_distill_from_v6a

Name: round1_retry1 train v10_ik2_input_distill_from_v6a

Status: completed

Type: train

Start: 2026-06-06T20:01:41

End: 2026-06-06T20:01:56

GPU: 1

PID: 1932101

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train --experiment-name v10_ik2_input_distill_from_v6a --epochs 3 --lr 3e-6 --dropout 0.15 --batch-size 16 --window 61 --init-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.25 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --ik1-distill-pRJ-weight 0.5 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/last.pt`
- `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.001099719398189336 |
| last_epoch | 3 |
| last_val_loss | 0.001099719398189336 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.5 |
| ik2_input_distill | 2.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.002 |
| pRJ_dot | 0.05 |

### Orchestrator Task: round1_retry1_s4_v10_ik2_input_distill_from_v6a

Name: round1_retry1 S4 full-pipeline v10_ik2_input_distill_from_v6a

Status: completed

Type: eval

Start: 2026-06-06T20:01:56

End: 2026-06-06T20:08:32

GPU: 1

PID: 1932375

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.41680224156379 |
| Local SIP | 10.076177978515625 |
| Local Angle | 8.730329704284667 |
| Local Joint | 4.4710526943206785 |
| Local Mesh | 5.111426067352295 |
| Global SIP | 10.240815353393554 |
| Global Angle | 8.487079906463624 |
| Global Joint | 4.305175113677978 |
| Global Mesh | 4.841847515106201 |
| Root Jitter | 0.2866271585226059 |
| Joint Jitter | 0.4776518106460571 |

### Orchestrator Task: round1_retry1_real_s4_v10_ik2_input_distill_from_v6a

Name: round1_retry1 S4 real streaming IK1 audit v10_ik2_input_distill_from_v6a

Status: completed

Type: audit

Start: 2026-06-06T20:08:32

End: 2026-06-06T20:13:21

GPU: 1

PID: 1937248

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v10_ik2_input_distill_from_v6a --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.41680224156379 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v10_ik2_input_distill_from_v6a |

### Orchestrator Task: round1_retry1_s5_v10_residual_pRJ_only_alpha025_from_v6a

Name: round1_retry1 S5 full-pipeline v10_residual_pRJ_only_alpha025_from_v6a

Status: completed

Type: eval

Start: 2026-06-06T20:14:49

End: 2026-06-06T20:20:54

GPU: 0

PID: 1942059

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.84817816592753 |
| Local SIP | 9.646694779396057 |
| Local Angle | 12.468416452407837 |
| Local Joint | 4.341636061668396 |
| Local Mesh | 5.115289092063904 |
| Global SIP | 9.137168526649475 |
| Global Angle | 11.771339654922485 |
| Global Joint | 3.823734700679779 |
| Global Mesh | 4.422149419784546 |
| Root Jitter | 0.3735490506514907 |
| Joint Jitter | 0.8021676316857338 |

### Orchestrator Task: round1_retry1_s5_v10_residual_pRJ_only_alpha05_from_v6a

Name: round1_retry1 S5 full-pipeline v10_residual_pRJ_only_alpha05_from_v6a

Status: completed

Type: eval

Start: 2026-06-06T20:14:49

End: 2026-06-06T20:21:09

GPU: 1

PID: 1942061

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.867689362633975 |
| Local SIP | 9.642346501350403 |
| Local Angle | 12.483997821807861 |
| Local Joint | 4.327491879463196 |
| Local Mesh | 5.121562480926514 |
| Global SIP | 9.137211203575134 |
| Global Angle | 11.781686544418335 |
| Global Joint | 3.815983235836029 |
| Global Mesh | 4.409284234046936 |
| Root Jitter | 0.3772413032129407 |
| Joint Jitter | 0.8099779952317476 |

### Orchestrator Task: round1_retry1_real_s5_v10_residual_pRJ_only_alpha025_from_v6a

Name: round1_retry1 S5 real streaming IK1 audit v10_residual_pRJ_only_alpha025_from_v6a

Status: completed

Type: audit

Start: 2026-06-06T20:20:54

End: 2026-06-06T20:25:13

GPU: 0

PID: 1946781

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v10_residual_pRJ_only_alpha025_from_v6a --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.84817816592753 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v10_residual_pRJ_only_alpha025_from_v6a |

### Orchestrator Task: round1_retry1_real_s5_v10_residual_pRJ_only_alpha05_from_v6a

Name: round1_retry1 S5 real streaming IK1 audit v10_residual_pRJ_only_alpha05_from_v6a

Status: completed

Type: audit

Start: 2026-06-06T20:21:09

End: 2026-06-06T20:25:28

GPU: 1

PID: 1947092

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v10_residual_pRJ_only_alpha05_from_v6a --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.867689362633975 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v10_residual_pRJ_only_alpha05_from_v6a |

### Orchestrator Task: round1_retry1_s5_v10_ik2_input_distill_from_v6a

Name: round1_retry1 S5 full-pipeline v10_ik2_input_distill_from_v6a

Status: completed

Type: eval

Start: 2026-06-06T20:25:13

End: 2026-06-06T20:31:18

GPU: 0

PID: 1950213

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.807055798713115 |
| Local SIP | 9.637646913528442 |
| Local Angle | 12.45914602279663 |
| Local Joint | 4.340159773826599 |
| Local Mesh | 5.110072493553162 |
| Global SIP | 9.124706625938416 |
| Global Angle | 11.761341333389282 |
| Global Joint | 3.8217774033546448 |
| Global Mesh | 4.420398771762848 |
| Root Jitter | 0.37341867480427027 |
| Joint Jitter | 0.8021185342222452 |

### Orchestrator Task: round1_retry1_real_s5_v10_ik2_input_distill_from_v6a

Name: round1_retry1 S5 real streaming IK1 audit v10_ik2_input_distill_from_v6a

Status: completed

Type: audit

Start: 2026-06-06T20:31:18

End: 2026-06-06T20:36:08

GPU: 0

PID: 1954769

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v10_ik2_input_distill_from_v6a --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.807055798713115 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v10_ik2_input_distill_from_v6a |

### Orchestrator Task: round2_train_v11_alpha025_ik2w1_from_v10

Name: Round2 train v11_alpha025_ik2w1_from_v10

Status: completed

Type: train

Start: 2026-06-06T21:17:21

End: 2026-06-06T21:17:36

GPU: 0

PID: 1978501

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train --experiment-name v11_alpha025_ik2w1_from_v10 --epochs 3 --lr 2e-6 --dropout 0.12 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.25 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.7 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 1.0`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train/last.pt`
- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010799324430990965 |
| last_epoch | 3 |
| last_val_loss | 0.0010799324430990965 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.7 |
| ik2_input_distill | 1.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round2_train_v11_alpha025_ik2w3_from_v10

Name: Round2 train v11_alpha025_ik2w3_from_v10

Status: completed

Type: train

Start: 2026-06-06T21:17:21

End: 2026-06-06T21:17:36

GPU: 1

PID: 1978502

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train --experiment-name v11_alpha025_ik2w3_from_v10 --epochs 3 --lr 2e-6 --dropout 0.12 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.25 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.7 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 3.0`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train/last.pt`
- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0011294037627521901 |
| last_epoch | 3 |
| last_val_loss | 0.0011294037627521901 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.7 |
| ik2_input_distill | 3.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round2_s4_v11_alpha025_ik2w3_from_v10

Name: Round2 S4 full-pipeline v11_alpha025_ik2w3_from_v10

Status: completed

Type: eval

Start: 2026-06-06T21:17:36

End: 2026-06-06T21:23:56

GPU: 1

PID: 1978819

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.416982645660646 |
| Local SIP | 10.077760219573975 |
| Local Angle | 8.729902648925782 |
| Local Joint | 4.47086877822876 |
| Local Mesh | 5.111277294158936 |
| Global SIP | 10.241942882537842 |
| Global Angle | 8.485101509094239 |
| Global Joint | 4.304106378555298 |
| Global Mesh | 4.840069007873535 |
| Root Jitter | 0.2867006339132786 |
| Joint Jitter | 0.477786985039711 |

### Orchestrator Task: round2_s4_v11_alpha025_ik2w1_from_v10

Name: Round2 S4 full-pipeline v11_alpha025_ik2w1_from_v10

Status: completed

Type: eval

Start: 2026-06-06T21:17:36

End: 2026-06-06T21:24:11

GPU: 0

PID: 1978818

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.40251200318336 |
| Local SIP | 10.071526050567627 |
| Local Angle | 8.727390766143799 |
| Local Joint | 4.470222616195679 |
| Local Mesh | 5.1106737613677975 |
| Global SIP | 10.237113285064698 |
| Global Angle | 8.484315109252929 |
| Global Joint | 4.303654050827026 |
| Global Mesh | 4.838948774337768 |
| Root Jitter | 0.28685255348682404 |
| Joint Jitter | 0.4779125452041626 |

### Orchestrator Task: round2_s5_v11_alpha025_ik2w3_from_v10

Name: Round2 S5 full-pipeline v11_alpha025_ik2w3_from_v10

Status: completed

Type: eval

Start: 2026-06-06T21:23:56

End: 2026-06-06T21:30:16

GPU: 1

PID: 1982547

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.80445552650839 |
| Local SIP | 9.638152122497559 |
| Local Angle | 12.459270000457764 |
| Local Joint | 4.340410530567169 |
| Local Mesh | 5.111134886741638 |
| Global SIP | 9.122875571250916 |
| Global Angle | 11.759988784790039 |
| Global Joint | 3.821051597595215 |
| Global Mesh | 4.4197933077812195 |
| Root Jitter | 0.37349481880664825 |
| Joint Jitter | 0.8022834695875645 |

### Orchestrator Task: round2_s5_v11_alpha025_ik2w1_from_v10

Name: Round2 S5 full-pipeline v11_alpha025_ik2w1_from_v10

Status: completed

Type: eval

Start: 2026-06-06T21:24:11

End: 2026-06-06T21:30:16

GPU: 0

PID: 1982859

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.84088326931 |
| Local SIP | 9.646111488342285 |
| Local Angle | 12.467326879501343 |
| Local Joint | 4.3410563468933105 |
| Local Mesh | 5.115470290184021 |
| Global SIP | 9.134477972984314 |
| Global Angle | 11.768590450286865 |
| Global Joint | 3.8224847316741943 |
| Global Mesh | 4.4208972454071045 |
| Root Jitter | 0.37358420994132757 |
| Joint Jitter | 0.8022370338439941 |

### Orchestrator Task: round2_real_s4_v11_alpha025_ik2w3_from_v10

Name: Round2 S4 real streaming IK1 audit v11_alpha025_ik2w3_from_v10

Status: completed

Type: audit

Start: 2026-06-06T21:30:17

End: 2026-06-06T21:34:50

GPU: 1

PID: 1986384

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v11_alpha025_ik2w3_from_v10 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.416982645660646 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v11_alpha025_ik2w3_from_v10 |

### Orchestrator Task: round2_real_s4_v11_alpha025_ik2w1_from_v10

Name: Round2 S4 real streaming IK1 audit v11_alpha025_ik2w1_from_v10

Status: completed

Type: audit

Start: 2026-06-06T21:30:17

End: 2026-06-06T21:35:06

GPU: 0

PID: 1986383

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v11_alpha025_ik2w1_from_v10 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.40251200318336 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v11_alpha025_ik2w1_from_v10 |

### Orchestrator Task: round2_real_s5_v11_alpha025_ik2w3_from_v10

Name: Round2 S5 real streaming IK1 audit v11_alpha025_ik2w3_from_v10

Status: completed

Type: audit

Start: 2026-06-06T21:34:50

End: 2026-06-06T21:39:09

GPU: 1

PID: 1989145

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v11_alpha025_ik2w3_from_v10 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w3_from_v10/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.80445552650839 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v11_alpha025_ik2w3_from_v10 |

### Orchestrator Task: round2_real_s5_v11_alpha025_ik2w1_from_v10

Name: Round2 S5 real streaming IK1 audit v11_alpha025_ik2w1_from_v10

Status: completed

Type: audit

Start: 2026-06-06T21:35:06

End: 2026-06-06T21:39:24

GPU: 0

PID: 1989426

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v11_alpha025_ik2w1_from_v10 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha025_ik2w1_from_v10/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.84088326931 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v11_alpha025_ik2w1_from_v10 |

### Orchestrator Task: round2_train_v11_pRJ_only_ik2w2_from_v10

Name: Round2 train v11_pRJ_only_ik2w2_from_v10

Status: completed

Type: train

Start: 2026-06-06T21:39:09

End: 2026-06-06T21:39:24

GPU: 1

PID: 1991818

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train --experiment-name v11_pRJ_only_ik2w2_from_v10 --epochs 3 --lr 1e-6 --dropout 0.12 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt --output-mode pRJ_only --residual-alpha 1.0 --pRJ-weight 0.8 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 1.0 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train/last.pt`
- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0017022560932673514 |
| last_epoch | 3 |
| last_val_loss | 0.0017022560932673514 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 1.0 |
| ik2_input_distill | 2.0 |
| pRJ | 0.8 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round2_train_v11_alpha035_ik2w2_from_v10

Name: Round2 train v11_alpha035_ik2w2_from_v10

Status: completed

Type: train

Start: 2026-06-06T21:39:24

End: 2026-06-06T21:39:39

GPU: 0

PID: 1992056

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train --experiment-name v11_alpha035_ik2w2_from_v10 --epochs 3 --lr 2e-6 --dropout 0.12 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.35 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.7 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/last.pt`
- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010508300038054585 |
| last_epoch | 3 |
| last_val_loss | 0.0010508300038054585 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.7 |
| ik2_input_distill | 2.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round2_s4_v11_pRJ_only_ik2w2_from_v10

Name: Round2 S4 full-pipeline v11_pRJ_only_ik2w2_from_v10

Status: completed

Type: eval

Start: 2026-06-06T21:39:24

End: 2026-06-06T21:46:00

GPU: 1

PID: 1992057

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.39572076609731 |
| Local SIP | 10.052773666381835 |
| Local Angle | 8.771314716339111 |
| Local Joint | 4.521404552459717 |
| Local Mesh | 5.171199131011963 |
| Global SIP | 10.191976642608642 |
| Global Angle | 8.492659854888917 |
| Global Joint | 4.300215053558349 |
| Global Mesh | 4.7997174739837645 |
| Root Jitter | 0.2938575528562069 |
| Joint Jitter | 0.4833925276994705 |

### Orchestrator Task: round2_s4_v11_alpha035_ik2w2_from_v10

Name: Round2 S4 full-pipeline v11_alpha035_ik2w2_from_v10

Status: completed

Type: eval

Start: 2026-06-06T21:39:40

End: 2026-06-06T21:46:15

GPU: 0

PID: 1992446

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.37255207578838 |
| Local SIP | 10.06448392868042 |
| Local Angle | 8.722952556610107 |
| Local Joint | 4.46697883605957 |
| Local Mesh | 5.1072286605834964 |
| Global SIP | 10.230513000488282 |
| Global Angle | 8.473731422424317 |
| Global Joint | 4.293906354904175 |
| Global Mesh | 4.824944019317627 |
| Root Jitter | 0.2870724491775036 |
| Joint Jitter | 0.47826484888792037 |

### Orchestrator Task: round2_s5_v11_pRJ_only_ik2w2_from_v10

Name: Round2 S5 full-pipeline v11_pRJ_only_ik2w2_from_v10

Status: completed

Type: eval

Start: 2026-06-06T21:46:00

End: 2026-06-06T21:51:50

GPU: 1

PID: 1996087

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.845466667562725 |
| Local SIP | 9.625837564468384 |
| Local Angle | 12.499703884124756 |
| Local Joint | 4.309311866760254 |
| Local Mesh | 5.130159974098206 |
| Global SIP | 9.117349624633789 |
| Global Angle | 11.782392978668213 |
| Global Joint | 3.809548497200012 |
| Global Mesh | 4.3971922397613525 |
| Root Jitter | 0.38753859512507915 |
| Joint Jitter | 0.8296579271554947 |

### Orchestrator Task: round2_s5_v11_alpha035_ik2w2_from_v10

Name: Round2 S5 full-pipeline v11_alpha035_ik2w2_from_v10

Status: completed

Type: eval

Start: 2026-06-06T21:46:15

End: 2026-06-06T21:52:35

GPU: 0

PID: 1996395

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.77819666691124 |
| Local SIP | 9.628082871437073 |
| Local Angle | 12.458468437194824 |
| Local Joint | 4.333025872707367 |
| Local Mesh | 5.108862519264221 |
| Global SIP | 9.112644672393799 |
| Global Angle | 11.756129264831543 |
| Global Joint | 3.8151710629463196 |
| Global Mesh | 4.41172057390213 |
| Root Jitter | 0.3747247625142336 |
| Joint Jitter | 0.805172748863697 |

### Orchestrator Task: round2_real_s4_v11_pRJ_only_ik2w2_from_v10

Name: Round2 S4 real streaming IK1 audit v11_pRJ_only_ik2w2_from_v10

Status: completed

Type: audit

Start: 2026-06-06T21:51:50

End: 2026-06-06T21:56:39

GPU: 1

PID: 1999570

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v11_pRJ_only_ik2w2_from_v10 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.39572076609731 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v11_pRJ_only_ik2w2_from_v10 |

### Orchestrator Task: round2_real_s4_v11_alpha035_ik2w2_from_v10

Name: Round2 S4 real streaming IK1 audit v11_alpha035_ik2w2_from_v10

Status: completed

Type: audit

Start: 2026-06-06T21:52:36

End: 2026-06-06T21:57:25

GPU: 0

PID: 2000168

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v11_alpha035_ik2w2_from_v10 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.37255207578838 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v11_alpha035_ik2w2_from_v10 |

### Orchestrator Task: round2_real_s5_v11_pRJ_only_ik2w2_from_v10

Name: Round2 S5 real streaming IK1 audit v11_pRJ_only_ik2w2_from_v10

Status: completed

Type: audit

Start: 2026-06-06T21:56:39

End: 2026-06-06T22:00:58

GPU: 1

PID: 2002516

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v11_pRJ_only_ik2w2_from_v10 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_pRJ_only_ik2w2_from_v10/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.845466667562725 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v11_pRJ_only_ik2w2_from_v10 |

### Orchestrator Task: round2_real_s5_v11_alpha035_ik2w2_from_v10

Name: Round2 S5 real streaming IK1 audit v11_alpha035_ik2w2_from_v10

Status: completed

Type: audit

Start: 2026-06-06T21:57:25

End: 2026-06-06T22:01:58

GPU: 0

PID: 2003071

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v11_alpha035_ik2w2_from_v10 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.77819666691124 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v11_alpha035_ik2w2_from_v10 |

### Orchestrator Task: round3_train_v12_alpha040_ik2w2_from_v11

Name: Round3 train v12_alpha040_ik2w2_from_v11

Status: completed

Type: train

Start: 2026-06-06T22:06:54

End: 2026-06-06T22:07:09

GPU: 1

PID: 2011132

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train --experiment-name v12_alpha040_ik2w2_from_v11 --epochs 3 --lr 1.5e-6 --dropout 0.10 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.4 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/last.pt`
- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.001047273859148845 |
| last_epoch | 3 |
| last_val_loss | 0.001047273859148845 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.8 |
| ik2_input_distill | 2.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round3_train_v12_alpha030_ik2w2_from_v11

Name: Round3 train v12_alpha030_ik2w2_from_v11

Status: completed

Type: train

Start: 2026-06-06T22:06:54

End: 2026-06-06T22:07:25

GPU: 0

PID: 2011131

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train --experiment-name v12_alpha030_ik2w2_from_v11 --epochs 3 --lr 1.5e-6 --dropout 0.10 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.3 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train/last.pt`
- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010780938668176533 |
| last_epoch | 3 |
| last_val_loss | 0.0010780938668176533 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.8 |
| ik2_input_distill | 2.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round3_s4_v12_alpha040_ik2w2_from_v11

Name: Round3 S4 full-pipeline v12_alpha040_ik2w2_from_v11

Status: completed

Type: eval

Start: 2026-06-06T22:07:10

End: 2026-06-06T22:13:45

GPU: 1

PID: 2011349

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.36728921282291 |
| Local SIP | 10.064269065856934 |
| Local Angle | 8.722883987426759 |
| Local Joint | 4.466901874542236 |
| Local Mesh | 5.1075506687164305 |
| Global SIP | 10.229679298400878 |
| Global Angle | 8.46994161605835 |
| Global Joint | 4.290406703948975 |
| Global Mesh | 4.819797563552856 |
| Root Jitter | 0.2872091166675091 |
| Joint Jitter | 0.4784387230873108 |

### Orchestrator Task: round3_s4_v12_alpha030_ik2w2_from_v11

Name: Round3 S4 full-pipeline v12_alpha030_ik2w2_from_v11

Status: completed

Type: eval

Start: 2026-06-06T22:07:25

End: 2026-06-06T22:14:00

GPU: 0

PID: 2011559

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.39537861308455 |
| Local SIP | 10.071556854248048 |
| Local Angle | 8.726690864562988 |
| Local Joint | 4.468818616867066 |
| Local Mesh | 5.109324741363525 |
| Global SIP | 10.236503601074219 |
| Global Angle | 8.47910213470459 |
| Global Joint | 4.298629951477051 |
| Global Mesh | 4.832332468032837 |
| Root Jitter | 0.2869093529880047 |
| Joint Jitter | 0.4780301660299301 |

### Orchestrator Task: round3_s5_v12_alpha040_ik2w2_from_v11

Name: Round3 S5 full-pipeline v12_alpha040_ik2w2_from_v11

Status: completed

Type: eval

Start: 2026-06-06T22:13:45

End: 2026-06-06T22:19:35

GPU: 1

PID: 2015996

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.73998444685712 |
| Local SIP | 9.61759603023529 |
| Local Angle | 12.450390100479126 |
| Local Joint | 4.328210830688477 |
| Local Mesh | 5.104433059692383 |
| Global SIP | 9.101807713508606 |
| Global Angle | 11.748084783554077 |
| Global Joint | 3.8121743202209473 |
| Global Mesh | 4.407786011695862 |
| Root Jitter | 0.3754016747698188 |
| Joint Jitter | 0.8067303989082575 |

### Orchestrator Task: round3_real_s4_v12_alpha040_ik2w2_from_v11

Name: Round3 S4 real streaming IK1 audit v12_alpha040_ik2w2_from_v11

Status: completed

Type: audit

Start: 2026-06-06T22:19:35

End: 2026-06-06T22:24:24

GPU: 1

PID: 2019097

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v12_alpha040_ik2w2_from_v11 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.36728921282291 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v12_alpha040_ik2w2_from_v11 |

### Orchestrator Task: round3_real_s5_v12_alpha040_ik2w2_from_v11

Name: Round3 S5 real streaming IK1 audit v12_alpha040_ik2w2_from_v11

Status: completed

Type: audit

Start: 2026-06-06T22:24:24

End: 2026-06-06T22:28:58

GPU: 1

PID: 2021713

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v12_alpha040_ik2w2_from_v11 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.73998444685712 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v12_alpha040_ik2w2_from_v11 |

### Orchestrator Task: round3_train_v12_alpha035_s5stable_from_v11

Name: Round3 train v12_alpha035_s5stable_from_v11

Status: completed

Type: train

Start: 2026-06-06T22:28:58

End: 2026-06-06T22:29:13

GPU: 1

PID: 2024225

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train --experiment-name v12_alpha035_s5stable_from_v11 --epochs 3 --lr 1e-6 --dropout 0.10 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.35 --pRJ-weight 0.8 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.04 --pRJ-ddot-weight 0.0015 --ik1-distill-pRJ-weight 1.0 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.5`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train/last.pt`
- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0009340953227365389 |
| last_epoch | 3 |
| last_val_loss | 0.0009340953227365389 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 1.0 |
| ik2_input_distill | 2.5 |
| pRJ | 0.8 |
| pRJ_ddot | 0.0015 |
| pRJ_dot | 0.04 |

### Orchestrator Task: round3_s4_v12_alpha035_s5stable_from_v11

Name: Round3 S4 full-pipeline v12_alpha035_s5stable_from_v11

Status: completed

Type: eval

Start: 2026-06-06T22:29:13

End: 2026-06-06T22:35:49

GPU: 1

PID: 2024444

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.38921534772218 |
| Local SIP | 10.07146110534668 |
| Local Angle | 8.72591257095337 |
| Local Joint | 4.4681251525878904 |
| Local Mesh | 5.108399391174316 |
| Global SIP | 10.235895729064941 |
| Global Angle | 8.474865913391113 |
| Global Joint | 4.294867515563965 |
| Global Mesh | 4.826852416992187 |
| Root Jitter | 0.28685052022337915 |
| Joint Jitter | 0.47807621508836745 |

### Orchestrator Task: round3_s5_v12_alpha030_ik2w2_from_v11

Name: Round3 S5 full-pipeline v12_alpha030_ik2w2_from_v11

Status: completed

Type: eval

Start: 2026-06-06T22:29:44

End: 2026-06-06T22:35:49

GPU: 0

PID: 2024931

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.78341374022886 |
| Local SIP | 9.631479263305664 |
| Local Angle | 12.456440687179565 |
| Local Joint | 4.336070895195007 |
| Local Mesh | 5.109046816825867 |
| Global SIP | 9.116354823112488 |
| Global Angle | 11.755742311477661 |
| Global Joint | 3.8175302743911743 |
| Global Mesh | 4.415165901184082 |
| Root Jitter | 0.3740515233948827 |
| Joint Jitter | 0.8036538194864988 |

### Orchestrator Task: round3_s5_v12_alpha035_s5stable_from_v11

Name: Round3 S5 full-pipeline v12_alpha035_s5stable_from_v11

Status: completed

Type: eval

Start: 2026-06-06T22:35:50

End: 2026-06-06T22:41:55

GPU: 1

PID: 2027416

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.737316830046474 |
| Local SIP | 9.619295358657837 |
| Local Angle | 12.448022603988647 |
| Local Joint | 4.331571042537689 |
| Local Mesh | 5.103365182876587 |
| Global SIP | 9.101602673530579 |
| Global Angle | 11.745819091796875 |
| Global Joint | 3.8136839866638184 |
| Global Mesh | 4.410418272018433 |
| Root Jitter | 0.37463897094130516 |
| Joint Jitter | 0.8051599152386189 |

### Orchestrator Task: round3_real_s4_v12_alpha035_s5stable_from_v11

Name: Round3 S4 real streaming IK1 audit v12_alpha035_s5stable_from_v11

Status: completed

Type: audit

Start: 2026-06-06T22:41:55

End: 2026-06-06T22:46:44

GPU: 1

PID: 2028489

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v12_alpha035_s5stable_from_v11 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.38921534772218 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v12_alpha035_s5stable_from_v11 |

### Orchestrator Task: round3_real_s5_v12_alpha035_s5stable_from_v11

Name: Round3 S5 real streaming IK1 audit v12_alpha035_s5stable_from_v11

Status: completed

Type: audit

Start: 2026-06-06T22:46:44

End: 2026-06-06T22:51:18

GPU: 1

PID: 2029337

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v12_alpha035_s5stable_from_v11 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.737316830046474 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v12_alpha035_s5stable_from_v11 |

### Orchestrator Task: round3_real_s4_v12_alpha030_ik2w2_from_v11

Name: Round3 S4 real streaming IK1 audit v12_alpha030_ik2w2_from_v11

Status: completed

Type: audit

Start: 2026-06-06T22:59:03

End: 2026-06-06T23:03:37

GPU: 1

PID: 2031800

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v12_alpha030_ik2w2_from_v11 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.39537861308455 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v12_alpha030_ik2w2_from_v11 |

### Orchestrator Task: round3_real_s5_v12_alpha030_ik2w2_from_v11

Name: Round3 S5 real streaming IK1 audit v12_alpha030_ik2w2_from_v11

Status: completed

Type: audit

Start: 2026-06-06T23:03:37

End: 2026-06-06T23:08:11

GPU: 1

PID: 2032712

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v12_alpha030_ik2w2_from_v11 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.78341374022886 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v12_alpha030_ik2w2_from_v11 |

### Orchestrator Task: round3_train_v12_alpha035_ik2w25_from_v11

Name: Round3 train v12_alpha035_ik2w25_from_v11

Status: completed

Type: train

Start: 2026-06-06T23:08:11

End: 2026-06-06T23:08:26

GPU: 1

PID: 2033620

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train --experiment-name v12_alpha035_ik2w25_from_v11 --epochs 3 --lr 1.5e-6 --dropout 0.10 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.35 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.5`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train/last.pt`
- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010818190639838576 |
| last_epoch | 3 |
| last_val_loss | 0.0010818190639838576 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.8 |
| ik2_input_distill | 2.5 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round3_s4_v12_alpha035_ik2w25_from_v11

Name: Round3 S4 full-pipeline v12_alpha035_ik2w25_from_v11

Status: completed

Type: eval

Start: 2026-06-06T23:08:26

End: 2026-06-06T23:15:17

GPU: 1

PID: 2033732

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.386719339862466 |
| Local SIP | 10.070025157928466 |
| Local Angle | 8.725553798675538 |
| Local Joint | 4.468177127838135 |
| Local Mesh | 5.108660507202148 |
| Global SIP | 10.235164165496826 |
| Global Angle | 8.474906826019287 |
| Global Joint | 4.294702768325806 |
| Global Mesh | 4.826364660263062 |
| Root Jitter | 0.2869244858622551 |
| Joint Jitter | 0.4781402125954628 |

### Orchestrator Task: round3_s5_v12_alpha035_ik2w25_from_v11

Name: Round3 S5 full-pipeline v12_alpha035_ik2w25_from_v11

Status: completed

Type: eval

Start: 2026-06-06T23:15:17

End: 2026-06-06T23:21:22

GPU: 1

PID: 2035002

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.75161533830688 |
| Local SIP | 9.62232220172882 |
| Local Angle | 12.45108699798584 |
| Local Joint | 4.332240521907806 |
| Local Mesh | 5.105639100074768 |
| Global SIP | 9.106214761734009 |
| Global Angle | 11.749263763427734 |
| Global Joint | 3.8145220279693604 |
| Global Mesh | 4.411274552345276 |
| Root Jitter | 0.3746604258194566 |
| Joint Jitter | 0.8051358442753553 |

### Orchestrator Task: round3_real_s4_v12_alpha035_ik2w25_from_v11

Name: Round3 S4 real streaming IK1 audit v12_alpha035_ik2w25_from_v11

Status: completed

Type: audit

Start: 2026-06-06T23:21:22

End: 2026-06-06T23:25:56

GPU: 1

PID: 2036145

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v12_alpha035_ik2w25_from_v11 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.386719339862466 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v12_alpha035_ik2w25_from_v11 |

### Orchestrator Task: round3_real_s5_v12_alpha035_ik2w25_from_v11

Name: Round3 S5 real streaming IK1 audit v12_alpha035_ik2w25_from_v11

Status: completed

Type: audit

Start: 2026-06-06T23:25:56

End: 2026-06-06T23:30:14

GPU: 1

PID: 2037099

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v12_alpha035_ik2w25_from_v11 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.75161533830688 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v12_alpha035_ik2w25_from_v11 |

### Orchestrator Task: round4_train_v13_alpha042_ik2w2_from_v12

Name: Round4 train v13_alpha042_ik2w2_from_v12

Status: completed

Type: train

Start: 2026-06-06T23:38:00

End: 2026-06-06T23:38:15

GPU: 1

PID: 2040135

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train --experiment-name v13_alpha042_ik2w2_from_v12 --epochs 3 --lr 1.2e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.42 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.001044887414900586 |
| last_epoch | 3 |
| last_val_loss | 0.001044887414900586 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.8 |
| ik2_input_distill | 2.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round4_train_v13_alpha038_ik2w2_from_v12

Name: Round4 train v13_alpha038_ik2w2_from_v12

Status: completed

Type: train

Start: 2026-06-06T23:38:00

End: 2026-06-06T23:38:31

GPU: 0

PID: 2040134

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train --experiment-name v13_alpha038_ik2w2_from_v12 --epochs 3 --lr 1.2e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.38 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010502463032025845 |
| last_epoch | 3 |
| last_val_loss | 0.0010502463032025845 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.8 |
| ik2_input_distill | 2.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round4_s4_v13_alpha042_ik2w2_from_v12

Name: Round4 S4 full-pipeline v13_alpha042_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-06T23:38:16

End: 2026-06-06T23:44:36

GPU: 1

PID: 2040330

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.370207245603204 |
| Local SIP | 10.065584087371827 |
| Local Angle | 8.723765754699707 |
| Local Joint | 4.467439889907837 |
| Local Mesh | 5.108120441436768 |
| Global SIP | 10.230829811096191 |
| Global Angle | 8.469501781463624 |
| Global Joint | 4.2899498462677 |
| Global Mesh | 4.81909556388855 |
| Root Jitter | 0.2873816750943661 |
| Joint Jitter | 0.4786837354302406 |

### Orchestrator Task: round4_s4_v13_alpha038_ik2w2_from_v12

Name: Round4 S4 full-pipeline v13_alpha038_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-06T23:38:31

End: 2026-06-06T23:45:06

GPU: 0

PID: 2040540

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.376336369305854 |
| Local SIP | 10.067440795898438 |
| Local Angle | 8.724551963806153 |
| Local Joint | 4.4677411079406735 |
| Local Mesh | 5.10846700668335 |
| Global SIP | 10.231966114044189 |
| Global Angle | 8.471617317199707 |
| Global Joint | 4.292027521133423 |
| Global Mesh | 4.822668361663818 |
| Root Jitter | 0.2870557144284248 |
| Joint Jitter | 0.47833154499530794 |

### Orchestrator Task: round4_s5_v13_alpha042_ik2w2_from_v12

Name: Round4 S5 full-pipeline v13_alpha042_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-06T23:44:36

End: 2026-06-06T23:50:56

GPU: 1

PID: 2041642

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.71751535784453 |
| Local SIP | 9.612090110778809 |
| Local Angle | 12.444246053695679 |
| Local Joint | 4.325650990009308 |
| Local Mesh | 5.1008830070495605 |
| Global SIP | 9.096914529800415 |
| Global Angle | 11.742501974105835 |
| Global Joint | 3.8112383484840393 |
| Global Mesh | 4.406506538391113 |
| Root Jitter | 0.37572029046714306 |
| Joint Jitter | 0.8073755614459515 |

### Orchestrator Task: round4_s5_v13_alpha038_ik2w2_from_v12

Name: Round4 S5 full-pipeline v13_alpha038_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-06T23:45:06

End: 2026-06-06T23:51:11

GPU: 0

PID: 2041889

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.73781035907567 |
| Local SIP | 9.618024587631226 |
| Local Angle | 12.448243856430054 |
| Local Joint | 4.329896986484528 |
| Local Mesh | 5.10431432723999 |
| Global SIP | 9.103167057037354 |
| Global Angle | 11.746022939682007 |
| Global Joint | 3.8130151629447937 |
| Global Mesh | 4.409253895282745 |
| Root Jitter | 0.37509846314787865 |
| Joint Jitter | 0.8060703352093697 |

### Orchestrator Task: round4_real_s4_v13_alpha042_ik2w2_from_v12

Name: Round4 S4 real streaming IK1 audit v13_alpha042_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-06T23:50:56

End: 2026-06-06T23:55:45

GPU: 1

PID: 2042973

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v13_alpha042_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.370207245603204 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v13_alpha042_ik2w2_from_v12 |

### Orchestrator Task: round4_real_s4_v13_alpha038_ik2w2_from_v12

Name: Round4 S4 real streaming IK1 audit v13_alpha038_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-06T23:51:12

End: 2026-06-06T23:56:00

GPU: 0

PID: 2043129

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v13_alpha038_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.376336369305854 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v13_alpha038_ik2w2_from_v12 |

### Orchestrator Task: round4_real_s5_v13_alpha042_ik2w2_from_v12

Name: Round4 S5 real streaming IK1 audit v13_alpha042_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-06T23:55:45

End: 2026-06-07T00:00:19

GPU: 1

PID: 2044064

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v13_alpha042_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.71751535784453 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v13_alpha042_ik2w2_from_v12 |

### Orchestrator Task: round4_real_s5_v13_alpha038_ik2w2_from_v12

Name: Round4 S5 real streaming IK1 audit v13_alpha038_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-06T23:56:01

End: 2026-06-07T00:00:19

GPU: 0

PID: 2044219

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v13_alpha038_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.73781035907567 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v13_alpha038_ik2w2_from_v12 |

### Orchestrator Task: round4_train_v13_alpha040_ik2w225_from_v12

Name: Round4 train v13_alpha040_ik2w225_from_v12

Status: completed

Type: train

Start: 2026-06-07T00:00:19

End: 2026-06-07T00:00:34

GPU: 0

PID: 2045261

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train --experiment-name v13_alpha040_ik2w225_from_v12 --epochs 3 --lr 1.2e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.4 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.25`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010616357845719903 |
| last_epoch | 3 |
| last_val_loss | 0.0010616357845719903 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.8 |
| ik2_input_distill | 2.25 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round4_train_v13_alpha040_s5stable_from_v12

Name: Round4 train v13_alpha040_s5stable_from_v12

Status: completed

Type: train

Start: 2026-06-07T00:00:19

End: 2026-06-07T00:00:34

GPU: 1

PID: 2045262

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train --experiment-name v13_alpha040_s5stable_from_v12 --epochs 3 --lr 8e-7 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.4 --pRJ-weight 0.8 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.04 --pRJ-ddot-weight 0.0015 --ik1-distill-pRJ-weight 1.0 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.25`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0009287933062296361 |
| last_epoch | 3 |
| last_val_loss | 0.0009287933062296361 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 1.0 |
| ik2_input_distill | 2.25 |
| pRJ | 0.8 |
| pRJ_ddot | 0.0015 |
| pRJ_dot | 0.04 |

### Orchestrator Task: round4_s4_v13_alpha040_ik2w225_from_v12

Name: Round4 S4 full-pipeline v13_alpha040_ik2w225_from_v12

Status: completed

Type: eval

Start: 2026-06-07T00:00:34

End: 2026-06-07T00:06:55

GPU: 0

PID: 2045900

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.3787378231883 |
| Local SIP | 10.068017387390137 |
| Local Angle | 8.725085735321045 |
| Local Joint | 4.467905569076538 |
| Local Mesh | 5.108921670913697 |
| Global SIP | 10.233187484741212 |
| Global Angle | 8.471712875366212 |
| Global Joint | 4.291605520248413 |
| Global Mesh | 4.8217607021331785 |
| Root Jitter | 0.2870852373540401 |
| Joint Jitter | 0.47832314372062684 |

### Orchestrator Task: round4_s4_v13_alpha040_s5stable_from_v12

Name: Round4 S4 full-pipeline v13_alpha040_s5stable_from_v12

Status: completed

Type: eval

Start: 2026-06-07T00:00:34

End: 2026-06-07T00:07:10

GPU: 1

PID: 2045901

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.3813811891675 |
| Local SIP | 10.069668865203857 |
| Local Angle | 8.725146579742432 |
| Local Joint | 4.467659664154053 |
| Local Mesh | 5.108443784713745 |
| Global SIP | 10.234616470336913 |
| Global Angle | 8.471254444122314 |
| Global Joint | 4.2914573669433596 |
| Global Mesh | 4.821690893173217 |
| Root Jitter | 0.2870588548481464 |
| Joint Jitter | 0.47831266522407534 |

### Orchestrator Task: round4_s5_v13_alpha040_ik2w225_from_v12

Name: Round4 S5 full-pipeline v13_alpha040_ik2w225_from_v12

Status: completed

Type: eval

Start: 2026-06-07T00:06:55

End: 2026-06-07T00:13:00

GPU: 0

PID: 2050136

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.71488639246672 |
| Local SIP | 9.61136519908905 |
| Local Angle | 12.443263053894043 |
| Local Joint | 4.326895356178284 |
| Local Mesh | 5.100820183753967 |
| Global SIP | 9.096797227859497 |
| Global Angle | 11.741515398025513 |
| Global Joint | 3.811891257762909 |
| Global Mesh | 4.407286643981934 |
| Root Jitter | 0.3753936253488064 |
| Joint Jitter | 0.8066852204501629 |

### Orchestrator Task: round4_s5_v13_alpha040_s5stable_from_v12

Name: Round4 S5 full-pipeline v13_alpha040_s5stable_from_v12

Status: completed

Type: eval

Start: 2026-06-07T00:07:10

End: 2026-06-07T00:13:15

GPU: 1

PID: 2050337

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.707128487303855 |
| Local SIP | 9.60938012599945 |
| Local Angle | 12.441551685333252 |
| Local Joint | 4.326491951942444 |
| Local Mesh | 5.099248051643372 |
| Global SIP | 9.094338417053223 |
| Global Angle | 11.740009546279907 |
| Global Joint | 3.811325490474701 |
| Global Mesh | 4.406897842884064 |
| Root Jitter | 0.37536901514977217 |
| Joint Jitter | 0.8066968396306038 |

### Orchestrator Task: round4_real_s4_v13_alpha040_ik2w225_from_v12

Name: Round4 S4 real streaming IK1 audit v13_alpha040_ik2w225_from_v12

Status: completed

Type: audit

Start: 2026-06-07T00:13:00

End: 2026-06-07T00:17:49

GPU: 0

PID: 2054578

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v13_alpha040_ik2w225_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.3787378231883 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v13_alpha040_ik2w225_from_v12 |

### Orchestrator Task: round4_real_s4_v13_alpha040_s5stable_from_v12

Name: Round4 S4 real streaming IK1 audit v13_alpha040_s5stable_from_v12

Status: completed

Type: audit

Start: 2026-06-07T00:13:15

End: 2026-06-07T00:17:49

GPU: 1

PID: 2054835

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v13_alpha040_s5stable_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.3813811891675 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v13_alpha040_s5stable_from_v12 |

### Orchestrator Task: round4_real_s5_v13_alpha040_s5stable_from_v12

Name: Round4 S5 real streaming IK1 audit v13_alpha040_s5stable_from_v12

Status: completed

Type: audit

Start: 2026-06-07T00:17:49

End: 2026-06-07T00:22:23

GPU: 1

PID: 2059380

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v13_alpha040_s5stable_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.707128487303855 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v13_alpha040_s5stable_from_v12 |

### Orchestrator Task: round4_real_s5_v13_alpha040_ik2w225_from_v12

Name: Round4 S5 real streaming IK1 audit v13_alpha040_ik2w225_from_v12

Status: completed

Type: audit

Start: 2026-06-07T00:17:49

End: 2026-06-07T00:22:38

GPU: 0

PID: 2059379

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v13_alpha040_ik2w225_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.71488639246672 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v13_alpha040_ik2w225_from_v12 |

### Orchestrator Task: round5_train_v14_alpha0405_ik2w2_from_v12

Name: Round5 train v14_alpha0405_ik2w2_from_v12

Status: completed

Type: train

Start: 2026-06-07T00:32:11

End: 2026-06-07T00:32:26

GPU: 1

PID: 2070359

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train --experiment-name v14_alpha0405_ik2w2_from_v12 --epochs 3 --lr 1.0e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.405 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.001046295097330585 |
| last_epoch | 3 |
| last_val_loss | 0.001046295097330585 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.8 |
| ik2_input_distill | 2.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round5_train_v14_alpha0410_ik2w2_from_v12

Name: Round5 train v14_alpha0410_ik2w2_from_v12

Status: completed

Type: train

Start: 2026-06-07T00:33:27

End: 2026-06-07T00:33:42

GPU: 0

PID: 2071433

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train --experiment-name v14_alpha0410_ik2w2_from_v12 --epochs 3 --lr 1.0e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.41 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010457895637955516 |
| last_epoch | 3 |
| last_val_loss | 0.0010457895637955516 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.8 |
| ik2_input_distill | 2.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round5_s4_v14_alpha0405_ik2w2_from_v12

Name: Round5 S4 full-pipeline v14_alpha0405_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T00:32:26

End: 2026-06-07T00:39:32

GPU: 1

PID: 2070590

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.372773325160146 |
| Local SIP | 10.066407489776612 |
| Local Angle | 8.723906421661377 |
| Local Joint | 4.467434072494507 |
| Local Mesh | 5.108246803283691 |
| Global SIP | 10.231533432006836 |
| Global Angle | 8.470336151123046 |
| Global Joint | 4.29062066078186 |
| Global Mesh | 4.820356750488282 |
| Root Jitter | 0.2872064270079136 |
| Joint Jitter | 0.47843572646379473 |

### Orchestrator Task: round5_s4_v14_alpha0410_ik2w2_from_v12

Name: Round5 S4 full-pipeline v14_alpha0410_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T00:33:42

End: 2026-06-07T00:40:02

GPU: 0

PID: 2071659

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.37348917667567 |
| Local SIP | 10.066746234893799 |
| Local Angle | 8.724196338653565 |
| Local Joint | 4.467832136154175 |
| Local Mesh | 5.108583879470825 |
| Global SIP | 10.231786918640136 |
| Global Angle | 8.470143032073974 |
| Global Joint | 4.290491676330566 |
| Global Mesh | 4.82004132270813 |
| Root Jitter | 0.2871894180774689 |
| Joint Jitter | 0.4784271165728569 |

### Orchestrator Task: round5_s5_v14_alpha0405_ik2w2_from_v12

Name: Round5 S5 full-pipeline v14_alpha0405_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T00:39:32

End: 2026-06-07T00:45:53

GPU: 1

PID: 2075371

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.725488430112605 |
| Local SIP | 9.613972902297974 |
| Local Angle | 12.44599986076355 |
| Local Joint | 4.327340245246887 |
| Local Mesh | 5.10241973400116 |
| Global SIP | 9.099478006362915 |
| Global Angle | 11.744041919708252 |
| Global Joint | 3.8119316697120667 |
| Global Mesh | 4.407502233982086 |
| Root Jitter | 0.37544210627675056 |
| Joint Jitter | 0.8068549484014511 |

### Orchestrator Task: round5_s5_v14_alpha0410_ik2w2_from_v12

Name: Round5 S5 full-pipeline v14_alpha0410_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T00:40:02

End: 2026-06-07T00:46:23

GPU: 0

PID: 2075815

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.72153897484764 |
| Local SIP | 9.613159656524658 |
| Local Angle | 12.445382833480835 |
| Local Joint | 4.327044785022736 |
| Local Mesh | 5.102129817008972 |
| Global SIP | 9.097949385643005 |
| Global Angle | 11.743125438690186 |
| Global Joint | 3.811467170715332 |
| Global Mesh | 4.407057285308838 |
| Root Jitter | 0.37558136228471994 |
| Joint Jitter | 0.8070464935153723 |

### Orchestrator Task: round5_real_s4_v14_alpha0405_ik2w2_from_v12

Name: Round5 S4 real streaming IK1 audit v14_alpha0405_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T00:45:53

End: 2026-06-07T00:50:27

GPU: 1

PID: 2081642

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v14_alpha0405_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.372773325160146 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha0405_ik2w2_from_v12 |

### Orchestrator Task: round5_real_s4_v14_alpha0410_ik2w2_from_v12

Name: Round5 S4 real streaming IK1 audit v14_alpha0410_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T00:49:26

End: 2026-06-07T00:54:00

GPU: 0

PID: 2085826

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v14_alpha0410_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.37348917667567 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha0410_ik2w2_from_v12 |

### Orchestrator Task: round5_real_s5_v14_alpha0405_ik2w2_from_v12

Name: Round5 S5 real streaming IK1 audit v14_alpha0405_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T00:50:27

End: 2026-06-07T00:54:45

GPU: 1

PID: 2086534

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v14_alpha0405_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0405_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.725488430112605 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha0405_ik2w2_from_v12 |

### Orchestrator Task: round5_train_v14_alpha0415_ik2w2_from_v12

Name: Round5 train v14_alpha0415_ik2w2_from_v12

Status: completed

Type: train

Start: 2026-06-07T00:54:46

End: 2026-06-07T00:55:01

GPU: 1

PID: 2088782

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train --experiment-name v14_alpha0415_ik2w2_from_v12 --epochs 3 --lr 1.0e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.415 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010452247457578777 |
| last_epoch | 3 |
| last_val_loss | 0.0010452247457578777 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.8 |
| ik2_input_distill | 2.0 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round5_real_s5_v14_alpha0410_ik2w2_from_v12

Name: Round5 S5 real streaming IK1 audit v14_alpha0410_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T00:54:00

End: 2026-06-07T00:58:18

GPU: 0

PID: 2088267

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v14_alpha0410_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.72153897484764 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha0410_ik2w2_from_v12 |

### Orchestrator Task: round5_train_v14_alpha0410_ik2w21_from_v12

Name: Round5 train v14_alpha0410_ik2w21_from_v12

Status: completed

Type: train

Start: 2026-06-07T00:58:19

End: 2026-06-07T00:58:34

GPU: 0

PID: 2090392

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train --experiment-name v14_alpha0410_ik2w21_from_v12 --epochs 3 --lr 1.0e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.41 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.1`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010521127027459443 |
| last_epoch | 3 |
| last_val_loss | 0.0010521127027459443 |
Weights:

| metric | value |
|---|---:|
| bone_length | 0.5 |
| gR2 | 0.0 |
| gR2_ddot | 0.0 |
| gR2_dot | 0.0 |
| ik1_distill_gR2 | 1.0 |
| ik1_distill_pRJ | 0.8 |
| ik2_input_distill | 2.1 |
| pRJ | 1.0 |
| pRJ_ddot | 0.001 |
| pRJ_dot | 0.03 |

### Orchestrator Task: round5_s4_v14_alpha0415_ik2w2_from_v12

Name: Round5 S4 full-pipeline v14_alpha0415_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T00:55:01

End: 2026-06-07T01:01:36

GPU: 1

PID: 2089010

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.37063280807435 |
| Local SIP | 10.065557193756103 |
| Local Angle | 8.724200439453124 |
| Local Joint | 4.4676093578338625 |
| Local Mesh | 5.108386039733887 |
| Global SIP | 10.230517864227295 |
| Global Angle | 8.469825649261475 |
| Global Joint | 4.289862251281738 |
| Global Mesh | 4.819163751602173 |
| Root Jitter | 0.2872119329869747 |
| Joint Jitter | 0.478450046479702 |

### Orchestrator Task: round5_s4_v14_alpha0410_ik2w21_from_v12

Name: Round5 S4 full-pipeline v14_alpha0410_ik2w21_from_v12

Status: completed

Type: eval

Start: 2026-06-07T00:58:34

End: 2026-06-07T01:05:24

GPU: 0

PID: 2090598

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.36968127383291 |
| Local SIP | 10.06557502746582 |
| Local Angle | 8.723701572418213 |
| Local Joint | 4.467196798324585 |
| Local Mesh | 5.107817125320435 |
| Global SIP | 10.230309104919433 |
| Global Angle | 8.469593906402588 |
| Global Joint | 4.2899760723114015 |
| Global Mesh | 4.819415521621704 |
| Root Jitter | 0.2871938951313496 |
| Joint Jitter | 0.4784375563263893 |

### Orchestrator Task: round5_s5_v14_alpha0415_ik2w2_from_v12

Name: Round5 S5 full-pipeline v14_alpha0415_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T01:01:36

End: 2026-06-07T01:07:26

GPU: 1

PID: 2092414

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.718751320485026 |
| Local SIP | 9.612117290496826 |
| Local Angle | 12.445310354232788 |
| Local Joint | 4.326860070228577 |
| Local Mesh | 5.102001309394836 |
| Global SIP | 9.09680986404419 |
| Global Angle | 11.742653369903564 |
| Global Joint | 3.8110249638557434 |
| Global Mesh | 4.4064154624938965 |
| Root Jitter | 0.37565048690885305 |
| Joint Jitter | 0.8071938399225473 |

### Orchestrator Task: round5_s5_v14_alpha0410_ik2w21_from_v12

Name: Round5 S5 full-pipeline v14_alpha0410_ik2w21_from_v12

Status: completed

Type: eval

Start: 2026-06-07T01:05:25

End: 2026-06-07T01:11:30

GPU: 0

PID: 2094667

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.719431065004315 |
| Local SIP | 9.612767815589905 |
| Local Angle | 12.445660591125488 |
| Local Joint | 4.326795279979706 |
| Local Mesh | 5.101636171340942 |
| Global SIP | 9.096270203590393 |
| Global Angle | 11.742905139923096 |
| Global Joint | 3.810771942138672 |
| Global Mesh | 4.406194746494293 |
| Root Jitter | 0.3755748961120844 |
| Joint Jitter | 0.8070592563599348 |

### Orchestrator Task: round5_real_s4_v14_alpha0415_ik2w2_from_v12

Name: Round5 S4 real streaming IK1 audit v14_alpha0415_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T01:07:26

End: 2026-06-07T01:12:15

GPU: 1

PID: 2095736

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v14_alpha0415_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.37063280807435 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha0415_ik2w2_from_v12 |

### Orchestrator Task: round5_real_s4_v14_alpha0410_ik2w21_from_v12

Name: Round5 S4 real streaming IK1 audit v14_alpha0410_ik2w21_from_v12

Status: completed

Type: audit

Start: 2026-06-07T01:11:30

End: 2026-06-07T01:16:19

GPU: 0

PID: 2098534

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v14_alpha0410_ik2w21_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.36968127383291 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha0410_ik2w21_from_v12 |

### Orchestrator Task: round5_real_s5_v14_alpha0415_ik2w2_from_v12

Name: Round5 S5 real streaming IK1 audit v14_alpha0415_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T01:12:15

End: 2026-06-07T01:16:49

GPU: 1

PID: 2099621

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v14_alpha0415_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0415_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.718751320485026 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha0415_ik2w2_from_v12 |

### Orchestrator Task: round5_real_s5_v14_alpha0410_ik2w21_from_v12

Name: Round5 S5 real streaming IK1 audit v14_alpha0410_ik2w21_from_v12

Status: completed

Type: audit

Start: 2026-06-07T01:16:19

End: 2026-06-07T01:20:37

GPU: 0

PID: 2103151

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v14_alpha0410_ik2w21_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha0410_ik2w21_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.719431065004315 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha0410_ik2w21_from_v12 |

## EXP-ik1-auto-search-round0-complete

Date: 2026-06-06

Status: completed.

Task result:

- Completed: 14
- Failed: 0
- State: `data/experiments/orchestrator_states/ik1_auto_search_queue.json`
- Results CSV: `experiments/ik1_auto_search_results.csv`
- Logs: `logs/orchestrator/ik1_auto_search/round0/*.log`
- New JSON root: `data/experiments/ik1_auto_search/round0/`

Best S4 ranking:

| Rank | Version | S4 Score ↓ | Δ vs PL-only |
|---:|---|---:|---:|
| 1 | baseline_official_ik1 | `38.625657482802865` | `0.0` |
| 2 | newik1_v6_stage_a | `38.649136830300094` | `+0.02347934749722924` |
| 3 | newik1_v9_C8_last | `38.69384564702212` | `+0.06818816421925789` |
| 4 | newik1_v8_B4_last | `38.69415222530066` | `+0.06849474249779774` |

Conclusion:

- No NewIK1 seed beats PL-only best under fixed `processed IMU + newpl_v4_init36`.
- `newik1_v6_stage_a` is the best NewIK1 seed by S4, but remains worse than PL-only.
- S4/S5 real streaming IK1 diagnostics are now present for the seed set and remain diagnostic only.
- Round 1 should use `newik1_v6_stage_a` as parent, but pRJ-only/residual `from_v6a` requires an official-input backend change before launch; otherwise the experiment would not match the requested route.

## EXP-ik1-auto-search-round1-launch

Date: 2026-06-06

Implemented backend changes before launch:

- `GPNet` now supports official-input IK1 output modes: `full`, `pRJ_only`, `residual`, and `residual_pRJ_only`.
- For official-input replacement modes, `GPNet` keeps separate recurrent hidden states for official IK1 baseline and NewIK1 candidate, so pRJ-only routes preserve official/base `gR2` during streaming.
- `newik1_official_input_train.py` now records `output_mode` and `residual_alpha` in checkpoint config.
- `newik1_official_input_train.py` now supports real `ik2_input_distill` loss using:
  `RRB_after_ik1[45] + gR2[3] + pRJ[69] = 117D`.
- Smoke checks completed:
  - full v6a official-input eval: `data/experiments/ik1_auto_search/smoke/official_full_v6a_eval.json`
  - full v6a real streaming audit: `data/experiments/ik1_auto_search/smoke/real_full_v6a_eval.json`
  - residual pRJ-only train/eval smoke: `data/experiments/ik1_auto_search/smoke/residual_pRJ_only_alpha025_train/`
  - IK2-input distill smoke: `data/experiments/ik1_auto_search/smoke/ik2_distill_train/`

Round 1 queue:

- Queue: `experiments/ik1_auto_search_round1_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round1_queue.json`
- Logs: `logs/orchestrator/ik1_auto_search/round1/*/*.log`

Round 1 experiments:

| Experiment | Route | Parent | Output mode | Key loss |
|---|---|---|---|---|
| `v10_residual_pRJ_only_alpha025_from_v6a` | residual pRJ-only | `newik1_v6_stage_a` | `residual_pRJ_only`, α=`0.25` | pRJ + official gR2 distill + light IK2-input distill |
| `v10_residual_pRJ_only_alpha05_from_v6a` | residual pRJ-only | `newik1_v6_stage_a` | `residual_pRJ_only`, α=`0.5` | pRJ + official gR2 distill + light IK2-input distill |
| `v10_stage_a_low_lr_distill_official` | stage_a conservative finetune | `newik1_v6_stage_a` | `full` | low LR + strong official distill |
| `v10_ik2_input_distill_from_v6a` | downstream-aware IK2-input distill | `newik1_v6_stage_a` | `residual_pRJ_only`, α=`0.25` | strong IK2-input distill |

Dry-run result:

- Task count: 12
- Ready train tasks: 4
- Conflicts: none.
- GPU status: GPU1 free; GPU0 has a foreign-user process (`haonan`, pid `1913542`), so scheduler will not share GPU0 until it becomes available.


### Orchestrator Task: round0_real_s5_baseline_official_ik1

Name: Round0 S5 real streaming IK1 audit baseline_official_ik1

Status: completed

Type: audit

Start: 2026-06-06T18:49:43

End: 2026-06-06T18:54:01

GPU: 1

PID: 1884947

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/real_streaming/s5/baseline_official_ik1/result.json --split-label S5 --version-name baseline_official_ik1 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed --ik1-backend original`

Log: `logs/orchestrator/ik1_auto_search/round0/real_s5_baseline_official_ik1.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/real_streaming/s5/baseline_official_ik1/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | original |
| ik1_checkpoint | None |
| ik1_checkpoint_config | None |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.81127653867006 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | baseline_official_ik1 |

### Orchestrator Task: round0_full_s4_baseline_official_ik1

Name: Round0 S4 full-pipeline baseline_official_ik1

Status: completed

Type: eval

Start: 2026-06-06T18:49:43

End: 2026-06-06T18:56:03

GPU: 0

PID: 1884946

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" pl_curve_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/full_pipeline/s4/baseline_official_ik1/result.json --checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round0/full_s4_baseline_official_ik1.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/full_pipeline/s4/baseline_official_ik1/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.625657482802865 |
| Local SIP | 10.13525094985962 |
| Local Angle | 8.77213363647461 |
| Local Joint | 4.4951457500457765 |
| Local Mesh | 5.13820219039917 |
| Global SIP | 10.290855979919433 |
| Global Angle | 8.538510513305663 |
| Global Joint | 4.346226930618286 |
| Global Mesh | 4.898400592803955 |
| Root Jitter | 0.28584847748279574 |
| Joint Jitter | 0.47691351771354673 |

### Orchestrator Task: round0_real_s5_newik1_v4_official_input

Name: Round0 S5 real streaming IK1 audit newik1_v4_official_input

Status: completed

Type: audit

Start: 2026-06-06T18:54:01

End: 2026-06-06T18:58:04

GPU: 1

PID: 1887266

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/real_streaming/s5/newik1_v4_official_input/result.json --split-label S5 --version-name newik1_v4_official_input --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed --ik1-checkpoint data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt --ik1-backend official_input_v1`

Log: `logs/orchestrator/ik1_auto_search/round0/real_s5_newik1_v4_official_input.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/real_streaming/s5/newik1_v4_official_input/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.92881545905024 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | newik1_v4_official_input |

### Orchestrator Task: round0_real_s4_newik1_v4_official_input

Name: Round0 S4 real streaming IK1 audit newik1_v4_official_input

Status: completed

Type: audit

Start: 2026-06-06T18:56:03

End: 2026-06-06T19:00:21

GPU: 0

PID: 1888255

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/real_streaming/s4/newik1_v4_official_input/result.json --split-label S4 --version-name newik1_v4_official_input --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed --ik1-checkpoint data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt --ik1-backend official_input_v1`

Log: `logs/orchestrator/ik1_auto_search/round0/real_s4_newik1_v4_official_input.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/real_streaming/s4/newik1_v4_official_input/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.6972478222847 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | newik1_v4_official_input |

### Orchestrator Task: round0_real_s5_newik1_v6_stage_a

Name: Round0 S5 real streaming IK1 audit newik1_v6_stage_a

Status: completed

Type: audit

Start: 2026-06-06T18:58:05

End: 2026-06-06T19:02:38

GPU: 1

PID: 1889178

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/real_streaming/s5/newik1_v6_stage_a/result.json --split-label S5 --version-name newik1_v6_stage_a --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --ik1-backend official_input_v1`

Log: `logs/orchestrator/ik1_auto_search/round0/real_s5_newik1_v6_stage_a.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/real_streaming/s5/newik1_v6_stage_a/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.85181389780715 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | newik1_v6_stage_a |

### Orchestrator Task: round0_full_s4_newik1_v6_stage_a

Name: Round0 S4 full-pipeline newik1_v6_stage_a

Status: completed

Type: eval

Start: 2026-06-06T19:00:21

End: 2026-06-06T19:06:41

GPU: 0

PID: 1890254

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/full_pipeline/s4/newik1_v6_stage_a/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round0/full_s4_newik1_v6_stage_a.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/full_pipeline/s4/newik1_v6_stage_a/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.649136830300094 |
| Local SIP | 10.066027450561524 |
| Local Angle | 8.79630355834961 |
| Local Joint | 4.52389087677002 |
| Local Mesh | 5.174596786499023 |
| Global SIP | 10.31864824295044 |
| Global Angle | 8.571873569488526 |
| Global Joint | 4.3895911693573 |
| Global Mesh | 4.904520702362061 |
| Root Jitter | 0.30062844827771185 |
| Joint Jitter | 0.4935804337263107 |

### Orchestrator Task: round0_real_s4_newik1_v7_best

Name: Round0 S4 real streaming IK1 audit newik1_v7_best

Status: completed

Type: audit

Start: 2026-06-06T19:02:38

End: 2026-06-06T19:07:57

GPU: 1

PID: 1891367

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/real_streaming/s4/newik1_v7_best/result.json --split-label S4 --version-name newik1_v7_best --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed --ik1-checkpoint data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt --ik1-backend auto_control_point`

Log: `logs/orchestrator/ik1_auto_search/round0/real_s4_newik1_v7_best.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/real_streaming/s4/newik1_v7_best/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | control_point_last_v1 |
| ik1_checkpoint | data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.69478097228706 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | newik1_v7_best |

### Orchestrator Task: round0_full_s4_newik1_v7_best

Name: Round0 S4 full-pipeline newik1_v7_best

Status: completed

Type: eval

Start: 2026-06-06T19:06:41

End: 2026-06-06T19:14:02

GPU: 0

PID: 1893122

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/full_pipeline/s4/newik1_v7_best/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round0/full_s4_newik1_v7_best.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/full_pipeline/s4/newik1_v7_best/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.69478097228706 |
| Local SIP | 10.155638122558594 |
| Local Angle | 8.780602645874023 |
| Local Joint | 4.506718969345092 |
| Local Mesh | 5.149347591400146 |
| Global SIP | 10.316577625274657 |
| Global Angle | 8.55061378479004 |
| Global Joint | 4.360230112075806 |
| Global Mesh | 4.9109173774719235 |
| Root Jitter | 0.2785334773361683 |
| Joint Jitter | 0.4653885647654533 |

### Orchestrator Task: round0_full_s4_newik1_v8_B4_last

Name: Round0 S4 full-pipeline newik1_v8_B4_last

Status: completed

Type: eval

Start: 2026-06-06T19:07:57

End: 2026-06-06T19:14:48

GPU: 1

PID: 1893737

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/full_pipeline/s4/newik1_v8_B4_last/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round0/full_s4_newik1_v8_B4_last.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/full_pipeline/s4/newik1_v8_B4_last/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.69415222530066 |
| Local SIP | 10.155435752868652 |
| Local Angle | 8.780413722991943 |
| Local Joint | 4.506601095199585 |
| Local Mesh | 5.149188375473022 |
| Global SIP | 10.31648416519165 |
| Global Angle | 8.550490474700927 |
| Global Joint | 4.360141086578369 |
| Global Mesh | 4.910782766342163 |
| Root Jitter | 0.2785340346395969 |
| Joint Jitter | 0.4653891369700432 |

### Orchestrator Task: round0_real_s5_newik1_v7_best

Name: Round0 S5 real streaming IK1 audit newik1_v7_best

Status: completed

Type: audit

Start: 2026-06-06T19:14:02

End: 2026-06-06T19:19:22

GPU: 0

PID: 1896254

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/real_streaming/s5/newik1_v7_best/result.json --split-label S5 --version-name newik1_v7_best --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed --ik1-checkpoint data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt --ik1-backend auto_control_point`

Log: `logs/orchestrator/ik1_auto_search/round0/real_s5_newik1_v7_best.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/real_streaming/s5/newik1_v7_best/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | control_point_last_v1 |
| ik1_checkpoint | data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.85909186106175 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | newik1_v7_best |

### Orchestrator Task: round0_real_s5_newik1_v8_B4_last

Name: Round0 S5 real streaming IK1 audit newik1_v8_B4_last

Status: completed

Type: audit

Start: 2026-06-06T19:14:48

End: 2026-06-06T19:19:52

GPU: 1

PID: 1896806

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/real_streaming/s5/newik1_v8_B4_last/result.json --split-label S5 --version-name newik1_v8_B4_last --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed --ik1-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --ik1-backend auto_control_point`

Log: `logs/orchestrator/ik1_auto_search/round0/real_s5_newik1_v8_B4_last.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/real_streaming/s5/newik1_v8_B4_last/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | control_point_last_v1 |
| ik1_checkpoint | data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.860368536058814 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | newik1_v8_B4_last |

### Orchestrator Task: round0_real_s4_newik1_v8_B4_last

Name: Round0 S4 real streaming IK1 audit newik1_v8_B4_last

Status: completed

Type: audit

Start: 2026-06-06T19:19:22

End: 2026-06-06T19:24:56

GPU: 0

PID: 1899536

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/real_streaming/s4/newik1_v8_B4_last/result.json --split-label S4 --version-name newik1_v8_B4_last --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed --ik1-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --ik1-backend auto_control_point`

Log: `logs/orchestrator/ik1_auto_search/round0/real_s4_newik1_v8_B4_last.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/real_streaming/s4/newik1_v8_B4_last/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | control_point_last_v1 |
| ik1_checkpoint | data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.69415222530066 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | newik1_v8_B4_last |

### Orchestrator Task: round0_real_s5_newik1_v9_C8_last

Name: Round0 S5 real streaming IK1 audit newik1_v9_C8_last

Status: completed

Type: audit

Start: 2026-06-06T19:19:52

End: 2026-06-06T19:25:11

GPU: 1

PID: 1899958

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/real_streaming/s5/newik1_v9_C8_last/result.json --split-label S5 --version-name newik1_v9_C8_last --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt --ik1-backend auto_control_point`

Log: `logs/orchestrator/ik1_auto_search/round0/real_s5_newik1_v9_C8_last.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/real_streaming/s5/newik1_v9_C8_last/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | control_point_last_v1 |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.8609724480845 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | newik1_v9_C8_last |

### Orchestrator Task: round0_full_s4_newik1_v9_C8_last

Name: Round0 S4 full-pipeline newik1_v9_C8_last

Status: completed

Type: eval

Start: 2026-06-06T19:24:56

End: 2026-06-06T19:32:02

GPU: 0

PID: 1902834

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round0/full_pipeline/s4/newik1_v9_C8_last/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round0/full_s4_newik1_v9_C8_last.log`

Outputs:

- `data/experiments/ik1_auto_search/round0/full_pipeline/s4/newik1_v9_C8_last/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.69384564702212 |
| Local SIP | 10.15534267425537 |
| Local Angle | 8.78029670715332 |
| Local Joint | 4.506520175933838 |
| Local Mesh | 5.149078845977783 |
| Global SIP | 10.316460132598877 |
| Global Angle | 8.550429248809815 |
| Global Joint | 4.360109663009643 |
| Global Mesh | 4.9107245922088625 |
| Root Jitter | 0.2785369522869587 |
| Joint Jitter | 0.4653900310397148 |

## IK1 Auto Search Round 1 Retry 1

Date: 2026-06-06

Status: retry prepared after preserving the failed Round 1 evidence.

Original Round 1 outcome:

- Completed tasks: 0
- Failed train tasks: 4
- Blocked downstream S4 eval/audit tasks: 8
- State: `data/experiments/orchestrator_states/ik1_auto_search_round1_queue.json`
- Logs: `logs/orchestrator/ik1_auto_search/round1/*/train.log`
- Failure signature: `RuntimeError: The size of tensor a (976) must match the size of tensor b (16) at non-singleton dimension 1`
- Failure location: `newik1_official_input_train.py`, `ik2_input_feature`

Fix:

- `ik2_input_feature` now explicitly aligns all leading dimensions before computing `RRB_after_ik1`.
- Shape smoke passed for unbatched `[T,72]`, batched `[T,B,72]`, and single-frame `[72]`, each producing the required `117D` IK2 input.

Retry queue:

- Queue: `experiments/ik1_auto_search_round1_retry1_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round1_retry1_queue.json`
- Output root: `data/experiments/ik1_auto_search/round1_retry1/`
- Logs: `logs/orchestrator/ik1_auto_search/round1_retry1/*/*.log`
- Dry-run status: 12 tasks, 4 train ready, 8 dependency-blocked until train checkpoints exist; no duplicate output/log conflicts and no existing output/log conflicts.

Retry experiments remain unchanged:

| Experiment | Route | Parent | Output mode | Selection gate |
|---|---|---|---|---|
| `v10_residual_pRJ_only_alpha025_from_v6a` | residual pRJ-only | `newik1_v6_stage_a` | `residual_pRJ_only`, alpha `0.25` | S4 full-pipeline first |
| `v10_residual_pRJ_only_alpha05_from_v6a` | residual pRJ-only | `newik1_v6_stage_a` | `residual_pRJ_only`, alpha `0.5` | S4 full-pipeline first |
| `v10_stage_a_low_lr_distill_official` | stage_a conservative finetune | `newik1_v6_stage_a` | `full` | S4 full-pipeline first |
| `v10_ik2_input_distill_from_v6a` | downstream-aware IK2 input distill | `newik1_v6_stage_a` | `residual_pRJ_only`, alpha `0.25` | S4 full-pipeline first |

Fixed baseline remains `processed IMU + newpl_v4_init36 + official IK1`, S4 Score `38.625657482802865`. AMASS/cache/local losses remain diagnostic only.

### IK1 Auto Search Round 1 Final Evidence

Date: 2026-06-06

Round 1 retry1 S4 queue:

- Queue: `experiments/ik1_auto_search_round1_retry1_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round1_retry1_queue.json`
- Completed tasks: 9
- Failed tasks: 1
- Blocked tasks: 2
- Failed route: `v10_stage_a_low_lr_distill_official`; train log contains NaN pRJ/bone/IK2-input losses, `best_loss` stayed `Infinity`, and `best_loss.pt` was not produced.

S5 completion queue:

- Queue: `experiments/ik1_auto_search_round1_retry1_s5_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round1_retry1_s5_queue.json`
- Completed tasks: 6
- Failed tasks: 0

Full-pipeline ranking:

| Rank | Version | S4 Score ↓ | S4 Δ vs PL-only | S5 Score ↓ | S5 Δ vs official IK1 | Decision |
|---:|---|---:|---:|---:|---:|---|
| 1 S4 | `v10_residual_pRJ_only_alpha05_from_v6a` | `38.30681182323395` | `-0.31884565956891464` | `43.867689362633975` | `+0.05641282396391745` | S4 improves, S5 regresses |
| 2 S4 | `v10_residual_pRJ_only_alpha025_from_v6a` | `38.401125624060626` | `-0.2245318587422389` | `43.84817816592753` | `+0.03690162725747115` | S4 improves, S5 regresses |
| 3 S4 / best overall | `v10_ik2_input_distill_from_v6a` | `38.41680224156379` | `-0.20885524123907118` | `43.807055798713115` | `-0.0042207399569420545` | beats PL-only on S4 and S5 |
| baseline | `baseline_official_ik1` | `38.625657482802865` | `0.0` | `43.81127653867006` | `0.0` | previous best |

Real streaming IK1 diagnostics:

| Version | S4 pRJ L2 cm ↓ | S4 gR2 deg ↓ | S5 pRJ L2 cm ↓ | S5 gR2 deg ↓ |
|---|---:|---:|---:|---:|
| baseline_official_ik1 | `4.9430084228515625` | `25.584686279296875` | `4.610454082489014` | `15.153098106384277` |
| `v10_residual_pRJ_only_alpha025_from_v6a` | `4.875001907348633` | `25.584686279296875` | `4.568667411804199` | `15.153098106384277` |
| `v10_residual_pRJ_only_alpha05_from_v6a` | `4.86588716506958` | `25.584686279296875` | `4.551676273345947` | `15.153098106384277` |
| `v10_ik2_input_distill_from_v6a` | `4.876088619232178` | `25.584686279296875` | `4.568141937255859` | `15.153098106384277` |

Diagnostic interpretation:

- pRJ improves for all completed v10 routes, while gR2 is unchanged because `residual_pRJ_only` preserves official/base gR2.
- `alpha05` has the strongest S4 gain but worsens S5, so it is not the final candidate.
- `v10_ik2_input_distill_from_v6a` has smaller S4 gain but is the only route with S4 and S5 full-pipeline improvement; this supports the downstream-compatibility hypothesis.

Artifacts:

- Best overall checkpoint: `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt`
- Best overall S4 JSON: `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/s4/best_loss/result.json`
- Best overall S5 JSON: `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/s5/best_loss/result.json`
- Results CSV: `experiments/ik1_auto_search_results.csv`

## IK1 Auto Search Round 2 Launch

Date: 2026-06-06

Goal: continue narrow downstream-aware search from the only Round 1 candidate that beat both S4 and S5 baselines.

Parent:

```text
v10_ik2_input_distill_from_v6a
data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt
```

Queue:

- Queue: `experiments/ik1_auto_search_round2_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round2_queue.json`
- Output root: `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/`
- Log root: `logs/orchestrator/ik1_auto_search/round2_downstream_aware_from_v10/`

Scope restrictions:

- Fixed PL: `newpl_v4_init36`
- PL checkpoint: `data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt`
- IMU input: processed
- Selection rule: S4/S5 full-pipeline 11 metrics only.
- Reject S4-only gains if S5 regresses.
- Real streaming IK1 pRJ/gR2 diagnostics are diagnostic only.
- AMASS/cache/local losses are diagnostic only.

Round 2 candidates:

| Experiment | Route | Init | alpha | IK2-input distill | gR2 handling |
|---|---|---|---:|---:|---|
| `v11_alpha025_ik2w1_from_v10` | IK2 distill weight low | v10 best | `0.25` | `1.0` | preserved + official distill |
| `v11_alpha025_ik2w3_from_v10` | IK2 distill weight high | v10 best | `0.25` | `3.0` | preserved + official distill |
| `v11_alpha035_ik2w2_from_v10` | alpha midpoint | v10 best | `0.35` | `2.0` | preserved + official distill |
| `v11_pRJ_only_ik2w2_from_v10` | pRJ-only + IK2 distill | v10 best | n/a | `2.0` | preserved + official distill |

Dry-run result:

- Task count: 20
- Ready tasks: 4 train tasks
- Planned local starts: GPU0 `v11_alpha025_ik2w1_from_v10`, GPU1 `v11_alpha025_ik2w3_from_v10`
- Remaining train tasks wait for GPU.
- Conflicts: none.

GPU/server note: the project-local orchestrator currently verified only local `node01` GPU0/GPU1. The second server is still not verified in this workspace; do not invent remote execution records. Add remote execution only after host, project path, environment, and GPU status are verified.

### IK1 Auto Search Round 2 Evidence

Date: 2026-06-06

Status: completed.

Completed tasks: 20

Failed tasks: 0

Full-pipeline ranking:

| Rank | Version | S4 Score ↓ | S4 Δ vs official | S5 Score ↓ | S5 Δ vs official | Beats both baseline |
|---:|---|---:|---:|---:|---:|---|
| 1 | `v11_alpha035_ik2w2_from_v10` | `38.37255207578838` | `-0.2531054070144876` | `43.77819666691124` | `-0.033079871758815216` | yes |
| 2 | `v11_alpha025_ik2w3_from_v10` | `38.416982645660646` | `-0.2086748371422189` | `43.80445552650839` | `-0.0068210121616658625` | yes |
| 3 | `v11_alpha025_ik2w1_from_v10` | `38.40251200318336` | `-0.22314547961950382` | `43.84088326931` | `+0.029606730639940793` | no; S5 regresses |
| 4 | `v11_pRJ_only_ik2w2_from_v10` | `38.39572076609731` | `-0.22993671670555216` | `43.845466667562725` | `+0.034190128892667815` | no; S5 regresses |
| previous | `v10_ik2_input_distill_from_v6a` | `38.41680224156379` | `-0.20885524123907118` | `43.807055798713115` | `-0.0042207399569420545` | yes |

Real streaming IK1 diagnostics:

| Version | S4 pRJ L2 cm ↓ | S4 gR2 deg ↓ | S5 pRJ L2 cm ↓ | S5 gR2 deg ↓ |
|---|---:|---:|---:|---:|
| `v11_alpha025_ik2w1_from_v10` | `4.874484062194824` | `25.584686279296875` | `4.566864967346191` | `15.153098106384277` |
| `v11_alpha025_ik2w3_from_v10` | `4.87534761428833` | `25.584686279296875` | `4.567495822906494` | `15.153098106384277` |
| `v11_alpha035_ik2w2_from_v10` | `4.863900184631348` | `25.584686279296875` | `4.556811332702637` | `15.153098106384277` |
| `v11_pRJ_only_ik2w2_from_v10` | `4.986303806304932` | `25.584686279296875` | `4.583155632019043` | `15.153098106384277` |

Decision: select `v11_alpha035_ik2w2_from_v10` as the new current best because it improves both S4 and S5 over `newpl_v4_init36 + official IK1`, and also improves both S4/S5 over the Round 1 selected `v10_ik2_input_distill_from_v6a`.

Artifacts:

- Best checkpoint: `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt`
- Best S4 JSON: `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/s4/best_loss/result.json`
- Best S5 JSON: `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/s5/best_loss/result.json`
- Best S4 real audit: `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/real_streaming/s4/best_loss/result.json`
- Best S5 real audit: `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/real_streaming/s5/best_loss/result.json`
- Results CSV: `experiments/ik1_auto_search_results.csv`

Next round planned experiments: narrow around `v11_alpha035_ik2w2_from_v10`; test alpha `0.30`, alpha `0.40`, IK2-input distill `2.5`, and one conservative S5-stability recipe. Keep gR2 preserved or official-distilled only.

## IK1 Auto Search Round 3 Launch

Date: 2026-06-06

Goal: run a second narrow downstream-aware iteration from the Round 2 best candidate.

Parent:

```text
v11_alpha035_ik2w2_from_v10
data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt
```

Queue:

- Queue: `experiments/ik1_auto_search_round3_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round3_queue.json`
- Output root: `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/`
- Log root: `logs/orchestrator/ik1_auto_search/round3_downstream_aware_from_v11/`

Round 3 candidates:

| Experiment | Route | Init | alpha | IK2-input distill | S5/generalization intent |
|---|---|---|---:|---:|---|
| `v12_alpha030_ik2w2_from_v11` | alpha lower | v11 best | `0.30` | `2.0` | check if slightly lower residual improves S5 |
| `v12_alpha040_ik2w2_from_v11` | alpha higher | v11 best | `0.40` | `2.0` | check S4 gain without S5 regression |
| `v12_alpha035_ik2w25_from_v11` | stronger IK2 distill | v11 best | `0.35` | `2.5` | improve downstream compatibility |
| `v12_alpha035_s5stable_from_v11` | conservative S5 stability | v11 best | `0.35` | `2.5` | stronger pRJ distill/dynamics, lower LR |

Dry-run result:

- Task count: 20
- Ready tasks: 4 train tasks
- Planned local starts: GPU0 `v12_alpha030_ik2w2_from_v11`, GPU1 `v12_alpha040_ik2w2_from_v11`
- Remaining train tasks wait for GPU.
- Conflicts: none.

Selection rule remains S4/S5 full-pipeline 11 metrics only. S4-only gains with S5 regression are rejected.

### IK1 Auto Search Round 2 Evidence

Date: 2026-06-06

Status: completed 20/20 tasks, failed 0.

Selection baseline:

| Baseline | S4 Score ↓ | S5 Score ↓ |
|---|---:|---:|
| `newpl_v4_init36 + official IK1` | `38.625657482802865` | `43.81127653867006` |
| previous best `v10_ik2_input_distill_from_v6a` | `38.41680224156379` | `43.807055798713115` |

Round 2 full-pipeline ranking:

| Rank | Version | S4 Score ↓ | S4 Δ vs baseline | S5 Score ↓ | S5 Δ vs baseline | Decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | `v11_alpha035_ik2w2_from_v10` | `38.37255207578838` | `-0.2531054070144876` | `43.77819666691124` | `-0.033079871758815216` | selected Round 2 best |
| 2 | `v11_alpha025_ik2w3_from_v10` | `38.416982645660646` | `-0.2086748371422189` | `43.80445552650839` | `-0.0068210121616658625` | improves S5 but not v10 S4 |
| 3 | `v11_alpha025_ik2w1_from_v10` | `38.40251200318336` | `-0.22314547961950382` | `43.84088326931` | `+0.029606730639940793` | reject; S5 regresses |
| 4 | `v11_pRJ_only_ik2w2_from_v10` | `38.39572076609731` | `-0.22993671670555216` | `43.845466667562725` | `+0.034190128892667815` | reject; S5 regresses |

Real streaming IK1 diagnostics:

| Version | S4 pRJ L2 cm ↓ | S4 gR2 deg ↓ | S5 pRJ L2 cm ↓ | S5 gR2 deg ↓ |
|---|---:|---:|---:|---:|
| `v11_alpha025_ik2w1_from_v10` | `4.874484062194824` | `25.584686279296875` | `4.566864967346191` | `15.153098106384277` |
| `v11_alpha025_ik2w3_from_v10` | `4.87534761428833` | `25.584686279296875` | `4.567495822906494` | `15.153098106384277` |
| `v11_alpha035_ik2w2_from_v10` | `4.863900184631348` | `25.584686279296875` | `4.556811332702637` | `15.153098106384277` |
| `v11_pRJ_only_ik2w2_from_v10` | `4.986303806304932` | `25.584686279296875` | `4.583155632019043` | `15.153098106384277` |

Key trend:

- `alpha=0.35, ik2_input_distill=2.0` improves both S4 and S5 versus v10 and the official IK1 baseline.
- Increasing IK2 distill to `3.0` at alpha `0.25` improves S5 but loses S4 relative to v10.
- Lowering IK2 distill to `1.0` or switching to direct `pRJ_only` regresses S5 and should not be selected.
- gR2 remained unchanged as intended; gains are from pRJ residual/downstream compatibility.

Selected Round 2 checkpoint:

```text
data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt
```

Round 3 plan:

- Parent: `v11_alpha035_ik2w2_from_v10`
- Search only around `alpha=0.35` and IK2-input distill weight near `2.0`.
- Candidate directions: alpha `0.30/0.40`, IK2 weight `1.5/2.5`, keep gR2 preserved + official distill.
- Keep S4/S5 full 11 metrics as the only selection rule.

## IK1 Auto Search Round 3 Evidence

Date: 2026-06-06

Status: completed 20/20 tasks, failed 0.

Operational note: the first Round 3 resume stalled because remaining pending tasks were hard-bound to GPU0 while GPU0 was occupied by an unrelated long `test.py` process. The pending tasks only were remapped to GPU1 in `experiments/ik1_auto_search_round3_queue.yaml`; completed outputs were not rerun. Backup: `experiments/ik1_auto_search_round3_queue.yaml.bak_before_gpu1_resume_20260606_2253`.

Selection baseline:

| Baseline | S4 Score ↓ | S5 Score ↓ |
|---|---:|---:|
| `newpl_v4_init36 + official IK1` | `38.625657482802865` | `43.81127653867006` |
| previous best `v11_alpha035_ik2w2_from_v10` | `38.37255207578838` | `43.77819666691124` |

Round 3 full-pipeline ranking:

| Rank | Version | Route | S4 Score ↓ | S4 Δ vs official | S5 Score ↓ | S5 Δ vs official | Δ vs v11 S4/S5 | Decision |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `v12_alpha040_ik2w2_from_v11` | alpha `0.40`, IK2 distill `2.0` | `38.36728921282291` | `-0.2583682699799539` | `43.73998444685712` | `-0.07129209181293561` | `-0.005262862965466297` / `-0.038212220054120394` | selected current best |
| 2 | `v12_alpha035_ik2w25_from_v11` | alpha `0.35`, IK2 distill `2.5` | `38.386719339862466` | `-0.23893814294039828` | `43.75161533830688` | `-0.059661200363180455` | `0.014167264074089303` / `-0.02658132860436524` | beats both baselines; not selected |
| 3 | `v12_alpha035_s5stable_from_v11` | alpha `0.35`, IK2 distill `2.5`, LR `1e-6`, stronger pRJ distill/dynamics | `38.38921534772218` | `-0.23644213508068646` | `43.737316830046474` | `-0.07395970862358325` | `0.016663271933801127` / `-0.04087983686476804` | best S5 diagnostic candidate |
| 4 | `v12_alpha030_ik2w2_from_v11` | alpha `0.30`, IK2 distill `2.0` | `38.39537861308455` | `-0.2302788697183118` | `43.78341374022886` | `-0.027862798441198322` | `0.022826537296175786` / `0.005217073317616894` | beats both baselines; not selected |

Round 3 S4 full-pipeline 11 metrics:

| Metric | `v12_alpha040_ik2w2_from_v11` | `v12_alpha035_s5stable_from_v11` | `v12_alpha035_ik2w25_from_v11` | `v12_alpha030_ik2w2_from_v11` |
|---|---:|---:|---:|---:|
| Score | `38.36728921282291` | `38.38921534772218` | `38.386719339862466` | `38.39537861308455` |
| Local SIP | `10.064269065856934` | `10.07146110534668` | `10.070025157928466` | `10.071556854248048` |
| Local Angle | `8.722883987426759` | `8.72591257095337` | `8.725553798675538` | `8.726690864562988` |
| Local Joint | `4.466901874542236` | `4.4681251525878904` | `4.468177127838135` | `4.468818616867066` |
| Local Mesh | `5.1075506687164305` | `5.108399391174316` | `5.108660507202148` | `5.109324741363525` |
| Global SIP | `10.229679298400878` | `10.235895729064941` | `10.235164165496826` | `10.236503601074219` |
| Global Angle | `8.46994161605835` | `8.474865913391113` | `8.474906826019287` | `8.47910213470459` |
| Global Joint | `4.290406703948975` | `4.294867515563965` | `4.294702768325806` | `4.298629951477051` |
| Global Mesh | `4.819797563552856` | `4.826852416992187` | `4.826364660263062` | `4.832332468032837` |
| Root Jitter | `0.2872091166675091` | `0.28685052022337915` | `0.2869244858622551` | `0.2869093529880047` |
| Joint Jitter | `0.4784387230873108` | `0.47807621508836745` | `0.4781402125954628` | `0.4780301660299301` |

Round 3 S5 full-pipeline 11 metrics:

| Metric | `v12_alpha040_ik2w2_from_v11` | `v12_alpha035_s5stable_from_v11` | `v12_alpha035_ik2w25_from_v11` | `v12_alpha030_ik2w2_from_v11` |
|---|---:|---:|---:|---:|
| Score | `43.73998444685712` | `43.737316830046474` | `43.75161533830688` | `43.78341374022886` |
| Local SIP | `9.61759603023529` | `9.619295358657837` | `9.62232220172882` | `9.631479263305664` |
| Local Angle | `12.450390100479126` | `12.448022603988647` | `12.45108699798584` | `12.456440687179565` |
| Local Joint | `4.328210830688477` | `4.331571042537689` | `4.332240521907806` | `4.336070895195007` |
| Local Mesh | `5.104433059692383` | `5.103365182876587` | `5.105639100074768` | `5.109046816825867` |
| Global SIP | `9.101807713508606` | `9.101602673530579` | `9.106214761734009` | `9.116354823112488` |
| Global Angle | `11.748084783554077` | `11.745819091796875` | `11.749263763427734` | `11.755742311477661` |
| Global Joint | `3.8121743202209473` | `3.8136839866638184` | `3.8145220279693604` | `3.8175302743911743` |
| Global Mesh | `4.407786011695862` | `4.410418272018433` | `4.411274552345276` | `4.415165901184082` |
| Root Jitter | `0.3754016747698188` | `0.37463897094130516` | `0.3746604258194566` | `0.3740515233948827` |
| Joint Jitter | `0.8067303989082575` | `0.8051599152386189` | `0.8051358442753553` | `0.8036538194864988` |

Round 3 real streaming IK1 diagnostics:

| Version | S4 pRJ L2 cm ↓ | S4 gR2 deg ↓ | S5 pRJ L2 cm ↓ | S5 gR2 deg ↓ |
|---|---:|---:|---:|---:|
| `v12_alpha040_ik2w2_from_v11` | `4.862191200256348` | `25.584686279296875` | `4.553211212158203` | `15.153098106384277` |
| `v12_alpha035_ik2w25_from_v11` | `4.865252494812012` | `25.584686279296875` | `4.557664394378662` | `15.153098106384277` |
| `v12_alpha035_s5stable_from_v11` | `4.865253925323486` | `25.584686279296875` | `4.557403087615967` | `15.153098106384277` |
| `v12_alpha030_ik2w2_from_v11` | `4.86892032623291` | `25.584686279296875` | `4.561491012573242` | `15.153098106384277` |

Selected Round 3 checkpoint:

```text
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Artifacts:

- Queue: `experiments/ik1_auto_search_round3_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round3_queue.json`
- Results CSV: `experiments/ik1_auto_search_results.csv`
- Selected S4 JSON: `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/s4/best_loss/result.json`
- Selected S5 JSON: `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/s5/best_loss/result.json`
- Selected S4 real audit: `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/real_streaming/s4/best_loss/result.json`
- Selected S5 real audit: `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/real_streaming/s5/best_loss/result.json`

Key trend: increasing residual alpha from `0.35` to `0.40` at IK2-input distill `2.0` gives the best S4 and still improves S5. IK2 distill `2.5` improves S5 stability but weakens S4. Alpha `0.30` is too conservative and loses the Round 2 S5 improvement. gR2 metrics are unchanged by design.

Next round planned experiments: optional Round 4 should test only alpha `0.38`/`0.42` at IK2-input distill `2.0`, plus one alpha `0.40` with IK2 distill `2.25` S5-stability check. No random search and no gR2-heavy route.

## IK1 Auto Search Round 4 Launch

Date: 2026-06-06

Goal: run a third narrow downstream-aware iteration from the Round 3 best candidate.

Parent:

```text
v12_alpha040_ik2w2_from_v11
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Queue:

- Queue: `experiments/ik1_auto_search_round4_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round4_queue.json`
- Output root: `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/`
- Log root: `logs/orchestrator/ik1_auto_search/round4_downstream_aware_from_v12/`

Round 4 candidates:

| Experiment | Route | Init | alpha | IK2-input distill | S5/generalization intent |
|---|---|---|---:|---:|---|
| `v13_alpha038_ik2w2_from_v12` | alpha lower near best | v12 best | `0.38` | `2.0` | check if slight alpha reduction preserves S5 without S4 loss |
| `v13_alpha042_ik2w2_from_v12` | alpha higher near best | v12 best | `0.42` | `2.0` | test whether S4 continues improving without S5 regression |
| `v13_alpha040_ik2w225_from_v12` | stronger IK2 distill | v12 best | `0.40` | `2.25` | small S5-stability increase |
| `v13_alpha040_s5stable_from_v12` | conservative S5 stability | v12 best | `0.40` | `2.25` | lower LR, stronger pRJ distill/dynamics |

Dry-run result:

- Task count: 20
- Ready tasks: 4 train tasks
- Planned local starts: GPU0 `v13_alpha038_ik2w2_from_v12`, GPU1 `v13_alpha042_ik2w2_from_v12`
- Remaining train tasks wait for GPU.
- Conflicts: none.

Selection rule remains S4/S5 full-pipeline 11 metrics only. Real streaming IK1 metrics are diagnostic only. gR2 remains preserved / official-distilled only.

## IK1 Auto Search Remote Scheduler Health Check

Date: 2026-06-07

Purpose: verify whether the second server can be used for future two-server / four-GPU scheduling. This was a non-training health check only; it does not change Round 2-5 experimental evidence.

Command:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 zktitan 'hostname; pwd; test -d /home/lingfeng/projects/GlobalposeMy/GlobalPose && echo PROJECT_OK || echo PROJECT_MISSING; test -x /home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python && echo PY_OK || echo PY_MISSING; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
```

Result:

```text
zktitan
/home/lingfeng
PROJECT_OK
PY_OK
0, NVIDIA GeForce RTX 5090, 29, 32607, 0
1, NVIDIA GeForce RTX 5090, 29, 32607, 0
```

Conclusion: `zktitan` is reachable and has two idle RTX 5090 GPUs. Future queues may use `node01` plus `zktitan` with explicit per-task `CUDA_VISIBLE_DEVICES`. The completed Round 2-5 runs remain local `node01` evidence; do not rewrite them as remote runs.

## IK1 Auto Search Round 5 Split Launch

Date: 2026-06-07

Purpose: start a real two-server confirmation-only queue. This is not a random search and does not change the current selected best until full S4/S5 evidence is complete.

Parent:

```text
v12_alpha040_ik2w2_from_v11
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Candidate set:

| Experiment | Server | GPU | Route |
|---|---|---:|---|
| `v14_repeat_alpha040_ik2w2_from_v12` | node01 | 1 | repeat current alpha `0.40`, IK2-input distill `2.0` |
| `v14_alpha041_ik2w2_from_v12` | node01 | 0 | midpoint alpha `0.41`, IK2-input distill `2.0` |
| `v14_confirm_alpha042_ik2w2_from_v12` | zktitan | 0 | confirm alpha `0.42`, IK2-input distill `2.0` |
| `v14_alpha041_ik2w21_from_v12` | zktitan | 1 | midpoint alpha `0.41`, IK2-input distill `2.1` |

Dry-run evidence:

- Local queue: 10 tasks, 2 train tasks ready, planned node01 GPU1/GPU0, no output/log conflicts.
- Remote queue: 10 tasks, 2 train tasks ready, planned zktitan GPU0/GPU1, no output/log conflicts.

Launch commands:

```bash
screen -dmS ik1_r5_local bash -lc 'cd /home/lingfeng/projects/GlobalposeMy/GlobalPose && /home/lingfeng/bin/longrun -- python tools/experiment_orchestrator.py --task-file experiments/ik1_auto_search_round5_local_queue.yaml --run > logs/orchestrator/ik1_auto_search/round5_split_launch/local_screen_longrun.log 2>&1'
screen -dmS ik1_r5_remote bash -lc 'cd /home/lingfeng/projects/GlobalposeMy/GlobalPose && /home/lingfeng/bin/longrun -- ssh -o BatchMode=yes zktitan "cd /home/lingfeng/projects/GlobalposeMy/GlobalPose && /home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python tools/experiment_orchestrator.py --task-file experiments/ik1_auto_search_round5_remote_queue.yaml --run" > logs/orchestrator/ik1_auto_search/round5_split_launch/remote_screen_longrun.log 2>&1'
```

Initial state check:

| State file | Status summary |
|---|---|
| `data/experiments/orchestrator_states/ik1_auto_search_round5_local_queue.json` | 2 completed train tasks, 2 S4 eval tasks running, 6 pending |
| `data/experiments/orchestrator_states/ik1_auto_search_round5_remote_queue.json` | 2 train tasks running, 8 pending |

Launch logs:

- `logs/orchestrator/ik1_auto_search/round5_split_launch/local_screen_longrun.log`
- `logs/orchestrator/ik1_auto_search/round5_split_launch/remote_screen_longrun.log`

Completion notification: local `/home/lingfeng/bin/longrun` wraps both the local orchestrator and the SSH remote orchestrator, so completion/failure emails should be sent when each queue exits.

## IK1 Auto Search Round 5 Evidence

Date: 2026-06-07

Status: completed 20/20 tasks, failed 0.

Baseline:

| Version | S4 Score | S5 Score |
|---|---:|---:|
| official `newpl_v4_init36 + official IK1` | `38.625657482802865` | `43.81127653867006` |
| current best `v12_alpha040_ik2w2_from_v11` | `38.36728921282291` | `43.73998444685712` |

Round 5 full-pipeline ranking:

| Rank | Version | Route | S4 Score | S4 Delta vs official | S5 Score | S5 Delta vs official | Delta vs current best S4/S5 | Decision |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `v14_alpha0410_ik2w21_from_v12` | alpha `0.410`, IK2 distill `2.1` | `38.36968127383291` | `-0.25597620896995466` | `43.719431065004315` | `-0.09184547366574236` | `+0.0023920610099992246` / `-0.02055338185280675` | best Round 5 S4, but does not beat current v12 S4 |
| 2 | `v14_alpha0415_ik2w2_from_v12` | alpha `0.415`, IK2 distill `2.0` | `38.37063280807435` | `-0.2550246747285172` | `43.718751320485026` | `-0.09252521818503112` | `+0.003343595251436682` / `-0.021233126372095512` | best Round 5 S5, diagnostic only because S4 is weaker |
| 3 | `v14_alpha0405_ik2w2_from_v12` | alpha `0.405`, IK2 distill `2.0` | `38.372773325160146` | `-0.2528841576427183` | `43.725488430112605` | `-0.08578810855745189` | `+0.005484112337235558` / `-0.014496016744516282` | beats official baseline, not current best |
| 4 | `v14_alpha0410_ik2w2_from_v12` | alpha `0.410`, IK2 distill `2.0` | `38.37348917667567` | `-0.25216830612719576` | `43.72153897484764` | `-0.0897375638224176` | `+0.006199963852758117` / `-0.01844547200948199` | beats official baseline, not current best |

Round 5 S4 full-pipeline 11 metrics:

| Metric | `v14_alpha0405_ik2w2_from_v12` | `v14_alpha0410_ik2w2_from_v12` | `v14_alpha0415_ik2w2_from_v12` | `v14_alpha0410_ik2w21_from_v12` |
|---|---:|---:|---:|---:|
| Score | `38.372773325160146` | `38.37348917667567` | `38.37063280807435` | `38.36968127383291` |
| Local SIP | `10.066407489776612` | `10.066746234893799` | `10.065557193756103` | `10.06557502746582` |
| Local Angle | `8.723906421661377` | `8.724196338653565` | `8.724200439453124` | `8.723701572418213` |
| Local Joint | `4.467434072494507` | `4.467832136154175` | `4.4676093578338625` | `4.467196798324585` |
| Local Mesh | `5.108246803283691` | `5.108583879470825` | `5.108386039733887` | `5.107817125320435` |
| Global SIP | `10.231533432006836` | `10.231786918640136` | `10.230517864227295` | `10.230309104919433` |
| Global Angle | `8.470336151123046` | `8.470143032073974` | `8.469825649261475` | `8.469593906402588` |
| Global Joint | `4.29062066078186` | `4.290491676330566` | `4.289862251281738` | `4.2899760723114015` |
| Global Mesh | `4.820356750488282` | `4.82004132270813` | `4.819163751602173` | `4.819415521621704` |
| Root Jitter | `0.2872064270079136` | `0.2871894180774689` | `0.2872119329869747` | `0.2871938951313496` |
| Joint Jitter | `0.47843572646379473` | `0.4784271165728569` | `0.478450046479702` | `0.4784375563263893` |

Round 5 S5 full-pipeline 11 metrics:

| Metric | `v14_alpha0405_ik2w2_from_v12` | `v14_alpha0410_ik2w2_from_v12` | `v14_alpha0415_ik2w2_from_v12` | `v14_alpha0410_ik2w21_from_v12` |
|---|---:|---:|---:|---:|
| Score | `43.725488430112605` | `43.72153897484764` | `43.718751320485026` | `43.719431065004315` |
| Local SIP | `9.613972902297974` | `9.613159656524658` | `9.612117290496826` | `9.612767815589905` |
| Local Angle | `12.44599986076355` | `12.445382833480835` | `12.445310354232788` | `12.445660591125488` |
| Local Joint | `4.327340245246887` | `4.327044785022736` | `4.326860070228577` | `4.326795279979706` |
| Local Mesh | `5.10241973400116` | `5.102129817008972` | `5.102001309394836` | `5.101636171340942` |
| Global SIP | `9.099478006362915` | `9.097949385643005` | `9.09680986404419` | `9.096270203590393` |
| Global Angle | `11.744041919708252` | `11.743125438690186` | `11.742653369903564` | `11.742905139923096` |
| Global Joint | `3.8119316697120667` | `3.811467170715332` | `3.8110249638557434` | `3.810771942138672` |
| Global Mesh | `4.407502233982086` | `4.407057285308838` | `4.4064154624938965` | `4.406194746494293` |
| Root Jitter | `0.37544210627675056` | `0.37558136228471994` | `0.37565048690885305` | `0.3755748961120844` |
| Joint Jitter | `0.8068549484014511` | `0.8070464935153723` | `0.8071938399225473` | `0.8070592563599348` |

Round 5 real streaming IK1 diagnostics:

| Version | S4 pRJ L1 cm | S4 pRJ L2 cm | S4 gR2 deg | S4 pRJ dot | S4 pRJ ddot | S5 pRJ L1 cm | S5 pRJ L2 cm | S5 gR2 deg | S5 pRJ dot | S5 pRJ ddot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v14_alpha0405_ik2w2_from_v12` | `2.4081618785858154` | `4.862767696380615` | `25.584686279296875` | `0.026267537847161293` | `0.022065144032239914` | `2.2199244499206543` | `4.553722381591797` | `15.153098106384277` | `0.025460511445999146` | `0.0199979767203331` |
| `v14_alpha0410_ik2w2_from_v12` | `2.4083924293518066` | `4.863192558288574` | `25.584686279296875` | `0.026264948770403862` | `0.022061819210648537` | `2.219696044921875` | `4.553475379943848` | `15.153098106384277` | `0.025449691340327263` | `0.01998515985906124` |
| `v14_alpha0415_ik2w2_from_v12` | `2.40818190574646` | `4.8627705574035645` | `25.584686279296875` | `0.026262899860739708` | `0.022060329094529152` | `2.219625473022461` | `4.553338527679443` | `15.153098106384277` | `0.02544444054365158` | `0.01997881568968296` |
| `v14_alpha0410_ik2w21_from_v12` | `2.4077656269073486` | `4.861997127532959` | `25.584686279296875` | `0.026264097541570663` | `0.022062087431550026` | `2.2190394401550293` | `4.552035808563232` | `15.153098106384277` | `0.025449231266975403` | `0.019985657185316086` |

Diagnostic relation:

- All four v14 candidates improve both S4 and S5 versus the original official IK1 baseline.
- No v14 candidate improves both S4 and S5 versus the current selected `v12_alpha040_ik2w2_from_v11`.
- pRJ diagnostics improve slightly as alpha/IK2 distill increase, while gR2 stays unchanged as intended.
- Because S4 full-pipeline score worsens whenever S5 improves, the likely bottleneck is downstream compatibility or per-sequence S4 sensitivity, not local pRJ/gR2 closeness alone.

Selected candidate remains:

```text
processed IMU + newpl_v4_init36 + v12_alpha040_ik2w2_from_v11
```

Selected checkpoint remains:

```text
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Artifacts:

- Queue: `experiments/ik1_auto_search_round5_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round5_queue.json`
- Results CSV: `experiments/ik1_auto_search_results.csv`
- Output root: `data/experiments/ik1_auto_search/round5_confirmation_from_v12/`
- Log root: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/`

Next round planned experiments: stop broad alpha/IK2-weight search. If continuing, run a targeted bottleneck audit of S4 per-sequence failures and downstream IK2/VR compatibility for `v12` versus `v14_alpha0410_ik2w21_from_v12` and `v14_alpha0415_ik2w2_from_v12`; do not launch another four-candidate blind sweep.

## IK1 Auto Search Round 4 Evidence

Date: 2026-06-07

Status: completed 20/20 tasks, failed 0.

Selection baseline:

| Baseline | S4 Score ↓ | S5 Score ↓ |
|---|---:|---:|
| `newpl_v4_init36 + official IK1` | `38.625657482802865` | `43.81127653867006` |
| current best `v12_alpha040_ik2w2_from_v11` | `38.36728921282291` | `43.73998444685712` |

Round 4 full-pipeline ranking:

| Rank | Version | Route | S4 Score ↓ | S4 Δ vs official | S5 Score ↓ | S5 Δ vs official | Δ vs v12 S4/S5 | Decision |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `v13_alpha042_ik2w2_from_v12` | alpha `0.42`, IK2 distill `2.0` | `38.370207245603204` | `-0.25545023719966053` | `43.71751535784453` | `-0.09376118082552409` | `+0.002918032780293345` / `-0.022469089012588483` | best Round 4 S4, but does not beat current v12 S4 |
| 2 | `v13_alpha038_ik2w2_from_v12` | alpha `0.38`, IK2 distill `2.0` | `38.376336369305854` | `-0.24932111349701103` | `43.73781035907567` | `-0.07346617959439072` | `+0.009047156482942853` / `-0.002174087781455114` | beats official baseline, not current best |
| 3 | `v13_alpha040_ik2w225_from_v12` | alpha `0.40`, IK2 distill `2.25` | `38.3787378231883` | `-0.2469196596145622` | `43.71488639246672` | `-0.09639014620334052` | `+0.011448610365391687` / `-0.02509805439040491` | beats official baseline, not current best |
| 4 | `v13_alpha040_s5stable_from_v12` | alpha `0.40`, IK2 distill `2.25`, LR `8e-7`, stronger pRJ distill/dynamics | `38.3813811891675` | `-0.24427629363536596` | `43.707128487303855` | `-0.10414805136620231` | `+0.01409197634458792` / `-0.0328559595532667` | best Round 4 S5, diagnostic only because S4 weaker |

Round 4 S4 full-pipeline 11 metrics:

| Metric | `v13_alpha042_ik2w2_from_v12` | `v13_alpha038_ik2w2_from_v12` | `v13_alpha040_ik2w225_from_v12` | `v13_alpha040_s5stable_from_v12` |
|---|---:|---:|---:|---:|
| Score | `38.370207245603204` | `38.376336369305854` | `38.3787378231883` | `38.3813811891675` |
| Local SIP | `10.065584087371827` | `10.067440795898438` | `10.068017387390137` | `10.069668865203857` |
| Local Angle | `8.723765754699707` | `8.724551963806153` | `8.725085735321045` | `8.725146579742432` |
| Local Joint | `4.467439889907837` | `4.4677411079406735` | `4.467905569076538` | `4.467659664154053` |
| Local Mesh | `5.108120441436768` | `5.10846700668335` | `5.108921670913697` | `5.108443784713745` |
| Global SIP | `10.230829811096191` | `10.231966114044189` | `10.233187484741212` | `10.234616470336913` |
| Global Angle | `8.469501781463624` | `8.471617317199707` | `8.471712875366212` | `8.471254444122314` |
| Global Joint | `4.2899498462677` | `4.292027521133423` | `4.291605520248413` | `4.2914573669433596` |
| Global Mesh | `4.81909556388855` | `4.822668361663818` | `4.8217607021331785` | `4.821690893173217` |
| Root Jitter | `0.2873816750943661` | `0.2870557144284248` | `0.2870852373540401` | `0.2870588548481464` |
| Joint Jitter | `0.4786837354302406` | `0.47833154499530794` | `0.47832314372062684` | `0.47831266522407534` |

Round 4 S5 full-pipeline 11 metrics:

| Metric | `v13_alpha042_ik2w2_from_v12` | `v13_alpha038_ik2w2_from_v12` | `v13_alpha040_ik2w225_from_v12` | `v13_alpha040_s5stable_from_v12` |
|---|---:|---:|---:|---:|
| Score | `43.71751535784453` | `43.73781035907567` | `43.71488639246672` | `43.707128487303855` |
| Local SIP | `9.612090110778809` | `9.618024587631226` | `9.61136519908905` | `9.60938012599945` |
| Local Angle | `12.444246053695679` | `12.448243856430054` | `12.443263053894043` | `12.441551685333252` |
| Local Joint | `4.325650990009308` | `4.329896986484528` | `4.326895356178284` | `4.326491951942444` |
| Local Mesh | `5.1008830070495605` | `5.10431432723999` | `5.100820183753967` | `5.099248051643372` |
| Global SIP | `9.096914529800415` | `9.103167057037354` | `9.096797227859497` | `9.094338417053223` |
| Global Angle | `11.742501974105835` | `11.746022939682007` | `11.741515398025513` | `11.740009546279907` |
| Global Joint | `3.8112383484840393` | `3.8130151629447937` | `3.811891257762909` | `3.811325490474701` |
| Global Mesh | `4.406506538391113` | `4.409253895282745` | `4.407286643981934` | `4.406897842884064` |
| Root Jitter | `0.37572029046714306` | `0.37509846314787865` | `0.3753936253488064` | `0.37536901514977217` |
| Joint Jitter | `0.8073755614459515` | `0.8060703352093697` | `0.8066852204501629` | `0.8066968396306038` |

Round 4 real streaming IK1 diagnostics:

| Version | S4 pRJ L2 cm ↓ | S4 gR2 deg ↓ | S5 pRJ L2 cm ↓ | S5 gR2 deg ↓ |
|---|---:|---:|---:|---:|
| `v13_alpha042_ik2w2_from_v12` | `4.862273693084717` | `25.584686279296875` | `4.551540374755859` | `15.153098106384277` |
| `v13_alpha038_ik2w2_from_v12` | `4.86371374130249` | `25.584686279296875` | `4.555866718292236` | `15.153098106384277` |
| `v13_alpha040_ik2w225_from_v12` | `4.863152503967285` | `25.584686279296875` | `4.553428649902344` | `15.153098106384277` |
| `v13_alpha040_s5stable_from_v12` | `4.862794399261475` | `25.584686279296875` | `4.552684783935547` | `15.153098106384277` |

Selected checkpoint remains:

```text
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Artifacts:

- Queue: `experiments/ik1_auto_search_round4_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round4_queue.json`
- Results CSV: `experiments/ik1_auto_search_results.csv`
- Best Round 4 S4 JSON: `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/s4/best_loss/result.json`
- Best Round 4 S5 JSON: `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/s5/best_loss/result.json`

Key trend: Round 4 improves S5 at the cost of S4. The S4/S5 tradeoff suggests the current best `v12_alpha040_ik2w2_from_v11` is still the best combined full-pipeline candidate.

Next round planned experiments: stop broad search. If more evidence is needed, run only confirmation/midpoint checks: repeat `v12_alpha040_ik2w2_from_v11` and `v13_alpha042_ik2w2_from_v12`, or one `alpha=0.41, ik2_input_distill=2.0` candidate with full S4/S5 eval.

## Official GPNet TotalCapture fine-tune diagnostic

Date: 2026-06-07

This is a diagnostic adaptation experiment, not the official training protocol.

Purpose: test whether the official GPNet checkpoint itself improves when fine-tuned on TotalCapture before TotalCapture testing. This directly checks whether a TotalCapture adaptation advantage exists before judging TotalCapture-finetuned NewIK1 results against an unfine-tuned official baseline.

Protocol note:

- The official paper protocol is AMASS pretrain -> DIP-IMU train split fine-tune -> DIP-IMU test / TotalCapture evaluation.
- This diagnostic uses TotalCapture for fine-tune and TotalCapture for test.
- It is only evidence about TotalCapture adaptation advantage.
- It must not be reported as the official protocol or as a paper-style cross-dataset generalization result.

Implementation:

| Field | Value |
|---|---|
| Script | `scripts/finetune_official_gpnet_totalcapture.py` |
| Official checkpoint | `data/weights.pt` |
| Checkpoint loading | full `GPNet.state_dict`; not split-module checkpoint loading |
| Model structure | unchanged official `GPNet` |
| Replacements | none; no NewPL init36, no IK1 replacement, no PL/IK2/VR replacement |
| Train data | `data/dataset_work/TotalCapture_globalpose_official/train.pt` |
| Validation data | `data/dataset_work/TotalCapture_globalpose_official/val.pt` |
| Test data | `data/dataset_work/TotalCapture_globalpose_official/test.pt` |
| Trainable modules | `plnet`, `iknet.net1`, `iknet.net2`, `vrnet` |
| Frozen modules | none |
| Loss | PL `pRB/gR`, IK1 `pRJ/gR2`, IK2 reduced-global 6D rotation, VR root velocity; stationary/contact GT not measured |
| Selection | best validation loss checkpoint |

Commands:

```bash
ENV=/home/lingfeng/.conda/envs/globalpose-gpu
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
PY="$ENV/bin/python"
ROOT=data/experiments/official_gpnet_totalcapture_finetune_diagnostic
SCRIPT=scripts/finetune_official_gpnet_totalcapture.py

$PY "$SCRIPT" --action eval --official_ckpt data/weights.pt --eval_ckpt data/weights.pt --data totalcapture --eval_split test --output_json "$ROOT/baseline/eval_test.json" --device cuda
$PY "$SCRIPT" --action train --official_ckpt data/weights.pt --data totalcapture --lr 1e-5 --epochs 2 --batch_size 2 --window 180 --save_dir "$ROOT/FT-A_lr1e-5_ep2" --device cuda
$PY "$SCRIPT" --action eval --official_ckpt data/weights.pt --eval_ckpt "$ROOT/FT-A_lr1e-5_ep2/best_weights.pt" --data totalcapture --eval_split test --output_json "$ROOT/FT-A_lr1e-5_ep2/eval_test.json" --device cuda
$PY "$SCRIPT" --action train --official_ckpt data/weights.pt --data totalcapture --lr 3e-6 --epochs 2 --batch_size 2 --window 180 --save_dir "$ROOT/FT-B_lr3e-6_ep2" --device cuda
$PY "$SCRIPT" --action eval --official_ckpt data/weights.pt --eval_ckpt "$ROOT/FT-B_lr3e-6_ep2/best_weights.pt" --data totalcapture --eval_split test --output_json "$ROOT/FT-B_lr3e-6_ep2/eval_test.json" --device cuda
$PY "$SCRIPT" --action train --official_ckpt data/weights.pt --data totalcapture --lr 1e-6 --epochs 2 --batch_size 2 --window 180 --save_dir "$ROOT/FT-C_lr1e-6_ep2" --device cuda
$PY "$SCRIPT" --action eval --official_ckpt data/weights.pt --eval_ckpt "$ROOT/FT-C_lr1e-6_ep2/best_weights.pt" --data totalcapture --eval_split test --output_json "$ROOT/FT-C_lr1e-6_ep2/eval_test.json" --device cuda
```

Training results:

| Version | LR | Epochs | Status | Best epoch | Best validation loss | Trainable modules | Frozen modules |
|---|---:|---:|---|---:|---:|---|---|
| FT-A | `1e-5` | 2 | ok | 2 | `1.8420624017715455` | `plnet`, `iknet.net1`, `iknet.net2`, `vrnet` | none |
| FT-B | `3e-6` | 2 | ok | 2 | `1.8670690298080443` | `plnet`, `iknet.net1`, `iknet.net2`, `vrnet` | none |
| FT-C | `1e-6` | 2 | ok | 2 | `1.8737063884735108` | `plnet`, `iknet.net1`, `iknet.net2`, `vrnet` | none |

TotalCapture test full 11 metrics:

| Version | Train data | LR | Epochs | Trainable modules | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_gpnet_original | none | not found | 0 | none | 44.477380 | 9.989696 | 12.550694 | 4.519671 | 5.300476 | 9.314009 | 11.781373 | 3.810419 | 4.470022 | 0.399826 | 0.859838 |
| FT-A | TotalCapture train | 1e-5 | 2 | full GPNet neural modules | 43.349149 | 9.603109 | 12.205352 | 4.340323 | 5.041633 | 9.143327 | 11.585930 | 3.688516 | 4.298988 | 0.397926 | 0.854695 |
| FT-B | TotalCapture train | 3e-6 | 2 | full GPNet neural modules | 43.782556 | 9.765441 | 12.379111 | 4.433093 | 5.167890 | 9.155503 | 11.656312 | 3.742955 | 4.361113 | 0.399393 | 0.858540 |
| FT-C | TotalCapture train | 1e-6 | 2 | full GPNet neural modules | 44.191540 | 9.900780 | 12.483899 | 4.487517 | 5.249972 | 9.241099 | 11.730035 | 3.783797 | 4.426778 | 0.399826 | 0.859736 |

Delta Score versus original:

| Version | Delta Score ↓ |
|---|---:|
| FT-A | `-1.1282314625568688` |
| FT-B | `-0.6948241149820404` |
| FT-C | `-0.28584031155332923` |

Conclusion: official GPNet does improve after TotalCapture fine-tuning in this diagnostic setup. Best version is FT-A, improving Score by `1.1282314625568688` lower-is-better points. This means previous comparisons where NewIK1 was TotalCapture-finetuned but official GPNet was not TotalCapture-finetuned are not fairness-complete. Future NewIK1 TotalCapture-finetuned results should compare against a similarly TotalCapture-finetuned official GPNet baseline, with FT-A as the current diagnostic baseline for this split.

Artifacts:

- Baseline JSON: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/baseline/eval_test.json`
- FT-A checkpoint: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-A_lr1e-5_ep2/best_weights.pt`
- FT-A JSON: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-A_lr1e-5_ep2/eval_test.json`
- FT-B checkpoint: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-B_lr3e-6_ep2/best_weights.pt`
- FT-B JSON: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-B_lr3e-6_ep2/eval_test.json`
- FT-C checkpoint: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-C_lr1e-6_ep2/best_weights.pt`
- FT-C JSON: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-C_lr1e-6_ep2/eval_test.json`
- Batch log: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/logs/run_all.log`

## IK1 Auto Search Round 5 Launch

Date: 2026-06-07

Goal: confirmation-only narrow search from the strict S4 best Round 3 candidate.

Parent:

```text
v12_alpha040_ik2w2_from_v11
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Queue:

- Queue: `experiments/ik1_auto_search_round5_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round5_queue.json`
- Output root: `data/experiments/ik1_auto_search/round5_confirmation_from_v12/`
- Log root: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/`

Round 5 candidates:

| Experiment | Route | Init | alpha | IK2-input distill | S5/generalization intent |
|---|---|---|---:|---:|---|
| `v14_alpha0405_ik2w2_from_v12` | alpha midpoint | v12 strict best | `0.405` | `2.0` | check whether a tiny alpha increase preserves S4 |
| `v14_alpha0410_ik2w2_from_v12` | alpha midpoint | v12 strict best | `0.410` | `2.0` | midpoint between v12 and v13 |
| `v14_alpha0415_ik2w2_from_v12` | alpha midpoint | v12 strict best | `0.415` | `2.0` | S5-stability check while keeping S4 close |
| `v14_alpha0410_ik2w21_from_v12` | slight IK2 distill increase | v12 strict best | `0.410` | `2.1` | small S5-stability increase |

Dry-run result:

- Task count: 20
- Ready tasks: 4 train tasks
- Planned local starts: GPU1 `v14_alpha0405_ik2w2_from_v12`
- GPU0 was occupied by unrelated `official_gpnet_totalcapture_finetune_diagnostic` eval; no process was killed.
- Remaining train tasks wait for GPU.
- Conflicts: none.

Selection rule remains S4/S5 full-pipeline 11 metrics only. Real streaming IK1 metrics are diagnostic only. gR2 remains preserved / official-distilled only.

## IMUOffsetNet Stage A/B/C Smoke + Task Setup

Date: 2026-06-07

Purpose: add an IMU position-offset estimator without fabricating real-data offset GT. Offset contract is `r_JS`: IMU origin relative to mapped joint J, expressed in joint-local coordinates; `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS`.

Implemented scripts:

- `imu_offset_net.py`: `offset_v1_mlp_frame`, `offset_v2_temporal_rnn`, `offset_v3_residual_prior`, supervised loss, regularizers, acceleration consistency audit.
- `imu_offset_train.py`: Stage A synthetic supervised training on AMASS shards with `imu_offset_r/r_JS` GT.
- `imu_offset_finetune_dip.py`: Stage B DIP official-input fine-tune using `aM/wM/RMB`, no processed IMU, no trans loss, no offset GT loss.
- `imu_offset_infer.py`: checkpoint inference into sequence-level cache `offset_r [6,3]`, with real-data offset accuracy marked not available.
- `configs/imu_offset_net_20260607_tasks.json`: formal Stage A/B/C task graph.

Smoke commands used GPU1 unless noted:

```bash
python imu_offset_train.py --version offset_v1_mlp_frame --output-dir data/experiments/imu_offset_net_stageA_smoke/v1 --max-shards 1 --max-sequences 4 --max-frames 180 --window 60 --windows-per-sequence 1 --val-windows-per-sequence 1 --epochs 1 --hidden-size 64 --acc-device cpu
python imu_offset_train.py --version offset_v2_temporal_rnn --output-dir data/experiments/imu_offset_net_stageA_smoke/v2 --max-shards 1 --max-sequences 4 --max-frames 180 --window 60 --windows-per-sequence 1 --val-windows-per-sequence 1 --epochs 1 --hidden-size 64 --acc-device cpu
python imu_offset_train.py --version offset_v3_residual_prior --output-dir data/experiments/imu_offset_net_stageA_smoke/v3 --max-shards 1 --max-sequences 4 --max-frames 180 --window 60 --windows-per-sequence 1 --val-windows-per-sequence 1 --epochs 1 --hidden-size 64 --prior-weight 0.01 --smooth-weight 0.001 --acc-device cpu
python imu_offset_finetune_dip.py --init-checkpoint data/experiments/imu_offset_net_stageA_smoke/v3/best_loss.pt --output-dir data/experiments/imu_offset_net_stageB_smoke/v3_dip_ft --epochs 1 --max-train-sequences 1 --max-val-sequences 1 --max-frames 180 --window 60 --windows-per-sequence 1 --val-windows-per-sequence 1 --lr 1e-5
python imu_offset_infer.py --checkpoint data/experiments/imu_offset_net_stageA_smoke/v3/best_loss.pt --input-cache data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json --output-dir data/experiments/imu_offset_net_stageB_smoke/dip_train_v3_pred_offset --imu-input-mode official --pl-source pose_prephysics --offset-gt-mode unavailable --acc-audit-mode auto --max-sequences 1 --max-frames 200 --window 80 --stride 80
python imu_offset_infer.py --checkpoint data/experiments/imu_offset_net_stageA_smoke/v3/best_loss.pt --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/imu_offset_net_stageC_smoke/tc_val_v3_pred_offset --imu-input-mode processed --pl-source pose_prephysics --offset-gt-mode unavailable --acc-audit-mode pose_gt --max-sequences 1 --max-frames 200 --window 80 --stride 80
```

Smoke results:

| Version | Stage | Input data | Offset L2 synthetic ↓ | Acc consistency ↓ | Real offset GT |
|---|---|---|---:|---:|---|
| `offset_v1_mlp_frame` | A smoke | AMASS synthetic | `12.3658 cm` | `4.0917` | available |
| `offset_v2_temporal_rnn` | A smoke | AMASS synthetic | `13.2249 cm` | `4.2218` | available |
| `offset_v3_residual_prior` | A smoke | AMASS synthetic | `4.3275 cm` | `1.5181` | available |
| `offset_v3_residual_prior` | B smoke | DIP train official `aM/wM/RMB` | not available | pose-proxy `0.00804` | not available |
| `offset_v3_residual_prior` | C smoke | TotalCapture processed IMU | not available | `2.9717` | not available |

Downstream smoke, TotalCapture S4 one sequence, processed IMU, same PL checkpoint:

| Cache offset source | Score ↓ | Delta |
|---|---:|---:|
| existing baseline/cache offset | `42.153346` | reference |
| IMUOffsetNet predicted offset | `42.153350` | `+0.000004` |

Dry-run:

```bash
python tools/experiment_orchestrator.py --task-file configs/imu_offset_net_20260607_tasks.json --dry-run
```

Dry-run status: 6 tasks, no output/log conflicts. Ready tasks are Stage A v1/v2/v3; Stage B and Stage C wait on v3 Stage A/B dependencies.

Current limitation: formal long training and full DIP/TotalCapture downstream tables are not run yet. DIP/TotalCapture offset accuracy is deliberately `not available`; only synthetic AMASS reports offset accuracy.

## IMUOffsetNet Formal Stage A/B/C Run

Date: 2026-06-07

Runner:

```bash
/home/lingfeng/bin/longrun -- python tools/experiment_orchestrator.py --task-file configs/imu_offset_net_20260607_tasks.json --run
```

State: `data/experiments/orchestrator_states/imu_offset_net_20260607.json`

All 6 tasks completed successfully. Logs:

- `logs/orchestrator/imu_offset_net_20260607/stageA_v1.log`
- `logs/orchestrator/imu_offset_net_20260607/stageA_v2.log`
- `logs/orchestrator/imu_offset_net_20260607/stageA_v3.log`
- `logs/orchestrator/imu_offset_net_20260607/stageB_v3_dip_ft.log`
- `logs/orchestrator/imu_offset_net_20260607/dip_test_v3_pred_offset.log`
- `logs/orchestrator/imu_offset_net_20260607/tc_val_v3_pred_offset.log`

Formal Stage A synthetic offset accuracy:

| Version | Best epoch | Best offset L2 cm ↓ | Last offset L1 cm ↓ | Last offset L2 cm ↓ | Temporal stability cm ↓ | Acc consistency pred ↓ |
|---|---:|---:|---:|---:|---:|---:|
| `offset_v1_mlp_frame` | 2 | `4.26099` | `2.30349` | `4.47110` | `0.83656` | `1.01356` |
| `offset_v2_temporal_rnn` | 2 | `4.24690` | `2.28499` | `4.41215` | `0.32516` | `1.01638` |
| `offset_v3_residual_prior` | 2 | `4.25810` | `2.18759` | `4.25904` | `0.00007` | `1.00229` |

Stage B DIP official-input fine-tune:

| Field | Value |
|---|---|
| Train cache | `data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json` |
| Val cache | `data/dataset_work/L4Cache/prephysics_pose_velocity_dip_val_globalpose_neural_only/baseline_cache_manifest.json` |
| IMU input | official `aM/wM/RMB` |
| Processed IMU | not used |
| Trans loss | not used |
| Offset GT | `not available` |
| Best selection | `2.260830` |
| Val pose-acc proxy | `2.259088` |
| Val offset norm mean | `0.174245 m` |

DIP test inference:

| Field | Value |
|---|---|
| Output manifest | `data/experiments/imu_offset_net_20260607/dip_test_v3_pred_offset/baseline_cache_manifest.json` |
| Num sequences | 19 |
| Offset GT | `not available` |
| Pred offset norm mean | `0.174245 m` |
| Pred temporal stability | `0.000201 cm` |
| Acc consistency | `not measured`; DIP trans is not reliable |

TotalCapture processed inference:

| Split | Num sequences | Offset GT | Pred offset norm mean | Acc consistency mean |
|---|---:|---|---:|---:|
| S4 val | 5 | not available | `0.174245 m` | `8.295448` |
| S5 test | 4 | not available | `0.174245 m` | `2.968832` |

Downstream utility, DIP official-input original GPNet:

| Cache | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no predicted offset | `44.642049` | `13.548559` | `8.469930` | `4.648240` | `5.408348` | `13.409634` | `8.291749` | `4.547626` | `5.265771` | `0.157846` | `0.259061` |
| IMUOffsetNet predicted offset | `44.642049` | `13.548559` | `8.469930` | `4.648240` | `5.408348` | `13.409634` | `8.291749` | `4.547626` | `5.265771` | `0.157846` | `0.259061` |

Downstream utility, TotalCapture processed PL evaluation:

| Split | Cache | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S4 | existing cache offset | `38.625657` | `10.135251` | `8.772134` | `4.495146` | `5.138202` | `10.290856` | `8.538511` | `4.346227` | `4.898401` | `0.285848` | `0.476914` |
| S4 | IMUOffsetNet predicted offset | `38.625657` | `10.135252` | `8.772134` | `4.495146` | `5.138202` | `10.290856` | `8.538509` | `4.346227` | `4.898400` | `0.285848` | `0.476914` |
| S5 | existing cache offset | `43.811277` | `9.647518` | `12.447101` | `4.358768` | `5.107238` | `9.131767` | `11.757354` | `3.837041` | `4.442018` | `0.370553` | `0.795591` |
| S5 | IMUOffsetNet predicted offset | `43.811278` | `9.647518` | `12.447102` | `4.358768` | `5.107239` | `9.131767` | `11.757355` | `3.837041` | `4.442018` | `0.370553` | `0.795591` |

Conclusion: formal run does not show meaningful downstream utility. Synthetic offset accuracy is measurable and around `4.25 cm` L2, but DIP/TotalCapture offset GT is not available and real-data predicted offsets do not improve DIP original GPNet or TotalCapture processed PL metrics.

## IMU position offset estimation for NewPL

Date: 2026-06-07

Question: Can an explicit IMU position offset estimator improve NewPL output `pRB[15]+gR1[3]` on real sparse-IMU data where DIP has no reliable `trans` and no real offset GT?

Protocol notes:

- This is a diagnostic module experiment, not an official protocol.
- DIP-IMU uses official baseline `aM/wM/RMB`; processed IMU and `trans` loss are forbidden.
- TotalCapture processed IMU is diagnostic/adaptation only.
- Real offset GT is `not available` for DIP/TotalCapture; AMASS synthetic offset accuracy is sanity-only.

Implemented:

```text
imu_position_offset.py
scripts/build_imu_position_offsets.py
pl_curve_offset_sensitivity_eval.py
scripts/summarize_imu_offset_newpl.py
pl_curve_cache.py --offset-cache / --max-sequences
pl_curve_train.py --offset-contrast-* --offset-init-dropout-prob --offset-init-noise-std
```

Offset coordinate contract: `r_JS` is the IMU origin position relative to mapped joint `J`, expressed in the joint-local frame. World reconstruction is `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS`.

Design summary: the diagnostic covers kinematic consistency, lever-arm acceleration, optimization-based calibration, self-supervised temporal consistency, and learned residual refinement. AMASS synthetic offset L1/L2 is sanity-only. DIP uses official `aM/wM/RMB` without `trans`; TotalCapture is diagnostic/adaptation only. NewPL external 84D input and 18D `pRB[15]+gR1[3]` output are preserved.

Routes:

| Route | Method | Training/eval data | Artifact |
|---|---|---|---|
| `offset_solver_v1_kinematic_opt` | lever-arm acceleration least-squares | TotalCapture diagnostic smoke | `data/experiments/imu_position_offset_newpl/tc_val_2seq/solver_v1_offsets.pt` |
| `offset_net_v2_selfsup` | StageA synthetic net + StageB DIP self-supervised fine-tune | AMASS sanity, DIP official-input fine-tune | `data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft/best_loss.pt` |
| `offset_hybrid_v3_opt_init_net_refine` | solver init plus net residual/blend | TotalCapture diagnostic smoke | `data/experiments/imu_position_offset_newpl/tc_val_2seq/hybrid_v3_offsets.pt` |

Commands run:

```bash
python scripts/build_imu_position_offsets.py --input data/dataset_work/TotalCapture_globalpose_official/val.pt --dataset totalcapture --method solver_v1 --max-sequences 2 --output data/experiments/imu_position_offset_newpl/tc_val_2seq/solver_v1_offsets.pt --summary-json data/experiments/imu_position_offset_newpl/tc_val_2seq/solver_v1_offsets.json --device cpu
python pl_curve_offset_sensitivity_eval.py --checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --cache zero=data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_cache_zero/pl_curve_cache_manifest.json --cache random=data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_cache_random/pl_curve_cache_manifest.json --cache solver_v1=data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_cache_solver_v1/pl_curve_cache_manifest.json --cache net_v2=data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_cache_net_v2/pl_curve_cache_manifest.json --cache hybrid_v3=data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_cache_hybrid_v3/pl_curve_cache_manifest.json --reference zero --output-json data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_offset_sensitivity_eval_current.json --max-sequences 2
python pl_curve_train.py --train-cache data/dataset_work/L4Cache/pl_curve_init36_processed_tc_train_Roffset_A/pl_curve_cache_manifest.json --val-cache data/dataset_work/L4Cache/pl_curve_init36_processed_tc_val_Roffset_A/pl_curve_cache_manifest.json --output-dir data/experiments/imu_position_offset_newpl/newpl_offset_sensitive_smoke_v2 --experiment-name newpl_offset_sensitive_smoke_v2 --epochs 3 --window 41 --batch-size 2 --lr 1e-4 --hidden-size 512 --tail-length 4 --residual-scale 0.1 --dropout 0.1 --init-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --init-size 36 --max-train-sequences 8 --max-val-sequences 2 --disable-ik-distill --baseline-pRB-weight 0.0 --baseline-gR1-weight 0.0 --control-point-prior-weight 0.05 --tail-update-prior-weight 0.001 --offset-contrast-weight 1.0 --offset-contrast-margin 0.005 --offset-contrast-mode roll_random --offset-init-dropout-prob 0.15 --offset-init-noise-std 0.03
```

TotalCapture 2-sequence NewPL smoke with current `newpl_v4_init36`:

| Method | Offset median m | PL pRB orig cm | PL pRB NewPL cm | Delta cm | gR1 orig deg | gR1 NewPL deg | Delta deg | Output diff vs zero cm | IK1 | Full 11 metrics |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| zero | `0` | `9.07321` | `8.69193` | `-0.38128` | `26.8333` | `26.6172` | `-0.21607` | `0` | not measured | not measured |
| random | `0.224814` | `9.07321` | `8.69193` | `-0.38128` | `26.8333` | `26.6172` | `-0.21607` | `1.66547e-07` | not measured | not measured |
| solver_v1 | `0.0654287` | `9.07321` | `8.69193` | `-0.38128` | `26.8333` | `26.6172` | `-0.21607` | `3.72451e-08` | not measured | not measured |
| net_v2 | `0.153126` | `9.07321` | `8.69193` | `-0.38128` | `26.8333` | `26.6172` | `-0.21607` | `8.12792e-08` | not measured | not measured |
| hybrid_v3 | `0.184449` | `9.07321` | `8.69193` | `-0.38128` | `26.8333` | `26.6172` | `-0.21607` | `6.61554e-08` | not measured | not measured |

Offset-sensitive smoke v2:

- Checkpoint: `data/experiments/imu_position_offset_newpl/newpl_offset_sensitive_smoke_v2/best_loss.pt`
- Best validation loss: `0.513764` at epoch 2.
- Sensitivity after smoke: random-vs-zero output diff mean `2.489e-06 cm`, solver-vs-zero `6.927e-07 cm`, net-vs-zero `3.980e-08 cm`, hybrid-vs-zero `6.272e-07 cm`.

Conclusion: offset cache injection works, and the three requested routes now exist, but current NewPL is effectively insensitive to `offset_r`. The five-method IK1/full-pipeline 11-metric comparison is intentionally not run yet because PL-level evidence says changing offset will not meaningfully change downstream behavior. Next step is to retrain/redesign PL-s1 so the offset condition affects `pRB/gR1`, then rerun `zero/random/solver_v1/net_v2/hybrid_v3` through NewPL, IK1, and full-pipeline metrics.

### Offset-conditioned NewPL diagnostic smoke

Date: 2026-06-07

Question: If `offset_r` is injected at every recurrent step instead of only initializing hidden state, do the five offset methods become distinguishable at NewPL output level?

Change implemented:

- `pl_curve.py`: added `PLCurveOffsetConditionedModule`.
- `pl_curve_train.py`: added `--model-variant offset_conditioned` and `--condition-scale`.
- `pl_curve_offset_sensitivity_eval.py` and `pl_curve_pl_accuracy_eval.py`: checkpoint-aware model construction through `build_pl_curve_model`.
- External PL frame input remains 84D; PL output remains 18D `pRB[15]+gR1[3]`; init36 remains `offset_r[18]+pRL[15]+gR0[3]`.

Training command:

```bash
python pl_curve_train.py --train-cache data/dataset_work/L4Cache/pl_curve_init36_processed_tc_train_Roffset_A/pl_curve_cache_manifest.json --val-cache data/dataset_work/L4Cache/pl_curve_init36_processed_tc_val_Roffset_A/pl_curve_cache_manifest.json --output-dir data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_smoke_v1 --experiment-name newpl_offset_conditioned_smoke_v1 --model-variant offset_conditioned --condition-scale 1.0 --epochs 3 --window 41 --batch-size 2 --lr 1e-4 --hidden-size 512 --tail-length 4 --residual-scale 0.05 --dropout 0.1 --init-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --init-size 36 --max-train-sequences 8 --max-val-sequences 2 --disable-ik-distill --baseline-pRB-weight 0.0 --baseline-gR1-weight 0.0 --control-point-prior-weight 0.05 --tail-update-prior-weight 0.001 --offset-contrast-weight 1.0 --offset-contrast-margin 0.005 --offset-contrast-mode roll_random --offset-init-dropout-prob 0.10 --offset-init-noise-std 0.02
```

Artifacts:

- Checkpoint: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_smoke_v1/best_loss.pt`
- Train result: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_smoke_v1/train_result.json`
- Sensitivity JSON: `data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_offset_sensitivity_eval_offset_conditioned_smoke_v1.json`
- PL eval JSONs: `data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_eval_offset_conditioned_{zero,random,solver_v1,net_v2,hybrid_v3}.json`
- PL table JSON: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_conditioned_pl_eval_table.json`

Training result:

| Metric | Value |
|---|---:|
| best validation loss | `0.515810` |
| best epoch | `3` |
| train sequences | `8` |
| val sequences | `2` |

Sensitivity versus zero offset:

| Method | Output diff cm | gR diff deg |
|---|---:|---:|
| random | `5.493e-04` | `1.908e-04` |
| solver_v1 | `2.802e-04` | `1.173e-04` |
| net_v2 | `1.628e-03` | `1.736e-04` |
| hybrid_v3 | `2.041e-03` | `1.852e-04` |

PL-level metrics on 2 TotalCapture validation sequences:

| Method | pRB orig cm | pRB NewPL cm | Delta cm | gR1 orig deg | gR1 NewPL deg | Delta deg |
|---|---:|---:|---:|---:|---:|---:|
| zero | `9.073207` | `9.123797` | `0.050590` | `26.833309` | `25.337427` | `-1.495884` |
| random | `9.073207` | `9.123161` | `0.049954` | `26.833309` | `25.337582` | `-1.495731` |
| solver_v1 | `9.073207` | `9.123602` | `0.050395` | `26.833309` | `25.337492` | `-1.495819` |
| net_v2 | `9.073207` | `9.122195` | `0.048988` | `26.833309` | `25.337856` | `-1.495458` |
| hybrid_v3 | `9.073207` | `9.121889` | `0.048682` | `26.833309` | `25.337954` | `-1.495360` |

Conclusion: per-frame conditioning makes offset differences measurable, but the effect is still too small and not yet useful. It improves gR1 relative to original PL but worsens pRB slightly, and net/hybrid only beat zero by about `0.001-0.002 cm` on pRB. IK1 and full-pipeline 11 metrics are still `not measured`; this smoke is evidence for the next PL training direction, not a selected replacement.

### Offset-conditioned pairwise v2

Date: 2026-06-07

Question: Does a direct good-vs-bad offset objective make NewPL prefer the correct offset in downstream PL GT loss?

Change: modified `pl_curve_train.py` offset contrast from a detached-good diagnostic to a pairwise training term:

```text
good_metric + relu(good_metric + margin - bad_metric)
```

This preserves the PL 84D input, 18D output, and init36 contract.

Training artifact:

```text
data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_pairwise_v2/best_loss.pt
data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_pairwise_v2/train_result.json
```

Training result:

| Metric | Value |
|---|---:|
| best validation loss | `0.505689` |
| best epoch | `5` |
| final train offset_bad_minus_good_metric | `3.59e-08` |

Swap eval artifact:

```text
data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_swap_eval_pairwise_v2_hybrid_cache.json
```

Swap eval result on 2 TotalCapture validation sequences, hybrid offset as good:

| Bad init | pRB delta vs good cm | gR1 delta vs good deg | PL GT loss delta vs good |
|---|---:|---:|---:|
| zero | `+0.020648` | `-0.011161` | `-1.132e-04` |
| roll_sensors | `+0.012551` | `-0.006490` | `-6.156e-05` |
| other_sequence | `+0.000510` | `-0.000966` | `-1.961e-05` |
| negate | `+0.027347` | `-0.014600` | `-1.457e-04` |

Conclusion: pairwise training creates pRB-level separability, but the combined PL GT loss still prefers bad offsets because gR1 improves in the wrong direction. The current offset-to-NewPL path is therefore not ready for IK1/full-pipeline 11 metrics. Next useful work should isolate the offset objective to pRB or add a physically grounded forward-IMU consistency loss, rather than treating full PL loss as the offset selection signal.

### Offset-conditioned pRB contrast v1

Date: 2026-06-07

Question: If the offset contrast metric is restricted to `pRB[15]` only, can NewPL learn a stronger preference for the correct offset without the gR1 term cancelling the signal?

Change: added `--offset-contrast-target {full_pl,pRB}` to `pl_curve_train.py`. The default remains `full_pl`; `pRB` uses only:

```text
SmoothL1(pred[..., :15], target[..., :15])
```

This preserves the external PL 84D input, 18D output, and init36 `offset_r[18]+pRL[15]+gR0[3]` contract.

Training command:

```bash
python pl_curve_train.py --train-cache data/dataset_work/L4Cache/pl_curve_init36_processed_tc_train_Roffset_A/pl_curve_cache_manifest.json --val-cache data/dataset_work/L4Cache/pl_curve_init36_processed_tc_val_Roffset_A/pl_curve_cache_manifest.json --output-dir data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_prb_contrast_v1 --experiment-name newpl_offset_conditioned_prb_contrast_v1 --model-variant offset_conditioned --condition-scale 1.0 --epochs 5 --window 41 --batch-size 2 --lr 5e-5 --hidden-size 512 --tail-length 4 --residual-scale 0.05 --dropout 0.1 --init-checkpoint data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_pairwise_v2/best_loss.pt --init-size 36 --max-train-sequences 8 --max-val-sequences 2 --disable-ik-distill --baseline-pRB-weight 0.0 --baseline-gR1-weight 0.0 --control-point-prior-weight 0.05 --tail-update-prior-weight 0.001 --offset-contrast-weight 2.0 --offset-contrast-margin 0.005 --offset-contrast-mode roll_random --offset-contrast-target pRB --offset-init-dropout-prob 0.05 --offset-init-noise-std 0.01
```

Artifacts:

- Checkpoint: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_prb_contrast_v1/best_loss.pt`
- Train result: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_prb_contrast_v1/train_result.json`
- Swap eval: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_swap_eval_prb_contrast_v1_hybrid_cache.json`

Training result:

| Metric | Value |
|---|---:|
| best validation loss | `0.502297` |
| best epoch | `1` |
| final validation loss | `0.526568` |
| final train offset_bad_minus_good_metric | `-5.22e-08` |

The best checkpoint is epoch 1; later epochs overfit the tiny TotalCapture diagnostic set.

Swap eval on 2 TotalCapture validation sequences, hybrid offset as good:

| Bad init | pRB delta vs good cm | gR1 delta vs good deg | PL GT loss delta vs good |
|---|---:|---:|---:|
| zero | `+0.018504` | `-0.010733` | `-1.143e-04` |
| roll_sensors | `+0.011441` | `-0.006266` | `-6.256e-05` |
| other_sequence | `+0.000239` | `-0.001027` | `-2.113e-05` |
| negate | `+0.024665` | `-0.014092` | `-1.479e-04` |

Conclusion: pRB-only contrast slightly preserves the earlier pRB preference for the good offset, but it does not strengthen it. The gain is still only about `0.01-0.025 cm`, train-time separability is effectively zero, and full PL loss still prefers bad offsets because gR1 moves in the opposite direction. IK1 output metrics and full-pipeline 11 metrics remain `not measured`; the current evidence does not justify treating any offset route as a selected NewPL improvement.

### Offset forward-consistency eval v1

Date: 2026-06-07

Question: Do the estimated offsets improve a direct lever-arm acceleration reconstruction objective on real TotalCapture data, even though real offset GT is not available?

Change implemented:

- Added `imu_position_offset_consistency_eval.py`.
- The script evaluates `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS` through a forward acceleration residual against measured IMU acceleration.
- It reports residual, zero-offset residual, improvement, and offset magnitude.
- This is diagnostic consistency only; it is not real offset GT accuracy.

Command:

```bash
python imu_position_offset_consistency_eval.py --input data/dataset_work/TotalCapture_globalpose_official/val.pt --dataset totalcapture --offset-cache zero=data/experiments/imu_position_offset_newpl/tc_val_2seq/zero_offsets.pt --offset-cache random=data/experiments/imu_position_offset_newpl/tc_val_2seq/random_offsets.pt --offset-cache solver_v1=data/experiments/imu_position_offset_newpl/tc_val_2seq/solver_v1_offsets.pt --offset-cache net_v2=data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_offsets.pt --offset-cache hybrid_v3=data/experiments/imu_position_offset_newpl/tc_val_2seq/hybrid_v3_offsets.pt --output-json data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_consistency_eval_v1.json --max-sequences 2 --device cpu
```

Artifact:

```text
data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_consistency_eval_v1.json
```

Aggregate on 2 TotalCapture validation sequences:

| Method | Mean residual m/s^2 | Mean improvement vs zero m/s^2 | Median offset norm m |
|---|---:|---:|---:|
| zero | `13.416491` | `0.000000` | `0.000000` |
| random | `15.256863` | `-1.840372` | `0.224814` |
| solver_v1 | `13.337500` | `0.078991` | `0.065429` |
| net_v2 | `13.096867` | `0.319623` | `0.153126` |
| hybrid_v3 | `13.304113` | `0.112376` | `0.184449` |

Interpretation: the real-data offset estimates are not pure random noise under this physical diagnostic. Random offsets worsen acceleration consistency, while `solver_v1`, `net_v2`, and `hybrid_v3` improve it, with `net_v2` best on this tiny smoke. However, the previous NewPL diagnostics show that better acceleration consistency does not yet translate into meaningful `pRB/gR1` improvement. This matches the decision rule: physical consistency is useful evidence, but it is insufficient for selecting an offset route until NewPL module metrics and downstream metrics improve.

### DIP self-supervised OffsetNet v4 smoke

Date: 2026-06-07

Question: Does a small DIP-IMU official-input self-supervised fine-tune improve the pose-derived acceleration proxy when no `trans` and no real offset GT are used?

Change implemented:

- `imu_offset_finetune_dip.py` now writes `initial_val.json` before training.
- `train_result.json` records `initial_val` and `best_epoch`.
- The script still uses DIP official `aM/wM/RMB` only, no processed IMU, no `trans_loss`, and no `offset_gt_loss`.

Command:

```bash
python imu_offset_finetune_dip.py --init-checkpoint data/experiments/imu_offset_net_20260607/stageA_v3/best_loss.pt --train-cache data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_dip_val_globalpose_neural_only/baseline_cache_manifest.json --output-dir data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke --epochs 2 --lr 3e-6 --window 90 --windows-per-sequence 1 --val-windows-per-sequence 1 --max-train-sequences 4 --max-val-sequences 2 --max-frames 300 --acc-weight 1.0 --smooth-weight 0.001 --magnitude-weight 0.01 --std-weight 0.01
```

Artifacts:

- Initial validation: `data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/initial_val.json`
- Train result: `data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/train_result.json`
- Checkpoint: `data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/best_loss.pt`

Result:

| Metric | Initial | Last/best | Delta |
|---|---:|---:|---:|
| DIP val pose_acc_proxy | `0.018450846` | `0.018450512` | `-3.34e-07` |
| offset magnitude m | `0.174248368` | `0.174248084` | `-2.83e-07` |
| best epoch | not applicable | `2` | not applicable |
| offset L1/L2 cm | not available | not available | not available |

Interpretation: the DIP self-supervised fine-tune path is reproducible and correctly avoids forbidden supervision, but this tiny smoke does not show a meaningful improvement. The proxy change is numerical-scale, so this checkpoint should not replace the prior `stageB_v3_dip_ft` model or be treated as evidence that DIP self-supervision improves NewPL.

### Offset-NewPL decision matrix v1

Date: 2026-06-07

Question: Can the solver/net/hybrid offset routes be compared in one auditable table across physical consistency, NewPL module metrics, DIP self-supervision status, and downstream readiness?

Change:

- Extended `scripts/summarize_imu_offset_newpl.py` with optional `--decision-json` and `--decision-md`.
- The original summary output remains compatible.
- The decision matrix explicitly records `not measured` for IK1/full-pipeline metrics and `not available` for real offset GT.

Command:

```bash
python scripts/summarize_imu_offset_newpl.py --root data/experiments/imu_position_offset_newpl/tc_val_2seq --sensitivity-json data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_offset_sensitivity_eval_current.json --output-json data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_smoke_summary_current.json --output-md data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_smoke_summary_current.md --decision-json data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.json --decision-md data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.md
```

Artifacts:

- Decision JSON: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.json`
- Decision Markdown: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.md`

Decision matrix:

| Method | Forward residual m/s^2 | Forward improvement | Cond. PL pRB vs zero cm | Cond. PL gR1 vs zero deg | IK1 | Full 11 | Decision |
|---|---:|---:|---:|---:|---|---|---|
| zero | `13.416491` | `0.000000` | `0.000000` | `0.000000` | not measured | not measured | not selected |
| random | `15.256863` | `-1.840372` | `-0.000636` | `0.000154` | not measured | not measured | negative control |
| solver_v1 | `13.337500` | `0.078991` | `-0.000196` | `0.000065` | not measured | not measured | physical signal, downstream not selected |
| net_v2 | `13.096867` | `0.319623` | `-0.001602` | `0.000429` | not measured | not measured | physical signal, downstream not selected |
| hybrid_v3 | `13.304113` | `0.112376` | `-0.001908` | `0.000526` | not measured | not measured | physical signal, downstream not selected |

Machine-readable selection:

```text
best_offset_method_by_forward_consistency = net_v2
best_offset_method_for_newpl = not selected
run_ik1_or_full_pipeline = false
reason = Forward consistency improves for solver/net/hybrid, but NewPL pRB/gR1 gains are tiny or conflicting and full PL loss can prefer bad offsets.
```

Interpretation: this consolidates the current evidence. `net_v2` is the best physical-consistency route on the TotalCapture smoke, but its NewPL pRB gain over zero is only `0.001602 cm`, and gR1 is slightly worse. Therefore no offset route is selected for NewPL yet, and IK1/full 11 metrics remain intentionally `not measured`.

### StageB v4 OffsetNet TotalCapture transfer check

Date: 2026-06-07

Question: Does the auditable DIP self-supervised v4 checkpoint produce meaningfully better TotalCapture offsets than the existing `net_v2` artifact?

Commands:

```bash
python scripts/build_imu_position_offsets.py --input data/dataset_work/TotalCapture_globalpose_official/val.pt --dataset totalcapture --method net_v2 --net-ckpt data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/best_loss.pt --max-sequences 2 --output data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_stageB_v4_offsets.pt --summary-json data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_stageB_v4_offsets.json --device cpu
python imu_position_offset_consistency_eval.py --input data/dataset_work/TotalCapture_globalpose_official/val.pt --dataset totalcapture --offset-cache zero=data/experiments/imu_position_offset_newpl/tc_val_2seq/zero_offsets.pt --offset-cache net_v2=data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_offsets.pt --offset-cache net_v2_stageB_v4=data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_stageB_v4_offsets.pt --output-json data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_consistency_eval_stageB_v4_compare.json --max-sequences 2 --device cpu
python scripts/summarize_imu_offset_newpl.py --root data/experiments/imu_position_offset_newpl/tc_val_2seq --sensitivity-json data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_offset_sensitivity_eval_current.json --output-json data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_smoke_summary_current.json --output-md data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_smoke_summary_current.md --decision-json data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.json --decision-md data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.md --stageb-v4-consistency-json data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_consistency_eval_stageB_v4_compare.json
```

Artifacts:

- Offset cache: `data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_stageB_v4_offsets.pt`
- Offset summary: `data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_stageB_v4_offsets.json`
- Consistency compare: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_consistency_eval_stageB_v4_compare.json`
- Updated decision matrix: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.json`

Result:

| Method | Mean residual m/s^2 | Improvement vs zero m/s^2 | Median offset norm m |
|---|---:|---:|---:|
| net_v2 | `13.096867` | `0.319623` | `0.153126` |
| net_v2_stageB_v4 | `13.096842` | `0.319649` | `0.153162` |

Delta in forward improvement: `+2.56e-05 m/s^2`.

Interpretation: DIP stageB v4 changes the TotalCapture offset result only at numerical scale. It does not justify generating a separate NewPL cache/evaluation, and it does not change the decision matrix selection: `net_v2` remains the best physical-consistency route, but no offset method is selected for NewPL/downstream.

## NewPL-root module training and module-level evaluation

Date: 2026-06-07

Purpose: design and implement `newpl_root_v1`, a PL module that keeps the current PL input contract and extends output from `pRB[15]+gR1[3]` to `pRB[15]+gR1[3]+root_vel[3]`. This record is module-level only; no full-pipeline 11 metrics were run.

Contract:

| Item | Shape | Meaning |
|---|---:|---|
| PL input | `84D` | `aRB[18] + wRB[18] + RRB[45] + gR0[3]` |
| Current NewPL output | `18D` | `pRB[15] + gR1[3]` |
| `newpl_root_v1` output | `21D` | `pRB[15] + gR1[3] + root_vel[3]` |
| init36 | `36D` | `offset_r[18] + pRL[15] + gR0[3]` |

Root velocity frame: `root_vel` is root/body-frame root linear velocity in m/s. It is computed from world/mocap translation by finite difference, then right-multiplied by `pose[:,0]`, matching the existing row-vector root-frame projection used for `pRB`. This is preferred over world-frame velocity because PL inputs are root/body-relative (`aRB`, `wRB`, `RRB`) plus gravity, so world-frame heading velocity is not the natural module target.

Implemented files:

- `newpl_root.py`
- `newpl_root_train.py`
- `newpl_root_eval.py`

Loss by stage:

| Stage | Data | root_vel supervision | Losses |
|---|---|---|---|
| A | AMASS long pretrain | GT from `tran_gt` or decoded `q75_gt` translation | `pRB`, `gR1`, `root_vel`, temporal terms, `root_vel_smooth`, GT control terms, priors |
| B1 | TotalCapture fine-tune | GT from `tran_gt` when reliable | same as AMASS; otherwise use `--root-vel-mode smooth_only` |
| B2 | DIP-IMU fine-tune | GT disabled | `pRB/gR1` plus temporal/control terms; use `--root-vel-mode none` or `smooth_only`; optional `--freeze-root-head`; no DIP trans loss |

DIP guard: `--dataset dip --root-vel-mode gt` and `--dataset dip --root-vel-gt` are rejected. DIP cache `tran_gt` fields are not used for root velocity training/evaluation.

Planned training commands:

```bash
ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
"$ENV/bin/python" newpl_root_train.py --train-cache data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged_with_offset_r/baseline_cache_manifest.json --val-cache data/dataset_work/L4Cache/globalpose_amass_baseline_cache_diverse7_merged_with_offset_r/baseline_cache_manifest.json --output-dir data/experiments/newpl_root_v1/amass_pretrain --experiment-name newpl_root_v1_amass_pretrain --dataset amass --imu-input-mode official --root-vel-mode gt --epochs 60 --window 61 --lr 1e-4 --init-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
"$ENV/bin/python" newpl_root_train.py --train-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/newpl_root_v1/tc_finetune --experiment-name newpl_root_v1_tc_finetune --dataset totalcapture --imu-input-mode processed --root-vel-mode gt --epochs 20 --window 61 --lr 1e-5 --init-checkpoint data/experiments/newpl_root_v1/amass_pretrain/best_loss.pt
"$ENV/bin/python" newpl_root_train.py --train-cache data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json --val-cache data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only_with_offset_r/baseline_cache_manifest.json --output-dir data/experiments/newpl_root_v1/dip_finetune --experiment-name newpl_root_v1_dip_finetune --dataset dip --imu-input-mode official --root-vel-mode none --freeze-root-head --epochs 20 --window 61 --lr 1e-5 --init-checkpoint data/experiments/newpl_root_v1/amass_pretrain/best_loss.pt
```

Planned evaluation JSONs:

- `data/experiments/newpl_root_v1/eval/amass_module_metrics.json`
- `data/experiments/newpl_root_v1/eval/tc_test_module_metrics.json`
- `data/experiments/newpl_root_v1/eval/dip_test_module_metrics.json`

Smoke validation run:

| Artifact | Path |
|---|---|
| checkpoint | `data/experiments/newpl_root_v1/smoke/tc_train_smoke/best_loss.pt` |
| train log | `data/experiments/newpl_root_v1/smoke/tc_train_smoke/train_log.jsonl` |
| train result | `data/experiments/newpl_root_v1/smoke/tc_train_smoke/train_result.json` |
| TC JSON | `data/experiments/newpl_root_v1/smoke/tc_multi_module_smoke.json` |
| DIP JSON | `data/experiments/newpl_root_v1/smoke/dip_root_module_smoke.json` |

Smoke TotalCapture single-sequence table:

| Version | Fine-tune data | Input mode | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | root_vel L1 ↓ | root_vel L2 ↓ | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| official_PL | none | processed | `3.607639` | `7.687036` | `11.658890` | not applicable | not applicable | smoke only |
| newpl_v4_init36 | historical TC | processed | `3.412260` | `7.253721` | `11.529694` | not applicable | not applicable | smoke only |
| newpl_root_v1_smoke | TC 1-seq/1-epoch smoke | processed | `3.411824` | `7.252498` | `11.529869` | `0.260059` | `0.553485` | smoke only |

Smoke DIP single-sequence table:

| Version | Fine-tune data | Input mode | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | root_vel L1 ↓ | root_vel L2 ↓ | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| newpl_root_v1_smoke | TC smoke only | official | `3.462420` | `7.301293` | `12.611866` | root_vel GT not available | root_vel GT not available | DIP trans not used |

Full requested AMASS/TC/DIP module tables are implemented but not measured beyond smoke. Long training was not launched in this turn.

### Fair baseline comparison requirement

New requirement recorded on 2026-06-07: NewPL-root is only useful if it improves the module outputs against fair baselines, not merely because the extra head emits a velocity.

Required PL output comparison for every AMASS/TotalCapture/DIP module eval:

| Dataset | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | Notes |
|---|---|---:|---:|---:|---|

Required versions:

- `official_PL`
- `newpl_v4_init36`
- `newpl_root_v1`

Required root velocity comparison when GT is available:

| Dataset | Version | root_vel source | root_vel L1 ↓ | root_vel L2 ↓ | root_vel angle ↓ | Notes |
|---|---|---|---:|---:|---:|---|

Root velocity baseline priority:

1. Use official pipeline root velocity if exposed.
2. Otherwise finite-difference official/newpl pipeline final translation.
3. Project baseline world velocity to the GT root/body frame for comparison with the `newpl_root_v1` root-frame target.
4. If GT root velocity is not reliable, write `root_vel GT not available` / `baseline root_vel not comparable` and do not infer metrics.

Implementation update: `newpl_root_eval.py` now writes `pl_output_comparison_table` and `root_velocity_comparison_table`. For TotalCapture, official and `newpl_v4_init36` root velocity baselines are computed from final pipeline translation. For DIP, root velocity GT remains disabled, and baseline velocity metrics are not computed.

### Long training run started

Date: 2026-06-07

Command:

```bash
CUDA_VISIBLE_DEVICES=0 /home/lingfeng/bin/longrun -- bash scripts/run_newpl_root_v1_longtrain_20260607.sh
```

Status: running after the initial 60-second failure check.

Run script: `scripts/run_newpl_root_v1_longtrain_20260607.sh`

Output root: `data/experiments/newpl_root_v1/longrun_20260607/`

Expected checkpoints:

- `data/experiments/newpl_root_v1/longrun_20260607/amass_pretrain/best_loss.pt`
- `data/experiments/newpl_root_v1/longrun_20260607/amass_pretrain/last.pt`
- `data/experiments/newpl_root_v1/longrun_20260607/tc_finetune/best_loss.pt`
- `data/experiments/newpl_root_v1/longrun_20260607/tc_finetune/last.pt`
- `data/experiments/newpl_root_v1/longrun_20260607/dip_finetune/best_loss.pt`
- `data/experiments/newpl_root_v1/longrun_20260607/dip_finetune/last.pt`

Expected fair-eval JSONs:

- `data/experiments/newpl_root_v1/longrun_20260607/eval/amass_module_metrics.json`
- `data/experiments/newpl_root_v1/longrun_20260607/eval/tc_test_module_metrics.json`
- `data/experiments/newpl_root_v1/longrun_20260607/eval/dip_test_module_metrics.json`

Notes:

- AMASS train cache: `prephysics_pose_velocity_amass_k2_paired_offset_overlay`, official IMU fields, GT root velocity enabled.
- TotalCapture fine-tune: processed input cache, GT root velocity enabled.
- DIP fine-tune: official input cache, `--root-vel-mode none`, `--freeze-root-head`, `--allow-zero-offset-init`; no DIP trans/root_vel GT is used.
- AMASS eval is currently configured as `AMASS-val20` because fair baseline velocity requires final pipeline translation and is expensive on the full AMASS cache.
## NewPL-root module training and module-level evaluation final results

Date: 2026-06-07

Final artifact root: `data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/`

Training summary:

| Stage | Status | Batch | Selection metric | Best epoch | Best selection | Last epoch | Notes |
|---|---|---:|---|---:|---:|---:|---|
| AMASS pretrain | early_stopped | 2048 | control_root_physical | 20 | `0.050948905052791815` | 28 | selected by fitted GT control pRB+gR1+root_vel |
| TotalCapture fine-tune | ok | 64 | control_root_physical | 19 | `0.11094439717999194` | 20 | selected by fitted GT control pRB+gR1+root_vel; epoch 20 lower by less than min_delta |
| DIP fine-tune | early_stopped | 64 | control_physical | 1 | `0.018623053280048464` | 7 | DIP root_vel GT disabled; selected by fitted GT control pRB+gR1 |

Fair PL output comparison:

| Dataset | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | Conclusion |
|---|---|---:|---:|---:|---|
| AMASS-val20 | official_PL | `1.635521` | `3.371232` | `4.867950` | best overall |
| AMASS-val20 | newpl_v4_init36 | `1.720172` | `3.572181` | `4.909946` | worse than NewPL-root |
| AMASS-val20 | newpl_root_v1_amass_pretrain | `1.671904` | `3.460373` | `4.869618` | improves v4, not official |
| TotalCapture-test | official_PL | `3.370257` | `6.995536` | `13.450453` | worse than v4/NewPL-root |
| TotalCapture-test | newpl_v4_init36 | `3.210470` | `6.654393` | `13.329531` | best |
| TotalCapture-test | newpl_root_v1_tc_finetune | `3.268334` | `6.779195` | `13.376897` | better than official, worse than v4 |
| DIP-IMU-test | official_PL | `3.115170` | `6.419473` | `12.947709` | best pRB L1/L2 |
| DIP-IMU-test | newpl_v4_init36 | `3.116842` | `6.441447` | `12.765167` | best gR1 |
| DIP-IMU-test | newpl_root_v1_dip_finetune | `3.115812` | `6.429121` | `12.854242` | not best |

Root velocity comparison:

| Dataset | Version | root_vel source | root_vel L1 ↓ | root_vel L2 ↓ | root_vel angle ↓ | Conclusion |
|---|---|---|---:|---:|---:|---|
| AMASS-val20 | official_PL | final pipeline translation diff | `0.153637` | `0.337882` | `60.888321` | baseline better than root head |
| AMASS-val20 | newpl_v4_init36 | final pipeline translation diff | `0.152898` | `0.337115` | `60.764704` | best baseline |
| AMASS-val20 | newpl_root_v1_amass_pretrain | direct root_vel head | `0.193051` | `0.427344` | `76.915063` | worse |
| TotalCapture-test | official_PL | final pipeline translation diff | `0.115552` | `0.234634` | `36.327946` | baseline better than root head |
| TotalCapture-test | newpl_v4_init36 | final pipeline translation diff | `0.114272` | `0.232193` | `36.147620` | best baseline |
| TotalCapture-test | newpl_root_v1_tc_finetune | direct root_vel head | `0.268907` | `0.587701` | `75.600269` | much worse |
| DIP-IMU-test | GT | not available | not available | not available | not available | no root_vel comparison allowed |

Decision:

- `newpl_root_v1` is not selected.
- It does not beat `newpl_v4_init36` on TotalCapture pRB/gR1.
- It does not beat the pipeline-derived baseline velocity on AMASS or TotalCapture.
- DIP root velocity remains not comparable; no DIP trans/root_vel GT was used.
- Do not connect this checkpoint to IK1/full pipeline unless a new design improves both control-point PL outputs and root velocity against the fair baselines.

## NewPL-root fine-tune before/after checkpoint audit

Date: 2026-06-07

Purpose: answer whether fine-tuning actually helped by evaluating AMASS best/last, TotalCapture before/after best/last, and DIP before/after best/last at module-output level. This audit still does not run full-pipeline 11 metrics and does not use DIP translation.

Additional JSONs:

- `data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/amass_best_last_module_metrics.json`
- `data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/tc_before_after_pl_only_metrics.json`
- `data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/tc_before_after_root_head_only_metrics.json`
- `data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/dip_before_after_pl_metrics.json`

PL before/after summary:

| Dataset | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | Interpretation |
|---|---|---:|---:|---:|---|
| AMASS-val20 | newpl_root_amass_best | `1.671904` | `3.460373` | `4.869618` | selected epoch 20 |
| AMASS-val20 | newpl_root_amass_last | `1.654693` | `3.415629` | `4.865114` | last is better on decoded PL, though control selection picked best |
| TotalCapture-test | before_tc_amass_best | `3.290194` | `6.825256` | `13.408251` | before TC fine-tune |
| TotalCapture-test | tc_finetune_best | `3.268334` | `6.779195` | `13.376897` | small improvement |
| TotalCapture-test | tc_finetune_last | `3.267189` | `6.776795` | `13.375096` | slightly better than best on decoded PL |
| DIP-IMU-test | before_dip_amass_best | `3.115633` | `6.428928` | `12.852417` | before DIP fine-tune |
| DIP-IMU-test | dip_finetune_best | `3.115812` | `6.429121` | `12.854242` | no improvement |
| DIP-IMU-test | dip_finetune_last | `3.117006` | `6.430589` | `12.864961` | worse than before fine-tune |

Root-head before/after summary:

| Dataset | Version | root_vel L1 ↓ | root_vel L2 ↓ | root_vel angle ↓ | Interpretation |
|---|---|---:|---:|---:|---|
| AMASS-val20 | newpl_root_amass_best | `0.193051` | `0.427344` | `76.915063` | worse than pipeline baseline |
| AMASS-val20 | newpl_root_amass_last | `0.193132` | `0.427571` | `74.706794` | L1/L2 slightly worse, angle better |
| TotalCapture-test | before_tc_amass_best | `0.269022` | `0.587931` | `76.761979` | before TC fine-tune |
| TotalCapture-test | tc_finetune_best | `0.268907` | `0.587701` | `75.600269` | tiny improvement |
| TotalCapture-test | tc_finetune_last | `0.268900` | `0.587688` | `75.533853` | tiny improvement, still much worse than pipeline baseline |
| DIP-IMU-test | all NewPL-root checkpoints | not available | not available | not available | DIP root_vel GT not available |

Conclusion: TotalCapture fine-tune has a small positive effect on `newpl_root_v1` decoded PL outputs and a tiny effect on direct root velocity, but it still does not beat `newpl_v4_init36` for PL or pipeline-derived baseline velocity for root motion. DIP fine-tune does not help. AMASS last is better than AMASS selected best on decoded PL, showing the control-point selection metric and decoded PL metric do not perfectly align, so both best and last should remain evaluated for this module.

## IMU position offset experiments completion summary

Date: 2026-06-07

Status: completed at diagnostic/module level; not selected for integration. No further long training or full-pipeline evaluation is currently justified from this branch.

Scope completed:

- Implemented three neural offset estimators: `offset_v1_mlp_frame`, `offset_v2_temporal_rnn`, `offset_v3_residual_prior`.
- Trained Stage A synthetic supervised AMASS offset models with real synthetic `r_JS` offset GT.
- Fine-tuned/evaluated `offset_v3_residual_prior` on DIP official-input consistency without DIP `trans`, processed IMU, or real offset GT.
- Generated predicted-offset caches for DIP test and TotalCapture S4/S5 diagnostics.
- Evaluated downstream PL scores with existing/cache offsets versus predicted offsets.
- Tested solver/net/hybrid offset routes and offset-conditioned NewPL diagnostics.

Coordinate/data contract:

- Offset is `r_JS`: IMU origin relative to mapped joint `J`, expressed in joint-local coordinates.
- World reconstruction is `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS`.
- DIP/TotalCapture real offset GT is `not available`; synthetic AMASS offset accuracy must not be reported as real-data accuracy.
- DIP `trans` is not used as supervision.

Main artifacts:

- Stage A v1/v2/v3: `data/experiments/imu_offset_net_20260607/stageA_v*/train_result.json`
- DIP fine-tune: `data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft/train_result.json`
- DIP predicted-offset cache: `data/experiments/imu_offset_net_20260607/dip_test_v3_pred_offset/baseline_cache_manifest.json`
- TotalCapture predicted-offset caches: `data/experiments/imu_offset_net_20260607/tc_val_v3_pred_offset/baseline_cache_manifest.json`, `data/experiments/imu_offset_net_20260607/tc_test_v3_pred_offset/baseline_cache_manifest.json`
- Downstream PL JSONs: `data/experiments/imu_offset_net_20260607/downstream/`
- Offset/NewPL decision matrix: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.json`

Synthetic Stage A offset accuracy:

Note: the `2.x cm` number from this branch is offset L1. The `4.x cm` number is vector offset L2. Both are AMASS synthetic-only diagnostics, not real DIP/TotalCapture offset accuracy.

| Version | Last offset L1 cm ↓ | Best offset L2 cm ↓ | Last offset L2 cm ↓ | Selected? |
|---|---:|---:|---:|---|
| `offset_v1_mlp_frame` | `2.30349` | `4.26099` | `4.47110` | no |
| `offset_v2_temporal_rnn` | `2.28499` | `4.24690` | `4.41215` | no |
| `offset_v3_residual_prior` | `2.18759` | `4.25810` | `4.25904` | no; used as stable diagnostic source |

Downstream utility:

| Dataset/protocol | Baseline/cache score ↓ | Predicted-offset score ↓ | Result |
|---|---:|---:|---|
| DIP test, official input | `44.642049` | `44.642049` | unchanged |
| TotalCapture S4 PL eval | `38.625657` | `38.625657` | unchanged |
| TotalCapture S5 PL eval | `43.811277` | `43.811278` | no improvement |

Physical consistency diagnostic on TotalCapture 2-seq:

| Method | Forward residual m/s^2 ↓ | Improvement vs zero | Decision |
|---|---:|---:|---|
| zero | `13.416491` | `0.000000` | baseline |
| random | `15.256863` | `-1.840372` | negative control |
| solver_v1 | `13.337500` | `0.078991` | physical signal only |
| net_v2 | `13.096867` | `0.319623` | best physical consistency |
| hybrid_v3 | `13.304113` | `0.112376` | physical signal only |

Offset-NewPL decision:

- `net_v2` is best by forward acceleration consistency, but NewPL pRB/gR1 gains are tiny and conflicting.
- Current `newpl_v4_init36` is nearly insensitive to offset changes; earlier offset swaps changed outputs only at numerical-noise scale.
- Offset-conditioned NewPL made offset effects measurable, but pRB gains over zero were around `0.001-0.002 cm`, far below a useful threshold.
- IK1 and full-pipeline 11 metrics for `zero/random/solver_v1/net_v2/hybrid_v3` remain `not measured` by design, because PL-level evidence does not justify the expensive downstream run.

Final decision: this experiment is done for the current design and is not selected. The next useful work is not more training of the same OffsetNet; it would require redesigning how offset enters PL, adding a stronger physical forward-IMU loss, or creating a downstream module that explicitly consumes `r_JS`.

## EXP-20260607-newpl_v5_official_protocol — NewPL AMASS-to-DIP official-like module evaluation

Date: 2026-06-07

Status: completed at module level; not selected as mainline. This is not a full-pipeline 11-metric run.

Question: If NewPL follows an official-like route, AMASS pretrain followed by DIP-IMU train fine-tune, does it beat `official_PL` and the current `newpl_v4_init36` baseline on PL module outputs?

Protocol:

- Stage A: AMASS pretrain on official-input PL curve cache.
- Stage B: DIP-IMU train fine-tune from the AMASS checkpoint.
- Evaluation: DIP-IMU test and TotalCapture official-input test after AMASS pretrain, then again after DIP fine-tune.
- TotalCapture train split was not used in this protocol.
- DIP `trans` and root velocity supervision were not used.
- Evaluation is module-level `pRB/gR1` only; full-pipeline S4/S5 11 metrics are `not measured`.

Input/output contract:

- PL input: `aRB[18] + wRB[18] + RRB[45] + gR0[3] = 84D`.
- PL output: `pRB[15] + gR1[3] = 18D`, preserving the official downstream IK1 contract.
- Init36 feature: `offset_r[18] + pRL[15] + gR0[3] = 36D`.

Training artifacts:

```text
script: scripts/run_newpl_v5_official_protocol_20260607.sh
root: data/experiments/newpl_v5_official_protocol_20260607_tuned
caches: data/experiments/newpl_v5_official_protocol_20260607/caches
AMASS checkpoint best/last: data/experiments/newpl_v5_official_protocol_20260607_tuned/amass_pretrain/best_loss.pt, last.pt
DIP checkpoint best/last: data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt, last.pt
summary: data/experiments/newpl_v5_official_protocol_20260607_tuned/summary.json, summary.md
```

Training configuration summary:

- AMASS pretrain: `epochs=80`, `batch_size=256`, `lr=1e-4`, `window=61`, `selection_metric=control_physical`, best epoch `80`, best selection value `0.002173126090565347`.
- DIP fine-tune: `epochs=40`, `batch_size=12`, `lr=5e-6`, initialized from AMASS best, best epoch `40`, best selection value `0.038939811958698556`.
- Checkpoint selection used control-point physical output closeness (`gt_control_pRB + gt_control_gR1`), not arbitrary weighted loss.

Evaluation JSONs:

```text
data/experiments/newpl_v5_official_protocol_20260607_tuned/eval/dip_test_after_amass_pretrain.json
data/experiments/newpl_v5_official_protocol_20260607_tuned/eval/tc_test_after_amass_pretrain.json
data/experiments/newpl_v5_official_protocol_20260607_tuned/eval/dip_test_after_dip_finetune.json
data/experiments/newpl_v5_official_protocol_20260607_tuned/eval/tc_test_after_dip_finetune.json
```

AMASS-only module results:

- DIP test: `official_PL` has pRB L1 `3.115170 cm`, pRB L2 `6.419473 cm`, gR1 `12.947709 deg`; `newpl_v4_init36` has `3.116842 cm`, `6.441447 cm`, `12.765167 deg`; `newpl_v5_amass` has `3.127556 cm`, `6.454484 cm`, `12.551949 deg`.
- TotalCapture test: `official_PL` has pRB L1 `3.370257 cm`, pRB L2 `6.995536 cm`, gR1 `13.450453 deg`; `newpl_v4_init36` has `3.210470 cm`, `6.654393 cm`, `13.329531 deg`; `newpl_v5_amass` has `3.264119 cm`, `6.783332 cm`, `13.415420 deg`.

DIP-fine-tuned module results:

- DIP test: `newpl_v5_dip_best` has pRB L1 `3.120847 cm`, pRB L2 `6.445578 cm`, gR1 `12.552613 deg`.
- Compared with AMASS-only on DIP test, DIP fine-tune improves pRB L1 by `0.006709 cm` and pRB L2 by `0.008906 cm`, while gR1 worsens only `0.000664 deg`.
- Compared with `official_PL` on DIP test, DIP-finetuned v5 is still worse on pRB L2 by `0.026105 cm`, but better on gR1 by `0.395096 deg`.
- Compared with `newpl_v4_init36` on DIP test, DIP-finetuned v5 is worse on pRB L2 by `0.004131 cm`, but better on gR1 by `0.212554 deg`.
- TotalCapture test after DIP fine-tune: `newpl_v5_dip_best` has pRB L1 `3.264551 cm`, pRB L2 `6.780749 cm`, gR1 `13.415189 deg`.
- Compared with AMASS-only on TotalCapture, DIP fine-tune changes little: pRB L2 improves by `0.002583 cm`, gR1 improves by `0.000231 deg`, and pRB L1 worsens by `0.000432 cm`.
- Compared with `official_PL` on TotalCapture, DIP-finetuned v5 remains better on pRB L2 by `0.214787 cm` and gR1 by `0.035276 deg`.
- Compared with `newpl_v4_init36` on TotalCapture, DIP-finetuned v5 is worse on pRB L2 by `0.126356 cm` and worse on gR1 by `0.085658 deg`.

Conclusion:

- DIP fine-tune has a real but small positive effect on v5 pRB, especially on DIP test.
- The official-like AMASS -> DIP route does not prove v5 is better than the current `newpl_v4_init36` mainline. v5 improves gR1 on DIP, but pRB remains slightly worse than both official_PL and v4 on DIP, and v5 remains clearly worse than v4 on TotalCapture.
- Current `newpl_v4_init36` should remain the selected PL mainline until v5 improves both pRB and gR1, or until a full-pipeline run shows a downstream benefit that justifies the pRB tradeoff.

## EXP-20260607-newpl_v5_delay_eval — future-output delay on PL-cache module metrics

Question: Because the curve control-point tail can be unstable, can `newpl_v5` improve module output by delaying PL output 1 or 2 frames without retraining?

Implemented change: eval-only future-output delay. The checkpoint output is not changed or retrained. For delay `d`, module metrics compare `pred[t+d]` against `GT[t]`; boundary frames are cropped from the evaluated interval.

Files:

```text
pl_curve_delay_eval.py
scripts/run_newpl_v5_delay_eval_20260607.sh
newpl_root_eval.py
```

Artifacts:

```text
root: data/experiments/newpl_v5_delay_eval_20260607
DIP JSON: data/experiments/newpl_v5_delay_eval_20260607/eval/dip_test_delay_module_metrics.json
TC JSON: data/experiments/newpl_v5_delay_eval_20260607/eval/tc_test_delay_module_metrics.json
summary: data/experiments/newpl_v5_delay_eval_20260607/summary.json
summary md: data/experiments/newpl_v5_delay_eval_20260607/summary.md
```

Runtime note: the first generic `newpl_root_eval.py` route was too slow because it recomputed GPNet/SMPL targets. The final route uses `pl_curve_cache_v2` directly: official_PL is `pl_base`, GT is `pl_target`, and v5 checkpoints are evaluated with one raw forward per checkpoint plus delay-aligned metrics. No training was run.

Key PL output results:

| Dataset | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ |
|---|---|---:|---:|---:|
| DIP test | official_PL | `3.115170` | `6.419473` | `12.947689` |
| DIP test | newpl_v5_dip_delay0 | `3.120847` | `6.445578` | `12.552613` |
| DIP test | newpl_v5_dip_delay1 | `3.129481` | `6.455922` | `12.554120` |
| DIP test | newpl_v5_dip_delay2 | `3.234059` | `6.655651` | `12.553713` |
| TotalCapture test | official_PL | `3.370257` | `6.995536` | `13.450445` |
| TotalCapture test | newpl_v5_dip_delay0 | `3.264551` | `6.780749` | `13.415190` |
| TotalCapture test | newpl_v5_dip_delay1 | `3.279620` | `6.815473` | `13.416209` |
| TotalCapture test | newpl_v5_dip_delay2 | `3.381457` | `7.026811` | `13.417055` |

Temporal result: delay1/2 slightly reduce some jitter/temporal values, but pRB L1/L2 regress. The best pRB setting is delay0 on both DIP and TotalCapture.

Decision: do not connect delayed `newpl_v5` to full-pipeline S4. The delay idea does not solve the pRB weakness; it only trades small smoothness/temporal changes for worse control-point accuracy.

## EXP-20260608-newik1_v10_official_protocol_last_control

Purpose: retrain NewIK1 with an official-like route and PL control-point input, then test whether replacing IK1 improves over official IK1 under the same NewPL upstream.

Protocol:

- Stage A: AMASS teacher-forced pretrain.
- Stage B: AMASS PL-streaming adaptation using `newpl_v5_amass`.
- Stage C: DIP-IMU PL-streaming fine-tune using `newpl_v5_dip`.
- Evaluation: DIP-IMU test and TotalCapture test.
- DIP translation loss was not used.
- TotalCapture train split was not used.

Implementation/artifacts:

```text
script: scripts/run_newik1_v10_official_protocol_last_control_20260607.sh
root: data/experiments/newik1_v10_official_protocol_last_control_20260607
log: data/experiments/newik1_v10_official_protocol_last_control_20260607/logs/run_full.log
summary json: data/experiments/newik1_v10_official_protocol_last_control_20260607/summary.json
summary md: data/experiments/newik1_v10_official_protocol_last_control_20260607/summary.md
```

NewIK1 contract:

- Input feature mode: `last_control`.
- Input: `RRB_after_pl[45] + last_control_gR1[3] + last_control_pRB[15] = 63D`.
- Output: official IK1 contract `pRJ[69] + gR2[3] = 72D`.
- Loss recipe: `pRJ=2.0`, `gR2=1.0`, `pRJ_dot=0.01`, `pRJ_ddot=0.0003`, `gR2_dot=0.03`, `gR2_ddot=0.001`, `control_pRJ=0.1`, `control_gR2=0.1`; control derivative, bone length, control-point prior, and tail-update prior weights are zero.

Training results:

| Stage | Status | Best epoch | Best loss | Checkpoint |
|---|---|---:|---:|---|
| Stage A AMASS teacher-forced | ok | 33 | `0.00048058280081022533` | `stage_a_amass_teacher_forced/best_loss.pt` |
| Stage B AMASS PL-streaming | ok | 20 | `0.013035929867764934` | `stage_b_amass_pl_streaming/best_loss.pt` |
| Stage C DIP PL-streaming | ok | 40 | `0.13303746217085669` | `stage_c_dip_pl_streaming/best_loss.pt` |

DIP-IMU test evidence:

| Version | Score ↓ | pRJ L1 cm ↓ | pRJ L2 cm ↓ | gR2 angle deg ↓ | Notes |
|---|---:|---:|---:|---:|---|
| official_gpnet | `44.642051` | `2.451623` | `5.082861` | `15.268174` | original official stack |
| newpl_v5_amass + official IK1 | `44.644916` | `2.463606` | `5.112111` | `14.864371` | AMASS-only NewPL |
| newpl_v5_dip + official IK1 | `44.598659` | `2.461882` | `5.107541` | `14.869393` | strongest DIP score in this run |
| newik1_v10 Stage A | `44.750028` | `2.467687` | `5.104484` | `15.297582` | worse score |
| newik1_v10 Stage B | `44.797428` | `2.459753` | `5.094634` | `15.155596` | best NewIK1 pRJ L2, worse score |
| newik1_v10 Stage C | `44.730331` | `2.456168` | `5.087737` | `15.128205` | improves pRJ vs NewPL+official IK1, but score and gR2 worse |

TotalCapture test evidence:

| Version | Score ↓ | pRJ L1 cm ↓ | pRJ L2 cm ↓ | gR2 angle deg ↓ | Notes |
|---|---:|---:|---:|---:|---|
| official_gpnet | `44.477381` | `2.330376` | `4.773225` | `15.323585` | original official stack |
| newpl_v5_amass + official IK1 | `43.868067` | `2.292379` | `4.699349` | `15.175444` | best TC score in this run |
| newpl_v5_dip + official IK1 | `43.872717` | `2.289634` | `4.693114` | `15.181975` | very close to AMASS-only NewPL baseline |
| newik1_v10 Stage A | `44.708547` | `2.292315` | `4.695573` | `15.578022` | score/gR2 worse |
| newik1_v10 Stage B | `44.991837` | `2.322065` | `4.764910` | `15.398288` | worse |
| newik1_v10 Stage C | `44.900650` | `2.317402` | `4.754637` | `15.368013` | worse |

Temporal/smoothness observations:

- DIP: NewIK1 Stage C has lower pRJ ddot (`0.022654`) than official (`0.022890`) and NewPL+official IK1 (`0.022744`), but gR2 angle is worse than NewPL+official IK1.
- TotalCapture: NewIK1 Stage C has better pRJ dot/ddot (`0.024761`/`0.018325`) than NewPL+official IK1 (`0.025832`/`0.020366`), but score and gR2 angle are worse.

Decision:

- `newik1_v10_official_protocol_last_control` is not selected.
- The official-like AMASS -> DIP route completed successfully, but it does not beat `newpl_v5_dip + official IK1` on DIP score and does not beat `newpl_v5_amass/newpl_v5_dip + official IK1` on TotalCapture score.
- The model can slightly improve `pRJ` module L2 and smoothness in places, but the gain is too small and is offset by worse `gR2` and worse downstream score.
- Keep official IK1 for this official-like NewPL v5 route. Future NewIK1 work should optimize downstream-aware objectives or preserve/distill `gR2` more strongly instead of selecting by local IK1 loss alone.

## EXP-20260608-newpose_ctrl_v1_official_protocol

Status: completed; `newpose_ctrl_v1` is rejected for current mainline promotion.

Purpose: test a control-point pose / IK2-slot replacement module (`newpose_ctrl_v1`) under the official-like route:

```text
AMASS pretrain -> DIP-IMU fine-tune -> DIP-IMU test + TotalCapture test
```

Contract:

| Item | Value |
|---|---|
| Replaced slot | IK-s2 / pose-control slot, before VR/physics |
| Frame input | `official IMU[90] + RRB_after_pl[45] + pRB/gR1[18] + last PL control[18] + gR0[3] = 174D` |
| Init-only input | `offset_r / r_JS`, used only in `reset_stream` hidden-state init |
| Output | `RRJ_control[90] + gR_pose_control[3] = 93D` |
| DIP trans/root loss | not used |
| TotalCapture train split | not used |

Implementation/artifacts:

| Artifact | Path |
|---|---|
| Module | `newpose_ctrl.py` |
| Cache builder | `newpose_ctrl_cache.py` |
| Trainer | `newpose_ctrl_train.py` |
| NewPose eval | `newpose_ctrl_eval.py` |
| Baseline IK2-slot module eval | `newpose_baseline_ik2_module_eval.py` |
| Runner | `scripts/run_newpose_ctrl_v1_official_protocol_20260608.sh` |
| Summarizer | `scripts/summarize_newpose_ctrl_v1.py` |
| Root | `data/experiments/newpose_ctrl_v1_20260608` |
| Full log | `data/experiments/newpose_ctrl_v1_20260608/logs/run_full.log` |
| Fixed module eval log | `data/experiments/newpose_ctrl_v1_20260608/logs/run_module_eval_fixed.log` |
| Fixed full eval log | `data/experiments/newpose_ctrl_v1_20260608/logs/run_full_eval_fixed.log` |

Training result:

| Stage | Status | Best epoch | Best selection value | Selection metric | Notes |
|---|---|---:|---:|---|---|
| Stage A AMASS pretrain | ok | 23 | `0.042461989620351234` | `control_pose_physical` | bounded validation: 50 AMASS seq, 300-frame windows |
| Stage B DIP fine-tune | ok | 40 | `0.03869467038415071` | `control_pose_physical` | bounded validation: 6 DIP val seq, 300-frame windows |

Throughput/config fixes made during run:

- Stage B was initially slow because validation evaluated full DIP-val sequences every epoch. It was changed to `--max-val-sequences 6 --val-window 300`; training then completed quickly.
- `pl_curve_eval.py` previously reran official GPNet inside every NewPL baseline evaluation to fill internal `baseline_metrics`. This was unnecessary because official baselines are evaluated separately. Added `--skip-baseline-rerun` and updated the runner to use it.
- The runner now skips smoke eval JSONs when they already exist.
- Added IK2-slot module eval for official/newpl_v5 baselines so module decoded-pose/FK metrics can be compared against `newpose_ctrl_v1` module metrics. The first helper was mistakenly launched with `CUDA_VISIBLE_DEVICES=` and was too slow; the runner was corrected to keep CUDA enabled.
- `scripts/run_newpose_ctrl_v1_official_protocol_20260608.sh` now defaults to `RUN_MODULE_EVAL=1` and `RUN_FULL_EVAL=0`, so long training no longer automatically chains full-pipeline 11-metric evaluation. Full-pipeline eval must be explicitly enabled and can be capped with `FULL_EVAL_MAX_SEQUENCES`.
- `newpose_baseline_ik2_module_eval.py` and `newpose_ctrl_eval.py` now reuse `GPNet`/body-model objects per process instead of rebuilding them for every sequence.
- Baseline module eval exposed a CPU/CUDA mismatch in GT pose-state target construction; the target path now stays on CPU while network inference stays on GPU. A `--max-eval-sequences 1` smoke for `official_gpnet` on DIP passed with `all_finite=true`.
- `scripts/summarize_newpose_ctrl_v1.py` now reads newpose module-only JSONs from `eval_module/` and treats `status != ok` as incomplete, not just missing files.
- `NewPoseControlModule.step()` originally evaluated the full spline prefix every frame and only consumed `curve[:, -2]`, causing O(T^2) time/memory on long eval sequences. It was changed to an equivalent local newest-control stencil: `state=(prev+5*current)/6`, `dot=(current-prev)/(2dt)`, `ddot=(prev-current)/dt^2`. Random-control equivalence checks passed locally and on zktitan before rerunning module eval.

Current completed full-pipeline JSONs:

| Dataset | Version | JSON | Score | all_finite |
|---|---|---|---:|---|
| DIP-IMU test | official_gpnet | `eval/dip_official_gpnet.json` | `44.642051114626234` | true |
| TotalCapture test | official_gpnet | `eval/tc_official_gpnet.json` | `44.477381113264705` | true |
| DIP-IMU test | newpl_v5_amass + official downstream | `eval/dip_newpl_v5_amass_official_ik2.json` | `44.644916377081294` | true |
| TotalCapture test | newpl_v5_amass + official downstream | `eval/tc_newpl_v5_amass_official_ik2.json` | `43.86806740006432` | true |
| DIP-IMU test | newpl_v5_dip + official downstream | `eval/dip_newpl_v5_dip_official_ik2.json` | `44.598659158501974` | true |
| TotalCapture test | newpl_v5_dip + official downstream | `eval/tc_newpl_v5_dip_official_ik2.json` | `43.872716777753084` | true |

The four NewPL v5 + official downstream full-pipeline JSONs were copied from `data/experiments/newik1_v10_official_protocol_last_control_20260607/eval/` after verifying the same official IMU mode, same DIP/TotalCapture test caches, same NewPL v5 checkpoints, and `ik1_backend=original`. They are marked with `reused_from` in the JSONs to avoid redundant full-pipeline baseline reruns.

Final summary artifacts:

| Artifact | Path |
|---|---|
| Summary JSON | `data/experiments/newpose_ctrl_v1_20260608/summary.json` |
| Summary tables | `data/experiments/newpose_ctrl_v1_20260608/summary_tables.md` |
| Module JSON dir | `data/experiments/newpose_ctrl_v1_20260608/eval_module/` |
| Full JSON dir | `data/experiments/newpose_ctrl_v1_20260608/eval/` |

Final full-pipeline table:

| Dataset | Version | Status | S4 score ↓ | Local angle ↓ | Global angle ↓ | all_finite |
| --- | --- | --- | --- | --- | --- | --- |
| DIP-IMU test | official_gpnet | ok | 44.642051 | 8.469930 | 8.291750 | 1.000000 |
| DIP-IMU test | newpl_v5_amass_official_ik2 | ok | 44.644916 | 8.479104 | 8.332290 | 1.000000 |
| DIP-IMU test | newpl_v5_dip_official_ik2 | ok | 44.598659 | 8.468339 | 8.315847 | 1.000000 |
| DIP-IMU test | newpose_ctrl_v1_stage_a_best | ok | 428.806986 | 100.590216 | 101.018856 | 1.000000 |
| DIP-IMU test | newpose_ctrl_v1_stage_b_best | ok | 432.122581 | 100.867367 | 101.714333 | 1.000000 |
| DIP-IMU test | newpose_ctrl_v1_stage_b_last | ok | 432.122581 | 100.867367 | 101.714333 | 1.000000 |
| TotalCapture test | official_gpnet | ok | 44.477381 | 12.550695 | 11.781375 | 1.000000 |
| TotalCapture test | newpl_v5_amass_official_ik2 | ok | 43.868067 | 12.423023 | 11.680961 | 1.000000 |
| TotalCapture test | newpl_v5_dip_official_ik2 | ok | 43.872717 | 12.423826 | 11.674817 | 1.000000 |
| TotalCapture test | newpose_ctrl_v1_stage_a_best | ok | 413.495453 | 95.741745 | 98.819038 | 1.000000 |
| TotalCapture test | newpose_ctrl_v1_stage_b_best | ok | 419.196776 | 96.512819 | 99.903576 | 1.000000 |
| TotalCapture test | newpose_ctrl_v1_stage_b_last | ok | 419.196776 | 96.512819 | 99.903576 | 1.000000 |

Final module IK2-slot / pose-control table:

| Dataset | Version | Status | Control RRJ deg ↓ | State RRJ deg ↓ | FK joint L2 cm ↓ | gR loss ↓ | all_finite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DIP-IMU test | official_gpnet | ok | not available | 102.407646 | 4.971125 | 0.001227 | 1.000000 |
| DIP-IMU test | newpl_v5_amass_official_ik2 | ok | not available | 102.249420 | 5.000293 | 0.001485 | 1.000000 |
| DIP-IMU test | newpl_v5_dip_official_ik2 | ok | not available | 102.242027 | 4.994349 | 0.001464 | 1.000000 |
| DIP-IMU test | newpose_ctrl_v1_stage_a_best | ok | 66.256638 | 66.101151 | 44.539856 | 0.015581 | 1.000000 |
| DIP-IMU test | newpose_ctrl_v1_stage_b_best | ok | 65.028152 | 64.682121 | 45.447945 | 0.015881 | 1.000000 |
| DIP-IMU test | newpose_ctrl_v1_stage_b_last | ok | 65.028152 | 64.682121 | 45.447945 | 0.015881 | 1.000000 |
| TotalCapture test | official_gpnet | ok | not available | 77.909523 | 4.705381 | 0.003662 | 1.000000 |
| TotalCapture test | newpl_v5_amass_official_ik2 | ok | not available | 77.640457 | 4.608555 | 0.004232 | 1.000000 |
| TotalCapture test | newpl_v5_dip_official_ik2 | ok | not available | 77.620567 | 4.600487 | 0.004123 | 1.000000 |
| TotalCapture test | newpose_ctrl_v1_stage_a_best | ok | 52.495689 | 52.634342 | 43.782135 | 0.027203 | 1.000000 |
| TotalCapture test | newpose_ctrl_v1_stage_b_best | ok | 51.712868 | 51.901062 | 44.701134 | 0.029040 | 1.000000 |
| TotalCapture test | newpose_ctrl_v1_stage_b_last | ok | 51.712868 | 51.901062 | 44.701134 | 0.029040 | 1.000000 |

Conclusion:

- `newpose_ctrl_v1` is not selected.
- DIP fine-tune does not improve the full-pipeline result. DIP score worsens from `428.806986` after AMASS pretrain to `432.122581` after DIP fine-tune.
- TotalCapture full-pipeline score also worsens after DIP fine-tune, from `413.495453` to `419.196776`.
- The module-level RRJ geodesic numbers for `newpose_ctrl_v1` look lower than official IK2-slot state geodesic, but this is not a valid win because decoded FK joint L2 is roughly `43.78-45.45 cm`, while official/newpl_v5 baselines are about `4.60-5.00 cm`.
- The full-pipeline metrics confirm the decoded pose/control state is not compatible with official VR/physics downstream. Do not connect this module to the mainline.

## EXP-20260608-newpose_ctrl_v2_fk_leaf

Status: implemented; smoke passed; formal AMASS -> DIP training not started in this entry.

Purpose: replace `newpose_ctrl_v1` with a pose-control module whose validation is tied to physically decoded SMPL FK outputs. The module still outputs control points, but loss/selection now checks whether those controls decode to GT-like body-space leaf and joint positions.

Contract:

| Item | Value |
|---|---|
| Version | `newpose_ctrl_v2_fk_leaf` |
| Frame input | `official IMU[90] + RRB_after_pl[45] + pRB/gR1[18] + last PL control[18] + gR0[3] = 174D` |
| Init-only input | `offset_r / r_JS`, used only for hidden-state init |
| Output | `RRJ_control[90] + gR_pose_control[3] = 93D` |
| Decoded physical frame | root/body frame; leaf positions use `(leaf_world - root_world) @ R_root` |
| FK vertex mask | five leaf vertices `(1961, 5424, 1176, 4662, 411)` plus root vertex `3021` |
| DIP trans/root loss | not used |

Implementation:

| Artifact | Path |
|---|---|
| Module/loss helpers | `newpose_ctrl.py` |
| Trainer | `newpose_ctrl_train.py` |
| NewPose evaluator | `newpose_ctrl_eval.py` |
| Baseline evaluator with same FK leaf metrics | `newpose_baseline_ik2_module_eval.py` |
| Runner | `scripts/run_newpose_ctrl_v2_fk_leaf_official_protocol_20260608.sh` |
| Smoke root | `data/experiments/newpose_ctrl_v2_fk_leaf_20260608_smoke3` |

Loss and checkpoint selection:

| Term | Meaning |
|---|---|
| existing v1 control/state terms | retained for `RRJ_control`, `gR_pose_control`, temporal control/state smoothness |
| `fk_leaf_pos` | SmoothL1 between decoded predicted and GT root-relative leaf vertices |
| `fk_leaf_vel` | first-difference decoded leaf velocity error |
| `fk_leaf_acc` | second-difference decoded leaf acceleration error |
| `fk_joint_pos` | SmoothL1 between decoded predicted and GT root-relative SMPL joints |
| `fk_leaf_physical` | best checkpoint metric: `FK_leaf_L2_cm + 0.25*FK_leaf_vel + 0.10*FK_leaf_acc + 0.10*FK_joint_L2_cm + 0.05*gR_pose` |

Engineering fixes made for v2:

- `newpose_ctrl_train.py` now loads and batches `pose_gt`, `RMB`, and `gR0` from the existing NewPose cache so FK losses are actually supervised.
- Training/eval FK uses a masked SMPL body model with only the required six vertices, avoiding full 6890-vertex mesh construction in every batch.
- The differentiable training decode path no longer SVD-projects pose matrices; SVD projection is kept in non-training decode/eval.
- `newpose_ctrl` internal matrix-to-6D encoding was corrected to match `art.math.r6d_to_rotation_matrix`'s two-column-vector convention. The base identity control is now `[1,0,0,0,1,0]`, avoiding the previous collinear `[1,0,0,1,0,0]` NaN-gradient point.
- Baseline module eval now reports the same FK leaf metrics, so future comparisons are fair against official GPNet and `newpl_v5 + official IK2`.

Smoke command:

```bash
SMOKE_ONLY=1 RUN_MODULE_EVAL=1 ROOT=data/experiments/newpose_ctrl_v2_fk_leaf_20260608_smoke3 \
  bash scripts/run_newpose_ctrl_v2_fk_leaf_official_protocol_20260608.sh
```

Smoke result:

| Check | Result |
|---|---|
| cache | `data/experiments/newpose_ctrl_v2_fk_leaf_20260608_smoke3/caches/smoke_dip_val_newpl_dip/newpose_ctrl_cache_manifest.json` |
| training | 1 epoch, finite train/validation loss |
| module eval | `data/experiments/newpose_ctrl_v2_fk_leaf_20260608_smoke3/eval/smoke_module.json` |
| all_finite | true |
| FK leaf metrics present | yes: L1/L2/per-leaf/velocity/acceleration/jitter |

Formal run command:

```bash
RUN_MODULE_EVAL=1 RUN_FULL_EVAL=0 \
  bash scripts/run_newpose_ctrl_v2_fk_leaf_official_protocol_20260608.sh
```

Default formal batches: Stage A AMASS `STAGE_A_BATCH=32`, Stage B DIP `STAGE_B_BATCH=16`. These can be increased after GPU memory probing; they are not batch size 1.

Formal run result:

| Artifact | Path |
|---|---|
| Root | `data/experiments/newpose_ctrl_v2_fk_leaf_20260608` |
| Full log | `data/experiments/newpose_ctrl_v2_fk_leaf_20260608/logs/run_full.log` |
| Summary JSON | `data/experiments/newpose_ctrl_v2_fk_leaf_20260608/summary.json` |
| Summary tables | `data/experiments/newpose_ctrl_v2_fk_leaf_20260608/summary_tables.md` |
| Stage A best | `data/experiments/newpose_ctrl_v2_fk_leaf_20260608/stage_a_amass_pretrain/best_loss.pt` |
| Stage B best | `data/experiments/newpose_ctrl_v2_fk_leaf_20260608/stage_b_dip_finetune/best_loss.pt` |

Formal command:

```bash
CUDA_VISIBLE_DEVICES=0 \
ROOT=data/experiments/newpose_ctrl_v2_fk_leaf_20260608 \
STAGE_A_BATCH=64 \
STAGE_B_BATCH=24 \
RUN_MODULE_EVAL=1 \
RUN_FULL_EVAL=0 \
/home/lingfeng/bin/longrun -- bash -lc 'set -euo pipefail; bash scripts/run_newpose_ctrl_v2_fk_leaf_official_protocol_20260608.sh 2>&1 | tee data/experiments/newpose_ctrl_v2_fk_leaf_20260608/logs/run_full.log'
```

Training:

| Stage | Status | Epochs run | Best epoch | Best selection | Last selection | Notes |
|---|---|---:|---:|---:|---:|---|
| Stage A AMASS pretrain | early_stopped | 29 | 19 | `16.734057` | `16.780566` | selected by `fk_leaf_physical` |
| Stage B DIP fine-tune | early_stopped | 11 | 1 | `18.760393` | `19.036661` | DIP fine-tune did not improve |

Final module comparison:

| Dataset | Version | FK leaf L2 cm ↓ | FK joint L2 cm ↓ | State RRJ deg ↓ | gR loss ↓ | all_finite |
|---|---|---:|---:|---:|---:|---|
| DIP-IMU test | official_gpnet | `6.234410` | `4.971124` | `9.994508` | `0.001227` | true |
| DIP-IMU test | newpl_v5_amass + official IK2 | `6.258889` | `5.000293` | `9.986519` | `0.001485` | true |
| DIP-IMU test | newpl_v5_dip + official IK2 | `6.254980` | `4.994349` | `9.979830` | `0.001464` | true |
| DIP-IMU test | newpose_ctrl_v2 Stage A best | `20.288481` | `14.479343` | `30.818228` | `0.015539` | true |
| DIP-IMU test | newpose_ctrl_v2 Stage B best | `20.285948` | `14.477565` | `30.793276` | `0.015544` | true |
| DIP-IMU test | newpose_ctrl_v2 Stage B last | `20.308220` | `14.487782` | `30.473463` | `0.015660` | true |
| TotalCapture test | official_gpnet | `5.933897` | `4.705381` | `11.220174` | `0.003662` | true |
| TotalCapture test | newpl_v5_amass + official IK2 | `5.770283` | `4.608556` | `11.040092` | `0.004232` | true |
| TotalCapture test | newpl_v5_dip + official IK2 | `5.766959` | `4.600485` | `11.047222` | `0.004123` | true |
| TotalCapture test | newpose_ctrl_v2 Stage A best | `18.867359` | `13.679826` | `25.630533` | `0.027010` | true |
| TotalCapture test | newpose_ctrl_v2 Stage B best | `18.869814` | `13.694618` | `25.636141` | `0.027035` | true |
| TotalCapture test | newpose_ctrl_v2 Stage B last | `19.102987` | `14.040706` | `25.847134` | `0.028097` | true |

Decision:

- `newpose_ctrl_v2_fk_leaf` is not selected.
- It is much better than `newpose_ctrl_v1` on decoded FK body geometry, but still far worse than official GPNet / `newpl_v5 + official IK2`.
- DIP fine-tune did not help; best Stage B checkpoint is epoch 1, and later epochs worsen the selection metric.
- Do not connect this module to full pipeline.

## EXP-20260608-bone_aux_v4_winner_rjs_dip_ft

Question: If the already AMASS-trained `bone_aux_newpl_20260608_v4` checkpoint is fine-tuned on the new winner foot-lock TransPose DIP pseudo-`r_JS` cache, does the module improve over old bone_aux v4?

Implementation:

- Added script: `scripts/run_bone_aux_v4_winner_rjs_dip_finetune_20260608.sh`.
- Init checkpoint: `data/experiments/bone_aux_newpl_20260608_v4/amass_bone_aux/best_loss.pt`.
- DIP train/val/test PL caches rebuilt with external winner pseudo-`r_JS`:
  - `data/experiments/footlock_transpose_rjs_20260608/dip_train_footlock_transpose_rjs.pt`
  - `data/experiments/footlock_transpose_rjs_20260608/dip_val_footlock_transpose_rjs.pt`
  - `data/experiments/footlock_transpose_rjs_20260608/dip_test_footlock_transpose_rjs.pt`
- Output root: `data/experiments/bone_aux_newpl_20260608_v4_winner_rjs_dip_ft`.
- Training: 40 DIP epochs, best epoch 40, best selection value `0.290444923738202`.

Fair DIP test comparison on the same winner-rJS cache:

| Checkpoint | pRB base cm | pRB new cm | delta pRB cm | gR1 base deg | gR1 new deg | delta gR1 deg | bone base deg | bone new deg | delta bone deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| new winner-rJS DIP ft best | `6.528883` | `6.524416` | `-0.004467` | `15.267228` | `15.191690` | `-0.075539` | `2.552935` | `3.493121` | `0.940186` |
| old v4 DIP best on winner-rJS | `6.528883` | `6.524327` | `-0.004555` | `15.267228` | `15.191858` | `-0.075371` | `2.552935` | `2.921854` | `0.368919` |
| AMASS v4 best on winner-rJS | `6.528883` | `6.520958` | `-0.007924` | `15.267228` | `15.171401` | `-0.095828` | `2.552935` | `3.700672` | `1.147737` |

Artifacts:

| Artifact | Path |
|---|---|
| Script | `scripts/run_bone_aux_v4_winner_rjs_dip_finetune_20260608.sh` |
| Train result | `data/experiments/bone_aux_newpl_20260608_v4_winner_rjs_dip_ft/dip_finetune/train_result.json` |
| Summary | `data/experiments/bone_aux_newpl_20260608_v4_winner_rjs_dip_ft/summary.md` |
| Fair comparison | `data/experiments/bone_aux_newpl_20260608_v4_winner_rjs_dip_ft/comparison_summary.md` |

Decision:

- Not selected.
- Winner-rJS DIP fine-tune is finite and reproducible, but it does not improve useful PL module metrics over old bone_aux v4.
- Compared with old v4 on the same winner-rJS DIP test cache, pRB is `+0.000089 cm` worse, gR1 is `-0.000168 deg` better, and bone orientation is `+0.571267 deg` worse.
- This supports the earlier conclusion that simply improving DIP pseudo-`r_JS` or doing a straightforward DIP fine-tune is not enough; the NewPL interface/loss still does not use `r_JS` in a helpful way.

<!-- BEGIN newpl-offset-v6-and-newik1-v11-control-only-training--2026-06-09 -->
## NewPL-offset v6 and NewIK1 v11 control-only training (2026-06-09)

# NewPL-offset v6 and NewIK1 v11 control-only training (2026-06-09)

## PL Module Comparison

| Dataset | Version | pRB L2 cm ↓ | gR1 angle deg ↓ | Notes |
| --- | --- | ---: | ---: | --- |
| amass | official PL baseline | 3.341522 | 6.395696 | official baseline from cache |
| amass | newpl_v4_init36 baseline | not available | not available | not available if checkpoint path is unset |
| amass | newpl_v5_dip_best | 3.424267 | 6.301991 | prior best official-route NewPL |
| amass | canonical_control_dip_best | 3.354085 | 6.389052 | canonical control target baseline |
| amass | newpl_offset_v6_best | 3.355032 | 6.391141 | control-only offset-aware |
| amass | newpl_offset_v6_acc_aux_best | 3.349999 | 6.391822 | offset-aware with IMU acceleration auxiliary |
| dip_test | official PL baseline | 6.528883 | 15.267228 | official baseline from cache |
| dip_test | newpl_v4_init36 baseline | not available | not available | not available if checkpoint path is unset |
| dip_test | newpl_v5_dip_best | 6.540273 | 14.801741 | prior best official-route NewPL |
| dip_test | canonical_control_dip_best | 6.527511 | 15.256069 | canonical control target baseline |
| dip_test | newpl_offset_v6_best | 6.532787 | 15.257532 | control-only offset-aware |
| dip_test | newpl_offset_v6_acc_aux_best | 6.531946 | 15.257427 | offset-aware with IMU acceleration auxiliary |
| tc_test | official PL baseline | 6.768144 | 14.014337 | official baseline from cache |
| tc_test | newpl_v4_init36 baseline | not available | not available | not available if checkpoint path is unset |
| tc_test | newpl_v5_dip_best | 6.567221 | 13.933652 | prior best official-route NewPL |
| tc_test | canonical_control_dip_best | 6.728185 | 14.042532 | canonical control target baseline |
| tc_test | newpl_offset_v6_best | 6.728234 | 14.034698 | control-only offset-aware |
| tc_test | newpl_offset_v6_acc_aux_best | 6.753339 | 14.030225 | offset-aware with IMU acceleration auxiliary |

## IK1 Module Comparison

| Dataset | Version | pRJ L2 cm ↓ | leaf pRJ L2 cm ↓ | gR2 angle deg ↓ | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| amass | official IK1 baseline | 1.822993 | 1.788325 | 6.536589 | canonical-control NewPL cache |
| amass | newik1_v10_stage_c_best | 1.838415 | 1.771354 | 6.946709 | canonical-control NewPL cache |
| amass | newik1_v11_best | 1.860457 | 1.798346 | 6.762720 | canonical-control NewPL cache |
| amass | newik1_v11_last | 1.860455 | 1.798345 | 6.762721 | canonical-control NewPL cache |
| dip_val | official IK1 baseline | 1.676843 | 1.723072 | 17.704771 | canonical-control NewPL cache |
| dip_val | newik1_v10_stage_c_best | 1.701203 | 1.750822 | 18.175951 | canonical-control NewPL cache |
| dip_val | newik1_v11_best | 1.698854 | 1.720655 | 18.001260 | canonical-control NewPL cache |
| dip_val | newik1_v11_last | 1.698852 | 1.720654 | 18.001256 | canonical-control NewPL cache |
| dip_test | official IK1 baseline | 3.864125 | 4.214985 | 15.261334 | canonical-control NewPL cache |
| dip_test | newik1_v10_stage_c_best | 3.864403 | 4.217302 | 15.543039 | canonical-control NewPL cache |
| dip_test | newik1_v11_best | 3.856695 | 4.222304 | 15.467358 | canonical-control NewPL cache |
| dip_test | newik1_v11_last | 3.856695 | 4.222304 | 15.467355 | canonical-control NewPL cache |
| tc_test | official IK1 baseline | 3.518937 | 3.455496 | 15.352110 | canonical-control NewPL cache |
| tc_test | newik1_v10_stage_c_best | 3.544581 | 3.509821 | 15.551805 | canonical-control NewPL cache |
| tc_test | newik1_v11_best | 3.492015 | 3.441835 | 15.513044 | canonical-control NewPL cache |
| tc_test | newik1_v11_last | 3.492016 | 3.441837 | 15.513038 | canonical-control NewPL cache |

## Contracts

- NewPL output remains `pRB[15] + gR1[3]`.
- IK1 output remains `pRJ[69] + gR2[3]`.
- Derivative and second-derivative loss terms are disabled; fitted GT control-point losses are used instead.
- `r_JS` is the IMU origin relative to mapped joint `J`, expressed in the joint-local frame; `p_WS=p_WJ+R_WJ@r_JS`.
- DIP trans/root velocity/global trajectory GT is not used.

## Measured Conclusions

### NewPL-offset v6

- amass: `newpl_offset_v6_best` pRB delta vs official/v5/canonical = +0.013510 cm / -0.069235 cm / +0.000947 cm; gR1 delta = -0.004555 deg / +0.089149 deg / +0.002089 deg.
- dip_test: `newpl_offset_v6_best` pRB delta vs official/v5/canonical = +0.003905 cm / -0.007485 cm / +0.005277 cm; gR1 delta = -0.009696 deg / +0.455791 deg / +0.001463 deg.
- tc_test: `newpl_offset_v6_best` pRB delta vs official/v5/canonical = -0.039909 cm / +0.161013 cm / +0.000050 cm; gR1 delta = +0.020362 deg / +0.101047 deg / -0.007833 deg.
- Verdict: `newpl_offset_v6_best` is close to the canonical-control baseline but does not clearly beat the prior `newpl_v5_dip_best` or the official PL baseline across pRB and gR1. Do not promote it as the selected PL mainline.
- `newpl_offset_v6_acc_aux_best` is not adopted: it gives a small AMASS pRB gain, but DIP/TC pRB and gR1 are not consistently better than the control-only branch or the prior PL baselines.

### Offset / IMU Acceleration Validation

- control-only dip_test: zero-offset minus true-offset pRB -0.004996 cm, gR1 -0.002186 deg; this is too small to prove strong offset usage.
- control-only tc_test: zero-offset minus true-offset pRB 0.010828 cm, gR1 -0.005676 deg; this is too small to prove strong offset usage.
- acc-aux dip_test: zero-offset minus true-offset pRB -0.000059 cm, gR1 0.001826 deg; this is too small to prove strong offset usage.
- acc-aux tc_test: zero-offset minus true-offset pRB 0.004996 cm, gR1 -0.003834 deg; this is too small to prove strong offset usage.
- Verdict: the current offset-aware NewPL variants do not yet demonstrate meaningful dependence on `r_JS`; the IMU acceleration auxiliary option remains an ablation, not a selected change.

### NewIK1 v11

- dip_test: `newik1_v11_best` pRJ delta vs official/v10 = -0.007430 cm / -0.007708 cm; leaf pRJ delta vs official = +0.007319 cm; gR2 delta vs official = +0.206024 deg.
- tc_test: `newik1_v11_best` pRJ delta vs official/v10 = -0.026921 cm / -0.052565 cm; leaf pRJ delta vs official = -0.013661 cm; gR2 delta vs official = +0.160934 deg.
- Verdict: v11 improves pRJ on DIP test and TotalCapture test, and improves TotalCapture leaf pRJ, but it regresses gR2 versus the official IK1 baseline and has mixed leaf behavior on DIP. Keep it as diagnostic evidence; do not connect it to the full pipeline until gR2 is fixed or a downstream run justifies the tradeoff.

### Protocol Notes

- No full-pipeline 11 metrics were run for this goal.
- DIP trans/root velocity/global trajectory GT was not used.
- `newpl_v4_init36 baseline` rows are marked `not available` in this specific canonical-control summary because the checkpoint path was not provided to the runner; historical v4 results remain recorded in earlier sections.
<!-- END newpl-offset-v6-and-newik1-v11-control-only-training--2026-06-09 -->

<!-- BEGIN newpl-v6-next-control-one-step-predictive-pl--2026-06-11 -->
## NewPL next-control / one-step predictive PL experiment (2026-06-11)

Status: completed. The selected full run is `full_fastval1`, with AMASS pretrain `80` epochs, DIP fine-tune `40` epochs, and module-level evaluation on AMASS, DIP test, and TotalCapture test. Full-pipeline 11 metrics were not run.

Implemented files:

```text
pl_curve.py
pl_next_control_cache.py
pl_next_control_train.py
pl_next_control_eval.py
scripts/run_newpl_v6_next_control_20260611.sh
```

Module contract:

```text
variant: newpl_v6_next_control
input: aRB[18] + wRB[18] + RRB[45] + gR0[3] = 84D
init: offset_r[18] + pRL[15] + gR0[3] = 36D
current output: pRB_t[15] + gR1_t[3] = 18D, unchanged for IK1
aux output: next_pl = pRB_{t+1}[15] + gR1_{t+1}[3]
aux dynamics: next_pldot / next_plddot from the corrected one-step preview UniformCubicBSpline decode
```

Control-point time semantics:

```text
Current PLCurve appends a tail control point each frame.
Current pl_t is decoded from UniformCubicBSpline(control_buffer + ghost(last)) at index [-2].
Therefore new_control_t is a tail control point, not the direct frame output.
The verified current-frame PLCurve also adjusts the last up-to-four existing controls before appending the current new control.
Corrected newpl_v6 predicts exactly one extra next tail control point from hidden_t.
It also adjusts the last up-to-four preview controls, then decodes adjusted_tail + next_control + ghost(next_control) at [-2].
The preview is not appended to the live stream.
```

Cache schema:

```text
type: pl_next_control_cache_v2
source: existing PL cache + data/dataset_work/GTControlCache/*/gt_control_cache_manifest.json
fields:
  pl_input [T,84]
  pl_init_feature [36]
  pl_target [T,18]
  pl_target_next [T,18]
  valid_next_mask [T]
  pl_target_control [T,18]
  pl_target_control_next [T,18]
  tail_control_target [T,4,18]
  tail_control_valid_mask [T,4]
  last_control_target [T,18]
  gt_pldot_next [T,18]
  gt_plddot_next [T,18]
  pl_base [T,18]
  pl_base_next [T,18]
  baseline_fd_vel [T,18]
  baseline_fd_acc [T,18]
```

Loss:

```text
current:
  pRB=1.0, gR1=1.0
  gt_control_pRB=0.3, gt_control_gR1=0.1
  pRB_dot=0.03, gR1_dot=0.03, gR1_ddot=0.001, pRB_ddot_smooth=1e-6
next:
  next_pRB=1.0, next_gR1=1.0
  next_gt_control_pRB=0.3, next_gt_control_gR1=0.1
  next_pRB_vel=0.03, next_pRB_acc=0.0003
  next_gR1_vel=0.03, next_gR1_acc=0.001
  next_control_delta_prior=0.01
  last_control_pRB=0.3, last_control_gR1=0.1
  next_tail4_control_pRB=0.15, next_tail4_control_gR1=0.05
```

Checkpoint selection:

```text
best_total_loss.pt
best_current_module_metric.pt: pRB_t_L2_cm + 0.1*gR1_t_angle_deg
best_next_module_metric.pt: pRB_t+1_L2_cm + 0.1*gR1_t+1_angle_deg
best_dynamics_metric.pt: pRB_vel_L2_cm_s + 0.01*pRB_acc_L2_cm_s2
best_control_metric.pt: last/next/tail4 control pRB L2 plus small gR1-angle term
last.pt
```

Commands:

```text
smoke:
/home/lingfeng/bin/longrun -- bash scripts/run_newpl_v6_next_control_20260611.sh smoke

full:
CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=512 WINDOW=81 MAX_TRAIN_VAL_SEQS=128 VAL_BATCH_SIZE=64 RUN_SUFFIX=fastval1 /home/lingfeng/bin/longrun -- bash scripts/run_newpl_v6_next_control_20260611.sh full
```

Smoke artifacts:

```text
root: data/experiments/newpl_v6_next_control_tail4_20260611
AMASS checkpoint dir: data/experiments/newpl_v6_next_control_tail4_20260611/smoke/amass_pretrain
DIP checkpoint dir: data/experiments/newpl_v6_next_control_tail4_20260611/smoke/dip_finetune
AMASS eval: data/experiments/newpl_v6_next_control_tail4_20260611/smoke/eval_amass_after_pretrain.json
DIP eval: data/experiments/newpl_v6_next_control_tail4_20260611/smoke/eval_dip_test_after_dip_finetune.json
TC eval after AMASS: data/experiments/newpl_v6_next_control_tail4_20260611/smoke/eval_totalcapture_test_after_amass_pretrain.json
TC eval after DIP: data/experiments/newpl_v6_next_control_tail4_20260611/smoke/eval_totalcapture_test_after_dip_finetune.json
full output root: data/experiments/newpl_v6_next_control_tail4_20260611/full
selected full output root: data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1
full AMASS eval: data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_amass_after_pretrain.json
full TC after AMASS eval: data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_totalcapture_test_after_amass_pretrain.json
full DIP eval: data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_dip_test_after_dip_finetune.json
full TC after DIP eval: data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_totalcapture_test_after_dip_finetune.json
```

Smoke training:

```text
AMASS smoke: train/val sequences=4/4, epoch=1, best_total=0.01991503080353141, best_current=4.308073937892914, best_next=4.436356425285339, best_dynamics=45.75109577178955, best_control=13.197566032409668.
DIP smoke fine-tune: train/val sequences=4/4, epoch=1, best_total=0.020547525607980788, best_current=2.6297619938850403, best_next=2.834542155265808, best_dynamics=44.51901865005493, best_control=8.212532877922058.
TotalCapture train/val was not used.
```

Current-frame smoke metrics:

```text
AMASS: official pRB_L2=3.9585 cm gR1=4.5255 deg; v4 pRB_L2=4.3979 cm gR1=4.6766 deg; v6_amass pRB_L2=3.9703 cm gR1=4.5235 deg.
DIP test: official pRB_L2=6.1285 cm gR1=10.1790 deg; v4 pRB_L2=6.2891 cm gR1=10.1212 deg; v6_dip pRB_L2=6.1153 cm gR1=10.1778 deg.
TC test after DIP: official pRB_L2=6.9955 cm gR1=13.4504 deg; v4 pRB_L2=6.6544 cm gR1=13.3295 deg; v6_dip pRB_L2=6.9956 cm gR1=13.4468 deg.
```

Next-frame and dynamics smoke metrics:

```text
AMASS next pRB_L2: official=4.1225 cm, v4=4.5802 cm, v6_amass=4.1196 cm.
DIP next pRB_L2: official=6.1329 cm, v4=6.3227 cm, v6_dip=6.1325 cm.
TC next pRB_L2 after DIP: official=7.0954 cm, v4=6.7852 cm, v6_dip=7.0963 cm.
Control-point smoke: AMASS v6 current/next/tail4 pRB L2=3.9702/4.1305/3.9646 cm; DIP v6=6.1291/6.1335/6.1274 cm; TC after DIP v6=7.0016/7.0992/7.0016 cm.
```

Full training checkpoints:

```text
AMASS full_fastval1:
  best_total_loss.pt epoch=80
  best_current_module_metric.pt epoch=70
  best_next_module_metric.pt epoch=26
  best_dynamics_metric.pt epoch=59
  best_control_metric.pt epoch=70
DIP full_fastval1:
  best_total_loss.pt epoch=40
  best_current_module_metric.pt epoch=40
  best_next_module_metric.pt epoch=39
  best_dynamics_metric.pt epoch=39
  best_control_metric.pt epoch=39
```

Full current-frame metrics:

```text
AMASS:
official PL baseline: pRB L2=2.8275 cm, gR1=7.2199 deg.
newpl_v4_init36 baseline: pRB L2=2.8989 cm, gR1=7.1207 deg.
newpl_v6_next_control_amass: pRB L2=2.8093 cm, gR1=7.2515 deg.
newpl_v6_next_control_amass_control: pRB L2=2.8226 cm, gR1=7.1798 deg.

DIP test:
official PL baseline: pRB L2=6.4195 cm, gR1=12.9477 deg.
newpl_v4_init36 baseline: pRB L2=6.4414 cm, gR1=12.7652 deg.
newpl_v6_next_control_amass: pRB L2=6.4806 cm, gR1=12.7494 deg.
newpl_v6_next_control_dip: pRB L2=6.4688 cm, gR1=12.6560 deg.

TotalCapture test:
official PL baseline: pRB L2=6.9955 cm, gR1=13.4504 deg.
newpl_v4_init36 baseline: pRB L2=6.6544 cm, gR1=13.3295 deg.
newpl_v6_next_control_amass: pRB L2=6.8749 cm, gR1=13.3279 deg.
newpl_v6_next_control_dip: pRB L2=6.9808 cm, gR1=13.1385 deg.
```

Full next-frame and dynamics metrics:

```text
AMASS:
official PL baseline: next pRB L2=2.9476 cm, vel L2=12.1129 cm/s, acc L2=871.7360 cm/s^2.
newpl_v4_init36 baseline: next pRB L2=3.0444 cm, vel L2=11.9329 cm/s, acc L2=726.7491 cm/s^2.
newpl_v6_next_control_amass: next pRB L2=2.9236 cm, vel L2=32.2878 cm/s, acc L2=489.2382 cm/s^2.

DIP test:
official PL baseline: next pRB L2=6.5600 cm, vel L2=40.5799 cm/s, acc L2=2729.2442 cm/s^2.
newpl_v4_init36 baseline: next pRB L2=6.6091 cm, vel L2=40.6118 cm/s, acc L2=2702.1118 cm/s^2.
newpl_v6_next_control_dip: next pRB L2=6.5954 cm, vel L2=66.3422 cm/s, acc L2=2658.9849 cm/s^2.

TotalCapture test:
official PL baseline: next pRB L2=7.0954 cm, vel L2=34.3235 cm/s, acc L2=2173.8883 cm/s^2.
newpl_v4_init36 baseline: next pRB L2=6.7852 cm, vel L2=32.9648 cm/s, acc L2=1861.3888 cm/s^2.
newpl_v6_next_control_amass: next pRB L2=6.9776 cm, vel L2=57.5801 cm/s, acc L2=706.3869 cm/s^2.
newpl_v6_next_control_dip: next pRB L2=7.0852 cm, vel L2=57.5980 cm/s, acc L2=707.7693 cm/s^2.
```

Evaluation contract:

```text
current comparison: module current output vs GT pRB_t/gR1_t
next comparison: baselines use causal persistence output_t -> GT_{t+1}; v6 uses direct next_pl_t -> GT_{t+1}
dynamics comparison: baselines use finite differences of current PL output; v6 uses predicted next-control spline derivatives
full-pipeline 11 metrics: not run
DIP trans/root velocity GT: not used
TotalCapture fine-tune: not run; TC is eval-only for official-route fairness
```

Conclusion:

```text
The full run is completed.
newpl_v6_next_control improves AMASS pRB/next-pRB slightly and improves spline acceleration strongly.
It improves gR1 on DIP and TotalCapture, especially after DIP fine-tune.
It does not beat fixed official/newpl_v4 baselines on DIP current pRB, and it is clearly worse than newpl_v4 on TotalCapture current pRB.
DIP fine-tune helps DIP pRB/gR1 slightly but hurts TotalCapture pRB.
Do not select this module and do not connect it to IK1/full pipeline.
```
<!-- END newpl-v6-next-control-one-step-predictive-pl--2026-06-11 -->

<!-- BEGIN newpl-v6-next-control-smoothacc-gR1--2026-06-13 -->
## EXP-20260613-newpl_v6_next_control_smoothacc_gR1 — Smoothacc + next-control gR1 search

Status: completed. This is a module-level PL-s1 diagnostic experiment only; full-pipeline 11 metrics were not run.

Question: can centered smoothed acceleration plus corrected v6 next-control and gR1-specific checkpoint selection produce the most accurate `gR1` without breaking the `pRB[15]+gR1[3]` PL output contract?

Implementation:

```text
changed:
  pl_next_control_train.py
  pl_next_control_eval.py
added:
  scripts/run_newpl_v6_next_control_smoothacc_gR1_20260613.sh
  scripts/summarize_newpl_v6_next_control_smoothacc_gR1.py
```

Input/output contract:

```text
PL input: smooth aRB[18] + raw wRB[18] + raw RRB[45] + gR0[3] = 84D
init: offset_r[18] + pRL[15] + gR0[3] = 36D
current output consumed by IK1: pRB_t[15] + gR1_t[3] = 18D
aux output: next_pl, next_pldot, next_plddot, next_control, preview tail4 controls
DIP trans/root velocity: not used
TotalCapture fine-tune: not run
full-pipeline 11 metrics: not run
```

Loss and checkpoint selection:

```text
Loss is the corrected v6 next-control family:
  current pRB/gR1, current GT control, temporal pRB/gR1 terms,
  next pRB/gR1, next GT control, next velocity/acceleration,
  next control delta prior, last-control loss, tail4-control loss.

New checkpoint outputs:
  best_current_gR1.pt
  best_next_gR1.pt
  best_gravity_control.pt

Selection values added to train logs:
  current_gR1_metric
  next_gR1_metric
  gravity_control_metric
```

Commands:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
EXP=/tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613 \
CACHE_ROOT=data/experiments/newpl_v5_smoothacc_20260612/caches \
NEXT_CACHE_ROOT=/tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/next_cache_full \
BATCH_SIZE=768 VAL_BATCH_SIZE=96 WINDOW=81 \
EPOCHS_AMASS=80 EPOCHS_DIP=40 MAX_VAL_SEQS=128 \
AMASS_MAX_EVAL_SEQS=20 \
/home/lingfeng/bin/longrun -- bash scripts/run_newpl_v6_next_control_smoothacc_gR1_20260613.sh full
```

Operational notes:

```text
Project quota was near the hard limit, so large checkpoints/caches were written under /tmp.
Training used BATCH_SIZE=768 and VAL_BATCH_SIZE=96; no batch=1 training was used.
Validation is sequence/window batched and not run every batch.
The smoothacc cache was reused from data/experiments/newpl_v5_smoothacc_20260612/caches.
The next-control cache was prebuilt under /tmp/.../next_cache_full.
```

Artifacts:

```text
root:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full
log:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/logs/run_full.log
AMASS checkpoints:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/amass_pretrain/best_current_gR1.pt
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/amass_pretrain/best_next_gR1.pt
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/amass_pretrain/best_gravity_control.pt
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/amass_pretrain/last.pt
DIP checkpoints:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/dip_finetune/best_current_gR1.pt
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/dip_finetune/best_next_gR1.pt
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/dip_finetune/best_gravity_control.pt
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/dip_finetune/last.pt
summary:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/summary.json
eval JSONs:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_amass_after_pretrain.json
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_dip_test_after_amass_pretrain.json
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_totalcapture_test_after_amass_pretrain.json
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_dip_test_after_amass_pretrain_fast512.json
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_totalcapture_test_after_amass_pretrain_fast512.json
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_dip_test_after_dip_finetune.json
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_totalcapture_test_after_dip_finetune.json
```

Training result:

```text
AMASS: 1294 train sequences, 128 val sequences, 80 epochs, batch=768, val_batch=96.
AMASS best_current_gR1.pt: epoch 80, val current_gR1_metric=13.504858 deg.
AMASS best_current_module_metric.pt: epoch 71, val score=5.255308.
DIP: 36 train sequences, 6 val sequences, 40 epochs, batch=768, val_batch=96.
DIP initialized from AMASS best_current_gR1.pt.
DIP best_current_gR1.pt: epoch 40, val current_gR1_metric=19.409483 deg.
DIP best_current_module_metric.pt: epoch 40, val score=5.407512.
```

Evaluation contract:

```text
AMASS after-pretrain: full eval for this run.
DIP/TC after-AMASS full-sequence JSONs exist.
For before/after DIP claims, use fast512 JSONs so the evaluation window is identical.
DIP/TC after-DIP JSONs use max_frames_per_sequence=512.
No full-pipeline 11 metrics were run.
```

Key metrics:

```text
AMASS after AMASS:
official_PL_smoothacc: pRB L1=1.933659 cm, pRB L2=4.030455 cm, gR1=4.838765 deg.
newpl_v4_init36_smoothacc: pRB L1=2.008328 cm, pRB L2=4.211169 cm, gR1=4.876139 deg.
newpl_v6_smoothacc_amass_current_gR1: pRB L1=1.939113 cm, pRB L2=4.021821 cm, gR1=5.224865 deg.

DIP fast512 before DIP fine-tune:
official_PL_smoothacc: pRB L2=4.241512 cm, gR1=8.879523 deg.
newpl_v4_init36_smoothacc: pRB L2=4.221608 cm, gR1=8.741283 deg.
newpl_v6_smoothacc_amass_current_gR1: pRB L2=4.262976 cm, gR1=8.792514 deg.

TotalCapture fast512 before DIP fine-tune:
official_PL_smoothacc: pRB L2=7.566914 cm, gR1=9.873214 deg.
newpl_v4_init36_smoothacc: pRB L2=7.160630 cm, gR1=9.745194 deg.
newpl_v6_smoothacc_amass_balanced: pRB L2=7.470823 cm, gR1=9.616730 deg.

DIP fast512 after DIP fine-tune:
official_PL_smoothacc: pRB L2=4.241512 cm, gR1=8.879523 deg.
newpl_v4_init36_smoothacc: pRB L2=4.221608 cm, gR1=8.741283 deg.
newpl_v5_raw_dip_on_smoothinput: pRB L2=4.190160 cm, gR1=8.671933 deg.
newpl_v6_smoothacc_dip_current_gR1: pRB L2=4.226809 cm, gR1=8.719222 deg.

TotalCapture fast512 after DIP fine-tune:
official_PL_smoothacc: pRB L2=7.566914 cm, gR1=9.873214 deg.
newpl_v4_init36_smoothacc: pRB L2=7.160630 cm, gR1=9.745194 deg.
newpl_v5_raw_dip_on_smoothinput: pRB L2=7.318521 cm, gR1=9.890395 deg.
newpl_v6_smoothacc_dip_current_gR1: pRB L2=7.562171 cm, gR1=9.470963 deg.
```

Per-leaf and smoothness after DIP fine-tune, fast512:

```text
DIP official per-leaf pRB L2=4.570761/5.951653/3.342375/3.540965/3.801805 cm; pRB jitter=0.153908; gR1 jitter=0.001165.
DIP v4 per-leaf pRB L2=4.632858/5.952460/3.179092/3.580340/3.763289 cm; pRB jitter=0.143387; gR1 jitter=0.001071.
DIP v6 smoothacc gR1 per-leaf pRB L2=4.696512/6.002673/3.343599/3.426246/3.665018 cm; pRB jitter=0.143305; gR1 jitter=0.001027.
TC official per-leaf pRB L2=7.971572/7.320425/6.911801/7.418833/8.211942 cm; pRB jitter=0.721941; gR1 jitter=0.009477.
TC v4 per-leaf pRB L2=7.831799/7.416688/6.194060/6.959013/7.401588 cm; pRB jitter=0.622985; gR1 jitter=0.008366.
TC v6 smoothacc gR1 per-leaf pRB L2=7.992774/7.749657/6.972277/7.565431/7.530715 cm; pRB jitter=0.622620; gR1 jitter=0.008079.
```

Conclusion:

```text
Not selected.
The gR1-specific checkpoints help gravity on DIP and TotalCapture fast512 after DIP fine-tune.
On TotalCapture fast512, v6 gR1=9.470963 deg, better than official 9.873214 and v4 9.745194.
On DIP fast512, v6 gR1=8.719222 deg, better than official 8.879523 and v4 8.741283, but worse than raw-v5-on-smoothinput 8.671933.
The pRB output is not robust: v6 pRB L2=4.226809 on DIP and 7.562171 on TC, both worse than v4 and raw-v5-on-smoothinput.
Do not connect this checkpoint to IK1/full pipeline. Keep the gR1 checkpoint-selection mechanism for future variants that preserve pRB.
```
<!-- END newpl-v6-next-control-smoothacc-gR1--2026-06-13 -->

<!-- BEGIN newpl-v7-learned-offset-accaux--2026-06-12 -->
## NewPL v7 learned-offset acceleration auxiliary diagnostic (2026-06-12)

Status: implemented and smoke-tested. Final decision: `diagnostic only`.

### Goal

Remove external `offset_r` from NewPL initialization and test whether an internal bounded per-sensor learned offset can be supervised indirectly by an IMU acceleration explanation loss.

### Implemented files

```text
pl_curve.py
newpl_v7_learned_offset_accaux_smoke.py
PROJECT_STATUS.md
RECENT_REPLACEMENT_VERSIONS.md
EXPERIMENT_LOG.md
```

New symbols:

```text
PLCurveLearnedOffsetAccAuxModule
split_legacy_pl_imu_feature
bounded_offset_from_raw
raw_offset_from_bounded
root_frame_offsets_from_rrb
learned_offset_imu_acc_terms
learned_offset_imu_acc_loss
```

### Contract

```text
variant: newpl_v7_learned_offset_accaux
PL frame input: official 84D aRB[18] + wRB[18] + RRB[45] + gR0[3]
stream init: init18 = pRL[15] + gR0[3]
PL output: official 18D pRB[15] + gR1[3]
learned offset: offset = offset_max * tanh(raw_offset), raw_offset shape [6,3], offset_max=0.30 m
offset frame: r_BS, sensor origin relative to mapped body/sensor frame B, expressed in B
root-frame mapping: r_RB = r_BS @ R_RB using R_RB from the 84D PL input, root sensor uses identity
acc proxy: pRB_ddot + alpha_RB x r_RB + omega_RB x (omega_RB x r_RB)
observed target: aRB_leaf - aRB_root
forbidden supervision: DIP trans, DIP root velocity, real offset GT for DIP/TotalCapture
full-pipeline S4/S5/11 metrics: not measured
```

The acceleration residual is explicitly a root-frame non-inertial diagnostic proxy. It is not global acceleration supervision and does not use global translation.

### Commands run

```bash
python -m py_compile pl_curve.py newpl_v7_learned_offset_accaux_smoke.py

python newpl_v7_learned_offset_accaux_smoke.py \
  --output-dir data/experiments/newpl_v7_learned_offset_accaux_20260612 \
  --dataset-label 'AMASS smoke' \
  --max-sequences 2 \
  --max-train-sequences 2 \
  --batch-size 2 \
  --window 31 \
  --stage0-steps 12 \
  --stage1-steps 12 \
  --stage2-epochs 1 \
  --offset-lr 0.001
```

Initial bug found during smoke: offset prior used `sqrt(mean(offset^2))`, which has an undefined gradient at zero initialization and produced NaN at Stage 0 step 2. Fixed to `mean(offset^2)` and added raw-offset gradient clipping for Stage 0/1.

### Artifacts

```text
root: data/experiments/newpl_v7_learned_offset_accaux_20260612
config: data/experiments/newpl_v7_learned_offset_accaux_20260612/config.json
command: data/experiments/newpl_v7_learned_offset_accaux_20260612/command.txt
stage0: data/experiments/newpl_v7_learned_offset_accaux_20260612/stage0_identifiability.json
stage1: data/experiments/newpl_v7_learned_offset_accaux_20260612/stage1_freeze_offset.json
stage2: data/experiments/newpl_v7_learned_offset_accaux_20260612/stage2_tiny_joint.json
summary json: data/experiments/newpl_v7_learned_offset_accaux_20260612/summary.json
summary md: data/experiments/newpl_v7_learned_offset_accaux_20260612/summary.md
checkpoints:
  data/experiments/newpl_v7_learned_offset_accaux_20260612/checkpoints/stage0_learned_offset.pt
  data/experiments/newpl_v7_learned_offset_accaux_20260612/checkpoints/stage1_freeze_offset.pt
  data/experiments/newpl_v7_learned_offset_accaux_20260612/checkpoints/stage2_tiny_joint.pt
```

### Offset diagnostic table

| zero offset residual | random offset residual | learned offset residual | learned vs zero improvement | random vs zero degradation | all finite |
|---:|---:|---:|---:|---:|---|
| `8.955655` | `11.304108` | `8.928542` | `0.027113` | `2.348453` | yes |

Stage 0 learned offset norm mean/median/p95: `0.005772 / 0.006184 / 0.006223 m`.

### Stage 1 freeze-offset-only

| Metric | Value |
|---|---:|
| residual before | `15.339169 m/s^2` |
| residual after | `15.346781 m/s^2` |
| improvement after-minus-before | `-0.007612 m/s^2` |
| pRB L2 delta | `-0.000186 cm` |
| gR1 angle delta | `+0.000350 deg` |
| offset norm mean/median/p95 | `0.009578 / 0.008723 / 0.012401 m` |

The PL output is effectively unchanged, as expected because learned offset is not part of the forward PL output. The offset-only residual did not improve.

### PL module smoke comparison

| Dataset/split | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | IMU acc residual ↓ | offset norm mean/median/p95 | Notes |
|---|---|---:|---:|---:|---:|---|---|
| AMASS smoke | official PL baseline | `2.402170` | `4.903610` | `6.664704` | `15.211345` | `0/0/0` | cached official PL, zero-offset residual |
| AMASS smoke | newpl_v4_init36 baseline | `2.566700` | `5.360255` | `6.979082` | `23.585226` | `0/0/0` | historical processed-input checkpoint on same smoke cache |
| AMASS smoke | newpl_v5_dip_best baseline | `2.511445` | `5.193558` | `6.801924` | `24.289366` | `0/0/0` | official-protocol checkpoint |
| AMASS smoke | newpl_v7_learned_offset_accaux | `2.509856` | `5.190952` | `6.799719` | `24.048222` | `0.009579/0.008723/0.012402` | learned offset, full-pipeline not measured |

Per-leaf pRB L2 cm for v7: `6.1512 / 5.5759 / 4.1469 / 7.2443 / 2.8366`.

### Conclusion

- Stage 0 shows weak identifiability: learned offset reduces residual by only `0.027113 m/s^2` versus zero, while a random offset degrades strongly.
- Stage 1 fails the useful offset-only criterion: frozen NewPL + learned offset makes residual slightly worse.
- Stage 2 is finite and v7 is marginally better than `newpl_v5_dip_best` on this tiny smoke (`pRB L2 -0.002605 cm`, `gR1 -0.002204 deg`, acc residual -0.241144), but v7 remains worse than official PL by `+0.287342 cm` pRB L2 and `+0.135015 deg` gR1.
- Learned offset norm is small and not saturated, so the failure is not caused by unrealistic offset magnitude.
- Do not promote. Do not run full AMASS -> DIP or connect to IK1/full pipeline until a broader same-cache diagnostic proves pRB/gR1 are not weaker than official and the acceleration residual reduction is meaningful.
<!-- END newpl-v7-learned-offset-accaux--2026-06-12 -->

## NewPL v5 loss-family ablation

### Orchestrator Task: gradient_audit_dip_from_v5_amass

Name: Gradient audit on DIP batch from v5 AMASS checkpoint

Status: completed

Type: audit

Start: 2026-06-12T00:37:44

End: 2026-06-12T00:38:44

GPU: 1

PID: 2413424

Return code: 0

Command: `/home/lingfeng/.conda/envs/globalpose-gpu/bin/python newpl_v5_loss_gradient_audit.py --cache data/experiments/newpl_v5_official_protocol_20260607/caches/pl_dip_train_official_init36/pl_curve_cache_manifest.json --gt-control-cache data/dataset_work/GTControlCache/dip_train/gt_control_cache_manifest.json --checkpoint data/experiments/newpl_v5_official_protocol_20260607_tuned/amass_pretrain/best_loss.pt --output-json data/experiments/newpl_v5_loss_family_ablation_20260611/gradient_audit/dip_from_v5_amass/result.json --batch-size 8 --window 61`

Log: `logs/orchestrator/newpl_v5_loss_family_ablation_20260611/gradient_audit_dip_from_v5_amass.log`

Outputs:

- `data/experiments/newpl_v5_loss_family_ablation_20260611/gradient_audit/dip_from_v5_amass/result.json`

Summary:

| metric | value |
|---|---:|
| batch_name | s01_01[0:61]|s01_02[997:1058]|s01_03[1994:2055]|s01_04[961:1022]|s01_05[3988:4049]|s02_01[4985:5046]|s02_02[5982:6043]|s02_03[52:113] |
| batch_size | 8 |
| cache | data/experiments/newpl_v5_official_protocol_20260607/caches/pl_dip_train_official_init36/pl_curve_cache_manifest.json |
| checkpoint | data/experiments/newpl_v5_official_protocol_20260607_tuned/amass_pretrain/best_loss.pt |
| gt_control_cache | data/dataset_work/GTControlCache/dip_train/gt_control_cache_manifest.json |
| manifest_type | pl_curve_cache_v2 |
| seed | 42 |
| status | ok |
| window | 61 |

### Orchestrator Task: variant_q_control

Name: NewPL v5 ablation q plus control

Status: completed

Type: train

Start: 2026-06-12T00:38:44

End: 2026-06-12T01:02:51

GPU: 1

PID: 2415063

Return code: 0

Command: `bash scripts/run_newpl_v5_loss_family_variant_20260611.sh q_control`

Log: `logs/orchestrator/newpl_v5_loss_family_ablation_20260611/variant_q_control.log`

Outputs:

- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control/done.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control/amass_pretrain/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control/dip_finetune/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control/after_dip_dip_test.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control/after_dip_tc_test.json`

Summary:

| metric | value |
|---|---:|
| amass_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control/after_amass_dip_test.json |
| amass_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control/after_amass_tc_test.json |
| dip_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control/after_dip_dip_test.json |
| dip_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control/after_dip_tc_test.json |
| root | data/experiments/newpl_v5_loss_family_ablation_20260611/q_control |
| status | ok |
| variant | q_control |

### Orchestrator Task: variant_q_control_qddot

Name: NewPL v5 ablation q plus control plus qddot

Status: completed

Type: train

Start: 2026-06-12T01:02:51

End: 2026-06-12T01:25:58

GPU: 1

PID: 2451076

Return code: 0

Command: `bash scripts/run_newpl_v5_loss_family_variant_20260611.sh q_control_qddot`

Log: `logs/orchestrator/newpl_v5_loss_family_ablation_20260611/variant_q_control_qddot.log`

Outputs:

- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qddot/done.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qddot/amass_pretrain/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qddot/dip_finetune/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qddot/after_dip_dip_test.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qddot/after_dip_tc_test.json`

Summary:

| metric | value |
|---|---:|
| amass_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qddot/after_amass_dip_test.json |
| amass_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qddot/after_amass_tc_test.json |
| dip_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qddot/after_dip_dip_test.json |
| dip_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qddot/after_dip_tc_test.json |
| root | data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qddot |
| status | ok |
| variant | q_control_qddot |

### Orchestrator Task: variant_q_control_qdot

Name: NewPL v5 ablation q plus control plus qdot

Status: completed

Type: train

Start: 2026-06-12T01:25:58

End: 2026-06-12T01:50:06

GPU: 1

PID: 2497589

Return code: 0

Command: `bash scripts/run_newpl_v5_loss_family_variant_20260611.sh q_control_qdot`

Log: `logs/orchestrator/newpl_v5_loss_family_ablation_20260611/variant_q_control_qdot.log`

Outputs:

- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qdot/done.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qdot/amass_pretrain/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qdot/dip_finetune/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot/after_dip_dip_test.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot/after_dip_tc_test.json`

Summary:

| metric | value |
|---|---:|
| amass_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot/after_amass_dip_test.json |
| amass_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot/after_amass_tc_test.json |
| dip_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot/after_dip_dip_test.json |
| dip_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot/after_dip_tc_test.json |
| root | data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qdot |
| status | ok |
| variant | q_control_qdot |

### Orchestrator Task: variant_q_control_qdot_qddot

Name: NewPL v5 ablation q plus control plus qdot plus qddot

Status: completed

Type: train

Start: 2026-06-12T02:07:11

End: 2026-06-12T02:32:19

GPU: 1

PID: 2569689

Return code: 0

Command: `bash scripts/run_newpl_v5_loss_family_variant_20260611.sh q_control_qdot_qddot`

Log: `logs/orchestrator/newpl_v5_loss_family_ablation_20260611/variant_q_control_qdot_qddot.log`

Outputs:

- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qdot_qddot/done.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qdot_qddot/amass_pretrain/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qdot_qddot/dip_finetune/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot_qddot/after_dip_dip_test.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot_qddot/after_dip_tc_test.json`

Summary:

| metric | value |
|---|---:|
| amass_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot_qddot/after_amass_dip_test.json |
| amass_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot_qddot/after_amass_tc_test.json |
| dip_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot_qddot/after_dip_dip_test.json |
| dip_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_control_qdot_qddot/after_dip_tc_test.json |
| root | data/experiments/newpl_v5_loss_family_ablation_20260611/q_control_qdot_qddot |
| status | ok |
| variant | q_control_qdot_qddot |

### Orchestrator Task: variant_q_only

Name: NewPL v5 ablation q only

Status: completed

Type: train

Start: 2026-06-12T02:32:19

End: 2026-06-12T02:55:26

GPU: 1

PID: 2609696

Return code: 0

Command: `bash scripts/run_newpl_v5_loss_family_variant_20260611.sh q_only`

Log: `logs/orchestrator/newpl_v5_loss_family_ablation_20260611/variant_q_only.log`

Outputs:

- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_only/done.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_only/amass_pretrain/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_only/dip_finetune/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_only/after_dip_dip_test.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_only/after_dip_tc_test.json`

Summary:

| metric | value |
|---|---:|
| amass_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_only/after_amass_dip_test.json |
| amass_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_only/after_amass_tc_test.json |
| dip_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_only/after_dip_dip_test.json |
| dip_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_only/after_dip_tc_test.json |
| root | data/experiments/newpl_v5_loss_family_ablation_20260611/q_only |
| status | ok |
| variant | q_only |

### Orchestrator Task: variant_q_qddot

Name: NewPL v5 ablation q plus qddot

Status: completed

Type: train

Start: 2026-06-12T02:55:26

End: 2026-06-12T03:18:32

GPU: 1

PID: 2626819

Return code: 0

Command: `bash scripts/run_newpl_v5_loss_family_variant_20260611.sh q_qddot`

Log: `logs/orchestrator/newpl_v5_loss_family_ablation_20260611/variant_q_qddot.log`

Outputs:

- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_qddot/done.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_qddot/amass_pretrain/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_qddot/dip_finetune/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qddot/after_dip_dip_test.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qddot/after_dip_tc_test.json`

Summary:

| metric | value |
|---|---:|
| amass_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qddot/after_amass_dip_test.json |
| amass_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qddot/after_amass_tc_test.json |
| dip_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qddot/after_dip_dip_test.json |
| dip_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qddot/after_dip_tc_test.json |
| root | data/experiments/newpl_v5_loss_family_ablation_20260611/q_qddot |
| status | ok |
| variant | q_qddot |

### Orchestrator Task: variant_q_qdot

Name: NewPL v5 ablation q plus qdot

Status: completed

Type: train

Start: 2026-06-12T03:18:33

End: 2026-06-12T03:41:39

GPU: 1

PID: 2644436

Return code: 0

Command: `bash scripts/run_newpl_v5_loss_family_variant_20260611.sh q_qdot`

Log: `logs/orchestrator/newpl_v5_loss_family_ablation_20260611/variant_q_qdot.log`

Outputs:

- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_qdot/done.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_qdot/amass_pretrain/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_qdot/dip_finetune/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot/after_dip_dip_test.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot/after_dip_tc_test.json`

Summary:

| metric | value |
|---|---:|
| amass_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot/after_amass_dip_test.json |
| amass_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot/after_amass_tc_test.json |
| dip_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot/after_dip_dip_test.json |
| dip_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot/after_dip_tc_test.json |
| root | data/experiments/newpl_v5_loss_family_ablation_20260611/q_qdot |
| status | ok |
| variant | q_qdot |

### Orchestrator Task: variant_q_qdot_qddot

Name: NewPL v5 ablation q plus qdot plus qddot

Status: completed

Type: train

Start: 2026-06-12T03:41:39

End: 2026-06-12T04:04:46

GPU: 1

PID: 2661393

Return code: 0

Command: `bash scripts/run_newpl_v5_loss_family_variant_20260611.sh q_qdot_qddot`

Log: `logs/orchestrator/newpl_v5_loss_family_ablation_20260611/variant_q_qdot_qddot.log`

Outputs:

- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_qdot_qddot/done.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_qdot_qddot/amass_pretrain/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/q_qdot_qddot/dip_finetune/train_result.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot_qddot/after_dip_dip_test.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot_qddot/after_dip_tc_test.json`

Summary:

| metric | value |
|---|---:|
| amass_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot_qddot/after_amass_dip_test.json |
| amass_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot_qddot/after_amass_tc_test.json |
| dip_eval_dip | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot_qddot/after_dip_dip_test.json |
| dip_eval_tc | data/experiments/newpl_v5_loss_family_ablation_20260611/eval/q_qdot_qddot/after_dip_tc_test.json |
| root | data/experiments/newpl_v5_loss_family_ablation_20260611/q_qdot_qddot |
| status | ok |
| variant | q_qdot_qddot |

### Orchestrator Task: summarize_newpl_v5_loss_ablation

Name: Summarize NewPL v5 loss-family ablation

Status: completed

Type: parse

Start: 2026-06-12T04:04:46

End: 2026-06-12T04:05:46

GPU: CPU

PID: 2678442

Return code: 0

Command: `/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/summarize_newpl_v5_loss_ablation.py --root data/experiments/newpl_v5_loss_family_ablation_20260611 --output-json data/experiments/newpl_v5_loss_family_ablation_20260611/summary.json --output-csv data/experiments/newpl_v5_loss_family_ablation_20260611/summary_eval_rows.csv`

Log: `logs/orchestrator/newpl_v5_loss_family_ablation_20260611/summarize.log`

Outputs:

- `data/experiments/newpl_v5_loss_family_ablation_20260611/summary.json`
- `data/experiments/newpl_v5_loss_family_ablation_20260611/summary_eval_rows.csv`

Summary:

| metric | value |
|---|---:|
| question | Effect of q/control/qdot/qddot loss families on NewPL v5 training and generalization. |
| root | data/experiments/newpl_v5_loss_family_ablation_20260611 |
| status | ok |

## NewPL v7b local acceleration learned-offset diagnostic (2026-06-12)

Goal: revise `newpl_v7_learned_offset_accaux` so learned IMU position offset is supervised by a more physical local acceleration proxy. The v7 root-relative loss subtracted root acceleration; v7b instead uses root IMU gyro and finite-difference angular acceleration only to correct `pRB/pRBdot/pRBddot` for the rotating root frame.

Implemented files:

```text
pl_curve.py
newpl_v7b_local_accaux_smoke.py
PROJECT_STATUS.md
RECENT_REPLACEMENT_VERSIONS.md
EXPERIMENT_LOG.md
```

Module contract:

```text
variant: newpl_v7b_local_accaux
class: PLCurveLearnedLeafOffsetLocalAccAuxModule
input: official PL 84D aRB[18] + wRB[18] + RRB[45] + gR0[3]
init: init18 pRL[15] + gR0[3]
output: official PL 18D pRB[15] + gR1[3]
learned parameter: raw_leaf_offset[5,3]
bounded offset: leaf_offset = offset_max * tanh(raw_leaf_offset), offset_max=0.30 m
root offset: not trained and not used by local acc loss
```

Acceleration auxiliary:

```text
omega_root = wRB[5]
alpha_root = d omega_root / dt
anchor_acc_R =
  pRB_ddot
+ 2 * omega_root x pRB_dot
+ alpha_root x pRB
+ omega_root x (omega_root x pRB)

r_leaf_R = r_leaf_B @ RRB_leaf
offset_acc_R =
  alpha_leaf x r_leaf_R
+ omega_leaf x (omega_leaf x r_leaf_R)

pred_acc_R = anchor_acc_R + offset_acc_R
obs_acc_R = aRB_leaf
L_local_acc = SmoothL1(pred_acc_R / acc_scale, obs_acc_R / acc_scale)
```

Loss recipe:

```text
L = L_PL_current
  + imu_acc_weight * L_local_acc
  + offset_prior_weight * mean(leaf_offset^2)

Smoke defaults:
imu_acc_weight=0.005
offset_prior_weight=0.001
acc_scale=30
gravity_mode=none
dt=1/60
```

Commands:

```bash
python -m py_compile pl_curve.py newpl_v7b_local_accaux_smoke.py

python newpl_v7b_local_accaux_smoke.py \
  --output-dir data/experiments/newpl_v7b_local_accaux_20260612 \
  --max-sequences 2 \
  --max-train-sequences 2 \
  --batch-size 2 \
  --window 31 \
  --stage0-steps 20 \
  --stage1-steps 20 \
  --stage2-epochs 1

python newpl_v7b_local_accaux_smoke.py \
  --output-dir data/experiments/newpl_v7b_local_accaux_20260612_minus_g \
  --max-sequences 2 \
  --max-train-sequences 2 \
  --batch-size 2 \
  --window 31 \
  --stage0-steps 10 \
  --stage1-steps 10 \
  --stage2-epochs 1 \
  --skip-stage2 \
  --gravity-mode minus_gR0

python newpl_v7b_local_accaux_smoke.py \
  --output-dir data/experiments/newpl_v7b_local_accaux_20260612_plus_g \
  --max-sequences 2 \
  --max-train-sequences 2 \
  --batch-size 2 \
  --window 31 \
  --stage0-steps 10 \
  --stage1-steps 10 \
  --stage2-epochs 1 \
  --skip-stage2 \
  --gravity-mode plus_gR0
```

Artifacts:

```text
main root: data/experiments/newpl_v7b_local_accaux_20260612
config: data/experiments/newpl_v7b_local_accaux_20260612/config.json
command: data/experiments/newpl_v7b_local_accaux_20260612/command.txt
stage0: data/experiments/newpl_v7b_local_accaux_20260612/stage0_local_acc_identifiability.json
stage1: data/experiments/newpl_v7b_local_accaux_20260612/stage1_freeze_offset.json
stage2: data/experiments/newpl_v7b_local_accaux_20260612/stage2_tiny_joint.json
summary json: data/experiments/newpl_v7b_local_accaux_20260612/summary.json
summary md: data/experiments/newpl_v7b_local_accaux_20260612/summary.md
checkpoints:
  data/experiments/newpl_v7b_local_accaux_20260612/checkpoints/stage0_learned_leaf_offset.pt
  data/experiments/newpl_v7b_local_accaux_20260612/checkpoints/stage1_freeze_leaf_offset.pt
  data/experiments/newpl_v7b_local_accaux_20260612/checkpoints/stage2_tiny_joint.pt
gravity checks:
  data/experiments/newpl_v7b_local_accaux_20260612_minus_g/summary.json
  data/experiments/newpl_v7b_local_accaux_20260612_plus_g/summary.json
```

Stage metrics:

| Stage | Metric | Value |
|---|---|---:|
| Stage 0 | zero-offset local acc residual | `8.974846 m/s^2` |
| Stage 0 | random-offset local acc residual | `11.928852 m/s^2` |
| Stage 0 | init36 GT-offset local acc residual | `9.755350 m/s^2` |
| Stage 0 | learned-offset local acc residual | `8.963714 m/s^2` |
| Stage 0 | learned improvement vs zero | `0.011132 m/s^2` |
| Stage 0 | random degradation vs zero | `2.954006 m/s^2` |
| Stage 1 | frozen NewPL residual before | `15.081704 m/s^2` |
| Stage 1 | frozen NewPL residual after | `14.994161 m/s^2` |
| Stage 1 | pRB/gR1 output drift | `+0.000251 cm / +0.000093 deg` |
| Stage 2 | learned leaf offset norm mean/median/p95 | `0.014525 / 0.011973 / 0.019684 m` |

Gravity-mode sensitivity:

| gravity_mode | zero | random | learned | GT offset | Stage1 improvement | Decision |
|---|---:|---:|---:|---:|---:|---|
| none | `8.974846` | `11.928852` | `8.963714` | `9.755350` | `0.087543` | diagnostic only |
| minus_gR0 | `13.902242` | `17.051386` | `13.878829` | `14.238059` | `0.018126` | diagnostic only |
| plus_gR0 | `13.978133` | `15.994410` | `13.964892` | `14.994678` | `0.037277` | diagnostic only |

PL module smoke comparison:

| Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | local acc residual ↓ | offset norm mean/median/p95 | Notes |
|---|---:|---:|---:|---:|---|---|
| official PL baseline | `2.402170` | `4.903610` | `6.664704` | `15.179327` | `0/0/0` | cached official PL |
| newpl_v4_init36 baseline | `2.566700` | `5.360255` | `6.979082` | `23.832706` | `0/0/0` | historical checkpoint |
| newpl_v5_dip_best baseline | `2.511445` | `5.193558` | `6.801924` | `24.416668` | `0/0/0` | official-protocol checkpoint |
| newpl_v7_rootrel_accaux | `2.509856` | `5.190952` | `6.799719` | `24.186819` | `0.009008/0.008723/0.011767` | previous root-relative accaux |
| newpl_v7b_local_accaux | `2.509855` | `5.190949` | `6.799719` | `24.196295` | `0.014525/0.011973/0.019684` | local accaux |

Conclusion:

- v7b implements the intended physical correction: root angular velocity and angular acceleration are used to correct root-frame `pRB` derivatives, while each leaf IMU uses its own angular velocity/angular acceleration for the lever-arm term.
- The current smoke is finite and the learned offset remains realistic, but the evidence is weak: learned residual reduction is only `0.011132 m/s^2`, and the init36 GT offset residual is worse than zero under this proxy.
- Same-cache pRB/gR1 remain worse than official PL. v7b is therefore `diagnostic only`.
- Full-pipeline S4/S5 and 11 metrics were not measured. DIP trans/root velocity/global trajectory GT was not used.
- Next step before long training: audit acceleration convention with FK/RBDL-derived sensor/anchor acceleration, because this proxy still does not validate that GT offset explains IMU acceleration better than zero.

### v7b AMASS long diagnostic follow-up

User requested a longer training diagnostic after the initial smoke.

Command:

```bash
/home/lingfeng/bin/longrun -- bash -lc 'CUDA_VISIBLE_DEVICES=0 python newpl_v7b_local_accaux_smoke.py --output-dir data/experiments/newpl_v7b_local_accaux_20260612_longtrain --max-sequences 512 --max-train-sequences 512 --batch-size 96 --window 61 --stage0-steps 300 --stage1-steps 300 --stage2-epochs 50 --imu-acc-weight 0.005 --offset-prior-weight 0.001 --joint-lr 1e-5 --offset-lr 1e-3 --gravity-mode none'
```

Run note:

```text
GPU: CUDA_VISIBLE_DEVICES=0 on local RTX 5090.
longrun notification email sent.
This is still module-level AMASS diagnostic training, not full AMASS -> DIP and not full-pipeline S4/S5.
```

Artifacts:

```text
root: data/experiments/newpl_v7b_local_accaux_20260612_longtrain
config: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/config.json
command: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/command.txt
stage0: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/stage0_local_acc_identifiability.json
stage1: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/stage1_freeze_offset.json
stage2: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/stage2_tiny_joint.json
summary json: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/summary.json
summary md: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/summary.md
checkpoint: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/checkpoints/stage2_tiny_joint.pt
```

Setup:

| setting | value |
|---|---:|
| max_sequences | `512` |
| max_train_sequences | `512` |
| batch_size | `96` |
| window | `61` |
| stage0_steps | `300` |
| stage1_steps | `300` |
| stage2_epochs | `50` |
| imu_acc_weight | `0.005` |
| offset_prior_weight | `0.001` |
| joint_lr | `1e-5` |
| gravity_mode | `none` |

Offset/acc metrics:

| Metric | Value |
|---|---:|
| Stage 0 zero local acc residual | `11.099609 m/s^2` |
| Stage 0 random local acc residual | `12.552015 m/s^2` |
| Stage 0 init36 GT-offset local acc residual | `11.629623 m/s^2` |
| Stage 0 learned local acc residual | `11.071591 m/s^2` |
| Stage 0 learned improvement vs zero | `0.028018 m/s^2` |
| Stage 1 frozen residual before -> after | `24.562115 -> 24.461527 m/s^2` |
| Stage 1 pRB/gR1 drift | `-0.000182 cm / -0.000075 deg` |
| Stage 2 offset norm mean/median/p95 | `0.073905 / 0.076368 / 0.092299 m` |
| Last epoch local acc L2 train average | `21.144252 m/s^2` |

Same-cache AMASS module comparison:

| Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | local acc residual ↓ | offset norm mean/median/p95 |
|---|---:|---:|---:|---:|---|
| official PL baseline | `1.613324` | `3.284492` | `10.588459` | `11.674987` | `0/0/0` |
| newpl_v4_init36 baseline | `1.620193` | `3.298557` | `10.437762` | `28.840921` | `0/0/0` |
| newpl_v5_dip_best baseline | `1.633508` | `3.328090` | `10.252261` | `25.495672` | `0/0/0` |
| newpl_v7_rootrel_accaux | `1.633343` | `3.327924` | `10.253457` | `25.443991` | `0.009008/0.008723/0.011767` |
| newpl_v7b_local_accaux long | `1.634196` | `3.330573` | `10.174911` | `24.781027` | `0.073905/0.076368/0.092299` |

Long diagnostic conclusion:

- Longer training makes the v7b local acceleration loss more useful than the tiny smoke: v7b improves local acceleration residual versus v5/v7 and has the best gR1 among the NewPL checkpoints in this AMASS same-cache table.
- pRB remains slightly worse than official PL, v4, and v5. The run's automatic decision says `continue to full AMASS->DIP`, but this should be read as permission for the next module-level experiment, not selection.
- Full-pipeline S4/S5 and DIP test are still not measured.

## NewPL v5 loss-family ablation final interpretation (2026-06-12)

User question: if NewPL v5 completely removes control-point loss, and separately toggles q, qdot, and qddot losses, what changes in training and evaluation?

Contract:

```text
root: data/experiments/newpl_v5_loss_family_ablation_20260611
task file: configs/newpl_v5_loss_family_ablation_20260611_tasks.json
summary json: data/experiments/newpl_v5_loss_family_ablation_20260611/summary.json
summary csv: data/experiments/newpl_v5_loss_family_ablation_20260611/summary_eval_rows.csv
gradient audit: newpl_v5_loss_gradient_audit.py
runner: scripts/run_newpl_v5_loss_family_variant_20260611.sh
```

In this experiment `q` means the NewPL decoded PL state `pRB[15]+gR1[3]`, not the full-body RBDL `q75`. All variants preserve the official PL 84D input, init36 stream initialization, and 18D PL output contract. All use AMASS pretrain -> DIP fine-tune -> DIP/TotalCapture module eval, with `pl_physical` checkpoint selection.

Best DIP-finetuned module metrics:

| Dataset | Variant | pRB L2 cm | gR1 angle deg | Notes |
|---|---|---:|---:|---|
| DIP test | `q_control_qddot` | `6.426853` | `12.707329` | best DIP pRB among variants |
| DIP test | `q_qdot_qddot` | `6.427028` | `12.698597` | almost identical pRB without control loss |
| DIP test | `q_qddot` | `6.430836` | `12.698030` | qddot helps DIP pRB |
| DIP test | `q_only` | `6.437678` | `12.689232` | removing control loss is not best on DIP |
| TotalCapture test | `q_only` | `6.753091` | `13.575686` | best TC pRB among variants |
| TotalCapture test | `q_qdot` | `6.754438` | `13.574678` | qdot is effectively neutral |
| TotalCapture test | `q_qddot` | `6.756564` | `13.582943` | qddot does not transfer to TC |
| TotalCapture test | `q_control_qddot` | `6.771322` | `13.586917` | DIP-best variant regresses on TC |

Comparison to reference PL module numbers:

| Dataset | Reference | pRB L2 cm | gR1 angle deg | Comparison |
|---|---|---:|---:|---|
| DIP test | official_PL | `6.419473` | `12.947709` | still best pRB |
| DIP test | newpl_v4_init36 | `6.441447` | `12.765167` | loss variants improve pRB slightly |
| DIP test | raw newpl_v5_dip_best | `6.445578` | `12.552613` | loss variants improve pRB but lose gR1 |
| TotalCapture test | official_PL | `6.995536` | `13.450465` | loss variants improve pRB but lose gR1 |
| TotalCapture test | newpl_v4_init36 | `6.654393` | `13.329531` | still best TC pRB/gR1 here |
| TotalCapture test | raw newpl_v5_dip_best | `6.780749` | `13.415189` | `q_only` improves pRB but loses gR1 |

Loss-family interpretation:

| Loss Family | Evidence | Conclusion |
|---|---|---|
| control point | `q_control` improves DIP pRB by only `0.003169 cm` vs `q_only`, but worsens TC pRB by `0.009945 cm`. | not a reliable generalization gain |
| qdot | `q_qdot` changes DIP pRB by `0.000278 cm` and TC pRB by `+0.001347 cm` vs `q_only`; gradient norms are tiny. | current weight/target is negligible |
| qddot | `q_qddot` improves DIP pRB by `0.006842 cm` vs `q_only`, but worsens TC pRB by `0.003473 cm`. | weak DIP regularizer, not robust |
| control+qddot | `q_control_qddot` is best DIP pRB, but TC pRB is `6.771322`. | over-specializes to DIP module metric |

Gradient audit:

| Stage | q/control cosine | qdot/qddot cosine | Main Readout |
|---|---:|---:|---|
| AMASS init | `0.846724` | `-0.862329` | q and control initially agree, qdot/qddot oppose |
| DIP from v5 AMASS | `0.227377` | `-0.739343` | control weakly aligns after pretrain; qdot/qddot conflict remains |

qddot gradient is dominated by the huge `pRB_ddot_smooth` component, so current qddot behavior is closer to smoothness regularization than direct physical acceleration supervision.

Decision: do not select any variant. Best DIP pRB still does not beat official_PL pRB, and best TotalCapture pRB still does not beat `newpl_v4_init36`; all variants have worse TotalCapture gR1 than official/v4/raw v5. For future work, control-point loss should be treated as a tunable prior rather than mandatory, qdot can be dropped at current scale, and qddot only deserves follow-up with a better acceleration target and same-cache full-pipeline validation.

## NewPL v5 realtime smooth+residual acceleration input (2026-06-12)

User request: use a real-time smoothing method for IMU acceleration and also feed the removed acceleration noise/residual into NewPL, then retrain and check whether it helps.

Implemented variant:

```text
name: newpl_v5_realtime_smooth_residual
input: aRB_smooth[18] + aRB_residual_raw_minus_smooth[18] + wRB[18] + RRB[45] + gR0[3] = 102D
output: pRB[15] + gR1[3] = 18D
init: existing init36 offset_r[18] + pRL[15] + gR0[3]
filter: causal_iir, cutoff_hz=20, fs=60, filter_order=2, lookahead_frames=0
aRB_residual: raw root-frame acceleration - causal-smoothed root-frame acceleration
```

Implementation files:

```text
pl_curve.py:
  PL_SMOOTH_RESIDUAL_INPUT_SIZE
  causal_iir_lowpass_sequence
  pl_smooth_residual_sequence_features
  split_pl_feature support for 102D feature layout
pl_curve_cache.py:
  --feature-mode smooth_residual
  --acc-filter-mode, --cutoff-hz, --filter-fs, --filter-order
  manifest smooth_residual_contract
pl_curve_train.py:
  partial checkpoint remap from legacy 84D input.weight to 102D smooth+residual input.weight
scripts/run_newpl_v5_realtime_residual_20260612.sh:
  cache -> AMASS pretrain -> eval -> DIP fine-tune -> eval -> summary
scripts/summarize_newpl_v5_realtime_residual.py:
  module-level comparison summary
```

Validation before full run:

```bash
python -m py_compile pl_curve.py pl_curve_cache.py pl_curve_train.py pl_curve_pl_accuracy_eval.py scripts/summarize_newpl_v5_realtime_residual.py
SMOKE=1 ROOT=data/experiments/newpl_v5_realtime_residual_20260612_smoke_test RUN_SMOKE=1 CUDA_VISIBLE_DEVICES=0 scripts/run_newpl_v5_realtime_residual_20260612.sh
```

Smoke result: finite; 102D caches generated; manifest records `aRB_smooth[18] + aRB_residual_raw_minus_smooth[18] + wRB[18] + RRB[45] + gR0[3]`, `causal_iir`, cutoff `20 Hz`, `lookahead_frames=0`. Partial checkpoint loading copied legacy raw acceleration weights into both smooth and residual channels with no skipped critical keys.

Full training command:

```bash
CUDA_VISIBLE_DEVICES=0 ROOT=data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20 RUN_SMOKE=0 /home/lingfeng/bin/longrun -- bash scripts/run_newpl_v5_realtime_residual_20260612.sh
```

Run setup:

```text
GPU: local GPU0, RTX 5090
AMASS cache: data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
DIP train/val/test caches: data/experiments/newpl_v5_official_protocol_20260607/caches/*
TotalCapture test cache: data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json
AMASS batch_size: 512
DIP batch_size: 64
window: 61
selection_metric: control_physical
TotalCapture fine-tune: not used
DIP trans/root velocity/global trajectory supervision: not used
full-pipeline 11 metrics: not measured
```

Loss terms used:

| Term | Weight |
|---|---:|
| gt_control_pRB | `0.3` |
| gt_control_gR1 | `0.1` |
| gR1_dot | `0.03` |
| gR1_ddot | `0.001` |
| pRB_ddot_smooth | `0.000001` |
| baseline_pRB / baseline_gR1 distill | `0.0 / 0.0` |
| IK distill | disabled |

Training outcome:

| Stage | Status | Best epoch | Best selection value | Checkpoint |
|---|---|---:|---:|---|
| AMASS pretrain | early_stopped | `51` | `0.001814578774065012` | `data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/amass_pretrain/best_loss.pt` |
| DIP fine-tune | ok | `40` | `0.05415532860752137` | `data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/dip_finetune/best_loss.pt` |

Artifacts:

```text
root: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20
log: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/logs/run.log
summary md: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/summary.md
summary json: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/summary.json
AMASS eval: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/eval/amass_after_amass_pretrain_realtime_residual.json
DIP test after AMASS eval: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/eval/dip_test_after_amass_pretrain_realtime_residual.json
TC test after AMASS eval: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/eval/tc_test_after_amass_pretrain_realtime_residual.json
DIP test after DIP FT eval: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/eval/dip_test_after_dip_finetune_realtime_residual.json
TC test after DIP FT eval: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/eval/tc_test_after_dip_finetune_realtime_residual.json
```

Module-level comparison:

| Dataset/stage | Version | pRB L2 cm ↓ | gR1 angle deg ↓ | Notes |
|---|---|---:|---:|---|
| AMASS proxy after AMASS | official PL | `3.608209` | `5.361644` | 20-seq proxy |
| AMASS proxy after AMASS | realtime smooth+residual | `3.698426` | `5.178191` | pRB worse by `0.090216`, gR1 better by `0.183453` |
| DIP test after AMASS | official PL | `6.528883` | `15.267228` | no DIP FT |
| DIP test after AMASS | raw newpl_v5_amass_best | `6.454484` | `12.551949` | historical reference from `newpl_v5_official_protocol_20260607_tuned`; not same-cache fairness row |
| DIP test after AMASS | realtime smooth+residual | `6.568949` | `14.552094` | beats same-cache official gR1; historical raw-v5 row is context only |
| TC test after AMASS | official PL | `6.768144` | `14.014337` | no TC FT |
| TC test after AMASS | raw newpl_v5_amass_best | `6.783332` | `13.415420` | historical reference from `newpl_v5_official_protocol_20260607_tuned`; not same-cache fairness row |
| TC test after AMASS | realtime smooth+residual | `6.649841` | `13.733241` | improves same-cache official pRB/gR1; historical raw-v5 row is context only |
| DIP test after DIP FT | official PL | `6.528883` | `15.267228` | baseline |
| DIP test after DIP FT | raw newpl_v5_dip_best | `6.445578` | `12.552613` | historical reference from `newpl_v5_official_protocol_20260607_tuned`; not same-cache fairness row |
| DIP test after DIP FT | realtime smooth+residual | `6.557889` | `14.557139` | versus same-cache official: pRB worse by `0.029006`, gR1 better by `0.710089`; historical raw-v5 row is context only |
| TC test after DIP FT | official PL | `6.768144` | `14.014337` | baseline |
| TC test after DIP FT | raw newpl_v5_dip_best | `6.780749` | `13.415189` | historical reference from `newpl_v5_official_protocol_20260607_tuned`; not same-cache fairness row |
| TC test after DIP FT | realtime smooth+residual | `6.638172` | `13.736756` | versus same-cache official: pRB improves by `0.129972`, gR1 improves by `0.277580`; historical raw-v5 row is context only |

Per-leaf pRB L2 after DIP fine-tune:

| Dataset | Leaf | official PL cm ↓ | realtime smooth+residual cm ↓ | Delta new-official |
|---|---|---:|---:|---:|
| DIP test | L_LowArm | `8.733906` | `8.731975` | `-0.001931` |
| DIP test | R_LowArm | `8.825057` | `8.926571` | `+0.101514` |
| DIP test | L_LowLeg | `4.272925` | `4.252717` | `-0.020208` |
| DIP test | R_LowLeg | `5.603253` | `5.649345` | `+0.046092` |
| DIP test | Head | `5.209275` | `5.228836` | `+0.019561` |
| TC test | L_LowArm | `7.052147` | `7.113341` | `+0.061194` |
| TC test | R_LowArm | `6.368523` | `6.532339` | `+0.163816` |
| TC test | L_LowLeg | `6.015356` | `5.902597` | `-0.112758` |
| TC test | R_LowLeg | `5.889632` | `5.677678` | `-0.211954` |
| TC test | Head | `8.515058` | `7.964906` | `-0.550152` |

Conclusion:

- The real-time smooth+residual input is technically valid: cache generation, training, checkpoint remapping, and evaluation are finite, and the filter is causal with zero lookahead.
- It improves official PL gR1 on DIP and TotalCapture, and improves TotalCapture pRB after DIP fine-tune.
- The imported raw `newpl_v5_dip_best` rows are historical references from `data/experiments/newpl_v5_official_protocol_20260607_tuned`, not a same-cache fairness baseline for this run.
- Decision: diagnostic only, not selected. Continue only as a filter/loss ablation branch; do not connect to IK1/full pipeline until official PL, raw v5, and realtime smooth+residual are re-evaluated on exactly the same cache/protocol.

## IMU Neighbor Velocity-Control Module v1 implementation smoke

Date: 2026-06-12

Status: implemented and smoke-tested; long training not launched.

Purpose: add an independent module that consumes official/cache IMU `aM/wM/RMB` plus sequence-level `r_JS[6,3]` and predicts world-frame velocity control curves for IMU-adjacent skeleton nodes. The module is diagnostic only and is not connected to PL/IK1/full pipeline.

Training efficiency update: `imu_neighbor_vel_ctrl_train.py` now defaults to compact precompute before epochs. It builds per-sequence `neighbor_features[90]` and, when world GT is allowed, `neighbor_target` tensors for `vel_W/acc_W/control_vel_W`, then releases unused cache fields from the training records. Batch training slices these cached tensors and does not rerun SMPL FK target construction inside the epoch loop. The run writes `precompute_summary.json`.

Baseline velocity policy: eval first uses `pose_baseline/tran_baseline` if present to finite-difference a full 33D neighbor-node baseline. If that is absent but `v_root_vr` is a 3D root velocity, eval reports it as a root-only official VR baseline. Otherwise baseline velocity is explicitly `baseline velocity not available`.

Contract:

```text
input: aM[18] + wM[18] + RMB_6d[36] + r_JS[18] = 90D
output: neighbor_vel_W_control[33]
decoded: vel_W[33], acc_W[33], jerk_W[33]
frame: world/model frame W
r_JS: IMU origin relative to mapped joint J, expressed in joint-local frame
DIP policy: no world velocity/root velocity/acceleration GT, no DIP trans finite-difference
```

Output layout:

| Sensor | Mapped joint | Nodes | Channels |
|---|---:|---|---:|
| left_forearm | `18` | `18,20` | `6` |
| right_forearm | `19` | `19,21` | `6` |
| left_lowerleg | `4` | `4,7` | `6` |
| right_lowerleg | `5` | `5,8` | `6` |
| head | `15` | `12,15` | `6` |
| pelvis/root | `0` | `0` | `3` |

Implemented files:

```text
imu_neighbor_vel_ctrl.py
imu_neighbor_vel_ctrl_train.py
imu_neighbor_vel_ctrl_eval.py
PROJECT_STATUS.md
RECENT_REPLACEMENT_VERSIONS.md
EXPERIMENT_LOG.md
```

Loss terms:

```text
AMASS/TotalCapture:
  ctrl_vel 1.0
  decoded_vel 0.5
  acc_W 0.5
  root_vel 0.5
  root_acc_W 0.1
  segment_consistency 0.05
  vel_smooth 0.01
  jerk_smooth 0.005
  control_prior 0.001

DIP/no world GT:
  distill 0.2
  vel_smooth 0.01
  jerk_smooth 0.005
  control_prior 0.001
```

Validation commands:

```bash
python -m py_compile imu_neighbor_vel_ctrl.py imu_neighbor_vel_ctrl_train.py imu_neighbor_vel_ctrl_eval.py

python imu_neighbor_vel_ctrl_train.py \
  --train-cache /tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953/cache.pt \
  --val-cache /tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953/cache.pt \
  --output-dir /tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953/train \
  --dataset amass \
  --epochs 1 \
  --window 5 \
  --batch-size 2 \
  --hidden-size 32 \
  --num-layers 1 \
  --dropout 0.0 \
  --max-train-sequences 3 \
  --max-val-sequences 2

python imu_neighbor_vel_ctrl_eval.py \
  --cache /tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953/cache.pt \
  --checkpoint /tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953/train/best_loss.pt \
  --output-json /tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953/eval_amass.json \
  --dataset amass \
  --max-sequences 2

python imu_neighbor_vel_ctrl_eval.py \
  --cache /tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953/cache.pt \
  --checkpoint /tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953/train/best_loss.pt \
  --output-json /tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953/eval_dip.json \
  --dataset dip \
  --world-gt-mode auto \
  --max-sequences 2
```

Smoke results:

| Check | Result |
|---|---|
| Py compile | passed |
| compact precompute | passed; `precompute_summary.json` reports `in_memory_compact_precompute`, 3 train sequences, 3 world-GT sequences |
| synthetic train | finite, `best_epoch=1`, `best_value=31.239551186561584` on the latest precompute smoke |
| synthetic AMASS eval | finite, aggregate velocity/acceleration/root metrics emitted; root-only baseline source reported as `v_root_vr official VR root velocity` |
| synthetic DIP eval | finite, `world_gt_status = world velocity GT not available` |
| full-pipeline 11 metrics | not run |
| real AMASS/TotalCapture/DIP metrics | not measured |

Conclusion: implementation is ready for a real-cache AMASS pretrain and TotalCapture module-level evaluation. The current evidence is smoke-only; no claim is made yet about whether velocity-control beats a baseline velocity source, whether the acceleration loss improves real motion, or whether this should feed PL/IK/VR.

### Longtrain launch on 2026-06-13

Status: running.

Command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
CUDA_VISIBLE_DEVICES=1 /home/lingfeng/bin/longrun -- bash scripts/run_imu_neighbor_vel_ctrl_v1_longtrain_20260613.sh
```

Run root:

```text
data/experiments/imu_neighbor_vel_ctrl_v1_longtrain_20260613
```

Launch details:

```text
pid: 3520086
gpu: 1
AMASS batch: 1536
TC batch: 128
DIP distill batch: 64
window: 61
hidden/layers/dropout: 512 / 2 / 0.2
longrun outer log: data/experiments/imu_neighbor_vel_ctrl_v1_longtrain_20260613/logs/longrun_outer.log
run log: data/experiments/imu_neighbor_vel_ctrl_v1_longtrain_20260613/logs/run.log
summary target: data/experiments/imu_neighbor_vel_ctrl_v1_longtrain_20260613/summary.json
```

Stages in this run:

```text
1. AMASS pretrain, world velocity/acceleration GT enabled.
2. AMASS module eval after AMASS best.
3. TotalCapture test eval after AMASS best.
4. DIP test eval after AMASS best, world GT disabled.
5. TotalCapture train/val fine-tune from AMASS best.
6. TotalCapture test eval after TC fine-tune best.
7. DIP train/val distill from AMASS best with AMASS best as teacher, world GT disabled.
8. DIP test eval after DIP distill best, world GT disabled.
9. TotalCapture test eval after DIP distill best.
```

Boundary checks:

```text
No full-pipeline 11 metrics.
No DIP trans/root/world velocity GT.
DIP eval metrics requiring world GT must remain null / world velocity GT not available.
Baseline velocity source is pose_baseline/tran_baseline if present, otherwise 3D v_root_vr root-only, otherwise unavailable.
```

## IMU Neighbor Pos-From-Vel-Control Module v1 full implementation and evaluation

Date: 2026-06-13

Status: completed. Diagnostic module implemented, trained, evaluated, and documented. No full-pipeline 11 metrics were run.

Purpose: test a PL-like diagnostic module that consumes official IMU features plus frozen neighbor velocity-control information, then predicts root-relative position controls for the same 11 IMU-adjacent nodes.

Implemented files:

```text
imu_neighbor_pos_from_vel_ctrl.py
imu_neighbor_pos_from_vel_ctrl_train.py
imu_neighbor_pos_from_vel_ctrl_eval.py
scripts/run_imu_neighbor_pos_from_vel_ctrl_v1_20260613.sh
```

Contract:

```text
module: imu_neighbor_pos_from_vel_ctrl_v1
input: 189D = imu_feature[90] + neighbor_vel_W_control[33] + decoded neighbor_vel_W[33] + decoded neighbor_acc_W[33]
imu_feature: aM[18] + wM[18] + RMB_6d[36] + r_JS[18]
output: neighbor_pos_R_control[33]
decoded: pos_R[33], vel_R[33], acc_R[33]
node layout: [18,20], [19,21], [4,7], [5,8], [12,15], [0]
root-relative frame: p_RJ = (p_WJ - p_WR) @ R_WR, matching the existing GlobalPose row-vector PL convention
root channel: retained for 33D alignment only; root-relative root position is zero
DIP policy: no DIP trans, no DIP world/root velocity GT, no fabricated finite-difference translation
```

Training efficiency and validation:

```text
compile:
  python -m py_compile imu_neighbor_pos_from_vel_ctrl.py imu_neighbor_pos_from_vel_ctrl_train.py imu_neighbor_pos_from_vel_ctrl_eval.py

smoke train:
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613_smoke/amass_smoke

smoke eval:
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613_smoke/eval_amass_smoke.json
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613_smoke/eval_dip_smoke.json

precompute:
  all train stages write precompute_summary.json with feature_dim=189, target_dim=33, mode=in_memory_compact_precompute.
  AMASS train precompute sequences=1296, gt_velocity_input_sequences=1296.
  TC train precompute sequences=36, gt_velocity_input_sequences=36.
  DIP train precompute sequences=36, gt_velocity_input_sequences=0, dip_trans_policy=not used.
```

Full command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/bin/longrun -- bash scripts/run_imu_neighbor_pos_from_vel_ctrl_v1_20260613.sh
```

Run root:

```text
data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613
```

Selected batches from GPU preflight:

```text
AMASS batch: 1536
TotalCapture batch: 512
DIP batch: 512
GPU: 1
```

Caches:

```text
AMASS:
  data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
TotalCapture train:
  data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json
TotalCapture val:
  data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json
TotalCapture test:
  data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json
DIP train:
  data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json
DIP val:
  data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json
DIP test:
  data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
Frozen velocity checkpoint:
  data/experiments/imu_neighbor_vel_ctrl_v1_longtrain_20260613/amass_pretrain/best_loss.pt
```

Loss policy:

```text
AMASS/TotalCapture:
  ctrl_pos=1.0
  decoded_pos=1.0
  vel_R=0.2
  acc_R=0.05
  vel_input_consistency=0.05
  segment_length=0.05
  smooth=0.01
  jerk=0.005
  control_prior=0.001

DIP:
  ctrl_pos=1.0
  decoded_pos=1.0
  vel_R=0.0
  acc_R=0.0
  vel_input_consistency=0.0
  segment_length=0.05
  smooth=0.01
  jerk=0.005
  control_prior=0.001

best_loss selection:
  ctrl_pos + decoded_pos + 0.1 * vel_R
```

Checkpoints:

```text
AMASS best:
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/amass_pretrain/best_loss.pt
  best_epoch=80
  best_value=0.10381769346162742

TotalCapture fine-tune best:
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/totalcapture_finetune/best_loss.pt
  best_epoch=58
  best_value=0.10527179263532162

DIP fine-tune best:
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/dip_finetune/best_loss.pt
  best_epoch=30
  best_value=0.11427372195757926
```

Logs and JSONs:

```text
run log:
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/logs/run.log
summary:
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/summary.json
eval JSONs:
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/eval/eval_amass_after_amass_best.json
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/eval/eval_totalcapture_test_after_amass_best.json
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/eval/eval_totalcapture_test_after_tc_finetune_best.json
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/eval/eval_dip_test_after_amass_best.json
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/eval/eval_dip_test_after_dip_finetune_best.json
  data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/eval/eval_totalcapture_test_after_dip_finetune_best.json
```

Module-level metrics:

| Dataset / checkpoint | pos_R L1 cm ↓ | pos_R L2 cm ↓ | vel_R L2 cm/s ↓ | acc_R L2 cm/s² ↓ | segment err cm ↓ | baseline source | baseline pos_R L2 cm ↓ |
|---|---:|---:|---:|---:|---:|---|---:|
| AMASS / AMASS best | `21.570420` | `48.002480` | `42.634092` | `500.249292` | `28.246628` | pose_prephysics FK root-relative | `2.339005` |
| TotalCapture test / AMASS best | `21.906387` | `48.779191` | `49.979693` | `570.050407` | `28.245176` | pose_prephysics FK root-relative | `5.300923` |
| TotalCapture test / TC best | `21.810657` | `48.570789` | `49.976181` | `569.974197` | `28.263721` | pose_prephysics FK root-relative | `5.300923` |
| DIP test / AMASS best | `21.174810` | `47.895768` | `54.456256` | `1244.791155` | `28.247556` | pose_prephysics FK root-relative | `5.567279` |
| DIP test / DIP best | `21.118551` | `47.770363` | `54.456872` | `1244.797796` | `28.282076` | pose_prephysics FK root-relative | `5.567279` |
| TotalCapture test / DIP best | `21.861108` | `48.655766` | `49.979489` | `570.058731` | `28.280052` | pose_prephysics FK root-relative | `5.300923` |

Per-node pos_R L2 cm:

| Dataset / checkpoint | n18 | n20 | n19 | n21 | n4 | n7 | n5 | n8 | n12 | n15 | n0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMASS / AMASS best | `36.417` | `40.987` | `37.661` | `43.572` | `46.088` | `84.364` | `46.837` | `84.170` | `50.213` | `57.560` | `0.158` |
| TC test / TC best | `39.308` | `42.660` | `38.619` | `45.734` | `45.916` | `84.071` | `46.674` | `83.002` | `50.156` | `58.045` | `0.094` |
| DIP test / DIP best | `35.761` | `40.640` | `37.680` | `41.749` | `45.799` | `83.591` | `46.125` | `83.714` | `51.151` | `59.181` | `0.084` |

Baseline handling:

```text
official PL baseline: not applicable for the full 33D neighbor-node target; cache does not contain official PL output for nodes [18,20],[19,21],[4,7],[5,8],[12,15],[0].
newpl_v4_init36 baseline: not applicable for the full 33D neighbor-node target; cache does not contain newpl_v4 output for the same 33D layout.
available same-cache baseline: pose_prephysics FK root-relative.
velocity integration baseline: not measured; diagnostic only and not the main baseline for root-relative position.
```

Conclusions:

```text
1. Velocity controls did not improve root-relative node-position estimation in v1.
2. The model is much worse than pose_prephysics FK root-relative baseline on every dataset.
3. TotalCapture fine-tune has a tiny positive effect on TotalCapture test:
   pos_R L2 48.779191 -> 48.570789 cm, still far worse than baseline 5.300923 cm.
4. DIP fine-tune has a tiny positive effect on DIP test:
   pos_R L2 47.895768 -> 47.770363 cm, still far worse than baseline 5.567279 cm.
5. DIP world/root velocity remains unevaluated and unsupervised; DIP trans was not used.
6. The root channel is correctly near zero, but this is an alignment artifact, not a useful learned benefit.
7. Do not feed imu_neighbor_pos_from_vel_ctrl_v1 into IK/NewIK1/full pipeline yet.
8. Next useful work is architectural/target debugging: residualize against pose_prephysics, constrain segment lengths more strongly, or use velocity controls as auxiliary distillation rather than direct position conditioning.
```

## 2026-06-13 - IMU Joint Euler/Qdot/Velocity Control Module v1 Full Flow

User request:

```text
Follow the official PL-like structure, keep input as world/model-frame RMB, aM, wM, add init, and train a new diagnostic module whose outputs are IMU-mapped joint RMB Euler-angle controls, qdot controls, and velocity controls. Loss must include control-point loss, q loss, qdot loss, qddot loss, velocity loss, and acceleration loss. Try multiple loss mixes in parallel across the available GPUs. Run the full flow from implementation through training and evaluation.
```

Scope:

```text
module id: imu_joint_euler_qdot_vel_ctrl_v1
diagnostic only: yes
connected to PL/IK/NewIK1/full pipeline: no
official S4 11 metrics: not run
DIP trans/root/world velocity GT: not used
```

Implemented files:

```text
imu_joint_euler_qdot_vel_ctrl.py
imu_joint_euler_qdot_vel_ctrl_train.py
imu_joint_euler_qdot_vel_ctrl_eval.py
scripts/run_imu_joint_euler_qdot_vel_ctrl_v1_20260613.sh
```

Module contract:

```text
historical completed run input: official world/model-frame aM[18] + wM[18] + RMB_flat[54] = 90D
post-run code update input: official/project-frame aM[18] + wM[18] + R_rootIMU_sensorIMU_flat[54] = 90D
rotation input after update: R_rootIMU_sensorIMU = RMB[root_imu=5]^T @ RMB[sensor]
init: q_RJ_euler[0] + qdot_RJ_euler[0] + vel_RJ[0] = 54D
joints: [18, 19, 4, 5, 15, 0]
output heads:
  q_RJ_euler_control[18]
  qdot_RJ_euler_control[18]
  vel_RJ_control[18]
decoded from UniformCubicBSpline(return_derivatives=True):
  q_RJ_euler, qdot_from_q, qddot_from_q
  qdot_RJ, qddot_from_qdot, qdot_jerk
  vel_RJ, acc_RJ, velocity_jerk
```

Frame contract:

```text
R_RJ = R_WR^T R_WJ
p_RJ = (p_WJ - p_WR) @ R_WR
q_RJ_euler = unwrapped XYZ Euler of R_RJ
vel_RJ / acc_RJ = finite differences of p_RJ in root frame R
DIP policy = no DIP trans, no DIP world/root velocity GT, no fabricated translation finite difference
```

Post-run root-RMB input change:

```text
Changed files:
  imu_joint_euler_qdot_vel_ctrl.py
  imu_joint_euler_qdot_vel_ctrl_train.py
  imu_joint_euler_qdot_vel_ctrl_eval.py
  scripts/run_imu_joint_euler_qdot_vel_ctrl_v1_20260613.sh

New feature builder:
  imu_rootframe_features(aM, wM, RMB)

Definition:
  aM and wM keep the selected official/processed project-frame values.
  RMB input block is converted from world/model-frame RMB_flat[54] to root-frame R_rootIMU_sensorIMU_flat[54].
  R_rootIMU_sensorIMU = RMB[root_imu=5]^T @ RMB[sensor].

New default experiment root:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613

Important:
  The metrics below are the historical completed world-RMB run.
  They are not root-RMB results; root-RMB needs a fresh AMASS/TC/DIP rerun.
```

Loss terms:

```text
q_control
q
qdot_from_q
qdot_decoded
qdot_control
qddot_from_q
qddot_from_qdot
vel_control
vel
acc
consistency
smooth
jerk
control_prior
```

Best checkpoint selection:

```text
selection = q_control
          + 0.5 * q
          + 0.4 * qdot
          + 0.4 * qdot_control
          + 0.3 * qddot_from_q
          + 0.3 * qddot_from_qdot
          + 0.3 * vel
          + 0.3 * acc
```

Loss variants:

| Variant | Main emphasis |
|---|---|
| `A_qctrl_main` | stronger q-control and decoded-q supervision |
| `B_qdot_qddot_strong` | stronger qdot, qdot-control, and qddot supervision |
| `C_vel_acc_strong` | stronger velocity-control, decoded velocity, and acceleration supervision |
| `D_all_balanced` | balanced q/qdot/qddot/velocity/acceleration with stronger consistency |

Training command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/bin/longrun -- bash scripts/run_imu_joint_euler_qdot_vel_ctrl_v1_20260613.sh
```

Training data:

```text
AMASS:
  data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
TotalCapture train:
  data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json
TotalCapture val:
  data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json
TotalCapture test:
  data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json
DIP train:
  data/experiments/newpl_v5_official_protocol_20260607/caches/dip_train_with_offset_r/baseline_cache_manifest.json
DIP val:
  data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json
DIP test:
  data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
```

Training efficiency:

```text
shared compact precompute: enabled before variant training
batch preflight selected:
  AMASS batch = 1024
  TotalCapture batch = 512
  DIP batch = 512
validation: aggregate validation only; no full eval per batch
parallelism: four variants scheduled across available GPU list in the runner
```

Checkpoints and best values:

| Variant | AMASS best epoch/value | TC best epoch/value | DIP best epoch/value |
|---|---:|---:|---:|
| `A_qctrl_main` | `80 / 7.530985` | `15 / 13.682983` | `1 / 13.443324` |
| `B_qdot_qddot_strong` | `80 / 7.210382` | `1 / 13.577274` | `1 / 13.365485` |
| `C_vel_acc_strong` | `80 / 7.574993` | `1 / 13.704780` | `29 / 13.442946` |
| `D_all_balanced` | `80 / 7.571019` | `1 / 13.719992` | `10 / 13.432773` |

Artifact root:

```text
data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613
```

Logs and JSONs:

```text
run log:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/logs/run.log
summary:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/summary.json
per-variant eval JSONs:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/<variant>/eval/eval_amass_after_amass_best.json
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/<variant>/eval/eval_totalcapture_test_after_amass_best.json
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/<variant>/eval/eval_totalcapture_test_after_tc_finetune_best.json
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/<variant>/eval/eval_dip_test_after_amass_best.json
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/<variant>/eval/eval_dip_test_after_dip_finetune_best.json
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/<variant>/eval/eval_totalcapture_test_after_dip_finetune_best.json
```

Best-by-rotation module metrics:

| Dataset / stage | Best variant | Rotation geodesic deg ↓ | vel_RJ L2 cm/s ↓ | acc_RJ L2 cm/s² ↓ | Baseline rotation deg ↓ | Baseline vel L2 cm/s ↓ | Baseline acc L2 cm/s² ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| AMASS after AMASS | `D_all_balanced` | `30.5556` | `29.4059` | `365.0700` | `4.0610` | `13.1790` | `393.4799` |
| TotalCapture after AMASS | `D_all_balanced` | `29.3217` | `32.7078` | `406.7416` | `12.3839` | `19.8320` | `562.0359` |
| TotalCapture after TC fine-tune | `A_qctrl_main` | `29.1832` | `32.5447` | `402.6187` | `12.3839` | `19.8320` | `562.0359` |
| DIP after AMASS | `C_vel_acc_strong` | `32.2599` | `39.2093` | `975.2344` | `5.2618` | `28.3552` | `965.4908` |
| DIP after DIP fine-tune | `C_vel_acc_strong` | `32.6004` | `39.2226` | `975.2334` | `5.2618` | `28.3552` | `965.4908` |
| TotalCapture after DIP fine-tune | `D_all_balanced` | `29.8325` | `32.7034` | `406.7268` | `12.3839` | `19.8320` | `562.0359` |

Variant snapshot:

| Variant | AMASS rot deg ↓ | AMASS vel L2 cm/s ↓ | TC-after-TC rot deg ↓ | TC-after-TC vel L2 cm/s ↓ | DIP-after-DIP rot deg ↓ | DIP-after-DIP vel L2 cm/s ↓ |
|---|---:|---:|---:|---:|---:|---:|
| `A_qctrl_main` | `33.0077` | `29.0785` | `29.1832` | `32.5447` | `33.5444` | `39.0059` |
| `B_qdot_qddot_strong` | `36.9067` | `29.0488` | `34.7390` | `31.7835` | `38.9002` | `40.2202` |
| `C_vel_acc_strong` | `33.1815` | `29.3996` | `29.4819` | `32.7112` | `32.6004` | `39.2226` |
| `D_all_balanced` | `30.5556` | `29.4059` | `29.4100` | `32.7040` | `34.2014` | `39.2169` |

Baseline handling:

```text
baseline source: same-cache pose_prephysics FK root-relative state
official PL/newpl_v4 PL-output baseline: not directly comparable to this 18D q/qdot/velocity-control target
DIP world/root velocity GT: not available and not fabricated
```

Conclusion:

```text
1. The implementation, precompute, AMASS pretrain, TC fine-tune, DIP fine-tune, and module evals all completed.
2. The module learned smooth controls but did not learn useful joint orientation/velocity controls.
3. All four variants are much worse than same-cache baseline on rotation and velocity.
4. TotalCapture fine-tune helps A_qctrl_main slightly on TC rotation: 30.0296 -> 29.1832 deg, but baseline is 12.3839 deg.
5. DIP fine-tune does not improve the best DIP result: C_vel_acc_strong changes 32.2599 -> 32.6004 deg, while baseline is 5.2618 deg.
6. Acceleration-heavy losses reduce some acceleration/smoothness behavior but do not recover q/qdot/velocity accuracy.
7. Do not feed imu_joint_euler_qdot_vel_ctrl_v1 into IK/NewIK1/full pipeline.
8. Next work should debug the target representation/init dependency, or train residual controls against pose_prephysics instead of direct absolute controls.
```

## 2026-06-13 - IMU Joint Euler/Qdot/Velocity Control v1 Root-RMB Rerun

Question: after converting the input `RMB` block into root IMU frame, does `imu_joint_euler_qdot_vel_ctrl_v1` improve over the previous world-RMB input run?

Code/input change:

```text
feature builder: imu_rootframe_features(aM, wM, RMB)
input: aM[18] + wM[18] + R_rootIMU_sensorIMU_flat[54] = 90D
rotation definition: R_rootIMU_sensorIMU = RMB[root_imu=5]^T @ RMB[sensor]
aM/wM: unchanged selected official/processed project-frame fields
target/output/loss: unchanged from imu_joint_euler_qdot_vel_ctrl_v1
DIP policy: no DIP trans, no DIP world/root velocity GT
```

Execution notes:

```text
GPU 0 was busy; run used GPU 1 only.
User quota was at the hard limit.
First attempt failed while writing shared AMASS precompute .pt at about 1.2GB.
Runner was updated with SHARED_PRECOMPUTE=0 to avoid large disk precompute.
Second attempt failed in TC while writing last.pt.
Train script was updated with --no-save-last and runner KEEP_LAST=0 so only best_loss.pt is retained.
Because of quota, only D_all_balanced was rerun; the other three loss variants were not rerun under root-RMB.
```

Command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
GPU_LIST=1 VARIANTS='D_all_balanced' SHARED_PRECOMPUTE=0 KEEP_LAST=0 PREFLIGHT=0 AMASS_BATCH=1024 TC_BATCH=512 DIP_BATCH=512 /home/lingfeng/bin/longrun -- bash scripts/run_imu_joint_euler_qdot_vel_ctrl_v1_20260613.sh
```

Artifacts:

```text
root:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613
summary:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/summary.json
run log:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/logs/run.log
checkpoints:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/D_all_balanced/amass_pretrain/best_loss.pt
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/D_all_balanced/totalcapture_finetune/best_loss.pt
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/D_all_balanced/dip_finetune/best_loss.pt
eval JSONs:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/D_all_balanced/eval/eval_amass_after_amass_best.json
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/D_all_balanced/eval/eval_totalcapture_test_after_amass_best.json
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/D_all_balanced/eval/eval_totalcapture_test_after_tc_finetune_best.json
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/D_all_balanced/eval/eval_dip_test_after_amass_best.json
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/D_all_balanced/eval/eval_dip_test_after_dip_finetune_best.json
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/D_all_balanced/eval/eval_totalcapture_test_after_dip_finetune_best.json
```

Training best values:

| Stage | Best epoch | Best selection value |
|---|---:|---:|
| AMASS pretrain | `80` | `7.560637714181961` |
| TotalCapture fine-tune | `1` | `13.72087287703529` |
| DIP fine-tune | `16` | `13.425601844466291` |

Root-RMB D versus previous world-RMB D:

| Dataset / stage | Root-RMB rot deg ↓ | Root-RMB vel L2 cm/s ↓ | Root-RMB acc L2 cm/s² ↓ | Old world-RMB D rot deg ↓ | Old world-RMB D vel L2 cm/s ↓ | Old world-RMB D acc L2 cm/s² ↓ | Baseline rot deg ↓ | Baseline vel L2 cm/s ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMASS after AMASS | `30.8153` | `29.4072` | `364.9746` | `30.5556` | `29.4059` | `365.0700` | `4.0610` | `13.1790` |
| TotalCapture after AMASS | `29.7156` | `32.7157` | `406.7256` | `29.3217` | `32.7078` | `406.7416` | `12.3839` | `19.8320` |
| TotalCapture after TC fine-tune | `29.8054` | `32.7145` | `406.7241` | `29.4100` | `32.7040` | `406.7378` | `12.3839` | `19.8320` |
| DIP after AMASS | `33.4361` | `39.2532` | `975.2976` | `33.2060` | `39.2334` | `975.3056` | `5.2618` | `28.3552` |
| DIP after DIP fine-tune | `34.0631` | `39.2021` | `975.2924` | `34.2014` | `39.2169` | `975.2963` | `5.2618` | `28.3552` |
| TotalCapture after DIP fine-tune | `30.3944` | `32.6989` | `406.7251` | `29.8325` | `32.7034` | `406.7268` | `12.3839` | `19.8320` |

Conclusion:

```text
Root-frame RMB input does not materially improve this diagnostic module.
Compared with the previous world-RMB D_all_balanced run, AMASS and TotalCapture rotation become slightly worse.
DIP after DIP fine-tune improves only 0.1383 deg and velocity improves only 0.0148 cm/s, which is not meaningful relative to the large gap to baseline.
The same-cache pose_prephysics FK root-relative baseline remains much better: AMASS 4.0610 deg, TC 12.3839 deg, DIP 5.2618 deg versus root-RMB module roughly 29.8-34.1 deg.
Do not feed this module into IK/NewIK1/full pipeline.
```

## 2026-06-13 - NewPL v6 gR1-only Next-Control Smoothacc Long Run

Question: can v6 use the next-control idea only for gravity direction so that `gR1` improves while current-frame `pRB` stays close to the existing PL baselines?

Design:

```text
version: newpl_v6_gR1nextonly_smoothacc
base module: newpl_v6_next_control
input: smoothacc aM cache -> official/legacy 84D PL feature with init36
current output: pRB[15] + gR1[3]
auxiliary output: next-control/current-control diagnostics from v6
protocol: AMASS pretrain -> DIP-IMU fine-tune -> AMASS/DIP/TotalCapture module evaluation
TotalCapture fine-tune: not run in this official-style rerun
DIP trans/root velocity: not used
full-pipeline 11 metrics: not run
```

Loss design:

```text
Keep current pRB/gR1 fitting:
  pRB=1.0, gR1=1.0
  gt_control_pRB=0.3, gt_control_gR1=0.2
  pRB_dot=0.03, pRB_ddot_smooth=1e-6
  gR1_dot=0.03, gR1_ddot=0.001

Disable auxiliary pRB next-control:
  next_pRB=0.0
  next_gt_control_pRB=0.0
  next_pRB_vel=0.0
  next_pRB_acc=0.0
  last_control_pRB=0.0
  next_tail4_control_pRB=0.0

Enable auxiliary gR1 next-control:
  next_gR1=2.0
  next_gt_control_gR1=0.5
  next_gR1_vel=0.05
  next_gR1_acc=0.002
  last_control_gR1=0.5
  next_tail4_control_gR1=0.35
  next_control_delta_prior=0.01
```

Implementation and command:

```text
files:
  scripts/run_newpl_v6_gR1_nextonly_smoothacc_20260613.sh
  scripts/run_newpl_v6_next_control_smoothacc_gR1_20260613.sh
  scripts/summarize_newpl_v6_next_control_smoothacc_gR1.py

command:
  cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
  EXP=/tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613 \
  CACHE_ROOT=data/experiments/newpl_v5_smoothacc_20260612/caches \
  NEXT_CACHE_ROOT=/tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/next_cache_full \
  BATCH_SIZE=768 VAL_BATCH_SIZE=96 WINDOW=81 \
  EPOCHS_AMASS=80 EPOCHS_DIP=40 MAX_VAL_SEQS=128 \
  AMASS_MAX_EVAL_SEQS=20 \
  /home/lingfeng/bin/longrun -- bash scripts/run_newpl_v6_gR1_nextonly_smoothacc_20260613.sh full
```

Execution notes:

```text
GPU: local GPU1, explicitly allowed to share with ggxx.
AMASS train: 1294 sequences, val: 128 sequences.
DIP fine-tune train: 36 sequences, val: 6 sequences.
Next-control caches were precomputed before training.
Training used batch_size=768, val_batch_size=96; no per-batch full eval.
Longrun email notification completed successfully.
```

Artifacts:

```text
root:
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full
log:
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/logs/run_full.log
summary:
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/summary.json
AMASS checkpoints:
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/amass_pretrain/best_current_gR1.pt
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/amass_pretrain/best_current_module_metric.pt
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/amass_pretrain/last.pt
DIP checkpoints:
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/dip_finetune/best_current_gR1.pt
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/dip_finetune/best_current_module_metric.pt
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/dip_finetune/last.pt
eval JSONs:
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/eval_amass_after_pretrain.json
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/eval_dip_test_after_amass_pretrain.json
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/eval_totalcapture_test_after_amass_pretrain.json
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/eval_dip_test_after_dip_finetune.json
  /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/eval_totalcapture_test_after_dip_finetune.json
```

Training best values:

| Stage | Metric | Best epoch | Value |
|---|---|---:|---:|
| AMASS pretrain | total_loss | `80` | `0.210326` |
| AMASS pretrain | current_module_metric | `71` | `5.256350` |
| AMASS pretrain | current_gR1_metric | `80` | `13.497158` |
| AMASS pretrain | next_gR1_metric | `80` | `13.464520` |
| DIP fine-tune | total_loss | `40` | `0.627380` |
| DIP fine-tune | current_module_metric | `40` | `5.414863` |
| DIP fine-tune | current_gR1_metric | `40` | `19.419542` |
| DIP fine-tune | next_gR1_metric | `40` | `19.480003` |

Fair same-cache module output comparison:

| Dataset/stage | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ |
|---|---|---:|---:|---:|
| AMASS after AMASS | official PL | `1.933659` | `4.030455` | `4.838765` |
| AMASS after AMASS | newpl_v4 init36 | `2.008328` | `4.211169` | `4.876139` |
| AMASS after AMASS | newpl_v6 gR1-only | `1.941207` | `4.025294` | `5.228020` |
| DIP after AMASS | official PL | `3.080999` | `6.345701` | `12.902106` |
| DIP after AMASS | newpl_v4 init36 | `3.071188` | `6.349507` | `12.722391` |
| DIP after AMASS | newpl_v6 gR1-only | `3.112660` | `6.431476` | `12.569474` |
| TC after AMASS | official PL | `3.612753` | `7.508986` | `13.170870` |
| TC after AMASS | newpl_v4 init36 | `3.428041` | `7.119541` | `13.075061` |
| TC after AMASS | newpl_v6 gR1-only | `3.522017` | `7.356882` | `13.018311` |
| DIP after DIP FT | official PL | `3.080999` | `6.345701` | `12.902106` |
| DIP after DIP FT | newpl_v4 init36 | `3.071188` | `6.349507` | `12.722391` |
| DIP after DIP FT | newpl_v6 gR1-only | `3.093226` | `6.392231` | `12.474146` |
| TC after DIP FT | official PL | `3.612753` | `7.508986` | `13.170870` |
| TC after DIP FT | newpl_v4 init36 | `3.428041` | `7.119541` | `13.075061` |
| TC after DIP FT | newpl_v6 gR1-only | `3.573889` | `7.430918` | `12.848388` |

Per-leaf pRB L2 after DIP fine-tune:

| Dataset | Version | leaf_1 | leaf_2 | leaf_3 | leaf_4 | leaf_5 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP test | official PL | `8.598823` | `8.994866` | `4.326460` | `5.207593` | `5.197842` | `6.465117` |
| DIP test | newpl_v4 init36 | `8.724850` | `9.086111` | `4.103781` | `5.333484` | `5.232543` | `6.496154` |
| DIP test | newpl_v6 gR1-only | `8.726285` | `9.133858` | `4.350231` | `5.102146` | `5.197729` | `6.502050` |
| TC test | official PL | `7.056774` | `7.258003` | `7.163494` | `6.819727` | `9.691798` | `7.597959` |
| TC test | newpl_v4 init36 | `7.185838` | `7.323622` | `6.543214` | `6.298469` | `8.832665` | `7.236761` |
| TC test | newpl_v6 gR1-only | `6.973117` | `7.532082` | `7.247189` | `6.909495` | `8.954300` | `7.523237` |

Temporal readout after DIP fine-tune:

| Dataset | Version | pRB vel L2 cm/s ↓ | pRB acc L2 cm/s² ↓ | gR1 vel vector L2 ↓ |
|---|---|---:|---:|---:|
| DIP test | official PL | `40.244087` | `2684.142116` | `0.677733` |
| DIP test | newpl_v4 init36 | `40.243880` | `2666.394326` | `0.673141` |
| DIP test | newpl_v6 gR1-only | `66.402182` | `2691.549734` | `0.548376` |
| TC test | official PL | `34.038251` | `2099.110771` | `0.715926` |
| TC test | newpl_v4 init36 | `32.694543` | `1801.389069` | `0.690698` |
| TC test | newpl_v6 gR1-only | `57.844750` | `761.011162` | `0.383529` |

Analysis:

```text
1. AMASS after pretrain does not improve gR1 over official/v4: 5.228020 deg is worse than official 4.838765 and v4 4.876139. It does slightly improve pRB L2 over official: 4.025294 versus 4.030455.
2. DIP after AMASS improves gR1 versus official/v4 but loses pRB: 12.569474 deg versus official 12.902106 and v4 12.722391, but pRB L2 is 6.431476 versus official 6.345701.
3. DIP fine-tune helps the candidate: gR1 improves from 12.569474 to 12.474146 on DIP test, and pRB improves from 6.431476 to 6.392231. It still does not beat official/v4 pRB.
4. TotalCapture after DIP fine-tune has the best gR1 in the fair comparison: 12.848388 deg versus official 13.170870 and v4 13.075061. pRB L2 beats official but not v4: 7.430918 versus official 7.508986 and v4 7.119541.
5. Temporal gR1 velocity is better for the candidate, but pRB velocity L2 is much worse on DIP/TC, consistent with disabling pRB auxiliary next-control and relying only on current pRB losses.
6. This confirms the hypothesis that next-control can be used as a gravity-specific regularizer. It does not prove a full PL replacement because pRB and pRB temporal behavior are not yet strong enough.
```

Conclusion:

```text
newpl_v6_gR1nextonly_smoothacc is useful as a gR1-specialized diagnostic branch.
It should not replace PL globally yet.
The next iteration should either add a pRB-preserving temporal term that does not use next-control pRB targets, or test whether IK1 benefits from better gR1 despite slightly weaker pRB.
No DIP translation/root velocity was used or fabricated.
No full-pipeline 11 metrics were run.
```

<!-- BEGIN newpl-v6-next-p-pdot-pddot-strong--2026-06-16 -->
## EXP-20260616-newpl_v6_next_p_pdot_pddot_strong — Decoded p/pd/pdd strong supervision

Status: implemented, smoke-tested, and full AMASS->DIP trained/evaluated at
module level. Full-pipeline 11 metrics were not run because same-cache module
metrics do not justify escalation.

Question: can the v6 next-control route be trained with only normalized decoded
`next_pl / next_pldot / next_plddot` pRB[15] supervision, selecting the best
checkpoint by the validation normalized `p + pd + pdd` composite?

Implementation:

```text
changed:
  pl_next_control_train.py
  scripts/run_newpl_v6_next_control_smoothacc_gR1_20260613.sh
added:
  scripts/run_newpl_v6_next_p_pdot_pddot_strong_20260615.sh
  scripts/summarize_newpl_v6_next_p_pdot_pddot_strong.py
```

Input/output contract:

```text
PL input: smooth aRB[18] + raw wRB[18] + raw RRB[45] + gR0[3] = 84D
init: offset_r[18] + pRL[15] + gR0[3] = 36D
current output consumed by IK1: pRB_t[15] + gR1_t[3] = 18D
aux supervised outputs: next_pl[..., :15], next_pldot[..., :15], next_plddot[..., :15]
GT source: existing next-control cache targets derived from GTControlCache controls
DIP trans/root velocity: not used
TotalCapture fine-tune: not run
```

Loss and selection:

```text
loss_preset = p_pdot_pddot_strong
active loss terms:
  next_pRB_norm_pos = 1.0
  next_pRB_norm_vel = 1.0
  next_pRB_norm_acc = 1.0
all old current/control/gR1/smooth/prior terms = 0.0
normalization = train-cache RMS scales for p, pd, pdd
best checkpoint = best_p_pdot_pddot_strong.pt
selection metric = validation next_pRB_norm_composite
```

Static validation:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -m py_compile \
  pl_next_control_train.py \
  pl_next_control_cache.py \
  pl_curve.py \
  pl_curve_pl_accuracy_eval.py \
  pl_next_control_eval.py \
  scripts/summarize_newpl_v6_next_control_smoothacc_gR1.py \
  scripts/summarize_newpl_v6_next_p_pdot_pddot_strong.py

bash -n \
  scripts/run_newpl_v6_next_control_smoothacc_gR1_20260613.sh \
  scripts/run_newpl_v6_next_p_pdot_pddot_strong_20260615.sh
```

Result: passed.

Smoke command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/bin/longrun -- bash -lc \
'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} NO_TEE=1 CACHE_MAX=2 \
MAX_TRAIN_SEQS=2 MAX_VAL_SEQS=2 MAX_EVAL_SEQS=2 AMASS_MAX_EVAL_SEQS=2 \
MAX_EVAL_FRAMES=128 BATCH_SIZE=2 VAL_BATCH_SIZE=2 WINDOW=21 VAL_WINDOW=21 \
EPOCHS_AMASS=1 EPOCHS_DIP=1 \
bash scripts/run_newpl_v6_next_p_pdot_pddot_strong_20260615.sh smoke'
```

Operational note: the first smoke attempt wrote bounded caches under the old
`newpl_v6_next_control_smoothacc_gR1_20260613` cache root. That was corrected:
those generated old-root cache directories were removed, and the wrapper now
uses the new experiment cache root for smoke. Full mode may reuse compatible
old full caches only when the expected full cache manifests already exist.

Artifacts:

```text
smoke root:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke
cache root:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/caches
summary:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke/summary_p_pdot_pddot_strong.json
AMASS checkpoint:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke/amass_pretrain/best_p_pdot_pddot_strong.pt
DIP checkpoint:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke/dip_finetune/best_p_pdot_pddot_strong.pt
eval JSONs:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke/eval_amass_after_pretrain.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke/eval_dip_test_after_amass_pretrain.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke/eval_totalcapture_test_after_amass_pretrain.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke/eval_dip_test_after_dip_finetune.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke/eval_totalcapture_test_after_dip_finetune.json
```

Smoke train results:

| Stage | Best epoch | normalized p+pd+pdd | p scale | pd scale | pdd scale |
|---|---:|---:|---:|---:|---:|
| AMASS pretrain | `1` | `0.3779128194` | `0.3678918372` | `0.3269050665` | `6.0625984267` |
| DIP fine-tune | `1` | `0.5174257755` | `0.3709179209` | `0.3357553991` | `6.3741371968` |

Smoke module metrics, same-cache bounded eval:

| Dataset/stage | Version | next p L2 cm | next pd L2 cm/s | next pdd L2 cm/s2 | current gR1 deg |
|---|---|---:|---:|---:|---:|
| AMASS after AMASS | strong AMASS | `2.686361` | `6.666100` | `455.439850` | `5.544665` |
| AMASS after AMASS | prior raw v6 AMASS | `2.730000` | `6.692560` | `455.608688` | `6.184368` |
| DIP after DIP FT | strong DIP | `1.326445` | `1.070521` | `26.545074` | `0.837331` |
| DIP after DIP FT | prior raw v6 DIP | `1.226944` | `1.050245` | `31.675880` | `1.475745` |
| TC after DIP FT | strong DIP | `4.911571` | `28.242008` | `474.005539` | `3.421504` |
| TC after DIP FT | prior raw v6 DIP | `5.062399` | `28.298583` | `452.309418` | `2.629442` |

Interpretation:

```text
The implementation path is finite and writes the expected best/last checkpoints
and same-cache eval JSONs. The loss preset correctly zeroes all old loss terms
except normalized decoded p/pd/pdd. This smoke is too small to support a
replacement claim. A full run is still required, and the final decision must
compare best_p_pdot_pddot_strong.pt and last.pt against official PL,
newpl_v4_init36, raw newpl_v5_dip_best, and prior v6 on the same caches.
```

Full command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
CUDA_VISIBLE_DEVICES=1 \
NO_TEE=0 \
CACHE_ROOT=data/experiments/newpl_v5_smoothacc_20260612/caches \
NEXT_CACHE_ROOT=/tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/next_cache_full \
BATCH_SIZE=32 \
VAL_BATCH_SIZE=32 \
WINDOW=81 \
EPOCHS_AMASS=80 \
EPOCHS_DIP=40 \
/home/lingfeng/bin/longrun -- bash scripts/run_newpl_v6_next_p_pdot_pddot_strong_20260615.sh full
```

Full artifacts:

```text
root:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full
log:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/logs/run_full.log
summary:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/summary_p_pdot_pddot_strong.json
checkpoints:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/amass_pretrain/best_p_pdot_pddot_strong.pt
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/amass_pretrain/last.pt
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/dip_finetune/best_p_pdot_pddot_strong.pt
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/dip_finetune/last.pt
eval JSONs:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/eval_amass_after_pretrain.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/eval_dip_test_after_amass_pretrain.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/eval_totalcapture_test_after_amass_pretrain.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/eval_dip_test_after_dip_finetune.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/eval_totalcapture_test_after_dip_finetune.json
```

Full training selection:

| Stage | Train seqs | Val seqs | Best epoch | normalized p+pd+pdd | p scale | pd scale | pdd scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMASS pretrain | `1294` | `128` | `70` | `0.6011257470` | `0.3572728132` | `0.3579577642` | `7.3343620957` |
| DIP fine-tune | `36` | `6` | `39` | `0.9176913500` | `0.3663190813` | `0.4027485024` | `17.5228447399` |

Full same-cache module eval after DIP fine-tune:

| Dataset | Version | current pRB L2 cm | current gR1 deg | next p L2 cm | next pd L2 cm/s | next pdd L2 cm/s2 |
|---|---|---:|---:|---:|---:|---:|
| DIP test | official PL smoothacc | `6.345701` | `12.902106` | `6.465117` | `40.244087` | `2684.142116` |
| DIP test | newpl_v4_init36_smoothacc | `6.349541` | `12.722353` | `6.496246` | `40.243882` | `2666.394318` |
| DIP test | newpl_v5_raw_dip_on_smoothinput | `6.357881` | `12.512222` | `6.507716` | `40.250549` | `2666.365777` |
| DIP test | prior newpl_v6_raw_dip_on_smoothinput | `6.370279` | `12.615336` | `6.478526` | `66.331095` | `2660.687140` |
| DIP test | strong best_p_pdot_pddot | `6.353314` | `12.901381` | `6.454685` | `64.602965` | `2684.758215` |
| DIP test | strong last | `6.353314` | `12.901381` | `6.454691` | `64.593160` | `2685.167951` |
| TC test | official PL smoothacc | `7.508986` | `13.170870` | `7.597959` | `34.038251` | `2099.110771` |
| TC test | newpl_v4_init36_smoothacc | `7.119541` | `13.075061` | `7.236774` | `32.694513` | `1801.389069` |
| TC test | newpl_v5_raw_dip_on_smoothinput | `7.255848` | `13.138471` | `7.387153` | `32.724789` | `1800.290009` |
| TC test | prior newpl_v6_raw_dip_on_smoothinput | `7.484898` | `12.954138` | `7.578813` | `57.576946` | `711.415844` |
| TC test | strong best_p_pdot_pddot | `7.508233` | `13.168667` | `7.589151` | `55.491009` | `806.639374` |
| TC test | strong last | `7.508233` | `13.168667` | `7.589482` | `55.475717` | `807.941780` |

Full interpretation:

```text
The existing p_pdot_pddot_strong experiment primarily supervises and selects
next-frame decoded p/pdot/pddot. It is not sufficient to claim current-frame
position/velocity/acceleration accuracy until the current-frame p/pdot/pddot
evaluation below is added.

The full run is finite and selected best_p_pdot_pddot_strong.pt as intended.
The normalized p/pd/pdd target reduces the selected composite during training,
but it does not produce a stronger current PL module.

DIP after DIP fine-tune: current pRB is slightly worse than official PL and v4,
and gR1 is much worse than v4/raw-v5 while roughly official-level. The candidate
has slightly better next p than official/v4/v5, but next velocity is much worse.

TotalCapture after DIP fine-tune: the candidate preserves official-level
pRB/gR1 and improves next acceleration versus official/v4/v5, but it is clearly
worse than newpl_v4_init36_smoothacc on current pRB/gR1 and worse than official/v4
on next velocity.

Decision: diagnostic only; do not promote and do not run full-pipeline 11 metrics
from this checkpoint.
```

Current-frame p/pdot/pddot evaluation command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
PYTHONPATH=. /home/lingfeng/bin/longrun -- \
  /home/lingfeng/.conda/envs/globalpose-gpu/bin/python \
  scripts/evaluate_current_p_pdot_pddot.py \
  --output-dir data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval \
  --dip-cache /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/next_cache_full/next_dip_test/pl_next_control_cache_manifest.json \
  --totalcapture-cache /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/next_cache_full/next_tc_test/pl_next_control_cache_manifest.json \
  --version official_PL_smoothacc=official \
  --version newpl_v4_init36_smoothacc=data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt \
  --version newpl_v5_raw_dip_on_smoothinput=data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt \
  --version prior_newpl_v6_raw_dip_on_smoothinput=data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/dip_finetune/best_next_module_metric.pt \
  --version strong_best_p_pdot_pddot=data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/dip_finetune/best_p_pdot_pddot_strong.pt \
  --version strong_last=data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/dip_finetune/last.pt \
  --strong-version strong_best_p_pdot_pddot
```

Current-frame eval semantics:

```text
current output:
  output["pl"] = pRB_t[15] + gR1_t[3]

next output:
  output["next_pl"] = predicted pRB_{t+1}[15] + gR1_{t+1}[3]

next derivatives:
  output["next_pldot"], output["next_plddot"]
  are decoded from predicted next control via spline.

Current experiment selection:
  selected by validation normalized next p/pdot/pddot composite.

Therefore this experiment does not by itself prove current-frame p/pdot/pddot accuracy.
```

Current-frame eval masks and units:

| Metric | Prediction | GT | Mask | Unit |
|---|---|---|---|---|
| current p | `output["pl"][..., :15]` | `pl_target[..., :15]` | all current frames | L1/L2 cm |
| current pdot | central FD of `output["pl"][..., :15]` | `gt_pldot[..., :15]` | exclude first/last frames | L1/L2 cm/s |
| current pddot | central FD acceleration of `output["pl"][..., :15]` | `gt_plddot[..., :15]` | exclude first/last frames | L1/L2 cm/s^2 |
| next p/pdot/pddot | `output["next_pl"]`, `output["next_pldot"]`, `output["next_plddot"]` | `pl_target_next`, `gt_pldot_next`, `gt_plddot_next` | `valid_next_mask` | cm, cm/s, cm/s^2 |

Current-frame eval artifacts:

```text
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/eval_current_p_pdot_pddot_dip.json
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/eval_current_p_pdot_pddot_totalcapture.json
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/summary_current_p_pdot_pddot.md
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/per_sequence_current_p_pdot_pddot.csv
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/per_leaf_current_p_pdot_pddot.csv
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/alignment_sweep.csv
```

Current-frame eval frame counts:

| Dataset | sequences | current_eval_frames | current_derivative_eval_frames | next_eval_frames |
|---|---:|---:|---:|---:|
| DIP test | `19` | `57994` | `57956` | `57975` |
| TotalCapture test | `4` | `16124` | `16116` | `16120` |

Table 1: current-frame accuracy:

| Dataset | Version | current p L2 cm | current pdot L2 cm/s | current pddot L2 cm/s2 | current gR1 deg |
|---|---|---:|---:|---:|---:|
| DIP test | official PL smoothacc | `6.462137` | `31.449577` | `1807.055833` | `15.216293` |
| DIP test | newpl_v4_init36_smoothacc | `6.451342` | `31.421659` | `1792.007486` | `14.991070` |
| DIP test | newpl_v5_raw_dip_on_smoothinput | `6.459074` | `31.431964` | `1791.990877` | `14.753305` |
| DIP test | prior newpl_v6_raw_dip_on_smoothinput | `6.483322` | `31.425229` | `1791.925604` | `14.786131` |
| DIP test | strong best_p_pdot_pddot | `6.465181` | `31.419971` | `1792.008308` | `15.215611` |
| DIP test | strong last | `6.465181` | `31.419971` | `1792.008308` | `15.215611` |
| TC test | official PL smoothacc | `7.254956` | `30.784883` | `1882.019322` | `13.745581` |
| TC test | newpl_v4_init36_smoothacc | `6.879507` | `29.582701` | `1617.953217` | `13.626233` |
| TC test | newpl_v5_raw_dip_on_smoothinput | `7.014368` | `29.609305` | `1617.023291` | `13.670421` |
| TC test | prior newpl_v6_raw_dip_on_smoothinput | `7.230463` | `29.585913` | `1616.539405` | `13.462959` |
| TC test | strong best_p_pdot_pddot | `7.253737` | `29.581992` | `1618.041598` | `13.743610` |
| TC test | strong last | `7.253737` | `29.581992` | `1618.041598` | `13.743610` |

Table 2: next-frame accuracy:

| Dataset | Version | next p L2 cm | next pdot L2 cm/s | next pddot L2 cm/s2 | next gR1 deg |
|---|---|---:|---:|---:|---:|
| DIP test | official PL smoothacc | `6.544064` | `31.442348` | `1806.703345` | `15.219841` |
| DIP test | newpl_v4_init36_smoothacc | `6.557839` | `31.414378` | `1791.633494` | `14.994450` |
| DIP test | newpl_v5_raw_dip_on_smoothinput | `6.568955` | `31.424697` | `1791.614994` | `14.756244` |
| DIP test | prior newpl_v6_raw_dip_on_smoothinput | `6.558404` | `54.595606` | `1782.213776` | `14.796517` |
| DIP test | strong best_p_pdot_pddot | `6.536575` | `53.040941` | `1813.554983` | `15.219834` |
| DIP test | strong last | `6.536606` | `53.032184` | `1814.021152` | `15.219834` |
| TC test | official PL smoothacc | `7.332092` | `30.799751` | `1882.179417` | `13.751889` |
| TC test | newpl_v4_init36_smoothacc | `6.982701` | `29.597338` | `1618.117857` | `13.632039` |
| TC test | newpl_v5_raw_dip_on_smoothinput | `7.130036` | `29.624039` | `1617.171514` | `13.674679` |
| TC test | prior newpl_v6_raw_dip_on_smoothinput | `7.313069` | `52.591713` | `650.197336` | `13.476821` |
| TC test | strong best_p_pdot_pddot | `7.323794` | `50.714014` | `742.650343` | `13.751886` |
| TC test | strong last | `7.324095` | `50.699906` | `743.746064` | `13.751886` |

Table 3: alignment sweep:

```text
All full DIP and TotalCapture current p, current pdot_fd, and current pddot_fd
rows have best shift = 0 for all six versions. No nonzero-shift warning is
triggered in the full eval. Full per-version shift values are in:
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/alignment_sweep.csv
```

Current-frame p/pdot/pddot decision:

```text
Decision: diagnostic only.

The strong checkpoint fails the requested current-frame non-regression gate:
- DIP current p L2: strong 6.465181 cm vs best baseline 6.451342 cm.
- DIP current pddot L2: strong 1792.008308 cm/s^2 vs best baseline 1791.925604 cm/s^2.
- TotalCapture current p L2: strong 7.253737 cm vs best baseline 6.879507 cm.
- TotalCapture current pddot L2: strong 1618.041598 cm/s^2 vs best baseline 1616.539405 cm/s^2.

Current pdot is marginally best on both datasets, but that is not enough under
the requested gate because current p and current pddot regress. Do not promote.
```

Velocity / acceleration metric audit command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home/lingfeng/bin/longrun -- \
  /home/lingfeng/.conda/envs/globalpose-gpu/bin/python \
  scripts/audit_newpl_velocity_metric.py \
  --output-dir data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit \
  --cache DIP=/tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/next_cache_full/next_dip_test/pl_next_control_cache_manifest.json \
  --cache TotalCapture=/tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/next_cache_full/next_tc_test/pl_next_control_cache_manifest.json \
  --version official_PL_smoothacc=official \
  --version newpl_v4_init36_smoothacc=data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt \
  --version newpl_v5_raw_dip_on_smoothinput=data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt \
  --version prior_newpl_v6_raw_dip_on_smoothinput=data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/dip_finetune/best_next_module_metric.pt \
  --version strong_best_p_pdot_pddot=data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/dip_finetune/best_p_pdot_pddot_strong.pt \
  --version strong_last=data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/dip_finetune/last.pt
```

Velocity audit artifacts:

```text
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit/velocity_metric_audit.json
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit/velocity_metric_audit.md
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit/velocity_alignment_sweep.csv
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit/velocity_per_leaf.csv
data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit/velocity_per_sequence.csv
```

Velocity audit definitions:

| Metric | Prediction | GT | Mask | Unit |
|---|---|---|---|---|
| current FD velocity | central FD of `output["pl"][..., :15]` | `gt_pldot[..., :15]` | exclude first/last frames | cm/s |
| current FD acceleration | central FD acceleration of `output["pl"][..., :15]` | `gt_plddot[..., :15]` | exclude first/last frames | cm/s^2 |
| next-head velocity | `output["next_pldot"][..., :15]` | `gt_pldot_next[..., :15]` | `valid_next_mask` | cm/s |
| next-head acceleration | `output["next_plddot"][..., :15]` | `gt_plddot_next[..., :15]` | `valid_next_mask` | cm/s^2 |
| next-position FD velocity | central FD of `output["next_pl"][..., :15]` | `gt_pldot_next[..., :15]` | `valid_next_mask` minus FD boundaries | cm/s |
| next-position FD acceleration | central FD acceleration of `output["next_pl"][..., :15]` | `gt_plddot_next[..., :15]` | `valid_next_mask` minus FD boundaries | cm/s^2 |

Velocity audit masks:

| Dataset | sequences | frames | valid_next_frames | current_derivative_valid_frames | excluded_boundary_frames |
|---|---:|---:|---:|---:|---:|
| DIP test | `19` | `57994` | `57975` | `57956` | `38` |
| TotalCapture test | `4` | `16124` | `16120` | `16116` | `8` |

GT self-consistency and dt/unit audit:

| Dataset | decoded pl L2 | decoded dot L2 | decoded ddot L2 | FD GT vel L2 at dt=1/60 | FD GT acc L2 at dt=1/60 | FD GT vel L2 at dt=1 |
|---|---:|---:|---:|---:|---:|---:|
| DIP test | `0.086820` | `0.000000` | `0.000000` | `0.083706` | `0.943413` | `53.769033` |
| TotalCapture test | `0.060392` | `0.000000` | `0.000000` | `0.367573` | `4.401025` | `51.970507` |

Interpretation: the cache derivative targets are exactly consistent with the
control decoder (`decoded_dot/ddot` L2 = 0). The correct dt is manifest `1/60`.
The `dt=1` sweep intentionally shows how badly the metric would scale if dt
were wrong; it is not evidence of an actual dt mismatch because eval dt and
manifest dt are both `0.016666666666666666`.

Velocity audit model metrics:

| Dataset | Version | current FD vel | current FD acc | next-head vel | next-head acc | next-position FD vel | next-position FD acc |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP test | official PL smoothacc | `31.449577` | `1807.055855` | `32.361504` | `1810.301938` | `32.338752` | `1808.462783` |
| DIP test | prior newpl_v6_raw_dip_on_smoothinput | `31.425229` | `1791.925571` | `54.595607` | `1782.213774` | `32.341935` | `1808.370255` |
| DIP test | strong best_p_pdot_pddot | `31.419971` | `1792.008316` | `53.040942` | `1813.554982` | `32.216370` | `1807.979955` |
| TotalCapture test | official PL smoothacc | `30.784883` | `1882.019351` | `31.635313` | `1899.639228` | `31.559175` | `1895.471466` |
| TotalCapture test | prior newpl_v6_raw_dip_on_smoothinput | `29.585914` | `1616.539413` | `52.591712` | `650.197321` | `31.573507` | `1894.099732` |
| TotalCapture test | strong best_p_pdot_pddot | `29.581992` | `1618.041596` | `50.714015` | `742.650349` | `31.785827` | `1931.831008` |

Alignment summary:

```text
DIP:
  current_fd_velocity / current_fd_acceleration best shift = 0 for all versions.
  strong best next_head_velocity best shift = -2.
  strong best next_head_acceleration best shift = +2.

TotalCapture:
  current_fd_velocity / current_fd_acceleration best shift = 0 for all versions.
  strong best next_head_velocity best shift = -2.
  strong best next_head_acceleration best shift = +1.
```

Velocity audit A-E conclusion:

```text
A. Real current-frame model issue:
   partial only. Current FD velocity/acceleration are comparable to same-cache
   baselines, but the separate current-frame p/pddot gate still failed above.

B. Current/next or temporal/source mismatch:
   yes for next-head derivatives. The large velocity anomaly is concentrated in
   output["next_pldot"] / output["next_plddot"], and their best shift is nonzero.

C. dt/unit mismatch:
   no actual mismatch. dt=1 would be wrong, but manifest/eval dt are 1/60.

D. derivative target definition mismatch:
   no. decoded control dot/ddot vs cache gt_pldot/gt_plddot are 0 L2.

E. boundary/mask issue:
   no. current derivative metrics exclude first/last frames and next metrics use
   valid_next_mask with separate frame counts.

Final audit conclusion:
  Do not pool current FD velocity and next-head velocity into one claim.
  The current-frame FD metric is aligned and uses the correct dt/mask.
  The large velocity error that motivated the audit is mainly a next-head
  derivative temporal/source issue, not an actual dt/unit or GT-cache mismatch.
  This experiment remains diagnostic only; do not promote and do not change
  network structure based only on this metric.
```
<!-- END newpl-v6-next-p-pdot-pddot-strong--2026-06-16 -->

<!-- BEGIN newpl-v4-init36-dip-fullpipeline-11metrics--2026-06-16 -->
## 2026-06-16 - newpl_v4_init36 DIP Full-Pipeline 11 Metrics Backfill

Task: fill the missing DIP test full-pipeline 11 metrics for the historical
`newpl_v4_init36` checkpoint. Do not reuse PL module-level pRB/gR1 metrics as
full-pipeline evidence.

Run contract:

```text
root: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616
evaluator: newik1_real_streaming_audit.py
cache/protocol: data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
source raw DIP cache: data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json
checkpoint: data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
replacement: PL-s1 only; official IK-s1, IK-s2, VR, and carticulate physics downstream preserved
DIP trans/root-velocity supervision: false; evaluation only
metric: MotionEvaluator full-pipeline 11 metrics
```

Exact command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
CUDA_VISIBLE_DEVICES=0 METRIC_CHUNK_FRAMES=256 /home/lingfeng/bin/longrun -- bash scripts/run_newpl_v4_init36_dip_fullpipeline_11metrics_20260616.sh
```

Implementation note:

```text
newik1_real_streaming_audit.py already supported --pl-checkpoint with
--ik1-backend original, which preserves official downstream modules. The only
code change needed for this run was evaluation/runtime wrapping: GPU chunked
MotionEvaluator metrics via --metric-chunk-frames and --skip-module-metrics for
this full-pipeline-only task. The PL/IK/VR/physics forward path is unchanged.
```

Results:

| Dataset | Version | Score | Local SIP | Local Angle | Local Joint | Local Mesh | Global SIP | Global Angle | Global Joint | Global Mesh | Root Jitter | Joint Jitter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP-IMU-test | official_gpnet | `44.641437` | `13.548337` | `8.469859` | `4.648157` | `5.408259` | `13.409406` | `8.291682` | `4.547544` | `5.265691` | `0.157846` | `0.258183` |
| DIP-IMU-test | newpl_v4_init36_official_downstream | `44.708897` | `13.537034` | `8.484648` | `4.646514` | `5.426462` | `13.429909` | `8.329860` | `4.602831` | `5.356486` | `0.154876` | `0.251050` |

Conclusion:

```text
Score delta newpl_v4_init36 - official_gpnet = +0.067461 (worse).
不支持 DIP full-pipeline improvement claim.
These are full-pipeline 11 metrics, not PL module-level pRB/gR1 metrics.
```

Artifacts:

```text
summary: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/summary.md
result JSON: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/result_summary.json
baseline JSON: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/eval/dip_official_gpnet.json
newpl JSON: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/eval/dip_newpl_v4_init36_official_downstream.json
run log: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/logs/run.log
exact command: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/exact_command.txt
```
<!-- END newpl-v4-init36-dip-fullpipeline-11metrics--2026-06-16 -->

<!-- BEGIN newpl-joint-leaf-acc--2026-06-19 -->
## EXP-20260619-newpl_joint_leaf_acc

Line: `PL-s1 / joint-leaf acceleration`

Purpose: add a joint-leaf semantic NewPL cache/training/eval route while preserving the existing PLCurve forward/loss structure. This experiment intentionally does not run old IK/full-pipeline evaluation because those consumers interpret the first 15D as vertex `pRB`, while this cache uses joint-leaf `p_leaf_joint_R`.

Implementation:

```text
helpers: pl_curve.py
  PL_JOINT_LEAF_ACC_INPUT_SIZE = 102
  pl_joint_leaf_target_from_pose(pose, body_model)
  pl_joint_leaf_init_feature(offset_r, pl0, init_size)

cache builder: pl_joint_leaf_acc_cache.py
  manifest type: pl_curve_joint_leaf_acc_cache_v1
  fields: pl_input, pl_target, pl_base, pl_init_feature, pl_target_control
  target/control source: /home/lingfeng/projects/data/dataset_work/*/gt_control/*/joint_pos_R(_control)
  gravity/control source: pl_pRB_gR1(_control) gravity block
  base source: pose_prephysics FK in the same joint-leaf semantic space
  frozen acceleration checkpoint:
    /home/lingfeng/projects/imu_acc_explainability/code/outputs/imu_leaf_acc_predictor_v1/full_world_leaf5_no_trans_smoothed_gtacc_centered_ma_w9_20260619_155137/dip_finetune/best.pt

training compatibility: pl_curve_train.py accepts pl_curve_joint_leaf_acc_cache_v1
module eval: scripts/eval_newpl_joint_leaf_acc.py
```

Feature modes:

| Mode | Dim | Layout |
|---|---:|---|
| baseline_jointtarget_84D | 84 | `aRB[18]+wRB[18]+RRB[45]+gR0[3]` |
| acc_root_102D | 102 | baseline 84D + `a_smoothed_R[3]+a_output_R[15]` |
| acc_mixed_102D | 102 | baseline 84D + `a_smoothed_W[3]+a_output_W[15]` |

Smoke commands:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose

/home/lingfeng/.conda/envs/globalpose-gpu/bin/python pl_joint_leaf_acc_cache.py build \
  --input-cache data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json \
  --gt-control-cache /home/lingfeng/projects/data/dataset_work/dip/gt_control/val/manifest.json \
  --output-dir data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_baseline_jointtarget_84D \
  --feature-mode baseline_jointtarget_84D --max-sequences 2 --max-frames 120 --shard-size 2 --device cpu

/home/lingfeng/.conda/envs/globalpose-gpu/bin/python pl_joint_leaf_acc_cache.py build \
  --input-cache data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json \
  --gt-control-cache /home/lingfeng/projects/data/dataset_work/dip/gt_control/val/manifest.json \
  --output-dir data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_acc_root_102D \
  --feature-mode acc_root_102D --max-sequences 2 --max-frames 120 --shard-size 2 --device cpu

/home/lingfeng/.conda/envs/globalpose-gpu/bin/python pl_joint_leaf_acc_cache.py build \
  --input-cache data/experiments/newpl_v5_official_protocol_20260607/caches/dip_val_with_offset_r/baseline_cache_manifest.json \
  --gt-control-cache /home/lingfeng/projects/data/dataset_work/dip/gt_control/val/manifest.json \
  --output-dir data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_acc_mixed_102D \
  --feature-mode acc_mixed_102D --max-sequences 2 --max-frames 120 --shard-size 2 --device cpu

/home/lingfeng/.conda/envs/globalpose-gpu/bin/python pl_joint_leaf_acc_cache.py validate \
  --manifests \
  data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_baseline_jointtarget_84D/pl_curve_cache_manifest.json \
  data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_acc_root_102D/pl_curve_cache_manifest.json \
  data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_acc_mixed_102D/pl_curve_cache_manifest.json \
  --output-json data/experiments/newpl_joint_leaf_acc_20260619/smoke/cache_validation.json

/home/lingfeng/.conda/envs/globalpose-gpu/bin/python pl_curve_train.py \
  --train-cache data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_baseline_jointtarget_84D/pl_curve_cache_manifest.json \
  --val-cache data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_baseline_jointtarget_84D/pl_curve_cache_manifest.json \
  --output-dir data/experiments/newpl_joint_leaf_acc_20260619/smoke/train_baseline_jointtarget_84D \
  --experiment-name smoke_baseline_jointtarget_84D --epochs 1 --window 61 --lr 1e-4 \
  --hidden-size 64 --tail-length 4 --residual-scale 0.005 --dropout 0.1 --grad-clip 1.0 \
  --init-size 36 --batch-size 2 --max-train-sequences 2 --max-val-sequences 2 \
  --disable-ik-distill --baseline-pRB-weight 0.0 --baseline-gR1-weight 0.0 \
  --gt-control-pRB-weight 0.3 --gt-control-gR1-weight 0.1 \
  --pRB-ddot-smooth-weight 0.000001 --gR1-dot-weight 0.03 --gR1-ddot-weight 0.001 \
  --selection-metric control_physical

/home/lingfeng/.conda/envs/globalpose-gpu/bin/python scripts/eval_newpl_joint_leaf_acc.py \
  --cache data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_baseline_jointtarget_84D/pl_curve_cache_manifest.json \
  --checkpoint data/experiments/newpl_joint_leaf_acc_20260619/smoke/train_baseline_jointtarget_84D/best_loss.pt \
  --output-json data/experiments/newpl_joint_leaf_acc_20260619/smoke/eval_baseline_jointtarget_84D.json \
  --output-summary data/experiments/newpl_joint_leaf_acc_20260619/smoke/eval_baseline_jointtarget_84D.md
```

Smoke validation:

| Check | Value |
|---|---:|
| shared sequences | `2` |
| frames per mode | `240` |
| baseline feature dim | `84` |
| acc_root/acc_mixed feature dim | `102` |
| max target diff across modes | `0.0` |
| max base diff across modes | `0.0` |
| max first84 input diff across modes | `0.0` |
| max acc_root-vs-acc_mixed last18 diff | `0.1592593193` |
| same target/base/first84 invariant | pass |

Smoke train result:

| Epoch | train loss | val selection loss | weighted val loss |
|---:|---:|---:|---:|
| 1 | `0.0000731783` | `0.0000374396` | `0.0000806001` |

Smoke module eval:

| Metric | Value |
|---|---:|
| p_leaf_joint_R L1 cm | `0.447489` |
| p_leaf_joint_R L2 cm | `1.008317` |
| base p_leaf_joint_R L2 cm | `1.008697` |
| control p_leaf_joint_R L2 cm | `1.020434` |
| gR1 angle deg | `0.456831` |
| base gR1 angle deg | `0.454811` |
| control gR1 angle deg | `0.459057` |

Artifacts:

```text
full summary: data/experiments/newpl_joint_leaf_acc_20260619/full/summary.json
full summary md: data/experiments/newpl_joint_leaf_acc_20260619/full/summary.md
full eval JSONs: data/experiments/newpl_joint_leaf_acc_20260619/full/eval/*.json
full logs: data/experiments/newpl_joint_leaf_acc_20260619/full/logs/run.log
data/experiments/newpl_joint_leaf_acc_20260619/smoke/cache_validation.json
data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_baseline_jointtarget_84D/pl_curve_cache_manifest.json
data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_acc_root_102D/pl_curve_cache_manifest.json
data/experiments/newpl_joint_leaf_acc_20260619/smoke/caches/dip_val_acc_mixed_102D/pl_curve_cache_manifest.json
data/experiments/newpl_joint_leaf_acc_20260619/smoke/train_baseline_jointtarget_84D/best_loss.pt
data/experiments/newpl_joint_leaf_acc_20260619/smoke/train_baseline_jointtarget_84D/train_result.json
data/experiments/newpl_joint_leaf_acc_20260619/smoke/eval_baseline_jointtarget_84D.json
data/experiments/newpl_joint_leaf_acc_20260619/smoke/eval_baseline_jointtarget_84D.md
```

Decision:

```text
Implementation/smoke gate passed, then the full AMASS -> DIP module-only run
completed under data/experiments/newpl_joint_leaf_acc_20260619/full.
Final conclusion is negative: the acceleration modes do not reduce
p_leaf_joint_R L2 on DIP test or TotalCapture test. They improve gR1 by a
negligible amount only, so they fail the planned effective criterion.
No IK/full-pipeline run should be attached to this result because the 15D
semantic is joint-leaf, not legacy vertex pRB.
```

Full-run final same-cache module metrics:

| Stage | Mode | p_leaf_joint_R L2 cm | gR1 deg | Delta p_leaf L2 vs baseline | Delta gR1 vs baseline |
|---|---|---:|---:|---:|---:|
| DIP test after AMASS | baseline_jointtarget_84D | `5.702977` | `2.343299` | `0.000000` | `0.000000` |
| DIP test after AMASS | acc_root_102D | `5.703926` | `2.343206` | `+0.000949` | `-0.000093` |
| DIP test after AMASS | acc_mixed_102D | `5.703530` | `2.343286` | `+0.000553` | `-0.000013` |
| TC test after AMASS | baseline_jointtarget_84D | `5.174403` | `4.200520` | `0.000000` | `0.000000` |
| TC test after AMASS | acc_root_102D | `5.176007` | `4.200364` | `+0.001604` | `-0.000157` |
| TC test after AMASS | acc_mixed_102D | `5.176037` | `4.200342` | `+0.001634` | `-0.000178` |
| DIP test after DIP FT | baseline_jointtarget_84D | `5.702842` | `2.343188` | `0.000000` | `0.000000` |
| DIP test after DIP FT | acc_root_102D | `5.703787` | `2.343102` | `+0.000945` | `-0.000086` |
| DIP test after DIP FT | acc_mixed_102D | `5.703366` | `2.343174` | `+0.000524` | `-0.000014` |
| TC test after DIP FT | baseline_jointtarget_84D | `5.174598` | `4.200546` | `0.000000` | `0.000000` |
| TC test after DIP FT | acc_root_102D | `5.176143` | `4.200393` | `+0.001545` | `-0.000153` |
| TC test after DIP FT | acc_mixed_102D | `5.176165` | `4.200376` | `+0.001567` | `-0.000170` |

### Protocol repair 2026-06-20: official PL-s1 base

Critical bug found after the full run: `pl_joint_leaf_acc_cache.py` generated
`pl_base` from `pose_prephysics` FK, making the base too strong and invalid for
comparison with old NewPL v5 caches. The corrected builder now generates
`pl_base` from official GPNet PL-s1 (`GPNet.plnet`) on the legacy 84D IMU input,
initialized with the legacy first-frame PL target. It explicitly does not use
`pose_prephysics` for `pl_base`.

Code changes:

```text
pl_joint_leaf_acc_cache.py
  pl_base_source: official_pl_s1
  base_mode: official_pl_s1_prediction
  protocol_check.comparable_to_v5: true

scripts/validate_pl_base_protocol.py
  rejects non-official base source, base_gR1 < 5 deg, or pose_prephysics base pipeline

scripts/eval_newpl_joint_leaf_acc.py
  emits pl_base_source, pl_target_source, evaluation_protocol_version, protocol_check
```

Validation commands:

```bash
export LD_LIBRARY_PATH=/home/lingfeng/.conda/envs/globalpose-gpu/lib:${LD_LIBRARY_PATH:-}
PY=/home/lingfeng/.conda/envs/globalpose-gpu/bin/python

$PY pl_joint_leaf_acc_cache.py build --input-cache data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json --gt-control-cache /home/lingfeng/projects/data/dataset_work/dip/gt_control/test/manifest.json --output-dir data/experiments/newpl_joint_leaf_acc_20260619/full/caches/baseline_jointtarget_84D/dip_test --feature-mode baseline_jointtarget_84D --shard-size 100 --device cuda:0
$PY pl_joint_leaf_acc_cache.py build --input-cache data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json --gt-control-cache /home/lingfeng/projects/data/dataset_work/dip/gt_control/test/manifest.json --output-dir data/experiments/newpl_joint_leaf_acc_20260619/full/caches/acc_root_102D/dip_test --feature-mode acc_root_102D --shard-size 100 --device cuda:0
$PY pl_joint_leaf_acc_cache.py build --input-cache data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json --gt-control-cache /home/lingfeng/projects/data/dataset_work/dip/gt_control/test/manifest.json --output-dir data/experiments/newpl_joint_leaf_acc_20260619/full/caches/acc_mixed_102D/dip_test --feature-mode acc_mixed_102D --shard-size 100 --device cuda:0

$PY scripts/validate_pl_base_protocol.py --cache data/experiments/newpl_joint_leaf_acc_20260619/full/caches/baseline_jointtarget_84D/dip_test/pl_curve_cache_manifest.json --raw-cache data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json --output-json data/experiments/newpl_joint_leaf_acc_20260619/full/eval/base_protocol_validation_baseline_jointtarget_84D_dip_test.json --device cuda:0
$PY scripts/validate_pl_base_protocol.py --cache data/experiments/newpl_joint_leaf_acc_20260619/full/caches/acc_root_102D/dip_test/pl_curve_cache_manifest.json --raw-cache data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json --output-json data/experiments/newpl_joint_leaf_acc_20260619/full/eval/base_protocol_validation_acc_root_102D_dip_test.json --device cuda:0
$PY scripts/validate_pl_base_protocol.py --cache data/experiments/newpl_joint_leaf_acc_20260619/full/caches/acc_mixed_102D/dip_test/pl_curve_cache_manifest.json --raw-cache data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json --output-json data/experiments/newpl_joint_leaf_acc_20260619/full/eval/base_protocol_validation_acc_mixed_102D_dip_test.json --device cuda:0
```

Corrected DIP test cache validation:

| Mode | base source | base gR1 angle deg | protocol valid |
|---|---|---:|---|
| baseline_jointtarget_84D | `official_pl_s1` | `15.267228` | true |
| acc_root_102D | `official_pl_s1` | `15.267228` | true |
| acc_mixed_102D | `official_pl_s1` | `15.267228` | true |

Cross-check: old v5 official DIP test cache
`data/experiments/newpl_v5_official_protocol_20260607/caches/pl_dip_test_official_init36/pl_curve_cache_manifest.json`
also has `base_gR1=15.267228 deg`, so the corrected joint-leaf cache now matches
the official base protocol. The value is slightly above the initial 15 deg
expectation because that is the measured old v5 cache value.

Three-mode invariant after corrected DIP test rebuild:

| Check | Value |
|---|---:|
| shared sequences | `19` |
| max target diff | `0.0` |
| max base diff | `0.0` |
| max first84 input diff | `0.0` |
| max acc_root-vs-acc_mixed last18 diff | `30.88434410095215` |

Decision: the original full-run metrics in this section are retained only as a
historical invalid protocol record. Do not use them for v5 comparability or
method effectiveness claims. Regenerate all train/val/test caches and rerun
training before evaluating acceleration features.
<!-- END newpl-joint-leaf-acc--2026-06-19 -->

### EXP-20260704-totalcapture-imu-vs-vertex-diff-acc - TotalCapture IMU acceleration vs SMPL IMU-vertex finite-difference acceleration

Question: on TotalCapture, how close is the current repository's IMU acceleration to the second-difference acceleration of the five GlobalPose IMU vertices, and is the vertex signal credible as a later acceleration explainability/supervision target?

Change tested: added a pure diagnostic script only. No model training, no PL/IK/VR/network changes.

Changed files:

```text
code/tools/compare_totalcapture_imu_acc_vs_vertex_diff_acc.py
PROJECT_STATUS.md
EXPERIMENT_LOG.md
```

Dataset/split:

```text
data/dataset_work/TotalCapture_globalpose_official/test.pt
sequences: s5_freestyle1, s5_freestyle3, s5_rom3, s5_walking2
FPS: 60
```

Command:

```bash
python -m py_compile code/tools/compare_totalcapture_imu_acc_vs_vertex_diff_acc.py
python code/tools/compare_totalcapture_imu_acc_vs_vertex_diff_acc.py --split test
```

Coordinate and gravity contract:

```text
Raw TotalCapture field `aS` is treated as sensor-frame specific force.
R_WS = RIM^T @ RIS.
Sensor-like vertex comparison: acc_sensor_like = R_WS^T @ (acc_vertex_world - gravity_world).
World/model comparison: aM = RIM^T @ RIS @ aS + gravity_world, compared to acc_vertex_world.
Vertex positions come from fk_imu_joints_and_vertices(pose, tran), so root translation is included and no root-relative subtraction is applied.
IMU vertex ids are read from l4_sensor_offset_utils.IMU_VERTICES, not hand-entered in the result logic.
Compared five leaf vertices: left_forearm=1961, right_forearm=5424, left_lower_leg=1176, right_lower_leg=4662, head=411.
```

Artifacts:

```text
root: code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409
summary: code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409/SUMMARY.md
overall CSV: code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409/summary_overall.csv
per-sensor CSV: code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409/summary_per_sensor.csv
per-sequence CSV: code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409/summary_per_sequence.csv
frame-level CSV: code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409/frame_level_metrics.csv
GitHub committed copy: code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409/frame_level_metrics.csv.gz
config: code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409/config.json
vertex ids: code/outputs/totalcapture_imu_vs_vertex_diff_acc_20260704_134409/vertex_ids.json
plots: error_bar_rmse.png, corr_bar.png, timeseries_examples_*.png, scatter_*.png, residual_hist_*.png, boxplot_residuals.png
```

Main sensor-specific-force results:

| Method | mean L2 m/s^2 | RMSE m/s^2 | Pearson | cosine | magnitude MAE m/s^2 |
|---|---:|---:|---:|---:|---:|
| raw FD | `2.836418` | `3.335955` | `0.895983` | `0.896658` | `1.300606` |
| SavGol-9/poly3 + FD | `2.280664` | `2.742093` | `0.926891` | `0.927408` | `1.071608` |
| SavGol-15/poly3 + FD | `2.372107` | `2.871197` | `0.919483` | `0.920057` | `1.155638` |

Sensor readout for selected SavGol-9/poly3:

```text
worst sensor: right_lower_leg vertex 4662, RMSE 3.956423 m/s^2
best sensor: head vertex 411, RMSE 1.118298 m/s^2
largest aggregate bias sensor: left_lower_leg, bias (-0.3226, -0.2460, 0.3693) m/s^2
worst sequence: s5_freestyle3, RMSE 4.6446 m/s^2
```

Interpretation: SavGol-9 position smoothing before centered second difference is closest. The sensor-frame comparison has much higher correlation than the world/model-frame comparison, matching the raw `aS` specific-force contract. Residual p95 remains high, especially on lower legs, so the remaining error is not only a constant bias; it likely includes attachment/site mismatch, soft-tissue or strap motion, finite-difference noise/outliers, and possibly sequence/sensor time alignment.

Claim support: bounded diagnostic.

Conclusion: partially supports using GlobalPose five-vertex finite-difference acceleration as an IMU acceleration explainability target, especially with SavGol-9 smoothing and explicit sensor-frame gravity handling. It is not yet strong enough to use blindly as a supervision target; lower-leg offsets, residual outliers, and timing/calibration should be audited before training against it.

### EXP-20260704-totalcapture-imu-vs-rjs-diff-acc - TotalCapture IMU acceleration vs rJS IMU-site finite-difference acceleration

Question: does the existing sequence-level `r_JS` IMU-site position explain TotalCapture IMU acceleration better than the five GlobalPose IMU vertices from `EXP-20260704-totalcapture-imu-vs-vertex-diff-acc`?

Change tested: added a pure diagnostic script only. No model training, no PL/IK/VR/network changes, and no new rJS estimation.

Changed files:

```text
code/tools/compare_totalcapture_imu_acc_vs_rjs_diff_acc.py
PROJECT_STATUS.md
EXPERIMENT_LOG.md
```

Dataset/split:

```text
data/dataset_work/TotalCapture_globalpose_official/test.pt
sequences: s5_freestyle1, s5_freestyle3, s5_rom3, s5_walking2
FPS: 60
```

rJS source:

```text
data/experiments/footlock_transpose_rjs_smoothacc_20260609/totalcapture_test_footlock_transpose_rjs.pt
path selection: default_auto_search
field used by load_offset_cache: offset
sequence-specific rJS: yes, all 4 TotalCapture test sequences present
contract: r_JS is the IMU origin position relative to mapped joint J, expressed in joint-local coordinates; p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS
```

Command:

```bash
python -m py_compile code/tools/compare_totalcapture_imu_acc_vs_rjs_diff_acc.py
python code/tools/compare_totalcapture_imu_acc_vs_rjs_diff_acc.py
```

Artifacts:

```text
root: code/outputs/totalcapture_imu_vs_rjs_diff_acc_20260704_141121
summary: code/outputs/totalcapture_imu_vs_rjs_diff_acc_20260704_141121/SUMMARY.md
overall CSV: code/outputs/totalcapture_imu_vs_rjs_diff_acc_20260704_141121/summary_overall.csv
per-sensor CSV: code/outputs/totalcapture_imu_vs_rjs_diff_acc_20260704_141121/summary_per_sensor.csv
per-sequence CSV: code/outputs/totalcapture_imu_vs_rjs_diff_acc_20260704_141121/summary_per_sequence.csv
frame-level CSV local: code/outputs/totalcapture_imu_vs_rjs_diff_acc_20260704_141121/frame_level_metrics.csv
frame-level CSV committed: code/outputs/totalcapture_imu_vs_rjs_diff_acc_20260704_141121/frame_level_metrics.csv.gz
rJS offsets: code/outputs/totalcapture_imu_vs_rjs_diff_acc_20260704_141121/rjs_offsets.json
vertex comparison plot: code/outputs/totalcapture_imu_vs_rjs_diff_acc_20260704_141121/vertex_vs_rjs_rmse_bar.png
```

Main sensor-specific-force results:

| Method | mean L2 m/s^2 | RMSE m/s^2 | Pearson | cosine | magnitude MAE m/s^2 |
|---|---:|---:|---:|---:|---:|
| raw FD | `4.188702` | `4.843998` | `0.787203` | `0.787390` | `1.884606` |
| SavGol-9/poly3 + FD | `3.425655` | `3.856080` | `0.853215` | `0.853296` | `1.560262` |
| SavGol-15/poly3 + FD | `3.310745` | `3.689875` | `0.864366` | `0.864460` | `1.559029` |

Direct SavGol-9 vertex-vs-rJS comparison:

| Sensor | Vertex RMSE | rJS RMSE | Delta vertex-rJS |
|---|---:|---:|---:|
| left_forearm | `1.6518` | `4.6840` | `-3.0322` |
| right_forearm | `2.2542` | `4.4866` | `-2.2324` |
| left_lower_leg | `3.5891` | `3.4906` | `+0.0985` |
| right_lower_leg | `3.9564` | `4.1698` | `-0.2133` |
| head | `1.1183` | `1.6451` | `-0.5268` |

Interpretation: rJS does not improve the overall TotalCapture IMU acceleration match relative to the five-vertex baseline. The best rJS method is SavGol-15, but even rJS SavGol-15 overall RMSE `3.689875` is worse than the prior vertex SavGol-9 overall RMSE `2.742093`. The only per-sensor SavGol-9 improvement is a small left-lower-leg gain; right lower leg regresses.

Claim support: bounded diagnostic.

Conclusion: do not promote the current footlock/pseudo rJS position acceleration as a stronger IMU acceleration explainability or supervision target than the five-vertex baseline. If revisited, audit rJS coordinate convention, TotalCapture mount mismatch, time alignment/filtering, and lower-leg/right-leg residuals before training against it.
