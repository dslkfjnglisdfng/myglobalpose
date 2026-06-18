# GlobalPose Project Status

## ACTIVE SUMMARY

Current stage: acceleration residual audit, leaf-relative/root-reference only
Current task: record acc_leaf_relative_residual_v3_20260618 results, keep docs current, and push
Review state: Approved for next step
Current changed files: scripts/build_acc_leaf_relative_residual_v3_20260618.py, scripts/validate_acc_leaf_relative_residual_v3_20260618.py, data/experiments/acc_leaf_relative_residual_v3_20260618/*, PROJECT_STATUS.md, RECENT_REPLACEMENT_VERSIONS.md, EXPERIMENT_LOG.md
Current module: acc_leaf_relative_residual_v3_20260618
Current replacement version: residual-audit-only, no model retraining
Current experiment: leaf-only acceleration residual audit on AMASS/DIP/TotalCapture; root IMU index 5 is reference only and excluded from metrics
Current result: 1404 sequences / 1609025 valid frames; ALL raw leaf-relative L2/RMSE/corr = 1.372700 / 2.269143 / 0.857988; ALL smooth leaf-relative = 0.457061 / 0.562864 / 0.979057
Current blocker: none
Next action: stage docs/code and commit to GitHub
Git state: dirty worktree with existing unrelated edits plus new AccCurve v1 TC eval files
CodeGraph state: healthy indexed native backend
Detailed logs: data/experiments/acc_leaf_relative_residual_v3_20260618/summary.md and metrics.json

## 0. Version-Line Reading Guide

This status file is now organized by version line first, then by experiment date. Use this section as the current map; use `RECENT_REPLACEMENT_VERSIONS.md` for the version ledger and `EXPERIMENT_LOG.md` for full command/log/JSON evidence.

Important metric rule:

```text
Every reported metric must name its experiment root/cache/protocol.
Do not compare same-named checkpoints across different caches as if they were the same baseline.
Example: `newpl_v5_dip_best` under
  data/experiments/newpl_v5_official_protocol_20260607_tuned
has DIP gR1=12.552613 deg, while later offset/smooth/cache-family summaries can show
different `newpl_v5_dip_best` numbers such as gR1=14.801741 deg.
Those are different metric namespaces unless re-evaluated on the same cache.
```

Current version-line map:

| Version line | Scope | Current selected / status | Read first |
|---|---|---|---|
| `PL-s1 / historical processed` | PL replacement under processed/TC-oriented route | `newpl_v4_init36` remains best historical full S4 artifact; DIP full-pipeline 11 metrics do not beat official GPNet | `RECENT_REPLACEMENT_VERSIONS.md` section `PL-s1 Replacement Versions` |
| `PL-s1 / official-route v5 family` | AMASS -> DIP, DIP/TC module eval, no TC fine-tune | no selected replacement; v5 variants are diagnostic | `newpl_v5_official_protocol`, loss-family, smoothacc, butteracc, realtime smooth+residual |
| `PL-s1 / predictive/root/offset variants` | NewPL-root, next-control, offset/acc-aux, learned offset | diagnostic only unless explicitly promoted later | `newpl_root_v1`, `newpl_v6_next_control`, `newpl_v6_next_control_smoothacc_gR1`, `newpl_v6_next_p_pdot_pddot_strong`, `newpl_offset_v6`, `newpl_v7/v7b` |
| `IK-s1` | IK1 replacement preserving `pRJ[69]+gR2[3]` | none selected; best S4 still behind PL-only init36 | `newik1_v6`, v8/v9 adaptive search, v10/v11/v14 notes |
| `IK-s2 / NewPose` | IK2/pose-control replacement | rejected so far | `newpose_ctrl_v1/v2` |
| `IMU offset / r_JS data line` | DIP/TC pseudo offset synthesis and audits | active route is `footlock_transpose_v1` only | `Active r_JS Policy` |
| `Diagnostic IMU control modules` | neighbor velocity/position and joint q/qdot/velocity control diagnostics | diagnostic only; not connected to PL/IK/full pipeline | `imu_neighbor_vel_ctrl_v1`, `imu_neighbor_pos_from_vel_ctrl_v1`, `imu_joint_euler_qdot_vel_ctrl_v1` |
| `AccCurve / acceleration residual` | v1 diff-pos, v1 zero-trans TC eval, v2 strict GTFK diagnostics, and leaf-relative residual audit | current audit is residual-only: compare `aM_leaf-aM_root` vs `diff_acc(p_leaf_zero_trans)-diff_acc(p_root_zero_trans)` on leaf sensors 0..4; smoothing strongly lowers residual; no downstream pipeline claim | `acc_leaf_relative_residual_v3_20260618`, `acc_curve_v1_totalcapture_eval_20260618`, `acc_curve_v1_totalcapture_zero_trans_eval_20260618`, `acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617` |
| `Experiment evidence` | commands, logs, JSONs, failures | archive only, not current status | `EXPERIMENT_LOG.md` detailed log index |

Latest PL-s1 full-pipeline eval note on 2026-06-16:

```text
newpl_v4_init36 DIP test full-pipeline 11 metrics were backfilled under:
  data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616
Evaluator:
  newik1_real_streaming_audit.py
Cache/protocol:
  data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
Checkpoint:
  data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
Replacement:
  PL-s1 only; official IK-s1, IK-s2, VR, and carticulate physics downstream
  preserved. DIP trans/root-velocity supervision was not used.
Result:
  official_gpnet Score 44.641437
  newpl_v4_init36_official_downstream Score 44.708897
  delta +0.067461 (worse)
Conclusion:
  不支持 DIP full-pipeline improvement claim. These are full-pipeline 11
  metrics, not PL module-level pRB/gR1.
```

Latest PL-s1 implementation note on 2026-06-16:

```text
newpl_v6_next_p_pdot_pddot_strong is implemented, smoke-tested, and full
AMASS->DIP trained/evaluated at module level.
It preserves the v6 next-control model and downstream PL contract, but changes
training to decoded next p/pd/pdd pRB[15] supervision with train-cache RMS
normalization and best_p_pdot_pddot_strong.pt selection.
Full root: data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full
Full result: AMASS best composite 0.6011257469 at epoch 70; DIP best composite
0.9176913500 at epoch 39. Important correction: this composite supervises and
selects next-frame decoded p/pdot/pddot, not current-frame p/pdot/pddot.
Current-frame eval was added under:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval
DIP current p/pdot/pddot L2 for strong best:
  6.465181 cm / 31.419971 cm/s / 1792.008308 cm/s^2
Best same-cache baselines:
  6.451342 cm / 31.419971 cm/s / 1791.925604 cm/s^2
TotalCapture current p/pdot/pddot L2 for strong best:
  7.253737 cm / 29.581992 cm/s / 1618.041598 cm/s^2
Best same-cache baselines:
  6.879507 cm / 29.581992 cm/s / 1616.539405 cm/s^2
Decision: diagnostic only. It fails the current-frame non-regression gate on
DIP p and pddot and on TotalCapture p and pddot. No full-pipeline 11 metrics
were run; do not promote it.

Velocity/acceleration metric audit was added under:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit
Audit result: the large velocity error is not explained by GT decoder mismatch,
actual dt mismatch, current-frame temporal shift, or boundary masks. GT control
decode gives dot/ddot L2 = 0 on DIP/TC; manifest/eval dt are both 1/60; current
finite-difference velocity/acceleration best shift is 0. The large anomaly is
specific to next-head derivatives (`next_pldot`/`next_plddot`) with nonzero
best shifts; classify as diagnostic B, next-head temporal/source mismatch.
Do not train or promote a new model from this result.
```

Latest AccCurve module note on 2026-06-17:

```text
acc_curve_v1_20260617 is the earlier standalone acceleration-level module with
an `aFK_smooth` cache built from position finite differences.
acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617 is the strict version.
Input:
  aM_raw[18] + aM_smooth[18] + (aM_raw-aM_smooth)[18] + wM[18] + RMB_6d[36] = 108D
Base:
  aM_smooth[18]
Output:
  pred_aM_curve[18]
Target:
  aFK_gtfk_smooth[18] from centered smooth(GTFKacc(q,qdot,qddot,rJS))
Training:
  AMASS pretrain -> DIP finetune with fixed windows 240/120, batch 64
Smoke:
  2 AMASS + 2 DIP records, 1 epoch each, zero-init pred-base = 0
Full run:
  AMASS best epoch 29, DIP best epoch 19
DIP val:
  pred_l2=1.990773, base_l2=3.102919, pred_base_ratio=0.757471
DIP test:
  pred_l2=2.997944, base_l2=3.958857, pred_base_ratio=0.778348
Conclusion:
  strict module-level regression improvement only; not a PL/full-pipeline claim.
```

Latest AccCurve-to-PL input evaluation note on 2026-06-17:

```text
Experiment:
  data/experiments/acc_curve_pl_input_eval_20260617
Evaluator:
  scripts/eval_pl_with_acc_curve_input_20260617.py
Frozen PL:
  data/weights.pt, official GPNet.plnet only
DIP test cache/protocol:
  data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
Contract:
  replace only PL acceleration block aRB[18]; keep wRB[18]+RRB[45]+gR0[3],
  target, mask, sequence split, and PL checkpoint fixed.
Frame:
  AccCurve outputs are model/world-frame M acceleration and are converted to
  PL root frame by aRB = acc_M @ RMB_root before PL forward.
Validation:
  official_raw_acc vectorized 84D feature vs pl_input_feature max diff
  7.6293945e-06; non-acc 66D block max diff 0.
Result on DIP test:
  official_raw_acc pRB 6.529110 cm, gR1 15.267153 deg
  smooth_acc pRB 6.462386 cm, gR1 15.216247 deg
  acc_curve_v1_pred pRB 6.967961 cm, gR1 15.036875 deg
  acc_curve_v2_gtfk_pred pRB 8.347050 cm, gR1 15.229429 deg
Decision:
  smooth_acc helps frozen PL pRB and gR1. AccCurve v1/v2 do not improve pRB,
  so AccCurve acceleration-level gains do not transfer into a simultaneous
  PL module-output gain. Do not connect AccCurve to PL as-is.
```

Latest AccCurve v1 TotalCapture acceleration-level eval note on 2026-06-18:

```text
Experiment:
  data/experiments/acc_curve_v1_totalcapture_eval_20260618
Cache root:
  code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test
Evaluator:
  scripts/eval_acc_curve_v1_totalcapture_20260618.py
Checkpoint:
  data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt
Source TC cache/protocol:
  data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json
Target:
  v1 smooth(diff_acc(p_WS)), p_WS = p_WJ + R_WJ @ rJS; field aFK_smooth[18]
Frame:
  input/base/pred/target are model/world frame M
Scope:
  acceleration-level only; no AccCurve training, no PL retraining, no IK/VR/full pipeline, no S4
Result on TotalCapture test:
  aM_smooth base L2/RMSE/corr = 0.873843 / 0.693060 / 0.974734
  AccCurve v1 pred L2/RMSE/corr = 2.091960 / 1.539445 / 0.866428
  pred/base ratio = 2.393977 over 16084 valid frames
DIP v1 historical:
  pred/base ratio = 0.622049, pred L2 = 1.202067, base L2 = 2.368697, corr = 0.940837
Decision:
  v1 does not beat aM_smooth on TotalCapture and the ratio gap versus DIP is
  +1.771928. Do not proceed with v1 acceleration as a cross-dataset NewPL
  retrain input without revising the acceleration module or adding a stronger gate.
```

Latest AccCurve v1 TotalCapture zero-trans acceleration-level eval note on 2026-06-18:

```text
Experiment:
  data/experiments/acc_curve_v1_totalcapture_zero_trans_eval_20260618
Cache root:
  code/outputs/smooth_acc_cache_totalcapture_v1_zero_trans_20260618/tc_test
Evaluator:
  scripts/eval_acc_curve_v1_totalcapture_20260618.py
Checkpoint:
  data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt
Source TC cache/protocol:
  data/experiments/newpl_v5_official_protocol_20260607/caches/tc_test_official_with_offset_r/baseline_cache_manifest.json
Target:
  v1 smooth(diff_acc(p_WS_zero_trans)); p_WS = p_WJ + R_WJ @ rJS with tran forced to zero
Frame:
  input/base/pred/target are model/world frame M
Scope:
  acceleration-level only; no AccCurve training, no PL retraining, no IK/VR/full pipeline, no S4
Result on TotalCapture test:
  aM_smooth base L2/RMSE/corr = 1.832642 / 1.466451 / 0.883554
  AccCurve v1 pred L2/RMSE/corr = 1.415560 / 0.977232 / 0.945382
  pred/base ratio = 0.772415 over 16084 valid frames
DIP v1 historical:
  pred/base ratio = 0.622049, pred L2 = 1.202067, base L2 = 2.368697, corr = 0.940837
TC full-trans previous:
  pred/base ratio = 2.393977, pred L2 = 2.091960, base L2 = 0.873843, corr = 0.866428
Decision:
  zero-trans alignment removes the TC failure seen in the source-translation eval.
  The TC target-definition mismatch was the main issue, not a complete collapse
  of v1 residual structure on TotalCapture. Continue considering zero-trans v1
  acceleration as a NewPL retrain input candidate, but keep same-cache PL gates.
```

## 0.4 AccCurve / absolute acceleration residual module

Status: implemented, smoke-tested, and full AMASS -> DIP trained/evaluated at module level on 2026-06-17. Diagnostic standalone only.

Contract:

```text
module: acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617
input: aM_raw[18] + aM_smooth[18] + (aM_raw-aM_smooth)[18] + wM[18] + RMB_6d[36] = 108D
base: aM_smooth[18]
output: pred_aM_curve[18]
target: aFK_gtfk_smooth[18] from strict GTFKacc(q,qdot,qddot,rJS)
frame: model/world frame M
```

Implementation and smoke:

```text
files: acc_curve.py, acc_curve_train.py, scripts/build_acc_curve_gtfk_cache.py, scripts/run_acc_curve_v2_gtfk_20260617.sh
cache root: code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617
experiment root: data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617
smoke cache: code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617_smoke
smoke result: 2 AMASS + 2 DIP records, 1 epoch each, pred-base zero-init verified, window steps matched window count
```

Module-level results:

| Split | pred L2 | base L2 | pred/base ratio | corr | cosine | mag MAE | residual std | residual p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP val | `1.990773` | `3.102919` | `0.757471` | `0.821207` | `0.488962` | `1.060400` | `1.950429` | `6.949836` |
| DIP test | `2.997944` | `3.958857` | `0.778348` | `0.792049` | `0.493421` | `1.639596` | `2.357390` | `8.619020` |

Training summary:

| Stage | Train seq | Val seq | Train windows | Val windows | Best epoch | Best selection |
|---|---:|---:|---:|---:|---:|---:|
| AMASS pretrain | `1231` | `67` | `8231` | `407` | `29` | `0.8150924359` |
| DIP finetune | `36` | `6` | `1887` | `253` | `19` | `0.7294550687` |

Known risks:

- This module only improves the strict GTFK acceleration regression target against its smoothed base.
- It is not connected to PL/IK/VR or full-pipeline 11 metrics.
- Cache size is large; avoid staging `data/experiments/` or `code/outputs/` wholesale.

## 0.1 IMU Neighbor Velocity-Control Module v1

Status: implemented at code/smoke level; no long AMASS/TotalCapture/DIP training has been launched yet.

Contract:

```text
module: imu_neighbor_vel_ctrl_v1
input: aM[18] + wM[18] + RMB_6d[36] + r_JS[18] = 90D
output: neighbor_vel_W_control[33]
frame: velocity and acceleration are in world/model frame W
r_JS: IMU origin relative to mapped joint J, expressed in joint-local coordinates
DIP: world velocity/root velocity/acceleration GT is not available and must not be synthesized from DIP trans
```

Output node groups:

| Sensor | Mapped joint | Output nodes | Channels |
|---|---:|---|---:|
| left_forearm | 18 | 18, 20 | 6 |
| right_forearm | 19 | 19, 21 | 6 |
| left_lowerleg | 4 | 4, 7 | 6 |
| right_lowerleg | 5 | 5, 8 | 6 |
| head | 15 | 12, 15 | 6 |
| pelvis/root | 0 | 0 | 3 |

Loss policy:

| Dataset policy | Supervision |
|---|---|
| AMASS/TotalCapture with reliable `tran_gt` | velocity control, decoded velocity, decoded acceleration, root velocity, root acceleration, segment relative velocity/acceleration, smooth/jerk/prior |
| DIP | no world velocity/acceleration/root-velocity GT; only teacher distill plus smooth/jerk/prior if adaptation is run |

Implementation and smoke:

```text
files: imu_neighbor_vel_ctrl.py, imu_neighbor_vel_ctrl_train.py, imu_neighbor_vel_ctrl_eval.py
compile: python -m py_compile imu_neighbor_vel_ctrl.py imu_neighbor_vel_ctrl_train.py imu_neighbor_vel_ctrl_eval.py
train efficiency: default compact precompute of per-sequence 90D input plus velocity/acc/control targets before epochs; training batches slice cached tensors and do not rerun SMPL FK per batch
baseline velocity eval: use pose_baseline/tran_baseline finite difference when present; otherwise use 3D v_root_vr as root-only official VR baseline; if absent, report baseline velocity not available
smoke: /tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953 synthetic train/eval
smoke result: train/eval finite; output vel/acc/control shapes are [T,33]
DIP smoke: aggregate world_gt_status = "world velocity GT not available"
full-pipeline 11 metrics: not run
formal checkpoints: not available until long training is launched
```

Next step: run the planned AMASS pretrain and TotalCapture fine-tune/eval if this diagnostic module should be tested on real caches. Use batched training only; do not use batch size 1 for long runs.

## 0.2 IMU Neighbor Pos-From-Vel-Control Module v1

Status: implemented, trained, and evaluated on 2026-06-13. Diagnostic only; not connected to PL/IK1/full pipeline and no full-pipeline 11 metrics were run.

Contract:

```text
module: imu_neighbor_pos_from_vel_ctrl_v1
input: imu_feature[90] + neighbor_vel_W_control[33] + decoded neighbor_vel_W[33] + decoded neighbor_acc_W[33] = 189D
imu_feature: aM[18] + wM[18] + RMB_6d[36] + r_JS[18]
output: neighbor_pos_R_control[33]
decoded: pos_R[33], vel_R[33], acc_R[33]
node layout: [18,20], [19,21], [4,7], [5,8], [12,15], [0]
frame: root-relative position frame R, using row-vector contract p_RJ = (p_WJ - p_WR) @ R_WR
root channel: retained for 33D layout; root-relative root position is zero and is not a benefit metric
DIP policy: no DIP trans, no DIP world/root velocity GT, no fabricated finite-difference translation
```

Implementation:

```text
module: imu_neighbor_pos_from_vel_ctrl.py
train: imu_neighbor_pos_from_vel_ctrl_train.py
eval: imu_neighbor_pos_from_vel_ctrl_eval.py
runner: scripts/run_imu_neighbor_pos_from_vel_ctrl_v1_20260613.sh
experiment root: data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613
frozen velocity checkpoint: data/experiments/imu_neighbor_vel_ctrl_v1_longtrain_20260613/amass_pretrain/best_loss.pt
```

Training and loss:

| Stage | Data | Batch | Best epoch | Selection value | Loss policy |
|---|---|---:|---:|---:|---|
| AMASS pretrain | AMASS overlay cache | `1536` | `80` | `0.1038176935` | `ctrl_pos=1, decoded_pos=1, vel_R=0.2, acc_R=0.05, vel_input_consistency=0.05, segment_length=0.05, smooth=0.01, jerk=0.005, control_prior=0.001` |
| TotalCapture fine-tune | TC train/val official offset_r cache | `512` | `58` | `0.1052717926` | same as AMASS; TotalCapture `tran_gt` only used for allowed velocity-input mix, target position is root-relative from pose |
| DIP fine-tune | DIP train/val official offset_r cache | `512` | `30` | `0.1142737220` | no world velocity/root velocity/acceleration GT; `vel_R=0`, `acc_R=0`, `vel_input_consistency=0` |

Checkpoint selection used `ctrl_pos + decoded_pos + 0.1 * vel_R`; smooth/prior/jerk could not dominate best-loss selection. Precompute summaries confirm compact per-sequence precomputation and no per-batch SMPL FK.

Module-level results:

| Dataset / stage | pos_R L1 cm ↓ | pos_R L2 cm ↓ | vel_R L2 cm/s ↓ | acc_R L2 cm/s² ↓ | Segment length err cm ↓ | Baseline pos_R L2 cm ↓ |
|---|---:|---:|---:|---:|---:|---:|
| AMASS after AMASS best | `21.5704` | `48.0025` | `42.6341` | `500.2493` | `28.2466` | `2.3390` |
| TotalCapture test after AMASS best | `21.9064` | `48.7792` | `49.9797` | `570.0504` | `28.2452` | `5.3009` |
| TotalCapture test after TC fine-tune | `21.8107` | `48.5708` | `49.9762` | `569.9742` | `28.2637` | `5.3009` |
| DIP test after AMASS best | `21.1748` | `47.8958` | `54.4563` | `1244.7912` | `28.2476` | `5.5673` |
| DIP test after DIP fine-tune | `21.1186` | `47.7704` | `54.4569` | `1244.7978` | `28.2821` | `5.5673` |

Baseline note: cached official PL and `newpl_v4_init36` outputs do not provide this full 33D neighbor-node layout. Evaluation therefore reports them as `not applicable` for this module and uses `pose_prephysics FK root-relative` as the available same-cache pose-derived baseline. This is not a PL-output comparison and should not be used to claim official PL/newpl_v4 module superiority for the 33D diagnostic target.

Conclusion:

```text
The module is technically implemented and the full AMASS -> TC/DIP evaluation flow is finite.
It is not useful as trained in v1: predicted root-relative neighbor positions are far worse than the pose_prephysics FK baseline on AMASS, TotalCapture, and DIP.
TC fine-tune slightly improves TC pos_R L2 from 48.7792 cm to 48.5708 cm, but remains much worse than the 5.3009 cm baseline.
DIP fine-tune slightly improves DIP pos_R L2 from 47.8958 cm to 47.7704 cm, but remains much worse than the 5.5673 cm baseline.
Velocity controls did not produce a useful root-relative position estimator in this v1 setup.
Do not feed this module into IK/NewIK1 yet. The next useful step would be architectural debugging or changing the target/input design, not pipeline integration.
```

## 0.3 IMU Joint Euler/Qdot/Velocity Control Module v1

Status: implemented, trained, evaluated, and rejected on 2026-06-13 under the original world-RMB input. Post-run code update: the rotation input is now converted to root IMU frame before flattening; this updated root-RMB contract requires a fresh training run before producing comparable metrics. Diagnostic only; it does not replace PL/IK1, does not connect to the full pipeline, and no full-pipeline 11 metrics were run.

Contract:

```text
module: imu_joint_euler_qdot_vel_ctrl_v1
input: official/project-frame aM[18] + wM[18] + root-frame R_rootIMU_sensorIMU_flat[54] = 90D
rotation input: R_rootIMU_sensorIMU = RMB[root_imu=5]^T @ RMB[sensor]
init: PL-style sequence init, init_state = q_RJ_euler[0] + qdot_RJ_euler[0] + vel_RJ[0] = 54D
output heads: q_RJ_euler_control[18] + qdot_RJ_euler_control[18] + vel_RJ_control[18]
decoded: q_RJ_euler, qdot_from_q, qddot_from_q, qdot_RJ, qddot_from_qdot, vel_RJ, acc_RJ
joints: [18, 19, 4, 5, 15, 0]
frame: R_RJ = R_WR^T R_WJ; p_RJ = (p_WJ - p_WR) @ R_WR; vel/acc are finite differences in root frame R
DIP policy: no DIP trans, no DIP world/root velocity GT, no fabricated finite-difference translation
```

Implementation:

```text
module: imu_joint_euler_qdot_vel_ctrl.py
train: imu_joint_euler_qdot_vel_ctrl_train.py
eval: imu_joint_euler_qdot_vel_ctrl_eval.py
runner: scripts/run_imu_joint_euler_qdot_vel_ctrl_v1_20260613.sh
previous experiment root: data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613
new root-RMB default root: data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613
previous summary: data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/summary.json
```

Command and artifacts:

```text
command:
  /home/lingfeng/bin/longrun -- bash scripts/run_imu_joint_euler_qdot_vel_ctrl_v1_20260613.sh

run log:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/logs/run.log

checkpoints for each variant A/B/C/D:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/<variant>/amass_pretrain/best_loss.pt
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/<variant>/totalcapture_finetune/best_loss.pt
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/<variant>/dip_finetune/best_loss.pt

eval JSONs for each variant:
  <variant>/eval/eval_amass_after_amass_best.json
  <variant>/eval/eval_totalcapture_test_after_amass_best.json
  <variant>/eval/eval_totalcapture_test_after_tc_finetune_best.json
  <variant>/eval/eval_dip_test_after_amass_best.json
  <variant>/eval/eval_dip_test_after_dip_finetune_best.json
  <variant>/eval/eval_totalcapture_test_after_dip_finetune_best.json
```

Root-RMB input update:

```text
Code path now uses imu_rootframe_features() in train/eval.
Only the RMB block is converted to root IMU frame; aM/wM keep the selected official/processed project-frame values.
The runner default ROOT is now data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613 so old world-RMB artifacts are not reused.
Root-RMB rerun completed for D_all_balanced only because the user quota is at the hard limit; shared disk precompute and last.pt were disabled.
```

Root-RMB rerun result, `D_all_balanced` only:

| Dataset / stage | Root-RMB rotation deg ↓ | Root-RMB vel L2 cm/s ↓ | Old world-RMB D rotation deg ↓ | Old world-RMB D vel L2 cm/s ↓ | Baseline rotation deg ↓ | Baseline vel L2 cm/s ↓ |
|---|---:|---:|---:|---:|---:|---:|
| AMASS after AMASS | `30.8153` | `29.4072` | `30.5556` | `29.4059` | `4.0610` | `13.1790` |
| TotalCapture after AMASS | `29.7156` | `32.7157` | `29.3217` | `32.7078` | `12.3839` | `19.8320` |
| TotalCapture after TC fine-tune | `29.8054` | `32.7145` | `29.4100` | `32.7040` | `12.3839` | `19.8320` |
| DIP after AMASS | `33.4361` | `39.2532` | `33.2060` | `39.2334` | `5.2618` | `28.3552` |
| DIP after DIP fine-tune | `34.0631` | `39.2021` | `34.2014` | `39.2169` | `5.2618` | `28.3552` |
| TotalCapture after DIP fine-tune | `30.3944` | `32.6989` | `29.8325` | `32.7034` | `12.3839` | `19.8320` |

Root-RMB rerun artifacts:

```text
root: data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613
variant: D_all_balanced
command: GPU_LIST=1 VARIANTS='D_all_balanced' SHARED_PRECOMPUTE=0 KEEP_LAST=0 PREFLIGHT=0 AMASS_BATCH=1024 TC_BATCH=512 DIP_BATCH=512 /home/lingfeng/bin/longrun -- bash scripts/run_imu_joint_euler_qdot_vel_ctrl_v1_20260613.sh
checkpoints:
  D_all_balanced/amass_pretrain/best_loss.pt
  D_all_balanced/totalcapture_finetune/best_loss.pt
  D_all_balanced/dip_finetune/best_loss.pt
summary:
  data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/summary.json
```

Root-RMB conclusion: not improved. Converting the RMB block to root frame slightly worsens AMASS and TotalCapture rotation for `D_all_balanced`; DIP after DIP fine-tune improves only `0.1383 deg` versus old world-RMB D, still far worse than the same-cache baseline. Do not promote this module to IK/NewIK1.

Training efficiency notes:

```text
batch preflight selected AMASS=1024, TotalCapture=512, DIP=512.
shared compact precompute was added before long training so AMASS/TC/DIP targets are not recomputed per variant.
AMASS compact precompute was deleted after both AMASS training jobs loaded it to avoid the active quota limit.
train_log.jsonl was fixed to store aggregate validation only, not per-sequence rows.
```

Loss variants:

| Variant | Main emphasis |
|---|---|
| `A_qctrl_main` | stronger q control and q decoded losses |
| `B_qdot_qddot_strong` | stronger qdot/qddot and qdot-control losses |
| `C_vel_acc_strong` | stronger velocity-control, decoded velocity, and acceleration losses |
| `D_all_balanced` | balanced q/qdot/qddot/velocity/acceleration with stronger consistency |

Best module metrics:

| Dataset / stage | Best variant by rotation | Rotation geodesic deg ↓ | vel_RJ L2 cm/s ↓ | acc_RJ L2 cm/s² ↓ | Baseline rotation deg ↓ | Baseline vel L2 cm/s ↓ |
|---|---|---:|---:|---:|---:|---:|
| AMASS after AMASS | `D_all_balanced` | `30.5556` | `29.4059` | `365.0700` | `4.0610` | `13.1790` |
| TotalCapture after AMASS | `D_all_balanced` | `29.3217` | `32.7078` | `406.7416` | `12.3839` | `19.8320` |
| TotalCapture after TC fine-tune | `A_qctrl_main` | `29.1832` | `32.5447` | `402.6187` | `12.3839` | `19.8320` |
| DIP after AMASS | `C_vel_acc_strong` | `32.2599` | `39.2093` | `975.2344` | `5.2618` | `28.3552` |
| DIP after DIP fine-tune | `C_vel_acc_strong` | `32.6004` | `39.2226` | `975.2334` | `5.2618` | `28.3552` |

Conclusion:

```text
The module learned smooth controls but did not learn joint orientation/velocity controls close enough to GT.
All four loss variants are worse than the same-cache pose_prephysics FK baseline for rotation and velocity on AMASS, TotalCapture, and DIP.
TotalCapture fine-tune helps rotation only slightly in A (TC 30.0296 -> 29.1832 deg) but remains far worse than baseline 12.3839 deg.
DIP fine-tune does not improve the best DIP result (C 32.2599 -> 32.6004 deg).
Acceleration-heavy losses reduce some acceleration/jerk behavior but do not recover physically useful q/qdot/velocity outputs.
Do not feed imu_joint_euler_qdot_vel_ctrl_v1 into IK/NewIK1/full pipeline. Next work should debug target representation and init dependency before any pipeline integration.
```

## 1. Current Best Result

| Rank | Version | Replaced Module | Input Mode | Key Change | S4 Score ↓ | Selected? | Reason |
|---:|---|---|---|---|---:|---|---|
| 1 | newpl_v4_init36 | PL-s1 | processed IMU | 36D stream init | 38.625657482802865 | yes | Best full S4 among current replacements. |
| 2 | newik1_v6_official_input_init36_cascade | IK-s1 | processed IMU + NewPL init36 | v4 official-input cascade fine-tune | 38.649136830300094 | no | Module GT improves, but S4 remains worse than NewPL init36 by 0.023479. |
| 3 | newik1_v9_adaptive_loss_search | IK-s1 | processed IMU + NewPL init36 | v8 B4 continuation, 8-way adaptive loss micro-sweep | 38.693844566687936 | no | Best v9 slightly improves v8 S4, but full pRJ and leaf-pRJ module outputs are farther from GT than official IK1 baseline. |
| 4 | newik1_v8_parallel_adaptive_loss_search | IK-s1 | processed IMU + NewPL init36 | v7 last-control small loss-ratio sweep | 38.69415222530066 | no | Best v8 is slightly better than v7, but worse than NewPL init36 and does not beat official IK1 module-GT baseline. |
| 5 | newpl_v3_gtcontrol_rund | PL-s1 | processed IMU | GT control pRB/gR1 | 38.69484578047692 | superseded | Improved processed baseline, but init36 is better. |
| 6 | newik1_v4_official_input | IK-s1 | processed IMU + Run D PL | official-shape IK1 replacement | 38.70523069866002 | no | Does not beat PL-only Run D / init36. |
| 7 | official_processed | none | processed IMU | RMB-only input correction | 38.753659665048126 | baseline | Processed-input reference. |
| 8 | newik1_v5_last_pl_control | IK-s1 | processed IMU + NewPL init36 | last PL control input | 38.84357685862481 | no | Worse than NewPL init36 by 0.217919. |
| 9 | official_official | none | official IMU | official GPNet | 42.522402 | baseline | Original official reference. |

## 2. Current Mainline

- Current project mainline is changed on 2026-06-07: first reproduce the official baseline training route, then iterate replacement modules under that same route.
- Project-level control-point target policy changed on 2026-06-08: all newly synthesized GT control points now use `derivative_aware_v1` instead of exact position-only spline fitting. The default objective is `wp||S C-x||^2 + wv||D1 C-fd_dot(x)||^2 + wa||D2 C-fd_ddot(x)||^2 + wr||C||^2`, with `wp=1.0`, `wv=0.03`, `wa=0.0003`, `wr=1e-6`, `dt=1/60`. This applies to PL `pRB/gR1`, NewPL-root `pRB/gR1/root_vel`, IK1 `pRJ/gR2`, bone aux controls, and NewPose `RRJ/gR_pose` targets through `fit_uniform_cubic_spline_controls`.
- Canonical dataset-level GT-control caches are now generated under `data/dataset_work/GTControlCache/` instead of overwriting historical experiment caches. The builder is `scripts/build_gt_control_cache.py`, with runner `scripts/run_build_gt_control_caches_20260608.sh`. It stores reusable derivative-aware controls for local pose 6D, unwrapped Euler joint angles, root/body-frame SMPL joint positions, source-cache IMU orientation 6D, and PL `pRB/gR1`. AMASS and TotalCapture store reliable `root_trans_W/root_vel_W_fd`; DIP-IMU explicitly does not store root translation or root velocity GT.
- Historical exact sample reconstruction remains available only as `fit_uniform_cubic_spline_controls_position_only` for audits. Do not use it for new training/cache targets unless explicitly running a legacy comparison.
- Any cache that stores GT control tails must be regenerated before being used for a new experiment: `newik1_control_cache.py` caches (`ik1_target_control_tail`, `pl_control_tail_gt`) and `newpose_ctrl_cache.py` caches (`newpose_target_control_tail`). PL caches that store only `pl_target` do not need their target tensors rewritten, but their manifest now records the derivative-aware control contract and training/eval will synthesize controls with the new function.
- Historical best artifact remains `newpl_v4_init36` / NewPL init36 RunD-style with processed TotalCapture-oriented training, but it is no longer sufficient evidence for a new mainline because it does not follow the official AMASS -> DIP fine-tune generalization protocol.
- The comparison protocol for new PL/IK/VR replacements is now:
  1. train/pretrain on AMASS using the official/baseline input contract where applicable;
  2. fine-tune on DIP-IMU train split, without DIP translation/root-velocity supervision;
  3. evaluate on DIP-IMU test and TotalCapture test;
  4. compare against official PL/IK/VR baselines and the current best replacement modules under the same input/evaluation contract;
  5. only promote a module if its physical module outputs are closer to GT and it does not regress downstream/full-pipeline metrics when those are run.
- Current PL frame input contract remains official PL 84D; PL output contract remains `pRB[15] + gR1[3] = 18D` unless a version explicitly defines a compatible extended head. Init36 only changes stream initialization to `offset_r[18] + pRL[15] + gR0[3] = 36D`.
- `newpl_v5_official_protocol` is the first check under this revised policy. It is not selected: it improves DIP gR1 but does not beat the pRB baselines, and it remains worse than `newpl_v4_init36` on TotalCapture module metrics. Full-pipeline 11 metrics were not run.
- `newpl_v5_loss_family_ablation` completed the requested q/control/qdot/qddot loss-family test. No variant is selected: qddot helps DIP pRB slightly, control-point loss is not robust, qdot is effectively negligible, and the best TotalCapture pRB comes from `q_only` while still losing to `newpl_v4_init36`.
- `newpl_v5_butteracc` completed the realtime causal acceleration-filter gate and the requested forced fc12 longtrain. It applies an order-2 causal Butterworth low-pass to official `aM` only, preserves `wM/RMB`, and evaluates fc8/fc10/fc12 with official PL, `newpl_v4_init36`, and raw `newpl_v5`. No cutoff passed the TotalCapture pRB guard; forced fc12 AMASS -> DIP training also does not recover pRB, so this branch is not selected.
- Current IK1 work is still exploratory. `newik1_v9_adaptive_loss_search` completed 8 micro-finetune variants with `best_loss.pt` and `last.pt` S4/module-GT evaluation for each checkpoint.
- Best v9 S4 is `v9_C8_no_control_dyn/last.pt` at `38.693844566687936`, slightly better than v8 by `0.000307658612724`, but still worse than NewPL init36 by `0.068187083885071`.
- NewIK1 v9 does not satisfy the module-output criterion under NewPL streaming TC validation input: best `pRJ_cm_l2_delta = +0.014219195553042852`, `leaf_pRJ_cm_l2_delta = +0.004263760473665279`, `gR2_angle_delta = -0.005290929126193333`, and `state_l2_delta = +0.00007313516150815602` versus official IK1 baseline.
- NewIK2 / pose-control replacement is unresolved. `newpose_ctrl_v1` completed the official-like AMASS -> DIP route but is rejected: module FK error is about `43.8-45.4 cm` versus `4.6-5.0 cm` for official/newpl_v5 baselines, and full-pipeline scores collapse to `413-432`.
- Real-data IMU position offset / `r_JS` mainline changed on 2026-06-09: the only active DIP/TotalCapture `r_JS` synthesis route is `footlock_transpose_v1`. Older `zero/random/solver_v1/net_v2/hybrid_v3/full_diagnostic/rawlike_se3/generic smoothed LS` routes are retired and their old generated artifacts were deleted. DIP `trans` remains untrusted and is not used as GT.

## 2.1 Active r_JS Policy: footlock_transpose_v1 Only

Contract:

```text
r_JS: IMU origin position relative to mapped joint J, expressed in joint-local coordinates.
p_WS(t) = p_WJ(t) + R_WJ(t) @ r_JS.
DIP r_JS is a pseudo-label only; DIP trans is not trusted and not used as GT.
```

Active method: `footlock_transpose_v1` uses frozen TransPose `pose_s1 + tran_b1` contact logits from raw official `aM/RMB`, selects one winner foot per stance frame with `contact_selection_mode=transpose_winner`, infers root motion by foot lock as `p_WR(t) = -p_WC_zero_tran(t) + constant`, then solves sequence-level lever-arm offset. The stance-side lower-leg sensor is not solved from its own stance window.

Smoothed-acc update: smoothed acceleration is not a separate method. The active builder keeps TransPose contact input raw (`contact_input=raw_official_aM_RMB`) and applies smoothing only inside the lever-arm fit (`fit_input=smoothed_aM_and_zero_translation_FK_window_9`, `derivative_mode=centered`).

Active files and outputs:

```text
builder: scripts/build_imu_position_offsets.py
runner: scripts/run_footlock_transpose_rjs_20260608.sh
historical winner root kept: data/experiments/footlock_transpose_rjs_20260608
new smoothed-fit root: data/experiments/footlock_transpose_rjs_smoothacc_20260609
smoke root: data/experiments/footlock_transpose_rjs_smoothacc_20260609_smoke
```

Validation smoke on 2026-06-09:

```text
DIP test, 1 sequence, 300 frames: all_finite=true, median offset norm=0.179266 m, contact windows=3.
TotalCapture test, 1 sequence, 300 frames: all_finite=true, median offset norm=0.199703 m, contact windows=5.
```

Full cache generation on 2026-06-09:

```text
DIP train: sequences=36, all_finite=true, mean/median/p95=0.182025/0.175146/0.333083 m, fallback sensors=9.
DIP val: sequences=6, all_finite=true, mean/median/p95=0.197334/0.185467/0.418209 m, fallback sensors=1.
DIP test: sequences=19, all_finite=true, mean/median/p95=0.175151/0.169691/0.297217 m, fallback sensors=2.
TotalCapture train: sequences=36, all_finite=true, mean/median/p95=0.164509/0.157523/0.307779 m, fallback sensors=0.
TotalCapture val: sequences=5, all_finite=true, mean/median/p95=0.149411/0.139848/0.269620 m, fallback sensors=0.
TotalCapture test: sequences=4, all_finite=true, mean/median/p95=0.182435/0.199503/0.287975 m, fallback sensors=0.
```

Acceleration explainability audit on 2026-06-09:

```text
script: scripts/audit_smoothed_rjs_acc_controlcurve.py
root: data/experiments/smoothed_rjs_acc_controlcurve_audit_20260609
primary metric: SMPL-contract root-relative lower-body acceleration L2 against aM_smooth_w9.
DIP trans policy: not used; DIP root translation is zeroed for this audit.
RBDL q/qdot/qddot: derivative-aware control-curve smoke only; direct rJS injection into RBDL is not approved because frame equivalence is not proven.
```

Primary result in `m/s^2`:

```text
DIP train: zero=3.479233, old=3.355208, smoothacc=3.524253.
DIP val: zero=3.450705, old=3.284543, smoothacc=3.646945.
DIP test: zero=4.250496, old=4.170551, smoothacc=4.230922.
TotalCapture train: zero=5.628481, smoothacc=5.529490.
TotalCapture val: zero=7.359807, smoothacc=6.957408.
TotalCapture test: zero=5.659256, smoothacc=5.470786.
```

Decision: smoothed-footlock `r_JS` improves acceleration explainability versus zero on TotalCapture train/val/test, but it is not validated as better on DIP. On DIP it is worse than zero on train/val and worse than the June 8 old footlock winner on all three splits. Do not use this result to claim improved DIP pseudo-`r_JS`.

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

## 3. Baseline References

| Baseline | Input | Replaced Module | S4 Score ↓ | Purpose |
|---|---|---|---:|---|
| Official GPNet + official IMU | official `aM/wM/RMB` | none | `42.522402` | Original official reference. |
| Official GPNet + processed IMU | orientation-only / RMB-only processed input | none | `38.753659665048126` | Processed-input baseline for replacement claims. |
| NewPL Run D | processed IMU | PL-s1 | `38.69484578047692` | Historical best PL baseline before init36. |
| NewPL init36 | processed IMU | PL-s1 | `38.625657482802865` | Historical best processed-input artifact and prior IK1 upstream checkpoint. |

Processed IMU convention: `l4_aM == aM`, `l4_wM == wM`, `l4_RMB != RMB`; current gain is from orientation/RMB correction, not changed stored acceleration or gyro.

## 3.1 NewPL v5 Official-Protocol Module Check

Status: completed on 2026-06-07; diagnostic only, not selected.

Protocol: AMASS pretrain on official-input init36 PL cache, then DIP-IMU train fine-tune, then module-level evaluation on DIP test and TotalCapture official-input test. TotalCapture train was not used. DIP `trans` and root velocity GT were not used. Full-pipeline 11 metrics are `not measured`.

Artifacts:

```text
script: scripts/run_newpl_v5_official_protocol_20260607.sh
root: data/experiments/newpl_v5_official_protocol_20260607_tuned
AMASS checkpoint: data/experiments/newpl_v5_official_protocol_20260607_tuned/amass_pretrain/best_loss.pt
DIP checkpoint: data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt
eval JSONs: data/experiments/newpl_v5_official_protocol_20260607_tuned/eval/
summary: data/experiments/newpl_v5_official_protocol_20260607_tuned/summary.json
```

Training outcome: AMASS best epoch `80`, selection value `0.002173126090565347`; DIP fine-tune best epoch `40`, selection value `0.038939811958698556`.

Result summary:

- DIP test after AMASS pretrain: `newpl_v5_amass` has pRB L1 `3.127556 cm`, pRB L2 `6.454484 cm`, gR1 `12.551949 deg`. It is better than official_PL on gR1 by `0.395760 deg`, but worse on pRB L2 by `0.035011 cm`.
- DIP test after DIP fine-tune: `newpl_v5_dip_best` improves to pRB L1 `3.120847 cm`, pRB L2 `6.445578 cm`, gR1 `12.552613 deg`. It remains worse than official_PL on pRB L2 by `0.026105 cm`, and worse than `newpl_v4_init36` on pRB L2 by `0.004131 cm`, while still better than both on gR1.
- TotalCapture test after AMASS pretrain: `newpl_v5_amass` has pRB L1 `3.264119 cm`, pRB L2 `6.783332 cm`, gR1 `13.415420 deg`. It is better than official_PL, but worse than `newpl_v4_init36`.
- TotalCapture test after DIP fine-tune: `newpl_v5_dip_best` has pRB L1 `3.264551 cm`, pRB L2 `6.780749 cm`, gR1 `13.415189 deg`. DIP fine-tune barely changes TC behavior and remains worse than `newpl_v4_init36` by `0.126356 cm` pRB L2 and `0.085658 deg` gR1.

Delay-output check: `newpl_v5` was evaluated with future-output delays `0/1/2` on PL-cache module metrics. Delay did not improve pRB. On DIP, best pRB L2 is still delay0 (`newpl_v5_dip_delay0 = 6.445578 cm`) and remains worse than official_PL (`6.419473 cm`). On TotalCapture, delay0 is also best (`newpl_v5_dip_delay0 = 6.780749 cm`), while delay1/2 regress pRB.

Decision: do not select `newpl_v5_official_protocol` or its 1/2-frame delayed-output variant. `newpl_v4_init36` remains the historical best processed-input artifact, while the active research mainline remains official-route reproduction plus same-protocol module iteration.

## 3.1.1 NewPL v5 loss-family ablation

Status: completed on 2026-06-12; diagnostic only, not selected. Full-pipeline 11 metrics were not run.

Question: under the unchanged NewPL v5 official-input contract, what do q, control-point, qdot, and qddot loss families contribute?

Contract:

```text
PL input shape: 84D unchanged.
Init: offset_r[18] + pRL[15] + gR0[3] = 36D unchanged.
Output: pRB[15] + gR1[3] = 18D unchanged.
Here q means decoded PL state pRB[15] + gR1[3], not full RBDL q75.
Protocol: AMASS pretrain -> DIP-IMU train fine-tune -> DIP test and TotalCapture official-input test.
Selection: pl_physical for every ablation variant.
```

Best after DIP fine-tune:

| Dataset | Best Variant | pRB L2 cm | gR1 angle deg | Baseline Context |
|---|---|---:|---:|---|
| DIP test | `q_control_qddot` | `6.426853` | `12.707329` | better pRB than raw v5/v4, still worse than official_PL `6.419473`; worse gR1 than raw v5 `12.552613` |
| TotalCapture test | `q_only` | `6.753091` | `13.575686` | better pRB than raw v5 `6.780749`, worse than v4 `6.654393`; gR1 worse than official/v4/raw v5 |

Loss-family readout:

| Change | DIP pRB Effect | TotalCapture pRB Effect | Interpretation |
|---|---:|---:|---|
| Remove control loss (`q_only`) | not best; `6.437678` | best; `6.753091` | control supervision is not a reliable generalization gain |
| Add control only (`q_control`) | `6.434509` | `6.763036` | tiny DIP improvement, TC regression |
| Add qdot (`q_qdot`) | `6.437400` | `6.754438` | negligible effect; gradient audit also shows tiny qdot gradients |
| Add qddot (`q_qddot`) | `6.430836` | `6.756564` | helps DIP pRB slightly, does not transfer to TC |
| Add control+qddot | best DIP `6.426853` | regresses TC `6.771322` | over-specializes to DIP module metric |

Gradient audit: q and control gradients are aligned at random/AMASS init (`cos=0.846724`) but only weakly aligned after v5 AMASS pretrain on DIP (`cos=0.227377`). qdot gradients are tiny, and qdot/qddot gradients are opposed (`cos=-0.862329` at init, `-0.739343` after v5 AMASS). qddot is dominated by the `pRB_ddot_smooth` term, not direct acceleration-state supervision.

Artifacts:

```text
runner: scripts/run_newpl_v5_loss_family_variant_20260611.sh
orchestrator tasks: configs/newpl_v5_loss_family_ablation_20260611_tasks.json
summary: data/experiments/newpl_v5_loss_family_ablation_20260611/summary.json
csv: data/experiments/newpl_v5_loss_family_ablation_20260611/summary_eval_rows.csv
gradient audit: newpl_v5_loss_gradient_audit.py
logs: logs/orchestrator/newpl_v5_loss_family_ablation_20260611/
```

Decision: do not promote any loss variant. If the next experiment continues from this branch, prioritize qddot only as a weak DIP pRB regularizer and treat control-point loss as a tunable prior, not a required objective.

## 3.1.2 NewPL v5 smooth-acc input experiment

Status: completed on 2026-06-12; diagnostic only, not selected. Full-pipeline 11 metrics were not run.

Question: can NewPL v5 improve if official `aM` is replaced by centered smoothed acceleration, while preserving the same PL output contract and loss?

Contract:

```text
PL input shape: 84D unchanged.
Input change: source raw-cache `aM` is replaced by centered moving-average smoothed `aM` with window=9.
Audit field: `aM_raw` preserves the original source acceleration.
Unchanged fields: `wM`, `RMB`, `pose_gt`, `tran_gt`, `offset_r`, PL target/control target.
PL output: pRB[15] + gR1[3] = 18D unchanged.
Init: offset_r[18] + pRL[15] + gR0[3] = 36D unchanged.
Loss: same NewPL v5 pRB/gR1/control/dynamics loss; best checkpoint selected by `control_physical`.
Forbidden supervision: DIP `trans`, DIP root velocity, full-pipeline 11 metrics.
```

Training/eval artifacts:

```text
cache builder: scripts/build_smooth_acc_cache.py
runner: scripts/run_newpl_v5_smoothacc_20260612.sh
summary: scripts/summarize_newpl_v5_smoothacc.py
full root: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256
shared smooth cache root: data/experiments/newpl_v5_smoothacc_20260612/caches
AMASS checkpoint: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/amass_pretrain/best_loss.pt
DIP checkpoint: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/dip_finetune/best_loss.pt
log: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/logs/run.log
summary json: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/summary.json
eval JSONs: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/eval/
```

Training outcome:

```text
AMASS pretrain: batch=256, lr=1e-4, best_epoch=3, best_control_physical=0.0025220187869422262, early_stopped at epoch 15.
DIP fine-tune: batch=24, lr=5e-6, best_epoch=39, best_control_physical=0.057518671217621886, ran 40 epochs.
Validation speed fix: training validation uses one 61-frame window per val sequence; final module eval uses full DIP/TC test sequences.
```

Key module-level results:

```text
DIP test after DIP fine-tune:
  raw official_PL: pRB L2=6.419473 cm, gR1=12.947709 deg.
  smooth official_PL_smoothacc: pRB L2=6.345701 cm, gR1=12.902131 deg.
  raw newpl_v5_dip_best: pRB L2=6.445578 cm, gR1=12.552613 deg.
  newpl_v5_smoothacc_dip_best: pRB L2=6.350327 cm, gR1=12.894731 deg.
  newpl_v4_init36 on smooth input: pRB L2=6.349507 cm, gR1=12.722395 deg.

TotalCapture test after DIP fine-tune:
  raw official_PL: pRB L2=6.995536 cm, gR1=13.450465 deg.
  smooth official_PL_smoothacc: pRB L2=7.508985 cm, gR1=13.170880 deg.
  raw newpl_v5_dip_best: pRB L2=6.780749 cm, gR1=13.415189 deg.
  newpl_v5_smoothacc_dip_best: pRB L2=7.473741 cm, gR1=13.185523 deg.
  newpl_v4_init36 on smooth input: pRB L2=7.119541 cm, gR1=13.075063 deg.
```

Decision: do not select `newpl_v5_smoothacc`. Smoothed `aM` helps DIP pRB slightly and helps DIP/TC gR1, but it hurts TotalCapture pRB badly. The retrained smooth-acc NewPL is only marginally better than `official_PL_smoothacc` on TC pRB and is worse than `newpl_v4_init36_smoothacc`; on DIP it is also slightly worse than the smoothed official PL on pRB and worse than `newpl_v4_init36_smoothacc` on gR1. The value here is diagnostic: acceleration smoothing changes PL behavior, but using it as the default NewPL v5 input is not justified.

## 3.1.3 NewPL v5 causal ButterAcc realtime input gate

Status: completed on 2026-06-12; diagnostic only, not selected. The input-only gate failed, then a user-requested forced fc12 AMASS -> DIP longtrain was run. Full-pipeline 11 metrics were not run.

Question: can a realtime causal low-pass acceleration filter replace the offline centered smooth-acc input without damaging PL `pRB/gR1` module outputs?

Contract:

```text
PL input shape: 84D unchanged.
Input change: source raw-cache `aM` is replaced by a causal Butterworth low-pass filtered `aM`.
Filter: order=2, fs=60 Hz, cutoff sweep=8/10/12 Hz.
Realtime contract: output[t] depends only on samples <= t; lookahead_frames=0; latency_ms=0.
Audit field: `aM_raw` preserves the original source acceleration.
Unchanged fields: `wM`, `RMB`, `pose_gt`, `tran_gt`, `offset_r`, PL target/control target.
PL output: pRB[15] + gR1[3] = 18D unchanged.
Init: offset_r[18] + pRL[15] + gR0[3] = 36D unchanged.
Root velocity: not part of this PL v5 input-gate experiment; no root-velocity training or GT metric was run.
```

Artifacts:

```text
filter code: l4_sensor_offset_utils.py::causal_butterworth_lowpass_sequence
cache builder: scripts/build_smooth_acc_cache.py --mode causal_butterworth
runner: scripts/run_newpl_v5_butteracc_20260612.sh
summary: scripts/summarize_newpl_v5_butteracc.py
smoke root: data/experiments/newpl_v5_butteracc_20260612_smoke
full root: data/experiments/newpl_v5_butteracc_20260612_full
forced fc12 longtrain root: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain
summary json: data/experiments/newpl_v5_butteracc_20260612_full/summary.json
selection json: data/experiments/newpl_v5_butteracc_20260612_full/selection.json
eval JSONs: data/experiments/newpl_v5_butteracc_20260612_full/eval/
log: data/experiments/newpl_v5_butteracc_20260612_full/logs/run.log
```

Input-only selection gate:

| Cutoff | DIP official pRB L2 cm | DIP official gR1 deg | TC official pRB L2 cm | TC official gR1 deg | TC guard |
|---:|---:|---:|---:|---:|---|
| 8 Hz | `6.907898` | `12.889588` | `7.506753` | `13.217498` | fail: raw official TC pRB + `0.511217 cm` |
| 10 Hz | `6.780699` | `12.893975` | `7.404335` | `13.211574` | fail: raw official TC pRB + `0.408799 cm` |
| 12 Hz | `6.699946` | `12.900018` | `7.267198` | `13.254831` | fail: raw official TC pRB + `0.271662 cm` |

Baseline context:

```text
Raw official_PL: DIP pRB L2=6.419473 cm, gR1=12.947709 deg; TC pRB L2=6.995536 cm, gR1=13.450465 deg.
Raw newpl_v5_dip_best: DIP pRB L2=6.445578 cm, gR1=12.552613 deg; TC pRB L2=6.780749 cm, gR1=13.415189 deg.
Best ButterAcc official cutoff by pRB is fc12, but it still worsens DIP pRB by +0.280473 cm and TC pRB by +0.271662 cm versus raw official_PL.
```

Forced fc12 longtrain:

```text
Command: CUDA_VISIBLE_DEVICES=1 ROOT=data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain CACHE_ROOT=data/experiments/newpl_v5_butteracc_20260612_full/caches CUTOFFS=12 FORCE_CUTOFF_HZ=12 RUN_SMOKE=0 AMASS_BATCH=512 DIP_BATCH=32 /home/lingfeng/bin/longrun -- bash scripts/run_newpl_v5_butteracc_20260612.sh
AMASS pretrain: batch=512, best_epoch=3, best_control_physical=0.0024848974486531006, early_stopped at epoch 15.
DIP fine-tune: batch=32, best_epoch=38, best_control_physical=0.057614331105772486, ran 40 epochs.
AMASS checkpoint: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/amass_pretrain/best_loss.pt
DIP checkpoint: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/dip_finetune/best_loss.pt
```

Forced fc12 module result:

```text
DIP after AMASS: newpl_v5_butteracc_amass_best pRB L2=6.722528 cm, gR1=12.894359 deg.
DIP after DIP FT: newpl_v5_butteracc_dip_best pRB L2=6.721462 cm, gR1=12.896323 deg.
TC after AMASS: newpl_v5_butteracc_amass_best pRB L2=7.271788 cm, gR1=13.251097 deg.
TC after DIP FT: newpl_v5_butteracc_dip_best pRB L2=7.266006 cm, gR1=13.257684 deg.
```

Decision: do not select `newpl_v5_butteracc`. The causal filter is realtime-valid and reduces acceleration second-difference jitter, but the PL output gate fails and forced fc12 training does not recover pRB. It remains worse than raw official PL on DIP and TotalCapture pRB, worse than raw `newpl_v5_dip_best`, and worse than `newpl_v4_init36_butter_fc12` on TotalCapture.

## 3.1.4 NewPL next-control / one-step predictive PL experiment

Status: completed on 2026-06-11; diagnostic only, not selected. The usable full run is `full_fastval1`, which uses the same AMASS -> DIP protocol with batched validation for speed (`VAL_BATCH_SIZE=64`). Full-pipeline 11 metrics were not run.

Contract:

```text
model variant: newpl_v6_next_control
PL input: aRB[18] + wRB[18] + RRB[45] + gR0[3] = 84D
init: offset_r[18] + pRL[15] + gR0[3] = 36D
current output consumed by IK1: pRB_t[15] + gR1_t[3] = 18D, unchanged
aux output: next_pl = pRB_{t+1}[15] + gR1_{t+1}[3]
aux dynamics: next_pldot / next_plddot decoded from a one-step preview control curve
```

Control semantics: the verified current PLCurve path predicts a new tail control point and also adjusts the last up-to-four existing controls before decoding the current frame. The corrected v6 preview branch mirrors that idea for one-step prediction: from hidden state at `t`, it predicts exactly one `next_control`, adjusts the last up-to-four preview controls, appends `next_control + ghost(next_control)`, and decodes `next_pl`/derivatives at preview index `[-2]`. The preview is not written back to the live streaming buffer, so IK1 still receives only the official current 18D PL output.

Loss and cache: `pl_next_control_cache.py` now writes `pl_next_control_cache_v2`, adding `tail_control_target [T,4,18]`, `tail_control_valid_mask [T,4]`, and `last_control_target [T,18]` to the earlier t/t+1 targets and derivative targets. Current loss keeps `pRB/gR1`, GT current control, and temporal terms. Next loss adds `next_pRB`, `next_gR1`, `next_gt_control_pRB/gR1`, `next_pRB_vel/acc`, `next_gR1_vel/acc`, `next_control_delta_prior`, plus explicit `last_control_pRB/gR1` and `next_tail4_control_pRB/gR1`.

Training protocol:

```text
AMASS long pretrain -> module eval on AMASS and TotalCapture test
DIP-IMU train fine-tune -> module eval on DIP test and TotalCapture test
TotalCapture train/val is not used and there is no TC fine-tune in this corrected protocol.
Full-pipeline 11 metrics are not run for this module-level experiment.
```

Artifacts:

```text
model/code: pl_curve.py::PLCurveNextControlModule
cache builder: pl_next_control_cache.py
trainer: pl_next_control_train.py
evaluator: pl_next_control_eval.py
runner: scripts/run_newpl_v6_next_control_20260611.sh
root: data/experiments/newpl_v6_next_control_tail4_20260611
smoke JSONs: data/experiments/newpl_v6_next_control_tail4_20260611/smoke/*.json
full run root: data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1
full JSONs:
  data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_amass_after_pretrain.json
  data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_totalcapture_test_after_amass_pretrain.json
  data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_dip_test_after_dip_finetune.json
  data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_totalcapture_test_after_dip_finetune.json
```

Smoke result, 4 sequences per cache, 1 epoch per stage:

```text
AMASS after pretrain:
official current pRB L2=3.9585 cm, gR1=4.5255 deg; next pRB L2=4.1225 cm.
newpl_v4 current pRB L2=4.3979 cm, gR1=4.6766 deg; next pRB L2=4.5802 cm.
newpl_v6_amass current pRB L2=3.9703 cm, gR1=4.5235 deg; next pRB L2=4.1196 cm; current/next/tail4 control pRB L2=3.9702/4.1305/3.9646 cm.

DIP test after DIP smoke fine-tune:
official current pRB L2=6.1285 cm, gR1=10.1790 deg; next pRB L2=6.1329 cm.
newpl_v4 current pRB L2=6.2891 cm, gR1=10.1212 deg; next pRB L2=6.3227 cm.
newpl_v6_dip current pRB L2=6.1153 cm, gR1=10.1778 deg; next pRB L2=6.1325 cm; current/next/tail4 control pRB L2=6.1291/6.1335/6.1274 cm.

TotalCapture test after DIP smoke fine-tune:
official current pRB L2=6.9955 cm, gR1=13.4504 deg; next pRB L2=7.0954 cm.
newpl_v4 current pRB L2=6.6544 cm, gR1=13.3295 deg; next pRB L2=6.7852 cm.
newpl_v6_dip current pRB L2=6.9956 cm, gR1=13.4468 deg; next pRB L2=7.0963 cm; current/next/tail4 control pRB L2=7.0016/7.0992/7.0016 cm.
```

Full module result:

```text
AMASS eval:
official current pRB/gR1=2.8275 cm / 7.2199 deg; next pRB=2.9476 cm; acc=871.7360.
newpl_v4 current pRB/gR1=2.8989 cm / 7.1207 deg; next pRB=3.0444 cm; acc=726.7491.
v6 AMASS current pRB/gR1=2.8093 cm / 7.2515 deg; next pRB=2.9236 cm; acc=489.2382.

DIP test eval:
official current pRB/gR1=6.4195 cm / 12.9477 deg; next pRB=6.5600 cm; acc=2729.2442.
newpl_v4 current pRB/gR1=6.4414 cm / 12.7652 deg; next pRB=6.6091 cm; acc=2702.1118.
v6 DIP current pRB/gR1=6.4688 cm / 12.6560 deg; next pRB=6.5954 cm; acc=2658.9849.

TotalCapture test eval:
official current pRB/gR1=6.9955 cm / 13.4504 deg; next pRB=7.0954 cm; acc=2173.8883.
newpl_v4 current pRB/gR1=6.6544 cm / 13.3295 deg; next pRB=6.7852 cm; acc=1861.3888.
v6 AMASS current pRB/gR1=6.8749 cm / 13.3279 deg; next pRB=6.9776 cm; acc=706.3869.
v6 DIP current pRB/gR1=6.9808 cm / 13.1385 deg; next pRB=7.0852 cm; acc=707.7693.
```

Decision: do not select `newpl_v6_next_control`. It slightly improves AMASS pRB/next-pRB and greatly lowers spline acceleration, and it improves gR1 on DIP/TotalCapture, but current-frame pRB does not beat the fixed baselines on DIP test or TotalCapture test. DIP fine-tune helps DIP gR1/pRB slightly but hurts TotalCapture pRB. Do not connect this module to IK1/full pipeline.

## 3.1.5 NewPL v6 next-control + smoothacc gR1 search

Status: completed on 2026-06-13; diagnostic only, not selected. This run combines centered smoothed acceleration input with the corrected `newpl_v6_next_control` architecture, adds gR1-specific checkpoint selection, and evaluates module outputs only. Full-pipeline 11 metrics were not run.

Contract and protocol:

```text
variant: newpl_v6_next_control_smoothacc_gR1
input: smooth aRB[18] + raw wRB[18] + raw RRB[45] + gR0[3] = 84D
init: offset_r[18] + pRL[15] + gR0[3] = 36D
current output: pRB_t[15] + gR1_t[3] = 18D, unchanged for IK1
aux output: next_pl, next_pldot, next_plddot, next_control, preview tail4 controls
training route: AMASS 80 epochs -> eval; DIP-IMU 40 epoch fine-tune -> eval
TotalCapture: eval-only, no TC fine-tune
DIP trans/root velocity: not used
```

Training and selection:

```text
root: /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full
smooth cache: data/experiments/newpl_v5_smoothacc_20260612/caches
next-control cache: /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/next_cache_full
batch/window: BATCH_SIZE=768, VAL_BATCH_SIZE=96, WINDOW=81
AMASS: 1294 train sequences, 128 val sequences, best_current_gR1 epoch=80
DIP: 36 train sequences, 6 val sequences, initialized from AMASS best_current_gR1, best_current_gR1 epoch=40
new gR1 checkpoints: best_current_gR1.pt, best_next_gR1.pt, best_gravity_control.pt
```

Evaluation contract: AMASS eval is full for this run; same-window before/after DIP comparisons on DIP and TotalCapture use `max_frames_per_sequence=512`. The full-sequence after-AMASS DIP/TC JSONs also exist, but the key before/after claims below use the fast512 JSONs so the window is consistent.

Key module results:

```text
AMASS after AMASS pretrain:
official_PL_smoothacc: pRB L2=4.030455 cm, gR1=4.838765 deg.
newpl_v4_init36_smoothacc: pRB L2=4.211169 cm, gR1=4.876139 deg.
newpl_v6_smoothacc_amass_current_gR1: pRB L2=4.021821 cm, gR1=5.224865 deg.
Interpretation: v6 smoothacc is slightly better on AMASS pRB than official/v4, but gR1 is worse by +0.386100 deg vs official and +0.348726 deg vs v4.

DIP test fast512 before DIP fine-tune:
official_PL_smoothacc: pRB L2=4.241512 cm, gR1=8.879523 deg.
newpl_v4_init36_smoothacc: pRB L2=4.221608 cm, gR1=8.741283 deg.
newpl_v6_smoothacc_amass_current_gR1: pRB L2=4.262976 cm, gR1=8.792514 deg.
Interpretation: AMASS-only v6 improves gR1 vs official, but not vs v4; pRB is worse than both.

TotalCapture test fast512 before DIP fine-tune:
official_PL_smoothacc: pRB L2=7.566914 cm, gR1=9.873214 deg.
newpl_v4_init36_smoothacc: pRB L2=7.160630 cm, gR1=9.745194 deg.
newpl_v6_smoothacc_amass_balanced: pRB L2=7.470823 cm, gR1=9.616730 deg.
Interpretation: AMASS-only v6 gives the best gR1 on TC fast512, but still loses pRB badly to v4.

DIP test fast512 after DIP fine-tune:
official_PL_smoothacc: pRB L2=4.241512 cm, gR1=8.879523 deg.
newpl_v4_init36_smoothacc: pRB L2=4.221608 cm, gR1=8.741283 deg.
newpl_v5_raw_dip_on_smoothinput: pRB L2=4.190160 cm, gR1=8.671933 deg.
newpl_v6_smoothacc_dip_current_gR1: pRB L2=4.226809 cm, gR1=8.719222 deg.
Interpretation: DIP fine-tune improves v6 gR1 vs official by 0.160301 deg and vs v4 by 0.022061 deg, but raw-v5-on-smoothinput remains better; pRB remains worse than v4/raw-v5.

TotalCapture test fast512 after DIP fine-tune:
official_PL_smoothacc: pRB L2=7.566914 cm, gR1=9.873214 deg.
newpl_v4_init36_smoothacc: pRB L2=7.160630 cm, gR1=9.745194 deg.
newpl_v5_raw_dip_on_smoothinput: pRB L2=7.318521 cm, gR1=9.890395 deg.
newpl_v6_smoothacc_dip_current_gR1: pRB L2=7.562171 cm, gR1=9.470963 deg.
Interpretation: v6 after DIP gives the best TC fast512 gR1, improving official by 0.402251 deg and v4 by 0.274231 deg, but pRB is much worse than v4 and worse than raw-v5.
```

Decision: do not select `newpl_v6_next_control_smoothacc_gR1` for IK1/full-pipeline. It is useful evidence that smoothacc plus gR1-specific selection can reduce gR1 on TotalCapture and slightly on DIP, but the pRB contract is not robust enough. Since IK1 consumes both `pRB` and `gR1`, better gravity alone is not sufficient.

## 3.2 NewPose Control v1 IK2-Slot Check

Status: completed on 2026-06-08; rejected.

Protocol: `newpose_ctrl_v1` was trained under the official-like route: AMASS pretrain, DIP-IMU train fine-tune, then DIP-IMU test and TotalCapture test. It uses official IMU input plus NewPL stream/control/geometric features. `offset_r / r_JS` is used only for hidden-state initialization. DIP translation/root loss was not used. TotalCapture train split was not used.

Contract:

```text
input: official IMU[90] + RRB_after_pl[45] + pRB/gR1[18] + last PL control[18] + gR0[3] = 174D
output: RRJ_control[90] + gR_pose_control[3] = 93D
```

Artifacts:

```text
root: data/experiments/newpose_ctrl_v1_20260608
runner: scripts/run_newpose_ctrl_v1_official_protocol_20260608.sh
summary: data/experiments/newpose_ctrl_v1_20260608/summary.json
tables: data/experiments/newpose_ctrl_v1_20260608/summary_tables.md
AMASS checkpoint: data/experiments/newpose_ctrl_v1_20260608/stage_a_amass_pretrain/best_loss.pt
DIP checkpoint: data/experiments/newpose_ctrl_v1_20260608/stage_b_dip_finetune/best_loss.pt
```

Training outcome: AMASS pretrain best epoch `23`, selection value `0.042461989620351234`; DIP fine-tune best epoch `40`, selection value `0.03869467038415071`.

Full-pipeline result:

| Dataset | Version | Score ↓ | Local angle ↓ | Global angle ↓ |
|---|---|---:|---:|---:|
| DIP test | official_gpnet | `44.642051` | `8.469930` | `8.291750` |
| DIP test | newpl_v5_dip + official downstream | `44.598659` | `8.468339` | `8.315847` |
| DIP test | newpose_ctrl_v1 Stage A | `428.806986` | `100.590216` | `101.018856` |
| DIP test | newpose_ctrl_v1 Stage B | `432.122581` | `100.867367` | `101.714333` |
| TotalCapture test | official_gpnet | `44.477381` | `12.550695` | `11.781375` |
| TotalCapture test | newpl_v5_amass + official downstream | `43.868067` | `12.423023` | `11.680961` |
| TotalCapture test | newpose_ctrl_v1 Stage A | `413.495453` | `95.741745` | `98.819038` |
| TotalCapture test | newpose_ctrl_v1 Stage B | `419.196776` | `96.512819` | `99.903576` |

Module result: `newpose_ctrl_v1` has lower RRJ geodesic numbers than the official IK2-slot state comparison, but this is not a valid win. Decoded FK joint L2 is `44.539856/45.447945 cm` on DIP and `43.782135/44.701134 cm` on TotalCapture, while official/newpl_v5 baselines are about `4.6-5.0 cm`. Its gR loss is also worse.

Decision: reject `newpose_ctrl_v1`. Do not continue this module to IK2/mainline. Any next IK2-slot attempt must first make decoded FK/body-space pose comparable to the official IK2 output before full-pipeline physics evaluation is meaningful.

## 3.3 DIP Foot-Lock TransPose pseudo-r_JS Cache

Status: completed on 2026-06-08; selected as the next DIP pseudo-`r_JS` source for NewPL cache/eval, not a module promotion.

Contract:

```text
r_JS: IMU origin position relative to mapped joint J, expressed in joint-local coordinates.
World prediction: p_WS(t) = p_WJ(t) + R_WJ(t) @ r_JS.
DIP output is pseudo-r_JS only; DIP trans is not trusted and this is not GT.
```

Method: `footlock_transpose_v1` uses frozen TransPose `pose_s1 + tran_b1` contact logits, then selects a single stance foot per frame with TransPose-style `max(contact)` winner logic plus a PIP-style near-ground foot-height sanity check. For a selected stance foot contact point `C`, zero-translation SMPL FK gives `p_WC_zero_tran(t)`, and root translation is used only through `p_WR(t) = -p_WC_zero_tran(t) + constant` over stance windows. The resulting root/joint accelerations are used to solve the sequence-level lever-arm offset. The stance-side lower-leg sensor is not solved from its own stance window.

Artifacts:

```text
winner root: data/experiments/footlock_transpose_rjs_20260608
independent-threshold comparison root: data/experiments/footlock_transpose_rjs_20260608_independent
script: scripts/run_footlock_transpose_rjs_20260608.sh
builder: scripts/build_imu_position_offsets.py --method footlock_transpose_v1
comparison: data/experiments/footlock_transpose_rjs_20260608/compare_with_independent.md
```

Full split cache outputs:

| Split | Sequences | Winner offset norm mean / median / p95 (m) | All finite? |
|---|---:|---:|---|
| DIP train | 36 | `0.1775 / 0.1546 / 0.3302` | yes |
| DIP val | 6 | `0.1660 / 0.1641 / 0.2719` | yes |
| DIP test | 19 | `0.1676 / 0.1553 / 0.2809` | yes |

Comparison with the previous independent-threshold contact selection:

| Split | Offset delta mean / median / p95 / max (m) | Winner fallback sensors | Independent fallback sensors | Winner/independent window-count ratio |
|---|---:|---:|---:|---:|
| DIP train | `0.0714 / 0.0482 / 0.2379 / 0.3517` | 4 | 6 | 1.69 |
| DIP val | `0.0681 / 0.0449 / 0.2398 / 0.4685` | 1 | 0 | 1.29 |
| DIP test | `0.0587 / 0.0320 / 0.1866 / 0.4265` | 3 | 3 | 1.62 |

Decision: use the winner version (`data/experiments/footlock_transpose_rjs_20260608`) for the next NewPL `offset_aware` / `r_JS`-sensitive cache generation and evaluation. Do not use the independent-threshold version as the default because it can treat both feet as simultaneously locked when TransPose assigns high probability to both feet.

## 4. Document Map

| Document | Purpose |
|---|---|
| `RECENT_REPLACEMENT_VERSIONS.md` | Version iterations, structure/loss/training changes, module GT comparison, official S4 11 metrics, decisions, artifacts. |
| `EXPERIMENT_LOG.md` | Detailed commands, orchestrator tasks, log paths, failure traces, JSON paths, timeline. |
| `PROJECT_STATUS.md` | Current best, current mainline, baseline references, known issues, next actions. |

## 5. Known Issues

- NewIK1 local validation loss and isolated module-GT gains do not reliably predict full S4 score.
- NewIK1 v8/v9 loss-ratio sweeps improved S4 only at the fourth decimal place and still worsened full `pRJ`, leaf `pRJ`, and module `state_l2` versus official IK1 baseline.
- NewIK2 artifacts are not confirmed; do not relabel NewIK1 artifacts as NewIK2.
- Do not mix batch/cache diagnostics with official full-pipeline streaming S4 metrics.
- Do not report `newpl_v5_official_protocol` as a full-pipeline improvement; it only has module-level pRB/gR1 evidence.
- Do not report `newpose_ctrl_v1` as an IK2/pose-control improvement. Its rotation-space control metric is misleading; decoded FK and full-pipeline metrics fail.
- Missing metrics must be recorded as `not measured` or `not found`, not inferred.

## 6. Next Actions

1. Treat `newpl_v4_init36` as the historical best processed-input artifact, not as the sole training recipe to extend.
2. Reproduce the official baseline training route as the new reference: AMASS pretrain -> DIP fine-tune -> DIP test and TotalCapture test.
3. Re-train NewPL variants under this official-like route before claiming superiority over official PL or processed-input NewPL.
4. For NewPL v5/v6, target pRB improvement without losing the observed DIP gR1 gain; module selection should use physical control-point closeness to GT.
5. For the next `offset_aware` / `r_JS`-sensitive NewPL cache/eval, use the winner DIP pseudo-`r_JS` cache at `data/experiments/footlock_transpose_rjs_20260608` rather than the independent-threshold comparison cache.
6. For IK1/IK2/VR, only iterate after the upstream PL protocol is fixed, and compare replacements under the same AMASS -> DIP route.
7. For IK2-slot / pose-control designs, add decoded FK/body-space losses and compatibility checks before long full-pipeline runs; do not rely on RRJ geodesic alone.
8. Do not promote a module based only on TotalCapture train/fine-tune gains unless the experiment is explicitly labeled TotalCapture-specialized rather than official-generalization mainline.

## EXP-fixed-init36-ik1-trend

### Orchestrator Task: s4_v3_strong_pRJ_control

Name: S4 fixed init36 newik1_v3_strong_pRJ_control

Status: completed

Type: eval

Start: 2026-06-06T12:19:30

End: 2026-06-06T12:26:35

GPU: 1

PID: 1649358

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s4/newik1_v3_strong_pRJ_control/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s4_v3_strong_pRJ_control.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s4/newik1_v3_strong_pRJ_control/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.917825568050155 |
| Local SIP | 10.206472396850586 |
| Local Angle | 8.799551105499267 |
| Local Joint | 4.530084133148193 |
| Local Mesh | 5.171298837661743 |
| Global SIP | 10.353433609008789 |
| Global Angle | 8.655188846588135 |
| Global Joint | 4.455532693862915 |
| Global Mesh | 5.0273336410522464 |
| Root Jitter | 0.27822756841778756 |
| Joint Jitter | 0.4617927402257919 |

### Orchestrator Task: s4_v4_official_input

Name: S4 fixed init36 newik1_v4_official_input

Status: completed

Type: eval

Start: 2026-06-06T12:26:35

End: 2026-06-06T12:33:20

GPU: 1

PID: 1654283

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s4/newik1_v4_official_input/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s4_v4_official_input.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s4/newik1_v4_official_input/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.6972478222847 |
| Local SIP | 10.06729621887207 |
| Local Angle | 8.804924583435058 |
| Local Joint | 4.529463863372802 |
| Local Mesh | 5.17761197090149 |
| Global SIP | 10.334024906158447 |
| Global Angle | 8.592272472381591 |
| Global Joint | 4.408935022354126 |
| Global Mesh | 4.929739141464234 |
| Root Jitter | 0.2972386389970779 |
| Joint Jitter | 0.48897528648376465 |

### Orchestrator Task: s5_baseline_official_ik1

Name: S5 fixed init36 official IK1 baseline

Status: completed

Type: eval

Start: 2026-06-06T12:33:20

End: 2026-06-06T12:39:04

GPU: 1

PID: 1659539

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" pl_curve_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s5/baseline_official_ik1/result.json --checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s5_baseline_official_ik1.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s5/baseline_official_ik1/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.81127653867006 |
| Local SIP | 9.647517800331116 |
| Local Angle | 12.44710111618042 |
| Local Joint | 4.358768105506897 |
| Local Mesh | 5.107238292694092 |
| Global SIP | 9.131767272949219 |
| Global Angle | 11.75735354423523 |
| Global Joint | 3.8370408415794373 |
| Global Mesh | 4.44201785326004 |
| Root Jitter | 0.3705526255071163 |
| Joint Jitter | 0.7955910265445709 |

### Orchestrator Task: s5_v1_control_tail

Name: S5 fixed init36 newik1_v1_control_tail

Status: completed

Type: eval

Start: 2026-06-06T12:39:04

End: 2026-06-06T12:45:28

GPU: 1

PID: 1664316

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s5/newik1_v1_control_tail/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s5_v1_control_tail.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s5/newik1_v1_control_tail/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.96522354094312 |
| Local SIP | 9.71082055568695 |
| Local Angle | 12.474984645843506 |
| Local Joint | 4.390320301055908 |
| Local Mesh | 5.144239544868469 |
| Global SIP | 9.159987926483154 |
| Global Angle | 11.780718088150024 |
| Global Joint | 3.9176366329193115 |
| Global Mesh | 4.526580810546875 |
| Root Jitter | 0.36807897686958313 |
| Joint Jitter | 0.7916631381958723 |

### Orchestrator Task: s5_v2_bonelength

Name: S5 fixed init36 newik1_v2_bonelength

Status: completed

Type: eval

Start: 2026-06-06T12:45:28

End: 2026-06-06T12:51:53

GPU: 1

PID: 1668471

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s5/newik1_v2_bonelength/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s5_v2_bonelength.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s5/newik1_v2_bonelength/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.90552373392508 |
| Local SIP | 9.68858516216278 |
| Local Angle | 12.442085266113281 |
| Local Joint | 4.3761022090911865 |
| Local Mesh | 5.116960406303406 |
| Global SIP | 9.169309973716736 |
| Global Angle | 11.76706862449646 |
| Global Joint | 3.9334908723831177 |
| Global Mesh | 4.528445243835449 |
| Root Jitter | 0.3581745456904173 |
| Joint Jitter | 0.7515399288386106 |

### Orchestrator Task: s5_v3_strong_pRJ_control

Name: S5 fixed init36 newik1_v3_strong_pRJ_control

Status: completed

Type: eval

Start: 2026-06-06T12:51:53

End: 2026-06-06T12:58:38

GPU: 1

PID: 1672887

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s5/newik1_v3_strong_pRJ_control/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s5_v3_strong_pRJ_control.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s5/newik1_v3_strong_pRJ_control/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.68835979767144 |
| Local SIP | 9.603381991386414 |
| Local Angle | 12.388200998306274 |
| Local Joint | 4.347099721431732 |
| Local Mesh | 5.066087603569031 |
| Global SIP | 9.111295223236084 |
| Global Angle | 11.751279830932617 |
| Global Joint | 3.923874020576477 |
| Global Mesh | 4.539568901062012 |
| Root Jitter | 0.33925189916044474 |
| Joint Jitter | 0.710437960922718 |

### Orchestrator Task: s5_v4_official_input

Name: S5 fixed init36 newik1_v4_official_input

Status: completed

Type: eval

Start: 2026-06-06T12:58:38

End: 2026-06-06T13:05:02

GPU: 1

PID: 1678902

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s5/newik1_v4_official_input/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s5_v4_official_input.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s5/newik1_v4_official_input/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.92881545905024 |
| Local SIP | 9.624021530151367 |
| Local Angle | 12.546680927276611 |
| Local Joint | 4.322596848011017 |
| Local Mesh | 5.1406508684158325 |
| Global SIP | 9.119454860687256 |
| Global Angle | 11.811805963516235 |
| Global Joint | 3.8605257272720337 |
| Global Mesh | 4.431929707527161 |
| Root Jitter | 0.39605626929551363 |
| Joint Jitter | 0.8539919890463352 |

### Orchestrator Task: s5_v5_last_pl_control

Name: S5 fixed init36 newik1_v5_last_pl_control

Status: completed

Type: eval

Start: 2026-06-06T13:05:02

End: 2026-06-06T13:11:47

GPU: 1

PID: 1685411

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s5/newik1_v5_last_pl_control/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_last_pl_control_20260605_v2/stage_c_tc_finetune/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s5_v5_last_pl_control.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s5/newik1_v5_last_pl_control/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.35959049385041 |
| Local SIP | 9.542999505996704 |
| Local Angle | 12.313348054885864 |
| Local Joint | 4.304443418979645 |
| Local Mesh | 5.0063745975494385 |
| Global SIP | 9.019214630126953 |
| Global Angle | 11.665404081344604 |
| Global Joint | 3.8085469603538513 |
| Global Mesh | 4.415583074092865 |
| Root Jitter | 0.3448094669729471 |
| Joint Jitter | 0.7325183562934399 |

### Orchestrator Task: s5_v6_stage_a

Name: S5 fixed init36 newik1_v6_stage_a

Status: completed

Type: eval

Start: 2026-06-06T13:11:47

End: 2026-06-06T13:17:52

GPU: 1

PID: 1691677

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s5/newik1_v6_stage_a/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s5_v6_stage_a.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s5/newik1_v6_stage_a/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.85181389780715 |
| Local SIP | 9.617714643478394 |
| Local Angle | 12.505094766616821 |
| Local Joint | 4.309996843338013 |
| Local Mesh | 5.126302123069763 |
| Global SIP | 9.118677496910095 |
| Global Angle | 11.785181283950806 |
| Global Joint | 3.854469418525696 |
| Global Mesh | 4.421529710292816 |
| Root Jitter | 0.4033846762031317 |
| Joint Jitter | 0.8699080664664507 |

### Orchestrator Task: s5_v7_best

Name: S5 fixed init36 newik1_v7_best

Status: completed

Type: eval

Start: 2026-06-06T13:17:52

End: 2026-06-06T13:24:36

GPU: 1

PID: 1697414

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s5/newik1_v7_best/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s5_v7_best.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s5/newik1_v7_best/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.85909186106175 |
| Local SIP | 9.671209216117859 |
| Local Angle | 12.452542066574097 |
| Local Joint | 4.363961756229401 |
| Local Mesh | 5.117542624473572 |
| Global SIP | 9.14937424659729 |
| Global Angle | 11.757962226867676 |
| Global Joint | 3.8410638570785522 |
| Global Mesh | 4.445083498954773 |
| Root Jitter | 0.3510137936100364 |
| Joint Jitter | 0.7501543574035168 |

### Orchestrator Task: s5_v8_B4_last

Name: S5 fixed init36 newik1_v8_B4_last

Status: completed

Type: eval

Start: 2026-06-06T13:24:37

End: 2026-06-06T13:31:21

GPU: 1

PID: 1702661

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s5/newik1_v8_B4_last/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s5_v8_B4_last.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s5/newik1_v8_B4_last/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.860368536058814 |
| Local SIP | 9.671541094779968 |
| Local Angle | 12.453015565872192 |
| Local Joint | 4.364069819450378 |
| Local Mesh | 5.117747902870178 |
| Global SIP | 9.149513721466064 |
| Global Angle | 11.758282899856567 |
| Global Joint | 3.8410672545433044 |
| Global Mesh | 4.445103466510773 |
| Root Jitter | 0.3510159496217966 |
| Joint Jitter | 0.7501546684652567 |

### Orchestrator Task: s5_v9_C8_last

Name: S5 fixed init36 newik1_v9_C8_last

Status: completed

Type: eval

Start: 2026-06-06T13:31:21

End: 2026-06-06T13:38:06

GPU: 1

PID: 1710011

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_control_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/s5/newik1_v9_C8_last/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/s5_v9_C8_last.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/s5/newik1_v9_C8_last/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.8609724480845 |
| Local SIP | 9.671691298484802 |
| Local Angle | 12.453238487243652 |
| Local Joint | 4.364092111587524 |
| Local Mesh | 5.117820978164673 |
| Global SIP | 9.149576425552368 |
| Global Angle | 11.758445739746094 |
| Global Joint | 3.8410972356796265 |
| Global Mesh | 4.4451345801353455 |
| Root Jitter | 0.3510171761736274 |
| Joint Jitter | 0.750156233087182 |

### Orchestrator Task: audit_s4_baseline

Name: S4 real streaming IK1 audit official IK1

Status: completed

Type: audit

Start: 2026-06-06T13:38:06

End: 2026-06-06T13:42:49

GPU: 1

PID: 1716018

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/real_streaming/s4/baseline_official_ik1/result.json --split-label S4 --version-name baseline_official_ik1 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-backend original --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/audit_s4_baseline.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/real_streaming/s4/baseline_official_ik1/result.json`

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
| score | 38.625657482802865 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | baseline_official_ik1 |

### Orchestrator Task: audit_s4_v6_stage_a

Name: S4 real streaming IK1 audit newik1_v6_stage_a

Status: completed

Type: audit

Start: 2026-06-06T13:42:49

End: 2026-06-06T13:47:33

GPU: 1

PID: 1718791

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/real_streaming/s4/newik1_v6_stage_a/result.json --split-label S4 --version-name newik1_v6_stage_a --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/audit_s4_v6_stage_a.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/real_streaming/s4/newik1_v6_stage_a/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.649136830300094 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | newik1_v6_stage_a |

### Orchestrator Task: audit_s4_v9_C8_last

Name: S4 real streaming IK1 audit newik1_v9_C8_last

Status: completed

Type: audit

Start: 2026-06-06T13:47:33

End: 2026-06-06T13:52:56

GPU: 1

PID: 1721687

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/fixed_init36_ik1_trend/real_streaming/s4/newik1_v9_C8_last/result.json --split-label S4 --version-name newik1_v9_C8_last --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt --ik1-backend auto_control_point --imu-input-mode processed`

Log: `logs/orchestrator/fixed_init36_ik1_trend/audit_s4_v9_C8_last.log`

Outputs:

- `data/experiments/fixed_init36_ik1_trend/real_streaming/s4/newik1_v9_C8_last/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | control_point_last_v1 |
| ik1_checkpoint | data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.69384564702212 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | newik1_v9_C8_last |
## IK1 Auto Search Status

Date: 2026-06-06

Current best remains `newpl_v4_init36 + official IK1` with S4 Score `38.625657482802865`.

An IK1 auto-search Round 0 queue has been created to complete missing fixed-upstream evidence under:

```text
processed IMU + newpl_v4_init36 + IK1 candidate
```

Artifacts:

- Queue: `experiments/ik1_auto_search_queue.yaml`
- Results CSV: `experiments/ik1_auto_search_results.csv`
- State: `data/experiments/orchestrator_states/ik1_auto_search_queue.json`
- Logs: `logs/orchestrator/ik1_auto_search/round0/*.log`
- New JSON root: `data/experiments/ik1_auto_search/round0/`

Round 0 current evidence before launching missing tasks:

- `newik1_v4_official_input` is the only seed with existing fixed-upstream S4 in the CSV; S4 `38.6972478222847`, worse than PL-only by `+0.0715903394818369`.
- Existing S5 seed scores are all worse than the fixed-upstream official IK1 S5 baseline in the current CSV.
- Real streaming IK1 diagnostics are incomplete, especially S5.

Next plan:

1. Finish Round 0 missing S4/S5 full-pipeline and S4/S5 real streaming IK1-vs-GT diagnostics.
2. Regenerate `experiments/ik1_auto_search_results.csv`.
3. Only after Round 0 evidence is complete, generate Round 1 with at most four conservative IK1 experiments: `v10_residual_pRJ_only_alpha025_from_v6a`, `v10_residual_pRJ_only_alpha05_from_v6a`, `v10_stage_a_low_lr_distill_official`, and `v10_ik2_input_distill_from_v6a`.
4. Do not select by AMASS/cache/local loss.

### Round 0 Completion

Status: completed 14/14 tasks, failed 0.

Best fixed-upstream S4 ranking from `experiments/ik1_auto_search_results.csv`:

| Rank | Version | S4 Score ↓ | Δ vs PL-only |
|---:|---|---:|---:|
| 1 | baseline_official_ik1 | `38.625657482802865` | `0.0` |
| 2 | newik1_v6_stage_a | `38.649136830300094` | `+0.02347934749722924` |
| 3 | newik1_v9_C8_last | `38.69384564702212` | `+0.06818816421925789` |
| 4 | newik1_v8_B4_last | `38.69415222530066` | `+0.06849474249779774` |

Conclusion: no current IK1 seed beats PL-only best. `newik1_v6_stage_a` is the best NewIK1 seed under fixed `newpl_v4_init36`, but it is still worse than the baseline by `+0.023479` S4 Score.

Round 1 status: not launched yet. The requested `from_v6a` pRJ-only/residual routes need an official-input backend change that preserves official/base `gR2` during streaming eval. Starting those experiments before implementing that backend would create mislabeled artifacts.

### Round 1 Launch

Status: launched after backend changes.

Round 1 queue: `experiments/ik1_auto_search_round1_queue.yaml`

Experiments:

- `v10_residual_pRJ_only_alpha025_from_v6a`
- `v10_residual_pRJ_only_alpha05_from_v6a`
- `v10_stage_a_low_lr_distill_official`
- `v10_ik2_input_distill_from_v6a`

Selection gate: S4 full-pipeline first; S4 real streaming IK1 output vs GT diagnostic; S5 only for promising S4 candidates.

GPU note: local GPU0 was occupied by a foreign-user process at dry-run, so the scheduler will not share it. GPU1 can proceed; GPU0 tasks remain pending until the GPU is free.

### Round 1 Retry 1

Status: prepared after failed first launch.

The first Round 1 launch failed before producing checkpoints. All 4 training tasks hit the same batched-window shape bug in `newik1_official_input_train.py::ik2_input_feature`; the 8 dependent S4 full-pipeline and S4 real-streaming audit tasks were blocked. Failure evidence remains in:

- State: `data/experiments/orchestrator_states/ik1_auto_search_round1_queue.json`
- Logs: `logs/orchestrator/ik1_auto_search/round1/*/train.log`

Fix validation:

- `ik2_input_feature` now preserves leading dimensions for IK2 input distillation.
- Shape smoke passed for `[T,72]`, `[T,B,72]`, and `[72]`, all returning `117D`.

Retry artifacts:

- Queue: `experiments/ik1_auto_search_round1_retry1_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round1_retry1_queue.json`
- Output root: `data/experiments/ik1_auto_search/round1_retry1/`
- Logs: `logs/orchestrator/ik1_auto_search/round1_retry1/*/*.log`

Dry-run result: 12 tasks, 4 train tasks ready, no output/log conflicts. Current best remains `newpl_v4_init36 + official IK1` with S4 Score `38.625657482802865`; no v10 candidate can be called best until S4 full-pipeline metrics exist.

### Round 1 Completion

Status: completed with one failed route and one candidate beating PL-only on both S4 and S5.

Task summary:

- Round 1 retry1 S4 queue: 9 completed, 1 failed, 2 blocked.
- S5 completion queue for promising S4 candidates: 6 completed, 0 failed.
- Failed route: `v10_stage_a_low_lr_distill_official`; full-mode training produced NaN losses and no `best_loss.pt`, so S4/S5 eval was blocked.

Best S4 version: `v10_residual_pRJ_only_alpha05_from_v6a`, S4 Score `38.30681182323395`, delta vs PL-only `-0.31884565956891464`, but S5 Score `43.867689362633975` is worse than baseline by `+0.05641282396391745`.

Best overall version: `v10_ik2_input_distill_from_v6a`.

| Version | S4 Score ↓ | S4 Δ vs PL-only | S5 Score ↓ | S5 Δ vs official IK1 | Decision |
|---|---:|---:|---:|---:|---|
| `v10_residual_pRJ_only_alpha05_from_v6a` | `38.30681182323395` | `-0.31884565956891464` | `43.867689362633975` | `+0.05641282396391745` | reject as overall best; S5 regresses |
| `v10_residual_pRJ_only_alpha025_from_v6a` | `38.401125624060626` | `-0.2245318587422389` | `43.84817816592753` | `+0.03690162725747115` | reject as overall best; S5 regresses |
| `v10_ik2_input_distill_from_v6a` | `38.41680224156379` | `-0.20885524123907118` | `43.807055798713115` | `-0.0042207399569420545` | current best IK1 candidate |
| `baseline_official_ik1` | `38.625657482802865` | `0.0` | `43.81127653867006` | `0.0` | previous baseline |

Current best should be treated as:

```text
processed IMU + newpl_v4_init36 + v10_ik2_input_distill_from_v6a
```

Checkpoint:

```text
data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt
```

Stop condition reached: a candidate improves both S4 and S5 full-pipeline scores versus PL-only best. Next action is not blind search; preserve this candidate as current best, then optionally run a confirmation rerun or a narrow Round 2 around `ik2_input_distill` only if more margin is needed.

### Round 2 Downstream-Aware Search

Status: completed 20/20 tasks, failed 0.

Parent seed:

```text
v10_ik2_input_distill_from_v6a
data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt
```

Round 2 queue:

- Queue: `experiments/ik1_auto_search_round2_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round2_queue.json`
- Output root: `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/`

Ranking by S4/S5 full-pipeline selection:

| Version | S4 Score ↓ | S4 Δ vs official | S5 Score ↓ | S5 Δ vs official | Decision |
|---|---:|---:|---:|---:|---|
| `v11_alpha035_ik2w2_from_v10` | `38.37255207578838` | `-0.2531054070144876` | `43.77819666691124` | `-0.033079871758815216` | new current best |
| `v11_alpha025_ik2w3_from_v10` | `38.416982645660646` | `-0.2086748371422189` | `43.80445552650839` | `-0.0068210121616658625` | beats both baseline, worse than v11 alpha035 |
| `v11_alpha025_ik2w1_from_v10` | `38.40251200318336` | `-0.22314547961950382` | `43.84088326931` | `+0.029606730639940793` | reject as main seed; S5 regresses |
| `v11_pRJ_only_ik2w2_from_v10` | `38.39572076609731` | `-0.22993671670555216` | `43.845466667562725` | `+0.034190128892667815` | reject as main seed; S5 regresses |
| `v10_ik2_input_distill_from_v6a` | `38.41680224156379` | `-0.20885524123907118` | `43.807055798713115` | `-0.0042207399569420545` | previous current best |

Current best:

```text
processed IMU + newpl_v4_init36 + v11_alpha035_ik2w2_from_v10
```

Checkpoint:

```text
data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt
```

Key trend: alpha `0.35` with IK2-input distill `2.0` improves both S4 and S5. Lower IK2 distill (`1.0`) and pRJ-only improve S4 but regress S5, so they should not seed Round 3. Higher IK2 distill (`3.0`) preserves S5 better than Round 1 but is weaker than alpha `0.35`.

Round 3 plan: narrow around `v11_alpha035_ik2w2_from_v10`, with alpha `0.30/0.40`, IK2-input distill `2.5`, and a conservative S5-stability variant. Keep gR2 preserved or official-distilled only.

### Round 2 Completion

Status: completed 20/20 tasks, failed 0.

Round 2 parent was `v10_ik2_input_distill_from_v6a`. The best Round 2 candidate is:

```text
processed IMU + newpl_v4_init36 + v11_alpha035_ik2w2_from_v10
```

Checkpoint:

```text
data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt
```

Ranking:

| Version | S4 Score ↓ | S4 Δ vs official | S5 Score ↓ | S5 Δ vs official | Decision |
|---|---:|---:|---:|---:|---|
| `v11_alpha035_ik2w2_from_v10` | `38.37255207578838` | `-0.2531054070144876` | `43.77819666691124` | `-0.033079871758815216` | current best |
| `v11_alpha025_ik2w3_from_v10` | `38.416982645660646` | `-0.2086748371422189` | `43.80445552650839` | `-0.0068210121616658625` | improves S5 but not better than v10 S4 |
| `v11_alpha025_ik2w1_from_v10` | `38.40251200318336` | `-0.22314547961950382` | `43.84088326931` | `+0.029606730639940793` | reject; S5 regresses |
| `v11_pRJ_only_ik2w2_from_v10` | `38.39572076609731` | `-0.22993671670555216` | `43.845466667562725` | `+0.034190128892667815` | reject; S5 regresses |

Current trend: the useful region is residual pRJ-only with alpha near `0.35` and IK2-input distill near `2.0`. Direct pRJ-only and too-low IK2 distill are not S5-stable.

Next plan: Round 3 narrow search around `v11_alpha035_ik2w2_from_v10`; do not expand to random loss combinations.

### Round 3 Completion

Status: completed 20/20 tasks, failed 0.

Round 3 parent was `v11_alpha035_ik2w2_from_v10`. During completion, pending GPU0 tasks were remapped to GPU1 because GPU0 was occupied by an unrelated long `test.py` process; completed outputs were not rerun.

Current best:

```text
processed IMU + newpl_v4_init36 + v12_alpha040_ik2w2_from_v11
```

Checkpoint:

```text
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Ranking by S4-first full-pipeline selection:

| Version | S4 Score ↓ | S4 Δ vs official | S5 Score ↓ | S5 Δ vs official | Decision |
|---|---:|---:|---:|---:|---|
| `v12_alpha040_ik2w2_from_v11` | `38.36728921282291` | `-0.2583682699799539` | `43.73998444685712` | `-0.07129209181293561` | selected current best |
| `v12_alpha035_ik2w25_from_v11` | `38.386719339862466` | `-0.23893814294039828` | `43.75161533830688` | `-0.059661200363180455` | beats both baselines, not selected |
| `v12_alpha035_s5stable_from_v11` | `38.38921534772218` | `-0.23644213508068646` | `43.737316830046474` | `-0.07395970862358325` | best S5 score, not selected because S4 is weaker |
| `v12_alpha030_ik2w2_from_v11` | `38.39537861308455` | `-0.2302788697183118` | `43.78341374022886` | `-0.027862798441198322` | beats both baselines, weaker than v11 on S5 |

Round 3 improves the previous best `v11_alpha035_ik2w2_from_v10` by `-0.005262862965466297` on S4 and `-0.038212220054120394` on S5.

Key trend: alpha `0.40` with IK2-input distill `2.0` is the strongest S4/S5 combined point. Increasing IK2 distill to `2.5` improves S5 stability but weakens S4. Lowering alpha to `0.30` weakens S4 and no longer improves over v11 on S5. gR2 stayed unchanged as intended; the improvements come from pRJ residual/downstream compatibility.

Next plan: one more narrow Round 4 only if more margin is needed. Recommended candidates are alpha `0.38`/`0.42` with IK2 distill `2.0`, and one S5-stability check around alpha `0.40` with IK2 distill `2.25`; keep gR2 official-distilled only and continue rejecting S4-only gains that regress S5.

### Round 4 Completion

Status: completed 20/20 tasks, failed 0.

Round 4 parent was `v12_alpha040_ik2w2_from_v11`. All candidates improve both S4 and S5 versus the original `newpl_v4_init36 + official IK1` baseline, but none improves both S4 and S5 versus the current Round 3 best.

Current best remains:

```text
processed IMU + newpl_v4_init36 + v12_alpha040_ik2w2_from_v11
```

Checkpoint:

```text
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Round 4 ranking by S4-first full-pipeline selection:

| Version | S4 Score ↓ | S4 Δ vs official | S5 Score ↓ | S5 Δ vs official | Δ vs current best S4/S5 | Decision |
|---|---:|---:|---:|---:|---:|---|
| `v13_alpha042_ik2w2_from_v12` | `38.370207245603204` | `-0.25545023719966053` | `43.71751535784453` | `-0.09376118082552409` | `+0.002918032780293345` / `-0.022469089012588483` | best Round 4 S4, not selected because S4 is worse than current best |
| `v13_alpha038_ik2w2_from_v12` | `38.376336369305854` | `-0.24932111349701103` | `43.73781035907567` | `-0.07346617959439072` | `+0.009047156482942853` / `-0.002174087781455114` | beats official baseline, not current best |
| `v13_alpha040_ik2w225_from_v12` | `38.3787378231883` | `-0.2469196596145622` | `43.71488639246672` | `-0.09639014620334052` | `+0.011448610365391687` / `-0.02509805439040491` | beats official baseline, not current best |
| `v13_alpha040_s5stable_from_v12` | `38.3813811891675` | `-0.24427629363536596` | `43.707128487303855` | `-0.10414805136620231` | `+0.01409197634458792` / `-0.0328559595532667` | best Round 4 S5, diagnostic only because S4 is worse |

Key trend: pushing alpha from `0.40` to `0.42` or increasing IK2-input distill to `2.25` improves S5 but sacrifices S4. This is a downstream-compatibility/stability tradeoff, not a new overall best. Do not replace `v12_alpha040_ik2w2_from_v11` unless a later candidate improves both S4 and S5 full-pipeline scores.

Next plan: stop broad search. If one more check is needed, run confirmation only: repeat `v12_alpha040_ik2w2_from_v11` and `v13_alpha042_ik2w2_from_v12` once, or test a single midpoint `alpha=0.41, ik2_input_distill=2.0` with the same fixed PL and full S4/S5 eval.

### Official GPNet TotalCapture fine-tune diagnostic

Status: completed on 2026-06-07.

This is a diagnostic adaptation experiment, not the official training protocol.

Why: previous NewIK1 comparisons included TotalCapture-finetuned variants. To test fairness, we checked whether official GPNet itself also improves when adapted on TotalCapture before TotalCapture testing.

Implementation:

- Added minimal training/eval entry: `scripts/finetune_official_gpnet_totalcapture.py`.
- Loaded official checkpoint `data/weights.pt` as a full `GPNet.state_dict`; no split-module checkpoint was used.
- Kept official model structure and official PL/IK1/IK2/VR modules unchanged.
- Did not use NewPL init36, did not replace IK1, and did not replace PL/IK2/VR.
- Fine-tuned trainable neural modules: `plnet`, `iknet.net1`, `iknet.net2`, `vrnet`.
- Frozen modules: none.
- Loss supervision used available TotalCapture GT for PL `pRB/gR`, IK1 `pRJ/gR2`, IK2 reduced-global 6D rotations, and VR root velocity. Stationary/contact GT was not measured.

Data leakage / protocol label:

- TotalCapture train split was used for fine-tune.
- TotalCapture val split was used for best-loss checkpoint selection.
- TotalCapture test split was used for final 11-metric evaluation.
- This only validates TotalCapture adaptation advantage and is not a paper-style generalization result.
- It must not be written as the official protocol.

TotalCapture test result:

| Version | Train data | LR | Epochs | Trainable modules | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_gpnet_original | none | not found | 0 | none | 44.477380 | 9.989696 | 12.550694 | 4.519671 | 5.300476 | 9.314009 | 11.781373 | 3.810419 | 4.470022 | 0.399826 | 0.859838 |
| FT-A | TotalCapture train | 1e-5 | 2 | full GPNet neural modules | 43.349149 | 9.603109 | 12.205352 | 4.340323 | 5.041633 | 9.143327 | 11.585930 | 3.688516 | 4.298988 | 0.397926 | 0.854695 |
| FT-B | TotalCapture train | 3e-6 | 2 | full GPNet neural modules | 43.782556 | 9.765441 | 12.379111 | 4.433093 | 5.167890 | 9.155503 | 11.656312 | 3.742955 | 4.361113 | 0.399393 | 0.858540 |
| FT-C | TotalCapture train | 1e-6 | 2 | full GPNet neural modules | 44.191540 | 9.900780 | 12.483899 | 4.487517 | 5.249972 | 9.241099 | 11.730035 | 3.783797 | 4.426778 | 0.399826 | 0.859736 |

Conclusion: TotalCapture fine-tuning does naturally improve official GPNet under this diagnostic setup. Best improvement is FT-A: Delta Score `-1.1282314625568688` versus original. Future TotalCapture-finetuned NewIK1 comparisons should use a similarly TotalCapture-finetuned official GPNet baseline, currently `FT-A_lr1e-5_ep2`, rather than only the original official checkpoint.

Artifacts:

- Log detail: `EXPERIMENT_LOG.md` section `Official GPNet TotalCapture fine-tune diagnostic`
- Script: `scripts/finetune_official_gpnet_totalcapture.py`
- Root: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/`
- Best diagnostic baseline checkpoint: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-A_lr1e-5_ep2/best_weights.pt`
- Best diagnostic baseline JSON: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-A_lr1e-5_ep2/eval_test.json`

### IK1 Auto Search Round 5 Completion

Status: completed 20/20 tasks, failed 0.

Round 5 parent was `v12_alpha040_ik2w2_from_v11`. All candidates improve both S4 and S5 versus the original `newpl_v4_init36 + official IK1` baseline, but none improves both S4 and S5 versus the current Round 3 best.

Current best remains:

```text
processed IMU + newpl_v4_init36 + v12_alpha040_ik2w2_from_v11
```

Checkpoint:

```text
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Round 5 ranking by S4-first full-pipeline selection:

| Version | S4 Score | S4 Delta vs official | S5 Score | S5 Delta vs official | Delta vs current best S4/S5 | Decision |
|---|---:|---:|---:|---:|---:|---|
| `v14_alpha0410_ik2w21_from_v12` | `38.36968127383291` | `-0.25597620896995466` | `43.719431065004315` | `-0.09184547366574236` | `+0.0023920610099992246` / `-0.02055338185280675` | best Round 5 S4, not selected because S4 is worse than current best |
| `v14_alpha0415_ik2w2_from_v12` | `38.37063280807435` | `-0.2550246747285172` | `43.718751320485026` | `-0.09252521818503112` | `+0.003343595251436682` / `-0.021233126372095512` | best Round 5 S5, diagnostic only because S4 is worse |
| `v14_alpha0405_ik2w2_from_v12` | `38.372773325160146` | `-0.2528841576427183` | `43.725488430112605` | `-0.08578810855745189` | `+0.005484112337235558` / `-0.014496016744516282` | beats official baseline, not current best |
| `v14_alpha0410_ik2w2_from_v12` | `38.37348917667567` | `-0.25216830612719576` | `43.72153897484764` | `-0.0897375638224176` | `+0.006199963852758117` / `-0.01844547200948199` | beats official baseline, not current best |

Key trend: Round 5 confirms the Round 4 tradeoff. Increasing alpha from `0.40` toward `0.405`-`0.415`, or increasing IK2-input distill from `2.0` to `2.1`, improves S5 stability but gives back S4. gR2 stayed unchanged/official-distilled; pRJ real diagnostics improve slightly, but the full-pipeline S4/S5 selection does not justify replacing `v12_alpha040_ik2w2_from_v11`.

Next plan: stop broad search. If another round is required, it should be a targeted bottleneck check, not more alpha/IK2-weight sweeping: inspect why S4 loses when S5 improves, with emphasis on downstream compatibility, jitter/global terms, and per-sequence failures. Do not select any v14 candidate as best unless a future rerun improves both S4 and S5 versus `v12`.

### IK1 Auto Search Scheduler Status

Status checked on 2026-06-07.

The completed Round 2-5 evidence above used verified local `node01` GPU scheduling only. Do not relabel those completed runs as two-server runs.

Second-server health check now passes for `zktitan`:

| Field | Result |
|---|---|
| SSH host | `zktitan` |
| Project path | `/home/lingfeng/projects/GlobalposeMy/GlobalPose` exists |
| Python env | `/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python` executable |
| GPU 0 | RTX 5090, 29 MiB used / 32607 MiB total, 0% util |
| GPU 1 | RTX 5090, 29 MiB used / 32607 MiB total, 0% util |

Future runs can use true two-server scheduling across `node01` and `zktitan`, with explicit `CUDA_VISIBLE_DEVICES` per task. Existing Round 5 local/remote split queue drafts are still pending artifacts, not completed evidence.

### IK1 Auto Search Round 5 Split Launch

Status: launched on 2026-06-07 as a confirmation-only two-server run.

This is not a new random search. It is a narrow split confirmation around the current `v12_alpha040_ik2w2_from_v11` best, using fixed `newpl_v4_init36`, residual pRJ alpha in the already tested `0.40`-`0.42` band, IK2-input distill `2.0`/`2.1`, and `gR2` preserved / official-distilled only.

Queues:

- Local node01: `experiments/ik1_auto_search_round5_local_queue.yaml`
- Remote zktitan: `experiments/ik1_auto_search_round5_remote_queue.yaml`
- Local state: `data/experiments/orchestrator_states/ik1_auto_search_round5_local_queue.json`
- Remote state: `data/experiments/orchestrator_states/ik1_auto_search_round5_remote_queue.json`
- Launch logs: `logs/orchestrator/ik1_auto_search/round5_split_launch/`

Initial state check:

| Server | GPU | Initial task | Status at launch check |
|---|---:|---|---|
| node01 | 1 | `v14_repeat_alpha040_ik2w2_from_v12` train then S4 eval | train completed, S4 eval running |
| node01 | 0 | `v14_alpha041_ik2w2_from_v12` train then S4 eval | train completed, S4 eval running |
| zktitan | 0 | `v14_confirm_alpha042_ik2w2_from_v12` train | running |
| zktitan | 1 | `v14_alpha041_ik2w21_from_v12` train | running |

Selection rule remains unchanged: only S4/S5 full-pipeline 11 metrics decide whether a candidate replaces `v12`; S4-only or S5-only gains are diagnostic.

## EXP-ik1-auto-search

### Orchestrator Task: round5_train_v14_repeat_alpha040_ik2w2_from_v12

Name: Round5 train v14_repeat_alpha040_ik2w2_from_v12

Status: completed

Type: train

Start: 2026-06-07T01:33:07

End: 2026-06-07T01:33:22

GPU: 1

PID: 2112370

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train --experiment-name v14_repeat_alpha040_ik2w2_from_v12 --epochs 3 --lr 1.0e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.4 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010469380300492047 |
| last_epoch | 3 |
| last_val_loss | 0.0010469380300492047 |
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

### Orchestrator Task: round5_train_v14_alpha041_ik2w2_from_v12

Name: Round5 train v14_alpha041_ik2w2_from_v12

Status: completed

Type: train

Start: 2026-06-07T01:33:07

End: 2026-06-07T01:33:22

GPU: 0

PID: 2112371

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train --experiment-name v14_alpha041_ik2w2_from_v12 --epochs 3 --lr 1.0e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.41 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.00104588529211469 |
| last_epoch | 3 |
| last_val_loss | 0.00104588529211469 |
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

### Orchestrator Task: round5_train_v14_confirm_alpha042_ik2w2_from_v12

Name: Round5 train v14_confirm_alpha042_ik2w2_from_v12

Status: completed

Type: train

Start: 2026-06-07T01:33:07

End: 2026-06-07T01:33:38

GPU: 0

PID: 2044000

Return code: 0

Command: `ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train --experiment-name v14_confirm_alpha042_ik2w2_from_v12 --epochs 3 --lr 1.0e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.42 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.0`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010450007568579168 |
| last_epoch | 3 |
| last_val_loss | 0.0010450007568579168 |
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

### Orchestrator Task: round5_train_v14_alpha041_ik2w21_from_v12

Name: Round5 train v14_alpha041_ik2w21_from_v12

Status: completed

Type: train

Start: 2026-06-07T01:33:07

End: 2026-06-07T01:33:38

GPU: 1

PID: 2044004

Return code: 0

Command: `ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python" newik1_official_input_train.py --train-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_amass/newik1_official_input_cache_manifest.json --val-cache data/experiments/newik1_v6_official_input_init36_cascade_rerun/caches/teacher_forced_tc_val/newik1_official_input_cache_manifest.json --output-dir data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train --experiment-name v14_alpha041_ik2w21_from_v12 --epochs 3 --lr 1.0e-6 --dropout 0.1 --batch-size 16 --window 61 --init-checkpoint data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt --output-mode residual_pRJ_only --residual-alpha 0.41 --pRJ-weight 1.0 --gR2-weight 0.0 --gR2-dot-weight 0.0 --gR2-ddot-weight 0.0 --pRJ-dot-weight 0.03 --pRJ-ddot-weight 0.001 --ik1-distill-pRJ-weight 0.8 --ik1-distill-gR2-weight 1.0 --ik2-input-distill-weight 2.1`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train/best_loss.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train/last.pt`
- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 3 |
| best_loss | 0.0010521824238821863 |
| last_epoch | 3 |
| last_val_loss | 0.0010521824238821863 |
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

### Orchestrator Task: round5_s4_v14_confirm_alpha042_ik2w2_from_v12

Name: Round5 S4 full-pipeline v14_confirm_alpha042_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T01:33:38

End: 2026-06-07T01:39:28

GPU: 0

PID: 2051044

Return code: 0

Command: `ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.37247389644385 |
| Local SIP | 10.066790580749512 |
| Local Angle | 8.723737812042236 |
| Local Joint | 4.467641305923462 |
| Local Mesh | 5.108529663085937 |
| Global SIP | 10.232226085662841 |
| Global Angle | 8.469191455841065 |
| Global Joint | 4.289770174026489 |
| Global Mesh | 4.818762254714966 |
| Root Jitter | 0.28740948662161825 |
| Joint Jitter | 0.47868141531944275 |

### Orchestrator Task: round5_s4_v14_alpha041_ik2w21_from_v12

Name: Round5 S4 full-pipeline v14_alpha041_ik2w21_from_v12

Status: completed

Type: eval

Start: 2026-06-07T01:33:38

End: 2026-06-07T01:39:28

GPU: 1

PID: 2051046

Return code: 0

Command: `ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.36998434019089 |
| Local SIP | 10.065641403198242 |
| Local Angle | 8.723715782165527 |
| Local Joint | 4.467381954193115 |
| Local Mesh | 5.108052682876587 |
| Global SIP | 10.230408477783204 |
| Global Angle | 8.469679355621338 |
| Global Joint | 4.2901660919189455 |
| Global Mesh | 4.819606161117553 |
| Root Jitter | 0.2872037500143051 |
| Joint Jitter | 0.47845168113708497 |

### Orchestrator Task: round5_s4_v14_repeat_alpha040_ik2w2_from_v12

Name: Round5 S4 full-pipeline v14_repeat_alpha040_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T01:33:22

End: 2026-06-07T01:39:58

GPU: 1

PID: 2112683

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.37586713039875 |
| Local SIP | 10.066961097717286 |
| Local Angle | 8.724554443359375 |
| Local Joint | 4.46775074005127 |
| Local Mesh | 5.108734178543091 |
| Global SIP | 10.232294654846191 |
| Global Angle | 8.471344089508056 |
| Global Joint | 4.291537094116211 |
| Global Mesh | 4.821682024002075 |
| Root Jitter | 0.2871789067983627 |
| Joint Jitter | 0.47840615510940554 |

### Orchestrator Task: round5_s4_v14_alpha041_ik2w2_from_v12

Name: Round5 S4 full-pipeline v14_alpha041_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T01:33:22

End: 2026-06-07T01:40:29

GPU: 0

PID: 2112684

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/s4/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 38.37346911524236 |
| Local SIP | 10.066622066497803 |
| Local Angle | 8.724538898468017 |
| Local Joint | 4.467956304550171 |
| Local Mesh | 5.1086304664611815 |
| Global SIP | 10.231480598449707 |
| Global Angle | 8.470207786560058 |
| Global Joint | 4.290395879745484 |
| Global Mesh | 4.819811105728149 |
| Root Jitter | 0.2872063972055912 |
| Joint Jitter | 0.47845468372106553 |

### Orchestrator Task: round5_s5_v14_confirm_alpha042_ik2w2_from_v12

Name: Round5 S5 full-pipeline v14_confirm_alpha042_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T01:39:28

End: 2026-06-07T01:44:48

GPU: 0

PID: 2127714

Return code: 0

Command: `ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.71934455584735 |
| Local SIP | 9.611623167991638 |
| Local Angle | 12.444964170455933 |
| Local Joint | 4.325854003429413 |
| Local Mesh | 5.101464867591858 |
| Global SIP | 9.098152041435242 |
| Global Angle | 11.742849826812744 |
| Global Joint | 3.810960590839386 |
| Global Mesh | 4.406104743480682 |
| Root Jitter | 0.375731754116714 |
| Joint Jitter | 0.8073889724910259 |

### Orchestrator Task: round5_s5_v14_alpha041_ik2w21_from_v12

Name: Round5 S5 full-pipeline v14_alpha041_ik2w21_from_v12

Status: completed

Type: eval

Start: 2026-06-07T01:39:28

End: 2026-06-07T01:44:48

GPU: 1

PID: 2127717

Return code: 0

Command: `ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.723319755326955 |
| Local SIP | 9.61390769481659 |
| Local Angle | 12.445979118347168 |
| Local Joint | 4.327014863491058 |
| Local Mesh | 5.1022831201553345 |
| Global SIP | 9.098065972328186 |
| Global Angle | 11.743443727493286 |
| Global Joint | 3.8115137219429016 |
| Global Mesh | 4.407037138938904 |
| Root Jitter | 0.3755719056352973 |
| Joint Jitter | 0.8070383798331022 |

### Orchestrator Task: round5_s5_v14_alpha041_ik2w2_from_v12

Name: Round5 S5 full-pipeline v14_alpha041_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T01:40:29

End: 2026-06-07T01:46:19

GPU: 0

PID: 2116749

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.718747972752894 |
| Local SIP | 9.612440347671509 |
| Local Angle | 12.444794178009033 |
| Local Joint | 4.326958894729614 |
| Local Mesh | 5.1019874811172485 |
| Global SIP | 9.09704864025116 |
| Global Angle | 11.742574453353882 |
| Global Joint | 3.8112422227859497 |
| Global Mesh | 4.4067511558532715 |
| Root Jitter | 0.375562047585845 |
| Joint Jitter | 0.8070241715759039 |

### Orchestrator Task: round5_s5_v14_repeat_alpha040_ik2w2_from_v12

Name: Round5 S5 full-pipeline v14_repeat_alpha040_ik2w2_from_v12

Status: completed

Type: eval

Start: 2026-06-07T01:41:45

End: 2026-06-07T01:48:05

GPU: 1

PID: 2117433

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_official_input_eval.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/s5/best_loss/result.json --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train/best_loss.pt --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| Score | 43.722780938427896 |
| Local SIP | 9.612994074821472 |
| Local Angle | 12.444732189178467 |
| Local Joint | 4.326852738857269 |
| Local Mesh | 5.101378083229065 |
| Global SIP | 9.099579572677612 |
| Global Angle | 11.743505954742432 |
| Global Joint | 3.812170624732971 |
| Global Mesh | 4.407671749591827 |
| Root Jitter | 0.3753584446385503 |
| Joint Jitter | 0.806681064888835 |

### Orchestrator Task: round5_real_s4_v14_confirm_alpha042_ik2w2_from_v12

Name: Round5 S4 real streaming IK1 audit v14_confirm_alpha042_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T01:44:48

End: 2026-06-07T01:49:07

GPU: 0

PID: 2198067

Return code: 0

Command: `ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v14_confirm_alpha042_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.37247389644385 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_confirm_alpha042_ik2w2_from_v12 |

### Orchestrator Task: round5_real_s4_v14_alpha041_ik2w21_from_v12

Name: Round5 S4 real streaming IK1 audit v14_alpha041_ik2w21_from_v12

Status: completed

Type: audit

Start: 2026-06-07T01:44:48

End: 2026-06-07T01:49:07

GPU: 1

PID: 2198070

Return code: 0

Command: `ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v14_alpha041_ik2w21_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.36998434019089 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha041_ik2w21_from_v12 |

### Orchestrator Task: round5_real_s4_v14_alpha041_ik2w2_from_v12

Name: Round5 S4 real streaming IK1 audit v14_alpha041_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T01:46:19

End: 2026-06-07T01:50:53

GPU: 0

PID: 2120347

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v14_alpha041_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.37346911524236 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha041_ik2w2_from_v12 |

### Orchestrator Task: round5_real_s5_v14_confirm_alpha042_ik2w2_from_v12

Name: Round5 S5 real streaming IK1 audit v14_confirm_alpha042_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T01:49:07

End: 2026-06-07T01:52:55

GPU: 0

PID: 2254863

Return code: 0

Command: `ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v14_confirm_alpha042_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_confirm_alpha042_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.71934455584735 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_confirm_alpha042_ik2w2_from_v12 |

### Orchestrator Task: round5_real_s5_v14_alpha041_ik2w21_from_v12

Name: Round5 S5 real streaming IK1 audit v14_alpha041_ik2w21_from_v12

Status: completed

Type: audit

Start: 2026-06-07T01:49:07

End: 2026-06-07T01:52:55

GPU: 1

PID: 2254865

Return code: 0

Command: `ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/remote-envs/globalpose-gpu-py310/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v14_alpha041_ik2w21_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w21_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.723319755326955 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha041_ik2w21_from_v12 |

### Orchestrator Task: round5_real_s5_v14_alpha041_ik2w2_from_v12

Name: Round5 S5 real streaming IK1 audit v14_alpha041_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T01:50:53

End: 2026-06-07T01:55:11

GPU: 0

PID: 2122572

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=0; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v14_alpha041_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_alpha041_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.718747972752894 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_alpha041_ik2w2_from_v12 |

### Orchestrator Task: round5_real_s4_v14_repeat_alpha040_ik2w2_from_v12

Name: Round5 S4 real streaming IK1 audit v14_repeat_alpha040_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T02:06:20

End: 2026-06-07T02:10:54

GPU: 1

PID: 2129146

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/real_streaming/s4/best_loss/result.json --split-label S4 --version-name v14_repeat_alpha040_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/real_s4_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/real_streaming/s4/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 38.37586713039875 |
| split_label | S4 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_repeat_alpha040_ik2w2_from_v12 |

## IMUOffsetNet Status

Date: 2026-06-07

Current state: implemented and smoke-tested, not yet proven useful downstream.

What is implemented:

- `IMUOffsetNet` with `offset_v1_mlp_frame`, `offset_v2_temporal_rnn`, `offset_v3_residual_prior`.
- Offset contract: `r_JS`, joint-local sensor origin offset, sequence-level cache field `offset_r [6,3]`.
- Stage A synthetic supervised training with true synthetic offset GT.
- Stage B DIP fine-tune script using only official `aM/wM/RMB`, no processed IMU, no trans loss, no real offset GT.
- Stage C TotalCapture processed-IMU inference/diagnostic.
- Formal task file: `configs/imu_offset_net_20260607_tasks.json`.

Smoke evidence:

| Version | Synthetic offset L2 ↓ | Acc consistency ↓ | Real-data downstream |
|---|---:|---:|---|
| `offset_v1_mlp_frame` | `12.36579 cm` | `4.09173` | not measured |
| `offset_v2_temporal_rnn` | `13.22492 cm` | `4.22182` | not measured |
| `offset_v3_residual_prior` | `4.32751 cm` | `1.51809` | TC one-seq PL smoke no improvement |

TotalCapture one-sequence processed-IMU downstream smoke:

| Cache | Score ↓ |
|---|---:|
| Existing baseline/cache offset | `42.153346` |
| IMUOffsetNet predicted offset | `42.153350` |

Conclusion: continue only with `offset_v3_residual_prior` if scaling this direction. Current evidence says synthetic offset learning works, but predicted real-data offsets do not yet improve PL/full-pipeline metrics. DIP/TotalCapture offset accuracy remains `not available`; do not report synthetic offset accuracy as real-data accuracy.

Formal update, 2026-06-07:

| Requirement | Status |
|---|---|
| Three modules implemented | done: v1 MLP, v2 GRU, v3 residual-prior |
| AMASS synthetic pretrain | done for v1/v2/v3 |
| DIP official-input fine-tune | done for v3; uses `aM/wM/RMB`, no processed IMU |
| DIP/TC offset GT handling | done; real offset accuracy marked `not available` |
| TotalCapture processed diagnostic | done for S4/S5 inference and PL downstream |
| Downstream utility | measured; no meaningful improvement |

Formal Stage A synthetic best offset L2:

| Version | Best offset L2 cm ↓ |
|---|---:|
| `offset_v1_mlp_frame` | `4.26099` |
| `offset_v2_temporal_rnn` | `4.24690` |
| `offset_v3_residual_prior` | `4.25810` |

Downstream utility:

| Dataset/protocol | Baseline/cache Score ↓ | IMUOffsetNet predicted-offset Score ↓ | Decision |
|---|---:|---:|---|
| DIP test, official input, original GPNet | `44.642049` | `44.642049` | no improvement |
| TotalCapture S4, processed input, PL eval | `38.625657` | `38.625657` | no meaningful improvement |
| TotalCapture S5, processed input, PL eval | `43.811277` | `43.811278` | no improvement |

Current decision: `IMUOffsetNet` is not worth integrating into the selected NewPL/NewIK1 pipeline in its current form. The next useful change is not longer training; it should be a different real-data consistency objective, offset coordinate/interface redesign, or explicit downstream use of the predicted offset beyond GPNet initialization.

### Orchestrator Task: round5_real_s5_v14_repeat_alpha040_ik2w2_from_v12

Name: Round5 S5 real streaming IK1 audit v14_repeat_alpha040_ik2w2_from_v12

Status: completed

Type: audit

Start: 2026-06-07T02:10:54

End: 2026-06-07T02:14:57

GPU: 1

PID: 2130441

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export CUDA_VISIBLE_DEVICES=1; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "/home/lingfeng/.conda/envs/globalpose-gpu/bin/python" newik1_real_streaming_audit.py --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json --output-json data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/real_streaming/s5/best_loss/result.json --split-label S5 --version-name v14_repeat_alpha040_ik2w2_from_v12 --pl-checkpoint data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt --ik1-checkpoint data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train/best_loss.pt --ik1-backend official_input_v1 --imu-input-mode processed`

Log: `logs/orchestrator/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/real_s5_best_loss.log`

Outputs:

- `data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/real_streaming/s5/best_loss/result.json`

Summary:

| metric | value |
|---|---:|
| all_finite | True |
| ik1_backend | official_input_v1 |
| ik1_checkpoint | data/experiments/ik1_auto_search/round5_confirmation_from_v12/v14_repeat_alpha040_ik2w2_from_v12/train/best_loss.pt |
| imu_input_mode | processed |
| metric_contract | Real dataset streaming/full-pipeline run. IK1 module metrics compare raw IK1 pRJ[69]+gR2[3] from GPNet forward_until_ik1 against GT from the same S4/S5 sequence. |
| pl_checkpoint | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt |
| score | 43.722780938427896 |
| split_label | S5 |
| status | ok |
| val_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/test_Roffset_A/baseline_cache_manifest.json |
| version_name | v14_repeat_alpha040_ik2w2_from_v12 |

### Orchestrator Task: stageA_v1_synthetic

Name: Stage A synthetic supervised offset_v1_mlp_frame

Status: completed

Type: train

Start: 2026-06-07T03:19:26

End: 2026-06-07T03:19:57

GPU: 1

PID: 2154840

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" imu_offset_train.py --version offset_v1_mlp_frame --output-dir data/experiments/imu_offset_net_20260607/stageA_v1 --max-shards 2 --max-sequences 96 --max-frames 900 --window 120 --windows-per-sequence 2 --val-windows-per-sequence 1 --epochs 5 --hidden-size 256 --acc-device cpu`

Log: `logs/orchestrator/imu_offset_net_20260607/stageA_v1.log`

Outputs:

- `data/experiments/imu_offset_net_20260607/stageA_v1/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 2 |
| best_loss | None |

### Orchestrator Task: stageA_v2_synthetic

Name: Stage A synthetic supervised offset_v2_temporal_rnn

Status: completed

Type: train

Start: 2026-06-07T03:19:26

End: 2026-06-07T03:19:57

GPU: 0

PID: 2154841

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" imu_offset_train.py --version offset_v2_temporal_rnn --output-dir data/experiments/imu_offset_net_20260607/stageA_v2 --max-shards 2 --max-sequences 96 --max-frames 900 --window 120 --windows-per-sequence 2 --val-windows-per-sequence 1 --epochs 5 --hidden-size 256 --acc-device cpu`

Log: `logs/orchestrator/imu_offset_net_20260607/stageA_v2.log`

Outputs:

- `data/experiments/imu_offset_net_20260607/stageA_v2/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 2 |
| best_loss | None |

### Orchestrator Task: stageA_v3_synthetic

Name: Stage A synthetic supervised offset_v3_residual_prior

Status: completed

Type: train

Start: 2026-06-07T03:19:57

End: 2026-06-07T03:20:12

GPU: 1

PID: 2155443

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" imu_offset_train.py --version offset_v3_residual_prior --output-dir data/experiments/imu_offset_net_20260607/stageA_v3 --max-shards 2 --max-sequences 96 --max-frames 900 --window 120 --windows-per-sequence 2 --val-windows-per-sequence 1 --epochs 5 --hidden-size 256 --prior-weight 0.01 --smooth-weight 0.001 --acc-device cpu`

Log: `logs/orchestrator/imu_offset_net_20260607/stageA_v3.log`

Outputs:

- `data/experiments/imu_offset_net_20260607/stageA_v3/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | 2 |
| best_loss | None |

### Orchestrator Task: stageB_v3_dip_finetune

Name: Stage B DIP official-input consistency fine-tune from v3

Status: completed

Type: train

Start: 2026-06-07T03:20:12

End: 2026-06-07T03:20:27

GPU: 1

PID: 2155729

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" imu_offset_finetune_dip.py --init-checkpoint data/experiments/imu_offset_net_20260607/stageA_v3/best_loss.pt --output-dir data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft --epochs 3 --lr 1e-5 --window 120 --windows-per-sequence 1 --val-windows-per-sequence 1`

Log: `logs/orchestrator/imu_offset_net_20260607/stageB_v3_dip_ft.log`

Outputs:

- `data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft/train_result.json`

Summary:

| metric | value |
|---|---:|
| status | ok |
| best_epoch | None |
| best_loss | None |

### Orchestrator Task: stageB_v3_dip_test_infer

Name: Infer DIP test offsets with official IMU, no offset GT

Status: completed

Type: cache

Start: 2026-06-07T03:20:27

End: 2026-06-07T03:20:42

GPU: 1

PID: 2156065

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" imu_offset_infer.py --checkpoint data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft/best_loss.pt --input-cache data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json --output-dir data/experiments/imu_offset_net_20260607/dip_test_v3_pred_offset --imu-input-mode official --pl-source pose_prephysics --offset-gt-mode unavailable --acc-audit-mode auto --window 120 --stride 120`

Log: `logs/orchestrator/imu_offset_net_20260607/dip_test_v3_pred_offset.log`

Outputs:

- `data/experiments/imu_offset_net_20260607/dip_test_v3_pred_offset/imu_offset_infer_result.json`

Summary:

| metric | value |
|---|---:|
| checkpoint | data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft/best_loss.pt |
| coordinate_contract | offset_r is r_JS: IMU origin position relative to mapped joint J, expressed in joint-local coordinates. World position is p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS. |
| imu_input_mode | official |
| input_cache | data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json |
| offset_gt_mode | unavailable |
| output_manifest | data/experiments/imu_offset_net_20260607/dip_test_v3_pred_offset/baseline_cache_manifest.json |
| pl_source | pose_prephysics |
| status | ok |
| version | offset_v3_residual_prior |

## NewPL-root module training and module-level evaluation

Status date: 2026-06-07

`newpl_root_v1` is implemented as a module-level PL diagnostic, not a current mainline replacement.

| Item | Status |
|---|---|
| Module structure | PLCurve-style 21D state |
| Input | `aRB[18]+wRB[18]+RRB[45]+gR0[3] = 84D` |
| Output | `pRB[15]+gR1[3]+root_vel[3] = 21D` |
| Init | `offset_r[18]+pRL[15]+gR0[3] = 36D` |
| root_vel frame | root/body frame, m/s |
| AMASS long pretrain | implemented, not run |
| TotalCapture fine-tune | implemented, not run beyond smoke |
| DIP fine-tune | implemented with DIP root_vel GT disabled, not run |
| Full-pipeline 11 metrics | intentionally not run |
| Mainline selection | no |

Files:

- `newpl_root.py`
- `newpl_root_train.py`
- `newpl_root_eval.py`

Root velocity rule: AMASS and TotalCapture may use translation-derived root velocity. DIP must not use `tran_gt` for root velocity supervision or evaluation; the scripts reject DIP root-velocity GT mode.

Smoke evidence:

| Dataset | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | root_vel L1 ↓ | root_vel L2 ↓ | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| TotalCapture | official_PL | `3.607639` | `7.687036` | `11.658890` | not applicable | not applicable | single-sequence smoke |
| TotalCapture | newpl_v4_init36 | `3.412260` | `7.253721` | `11.529694` | not applicable | not applicable | single-sequence smoke |
| TotalCapture | newpl_root_v1_smoke | `3.411824` | `7.252498` | `11.529869` | `0.260059` | `0.553485` | single-sequence smoke |
| DIP-IMU | newpl_root_v1_smoke | `3.462420` | `7.301293` | `12.611866` | root_vel GT not available | root_vel GT not available | DIP trans not used |

Artifacts:

- Smoke checkpoint: `data/experiments/newpl_root_v1/smoke/tc_train_smoke/best_loss.pt`
- Smoke TC JSON: `data/experiments/newpl_root_v1/smoke/tc_multi_module_smoke.json`
- Smoke DIP JSON: `data/experiments/newpl_root_v1/smoke/dip_root_module_smoke.json`

Next decision gate: run AMASS long pretrain and module-level AMASS/TotalCapture/DIP eval tables before considering IK1/full-pipeline integration. Current answers to improvement questions are `not measured`, except that DIP root velocity is not reliably evaluable because official DIP-IMU does not provide trustworthy global translation.

Fair comparison requirement: every NewPL-root evaluation must compare `official_PL`, `newpl_v4_init36`, and `newpl_root_v1` on `pRB L1 cm`, `pRB L2 cm`, per-leaf pRB L2, and `gR1 angle deg`. For root velocity, compare `GT`, official baseline velocity, `newpl_v4_init36` baseline velocity, and `newpl_root_v1` direct head when GT is available. Baseline velocity comes from official pipeline velocity if exposed, otherwise finite-difference final pipeline translation. If root velocity GT is not reliable, mark it unavailable and do not infer a comparison.

Long training status: `scripts/run_newpl_root_v1_longtrain_20260607.sh` was started with `longrun` on GPU 0. Results are expected under `data/experiments/newpl_root_v1/longrun_20260607/`; until those JSONs exist, final AMASS/TotalCapture/DIP conclusions remain `not measured`.

## IMU position offset estimation for NewPL

Date: 2026-06-07

Purpose: estimate explicit IMU position offsets for NewPL without fabricating real-data offset GT. This is a diagnostic module study, not an official protocol change.

Coordinate/data contract:

- Offset is `r_JS`: IMU origin relative to mapped joint `J`, expressed in the joint-local frame.
- World reconstruction is `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS`.
- DIP-IMU uses official baseline `aM/wM/RMB`; processed IMU, trusted `trans`, and real offset GT loss are not used.
- TotalCapture processed IMU is allowed only for diagnostic/adaptation; TotalCapture offset GT remains `not available`.

Implemented files:

- `imu_position_offset.py`: shared solver/net/hybrid utilities and `OFFSET_POSITION_CONTRACT`.
- `scripts/build_imu_position_offsets.py`: builds `zero`, `random`, `solver_v1`, `net_v2`, and `hybrid_v3` sequence offset caches.
- `pl_curve_cache.py`: accepts external `--offset-cache` and `--max-sequences`.
- `pl_curve_offset_sensitivity_eval.py`: audits whether different offset caches change NewPL output.
- `scripts/summarize_imu_offset_newpl.py`: creates summary JSON/Markdown tables.
- `pl_curve_train.py`: optional diagnostic `--offset-contrast-*`, `--offset-init-dropout-prob`, and `--offset-init-noise-std`; defaults preserve old behavior.

Method/design summary:

- Surveyed offset estimation families: kinematic consistency, lever-arm acceleration, angular-acceleration relation, bone/magnitude constraints, optimization-based calibration, self-supervised temporal consistency, and learned residual refinement.
- Chosen coordinate frame is `r_JS`, the IMU origin relative to mapped joint `J`, expressed in joint-local coordinates. World reconstruction is `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS`.
- `offset_solver_v1_kinematic_opt`: ridge-regularized lever-arm acceleration least-squares with plausibility projection.
- `offset_net_v2_selfsup`: AMASS synthetic sanity pretrain plus DIP official-input self-supervised pose-acc proxy; no DIP `trans`, no real offset GT.
- `offset_hybrid_v3_opt_init_net_refine`: solver initialization plus learned OffsetNet residual/blend.
- AMASS synthetic offset L1/L2 is diagnostic only; DIP/TotalCapture real offset GT is `not available`.
- NewPL external 84D input and 18D `pRB[15]+gR1[3]` output are preserved; init36 remains `offset_r[18]+pRL[15]+gR0[3]`.

Implemented routes:

| Route | Algorithm | Offset frame | Current artifact |
|---|---|---|---|
| `offset_solver_v1_kinematic_opt` | lever-arm acceleration least-squares with ridge and magnitude projection | `r_JS` joint-local | `data/experiments/imu_position_offset_newpl/tc_val_2seq/solver_v1_offsets.pt` |
| `offset_net_v2_selfsup` | StageA synthetic OffsetNet, StageB DIP self-supervised fine-tune | `r_JS` joint-local | `data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft/best_loss.pt` |
| `offset_hybrid_v3_opt_init_net_refine` | solver initial offset plus learned residual/blend | `r_JS` joint-local | `data/experiments/imu_position_offset_newpl/tc_val_2seq/hybrid_v3_offsets.pt` |

TotalCapture 2-sequence NewPL smoke, current selected `newpl_v4_init36`:

| Method | Offset median m | PL pRB orig cm | PL pRB NewPL cm | Delta cm | gR1 orig deg | gR1 NewPL deg | Delta deg | Output diff vs zero cm | IK1 | Full 11 metrics |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| zero | `0` | `9.07321` | `8.69193` | `-0.38128` | `26.8333` | `26.6172` | `-0.21607` | `0` | not measured | not measured |
| random | `0.224814` | `9.07321` | `8.69193` | `-0.38128` | `26.8333` | `26.6172` | `-0.21607` | `1.66547e-07` | not measured | not measured |
| solver_v1 | `0.0654287` | `9.07321` | `8.69193` | `-0.38128` | `26.8333` | `26.6172` | `-0.21607` | `3.72451e-08` | not measured | not measured |
| net_v2 | `0.153126` | `9.07321` | `8.69193` | `-0.38128` | `26.8333` | `26.6172` | `-0.21607` | `8.12792e-08` | not measured | not measured |
| hybrid_v3 | `0.184449` | `9.07321` | `8.69193` | `-0.38128` | `26.8333` | `26.6172` | `-0.21607` | `6.61554e-08` | not measured | not measured |

Offset-sensitive NewPL smoke:

- Checkpoint: `data/experiments/imu_position_offset_newpl/newpl_offset_sensitive_smoke_v2/best_loss.pt`.
- Train config: 8 TotalCapture train sequences, 2 val sequences, 3 epochs, lr `1e-4`, residual scale `0.1`, offset contrast weight `1.0`, offset dropout `0.15`, offset noise std `0.03`.
- Best validation loss: `0.513764` at epoch 2.
- Sensitivity still remains numerical-noise scale: random-vs-zero output diff mean `2.489e-06 cm`, solver-vs-zero `6.927e-07 cm`, net-vs-zero `3.980e-08 cm`, hybrid-vs-zero `6.272e-07 cm`.

Decision: the offset cache path works, but current NewPL does not use `offset_r` meaningfully. Do not run expensive IK1/full-pipeline five-method comparisons until NewPL is retrained or redesigned to be offset-sensitive. IK1 metrics, full-pipeline 11 metrics, DIP test metrics, and TotalCapture full metrics for `zero/random/solver_v1/net_v2/hybrid_v3` remain `not measured`.

### Offset-conditioned NewPL smoke v1

Date: 2026-06-07

Change: added diagnostic `PLCurveOffsetConditionedModule` in `pl_curve.py`. The PL frame input remains official 84D and output remains official 18D, but the init36 encoding is injected at every recurrent step through a condition branch. This is a diagnostic NewPL variant, not an official protocol.

Artifacts:

- Checkpoint: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_smoke_v1/best_loss.pt`
- Train log: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_smoke_v1/train_log.jsonl`
- Train result: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_smoke_v1/train_result.json`
- Sensitivity JSON: `data/experiments/imu_position_offset_newpl/tc_val_2seq/pl_offset_sensitivity_eval_offset_conditioned_smoke_v1.json`
- PL eval table JSON: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_conditioned_pl_eval_table.json`

Training config: 8 TotalCapture train sequences, 2 val sequences, 3 epochs, lr `1e-4`, residual scale `0.05`, condition scale `1.0`, offset contrast weight `1.0`, offset dropout `0.10`, offset noise std `0.02`. Best validation loss is `0.515810` at epoch 3.

Sensitivity versus zero offset improved from numerical noise to small but measurable output changes:

| Method vs zero | Output diff cm | gR diff deg |
|---|---:|---:|
| random | `5.493e-04` | `1.908e-04` |
| solver_v1 | `2.802e-04` | `1.173e-04` |
| net_v2 | `1.628e-03` | `1.736e-04` |
| hybrid_v3 | `2.041e-03` | `1.852e-04` |

TotalCapture 2-sequence PL-level metrics with offset-conditioned checkpoint:

| Method | pRB orig cm | pRB NewPL cm | Delta cm | gR1 orig deg | gR1 NewPL deg | Delta deg |
|---|---:|---:|---:|---:|---:|---:|
| zero | `9.073207` | `9.123797` | `0.050590` | `26.833309` | `25.337427` | `-1.495884` |
| random | `9.073207` | `9.123161` | `0.049954` | `26.833309` | `25.337582` | `-1.495731` |
| solver_v1 | `9.073207` | `9.123602` | `0.050395` | `26.833309` | `25.337492` | `-1.495819` |
| net_v2 | `9.073207` | `9.122195` | `0.048988` | `26.833309` | `25.337856` | `-1.495458` |
| hybrid_v3 | `9.073207` | `9.121889` | `0.048682` | `26.833309` | `25.337954` | `-1.495360` |

Decision: the conditioned structure makes offset changes measurable, but this smoke does not show clear NewPL improvement from better offsets. gR1 improves relative to original PL, pRB worsens slightly, and net/hybrid are only about `0.001-0.002 cm` better than zero on pRB. IK1/full-pipeline 11 metrics remain `not measured`; the next useful step is stronger offset-conditioned training or a loss that ties offset to pRB improvement before full-pipeline evaluation.

### Offset-conditioned pairwise v2

Date: 2026-06-07

Change: adjusted `pl_curve_train.py` offset contrast so the good-offset branch also receives supervision gradient: `good_metric + relu(good_metric + margin - bad_metric)`. This tests whether NewPL can learn a direct preference for correct offset over swapped/rolled offsets.

Artifacts:

- Checkpoint: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_pairwise_v2/best_loss.pt`
- Train result: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_pairwise_v2/train_result.json`
- Swap eval: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_swap_eval_pairwise_v2_hybrid_cache.json`
- Script: `pl_curve_offset_swap_eval.py`

Training result: best validation loss improved to `0.505689` at epoch 5. Training-set `offset_bad_minus_good_metric` ended at only `3.59e-08`, so the pairwise loss did not create strong train-time separability.

Swap eval on 2 TotalCapture validation sequences, using hybrid offset as good and replacing only `offset_r[18]`:

| Bad init | pRB delta vs good cm | gR1 delta vs good deg | PL GT loss delta vs good |
|---|---:|---:|---:|
| zero | `+0.020648` | `-0.011161` | `-1.132e-04` |
| roll_sensors | `+0.012551` | `-0.006490` | `-6.156e-05` |
| other_sequence | `+0.000510` | `-0.000966` | `-1.961e-05` |
| negate | `+0.027347` | `-0.014600` | `-1.457e-04` |

Interpretation: pRB alone now prefers the good offset over zero/rolled/negated offsets, but the combined PL GT loss still prefers bad offsets because the gR1 term moves in the opposite direction. Therefore the offset signal is only partially useful and still not a reliable NewPL selection objective. IK1/full-pipeline 11 metrics remain `not measured`.

### Offset-conditioned pRB contrast v1

Date: 2026-06-07

Change: `pl_curve_train.py` now supports `--offset-contrast-target {full_pl,pRB}`. The default `full_pl` preserves previous behavior; `pRB` restricts the diagnostic contrast metric to the first 15 NewPL output channels. The NewPL external input/output contract is unchanged.

Artifacts:

- Checkpoint: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_prb_contrast_v1/best_loss.pt`
- Train result: `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_prb_contrast_v1/train_result.json`
- Swap eval: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_swap_eval_prb_contrast_v1_hybrid_cache.json`

Training config: initialized from `newpl_offset_conditioned_pairwise_v2/best_loss.pt`, 8 TotalCapture train sequences, 2 validation sequences, 5 epochs, lr `5e-5`, offset contrast weight `2.0`, contrast target `pRB`, offset dropout `0.05`, offset noise std `0.01`. Best validation loss was `0.502297` at epoch 1; final validation loss degraded to `0.526568`, so the run overfit the tiny diagnostic set.

Swap eval on 2 TotalCapture validation sequences, using hybrid offset as good:

| Bad init | pRB delta vs good cm | gR1 delta vs good deg | PL GT loss delta vs good |
|---|---:|---:|---:|
| zero | `+0.018504` | `-0.010733` | `-1.143e-04` |
| roll_sensors | `+0.011441` | `-0.006266` | `-6.256e-05` |
| other_sequence | `+0.000239` | `-0.001027` | `-2.113e-05` |
| negate | `+0.024665` | `-0.014092` | `-1.479e-04` |

Decision: pRB-only contrast did not solve the NewPL offset problem. Good offset is still slightly better for pRB, but the effect is only `0.01-0.025 cm`, train-time good/bad separability is approximately zero, and the full PL loss still prefers bad offsets due to gR1. IK1 metrics and full-pipeline 11 metrics remain `not measured`; next work should use a stronger physical forward-IMU consistency objective or redesign the downstream offset injection before spending on full-pipeline evaluation.

### Offset forward-consistency eval v1

Date: 2026-06-07

Added `imu_position_offset_consistency_eval.py` to evaluate offsets without real offset GT. It computes a forward lever-arm acceleration residual from `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS` and compares it with measured IMU acceleration. This is diagnostic consistency only, not true sensor-position accuracy.

Artifact:

- JSON: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_consistency_eval_v1.json`

TotalCapture 2-sequence aggregate:

| Method | Mean residual m/s^2 | Mean improvement vs zero m/s^2 | Median offset norm m |
|---|---:|---:|---:|
| zero | `13.416491` | `0.000000` | `0.000000` |
| random | `15.256863` | `-1.840372` | `0.224814` |
| solver_v1 | `13.337500` | `0.078991` | `0.065429` |
| net_v2 | `13.096867` | `0.319623` | `0.153126` |
| hybrid_v3 | `13.304113` | `0.112376` | `0.184449` |

Interpretation: the offset estimates are physically meaningful under this diagnostic because random offsets degrade acceleration consistency, while solver/net/hybrid improve it. `net_v2` is best on this tiny TotalCapture smoke. This still does not select a NewPL replacement: previous PL metrics show the downstream NewPL currently does not convert this consistency improvement into clear `pRB/gR1` gains.

### DIP self-supervised OffsetNet v4 smoke

Date: 2026-06-07

`imu_offset_finetune_dip.py` now records `initial_val.json` before training and writes `initial_val` / `best_epoch` into `train_result.json`. This makes the no-GT DIP self-supervised fine-tune auditable against its initialization checkpoint.

Contract:

- Input: DIP official baseline `aM/wM/RMB`.
- Forbidden: processed IMU, DIP `trans_loss`, real offset GT loss.
- Supervision: pose-derived acceleration proxy, temporal/magnitude/stability regularizers.
- Offset GT metrics: `not available`.

Artifacts:

- Initial validation: `data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/initial_val.json`
- Train result: `data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/train_result.json`
- Checkpoint: `data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/best_loss.pt`

Tiny smoke result, 4 DIP train sequences / 2 val sequences / max 300 frames:

| Metric | Initial | Last/best | Delta |
|---|---:|---:|---:|
| DIP val pose_acc_proxy | `0.018450846` | `0.018450512` | `-3.34e-07` |
| offset magnitude m | `0.174248368` | `0.174248084` | `-2.83e-07` |
| best epoch | not applicable | `2` | not applicable |
| offset L1/L2 cm | not available | not available | not available |

Decision: the DIP self-supervised fine-tune path is valid and auditable, but this smoke shows only numerical-scale improvement. Do not select `stageB_v4_dip_consistency_smoke` as a better offset checkpoint, and do not infer NewPL/full-pipeline improvement from it.

### Offset-NewPL decision matrix v1

Date: 2026-06-07

`scripts/summarize_imu_offset_newpl.py` now emits a combined decision matrix with `--decision-json` / `--decision-md`. This is the current authoritative comparison for the offset-to-NewPL diagnostic because it combines:

- TotalCapture forward lever-arm consistency.
- Offset-conditioned NewPL `pRB/gR1` module metrics.
- DIP self-supervised fine-tune status.
- Explicit `not measured` downstream IK1/full-pipeline fields.
- Real offset GT and DIP trans leakage notes.

Artifacts:

- JSON: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.json`
- Markdown: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.md`

Current decision matrix:

| Method | Forward residual m/s^2 | Forward improvement | Cond. PL pRB vs zero cm | Cond. PL gR1 vs zero deg | IK1 | Full 11 | Decision |
|---|---:|---:|---:|---:|---|---|---|
| zero | `13.416491` | `0.000000` | `0.000000` | `0.000000` | not measured | not measured | not selected |
| random | `15.256863` | `-1.840372` | `-0.000636` | `0.000154` | not measured | not measured | negative control |
| solver_v1 | `13.337500` | `0.078991` | `-0.000196` | `0.000065` | not measured | not measured | physical signal, downstream not selected |
| net_v2 | `13.096867` | `0.319623` | `-0.001602` | `0.000429` | not measured | not measured | physical signal, downstream not selected |
| hybrid_v3 | `13.304113` | `0.112376` | `-0.001908` | `0.000526` | not measured | not measured | physical signal, downstream not selected |

Selection:

```text
best_offset_method_by_forward_consistency = net_v2
best_offset_method_for_newpl = not selected
run_ik1_or_full_pipeline = false
```

Reason: solver/net/hybrid offsets improve physical acceleration consistency, but NewPL gains are tiny and partly conflicting. The current downstream interface still does not make reliable use of `r_JS`.

### StageB v4 OffsetNet transfer check

Date: 2026-06-07

Checked whether `stageB_v4_dip_consistency_smoke/best_loss.pt` improves TotalCapture offset consistency enough to warrant a separate NewPL cache/evaluation.

Artifacts:

- Offset cache: `data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_stageB_v4_offsets.pt`
- Offset summary: `data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_stageB_v4_offsets.json`
- Consistency compare: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_consistency_eval_stageB_v4_compare.json`
- Updated decision matrix: `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.json`

Comparison:

| Method | Mean residual m/s^2 | Improvement vs zero m/s^2 | Median offset norm m |
|---|---:|---:|---:|
| net_v2 | `13.096867` | `0.319623` | `0.153126` |
| net_v2_stageB_v4 | `13.096842` | `0.319649` | `0.153162` |

Decision: v4 improves forward consistency over the existing `net_v2` by only `2.56e-05 m/s^2`. This is numerical-scale and does not justify building a separate NewPL cache or running IK1/full-pipeline metrics. The decision matrix remains unchanged: best physical-consistency method is `net_v2`, best NewPL method is `not selected`.

### Orchestrator Task: stageC_v3_tc_val_infer

Name: Infer TotalCapture S4 offsets with processed IMU diagnostic

Status: completed

Type: cache

Start: 2026-06-07T03:20:27

End: 2026-06-07T03:20:42

GPU: 0

PID: 2156066

Return code: 0

Command: `ENV=/home/lingfeng/.conda/envs/globalpose-gpu; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"; "$ENV/bin/python" imu_offset_infer.py --checkpoint data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft/best_loss.pt --input-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json --output-dir data/experiments/imu_offset_net_20260607/tc_val_v3_pred_offset --imu-input-mode processed --pl-source pose_prephysics --offset-gt-mode unavailable --acc-audit-mode pose_gt --window 120 --stride 120`

Log: `logs/orchestrator/imu_offset_net_20260607/tc_val_v3_pred_offset.log`

Outputs:

- `data/experiments/imu_offset_net_20260607/tc_val_v3_pred_offset/imu_offset_infer_result.json`

Summary:

| metric | value |
|---|---:|
| checkpoint | data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft/best_loss.pt |
| coordinate_contract | offset_r is r_JS: IMU origin position relative to mapped joint J, expressed in joint-local coordinates. World position is p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS. |
| imu_input_mode | processed |
| input_cache | data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json |
| offset_gt_mode | unavailable |
| output_manifest | data/experiments/imu_offset_net_20260607/tc_val_v3_pred_offset/baseline_cache_manifest.json |
| pl_source | pose_prephysics |
| status | ok |
| version | offset_v3_residual_prior |
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

Fine-tune before/after checkpoint audit:

- JSONs: `data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/`
- AMASS: `last.pt` improves decoded PL over selected `best_loss.pt` (`pRB L2 3.415629` vs `3.460373`, `gR1 4.865114` vs `4.869618`), but root velocity L1/L2 is slightly worse (`0.193132/0.427571` vs `0.193051/0.427344`) and both are worse than pipeline-derived baseline velocity.
- TotalCapture: fine-tune helps slightly. Before fine-tune `pRB L2 6.825256`, `gR1 13.408251`; TC best `6.779195`, `13.376897`; TC last `6.776795`, `13.375096`. Direct root velocity also improves only minimally from `0.269022/0.587931` to `0.268900/0.587688`, still much worse than the baseline pipeline velocity (`newpl_v4_init36`: `0.114272/0.232193`).
- DIP: fine-tune does not help. Before fine-tune `pRB L2 6.428928`, `gR1 12.852417`; DIP best `6.429121`, `12.854242`; DIP last `6.430589`, `12.864961`. DIP root velocity GT remains not available and was not used.
- Operational note: full fair root-velocity baseline evaluation is slow because official/newpl_v4 velocity must be computed by running final pipeline translation and finite differencing it. For checkpoint audits, use existing baseline velocity JSONs and evaluate NewPL-root head directly unless a new baseline trajectory is required.

## IMU position offset completion status

Date: 2026-06-07

Answer: the previous IMU position-offset experiment branch is complete for the current design and is not selected.

Completed:

- `IMUOffsetNet` v1/v2/v3 implemented and trained on AMASS synthetic offset GT.
- DIP official-input fine-tune/eval was run without processed IMU, DIP `trans`, or real offset GT.
- TotalCapture predicted-offset caches were generated for S4/S5 diagnostics.
- Downstream PL utility was checked on DIP and TotalCapture.
- Solver/net/hybrid offset routes and offset-conditioned NewPL diagnostics were evaluated.

Key results:

| Evidence | Result |
|---|---|
| AMASS synthetic offset L1/L2 | L1 is around `2.x cm`; formal v3 last L1 `2.18759 cm`, best L2 `4.25810 cm`, last L2 `4.25904 cm` |
| DIP predicted-offset downstream | `44.642049 -> 44.642049`, no effect |
| TotalCapture S4 predicted-offset downstream | `38.625657 -> 38.625657`, no effect |
| TotalCapture S5 predicted-offset downstream | `43.811277 -> 43.811278`, no improvement |
| Best physical consistency route | `net_v2`, residual `13.096867 m/s^2` vs zero `13.416491` |
| Best NewPL route | not selected |
| IK1/full-pipeline 11 metrics | not measured by design |

Reason for stopping: predicted offsets have physical signal under forward acceleration consistency, but current NewPL/PL init path does not convert that signal into meaningful `pRB/gR1` or score gains. Running IK1/full-pipeline metrics for this branch would be low-value until offset injection or physical loss is redesigned.

Next action if revisited: redesign the downstream offset interface first, for example stronger per-frame offset conditioning, forward-IMU consistency tied directly to PL output, or a module that explicitly consumes `r_JS`; do not simply rerun longer OffsetNet training.

## NewIK1 official-like retraining status

Date: 2026-06-08

Experiment: `newik1_v10_official_protocol_last_control`

Status: complete and rejected as a replacement.

Route:

- AMASS teacher-forced pretrain.
- AMASS PL-streaming adaptation with `newpl_v5_amass`.
- DIP-IMU PL-streaming fine-tune with `newpl_v5_dip`.
- DIP translation loss was not used.
- TotalCapture train split was not used.

Artifacts:

```text
root: data/experiments/newik1_v10_official_protocol_last_control_20260607
summary: data/experiments/newik1_v10_official_protocol_last_control_20260607/summary.json
log: data/experiments/newik1_v10_official_protocol_last_control_20260607/logs/run_full.log
```

Training result:

| Stage | Best epoch | Best loss |
|---|---:|---:|
| Stage A AMASS teacher-forced | 33 | `0.00048058280081022533` |
| Stage B AMASS PL-streaming | 20 | `0.013035929867764934` |
| Stage C DIP PL-streaming | 40 | `0.13303746217085669` |

Key evaluation:

| Dataset | Best baseline in this protocol | NewIK1 v10 Stage C | Result |
|---|---:|---:|---|
| DIP-IMU test Score | `44.598659` (`newpl_v5_dip + official IK1`) | `44.730331` | worse |
| DIP-IMU test pRJ L2 cm | `5.107541` (`newpl_v5_dip + official IK1`) | `5.087737` | local pRJ better |
| DIP-IMU test gR2 angle deg | `14.869393` (`newpl_v5_dip + official IK1`) | `15.128205` | worse |
| TotalCapture test Score | `43.868067` (`newpl_v5_amass + official IK1`) | `44.900650` | worse |
| TotalCapture test pRJ L2 cm | `4.693114` (`newpl_v5_dip + official IK1`) | `4.754637` | worse |
| TotalCapture test gR2 angle deg | `15.175444` (`newpl_v5_amass + official IK1`) | `15.368013` | worse |

Decision:

- Do not select `newik1_v10_official_protocol_last_control`.
- The official-like training route completed, but local `pRJ` gains on DIP are too small and are offset by worse `gR2` and worse downstream Score.
- Keep official IK1 for the `newpl_v5` official-like route.
- Future NewIK1 work should use downstream-aware objectives and stronger `gR2` preservation/distillation rather than choosing by local IK1 training loss alone.

## Current mainline: NewPose control v2 FK-leaf

Date: 2026-06-08

Status: formal module-level experiment complete; not selected.

Reason: `newpose_ctrl_v1` is rejected because its control/RRJ losses did not preserve decoded FK geometry. The next candidate keeps the control-point output idea but trains and validates against decoded SMPL FK leaf/joint positions.

Implemented:

- `newpose_ctrl.py`: `newpose_ctrl_v2_fk_leaf` loss preset and FK helpers.
- `newpose_ctrl_train.py`: loads `pose_gt/RMB/gR0`, batches them, passes a GPU masked SMPL body model into loss, and supports `--model-type newpose_ctrl_v2_fk_leaf --loss-preset v2_fk_leaf --selection-metric fk_leaf_physical`.
- `newpose_ctrl_eval.py`: module metrics now include `FK_leaf_L1_cm`, `FK_leaf_L2_cm`, per-leaf L2, leaf velocity/acceleration error, leaf jitter, and FK joint L2.
- `newpose_baseline_ik2_module_eval.py`: official/newpl_v5 baselines now report the same FK leaf metrics for fair comparison.
- `scripts/run_newpose_ctrl_v2_fk_leaf_official_protocol_20260608.sh`: smoke, AMASS pretrain, DIP fine-tune, and module-only DIP/TotalCapture eval route. Full-pipeline eval remains opt-in with `RUN_FULL_EVAL=1`.

Important technical notes:

- FK leaf positions are root/body-frame: `(leaf_world - root_world) @ R_root`.
- Only the five leaf vertices and one root vertex are requested from SMPL mesh, avoiding full mesh construction during training.
- The new best checkpoint metric is `fk_leaf_physical`, based on decoded FK leaf/joint closeness rather than raw control loss alone.
- DIP translation/root loss is not used.
- NewPose internal 6D target encoding was corrected to match `art.math.r6d_to_rotation_matrix`'s two-vector convention; this fixed a smoke-time NaN-gradient issue from collinear identity 6D.

Artifacts:

```text
root: data/experiments/newpose_ctrl_v2_fk_leaf_20260608
log: data/experiments/newpose_ctrl_v2_fk_leaf_20260608/logs/run_full.log
summary: data/experiments/newpose_ctrl_v2_fk_leaf_20260608/summary.json
summary tables: data/experiments/newpose_ctrl_v2_fk_leaf_20260608/summary_tables.md
stage_a_best: data/experiments/newpose_ctrl_v2_fk_leaf_20260608/stage_a_amass_pretrain/best_loss.pt
stage_b_best: data/experiments/newpose_ctrl_v2_fk_leaf_20260608/stage_b_dip_finetune/best_loss.pt
```

Training result:

| Stage | Epochs run | Best epoch | Best selection | Last selection | Result |
|---|---:|---:|---:|---:|---|
| AMASS pretrain | 29 | 19 | `16.734057` | `16.780566` | improved from early epochs, then plateaued |
| DIP fine-tune | 11 | 1 | `18.760393` | `19.036661` | did not improve after initialization |

Module-level comparison:

| Dataset | Best baseline FK leaf L2 cm ↓ | Best baseline FK joint L2 cm ↓ | newpose_ctrl_v2 best FK leaf L2 cm ↓ | newpose_ctrl_v2 best FK joint L2 cm ↓ | Result |
|---|---:|---:|---:|---:|---|
| DIP-IMU test | `6.234410` | `4.971124` | `20.285948` | `14.477565` | worse |
| TotalCapture test | `5.766959` | `4.600485` | `18.867359` | `13.679826` | worse |

Decision:

- Do not select `newpose_ctrl_v2_fk_leaf`.
- The FK physical losses fixed part of the `newpose_ctrl_v1` collapse, but the decoded FK leaf/joint gap is still too large.
- DIP fine-tune did not help; best Stage B checkpoint is epoch 1 and later epochs worsen the selection metric.
- Do not connect this module to full pipeline.

<!-- BEGIN newpl-offset-v6---newik1-v11-control-only -->
### NewPL-offset v6 / NewIK1 v11 control-only

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
<!-- END newpl-offset-v6---newik1-v11-control-only -->

## NewPL v7 learned-offset acceleration auxiliary diagnostic

Status: implemented and smoke-tested on 2026-06-12; diagnostic only, not selected. Full-pipeline S4/S5 and 11 metrics were not run.

Contract:

```text
variant: newpl_v7_learned_offset_accaux
PL input: official 84D aRB[18] + wRB[18] + RRB[45] + gR0[3]
stream init: init18 = pRL[15] + gR0[3]; external offset_r is removed from init
PL output: official 18D pRB[15] + gR1[3]
learned offset: bounded global raw_offset[6,3], offset = 0.30 * tanh(raw_offset)
offset frame: r_BS, sensor origin relative to mapped body/sensor frame B, expressed in B
acc auxiliary: compares root-frame non-root-minus-root IMU acceleration residual; no global translation, no DIP trans, no real offset GT
```

Smoke artifacts:

```text
root: data/experiments/newpl_v7_learned_offset_accaux_20260612
summary: data/experiments/newpl_v7_learned_offset_accaux_20260612/summary.md
json: data/experiments/newpl_v7_learned_offset_accaux_20260612/summary.json
checkpoints: checkpoints/stage0_learned_offset.pt, checkpoints/stage1_freeze_offset.pt, checkpoints/stage2_tiny_joint.pt
```

Smoke result on AMASS 2-sequence, 31-frame tiny diagnostic:

| Check | Result |
|---|---:|
| Stage 0 zero-offset acc residual | `8.955655 m/s^2` |
| Stage 0 random-offset acc residual | `11.304108 m/s^2` |
| Stage 0 learned-offset acc residual | `8.928542 m/s^2` |
| Stage 0 learned vs zero improvement | `0.027113 m/s^2` |
| Stage 1 freeze-offset before -> after | `15.339169 -> 15.346781 m/s^2` |
| Stage 1 PL delta pRB/gR1 | `-0.000186 cm / +0.000350 deg` |
| Stage 2 v7 offset norm mean/median/p95 | `0.009579 / 0.008723 / 0.012402 m` |

PL module smoke comparison:

| Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | IMU acc residual ↓ | Notes |
|---|---:|---:|---:|---:|---|
| official PL baseline | `2.402170` | `4.903610` | `6.664704` | `15.211345` | cached official PL, zero-offset residual |
| newpl_v4_init36 baseline | `2.566700` | `5.360255` | `6.979082` | `23.585226` | historical processed-input checkpoint on same smoke cache |
| newpl_v5_dip_best baseline | `2.511445` | `5.193558` | `6.801924` | `24.289366` | official-protocol checkpoint, zero-offset residual |
| newpl_v7_learned_offset_accaux | `2.509856` | `5.190952` | `6.799719` | `24.048222` | learned offset, full-pipeline not measured |

Decision: `diagnostic only`. Stage 0 shows the auxiliary residual is weakly identifiable, and Stage 2 is slightly better than v5 on this tiny cache, but v7 remains worse than the official PL baseline on pRB/gR1. Stage 1 offset-only training does not reduce the residual. Do not start full AMASS -> DIP or connect to IK1 until the acceleration formulation proves useful on broader same-cache module evaluation.

## NewPL v7b local acceleration learned-offset diagnostic

Status: implemented and smoke-tested on 2026-06-12; diagnostic only, not selected. Full-pipeline S4/S5 and 11 metrics were not run.

Contract:

```text
variant: newpl_v7b_local_accaux
PL input: official 84D aRB[18] + wRB[18] + RRB[45] + gR0[3]
stream init: init18 = pRL[15] + gR0[3]; external offset_r is removed from init
PL output: official 18D pRB[15] + gR1[3]
learned offset: bounded leaf-only raw_leaf_offset[5,3], offset = 0.30 * tanh(raw_leaf_offset)
root role: root gyro wRB[5] and alpha_root=dwRB[5]/dt only correct pRB derivatives in the rotating root frame
acc auxiliary: pred_leaf_acc_R = pRB_ddot + 2*w_root x pRB_dot + alpha_root x pRB + w_root x (w_root x pRB) + alpha_leaf x r_leaf + w_leaf x (w_leaf x r_leaf)
forbidden supervision: DIP trans, DIP root velocity, real DIP/TotalCapture offset GT
```

Smoke artifacts:

```text
main root: data/experiments/newpl_v7b_local_accaux_20260612
summary: data/experiments/newpl_v7b_local_accaux_20260612/summary.md
json: data/experiments/newpl_v7b_local_accaux_20260612/summary.json
gravity checks: data/experiments/newpl_v7b_local_accaux_20260612_minus_g, data/experiments/newpl_v7b_local_accaux_20260612_plus_g
checkpoints: checkpoints/stage0_learned_leaf_offset.pt, checkpoints/stage1_freeze_leaf_offset.pt, checkpoints/stage2_tiny_joint.pt
```

Smoke result on AMASS 2-sequence, 31-frame tiny diagnostic:

| Check | Result |
|---|---:|
| Stage 0 zero-offset local acc residual | `8.974846 m/s^2` |
| Stage 0 random-offset local acc residual | `11.928852 m/s^2` |
| Stage 0 GT-offset local acc residual from init36 | `9.755350 m/s^2` |
| Stage 0 learned-offset local acc residual | `8.963714 m/s^2` |
| Stage 0 learned vs zero improvement | `0.011132 m/s^2` |
| Stage 1 freeze-offset before -> after | `15.081704 -> 14.994161 m/s^2` |
| Stage 1 PL delta pRB/gR1 | `+0.000251 cm / +0.000093 deg` |
| Stage 2 v7b offset norm mean/median/p95 | `0.014525 / 0.011973 / 0.019684 m` |

Gravity-mode sensitivity:

| gravity_mode | zero | random | learned | GT offset | Stage1 improvement | Decision |
|---|---:|---:|---:|---:|---:|---|
| none | `8.974846` | `11.928852` | `8.963714` | `9.755350` | `0.087543` | diagnostic only |
| minus_gR0 | `13.902242` | `17.051386` | `13.878829` | `14.238059` | `0.018126` | diagnostic only |
| plus_gR0 | `13.978133` | `15.994410` | `13.964892` | `14.994678` | `0.037277` | diagnostic only |

PL module smoke comparison:

| Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | local acc residual ↓ | Notes |
|---|---:|---:|---:|---:|---|
| official PL baseline | `2.402170` | `4.903610` | `6.664704` | `15.179327` | cached official PL |
| newpl_v4_init36 baseline | `2.566700` | `5.360255` | `6.979082` | `23.832706` | historical checkpoint |
| newpl_v5_dip_best baseline | `2.511445` | `5.193558` | `6.801924` | `24.416668` | official-protocol checkpoint |
| newpl_v7_rootrel_accaux | `2.509856` | `5.190952` | `6.799719` | `24.186819` | previous root-relative accaux |
| newpl_v7b_local_accaux | `2.509855` | `5.190949` | `6.799719` | `24.196295` | local accaux, full-pipeline not measured |

Decision: diagnostic only. v7b fixes the physical issue in v7 by using root gyro/alpha for rotating-frame correction instead of subtracting root acceleration. However, the residual reduction is very small, AMASS init36 GT offset residual is worse than zero in this proxy, and same-cache pRB/gR1 still trail the official PL baseline. Do not start long AMASS -> DIP training or connect to IK1 until the acceleration convention is verified with a stronger FK/RBDL acceleration audit.

### v7b AMASS long diagnostic follow-up

Status: completed on 2026-06-12. This is a longer AMASS module-level diagnostic, not full AMASS -> DIP and not full-pipeline evaluation.

Command:

```text
CUDA_VISIBLE_DEVICES=0 python newpl_v7b_local_accaux_smoke.py --output-dir data/experiments/newpl_v7b_local_accaux_20260612_longtrain --max-sequences 512 --max-train-sequences 512 --batch-size 96 --window 61 --stage0-steps 300 --stage1-steps 300 --stage2-epochs 50 --imu-acc-weight 0.005 --offset-prior-weight 0.001 --joint-lr 1e-5 --offset-lr 1e-3 --gravity-mode none
```

Artifacts:

```text
root: data/experiments/newpl_v7b_local_accaux_20260612_longtrain
summary: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/summary.md
json: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/summary.json
checkpoint: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/checkpoints/stage2_tiny_joint.pt
```

Long diagnostic metrics:

| Check | Result |
|---|---:|
| Stage 0 zero/random/GT/learned local acc residual | `11.099609 / 12.552015 / 11.629623 / 11.071591 m/s^2` |
| Stage 0 learned vs zero improvement | `0.028018 m/s^2` |
| Stage 1 freeze-offset before -> after | `24.562115 -> 24.461527 m/s^2` |
| Stage 2 offset norm mean/median/p95 | `0.073905 / 0.076368 / 0.092299 m` |

Same-cache AMASS module comparison:

| Version | pRB L2 cm ↓ | gR1 angle deg ↓ | local acc residual ↓ |
|---|---:|---:|---:|
| official PL baseline | `3.284492` | `10.588459` | `11.674987` |
| newpl_v4_init36 baseline | `3.298557` | `10.437762` | `28.840921` |
| newpl_v5_dip_best baseline | `3.328090` | `10.252261` | `25.495672` |
| newpl_v7_rootrel_accaux | `3.327924` | `10.253457` | `25.443991` |
| newpl_v7b_local_accaux long | `3.330573` | `10.174911` | `24.781027` |

Conclusion: the longer AMASS diagnostic makes v7b look more useful for the acceleration auxiliary and gR1 than the tiny smoke: local acc residual improves over v5/v7, and gR1 is best among the NewPL variants in this table. pRB still trails official PL and v4/v5 slightly, and the evaluation is still same-cache AMASS module-level only. Treat this as permission to run a proper AMASS -> DIP module-level experiment, not as a selected PL replacement.

## NewPL v5 realtime smooth+residual input

Status: completed on 2026-06-12. This is a module-level NewPL experiment, not a full-pipeline S4/S5 run.

Design:

```text
variant: newpl_v5_realtime_smooth_residual
frame input: aRB_smooth[18] + aRB_residual_raw_minus_smooth[18] + wRB[18] + RRB[45] + gR0[3] = 102D
output: pRB[15] + gR1[3] = 18D
filter: causal_iir, cutoff_hz=20, fs=60, zero lookahead
residual: raw root-frame acceleration - causal-smoothed root-frame acceleration
protocol: AMASS pretrain -> DIP train fine-tune -> DIP/TotalCapture module eval
DIP trans/root velocity: not used
full-pipeline 11 metrics: not measured
```

Artifacts:

```text
root: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20
script: scripts/run_newpl_v5_realtime_residual_20260612.sh
summary: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/summary.md
json: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/summary.json
amass checkpoint: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/amass_pretrain/best_loss.pt
dip checkpoint: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20/dip_finetune/best_loss.pt
```

Key module results:

| Dataset/stage | Version | pRB L2 cm ↓ | gR1 angle deg ↓ | Readout |
|---|---|---:|---:|---|
| DIP test after DIP FT | official PL | `6.528883` | `15.267228` | same candidate eval cache/source |
| DIP test after DIP FT | raw newpl_v5_dip_best | `6.445578` | `12.552613` | historical reference from `newpl_v5_official_protocol_20260607_tuned`, not same-cache fairness row |
| DIP test after DIP FT | realtime smooth+residual | `6.557889` | `14.557139` | versus same-cache official: gR1 improves by `0.710089 deg`, pRB worsens by `0.029006 cm`; historical raw-v5 row is context only |
| TC test after DIP FT | official PL | `6.768144` | `14.014337` | baseline |
| TC test after DIP FT | raw newpl_v5_dip_best | `6.780749` | `13.415189` | historical reference from `newpl_v5_official_protocol_20260607_tuned`, not same-cache fairness row |
| TC test after DIP FT | realtime smooth+residual | `6.638172` | `13.736756` | versus same-cache official: pRB improves by `0.129972 cm`, gR1 improves by `0.277580 deg`; historical raw-v5 row is context only |

Decision: diagnostic only, not selected. Feeding the removed acceleration residual back into the model is feasible and real-time, and it improves same-cache official PL gR1 on DIP/TC and same-cache TotalCapture pRB. The imported raw `newpl_v5_dip_best` rows are historical references from a different metric namespace, so they are not a fairness baseline for this run unless re-evaluated on the same cache. Do not promote this branch without a same-cache raw-v5 comparison and a loss/filter sweep.

## NewPL v6 gR1-only next-control smoothacc

Status: completed on 2026-06-13. This is a module-level PL experiment; no full-pipeline 11 metrics were run.

Contract:

```text
variant: newpl_v6_gR1nextonly_smoothacc
input: smoothed acceleration cache -> legacy init36 PL features
output: current pRB[15] + gR1[3], plus v6 auxiliary next-control outputs
training: AMASS pretrain -> DIP-IMU fine-tune; TotalCapture test is eval-only
DIP trans/root velocity: not used
```

Loss decision: keep ordinary current-frame `pRB/gR1`, current control, and temporal losses, but set all auxiliary next-control pRB weights to `0`. Only gravity receives auxiliary next/control supervision: `next_gR1=2.0`, `next_gt_control_gR1=0.5`, `next_gR1_vel=0.05`, `next_gR1_acc=0.002`, `last_control_gR1=0.5`, `tail4_control_gR1=0.35`.

Artifacts:

```text
root: /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full
log: /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/logs/run_full.log
summary: /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/summary.json
AMASS best gR1 ckpt: /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/amass_pretrain/best_current_gR1.pt
DIP best gR1 ckpt: /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/dip_finetune/best_current_gR1.pt
```

Key fair same-cache module metrics:

| Dataset/stage | Version | pRB L2 cm ↓ | gR1 angle deg ↓ | Readout |
|---|---|---:|---:|---|
| AMASS after AMASS | official PL | `4.030455` | `4.838765` | official still best gR1 |
| AMASS after AMASS | newpl_v6 gR1-only | `4.025294` | `5.228020` | slightly better pRB L2 than official, worse gR1 |
| DIP test after AMASS | official PL | `6.345701` | `12.902106` | baseline |
| DIP test after AMASS | newpl_v6 gR1-only | `6.431476` | `12.569474` | better gR1, worse pRB |
| DIP test after DIP FT | official PL | `6.345701` | `12.902106` | baseline |
| DIP test after DIP FT | newpl_v4 init36 | `6.349507` | `12.722391` | v4 pRB close to official |
| DIP test after DIP FT | newpl_v6 gR1-only | `6.392231` | `12.474146` | best gR1, pRB worse than official/v4 |
| TC test after DIP FT | official PL | `7.508986` | `13.170870` | baseline |
| TC test after DIP FT | newpl_v4 init36 | `7.119541` | `13.075061` | best pRB |
| TC test after DIP FT | newpl_v6 gR1-only | `7.430918` | `12.848388` | best gR1, pRB better than official but worse than v4 |

Decision: this run validates the intended tradeoff. Using next-control only for `gR1` improves gravity on DIP and TotalCapture without the larger pRB damage seen when pRB also receives auxiliary next-control pressure, but it still does not make pRB better than the strongest baseline. Keep it as a gR1-specialized candidate; do not connect to IK/full pipeline until either pRB is recovered or IK tests explicitly show that the gravity gain offsets the pRB loss.

### NewPL v6 gR1-only downstream swap validation

Status: completed on 2026-06-14.

Purpose: test the user's proposed question directly: if `newpl_v6_gR1nextonly_smoothacc` has the most accurate module-level `gR1`, does it help downstream when `pRB` is held fixed? This is not a new training run. It is an IK/full-pipeline ablation with official/baseline PL `pRB[15]` fixed and only `gR1[3]` replaced by the v6 checkpoint.

Contract:

```text
baseline: official/baseline PL pRB + official/baseline PL gR1
hybrid:   official/baseline PL pRB + newpl_v6_gR1nextonly_smoothacc gR1
checkpoint: /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/dip_finetune/best_current_gR1.pt
datasets/protocols: DIP raw official, TotalCapture raw official, DIP smoothacc, TotalCapture smoothacc
no GT oracle: no GT gR1 is injected
DIP trans/root velocity: not used or fabricated
```

Artifacts:

```text
summary: data/experiments/newpl_v6_gR1_only_swap_20260614/summary.json
raw DIP JSON: data/experiments/newpl_v6_gR1_only_swap_20260614/eval/dip_raw_official_pl_curve_eval.json
raw TC JSON: data/experiments/newpl_v6_gR1_only_swap_20260614/eval/tc_raw_official_pl_curve_eval.json
smooth DIP JSON: /tmp/globalpose_hybrid_prb_base_gr1_v6_20260613/full/dip_test_hybrid_baseline_pRB_newpl_gR1.json
smooth TC JSON: /tmp/globalpose_hybrid_prb_base_gr1_v6_20260613/full/totalcapture_test_hybrid_baseline_pRB_newpl_gR1.json
```

Downstream result, lower score is better:

| Protocol | Official score | Hybrid score | Delta hybrid-official | Main readout |
|---|---:|---:|---:|---|
| DIP raw official | `44.642049` | `45.089955` | `+0.447906` | worse |
| TotalCapture raw official | `44.477380` | `44.546482` | `+0.069101` | worse |
| DIP smoothacc | `44.233519` | `44.709970` | `+0.476451` | worse |
| TotalCapture smoothacc | `45.293466` | `45.451494` | `+0.158027` | worse |

Conclusion: better module-level `gR1` is real, but it is not downstream-useful by itself in this ablation. The hybrid consistently reduces `Joint Jitter` slightly, but worsens local angle and joint errors, and all four full-pipeline scores regress. Do not promote `newpl_v6_gR1nextonly_smoothacc` to IK/full-pipeline as-is.
