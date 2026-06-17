# Recent Replacement Versions

This document is organized by replacement-version iterations. Each version entry describes the structural change, input/output contract, loss change, training recipe, module-output-vs-GT metrics, full S4 11 metrics, artifacts, and final decision. Orchestrator logs and task-level traces belong to EXPERIMENT_LOG.md, not this summary.

本文档按替换模块的版本迭代组织。每个版本必须说明结构改动、输入输出、loss 改动、训练方式、模块输出是否更接近 GT、官方 S4 11 项误差、artifact 以及最终是否保留。orchestrator 日志和任务级记录不放在本文档中，应放入 EXPERIMENT_LOG.md。

## 0. Version-Line Organization

From 2026-06-12 onward, this ledger should be read and updated by version line. Do not append free-floating experiment notes at the bottom unless they are explicitly marked as archival evidence. Put new entries under the matching line below.

Metric namespace rule:

```text
Version name alone is not enough. Always report:
  version id + experiment root + cache/protocol + checkpoint.
Only same-cache / same-protocol evaluations are fair comparisons.
Cross-cache values may be kept as historical references, but must be labeled `historical reference`.
```

Known example:

```text
newpl_v5_dip_best in data/experiments/newpl_v5_official_protocol_20260607_tuned:
  DIP test pRB L2=6.445578 cm, gR1=12.552613 deg.

newpl_v5_dip_best in later offset/smooth/cache-family summaries:
  can appear with different values, e.g. DIP test gR1=14.801741 deg.

Do not mix these rows without re-evaluating all versions on the same cache.
```

Version-line index:

| Line | Versions / branches | Current status |
|---|---|---|
| `PL-s1 / historical processed` | `newpl_v1_processed_no_baseline`, `newpl_v2_gRdyn`, `newpl_v3_gtcontrol_rund`, `newpl_v4_init36` | `newpl_v4_init36` selected only as historical processed-input full-S4 best |
| `PL-s1 / official-route v5` | `newpl_v5_official_protocol`, `newpl_v5_loss_family_ablation` | no selected replacement; official-route pRB criterion not met |
| `PL-s1 / acceleration-input filters` | `newpl_v5_smoothacc`, `newpl_v5_butteracc`, `newpl_v5_realtime_smooth_residual` | diagnostic only; smooth/residual/Butter variants not promoted |
| `PL-s1 / predictive dynamics` | `newpl_v6_next_control`, `newpl_v6_next_control_smoothacc_gR1`, `newpl_v6_next_p_pdot_pddot_strong`, `newpl_root_v1` | diagnostic only; no IK1/full-pipeline promotion |
| `PL-s1 / offset-aware and acc-aux` | `newpl_offset_conditioned_*`, `newpl_offset_v6`, `newpl_v7_learned_offset_accaux`, `newpl_v7b_local_accaux` | diagnostic only; offset signal not converted to robust pRB/gR1 gain |
| `IK-s1` | `newik1_v1` through `newik1_v14` search/diagnostics | none selected; best remains behind PL-only init36 |
| `IK-s2 / NewPose` | `newpose_ctrl_v1`, `newpose_ctrl_v2` | rejected/diagnostic |
| `Diagnostic IMU-control modules` | `imu_neighbor_vel_ctrl_v1`, `imu_neighbor_pos_from_vel_ctrl_v1`, `imu_joint_euler_qdot_vel_ctrl_v1` | diagnostic only; neighbor position v1 and joint Euler/qdot/velocity v1 are rejected for IK/NewIK1 input |
| `AccCurve / acceleration residual` | `acc_curve_v1_20260617`, `acc_curve_v1_totalcapture_eval_20260618`, `acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617`, `acc_curve_pl_input_eval_20260617` | standalone acceleration-level modules; v1 improves DIP diff-pos target but fails TotalCapture diff-pos target; v2 strict GTFK remains diagnostic; do not connect as-is |
| `IMU offset / r_JS` | `footlock_transpose_v1`, retired solver/net/hybrid routes | active pseudo-`r_JS` route is `footlock_transpose_v1` only |
| `Baselines / official fine-tune` | official GPNet official/processed, TotalCapture fine-tune diagnostic | references only |

## 1. Current Best Summary

Policy update on 2026-06-07: the project mainline is no longer to keep scaling the best TotalCapture/processed-input artifact directly. The new mainline is to first reproduce the official baseline route, then evaluate replacement modules under the same route:

```text
AMASS pretrain -> DIP-IMU train fine-tune -> DIP-IMU test + TotalCapture test
```

`newpl_v4_init36` remains the best historical processed-input full S4 artifact, but future PL/IK/VR promotion must be judged against official/baseline modules under this official-like generalization protocol. TotalCapture-specialized fine-tunes must be labeled as such and cannot be used alone to claim a better general-purpose baseline.

Control-target policy update on 2026-06-08: new control-point GT synthesis is now `derivative_aware_v1` for all replacement modules that call `fit_uniform_cubic_spline_controls`. It fits position/state plus finite-difference first and second derivatives with weights `position=1.0`, `velocity=0.03`, `acceleration=0.0003`, `ridge=1e-6`, `dt=1/60`. This affects new PL, NewPL-root, IK1 control, bone aux, and NewPose control targets. Existing stored control-tail caches are historical and must be rebuilt before new same-protocol comparisons.

Canonical GT-control cache update on 2026-06-08: reusable dataset-level control targets now live under `data/dataset_work/GTControlCache/` and are generated by `scripts/build_gt_control_cache.py` / `scripts/run_build_gt_control_caches_20260608.sh`. These caches include `pose_rot6d_control`, `joint_angle_euler_control`, `joint_pos_R_control`, `imu_RMB_6d_control`, and `pl_pRB_gR1_control` for AMASS, DIP-IMU, and TotalCapture splits. AMASS/TotalCapture include `root_trans_W/root_vel_W_fd`; DIP-IMU root velocity GT is marked unavailable and is not synthesized from unreliable translation.

Real-data `r_JS` policy update on 2026-06-09: only `footlock_transpose_v1` remains active for DIP/TotalCapture sequence-level IMU position offsets. This route uses TransPose winner-foot contact windows, does not trust DIP `trans`, infers root motion by foot lock, and solves `r_JS` from lever-arm equations. Smoothed acceleration is used only inside this same fit (`smooth_window=9`, `derivative_mode=centered`); TransPose contact still uses raw official `aM/RMB`. Old `zero/random/solver_v1/net_v2/hybrid_v3/full_diagnostic/rawlike_se3/generic smoothed LS` routes and their generated artifacts are retired/deleted, not candidates for future NewPL/NewIK1 comparisons.

| Rank | Version | Replaced Module | Input Mode | Main Change | S4 Score ↓ | Selected? | Reason |
|---:|---|---|---|---|---:|---|---|
| 1 | newpl_v4_init36 | PL-s1 | processed IMU | 36D stream init | `38.625657` | historical best | Best processed-input full S4 artifact; not sufficient alone for the new official-route mainline. |
| 2 | newik1_v6_official_input_init36_cascade | IK-s1 | processed + NewPL init36 | v4 official-input cascade fine-tune | `38.649137` | no | IK1 output is closer to GT, but S4 is still worse than NewPL init36. |
| 3 | newik1_v9_adaptive_loss_search | IK-s1 | processed + NewPL init36 | v8 B4 continuation, 8-way adaptive loss micro-sweep | `38.693845` | no | Slightly improves v8 S4, but full pRJ and leaf-pRJ are farther from GT than official IK1 baseline. |
| 4 | newik1_v8_parallel_adaptive_loss_search | IK-s1 | processed + NewPL init36 | v7 last-control small loss-ratio sweep | `38.694152` | no | Slightly improves v7 S4, but worse than NewPL init36 and module-GT state is worse than official IK1 baseline. |
| 5 | newpl_v3_gtcontrol_rund | PL-s1 | processed IMU | GT control Run D | `38.694846` | superseded | Previous PL best. |
| 6 | newik1_v4_official_input | IK-s1 | processed + Run D PL | official-shape IK1 | `38.705231` | no | Does not beat PL-only. |
| 7 | official_processed | none | processed IMU | RMB-only input correction | `38.753660` | baseline | Processed baseline. |
| 8 | newik1_v5_last_pl_control | IK-s1 | processed + NewPL init36 | last PL control input | `38.843577` | no | Worse than NewPL init36. |
| 9 | official_official | none | official IMU | official GPNet | `42.522402` | baseline | Official reference. |

## 2. Baseline References

| Baseline | Input | Replaced Module | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official GPNet + official IMU | official `aM/wM/RMB` | none | `42.522402` | `10.466050` | `10.133907` | `4.817390` | `5.537234` | `10.716775` | `10.255115` | `4.638654` | `5.289347` | `0.297783` | `0.495126` |
| Official GPNet + processed IMU | `l4_aM==aM`, `l4_wM==wM`, `l4_RMB!=RMB` | none | `38.753660` | `10.250858` | `8.825690` | `4.561564` | `5.181901` | `10.316410` | `8.463454` | `4.361697` | `4.872381` | `0.294599` | `0.492203` |

## 3. Official Baseline Project Structure

Official `GPNet` full-pipeline structure used as the reference for all replacement versions:

```text
aM / wM / RMB
-> GPNet.forward_frame feature construction
-> PL-s1 (`plnet`)
-> IK-s1 (`iknet.net1`)
-> IK-s2 (`iknet.net2`)
-> pose [24, 3, 3]
-> VR-s1 (`vrnet`)
-> velocity/contact fusion
-> carticulate physics backend
-> final pose / translation
```

Official sparse-IMU input contract:

| Item | Shape | Meaning |
|---|---:|---|
| `aM` | `[6, 3]` | model-frame acceleration |
| `wM` | `[6, 3]` | model-frame angular velocity |
| `RMB` | `[6, 3, 3]` | body-to-model IMU orientation |

Official stage contracts:

| Stage | Input | Output | Replacement rule |
|---|---|---|---|
| PL-s1 (`plnet`) | `aRB[18] + wRB[18] + RRB[45] + gR0[3] = 84D` | `pRB[15] + gR1[3] = 18D` | PL replacements must preserve the 18D output for IK1. |
| IK-s1 (`iknet.net1`) | `RRB_after_pl[45] + gR1[3] + pRB[15] = 63D` | `pRJ[69] + gR2[3] = 72D` | IK1 replacements must preserve the 72D output for IK2/VR. |
| IK-s2 (`iknet.net2`) | `RRB_after_ik1[45] + gR2[3] + pRJ[69] = 117D` | `15 reduced joints x 6D = 90D` | IK2 replacements must be confirmed as `iknet.net2`, not IK1 artifacts. |
| VR-s1 (`vrnet`) | `RRJ[135] + pRJ[69] + aRB[18] + wRB[18] + gR2[3] = 243D` | `9D` root velocity/contact representation | VR replacements must preserve physics backend compatibility. |

PL feature construction in `GPNet.forward_frame`:

```text
aRB = aM @ RMB[5]
wRB = wM @ RMB[5]
RRB = RMB[5]^T @ RMB[:5]
gR0 = -RMB[5, 1]
```

Processed IMU convention used in current replacement comparisons:

```text
processed IMU = orientation-only / RMB-only correction
l4_aM == aM
l4_wM == wM
l4_RMB != RMB
```

Interpretation: processed-input gains come from corrected orientation/RMB and induced root-relative feature changes, not from changed stored acceleration or gyro fields.

## 4. Version Timeline

| Version | Module | Main Change | Input Change | Output Change | Loss Change | Training Change | Module GT Improved? | S4 Improved? | Decision |
|---|---|---|---|---|---|---|---|---|---|
| newpl_v1_processed_no_baseline | PL-s1 | PLCurve processed no-baseline | official PL 84D from processed IMU | PL 18D unchanged | disable baseline/distill | TC finetune | not vs previous | no vs processed baseline | not selected |
| newpl_v2_gRdyn | PL-s1 | add gR dynamics | unchanged | unchanged | add gR1_dot/gR1_ddot | TC finetune | yes small local loss | not found | intermediate |
| newpl_v3_gtcontrol_rund | PL-s1 | GT control Run D | unchanged | unchanged | add gt_control pRB/gR1 | continue 10 epochs | local total worse, S4 better | yes vs processed | superseded |
| newpl_v4_init36 | PL-s1 | 36D stream init | stream init uses offset_r+pRL+gR0 | PL output unchanged | same RunD-style | 60 epochs from Run D | yes vs Run D local and S4 | yes | selected |
| newpl_v5_official_protocol | PL-s1 | official-like AMASS -> DIP route | official input; init36 cache | PL output unchanged | RunD-style control physical selection | AMASS 80 epochs -> DIP 40 epochs | mixed; DIP gR1 improves, pRB does not beat baselines | not measured | not selected |
| newpl_v5_loss_family_ablation | PL-s1 | q/control/qdot/qddot loss-family ablation | official input; init36 cache | PL output unchanged | 8 variants over q, control, qdot, qddot | AMASS -> DIP per variant | weak; qddot helps DIP pRB, control not robust | not measured | not selected |
| newpl_v5_smoothacc | PL-s1 | smooth official acceleration input | replace raw `aM` with centered smoothed `aM`, keep `wM/RMB` | PL output unchanged | same v5 loss and control_physical selection | AMASS -> DIP with cached smooth inputs | mixed; DIP pRB/gR1 improve vs raw input, TC pRB regresses | not measured | not selected |
| newpl_v5_butteracc | PL-s1 | realtime causal Butterworth acceleration gate | replace raw `aM` with causal Butterworth filtered `aM`, keep `wM/RMB` | PL output unchanged | same v5 loss and control_physical selection | input-only sweep over fc8/fc10/fc12, then forced fc12 AMASS -> DIP | no; gate fails and forced training does not recover pRB | not measured | not selected |
| newpl_v6_next_control | PL-s1 | one-step predictive next control with preview tail4 update | official 84D input + init36 | current PL 18D unchanged; adds aux next PL/dynamics | current PL/control plus next state/control/velocity/acceleration and last/tail4 control losses | full AMASS 80 -> DIP 40; TC eval-only | no; AMASS pRB/acc improves, but DIP/TC pRB does not beat baselines | not measured | not selected |
| newpl_v6_next_control_smoothacc_gR1 | PL-s1 | combine centered smooth acceleration with v6 next-control and gR1 checkpoint selection | smooth `aM`; raw `wM/RMB`; init36 | current PL 18D unchanged; aux next PL/dynamics unchanged | same v6 loss plus saved `best_current_gR1`, `best_next_gR1`, `best_gravity_control` | AMASS 80 -> DIP 40; TC eval-only; final DIP/TC eval fast512 | mixed; best TC/DIP gR1 among smooth baselines, but pRB is worse than v4/raw-v5 | not measured | not selected |
| newpl_v6_next_p_pdot_pddot_strong | PL-s1 | supervise decoded next `p/pd/pdd` trajectory outputs directly | smooth `aM`; raw `wM/RMB`; init36; reuses compatible next-control cache or builds under run root | current PL 18D unchanged; aux next PL/dynamics unchanged | only normalized decoded next pRB `p/pd/pdd = 1/1/1`; all old current/control/gR1/prior terms zero in preset | AMASS 80 -> DIP 40; batch 32; current/next p-pdot-pddot eval added; TC eval-only | no; next p/pd/pdd target converges but current-frame p/pddot fails same-cache non-regression | not measured | diagnostic only; not selected |
| imu_neighbor_pos_from_vel_ctrl_v1 | diagnostic neighbor-control | predict root-relative positions from official IMU plus frozen neighbor velocity-control features | 189D: `aM/wM/RMB_6d/r_JS` + velocity control/vel/acc | `neighbor_pos_R_control[33]`, decoded `pos_R/vel_R/acc_R` | position control, decoded position, optional root-relative vel/acc, segment length, smooth/jerk/prior | AMASS 80 -> TC 60 and AMASS 80 -> DIP 30; module eval only | no; pos_R L2 about `48 cm` vs pose_prephysics baseline `2.34-5.57 cm` | not run | rejected for IK/NewIK1 |
| imu_joint_euler_qdot_vel_ctrl_v1 | diagnostic IMU joint-control | PL-like init network predicts IMU-mapped joint Euler/qdot/velocity controls | current code: `aM[18]+wM[18]+R_rootIMU_sensorIMU_flat[54]=90D`; historical run used world `RMB_flat[54]`; init `q[0]+qdot[0]+vel[0]=54D` | `q_RJ_euler_control[18]+qdot_RJ_euler_control[18]+vel_RJ_control[18]` | q control/q/qdot/qdot-control/qddot/velocity/acceleration/consistency/smooth/jerk/prior; 4 loss variants | historical world-RMB run: 4 variants; root-RMB rerun: `D_all_balanced` only with shared precompute/last.pt disabled for quota | no; root-RMB D is not better than world-RMB D and remains far worse than baseline | not run | rejected for IK/NewIK1 |
| acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617 | AccCurve / acceleration residual | PL-style curve residual over absolute sensor-site acceleration with strict GTFK target | 108D: `aM_raw[18]+aM_smooth[18]+residual[18]+wM[18]+RMB_6d[36]`; all in model/world cache frame; `RMB_6d=rotation[..., :, :2].transpose(-1,-2).reshape(...,6)` | `pred_aM_curve[18]` absolute acceleration; base `aM_smooth[18]` | valid-mask MSE to `aFK_gtfk_smooth[18] = smooth(GTFKacc(q,qdot,qddot,rJS))`; selection by `val_pred_base_ratio` | AMASS 30 -> DIP 20; window 240, stride 120, batch 64; AMASS-train-only feature z-score; module eval only | yes; DIP val ratio `0.7575`, DIP test ratio `0.7783` vs same-cache base | not run | diagnostic only; strict GTFK acceleration target works vs base, but does not meet ratio<0.7/corr>0.9 effectiveness gate |
| acc_curve_v1_20260617 | AccCurve / acceleration residual | PL-style curve residual over absolute sensor-site acceleration | 108D: `aM_raw[18]+aM_smooth[18]+residual[18]+wM[18]+RMB_6d[36]` | `pred_aM_curve[18]` absolute acceleration; base `aM_smooth[18]` | valid-mask MSE to `aFK_smooth[18]`; selection by `val_pred_base_ratio` | AMASS 30 -> DIP 20; window 240, stride 120, batch 64; module eval only | mixed; DIP test ratio `0.6220`, but TotalCapture test ratio `2.3940` after `acc_curve_v1_totalcapture_eval_20260618` | not run | diagnostic only; do not use v1 acceleration directly for cross-dataset NewPL retrain |
| acc_curve_v1_totalcapture_eval_20260618 | AccCurve / acceleration residual | evaluate v1 checkpoint on TotalCapture v1 diff-pos acceleration target | same v1 108D input from TC official offset cache | `pred_aM_curve[18]` vs `aFK_smooth[18]` | no training; target is `smooth(diff_acc(p_WS))`, not strict GTFK | TC test cache build + eval only; checkpoint `acc_curve_v1_20260617/dip_finetune/best_loss.pt` | no; TC pred L2/RMSE `2.091960/1.539445` vs base `0.873843/0.693060`, ratio `2.393977`, corr `0.866428` | not run | rejects direct v1 cross-dataset acceleration replacement before NewPL retrain |
| acc_curve_pl_input_eval_20260617 | AccCurve / PL input evaluation | feed raw/smooth/v1/v2 acceleration into the same frozen official PL input | legacy PL 84D with only `aRB[18]` replaced; `wRB[18]+RRB[45]+gR0[3]` identical; AccCurve M-frame outputs converted by `aRB=acc_M @ RMB_root` | official PL `pRB[15]+gR1[3]` from `data/weights.pt` `GPNet.plnet` | no training; same `pl_target_from_pose(pose_gt)` target for all variants | DIP test evaluation only on `newpl_v5_official_protocol_20260607` cache | smooth_acc improves pRB/gR1; AccCurve v1/v2 improve gR1 only but regress pRB | module-level PL pRB/gR1 only, not full S4 | do not connect AccCurve v1/v2 to PL as-is; acceleration-level gains did not transfer to simultaneous PL output gain |
| newik1_v1_control_tail | IK-s1 | control-tail IK1 | 120D control-tail feature | 72D pRJ+gR2 unchanged | baseline IK1 control losses | PL streaming TC finetune | not vs previous | no | not selected |
| newik1_v2_bonelength | IK-s1 | bone length | unchanged | unchanged | add bone_length=0.5 | continue from v1 | yes local loss | no | not selected |
| newik1_v3_strong_pRJ_control | IK-s1 | strong pRJ/control | unchanged | unchanged | pRJ=2.0, control_pRJ=0.3 | continue from v2 | no local total | better than v1/v2 but not PL-only | not selected |
| newik1_v4_official_input | IK-s1 | official-shape IK1 | 63D official-shape input | 72D unchanged | pRJ=2, distill pRJ=0.2 | PL streaming finetune | local loss lower but output pRJ/gR2 larger | better than processed baseline, worse than PL-only | not selected |
| newik1_v5_last_pl_control | IK-s1 | last PL control input | 63D last-control feature | 72D unchanged | same control loss family | 3-stage GT/AMASS/TC | mixed; pRJ better, gR2 worse | worse than NewPL init36 | not selected |
| newik1_v6_official_input_init36_cascade | IK-s1 | v4 official-input cascade over NewPL init36 | unchanged official 63D | 72D unchanged | same official-input loss | v4 -> GT AMASS refresh -> PL AMASS adapt -> PL TC finetune | yes overall; pRJ L2 and gR2 angle improve, pRJ L1 slightly worse in B/C | stage A close but still worse; B/C much worse | not selected |
| newik1_v8_parallel_adaptive_loss_search | IK-s1 | v7 last-control loss-ratio sweep | unchanged last-control 63D | 72D unchanged | varied pRJ/gR2 and dynamic weights, removed bone/prior | 8 parallel TC micro-finetunes from v7 | no; best state L2 delta is positive vs official IK1 baseline | slight vs v7, no vs NewPL init36 | not selected |
| newik1_v9_adaptive_loss_search | IK-s1 | v8 B4 continuation with second adaptive loss sweep | unchanged last-control 63D | 72D unchanged | pRJ/gR2/control/dynamics micro-sweep; best disables control dynamics | 8 parallel 5-epoch TC micro-finetunes from v8 B4 last | no; full pRJ and leaf-pRJ deltas are positive vs official IK1 baseline | slight vs v8, no vs NewPL init36 | not selected |
| newpose_ctrl_v1 | IK-s2 / pose-control slot | direct pose-control state from official IMU + NewPL control features | 174D official IMU/NewPL feature; offset_r init-only | `RRJ_control[90]+gR_pose_control[3]=93D` | control RRJ/gR, decoded state, FK, temporal losses | AMASS pretrain -> DIP fine-tune; DIP/TC eval | no; FK joint L2 is `43.8-45.4 cm` vs baseline `4.6-5.0 cm` | no; full score `413-432` vs baseline `43-45` | rejected |

## 4.1 Diagnostic IMU-Control Modules

## Version: acc_curve_pl_input_eval_20260617

### 1. Purpose

Evaluate whether AccCurve v1/v2 predicted accelerations improve the frozen official baseline PL module when used only as the PL acceleration input. This is evaluation-only: no PL retraining and no PL network change.

### 2. Contract

| Item | Value |
|---|---|
| Experiment root | `data/experiments/acc_curve_pl_input_eval_20260617` |
| Evaluator | `scripts/eval_pl_with_acc_curve_input_20260617.py` |
| Frozen PL checkpoint | `data/weights.pt`, official `GPNet.plnet` weights only |
| DIP test cache/protocol | `data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json` |
| PL feature | `aRB[18] + wRB[18] + RRB[45] + gR0[3] = 84D` |
| Replacement rule | replace only `aRB[18]`; keep the other 66D, target, mask, split, and checkpoint fixed |
| Frame conversion | AccCurve output is model/world-frame M; PL input uses root-frame `aRB = acc_M @ RMB_root` |
| Target | same `pl_target_from_pose(pose_gt)` pRB/gR1 target for every variant |

Validation:

| Check | Value |
|---|---:|
| official vectorized 84D feature vs `pl_input_feature` max abs diff | `7.6293945e-06` |
| non-acc 66D block max abs diff across variants | `0` |
| AccCurve v1/v2 pred shape | `[T,6,3]` for every evaluated sequence |
| DIP test used for train/norm/checkpoint selection | no |

### 3. DIP Test Results

| Variant | Acc source | Target used by AccCurve | PL pRB L2 cm | PL pRB RMSE cm | PL gR1 deg | valid frames |
|---|---|---|---:|---:|---:|---:|
| official_raw_acc | raw aM | none | `6.529110` | `4.638030` | `15.267153` | `57994` |
| smooth_acc | smooth(aM) | none | `6.462386` | `4.589704` | `15.216247` | `57994` |
| acc_curve_v1_pred | AccCurve v1 pred | smooth(diff_acc(p_WS)) | `6.967961` | `4.866400` | `15.036875` | `57994` |
| acc_curve_v2_gtfk_pred | AccCurve v2 pred | smooth(GTFKacc(q,qdot,qddot,rJS)) | `8.347050` | `5.958994` | `15.229429` | `57994` |

Decision:

```text
smooth_acc improves both PL pRB and gR1 versus official_raw_acc.
acc_curve_v1_pred improves gR1 but regresses pRB by +0.438851 cm.
acc_curve_v2_gtfk_pred slightly improves gR1 but regresses pRB by +1.817940 cm.
Therefore AccCurve acceleration-level gains did not transfer into a simultaneous
PL pRB+gR1 module-output gain. Do not connect AccCurve v1/v2 into PL as-is.
This is not a full-pipeline S4/motion-quality result.
```

## Version: acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617

### 1. Purpose

Train the strict standalone AccCurve residual module on IMU sensor-site acceleration targets computed from `GTFKacc(q,qdot,qddot,rJS)`, not from sensor-site position finite differences. This is an acceleration-level diagnostic and does not replace PL-s1, IK-s1, IK-s2, or VR-s1.

### 2. Input / Output Contract

| Item | Value |
|---|---|
| Module | `PLStyleAccCurveModule` |
| Input | `aM_raw[18] + aM_smooth[18] + (aM_raw-aM_smooth)[18] + wM[18] + RMB_6d[36] = 108D` |
| Input frame | model/world frame from the cache for `aM_raw`, `aM_smooth`, `wM`, and `RMB`; no root-frame transform |
| `RMB_6d` | `rotation[..., :, :2].transpose(-1, -2).reshape(..., 6)`, matching PL |
| Base | `aM_smooth[18]` |
| Output | `pred_aM_curve[18]`, absolute sensor-site acceleration |
| Target | `aFK_gtfk_smooth[18] = centered_smooth(GTFKacc(q,qdot,qddot,rJS))` |
| Units | output and target stay in `m/s^2`; only input features are z-scored |
| Normalization | feature z-score fitted from AMASS train split only |
| Valid mask | strict GTFK finite frames with smooth trim excluded |

The v1 target `aFK_smooth` was built from a position finite-difference cache (`smooth(diff_acc(p_WS))` style). v2 uses the explicit `aFK_gtfk_smooth` field and refuses target keys or target metadata that are not strict GTFK.

### 3. Architecture

The network keeps the v1 PL-style AccCurve residual structure: `Linear(feature+base) -> GRUCell -> zero-initialized residual control/tail heads -> UniformCubicBSpline`, `state_dim=18`, `tail_update=4`, `dt=1/60`. The smoke run verified `zero_init_max_abs_pred_minus_base=0.0`.

### 4. Training / Evaluation

| Stage | Split | Train seq | Val seq | Train windows | Val windows | Best epoch | Best selection |
|---|---|---:|---:|---:|---:|---:|---:|
| AMASS pretrain | AMASS hash 95/5 | `1231` | `67` | `8231` | `407` | `29` | `0.8150924359` |
| DIP finetune | DIP train -> DIP val | `36` | `6` | `1887` | `253` | `19` | `0.7294550687` |

Checkpoint selection uses:

```text
val_pred_base_ratio = mean||pred_aM_curve-aFK_gtfk_smooth|| / mean||aM_smooth-aFK_gtfk_smooth||
```

### 5. Module Metrics

| Split | pred L2 | base L2 | pred/base ratio | pred RMSE | base RMSE | corr | cosine | mag MAE | residual std | residual p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP val | `1.990773` | `3.102919` | `0.757471` | `1.689920` | `2.535429` | `0.821207` | `0.488962` | `1.060400` | `1.950429` | `6.949836` |
| DIP test | `2.997944` | `3.958857` | `0.778348` | `2.588015` | `3.341077` | `0.792049` | `0.493421` | `1.639596` | `2.357390` | `8.619020` |

The model beats the `aM_smooth` base (`ratio < 1`) on DIP val/test, but does not meet the stronger effectiveness gate (`ratio < 0.7` and `corr > 0.9`).

### 6. Artifacts

| Item | Path |
|---|---|
| Module | `acc_curve.py` |
| Train/eval | `acc_curve_train.py` |
| Strict cache builder | `scripts/build_acc_curve_gtfk_cache.py` |
| Runner | `scripts/run_acc_curve_v2_gtfk_20260617.sh` |
| Cache root | `code/outputs/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617` |
| Experiment root | `data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617` |
| Final checkpoint | `data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/dip_finetune/best_loss.pt` |
| Summary | `data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/train_result.json` |
| Eval JSONs | `data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/eval/dip_val_eval.json`, `data/experiments/acc_curve_v2_gtfk_q_qdot_qddot_rjs_20260617/eval/dip_test_eval.json` |

### 7. Decision

Keep as a standalone acceleration-level diagnostic. It supports continued exploration as an acceleration alignment auxiliary because it improves strict GTFK target regression over the smoothed IMU base, but it is not connected to NewPL/IK/full pipeline and should not be used to claim motion-quality improvement.

## Version: acc_curve_v1_20260617

### 1. Purpose

Train a standalone PL-style acceleration curve module that predicts absolute sensor-site acceleration in the model/world frame. This is an acceleration-level diagnostic and does not replace PL-s1, IK-s1, IK-s2, or VR-s1.

### 2. Input / Output Contract

| Item | Value |
|---|---|
| Module | `PLStyleAccCurveModule` |
| Input | `aM_raw[18] + aM_smooth[18] + (aM_raw-aM_smooth)[18] + wM[18] + RMB_6d[36] = 108D` |
| Base | `aM_smooth[18]` |
| Output | `pred_aM_curve[18]`, absolute sensor-site acceleration |
| Target | `aFK_smooth[18]` from FK sensor-site acceleration |
| Frame | model/world frame `M` |
| Units | `m/s^2` |
| Valid mask | centered-difference FK acceleration endpoints plus smooth trim excluded |

### 3. Architecture

The module follows the PLCurve streaming style: input encoder, `GRUCell`, zero-initialized new-control and tail-delta heads, and `UniformCubicBSpline` decoding with `tail_update=4`, `dt=1/60`, and `state_dim=18`. Initial output is exactly the smoothed acceleration base.

### 4. Training / Evaluation

| Stage | Split | Train seq | Val seq | Train windows | Val windows | Best epoch | Best selection |
|---|---|---:|---:|---:|---:|---:|---:|
| AMASS pretrain | AMASS hash 95/5 | `1231` | `67` | `8231` | `407` | `24` | `0.9526165639` |
| DIP finetune | DIP train -> DIP val | `36` | `6` | `1887` | `253` | `20` | `0.5814286023` |

Training uses fixed windows (`window=240`, `stride=120`, `batch_size=64`), train-split-only feature z-score normalization, and same-cache module evaluation on DIP val/test. Checkpoint selection uses:

```text
val_pred_base_ratio = mean||pred_aM_curve-aFK_smooth|| / mean||aM_smooth-aFK_smooth||
```

### 5. Module Metrics

| Split | pred L2 | base L2 | pred/base ratio | pred RMSE | base RMSE | corr | residual std | residual p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP val | `0.846367` | `1.871923` | `0.655075` | `0.628144` | `1.324018` | `0.957387` | `1.204956` | `3.743450` |
| DIP test | `1.202067` | `2.368697` | `0.622049` | `0.930242` | `1.733464` | `0.940837` | `1.450683` | `4.660124` |

TotalCapture generalization check on 2026-06-18:

| Dataset | Target | Acc source | L2 error | RMSE | pred/base ratio | corr | valid frames |
|---|---|---|---:|---:|---:|---:|---:|
| TotalCapture test | `smooth(diff_acc(p_WS))` | `aM_smooth` | `0.873843` | `0.693060` | `1.000000` | `0.974734` | `16084` |
| TotalCapture test | `smooth(diff_acc(p_WS))` | AccCurve v1 pred | `2.091960` | `1.539445` | `2.393977` | `0.866428` | `16084` |

Interpretation: DIP test v1 ratio was `0.622049`; TotalCapture test ratio is `2.393977`, a gap of `+1.771928`. v1 improves the DIP diff-pos target but does not generalize to TotalCapture, where `aM_smooth` is already much closer to `aFK_smooth` than the prediction.

### 6. Artifacts

| Item | Path |
|---|---|
| Module | `acc_curve.py` |
| Train/eval | `acc_curve_train.py` |
| Cache builder | `scripts/build_acc_curve_cache.py` |
| Runner | `scripts/run_acc_curve_v1_20260617.sh` |
| Cache root | `code/outputs/smooth_acc_cache_amass_dip_20260617` |
| Experiment root | `data/experiments/acc_curve_v1_20260617` |
| Final checkpoint | `data/experiments/acc_curve_v1_20260617/dip_finetune/best_loss.pt` |
| Summary | `data/experiments/acc_curve_v1_20260617/train_result.json` |
| Eval JSONs | `data/experiments/acc_curve_v1_20260617/eval/dip_val_eval.json`, `data/experiments/acc_curve_v1_20260617/eval/dip_test_eval.json` |
| TC cache root | `code/outputs/smooth_acc_cache_totalcapture_v1_20260618/tc_test` |
| TC eval root | `data/experiments/acc_curve_v1_totalcapture_eval_20260618` |
| TC eval script | `scripts/eval_acc_curve_v1_totalcapture_20260618.py` |

### 7. Decision

This is a successful DIP-only standalone acceleration-regression diagnostic: it beats the `aM_smooth` base on DIP val/test under the same cache and metric. The 2026-06-18 TotalCapture check fails the cross-dataset gate (`pred/base ratio=2.393977 >= 1`). Do not use v1 acceleration directly as a cross-dataset NewPL retrain input unless the acceleration module is revised or a stronger same-cache gate is added. It is not a PL/NewPL replacement and no full-pipeline 11 metrics were run.

## Version: imu_joint_euler_qdot_vel_ctrl_v1

### 1. Purpose

Test the user's proposed PL-like diagnostic module that predicts the IMU-mapped joints' Euler-angle controls, Euler-rate controls, and root-relative joint-velocity controls. This is diagnostic only: it does not replace PL/IK1, does not connect to the full pipeline, and does not run official S4 11 metrics.

Post-run update: the original completed run used world/model-frame `RMB_flat[54]`. The code has now been changed so the rotation input is root-frame `R_rootIMU_sensorIMU = RMB[root_imu=5]^T @ RMB[sensor]`; root-RMB `D_all_balanced` was rerun under `data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613`.

### 2. Main Change

| Change Type | Previous diagnostics | This Version | Motivation |
|---|---|---|---|
| Input | neighbor velocity/position diagnostics used `r_JS` and/or velocity-control features | `aM/wM` plus root-frame IMU rotations | Remove global heading/root orientation burden from the rotation input while testing a new control target |
| Init | none or task-specific state | PL-style learned initial hidden state from `q_RJ_euler[0]+qdot_RJ[0]+vel_RJ[0] = 54D` | Keep the requested init path so sequence state is not started from zeros |
| Output | neighbor node velocity/position controls | `q_RJ_euler_control[18]+qdot_RJ_euler_control[18]+vel_RJ_control[18]` | Joint q/qdot/velocity control diagnostic |
| Loss search | single recipe | four parallel variants A/B/C/D | Test whether q-control, qdot/qddot, velocity/acceleration, or balanced losses produce better module outputs |

### 3. Input / Output Contract

| Item | Shape | Meaning |
|---|---:|---|
| Input feature | 90D | `aM[18]+wM[18]+R_rootIMU_sensorIMU_flat[54]`; only the RMB block is converted to root IMU frame |
| Init state | 54D | `q_RJ_euler[0]+qdot_RJ_euler[0]+vel_RJ[0]` |
| Joints | 6 x 3D | `[18, 19, 4, 5, 15, 0]` |
| Output head 1 | 18D | `q_RJ_euler_control`, XYZ Euler control for `R_RJ` |
| Output head 2 | 18D | `qdot_RJ_euler_control` |
| Output head 3 | 18D | `vel_RJ_control` |
| Decoded q branch | 54D per frame | `q_RJ_euler`, `qdot_from_q`, `qddot_from_q` |
| Decoded qdot branch | 54D per frame | `qdot_RJ`, `qddot_from_qdot`, qdot jerk |
| Decoded velocity branch | 54D per frame | `vel_RJ`, `acc_RJ`, velocity jerk |

Frame contract:

```text
R_RJ = R_WR^T R_WJ
p_RJ = (p_WJ - p_WR) @ R_WR
q_RJ_euler = unwrapped XYZ Euler of R_RJ
R_rootIMU_sensorIMU = RMB[root_imu=5]^T @ RMB[sensor]
vel_RJ / acc_RJ = finite differences of p_RJ in root frame R
DIP policy = no DIP trans, no DIP world/root velocity GT, no fabricated translation finite difference
```

### 4. Loss Design

All variants share the same terms and differ only by weights:

```text
q_control, q, qdot_from_q, qdot_decoded, qdot_control,
qddot_from_q, qddot_from_qdot,
vel_control, vel, acc,
consistency, smooth, jerk, control_prior
```

Best checkpoint selection uses the direct physical module outputs:

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

| Variant | Main emphasis |
|---|---|
| `A_qctrl_main` | stronger q-control and decoded-q supervision |
| `B_qdot_qddot_strong` | stronger qdot, qdot-control, and qddot supervision |
| `C_vel_acc_strong` | stronger velocity-control, decoded velocity, and acceleration supervision |
| `D_all_balanced` | balanced q/qdot/qddot/velocity/acceleration with stronger consistency |

### 5. Training Recipe

| Stage | Data | Init | Batch | Epochs | Notes |
|---|---|---|---:|---:|---|
| AMASS pretrain | `prephysics_pose_velocity_amass_k2_paired_offset_overlay` | scratch | `1024` | `80` | shared compact precompute; four variants trained across available GPUs |
| TotalCapture fine-tune | TC train/val official offset_r caches | AMASS best of each variant | `512` | `60` | eval on TC test after AMASS and after TC fine-tune |
| DIP fine-tune | DIP train/val official offset_r caches | AMASS best of each variant | `512` | `30` | no DIP trans/world/root velocity GT; eval on DIP test and TC test |

Run command:

```bash
cd /home/lingfeng/projects/GlobalposeMy/GlobalPose
/home/lingfeng/bin/longrun -- bash scripts/run_imu_joint_euler_qdot_vel_ctrl_v1_20260613.sh
```

Artifacts:

```text
experiment root: data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613
run log: data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/logs/run.log
summary: data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_20260613/summary.json
per-variant checkpoints:
  <variant>/amass_pretrain/best_loss.pt
  <variant>/totalcapture_finetune/best_loss.pt
  <variant>/dip_finetune/best_loss.pt
per-variant eval JSONs:
  <variant>/eval/eval_amass_after_amass_best.json
  <variant>/eval/eval_totalcapture_test_after_amass_best.json
  <variant>/eval/eval_totalcapture_test_after_tc_finetune_best.json
  <variant>/eval/eval_dip_test_after_amass_best.json
  <variant>/eval/eval_dip_test_after_dip_finetune_best.json
  <variant>/eval/eval_totalcapture_test_after_dip_finetune_best.json
```

### 6. Module GT Metrics

Root-RMB rerun, `D_all_balanced` only:

| Dataset / stage | Root-RMB rot deg ↓ | Root-RMB vel L2 cm/s ↓ | Old world-RMB D rot deg ↓ | Old world-RMB D vel L2 cm/s ↓ | Baseline rot deg ↓ | Baseline vel L2 cm/s ↓ |
|---|---:|---:|---:|---:|---:|---:|
| AMASS after AMASS | `30.8153` | `29.4072` | `30.5556` | `29.4059` | `4.0610` | `13.1790` |
| TotalCapture after AMASS | `29.7156` | `32.7157` | `29.3217` | `32.7078` | `12.3839` | `19.8320` |
| TotalCapture after TC fine-tune | `29.8054` | `32.7145` | `29.4100` | `32.7040` | `12.3839` | `19.8320` |
| DIP after AMASS | `33.4361` | `39.2532` | `33.2060` | `39.2334` | `5.2618` | `28.3552` |
| DIP after DIP fine-tune | `34.0631` | `39.2021` | `34.2014` | `39.2169` | `5.2618` | `28.3552` |
| TotalCapture after DIP fine-tune | `30.3944` | `32.6989` | `29.8325` | `32.7034` | `12.3839` | `19.8320` |

Root-RMB artifacts:

```text
root: data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613
variant: D_all_balanced
summary: data/experiments/imu_joint_euler_qdot_vel_ctrl_v1_rootrmb_20260613/summary.json
checkpoints:
  D_all_balanced/amass_pretrain/best_loss.pt
  D_all_balanced/totalcapture_finetune/best_loss.pt
  D_all_balanced/dip_finetune/best_loss.pt
```

Root-RMB conclusion: not improved. It slightly worsens AMASS and TotalCapture rotation versus old world-RMB D, only slightly improves DIP after DIP fine-tune by `0.1383 deg`, and remains far worse than the same-cache baseline.

Best variant by rotation for each stage:

| Dataset / stage | Best variant | Rotation geodesic deg ↓ | vel_RJ L2 cm/s ↓ | acc_RJ L2 cm/s² ↓ | Baseline rotation deg ↓ | Baseline vel L2 cm/s ↓ | Baseline acc L2 cm/s² ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| AMASS after AMASS | `D_all_balanced` | `30.5556` | `29.4059` | `365.0700` | `4.0610` | `13.1790` | `393.4799` |
| TotalCapture after AMASS | `D_all_balanced` | `29.3217` | `32.7078` | `406.7416` | `12.3839` | `19.8320` | `562.0359` |
| TotalCapture after TC fine-tune | `A_qctrl_main` | `29.1832` | `32.5447` | `402.6187` | `12.3839` | `19.8320` | `562.0359` |
| DIP after AMASS | `C_vel_acc_strong` | `32.2599` | `39.2093` | `975.2344` | `5.2618` | `28.3552` | `965.4908` |
| DIP after DIP fine-tune | `C_vel_acc_strong` | `32.6004` | `39.2226` | `975.2334` | `5.2618` | `28.3552` | `965.4908` |
| TotalCapture after DIP fine-tune | `D_all_balanced` | `29.8325` | `32.7034` | `406.7268` | `12.3839` | `19.8320` | `562.0359` |

Variant snapshot on each variant's selected fine-tune stage:

| Variant | AMASS rot deg ↓ | AMASS vel L2 cm/s ↓ | TC-after-TC rot deg ↓ | TC-after-TC vel L2 cm/s ↓ | DIP-after-DIP rot deg ↓ | DIP-after-DIP vel L2 cm/s ↓ |
|---|---:|---:|---:|---:|---:|---:|
| `A_qctrl_main` | `33.0077` | `29.0785` | `29.1832` | `32.5447` | `33.5444` | `39.0059` |
| `B_qdot_qddot_strong` | `36.9067` | `29.0488` | `34.7390` | `31.7835` | `38.9002` | `40.2202` |
| `C_vel_acc_strong` | `33.1815` | `29.3996` | `29.4819` | `32.7112` | `32.6004` | `39.2226` |
| `D_all_balanced` | `30.5556` | `29.4059` | `29.4100` | `32.7040` | `34.2014` | `39.2169` |

Baseline is the same-cache `pose_prephysics FK root-relative` state from the cache/eval path. It is the available direct state baseline for this diagnostic, not an official PL output comparison.

### 7. Official S4 11 Metrics

Not run. This diagnostic module is not connected to PL/IK/NewIK1/full pipeline.

### 8. Conclusion

- Module GT: not improved. All four variants are much worse than the same-cache baseline on rotation and velocity.
- TotalCapture fine-tune gives only a small rotation improvement for `A_qctrl_main`: `30.0296 -> 29.1832 deg`, still far worse than baseline `12.3839 deg`.
- DIP fine-tune does not improve the best DIP result: `C_vel_acc_strong` changes `32.2599 -> 32.6004 deg`.
- Acceleration-heavy terms improve some acceleration/smoothness behavior, but do not recover useful q/qdot/velocity controls.
- Decision: rejected for IK/NewIK1/full-pipeline integration. The next useful step is target/representation debugging or residualizing against `pose_prephysics`, not pipeline promotion.

## Version: imu_neighbor_pos_from_vel_ctrl_v1

### 1. Purpose

Test whether frozen IMU-neighbor velocity controls can help a new diagnostic module predict the same IMU-adjacent nodes' root-relative position controls before considering IK/NewIK1 integration.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Module scope | velocity-control diagnostic only | position-from-velocity-control diagnostic | Test whether velocity controls contain useful position information |
| Input | `imu_neighbor_vel_ctrl_v1` 90D IMU input | 189D IMU + velocity control/decoded velocity/decoded acceleration | Condition position on motion-control estimates |
| Output | world-frame velocity control `neighbor_vel_W_control[33]` | root-relative position control `neighbor_pos_R_control[33]` | Produce PL-like position controls for neighboring nodes |
| Pipeline | not connected | not connected | Avoid full-pipeline claims before module-level evidence |

### 3. Input / Output Contract

| Item | Shape | Meaning |
|---|---:|---|
| IMU feature | 90D | `aM[18]+wM[18]+RMB_6d[36]+r_JS[18]` |
| Velocity control input | 33D | frozen `imu_neighbor_vel_ctrl_v1` predicted or GT-mixed `neighbor_vel_W_control` |
| Decoded velocity input | 33D | frozen velocity module decoded `neighbor_vel_W` |
| Decoded acceleration input | 33D | frozen velocity module decoded `neighbor_acc_W` |
| Output control | 33D | `neighbor_pos_R_control` |
| Decoded outputs | 99D | `pos_R[33]+vel_R[33]+acc_R[33]` |

Node layout is `[18,20], [19,21], [4,7], [5,8], [12,15], [0]`. Root-relative position uses the project row-vector contract `p_RJ = (p_WJ - p_WR) @ R_WR`. The root channel is retained only for 33D alignment; it is zero by construction and is not a useful improvement target.

DIP policy: no DIP `trans`, no DIP world/root velocity GT, and no fabricated finite-difference translation. DIP fine-tune uses pose-derived root-relative position supervision only.

### 4. Loss Design

| Stage | Loss terms |
|---|---|
| AMASS / TotalCapture | `ctrl_pos=1.0`, `decoded_pos=1.0`, `vel_R=0.2`, `acc_R=0.05`, `vel_input_consistency=0.05`, `segment_length=0.05`, `smooth=0.01`, `jerk=0.005`, `control_prior=0.001` |
| DIP | `ctrl_pos=1.0`, `decoded_pos=1.0`, `segment_length=0.05`, `smooth=0.01`, `jerk=0.005`, `control_prior=0.001`; `vel_R=0`, `acc_R=0`, `vel_input_consistency=0` |

Best checkpoint selection: `ctrl_pos + decoded_pos + 0.1 * vel_R`; smooth, jerk, and prior cannot dominate the selected checkpoint.

### 5. Training Recipe

| Stage | Data | Init | Batch | Best epoch | Best value | Checkpoint |
|---|---|---|---:|---:|---:|---|
| AMASS pretrain | `prephysics_pose_velocity_amass_k2_paired_offset_overlay` | scratch + frozen velocity ckpt | `1536` | `80` | `0.1038176935` | `data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/amass_pretrain/best_loss.pt` |
| TC fine-tune | TC train/val official offset_r caches | AMASS best | `512` | `58` | `0.1052717926` | `data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/totalcapture_finetune/best_loss.pt` |
| DIP fine-tune | DIP train/val official offset_r caches | AMASS best | `512` | `30` | `0.1142737220` | `data/experiments/imu_neighbor_pos_from_vel_ctrl_v1_20260613/dip_finetune/best_loss.pt` |

Training efficiency: all stages precompute compact per-sequence 189D features and position targets before epochs. Training is batched and does not run full eval per batch.

### 6. Module GT Metrics

| Dataset / checkpoint | pos_R L1 cm ↓ | pos_R L2 cm ↓ | vel_R L2 cm/s ↓ | acc_R L2 cm/s² ↓ | segment err cm ↓ | pose_prephysics baseline pos_R L2 cm ↓ |
|---|---:|---:|---:|---:|---:|---:|
| AMASS / AMASS best | `21.5704` | `48.0025` | `42.6341` | `500.2493` | `28.2466` | `2.3390` |
| TotalCapture test / AMASS best | `21.9064` | `48.7792` | `49.9797` | `570.0504` | `28.2452` | `5.3009` |
| TotalCapture test / TC best | `21.8107` | `48.5708` | `49.9762` | `569.9742` | `28.2637` | `5.3009` |
| DIP test / AMASS best | `21.1748` | `47.8958` | `54.4563` | `1244.7912` | `28.2476` | `5.5673` |
| DIP test / DIP best | `21.1186` | `47.7704` | `54.4569` | `1244.7978` | `28.2821` | `5.5673` |

Official PL and `newpl_v4_init36` are marked `not applicable` for the full 33D neighbor-node target because the inspected caches do not contain their corresponding 33D node-position outputs. This version was compared against the same-cache `pose_prephysics FK root-relative` baseline instead.

### 7. Official S4 11 Metrics

Not run. This is a diagnostic module and is not connected to PL/IK/NewIK1/full pipeline.

### 8. Conclusion

- Module GT: not improved. The new module is much worse than the same-cache `pose_prephysics` root-relative FK baseline on every dataset.
- TC fine-tune has only a small effect: TotalCapture pos_R L2 improves from `48.7792 cm` to `48.5708 cm`, still far worse than `5.3009 cm` baseline.
- DIP fine-tune has only a small effect: DIP pos_R L2 improves from `47.8958 cm` to `47.7704 cm`, still far worse than `5.5673 cm` baseline.
- Velocity controls do not yet provide a useful position-control estimator in this v1 architecture.
- Decision: rejected for IK/NewIK1 integration. Keep artifacts as diagnostic evidence; next work should debug architecture/target design before any pipeline connection.

## 5. PL-s1 Replacement Versions

## Version: newpl_v1_processed_no_baseline

### 1. Purpose

Replace official PL with PLCurve under processed input while removing baseline preservation/distillation.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | Official GPNet processed baseline | Processed IMU PL features | Test replacement under corrected RMB input |
| Output | Official PL 18D | Same 18D | Preserve downstream IK1 |
| Loss | Official/legacy | No baseline preserve/distill | Avoid anchoring to weaker PL |
| Training | official frozen | TC finetune 10 epochs | Adapt PL replacement |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | 84D | `aRB[18]+wRB[18]+RRB[45]+gR0[3]` | `GPNet.forward_frame` PL features |
| Output | 18D | `pRB[15]+gR1[3]` | official PL downstream contract |

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRB` | `1.0` | `0.0015100243501365185` |
| `gR1` | `1.0` | `0.18030107840895654` |
| `pRB_dot` | `0.03` | `2.4217713144025765e-05` |
| `pRB_ddot_smooth` | `1e-06` | `708.9556671142578` |
| `gR1_dot` | `0.03` | `not measured` |
| `gR1_ddot` | `0.001` | `not measured` |
| `gt_control_pRB` | `varies` | `not measured` |
| `gt_control_gR1` | `varies` | `not measured` |
| `total` | `mixed` | `0.1825556218624115` |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| TC finetune | TotalCapture train | processed PL cache | AMASS/previous PLCurve checkpoint | 10 | best_loss.pt |

### 6. Module GT Delta

| Version | pRB ↓ | gR1 ↓ | pRB_dot ↓ | pRB_ddot_smooth ↓ | gR1_dot ↓ | gR1_ddot ↓ | Total Loss Delta vs Previous ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| newpl_v1_processed_no_baseline | `0.0015100243501365185` | `0.18030107840895654` | `2.4217713144025765e-05` | `708.9556671142578` | `not measured` | `not measured` | not measured |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newpl_v1_processed_no_baseline | `38.762657` | `10.210696` | `8.806678` | `4.539996` | `5.168105` | `10.330306` | `8.520222` | `4.359863` | `4.894816` | `0.285659` | `0.477006` |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| official_processed | `0.008997` | not measured | S4 slightly worse than processed official baseline, so not selected. |

### 9. Conclusion

- Module GT: see Module GT Delta table
- Official S4: `38.762657`
- Keep as mainline: no
- Reason: S4 slightly worse than processed official baseline, so not selected.

## Version: newpl_v2_gRdyn

### 1. Purpose

Add temporal gR dynamics losses to improve gravity/root-direction continuity.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | newpl_v1 | unchanged | isolate loss change |
| Output | 18D | unchanged | preserve IK1 |
| Loss | no gR derivatives | add gR1_dot/gR1_ddot | smooth gR1 |
| Training | TC finetune | TC finetune continuation | test loss effect |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | 84D | `aRB[18]+wRB[18]+RRB[45]+gR0[3]` | `GPNet.forward_frame` PL features |
| Output | 18D | `pRB[15]+gR1[3]` | official PL downstream contract |

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRB` | `1.0` | `0.0015013986267149448` |
| `gR1` | `1.0` | `0.18012921810150145` |
| `pRB_dot` | `0.03` | `2.4236946228484157e-05` |
| `pRB_ddot_smooth` | `1e-06` | `710.6029846191407` |
| `gR1_dot` | `0.03` | `6.579346336366143e-05` |
| `gR1_ddot` | `0.001` | `6.764548697901773e-05` |
| `gt_control_pRB` | `varies` | `not measured` |
| `gt_control_gR1` | `varies` | `not measured` |
| `total` | `mixed` | `0.18238026313483716` |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| TC finetune | TotalCapture train | processed PL cache | newpl_v1 checkpoint | 10 | best_loss.pt |

### 6. Module GT Delta

| Version | pRB ↓ | gR1 ↓ | pRB_dot ↓ | pRB_ddot_smooth ↓ | gR1_dot ↓ | gR1_ddot ↓ | Total Loss Delta vs Previous ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| newpl_v2_gRdyn | `0.0015013986267149448` | `0.18012921810150145` | `2.4236946228484157e-05` | `710.6029846191407` | `6.579346336366143e-05` | `6.764548697901773e-05` | `-0.000175359` |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newpl_v2_gRdyn | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newpl_v1_processed_no_baseline | not measured | `-0.000175` | Module GT loss improves slightly, but S4 JSON was not found in current artifacts. |

### 9. Conclusion

- Module GT: see Module GT Delta table
- Official S4: not measured
- Keep as mainline: no
- Reason: Module GT loss improves slightly, but S4 JSON was not found in current artifacts.

## Version: newpl_v3_gtcontrol_rund

### 1. Purpose

Add GT spline-control supervision, Run D weights.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | newpl_v2 | unchanged | isolate control loss |
| Output | 18D | unchanged | preserve IK1 |
| Loss | gR dynamics | add gt_control_pRB=0.3, gt_control_gR1=0.1 | supervise control points |
| Training | TC finetune | continue 10 epochs | Run D recipe |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | 84D | `aRB[18]+wRB[18]+RRB[45]+gR0[3]` | `GPNet.forward_frame` PL features |
| Output | 18D | `pRB[15]+gR1[3]` | official PL downstream contract |

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRB` | `1.0` | `0.0014918483793735504` |
| `gR1` | `1.0` | `0.17994229570031167` |
| `pRB_dot` | `0.03` | `2.426689088679268e-05` |
| `pRB_ddot_smooth` | `1e-06` | `713.4133941650391` |
| `gR1_dot` | `0.03` | `6.576632804353722e-05` |
| `gR1_ddot` | `0.001` | `6.763922087884566e-05` |
| `gt_control_pRB` | `varies` | `0.001501525123603642` |
| `gt_control_gR1` | `varies` | `0.05906848106533289` |
| `total` | `mixed` | `0.1885452039539814` |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| TC finetune | TotalCapture train | processed PL cache | gRdyn checkpoint | 10 | best_loss.pt |

### 6. Module GT Delta

| Version | pRB ↓ | gR1 ↓ | pRB_dot ↓ | pRB_ddot_smooth ↓ | gR1_dot ↓ | gR1_ddot ↓ | Total Loss Delta vs Previous ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| newpl_v3_gtcontrol_rund | `0.0014918483793735504` | `0.17994229570031167` | `2.426689088679268e-05` | `713.4133941650391` | `6.576632804353722e-05` | `6.763922087884566e-05` | `0.00616494` |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newpl_v3_gtcontrol_rund | `38.694846` | `10.183679` | `8.784614` | `4.520307` | `5.150876` | `10.317185` | `8.517355` | `4.352148` | `4.887925` | `0.285478` | `0.476805` |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newpl_v2_gRdyn | not measured | `0.006165` | S4 improves over processed official and earlier PL variants; selected until init36. |

### 9. Conclusion

- Module GT: see Module GT Delta table
- Official S4: `38.694846`
- Keep as mainline: no
- Reason: S4 improves over processed official and earlier PL variants; selected until init36.

## Version: newpl_v4_init36

### 1. Purpose

Change only stream initialization to include real IMU offsets while preserving PL frame I/O.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | 18D/legacy init | 36D stream init `offset_r[18]+pRL[15]+gR0[3]` | reduce train/runtime init mismatch |
| Output | 18D | unchanged | preserve downstream IK1 |
| Loss | RunD-style | unchanged | isolate init change |
| Training | 10 epoch Run D | 60 epoch TC finetune from Run D | train expanded init encoder |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | 84D | `aRB[18]+wRB[18]+RRB[45]+gR0[3]` | `GPNet.forward_frame` PL features |
| Output | 18D | `pRB[15]+gR1[3]` | official PL downstream contract |

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRB` | `1.0` | `0.0014582541072741151` |
| `gR1` | `1.0` | `0.17935603857040405` |
| `pRB_dot` | `0.03` | `2.41788986386382e-05` |
| `pRB_ddot_smooth` | `1e-06` | `704.4198699951172` |
| `gR1_dot` | `0.03` | `6.565149778907653e-05` |
| `gR1_ddot` | `0.001` | `6.761706231372955e-05` |
| `gt_control_pRB` | `varies` | `0.0014678791398182512` |
| `gt_control_gR1` | `varies` | `0.05890091136097908` |
| `total` | `mixed` | `0.18789918906986713` |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| TC finetune | TotalCapture train | processed PL cache with offset_r | Run D checkpoint partial load | 60 | best_loss.pt / last.pt |

### 6. Module GT Delta

| Version | pRB ↓ | gR1 ↓ | pRB_dot ↓ | pRB_ddot_smooth ↓ | gR1_dot ↓ | gR1_ddot ↓ | Total Loss Delta vs Previous ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| newpl_v4_init36 | `0.0014582541072741151` | `0.17935603857040405` | `2.41788986386382e-05` | `704.4198699951172` | `6.565149778907653e-05` | `6.761706231372955e-05` | `-0.000646015` |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newpl_v4_init36 | `38.625657` | `10.135251` | `8.772134` | `4.495146` | `5.138202` | `10.290856` | `8.538511` | `4.346227` | `4.898401` | `0.285848` | `0.476914` |

### 7.1 DIP Test Full-Pipeline 11 Metrics

Backfilled on 2026-06-16 to close the historical gap. This is a DIP test
official-protocol full-pipeline run, not PL module-level pRB/gR1 evaluation.

Evaluator and contract:

```text
evaluator: newik1_real_streaming_audit.py
cache/protocol: data/experiments/newpl_v5_official_protocol_20260607/caches/dip_test_with_offset_r/baseline_cache_manifest.json
source raw DIP cache: data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json
checkpoint: data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
replacement: PL-s1 only; official IK-s1, IK-s2, VR, and carticulate physics downstream preserved
DIP trans/root-velocity supervision: not used; evaluation only
metric implementation: MotionEvaluator full-pipeline 11 metrics, GPU chunked metric evaluation, IK1 module metrics skipped
```

| Dataset | Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP-IMU-test | official_gpnet | `44.641437` | `13.548337` | `8.469859` | `4.648157` | `5.408259` | `13.409406` | `8.291682` | `4.547544` | `5.265691` | `0.157846` | `0.258183` |
| DIP-IMU-test | newpl_v4_init36_official_downstream | `44.708897` | `13.537034` | `8.484648` | `4.646514` | `5.426462` | `13.429909` | `8.329860` | `4.602831` | `5.356486` | `0.154876` | `0.251050` |

Decision: `newpl_v4_init36` does not improve over official GPNet on DIP
full-pipeline Score (`+0.067461`, worse). 不支持 DIP full-pipeline improvement
claim. It remains a historical processed-input TotalCapture/S4 artifact, not a
DIP full-pipeline improvement.

Artifacts:

```text
root: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616
summary: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/summary.md
result JSON: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/result_summary.json
baseline JSON: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/eval/dip_official_gpnet.json
newpl JSON: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/eval/dip_newpl_v4_init36_official_downstream.json
run log: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/logs/run.log
exact command: data/experiments/newpl_v4_init36_dip_fullpipeline_11metrics_20260616/exact_command.txt
```

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newpl_v3_gtcontrol_rund | `-0.069188` | `-0.000646` | Best current S4; selected mainline. |

### 9. Conclusion

- Module GT: see Module GT Delta table
- Official S4: `38.625657`
- Keep as mainline: yes
- Reason: Best historical processed-input S4 artifact. The 2026-06-16 DIP
  full-pipeline 11-metric backfill does not support a DIP improvement claim.

## Version: newpl_v5_official_protocol

### 1. Purpose

Test whether NewPL remains competitive under an official-like training route: AMASS pretrain followed by DIP-IMU train fine-tune, then module-level evaluation on DIP test and TotalCapture test.

### 2. Main Change

| Change Type | Previous Mainline | This Version | Motivation |
|---|---|---|---|
| Input | processed-input NewPL init36 mainline | official-input PL caches with init36 feature | mimic official-style AMASS -> DIP adaptation route |
| Output | `pRB[15]+gR1[3]` | unchanged 18D | preserve IK1 contract |
| Loss | RunD-style control losses | same family, selected by `control_physical` | choose best checkpoint by GT control-point fit |
| Training | TotalCapture-focused processed route | AMASS pretrain 80 epochs -> DIP fine-tune 40 epochs | test cross-dataset generalization without TC train |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | 84D | `aRB[18]+wRB[18]+RRB[45]+gR0[3]` | official PL feature construction |
| Init | 36D | `offset_r[18]+pRL[15]+gR0[3]` | cached init36 feature |
| Output | 18D | `pRB[15]+gR1[3]` | official downstream PL contract |

### 4. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| AMASS pretrain | AMASS cache | official PL init36 | none | 80 | `data/experiments/newpl_v5_official_protocol_20260607_tuned/amass_pretrain/best_loss.pt` |
| DIP fine-tune | DIP-IMU train cache | official DIP init36 | AMASS best | 40 | `data/experiments/newpl_v5_official_protocol_20260607_tuned/dip_finetune/best_loss.pt` |

Training details: AMASS used `batch_size=256`, `lr=1e-4`, best epoch `80`, selection value `0.002173126090565347`. DIP fine-tune used `batch_size=12`, `lr=5e-6`, best epoch `40`, selection value `0.038939811958698556`. No DIP `trans` supervision was used.

### 5. Module-Level Results

| Dataset | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | Observation |
|---|---|---:|---:|---:|---|
| DIP test | official_PL | `3.115170` | `6.419473` | `12.947709` | pRB baseline best |
| DIP test | newpl_v4_init36 | `3.116842` | `6.441447` | `12.765167` | close pRB, better gR1 than official |
| DIP test | newpl_v5_amass | `3.127556` | `6.454484` | `12.551949` | best gR1 before DIP FT, pRB worse |
| DIP test | newpl_v5_dip_best | `3.120847` | `6.445578` | `12.552613` | DIP FT improves pRB, still not pRB-best |
| TotalCapture test | official_PL | `3.370257` | `6.995536` | `13.450465` | official baseline |
| TotalCapture test | newpl_v4_init36 | `3.210470` | `6.654393` | `13.329531` | best among compared versions |
| TotalCapture test | newpl_v5_amass | `3.264119` | `6.783332` | `13.415420` | better than official, worse than v4 |
| TotalCapture test | newpl_v5_dip_best | `3.264551` | `6.780749` | `13.415189` | tiny TC improvement after DIP FT, still worse than v4 |

Full-pipeline 11 metrics: `not measured`.

Delay-output diagnostic:

| Dataset | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | Decision |
|---|---|---:|---:|---:|---|
| DIP test | official_PL | `3.115170` | `6.419473` | `12.947689` | pRB baseline best |
| DIP test | newpl_v5_dip_delay0 | `3.120847` | `6.445578` | `12.552613` | best v5 pRB, not official-best |
| DIP test | newpl_v5_dip_delay1 | `3.129481` | `6.455922` | `12.554120` | pRB regresses |
| DIP test | newpl_v5_dip_delay2 | `3.234059` | `6.655651` | `12.553713` | pRB regresses strongly |
| TotalCapture test | official_PL | `3.370257` | `6.995536` | `13.450445` | official baseline |
| TotalCapture test | newpl_v5_dip_delay0 | `3.264551` | `6.780749` | `13.415190` | best v5 pRB |
| TotalCapture test | newpl_v5_dip_delay1 | `3.279620` | `6.815473` | `13.416209` | pRB regresses |
| TotalCapture test | newpl_v5_dip_delay2 | `3.381457` | `7.026811` | `13.417055` | worse than official pRB |

Delay result: future-output delay `pred[t+d] -> GT[t]` slightly reduces some jitter/temporal terms, but it worsens pRB L1/L2. Delay0 remains the best v5 setting, so the 1/2-frame delayed-output idea is not selected for full-pipeline S4.

### 6. Artifacts

```text
script: scripts/run_newpl_v5_official_protocol_20260607.sh
root: data/experiments/newpl_v5_official_protocol_20260607_tuned
eval JSONs: data/experiments/newpl_v5_official_protocol_20260607_tuned/eval/
summary: data/experiments/newpl_v5_official_protocol_20260607_tuned/summary.json
delay script: scripts/run_newpl_v5_delay_eval_20260607.sh
delay root: data/experiments/newpl_v5_delay_eval_20260607
delay summary: data/experiments/newpl_v5_delay_eval_20260607/summary.json
```

### 7. Conclusion

- Keep as mainline: no.
- Reason: v5 is useful evidence that the official-like route improves DIP gR1 and mildly improves pRB after DIP fine-tune, but it does not beat `newpl_v4_init36` on TotalCapture module metrics and still does not beat official_PL on DIP pRB.
- Next use: diagnostic branch only unless a later loss/selection change makes pRB no worse than baselines or a full-pipeline run proves downstream benefit.

## Version: newpl_v5_loss_family_ablation

### 1. Purpose

Test the user's requested NewPL v5 loss-family question: what happens if control-point loss is removed, and what do q, qdot, and qddot losses contribute to training?

### 2. Main Change

| Change Type | Previous | This Experiment | Motivation |
|---|---|---|---|
| Input | official PL 84D + init36 | unchanged | isolate losses |
| Output | `pRB[15]+gR1[3]` | unchanged 18D | preserve IK1 contract |
| Loss | fixed v5 loss mix | 8 variants over q, control, qdot, qddot | measure each family |
| Training | AMASS -> DIP | same route per variant | same-protocol comparison |
| Selection | control/physical v5 selection | `pl_physical` for every variant | select by decoded PL module state |

Important contract note: in this experiment `q` means decoded NewPL state `pRB[15]+gR1[3]`, not full RBDL `q75`.

### 3. Variant Definition

| Variant | q | Control | qdot | qddot |
|---|---:|---:|---:|---:|
| `q_only` | yes | no | no | no |
| `q_control` | yes | yes | no | no |
| `q_qdot` | yes | no | yes | no |
| `q_qddot` | yes | no | no | yes |
| `q_qdot_qddot` | yes | no | yes | yes |
| `q_control_qdot` | yes | yes | yes | no |
| `q_control_qddot` | yes | yes | no | yes |
| `q_control_qdot_qddot` | yes | yes | yes | yes |

### 4. Module GT Comparison

After AMASS pretrain and DIP fine-tune, best DIP-finetune checkpoints:

| Dataset | Variant | pRB L2 cm | gR1 angle deg | pRB smooth jitter cm | gR1 smooth jitter |
|---|---|---:|---:|---:|---:|
| DIP test | `q_control_qddot` | `6.426853` | `12.707329` | `0.212453` | `0.001227` |
| DIP test | `q_qdot_qddot` | `6.427028` | `12.698597` | `0.212451` | `0.001224` |
| DIP test | `q_qddot` | `6.430836` | `12.698030` | `0.212463` | `0.001225` |
| DIP test | `q_control` | `6.434509` | `12.698860` | `0.212400` | `0.001224` |
| DIP test | `q_qdot` | `6.437400` | `12.687360` | `0.212404` | `0.001221` |
| DIP test | `q_only` | `6.437678` | `12.689232` | `0.212439` | `0.001221` |
| TotalCapture test | `q_only` | `6.753091` | `13.575686` | `0.531764` | `0.006081` |
| TotalCapture test | `q_qdot` | `6.754438` | `13.574678` | `0.531764` | `0.006081` |
| TotalCapture test | `q_qddot` | `6.756564` | `13.582943` | `0.531758` | `0.006082` |
| TotalCapture test | `q_control` | `6.763036` | `13.580624` | `0.531750` | `0.006095` |
| TotalCapture test | `q_control_qddot` | `6.771322` | `13.586917` | `0.531899` | `0.006105` |
| TotalCapture test | `q_control_qdot_qddot` | `6.778233` | `13.589017` | `0.531641` | `0.006101` |

Baseline context:

| Dataset | Version | pRB L2 cm | gR1 angle deg | Observation |
|---|---|---:|---:|---|
| DIP test | official_PL | `6.419473` | `12.947709` | still best pRB |
| DIP test | newpl_v4_init36 | `6.441447` | `12.765167` | ablations beat pRB, but not selected without TC/full-pipeline gain |
| DIP test | raw newpl_v5_dip_best | `6.445578` | `12.552613` | ablations improve pRB but lose gR1 |
| TotalCapture test | official_PL | `6.995536` | `13.450465` | ablations beat pRB but lose gR1 |
| TotalCapture test | newpl_v4_init36 | `6.654393` | `13.329531` | still strongest TC pRB/gR1 among these PL replacements |
| TotalCapture test | raw newpl_v5_dip_best | `6.780749` | `13.415189` | `q_only` improves pRB, but all ablations lose gR1 |

### 5. Loss-Family Interpretation

| Question | Evidence | Interpretation |
|---|---|---|
| Does dropping control-point loss help? | `q_only` is best on TC pRB, but not best on DIP pRB; `q_control` improves DIP pRB by only `0.003169 cm` and worsens TC pRB by `0.009945 cm` vs `q_only`. | Control-point loss is not a reliable generalization gain in this recipe. |
| Does qdot help? | `q_qdot` changes DIP pRB by `0.000278 cm` and TC pRB by `+0.001347 cm` vs `q_only`. | qdot is effectively negligible at current weights. |
| Does qddot help? | `q_qddot` improves DIP pRB by `0.006842 cm` vs `q_only`, but worsens TC pRB by `0.003473 cm`. | qddot is a weak DIP pRB regularizer, not a robust TC gain. |
| Does control+qddot help? | `q_control_qddot` is best DIP pRB at `6.426853`, but TC pRB is `6.771322`. | Best local DIP number over-specializes and is not selected. |

Gradient audit:

| Stage | q/control cosine | qdot/qddot cosine | Readout |
|---|---:|---:|---|
| AMASS init | `0.846724` | `-0.862329` | q and control initially agree; qdot and qddot strongly oppose |
| DIP from v5 AMASS | `0.227377` | `-0.739343` | control becomes weakly aligned after pretraining; qdot/qddot conflict remains |

qdot gradient norms are tiny in both audits. qddot gradients are mainly from `pRB_ddot_smooth`, so current qddot behavior should be read as smoothness regularization more than true physical acceleration supervision.

### 6. Artifacts

```text
root: data/experiments/newpl_v5_loss_family_ablation_20260611
task file: configs/newpl_v5_loss_family_ablation_20260611_tasks.json
variant runner: scripts/run_newpl_v5_loss_family_variant_20260611.sh
summary script: scripts/summarize_newpl_v5_loss_ablation.py
gradient audit: newpl_v5_loss_gradient_audit.py
summary: data/experiments/newpl_v5_loss_family_ablation_20260611/summary.json
csv: data/experiments/newpl_v5_loss_family_ablation_20260611/summary_eval_rows.csv
logs: logs/orchestrator/newpl_v5_loss_family_ablation_20260611/
```

### 7. Conclusion

- Keep as mainline: no.
- Reason: no loss-family variant gives a robust win across DIP and TotalCapture. The best DIP pRB variant still misses official_PL pRB, and the best TC pRB variant still misses `newpl_v4_init36` while worsening gR1.
- Next use: keep qddot as a possible weak regularizer to test with a better acceleration target; do not treat control-point loss as mandatory for NewPL v5, and do not promote this branch without full-pipeline evidence.

## Version: newpl_v5_smoothacc

### 1. Purpose

Test whether replacing official raw acceleration with smoothed acceleration improves the NewPL v5 `pRB/gR1` module output under the same AMASS -> DIP protocol.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | official raw `aM/wM/RMB` | `aM` is centered moving-average smoothed, window=9; `wM/RMB` unchanged | Test acceleration smoothing as a direct PL input change |
| Output | `pRB[15]+gR1[3]` | unchanged 18D | Preserve IK1 contract |
| Init | `offset_r[18]+pRL[15]+gR0[3]` | unchanged 36D | Keep v5 init36 route |
| Loss | v5 pRB/gR1/control/dynamics | unchanged | Isolate input smoothing |
| Training | AMASS -> DIP | same route, with precomputed smooth raw and PL caches | Avoid repeated smoothing/feature construction |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | 84D | `aRB[18]+wRB[18]+RRB[45]+gR0[3]` | built from smoothed `aM`, unchanged `wM/RMB` |
| Audit input | same as source | original acceleration | saved as `aM_raw` in smooth raw caches |
| Output | 18D | `pRB[15]+gR1[3]` | official PL downstream contract |
| Init | 36D | `offset_r[18]+pRL[15]+gR0[3]` | unchanged v5 init36 |

### 4. Loss Design

Same as `newpl_v5_official_protocol`:

| Loss | Weight | Notes |
|---|---:|---|
| `pRB` | `1.0` | decoded PL output vs GT |
| `gR1` | `1.0` | decoded gravity direction vs GT |
| `gt_control_pRB` | `0.3` | canonical GT control cache |
| `gt_control_gR1` | `0.1` | canonical GT control cache |
| `pRB_ddot_smooth` | `0.000001` | smoothness |
| `gR1_dot` | `0.03` | temporal |
| `gR1_ddot` | `0.001` | temporal |
| `baseline_pRB/gR1` | `0.0/0.0` | disabled |
| IK distill | disabled | cache-only PL training |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs / Stop | Output |
|---|---|---|---|---:|---|
| Smooth cache | AMASS + DIP train/val/test + TC test | smooth `aM`, raw `wM/RMB` | none | preprocessing | `data/experiments/newpl_v5_smoothacc_20260612/caches` |
| AMASS pretrain | AMASS PL cache | smooth official input | none | best epoch `3`, stopped at `15/80` | `data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/amass_pretrain/best_loss.pt` |
| DIP fine-tune | DIP train/val PL cache | smooth official input | AMASS best | best epoch `39/40` | `data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/dip_finetune/best_loss.pt` |
| Eval | AMASS proxy, DIP test, TC test | smooth official input | fixed checkpoints | module-level only | `data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/eval/*.json` |

Training settings: AMASS batch `256`, DIP batch `24`, `val_window_length=61`, final DIP/TC eval full sequences. Full-pipeline 11 metrics were not run. DIP translation/root velocity was not used.

### 6. Module GT Comparison

| Dataset / Stage | Version | pRB L2 cm ↓ | gR1 angle deg ↓ | Decision |
|---|---|---:|---:|---|
| DIP test after AMASS | smooth official_PL | `6.345701` | `12.902131` | input smoothing improves raw official slightly |
| DIP test after AMASS | newpl_v5_smoothacc_amass_best | `6.354993` | `12.891370` | worse pRB than smooth official, slightly better gR1 |
| TC test after AMASS | smooth official_PL | `7.508985` | `13.170892` | TC pRB regresses vs raw official |
| TC test after AMASS | newpl_v5_smoothacc_amass_best | `7.484481` | `13.174086` | slightly better pRB than smooth official, worse than v4/raw v5 |
| DIP test after DIP FT | smooth official_PL | `6.345701` | `12.902131` | smoothed input baseline |
| DIP test after DIP FT | newpl_v5_smoothacc_dip_best | `6.350327` | `12.894731` | pRB slightly worse than smooth official; gR1 slightly better |
| TC test after DIP FT | smooth official_PL | `7.508985` | `13.170880` | smoothed input baseline |
| TC test after DIP FT | newpl_v5_smoothacc_dip_best | `7.473741` | `13.185523` | pRB slightly better than smooth official, but worse than v4/raw v5 |
| TC test after DIP FT | newpl_v4_init36_smoothacc | `7.119541` | `13.075063` | stronger than smoothacc v5 under the same smooth input |

Per-leaf examples after DIP fine-tune:

| Dataset | Version | leaf_1 | leaf_2 | leaf_3 | leaf_4 | leaf_5 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP test | smooth official_PL | `8.350630` | `8.765077` | `4.303051` | `5.137257` | `5.172492` | `6.345701` |
| DIP test | newpl_v5_smoothacc_dip_best | `8.372464` | `8.792264` | `4.272254` | `5.150879` | `5.163776` | `6.350327` |
| TC test | smooth official_PL | `6.812538` | `7.092184` | `7.168602` | `6.805456` | `9.666146` | `7.508985` |
| TC test | newpl_v5_smoothacc_dip_best | `6.816790` | `7.088333` | `7.113846` | `6.760619` | `9.589119` | `7.473741` |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newpl_v5_smoothacc | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |

### 8. Artifacts

```text
cache builder: scripts/build_smooth_acc_cache.py
runner: scripts/run_newpl_v5_smoothacc_20260612.sh
summary script: scripts/summarize_newpl_v5_smoothacc.py
full root: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256
shared smooth cache root: data/experiments/newpl_v5_smoothacc_20260612/caches
summary: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/summary.json
log: data/experiments/newpl_v5_smoothacc_20260612_fastval2_b256/logs/run.log
```

### 9. Conclusion

- Keep as mainline: no.
- Reason: smoothed acceleration is not uniformly beneficial. It improves DIP pRB slightly and improves gR1, but it strongly worsens TotalCapture pRB. The retrained smooth-acc v5 does not beat the smoothed official PL on DIP pRB, and it is worse than `newpl_v4_init36_smoothacc` on TotalCapture pRB/gR1.
- Next use: keep only as diagnostic evidence that acceleration smoothing changes PL behavior; do not connect to IK1/full pipeline.


## Version: newpl_v5_butteracc

### 1. Purpose

Test whether a realtime-valid causal acceleration filter can replace raw official `aM` for NewPL v5 without the offline lookahead used by centered smooth-acc.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | official raw `aM/wM/RMB` | `aM` is causal Butterworth low-pass filtered; `wM/RMB` unchanged | Test realtime acceleration smoothing |
| Filter | none | order-2 Butterworth, fs=60 Hz, cutoff sweep 8/10/12 Hz | low latency, no future frames |
| Output | `pRB[15]+gR1[3]` | unchanged 18D | Preserve IK1 contract |
| Init | `offset_r[18]+pRL[15]+gR0[3]` | unchanged 36D | Keep v5 init36 route |
| Loss | v5 pRB/gR1/control/dynamics | unchanged in forced fc12 longtrain | Test whether training can compensate causal filter shift |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | 84D | `aRB[18]+wRB[18]+RRB[45]+gR0[3]` | built from causal-filtered `aM`, unchanged `wM/RMB` |
| Audit input | same as source | original acceleration | saved as `aM_raw` in ButterAcc raw caches |
| Output | 18D | `pRB[15]+gR1[3]` | official PL downstream contract |
| Init | 36D | `offset_r[18]+pRL[15]+gR0[3]` | unchanged v5 init36 |

Realtime contract: zero lookahead; `lookahead_frames=0`; `latency_ms=0`; output at frame `t` depends only on frames `<=t`. The cache builder initializes the filter state from the first sample, matching a streaming filter started from the first available IMU frame.

Root velocity is not part of this experiment. No root velocity head, root velocity training, or root velocity GT metric was used.

### 4. Loss Design

Same as `newpl_v5_official_protocol`; executed only in the forced fc12 longtrain after the input-only gate failed:

| Loss | Weight | Status |
|---|---:|---|
| `pRB` | `1.0` | trained |
| `gR1` | `1.0` | trained |
| `gt_control_pRB` | `0.3` | trained |
| `gt_control_gR1` | `0.1` | trained |
| `pRB_ddot_smooth` | `0.000001` | trained |
| `gR1_dot` | `0.03` | trained |
| `gR1_ddot` | `0.001` | trained |
| `baseline_pRB/gR1` | `0.0/0.0` | disabled |
| IK distill | disabled | disabled |

### 5. Training / Gate Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs / Stop | Output |
|---|---|---|---|---:|---|
| Smoke | DIP/TC small subsets | forced fc10 causal ButterAcc | none | 1 epoch smoke only | `data/experiments/newpl_v5_butteracc_20260612_smoke` |
| Input-only sweep | DIP test + TC test | causal ButterAcc fc8/fc10/fc12 | official PL, `newpl_v4_init36`, raw `newpl_v5_dip_best` | module eval only | `data/experiments/newpl_v5_butteracc_20260612_full/eval/input_only_*.json` |
| Selection | TC guard | require TC official ButterAcc pRB L2 <= raw official TC pRB L2 + 0.10 cm | none | no candidate | `selection.json` |
| Forced AMASS pretrain | AMASS | fc12 causal ButterAcc | none | best epoch `3`, stopped at `15/80` | `data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/amass_pretrain/best_loss.pt` |
| Forced DIP fine-tune | DIP train/val | fc12 causal ButterAcc | AMASS best | best epoch `38/40` | `data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/dip_finetune/best_loss.pt` |

Full-pipeline 11 metrics were not run. DIP translation/root velocity was not used.

Forced longtrain settings: AMASS batch `512`, DIP batch `32`, `val_window_length=61`, best checkpoint selected by `control_physical`.

### 6. Module GT Comparison

Input-only gate compared the same filtered input across official PL, `newpl_v4_init36`, and raw `newpl_v5_dip_best` checkpoints.

| Dataset / Cutoff | Version | pRB L1 cm | pRB L2 cm | gR1 deg | Decision |
|---|---|---:|---:|---:|---|
| DIP fc8 | official_PL_butter | `3.351340` | `6.907898` | `12.889588` | worse pRB than raw official |
| DIP fc8 | newpl_v4_init36_butter | `3.357099` | `6.934424` | `12.711126` | worse pRB than official butter |
| DIP fc8 | newpl_v5_raw_dip_butter | `3.366243` | `6.950735` | `12.503217` | best gR1, worse pRB |
| TC fc8 | official_PL_butter | `3.603690` | `7.506753` | `13.217498` | fails TC guard by `+0.511217 cm` |
| TC fc8 | newpl_v4_init36_butter | `3.436870` | `7.147257` | `13.118172` | still worse than raw v4 TC |
| TC fc8 | newpl_v5_raw_dip_butter | `3.494429` | `7.287691` | `13.173159` | worse than raw v5 TC |
| DIP fc10 | official_PL_butter | `3.290294` | `6.780699` | `12.893975` | worse pRB than raw official |
| DIP fc10 | newpl_v4_init36_butter | `3.295261` | `6.806182` | `12.714799` | worse pRB than official butter |
| DIP fc10 | newpl_v5_raw_dip_butter | `3.303543` | `6.820206` | `12.506282` | best gR1, worse pRB |
| TC fc10 | official_PL_butter | `3.557797` | `7.404335` | `13.211574` | fails TC guard by `+0.408799 cm` |
| TC fc10 | newpl_v4_init36_butter | `3.386474` | `7.036701` | `13.113819` | better than official butter, worse than raw v4 TC |
| TC fc10 | newpl_v5_raw_dip_butter | `3.443299` | `7.172809` | `13.173770` | worse than raw v5 TC |
| DIP fc12 | official_PL_butter | `3.251481` | `6.699946` | `12.900018` | best ButterAcc pRB, still worse than raw official by `+0.280473 cm` |
| DIP fc12 | newpl_v4_init36_butter | `3.255420` | `6.724124` | `12.720327` | worse pRB than official butter |
| DIP fc12 | newpl_v5_raw_dip_butter | `3.263422` | `6.736792` | `12.510739` | best gR1, worse pRB |
| TC fc12 | official_PL_butter | `3.496175` | `7.267198` | `13.254831` | fails TC guard by `+0.271662 cm` |
| TC fc12 | newpl_v4_init36_butter | `3.325723` | `6.903734` | `13.152519` | best ButterAcc TC pRB, still worse than raw v4 TC |
| TC fc12 | newpl_v5_raw_dip_butter | `3.381516` | `7.034841` | `13.218887` | worse than raw v5 TC |
| AMASS after AMASS | newpl_v5_butteracc_amass_best | `1.816958` | `3.782071` | `4.859811` | close to official, worse pRB than official fc12 |
| DIP after AMASS | newpl_v5_butteracc_amass_best | `3.262136` | `6.722528` | `12.894359` | worse pRB than official fc12 |
| TC after AMASS | newpl_v5_butteracc_amass_best | `3.498118` | `7.271788` | `13.251097` | worse pRB than official fc12 |
| DIP after DIP FT | newpl_v5_butteracc_dip_best | `3.261222` | `6.721462` | `12.896323` | DIP fine-tune gives only `-0.001066 cm` pRB vs AMASS checkpoint |
| TC after DIP FT | newpl_v5_butteracc_dip_best | `3.495333` | `7.266006` | `13.257684` | near official fc12, still worse than raw official/raw v5/v4 |

Per-leaf examples at the least-bad fc12 cutoff:

| Dataset | Version | leaf_1 | leaf_2 | leaf_3 | leaf_4 | leaf_5 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP fc12 | official_PL_butter | `9.087043` | `9.425798` | `4.402759` | `5.353347` | `5.230786` | `6.699947` |
| DIP fc12 | newpl_v4_init36_butter | `9.153128` | `9.497437` | `4.230015` | `5.426576` | `5.313463` | `6.724124` |
| DIP fc12 | newpl_v5_raw_dip_butter | `9.092113` | `9.579000` | `4.309003` | `5.394506` | `5.309338` | `6.736792` |
| TC fc12 | official_PL_butter | `6.707833` | `6.984410` | `6.754984` | `6.915042` | `8.973717` | `7.267197` |
| TC fc12 | newpl_v4_init36_butter | `6.782565` | `7.052000` | `6.184772` | `6.385552` | `8.113783` | `6.903734` |
| TC fc12 | newpl_v5_raw_dip_butter | `6.756134` | `7.184214` | `6.587662` | `6.444391` | `8.201804` | `7.034841` |
| DIP after DIP FT | newpl_v5_butteracc_dip_best | `9.129390` | `9.476355` | `4.399704` | `5.370505` | `5.231353` | `6.721461` |
| TC after DIP FT | newpl_v5_butteracc_dip_best | `6.740865` | `7.013272` | `6.733105` | `6.903114` | `8.939674` | `7.266006` |

### 7. Official S4 11 Metrics

| Version | Score | Local SIP | Local Angle | Local Joint | Local Mesh | Global SIP | Global Angle | Global Joint | Global Mesh | Root Jitter | Joint Jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newpl_v5_butteracc | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |

### 8. Artifacts

```text
filter code: l4_sensor_offset_utils.py::causal_butterworth_lowpass_sequence
cache builder: scripts/build_smooth_acc_cache.py
runner: scripts/run_newpl_v5_butteracc_20260612.sh
summary script: scripts/summarize_newpl_v5_butteracc.py
smoke root: data/experiments/newpl_v5_butteracc_20260612_smoke
full root: data/experiments/newpl_v5_butteracc_20260612_full
forced fc12 longtrain root: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain
summary: data/experiments/newpl_v5_butteracc_20260612_full/summary.json
selection: data/experiments/newpl_v5_butteracc_20260612_full/selection.json
eval: data/experiments/newpl_v5_butteracc_20260612_full/eval/*.json
forced summary: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/summary.json
forced checkpoints: data/experiments/newpl_v5_butteracc_20260612_forced_fc12_longtrain/{amass_pretrain,dip_finetune}/best_loss.pt
log: data/experiments/newpl_v5_butteracc_20260612_full/logs/run.log
```

### 9. Conclusion

- Keep as mainline: no.
- Reason: every causal ButterAcc cutoff fails the TotalCapture pRB guard and also worsens DIP pRB versus raw official input. The forced fc12 longtrain does not fix this: after DIP fine-tune, pRB L2 is `6.721462 cm` on DIP and `7.266006 cm` on TotalCapture, both worse than raw official PL and raw `newpl_v5_dip_best`.
- Next use: do not train or connect this input replacement to IK1/full pipeline. A future realtime filter needs either different state compensation or a model trained with a robust TC-preserving gate.

## Version: newpl_v6_next_control

### 1. Purpose

Test whether a PLCurve/NewPL module can predict the next PL control point and decode a better one-step-ahead `pRB/gR1` plus spline velocity/acceleration, without changing the official current-frame PL output consumed by IK1.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | official PL 84D | unchanged: `aRB[18]+wRB[18]+RRB[45]+gR0[3]` | fair PL replacement contract |
| Init | init36 available | unchanged: `offset_r[18]+pRL[15]+gR0[3]` | preserve NewPL/init36 setup |
| Current output | `pRB_t[15]+gR1_t[3]` | unchanged 18D `pl` output | keep official IK1 interface |
| Aux output | none | `next_pl`, `next_pldot`, `next_plddot`, `next_control`, preview tail4 controls | evaluate one-step prediction and dynamics |
| Training cache | PL cache + canonical GTControlCache | `pl_next_control_cache_v2` with t+1 targets, derivatives, last-control target, and tail4 target/mask | avoid recomputing upstream/FK per batch |

### 3. Control Point Time Semantics

Current PLCurve appends a tail control point each frame and decodes current `pl_t` from uniform-cubic curve index `[-2]`; `new_control_t` is therefore a tail control point, not the direct frame output. The verified current-frame PLCurve path also predicts a `tail_delta` that adjusts the last up-to-four existing controls before appending the new one. Corrected `newpl_v6_next_control` mirrors that for preview only: it predicts exactly one future `next_control`, adjusts the last up-to-four preview controls, appends `next_control + ghost(next_control)`, and decodes `next_pl`/derivatives from preview index `[-2]`. The preview control is not appended to the persistent streaming buffer.

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRB`, `gR1` | `1.0`, `1.0` | recorded in train JSON |
| `gt_control_pRB`, `gt_control_gR1` | `0.3`, `0.1` | recorded in train JSON |
| `pRB_dot`, `gR1_dot`, `gR1_ddot`, `pRB_ddot_smooth` | `0.03`, `0.03`, `0.001`, `1e-6` | recorded in train JSON |
| `next_pRB`, `next_gR1` | `1.0`, `1.0` | recorded in train JSON |
| `next_gt_control_pRB`, `next_gt_control_gR1` | `0.3`, `0.1` | recorded in train JSON |
| `next_pRB_vel`, `next_pRB_acc` | `0.03`, `0.0003` | recorded in train JSON |
| `next_gR1_vel`, `next_gR1_acc` | `0.03`, `0.001` | recorded in train JSON |
| `next_control_delta_prior` | `0.01` | recorded in train JSON |
| `last_control_pRB`, `last_control_gR1` | `0.3`, `0.1` | recorded in train JSON |
| `next_tail4_control_pRB`, `next_tail4_control_gR1` | `0.15`, `0.05` | recorded in train JSON |

Checkpoint selection saves `best_total_loss.pt`, `best_current_module_metric.pt`, `best_next_module_metric.pt`, `best_dynamics_metric.pt`, `best_control_metric.pt`, and `last.pt`. Current score is `pRB_t_L2_cm + 0.1*gR1_t_angle_deg`; next score uses the same formula on `t+1`; dynamics score is `pRB_vel_L2_cm_s + 0.01*pRB_acc_L2_cm_s2`; control score combines last/next/tail4 control pRB L2 plus a small gR1-angle term.

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| Smoke AMASS pretrain | AMASS next-control cache, 4 seq | official PL cache + init36 | none | 1 | `data/experiments/newpl_v6_next_control_tail4_20260611/smoke/amass_pretrain/best_next_module_metric.pt` |
| Smoke DIP fine-tune | DIP train/val next-control cache, 4 seq | official DIP IMU input cache | AMASS smoke best | 1 | `data/experiments/newpl_v6_next_control_tail4_20260611/smoke/dip_finetune/best_next_module_metric.pt` |
| Smoke TotalCapture eval | TotalCapture test next-control cache, 4 seq | official TC IMU input cache | AMASS/DIP smoke checkpoints | 0 | `data/experiments/newpl_v6_next_control_tail4_20260611/smoke/eval_totalcapture_test_after_dip_finetune.json` |
| Full AMASS pretrain | AMASS next-control cache | official PL cache + init36 | none | 80 | `data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/amass_pretrain/best_*.pt` |
| Full DIP fine-tune | DIP train/val next-control cache | official DIP IMU input cache | AMASS best-next checkpoint | 40 | `data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/dip_finetune/best_*.pt` |
| Full eval | AMASS, DIP test, TotalCapture test | official input caches | fixed official/v4 baselines plus v6 checkpoints | 0 | `data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/*.json` |

Full run command:

```text
CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=512 WINDOW=81 MAX_TRAIN_VAL_SEQS=128 VAL_BATCH_SIZE=64 RUN_SUFFIX=fastval1 /home/lingfeng/bin/longrun -- bash scripts/run_newpl_v6_next_control_20260611.sh full
```

### 6. Module GT Metrics

Smoke current-frame comparison:

| Dataset | Version | pRB L2 cm ↓ | gR1 angle deg ↓ | Notes |
|---|---|---:|---:|---|
| AMASS smoke | official PL baseline | `3.9585` | `4.5255` | cached official PL |
| AMASS smoke | newpl_v4_init36 baseline | `4.3979` | `4.6766` | same smoke cache |
| AMASS smoke | newpl_v6_next_control_amass | `3.9703` | `4.5235` | 1 epoch smoke |
| DIP test smoke | official PL baseline | `6.1285` | `10.1790` | cached official PL |
| DIP test smoke | newpl_v4_init36 baseline | `6.2891` | `10.1212` | same smoke cache |
| DIP test smoke | newpl_v6_next_control_dip | `6.1153` | `10.1778` | 1 epoch DIP smoke FT |
| TotalCapture test smoke | official PL baseline | `6.9955` | `13.4504` | cached official PL |
| TotalCapture test smoke | newpl_v4_init36 baseline | `6.6544` | `13.3295` | same smoke cache |
| TotalCapture test smoke | newpl_v6_next_control_dip | `6.9956` | `13.4468` | TC eval-only after DIP smoke FT |

Smoke next/dynamics comparison:

| Dataset | Version | next pRB L2 cm ↓ | pRB vel L2 cm/s ↓ | pRB acc L2 cm/s^2 ↓ | Notes |
|---|---|---:|---:|---:|---|
| AMASS smoke | official PL baseline | `4.1225` | `18.7863` | `1282.4579` | causal persistence + finite differences |
| AMASS smoke | newpl_v4_init36 baseline | `4.5802` | `18.5558` | `1084.0737` | causal persistence + finite differences |
| AMASS smoke | newpl_v6_next_control_amass | `4.1196` | not measured in summary | not measured in summary | direct next-control decode |
| DIP test smoke | official PL baseline | `6.1329` | `22.0227` | `787.6055` | no DIP trans/root velocity GT used |
| DIP test smoke | newpl_v4_init36 baseline | `6.3227` | `21.9740` | `767.7273` | causal persistence + finite differences |
| DIP test smoke | newpl_v6_next_control_dip | `6.1325` | not measured in summary | not measured in summary | direct next-control decode |
| TotalCapture test smoke | official PL baseline | `7.0954` | `34.3235` | `2173.8883` | causal persistence + finite differences |
| TotalCapture test smoke | newpl_v4_init36 baseline | `6.7852` | `32.9648` | `1861.3888` | causal persistence + finite differences |
| TotalCapture test smoke | newpl_v6_next_control_dip | `7.0963` | not measured in summary | not measured in summary | TC eval-only after DIP smoke FT |

Smoke control-point comparison:

| Dataset | Version | current control pRB L2 cm ↓ | next control pRB L2 cm ↓ | tail4 control pRB L2 cm ↓ | Notes |
|---|---|---:|---:|---:|---|
| AMASS smoke | official PL baseline | not available | not available | not available | cached official PL has no learned controls |
| AMASS smoke | newpl_v4_init36 baseline | `4.3922` | not available | not available | current control only |
| AMASS smoke | newpl_v6_next_control_amass | `3.9702` | `4.1305` | `3.9646` | corrected preview tail4 branch |
| DIP test smoke | newpl_v6_next_control_dip | `6.1291` | `6.1335` | `6.1274` | corrected preview tail4 branch |
| TotalCapture test smoke | newpl_v6_next_control_dip | `7.0016` | `7.0992` | `7.0016` | TC eval-only |

Full current/next/dynamics comparison:

| Dataset | Version | Training route | current pRB L2 cm ↓ | current gR1 deg ↓ | next pRB L2 cm ↓ | pRB vel L2 cm/s ↓ | pRB acc L2 cm/s^2 ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| AMASS | official PL baseline | fixed baseline | `2.8275` | `7.2199` | `2.9476` | `12.1129` | `871.7360` |
| AMASS | newpl_v4_init36 baseline | fixed baseline | `2.8989` | `7.1207` | `3.0444` | `11.9329` | `726.7491` |
| AMASS | newpl_v6_next_control_amass | AMASS pretrain | `2.8093` | `7.2515` | `2.9236` | `32.2878` | `489.2382` |
| DIP test | official PL baseline | fixed baseline | `6.4195` | `12.9477` | `6.5600` | `40.5799` | `2729.2442` |
| DIP test | newpl_v4_init36 baseline | fixed baseline | `6.4414` | `12.7652` | `6.6091` | `40.6118` | `2702.1118` |
| DIP test | newpl_v6_next_control_amass | AMASS pretrain | `6.4806` | `12.7494` | `6.6094` | `66.3748` | `2661.7243` |
| DIP test | newpl_v6_next_control_dip | AMASS -> DIP | `6.4688` | `12.6560` | `6.5954` | `66.3422` | `2658.9849` |
| TotalCapture test | official PL baseline | fixed baseline | `6.9955` | `13.4504` | `7.0954` | `34.3235` | `2173.8883` |
| TotalCapture test | newpl_v4_init36 baseline | fixed baseline | `6.6544` | `13.3295` | `6.7852` | `32.9648` | `1861.3888` |
| TotalCapture test | newpl_v6_next_control_amass | AMASS pretrain | `6.8749` | `13.3279` | `6.9776` | `57.5801` | `706.3869` |
| TotalCapture test | newpl_v6_next_control_dip | AMASS -> DIP | `6.9808` | `13.1385` | `7.0852` | `57.5980` | `707.7693` |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newpl_v6_next_control | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |

### 8. Artifacts

```text
implemented files:
  pl_curve.py
  pl_next_control_cache.py
  pl_next_control_train.py
  pl_next_control_eval.py
  scripts/run_newpl_v6_next_control_20260611.sh
smoke root:
  data/experiments/newpl_v6_next_control_tail4_20260611/smoke
eval JSONs:
  data/experiments/newpl_v6_next_control_tail4_20260611/smoke/eval_amass_after_pretrain.json
  data/experiments/newpl_v6_next_control_tail4_20260611/smoke/eval_dip_test_after_dip_finetune.json
  data/experiments/newpl_v6_next_control_tail4_20260611/smoke/eval_totalcapture_test_after_amass_pretrain.json
  data/experiments/newpl_v6_next_control_tail4_20260611/smoke/eval_totalcapture_test_after_dip_finetune.json
full run output:
  data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/
  data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_amass_after_pretrain.json
  data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_totalcapture_test_after_amass_pretrain.json
  data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_dip_test_after_dip_finetune.json
  data/experiments/newpl_v6_next_control_tail4_20260611/full_fastval1/eval_totalcapture_test_after_dip_finetune.json
```

### 9. Conclusion

- Keep as mainline: no.
- Current pRB/gR1: AMASS pRB is slightly better than official/v4, but DIP test pRB is worse than official/v4 and TotalCapture pRB is much worse than `newpl_v4_init36`.
- Next prediction: AMASS next pRB improves slightly; DIP/TotalCapture next pRB does not beat v4.
- Dynamics: spline acceleration improves strongly, especially on TotalCapture, but velocity L2 is much worse than finite-difference baselines.
- DIP fine-tune: improves DIP gR1 and pRB slightly versus the v6 AMASS checkpoint, but hurts TotalCapture pRB.
- Next use: do not connect to IK1/full pipeline unless a later version preserves current pRB while keeping the gR1/acceleration gains.

## Version: newpl_v6_next_control_smoothacc_gR1

### 1. Purpose

Find whether the best gravity-estimation NewPL can be obtained by combining the corrected v6 next-control branch with centered smoothed acceleration input, while selecting checkpoints directly by gR1/control-point gravity metrics.

### 2. Main Change

| Change Type | Previous v6 | This Version | Motivation |
|---|---|---|---|
| Input | raw official `aM/wM/RMB` converted to 84D PL input | centered smoothed `aM`, raw `wM/RMB`, converted to 84D PL input | reuse the smoothacc evidence that acceleration is more compatible with GT+offset fits |
| Output | current PL 18D plus aux next PL/dynamics | unchanged | preserve IK1 contract |
| Cache | next-control cache from raw PL cache | next-control cache rebuilt from smoothacc PL cache | avoid recomputing feature/target construction inside training |
| Selection | total/current/next/dynamics/control checkpoints | adds `best_current_gR1.pt`, `best_next_gR1.pt`, `best_gravity_control.pt` | target the user's gravity-estimation search directly |

### 3. Loss Design

The loss terms are the same corrected v6 next-control family: current `pRB/gR1`, GT current control, current temporal terms, next `pRB/gR1`, next GT control, next velocity/acceleration, next control delta prior, last-control loss, and tail4-control loss. The new part is checkpoint selection, not a new loss weight family. `best_current_gR1.pt` is selected by current decoded `gR1` angle against GT; `best_next_gR1.pt` by next decoded `gR1`; `best_gravity_control.pt` by current/next/last/tail4 gravity-control terms.

### 4. Training Recipe

| Stage | Data | Input Mode | Init | Epochs | Notes |
|---|---|---|---|---:|---|
| AMASS pretrain | AMASS smoothacc next-control cache | smooth `aM`, raw `wM/RMB` | none | `80` | train seq `1294`, val seq capped at `128`, batch `768` |
| DIP fine-tune | DIP-IMU smoothacc next-control cache | smooth `aM`, raw `wM/RMB` | AMASS `best_current_gR1.pt` | `40` | train seq `36`, val seq `6`, batch `768` |
| TotalCapture | TotalCapture test smoothacc next-control cache | smooth `aM`, raw `wM/RMB` | eval only | `0` | no TC fine-tune |

Evaluation note: AMASS after-pretrain eval is full for this run. Same-window DIP/TotalCapture before/after claims use `max_frames_per_sequence=512`; full-sequence after-AMASS DIP/TC JSONs are retained but not mixed into fast512 conclusions.

### 5. Module GT Metrics

| Dataset / Stage | Version | pRB L2 cm ↓ | gR1 angle deg ↓ | Interpretation |
|---|---|---:|---:|---|
| AMASS after AMASS | official_PL_smoothacc | `4.030455` | `4.838765` | best gR1 |
| AMASS after AMASS | newpl_v4_init36_smoothacc | `4.211169` | `4.876139` | worse pRB than v6 |
| AMASS after AMASS | newpl_v6_smoothacc_amass_current_gR1 | `4.021821` | `5.224865` | best pRB, but worse gR1 |
| DIP fast512 before DIP FT | official_PL_smoothacc | `4.241512` | `8.879523` | fixed baseline |
| DIP fast512 before DIP FT | newpl_v4_init36_smoothacc | `4.221608` | `8.741283` | best pRB among fixed baselines |
| DIP fast512 before DIP FT | newpl_v6_smoothacc_amass_current_gR1 | `4.262976` | `8.792514` | gR1 beats official, pRB loses |
| TC fast512 before DIP FT | official_PL_smoothacc | `7.566914` | `9.873214` | fixed baseline |
| TC fast512 before DIP FT | newpl_v4_init36_smoothacc | `7.160630` | `9.745194` | best pRB |
| TC fast512 before DIP FT | newpl_v6_smoothacc_amass_balanced | `7.470823` | `9.616730` | best gR1, pRB loses to v4 |
| DIP fast512 after DIP FT | official_PL_smoothacc | `4.241512` | `8.879523` | fixed baseline |
| DIP fast512 after DIP FT | newpl_v4_init36_smoothacc | `4.221608` | `8.741283` | stronger pRB than v6 |
| DIP fast512 after DIP FT | newpl_v5_raw_dip_on_smoothinput | `4.190160` | `8.671933` | best overall on DIP fast512 |
| DIP fast512 after DIP FT | newpl_v6_smoothacc_dip_current_gR1 | `4.226809` | `8.719222` | gR1 beats official/v4, loses to raw-v5 |
| TC fast512 after DIP FT | official_PL_smoothacc | `7.566914` | `9.873214` | fixed baseline |
| TC fast512 after DIP FT | newpl_v4_init36_smoothacc | `7.160630` | `9.745194` | best pRB |
| TC fast512 after DIP FT | newpl_v5_raw_dip_on_smoothinput | `7.318521` | `9.890395` | better pRB than v6 |
| TC fast512 after DIP FT | newpl_v6_smoothacc_dip_current_gR1 | `7.562171` | `9.470963` | best gR1, pRB nearly official and far worse than v4 |

Per-leaf after DIP fine-tune, fast512:

| Dataset | Version | leaf_1 | leaf_2 | leaf_3 | leaf_4 | leaf_5 | smooth notes |
|---|---|---:|---:|---:|---:|---:|---|
| DIP | official_PL_smoothacc | `4.570761` | `5.951653` | `3.342375` | `3.540965` | `3.801805` | pRB jitter `0.153908`, gR1 jitter `0.001165` |
| DIP | newpl_v4_init36_smoothacc | `4.632858` | `5.952460` | `3.179092` | `3.580340` | `3.763289` | pRB jitter `0.143387`, gR1 jitter `0.001071` |
| DIP | newpl_v6_smoothacc_dip_current_gR1 | `4.696512` | `6.002673` | `3.343599` | `3.426246` | `3.665018` | pRB jitter `0.143305`, gR1 jitter `0.001027` |
| TC | official_PL_smoothacc | `7.971572` | `7.320425` | `6.911801` | `7.418833` | `8.211942` | pRB jitter `0.721941`, gR1 jitter `0.009477` |
| TC | newpl_v4_init36_smoothacc | `7.831799` | `7.416688` | `6.194060` | `6.959013` | `7.401588` | pRB jitter `0.622985`, gR1 jitter `0.008366` |
| TC | newpl_v6_smoothacc_dip_current_gR1 | `7.992774` | `7.749657` | `6.972277` | `7.565431` | `7.530715` | pRB jitter `0.622620`, gR1 jitter `0.008079` |

### 6. Official S4 11 Metrics

| Version | Score ↓ | Notes |
|---|---:|---|
| newpl_v6_next_control_smoothacc_gR1 | not measured | module-level diagnostic only; full-pipeline 11 metrics intentionally not run |

### 7. Artifacts

```text
implemented/changed:
  pl_next_control_train.py
  pl_next_control_eval.py
  scripts/run_newpl_v6_next_control_smoothacc_gR1_20260613.sh
  scripts/summarize_newpl_v6_next_control_smoothacc_gR1.py
root:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full
log:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/logs/run_full.log
checkpoints:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/amass_pretrain/best_current_gR1.pt
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/dip_finetune/best_current_gR1.pt
summary:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/summary.json
eval JSONs:
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_amass_after_pretrain.json
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_dip_test_after_amass_pretrain_fast512.json
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_totalcapture_test_after_amass_pretrain_fast512.json
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_dip_test_after_dip_finetune.json
  /tmp/globalpose_newpl_v6_next_control_smoothacc_gR1_20260613/full/eval_totalcapture_test_after_dip_finetune.json
```

### 8. Conclusion

- Keep as mainline: no.
- Gravity: this is currently the best checked smoothacc/v6 route for TotalCapture fast512 gR1 and slightly better than official/v4 on DIP fast512 gR1 after DIP fine-tune.
- pRB: fails the PL guard. It loses to `newpl_v4_init36_smoothacc` on DIP and TotalCapture pRB, and loses to raw-v5-on-smoothinput on DIP.
- Next use: keep the gR1-selection code and metrics, but do not connect this checkpoint to IK1/full pipeline unless a later variant preserves pRB.

## Version: newpl_v6_next_p_pdot_pddot_strong

### 1. Purpose

Test the user's requested v6 next-control variant where supervision is applied
to the decoded trajectory outputs, not the control points: `next_pl`,
`next_pldot`, and `next_plddot` pRB[15].

### 2. Main Change

| Change Type | Previous v6 smoothacc | This Version | Motivation |
|---|---|---|---|
| Input/cache | smooth `aM`, raw `wM/RMB`, init36 next-control cache | same; compatible full caches may be reused, smoke builds under this run root | isolate loss/selection change |
| Output | current PL 18D plus aux next PL/dynamics | unchanged | preserve IK1/full-pipeline contract |
| Loss | mixed current/next/control/gR1/priors | only normalized decoded next pRB `p/pd/pdd = 1/1/1` | prevent control-point or gR1 terms from driving selection |
| Selection | current/next/control/gR1 metrics | `best_p_pdot_pddot_strong.pt` by validation normalized `p+pd+pdd` composite | align checkpoint selection with direct supervised outputs |

### 3. Loss Design

| Loss | Weight | Scale Source | Notes |
|---|---:|---|---|
| `next_pRB_norm_pos` | `1.0` | train-cache RMS of `pl_target_next[..., :15]` | decoded `next_pl` pRB only |
| `next_pRB_norm_vel` | `1.0` | train-cache RMS of `gt_pldot_next[..., :15]` | decoded `next_pldot` pRB only |
| `next_pRB_norm_acc` | `1.0` | train-cache RMS of `gt_plddot_next[..., :15]` | decoded `next_plddot` pRB only |
| all old current/control/gR1/prior terms | `0.0` | not applicable | preset `p_pdot_pddot_strong` zeroes them |

Full normalization scales:

| Stage | p scale | pd scale | pdd scale |
|---|---:|---:|---:|
| AMASS pretrain full | `0.3572728132` | `0.3579577642` | `7.3343620957` |
| DIP fine-tune full | `0.3663190813` | `0.4027485024` | `17.5228447399` |

### 4. Training Recipe

| Stage | Data | Epochs | Batch/window | Selection value | Output |
|---|---|---:|---|---:|---|
| AMASS pretrain | 1294 train / 128 val next-control sequences | `80` | batch `32`, window `81` | `0.6011257470` at epoch `70` | `full/amass_pretrain/best_p_pdot_pddot_strong.pt` |
| DIP fine-tune | 36 train / 6 val next-control sequences | `40` | batch `32`, window `81` | `0.9176913500` at epoch `39` | `full/dip_finetune/best_p_pdot_pddot_strong.pt` |

### 5. Full Module Metrics

Existing same-cache full eval after DIP fine-tune. This table is mostly
next-frame for `next p/pd/pdd`; it is not sufficient to claim current-frame
p/pdot/pddot accuracy.

| Dataset / Stage | Version | current pRB L2 cm | current gR1 deg | next p L2 cm | next pd L2 cm/s | next pdd L2 cm/s2 |
|---|---|---:|---:|---:|---:|---:|
| DIP after DIP FT | official PL smoothacc | `6.345701` | `12.902106` | `6.465117` | `40.244087` | `2684.142116` |
| DIP after DIP FT | newpl_v4_init36_smoothacc | `6.349541` | `12.722353` | `6.496246` | `40.243882` | `2666.394318` |
| DIP after DIP FT | newpl_v5_raw_dip_on_smoothinput | `6.357881` | `12.512222` | `6.507716` | `40.250549` | `2666.365777` |
| DIP after DIP FT | prior newpl_v6_raw_dip_on_smoothinput | `6.370279` | `12.615336` | `6.478526` | `66.331095` | `2660.687140` |
| DIP after DIP FT | strong best_p_pdot_pddot | `6.353314` | `12.901381` | `6.454685` | `64.602965` | `2684.758215` |
| DIP after DIP FT | strong last | `6.353314` | `12.901381` | `6.454691` | `64.593160` | `2685.167951` |
| TC after DIP FT | official PL smoothacc | `7.508986` | `13.170870` | `7.597959` | `34.038251` | `2099.110771` |
| TC after DIP FT | newpl_v4_init36_smoothacc | `7.119541` | `13.075061` | `7.236774` | `32.694513` | `1801.389069` |
| TC after DIP FT | newpl_v5_raw_dip_on_smoothinput | `7.255848` | `13.138471` | `7.387153` | `32.724789` | `1800.290009` |
| TC after DIP FT | prior newpl_v6_raw_dip_on_smoothinput | `7.484898` | `12.954138` | `7.578813` | `57.576946` | `711.415844` |
| TC after DIP FT | strong best_p_pdot_pddot | `7.508233` | `13.168667` | `7.589151` | `55.491009` | `806.639374` |
| TC after DIP FT | strong last | `7.508233` | `13.168667` | `7.589482` | `55.475717` | `807.941780` |

### 5.1 Current-frame p/pdot/pddot Eval

This added eval answers the intended current-frame question directly.

```text
current output:
  output["pl"] = pRB_t[15] + gR1_t[3]
next output:
  output["next_pl"] = predicted pRB_{t+1}[15] + gR1_{t+1}[3]
next derivatives:
  output["next_pldot"], output["next_plddot"] are decoded from predicted next control via spline.
current selection warning:
  best_p_pdot_pddot_strong.pt was selected by validation normalized next p/pdot/pddot composite.
  It does not by itself prove current-frame p/pdot/pddot accuracy.
```

Masks and units:

| Metric | Prediction | GT | Mask | Unit |
|---|---|---|---|---|
| current p | `output["pl"][..., :15]` | `pl_target[..., :15]` | all current frames | L1/L2 cm |
| current pdot | central FD of `output["pl"][..., :15]` | `gt_pldot[..., :15]` | exclude first/last frames | L1/L2 cm/s |
| current pddot | central FD acceleration of `output["pl"][..., :15]` | `gt_plddot[..., :15]` | exclude first/last frames | L1/L2 cm/s^2 |
| next p/pdot/pddot | `output["next_pl"]`, `output["next_pldot"]`, `output["next_plddot"]` | `pl_target_next`, `gt_pldot_next`, `gt_plddot_next` | `valid_next_mask` | cm, cm/s, cm/s^2 |

Current-frame full eval:

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

Alignment sweep result: all full DIP and TotalCapture current p/pdot/pddot
best shifts are `0`; no time-shift warning is triggered in the full eval.

### 5.2 Velocity / Acceleration Metric Audit

This audit checks whether the large velocity error is a model issue or an eval
artifact from current/next mixing, dt/unit mismatch, masks, alignment, or
derivative target definitions.

Metric definitions:

| Metric | Prediction | GT | Mask | Unit |
|---|---|---|---|---|
| `current_fd_velocity` | central FD of `output["pl"][..., :15]` | `gt_pldot[..., :15]` | exclude first/last frames | L2 cm/s |
| `current_fd_acceleration` | central FD acceleration of `output["pl"][..., :15]` | `gt_plddot[..., :15]` | exclude first/last frames | L2 cm/s^2 |
| `next_head_velocity` | `output["next_pldot"][..., :15]` | `gt_pldot_next[..., :15]` | `valid_next_mask` | L2 cm/s |
| `next_head_acceleration` | `output["next_plddot"][..., :15]` | `gt_plddot_next[..., :15]` | `valid_next_mask` | L2 cm/s^2 |
| `next_position_fd_velocity` | central FD of `output["next_pl"][..., :15]` | `gt_pldot_next[..., :15]` | `valid_next_mask` minus FD boundaries | L2 cm/s |
| `next_position_fd_acceleration` | central FD acceleration of `output["next_pl"][..., :15]` | `gt_plddot_next[..., :15]` | `valid_next_mask` minus FD boundaries | L2 cm/s^2 |

Mask counts:

| Dataset | sequences | frames | valid_next_frames | current_derivative_valid_frames | excluded_boundary_frames |
|---|---:|---:|---:|---:|---:|
| DIP test | `19` | `57994` | `57975` | `57956` | `38` |
| TotalCapture test | `4` | `16124` | `16120` | `16116` | `8` |

GT and dt audit:

| Dataset | decoded dot L2 | decoded ddot L2 | FD GT vel L2 at dt=1/60 | FD GT acc L2 at dt=1/60 | FD GT vel L2 at dt=1 | actual dt mismatch |
|---|---:|---:|---:|---:|---:|---|
| DIP test | `0.000000` | `0.000000` | `0.083706` | `0.943413` | `53.769033` | no |
| TotalCapture test | `0.000000` | `0.000000` | `0.367573` | `4.401025` | `51.970507` | no |

Velocity/acceleration split:

| Dataset | Version | current FD vel | current FD acc | next-head vel | next-head acc | next-position FD vel | next-position FD acc |
|---|---|---:|---:|---:|---:|---:|---:|
| DIP test | official PL smoothacc | `31.449577` | `1807.055855` | `32.361504` | `1810.301938` | `32.338752` | `1808.462783` |
| DIP test | prior newpl_v6_raw_dip_on_smoothinput | `31.425229` | `1791.925571` | `54.595607` | `1782.213774` | `32.341935` | `1808.370255` |
| DIP test | strong best_p_pdot_pddot | `31.419971` | `1792.008316` | `53.040942` | `1813.554982` | `32.216370` | `1807.979955` |
| TotalCapture test | official PL smoothacc | `30.784883` | `1882.019351` | `31.635313` | `1899.639228` | `31.559175` | `1895.471466` |
| TotalCapture test | prior newpl_v6_raw_dip_on_smoothinput | `29.585914` | `1616.539413` | `52.591712` | `650.197321` | `31.573507` | `1894.099732` |
| TotalCapture test | strong best_p_pdot_pddot | `29.581992` | `1618.041596` | `50.714015` | `742.650349` | `31.785827` | `1931.831008` |

Audit classification:

| Cause | Result | Evidence |
|---|---|---|
| A. real current-frame model issue | partial | current FD velocity/acc are comparable to same-cache baselines, but p/pddot gate still failed in the current-frame eval above |
| B. current/next or temporal/source mismatch | yes for next-head derivatives | current FD best shift is `0`; next-head velocity best shift is nonzero, including `-2` for strong best on DIP/TC |
| C. dt/unit mismatch | no actual mismatch | `dt=1` would be wrong, but manifest and eval dt are both `1/60` |
| D. derivative target definition mismatch | no | decoded control dot/ddot vs cache GT are exactly `0.000000` L2 |
| E. boundary/mask issue | no | derivative metrics exclude first/last frames and report separate current/next frame counts |

Conclusion: velocity error should not be treated as a single pooled number.
The current-frame finite-difference velocity metric is aligned and uses the
correct dt/mask. The large velocity anomaly is mainly in `next_pldot`/next-head
derivatives, where the best temporal alignment is nonzero. This remains
diagnostic only and does not justify promotion or network-structure changes.

### 6. Official S4 11 Metrics

| Version | Score | Notes |
|---|---:|---|
| newpl_v6_next_p_pdot_pddot_strong | not measured | module-level full AMASS->DIP run only; no full-pipeline 11 metrics because module metrics do not justify escalation |

### 7. Artifacts

```text
changed:
  pl_next_control_train.py
  scripts/run_newpl_v6_next_control_smoothacc_gR1_20260613.sh
added:
  scripts/run_newpl_v6_next_p_pdot_pddot_strong_20260615.sh
  scripts/summarize_newpl_v6_next_p_pdot_pddot_strong.py
smoke root:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke
smoke summary:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/smoke/summary_p_pdot_pddot_strong.json
full root:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full
full summary:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/summary_p_pdot_pddot_strong.json
full checkpoints:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/amass_pretrain/best_p_pdot_pddot_strong.pt
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/dip_finetune/best_p_pdot_pddot_strong.pt
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/dip_finetune/last.pt
current-frame eval:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/eval_current_p_pdot_pddot_dip.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/eval_current_p_pdot_pddot_totalcapture.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/summary_current_p_pdot_pddot.md
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/per_sequence_current_p_pdot_pddot.csv
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/per_leaf_current_p_pdot_pddot.csv
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/current_p_pdot_pddot_eval/alignment_sweep.csv
velocity metric audit:
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit/velocity_metric_audit.json
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit/velocity_metric_audit.md
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit/velocity_alignment_sweep.csv
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit/velocity_per_leaf.csv
  data/experiments/newpl_v6_next_p_pdot_pddot_strong_20260615/full/velocity_metric_audit/velocity_per_sequence.csv
```

### 8. Conclusion

- Keep as mainline: no.
- The intended loss contract works and training is finite, but the selected
  checkpoint mostly learns a next-frame trajectory/dynamics tradeoff rather than
  better current-frame p/pdot/pddot.
- DIP current-frame gate: strong best is worse than the best baseline on current
  p (`6.465181` vs `6.451342`) and current pddot (`1792.008308` vs `1791.925604`);
  current pdot is marginally best.
- TotalCapture current-frame gate: strong best is worse than the best baseline
  on current p (`7.253737` vs `6.879507`) and current pddot (`1618.041598` vs
  `1616.539405`); current pdot is marginally best.
- Full alignment sweep has best shift `0` for current p/pdot/pddot on both DIP
  and TotalCapture, so the failure is not explained by a global frame offset.
- Do not run full-pipeline 11 metrics or promote this checkpoint unless a later
  variant passes current-frame p/pdot/pddot non-regression while keeping any
  useful next-frame acceleration gain.

## 6. IK-s1 Replacement Versions

Official IK1 contract: input `RRB_after_pl[45] + gR1[3] + pRB[15] = 63D`; output `pRJ[69] + gR2[3] = 72D`. Control-tail variants use expanded features; last-control returns to 63D by using the final PL control point.

## Version: newik1_v1_control_tail

### 1. Purpose

Replace official IK1 with control-tail NewIK1 trained on PL streaming outputs.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | official IK1 63D | control-tail feature around PL controls | reduce teacher-forcing mismatch |
| Output | 72D | unchanged | preserve IK2/VR |
| Loss | official historical | pRJ/gR2/control temporal losses | train state/control prediction |
| Training | none | PL streaming TC finetune | adapt to upstream PL |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | varies | official-shape 63D or control-tail 120D, specified per version | after PL output |
| Output | 72D | `pRJ[69]+gR2[3]` | official IK1 downstream contract |

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRJ` | `varies` | `0.0008581769943702966` |
| `gR2` | `1.0` | `0.16139879822731018` |
| `pRJ_dot` | `0.03/0.05` | `1.6324850707860606e-05` |
| `pRJ_ddot` | `0.001/0.002` | `2.3687867133048714e-05` |
| `gR2_dot` | `0.03` | `6.529722668346949e-05` |
| `gR2_ddot` | `0.001` | `6.667937266229274e-05` |
| `bone_length` | `0/0.5` | `not measured` |
| `control_pRJ` | `0.1/0.3` | `0.0008616707404144108` |
| `control_gR2` | `0.1` | `0.1612615093588829` |
| `total` | `mixed` | `0.17848628610372544` |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| TC finetune | TotalCapture train | PL Run D streaming cache | AMASS-adapted checkpoint | 10 | best_loss.pt |

### 6. Module GT Delta

| Version | pRJ ↓ | gR2 ↓ | pRJ_dot ↓ | pRJ_ddot ↓ | gR2_dot ↓ | gR2_ddot ↓ | Total Loss Delta vs Previous ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| newik1_v1_control_tail | `0.0008581769943702966` | `0.16139879822731018` | `1.6324850707860606e-05` | `2.3687867133048714e-05` | `6.529722668346949e-05` | `6.667937266229274e-05` | `-0.0100589` |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newik1_v1_control_tail | `39.378624` | `10.306240` | `8.953209` | `4.601567` | `5.221055` | `10.463551` | `8.721832` | `4.688967` | `5.262477` | `0.284094` | `0.473890` |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newpl_v3_gtcontrol_rund | `0.683779` | `-0.010059` | Local loss converges but full S4 is much worse than PL-only. |

### 9. Conclusion

- Module GT: see Module GT Delta table
- Official S4: `39.378624`
- Keep as mainline: no
- Reason: Local loss converges but full S4 is much worse than PL-only.

## Version: newik1_v2_bonelength

### 1. Purpose

Add bone-length consistency to IK1 output geometry.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | control-tail | unchanged | isolate geometry loss |
| Output | 72D | unchanged | preserve downstream |
| Loss | v1 losses | add bone_length=0.5 | stabilize pRJ geometry |
| Training | v1 checkpoint | continue 10 epochs | test local improvement |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | varies | official-shape 63D or control-tail 120D, specified per version | after PL output |
| Output | 72D | `pRJ[69]+gR2[3]` | official IK1 downstream contract |

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRJ` | `varies` | `0.0008572387800086289` |
| `gR2` | `1.0` | `0.16118536964058877` |
| `pRJ_dot` | `0.03/0.05` | `1.6324809098478e-05` |
| `pRJ_ddot` | `0.001/0.002` | `2.3687857503773557e-05` |
| `gR2_dot` | `0.03` | `6.524095770146232e-05` |
| `gR2_ddot` | `0.001` | `6.667143768481764e-05` |
| `bone_length` | `0/0.5` | `0.00013660868571605534` |
| `control_pRJ` | `0.1/0.3` | `0.0008606041956227273` |
| `control_gR2` | `0.1` | `0.1609759509563446` |
| `total` | `mixed` | `0.1783127911388874` |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| TC finetune | TotalCapture train | PL Run D streaming cache | newik1_v1 | 10 | best_loss.pt |

### 6. Module GT Delta

| Version | pRJ ↓ | gR2 ↓ | pRJ_dot ↓ | pRJ_ddot ↓ | gR2_dot ↓ | gR2_ddot ↓ | Total Loss Delta vs Previous ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| newik1_v2_bonelength | `0.0008572387800086289` | `0.16118536964058877` | `1.6324809098478e-05` | `2.3687857503773557e-05` | `6.524095770146232e-05` | `6.667143768481764e-05` | `-0.000173495` |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newik1_v2_bonelength | `39.394920` | `10.310898` | `8.957259` | `4.603743` | `5.223369` | `10.466181` | `8.726370` | `4.691000` | `5.265961` | `0.284059` | `0.473769` |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newik1_v1_control_tail | `0.016295` | `-0.000173` | Slight local loss improvement, S4 worse than v1 and PL-only. |

### 9. Conclusion

- Module GT: see Module GT Delta table
- Official S4: `39.394920`
- Keep as mainline: no
- Reason: Slight local loss improvement, S4 worse than v1 and PL-only.

## Version: newik1_v3_strong_pRJ_control

### 1. Purpose

Increase pRJ and control_pRJ supervision.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | control-tail | unchanged | isolate loss strength |
| Output | 72D | unchanged | preserve downstream |
| Loss | pRJ=1, control_pRJ=0.1 | pRJ=2, control_pRJ=0.3 | emphasize Cartesian IK state |
| Training | v2 checkpoint | continue 10 epochs | test stronger loss |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | varies | official-shape 63D or control-tail 120D, specified per version | after PL output |
| Output | 72D | `pRJ[69]+gR2[3]` | official IK1 downstream contract |

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRJ` | `varies` | `0.0008561596681829542` |
| `gR2` | `1.0` | `0.16105071380734443` |
| `pRJ_dot` | `0.03/0.05` | `1.6324767375408556e-05` |
| `pRJ_ddot` | `0.001/0.002` | `2.368784554391823e-05` |
| `gR2_dot` | `0.03` | `6.520473489217693e-05` |
| `gR2_ddot` | `0.001` | `6.666635310921265e-05` |
| `bone_length` | `0/0.5` | `0.0001362629613140598` |
| `control_pRJ` | `0.1/0.3` | `0.0008591262041591108` |
| `control_gR2` | `0.1` | `0.16079552993178367` |
| `total` | `mixed` | `0.17918791025876998` |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| TC finetune | TotalCapture train | PL Run D streaming cache | newik1_v2 | 10 | best_loss.pt |

### 6. Module GT Delta

| Version | pRJ ↓ | gR2 ↓ | pRJ_dot ↓ | pRJ_ddot ↓ | gR2_dot ↓ | gR2_ddot ↓ | Total Loss Delta vs Previous ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| newik1_v3_strong_pRJ_control | `0.0008561596681829542` | `0.16105071380734443` | `1.6324767375408556e-05` | `2.368784554391823e-05` | `6.520473489217693e-05` | `6.666635310921265e-05` | `0.000875119` |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newik1_v3_strong_pRJ_control | `38.948675` | `10.252610` | `8.808721` | `4.553129` | `5.180627` | `10.362026` | `8.620636` | `4.447495` | `5.005212` | `0.278308` | `0.461912` |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newik1_v2_bonelength | `-0.446245` | `0.000875` | Local total worsens but S4 improves relative to v1/v2; still worse than PL-only. |

### 9. Conclusion

- Module GT: see Module GT Delta table
- Official S4: `38.948675`
- Keep as mainline: no
- Reason: Local total worsens but S4 improves relative to v1/v2; still worse than PL-only.

## Version: newik1_v4_official_input

### 1. Purpose

Use official-shape IK1 input and full streaming eval.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | control-tail 120D | official-shape 63D | match GPNet contract |
| Output | 72D | unchanged | preserve downstream |
| Loss | control-tail losses | pRJ/gR2 + distill pRJ | train official-shape module |
| Training | control-tail path | official input PL streaming finetune | reduce interface mismatch |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | varies | official-shape 63D or control-tail 120D, specified per version | after PL output |
| Output | 72D | `pRJ[69]+gR2[3]` | official IK1 downstream contract |

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRJ` | `varies` | `0.002114904811605811` |
| `gR2` | `1.0` | `0.1732551373541355` |
| `pRJ_dot` | `0.03/0.05` | `1.834258318922366e-05` |
| `pRJ_ddot` | `0.001/0.002` | `2.5820879136517762e-05` |
| `gR2_dot` | `0.03` | `6.748882951796986e-05` |
| `gR2_ddot` | `0.001` | `6.897390539961634e-05` |
| `bone_length` | `0/0.5` | `0.0006415344832930714` |
| `control_pRJ` | `0.1/0.3` | `not measured` |
| `control_gR2` | `0.1` | `not measured` |
| `total` | `mixed` | `0.1778486765921116` |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| TC finetune | TotalCapture train | PL Run D streaming official-shape cache | teacher-forced checkpoint | 10 | best_loss.pt |

### 6. Module GT Delta

| Version | pRJ ↓ | gR2 ↓ | pRJ_dot ↓ | pRJ_ddot ↓ | gR2_dot ↓ | gR2_ddot ↓ | Total Loss Delta vs Previous ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| newik1_v4_official_input | `0.002114904811605811` | `0.1732551373541355` | `1.834258318922366e-05` | `2.5820879136517762e-05` | `6.748882951796986e-05` | `6.897390539961634e-05` | `-0.00133923` |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newik1_v4_official_input | `38.705231` | `10.094597` | `8.814167` | `4.550563` | `5.196077` | `10.335713` | `8.560990` | `4.398241` | `4.911769` | `0.296159` | `0.488408` |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newik1_v3_strong_pRJ_control | `-0.243445` | `-0.001339` | Beats processed official baseline but not Run D PL-only; not selected. |

### 9. Conclusion

- Module GT: see Module GT Delta table
- Official S4: `38.705231`
- Keep as mainline: no
- Reason: Beats processed official baseline but not Run D PL-only; not selected.

## Version: newik1_v5_last_pl_control

### 1. Purpose

Use final PL control point as IK1 input with NewPL init36 upstream.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | official/current control variants | `RRB_after_pl[45]+last_control_pRB[15]+last_control_gR1[3]=63D` | use cleaner upstream state |
| Output | 72D | unchanged | preserve downstream |
| Loss | NewIK1 control loss family | same family with last-control feature | isolate input change |
| Training | single TC finetune | GT pretrain -> PL AMASS adapt -> PL TC finetune | reduce mismatch |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | varies | official-shape 63D or control-tail 120D, specified per version | after PL output |
| Output | 72D | `pRJ[69]+gR2[3]` | official IK1 downstream contract |

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRJ` | `varies` | `0.0008461097080726176` |
| `gR2` | `1.0` | `0.16928540915250778` |
| `pRJ_dot` | `0.03/0.05` | `1.6320449640261357e-05` |
| `pRJ_ddot` | `0.001/0.002` | `2.3684303312165866e-05` |
| `gR2_dot` | `0.03` | `6.732392575941049e-05` |
| `gR2_ddot` | `0.001` | `6.696426223697926e-05` |
| `bone_length` | `0/0.5` | `0.00013875475924578494` |
| `control_pRJ` | `0.1/0.3` | `0.0008498783106915652` |
| `control_gR2` | `0.1` | `0.16925255656242372` |
| `total` | `mixed` | `0.18721885234117508` |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| Stage A | AMASS | GT/teacher-forced | none | 30 | stage_a/best_loss.pt |
| Stage B | AMASS | NewPL init36 streaming | stage_a | 20 | stage_b/best_loss.pt |
| Stage C | TotalCapture | NewPL init36 streaming | stage_b | 15 | stage_c/best_loss.pt |

### 6. Module GT Delta

| Version | pRJ ↓ | gR2 ↓ | pRJ_dot ↓ | pRJ_ddot ↓ | gR2_dot ↓ | gR2_ddot ↓ | Total Loss Delta vs Previous ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| newik1_v5_last_pl_control | `0.0008461097080726176` | `0.16928540915250778` | `1.6320449640261357e-05` | `2.3684303312165866e-05` | `6.732392575941049e-05` | `6.696426223697926e-05` | `0.00937018` |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newik1_v5_last_pl_control | `38.843577` | `10.184999` | `8.797480` | `4.517464` | `5.162325` | `10.369052` | `8.599769` | `4.358955` | `4.932460` | `0.277014` | `0.463422` |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newik1_v4_official_input | `0.138346` | `0.009370` | pRJ local improves but gR2/S4 do not; worse than NewPL init36, not selected. |

### 9. Conclusion

- Module GT: see Module GT Delta table
- Official S4: `38.843577`
- Keep as mainline: no
- Reason: pRJ local improves but gR2/S4 do not; worse than NewPL init36, not selected.

## Version: newik1_v6_official_input_init36_cascade

### 1. Purpose

Fine-tune `newik1_v4_official_input` with the v5-style cascade while preserving the official IK1 input/output contract and using NewPL init36 as upstream.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | `newik1_v4_official_input` official-shape 63D on Run D/PL streaming | official-shape 63D with NewPL init36 upstream | keep GPNet IK1 contract while using current best PL |
| Output | `pRJ[69]+gR2[3]=72D` | unchanged | preserve IK2/VR/physics compatibility |
| Loss | official-input pRJ/gR2/bone/temporal/distill loss | unchanged | isolate staged cascade fine-tuning |
| Training | v4 single PL streaming TC finetune | v4 checkpoint -> teacher-forced AMASS refresh -> NewPL init36 streaming AMASS adapt -> NewPL init36 streaming TC finetune, each with continuation | test whether v5-style staged adaptation helps official-shape IK1 |
| Initialization | v4 checkpoint | `data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt` | start from best official-input IK1 |
| Evaluation | v4 full S4 only | every stage has Module GT diagnostic and full S4 11 metrics | measure both local IK1 quality and downstream S4 |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | 63D | `RRB_after_pl[45]+gR1[3]+pRB[15]` | official IK1 features after NewPL init36 output |
| Output | 72D | `pRJ[69]+gR2[3]` | official IK1 downstream contract |

### 4. Loss Design

| Loss | Weight | Validation Value |
|---|---:|---:|
| `pRJ` | `2.0` | stage A `0.000137963`; stage B `0.001733345`; stage C `0.001730118` |
| `gR2` | `1.0` | stage A `0.000047145`; stage B `0.135585785`; stage C `0.134571475` |
| `bone_length` | `0.5` | stage A `0.000036040`; stage B `0.000133265`; stage C `0.000133539` |
| `pRJ_dot` | `0.05` | stage A `0.000003693`; stage B `0.000014308`; stage C `0.000014308` |
| `gR2_dot` | `0.03` | stage A `0.000000391`; stage B `0.000038046`; stage C `0.000038102` |
| `pRJ_ddot` | `0.002` | stage A `0.000004368`; stage B `0.000020995`; stage C `0.000020989` |
| `gR2_ddot` | `0.001` | stage A `0.000000747`; stage B `0.000019829`; stage C `0.000019819` |
| `ik1_distill_pRJ` | `0.2` | stage A `0.000456925`; stage B `0.000063493`; stage C `0.000062118` |
| `ik1_distill_gR2` | `0.0` | measured but unweighted |
| `total` | mixed | stage A `0.000430632`; stage B `0.139396750`; stage C `0.138375943` |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| Stage A refresh | AMASS train, TC val | teacher-forced official IK1 cache | `newik1_v4_official_input` best | 12 | `stage_a_gt_refresh/best_loss.pt` |
| Stage A continue | AMASS train, TC val | teacher-forced official IK1 cache | Stage A refresh best | 6 | `stage_a_continue/best_loss.pt` |
| Stage B adapt | AMASS train, TC val | NewPL init36 streaming AMASS train, NewPL init36 streaming TC val | Stage A continue best | 16 | `stage_b_pl_amass_adapt/best_loss.pt` |
| Stage B continue | AMASS train, TC val | NewPL init36 streaming AMASS train, NewPL init36 streaming TC val | Stage B adapt best | 8 | `stage_b_continue/best_loss.pt` |
| Stage C finetune | TC train, TC val | NewPL init36 streaming TC train/val | Stage B continue best | 12 | `stage_c_tc_finetune/best_loss.pt` |
| Stage C continue | TC train, TC val | NewPL init36 streaming TC train/val | Stage C finetune best | 8 | `stage_c_continue/best_loss.pt` / `last.pt` |

AMASS streaming required `offset_r`; the original AMASS L4 baseline cache was enriched from original `globalpose_synth_shardXXXXX.pt` offsets before rerun.

### 6. Module GT Delta

Metric contract: compare NewIK1 output against `ik1_target` GT on the same cache as the official IK1 baseline (`ik1_base`). Negative delta means NewIK1 is closer to GT.

Important: `stage_a_teacher_forced` uses a teacher-forced / GT-like TC val cache, not PL streaming input. The PL-streaming comparison uses `pl_streaming_tc_val` and is the correct answer for "using PL as input".

| Stage / Cache | pRJ L1 cm Δ ↓ | pRJ L2 cm Δ ↓ | gR2 angle deg Δ ↓ | pRJ_dot cm L2 Δ ↓ | pRJ_ddot cm L2 Δ ↓ | gR2_dot L2 Δ ↓ | gR2_ddot L2 Δ ↓ | State L2 Δ ↓ | Better than baseline? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stage_a_teacher_forced | `-1.041487` | `-2.101508` | `-1.757093` | `-0.037492` | `-0.013573` | `-0.0000647` | `0.0001154` | `-0.021073` | yes, but teacher-forced only |
| stage_a_on_pl_streaming | `0.040988` | `0.011086` | `-0.044246` | `-0.005815` | `-0.017749` | `0.0000621` | `0.0000874` | `-0.0000278` | essentially tied; pRJ worse, gR2 slightly better |
| stage_b_pl_streaming | `0.025949` | `-0.450596` | `-1.624033` | `-0.000284` | `-0.023345` | `-0.0005511` | `0.0000097` | `-0.005932` | yes overall; pRJ L1 slightly worse |
| stage_c_pl_streaming | `0.008542` | `-0.470448` | `-1.715858` | `0.0000025` | `-0.023236` | `-0.0005710` | `0.0000071` | `-0.006220` | yes overall; pRJ L1 slightly worse |

### 7. Official S4 11 Metrics

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newik1_v6_stage_a | `38.649137` | `10.066027` | `8.796304` | `4.523891` | `5.174597` | `10.318648` | `8.571874` | `4.389591` | `4.904521` | `0.300628` | `0.493580` |
| newik1_v6_stage_b | `41.534289` | `10.964134` | `9.126915` | `4.873451` | `5.439083` | `11.366053` | `9.083804` | `5.011192` | `5.462034` | `0.292218` | `0.491840` |
| newik1_v6_stage_c_best | `41.543204` | `10.977999` | `9.124745` | `4.878132` | `5.443132` | `11.377183` | `9.069246` | `5.013017` | `5.463964` | `0.291755` | `0.491419` |
| newik1_v6_stage_c_last | `41.543204` | `10.977999` | `9.124745` | `4.878132` | `5.443132` | `11.377183` | `9.069246` | `5.013017` | `5.463964` | `0.291755` | `0.491419` |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newpl_v4_init36 | best `+0.023479` | state L2 better than official IK1 baseline | Stage A is close but still worse than PL-only NewPL init36. |
| newik1_v4_official_input | best `-0.056094` | pRJ L2 and gR2 angle improve against current-cache official IK1 baseline | v6 improves S4 over v4 but still not enough to become mainline. |
| stage_a -> stage_b | `+2.885152` | state L2 remains better but less so | PL streaming AMASS adaptation hurts downstream S4 badly. |

### 9. Conclusion

- Module GT: teacher-forced Stage A improves both pRJ and gR2, but that is not the PL-streaming deployment input. Under PL-streaming input, Stage A is essentially tied and pRJ is slightly worse while gR2 is only slightly better; Stage B/C improve pRJ L2 and gR2 angle, but pRJ L1 remains slightly worse. Do not claim every pRJ coordinate/node-position metric improves.
- Official S4: best is stage A `38.649136830300094`, worse than NewPL init36 `38.625657482802865`.
- Keep as mainline: no.
- Reason: IK1 local output is closer to GT, but downstream S4 does not beat PL-only; PL streaming AMASS/TC continuation degrades S4 to about `41.54`.
- Next step: diagnose why locally better IK1 outputs harm IK2/VR/physics, especially distribution/scale of `pRJ` and `gR2` under downstream IK2 input.

## Version: newik1_v8_parallel_adaptive_loss_search

### 1. Purpose

Continue from the v7 last-control IK1 checkpoint and run a small parallel loss-ratio search to identify whether changing pRJ/gR2 emphasis or dynamic loss weights improves either full S4 or IK1 module-output-vs-GT under NewPL init36 streaming input.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | `newik1_v7_last_pl_control_lightloss_amass` last-control 63D input | unchanged `RRB_after_pl[45] + last_control_pRB[15] + last_control_gR1[3] = 63D` | isolate loss-ratio and short fine-tune effects |
| Output | `pRJ[69] + gR2[3] = 72D` | unchanged | preserve official IK1 downstream contract |
| Loss | v7 light-loss recipe | eight variants changing `pRJ`, `gR2`, `pRJ_dot/ddot`, `gR2_dot/ddot`; bone/prior/distill remain zero | find which local terms give positive S4 or module-GT effect |
| Training | v7 AMASS light-loss checkpoint | 8 parallel 5-epoch TotalCapture micro-finetunes from v7 best | quick adaptive search before longer continuation |
| Initialization | v7 Stage A light-loss AMASS best | `data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt` | start from the preferred v5/v7 last-control direction |
| Evaluation | v7 best/last S4 and module-GT | every trial evaluates `best_loss.pt` and `last.pt` with Module GT and full S4 | select by both module-output-vs-GT and official streaming S4 |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | 63D | `RRB_after_pl[45] + last_control_pRB[15] + last_control_gR1[3]` | NewPL init36 streaming control cache |
| Output | 72D | `pRJ[69] + gR2[3]` | official IK1 downstream contract for IK2/VR |

### 4. Loss Design

Best v8 variant: `v8_B4_pRJ_x2_lowdyn`, selected only within the v8 sweep, not as project mainline.

| Loss | Weight | Purpose | New / Changed / Unchanged |
|---|---:|---|---|
| `pRJ` | `2.0` | current-frame IK1 joint state | changed from base sweep value |
| `gR2` | `1.0` | current-frame gravity/root direction | unchanged |
| `pRJ_dot` | `0.01` | pRJ first derivative | decreased |
| `pRJ_ddot` | `0.0003` | pRJ second derivative | decreased |
| `gR2_dot` | `0.03` | gR2 first derivative | unchanged |
| `gR2_ddot` | `0.001` | gR2 second derivative | unchanged |
| `control_pRJ` | `0.1` | control-point pRJ alignment | unchanged |
| `control_gR2` | `0.1` | control-point gR2 alignment | unchanged |
| `control_pRJ_dot` | `0.003` | control pRJ first derivative | unchanged |
| `control_gR2_dot` | `0.003` | control gR2 first derivative | unchanged |
| `control_pRJ_ddot` | `0.0001` | control pRJ second derivative | unchanged |
| `control_gR2_ddot` | `0.0001` | control gR2 second derivative | unchanged |
| `bone_length` | `0.0` | geometry prior | disabled |
| `control_point_prior` | `0.0` | residual/control prior | disabled |
| `tail_update_prior` | `0.0` | tail update regularizer | disabled |
| `gt_control_pRJ` | `0.0` | GT control supervision | disabled |
| `gt_control_gR2` | `0.0` | GT control supervision | disabled |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| A1 | TotalCapture train/val | NewPL init36 PL-streaming TC cache | v7 Stage A AMASS best | 5 | `v8_A1_gR2_x2/train/best_loss.pt`, `last.pt` |
| A2 | TotalCapture train/val | NewPL init36 PL-streaming TC cache | v7 Stage A AMASS best | 5 | `v8_A2_gR2_x4/train/best_loss.pt`, `last.pt` |
| A3 | TotalCapture train/val | NewPL init36 PL-streaming TC cache | v7 Stage A AMASS best | 5 | `v8_A3_gR2_half/train/best_loss.pt`, `last.pt` |
| A4 | TotalCapture train/val | NewPL init36 PL-streaming TC cache | v7 Stage A AMASS best | 5 | `v8_A4_gR2_x2_lowdyn/train/best_loss.pt`, `last.pt` |
| B1 | TotalCapture train/val | NewPL init36 PL-streaming TC cache | v7 Stage A AMASS best | 5 | `v8_B1_pRJ_x2/train/best_loss.pt`, `last.pt` |
| B2 | TotalCapture train/val | NewPL init36 PL-streaming TC cache | v7 Stage A AMASS best | 5 | `v8_B2_pRJ_x4/train/best_loss.pt`, `last.pt` |
| B3 | TotalCapture train/val | NewPL init36 PL-streaming TC cache | v7 Stage A AMASS best | 5 | `v8_B3_pRJ_half/train/best_loss.pt`, `last.pt` |
| B4 | TotalCapture train/val | NewPL init36 PL-streaming TC cache | v7 Stage A AMASS best | 5 | `v8_B4_pRJ_x2_lowdyn/train/best_loss.pt`, `last.pt` |

### 6. Module GT Delta

Metric contract: Module GT uses NewPL init36 PL-streaming TotalCapture validation input, compared against the official IK1 baseline on the same cache. Negative delta means NewIK1 is closer to GT.

| Version | pRJ L2 ↓ | gR2 angle deg ↓ | State L2 ↓ | pRJ L2 Delta ↓ | gR2 Angle Delta ↓ | State L2 Delta ↓ | Better than official IK1 baseline? |
|---|---:|---:|---:|---:|---:|---:|---|
| v8_B4_pRJ_x2_lowdyn last | `0.039556610492` | `25.579577928497` | `0.071334521093` | `0.000142750957` | `-0.005103604411` | `0.000073719792` | no |
| v8_A1_gR2_x2 last | `0.039556802543` | `25.579603020934` | `0.071334653073` | `0.000142943008` | `-0.005078511974` | `0.000073851772` | no |
| v8_B2_pRJ_x4 last | `0.039556628865` | `25.579573674366` | `0.071334553607` | `0.000142769330` | `-0.005107858542` | `0.000073752305` | no |
| v8_A2_gR2_x4 last | `0.039556728542` | `25.579615338577` | `0.071334638926` | `0.000142869007` | `-0.005066194331` | `0.000073837624` | no |
| v8_A4_gR2_x2_lowdyn last | `0.039556669752` | `25.579587571333` | `0.071334593424` | `0.000142810217` | `-0.005093961575` | `0.000073792122` | no |
| v8_B3_pRJ_half last | `0.039556806878` | `25.579605945303` | `0.071334674266` | `0.000142947343` | `-0.005075587605` | `0.000073872965` | no |
| v8_B1_pRJ_x2 last | `0.039556737731` | `25.579596263153` | `0.071334630816` | `0.000142878196` | `-0.005085269755` | `0.000073829514` | no |
| v8_A3_gR2_half last | `0.039556814502` | `25.579640740489` | `0.071334724577` | `0.000142954968` | `-0.005040792419` | `0.000073923276` | no |

### 7. Official S4 11 Metrics

Metric contract: official full-pipeline streaming TotalCapture S4 evaluation with NewPL init36 upstream and the tested IK1 checkpoint.

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v8_B4_pRJ_x2_lowdyn last | `38.694152` | `10.155436` | `8.780414` | `4.506601` | `5.149188` | `10.316484` | `8.550490` | `4.360141` | `4.910783` | `0.278534` | `0.465389` |
| v8_A1_gR2_x2 last | `38.694199` | `10.155470` | `8.780414` | `4.506621` | `5.149212` | `10.316510` | `8.550475` | `4.360147` | `4.910786` | `0.278534` | `0.465390` |
| v8_B2_pRJ_x4 last | `38.694224` | `10.155424` | `8.780419` | `4.506596` | `5.149182` | `10.316508` | `8.550541` | `4.360200` | `4.910848` | `0.278535` | `0.465391` |
| v8_A2_gR2_x4 last | `38.694236` | `10.155477` | `8.780402` | `4.506597` | `5.149179` | `10.316534` | `8.550494` | `4.360166` | `4.910810` | `0.278535` | `0.465389` |
| v8_A4_gR2_x2_lowdyn last | `38.694258` | `10.155476` | `8.780392` | `4.506599` | `5.149183` | `10.316557` | `8.550501` | `4.360193` | `4.910828` | `0.278535` | `0.465392` |
| v8_B3_pRJ_half last | `38.694301` | `10.155500` | `8.780433` | `4.506617` | `5.149208` | `10.316533` | `8.550502` | `4.360168` | `4.910813` | `0.278533` | `0.465390` |
| v8_B1_pRJ_x2 last | `38.694343` | `10.155510` | `8.780420` | `4.506605` | `5.149192` | `10.316566` | `8.550514` | `4.360173` | `4.910820` | `0.278538` | `0.465392` |
| v8_A3_gR2_half last | `38.694350` | `10.155521` | `8.780432` | `4.506609` | `5.149198` | `10.316554` | `8.550512` | `4.360174` | `4.910828` | `0.278534` | `0.465389` |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newik1_v7_last_pl_control_lightloss_amass | best `-0.000629` | state L2 remains worse than official IK1 baseline | v8 gives a tiny S4 gain over v7, led by B4 lower pRJ dynamics. |
| newpl_v4_init36 | best `+0.068495` | best state L2 delta `+0.00007372` | Still much worse than PL-only mainline, and module output is not closer to GT than official IK1 baseline. |
| official IK1 module baseline on NewPL streaming cache | not applicable | pRJ L2 worse, gR2 angle slightly better, state L2 worse | The sweep trades tiny gR2 angle improvement for worse pRJ/state distance. |

### 9. Conclusion

- Module GT: no v8 trial beats the official IK1 baseline on state L2 under NewPL streaming input.
- Official S4: best v8 is `v8_B4_pRJ_x2_lowdyn last` with `38.69415222530066`, a tiny gain over v7 but still worse than NewPL init36 by `0.06849474249779774`.
- Keep as mainline: no.
- Successful part: reducing pRJ dynamics while using `pRJ=2.0` is the best direction inside this micro-sweep.
- Failure reason: the changes do not solve the core IK1 problem; isolated IK1 output remains slightly worse than official IK1 baseline and downstream S4 remains worse than PL-only.
- Next step: do not scale this exact v8 recipe into long training unless a new diagnostic explains how to improve pRJ/state L2 versus official IK1 under PL streaming input.

## Version: newik1_v9_adaptive_loss_search

### 1. Purpose

Continue from the best v8 control-point-input IK1 checkpoint and run another adaptive loss-weight micro-sweep. The goal was to find a positive direction that improves both official full-pipeline S4 and IK1 module-output-vs-GT under the deployment input distribution.

### 2. Main Change

| Change Type | Previous | This Version | Motivation |
|---|---|---|---|
| Input | v8 last-control 63D input | unchanged `RRB_after_pl[45] + last_control_pRB[15] + last_control_gR1[3] = 63D` | isolate loss/fine-tune effects |
| Output | `pRJ[69] + gR2[3] = 72D` | unchanged | preserve IK2/VR downstream contract |
| Loss | v8 B4: `pRJ=2`, low pRJ dynamics, control dynamics enabled | 8 variants around pRJ/gR2/control/dynamics; best disables control dynamics | test whether control/dynamic weight changes yield positive module and S4 gains |
| Training | v8 B4 selected checkpoint | 8 parallel 5-epoch TC micro-finetunes from v8 B4 `last.pt` | small adaptive search without committing to long training |
| Initialization | `v8_B4_pRJ_x2_lowdyn/train/last.pt` | unchanged across all v9 trials | start from best control-point IK1 result before v9 |
| Evaluation | v8 had Module GT and S4 | every v9 trial evaluates `best_loss.pt` and `last.pt`; Module GT now includes full pRJ, leaf pRJ, gR2 angle, and derivative deltas | require both official S4 and standalone IK1 evidence |

### 3. Input / Output Contract

| Item | Shape | Meaning | Source |
|---|---:|---|---|
| Input | 63D | `RRB_after_pl[45] + last_control_pRB[15] + last_control_gR1[3]` | NewPL init36 PL-streaming control cache |
| Output | 72D | `pRJ[69] + gR2[3]` | official IK1 downstream contract for IK2/VR |

Leaf-node metric contract: `leaf_pRJ` uses project IK1 leaf joint IDs `(18, 19, 4, 5, 15)`, mapped to pRJ indices from `joints[:, 1:]`.

### 4. Loss Design

Best v9 variant: `v9_C8_no_control_dyn/last.pt`, selected only as the best v9 ablation result, not as project mainline.

| Loss | Weight | Purpose | New / Changed / Unchanged |
|---|---:|---|---|
| `pRJ` | `2.0` | current-frame IK1 joint state | unchanged from v8 B4 |
| `gR2` | `1.0` | current-frame gravity/root direction | unchanged |
| `pRJ_dot` | `0.01` | pRJ first derivative | unchanged |
| `pRJ_ddot` | `0.0003` | pRJ second derivative | unchanged |
| `gR2_dot` | `0.03` | gR2 first derivative | unchanged |
| `gR2_ddot` | `0.001` | gR2 second derivative | unchanged |
| `control_pRJ` | `0.1` | control-point pRJ alignment | unchanged |
| `control_gR2` | `0.1` | control-point gR2 alignment | unchanged |
| `control_pRJ_dot` | `0.0` | control pRJ first derivative | changed from `0.003` |
| `control_gR2_dot` | `0.0` | control gR2 first derivative | changed from `0.003` |
| `control_pRJ_ddot` | `0.0` | control pRJ second derivative | changed from `0.0001` |
| `control_gR2_ddot` | `0.0` | control gR2 second derivative | changed from `0.0001` |
| `bone_length` | `0.0` | geometry prior | disabled |
| `control_point_prior` | `0.0` | residual/control prior | disabled |
| `tail_update_prior` | `0.0` | tail update regularizer | disabled |
| `gt_control_pRJ` | `0.0` | GT control supervision | disabled |
| `gt_control_gR2` | `0.0` | GT control supervision | disabled |

Full v9 trial sweep:

| Trial | Main Loss Change |
|---|---|
| `v9_C1_pRJ_x3_lowdyn` | `pRJ=3.0` |
| `v9_C2_pRJ_x2_gR2_x2` | `gR2=2.0` |
| `v9_C3_gR2_x3` | `gR2=3.0` |
| `v9_C4_dyn_lower` | `pRJ_dot=0.003`, `pRJ_ddot=0.0001` |
| `v9_C5_dyn_higher` | `pRJ_dot=0.03`, `pRJ_ddot=0.001` |
| `v9_C6_control_x2` | `control_pRJ=0.2`, `control_gR2=0.2` |
| `v9_C7_control_dyn_x2` | control derivative weights doubled |
| `v9_C8_no_control_dyn` | control derivative weights zeroed |

### 5. Training Recipe

| Stage | Data | Input Distribution | Init Checkpoint | Epochs | Output Checkpoint |
|---|---|---|---|---:|---|
| C1-C8 micro-finetunes | TotalCapture train/val | NewPL init36 PL-streaming TC cache | `data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt` | 5 each | `data/experiments/newik1_v9_adaptive_loss_search/<trial>/train/best_loss.pt`, `last.pt` |
| Module audit | TotalCapture S4 val, 5 sequences | NewPL init36 PL-streaming TC val cache, not teacher-forced | each trial checkpoint | n/a | `data/experiments/newik1_v9_adaptive_loss_search/<trial>/module_gt/<checkpoint>/result.json` |
| Full S4 | TotalCapture S4 val, 5 sequences | official full-pipeline streaming eval with NewPL init36 upstream | each trial checkpoint | n/a | `data/experiments/newik1_v9_adaptive_loss_search/<trial>/s4/<checkpoint>/result.json` |

### 6. Module GT Delta

Metric contract: Module GT uses NewPL init36 PL-streaming TotalCapture validation input, compared against the official IK1 baseline on the same cache. Negative delta means NewIK1 is closer to GT. Positive pRJ/leaf-pRJ delta means node positions are worse, even if gR2 is slightly better.

| Version | pRJ L2 cm Δ ↓ | leaf pRJ L2 cm Δ ↓ | gR2 angle deg Δ ↓ | pRJ_dot L2 Δ ↓ | pRJ_ddot L2 Δ ↓ | leaf_dot L2 Δ ↓ | leaf_ddot L2 Δ ↓ | gR2_dot L2 Δ ↓ | gR2_ddot L2 Δ ↓ | State L2 Δ ↓ | Better than official IK1 baseline? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v9_C8_no_control_dyn last | `+0.014219196` | `+0.004263760` | `-0.005290929` | `+0.000005583` | `-0.000075801` | `-0.000049199` | `-0.000141235` | `-0.000041919` | `-0.000085427` | `+0.000073135` | no |
| v9_C7_control_dyn_x2 last | `+0.014225325` | `+0.004267049` | `-0.005281134` | `+0.000005583` | `-0.000075801` | `-0.000049199` | `-0.000141235` | `-0.000041920` | `-0.000085427` | `+0.000073177` | no |
| v9_C2_pRJ_x2_gR2_x2 last | `+0.014224009` | `+0.004265906` | `-0.005277513` | `+0.000005583` | `-0.000075801` | `-0.000049199` | `-0.000141235` | `-0.000041920` | `-0.000085427` | `+0.000073178` | no |
| v9_C6_control_x2 last | `+0.014227194` | `+0.004272676` | `-0.005292260` | `+0.000005583` | `-0.000075801` | `-0.000049199` | `-0.000141235` | `-0.000041918` | `-0.000085427` | `+0.000073175` | no |
| v9_C1_pRJ_x3_lowdyn last | `+0.014225283` | `+0.004270220` | `-0.005294314` | `+0.000005583` | `-0.000075801` | `-0.000049199` | `-0.000141236` | `-0.000041921` | `-0.000085427` | `+0.000073158` | no |
| v9_C5_dyn_higher last | `+0.014228201` | `+0.004269829` | `-0.005289152` | `+0.000005583` | `-0.000075801` | `-0.000049199` | `-0.000141235` | `-0.000041920` | `-0.000085427` | `+0.000073187` | no |
| v9_C3_gR2_x3 last | `+0.014228988` | `+0.004277372` | `-0.005296286` | `+0.000005583` | `-0.000075801` | `-0.000049200` | `-0.000141235` | `-0.000041920` | `-0.000085427` | `+0.000073181` | no |
| v9_C4_dyn_lower last | `+0.014230915` | `+0.004276625` | `-0.005300432` | `+0.000005583` | `-0.000075801` | `-0.000049200` | `-0.000141235` | `-0.000041919` | `-0.000085427` | `+0.000073196` | no |

### 7. Official S4 11 Metrics

Metric contract: official full-pipeline streaming TotalCapture S4 evaluation with NewPL init36 upstream and the tested IK1 checkpoint.

| Version | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v9_C8_no_control_dyn last | `38.693845` | `10.155343` | `8.780297` | `4.506520` | `5.149079` | `10.316460` | `8.550429` | `4.360109` | `4.910725` | `0.278536` | `0.465391` |
| v9_C7_control_dyn_x2 last | `38.693845` | `10.155346` | `8.780304` | `4.506531` | `5.149094` | `10.316455` | `8.550422` | `4.360107` | `4.910723` | `0.278536` | `0.465392` |
| v9_C2_pRJ_x2_gR2_x2 last | `38.693848` | `10.155346` | `8.780301` | `4.506528` | `5.149089` | `10.316459` | `8.550424` | `4.360107` | `4.910724` | `0.278538` | `0.465391` |
| v9_C6_control_x2 last | `38.693869` | `10.155347` | `8.780304` | `4.506535` | `5.149098` | `10.316468` | `8.550431` | `4.360116` | `4.910733` | `0.278536` | `0.465391` |
| v9_C1_pRJ_x3_lowdyn last | `38.693870` | `10.155360` | `8.780307` | `4.506535` | `5.149099` | `10.316469` | `8.550417` | `4.360100` | `4.910714` | `0.278536` | `0.465391` |
| v9_C5_dyn_higher last | `38.693888` | `10.155356` | `8.780311` | `4.506536` | `5.149099` | `10.316471` | `8.550431` | `4.360108` | `4.910726` | `0.278536` | `0.465392` |
| v9_C3_gR2_x3 last | `38.693897` | `10.155356` | `8.780309` | `4.506536` | `5.149099` | `10.316477` | `8.550437` | `4.360113` | `4.910730` | `0.278535` | `0.465391` |
| v9_C4_dyn_lower last | `38.693899` | `10.155344` | `8.780323` | `4.506549` | `5.149115` | `10.316464` | `8.550446` | `4.360126` | `4.910741` | `0.278536` | `0.465391` |

### 8. Comparison

| Compared With | Score Delta | Module GT Delta | Main Observation |
|---|---:|---:|---|
| newik1_v8_parallel_adaptive_loss_search | best `-0.000308` | pRJ/leaf/state remain worse than official IK1 baseline | v9 slightly improves S4 over v8; best direction is removing control derivative loss. |
| newpl_v4_init36 | best `+0.068187` | best pRJ L2 delta `+0.014219 cm`, best leaf pRJ L2 delta `+0.004264 cm`, best state L2 delta `+0.000073135` | Still worse than PL-only mainline and fails the standalone module-output criterion. |
| official IK1 module baseline on NewPL streaming cache | not applicable | gR2 angle slightly improves by about `0.0053 deg`, but full and leaf pRJ are worse | The sweep improves gravity direction and temporal smoothness but not node position accuracy. |

### 9. Conclusion

- Module GT: not better overall. Full pRJ and leaf-pRJ positions are farther from GT for every v9 trial; gR2 angle and several derivative metrics are slightly better.
- Official S4: best is `v9_C8_no_control_dyn/last.pt` with `38.693844566687936`, slightly better than v8 but worse than NewPL init36.
- Keep as mainline: no.
- Successful part: removing control derivative losses gave the best S4 within v9 and the smallest pRJ/leaf-pRJ regression among v9 trials.
- Failure reason: the positive S4 movement is too small and conflicts with the module criterion that IK1 node/leaf positions should be closer to GT than official IK1 under NewPL streaming input.
- Next step: stop this particular micro-finetune branch as a mainline candidate. Future IK1 work should change the mechanism, not only local loss weights, and must directly reduce pRJ/leaf-pRJ GT error before scaling training.


## 7. IK-s2 Replacement Versions

Status: unresolved; `newpose_ctrl_v1` is a tested but rejected pose-control / IK-s2-slot replacement.

Confirmed requirement:
- replaced module must be IK-s2 / `iknet.net2`;
- input contract = `RRB_after_ik1[45] + gR2[3] + pRJ[69] = 117D`;
- output contract = `15 reduced joints x 6D = 90D`.

Current issue: earlier verified artifacts are NewIK1 / official-shape IK1 artifacts, not confirmed IK-s2 replacements. Do not add `newik2_v*` until exact IK-s2 artifact paths are verified.

### Version: newpose_ctrl_v1

Date: 2026-06-08

Purpose: test a control-point pose / IK2-slot replacement that outputs a pose-control state directly, while using the official-like route:

```text
AMASS pretrain -> DIP-IMU train fine-tune -> DIP-IMU test + TotalCapture test
```

Contract:

| Item | Value |
|---|---|
| Replaced slot | IK-s2 / pose-control slot before VR/physics |
| Frame input | `official IMU[90] + RRB_after_pl[45] + pRB/gR1[18] + last PL control[18] + gR0[3] = 174D` |
| Init-only input | `offset_r / r_JS`, used only for hidden-state initialization |
| Output | `RRJ_control[90] + gR_pose_control[3] = 93D` |
| DIP trans/root loss | not used |
| TotalCapture train split | not used |

Artifacts:

| Artifact | Path |
|---|---|
| Module | `newpose_ctrl.py` |
| Cache builder | `newpose_ctrl_cache.py` |
| Trainer | `newpose_ctrl_train.py` |
| Eval | `newpose_ctrl_eval.py` |
| Baseline module eval | `newpose_baseline_ik2_module_eval.py` |
| Runner | `scripts/run_newpose_ctrl_v1_official_protocol_20260608.sh` |
| Summary JSON | `data/experiments/newpose_ctrl_v1_20260608/summary.json` |
| Summary tables | `data/experiments/newpose_ctrl_v1_20260608/summary_tables.md` |
| AMASS best | `data/experiments/newpose_ctrl_v1_20260608/stage_a_amass_pretrain/best_loss.pt` |
| DIP best | `data/experiments/newpose_ctrl_v1_20260608/stage_b_dip_finetune/best_loss.pt` |
| DIP last | `data/experiments/newpose_ctrl_v1_20260608/stage_b_dip_finetune/last.pt` |

Training:

| Stage | Best epoch | Best selection value | Selection metric | Status |
|---|---:|---:|---|---|
| AMASS pretrain | 23 | `0.042461989620351234` | `control_pose_physical` | ok, early stopped |
| DIP fine-tune | 40 | `0.03869467038415071` | `control_pose_physical` | ok |

Full-pipeline comparison:

| Dataset | Version | Score ↓ | Local angle ↓ | Global angle ↓ | Decision |
|---|---|---:|---:|---:|---|
| DIP test | official_gpnet | `44.642051` | `8.469930` | `8.291750` | baseline |
| DIP test | newpl_v5_dip + official downstream | `44.598659` | `8.468339` | `8.315847` | best DIP baseline here |
| DIP test | newpose_ctrl_v1 Stage A | `428.806986` | `100.590216` | `101.018856` | rejected |
| DIP test | newpose_ctrl_v1 Stage B | `432.122581` | `100.867367` | `101.714333` | rejected; worse after DIP fine-tune |
| TotalCapture test | official_gpnet | `44.477381` | `12.550695` | `11.781375` | baseline |
| TotalCapture test | newpl_v5_amass + official downstream | `43.868067` | `12.423023` | `11.680961` | best TC baseline here |
| TotalCapture test | newpose_ctrl_v1 Stage A | `413.495453` | `95.741745` | `98.819038` | rejected |
| TotalCapture test | newpose_ctrl_v1 Stage B | `419.196776` | `96.512819` | `99.903576` | rejected; worse after DIP fine-tune |

Module IK2-slot / pose-control comparison:

| Dataset | Version | Control RRJ deg ↓ | State RRJ deg ↓ | FK joint L2 cm ↓ | gR loss ↓ |
|---|---|---:|---:|---:|---:|
| DIP test | official_gpnet | not available | `102.407646` | `4.971125` | `0.001227` |
| DIP test | newpl_v5_dip + official downstream | not available | `102.242027` | `4.994349` | `0.001464` |
| DIP test | newpose_ctrl_v1 Stage A | `66.256638` | `66.101151` | `44.539856` | `0.015581` |
| DIP test | newpose_ctrl_v1 Stage B | `65.028152` | `64.682121` | `45.447945` | `0.015881` |
| TotalCapture test | official_gpnet | not available | `77.909523` | `4.705381` | `0.003662` |
| TotalCapture test | newpl_v5_dip + official downstream | not available | `77.620567` | `4.600487` | `0.004123` |
| TotalCapture test | newpose_ctrl_v1 Stage A | `52.495689` | `52.634342` | `43.782135` | `0.027203` |
| TotalCapture test | newpose_ctrl_v1 Stage B | `51.712868` | `51.901062` | `44.701134` | `0.029040` |

Decision: rejected. The lower RRJ geodesic value is not sufficient evidence because decoded FK joint error is roughly an order of magnitude worse than the official/newpl_v5 baselines, and full-pipeline scores collapse to `413-432`. DIP fine-tuning does not help. Do not continue this module to mainline; redesign the pose-control representation/decoder before any further IK2-slot replacement claim.

## 8. Loss and Training Strategy Comparison

| Version | Module | pRB/pRJ | gR1/gR2 | dot | ddot | GT control | bone length | baseline preserve | distill | Init Change | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| newpl_v1_processed_no_baseline | PL | 1.0 | 1.0 | pRB 0.03 | pRB smooth 1e-6 | none | n/a | 0 | 0 | none | not selected |
| newpl_v2_gRdyn | PL | 1.0 | 1.0 | pRB/gR1 0.03 | gR1 0.001, pRB smooth 1e-6 | none | n/a | 0 | 0 | none | intermediate |
| newpl_v3_gtcontrol_rund | PL | 1.0 | 1.0 | pRB/gR1 0.03 | gR1 0.001, pRB smooth 1e-6 | pRB 0.3, gR1 0.1 | n/a | 0 | 0 | none | S4 improves |
| newpl_v4_init36 | PL | 1.0 | 1.0 | pRB/gR1 0.03 | gR1 0.001, pRB smooth 1e-6 | pRB 0.3, gR1 0.1 | n/a | 0 | 0 | 36D init | selected |
| newik1_v1_control_tail | IK1 | 1.0 | 1.0 | 0.03 | 0.001 | control 0.1 | 0 | n/a | not used | control-tail | not selected |
| newik1_v2_bonelength | IK1 | 1.0 | 1.0 | 0.03 | 0.001 | control 0.1 | 0.5 | n/a | not used | control-tail | local improves |
| newik1_v3_strong_pRJ_control | IK1 | 2.0 | 1.0 | 0.03 | 0.001 | control_pRJ 0.3 | 0.5 | n/a | not used | control-tail | local worsens, S4 less bad |
| newik1_v4_official_input | IK1 | 2.0 | 1.0 | pRJ 0.05/gR2 0.03 | pRJ 0.002/gR2 0.001 | none | 0.5 | n/a | pRJ 0.2 | official 63D | not selected |
| newik1_v5_last_pl_control | IK1 | 1.0 | 1.0 | 0.03 | 0.001 | control 0.1 | 0.5 | n/a | not used | last-control 63D | not selected |
| newik1_v6_official_input_init36_cascade | IK1 | 2.0 | 1.0 | pRJ 0.05/gR2 0.03 | pRJ 0.002/gR2 0.001 | none | 0.5 | n/a | pRJ 0.2 | official 63D, NewPL init36 cascade | module GT improves, S4 not selected |
| newik1_v8_parallel_adaptive_loss_search | IK1 | 0.5/1/2/4 | 0.5/1/2/4 | 0.01/0.03 plus control 0.003 | 0.0003/0.001 plus control 0.0001 | control 0.1, no GT control | 0 | n/a | not used | last-control 63D, NewPL init36 | tiny S4 gain vs v7, module-GT not selected |
| newik1_v9_adaptive_loss_search | IK1 | 2/3 | 1/2/3 | pRJ 0.003/0.01/0.03, gR2 0.03, control 0/0.003/0.006 | pRJ 0.0001/0.0003/0.001, gR2 0.001, control 0/0.0001/0.0002 | control 0.1/0.2, no GT control | 0 | n/a | not used | last-control 63D, NewPL init36 | tiny S4 gain vs v8; pRJ and leaf-pRJ still worse than official IK1 |
| newpose_ctrl_v1 | IK2 / pose-control | RRJ/state/control | gR_pose | state/control 0.01-0.03 family | state/control 0.001 family | pose-control tail | 0 | n/a | disabled by default | 174D official IMU + NewPL control features, offset_r init-only | rejected; FK and full-pipeline collapse |

## 9. Lessons Learned

- NewPL: processed/RMB-only input improves the frozen official baseline; GT control supervision improves S4; init36 is currently the strongest verified PL change. gR dynamics has module-GT evidence but its S4 JSON was not found in current artifacts.
- NewIK1: local pRJ/gR2 losses and bone length can reduce local validation loss, but local loss does not reliably predict full S4. Stronger pRJ/control improved S4 relative to weaker IK1 variants but still did not beat PL-only. Last-control input improved pRJ local value but worsened final S4 versus NewPL init36.
- NewIK1 v6: under the same NewPL input, the trained IK1 output can be closer to GT than the official IK1 baseline while full S4 still worsens. This suggests downstream IK2/VR/physics compatibility is more sensitive than the isolated `pRJ/gR2` GT distance.
- NewIK1 v8: small pRJ/gR2 loss-ratio sweeps can move S4 by about `0.0006`, but all tested variants remain worse than the official IK1 module baseline on `state_l2` under PL-streaming input. The best internal direction is `pRJ=2.0` with reduced pRJ dynamics, but it is not a mainline result.
- NewIK1 v9: removing control derivative losses gives the best S4 inside the v9 sweep, but full pRJ and leaf-pRJ positions are still farther from GT than the official IK1 baseline. A tiny S4 improvement is not enough when node/leaf positions regress.
- NewPose/IK2-slot v1: lower rotation-space RRJ geodesic can be misleading. `newpose_ctrl_v1` reduces RRJ geodesic versus official IK2-slot state metrics, but decoded FK joint error is `43.8-45.4 cm` and full-pipeline score is `413-432`, so FK/body-space compatibility and downstream physics must be gating metrics.
- Selection must require both module-output-vs-GT evidence and official full-pipeline streaming S4.

## 10. Artifact Index

| Version | Config | Train Result | Checkpoint | Module Audit JSON | S4 JSON | Logs |
|---|---|---|---|---|---|---|
| newpl_v1_processed_no_baseline | data/experiments/pl_curve_v2_processed_no_baseline/tc_finetune_10ep/config.json | data/experiments/pl_curve_v2_processed_no_baseline/tc_finetune_10ep/train_result.json | data/experiments/pl_curve_v2_processed_no_baseline/tc_finetune_10ep/best_loss.pt | not found | data/experiments/pl_curve_v2_processed_no_baseline/tc_finetune_10ep/s4_validation_best_loss_processed_imu.json | data/experiments/pl_curve_v2_processed_no_baseline/tc_finetune_10ep/train_log.jsonl |
| newpl_v2_gRdyn | data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_finetune_v1/tc_finetune_10ep/config.json | data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_finetune_v1/tc_finetune_10ep/train_result.json | data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_finetune_v1/tc_finetune_10ep/best_loss.pt | not found | not found | data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_finetune_v1/tc_finetune_10ep/train_log.jsonl |
| newpl_v3_gtcontrol_rund | data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/config.json | data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/train_result.json | data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/best_loss.pt | data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/streaming_audit/result.json | data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/s4_best/result.json | data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/train_log.jsonl |
| newpl_v4_init36 | data/experiments/pl_curve_init36_processed_rund_style/config.json | data/experiments/pl_curve_init36_processed_rund_style/train_result.json | data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt | not found | data/experiments/pl_curve_init36_processed_rund_style/eval_best_final_streaming_processed.json | data/experiments/pl_curve_init36_processed_rund_style/train_log.jsonl |
| newik1_v1_control_tail | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/config.json | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/train_result.json | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/best_loss.pt | data/experiments/newik1_mainline_20260604/diagnostics/newik1_vs_original_ik1_decoded_s4.json | data/experiments/newik1_mainline_20260604/s4/pl1_newik1_tc_best/result.json | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/train_log.jsonl |
| newik1_v2_bonelength | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/config.json | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/train_result.json | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/best_loss.pt | data/experiments/newik1_mainline_20260604/diagnostics/newik1_bonelen_w0p5_vs_original_ik1_decoded_s4.json | data/experiments/newik1_mainline_20260604/s4/pl1_newik1_tc_best_bonelen_w0p5/result.json | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/train_log.jsonl |
| newik1_v3_strong_pRJ_control | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/config.json | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/train_result.json | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/best_loss.pt | data/experiments/newik1_mainline_20260604/diagnostics/newik1_bonelen_w0p5_pRJ2_controlpRJ0p3_vs_original_ik1_decoded_s4.json | data/experiments/newik1_mainline_20260604/s4/pl1_newik1_tc_best_bonelen_w0p5_pRJ2_controlpRJ0p3/result.json | data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/train_log.jsonl |
| newik1_v4_official_input | data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/config.json | data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/train_result.json | data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt | not found | data/experiments/newik1_official_input_20260604/eval_pl1_streaming_tc_val.json | data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/train_log.jsonl |
| newik1_v5_last_pl_control | configs/newik1_last_pl_control_20260605_v2_tasks.json | data/experiments/newik1_last_pl_control_20260605_v2/stage_c_tc_finetune/train_result.json | data/experiments/newik1_last_pl_control_20260605_v2/stage_c_tc_finetune/best_loss.pt | not found | data/experiments/newik1_last_pl_control_20260605_v2/s4/best_loss/result.json | logs/orchestrator/newik1_last_pl_control_20260605_v2/ |
| newik1_v6_official_input_init36_cascade | configs/newik1_v6_official_input_init36_cascade_rerun_tasks.json | data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/train_result.json | data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt | data/experiments/newik1_v6_official_input_init36_cascade_rerun/module_gt/stage_a/result.json | data/experiments/newik1_v6_official_input_init36_cascade_rerun/s4/stage_a/result.json | logs/orchestrator/newik1_v6_official_input_init36_cascade_rerun/ |
| newik1_v8_parallel_adaptive_loss_search | configs/newik1_v8_parallel_adaptive_loss_search_local_AB_tasks.json | data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/train_result.json | data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt | data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/module_gt/last/result.json | data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/s4/last/result.json | logs/orchestrator/newik1_v8_parallel_adaptive_loss_search/; summary: data/experiments/newik1_v8_parallel_adaptive_loss_search/summary/phase1_ranking.json |
| newik1_v9_adaptive_loss_search | configs/newik1_v9_adaptive_loss_search_phase1_tasks.json | data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/train_result.json | data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt | data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/module_gt/last/result.json | data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/s4/last/result.json | logs/orchestrator/newik1_v9_adaptive_loss_search/; remote S4 logs: logs/orchestrator/newik1_v9_adaptive_loss_search_remote_s4/; summary: data/experiments/newik1_v9_adaptive_loss_search/summary/phase1_ranking.json |
## IK-s1 Auto Search Ledger

### Version: ik1_auto_search_round0

Purpose: complete comparable fixed-upstream evidence before launching new IK1 variants.

Fixed contract:

| Field | Value |
|---|---|
| IMU input | processed |
| PL upstream | `newpl_v4_init36` |
| PL checkpoint | `data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt` |
| IK1 baseline | official IK1 |
| Selection metric | S4/S5 full-pipeline 11 metrics |
| Diagnostic only | S4/S5 real streaming IK1 output vs GT; AMASS/cache module metrics |

Seed checkpoint audit:

| Seed | Status | Checkpoint |
|---|---|---|
| `newik1_v4_official_input` | found | `data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt` |
| `newik1_v6_stage_a` | found | `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt` |
| `newik1_v7_best` | found | `data/experiments/newik1_v7_last_pl_control_lightloss_amass/stage_a_lightloss_amass/best_loss.pt` |
| `newik1_v8_B4_last` | found | `data/experiments/newik1_v8_parallel_adaptive_loss_search/v8_B4_pRJ_x2_lowdyn/train/last.pt` |
| `newik1_v9_C8_last` | found | `data/experiments/newik1_v9_adaptive_loss_search/v9_C8_no_control_dyn/train/last.pt` |

Current Round 0 artifacts:

- Queue: `experiments/ik1_auto_search_queue.yaml`
- Results CSV: `experiments/ik1_auto_search_results.csv`
- New JSON root: `data/experiments/ik1_auto_search/round0/`

Decision status: incomplete. Do not declare any NewIK1 seed or v10 route best until S4/S5 full-pipeline metrics are complete and compared against `newpl_v4_init36 + official IK1`.

Round 0 completion update:

| Version | S4 Score ↓ | S5 Score ↓ | S4 real pRJ L2 cm ↓ | S4 real gR2 angle deg ↓ | Decision |
|---|---:|---:|---:|---:|---|
| baseline_official_ik1 | `38.625657482802865` | `43.81127653867006` | `4.9430084228515625` | `25.584686279296875` | baseline remains best |
| newik1_v4_official_input | `38.6972478222847` | `43.92881545905024` | `5.008570671081543` | `25.48590660095215` | worse S4/S5 |
| newik1_v6_stage_a | `38.649136830300094` | `43.85181389780715` | `4.977619171142578` | `25.492042541503906` | best NewIK1 seed, still worse than baseline |
| newik1_v7_best | `38.69478097228706` | `43.85909186106175` | `4.959465503692627` | `25.57992935180664` | worse S4/S5 |
| newik1_v8_B4_last | `38.69415222530066` | `43.860368536058814` | `4.959296703338623` | `25.579580307006836` | worse S4/S5 |
| newik1_v9_C8_last | `38.69384564702212` | `43.8609724480845` | `4.959182262420654` | `25.579391479492188` | worse S4/S5 |

Decision: keep `newpl_v4_init36 + official IK1` as baseline/current best. Use `newik1_v6_stage_a` only as the parent seed for conservative Round 1 design; do not mark it selected.

### Version: ik1_auto_search_round1_retry1

Purpose: retry the conservative IK1 v10 route search after the first Round 1 launch failed before producing checkpoints.

Fixed contract:

| Field | Value |
|---|---|
| IMU input | processed |
| PL upstream | `newpl_v4_init36` |
| PL checkpoint | `data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt` |
| IK1 parent seed | `newik1_v6_stage_a` |
| Parent checkpoint | `data/experiments/newik1_v6_official_input_init36_cascade_rerun/stage_a_continue/best_loss.pt` |
| Selection metric | S4/S5 full-pipeline 11 metrics |
| Diagnostic only | S4/S5 real streaming IK1 output vs GT; AMASS/cache/local training loss |

First launch failure:

| Field | Value |
|---|---|
| Failed train tasks | 4 |
| Blocked eval/audit tasks | 8 |
| State | `data/experiments/orchestrator_states/ik1_auto_search_round1_queue.json` |
| Logs | `logs/orchestrator/ik1_auto_search/round1/*/train.log` |
| Cause | batched-window shape mismatch in IK2-input distillation feature construction |

Retry design:

| Version | Key change | Input/output contract | Loss/training recipe | S4 full metrics | S4 real IK1 metric | Decision |
|---|---|---|---|---|---|---|
| `v10_residual_pRJ_only_alpha025_from_v6a` | pRJ residual only, alpha `0.25`; official/base gR2 preserved | IK1 output remains `pRJ[69]+gR2[3]`; IK2 input remains `117D` | 3 epochs, LR `3e-6`, pRJ GT + pRJ distill + gR2 official distill + light IK2-input distill | pending | pending | pending S4 full-pipeline |
| `v10_residual_pRJ_only_alpha05_from_v6a` | pRJ residual only, alpha `0.5`; official/base gR2 preserved | IK1 output remains `pRJ[69]+gR2[3]`; IK2 input remains `117D` | 3 epochs, LR `3e-6`, pRJ GT + pRJ distill + gR2 official distill + light IK2-input distill | pending | pending | pending S4 full-pipeline |
| `v10_stage_a_low_lr_distill_official` | conservative stage_a finetune | IK1 output remains `pRJ[69]+gR2[3]`; IK2 input remains `117D` | 3 epochs, LR `1e-6`, strong official distill + low dynamics weights | pending | pending | pending S4 full-pipeline |
| `v10_ik2_input_distill_from_v6a` | downstream-aware IK2-input distill | IK1 output remains `pRJ[69]+gR2[3]`; IK2 input remains `117D` | 3 epochs, LR `3e-6`, residual pRJ-only alpha `0.25`, strong IK2-input distill | pending | pending | pending S4 full-pipeline |

Artifacts:

- Queue: `experiments/ik1_auto_search_round1_retry1_queue.yaml`
- Output root: `data/experiments/ik1_auto_search/round1_retry1/`
- Logs: `logs/orchestrator/ik1_auto_search/round1_retry1/`

Decision: pending. Do not select a v10 candidate until S4 full-pipeline metrics exist; S5 is required for any candidate close to or better than the PL-only best.

Round 1 completion update:

| Version | S4 Score ↓ | S5 Score ↓ | S4 real pRJ L2 cm ↓ | S5 real pRJ L2 cm ↓ | Decision reason |
|---|---:|---:|---:|---:|---|
| `v10_residual_pRJ_only_alpha025_from_v6a` | `38.401125624060626` | `43.84817816592753` | `4.875001907348633` | `4.568667411804199` | pRJ improves and S4 improves, but S5 regresses versus official baseline |
| `v10_residual_pRJ_only_alpha05_from_v6a` | `38.30681182323395` | `43.867689362633975` | `4.86588716506958` | `4.551676273345947` | best S4, but S5 regresses more; downstream generalization risk |
| `v10_ik2_input_distill_from_v6a` | `38.41680224156379` | `43.807055798713115` | `4.876088619232178` | `4.568141937255859` | selected current best; only completed route beating S4 and S5 full-pipeline baseline |
| `v10_stage_a_low_lr_distill_official` | not found | not found | not found | not found | failed training; NaN local losses, no `best_loss.pt` |

Full 11-metric JSON artifacts:

- `v10_residual_pRJ_only_alpha025_from_v6a`: S4 `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha025_from_v6a/s5/best_loss/result.json`
- `v10_residual_pRJ_only_alpha05_from_v6a`: S4 `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round1_retry1/v10_residual_pRJ_only_alpha05_from_v6a/s5/best_loss/result.json`
- `v10_ik2_input_distill_from_v6a`: S4 `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/s5/best_loss/result.json`

Selected candidate:

```text
processed IMU + newpl_v4_init36 + v10_ik2_input_distill_from_v6a
```

Selected checkpoint:

```text
data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt
```

Stop condition reached because the selected candidate improves both S4 and S5 full-pipeline Score against `newpl_v4_init36 + official IK1`. Further search should be narrow and downstream-aware, not random or AMASS/local-loss selected.

### Version: ik1_auto_search_round2_downstream_aware

Purpose: continue narrow downstream-aware IK1 search from the Round 1 selected candidate without random search.

Parent:

```text
v10_ik2_input_distill_from_v6a
data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt
```

Fixed contract:

| Field | Value |
|---|---|
| IMU input | processed |
| PL upstream | `newpl_v4_init36` |
| PL checkpoint | `data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt` |
| Selection metric | S4/S5 full-pipeline 11 metrics |
| Diagnostic only | S4/S5 real streaming IK1 output vs GT; AMASS/cache/local loss |
| gR2 policy | preserved / official distill only |

Round 2 results:

| Version | Key change | S4 Score ↓ | S5 Score ↓ | S4 real pRJ L2 cm ↓ | S5 real pRJ L2 cm ↓ | Decision |
|---|---|---:|---:|---:|---:|---|
| `v11_alpha025_ik2w1_from_v10` | alpha `0.25`, IK2 distill `1.0` | `38.40251200318336` | `43.84088326931` | `4.874484062194824` | `4.566864967346191` | reject as seed; S5 regresses |
| `v11_alpha025_ik2w3_from_v10` | alpha `0.25`, IK2 distill `3.0` | `38.416982645660646` | `43.80445552650839` | `4.87534761428833` | `4.567495822906494` | beats baseline, not best |
| `v11_alpha035_ik2w2_from_v10` | alpha `0.35`, IK2 distill `2.0` | `38.37255207578838` | `43.77819666691124` | `4.863900184631348` | `4.556811332702637` | selected current best |
| `v11_pRJ_only_ik2w2_from_v10` | pRJ-only + IK2 distill `2.0` | `38.39572076609731` | `43.845466667562725` | `4.986303806304932` | `4.583155632019043` | reject as seed; S5 regresses |

Selected candidate:

```text
processed IMU + newpl_v4_init36 + v11_alpha035_ik2w2_from_v10
```

Selected checkpoint:

```text
data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt
```

Decision reason: `v11_alpha035_ik2w2_from_v10` is the strongest current S4/S5 full-pipeline candidate. It improves both S4 and S5 versus the official IK1 baseline and versus the Round 1 selected v10 route. pRJ-only and low IK2-distill variants are not selected because they regress S5 despite acceptable S4.

### Version: ik1_auto_search_round2_downstream_aware

Purpose: continue narrow downstream-aware search from `v10_ik2_input_distill_from_v6a`; do not use S4-only improvements as selection.

Fixed contract:

| Field | Value |
|---|---|
| IMU input | processed |
| PL upstream | `newpl_v4_init36` |
| PL checkpoint | `data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt` |
| Parent IK1 | `v10_ik2_input_distill_from_v6a` |
| Parent checkpoint | `data/experiments/ik1_auto_search/round1_retry1/v10_ik2_input_distill_from_v6a/train/best_loss.pt` |
| Selection metric | S4/S5 full-pipeline 11 metrics |
| Diagnostic only | S4/S5 real streaming IK1 output vs GT; AMASS/cache/local loss |

Round 2 result:

| Version | Key change | S4 Score ↓ | S5 Score ↓ | S4 real pRJ L2 cm ↓ | S5 real pRJ L2 cm ↓ | Decision reason |
|---|---|---:|---:|---:|---:|---|
| `v11_alpha025_ik2w1_from_v10` | alpha `0.25`, IK2 distill `1.0` | `38.40251200318336` | `43.84088326931` | `4.874484062194824` | `4.566864967346191` | reject; S5 regresses |
| `v11_alpha025_ik2w3_from_v10` | alpha `0.25`, IK2 distill `3.0` | `38.416982645660646` | `43.80445552650839` | `4.87534761428833` | `4.567495822906494` | S5 improves, but not best S4/S5 combined |
| `v11_alpha035_ik2w2_from_v10` | alpha `0.35`, IK2 distill `2.0` | `38.37255207578838` | `43.77819666691124` | `4.863900184631348` | `4.556811332702637` | selected current best |
| `v11_pRJ_only_ik2w2_from_v10` | pRJ-only + IK2 distill `2.0` | `38.39572076609731` | `43.845466667562725` | `4.986303806304932` | `4.583155632019043` | reject; S5 regresses |

Selected checkpoint:

```text
data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt
```

Full 11-metric JSON artifacts:

- S4: `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/s4/best_loss/result.json`
- S5: `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/s5/best_loss/result.json`

Decision: `v11_alpha035_ik2w2_from_v10` replaces `v10_ik2_input_distill_from_v6a` as the current IK1 candidate. Round 3 should stay close to alpha `0.35` and IK2-input distill weight `2.0`.

### Version: ik1_auto_search_round3_downstream_aware

Purpose: continue narrow downstream-aware search from `v11_alpha035_ik2w2_from_v10`; no random search and no S4-only selection.

Fixed contract:

| Field | Value |
|---|---|
| IMU input | processed |
| PL upstream | `newpl_v4_init36` |
| PL checkpoint | `data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt` |
| Parent IK1 | `v11_alpha035_ik2w2_from_v10` |
| Parent checkpoint | `data/experiments/ik1_auto_search/round2_downstream_aware_from_v10/v11_alpha035_ik2w2_from_v10/train/best_loss.pt` |
| IK1 output contract | `pRJ[69] + gR2[3]`, IK2 input remains `117D` |
| Selection metric | S4/S5 full-pipeline 11 metrics |
| Diagnostic only | S4/S5 real streaming IK1 output vs GT; AMASS/cache/local loss |
| gR2 policy | preserved / official distill only |

Round 3 result:

| Version | Key change | S4 Score ↓ | S5 Score ↓ | S4 real pRJ L2 cm ↓ | S5 real pRJ L2 cm ↓ | Decision reason |
|---|---|---:|---:|---:|---:|---|
| `v12_alpha040_ik2w2_from_v11` | alpha `0.40`, IK2 distill `2.0` | `38.36728921282291` | `43.73998444685712` | `4.862191200256348` | `4.553211212158203` | selected current best; best S4 and improves S5 versus v11 |
| `v12_alpha035_ik2w25_from_v11` | alpha `0.35`, IK2 distill `2.5` | `38.386719339862466` | `43.75161533830688` | `4.865252494812012` | `4.557664394378662` | beats both baselines, but S4 weaker than v11 |
| `v12_alpha035_s5stable_from_v11` | alpha `0.35`, IK2 distill `2.5`, LR `1e-6`, stronger pRJ distill/dynamics | `38.38921534772218` | `43.737316830046474` | `4.865253925323486` | `4.557403087615967` | best S5, but S4 weaker than selected candidate |
| `v12_alpha030_ik2w2_from_v11` | alpha `0.30`, IK2 distill `2.0` | `38.39537861308455` | `43.78341374022886` | `4.86892032623291` | `4.561491012573242` | beats both baselines, but weaker than v11 on S5 |

Selected candidate:

```text
processed IMU + newpl_v4_init36 + v12_alpha040_ik2w2_from_v11
```

Selected checkpoint:

```text
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Full 11-metric JSON artifacts:

- `v12_alpha040_ik2w2_from_v11`: S4 `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/s5/best_loss/result.json`
- `v12_alpha035_ik2w25_from_v11`: S4 `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_ik2w25_from_v11/s5/best_loss/result.json`
- `v12_alpha035_s5stable_from_v11`: S4 `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha035_s5stable_from_v11/s5/best_loss/result.json`
- `v12_alpha030_ik2w2_from_v11`: S4 `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha030_ik2w2_from_v11/s5/best_loss/result.json`

Decision: `v12_alpha040_ik2w2_from_v11` replaces `v11_alpha035_ik2w2_from_v10` as current IK1 candidate. Round 4, if run, should stay near alpha `0.40` and IK2-input distill `2.0`; do not expand to random mixed-objective search.

### Version: ik1_auto_search_round4_downstream_aware

Purpose: final narrow downstream-aware check around `v12_alpha040_ik2w2_from_v11`; no random search and no S4-only or S5-only selection.

Fixed contract:

| Field | Value |
|---|---|
| IMU input | processed |
| PL upstream | `newpl_v4_init36` |
| PL checkpoint | `data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt` |
| Parent IK1 | `v12_alpha040_ik2w2_from_v11` |
| Parent checkpoint | `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt` |
| IK1 output contract | `pRJ[69] + gR2[3]`, IK2 input remains `117D` |
| Selection metric | S4/S5 full-pipeline 11 metrics |
| Diagnostic only | S4/S5 real streaming IK1 output vs GT; AMASS/cache/local loss |
| gR2 policy | preserved / official distill only |

Round 4 result:

| Version | Key change | S4 Score ↓ | S5 Score ↓ | S4 real pRJ L2 cm ↓ | S5 real pRJ L2 cm ↓ | Decision reason |
|---|---|---:|---:|---:|---:|---|
| `v13_alpha042_ik2w2_from_v12` | alpha `0.42`, IK2 distill `2.0` | `38.370207245603204` | `43.71751535784453` | `4.862273693084717` | `4.551540374755859` | best Round 4 S4, but S4 worse than selected v12 |
| `v13_alpha038_ik2w2_from_v12` | alpha `0.38`, IK2 distill `2.0` | `38.376336369305854` | `43.73781035907567` | `4.86371374130249` | `4.555866718292236` | beats official baseline, not current best |
| `v13_alpha040_ik2w225_from_v12` | alpha `0.40`, IK2 distill `2.25` | `38.3787378231883` | `43.71488639246672` | `4.863152503967285` | `4.553428649902344` | better S5, S4 weaker than current best |
| `v13_alpha040_s5stable_from_v12` | alpha `0.40`, IK2 distill `2.25`, LR `8e-7`, stronger pRJ distill/dynamics | `38.3813811891675` | `43.707128487303855` | `4.862794399261475` | `4.552684783935547` | best Round 4 S5, diagnostic only because S4 is weaker |

Selected candidate remains:

```text
processed IMU + newpl_v4_init36 + v12_alpha040_ik2w2_from_v11
```

Selected checkpoint remains:

```text
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Full 11-metric JSON artifacts:

- `v13_alpha042_ik2w2_from_v12`: S4 `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha042_ik2w2_from_v12/s5/best_loss/result.json`
- `v13_alpha038_ik2w2_from_v12`: S4 `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha038_ik2w2_from_v12/s5/best_loss/result.json`
- `v13_alpha040_ik2w225_from_v12`: S4 `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_ik2w225_from_v12/s5/best_loss/result.json`
- `v13_alpha040_s5stable_from_v12`: S4 `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/s4/best_loss/result.json`; S5 `data/experiments/ik1_auto_search/round4_downstream_aware_from_v12/v13_alpha040_s5stable_from_v12/s5/best_loss/result.json`

Decision: Round 4 does not replace the current IK1 candidate. It is useful evidence that S5 can be improved further, but only by giving back S4. Keep `v12_alpha040_ik2w2_from_v11` as the current best unless a later confirmation/midpoint candidate improves both S4 and S5.

## NewIK1 v14 Round 5 confirmation from v12

Purpose: confirmation-only downstream-aware check around `v12_alpha040_ik2w2_from_v11`; no random search, no PL change, no gR2-heavy route.

Input/output contract:

| Field | Value |
|---|---|
| Upstream PL | fixed `newpl_v4_init36` |
| PL checkpoint | `data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt` |
| Parent IK1 | `v12_alpha040_ik2w2_from_v11` |
| Parent checkpoint | `data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt` |
| IK1 contract | official-shape `pRJ[69] + gR2[3]`; `gR2` preserved/official-distilled |
| Selection metric | S4/S5 full-pipeline 11 metrics only |

Training recipe:

| Version | Key change | LR | Epochs | Loss weights |
|---|---|---:|---:|---|
| `v14_alpha0405_ik2w2_from_v12` | residual pRJ alpha `0.405`, IK2-input distill `2.0` | `1e-6` | 3 | pRJ `1.0`, pRJ dot `0.03`, pRJ ddot `0.001`, IK1 pRJ distill `0.8`, IK1 gR2 distill `1.0`, IK2-input distill `2.0` |
| `v14_alpha0410_ik2w2_from_v12` | residual pRJ alpha `0.410`, IK2-input distill `2.0` | `1e-6` | 3 | same |
| `v14_alpha0415_ik2w2_from_v12` | residual pRJ alpha `0.415`, IK2-input distill `2.0` | `1e-6` | 3 | same |
| `v14_alpha0410_ik2w21_from_v12` | residual pRJ alpha `0.410`, IK2-input distill `2.1` | `1e-6` | 3 | same except IK2-input distill `2.1` |

Round 5 full-pipeline metrics:

| Version | S4 Score | S5 Score | S4 pRJ L2 cm | S5 pRJ L2 cm | S4 gR2 deg | S5 gR2 deg | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `v14_alpha0410_ik2w21_from_v12` | `38.36968127383291` | `43.719431065004315` | `4.861997127532959` | `4.552035808563232` | `25.584686279296875` | `15.153098106384277` | best Round 5 S4, not selected because S4 is worse than current best |
| `v14_alpha0415_ik2w2_from_v12` | `38.37063280807435` | `43.718751320485026` | `4.8627705574035645` | `4.553338527679443` | `25.584686279296875` | `15.153098106384277` | best Round 5 S5, diagnostic only because S4 is worse |
| `v14_alpha0405_ik2w2_from_v12` | `38.372773325160146` | `43.725488430112605` | `4.862767696380615` | `4.553722381591797` | `25.584686279296875` | `15.153098106384277` | beats official baseline, not current best |
| `v14_alpha0410_ik2w2_from_v12` | `38.37348917667567` | `43.72153897484764` | `4.863192558288574` | `4.553475379943848` | `25.584686279296875` | `15.153098106384277` | beats official baseline, not current best |

Selected candidate remains:

```text
processed IMU + newpl_v4_init36 + v12_alpha040_ik2w2_from_v11
```

Selected checkpoint remains:

```text
data/experiments/ik1_auto_search/round3_downstream_aware_from_v11/v12_alpha040_ik2w2_from_v11/train/best_loss.pt
```

Artifact pointers:

- Queue: `experiments/ik1_auto_search_round5_queue.yaml`
- State: `data/experiments/orchestrator_states/ik1_auto_search_round5_queue.json`
- Results CSV: `experiments/ik1_auto_search_results.csv`
- Output root: `data/experiments/ik1_auto_search/round5_confirmation_from_v12/`

Decision: Round 5 does not replace the current IK1 candidate. It strengthens the evidence that the alpha/IK2-distill family has reached an S4/S5 tradeoff: S5 improves as alpha or IK2 distill increases, but S4 becomes worse than the selected `v12`.

## Official GPNet TotalCapture fine-tune diagnostic

This is a diagnostic adaptation experiment, not the official training protocol.

Purpose: this is not a new replacement module. It is a fairness diagnostic for TotalCapture adaptation. The official codebase has no complete training entry, so `scripts/finetune_official_gpnet_totalcapture.py` was added as a minimal reproducible fine-tune/eval script using the existing official `GPNet` model, TotalCapture data, available GT supervision, and `test.py`/`MotionEvaluator`-compatible final evaluation.

Protocol and leakage label:

| Field | Value |
|---|---|
| Official protocol? | no |
| Diagnostic purpose | check TotalCapture adaptation advantage |
| Fine-tune data | `data/dataset_work/TotalCapture_globalpose_official/train.pt` |
| Validation data | `data/dataset_work/TotalCapture_globalpose_official/val.pt` |
| Test data | `data/dataset_work/TotalCapture_globalpose_official/test.pt` |
| Data leakage note | TotalCapture is used for fine-tune and test; do not report as paper-style generalization |
| Official checkpoint | `data/weights.pt` |
| Checkpoint loading | full `GPNet.state_dict`; split-module checkpoint not used |
| Model replacements | none |
| NewPL init36 | not used |
| IK1 replacement | not used |
| Trainable modules | `plnet`, `iknet.net1`, `iknet.net2`, `vrnet` |
| Frozen modules | none |

Training configs:

| Version | LR | Epochs | Status | Best validation loss | Checkpoint |
|---|---:|---:|---|---:|---|
| FT-A | `1e-5` | 2 | ok | `1.8420624017715455` | `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-A_lr1e-5_ep2/best_weights.pt` |
| FT-B | `3e-6` | 2 | ok | `1.8670690298080443` | `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-B_lr3e-6_ep2/best_weights.pt` |
| FT-C | `1e-6` | 2 | ok | `1.8737063884735108` | `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-C_lr1e-6_ep2/best_weights.pt` |

TotalCapture test full 11 metrics:

| Version | Train data | LR | Epochs | Trainable modules | Score ↓ | Local SIP ↓ | Local Angle ↓ | Local Joint ↓ | Local Mesh ↓ | Global SIP ↓ | Global Angle ↓ | Global Joint ↓ | Global Mesh ↓ | Root Jitter ↓ | Joint Jitter ↓ |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_gpnet_original | none | not found | 0 | none | 44.477380 | 9.989696 | 12.550694 | 4.519671 | 5.300476 | 9.314009 | 11.781373 | 3.810419 | 4.470022 | 0.399826 | 0.859838 |
| FT-A | TotalCapture train | 1e-5 | 2 | full GPNet neural modules | 43.349149 | 9.603109 | 12.205352 | 4.340323 | 5.041633 | 9.143327 | 11.585930 | 3.688516 | 4.298988 | 0.397926 | 0.854695 |
| FT-B | TotalCapture train | 3e-6 | 2 | full GPNet neural modules | 43.782556 | 9.765441 | 12.379111 | 4.433093 | 5.167890 | 9.155503 | 11.656312 | 3.742955 | 4.361113 | 0.399393 | 0.858540 |
| FT-C | TotalCapture train | 1e-6 | 2 | full GPNet neural modules | 44.191540 | 9.900780 | 12.483899 | 4.487517 | 5.249972 | 9.241099 | 11.730035 | 3.783797 | 4.426778 | 0.399826 | 0.859736 |

Fairness conclusion:

| Question | Answer |
|---|---|
| Does official GPNet improve after TotalCapture fine-tuning? | yes |
| Best version | FT-A |
| Best Delta Score | `-1.1282314625568688` |
| Impact on previous NewIK1 comparisons | TotalCapture-finetuned NewIK1 versus unfine-tuned official GPNet is not fairness-complete |
| Future baseline | compare TotalCapture-finetuned NewIK1 against similarly TotalCapture-finetuned official GPNet, currently `FT-A_lr1e-5_ep2` |

Artifacts:

- Script: `scripts/finetune_official_gpnet_totalcapture.py`
- Baseline JSON: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/baseline/eval_test.json`
- FT-A JSON: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-A_lr1e-5_ep2/eval_test.json`
- FT-B JSON: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-B_lr3e-6_ep2/eval_test.json`
- FT-C JSON: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/FT-C_lr1e-6_ep2/eval_test.json`
- Batch log: `data/experiments/official_gpnet_totalcapture_finetune_diagnostic/logs/run_all.log`

## IMUOffsetNet

Contract:

| Field | Value |
|---|---|
| Output | sequence/frame offset `offset_r` with shape `[6,3]` |
| Coordinate frame | `r_JS`: IMU origin relative to mapped joint J, expressed in joint-local coordinates |
| World reconstruction | `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS` |
| DIP input policy | official baseline `aM/wM/RMB`; processed IMU forbidden |
| TotalCapture policy | processed IMU diagnostic/evaluation |
| Real offset GT | DIP/TotalCapture offset GT not available |

Versions:

| Version | Structure | Output contract | Loss / audit | Training data | Checkpoint/log/json |
|---|---|---|---|---|---|
| `offset_v1_mlp_frame` | single-frame MLP | `[T,6,3]`, averaged to sequence `[6,3]` for cache | Stage A SmoothL1 offset GT; magnitude/smooth optional; acc consistency audit | AMASS synthetic smoke | `data/experiments/imu_offset_net_stageA_smoke/v1/best_loss.pt`, `train_result.json` |
| `offset_v2_temporal_rnn` | MLP encoder + GRU | `[T,6,3]`, averaged to sequence `[6,3]` | same as v1 | AMASS synthetic smoke | `data/experiments/imu_offset_net_stageA_smoke/v2/best_loss.pt`, `train_result.json` |
| `offset_v3_residual_prior` | MLP residual around median prior | `[T,6,3]`, averaged to sequence `[6,3]` | Stage A offset GT plus residual prior/smooth; Stage B DIP pose-acc proxy + magnitude/smooth/std | AMASS synthetic smoke; DIP official-input smoke | `data/experiments/imu_offset_net_stageA_smoke/v3/best_loss.pt`, `data/experiments/imu_offset_net_stageB_smoke/v3_dip_ft/best_loss.pt` |

Stage A synthetic offset accuracy smoke:

| Version | Offset L1 cm ↓ | Offset L2 cm ↓ | Per-sequence temporal stability cm ↓ | Acc consistency pred ↓ |
|---|---:|---:|---:|---:|
| `offset_v1_mlp_frame` | `5.37795` | `12.36579` | `2.83199` | `4.09173` |
| `offset_v2_temporal_rnn` | `5.47529` | `13.22492` | `1.46447` | `4.22182` |
| `offset_v3_residual_prior` | `2.22688` | `4.32751` | `0.03528` | `1.51809` |

Real-data utility smoke:

| Version | Input data | Offset coord frame | Offset L2 synthetic ↓ | Acc consistency ↓ | DIP S4/S5 Score ↓ | TotalCapture Score ↓ | Improved? | Conclusion |
|---|---|---|---:|---:|---|---:|---|---|
| `offset_v3_residual_prior` | DIP official `aM/wM/RMB` | `r_JS` joint-local | `4.32751 cm` from Stage A smoke | DIP pose-proxy `0.00804` | not measured | not measured | not measured | DIP fine-tune smoke runs; offset GT not available |
| `offset_v3_residual_prior` | TotalCapture processed IMU | `r_JS` joint-local | `4.32751 cm` from Stage A smoke | `2.97170` | not measured | `42.153350` one-seq PL smoke | no | predicted offset was effectively tied/slightly worse than existing cache offset (`42.153346`) |

Decision: `offset_v3_residual_prior` is the only smoke candidate worth scaling. Evidence so far supports that synthetic offsets can be learned, but does not show real-data downstream improvement. Full DIP/TotalCapture downstream results are pending the formal task file `configs/imu_offset_net_20260607_tasks.json`.

Formal 2026-06-07 results:

| Version | Input data | Offset coord frame | Offset L2 synthetic ↓ | Acc consistency ↓ | DIP Score ↓ | TotalCapture S4/S5 Score ↓ | Improved? | Conclusion |
|---|---|---|---:|---:|---:|---|---|---|
| `offset_v1_mlp_frame` | AMASS synthetic | `r_JS` joint-local | `4.26099 cm` best | `1.01356` | not evaluated | not evaluated | no | synthetic-only baseline |
| `offset_v2_temporal_rnn` | AMASS synthetic | `r_JS` joint-local | `4.24690 cm` best | `1.01638` | not evaluated | not evaluated | no | similar L2 to v1/v3, less stable than v3 |
| `offset_v3_residual_prior` | AMASS synthetic + DIP official + TC processed | `r_JS` joint-local | `4.25810 cm` best | TC S5 `2.96883`; DIP physical acc not measured | `44.642049 -> 44.642049` | S4 `38.625657 -> 38.625657`; S5 `43.811277 -> 43.811278` | no | stable offsets, no downstream gain |

Formal artifacts:

- Stage A v1: `data/experiments/imu_offset_net_20260607/stageA_v1/train_result.json`
- Stage A v2: `data/experiments/imu_offset_net_20260607/stageA_v2/train_result.json`
- Stage A v3: `data/experiments/imu_offset_net_20260607/stageA_v3/train_result.json`
- Stage B DIP fine-tune: `data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft/train_result.json`
- DIP predicted-offset cache: `data/experiments/imu_offset_net_20260607/dip_test_v3_pred_offset/baseline_cache_manifest.json`
- TotalCapture S4/S5 predicted-offset caches: `data/experiments/imu_offset_net_20260607/tc_val_v3_pred_offset/baseline_cache_manifest.json`, `data/experiments/imu_offset_net_20260607/tc_test_v3_pred_offset/baseline_cache_manifest.json`
- Downstream JSONs: `data/experiments/imu_offset_net_20260607/downstream/`

Final decision for this iteration: do not adopt `IMUOffsetNet` into NewPL/NewIK1/full pipeline. If continuing, change the downstream接入方式 or loss first; larger training alone is unlikely to help because the predicted sequence offset is almost constant and downstream metrics are unchanged.

## IMU position offset estimation for NewPL

Date: 2026-06-07

This entry tracks the diagnostic offset-to-NewPL replacement path. It is not an official protocol change and must not be reported as real offset GT evaluation.

Contract:

| Item | Value |
|---|---|
| Offset | `r_JS`, IMU origin relative to mapped joint `J`, expressed in joint-local coordinates |
| World reconstruction | `p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS` |
| DIP input | official `aM/wM/RMB`; no processed IMU, no `trans` loss, no real offset GT |
| TotalCapture input | processed IMU for diagnostic/adaptation only |
| NewPL I/O | official 84D frame input and 18D `pRB[15]+gR1[3]` output preserved |
| Design summary | kinematic/lever-arm optimization, self-supervised OffsetNet, and hybrid solver+net residual; `r_JS` joint-local offset |

Implemented versions:

| Version | Module | Algorithm | Input/output change | Loss/objective | Artifacts | Selected? |
|---|---|---|---|---|---|---|
| `offset_solver_v1_kinematic_opt` | offset pre-NewPL | lever-arm acceleration least-squares | external offset cache `[N,6,3]`; NewPL contract unchanged | acceleration residual, ridge, magnitude projection | `imu_position_offset.py`, `scripts/build_imu_position_offsets.py`, `data/experiments/imu_position_offset_newpl/tc_val_2seq/solver_v1_offsets.pt` | no |
| `offset_net_v2_selfsup` | offset pre-NewPL | StageA synthetic OffsetNet + StageB DIP self-supervision | external offset cache `[N,6,3]`; NewPL contract unchanged | synthetic sanity offset loss; real pose/acc proxy, no real offset GT | `imu_offset_net.py`, `imu_offset_train.py`, `imu_offset_finetune_dip.py`, `data/experiments/imu_offset_net_20260607/stageB_v3_dip_ft/best_loss.pt` | no |
| `offset_hybrid_v3_opt_init_net_refine` | offset pre-NewPL | solver init plus net residual/blend | external offset cache `[N,6,3]`; NewPL contract unchanged | solver plausibility plus learned residual | `data/experiments/imu_position_offset_newpl/tc_val_2seq/hybrid_v3_offsets.pt` | no |
| `newpl_offset_sensitive_smoke_v2` | PL-s1 diagnostic | NewPL training with offset contrast/dropout/noise | 84D/18D preserved; init36 preserved | PL GT loss plus diagnostic offset contrast | `data/experiments/imu_position_offset_newpl/newpl_offset_sensitive_smoke_v2/best_loss.pt` | no |

TotalCapture 2-sequence PL-level smoke under current `newpl_v4_init36`:

| Method | Offset median m | pRB orig cm | pRB NewPL cm | gR1 orig deg | gR1 NewPL deg | Output diff vs zero cm | IK1 | Full 11 metrics |
|---|---:|---:|---:|---:|---:|---:|---|---|
| zero | `0` | `9.07321` | `8.69193` | `26.8333` | `26.6172` | `0` | not measured | not measured |
| random | `0.224814` | `9.07321` | `8.69193` | `26.8333` | `26.6172` | `1.66547e-07` | not measured | not measured |
| solver_v1 | `0.0654287` | `9.07321` | `8.69193` | `26.8333` | `26.6172` | `3.72451e-08` | not measured | not measured |
| net_v2 | `0.153126` | `9.07321` | `8.69193` | `26.8333` | `26.6172` | `8.12792e-08` | not measured | not measured |
| hybrid_v3 | `0.184449` | `9.07321` | `8.69193` | `26.8333` | `26.6172` | `6.61554e-08` | not measured | not measured |

Offset-sensitive smoke v2 did not solve the issue: random-vs-zero output diff mean is only `2.489e-06 cm`; solver/net/hybrid remain below `1e-06 cm` except random. Decision: none of the offset methods is selected yet, because the current NewPL does not meaningfully use `offset_r`. Next replacement work must first make PL-s1 offset-sensitive before running IK1/full-pipeline five-method comparisons.

### `newpl_offset_conditioned_smoke_v1`

Purpose: make PL-s1 able to observe the static offset at every frame while preserving the official PL external contract.

| Item | Value |
|---|---|
| Module | PL-s1 diagnostic |
| Architecture | `PLCurveOffsetConditionedModule` |
| Input contract | unchanged 84D PL frame input |
| Output contract | unchanged 18D `pRB[15]+gR1[3]` |
| Init contract | unchanged init36 `offset_r[18]+pRL[15]+gR0[3]` |
| Change | condition branch injects init36 encoding into every recurrent step |
| Checkpoint | `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_smoke_v1/best_loss.pt` |
| Selection | not selected |

Training: TotalCapture diagnostic smoke, 8 train sequences / 2 val sequences, 3 epochs, lr `1e-4`, residual scale `0.05`, condition scale `1.0`, offset contrast weight `1.0`, best validation loss `0.515810`.

Sensitivity versus zero offset:

| Method | Output diff cm | gR diff deg |
|---|---:|---:|
| random | `5.493e-04` | `1.908e-04` |
| solver_v1 | `2.802e-04` | `1.173e-04` |
| net_v2 | `1.628e-03` | `1.736e-04` |
| hybrid_v3 | `2.041e-03` | `1.852e-04` |

PL-level metrics on the same 2 TotalCapture validation sequences:

| Method | pRB NewPL cm | pRB delta cm | gR1 NewPL deg | gR1 delta deg |
|---|---:|---:|---:|---:|
| zero | `9.123797` | `0.050590` | `25.337427` | `-1.495884` |
| random | `9.123161` | `0.049954` | `25.337582` | `-1.495731` |
| solver_v1 | `9.123602` | `0.050395` | `25.337492` | `-1.495819` |
| net_v2 | `9.122195` | `0.048988` | `25.337856` | `-1.495458` |
| hybrid_v3 | `9.121889` | `0.048682` | `25.337954` | `-1.495360` |

Decision: this version proves offset conditioning can create measurable output differences, but it still does not make offset methods clearly better. pRB worsens slightly versus original PL, while gR1 improves; net/hybrid only beat zero by about `0.001-0.002 cm` on pRB. Keep as diagnostic evidence, not as a selected NewPL replacement. IK1 and full 11 metrics remain `not measured`.

### `newpl_offset_conditioned_pairwise_v2`

Purpose: test whether direct good-vs-bad offset training can make NewPL prefer correct offset.

| Item | Value |
|---|---|
| Module | PL-s1 diagnostic |
| Init | `newpl_offset_conditioned_smoke_v1/best_loss.pt` |
| Loss change | `good_metric + relu(good_metric + margin - bad_metric)` |
| Checkpoint | `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_pairwise_v2/best_loss.pt` |
| Swap eval | `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_swap_eval_pairwise_v2_hybrid_cache.json` |
| Selection | not selected |

Training result: best validation loss `0.505689` at epoch 5, but final train `offset_bad_minus_good_metric` is only `3.59e-08`.

Swap eval on 2 TotalCapture validation sequences:

| Bad init | pRB delta vs good cm | gR1 delta vs good deg | PL GT loss delta vs good |
|---|---:|---:|---:|
| zero | `+0.020648` | `-0.011161` | `-1.132e-04` |
| roll_sensors | `+0.012551` | `-0.006490` | `-6.156e-05` |
| other_sequence | `+0.000510` | `-0.000966` | `-1.961e-05` |
| negate | `+0.027347` | `-0.014600` | `-1.457e-04` |

Decision: pRB separability exists after pairwise training, but the combined PL target still prefers bad offsets due to the gR1 term moving in the opposite direction. This is not a selected replacement and still does not justify IK1/full 11-metric evaluation.

### `newpl_offset_conditioned_prb_contrast_v1`

Purpose: remove the gR1 term from the offset contrast metric and test whether correct offsets become more useful for `pRB[15]`.

| Item | Value |
|---|---|
| Module | PL-s1 diagnostic |
| Init | `newpl_offset_conditioned_pairwise_v2/best_loss.pt` |
| Loss change | `--offset-contrast-target pRB`, contrast metric is SmoothL1 over output channels `0:15` only |
| Checkpoint | `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_prb_contrast_v1/best_loss.pt` |
| Train result | `data/experiments/imu_position_offset_newpl/newpl_offset_conditioned_prb_contrast_v1/train_result.json` |
| Swap eval | `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_swap_eval_prb_contrast_v1_hybrid_cache.json` |
| Selection | not selected |

Training result: best validation loss `0.502297` at epoch 1, then validation degraded to `0.526568` by epoch 5. Final train `offset_bad_minus_good_metric` is `-5.22e-08`, so pRB-only contrast did not create meaningful train-time separability.

Swap eval on 2 TotalCapture validation sequences:

| Bad init | pRB delta vs good cm | gR1 delta vs good deg | PL GT loss delta vs good |
|---|---:|---:|---:|
| zero | `+0.018504` | `-0.010733` | `-1.143e-04` |
| roll_sensors | `+0.011441` | `-0.006266` | `-6.256e-05` |
| other_sequence | `+0.000239` | `-0.001027` | `-2.113e-05` |
| negate | `+0.024665` | `-0.014092` | `-1.479e-04` |

Decision: this version confirms that isolating the contrast to pRB is not enough. Correct offset remains only marginally better for pRB, and full PL loss still prefers bad offsets because gR1 improves in the opposite direction. It is not a selected NewPL replacement; IK1 and full-pipeline 11 metrics are `not measured`.

### `offset_consistency_eval_v1`

Purpose: add a real-data diagnostic that does not require real offset GT. The metric evaluates forward lever-arm acceleration consistency for each offset cache.

| Item | Value |
|---|---|
| Script | `imu_position_offset_consistency_eval.py` |
| Coordinate contract | `r_JS`, joint-local IMU origin offset |
| Evaluation data | TotalCapture validation smoke, 2 sequences |
| JSON | `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_consistency_eval_v1.json` |
| GT offset used? | no |
| Selected replacement? | no |

TotalCapture 2-sequence aggregate:

| Method | Mean residual m/s^2 | Mean improvement vs zero m/s^2 | Median offset norm m |
|---|---:|---:|---:|
| zero | `13.416491` | `0.000000` | `0.000000` |
| random | `15.256863` | `-1.840372` | `0.224814` |
| solver_v1 | `13.337500` | `0.078991` | `0.065429` |
| net_v2 | `13.096867` | `0.319623` | `0.153126` |
| hybrid_v3 | `13.304113` | `0.112376` | `0.184449` |

Decision: solver/net/hybrid offsets are physically meaningful under forward acceleration consistency, and random offsets are worse. `net_v2` is best on this diagnostic. This does not override the NewPL decision: because `pRB/gR1` and full PL loss still do not improve clearly, none of the offset routes is selected for downstream IK1/full-pipeline evaluation yet.

### `offset_net_v2_stageB_v4_dip_consistency_smoke`

Purpose: make DIP self-supervised OffsetNet fine-tune auditable by recording pre-train validation and checking whether the no-GT consistency loss improves from initialization.

| Item | Value |
|---|---|
| Module | offset pre-NewPL |
| Init | `data/experiments/imu_offset_net_20260607/stageA_v3/best_loss.pt` |
| Script | `imu_offset_finetune_dip.py` |
| Input | DIP official `aM/wM/RMB` |
| Forbidden supervision | processed IMU, DIP `trans_loss`, real offset GT loss |
| Initial validation | `data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/initial_val.json` |
| Train result | `data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/train_result.json` |
| Checkpoint | `data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/best_loss.pt` |
| Selection | not selected |

Tiny smoke result:

| Metric | Initial | Last/best | Delta |
|---|---:|---:|---:|
| DIP val pose_acc_proxy | `0.018450846` | `0.018450512` | `-3.34e-07` |
| offset magnitude m | `0.174248368` | `0.174248084` | `-2.83e-07` |
| best epoch | not applicable | `2` | not applicable |
| offset L1/L2 cm | not available | not available | not available |

Decision: valid diagnostic training path, but no meaningful improvement. Keep `stageB_v3_dip_ft` as the existing net_v2 artifact for cache generation unless a later run demonstrates a real proxy/NewPL gain. Do not use this smoke to claim NewPL or full-pipeline improvement.

### `offset_newpl_decision_matrix_v1`

Purpose: consolidate all current offset route evidence into one selection artifact.

| Item | Value |
|---|---|
| Script | `scripts/summarize_imu_offset_newpl.py --decision-json --decision-md` |
| JSON | `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.json` |
| Markdown | `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.md` |
| Real offset GT | not available |
| DIP trans | not used |
| TotalCapture role | diagnostic/adaptation only |
| IK1/full 11 metrics | not measured |
| Completion audit | requirement evidence is recorded in this section, `PROJECT_STATUS.md`, `EXPERIMENT_LOG.md`, and `offset_newpl_decision_matrix_v1.json` |

Decision matrix:

| Method | Forward residual m/s^2 | Forward improvement | Cond. PL pRB vs zero cm | Cond. PL gR1 vs zero deg | Decision |
|---|---:|---:|---:|---:|---|
| zero | `13.416491` | `0.000000` | `0.000000` | `0.000000` | not selected |
| random | `15.256863` | `-1.840372` | `-0.000636` | `0.000154` | negative control |
| solver_v1 | `13.337500` | `0.078991` | `-0.000196` | `0.000065` | physical signal, downstream not selected |
| net_v2 | `13.096867` | `0.319623` | `-0.001602` | `0.000429` | physical signal, downstream not selected |
| hybrid_v3 | `13.304113` | `0.112376` | `-0.001908` | `0.000526` | physical signal, downstream not selected |

Selection:

```text
best_offset_method_by_forward_consistency = net_v2
best_offset_method_for_newpl = not selected
run_ik1_or_full_pipeline = false
```

Decision: none of the offset routes is selected as a NewPL replacement input yet. `net_v2` is the best current physical-consistency route, but its NewPL pRB gain over zero is only `0.001602 cm` and gR1 slightly worsens. Future work should change the NewPL offset injection/loss before running IK1 or full-pipeline 11 metrics.

### Offset experiment completion status

Date: 2026-06-07

Status: completed for the current offset-estimation design; not selected.

| Category | Completed evidence | Result |
|---|---|---|
| Synthetic OffsetNet training | Stage A v1/v2/v3 on AMASS synthetic offset GT | L1 around `2.19-2.30 cm`, best L2 around `4.25 cm`; learnable on synthetic data |
| DIP real-data fine-tune | official `aM/wM/RMB`, no processed IMU, no `trans`, no offset GT | auditable, but no meaningful improvement |
| TotalCapture predicted-offset cache | S4/S5 processed diagnostic caches | generated successfully |
| Downstream PL utility | DIP and TotalCapture baseline/cache offset vs predicted offset | no improvement |
| Physical consistency | TotalCapture 2-seq forward lever-arm acceleration residual | `net_v2` best physical signal |
| NewPL offset sensitivity | zero/random/solver/net/hybrid and offset-conditioned PL diagnostics | current NewPL does not use offset strongly enough |
| IK1/full 11 metrics | intentionally not run | PL evidence too weak |

Synthetic offset metric note: the remembered `2.x cm` value is offset L1, while the completion summary's `4.x cm` value is vector offset L2. For formal Stage A, v3 has last offset L1 `2.18759 cm`, best offset L2 `4.25810 cm`, and last offset L2 `4.25904 cm`.

Final comparison:

| Dataset/protocol | Baseline/cache score ↓ | Predicted-offset score ↓ | Decision |
|---|---:|---:|---|
| DIP test official | `44.642049` | `44.642049` | no effect |
| TotalCapture S4 PL | `38.625657` | `38.625657` | no effect |
| TotalCapture S5 PL | `43.811277` | `43.811278` | slightly worse |

Selection: no offset route is selected for NewPL/NewIK1/full-pipeline integration. Keep `net_v2` only as the best physical-consistency reference (`13.096867 m/s^2` residual vs zero `13.416491`), not as a downstream replacement.

### `net_v2_stageB_v4_transfer_check`

Purpose: test whether the auditable DIP stageB v4 self-supervised checkpoint should replace the existing `net_v2` offset source for TotalCapture diagnostics.

| Item | Value |
|---|---|
| Init/checkpoint | `data/experiments/imu_offset_net_20260607/stageB_v4_dip_consistency_smoke/best_loss.pt` |
| Offset cache | `data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_stageB_v4_offsets.pt` |
| Offset summary | `data/experiments/imu_position_offset_newpl/tc_val_2seq/net_v2_stageB_v4_offsets.json` |
| Consistency compare | `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_consistency_eval_stageB_v4_compare.json` |
| Decision matrix | `data/experiments/imu_position_offset_newpl/tc_val_2seq/offset_newpl_decision_matrix_v1.json` |
| Selected? | no |

Comparison:

| Method | Mean residual m/s^2 | Improvement vs zero m/s^2 | Median offset norm m |
|---|---:|---:|---:|
| net_v2 | `13.096867` | `0.319623` | `0.153126` |
| net_v2_stageB_v4 | `13.096842` | `0.319649` | `0.153162` |

Decision: `stageB_v4` changes TotalCapture forward consistency by only `+2.56e-05 m/s^2` relative to existing `net_v2`. Do not generate a separate NewPL cache/eval for this checkpoint; keep `net_v2` as the current physical-consistency reference and keep NewPL selection as `not selected`.

## NewPL-root module training and module-level evaluation

### Version: `newpl_root_v1`

| Item | Value |
|---|---|
| Replaced module | PL-s1 diagnostic module |
| Input contract | official PL `84D = aRB[18]+wRB[18]+RRB[45]+gR0[3]` |
| Init contract | `init36 = offset_r[18]+pRL[15]+gR0[3]` |
| Output contract | `21D = pRB[15]+gR1[3]+root_vel[3]` |
| Downstream IK1 compatibility | not connected; IK1 still consumes 18D PL output |
| root_vel frame | root/body frame, m/s |
| root_vel GT | AMASS/TotalCapture only; DIP GT not available |
| Full 11 metrics | not measured |
| Status | implemented and smoke-tested; not selected for mainline |

Root velocity definition: `root_vel = finite_difference(tran_gt) @ pose[:,0]`, using the same row-vector root/body projection convention as current `pRB`. This avoids a heading-dependent world-frame target for a module whose inputs are root/body-relative.

Loss design:

| Loss | Weight default | AMASS | TotalCapture | DIP |
|---|---:|---|---|---|
| `pRB` | `1.0` | yes | yes | yes |
| `gR1` | `1.0` | yes | yes | yes |
| `root_vel` | `0.25` | GT | GT if reliable | no |
| `pRB_dot` | `0.03` | yes | yes | yes |
| `gR1_dot` | `0.03` | yes | yes | yes |
| `root_vel_smooth` | `0.01` | yes | yes | optional |
| `gt_control_pRB` | `0.3` | yes | yes | yes |
| `gt_control_gR1` | `0.1` | yes | yes | yes |

Artifacts:

| Artifact | Path |
|---|---|
| Module code | `newpl_root.py` |
| Training entry | `newpl_root_train.py` |
| Module eval entry | `newpl_root_eval.py` |
| Smoke checkpoint | `data/experiments/newpl_root_v1/smoke/tc_train_smoke/best_loss.pt` |
| Smoke log | `data/experiments/newpl_root_v1/smoke/tc_train_smoke/train_log.jsonl` |
| Smoke TC JSON | `data/experiments/newpl_root_v1/smoke/tc_multi_module_smoke.json` |
| Smoke DIP JSON | `data/experiments/newpl_root_v1/smoke/dip_root_module_smoke.json` |

Smoke module-output metrics:

| Dataset | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | root_vel L1 ↓ | root_vel L2 ↓ | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| TotalCapture | official_PL | `3.607639` | `7.687036` | `11.658890` | not applicable | not applicable | single-sequence smoke |
| TotalCapture | newpl_v4_init36 | `3.412260` | `7.253721` | `11.529694` | not applicable | not applicable | single-sequence smoke |
| TotalCapture | newpl_root_v1_smoke | `3.411824` | `7.252498` | `11.529869` | `0.260059` | `0.553485` | single-sequence smoke |
| DIP-IMU | newpl_root_v1_smoke | `3.462420` | `7.301293` | `12.611866` | root_vel GT not available | root_vel GT not available | DIP trans not used |

Decision: `newpl_root_v1` is implemented and mechanically valid, but the required AMASS long pretrain, TotalCapture fine-tune, and DIP fine-tune have not been run. Do not claim improvement over `newpl_v4_init36`, do not connect it to IK1/full pipeline yet, and keep all full-run metrics as `not measured`.

### Fair comparison gate for `newpl_root_v1`

Selection requires both tables below on each evaluated dataset.

PL output comparison must include `official_PL`, `newpl_v4_init36`, and `newpl_root_v1`:

| Dataset | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | Notes |
|---|---|---:|---:|---:|---|

Root velocity comparison must include `GT`, `official baseline`, `newpl_v4_init36 baseline`, and `newpl_root_v1` when GT is available:

| Dataset | Version | root_vel source | root_vel L1 ↓ | root_vel L2 ↓ | root_vel angle ↓ | Notes |
|---|---|---|---:|---:|---:|---|

Baseline velocity policy: use official pipeline velocity if available; otherwise finite-difference final pipeline translation and project it to the GT root/body frame. If GT root velocity is not available, record `root_vel GT not available` and do not compare velocity. `newpl_root_v1` cannot be selected unless `pRB/gR1` are not weaker than baseline PL and `root_vel` is clearly better than baseline velocity where baseline velocity is comparable.

Long-run status: `scripts/run_newpl_root_v1_longtrain_20260607.sh` has been started via `longrun`. Pending artifacts live under `data/experiments/newpl_root_v1/longrun_20260607/`. This version remains unselected until the fair PL-output and root-velocity tables are produced.
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

### Fine-tune before/after audit

Additional module-level JSONs:

- `data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/amass_best_last_module_metrics.json`
- `data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/tc_before_after_pl_only_metrics.json`
- `data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/tc_before_after_root_head_only_metrics.json`
- `data/experiments/newpl_root_v1/longrun_20260607_b2048_controlselect/eval_finetune_effect/dip_before_after_pl_metrics.json`

| Dataset | Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | root_vel L1 ↓ | root_vel L2 ↓ | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| AMASS-val20 | newpl_root_amass_best | `1.671904` | `3.460373` | `4.869618` | `0.193051` | `0.427344` | selected epoch 20 |
| AMASS-val20 | newpl_root_amass_last | `1.654693` | `3.415629` | `4.865114` | `0.193132` | `0.427571` | decoded PL better, root L1/L2 slightly worse |
| TotalCapture-test | before_tc_amass_best | `3.290194` | `6.825256` | `13.408251` | `0.269022` | `0.587931` | before TC fine-tune |
| TotalCapture-test | tc_finetune_best | `3.268334` | `6.779195` | `13.376897` | `0.268907` | `0.587701` | small improvement |
| TotalCapture-test | tc_finetune_last | `3.267189` | `6.776795` | `13.375096` | `0.268900` | `0.587688` | slightly best decoded PL/root-head, still unselected |
| DIP-IMU-test | before_dip_amass_best | `3.115633` | `6.428928` | `12.852417` | not available | not available | DIP root_vel GT not available |
| DIP-IMU-test | dip_finetune_best | `3.115812` | `6.429121` | `12.854242` | not available | not available | no improvement |
| DIP-IMU-test | dip_finetune_last | `3.117006` | `6.430589` | `12.864961` | not available | not available | worse |

Fine-tune judgment: TC fine-tune helps `newpl_root_v1` slightly, DIP fine-tune does not help, and the root velocity head remains worse than the fair pipeline-derived velocity baseline. Keep `newpl_root_v1` as a recorded but rejected PL replacement.

## IK-s1: `newik1_v10_official_protocol_last_control`

Date: 2026-06-08

Purpose: official-like NewIK1 retraining with PL control-point input, following AMASS pretrain -> AMASS PL-streaming adaptation -> DIP-IMU PL-streaming fine-tune. This tests whether a NewIK1 trained in the same spirit as the official route improves over the official IK1 baseline when paired with `newpl_v5`.

Contract:

| Item | Value |
|---|---|
| Input feature mode | `last_control` |
| Input dim | `63D = RRB_after_pl[45] + last_control_gR1[3] + last_control_pRB[15]` |
| Output dim | `72D = pRJ[69] + gR2[3]` |
| Downstream contract | unchanged official IK1 output shape |
| DIP trans loss | not used |
| TotalCapture train split | not used |

Artifacts:

| Artifact | Path |
|---|---|
| Script | `scripts/run_newik1_v10_official_protocol_last_control_20260607.sh` |
| Root | `data/experiments/newik1_v10_official_protocol_last_control_20260607` |
| Log | `data/experiments/newik1_v10_official_protocol_last_control_20260607/logs/run_full.log` |
| Summary JSON | `data/experiments/newik1_v10_official_protocol_last_control_20260607/summary.json` |
| Stage A best | `data/experiments/newik1_v10_official_protocol_last_control_20260607/stage_a_amass_teacher_forced/best_loss.pt` |
| Stage B best | `data/experiments/newik1_v10_official_protocol_last_control_20260607/stage_b_amass_pl_streaming/best_loss.pt` |
| Stage C best | `data/experiments/newik1_v10_official_protocol_last_control_20260607/stage_c_dip_pl_streaming/best_loss.pt` |

Training:

| Stage | Best epoch | Best loss | Status |
|---|---:|---:|---|
| AMASS teacher-forced | 33 | `0.00048058280081022533` | ok |
| AMASS PL-streaming | 20 | `0.013035929867764934` | ok |
| DIP PL-streaming | 40 | `0.13303746217085669` | ok |

Module and score evidence:

| Dataset | Version | Score ↓ | pRJ L2 cm ↓ | gR2 angle deg ↓ | Decision |
|---|---|---:|---:|---:|---|
| DIP test | official_gpnet | `44.642051` | `5.082861` | `15.268174` | baseline |
| DIP test | newpl_v5_dip + official IK1 | `44.598659` | `5.107541` | `14.869393` | best DIP score |
| DIP test | newik1_v10 Stage C | `44.730331` | `5.087737` | `15.128205` | rejected: worse score and gR2 |
| TotalCapture test | official_gpnet | `44.477381` | `4.773225` | `15.323585` | original baseline |
| TotalCapture test | newpl_v5_amass + official IK1 | `43.868067` | `4.699349` | `15.175444` | best TC score |
| TotalCapture test | newik1_v10 Stage C | `44.900650` | `4.754637` | `15.368013` | rejected |

Decision: not selected. `newik1_v10` improves some local `pRJ`/smoothness terms, but the improvement does not translate into better score and the `gR2` output worsens relative to `newpl_v5 + official IK1`. Keep official IK1 for this official-like NewPL v5 route.

## NewPose control v2 FK-leaf implementation

Date: 2026-06-08

Version: `newpose_ctrl_v2_fk_leaf`

Status: completed and not selected.

Why this version exists:

- `newpose_ctrl_v1` was rejected because RRJ/control metrics looked acceptable while decoded FK body geometry collapsed (`FK_joint_L2_cm` around `43-45 cm` versus baseline `4.6-5.0 cm`).
- v2 keeps the control-point output contract but adds losses that directly test whether decoded controls produce GT-like leaf and joint positions.

Contract:

| Item | Value |
|---|---|
| Input | `174D = official IMU[90] + RRB_after_pl[45] + pRB/gR1[18] + last PL control[18] + gR0[3]` |
| Init-only | `offset_r / r_JS` |
| Output | `93D = RRJ_control[90] + gR_pose_control[3]` |
| FK loss frame | root/body frame |
| Leaf vertices | `(1961, 5424, 1176, 4662, 411)` |
| Root vertex | `3021` |
| DIP trans loss | not used |

New implementation pieces:

| Artifact | Path |
|---|---|
| Module/loss | `newpose_ctrl.py` |
| Trainer | `newpose_ctrl_train.py` |
| NewPose eval | `newpose_ctrl_eval.py` |
| Fair baseline module eval | `newpose_baseline_ik2_module_eval.py` |
| Runner | `scripts/run_newpose_ctrl_v2_fk_leaf_official_protocol_20260608.sh` |

Loss preset: `v2_fk_leaf`

- Retains v1 control/state losses.
- Adds `fk_leaf_pos`, `fk_leaf_vel`, `fk_leaf_acc`, and `fk_joint_pos`.
- Selects best checkpoint by `fk_leaf_physical`, not by full-pipeline score or by a convenience weighted loss.
- Uses masked SMPL FK vertices for speed, not full mesh.

Artifacts:

```text
root: data/experiments/newpose_ctrl_v2_fk_leaf_20260608
log: data/experiments/newpose_ctrl_v2_fk_leaf_20260608/logs/run_full.log
summary: data/experiments/newpose_ctrl_v2_fk_leaf_20260608/summary.json
summary tables: data/experiments/newpose_ctrl_v2_fk_leaf_20260608/summary_tables.md
```

Training:

| Stage | Epochs run | Best epoch | Best selection | Status |
|---|---:|---:|---:|---|
| AMASS pretrain | 29 | 19 | `16.734057` | early stopped |
| DIP fine-tune | 11 | 1 | `18.760393` | early stopped |

Module result:

| Dataset | Version | FK leaf L2 cm ↓ | FK joint L2 cm ↓ | State RRJ deg ↓ | Decision |
|---|---|---:|---:|---:|---|
| DIP-IMU test | official_gpnet | `6.234410` | `4.971124` | `9.994508` | baseline |
| DIP-IMU test | newpl_v5_dip + official IK2 | `6.254980` | `4.994349` | `9.979830` | baseline |
| DIP-IMU test | newpose_ctrl_v2 Stage B best | `20.285948` | `14.477565` | `30.793276` | rejected |
| TotalCapture test | newpl_v5_dip + official IK2 | `5.766959` | `4.600485` | `11.047222` | baseline |
| TotalCapture test | newpose_ctrl_v2 Stage A best | `18.867359` | `13.679826` | `25.630533` | rejected |
| TotalCapture test | newpose_ctrl_v2 Stage B best | `18.869814` | `13.694618` | `25.636141` | rejected |

Decision: not selected. v2 is materially better than the rejected v1 failure mode, but it is still much worse than the official/newpl_v5 + official IK2 baselines on decoded FK leaf/joint metrics. DIP fine-tune did not help; the best DIP fine-tune checkpoint is epoch 1. Do not connect this module to full pipeline.

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

## PL-s1: newpl_v7_learned_offset_accaux

Status: diagnostic only, implemented and smoke-tested on 2026-06-12. Not selected.

Contract:

```text
input preserved: official PL 84D aRB[18] + wRB[18] + RRB[45] + gR0[3]
output preserved: pRB[15] + gR1[3]
init changed: init18 = pRL[15] + gR0[3]; external offset_r removed
internal parameter: learned_offset[6,3] = offset_max * tanh(raw_offset), offset_max=0.30 m
acc loss frame: root-frame relative residual, non-root IMU acceleration minus root IMU acceleration
forbidden supervision: DIP trans, DIP root velocity, real DIP/TotalCapture offset GT
```

Implementation:

```text
model: pl_curve.py::PLCurveLearnedOffsetAccAuxModule
helpers: split_legacy_pl_imu_feature, learned_offset_imu_acc_loss, learned_offset_imu_acc_terms
smoke runner: newpl_v7_learned_offset_accaux_smoke.py
artifacts: data/experiments/newpl_v7_learned_offset_accaux_20260612
```

Loss recipe:

```text
Base PL loss: pRB + gR1 + gt_control_pRB + gt_control_gR1 + temporal/smooth terms.
Aux loss: imu_acc_weight * SmoothL1((pRB_ddot + lever_leaf - lever_root) / acc_scale,
                                    (aRB_leaf - aRB_root) / acc_scale)
Prior: offset_prior_weight * mean(offset^2)
Smoke defaults: imu_acc_weight=0.01, offset_prior_weight=0.001, acc_scale=30, offset_max=0.30.
```

Smoke validation:

| Stage | Metric | Value |
|---|---|---:|
| Stage 0 | zero-offset residual | `8.955655 m/s^2` |
| Stage 0 | random-offset residual | `11.304108 m/s^2` |
| Stage 0 | learned-offset residual | `8.928542 m/s^2` |
| Stage 0 | learned improvement vs zero | `0.027113 m/s^2` |
| Stage 1 | frozen NewPL residual before | `15.339169 m/s^2` |
| Stage 1 | frozen NewPL residual after | `15.346781 m/s^2` |
| Stage 1 | pRB/gR1 output drift | `-0.000186 cm / +0.000350 deg` |
| Stage 2 | learned offset norm mean/median/p95 | `0.009579 / 0.008723 / 0.012402 m` |

PL module smoke comparison:

| Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | IMU acc residual ↓ | Notes |
|---|---:|---:|---:|---:|---|
| official PL baseline | `2.402170` | `4.903610` | `6.664704` | `15.211345` | cached official PL |
| newpl_v4_init36 baseline | `2.566700` | `5.360255` | `6.979082` | `23.585226` | historical processed-input checkpoint |
| newpl_v5_dip_best baseline | `2.511445` | `5.193558` | `6.801924` | `24.289366` | official-protocol checkpoint |
| newpl_v7_learned_offset_accaux | `2.509856` | `5.190952` | `6.799719` | `24.048222` | learned offset acc auxiliary |

Decision: diagnostic only. The learned offset is finite and small, and Stage 0 shows a weak residual reduction, but frozen offset-only training does not improve the residual and v7 does not beat the official PL baseline on pRB/gR1. Full-pipeline S4/S5 not measured.

## PL-s1: newpl_v7b_local_accaux

Status: diagnostic only, implemented and smoke-tested on 2026-06-12. Not selected.

Contract:

```text
input preserved: official PL 84D aRB[18] + wRB[18] + RRB[45] + gR0[3]
output preserved: pRB[15] + gR1[3]
init changed: init18 = pRL[15] + gR0[3]; external offset_r removed
internal parameter: learned leaf offset[5,3] = offset_max * tanh(raw_leaf_offset), offset_max=0.30 m
acc loss frame: root frame with root gyro/alpha rotating-frame correction, not root-acceleration subtraction
forbidden supervision: DIP trans, DIP root velocity, real DIP/TotalCapture offset GT
```

Implementation:

```text
model: pl_curve.py::PLCurveLearnedLeafOffsetLocalAccAuxModule
helpers: learned_leaf_offset_local_imu_acc_terms, learned_leaf_offset_local_imu_acc_loss
smoke runner: newpl_v7b_local_accaux_smoke.py
artifacts: data/experiments/newpl_v7b_local_accaux_20260612
gravity sensitivity artifacts: data/experiments/newpl_v7b_local_accaux_20260612_minus_g, data/experiments/newpl_v7b_local_accaux_20260612_plus_g
```

Loss recipe:

```text
Base PL loss: pRB + gR1 + gt_control_pRB + gt_control_gR1 + temporal/smooth terms.
Root rotating-frame correction:
  anchor_acc_R = pRB_ddot + 2*w_root x pRB_dot + alpha_root x pRB + w_root x (w_root x pRB)
Leaf lever term:
  offset_acc_R = alpha_leaf x r_leaf + w_leaf x (w_leaf x r_leaf)
Aux loss:
  imu_acc_weight * SmoothL1((anchor_acc_R + offset_acc_R) / acc_scale, aRB_leaf / acc_scale)
Prior: offset_prior_weight * mean(leaf_offset^2)
Smoke defaults: imu_acc_weight=0.005, offset_prior_weight=0.001, acc_scale=30, offset_max=0.30, gravity_mode=none.
```

Smoke validation:

| Stage | Metric | Value |
|---|---|---:|
| Stage 0 | zero-offset local acc residual | `8.974846 m/s^2` |
| Stage 0 | random-offset local acc residual | `11.928852 m/s^2` |
| Stage 0 | init36 GT-offset local acc residual | `9.755350 m/s^2` |
| Stage 0 | learned-offset local acc residual | `8.963714 m/s^2` |
| Stage 0 | learned improvement vs zero | `0.011132 m/s^2` |
| Stage 1 | frozen NewPL residual before | `15.081704 m/s^2` |
| Stage 1 | frozen NewPL residual after | `14.994161 m/s^2` |
| Stage 1 | pRB/gR1 output drift | `+0.000251 cm / +0.000093 deg` |
| Stage 2 | learned leaf offset norm mean/median/p95 | `0.014525 / 0.011973 / 0.019684 m` |

Gravity-mode sensitivity:

| gravity_mode | zero | random | learned | GT offset | Stage1 improvement |
|---|---:|---:|---:|---:|---:|
| none | `8.974846` | `11.928852` | `8.963714` | `9.755350` | `0.087543` |
| minus_gR0 | `13.902242` | `17.051386` | `13.878829` | `14.238059` | `0.018126` |
| plus_gR0 | `13.978133` | `15.994410` | `13.964892` | `14.994678` | `0.037277` |

PL module smoke comparison:

| Version | pRB L1 cm ↓ | pRB L2 cm ↓ | gR1 angle deg ↓ | local acc residual ↓ | Notes |
|---|---:|---:|---:|---:|---|
| official PL baseline | `2.402170` | `4.903610` | `6.664704` | `15.179327` | cached official PL |
| newpl_v4_init36 baseline | `2.566700` | `5.360255` | `6.979082` | `23.832706` | historical checkpoint |
| newpl_v5_dip_best baseline | `2.511445` | `5.193558` | `6.801924` | `24.416668` | official-protocol checkpoint |
| newpl_v7_rootrel_accaux | `2.509856` | `5.190952` | `6.799719` | `24.186819` | previous root-relative accaux |
| newpl_v7b_local_accaux | `2.509855` | `5.190949` | `6.799719` | `24.196295` | local accaux, full-pipeline not measured |

Decision: diagnostic only. v7b is a better physical formulation than v7 because root gyro/alpha are used as rotating-frame correction instead of treating root acceleration as the comparison target. The current smoke still does not justify selection: learned residual reduction is tiny, GT offset is not better than zero under this proxy, and pRB/gR1 remain worse than the official PL baseline. Next evidence should come from an FK/RBDL acceleration audit before any long AMASS -> DIP training.

### v7b AMASS long diagnostic follow-up

Status: completed on 2026-06-12. This is a longer AMASS module-level diagnostic, not full AMASS -> DIP and not full-pipeline evaluation.

Artifacts:

```text
root: data/experiments/newpl_v7b_local_accaux_20260612_longtrain
summary: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/summary.md
json: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/summary.json
checkpoint: data/experiments/newpl_v7b_local_accaux_20260612_longtrain/checkpoints/stage2_tiny_joint.pt
```

Run setup:

```text
max_sequences=512
max_train_sequences=512
batch_size=96
window=61
stage0_steps=300
stage1_steps=300
stage2_epochs=50
imu_acc_weight=0.005
gravity_mode=none
```

Offset/acc diagnostic:

| Metric | Value |
|---|---:|
| Stage 0 zero local acc residual | `11.099609 m/s^2` |
| Stage 0 random local acc residual | `12.552015 m/s^2` |
| Stage 0 init36 GT-offset local acc residual | `11.629623 m/s^2` |
| Stage 0 learned local acc residual | `11.071591 m/s^2` |
| Stage 0 learned improvement vs zero | `0.028018 m/s^2` |
| Stage 1 frozen residual before -> after | `24.562115 -> 24.461527 m/s^2` |
| Stage 2 offset norm mean/median/p95 | `0.073905 / 0.076368 / 0.092299 m` |

Same-cache AMASS module comparison:

| Version | pRB L2 cm ↓ | gR1 angle deg ↓ | local acc residual ↓ | Notes |
|---|---:|---:|---:|---|
| official PL baseline | `3.284492` | `10.588459` | `11.674987` | cached official PL |
| newpl_v4_init36 baseline | `3.298557` | `10.437762` | `28.840921` | historical checkpoint |
| newpl_v5_dip_best baseline | `3.328090` | `10.252261` | `25.495672` | official-protocol checkpoint |
| newpl_v7_rootrel_accaux | `3.327924` | `10.253457` | `25.443991` | previous root-relative accaux |
| newpl_v7b_local_accaux long | `3.330573` | `10.174911` | `24.781027` | local accaux long diagnostic |

Decision update: continue to a proper AMASS -> DIP module-level experiment is reasonable, but v7b is still not selected. It improves local acceleration residual versus v5/v7 and improves gR1, but pRB is slightly worse than official/v4/v5 on this AMASS same-cache table. Full-pipeline S4/S5 and DIP test are still not measured.

## PL-s1: newpl_v5_realtime_smooth_residual

Status: diagnostic only, completed on 2026-06-12. Not selected.

Contract:

```text
input changed for NewPL only:
  aRB_smooth[18] + aRB_residual_raw_minus_smooth[18] + wRB[18] + RRB[45] + gR0[3] = 102D
output preserved:
  pRB[15] + gR1[3] = 18D
downstream IK/VR/physics contract:
  not changed; module-level evaluation only
```

Implementation:

```text
constants/helpers: pl_curve.py::PL_SMOOTH_RESIDUAL_INPUT_SIZE, causal_iir_lowpass_sequence, pl_smooth_residual_sequence_features
cache path: pl_curve_cache.py --feature-mode smooth_residual
checkpoint remap: pl_curve_train.py partial 84D -> 102D input.weight initialization
runner: scripts/run_newpl_v5_realtime_residual_20260612.sh
summary: scripts/summarize_newpl_v5_realtime_residual.py
artifacts: data/experiments/newpl_v5_realtime_residual_20260612_full_causal_iir20
```

Input feature semantics:

```text
filter: causal_iir, cutoff_hz=20, fs=60, lookahead_frames=0
aRB_smooth: causal low-pass root-frame acceleration
aRB_residual: raw root-frame acceleration - aRB_smooth
reason: keep the real-time smoothing benefit while exposing the removed high-frequency content to the network instead of discarding it
```

Training recipe:

```text
AMASS pretrain: 80 epoch cap, early stopped, batch_size=512, lr=1e-4, best_epoch=51
DIP fine-tune: 40 epochs, batch_size=64, lr=5e-6, best_epoch=40
selection_metric: control_physical = gt_control_pRB + gt_control_gR1
loss terms: gt_control_pRB 0.3, gt_control_gR1 0.1, gR1_dot 0.03, gR1_ddot 0.001, pRB_ddot_smooth 1e-6
forbidden supervision: DIP trans/root velocity/global trajectory not used
full-pipeline 11 metrics: not measured
```

Module-level validation:

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
| DIP test after DIP FT | realtime smooth+residual | `6.557889` | `14.557139` | improves same-cache official gR1, slightly worsens same-cache official pRB |
| TC test after DIP FT | official PL | `6.768144` | `14.014337` | baseline |
| TC test after DIP FT | raw newpl_v5_dip_best | `6.780749` | `13.415189` | historical reference from `newpl_v5_official_protocol_20260607_tuned`; not same-cache fairness row |
| TC test after DIP FT | realtime smooth+residual | `6.638172` | `13.736756` | improves same-cache official pRB/gR1; historical raw-v5 row is context only |

Per-leaf readout after DIP fine-tune:

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

Decision: do not select. The real-time smooth+residual idea is implementation-valid and helps same-cache official TotalCapture pRB plus DIP/TC gR1. However, the imported raw official-protocol `newpl_v5_dip_best` rows are historical references from a different metric namespace, not a same-cache fairness baseline. Continue only as a filter/loss ablation branch until raw v5, official PL, and realtime smooth+residual are re-evaluated on exactly the same cache/protocol.

## PL-s1: newpl_v6_gR1nextonly_smoothacc

Status: diagnostic candidate, completed on 2026-06-13. Not selected for full pipeline.

Contract:

```text
module variant: newpl_v6_next_control
version prefix: newpl_v6_gR1nextonly_smoothacc
input: smoothacc cache -> legacy 84D PL feature with init36
current output contract: pRB[15] + gR1[3] = 18D
auxiliary outputs: v6 next-control branch for control/preview diagnostics
training protocol: AMASS pretrain -> DIP-IMU train fine-tune -> DIP/TotalCapture test module eval
TotalCapture fine-tune: not run in this official-style rerun
DIP trans/root velocity: not used
full-pipeline 11 metrics: not measured
```

Implementation:

```text
runner wrapper: scripts/run_newpl_v6_gR1_nextonly_smoothacc_20260613.sh
base runner: scripts/run_newpl_v6_next_control_smoothacc_gR1_20260613.sh
summary script: scripts/summarize_newpl_v6_next_control_smoothacc_gR1.py
root: /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full
summary json: /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/summary.json
run log: /tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/logs/run_full.log
```

Loss weights:

| Term | Weight | Notes |
|---|---:|---|
| current pRB | `1.0` | preserved ordinary pRB fitting |
| current gR1 | `1.0` | preserved ordinary gR1 fitting |
| gt_control_pRB / gt_control_gR1 | `0.3 / 0.2` | current control-point supervision |
| pRB_dot / pRB_ddot_smooth | `0.03 / 0.000001` | current pRB dynamics |
| gR1_dot / gR1_ddot | `0.03 / 0.001` | current gR1 dynamics |
| next_pRB / next_gt_control_pRB | `0.0 / 0.0` | disabled |
| next_pRB_vel / next_pRB_acc | `0.0 / 0.0` | disabled |
| last_control_pRB / tail4_control_pRB | `0.0 / 0.0` | disabled |
| next_gR1 / next_gt_control_gR1 | `2.0 / 0.5` | enabled |
| next_gR1_vel / next_gR1_acc | `0.05 / 0.002` | enabled |
| last_control_gR1 / tail4_control_gR1 | `0.5 / 0.35` | enabled |

Training command:

```bash
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

Training best checkpoints:

| Stage | Best current gR1 epoch/value | Best current module epoch/value | Checkpoint |
|---|---:|---:|---|
| AMASS pretrain | `80 / 13.497158` | `71 / 5.256350` | `/tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/amass_pretrain/best_current_gR1.pt` |
| DIP fine-tune | `40 / 19.419542` | `40 / 5.414863` | `/tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/dip_finetune/best_current_gR1.pt` |

Fair same-cache current-frame output comparison:

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

Temporal/smooth readout after DIP fine-tune:

| Dataset | Version | pRB vel L2 cm/s ↓ | pRB acc L2 cm/s² ↓ | gR1 vel vector L2 ↓ |
|---|---|---:|---:|---:|
| DIP test | official PL | `40.244087` | `2684.142116` | `0.677733` |
| DIP test | newpl_v4 init36 | `40.243880` | `2666.394326` | `0.673141` |
| DIP test | newpl_v6 gR1-only | `66.402182` | `2691.549734` | `0.548376` |
| TC test | official PL | `34.038251` | `2099.110771` | `0.715926` |
| TC test | newpl_v4 init36 | `32.694543` | `1801.389069` | `0.690698` |
| TC test | newpl_v6 gR1-only | `57.844750` | `761.011162` | `0.383529` |

Decision:

```text
The experiment achieved the intended gR1 improvement on DIP and TotalCapture.
DIP after DIP fine-tune: gR1 12.474146 deg versus official 12.902106 and v4 12.722391.
TotalCapture after DIP fine-tune: gR1 12.848388 deg versus official 13.170870 and v4 13.075061.
pRB is not fully recovered: DIP pRB L2 6.392231 is worse than official 6.345701 and v4 6.349507; TC pRB L2 7.430918 beats official 7.508986 but remains worse than v4 7.119541.
This is a useful gravity-specialized branch, not a selected PL replacement yet.
Next-control for pRB should stay disabled unless a later loss can recover pRB and pRB temporal velocity.
Do not connect to IK/full pipeline until a pRB-preserving gR1 variant or a small IK ablation proves the gravity gain is worth the pRB/temporal cost.
```

### Downstream gR1-only swap validation

Status: completed on 2026-06-14.

This validation tests whether the accurate `gR1` branch helps downstream when `pRB` is held fixed. It does not introduce a new checkpoint; it evaluates the existing DIP-fine-tuned `best_current_gR1` checkpoint inside official GPNet / hybrid PL.

Contract:

| Item | Value |
|---|---|
| Baseline | official/baseline PL `pRB[15]+gR1[3]` |
| Hybrid | official/baseline PL `pRB[15]` + `newpl_v6_gR1nextonly_smoothacc gR1[3]` |
| Checkpoint | `/tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/dip_finetune/best_current_gR1.pt` |
| Raw official eval | `data/experiments/newpl_v6_gR1_only_swap_20260614/eval/*.json` |
| Smoothacc eval | `/tmp/globalpose_hybrid_prb_base_gr1_v6_20260613/full/*.json` |
| Summary | `data/experiments/newpl_v6_gR1_only_swap_20260614/summary.json` |
| GT oracle | none |
| DIP trans/root velocity | not used |

Full-pipeline validation:

| Dataset/protocol | Official score ↓ | Hybrid score ↓ | Delta ↓ | L angle delta | G angle delta | G joint delta cm | Joint jitter delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| DIP raw official | `44.642049` | `45.089955` | `+0.447906` | `+0.171669` | `+0.096508` | `+0.216007` | `-0.004184` |
| TotalCapture raw official | `44.477380` | `44.546482` | `+0.069101` | `+0.124946` | `-0.013503` | `+0.131201` | `-0.082612` |
| DIP smoothacc | `44.233519` | `44.709970` | `+0.476451` | `+0.178521` | `+0.088901` | `+0.204033` | `-0.003891` |
| TotalCapture smoothacc | `45.293466` | `45.451494` | `+0.158027` | `+0.162359` | `+0.004417` | `+0.121283` | `-0.077978` |

Decision update: do not promote this branch to IK/full-pipeline. The module-level `gR1` improvement does not survive downstream by itself. In all four protocols, fixing `pRB` and swapping only v6 `gR1` worsens the full-pipeline score, mainly through local angle and joint-error regressions. The small jitter reduction is not enough to compensate.

## Version: imu_neighbor_vel_ctrl_v1

### 1. Purpose

Add an independent module-level diagnostic that predicts world-frame velocity control curves for the two skeleton nodes adjacent to each non-root IMU, plus root velocity for the pelvis/root IMU. This does not replace PL/IK1 and does not run full-pipeline 11 metrics.

### 2. Contract

| Item | Value |
|---|---|
| Module | `imu_neighbor_vel_ctrl_v1` |
| Input | `aM[18] + wM[18] + RMB_6d[36] + r_JS[18] = 90D` |
| Output | `neighbor_vel_W_control[33]` |
| Frame | world/model frame `W` |
| `r_JS` | IMU origin relative to mapped joint `J`, expressed in joint-local coordinates |
| DIP policy | no DIP world velocity/root velocity/acceleration GT; no DIP trans finite-difference |

Output node groups:

| Sensor | Mapped joint | Nodes | Channels |
|---|---:|---|---:|
| left_forearm | `18` | `18,20` | `6` |
| right_forearm | `19` | `19,21` | `6` |
| left_lowerleg | `4` | `4,7` | `6` |
| right_lowerleg | `5` | `5,8` | `6` |
| head | `15` | `12,15` | `6` |
| pelvis/root | `0` | `0` | `3` |

### 3. Loss

AMASS/TotalCapture with reliable translation use velocity control, decoded velocity, decoded acceleration, root velocity, root acceleration, segment relative velocity/acceleration, smoothness, jerk, and control-prior terms. DIP uses no world GT terms; only teacher distill plus smooth/jerk/prior is allowed if adaptation is run.

### 4. Implementation Status

| Item | Value |
|---|---|
| Files | `imu_neighbor_vel_ctrl.py`, `imu_neighbor_vel_ctrl_train.py`, `imu_neighbor_vel_ctrl_eval.py` |
| Compile | passed |
| Train efficiency | default compact precompute of 90D inputs and velocity/acc/control targets; epochs slice cached tensors instead of rebuilding FK targets per batch |
| Baseline velocity eval | full `pose_baseline/tran_baseline` finite-difference baseline when present; otherwise 3D `v_root_vr` as root-only official VR baseline; otherwise `baseline velocity not available` |
| Smoke train/eval | passed on `/tmp/imu_neighbor_vel_ctrl_verify_precompute_3505953` synthetic cache |
| DIP GT guard | passed; eval reports `world velocity GT not available` |
| Formal AMASS/TC/DIP run | not measured |
| Full-pipeline 11 metrics | not measured |
| Selected | no; implementation only, needs real-cache training/eval |
