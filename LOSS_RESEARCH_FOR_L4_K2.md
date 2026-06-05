# Loss Research for L4/K2

Date: 2026-05-29

Scope: document/code reading plus loss research only. I did not train, modify model code, touch official weights, modify `test.py`/`MotionEvaluator`, or run S5.

## 1. Current project problem

The official GlobalPose baseline is:

```text
aM / wM / RMB -> PL-s1 -> IK-s1 -> IK-s2 -> pose -> VR-s1 -> velocity fusion -> physics
```

The external K2/L4 prototype is:

```text
IK-s2 pose -> K2/L4 refiner -> refined q75 / pose / qdot / qddot -> VR-s1 / physics
```

The mature K2/L4 contract uses rot6d pose input, not 6D rotation matrices. Per frame it uses 144D rot6d pose + 90D IMU feature = 234D. The IMU feature is `aM` 18D + `wM` 18D + `RMB` 54D. RNN init is `r_JS` 18D + first-frame rot6d pose 144D + first-frame IMU 90D = 252D. `r_JS` has useful evidence; `R_JS` is not proven, because the Roffset sidecar produced only a small response.

The current problem is a three-way trade-off:

- recent-L4/K2 improves Local metrics but still worsens Global Mesh and Jitter against the GlobalPose baseline.
- Continuing L4-loss improves Global Joint/Mesh a little but hurts Local, Jitter, and Score.
- qdot-aware physics adapter P2 reduces Root/Joint Jitter but hurts Local/Angle/Score.

Representative S4 evidence:

- `l4_K2_TC_recentL4loss_10ep_v1`: score `42.159255`; Local improves, Global Mesh and Jitter worsen.
- `l4_K2_recentL4_continue_L4loss_10ep_v1`: original physics score `42.261841`; Local remains slightly better than baseline, Global Joint slightly better, but Global Mesh and Jitter worsen.
- Same checkpoint with `l4_pip_v2`: score `42.346873`; Root Jitter improves from baseline `0.297783` to `0.239058`, Joint Jitter improves from `0.495126` to `0.480032`, but Local/Global pose metrics degrade. Mean root-velocity delta is `0.1648`, max `1.5027`, so the current 0.5 qdot blend is too blunt.

Conclusion: the next loss design should separate local-pose preservation, root-relative kinematics, global/root velocity, and contact/physics signals. Stronger global FK/mesh pressure or direct qdot-to-root-velocity blending is likely to keep reproducing the same conflict.

## 2. Survey summary table

| Paper / project | Modality | Predicted outputs | Loss terms / optimization terms | Code availability | Useful? | Notes |
|---|---|---|---|---|---|---|
| TransPose | 6 IMU | leaf joints, full joints, pose, velocity | staged MSE; local spline variant uses value, velocity, acceleration, geodesic, control prior, preserve | local repo `/home/lingfeng/projects/TransPose` | High | Derivative losses are small relative to pose/value. |
| PIP | 6 IMU + physics | leaf/full joints, global 6D pose, joint velocity, contact | dynamics uses pose/joint PD, joint velocity target, contact/friction constraints, torque regularization | local repo `/home/lingfeng/projects/paper_code/PIP` | High | Use velocity/contact as physics targets, not direct pose overwrite. |
| PNP | 6 IMU + non-inertial dynamics | pose/velocity/contact feeding PIP-like dynamics | PIP-like optimizer; paper handles non-inertial root-frame acceleration | local repo `/home/lingfeng/projects/PNP` | Medium-high | Warning for IMU acceleration proxy. |
| GlobalPose Physics 2025 | 6 IMU + 3D contacts | pose, root velocity, stationary probability, contacts, forces, torques | separated pose, translation, physics modules; gravity/contact reasoning | local paper + current repo | High | Supports separating Local pose from global physics. |
| TIP | 6 IMU transformer | pose q, root XY/Z, contact/residual | pose 6D MSE, root XY/Z weighted separately, contact BCE, residual regression, jerk | local repo `/home/lingfeng/projects/paper_code/transformer-inertial-poser` | High | Contact and root should be explicit terms. |
| MobilePoser | sparse IMU | reduced pose | pose MSE, optional FK MSE, jerk `1e-5` | local repo | Medium | Jerk is a tiny regularizer. |
| IMUPoser | sparse IMU | pose | pose MSE/L1, optional FK joint loss | local repo | Low-medium | Too simple for current physics conflict. |
| UltraInertialPoser | sparse IMU + dynamics | pose/translation/dynamics | no clear training loss found in local clone | local repo | Low | Insufficient loss evidence. |
| SPIN | image HMR | SMPL pose/shape, joints, vertices | 2D keypoint, pelvis-centered 3D keypoint, vertices, SMPL pose/shape, depth prior, SMPLify teacher gating | GitHub inspected | High | Strong reference for teacher/preserve and root-relative separation. |
| VIBE | video HMR | temporal SMPL pose/shape/camera | 2D/3D keypoints, SMPL pose/shape, adversarial motion prior | GitHub inspected | Medium | Motion prior concept useful, adversarial term too big for next step. |
| PARE / CLIFF | image HMR | SMPL pose/shape/camera | expected HMR-like; exact raw loss paths returned 404 in this pass | not verified | Low now | Do not use as code evidence yet. |
| mmhuman3d | HMR toolkit | SMPL/keypoints/translation | shape prior, pose reg, joint-angle prior, smooth joints/pelvis/translation | GitHub raw inspected | Medium | Useful taxonomy of small priors/smoothness. |
| HuMoR | RGB/RGB-D/3D + motion prior | SMPL motion, rollout joints, contacts, floor | 2D/3D joints, priors, joint consistency, bone length, smoothness, contact velocity/height, floor | local repo `/home/lingfeng/projects/paper_code/humor` | High | Best reference for gated contact velocity/height. |

## 3. Detailed notes by method

### TransPose

Local repo: `/home/lingfeng/projects/TransPose`

The original family uses staged sparse-IMU supervision for leaf joints, full joints, pose, and translation/velocity. The local repo also contains spline/dynamics variants. `PoseS3SplineCriterion` uses SmoothL1 value, velocity, acceleration, geodesic, control prior, and optional preserve/position preserve. Representative weights are value `1`, velocity `0.01`, acceleration `0.001`, geodesic `0.05`, control prior `0.001`. `DynamicsLoss` uses q `1`, qdot `0.05`, qddot `0.005`, FK joint `0.1`, optional IMU acceleration.

Transfer: use derivative losses as small regularizers. Preserve losses are appropriate for guarding a good baseline. qdot/qddot should be treated as generalized-coordinate finite-difference targets, not physical angular velocity/acceleration.

### PIP

Local repo: `/home/lingfeng/projects/paper_code/PIP`

PIP predicts leaf joints, full joints, global 6D pose, all-joint velocity, and foot contact. The local clone does not include a complete network training script, so exact training loss weights are not verified. The dynamics code does show the important interface: pose/joint PD, joint position/velocity PD, contact constraints, friction/signorini-style terms, and torque regularization. An IMU acceleration block exists but is disabled in the local code.

Transfer: velocity and contact should be separate physics targets. Do not directly force qdot into root velocity with a large blend. PIP itself mainly improves translation/plausibility; pose/SIP may not improve.

### PNP

Local repo: `/home/lingfeng/projects/PNP`

PNP is PIP-like but emphasizes non-inertial effects when using root-frame acceleration. This matters because GlobalPose-style root-relative IMU acceleration is not automatically an inertial target.

Transfer: the current `imu_proxy_offset_acc` should remain low-priority and robust/ramped if used. Strong acceleration reconstruction can easily teach the wrong thing if frame, gravity, offset, and non-inertial terms are not fully validated.

### GlobalPose Physics 2025

Local paper: `/home/lingfeng/projects/论文/markdown_converted/04_physics_dynamics_optimization_GlobalPose_Physics_2025.md`

The paper separates pose estimation, translation estimation, and physics optimization. It refines root-frame gravity in pose estimation, estimates root velocity/stationary probability, then uses contacts/physics for global motion. This is directly relevant to the current L4 issue: Local pose should be guarded by local/root-relative losses, while global translation/jitter should be handled by root velocity, contact, and physics interface terms.

### TIP

Local repo: `/home/lingfeng/projects/paper_code/transformer-inertial-poser`

The code uses pose 6D MSE scaled high, root XY and Z losses with separate weights, contact BCE plus residual regression, and jerk on pose. Transferable idea: root/global and contact should be explicit, separate objectives. Do not copy raw constants, because scales are representation-specific.

### MobilePoser and IMUPoser

MobilePoser uses pose MSE, optional FK joint MSE, and a small jerk regularizer around `1e-5`. IMUPoser uses pose regression plus optional FK. These confirm that FK and jerk are common, but they do not solve the current Local/Global/physics conflict by themselves.

### SPIN and VIBE

SPIN uses confidence-weighted 2D keypoints, pelvis-centered 3D keypoints, vertex loss, SMPL pose/shape losses, depth prior, and valid SMPLify teacher/pseudo-label gating. The key transferable concept is teacher/preserve with root-relative separation.

VIBE adds temporal sequence training and an adversarial motion prior on SMPL pose. This is useful as a long-term idea, but not for the next L4 ablation because it would introduce a new prior system and obscure the current loss-interface diagnosis.

### mmhuman3d

GitHub raw code for `prior_loss.py` and `smooth_loss.py` shows shape prior, pose regularization, joint-angle prior, and smooth joints/pelvis/translation. These are useful as small priors and taxonomy, but less targeted than baseline preserve and root-relative velocity for our current module.

### HuMoR

Local repo: `/home/lingfeng/projects/paper_code/humor`

HuMoR fitting loss includes robust 2D joints, 3D joints, vertices/points, pose/shape priors, motion/init priors, joint consistency between direct SMPL joints and rollout joints, bone length consistency, joint smoothness, contact velocity, contact height, floor regularization, and overlap consistency. The contact velocity formula is essentially:

```text
sum_t,j contact_conf[t,j] * ||joints[t,j] - joints[t-1,j]||^2
```

Transfer: contact velocity should be gated and local to contacting feet/joints. Contact height/floor is riskier until the floor convention is verified.

## 4. Loss taxonomy

Pose/rotation loss: geodesic rotation, 6D rotation MSE/SmoothL1, SMPL pose parameter loss. For K2/L4, keep `pose_geodesic`, `q_body`, `q_root_ori`, and add explicit local preserve.

q/SMPL parameter loss: SmoothL1/MSE on q, residual prior, edge losses. qdot/qddot are generalized-coordinate derivatives, not physical angular velocity/acceleration.

Joint/FK loss: root-relative 3D joints, global joints, pelvis-centered HMR keypoints. For K2/L4, root-relative FK should dominate; avoid strong global FK.

Mesh/vertex loss: common in HMR, but not recommended next because it can pull broad pose corrections and hurt Local if root/body separation is not precise.

Velocity loss: finite-difference joint velocity, root velocity, teacher velocity. For K2/L4, root-relative joint velocity is high priority; keep root velocity weak/off in pose fine-tuning.

Acceleration/jerk loss: GT acceleration matching or smoothness. Keep jerk tiny; do not use large jerk as the main jitter fix.

Contact/foot loss: contact BCE, contact-gated foot velocity, contact height/floor, hard physics constraints. Start with small contact-gated foot velocity only after L2 is stable.

Root/global loss: translation/root velocity, axis-separated root losses, gravity-aware orientation. Attack Global Mesh/Jitter through root velocity/contact consistency, not strong body-pose FK.

IMU proxy/reconstruction loss: reconstruct sensor orientation/gyro/acceleration from FK/RBDL and offsets. Current offset-acc raw losses are huge, so use only as low-weight robust diagnostic later.

Physics/dynamics loss: PD consistency, torque/action regularization, contact/friction, ZMP/equilibrium, motion priors. For now prefer bounded adapter consistency over direct physics output overwrite.

Distillation/preserve loss: teacher/pseudo-label gating and baseline preserve. This is one of the most important tools for preventing Local regression.

Multi-frame/window loss: full-window or recent-window temporal supervision. Keep `recent_l4` last 4 frames for the next ablations to isolate loss changes.

## 5. Recommendation for next loss ablation

Use `data/experiments/l4_K2_TC_recentL4loss_10ep_v1/best_loss.pt` as the starting checkpoint for the first set, not the later damaging continuation, unless explicitly testing recovery from that continuation.

### Priority 1: L1 balanced local preserve + reduced FK/global pressure

Purpose: protect Local/Angle while preserving the useful recent-L4 signal.

Formula:

```text
L = L_pose_geodesic
  + L_q_body
  + 0.5 L_q_root_ori
  + w_pres_body geodesic(R_pred_body, R_base_body)
  + w_pres_root geodesic(R_pred_root, R_base_root)
  + w_fk_rootrel SmoothL1(FK_rootrel(pred), FK_rootrel(gt))
  + existing small qdot/qddot/edge/residual/tail/jerk
  + existing baseline_velocity
```

Suggested weights:

- `local_pose_preserve_body`: `1.0-2.0`
- `local_pose_preserve_root_ori`: `0.1-0.25`
- `baseline_body`: `0` if explicit preserve is used, or at most `1.0`
- `baseline_root_ori`: `0-1.0`, not `5.0`
- `fk_joint_rootrel`: `0.03-0.08`
- `root_velocity`: `0`
- Keep global FK/mesh, contact, IMU proxy off

Data: TotalCapture first. Use AMASS later only if TC S4 is stable.

S4 standard: Local Angle/Joint/Vertex should not worsen versus `l4_K2_TC_recentL4loss_10ep_v1`; Global Mesh should not get worse; Jitter should remain neutral.

### Priority 2: L2 = L1 + root-relative joint velocity consistency

Purpose: reduce body-motion jitter without changing global translation semantics.

Formula:

```text
L_L2 = L_L1
     + w_jvel SmoothL1(
         (FK_rootrel(pred)[t] - FK_rootrel(pred)[t-1]) / dt,
         (FK_rootrel(gt)[t]   - FK_rootrel(gt)[t-1]) / dt
       )
```

Suggested weights:

- `rootrel_joint_velocity`: `0.02-0.10`
- `qdot_consistency_body`: `0`
- contact and IMU proxy off
- physics adapter original/off

S4 standard: Joint Jitter should improve or stay neutral while Local remains within L1 tolerance.

### Priority 3: L3 = L2 + body-only qdot finite-difference consistency

Purpose: make qdot match the q trajectory before using it in physics.

Formula:

```text
L_qdot_cons_body =
  SmoothL1(wrap(q_body[t] - q_body[t-1]), dt * qdot_body[t])

L_L3 = L_L2 + w_qdot_cons L_qdot_cons_body
```

Suggested weights:

- `qdot_consistency_body`: `0.1-0.5`; consider `1.0` only if raw scale stays tiny and gradients are ineffective
- `qdot_consistency_all`: `0`
- P2 physics adapter off

S4 standard: same as L2, plus qdot consistency logs should move in the intended direction. Do not call this a physics win until a later adapter test validates motion metrics.

### Secondary 4: contact-gated foot velocity

Purpose: reduce foot skating/root jitter through stance constraints.

Formula:

```text
foot_world_vel[t,f] ~= d(FK_rootrel_foot(pred)[t,f]) / dt + v_root_refined[t]
L_contact = sum gate[t,f] ||foot_world_vel[t,f]||^2 / max(sum gate, 1)
```

Suggested weights:

- `contact_foot_velocity`: `0.001-0.01`
- `contact_foot_height`: `0` until floor convention is validated

Only try after L2 is stable. Wrong gates can pin the wrong foot and hurt pose.

### Secondary 5: bounded P2b physics adapter

Purpose: recover some P2 jitter reduction without its Local/Global damage.

Later design, not part of this no-code task:

```text
v_root_adapt = v_vr + alpha * gate * clamp(qdot_root_translation - v_vr, max_delta)
```

Suggested ranges:

- `alpha`: `0.05-0.15`, not `0.5`
- gate by contact/stationary confidence and/or VR confidence
- cap velocity delta norm

Test as eval/inference adapter first. Train it only if S4 improves Root Jitter without worse Global Mesh/Joint or Score.

## 6. Papers I need you to download

None required before the next loss ablation. The strongest needed evidence came from local code/paper notes and accessible GitHub code.

Optional later:

- PARE and CLIFF exact loss code/supplements, because raw paths attempted in this pass returned 404.
- PhysCap / SimPoE / PHC details only if we decide to build a full physics-learning objective instead of a bounded GlobalPose adapter.

## 7. Evidence block

Project files read:

- `PROJECT_STATUS.md`
- `PROJECT_DOC_BACKUP.md`
- `net.py` via project documentation and CodeGraph context
- `l4_tail_update_qstate.py`
- `l4_train_loss_ablation.py`
- `l4_q75_utils.py`
- `l4_velocity_losses.py`
- `l4_physics_adapter.py`
- `l4_physics_adapter_eval.py`
- `curve_state_decoder.py`
- recent K2 / recent-L4 / L4-loss / physics adapter / IMU proxy experiment JSON and JSONL artifacts under `data/experiments/`

Research surveyed:

- TransPose, PIP, PNP, GlobalPose Physics 2025, TIP, MobilePoser, IMUPoser, UltraInertialPoser, SPIN, VIBE, PARE/CLIFF attempted, mmhuman3d, HuMoR, and physics/motion-imitation methods at taxonomy level.

Repositories with actual loss/optimization code inspected:

- `/home/lingfeng/projects/TransPose`
- `/home/lingfeng/projects/paper_code/PIP`
- `/home/lingfeng/projects/PNP`
- `/home/lingfeng/projects/paper_code/transformer-inertial-poser`
- `/home/lingfeng/projects/MobilePoser-main`
- `/home/lingfeng/projects/paper_code/IMUPoser`
- `/home/lingfeng/projects/paper_code/humor`
- GitHub SPIN and VIBE code paths inspected earlier in this research pass
- GitHub mmhuman3d `prior_loss.py` and `smooth_loss.py`

Most suitable losses for K2/L4:

- baseline local pose preserve / teacher distillation
- root-relative FK with reduced weight
- root-relative joint velocity consistency
- body-only qdot finite-difference consistency
- small contact-gated foot velocity
- bounded/gated physics adapter consistency later

Not recommended now:

- strong global FK/mesh supervision on L4 pose
- strong IMU acceleration reconstruction loss
- contact height/floor loss before floor validation
- adversarial motion prior or policy/torque learning
- direct `qdot_velocity_blend=0.5` root velocity replacement

Recommended next ablations:

1. L1 balanced local preserve + reduced FK/global pressure.
2. L2 add root-relative joint velocity.
3. L3 add body-only qdot consistency.
4. Contact-gated foot velocity after L2 stability.
5. Bounded P2b adapter after L3 evidence.

Document path: `LOSS_RESEARCH_FOR_L4_K2.md`.
