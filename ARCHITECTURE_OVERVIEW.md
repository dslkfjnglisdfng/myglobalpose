# GlobalPose Architecture Overview

This document is a code-checked architecture guide for the current GlobalPose project state. It explains the official GlobalPose baseline, the added L4 module, the H1/J1/K1 branches, and the data/cache/training/evaluation flow without requiring code reading.

Scope of this document:

- Architecture only, not a new experiment log.
- No training or evaluation was started while writing this document.
- `test.py`, `MotionEvaluator`, official weights, official test datasets, and final test splits were not modified.

## 1. Frames, Sensors, and Main Tensors

### Sensor Set

GlobalPose uses 6 sparse IMUs. The code-level IMU vertex IDs are:

```text
v_imu = (1961, 5424, 1176, 4662, 411, 3021)
```

The synthetic AMASS generator associates these sensors with SMPL joints:

```text
IMU_JOINTS = (18, 19, 4, 5, 15, 0)
```

Sensor index `5` is the pelvis/root IMU in `GPNet.forward_frame`.

### Official Input Fields

The official real/synthetic data path stores raw-style IMU fields:

| field | shape per sequence | meaning in code |
|---|---:|---|
| `RIM` | `[6,3,3]` or sequence-level list item | IMU/magnetic-world calibration rotation used by the official conversion path |
| `RIS` | `[T,6,3,3]` | sensor orientation sequence before `RSB` correction |
| `RSB` | `[6,3,3]` | fixed sensor-to-body correction used by official conversion |
| `aS` | `[T,6,3]` | raw/sensor-frame acceleration-like vector before model-frame conversion |
| `wS` | `[T,6,3]` | raw/sensor-frame angular velocity before model-frame conversion |
| `mS` | `[T,6,3]` when present | raw/sensor magnetic vector |

The baseline input to `GPNet.forward_frame` is the converted triplet:

| field | shape per frame | sequence shape | meaning |
|---|---:|---:|---|
| `aM` | `[6,3]` | `[T,6,3]` | IMU acceleration in the model/GlobalPose input frame, with gravity `[0,-9.8,0]` added by the official conversion |
| `wM` | `[6,3]` | `[T,6,3]` | IMU angular velocity in the model/GlobalPose input frame |
| `RMB` | `[6,3,3]` | `[T,6,3,3]` | IMU orientation matrix used by GlobalPose as model/body-frame sensor orientation |

The conversion used in `test.py` and `l4_generate_baseline_cache.py` is:

```text
RMB = RIM.transpose(1, 2) @ RIS @ RSB
aM  = RIM.transpose(1, 2) @ RIS @ aS + [0, -9.8, 0]
wM  = RIM.transpose(1, 2) @ RIS @ wS
```

The code uses row-vector style multiplications in several places, for example `a.mm(R[5])`. This document therefore describes frames by the exact code contract rather than asserting a separate external transform convention.

## 2. Official GlobalPose Baseline

The true online inference order is `GPNet.forward_frame(a, w, R)`:

```text
input aM/wM/RMB
-> PL-s1
-> IK-s1
-> IK-s2
-> optional L4 pose/q75 refiner
-> recompute dependent tensors if L4 changed pose
-> VR-s1
-> optional L4 velocity refiner
-> velocity fusion
-> carticulate physics optimization
-> final pose/tran
```

### 2.1 Root-IMU Relative Pre-Features

Before PL-s1:

```text
aRB = aM @ RMB[5]
wRB = wM @ RMB[5]
RRB = RMB[5].T @ RMB[:5]
gR0 = -RMB[5, 1]
```

Shapes:

| tensor | shape | meaning |
|---|---:|---|
| `aRB` | `[6,3]` | all IMU accelerations expressed relative to the root/pelvis IMU orientation convention used by the code |
| `wRB` | `[6,3]` | all IMU angular velocities in the same root-IMU-relative convention |
| `RRB` | `[5,3,3]` | relative orientations from root/pelvis IMU to the other 5 IMUs |
| `gR0` | `[3]` | gravity direction estimate derived from the root/pelvis IMU orientation |

### 2.2 PL-s1

Code module:

```text
self.plnet = RNNWithInit(input_size=84, output_size=18, hidden_size=512, num_rnn_layer=3, dropout=0.4)
```

Input:

```text
concat(aRB.ravel, wRB.ravel, RRB.ravel, gR0)
= 18 + 18 + 45 + 3 = 84D
```

Output:

```text
x = 18D
x[:15]  -> pRB
x[15:]  -> gR1, normalized to unit vector
```

Physical meaning:

- `pRB` is a 15D intermediate relative body/leaf position feature used by IK-s1.
- `gR1` is a refined gravity direction. It corrects the previous `RRB` by `from_to_rotation_matrix(gR0, gR1)`.

Downstream dependency:

- IK-s1 uses updated `RRB`, `gR1`, and `pRB`.

### 2.3 IK-s1

Code module:

```text
self.iknet["net1"] = RNN(input_size=63, output_size=72, hidden_size=512, num_rnn_layer=3, dropout=0.4)
```

Input:

```text
concat(RRB.ravel, gR1, pRB)
= 45 + 3 + 15 = 63D
```

Output:

```text
x = 72D
x[:69] -> pRJ
x[69:] -> gR2, normalized to unit vector
```

Physical meaning:

- `pRJ` is a 69D relative joint-position feature for 23 non-root SMPL joints: `23 * 3 = 69`.
- `gR2` is a second refined gravity direction.
- `RRB` is updated again by `from_to_rotation_matrix(gR1, gR2)`.

Downstream dependency:

- IK-s2 uses updated `RRB`, `gR2`, and `pRJ`.
- VR-s1 later uses `gR2` and a recomputed `pRJ`.

### 2.4 IK-s2

Code module:

```text
self.iknet["net2"] = RNN(input_size=117, output_size=90, hidden_size=512, num_rnn_layer=3, dropout=0.4)
```

Input:

```text
concat(RRB.ravel, gR2, pRJ)
= 45 + 3 + 69 = 117D
```

Output:

```text
x = 90D = 15 reduced joints * 6D rotation representation
```

Reduced and ignored joints:

```text
j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)
```

Rotation conversion:

```text
RRJ = r6d_to_rotation_matrix(x)  # shape [15,3,3]
```

Full pose recovery:

1. Create `glb_pose = eye(3)` for all 24 joints.
2. Put the 15 reduced global rotations into `glb_pose[:, j_reduce]`.
3. Run `body_model.inverse_kinematics_R(glb_pose)` to get local SMPL pose.
4. Set ignored joints to identity.
5. Recompute non-root joint positions:

```text
pRJ = body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
```

Root orientation recovery:

```text
pose[0] = RMB[5] @ from_to_rotation_matrix(gR2, gR0)
```

Output:

| tensor | shape | meaning |
|---|---:|---|
| `RRJ` | `[15,3,3]` | reduced joint global rotations after 6D-to-matrix conversion |
| `pose` | `[24,3,3]` | full local SMPL pose, with ignored joints set to identity and root restored from root IMU/gravity |
| `pRJ` | `[23,3]` | non-root joint positions from SMPL FK, root-relative/body-local in the model convention |

### 2.5 VR-s1

Code module:

```text
self.vrnet = RNNWithInit(input_size=243, output_size=9, hidden_size=512, num_rnn_layer=3, dropout=0.4)
```

Before VR-s1:

```text
aRB = aM @ pose[0]
wRB = wM @ pose[0]
```

Input:

```text
concat(RRJ.ravel, pRJ.ravel, aRB.ravel, wRB.ravel, gR2)
= 135 + 69 + 18 + 18 + 3 = 243D
```

Output:

```text
x = 9D
x[0]   -> vRR_V
x[1:4] -> vRR_H
x[4:9] -> stationary logits
```

Interpretation:

- `vRR_V`: scalar vertical root velocity component.
- `vRR_H`: 3D root-frame horizontal velocity vector before world/root orientation recovery.
- `stationary_prob = sigmoid(x[4:9])`: 5 probabilities for contact/stationary joints.

Root/world velocity reconstruction:

```text
vWR = pose[0] @ vRR_H
vWR[1] = vRR_V
```

`vWR` is the network root velocity estimate used by velocity fusion.

### 2.6 Velocity Fusion

Contact joints:

```text
j_contact = (0, 10, 11, 22, 23)
```

The current contact joint positions are:

```text
cjoint = cat(root_zero, pRJ @ pose[0].T)[j_contact]
```

Stationary weight:

```text
stationary_weight = (stationary_prob * 5 - 3).clip(0, 1)
```

Velocity fusion:

```text
velocity =
  (stationary_weight @ (last_cjoint - cjoint) / dt + beta_velocity * vWR)
  / (beta_velocity + stationary_weight.sum())
```

Physical meaning:

- `vWR` is the VR-s1 predicted root velocity.
- `last_cjoint - cjoint` is a contact-derived root displacement estimate when stationary/contact joints should stay fixed in world space.
- `stationary_weight` gates how much contact correction affects root velocity.
- The fused `velocity` is used as the translation tracking target inside carticulate physics.

### 2.7 Carticulate Physics

On the first frame:

```text
physics_model.set_state_R(pose, initial_tran, zeros(75))
```

On later frames, the physics optimizer:

1. Reads current physics state: `pose_cur`, `tran_cur`, `qdot`.
2. Computes contact joint positions, velocities, Jacobians, mass matrix, and inverse dynamics.
3. Builds stable-PD desired pose acceleration `thetaddotdes`.
4. Builds contact/root translation target from fused `velocity` and contact joints.
5. Solves least-squares dynamics/QP-related systems for `qddot`.
6. Explains residual root force through contact forces when possible.
7. Updates the carticulate dynamic model state.

Final output:

```text
refined_pose, refined_tran, qdot = physics_model.get_state_R()
return refined_pose, refined_tran
```

Upstream variables that affect physics:

- IK pose from `IK-s2`;
- optional L4-refined pose;
- VR-s1 `vWR`;
- optional L4-refined velocity;
- `stationary_prob`;
- `pRJ` and contact joint positions;
- contact history `last_cjoint` and `self.contact`.

## 3. Current L4 Mainline Module

Current active method:

```text
H1: L=4 streaming Euler-q tail-update pose+velocity refiner
```

Important: this is one module with two call points, not two independent modules.

Call points in `GPNet.forward_frame`:

1. After IK-s2 and before VR-s1: refine pose/q75.
2. After VR-s1 and before velocity fusion: refine `vWR`.

Current retained H1 configuration:

| setting | value |
|---|---:|
| pose input mode | `euler_q75` |
| pose feature dim | 75 |
| IMU feature dim | 90 |
| total frame input dim | 165 |
| hidden size | 256 |
| tail length | 4 |
| residual scale | 0.005 |
| velocity residual scale | 0.0 |
| checkpoint | `data/experiments/l4_ablation_H1_velocity_scale0_tc_v1/best.pt` |

Since `velocity_residual_scale=0.0`, H1's most stable retained version changes pose/q75 but does not change root velocity. The velocity branch is still called, but its output delta is zero.

### 3.1 q75 State Definition

`q75` is an Euler generalized coordinate vector:

```text
q75 = root translation 3D
    + root Euler orientation 3D
    + 23 non-root local Euler joints 69D
    = 75D
```

Conversion functions:

```text
pose_tran_to_q75(pose [T,24,3,3], tran [T,3]) -> [T,75]
q75_to_pose_tran(q75 [T,75]) -> pose [T,24,3,3], tran [T,3]
```

`qdot` and `qddot` in L4 are finite-difference/spline derivatives of Euler generalized coordinates. They are not physical angular velocities.

Root translation residual:

- `freeze_root_translation=True` by default.
- Any residual component in `q75[:3]` is set to zero.
- L4 does not directly refine root translation in the retained H1/J1/K1 paths.

Root orientation:

- `q75[3:6]` can be refined.
- The output pose root orientation can therefore change through `q75_to_pose_tran`.

### 3.2 StreamingTailUpdateQState

File:

```text
l4_tail_update_qstate.py
```

Frame input feature:

```text
prephysics_feature =
  pose_input_feature + aM.ravel + wM.ravel + RMB.ravel
```

H1:

```text
q75_prephysics 75D + aM 18D + wM 18D + RMB 54D = 165D
```

J1/K1:

```text
rot6d pose 144D + aM 18D + wM 18D + RMB 54D = 234D
```

Internal state:

| item | meaning |
|---|---|
| `hidden` | GRUCell hidden state `[batch, hidden_size]`; default zero unless K1 hidden-init is used |
| `control_buffer` | history of refined spline control q states |
| `base_buffer` | matching history of original prephysics q states |
| `velocity_buffer` | short history of VR root velocities for the velocity residual branch |

Step logic:

1. Project feature through `Linear(n_input, hidden_size)` and ReLU.
2. Update `hidden` with `GRUCell`.
3. Predict current `new_control` residual:

```text
new_delta = new_control_head(hidden) * residual_scale
new_control = base_q_t + frozen_root(new_delta)
```

4. Predict tail updates for the previous `L=4` controls:

```text
tail_delta = tail_delta_head(hidden).reshape(batch, 4, 75) * residual_scale
```

5. Update the last up to 4 control states plus append current control.
6. Decode control and base buffers through a uniform cubic B-spline.
7. Output the current residual:

```text
residual_t = q_control[current] - q_base[current]
q_t = base_q_t + residual_t
```

Outputs:

| output | shape per frame | meaning |
|---|---:|---|
| `q_t` | `[1,75]` | refined q75 |
| `qdot_t` | `[1,75]` | spline derivative of q control state |
| `qddot_t` | `[1,75]` | spline second derivative of q control state |
| `residual_t` | `[1,75]` | q75 residual applied to current frame |
| `new_delta_norm` | scalar | magnitude diagnostic for new control update |
| `tail_delta_norm` | scalar | magnitude diagnostic for tail updates |

`residual_scale` controls the magnitude of both the new control residual and the tail update residual. H1/J1/K1 use `0.005`.

### 3.3 L4PrePhysicsRefiner

File:

```text
l4_tail_update_qstate.py
```

Purpose:

- Bridge `GPNet.forward_frame` tensors and `StreamingTailUpdateQState`.
- Convert pose/tran to `q75`, build the L4 feature, run the L4 q-state model, and convert refined `q75` back to pose/tran.

Inputs:

| input | shape | source |
|---|---:|---|
| `pose` | `[24,3,3]` | IK-s2 full local SMPL pose |
| `prephysics_tran` | `[3]` | current physics translation if initialized, otherwise zeros |
| `a` | `[6,3]` | current `aM` |
| `w` | `[6,3]` | current `wM` |
| `R` | `[6,3,3]` | current `RMB` |

Process:

```text
pose/tran -> q75
q75 + pose + a/w/R -> prephysics_feature
StreamingTailUpdateQState.step(feature, q75)
q_refined -> pose_refined/tran_refined
```

After pose is refined, `GPNet.forward_frame` must recompute dependent tensors before VR-s1:

```text
pose_body = pose with root set to identity
glb_pose_body, joint_body = body_model.forward_kinematics(pose_body)
RRJ = glb_pose_body[j_reduce]
pRJ = joint_body[1:]
root_from_imu = RMB[5].T @ pose[0]
gR2 = root_from_imu.T @ gR0
aRB = aM @ pose[0]
wRB = wM @ pose[0]
```

Why recomputation is required:

- VR-s1 consumes `RRJ`, `pRJ`, `aRB`, `wRB`, and `gR2`.
- If pose changes but these tensors remain from the old IK-s2 pose, VR-s1 receives internally inconsistent pose/IMU features.
- That inconsistency would contaminate root velocity, contact probabilities, velocity fusion, and physics.

### 3.4 L4 Velocity Residual Path

The same `StreamingTailUpdateQState` module also owns a velocity residual branch:

```text
velocity_feature = concat(hidden, last_4_v_root_vr.flatten, stationary_prob)
delta_v = velocity_delta(velocity_feature) * velocity_residual_scale
v_root_refined = v_root_vr + delta_v
```

Inputs:

| input | shape | meaning |
|---|---:|---|
| `v_root_vr` / `vWR` | `[3]` | VR-s1 root velocity before fusion |
| `stationary_prob` | `[5]` | VR-s1 contact/stationary probabilities |
| `hidden` | `[hidden_size]` | same GRU hidden used by q refinement |
| `velocity_buffer` | up to `[4,3]` | recent VR root velocities |

H1 sets:

```text
velocity_residual_scale = 0.0
```

Therefore:

- `delta_v_root = 0`;
- `v_root_refined = v_root_vr`;
- the module still runs the velocity-refine call for logging/contract consistency, but does not change velocity.

H1 uses this because earlier velocity residual variants increased risk in global/jitter behavior. The retained stable version only changes pose/q75.

## 4. J1 and K1 Branches

### 4.1 H1 vs J1 vs K1 Summary

| branch | active status | pose input | frame dim | offset use | velocity residual | losses added | result |
|---|---|---|---:|---|---:|---|---|
| H1 | retained candidate | Euler `q75_prephysics` | 165 | none | 0.0 | no contact/IMU/offset active weights | kept |
| J1 | rejected as active | full 24-joint 6D rotation representation | 234 | none | 0.0 | no IMU proxy, no offset | Local improves, Global/Mesh/Jitter regress |
| K1 | rejected as active | same as J1 rot6d | 234 | hidden init from `offset_r` | 0.0 | no IMU proxy, no contact, no offset loss | Local further improves, Global/Mesh/Jitter further regress |

### 4.2 J1: rot6d Pose Input

Original H1 input:

```text
q75_prephysics 75D + aM/wM/RMB 90D = 165D
```

J1 input:

```text
pose_prephysics [24,3,3]
-> first two columns per joint
-> 24 * 6 = 144D
144D + aM/wM/RMB 90D = 234D
```

Important terminology:

- This is a 6D rotation representation.
- It is not a full 6D rotation matrix.
- It is not a full `3x3` rotation matrix.

Output contract:

- unchanged q75 residual/refinement;
- still predicts refined `q75`, converts back to pose/tran;
- target/loss still use `q75_gt`, `pose_gt`, `tran_gt`, and related terms.

Experiment controls:

- sensor offset input: disabled;
- IMU position offset augmented datasets: excluded;
- IMU proxy loss: disabled;
- offset loss: disabled;
- velocity residual scale: `0.0`;
- residual scale: `0.005`.

Observed result:

- TotalCapture Local metrics improved versus H1.
- TotalCapture Global SIP/Angle/Joint/Mesh and Root/Joint Jitter worsened versus H1.
- DIP showed a similar tradeoff pattern.
- J1 is not the active method.

### 4.3 K1: rot6d + Offset Hidden Initialization

K1 is based on J1.

Per-frame input:

```text
rot6d pose 144D + aM/wM/RMB 90D = 234D
```

The per-frame input dimension is unchanged from J1.

Offset field:

| field | shape | meaning |
|---|---:|---|
| `imu_offset_r` / `r_JS` | `[6,3]` per sequence | joint-local IMU installation position offset |
| `offset_r` | `[6,3]` per sequence | same data copied into enriched neural-only cache |

Offset semantics:

- sequence-level constant;
- position offset only, not full SE(3);
- joint-local;
- installation/mounting offset, not accelerometer/gyroscope measurement bias;
- not copied into every frame as input.

Hidden initialization:

```text
offset_r [6,3]
-> flatten 18D
-> Linear(18, hidden_size)
-> ReLU
-> Linear(hidden_size, hidden_size)
-> h0 [hidden_size]
```

Initialization strategy:

- final offset-encoder layer is zero-initialized;
- output is scaled by `offset_init_scale=0.1`;
- used once at sequence reset through `reset_stream(offset_r)`.

Experiment controls:

- no offset augmented dataset training;
- no IMU proxy loss;
- no contact loss;
- no offset loss;
- no per-frame offset concatenation.

Observed result:

- TotalCapture Local metrics further improved versus J1.
- TotalCapture Global SIP/Angle/Joint/Mesh and Root/Joint Jitter further worsened versus J1/H1.
- K1 is not the active method.

## 5. Data and Cache Flow

### 5.1 Synthetic AMASS Flow

```text
Raw AMASS .npz
-> process_amass_globalpose.py
   - resample/convert AMASS pose/tran to 60 FPS
   - align AMASS frame to SMPL/GlobalPose frame
   - run SMPL FK
   - synthesize GlobalPose-style sparse IMU
   - save processed shards
-> l4_generate_baseline_cache.py
   - run frozen official GlobalPose neural PL/IK/VR path
   - save neural-only prephysics cache
-> L4 AMASS loss-only training
   - train on q/pose/velocity losses
   - no MotionEvaluator physics validation
   - best.pt selected by training/cache loss only
-> real-data fine-tune/ablation
```

AMASS processed shard fields from `process_amass_globalpose.py`:

| field | shape per sequence | meaning |
|---|---:|---|
| `name` | string | source sequence name |
| `pose` | `[T,72]` | SMPL local axis-angle pose |
| `tran` | `[T,3]` | SMPL root translation |
| `joint` | `[T,24,3]` | SMPL FK joints |
| `v_imu` | `[T,6,3]` | SMPL mesh vertices at IMU attachment vertices |
| `aM` | `[T,6,3]` | synthesized model-frame acceleration |
| `wM` | `[T,6,3]` | synthesized model-frame angular velocity |
| `RMB` | `[T,6,3,3]` | synthesized model/body-frame IMU orientation |
| `RIM`, `RIS`, `RSB` | raw-style orientation fields | compatibility with official conversion path |
| `aS`, `wS`, `mS` | raw-style sensor fields | compatibility with official conversion path |
| `gender`, `shape` | metadata | AMASS body metadata |

### 5.2 Real DIP / TotalCapture Flow

```text
DIP / TotalCapture processed data
-> official train/val/test split preparation
-> l4_generate_baseline_cache.py neural-only cache
-> fine-tune / ablation training
-> validation MotionEvaluator on val split
-> held-out final test
```

Current policy:

- final test is not used for tuning;
- official weights and test datasets are not modified;
- if a dataset/cache includes offset augmentation and the experiment does not explicitly require it, it is excluded.

### 5.3 Processed Dataset Fields

Observed TotalCapture official train/val processed fields:

| field | example shape | status |
|---|---:|---|
| `pose` | list item `[T,72]` | present |
| `tran` | list item `[T,3]` | present |
| `RIM` | list item `[6,3,3]` | present |
| `RIS` | list item `[T,6,3,3]` | present |
| `RSB` | list item `[6,3,3]` | present |
| `aS` | list item `[T,6,3]` | present |
| `wS` | list item `[T,6,3]` | present |
| `mS` | list item, when saved | present in observed TotalCapture |
| `aM/wM/RMB` | may be absent in official raw-style splits | computed by cache generator if raw-style fields exist |
| `joint` | varies by source | not present in observed TotalCapture official train/val |
| `v_imu` | varies by source | not present in observed TotalCapture official train/val |

### 5.4 Neural-Only Prephysics Cache Fields

Observed TotalCapture neural-only cache fields:

| field | shape per sequence | meaning |
|---|---:|---|
| `name` | string | sequence name |
| `num_frames` | int | sequence length |
| `pose_gt` | `[T,24,3,3]` | ground-truth SMPL pose as rotation matrices |
| `tran_gt` | `[T,3]` | ground-truth translation |
| `q75_gt` | `[T,75]` | ground-truth q75 |
| `q75_prephysics` | `[T,75]` | frozen official PL/IK prephysics pose converted to q75 |
| `pose_prephysics` | `[T,24,3,3]` | frozen official PL/IK prephysics pose |
| `v_root_vr` | `[T,3]` | VR-s1 root velocity before velocity fusion |
| `stationary_prob` | `[T,5]` | VR-s1 contact/stationary probabilities |
| `aM` | `[T,6,3]` | model-frame IMU acceleration |
| `wM` | `[T,6,3]` | model-frame IMU angular velocity |
| `RMB` | `[T,6,3,3]` | model/body-frame IMU orientation |
| `pose_baseline` | `[T,24,3,3]` | full official GlobalPose baseline output when saved |
| `tran_baseline` | `[T,3]` | full official GlobalPose baseline translation when saved |
| `q75_baseline` | `[T,75]` | baseline pose/tran converted to q75 when saved |

J1/H1 use the no-offset neural-only cache.

K1 uses an enriched copy that adds:

| field | shape per sequence | meaning |
|---|---:|---|
| `offset_r` | `[6,3]` | sequence-level joint-local IMU installation position offset copied from processed `imu_offset_r` / `r_JS` |

### 5.4.1 K2 Paired Original/Offset-Augmented AMASS Cache

K2 is an executed experimental data/training branch. It uses two AMASS views for the same source sequence, but the executed version is a lightweight overlay cache: it reuses old no-offset neural-only pose/prephysics/target fields and overlays current offset-augmented acceleration plus `imu_offset_r/r_JS`. It does not rerun official GlobalPose prephysics.

```text
source AMASS pose/tran
-> original view
   - old synthetic aM/wM/RMB from the no-offset neural-only cache
   - original/default sequence-level offset_r
   - shared pose_gt/tran_gt/q75_gt target
-> offset_aug view
   - current offset-augmented aM from resynthesized r_JS
   - wM/RMB preserved from the old neural cache after consistency checks
   - current sequence-level imu_offset_r/r_JS
   - shared pose_gt/tran_gt/q75_gt target
```

The earlier smoke paired cache lives at:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_amass_paired_offset_aug_neural_only_smoke/
```

The executed K2 overlay cache lives at:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
```

Executed overlay result:

| item | value |
|---|---:|
| original/offset pairs | `649` |
| records | `1298` |
| frames across records | `1118012` |
| skipped sequences | `20`, all due to strict `wM_diff_too_large` near `1e-4` |
| `aM` mean norm diff, mean / max over sequences | `3.5920 / 10.8336` |
| `wM` max-abs diff, mean / max over sequences | `1.92e-5 / 9.97e-5` |
| `RMB` max-abs diff | `0.0` |
| offset norm mean / median / p90 / max | `0.1755 / 0.1812 / 0.2327 / 0.2852 m` |

Each K2 overlay cache record stores:

| field | shape per record | meaning |
|---|---:|---|
| `name` | string | view-qualified sequence name |
| `source_name` | string | original AMASS sequence name |
| `view_type` | `original` or `offset_aug_overlay` | training view identifier |
| `pair_id` | string | key linking the two views |
| `q75_prephysics` | `[T,75]` | frozen prephysics q state reused from old neural cache |
| `pose_prephysics` | `[T,24,3,3]` | frozen prephysics pose reused from old neural cache |
| `v_root_vr` | `[T,3]` | VR root velocity reused from old neural cache |
| `stationary_prob` | `[T,5]` | stationary/contact signal reused from old neural cache |
| `aM` | `[T,6,3]` | original or offset-augmented acceleration view |
| `wM` | `[T,6,3]` | old-cache angular velocity; consistency checked against augmented shard |
| `RMB` | `[T,6,3,3]` | old-cache IMU orientation; consistency checked against augmented shard |
| `q75_gt` | `[T,75]` | shared supervision target |
| `pose_gt` | `[T,24,3,3]` | shared supervision target |
| `tran_gt` | `[T,3]` | shared supervision target |
| `offset_r` | `[6,3]` | sequence-level joint-local installation position offset |
| `num_frames` | int | frame count |

K2 frame input remains the J1/K1 rot6d frame contract:

```text
rot6d pose 144D + aM/wM/RMB 90D = 234D
```

K2 RNN initialization input is:

```text
offset_r 18D
+ first-frame rot6d pose 144D
+ first-frame aM/wM/RMB 90D
= 252D
-> Linear(252, hidden_size)
-> ReLU
-> Linear(hidden_size, hidden_size)
-> h0 [hidden_size]
```

K2 does not concatenate offset to each frame, does not predict offset, and does not enable offset/IMU-proxy/contact loss. Pair consistency exists only as a default-off interface with `pair_consistency_weight=0.0`.

K2 TotalCapture result on S4 validation is partial positive: it improves the scalar validation score versus H1/J1/K1 and reduces J1/K1 global regression, but still trails the GlobalPose baseline on Global Joint/Mesh and jitter. It is not final-test ready as a retained method.

### 5.4.2 TotalCapture Roffset Sidecar Cache

The TotalCapture Roffset ablation is a sidecar/enriched neural-only cache for the L4 path only. It does not overwrite official TotalCapture processed data and does not change the frozen GlobalPose PL/IK/VR inputs.

Sidecar cache paths:

```text
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_B/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_B/baseline_cache_manifest.json
```

Record contract:

| field | shape per sequence | meaning |
|---|---:|---|
| `aM`, `wM`, `RMB` | `[T,6,3]`, `[T,6,3]`, `[T,6,3,3]` | original TotalCapture IMU features used by frozen GlobalPose |
| `l4_aM`, `l4_wM`, `l4_RMB` | same as above | L4-only IMU feature override for the Roffset ablation |
| `aM_orig`, `wM_orig`, `RMB_orig` | same as above | explicit original copies for audit |
| `aM_Roffset`, `wM_Roffset`, `RMB_Roffset` | same as above | selected Roffset corrected stream |
| `aM_RT_offset`, `wM_RT_offset`, `RMB_RT_offset` | same as above | transpose-direction candidate stream |
| `imu_offset_R` / `R_JS` | `[6,3,3]` | sequence-level estimated orientation installation offset |
| `offset_r` / `imu_offset_r` | `[6,3]` | sequence-level estimated position installation offset |
| `orientation_offset_mode` | string | `Roffset_A` or `Roffset_B` |

Roffset candidates:

```text
Official: RMB = RIM^T RIS RSB
Roffset A: RSB_corr = R_JS^T, RMB_A = RIM^T RIS RSB_corr
Roffset B: RSB_corr = R_JS,   RMB_B = RIM^T RIS RSB_corr
```

In this ablation, `aM` and `wM` are not directly rotated by `R_JS` because official `aM=RIM^T RIS aS + g` and `wM=RIM^T RIS wS` are already model/world-frame vectors. The sidecar stores corrected `l4_RMB` and original `l4_aM/l4_wM`.

Roffset A/B produced effectively identical TotalCapture S4 metrics to the K2 original-input control, so this orientation-offset correction is not retained.

### 5.5 Sensor-Offset Fields

Offset fields observed in TotalCapture official train/val processed data:

| field | shape per sequence | meaning |
|---|---:|---|
| `imu_offset_r` | `[6,3]` | joint-local IMU installation position offset |
| `r_JS` | `[6,3]` | alias for `imu_offset_r`; position of sensor origin relative to joint, expressed in joint-local coordinates |
| `imu_offset_R` | `[6,3,3]` | joint-local installation rotation component |
| `R_JS` | `[6,3,3]` | alias for `imu_offset_R` |
| `imu_offset_T` | `[6,4,4]` | full homogeneous installation transform |
| `T_JS` | `[6,4,4]` | alias for `imu_offset_T` |

Current active method H1:

- does not use any sensor offset field;
- does not use offset-augmented datasets.

J1:

- does not use any sensor offset field.

K1:

- uses only `offset_r` / `r_JS` as sequence-level hidden-state initialization;
- does not use `R_JS` or `T_JS`;
- rejected as active method.

## 6. Training and Evaluation Flow

### 6.1 AMASS Loss-Only Training

Scripts:

- `l4_train_full_loss.py`
- earlier AMASS experiment folders under `data/experiments/l4_prephysics_pose_velocity_*`

Characteristics:

- Uses neural-only cache records.
- Trains L4 on q/pose/velocity/smoothness losses.
- Does not run full carticulate MotionEvaluator validation during training.
- `best.pt` is selected by training/cache loss.
- It is not by itself method-success evidence.

### 6.2 DIP / TotalCapture Fine-Tune and Ablation

Scripts:

- `l4_train_diverse_short.py`
- `l4_train_loss_ablation.py`

Characteristics:

- Load cached prephysics records.
- Initialize from a previous L4 checkpoint when requested.
- Train one cached sequence/window per optimizer step.
- `window=61` in H1/J1/K1.
- MotionEvaluator validation can run during training and select `best.pt` by lowest full-validation score.
- DIP protocol can disable root velocity/root translation supervision when GT translation is unreliable.
- TotalCapture can use root velocity supervision from GT translation.

### 6.3 Loss Ablation Summary

Observed ablation families from local `train_result.json` files:

| family | role | outcome summary |
|---|---|---|
| A / A2 | contact-mode TotalCapture ablation with `residual_scale=0.005`, `velocity_residual_scale=0.005`; contact weights effectively zero in recorded configs | not retained over H1 |
| D0 | control/check ablation at same scale | not retained |
| E1 / E2 | smaller residual scales `0.0025` / `0.00375` | worse TotalCapture validation scores |
| F1 | stronger baseline preservation variant | not retained |
| G1 / G2 | FK baseline root-relative variants | not retained |
| H1 | `velocity_residual_scale=0.0`, `residual_scale=0.005` | retained current active candidate |
| H2 / H3 | H-family variants with residual or longer training changes | not current retained method |
| I1 | q-root-orientation weight variant | not current retained method |
| J1 | rot6d pose input | Local improves, Global/Mesh/Jitter regress; rejected |
| K1 | rot6d + offset hidden init | Local further improves, Global/Mesh/Jitter further regress; rejected |

### 6.4 MotionEvaluator Metrics

`MotionEvaluator.names`:

```text
L SIP Err (deg)
L Angle Err (deg)
L Joint Err (cm)
L Vertex Err (cm)
G SIP Err (deg)
G Angle Err (deg)
G Joint Err (cm)
G Vertex Err (cm)
Root Jitter (km/s^3)
Joint Jitter (km/s^3)
```

Local vs Global:

- Global metrics evaluate pose and translation with root/global motion preserved.
- Local metrics set root pose to identity before evaluation and focus on body-local pose quality.
- The current J1/K1 pattern is: Local improves, but Global/root/mesh/jitter regress.

Metric aliases used in experiment notes:

| note name | MotionEvaluator name |
|---|---|
| Local SIP | `L SIP Err (deg)` |
| Local Angle | `L Angle Err (deg)` |
| Local Joint | `L Joint Err (cm)` |
| Local Mesh | `L Vertex Err (cm)` |
| Global SIP | `G SIP Err (deg)` |
| Global Angle | `G Angle Err (deg)` |
| Global Joint | `G Joint Err (cm)` |
| Global Mesh | `G Vertex Err (cm)` |
| Root Jitter | `Root Jitter (km/s^3)` |
| Joint Jitter | `Joint Jitter (km/s^3)` |

## 7. Complete Text Flow Diagram

```text
Input per frame:
  aM [6,3], wM [6,3], RMB [6,3,3]
  active in baseline/H1/J1/K1

-> Root-IMU relative pre-features
   input:
     aM, wM, RMB
   output:
     aRB [6,3], wRB [6,3], RRB [5,3,3], gR0 [3]
   active:
     baseline/H1/J1/K1

-> PL-s1
   input:
     aRB 18D + wRB 18D + RRB 45D + gR0 3D = 84D
   output:
     pRB 15D, gR1 3D
   active:
     baseline/H1/J1/K1

-> IK-s1
   input:
     RRB 45D + gR1 3D + pRB 15D = 63D
   output:
     pRJ 69D, gR2 3D
   active:
     baseline/H1/J1/K1

-> IK-s2
   input:
     RRB 45D + gR2 3D + pRJ 69D = 117D
   output:
     reduced 6D rotations 90D -> RRJ [15,3,3]
     full pose [24,3,3] through inverse_kinematics_R
     pRJ [23,3] through FK
   active:
     baseline/H1/J1/K1

-> Optional L4 pose/q75 refiner
   input H1:
     q75_prephysics 75D + aM/wM/RMB 90D = 165D
   input J1:
     rot6d pose 144D + aM/wM/RMB 90D = 234D
   input K1:
     same 234D frame input as J1
     plus sequence-level offset_r [6,3] only for hidden initialization
   output:
     refined q75 [75], refined pose [24,3,3]
   active/default:
     baseline: absent
     H1: active retained candidate
     J1: rejected
     K1: rejected

-> Recompute dependent tensors if L4 changed pose
   input:
     refined pose, aM, wM, RMB, gR0
   output:
     RRJ [15,3,3], pRJ [23,3], gR2 [3], aRB [6,3], wRB [6,3]
   active:
     H1/J1/K1 when L4 pose residual is nonzero

-> VR-s1
   input:
     RRJ 135D + pRJ 69D + aRB 18D + wRB 18D + gR2 3D = 243D
   output:
     vRR_V scalar, vRR_H [3], stationary_prob [5]
     vWR [3]
   active:
     baseline/H1/J1/K1

-> Optional L4 velocity refiner
   input:
     vWR [3], stationary_prob [5], L4 hidden [256], recent velocity buffer up to [4,3]
   output:
     v_root_refined [3], delta_v_root [3]
   active/default:
     baseline: absent
     H1/J1/K1: called, but velocity_residual_scale=0.0, so delta_v_root=0

-> Velocity fusion
   input:
     vWR or v_root_refined [3]
     stationary_prob [5]
     cjoint [5,3]
     last_cjoint [5,3]
   output:
     fused velocity [3]
   active:
     baseline/H1/J1/K1

-> Carticulate physics
   input:
     target pose [24,3,3]
     fused velocity [3]
     stationary/contact state
     current physics state
   output:
     final pose [24,3,3], final tran [3]
   active:
     baseline/H1/J1/K1

-> MotionEvaluator
   input:
     predicted pose/tran and ground-truth pose/tran
   output:
     Local/Global SIP, Angle, Joint, Mesh, Root Jitter, Joint Jitter
   active:
     validation only; final test not used for tuning
```

## 8. Current Status

Current active method:

```text
H1: L=4 streaming Euler-q tail-update pose+velocity refiner
pose input: euler_q75
frame input dim: 165
residual_scale: 0.005
velocity_residual_scale: 0.0
sensor offset: disabled
IMU proxy loss: disabled
offset loss: disabled
```

Rejected/deprecated branches:

- J1: rot6d pose input, rejected because Global/Mesh/Jitter regress despite Local gains.
- K1: rot6d + offset hidden init, rejected because Global/Mesh/Jitter regress further.
- IMU proxy loss branches: not active.
- Contact loss branches with nonzero contact weights: not active.
- IMU position offset augmented datasets: not used for current active method.

Next architecture-level recommendation:

- Keep H1 as the reference active method.
- Treat J1/K1 as diagnostic evidence: rot6d helps local pose but currently harms global/root dynamics.
- Any future architecture change should explicitly target Global/Mesh/Jitter, not just Local pose loss.
