# GlobalPose Implementation Plan and Review

## ACTIVE SUMMARY

Current stage: K2 AMASS paired-overlay training and TotalCapture Roffset validation completed.

Current task: Compare K2 AMASS paired offset training plus TotalCapture original-input fine-tune against TotalCapture R_JS orientation-offset A/B ablations.

Review state: Implemented, validation pending for broader interpretation. AMASS K2 training and TotalCapture S4 validation completed; DIP validation was not run.

Current changed files: `net.py`, `l4_q75_utils.py`, `l4_tail_update_qstate.py`, `l4_train_diverse_short.py`, `l4_train_loss_ablation.py`, `l4_generate_k2_overlay_cache.py`, `l4_generate_totalcapture_orientation_offset_cache.py`, `l4_generate_paired_offset_aug_cache.py`, generated K2 overlay caches, generated TotalCapture Roffset sidecar caches, and generated K2 experiment artifacts.

Current module: `StreamingTailUpdateQState` L4 tail update/refinement module.

Current experiment: `K2_amass_paired_offset_overlay_rot6d_init_20ep_v1`, `K2_tc_original_input_control_20ep_v2`, `K2_tc_Roffset_A_20ep_v1`, `K2_tc_Roffset_B_20ep_v1`.

Current result: K2 overlay cache produced 649 original/offset pairs and 1298 view records. AMASS K2 20-epoch training completed. TotalCapture S4 validation score improved versus H1/J1/K1 in the scalar selection score (`42.1916` vs H1 `42.2851`, J1 `42.2495`, K1 `42.2962`), mainly by reducing J1/K1 global regression. Roffset A/B produced effectively identical S4 metrics to the original-input K2 control. A follow-up input-path audit confirmed `l4_RMB` did enter the L4 feature path, but the trained L4 residual response was extremely small; this is not evidence that `R_JS` is intrinsically invalid.

Current dropout status: K2/J1/K1 historical runs used no dropout. A new default-off feature/dropout implementation has been added, and a 1-epoch/4-train/1-val S4 smoke ablation found no meaningful trend; it is a wiring smoke only, not a full S4 result.

Current blocker: none for the completed K2/TC runs. Remaining uncertainty: the learned K2 gain is small, AMASS training loss rose over epochs, and Roffset A/B did not affect S4 metrics enough to support keeping orientation correction.

Next action: Treat K2 as partial positive only; do not promote Roffset. If continuing, run a zero/shuffled offset sanity check and optionally a shorter cache-loss-selected AMASS variant before final-test consideration.

Git state: branch `main`; many L4 files and generated data are currently untracked/modified. No commit was made.

CodeGraph state: available and indexed for this project: 71 files, 1557 nodes, 2764 edges.

## Project Overview

This project modifies GlobalPose's L4 post/pre-physics pose refinement path. The current retained direction is a conservative L4 tail update module that refines a `q75` generalized coordinate trajectory while preserving the official GlobalPose evaluation path.

Detailed architecture is summarized in `ARCHITECTURE_OVERVIEW.md`.

The latest experiments compare:

- H1: Euler `q75_prephysics` input, no velocity residual, retained candidate.
- J1: full-body 6D rotation representation as pose input, no sensor offset input.
- K1: J1 plus sequence-level IMU installation offset conditioning through RNN hidden initialization.
- K2: J1/K1-style rot6d L4 module trained on AMASS original/offset-augmented paired-overlay views, then fine-tuned on TotalCapture.
- TC Roffset A/B: TotalCapture-only orientation-offset ablation using `R_JS` / `R_JS^T` corrected `RMB` as an L4-only sidecar input.

Do not modify `test.py`, `MotionEvaluator`, official weights, official test datasets, or final test splits for these ablations.

## Data and Preprocessing

### IMU Input Contract

The L4 frame feature uses original IMU features from the current mainline neural-only cache:

- `aM`: IMU acceleration feature.
- `wM`: IMU angular-velocity feature.
- `RMB`: IMU orientation feature.
- Combined IMU feature dimension: 90D.

For J1/K1 the frame input is:

- pose feature: full 24-joint 6D rotation representation, `24 * 6 = 144D`;
- IMU feature: `aM/wM/RMB = 90D`;
- total frame input dimension: `234D`.

The 6D pose representation is the first two columns of each `3x3` rotation matrix flattened per joint. It is not a full `3x3` rotation matrix and should not be called a 6D rotation matrix.

### IMU Installation Offset Contract

The offset used in K1 is IMU installation position offset, not IMU measurement bias.

- IMU measurement bias means additive accelerometer/gyroscope bias such as `b_a` or `b_g`.
- IMU installation offset means a geometric sensor mounting displacement relative to the body/joint.

K1 uses only the position part:

- processed source field: `imu_offset_r` when present, equivalent alias `r_JS`;
- enriched neural-cache field: `offset_r`;
- per-sequence shape: `[6, 3]`;
- dataset-level shape in processed TotalCapture train/val: `[N, 6, 3]`;
- meaning: `r_JS`, the position of each IMU origin relative to the mapped installation joint, expressed in that joint's local frame;
- frame: joint-local, not world-frame, not sensor-local;
- type: 3D translation offset only, not full SE(3);
- temporal behavior: sequence-level constant, not per-frame;
- source: copied from processed TotalCapture cache into the current no-offset neural-only cache. No offset-augmented dataset is used for training.

K1 does not use `imu_offset_R/R_JS`, `imu_offset_T/T_JS`, `offset_BS`, `offset_JS`, or any per-frame offset feature.

### Enriched Cache

The J1 neural-only cache did not contain offset fields. K1 therefore uses an enriched copy that preserves the original no-offset train/val split and all existing fields, and only adds `offset_r`.

Tool:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python l4_enrich_cache_with_offsets.py \
  --cache-manifest <no-offset neural-only manifest> \
  --processed-dataset <processed TotalCapture train/val pt> \
  --output-dir <new output dir>
```

Train enriched cache:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json
```

Val enriched cache:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json
```

Lightweight validation:

| split | sequences | `offset_r` shape | finite | mean norm | median norm | p90 norm | max norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| TotalCapture train | 36 | `[36,6,3]` | yes | 0.173611 | 0.182150 | 0.227505 | 0.256172 |
| TotalCapture val S4 | 5 | `[5,6,3]` | yes | 0.188401 | 0.206970 | 0.227659 | 0.236213 |

Original no-offset neural-only caches were not overwritten.

## Module Registry

### Module: L4 Pose Feature Builder

Purpose: Build the L4 per-frame feature from prephysics pose and original IMU observations.

Related files: `l4_q75_utils.py`, `l4_train_loss_ablation.py`, `l4_train_diverse_short.py`.

Inputs:

- `euler_q75`: original `q75_prephysics`, `75D`; combined with IMU gives `165D`.
- `rot6d`: full 24-joint 6D rotation representation, `144D`; combined with IMU gives `234D`.
- IMU feature: `aM/wM/RMB`, `90D`.

Outputs: per-frame feature passed to `StreamingTailUpdateQState`.

Current evidence: J1 showed Local TotalCapture improvements, but worsened Global/Mesh/Jitter versus H1.

### Module: Offset Hidden Initialization

Purpose: Condition the L4 recurrent hidden state on a sequence-level IMU installation position offset without changing the per-frame input feature.

Related files: `l4_tail_update_qstate.py`, `l4_train_loss_ablation.py`, `l4_train_diverse_short.py`.

User requirement: Do not concatenate offset to every frame in the first version. Use offset only to initialize the GRU hidden state at sequence reset.

Inputs:

- `offset_r`: `[6,3]`, joint-local IMU position offsets, sequence-level constant.
- flattened offset: `18D`.

Internal structure:

- `Linear(18, hidden_size)`;
- ReLU;
- `Linear(hidden_size, hidden_size)`;
- final layer is zero-initialized;
- output is multiplied by `offset_init_scale`, default `0.1`.

Outputs:

- `h0`: `[hidden_size]`, used as the initial GRU hidden state.

Trainable/frozen parts:

- Offset encoder is newly initialized and trainable.
- Existing J1-compatible L4 parameters are loaded from J1 `best.pt`.

Connected losses:

- No offset loss.
- No IMU proxy loss.
- No contact loss.
- Existing base L4 q/pose/velocity/smoothness losses only.

Expected effect: Provide fixed installation geometry context to the RNN state, potentially reducing J1's Global/Mesh/Jitter regression while preserving rot6d Local gains.

Current evidence: K1 did not achieve this. It improved Local metrics but worsened Global/Mesh/Jitter and scalar validation score versus J1/H1.

Known risks:

- Since the final offset encoder layer starts at zero, a 20-epoch continuation may underuse offset conditioning.
- Hidden initialization can perturb sequence memory in ways that improve local pose while harming global root/mesh behavior.
- Current evidence is downstream MotionEvaluator only; there is no direct hidden-state or offset-use diagnostic yet.

## Loss Registry

### Loss: Base L4 Training Loss

Purpose: Train the L4 q75 residual/refinement while preserving original output contract.

Applied to: q/pose trajectory, qdot/qddot smoothness, root velocity, residual/tail priors, and FK joint terms as configured in `l4_train_loss_ablation.py`.

K1 weights from `train_result.json`:

- `pose_geodesic`: 1.0
- `q_body`: 1.0
- `q_root_ori`: 0.5
- `baseline_body`: 2.0
- `baseline_root_ori`: 5.0
- `qdot`: 0.03
- `qddot`: 0.003
- `fk_joint_rootrel`: 0.1
- `fk_joint_baseline_rootrel`: 0.0
- `residual_prior`: 0.001
- `tail_update_prior`: 0.005
- `edge_q`: 0.01
- `edge_qdot`: 0.03
- `edge_qddot`: 0.003
- `jerk_smooth`: 0.00001
- `root_velocity`: 1.0
- `baseline_velocity`: 2.0
- `velocity_smooth`: 0.03
- `contact_foot_velocity`: 0.0
- `contact_foot_height`: 0.0
- `imu_orientation_proxy`: 0.0
- `imu_acc_proxy`: 0.0
- `imu_gyro_proxy`: 0.0

K1 explicitly disables contact loss, IMU proxy loss, and offset loss.

## Experiment Log

### H1 - Euler q75 Retained Candidate

Question: Can L4 q75 residual refinement improve the current TotalCapture validation baseline without adding unstable velocity residual behavior?

Known configuration:

- pose input mode: Euler `q75_prephysics`;
- frame input dimension: `75 + 90 = 165D`;
- residual scale: `0.005`;
- velocity residual scale: `0.0`;
- checkpoint: `data/experiments/l4_ablation_H1_velocity_scale0_tc_v1/best.pt`.

Status: retained candidate before J1/K1.

### J1 - Pose Input Representation Ablation: Euler q75 vs 6D Rotation Input

Question: Does replacing Euler `q75_prephysics` pose input with full-body 6D rotation representation improve L4 stability and MotionEvaluator metrics?

Configuration:

- pose input mode: `rot6d`;
- pose feature: full 24 joints `* 6D = 144D`;
- IMU feature: `aM/wM/RMB = 90D`;
- frame input dimension: `234D`;
- output: unchanged q75 residual/refinement;
- residual scale: `0.005`;
- velocity residual scale: `0.0`;
- sensor offset input: disabled;
- IMU position offset augmented datasets: excluded;
- IMU proxy loss: disabled;
- offset loss: disabled.

Training command:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -u l4_train_loss_ablation.py \
  --train-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only/baseline_cache_manifest.json \
  --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only/baseline_cache_manifest.json \
  --output-dir data/experiments/l4_ablation_J1_rot6d_poseinput_tc_20ep_v1 \
  --init-checkpoint data/experiments/l4_ablation_H1_velocity_scale0_tc_v1/best.pt \
  --experiment-name J1_rot6d_poseinput_tc_20ep_v1 \
  --loss-mode contact \
  --epochs 20 \
  --window 61 \
  --lr 1e-5 \
  --hidden-size 256 \
  --tail-length 4 \
  --residual-scale 0.005 \
  --velocity-residual-scale 0.0 \
  --contact-foot-velocity-weight 0.0 \
  --contact-foot-height-weight 0.0 \
  --pose-input-mode rot6d \
  --allow-partial-init \
  --validate-every 20
```

Artifacts:

- best checkpoint: `data/experiments/l4_ablation_J1_rot6d_poseinput_tc_20ep_v1/best.pt`;
- last checkpoint: `data/experiments/l4_ablation_J1_rot6d_poseinput_tc_20ep_v1/last.pt`;
- summary: `data/experiments/l4_ablation_J1_rot6d_poseinput_tc_20ep_v1/pose_input_ablation_summary.json`.

Conclusion: J1 is not active method. It improves TotalCapture Local metrics, but Global SIP/Angle/Joint/Mesh and Root/Joint Jitter regress versus H1.

### K1 - Rot6D Pose Input + Sequence-level IMU Offset Hidden Initialization

Question: Can a fixed per-sequence IMU installation offset condition repair J1's Global/Mesh/Jitter regression while preserving its Local pose gains?

Hypothesis: The RNN may benefit from knowing the sequence-level IMU mounting geometry. Passing `offset_r` through an MLP to initialize the GRU hidden state may provide this context without changing the per-frame input dimension.

Controlled variables:

- pose input remains `rot6d`;
- do not return to Euler `q75`;
- frame input remains `234D`;
- original `aM/wM/RMB` IMU feature remains `90D`;
- no per-frame offset concatenation;
- no IMU position offset augmented dataset training;
- no IMU proxy loss;
- no contact loss;
- no offset loss;
- original no-offset cache was not overwritten.

Offset source:

- processed train: `data/dataset_work/TotalCapture_globalpose_official/train.pt`;
- processed val: `data/dataset_work/TotalCapture_globalpose_official/val.pt`;
- source field: `imu_offset_r`, equivalent to `r_JS`;
- enriched cache field: `offset_r`;
- shape per sequence: `[6,3]`;
- sequence-level constant: yes;
- semantic: joint-local IMU origin position relative to its mapped joint.

Enriched cache paths:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json
data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json
```

Network input:

- frame input: `rot6d pose 144D + aM/wM/RMB 90D = 234D`;
- offset condition: `offset_r [6,3] -> flatten 18D -> MLP -> h0 [256]`;
- offset MLP: `18 -> 256 -> 256`, ReLU after the first layer;
- initialization: final offset-encoder layer zero-initialized, `offset_init_scale = 0.1`;
- usage: only once per sequence at `reset_stream(offset_r)`, not every frame.

Training command:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -u l4_train_loss_ablation.py \
  --train-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json \
  --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json \
  --output-dir data/experiments/l4_ablation_K1_offset_init_rot6d_tc_20ep_v1 \
  --init-checkpoint data/experiments/l4_ablation_J1_rot6d_poseinput_tc_20ep_v1/best.pt \
  --experiment-name K1_offset_init_rot6d_tc_20ep_v1 \
  --loss-mode base \
  --epochs 20 \
  --window 61 \
  --lr 1e-5 \
  --hidden-size 256 \
  --tail-length 4 \
  --residual-scale 0.005 \
  --velocity-residual-scale 0.0 \
  --pose-input-mode rot6d \
  --offset-conditioning hidden_init \
  --offset-init-scale 0.1 \
  --allow-partial-init \
  --validate-every 20
```

Checkpoint and logs:

- best checkpoint: `data/experiments/l4_ablation_K1_offset_init_rot6d_tc_20ep_v1/best.pt`;
- last checkpoint: `data/experiments/l4_ablation_K1_offset_init_rot6d_tc_20ep_v1/last.pt`;
- train result: `data/experiments/l4_ablation_K1_offset_init_rot6d_tc_20ep_v1/train_result.json`;
- train log: `data/experiments/l4_ablation_K1_offset_init_rot6d_tc_20ep_v1/train_log.jsonl`;
- summary: `data/experiments/l4_ablation_K1_offset_init_rot6d_tc_20ep_v1/offset_hidden_init_summary.json`.

Checkpoint initialization:

- initialized from J1 best checkpoint;
- shape-compatible J1 parameters loaded;
- newly initialized parameters: `offset_encoder.0.weight`, `offset_encoder.0.bias`, `offset_encoder.2.weight`, `offset_encoder.2.bias`.

TotalCapture S4 validation MotionEvaluator:

| metric | GlobalPose baseline | H1 Euler q75 | J1 rot6d | K1 rot6d+offset h0 | K1-J1 | K1-H1 |
|---|---:|---:|---:|---:|---:|---:|
| Local SIP | 10.466050 | 10.387464 | 10.349230 | 10.333938 | -0.015291 | -0.053526 |
| Local Angle | 10.133907 | 10.014944 | 9.936460 | 9.880548 | -0.055912 | -0.134396 |
| Local Joint | 4.817390 | 4.780045 | 4.748529 | 4.729209 | -0.019320 | -0.050836 |
| Local Mesh | 5.537234 | 5.488123 | 5.457066 | 5.439651 | -0.017414 | -0.048471 |
| Global SIP | 10.716775 | 10.721214 | 10.761651 | 10.824536 | +0.062884 | +0.103322 |
| Global Angle | 10.255115 | 10.214982 | 10.255812 | 10.308572 | +0.052760 | +0.093590 |
| Global Joint | 4.638654 | 4.635137 | 4.664792 | 4.707159 | +0.042367 | +0.072021 |
| Global Mesh | 5.289347 | 5.310025 | 5.360506 | 5.420009 | +0.059503 | +0.109984 |
| Root Jitter | 0.297783 | 0.298330 | 0.299706 | 0.301179 | +0.001473 | +0.002849 |
| Joint Jitter | 0.495126 | 0.496545 | 0.498444 | 0.499747 | +0.001303 | +0.003202 |

Scalar validation score:

- H1: `42.285087`;
- J1: `42.249469`;
- K1: `42.296228`.

DIP S08 validation:

- Not run for K1 in this pass.
- Reason: the allowed DIP neural-only cache / processed split available for the current mainline path does not contain `r_JS/offset_r`; using zero or missing offset would not validate the requested installation-offset hidden initialization.

Conclusion: discard K1 as an active method. K1 preserves and slightly improves J1's Local gains, but it further worsens Global SIP/Angle/Joint/Mesh and Root/Joint Jitter. It is not a partial positive under the stated criterion because the target was specifically to reduce J1's Global/Mesh/Jitter regression.

### K2 - Rot6D Input with Offset-Augmented IMU and Offset/First-Frame RNN Init

Status: historical plan/smoke record. The executed K2 branch is documented in `K2 Executed Run - AMASS Overlay + TotalCapture Roffset A/B` below and supersedes this initial plan.

Why this branch exists:

- K1 used `offset_r` as a hidden-state condition, but it trained/evaluated on the original no-offset neural-only cache except for the merged `offset_r` label.
- K1 did not use offset-augmented AMASS IMU signals.
- K2 is a separate branch: it trains on paired AMASS views, where one view is the original synthetic IMU stream and the other view is the offset-augmented synthetic IMU stream for the same pose/tran target.
- The intended first training version is mixed-view training: original and offset-augmented records are ordinary samples in one cache. Pair-consistency loss is left default-off.

Data audit result:

- Current offset-augmented AMASS processed manifest: `data/dataset_work/AMASS/globalpose_synth_manifest.json`.
- Current offset-augmented AMASS processed shards: `data/dataset_work/AMASS/globalpose_synth_shard*.pt`.
- Offset append backup exists: `*.pt.bak_before_imu_offset`.
- Single-random-placement backup exists: `*.pt.bak_before_single_random_placement`.
- Offset resynthesis summary: `data/dataset_work/SensorOffset/amass_single_random_placement_resynthesis_summary.json`.

Observed current AMASS processed shard fields:

- `pose`: list, first sampled shape `[T,72]`;
- `tran`: list, first sampled shape `[T,3]`;
- `aM`: list, first sampled shape `[T,6,3]`;
- `wM`: list, first sampled shape `[T,6,3]`;
- `RMB`: list, first sampled shape `[T,6,3,3]`;
- `aS`: list, first sampled shape `[T,6,3]`;
- `wS`: list, first sampled shape `[T,6,3]`;
- `RIS`: list, first sampled shape `[T,6,3,3]`;
- `RIM`: list, first sampled shape `[6,3,3]`;
- `RSB`: list, first sampled shape `[6,3,3]`;
- `joint`: list, first sampled shape `[T,24,3]`;
- `v_imu`: list, first sampled shape `[T,6,3]`;
- `imu_offset_r` / `r_JS`: list, per-sequence shape `[6,3]`;
- `imu_offset_R` / `R_JS`: list, per-sequence shape `[6,3,3]`;
- `imu_offset_T` / `T_JS`: list, per-sequence shape `[6,4,4]`;
- `original_imu_offset_r`: list, per-sequence shape `[6,3]`;
- `placement_sampling_config`: present.

Evidence that current processed AMASS has offset-augmented IMU:

- `amass_single_random_placement_resynthesis_summary.json` reports 107 AMASS shards and 10669 sequences.
- Aggregate offset norm median: `0.1820587 m`.
- Aggregate `aM` difference from pre-resynthesis data: mean-of-means `1.394519`, median-of-medians `1.090200`, p95 median `2.478911`, max `1495.485596`.
- Direct sampled comparison showed current processed AMASS `aM` differs from the old neural-only cache/source by about `1.0-2.0` mean norm per IMU sample on sampled shards.

Important cache finding:

- Existing AMASS neural-only cache `data/dataset_work/L4Cache/prephysics_pose_velocity_amass_cache_diverse7_neural_only/baseline_cache_manifest.json` was generated before the May 27 AMASS single-random-placement resynthesis.
- Its shard files do not contain `offset_r`, `imu_offset_r`, `r_JS`, `R_JS`, or `T_JS`.
- Its `aM` values match the older no-offset data, not the current offset-augmented processed shards.
- The initial plan expected a new paired neural-only cache. The executed branch instead used a lighter overlay cache that reuses old no-offset neural-only prephysics/target fields and overlays offset-augmented `aM` plus `imu_offset_r/r_JS`. The old no-offset cache was not overwritten.

Planned full paired AMASS cache path:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_amass_paired_offset_aug_neural_only/
```

Smoke paired AMASS cache path:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_amass_paired_offset_aug_neural_only_smoke/
```

Paired cache requirement:

- One `pair_id` links the original and offset-augmented views of the same source sequence.
- `original` view source: existing old/no-offset AMASS neural-only cache record.
- `offset_aug` view source: current offset-augmented AMASS processed shard passed through the frozen official prephysics path.
- Both views share the same `pose_gt`, `tran_gt`, and `q75_gt` supervision target.
- Both views save `q75_prephysics`, `pose_prephysics`, `v_root_vr`, `stationary_prob`, `aM`, `wM`, `RMB`, `q75_gt`, `pose_gt`, `tran_gt`, `name`, `source_name`, `view_type`, `pair_id`, `offset_r`, and `num_frames`.
- `offset_r` is sequence-level `[6,3]`, not repeated per frame.
- Original-view `offset_r` source is `original_imu_offset_r` from the current augmented shard when present; otherwise the generator falls back to an existing original-cache offset field, then to zero/default. In the smoke cache it used `original_imu_offset_r`.
- Offset-aug-view `offset_r` source is `imu_offset_r` / `r_JS` from the current augmented shard.

Paired smoke command already run:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -u l4_generate_paired_offset_aug_cache.py \
  --original-cache data/dataset_work/L4Cache/prephysics_pose_velocity_amass_cache_diverse7_neural_only/baseline_cache_manifest.json \
  --augmented-input data/dataset_work/AMASS/globalpose_synth_manifest.json \
  --output-dir data/dataset_work/L4Cache/prephysics_pose_velocity_amass_paired_offset_aug_neural_only_smoke \
  --shard-indices 18 \
  --max-pairs 2 \
  --max-frames 120 \
  --require-original-match
```

Paired smoke result:

| item | result |
|---|---:|
| records | `4` |
| pairs | `2` |
| sequences | `CMU/55/55_28_poses`, `CMU/56/56_01_poses` |
| per-record frames | `120` |
| required tensor fields finite | yes |
| `pose_gt/tran_gt/q75_gt` original-vs-aug diff | `0.0` max |
| `RMB` original-vs-aug diff | `0.0` max |
| `wM` original-vs-aug diff | `<= 7.9e-6` max, numerical drift only |
| `aM` original-vs-aug mean norm diff | `3.2666`, `4.5312` |
| original offset source | `original_imu_offset_r` |
| augmented offset source | `imu_offset_r` |

Smoke cache field shapes:

| field | shape |
|---|---:|
| `q75_prephysics` | `[120,75]` |
| `pose_prephysics` | `[120,24,3,3]` |
| `v_root_vr` | `[120,3]` |
| `stationary_prob` | `[120,5]` |
| `aM` | `[120,6,3]` |
| `wM` | `[120,6,3]` |
| `RMB` | `[120,6,3,3]` |
| `q75_gt` | `[120,75]` |
| `pose_gt` | `[120,24,3,3]` |
| `tran_gt` | `[120,3]` |
| `offset_r` | `[6,3]` |

Full-cache caution:

- The existing original no-offset AMASS neural-only cache is the diverse7 cache, not a full 10669-sequence original-view cache.
- A full paired cache can only cover sequences that have an available original-view neural-only record unless a new original-view neural-only cache is regenerated from the pre-resynthesis backup shards.
- Using the existing original cache gives a paired diverse7 subset; using all offset-augmented AMASS requires rebuilding or recovering the matching original view for all shards.

Planned model change:

- Keep per-frame J1 feature:
  - rot6d pose: `144D`;
  - offset-augmented `aM/wM/RMB`: `90D`;
  - total frame input: `234D`.
- Add `--rnn-init-mode {none, offset_only, offset_firstframe}`.
- K2 uses `--rnn-init-mode offset_firstframe`.
- Init feature:
  - `offset_r [6,3] -> 18D`;
  - first-frame rot6d pose `[24,6] -> 144D`;
  - first-frame `aM/wM/RMB -> 90D`;
  - total init dim: `252D`.
- Init MLP:
  - `Linear(252, hidden_size)`;
  - ReLU;
  - `Linear(hidden_size, hidden_size)`;
  - output initializes the GRU hidden state once at sequence reset.
- Do not concatenate offset or first-frame features to every frame.
- Add `--paired-offset-training {true,false}` as a training/data flag. First K2 version uses mixed-view training with this flag false or only as metadata.
- Add `--pair-consistency-weight`, default `0.0`. First K2 version does not enable pair consistency.

Planned losses:

- No IMU proxy loss.
- No contact loss.
- No offset loss.
- Keep H1/J1-style q/pose/velocity/smoothness loss family.

Draft full paired-cache command, not yet run:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -u l4_generate_paired_offset_aug_cache.py \
  --original-cache data/dataset_work/L4Cache/prephysics_pose_velocity_amass_cache_diverse7_neural_only/baseline_cache_manifest.json \
  --augmented-input data/dataset_work/AMASS/globalpose_synth_manifest.json \
  --output-dir data/dataset_work/L4Cache/prephysics_pose_velocity_amass_paired_offset_aug_neural_only \
  --shard-indices 0,18,36,54,72,90,106 \
  --require-original-match
```

This command was the early full-paired plan. The executed K2 run used `l4_generate_k2_overlay_cache.py` and the overlay path documented below.

Draft AMASS mixed-view training command, not yet run:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -u l4_train_loss_ablation.py \
  --train-cache data/dataset_work/L4Cache/prephysics_pose_velocity_amass_paired_offset_aug_neural_only/paired_cache_manifest.json \
  --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_amass_paired_offset_aug_neural_only/paired_cache_manifest.json \
  --output-dir data/experiments/l4_K2_paired_offset_aug_amass_rot6d_init_20ep_v1 \
  --init-checkpoint data/experiments/l4_ablation_J1_rot6d_poseinput_tc_20ep_v1/best.pt \
  --experiment-name K2_paired_offset_aug_amass_rot6d_init_20ep_v1 \
  --loss-mode base \
  --epochs 20 \
  --window 61 \
  --lr 1e-5 \
  --hidden-size 256 \
  --tail-length 4 \
  --residual-scale 0.005 \
  --velocity-residual-scale 0.0 \
  --pose-input-mode rot6d \
  --rnn-init-mode offset_firstframe \
  --pair-consistency-weight 0.0 \
  --allow-partial-init \
  --validate-every 20
```

This early command is retained only for provenance. The actual executed commands and checkpoints are listed in the next section.

### K2 Executed Run - AMASS Overlay + TotalCapture Roffset A/B

Status: completed on 2026-05-27. This is the executed K2 branch and supersedes the earlier paired-cache smoke plan above.

AMASS K2 overlay cache:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
```

Cache contract:

- No official GlobalPose prephysics forward was rerun for AMASS K2 overlay generation.
- `q75_prephysics`, `pose_prephysics`, `v_root_vr`, `stationary_prob`, `q75_gt`, `pose_gt`, and `tran_gt` are reused from the old AMASS no-offset neural-only cache.
- `original` view uses old-cache `aM/wM/RMB` and `original_imu_offset_r`.
- `offset_aug_overlay` view uses current offset-augmented AMASS `aM` and current `imu_offset_r/r_JS`; `wM/RMB` stay from the old cache after consistency checks.
- Both views share the same `pose_gt/tran_gt/q75_gt`.
- `offset_r` is sequence-level `[6,3]`; it is IMU installation position offset, not acc/gyro measurement bias.

Overlay sanity:

| item | result |
|---|---:|
| pairs | `649` |
| records | `1298` |
| frames across records | `1118012` |
| skipped sequences | `20`, all due to strict `wM_diff_too_large` at about `1e-4` |
| `aM` mean norm diff, mean / max over sequences | `3.5920 / 10.8336` |
| `wM` max-abs diff, mean / max over sequences | `1.92e-5 / 9.97e-5` |
| `RMB` max-abs diff | `0.0` |
| offset norm mean / median / p90 / max | `0.1755 / 0.1812 / 0.2327 / 0.2852 m` |

K2 model/input:

- per-frame input: rot6d pose `144D` + IMU `aM/wM/RMB 90D` = `234D`;
- RNN init input: `offset_r 18D` + first-frame rot6d pose `144D` + first-frame IMU `90D` = `252D`;
- init MLP: `252 -> hidden_size -> hidden_size`;
- `--pair-consistency-weight 0.0`;
- no contact loss, no IMU proxy loss, no offset loss;
- `velocity_residual_scale=0.0`, so this run does not learn root velocity residual.

AMASS training command:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -u l4_train_loss_ablation.py \
  --train-cache data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json \
  --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json \
  --output-dir data/experiments/l4_K2_amass_paired_offset_overlay_rot6d_init_20ep_v1 \
  --init-checkpoint data/experiments/l4_ablation_J1_rot6d_poseinput_tc_20ep_v1/best.pt \
  --experiment-name K2_amass_paired_offset_overlay_rot6d_init_20ep_v1 \
  --loss-mode base --epochs 20 --window 61 --lr 1e-5 \
  --hidden-size 256 --tail-length 4 --residual-scale 0.005 \
  --velocity-residual-scale 0.0 --pose-input-mode rot6d \
  --rnn-init-mode offset_firstframe --paired-offset-training \
  --pair-consistency-weight 0.0 --allow-partial-init \
  --max-val-sequences 20 --disable-root-velocity-loss --validate-every 20
```

AMASS checkpoint:

```text
data/experiments/l4_K2_amass_paired_offset_overlay_rot6d_init_20ep_v1/best.pt
data/experiments/l4_K2_amass_paired_offset_overlay_rot6d_init_20ep_v1/last.pt
data/experiments/l4_K2_amass_paired_offset_overlay_rot6d_init_20ep_v1/train_result.json
data/experiments/l4_K2_amass_paired_offset_overlay_rot6d_init_20ep_v1/train_log.jsonl
```

AMASS result:

- best epoch: `20`;
- validation score on 20 AMASS overlay records: `23.070879`;
- final train loss: `0.427386`;
- q residual norm mean: `0.720980`;
- tail update norm mean: `0.000443`.

TotalCapture original-input control:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -u l4_train_loss_ablation.py \
  --train-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json \
  --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json \
  --output-dir data/experiments/l4_K2_tc_original_input_control_20ep_v2 \
  --init-checkpoint data/experiments/l4_K2_amass_paired_offset_overlay_rot6d_init_20ep_v1/best.pt \
  --experiment-name K2_tc_original_input_control_20ep_v2 \
  --loss-mode base --epochs 20 --window 61 --lr 1e-5 \
  --hidden-size 256 --tail-length 4 --residual-scale 0.005 \
  --velocity-residual-scale 0.0 --pose-input-mode rot6d \
  --rnn-init-mode offset_firstframe --pair-consistency-weight 0.0 \
  --allow-partial-init --disable-root-velocity-loss --validate-every 20
```

TotalCapture original-input checkpoint:

```text
data/experiments/l4_K2_tc_original_input_control_20ep_v2/best.pt
data/experiments/l4_K2_tc_original_input_control_20ep_v2/last.pt
data/experiments/l4_K2_tc_original_input_control_20ep_v2/train_result.json
data/experiments/l4_K2_tc_original_input_control_20ep_v2/train_log.jsonl
```

TotalCapture original-input result:

- best epoch: `20`;
- S4 validation score: `42.191594`;
- final train loss: `0.552613`;
- q residual norm mean: `0.690507`;
- tail update norm mean: `0.005833`.

TotalCapture Roffset sidecar caches:

```text
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_B/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_B/baseline_cache_manifest.json
```

TotalCapture Roffset rules:

- The real TotalCapture processed cache is not overwritten.
- Official `aM/wM/RMB` remain the original inputs for frozen GlobalPose PL/IK/VR.
- Corrected IMU fields are stored as `l4_aM/l4_wM/l4_RMB` and consumed only by the L4 feature path.
- `R_JS` shape is `[6,3,3]` per sequence. The audit convention treats `R_JS` as mapping sensor-frame vectors into the estimated joint/body proxy.
- Roffset A: `RSB_corr = R_JS^T`, `RMB_A = RIM^T RIS RSB_corr`.
- Roffset B: `RSB_corr = R_JS`, `RMB_B = RIM^T RIS RSB_corr`.
- `aM` and `wM` are not rotated by `R_JS` because they are already converted by `RIM^T RIS` into the model/world frame in the official formula.

TotalCapture Roffset A fine-tune command:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -u l4_train_loss_ablation.py \
  --train-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json \
  --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json \
  --output-dir data/experiments/l4_K2_tc_Roffset_A_20ep_v1 \
  --init-checkpoint data/experiments/l4_K2_amass_paired_offset_overlay_rot6d_init_20ep_v1/best.pt \
  --experiment-name K2_tc_Roffset_A_20ep_v1 \
  --loss-mode base --epochs 20 --window 61 --lr 1e-5 \
  --hidden-size 256 --tail-length 4 --residual-scale 0.005 \
  --velocity-residual-scale 0.0 --pose-input-mode rot6d \
  --rnn-init-mode offset_firstframe --pair-consistency-weight 0.0 \
  --allow-partial-init --disable-root-velocity-loss --validate-every 20
```

TotalCapture Roffset B fine-tune command:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -u l4_train_loss_ablation.py \
  --train-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_B/baseline_cache_manifest.json \
  --val-cache data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_B/baseline_cache_manifest.json \
  --output-dir data/experiments/l4_K2_tc_Roffset_B_20ep_v1 \
  --init-checkpoint data/experiments/l4_K2_amass_paired_offset_overlay_rot6d_init_20ep_v1/best.pt \
  --experiment-name K2_tc_Roffset_B_20ep_v1 \
  --loss-mode base --epochs 20 --window 61 --lr 1e-5 \
  --hidden-size 256 --tail-length 4 --residual-scale 0.005 \
  --velocity-residual-scale 0.0 --pose-input-mode rot6d \
  --rnn-init-mode offset_firstframe --pair-consistency-weight 0.0 \
  --allow-partial-init --disable-root-velocity-loss --validate-every 20
```

TotalCapture Roffset checkpoints:

```text
data/experiments/l4_K2_tc_Roffset_A_20ep_v1/best.pt
data/experiments/l4_K2_tc_Roffset_A_20ep_v1/last.pt
data/experiments/l4_K2_tc_Roffset_A_20ep_v1/train_result.json
data/experiments/l4_K2_tc_Roffset_A_20ep_v1/train_log.jsonl

data/experiments/l4_K2_tc_Roffset_B_20ep_v1/best.pt
data/experiments/l4_K2_tc_Roffset_B_20ep_v1/last.pt
data/experiments/l4_K2_tc_Roffset_B_20ep_v1/train_result.json
data/experiments/l4_K2_tc_Roffset_B_20ep_v1/train_log.jsonl
```

TotalCapture Roffset training result:

| run | best epoch | S4 score | train loss | q residual norm mean | tail update norm mean |
|---|---:|---:|---:|---:|---:|
| Roffset A | `20` | `42.191598` | `0.552597` | `0.690466` | `0.005832` |
| Roffset B | `20` | `42.191546` | `0.552578` | `0.690396` | `0.005831` |

TotalCapture S4 MotionEvaluator:

| method | score | Local SIP | Local Angle | Local Joint | Local Mesh | Global SIP | Global Angle | Global Joint | Global Mesh | Root Jitter | Joint Jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GlobalPose baseline | n/a | 10.4660 | 10.1339 | 4.8174 | 5.5372 | 10.7168 | 10.2551 | 4.6387 | 5.2893 | 0.2978 | 0.4951 |
| H1 Euler q75 retained | 42.2851 | 10.3875 | 10.0149 | 4.7800 | 5.4881 | 10.7212 | 10.2150 | 4.6351 | 5.3100 | 0.2983 | 0.4965 |
| J1 rot6d no-offset | 42.2495 | 10.3492 | 9.9365 | 4.7485 | 5.4571 | 10.7617 | 10.2558 | 4.6648 | 5.3605 | 0.2997 | 0.4984 |
| K1 offset hidden init, no offset-aug AMASS | 42.2962 | 10.3339 | 9.8805 | 4.7292 | 5.4397 | 10.8245 | 10.3086 | 4.7072 | 5.4200 | 0.3012 | 0.4997 |
| K2 TC original-input control | 42.1916 | 10.3406 | 9.9753 | 4.7597 | 5.4679 | 10.7150 | 10.2134 | 4.6637 | 5.3589 | 0.2987 | 0.4979 |
| K2 TC Roffset A | 42.1916 | 10.3406 | 9.9753 | 4.7597 | 5.4679 | 10.7150 | 10.2134 | 4.6637 | 5.3589 | 0.2987 | 0.4979 |
| K2 TC Roffset B | 42.1915 | 10.3406 | 9.9753 | 4.7597 | 5.4679 | 10.7150 | 10.2134 | 4.6637 | 5.3589 | 0.2987 | 0.4979 |

Conclusion:

- K2 is a partial positive: it improves the scalar S4 validation score versus H1/J1/K1 and reduces J1/K1 Global SIP/Angle/Joint/Mesh regression, while retaining better Local SIP/Angle than H1.
- K2 still has worse Global Joint/Mesh and jitter than the GlobalPose baseline, so it is not yet final-test ready as a clearly retained method.
- Roffset A/B should not be promoted from this run: both produce effectively identical S4 metrics to K2 original-input control. A follow-up audit below confirms the sidecar fields did enter the L4 feature path, but the trained L4 residual response to those feature changes is extremely small. This is evidence that the current Roffset injection is ineffective for the current L4 configuration, not proof that `R_JS` is intrinsically useless.
- DIP validation was not run in this pass because the requested focus was TotalCapture S4 and DIP offset fields/contract remain less direct for this branch.

### 2026-05-28 - Roffset A/B Input-Path Audit

Question: Why are `K2_tc_Roffset_A_20ep_v1` and `K2_tc_Roffset_B_20ep_v1` almost identical to `K2_tc_original_input_control_20ep_v2` on TotalCapture S4?

Hypotheses checked:

- The sidecar cache might not contain different corrected fields.
- The dataloader / feature builder might ignore `l4_RMB` and keep using original `RMB`.
- The physics validation path might pass original `RMB` into L4.
- The experiment directories might contain reused or identical checkpoints.
- The corrected field may enter the network but have very small effect on the learned residual.

Val caches audited:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_B/baseline_cache_manifest.json
```

Common validation sequences:

```text
s4_acting3
s4_freestyle1
s4_freestyle3
s4_rom3
s4_walking2
```

Cache field diff:

| comparison | max abs diff | mean abs diff | rotation geodesic mean / max |
|---|---:|---:|---:|
| Roffset A `RMB` vs `l4_RMB` | `0.327579` | `0.073026` | `11.0498 / 18.9299 deg` |
| Roffset B `RMB` vs `l4_RMB` | `1.996841` | `0.387976` | `89.2275 / 178.1215 deg` |
| Roffset A `l4_RMB` vs Roffset B `l4_RMB` | `1.991479` | `0.369026` | `85.3395 / 170.1719 deg` |
| Roffset A `aM` vs `l4_aM` | `0.0` | `0.0` | n/a |
| Roffset B `aM` vs `l4_aM` | `0.0` | `0.0` | n/a |
| Roffset A `wM` vs `l4_wM` | `0.0` | `0.0` | n/a |
| Roffset B `wM` vs `l4_wM` | `0.0` | `0.0` | n/a |

Per-sensor mean geodesic difference:

| comparison | sensor 0 | sensor 1 | sensor 2 | sensor 3 | sensor 4 | sensor 5 |
|---|---:|---:|---:|---:|---:|---:|
| A `RMB` vs `l4_RMB` | `14.6459` | `10.8777` | `13.0164` | `8.2330` | `8.4707` | `10.6233` |
| B `RMB` vs `l4_RMB` | `163.8097` | `11.9839` | `161.2822` | `161.3653` | `6.1751` | `28.1864` |
| A `l4_RMB` vs B `l4_RMB` | `152.0844` | `4.9564` | `159.8783` | `162.3682` | `6.0602` | `22.6663` |

Conclusion from cache diff:

- Roffset sidecar `l4_RMB` is not identical to original `RMB`.
- Roffset A and Roffset B are strongly different from each other.
- `l4_aM/l4_wM` intentionally equal original `aM/wM`; this matches the Roffset sidecar design, where only the orientation feature is corrected and model/world-frame acceleration/angular velocity are not directly rotated by local `R_JS`.

Dataloader / feature-builder audit:

- `l4_train_diverse_short.load_records` loads optional `l4_aM`, `l4_wM`, and `l4_RMB` when present.
- `l4_train_loss_ablation.firstframe_init_feature` uses:

```python
a0 = record.get('l4_aM', record['aM'])[0]
w0 = record.get('l4_wM', record['wM'])[0]
R0 = record.get('l4_RMB', record['RMB'])[0]
```

- `l4_train_loss_ablation.run_cached_sequence` uses the same fallback rule for every frame.
- `l4_train_diverse_short.evaluate_physics` calls:

```python
net.forward_frame(
    record['aM'][frame_idx],
    record['wM'][frame_idx],
    record['RMB'][frame_idx],
    l4_a=record.get('l4_aM', record['aM'])[frame_idx],
    l4_w=record.get('l4_wM', record['wM'])[frame_idx],
    l4_R=record.get('l4_RMB', record['RMB'])[frame_idx],
)
```

- `net.GPNet.forward_frame` uses original `a/w/R` for frozen PL/IK/VR, then passes `l4_a/l4_w/l4_R` only into `self.l4_prephysics.refine`.
- There is no `--use-l4-imu-fields` or `--l4-imu-field-prefix` switch in this implementation. The behavior is automatic: if `l4_*` fields exist in the cache, the L4 path uses them; otherwise it falls back to original `aM/wM/RMB`.

One-window actual feature diff:

Sequence/window:

```text
s4_acting3, first 61 frames
```

Actual `StreamingTailUpdateQState` frame feature is:

```text
rot6d pose 144D + selected IMU feature 90D
```

| feature comparison | full feature max / mean | rot6d max / mean | IMU part max / mean |
|---|---:|---:|---:|
| original vs Roffset A | `0.212490 / 0.014880` | `0.0 / 0.0` | `0.212490 / 0.038688` |
| original vs Roffset B | `1.953654 / 0.088381` | `0.0 / 0.0` | `1.953654 / 0.229791` |
| Roffset A vs Roffset B | `1.955284 / 0.081947` | `0.0 / 0.0` | `1.955284 / 0.213063` |

Hidden-init feature diff for the same window:

| init comparison | full init max / mean | offset part max | frame-feature part max |
|---|---:|---:|---:|
| original vs Roffset A | `0.210516 / 0.014060` | `0.0` | `0.210516` |
| original vs Roffset B | `1.937921 / 0.084381` | `0.0` | `1.937921` |
| Roffset A vs Roffset B | `1.941835 / 0.078418` | `0.0` | `1.941835` |

Conclusion from feature diff:

- Corrected `l4_RMB` does enter the actual L4 frame feature and first-frame hidden-init feature.
- The pose/rot6d feature is unchanged, so the audit isolates the IMU feature difference.
- The offset part of hidden init is unchanged across original/A/B, as expected; only first-frame IMU orientation changes.

Checkpoint/result uniqueness:

| run | best.pt sha256 | last.pt sha256 | best epoch | S4 score | train loss first / last / min |
|---|---|---|---:|---:|---:|
| K2 original-input control | `e4a747472c6f6c7f6dad97d03267631ffbe88136bf8bf8c4d698836f400882b5` | `bdca971f67ad4b0d376359ff3d7a3c91b1ddf8f34f194a97cd716ee1605a230a` | `20` | `42.191594` | `0.511370 / 0.552613 / 0.504150` |
| K2 Roffset A | `6ee25ec8e2d86f194e3fb7a892bc6ad6c4f2aaa5d3c3382398cacf70a4b07c25` | `b5d081cdc392d22ffc0b2f20cb05b9b29ecbfdd6abf4f2ec28222dfed27cfe5ce` | `20` | `42.191598` | `0.511353 / 0.552597 / 0.504128` |
| K2 Roffset B | `12ca92c38a9d05e71b5a034df0672e51c4fd0a992fb7a53107afb21acf21ec7b` | `37e70ebe02c23fc18914db9cdb4fb1cb7b10125bb0980df27f1de3c10c0ef532` | `20` | `42.191546` | `0.511204 / 0.552578 / 0.503988` |

Conclusion from checkpoint/result uniqueness:

- Checkpoints are not byte-identical.
- Training logs are not byte-identical.
- The three runs were not simple file reuse.

Forward-response diagnostic:

Same sequence/window:

```text
s4_acting3, first 61 frames
```

For each trained checkpoint, the same model was run on original, Roffset A, and Roffset B records using the cached-sequence feature path. This isolates model sensitivity to the changed feature while keeping the checkpoint fixed.

| checkpoint | q residual diff original-vs-A max / mean | q residual diff original-vs-B max / mean | q residual norm mean original / A / B |
|---|---:|---:|---:|
| K2 original-input model | `7.45e-5 / 8.23e-7` | `1.53e-3 / 2.72e-5` | `0.709317 / 0.709318 / 0.709898` |
| K2 Roffset A model | `9.55e-5 / 6.96e-7` | `1.53e-3 / 2.74e-5` | `0.709314 / 0.709302 / 0.709899` |
| K2 Roffset B model | `1.01e-4 / 1.63e-6` | `5.24e-4 / 2.36e-5` | `0.710171 / 0.710136 / 0.709665` |

Input-layer weight norms for the three checkpoints:

| checkpoint | pose columns | aM columns | wM columns | RMB columns |
|---|---:|---:|---:|---:|
| K2 original-input model | `7.2343` | `2.3492` | `2.4894` | `4.4286` |
| K2 Roffset A model | `7.2341` | `2.3493` | `2.4895` | `4.4309` |
| K2 Roffset B model | `7.2344` | `2.3474` | `2.4889` | `4.4424` |

Interpretation:

- The RMB input columns are not zero, so this is not a completely disconnected feature.
- Despite sizeable `l4_RMB` changes, the L4 output residual changes are tiny: about `1e-4` to `1e-3` max in q75 residual for this diagnostic window.
- This explains why MotionEvaluator metrics are nearly unchanged. The current L4 residual branch is strongly constrained by small `residual_scale=0.005`, baseline-preserving losses, frozen root-velocity residual, and a learned residual trajectory that is dominated by pose/prephysics context rather than the orientation sidecar perturbation.

Audit decision:

```text
ROFFSET_SIDECAR_FIELDS_ENTER_L4_FEATURE_PATH = YES
ROFFSET_A_B_CACHES_ARE_NUMERICALLY_DIFFERENT = YES
ROFFSET_A_B_CHECKPOINTS_ARE_UNIQUE = YES
CURRENT_L4_RESPONSE_TO_ROFFSET_INPUT = VERY_SMALL
```

Implication:

- The previous Roffset A/B result should not be described as "R_JS is invalid" or "R_JS has no value".
- It should be described as: under the current K2 L4 configuration, using `R_JS` only as an L4-sidecar `RMB` correction changes the input but barely changes the learned residual or S4 metrics.
- A stronger diagnostic would freeze the trained model and run controlled forward-only perturbations with larger/known `RMB` perturbations, or train an explicit diagnostic head/loss to confirm whether the L4 hidden state can use the orientation-offset signal.

### 2026-05-28 - K2 Dropout Audit and Smoke Ablation

Question: Did current J1/K1/K2 training use dropout, and can a small dropout ablation improve K2 Global/Mesh/Jitter?

Current-code audit before this change:

| item | finding |
|---|---|
| `l4_train_loss_ablation.py --dropout` default | no such parameter existed |
| `StreamingTailUpdateQState` dropout | no dropout existed |
| GRU internal dropout | not applicable; the module uses one `torch.nn.GRUCell`, not multi-layer `torch.nn.GRU` |
| input MLP / feature dropout | none |
| K2 training command explicit dropout | no dropout flag in historical K2 commands |
| IMU feature dropout | none |
| acceleration-specific dropout | none |
| gyro-specific dropout | none |
| orientation/RMB-specific dropout | none |

Conclusion from audit:

```text
J1/K1/K2 historical runs used dropout = 0.0 everywhere.
```

Implemented default-off dropout parameters:

```text
--dropout
--imu-feature-dropout
--acc-dropout
--gyro-dropout
--orientation-dropout
--offset-init-scale
```

Implementation details:

- File: `l4_tail_update_qstate.py`.
- Dropout is applied inside `StreamingTailUpdateQState` only when `model.training == True`.
- Validation/test/MotionEvaluator calls use `model.eval()`, so dropout is disabled there.
- `--dropout` applies to the full L4 frame feature before the input MLP.
- `--imu-feature-dropout` applies to the whole 90D `aM/wM/RMB` slice.
- `--acc-dropout` applies to the 18D `aM` slice.
- `--gyro-dropout` applies to the 18D `wM` slice.
- `--orientation-dropout` applies to the 54D `RMB` slice.
- For `rnn_init_mode=offset_firstframe`, the same feature dropout function is also applied to the first-frame init feature during training only.
- File: `l4_train_loss_ablation.py`.
- The trainer now records dropout settings in `train_result.json`.
- The trainer now prints and records the actual L4 IMU field contract:

```text
actual a field: l4_aM if present else aM
actual w field: l4_wM if present else wM
actual R field: l4_RMB if present else RMB
```

Important constraint:

- A first attempt to run five full-S4 1-epoch branches was stopped because even D0 exceeded the intended "small ablation" budget before finishing. No result from that aborted directory should be used.
- The completed run below is intentionally a smoke ablation, not a complete S4 method comparison.

Smoke ablation setup:

```text
train cache: data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json
val cache: data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json
init checkpoint: data/experiments/l4_K2_amass_paired_offset_overlay_rot6d_init_20ep_v1/best.pt
epochs: 1
max train sequences: 4
max val sequences: 1
pose_input_mode: rot6d
rnn_init_mode: offset_firstframe
residual_scale: 0.005
velocity_residual_scale: 0.0
contact loss: disabled
IMU proxy loss: disabled
offset loss: disabled
pair consistency: 0.0
```

Base command pattern:

```bash
/home/lingfeng/.conda/envs/globalpose-gpu/bin/python -u l4_train_loss_ablation.py \
  --train-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json \
  --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json \
  --init-checkpoint data/experiments/l4_K2_amass_paired_offset_overlay_rot6d_init_20ep_v1/best.pt \
  --loss-mode base --epochs 1 --window 61 --lr 1e-5 \
  --hidden-size 256 --tail-length 4 --residual-scale 0.005 \
  --velocity-residual-scale 0.0 --pose-input-mode rot6d \
  --rnn-init-mode offset_firstframe --pair-consistency-weight 0.0 \
  --allow-partial-init --disable-root-velocity-loss --validate-every 1 \
  --max-train-sequences 4 --max-val-sequences 1
```

Smoke branches:

| branch | extra args | output dir |
|---|---|---|
| K2-D0 | none | `data/experiments/l4_K2_D0_dropout_none_tc_1ep_smoke_v1` |
| K2-D1 | `--dropout 0.1` | `data/experiments/l4_K2_D1_dropout01_tc_1ep_smoke_v1` |
| K2-D2 | `--imu-feature-dropout 0.1` | `data/experiments/l4_K2_D2_imu_feature_dropout01_tc_1ep_smoke_v1` |
| K2-D3 | `--acc-dropout 0.1` | `data/experiments/l4_K2_D3_acc_dropout01_tc_1ep_smoke_v1` |
| K2-D4 | `--acc-dropout 0.1 --offset-init-scale 0.2` | `data/experiments/l4_K2_D4_acc_dropout01_offsetscale02_tc_1ep_smoke_v1` |

Smoke TotalCapture S4 one-sequence MotionEvaluator:

| run | score | Local SIP | Local Angle | Local Joint | Local Mesh | Global SIP | Global Angle | Global Joint | Global Mesh | Root Jitter | Joint Jitter | train loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K2-D0 smoke | 43.203086 | 10.9465 | 10.1989 | 4.9612 | 5.8274 | 10.9530 | 10.1088 | 4.9269 | 5.6587 | 0.4032 | 0.7126 | 0.493209 |
| K2-D1 dropout=0.1 | 43.203085 | 10.9465 | 10.1989 | 4.9612 | 5.8274 | 10.9530 | 10.1088 | 4.9269 | 5.6587 | 0.4032 | 0.7126 | 0.493209 |
| K2-D2 imu_feature_dropout=0.1 | 43.203085 | 10.9465 | 10.1989 | 4.9612 | 5.8274 | 10.9530 | 10.1088 | 4.9269 | 5.6587 | 0.4032 | 0.7126 | 0.493235 |
| K2-D3 acc_dropout=0.1 | 43.203085 | 10.9465 | 10.1989 | 4.9612 | 5.8274 | 10.9530 | 10.1088 | 4.9269 | 5.6587 | 0.4032 | 0.7126 | 0.493233 |
| K2-D4 acc_dropout=0.1, offset_init_scale=0.2 | 43.203085 | 10.9465 | 10.1989 | 4.9612 | 5.8274 | 10.9530 | 10.1088 | 4.9269 | 5.6587 | 0.4032 | 0.7126 | 0.493194 |

Smoke result:

- The dropout implementation runs and checkpoints are distinct.
- In this very small run, all variants are numerically indistinguishable in the one-sequence S4 MotionEvaluator table.
- Train loss changes only at `~1e-5` to `~4e-5` scale.
- This does not support promoting any dropout setting.
- It also does not rule out dropout for a larger controlled run, because 1 epoch / 4 train sequences / 1 validation sequence is only a wiring smoke.

Dropout decision:

```text
K2_DROPOUT_IMPLEMENTATION = OK_FOR_FUTURE_ABLATION
K2_DROPOUT_SMOKE_RESULT = NO_TREND
DO_NOT_PROMOTE_D1_D2_D3_D4_FROM_SMOKE
```

## Evaluation Protocol

Primary validation for these L4 ablations is MotionEvaluator on TotalCapture S4 validation. DIP S08 validation is used when the required fields and cache contract are available without changing the intended experiment variable.

Metrics to report:

- Local SIP;
- Local Angle;
- Local Joint;
- Local Mesh;
- Global SIP;
- Global Angle;
- Global Joint;
- Global Mesh;
- Root Jitter;
- Joint Jitter.

Do not judge candidates by scalar loss alone. For J1/K1, Global and Jitter regressions dominate the decision even when Local metrics improve.

## Known Issues and Risks

- The original `IMPLEMENTATION_PLAN_AND_REVIEW.md` was not present in the project when K1 documentation was updated. This recreated file captures the verifiable current state but may not include earlier narrative that was not recoverable from disk or git.
- Many L4 files are untracked, so git does not currently provide a clean reproducibility boundary.
- K1 does not include direct validation that the learned hidden state uses `offset_r` meaningfully.
- K2 also does not yet include direct zero-offset or shuffled-offset sanity checks, so the contribution of the actual offset value versus the extra RNN-init MLP capacity is not isolated.
- AMASS K2 training loss rose across the 20-epoch run; TotalCapture S4 is the relevant validation evidence, but this makes the AMASS stage a weak success signal by itself.
- Roffset A/B changed the L4-only sidecar `RMB` input but produced effectively unchanged S4 metrics, so orientation-offset correction should not be promoted without a more diagnostic experiment.
- The offset field is geometric installation offset, not accelerometer or gyroscope measurement bias. Future naming and documentation must keep this distinction.
- Do not use S5/final test for tuning K2 or Roffset; the current evidence is S4 validation only.

## Next Actions

1. Keep H1 as the retained candidate unless a future ablation improves Global/Mesh/Jitter while preserving Local gains.
2. Do not promote K1.
3. Treat K2 as partial positive only; do not move it to final test until a zero/shuffled offset sanity check and a stability check support the causal value of `offset_r`.
4. Do not promote TotalCapture Roffset A/B; they are indistinguishable from K2 original-input control in the current S4 metrics.
5. If offset conditioning is revisited, add direct diagnostics for offset encoder activation, per-sensor/sequence consistency, and real-vs-synthetic acceleration distribution.
6. Keep final test split untouched for tuning decisions.
