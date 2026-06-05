# GlobalPose Project Status

## ACTIVE SUMMARY

Current stage: GlobalPose sparse-IMU mocap module-replacement research.

Current task: Archive the completed NewPL init36 RunD-style training and streaming evaluation.

Review state: Approved for next step.

Current changed files:

```text
PROJECT_STATUS.md
RECENT_REPLACEMENT_VERSIONS.md
EXPERIMENT_LOG.md
```

Current module: PL-s1 replacement remains the selected current mainline direction; IK-s1 replacements are recorded but not selected over PL-only results.

Current experiment: NewPL init36 / K2-like initialization with `offset_r[18] + pRL[15] + gR0[3]`, RunD-style training, processed TotalCapture S4 streaming evaluation.

Current result: Best current processed-input replacement result is NewPL init36 RunD-style, S4 score `38.625657482802865`. It improves Run D PL-only `38.69484578047692` by `0.069188297674055` and Original GPNet + processed `38.753660` by `0.128002517197135`.

Current blocker: No blocker for NewPL init36. NewIK2 remains unresolved: verified artifacts found earlier are NewIK1 / official-shape IK1 artifacts, not confirmed IK-s2 (`iknet.net2`) replacement artifacts.

Next action: Use NewPL init36 `best_loss.pt` / `last.pt` as the current PL1 upstream checkpoint for future downstream replacement experiments, then locate exact NewIK2 artifacts if they exist.

Git state: The project root `/home/lingfeng/projects/GlobalposeMy/GlobalPose` is not reported as a git repository in this Claude session, so no git status was recorded for this documentation update.

CodeGraph state: Initialized under `/home/lingfeng/projects/GlobalposeMy/GlobalPose/.codegraph`; use CodeGraph for structural code questions, not for immediate post-edit verification.

## Document Map

Use these three documents as the project source of truth:

| Document | Role | Read policy |
|---|---|---|
| `PROJECT_STATUS.md` | First-read lightweight project status, baseline architecture, current best evidence, concise experiment summaries, and pointers to details. | Read first when designing a new experiment. |
| `RECENT_REPLACEMENT_VERSIONS.md` | Current replacement-version series; every new module replacement gets a version entry with design/change/effect/comparison/conclusion/artifacts. | Update after every replacement experiment. |
| `EXPERIMENT_LOG.md` | Detailed archive for commands, metrics, checkpoints, failures, and interpretation. | Do not read by default; read targeted line ranges through indices in this file or `RECENT_REPLACEMENT_VERSIONS.md`. |

## Project Overview

This repository is a research fork of GlobalPose for sparse-IMU full-body motion capture. The current research direction is to replace individual official GlobalPose stages while preserving each stage's official input/output contract and evaluating through the official TotalCapture S4 protocol.

Current replacement order:

```text
Official PL  -> NewPL
Official IK1 -> NewIK1
Official IK2 -> NewIK2
Official VR  -> NewVR
```

Replacement rules:

1. Keep downstream official modules unchanged unless the experiment explicitly replaces them.
2. Match the replaced module's official output shape and semantics.
3. Train/evaluate module-level targets first, then validate through full S4 when making a claim.
4. Compare only runs that share the same dataset split and evaluation metric.
5. Record every replacement attempt in `RECENT_REPLACEMENT_VERSIONS.md` and detailed evidence in `EXPERIMENT_LOG.md`.

## Repository Map

Important files and groups:

| Path | Role |
|---|---|
| `net.py` | Central `GPNet` implementation and official GlobalPose stage integration points. |
| `process.py` | Dataset and cache processing for TotalCapture, DIP-IMU, AMASS-style data. |
| `test.py` | Official-style evaluation path and `MotionEvaluator` metrics. |
| `pl_curve*` | NewPL / PLCurve training, evaluation, cache usage, and streaming control-point prediction. |
| `newik1_*`, `ik1_curve*` | IK1 replacement/refinement experiments. |
| `l4_*`, `k2_*` | Prephysics q-state and SO(3)-curve experiments. |
| `full_curve_globalpose.py` | Full-chain curve modules using the streaming control-point pattern. |
| `tools/experiment_orchestrator.py` | Task-file experiment scheduling, GPU allocation, logs, parsing, and status writeback. |
| `articulate/`, `carticulate/` | Local math, body-model, and original physics backend support packages. |

## Current Architecture

`GPNet` in `net.py` is the central model. The official GlobalPose pipeline is:

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

### Input Contract

Official per-frame sparse-IMU inputs use six sensors:

```text
aM:  [6, 3]    model-frame acceleration
wM:  [6, 3]    model-frame angular velocity
RMB: [6, 3, 3] body-to-model IMU orientation
```

`GPNet.forward_frame` constructs neural PL features from these inputs:

```text
aRB = aM @ RMB[5]
wRB = wM @ RMB[5]
RRB = RMB[5]^T @ RMB[:5]
gR0 = -RMB[5, 1]
```

Reduced joint set:

```text
(1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
```

### PL-s1 Contract

Input:

```text
aRB 18D + wRB 18D + RRB 45D + gR0 3D = 84D
```

Output:

```text
pRB 15D + gR1 3D = 18D
```

Any PL replacement must preserve this 18D output contract for downstream IK1.

### IK-s1 Contract

Input:

```text
RRB_after_pl 45D + gR1 3D + pRB 15D = 63D
```

Output:

```text
pRJ 69D + gR2 3D = 72D
```

Any IK1 replacement must preserve this 72D output contract for downstream IK2/VR.

### IK-s2 Contract

Input:

```text
RRB_after_ik1 45D + gR2 3D + pRJ 69D = 117D
```

Output:

```text
15 reduced joints × 6D rotation = 90D
```

Any NewIK2 replacement must preserve the official 90D reduced-rotation output contract and postprocessing compatibility.

### VR-s1 and Physics Contract

Input:

```text
RRJ 135D + pRJ 69D + aRB 18D + wRB 18D + gR2 3D = 243D
```

Output:

```text
9D root velocity / stationary-contact representation
```

VR output feeds velocity fusion and the `carticulate` physics backend.

## Data and Preprocessing

Baseline IMU processing formulas:

```text
RMB = RIM^T @ RIS @ RSB
aM  = RIM^T @ RIS @ aS + [0, -9.8, 0]
wM  = RIM^T @ RIS @ wS
```

Current processed IMU convention:

```text
processed IMU = orientation-only processed IMU / RMB-only correction
l4_aM == aM
l4_wM == wM
l4_RMB != RMB
```

Consistent processed v2 formula:

```text
RSB_new = R_JS^T
l4_RMB = RIM^T @ RIS @ RSB_new
l4_aM  = RIM^T @ RIS @ aS + [0, -9.8, 0] = aM
l4_wM  = RIM^T @ RIS @ wS = wM
```

Interpretation: the useful processed-input gain comes from corrected orientation and induced root-relative feature changes in `aRB/wRB/RRB/gR0`, not from changed stored acceleration or gyro fields.

Detailed records:

- Processed IMU direct S4 baseline: `EXPERIMENT_LOG.md:5-36`.
- Consistent processed IMU v2 audit: `EXPERIMENT_LOG.md:38-105`.

## Module Registry

### Module: Official GlobalPose `GPNet`

Purpose: Baseline sparse-IMU full-body pose and translation pipeline.

User requirement: Preserve official module contracts so each stage can be replaced independently and fairly compared.

Inserted location: `net.py`.

Related files: `net.py`, `test.py`, `process.py`, `carticulate/`, `articulate/`.

Inputs: `aM [6,3]`, `wM [6,3]`, `RMB [6,3,3]` per frame.

Outputs: final pose and translation through official evaluator path.

Internal structure: PL-s1, IK-s1, IK-s2, VR-s1, velocity/contact fusion, physics backend.

Trainable/frozen parts: Frozen official weights for baseline comparisons; individual stages may be replaced in controlled experiments.

Connected losses: Official training losses are historical baseline context; current replacement losses are listed below.

Training strategy: Use official baseline/evaluator as the reference. Replacement modules should preserve I/O contracts and validate on TotalCapture S4.

Expected effect: Stable reference for all replacement comparisons.

Current evidence: Official input S4 score `42.522402`; processed orientation input score `38.753660`.

Known risks: Mixing official-input and processed-input metrics can produce invalid claims.

### Module: NewPL / PLCurve init36

Purpose: Replace official PL-s1 while preserving `pRB[15] + gR1[3]` output.

User requirement: Use a K2-like NewPL initialization with first-frame IMU position offset, `pRL`, and `gR0`, then train it with the successful RunD-style recipe.

Inserted location: PL backend selected through `GPNet` options and `pl_curve*` scripts.

Related files: `pl_curve.py`, `pl_curve_cache.py`, `pl_curve_train.py`, `pl_curve_eval.py`, `net.py`.

Inputs: Official PL neural frame features from processed orientation input: `aRB/wRB/RRB/gR0`, 84D total. Stream initialization uses `offset_r[18] + pRL[15] + gR0[3] = 36D`.

Outputs: `pRB[15] + gR1[3] = 18D` for downstream IK1.

Internal structure: PLCurve-style streaming control-point module with configurable `init_size`; legacy 18D checkpoints remain compatible, and init36 checkpoints require `offset_r` at runtime.

Trainable/frozen parts: NewPL trainable; downstream official modules frozen for PL-only evaluation.

Connected losses: RunD-style pRB/gR1 output losses, gR1 derivative losses, pRB ddot smoothing, and GT spline-control supervision.

Training strategy: TotalCapture processed-input fine-tune from historical Run D checkpoint using partial loading into expanded 36D init encoder and plateau early-stop controls.

Expected effect: Reduce NewPL train/runtime init mismatch and improve final S4 under processed orientation input.

Current evidence: NewPL init36 S4 score `38.625657482802865`, best current processed-input replacement result.

Known risks: `tail_update=4` is a control-point revision window, not a declared four-frame output latency. Init36 runtime requires `offset_r`.

Version entry: `RECENT_REPLACEMENT_VERSIONS.md` `newpl_init36_v1`.

Detailed record: `EXPERIMENT_LOG.md:297-354`.

### Module: NewIK1 Control-Point Variants

Purpose: Replace official IK-s1 while preserving `pRJ[69] + gR2[3]` output.

User requirement: Record each IK replacement version with design, effect, comparison, conclusion, and artifacts.

Inserted location: IK1 backend selected through `GPNet` options and `newik1_*` scripts.

Related files: `newik1_control_point.py`, `newik1_control_train.py`, `newik1_control_eval.py`, `newik1_official_input_eval.py`, `scripts/run_newik1_official_input_full.sh`, `net.py`.

Inputs: IK1 features derived after PL output; official-shape input is `RRB_after_pl 45D + gR1 3D + pRB 15D = 63D`. Control-point variants also use streaming control tails where applicable.

Outputs: `pRJ[69] + gR2[3] = 72D`.

Internal structure: GRU/control-point module for streaming IK1 state prediction.

Trainable/frozen parts: NewIK1 trainable; official downstream IK2/VR/physics frozen for full-pipeline validation.

Connected losses: pRJ/gR2 SmoothL1 or directional losses, pRJ/gR first/second difference losses, control pRJ/gR2 losses, control-point prior, tail-update prior, bone-length consistency, weak distillation where configured.

Training strategy: AMASS adaptation, TotalCapture PL1-output fine-tuning, optional bone-length continuation, then full S4 validation.

Expected effect: Reduce teacher-forcing mismatch and improve downstream IK state quality.

Current evidence: Bone-length continuation improved local loss to `0.1783127911388874`; full official-input NewIK1 pipeline scored `38.70523069866002`, which does not beat Run D PL-only `38.694846`.

Known risks: Local decoded-state validation loss may not track final S4; verified artifacts are NewIK1/IK, not confirmed NewIK2.

Version entries: `RECENT_REPLACEMENT_VERSIONS.md` `newik1_v1`, `newik1_v2`, `newik1_v3`, `newik1_official_input_v1`.

Detailed records: `EXPERIMENT_LOG.md:138-295`.

## Loss Registry

### Loss: NewPL RunD-style GT spline-control supervision

Purpose: Improve PLCurve control points rather than only current-frame decoded outputs.

Formula or description: Supervise predicted PL spline controls against fitted GT controls for pRB and gR1 while preserving the official PL output contract.

Inputs: Predicted and GT/fitted PL controls for `pRB[15]` and `gR1[3]`.

Weight: NewPL init36 used `gt_control_pRB=0.3`, `gt_control_gR1=0.1`, `gR1_dot=0.03`, `gR1_ddot=0.001`, and `pRB_ddot_smooth=1e-6`.

Applied to: NewPL / PLCurve init36 RunD-style training.

Connected module: NewPL / PLCurve init36.

Expected effect: Better streaming PL state and improved final S4.

Observed effect: Historical Run D improved S4 to `38.694846`; NewPL init36 further improved S4 to `38.625657482802865`.

### Loss: NewIK1 pRJ/gR2 state losses

Purpose: Match the official IK1 state target while preserving downstream contract.

Formula or description: pRJ Cartesian/state regression plus normalized gR2 direction loss.

Inputs: Predicted and target `pRJ[69] + gR2[3]`.

Weight: Common baseline used `pRJ=1.0`, `gR2=1.0`; stronger pRJ/control variant used `pRJ=2.0`.

Applied to: NewIK1 control-point variants.

Connected module: NewIK1.

Expected effect: Improve decoded IK1 state quality.

Observed effect: Stronger pRJ/control continuation worsened local validation loss from `0.1783127911388874` to `0.17918791025876998`.

### Loss: NewIK1 temporal derivative losses

Purpose: Reduce jitter and improve temporal consistency of IK1 outputs.

Formula or description: First- and second-difference losses on pRJ and gR2 trajectories.

Inputs: Predicted and target pRJ/gR2 finite differences.

Weight: Recorded configuration used `pRJ_dot=0.03`, `pRJ_ddot=0.001`, `gR2_dot=0.03`, `gR2_ddot=0.001`.

Applied to: NewIK1 control-point training.

Connected module: NewIK1.

Expected effect: Smoother IK1 state predictions.

Observed effect: Contributed to local convergence, but isolated effect was not separately validated in the current record.

### Loss: NewIK1 control-point losses and priors

Purpose: Keep streaming control-point states aligned with target controls and avoid excessive residual/tail revisions.

Formula or description: Control pRJ/gR2 losses, control-point prior, and tail-update prior.

Inputs: Predicted control tails, target control tails, base IK1 state, and residual/tail deltas.

Weight: Recorded baseline used `control_pRJ=0.1`, `control_gR2=0.1`, `control_point_prior=0.3`, `tail_update_prior=0.005`; stronger control variant used `control_pRJ=0.3`.

Applied to: NewIK1 control-point variants.

Connected module: NewIK1.

Expected effect: Better streaming control behavior and lower teacher-forcing mismatch.

Observed effect: Stronger control pRJ with stronger pRJ did not help local validation loss.

### Loss: NewIK1 bone-length consistency

Purpose: Stabilize pRJ geometry and possibly improve gravity/root-direction behavior.

Formula or description: Bone-length consistency penalty over decoded joint geometry.

Inputs: Predicted pRJ/joint geometry and skeleton bone-length constraints.

Weight: `bone_length=0.5`.

Applied to: NewIK1 bone-length continuation.

Connected module: NewIK1.

Expected effect: Better Cartesian IK1 geometry.

Observed effect: Slight local validation improvement from `0.17848628610372544` to `0.1783127911388874`; final S4 benefit not established.

## Training Strategy

For GlobalPose staged replacement, use a resumable cascade:

```text
synthetic/AMASS teacher-forced pretrain
-> real-dataset teacher-forced fine-tune
-> upstream-streaming cache generation
-> upstream-streaming fine-tune
-> final full-pipeline evaluation
```

Future mainline training caches must be streaming-compatible:

1. reset or initialize RNN/module state for each sequence;
2. call official `GPNet.rnn_initialize(init_pose, init_velocity)`;
3. run frame-by-frame forward;
4. save upstream module streaming outputs;
5. train downstream stages on those streaming outputs.

Do not use this old batch/cache contract as a new mainline cache:

```python
gpnet.plnet([(pl_input, pl_target[0])])
```

That form is reserved for historical diagnostics or explicit ablation controls.

## Evaluation Protocol

Primary validation target: TotalCapture S4 validation, 5 sequences, 17223 frames, official-style `MotionEvaluator` metrics.

Selection rule: Do not claim improvement unless the baseline and new method use the same data and same metric.

Key S4 scores, lower is better:

| Run | Score | Evidence |
|---|---:|---|
| Official GPNet + official input | `42.522402` | `EXPERIMENT_LOG.md:5-36` |
| Official GPNet + processed orientation input | `38.753660` | `EXPERIMENT_LOG.md:5-36` |
| Consistent processed v2 | `38.753660` | `EXPERIMENT_LOG.md:38-105` |
| NewPL Run D | `38.694846` | `EXPERIMENT_LOG.md:107-136` |
| NewIK1 official-input full pipeline | `38.70523069866002` | `EXPERIMENT_LOG.md:256-295` |
| NewPL init36 RunD-style | `38.625657482802865` | `EXPERIMENT_LOG.md:297-354` |

Current selection: NewPL init36 RunD-style is the best current processed-input replacement checkpoint. NewIK1 official-input full pipeline is not selected because it is worse than both Run D PL-only and NewPL init36.

## Experiment Log Index

Detailed archive: `EXPERIMENT_LOG.md`.

| Experiment | Summary | Claim support | Detail lines |
|---|---|---|---|
| EXP-20260604-001 | Processed orientation input improves frozen official GPNet S4 from `42.522402` to `38.753660`. | validation result | `EXPERIMENT_LOG.md:5-36` |
| EXP-20260604-002 | Consistent processed IMU v2 audit confirms orientation-only/RMB-only correction and identical v1/v2 S4 `38.753660`. | validation result | `EXPERIMENT_LOG.md:38-105` |
| EXP-20260604-003 | NewPL Run D with GT control supervision reaches processed-input score `38.694846`. | validation result | `EXPERIMENT_LOG.md:107-136` |
| EXP-20260604-004 | NewIK1 PL1-output TC fine-tune reaches local val loss `0.17848628610372544`. | bounded diagnostic | `EXPERIMENT_LOG.md:138-174` |
| EXP-20260604-005 | NewIK1 bone-length continuation improves local loss to `0.1783127911388874`. | bounded diagnostic | `EXPERIMENT_LOG.md:176-214` |
| EXP-20260604-006 | Stronger pRJ/control NewIK1 continuation worsens local loss to `0.17918791025876998`. | bounded diagnostic | `EXPERIMENT_LOG.md:216-254` |
| EXP-20260604-007 | NewIK1 official-input full pipeline scores `38.70523069866002`, worse than Run D PL-only. | validation result | `EXPERIMENT_LOG.md:256-295` |
| EXP-20260605-001 | NewPL init36 RunD-style reaches best current processed-input score `38.625657482802865`. | validation result | `EXPERIMENT_LOG.md:297-354` |

## Implementation Records

### 2026-06-05 — Three-document experiment documentation split

User request: Split bulky project status into three documents; keep `PROJECT_STATUS.md` lightweight, move detailed experiment evidence to `EXPERIMENT_LOG.md`, and maintain recent replacement versions in `RECENT_REPLACEMENT_VERSIONS.md`. Also update related skills so future work follows this structure.

Changed files:

```text
/home/lingfeng/.claude/skills/universal-project-document-controller/SKILL.md
/home/lingfeng/.claude/skills/experiment-orchestrator/SKILL.md
/home/lingfeng/.claude/skills/sparse-imu-mocap/SKILL.md
EXPERIMENT_LOG.md
RECENT_REPLACEMENT_VERSIONS.md
PROJECT_STATUS.md
```

Work performed:

- Updated the project-document controller skill to recognize the three-document workflow.
- Updated the experiment-orchestrator skill so status writeback separates concise status, version records, and detailed logs.
- Updated the sparse-IMU mocap skill so every GlobalPose module-replacement iteration receives an explicit version name and version record.
- Created `EXPERIMENT_LOG.md` as the detailed archive.
- Created `RECENT_REPLACEMENT_VERSIONS.md` as the replacement-version index.
- Rewrote `PROJECT_STATUS.md` as the lightweight first-read document with line references into `EXPERIMENT_LOG.md`.

Design details:

- `PROJECT_STATUS.md` keeps baseline architecture, module contracts, current best evidence, and indices.
- `RECENT_REPLACEMENT_VERSIONS.md` records version-level design/effect/comparison/conclusion/artifacts.
- `EXPERIMENT_LOG.md` records detailed commands/artifacts/metrics/interpretation and should not be read by default for new experiment design.

Validation:

- Documentation content was generated from verified existing experiment records and artifacts already read in this session.
- Not checked with a final shell validation yet in this record.

Documentation updated:

- `PROJECT_STATUS.md`
- `RECENT_REPLACEMENT_VERSIONS.md`
- `EXPERIMENT_LOG.md`
- related skills under `/home/lingfeng/.claude/skills/`

Risks/blockers:

- NewIK2 remains unresolved: exact confirmed NewIK2 artifact paths were not found in this documentation pass.
- `EXPERIMENT_LOG.md` line references assume the current file structure and should be rechecked if entries are inserted above existing records.

Next action:

- Verify file status and line references, then locate exact NewIK2 artifacts or ask the user for paths.

### 2026-06-05 — Archive NewPL init36 RunD-style experiment

User request: Archive the completed recent experiment according to the latest three-document project-document workflow.

Changed files:

```text
PROJECT_STATUS.md
RECENT_REPLACEMENT_VERSIONS.md
EXPERIMENT_LOG.md
```

Work performed:

- Updated ACTIVE SUMMARY to select NewPL init36 RunD-style as the current best processed-input replacement.
- Added `newpl_init36_v1` / `newpl_init36_第一版` to `RECENT_REPLACEMENT_VERSIONS.md`.
- Appended `EXP-20260605-001` to `EXPERIMENT_LOG.md` with the init36 design, cache/training/eval artifacts, metrics, and interpretation.
- Updated the NewPL module card, NewPL loss record, S4 score table, experiment index, known risks, and next actions in this status file.

Design details:

- Init feature: `offset_r[18] + pRL[15] + gR0[3] = 36D`.
- Frame input remains official PL 84D.
- Frame output remains official PL 18D.
- Training follows historical RunD-style losses with partial loading from Run D into the expanded init encoder.

Validation:

- Training completed 60 epochs with `best_loss=0.18789918906986713` at epoch 60.
- Streaming S4 evaluation completed for both `best_loss.pt` and `last.pt`; both scored `38.625657482802865` with finite outputs.
- Documentation consistency checked by targeted reads/edits in this session; shell validation was not run because this directory is not reported as a git repository in the Claude environment.

Documentation updated:

- `PROJECT_STATUS.md`
- `RECENT_REPLACEMENT_VERSIONS.md`
- `EXPERIMENT_LOG.md`

Risks/blockers:

- `EXPERIMENT_LOG.md` line references are approximate after appending; use section IDs if exact line numbers drift.
- NewIK2 remains unresolved: exact confirmed NewIK2 artifact paths were not found.

Next action:

- Use NewPL init36 `best_loss.pt` / `last.pt` as the PL1 upstream checkpoint for future downstream replacement experiments.

## Known Issues and Risks

- Do not call the processed input a full IMU reprocessing result. It is orientation-only / RMB-only correction under the current stored-field convention.
- Do not describe `tail_update=4` as a declared four-frame output latency. It is a control-point revision window.
- NewPL init36 runtime requires `offset_r` for its 36D stream initialization; do not silently fall back to `target[0]` for mainline init36 training/eval.
- Do not relabel NewIK1/IK artifacts as NewIK2 without exact evidence that the replaced module is IK-s2 (`iknet.net2`) and the output contract is 90D reduced rotations.
- Local NewIK1 validation loss does not necessarily predict full S4 score.
- Detailed experiment logs should not be read wholesale by default because they are intended to grow large.

## Next Actions

1. Use NewPL init36 as the current PL1 upstream checkpoint for future downstream replacement experiments.
2. Locate exact NewIK2 artifact paths if they exist.
3. If NewIK2 artifacts are confirmed, add version entries such as `newik2_v1` / `newik2_第一版` and `newik2_v2` / `newik2_第二版` with design, effect, comparison, conclusion, and artifacts.
4. When running future long training, keep plateau early stopping and evaluate both `best_loss.pt` and `last.pt` because PL validation loss and streaming S4 can diverge.
