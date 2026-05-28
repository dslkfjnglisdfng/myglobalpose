# Sensor Offset Plan And Review

Last updated: 2026-05-27

This document is the sensor-offset data-processing control document for the GlobalPose working cache. It is separate from `IMPLEMENTATION_PLAN_AND_REVIEW.md`, which tracks L4 network implementation and training plans.

## ACTIVE SUMMARY

Current stage: K2 AMASS paired-overlay cache, AMASS training, TotalCapture original-input fine-tune, and TotalCapture Roffset A/B validation completed.
Current task: document the executed K2 data/cache contract and the TotalCapture `R_JS` orientation-offset sidecar ablation.
Review state: Implemented, validation completed on TotalCapture S4; not approved for final test.
Current changed files: `SENSOR_OFFSET_PLAN_AND_REVIEW.md`, `IMPLEMENTATION_PLAN_AND_REVIEW.md`, `ARCHITECTURE_OVERVIEW.md`, `net.py`, `l4_tail_update_qstate.py`, `l4_train_loss_ablation.py`, `l4_train_diverse_short.py`, `l4_generate_k2_overlay_cache.py`, `l4_generate_totalcapture_orientation_offset_cache.py`, and generated K2/TC cache and experiment artifacts.
Current module: K2 rot6d L4 input with sequence-level `offset_r` + first-frame RNN hidden initialization.
Current experiment: K2 AMASS paired-overlay training plus TotalCapture original-input and `R_JS` orientation-offset A/B fine-tunes.
Current result: K2 overlay cache produced 649 pairs / 1298 records from the old no-offset AMASS neural cache plus current offset-augmented `aM` and `imu_offset_r/r_JS`. TotalCapture S4 K2 original-input control improved the scalar validation score versus H1/J1/K1, but still trailed the GlobalPose baseline on Global Joint/Mesh and jitter. Roffset A/B produced effectively identical metrics to K2 original-input control.
Current blocker: no execution blocker. Scientific blocker: K2 is only partial positive, and `R_JS` orientation correction has no measurable validation benefit in this run.
Next action: do not promote Roffset. If K2 continues, run zero/shuffled offset sanity checks and a smaller cache-loss-selected AMASS variant before any final-test consideration.
Git state: branch `main`; worktree has many existing untracked/modified L4 and data files.
CodeGraph state: indexed, 71 files, 1557 nodes, 2764 edges.

## 2026-05-27 - IMU offset participation audit: AMASS synthetic vs TotalCapture real

Purpose:

- Confirm the semantic role of `imu_offset_r` / `r_JS` in the processed caches.
- AMASS is synthetic: offset should participate in IMU acceleration construction.
- TotalCapture is real: offset should be calibration metadata / conditioning, not a replacement for real IMU observations.
- Do not train, do not modify networks, do not touch `test.py`, `MotionEvaluator`, official weights, or official raw datasets.

Cache paths audited:

```text
data/dataset_work/AMASS/globalpose_synth_shard*.pt
data/dataset_work/TotalCapture_globalpose_official/train.pt
data/dataset_work/TotalCapture_globalpose_official/val.pt
```

AMASS result:

- Current AMASS processed cache is the single-random-placement synthetic version.
- Each sequence has one saved sequence-level `r_JS[6,3]`.
- That `r_JS` was used to resynthesize `aM` / `aS`.
- `wM`, `RMB`, and `wS` were preserved because this pass sampled only IMU position, not orientation.

AMASS forward model:

```text
p_WS(t) = p_WJ(t) + R_WJ(t) r_JS
aM(t) = d2/dt2 p_WS(t)
aS(t) = RMB(t)^T (aM(t) - g_W)
```

AMASS evidence:

| check | result |
| --- | ---: |
| sequence count | `10669` |
| previous full resynthesis reconstruction error max | `0.0` |
| current cross-shard FK sample count | `107` |
| current sampled `r_JS -> aM` reconstruction error max | `0.0` |
| current sampled `aS=RMB^T(aM-g)` consistency error max | `0.0` |
| `wM` max change vs pre-single-random-placement backup | `0.0` |
| `RMB` max change vs pre-single-random-placement backup | `0.0` |
| `wS` max change vs pre-single-random-placement backup | `0.0` |
| `aM` diff p50 / p95 / p99 / p99.9 vs pre-single-random-placement | `1.153 / 3.312 / 7.141 / 17.842 m/s^2` |

AMASS conclusion:

```text
AMASS synthetic processing is correct: offset participates in acceleration construction.
```

TotalCapture result:

- Current TotalCapture processed cache does **not** store direct `aM/wM/RMB` fields.
- It stores raw-like real IMU fields:

```text
RIM, RSB, RIS, aS, wS, mS
```

- Estimated `imu_offset_r/r_JS`, `imu_offset_R/R_JS`, and `imu_offset_T/T_JS` were appended to the processed cache.
- The real IMU fields were not rewritten by GT pose + estimated offset forward prediction.

TotalCapture backup comparison:

| cache | sequences | direct `aM/wM/RMB` fields | max dynamic-field change vs `*.pt.bak_before_imu_offset` |
| --- | ---: | --- | ---: |
| `TotalCapture_globalpose_official/train.pt` | `36` | no | `0.0` |
| `TotalCapture_globalpose_official/val.pt` | `5` | no | `0.0` |
| TotalCapture total | `41` | no | `0.0` |

Fields compared against backup:

```text
pose
tran
RIM
RSB
RIS
aS
wS
mS
```

TotalCapture conclusion:

```text
TotalCapture real processing is correct: estimated offsets are metadata / conditioning fields only.
```

Why this distinction matters:

- AMASS synthetic data has no real IMU sensor stream; therefore changing `r_JS` should generate a matching synthetic acceleration stream.
- TotalCapture has real IMU observations; replacing them with `GT pose + estimated offset` predictions would leak mocap information into the IMU input and destroy the meaning of real-data evaluation.
- For TotalCapture, predicted acceleration from offset is allowed only as diagnostic evidence or reliability metadata, not as the default network input.

Diagnostic-only prediction fields:

- This pass did **not** add `aM_pred_from_offset`, `aS_pred_from_offset`, `acceleration_residual`, `residual_fit_rms`, or `reliability_score` into the processed TotalCapture cache.
- Reason: existing residual diagnostic side outputs already cover this analysis, and the immediate question was whether default IMU inputs were overwritten.
- If these fields are needed later, they should be added as explicitly diagnostic fields or sidecar caches, not by replacing `aS/wS/RIS` or derived `aM/wM/RMB`.

Summary artifact:

```text
data/dataset_work/SensorOffset/imu_offset_input_participation_audit.json
```

Decision:

```text
OFFSET_INPUT_SEMANTICS_OK
```

## 2026-05-27 - Paired original/offset-augmented AMASS cache smoke

Purpose:

- Build the K2 data contract without starting long training.
- Represent each AMASS source sequence as two training views:
  - `original`: old synthetic IMU stream from the existing no-offset neural-only cache.
  - `offset_aug`: current synthetic IMU stream generated with newly sampled `r_JS`.
- Keep the same `pose/tran/q75` target for both views so the network sees that installation offset changes acceleration input, not the supervised body motion.

Two-view semantics:

| view | IMU source | offset source | target source |
|---|---|---|---|
| `original` | existing old/no-offset AMASS neural-only cache `aM/wM/RMB` | `original_imu_offset_r` from current augmented shard when available; fallback to cache offset field, then zero/default | old-cache `pose_gt/tran_gt/q75_gt` |
| `offset_aug` | current AMASS processed shard after single-random-placement resynthesis | current `imu_offset_r` / `r_JS` | rebuilt from the same processed `pose/tran` |

Important boundaries:

- `imu_offset_r` / `r_JS` is IMU installation position offset, not accelerometer or gyroscope measurement bias.
- Offset is sequence-level `[6,3]`, not a per-frame signal.
- The K2 first version uses offset only for RNN hidden-state initialization together with first-frame pose/IMU; it is not an offset prediction target.
- Offset is not concatenated to every frame.
- Pair consistency is designed as a future/default-off loss; the first planned K2 training is mixed-view training.

Paired cache generator:

```text
l4_generate_paired_offset_aug_cache.py
```

Smoke command:

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

Smoke output:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_amass_paired_offset_aug_neural_only_smoke/paired_cache_manifest.json
data/dataset_work/L4Cache/prephysics_pose_velocity_amass_paired_offset_aug_neural_only_smoke/paired_cache_shard00000.pt
data/dataset_work/L4Cache/prephysics_pose_velocity_amass_paired_offset_aug_neural_only_smoke/paired_cache_smoke_summary.json
```

Smoke result:

| check | result |
|---|---:|
| records | `4` |
| pairs | `2` |
| paired sequences | `CMU/55/55_28_poses`, `CMU/56/56_01_poses` |
| frame count per record | `120` |
| finite check | passed |
| `pose_gt/tran_gt/q75_gt` original-vs-aug max diff | `0.0` |
| `RMB` original-vs-aug max diff | `0.0` |
| `wM` original-vs-aug max diff | `7.9e-6` |
| `aM` original-vs-aug mean norm diff | `3.2666`, `4.5312` |

Smoke field contract:

| field | shape | note |
|---|---:|---|
| `q75_prephysics` | `[120,75]` | per-view frozen prephysics q state |
| `pose_prephysics` | `[120,24,3,3]` | per-view frozen prephysics pose |
| `v_root_vr` | `[120,3]` | per-view VR root velocity |
| `stationary_prob` | `[120,5]` | per-view VR contact/stationary logits/probabilities |
| `aM` | `[120,6,3]` | original or offset-augmented acceleration view |
| `wM` | `[120,6,3]` | unchanged up to numerical drift |
| `RMB` | `[120,6,3,3]` | unchanged |
| `q75_gt` | `[120,75]` | shared target |
| `pose_gt` | `[120,24,3,3]` | shared target |
| `tran_gt` | `[120,3]` | shared target |
| `offset_r` | `[6,3]` | sequence-level installation offset |

Observed offset norms:

| sequence | original mean / max | offset_aug mean / max |
|---|---:|---:|
| `CMU/55/55_28_poses` | `0.1732 / 0.2311 m` | `0.1746 / 0.2409 m` |
| `CMU/56/56_01_poses` | `0.1742 / 0.2320 m` | `0.1903 / 0.2796 m` |

Full-cache caveat:

- The current original-view AMASS neural-only cache is `prephysics_pose_velocity_amass_cache_diverse7_neural_only`; it is a diverse7 subset.
- A full 10669-sequence paired AMASS cache requires a full original-view neural-only cache, likely regenerated from the pre-single-random-placement backup shards.
- The existing smoke proves the paired contract, not the availability of a full original+augmented pair set.

## 2026-05-27 - K2 executed overlay cache and TotalCapture Roffset sidecar

Purpose:

- Execute the K2 branch without rerunning official GlobalPose prephysics on AMASS.
- Reuse old no-offset AMASS neural-only pose/prephysics/target fields, while overlaying current offset-augmented synthetic acceleration and sequence-level installation offset.
- Fine-tune on TotalCapture with real IMU preserved, then test whether `R_JS` orientation-offset corrected L4-only IMU inputs help on S4 validation.

AMASS K2 overlay cache:

```text
data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
```

Generator:

```text
l4_generate_k2_overlay_cache.py
```

AMASS K2 cache semantics:

| view | pose/q/prephysics/target source | IMU source | offset source | note |
|---|---|---|---|---|
| `original` | old no-offset neural-only cache | old-cache `aM/wM/RMB` | `original_imu_offset_r` when available, otherwise existing cache/default offset | original synthetic view |
| `offset_aug_overlay` | old no-offset neural-only cache | offset-augmented AMASS `aM`; old-cache `wM/RMB` after consistency checks | current `imu_offset_r/r_JS` | synthetic position-offset acceleration view |

K2 does not regenerate official prephysics. It intentionally keeps `q75_prephysics`, `pose_prephysics`, `v_root_vr`, `stationary_prob`, `q75_gt`, `pose_gt`, and `tran_gt` from the old cache so the experiment isolates the L4 module's response to offset-augmented acceleration plus `offset_r` conditioning.

K2 overlay sanity:

| check | result |
|---|---:|
| pairs | `649` |
| records | `1298` |
| frames across records | `1118012` |
| skipped sequences | `20`, all due to strict `wM_diff_too_large` at about `1e-4` |
| `aM` mean norm diff, mean / max over sequences | `3.5920 / 10.8336` |
| `wM` max-abs diff, mean / max over sequences | `1.92e-5 / 9.97e-5` |
| `RMB` max-abs diff | `0.0` |
| offset norm mean / median / p90 / max | `0.1755 / 0.1812 / 0.2327 / 0.2852 m` |

K2 network use:

- Per-frame L4 input remains rot6d pose `144D` + IMU `aM/wM/RMB 90D` = `234D`.
- RNN initialization uses `offset_r 18D` + first-frame rot6d pose `144D` + first-frame IMU `90D` = `252D`.
- `offset_r` is sequence-level `[6,3]`.
- Offset is not concatenated to every frame, is not predicted, and has no offset loss.
- IMU proxy loss and contact loss are disabled.

TotalCapture Roffset sidecar caches:

```text
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_A/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_A/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/train_Roffset_B/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_ablation/val_Roffset_B/baseline_cache_manifest.json
```

Generator:

```text
l4_generate_totalcapture_orientation_offset_cache.py
```

TotalCapture sidecar semantics:

- The real TotalCapture processed caches are not overwritten.
- Official `aM/wM/RMB` remain the inputs to frozen GlobalPose PL/IK/VR.
- Corrected fields are stored as `l4_aM/l4_wM/l4_RMB` and are consumed only by the optional L4 feature path.
- `R_JS` / `imu_offset_R` has shape `[6,3,3]` per sequence.
- `offset_r` / `imu_offset_r` remains `[6,3]`.
- In this ablation, `aM` and `wM` are not directly rotated by `R_JS` because official `aM=RIM^T RIS aS + g` and `wM=RIM^T RIS wS` are already model/world-frame vectors.

Roffset correction candidates:

```text
Official: RMB = RIM^T RIS RSB
Roffset A: RSB_corr = R_JS^T, RMB_A = RIM^T RIS RSB_corr
Roffset B: RSB_corr = R_JS,   RMB_B = RIM^T RIS RSB_corr
```

TotalCapture sidecar smoke:

| cache | sequences | frames | skipped | `R_JS` det mean | RMB diff norm mean / max |
|---|---:|---:|---:|---:|---:|
| train Roffset A | `36` | `142902` | `0` | `1.0000001` | `0.6532 / 1.3338` |
| val Roffset A | `5` | `17223` | `0` | `1.0000` | `0.6881 / 0.7497` |
| train Roffset B | `36` | `142902` | `0` | `1.0000001` | `5.9753 / 6.2710` |
| val Roffset B | `5` | `17223` | `0` | `1.0000` | `4.8825 / 4.9291` |

TotalCapture S4 validation summary:

| method | score | Local SIP | Local Angle | Local Joint | Local Mesh | Global SIP | Global Angle | Global Joint | Global Mesh | Root Jitter | Joint Jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GlobalPose baseline | n/a | 10.4660 | 10.1339 | 4.8174 | 5.5372 | 10.7168 | 10.2551 | 4.6387 | 5.2893 | 0.2978 | 0.4951 |
| H1 Euler q75 retained | 42.2851 | 10.3875 | 10.0149 | 4.7800 | 5.4881 | 10.7212 | 10.2150 | 4.6351 | 5.3100 | 0.2983 | 0.4965 |
| J1 rot6d no-offset | 42.2495 | 10.3492 | 9.9365 | 4.7485 | 5.4571 | 10.7617 | 10.2558 | 4.6648 | 5.3605 | 0.2997 | 0.4984 |
| K1 offset hidden init, no offset-aug AMASS | 42.2962 | 10.3339 | 9.8805 | 4.7292 | 5.4397 | 10.8245 | 10.3086 | 4.7072 | 5.4200 | 0.3012 | 0.4997 |
| K2 TC original-input control | 42.1916 | 10.3406 | 9.9753 | 4.7597 | 5.4679 | 10.7150 | 10.2134 | 4.6637 | 5.3589 | 0.2987 | 0.4979 |
| K2 TC Roffset A | 42.1916 | 10.3406 | 9.9753 | 4.7597 | 5.4679 | 10.7150 | 10.2134 | 4.6637 | 5.3589 | 0.2987 | 0.4979 |
| K2 TC Roffset B | 42.1915 | 10.3406 | 9.9753 | 4.7597 | 5.4679 | 10.7150 | 10.2134 | 4.6637 | 5.3589 | 0.2987 | 0.4979 |

Decision:

```text
K2_ORIGINAL_INPUT_CONTROL = PARTIAL_POSITIVE_ON_TOTALCAPTURE_S4
TC_ROFFSET_A_B = DISCARD_FOR_NOW
```

Reason:

- K2 improves the scalar S4 validation score versus H1/J1/K1 and reduces J1/K1 Global SIP/Angle regression.
- K2 still trails the GlobalPose baseline on Global Joint/Mesh and jitter.
- Roffset A/B are effectively identical to the K2 original-input control, so this run provides no evidence that `R_JS` orientation correction helps the L4 path.
