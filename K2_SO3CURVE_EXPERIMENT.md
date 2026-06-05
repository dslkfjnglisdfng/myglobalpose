# K2_SO3Curve_v1 Experiment

## Summary

`K2_SO3Curve_v1` was implemented and validated as a new K2/L4 external refiner variant whose curve state is:

```text
state_so3 = [root translation 3D, 24 local rotation vectors 72D] = 75D
```

The network keeps the proven K2 frame contract:

- frame input: `rot6d pose 144D + IMU aM/wM/RMB 90D = 234D`;
- RNN init: `r_JS 18D + first-frame frame feature 234D = 252D`;
- original + offset-augmented AMASS paired training;
- `rnn_init_mode=offset_firstframe`;
- recent-L4 supervision with `recent_loss_frames=4`;
- `residual_scale=0.005`;
- `velocity_residual_scale=0.0`;
- original carticulate physics for S4 validation.

Result:

- The implementation is functional: conversion smoke, one-batch training smoke, AMASS training, TotalCapture finetune, and S4 MotionEvaluator all completed.
- It is **not better** than K2 recent-L4 or K2 dropout continue10 on S4.
- Main failure mode: Local/Global pose metrics are worse and Root/Joint Jitter are much worse. The representation is stable enough to run, but the current SO3 training/eval bridge is not a retained mainline yet.

## Why AMASS Pretraining Was Required

This is a new output representation, not a small loss tweak. Training only on TotalCapture S1-S3 would test whether the model can overfit a small real dataset, not whether the new SO3 control-point representation learns a stable sparse-IMU motion prior. Therefore the executed pipeline was:

```text
Stage 0: conversion + one-batch + original-physics smoke
Stage 1: AMASS paired original/offset_aug long training
Stage 2: TotalCapture S1-S3 finetune
Stage 3: TotalCapture S4 MotionEvaluator validation
```

## Implementation

### Files

| file | role |
|---|---|
| `k2_so3_curve.py` | SO3 curve decoder, rotvec conversion utilities, `StreamingTailUpdateSO3State` |
| `k2_so3curve_smoke.py` | conversion smoke and one-batch training smoke |
| `k2_so3curve_train.py` | batched SO3 AMASS/TC trainer |
| `l4_train_loss_ablation.py` | added `--model-type k2_so3curve_v1` support and SO3 fast-mode loss path |
| `l4_physics_adapter_eval.py` | loads SO3Curve checkpoints by `config.model_type` for original physics S4 validation |

### Decoder Contract

`SO3CurveStateDecoder(control)`:

| output | shape | meaning |
|---|---:|---|
| `q_so3` | `[B,T,75]` | root translation + local rotvec state |
| `qdot_so3` | `[B,T,75]` | spline derivative in rotvec coordinates; not physical angular velocity |
| `qddot_so3` | `[B,T,75]` | spline second derivative in rotvec coordinates |
| `pose_R` | `[B,T,24,3,3]` | local rotation matrices from Exp map |
| `euler_q75` | `[B,T,75]` | Euler XYZ compatibility output for existing physics interface |
| `euler_qdot/euler_qddot` | `[B,T,75]` | compatibility finite-difference derivative proxies |
| `angular_velocity` | `[B,T,24,3]` | diagnostic from relative rotation, not `rotvec_dot` |
| `angular_acceleration` | `[B,T,24,3]` | diagnostic finite difference of angular velocity |

Important distinction:

- `qdot_so3` is the derivative of the rotvec coordinates.
- It is **not** physical angular velocity.
- Angular velocity is derived from rotation matrices through relative rotation / `R^T Rdot` style diagnostics.

## Smoke Results

Artifact:

```text
data/experiments/k2_so3curve_v1/smoke_8f.json
data/experiments/k2_so3curve_v1/smoke_original_physics_8f.json
```

Smoke summary:

| check | result |
|---|---:|
| conversion status | ok |
| one-batch training status | ok |
| original physics 8-frame status | all_finite=true |
| one-batch loss | 0.433929 |
| rotvec norm mean / max | 0.139811 / 0.553224 |
| pose round-trip geodesic mean rad | 0.001424 |
| Euler compatibility geodesic mean rad | 0.001424 |
| qdot_so3 norm mean / max | 4.316929 / 7.271492 |
| qddot_so3 norm mean / max | 252.453766 / 641.851685 |
| angular velocity norm mean / max | 0.706318 / 3.371108 |
| angular acceleration norm mean / max | 16.061354 / 163.086090 |

## Training

### Stage 1: AMASS Long Training

Command:

```bash
python k2_so3curve_train.py \
  --train-cache data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json \
  --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json \
  --output-dir data/experiments/k2_so3curve_v1_amass_20ep_batched_v2 \
  --init-checkpoint data/experiments/l4_ablation_J1_rot6d_poseinput_tc_20ep_v1/best.pt \
  --experiment-name K2_SO3Curve_v1_AMASS_20ep_batched_v2 \
  --epochs 20 --window 61 --batch-size 16 --lr 1e-5 \
  --hidden-size 256 --residual-scale 0.005 --velocity-residual-scale 0.0 \
  --offset-init-scale 0.1 --recent-loss-frames 4 --max-val-sequences 100
```

Notes:

- Full AMASS paired original/offset_aug training set was used: 1298 records.
- Cache validation used 100 AMASS records for best-loss selection to keep training tractable.
- The trainer uses SO3 fast mode: training loss is computed directly from `q_so3/pose_R`; Euler conversion is reserved for physics/eval compatibility.
- Old Euler output heads were not loaded into the SO3 residual heads.

Artifacts:

```text
data/experiments/k2_so3curve_v1_amass_20ep_batched_v2/amass_best_loss.pt
data/experiments/k2_so3curve_v1_amass_20ep_batched_v2/amass_last.pt
data/experiments/k2_so3curve_v1_amass_20ep_batched_v2/train_result.json
```

AMASS loss:

| epoch | train loss | cache-val loss | pose geodesic | q_body | root_velocity | tail_update | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.331060 | 0.273290 | 0.153144 | 0.011158 | 0.004083 | 0.000528 | 88.18 |
| 20 | 0.371239 | 0.273147 | 0.153651 | 0.011330 | 0.003775 | 0.000084 | 86.59 |

Best AMASS checkpoint: epoch 20, cache-val loss `0.273147`.

### Stage 2: TotalCapture Finetune

Command:

```bash
python k2_so3curve_train.py \
  --train-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_train_official_neural_only_offset_r/baseline_cache_manifest.json \
  --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json \
  --output-dir data/experiments/k2_so3curve_v1_tc_finetune_10ep \
  --init-checkpoint data/experiments/k2_so3curve_v1_amass_20ep_batched_v2/amass_best_loss.pt \
  --experiment-name K2_SO3Curve_v1_TC_finetune_10ep \
  --epochs 10 --window 61 --batch-size 8 --lr 1e-5 \
  --hidden-size 256 --residual-scale 0.005 --velocity-residual-scale 0.0 \
  --offset-init-scale 0.1 --recent-loss-frames 4 --max-val-sequences 0
```

Artifacts:

```text
data/experiments/k2_so3curve_v1_tc_finetune_10ep/tc_best_loss.pt
data/experiments/k2_so3curve_v1_tc_finetune_10ep/tc_last.pt
data/experiments/k2_so3curve_v1_tc_finetune_10ep/train_result.json
```

TC loss:

| epoch | train loss | S4 cache-val loss | pose geodesic | q_body | root_velocity | tail_update | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.242896 | 0.212376 | 0.216441 | 0.013117 | 0.000335 | 0.000046 | 4.87 |
| 10 | 0.235886 | 0.211894 | 0.210759 | 0.012214 | 0.000268 | 0.000833 | 4.15 |

Best TC checkpoint: epoch 10, cache-val loss `0.211894`.

## S4 MotionEvaluator

Validation commands:

```bash
python l4_physics_adapter_eval.py --checkpoint data/experiments/k2_so3curve_v1_amass_20ep_batched_v2/amass_best_loss.pt --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json --output-json data/experiments/k2_so3curve_v1_amass_20ep_batched_v2/s4_validation_amass_best_loss.json --physics-mode original

python l4_physics_adapter_eval.py --checkpoint data/experiments/k2_so3curve_v1_tc_finetune_10ep/tc_best_loss.pt --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json --output-json data/experiments/k2_so3curve_v1_tc_finetune_10ep/s4_validation_tc_best_loss.json --physics-mode original

python l4_physics_adapter_eval.py --checkpoint data/experiments/k2_so3curve_v1_tc_finetune_10ep/tc_last.pt --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json --output-json data/experiments/k2_so3curve_v1_tc_finetune_10ep/s4_validation_tc_last.json --physics-mode original
```

Full S4 comparison:

| Method | Score | Local SIP | Local Angle | Local Joint | Local Mesh | Global SIP | Global Angle | Global Joint | Global Mesh | Root Jitter | Joint Jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GlobalPose baseline | 42.522402 | 10.466050 | 10.133907 | 4.817390 | 5.537234 | 10.716775 | 10.255115 | 4.638654 | 5.289347 | 0.297783 | 0.495126 |
| K2 recent-L4 | 42.159255 | 10.331157 | 9.943425 | 4.749234 | 5.457094 | 10.721658 | 10.216917 | 4.661891 | 5.362637 | 0.299162 | 0.498475 |
| K2 dropout continue10 | 42.147905 | 10.327942 | 9.936177 | 4.745865 | 5.453482 | 10.723170 | 10.214663 | 4.663808 | 5.365855 | 0.299181 | 0.498454 |
| K2 L4loss continue | 42.261841 | 10.370317 | 10.010023 | 4.767443 | 5.486629 | 10.704034 | 10.232511 | 4.631958 | 5.316816 | 0.302661 | 0.501705 |
| K2 SO3 AMASS-only | 42.970838 | 10.510187 | 10.184102 | 4.845468 | 5.568089 | 10.882861 | 10.430471 | 4.705670 | 5.371423 | 0.445992 | 0.810248 |
| K2 SO3 TC best_loss | 42.950928 | 10.508704 | 10.172729 | 4.841045 | 5.563102 | 10.882927 | 10.423634 | 4.707289 | 5.373316 | 0.446006 | 0.810130 |
| K2 SO3 TC last | 42.950928 | 10.508704 | 10.172729 | 4.841045 | 5.563102 | 10.882927 | 10.423634 | 4.707289 | 5.373316 | 0.446006 | 0.810130 |

## Interpretation

Against K2 recent-L4:

- Score is worse by about `+0.791673`.
- Local Angle is worse by about `+0.229304`.
- Local Mesh is worse by about `+0.106008`.
- Global Mesh is slightly worse by about `+0.010679`.
- Root Jitter is much worse by about `+0.146844`.
- Joint Jitter is much worse by about `+0.311655`.

Against K2 dropout continue10:

- Score is worse by about `+0.803023`.
- Local Angle is worse by about `+0.236552`.
- Local Mesh is worse by about `+0.109620`.
- Root/Jitter are much worse.

AMASS-only and TC-finetuned SO3Curve are very close, which suggests TC finetune did not meaningfully correct the representation mismatch. This is not a simple under-finetuning issue after 10 epochs; the more likely issues are:

- SO3 rotvec residual heads are too conservative or poorly aligned with the existing Euler-q75 physics/eval interface.
- The Euler compatibility bridge may be introducing derivative/pose behavior that differs from the old Euler control-point model.
- The SO3 fast training loss is not identical to the old K2 Euler loss; it preserves pose supervision but changes q-body loss semantics to rotvec coordinates.
- The current external-refiner setup still passes final pose through the original carticulate physics path, whose tuning was developed around Euler-q75 target behavior.

## Decision

Do not promote `K2_SO3Curve_v1` as the main K2/L4 line yet.

It is useful as a working implementation and diagnostic branch, but the current evidence is worse than K2 recent-L4 and K2 dropout continue10. The next useful SO3 work would be:

1. Add a stricter equivalence test comparing Euler K2 and SO3 K2 when residuals are zero and when small residuals are injected.
2. Audit whether Euler compatibility `qdot/qddot` sent to diagnostics/physics are causing jitter amplification.
3. Try a smaller SO3 residual scale or trust-region on rotvec deltas.
4. Only after SO3 matches Euler K2 at small residuals, consider contact or physics LSQR qdot targets.

Do not move to contact/dynamics losses on SO3Curve until the basic representation no longer worsens Local and Jitter.
