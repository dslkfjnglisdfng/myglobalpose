# K2 Control Point Physics Refine v1-v3

This document records the test-time / inference-time control-point refinement experiment series for the current K2/L4 external refiner.

No neural network training was performed. No S5 final test was run. Official weights, `test.py`, `MotionEvaluator`, PIP physics, original carticulate physics, and C++ code were not modified.

## Why This Experiment

The previous K2/L4 loss sweeps showed a repeated failure mode:

- Global Joint / Global Mesh can improve.
- Local Angle / Local Mesh / Root Jitter / Joint Jitter / Score often get worse.

That suggests training-time loss weighting can still let the model trade local pose and temporal dynamics for global surface improvement. This experiment moves the robotics trajectory-optimization idea to test time: use the neural K2/L4 control points as a nominal trajectory and optimize only the control points locally at inference.

The intended source idea is:

```text
neural trajectory initialization
-> control point / spline trajectory refinement
-> q, qdot, qddot derived from the same curve
-> small consistency and smoothness objectives
```

## Module

Implemented files:

| File | Role |
|---|---|
| `control_point_physics_refine.py` | differentiable control-point refinement objective and optimizer; v1/v2/v3 losses |
| `l4_tail_update_qstate.py` | optional inference-time hook after normal K2/L4 control-buffer update |
| `l4_physics_adapter_eval.py` | CLI switches and diagnostics for S4 validation |

Default behavior is unchanged. The module is active only with:

```bash
--control-point-refine
```

## Inputs and Outputs

Inputs:

| Name | Shape | Meaning |
|---|---:|---|
| `C_net` / `control_buffer` | `[B, T, 75]` | K2/L4 causal spline control points after the network update |
| `base_buffer` | `[B, T, 75]` | baseline q75 controls used for diagnostics |
| `q_net` | `[B, T+1, 75]` | decoded q75 from `C_net` plus ghost endpoint |
| `qdot_net` | `[B, T+1, 75]` | decoded generalized Euler-coordinate derivative from `C_net` |
| `qddot_net` | `[B, T+1, 75]` | decoded second derivative from `C_net` |
| `qdot_ref` | `[B, T+1, 69]` | finite difference of `q_net[..., 6:]` |
| contact / IMU | reserved | not used in v1 |

Outputs:

| Name | Shape | Meaning |
|---|---:|---|
| `C_refined` | `[B, T, 75]` | optimized control buffer |
| `q_refined` | `[B, T+1, 75]` | decoded from `C_refined` |
| `qdot_refined` | `[B, T+1, 75]` | decoded from `C_refined` |
| `qddot_refined` | `[B, T+1, 75]` | decoded from `C_refined` |
| `pose_refined` | `[24,3,3]` per frame | converted from current refined q75 frame |
| diagnostics | JSON fields | loss, delta-control norm, q drift, qdot/qddot norms |

The current v1 optimization acts only on body slice `q[6:]`. Root translation and root orientation are frozen unless `--cp-refine-include-root` is explicitly used.

## Objective

The optimization variable is `C`, not any network weight.

```text
C = C_net + delta_C
```

v1 objective:

```text
L =
  lambda_prior * ||C_body - C_net_body||^2
+ lambda_q     * ||q_body(C) - q_body(C_net)||^2
+ lambda_v     * SmoothL1(qdot_body(C), finite_difference(q_body(C_net)))
+ lambda_a     * ||qddot_body(C)||^2
```

Term meanings:

| Term | Purpose |
|---|---|
| `L_prior` | keep the control points near the K2/L4 network trajectory |
| `L_q` | preserve body pose trajectory and protect Local metrics |
| `L_v` | make curve velocity consistent with finite differences of the same nominal q trajectory |
| `L_a` | small acceleration smoothness; disabled or kept tiny because qddot scale is large |

v1 intentionally does not use GT, contact loss, IMU proxy, full dynamics residual, physics internal qdot, or qddot targets.

v2 adds contact-gated foot velocity:

```text
L_contact = gate * ||foot_vel_rootrel(q(C))||^2
```

The implemented v2 gate is a detached heuristic from the nominal `q_net` trajectory:

```text
gate = sigmoid((height_threshold - foot_height) * sharpness)
     * sigmoid((velocity_threshold - ||foot_vel(q_net)||) * sharpness)
```

The first v2 implementation exposed an important runtime issue: using the whole historical control buffer makes FK/contact computation grow with sequence length. v2 therefore adds `--cp-refine-window`; the retained v2 S4 uses `--cp-refine-window 8`, so only the most recent 8 causal control points are optimized.

v3 adds an optional curve dynamics proxy:

```text
L_dyn_proxy = SmoothL1(qddot_body(C)[t], finite_difference(qdot_body(C))[t])
```

This is not a rigid-body dynamics residual. It does not use carticulate `M(q)`, `h`, contact forces, or `J^T lambda`; it is only a small derivative-consistency proxy because the original carticulate physics solver was intentionally not modified.

## CLI

S4 validation with control point refinement:

```bash
CUDA_VISIBLE_DEVICES=0 /home/lingfeng/.conda/envs/globalpose-gpu/bin/python l4_physics_adapter_eval.py \
  --checkpoint data/experiments/l4_K2_TC_recentL4loss_10ep_v1/best_loss.pt \
  --val-cache data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_val_official_neural_only_offset_r/baseline_cache_manifest.json \
  --output-json <output.json> \
  --physics-mode original \
  --control-point-refine \
  --cp-refine-steps <steps> \
  --cp-refine-lr <lr> \
  --cp-refine-lambda-prior 1.0 \
  --cp-refine-lambda-q 1.0 \
  --cp-refine-lambda-v <lambda_v> \
  --cp-refine-lambda-a <lambda_a> \
  --cp-refine-lambda-contact <lambda_contact> \
  --cp-refine-lambda-dyn <lambda_dyn> \
  --cp-refine-window <recent_control_points>
```

`--cp-refine-persist-buffer` is available but not recommended from v1 evidence. Without it, each frame can use refined control points for current output while the next frame continues from the original network stream.

## Smoke

8-frame smoke used `s4_acting3`, original carticulate physics, no S5.

| Config | Score | finite | mean delta C | q body drift | Notes |
|---|---:|---|---:|---:|---|
| no refine | `37.563591` | yes | `0.000000` | `0.000000` | reference |
| primary `steps=10 lr=3e-3 v=0.03 a=3e-4` | `37.660232` | yes | `0.048657` | `0.005515` | stable but worse |
| `steps=10 lr=1e-3 v=0.03 a=1e-4` persist | `37.593212` | yes | `0.032459` | `0.003369` | least-bad non-tiny smoke among original sweep |
| no-persist `steps=10 lr=1e-3 v=0.03 a=1e-4` | `37.747486` | yes | `0.036845` | `0.003577` | worse |
| no-persist tiny `steps=1 lr=1e-4 v=0.01 a=0` | `37.567683` | yes | `0.000694` | `0.000068` | near no-op |
| no-persist tiny `steps=3 lr=1e-4 v=0.01 a=0` | `37.575650` | yes | `0.001983` | `0.000195` | still near no-op |

Smoke result: all variants are finite and original physics runs, but non-tiny optimization already tends to worsen short-window score.

v2/v3 smoke after adding the recent-window optimizer:

| Config | Score | C shape | q/qdot/qddot shape | finite | contact gate mean/max | active fraction | mean delta C | q drift |
|---|---:|---|---|---|---:|---:|---:|---:|
| v2 contact `steps=3 lr=1e-4 v=0.01 contact=0.01 win=8` | `37.575650` | `[1,8,75]` | `[1,9,75]` each | yes | `0.406 / 0.961` | `0.428` | `0.001983` | `0.000195` |
| v3 contact+dyn `steps=3 lr=1e-4 v=0.01 contact=0.01 dyn=1e-5 win=8` | `37.578477` | `[1,8,75]` | `[1,9,75]` each | yes | `0.406 / 0.961` | `0.428` | `0.002039` | `0.000185` |

Smoke JSONs:

- `data/experiments/control_point_physics_refine_v2/smoke_v2_shape_contact01_s3_lr1e4_win8_s4_acting3_8f.json`
- `data/experiments/control_point_physics_refine_v2/smoke_v3_shape_contact01_dyn1e5_s3_lr1e4_win8_s4_acting3_8f.json`

Debug tensor payloads for the 8-frame smoke are saved with actual `C_refined`, `q_refined`, `qdot_refined`, and `qddot_refined` tensors:

| Config | Tensor path | Records | Tensor shapes |
|---|---|---:|---|
| v2 contact01 s3 win8 | `data/experiments/control_point_physics_refine_v2/smoke_v2_tensor_contact01_s3_lr1e4_win8_debug_tensors.pt` | 8 frames | `C_refined [1,8,75]`, `q/qdot/qddot [1,9,75]` |
| v3 contact01 dyn1e-5 s3 win8 | `data/experiments/control_point_physics_refine_v2/smoke_v3_tensor_contact01_dyn1e5_s3_lr1e4_win8_debug_tensors.pt` | 8 frames | `C_refined [1,8,75]`, `q/qdot/qddot [1,9,75]` |

## S4 Validation

Machine-readable files:

- `data/experiments/control_point_physics_refine_v1/control_point_physics_refine_summary.json`
- `data/experiments/control_point_physics_refine_v1/control_point_physics_refine_summary.csv`
- `data/experiments/control_point_physics_refine_v2/control_point_physics_refine_v123_summary.json`
- `data/experiments/control_point_physics_refine_v2/control_point_physics_refine_v123_summary.csv`

Full S4 metrics, lower is better:

| Run | Score | Local SIP | Local Angle | Local Joint | Local Mesh | Global SIP | Global Angle | Global Joint | Global Mesh | Root Jitter | Joint Jitter | balanced | dC mean | q drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GlobalPose baseline | `42.522402` | `10.466050` | `10.133907` | `4.817390` | `5.537234` | `10.716775` | `10.255115` | `4.638654` | `5.289347` | `0.297783` | `0.495126` | `45.228619` | `0.000000` | `0.000000` |
| K2 dropout continue10 + original | `42.147905` | `10.327900` | `9.936200` | `4.745900` | `5.453500` | `10.723200` | `10.214700` | `4.663800` | `5.365900` | `0.299200` | `0.498500` | `42.149176` | `0.000000` | `0.000000` |
| P2 old qdot blend 0.5 | `42.346873` | `10.392272` | `10.032112` | `4.785996` | `5.503079` | `10.720663` | `10.252604` | `4.658218` | `5.334567` | `0.239058` | `0.480032` | `43.693590` | `0.000000` | `0.000000` |
| K2 recent-L4-loss | `42.159255` | `10.331157` | `9.943425` | `4.749234` | `5.457094` | `10.721658` | `10.216917` | `4.661891` | `5.362637` | `0.299162` | `0.498475` | `42.159255` | `0.000000` | `0.000000` |
| K2 CPRefine tiny s1 lr1e-4 v0.01 a0 no-persist | `42.157324` | `10.330729` | `9.942564` | `4.748854` | `5.456590` | `10.721444` | `10.216526` | `4.661818` | `5.362620` | `0.299575` | `0.499387` | `42.183834` | `0.000532` | `0.000054` |
| K2 CPRefine tiny s3 lr1e-4 v0.01 a0 no-persist | `42.152886` | `10.329308` | `9.940519` | `4.747753` | `5.455037` | `10.721083` | `10.216015` | `4.661718` | `5.362672` | `0.300571` | `0.501333` | `42.238233` | `0.001400` | `0.000148` |
| K2 CPRefine mid s10 lr1e-3 v0.03 a1e-4 persist | `42.512281` | `10.411536` | `10.027402` | `4.802406` | `5.526900` | `10.806404` | `10.309693` | `4.719404` | `5.430393` | `0.308016` | `0.506404` | `44.385770` | `0.004800` | `0.000450` |
| K2 CPRefine v2 contact01 s3 win8 | `42.152430` | `10.329088` | `9.940315` | `4.747694` | `5.454951` | `10.721009` | `10.216059` | `4.661740` | `5.362753` | `0.300768` | `0.501601` | `42.247074` | `0.001950` | `0.000184` |
| K2 CPRefine v3 contact01 dyn1e-5 s3 win8 | `42.163116` | `10.332263` | `9.942979` | `4.749382` | `5.457510` | `10.723506` | `10.218095` | `4.663046` | `5.364401` | `0.301618` | `0.502935` | `42.305601` | `0.001589` | `0.000155` |

Delta vs K2 recent-L4:

| Run | dScore | dLocalAngle | dLocalMesh | dGlobalMesh | dRootJitter | dJointJitter | dScore vs dropout |
|---|---:|---:|---:|---:|---:|---:|---:|
| K2 CPRefine tiny s1 lr1e-4 v0.01 a0 no-persist | `-0.001930` | `-0.000861` | `-0.000504` | `-0.000017` | `+0.000414` | `+0.000912` | `+0.009419` |
| K2 CPRefine tiny s3 lr1e-4 v0.01 a0 no-persist | `-0.006369` | `-0.002906` | `-0.002057` | `+0.000035` | `+0.001410` | `+0.002858` | `+0.004981` |
| K2 CPRefine mid s10 lr1e-3 v0.03 a1e-4 persist | `+0.353026` | `+0.083977` | `+0.069806` | `+0.067756` | `+0.008854` | `+0.007929` | `+0.364376` |
| K2 CPRefine v2 contact01 s3 win8 | `-0.006825` | `-0.003110` | `-0.002144` | `+0.000116` | `+0.001606` | `+0.003126` | `+0.004525` |
| K2 CPRefine v3 contact01 dyn1e-5 s3 win8 | `+0.003861` | `-0.000446` | `+0.000416` | `+0.001764` | `+0.002456` | `+0.004460` | `+0.015211` |

## Selection

- Best by Score: `K2 dropout continue10 + original` (`42.147905`).
- Best by Global Mesh: `GlobalPose baseline` (`5.289347`).
- Best by Root/Joint Jitter: `P2 old qdot blend 0.5`, but it is not balanced.
- Best among CPRefine v1-v3 by Score: v2 contact01 `steps=3, lr=1e-4, lambda_v=0.01, lambda_contact=0.01, window=8`.
- Best balanced overall: `K2 dropout continue10 + original`.

## Interpretation

CPRefine v1 proves the engineering path works:

- control point optimization runs at inference;
- q/qdot/qddot are decoded from the refined curve;
- original carticulate physics still runs;
- outputs are finite;
- diagnostics are saved.

But v1 does not meet the requested success standard:

- tiny no-persist improves Score and Local slightly, but Root/Joint Jitter get worse.
- stronger or persistent refinement causes over-refinement and large Score/Local/Global/Jitter damage.
- no tested configuration reduces jitter while preserving Local and Global.
- v2 contact-gated foot velocity slightly improves Score/Local over v1 tiny s3, but still worsens Root/Joint Jitter relative to K2 recent-L4.
- v3 derivative-consistency proxy worsens Score, Local Mesh, Global Mesh, and Jitter. It should not be retained.

The best CPRefine result is therefore not a balanced improvement. It is a very small local pose correction, not a useful physics/jitter correction.

## Future Dynamics

Full rigid-body dynamics residual is still not enabled. A future version could consider:

```text
L_dyn = ||M(q) qddot + h||^2
L_dyn = ||M qddot + h - J^T lambda||^2
```

This should not be added before validating q/qdot/qddot layout and scale against carticulate.

## Recommendation

Do not promote K2_ControlPoint_PhysicsRefine_v1-v3 as a retained method for jitter reduction.

Recommended next step:

1. Keep the no-persist and recent-window design if this line continues.
2. Add a hard trust-region or line search that rejects a refinement step when jitter proxy, q drift, or short-window Local proxy increases.
3. Do not keep the v3 derivative proxy; it is not useful in S4.
4. For real jitter reduction, the more promising route remains a bounded qdot target inside the original Python LSQR physics solve after carticulate qdot layout diagnostics.
