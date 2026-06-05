# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository root

Work from `/home/lingfeng/projects/GlobalposeMy/GlobalPose`. The outer `/home/lingfeng/projects/GlobalposeMy` directory is not the project root and does not contain the initialized CodeGraph index.

## Environment and commands

This is a script-driven research repository; no top-level `README.md`, `pyproject.toml`, `setup.py`, or `requirements.txt` was found during initialization. Use each script's `--help` output for exact current options.

Common environment setup used by repository scripts:

```bash
ENV=/home/lingfeng/remote-envs/globalpose-gpu-py310
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
```

Useful command patterns:

```bash
python pl_curve_train.py --help
python l4_physics_adapter_eval.py --help
python newik1_control_train.py --help
python tools/experiment_orchestrator.py --help
```

Train PLCurve-style prephysics modules:

```bash
python pl_curve_train.py \
  --train-cache <train_manifest.json> \
  --val-cache <val_manifest.json> \
  --output-dir <output_dir> \
  --experiment-name <name> \
  --epochs <n> \
  --window 61 \
  --batch-size <n>
```

Evaluate L4/K2/PL-curve physics adapters:

```bash
python l4_physics_adapter_eval.py \
  --checkpoint <checkpoint.pt> \
  --val-cache <val_manifest.json> \
  --output-json <metrics.json> \
  --physics-mode original
```

Train NewIK1 control-point modules:

```bash
python newik1_control_train.py \
  --train-cache <train_manifest.json> \
  --val-cache <val_manifest.json> \
  --output-dir <output_dir> \
  --experiment-name <name> \
  --epochs <n> \
  --window 61
```

Run the full NewIK1 official-input pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_newik1_official_input_full.sh
```

Launch the configured background training/resume wrapper:

```bash
bash scripts/launch_train_background.sh
```

## Architecture overview

`net.py` is the central model file. It defines `GPNet`, the official GlobalPose pipeline, and the integration points for current research backends. The official stages are PL (`plnet`), IK (`iknet.net1` and `iknet.net2`), and VR/translation (`vrnet`). Constructor options select variants such as L4 prephysics, PL curve backends, IK1 curve/control-point/official-input backends, and physics modes.

`process.py` and related processing scripts convert TotalCapture, DIP-IMU, and AMASS-style data into the repository's working tensors and cache formats. `test.py` contains the official-style evaluation path and `MotionEvaluator` metrics.

Experiment families are organized as script/module groups rather than a conventional package layout:

- `pl_curve*` implements NewPL/PLCurve training, cache usage, and streaming control-point prediction.
- `newik1_*` and `ik1_curve*` implement IK1 replacement/refinement experiments.
- `l4_*` and `k2_*` implement prephysics q-state and SO(3)-curve variants before the physics backend.
- `full_curve_globalpose.py` contains full-chain curve modules using the same streaming control-point pattern.
- `tools/experiment_orchestrator.py` schedules task-file-driven experiments, GPU allocation, logging, summary parsing, and `PROJECT_STATUS.md` writeback.
- `articulate/` and `carticulate/` are local support packages for math, body models, and the original physics backend.

`PROJECT_STATUS.md` is the best high-level source for current experiment state and conclusions. It records that the strongest current mainline result is the processed-input PL direction, and that recent NewIK1 stage-2 work did not beat Run D PL-only or Original GPNet with processed input.

## IMU and model conventions

Sparse IMU processing uses six sensors. Baseline formulas documented in project status are:

```text
RMB = RIM^T @ RIS @ RSB
aM  = RIM^T @ RIS @ aS + [0, -9.8, 0]
wM  = RIM^T @ RIS @ wS
```

Current processed IMU data is orientation-only processed: `l4_aM == aM`, `l4_wM == wM`, and `l4_RMB != RMB`. Improvements from processed input should be attributed to corrected orientation, not changed acceleration or gyro values.

`GPNet.forward_frame` builds neural PL inputs from `aM`, `wM`, and `RMB` using body-relative acceleration/gyro, relative orientations, and gravity/root direction features.

K2/SO3Curve state uses:

```text
state_so3 = [root translation 3D, 24 local rotation vectors 72D] = 75D
```

For K2/SO3Curve, `qdot_so3` is the derivative of rotation-vector coordinates, not physical angular velocity.

NewIK1_ControlPoint_v1 predicts the 72D IK1 state:

```text
pRJ[69] + gR2[3]
```

For control-point modules, `tail_update=4` means the stream can revise the most recent four control points before appending the new one. Do not describe it as a declared four-frame output latency unless a specific experiment explicitly establishes that.

## Streaming cache rule

Future mainline training caches must be streaming-compatible:

1. reset or initialize RNN/module state for each sequence;
2. call official `GPNet.rnn_initialize(init_pose, init_velocity)`;
3. run frame-by-frame forward;
4. save upstream module streaming outputs;
5. train downstream stages on those streaming outputs.

Do not use this old batch/cache contract as a new mainline training cache:

```python
gpnet.plnet([(pl_input, pl_target[0])])
```

That batch/cache form is reserved for historical diagnostics or explicit ablation controls.

## CodeGraph

This repository has CodeGraph initialized under `/home/lingfeng/projects/GlobalposeMy/GlobalPose/.codegraph`. Use CodeGraph for structural questions such as symbol definitions, callers/callees, impact analysis, and unfamiliar module surveys. Use literal search/read for exact strings, comments, logs, or files already opened. Do not query CodeGraph immediately after editing a file; allow the watcher to sync first.
