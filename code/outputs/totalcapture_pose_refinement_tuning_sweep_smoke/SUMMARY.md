# TotalCapture Pose Refinement Tuning Sweep

Diagnostic only: no network training and no PL/IK/VR changes.

## Scope

- dataset: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/data/dataset_work/TotalCapture_globalpose_official/test.pt`
- max sequences / frames: `1` / `48`
- max configs per stage: `1`
- Stage 1 passed: `False`
- Stage 2 ran: `False`

## Result

- total verdict rows: `3`
- pass rows: `0`
- failed rows: `3`

No configuration passed all gates in this bounded run.

Failure causes to inspect:
- `fit_not_improved`: 3
- `jerk_regressed`: 3
- `foot_sliding_regressed`: 3
- `heldout_regressed`: 2

Interpretation candidates: acc target may be too noisy, second differences may make optimization unstable, translation-only may not explain the residual, pose/tran priors may be too strong or too weak, robust-loss gradients may be poorly scaled, or the next version may need low-frequency-only acceleration loss instead of raw acceleration loss.
