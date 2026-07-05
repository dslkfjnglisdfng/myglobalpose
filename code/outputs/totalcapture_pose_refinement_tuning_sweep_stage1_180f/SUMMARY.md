# TotalCapture Pose Refinement Tuning Sweep

Diagnostic only: no network training and no PL/IK/VR changes.

## Scope

- dataset: `/home/lingfeng/projects/GlobalposeMy/GlobalPose/data/dataset_work/TotalCapture_globalpose_official/test.pt`
- max sequences / frames: `1` / `180`
- max configs per stage: `full grid`
- Stage 1 passed: `False`
- Stage 2 ran: `False`

## Result

- total verdict rows: `18`
- pass rows: `0`
- failed rows: `18`

No configuration passed all gates in this bounded run.

Failure causes to inspect:
- `fit_not_improved`: 18
- `heldout_regressed`: 18
- `jerk_regressed`: 18
- `foot_sliding_regressed`: 18

Interpretation candidates: acc target may be too noisy, second differences may make optimization unstable, translation-only may not explain the residual, pose/tran priors may be too strong or too weak, robust-loss gradients may be poorly scaled, or the next version may need low-frequency-only acceleration loss instead of raw acceleration loss.
