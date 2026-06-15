#!/usr/bin/env bash
set -euo pipefail

# NewPL v6 diagnostic: keep the corrected next-control architecture, but use
# one-step next/control supervision only for gR1. Current-frame pRB remains
# trained by the ordinary current pRB/control/temporal losses.

export EXP="${EXP:-/tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613}"
export EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-newpl_v6_gR1_nextonly_smoothacc}"
export CANDIDATE_PREFIX="${CANDIDATE_PREFIX:-newpl_v6_gR1nextonly_smoothacc}"

# Preserve current-frame pRB fitting. Keep current gR1 ordinary supervision,
# then put the predictive/control branch pressure only on gR1.
export PRB_WEIGHT="${PRB_WEIGHT:-1.0}"
export GR1_WEIGHT="${GR1_WEIGHT:-1.0}"
export GT_CONTROL_PRB_WEIGHT="${GT_CONTROL_PRB_WEIGHT:-0.3}"
export GT_CONTROL_GR1_WEIGHT="${GT_CONTROL_GR1_WEIGHT:-0.2}"
export PRB_DOT_WEIGHT="${PRB_DOT_WEIGHT:-0.03}"
export PRB_DDOT_SMOOTH_WEIGHT="${PRB_DDOT_SMOOTH_WEIGHT:-0.000001}"
export GR1_DOT_WEIGHT="${GR1_DOT_WEIGHT:-0.03}"
export GR1_DDOT_WEIGHT="${GR1_DDOT_WEIGHT:-0.001}"

# Disable all pRB losses in the auxiliary next-control branch.
export NEXT_PRB_WEIGHT="${NEXT_PRB_WEIGHT:-0.0}"
export NEXT_GT_CONTROL_PRB_WEIGHT="${NEXT_GT_CONTROL_PRB_WEIGHT:-0.0}"
export NEXT_PRB_VEL_WEIGHT="${NEXT_PRB_VEL_WEIGHT:-0.0}"
export NEXT_PRB_ACC_WEIGHT="${NEXT_PRB_ACC_WEIGHT:-0.0}"
export LAST_CONTROL_PRB_WEIGHT="${LAST_CONTROL_PRB_WEIGHT:-0.0}"
export NEXT_TAIL4_CONTROL_PRB_WEIGHT="${NEXT_TAIL4_CONTROL_PRB_WEIGHT:-0.0}"

# Keep next/control supervision for gravity only.
export NEXT_GR1_WEIGHT="${NEXT_GR1_WEIGHT:-2.0}"
export NEXT_GT_CONTROL_GR1_WEIGHT="${NEXT_GT_CONTROL_GR1_WEIGHT:-0.5}"
export NEXT_GR1_VEL_WEIGHT="${NEXT_GR1_VEL_WEIGHT:-0.05}"
export NEXT_GR1_ACC_WEIGHT="${NEXT_GR1_ACC_WEIGHT:-0.002}"
export LAST_CONTROL_GR1_WEIGHT="${LAST_CONTROL_GR1_WEIGHT:-0.5}"
export NEXT_TAIL4_CONTROL_GR1_WEIGHT="${NEXT_TAIL4_CONTROL_GR1_WEIGHT:-0.35}"
export NEXT_CONTROL_DELTA_PRIOR_WEIGHT="${NEXT_CONTROL_DELTA_PRIOR_WEIGHT:-0.01}"

exec "$(dirname "$0")/run_newpl_v6_next_control_smoothacc_gR1_20260613.sh" "$@"
