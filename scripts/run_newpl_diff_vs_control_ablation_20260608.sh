#!/usr/bin/env bash
set -euo pipefail

ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
PY="$ENV/bin/python"

ROOT=${ROOT:-data/experiments/newpl_diff_vs_control_ablation_20260608}
CACHE_ROOT=${CACHE_ROOT:-data/experiments/newpl_v5_official_protocol_20260607/caches}
GT_CTRL_ROOT=${GT_CTRL_ROOT:-data/dataset_work/GTControlCache}
EVAL_ROOT="$ROOT/eval"
mkdir -p "$ROOT/logs" "$EVAL_ROOT"

NEWPL_V4=data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt

AMASS_PL_CACHE="$CACHE_ROOT/pl_amass_official_init36/pl_curve_cache_manifest.json"
DIP_TRAIN_PL_CACHE="$CACHE_ROOT/pl_dip_train_official_init36/pl_curve_cache_manifest.json"
DIP_VAL_PL_CACHE="$CACHE_ROOT/pl_dip_val_official_init36/pl_curve_cache_manifest.json"
DIP_TEST_RAW="$CACHE_ROOT/dip_test_with_offset_r/baseline_cache_manifest.json"
TC_TEST_RAW="$CACHE_ROOT/tc_test_official_with_offset_r/baseline_cache_manifest.json"

AMASS_CTRL="$GT_CTRL_ROOT/amass_train/gt_control_cache_manifest.json"
DIP_TRAIN_CTRL="$GT_CTRL_ROOT/dip_train/gt_control_cache_manifest.json"
DIP_VAL_CTRL="$GT_CTRL_ROOT/dip_val/gt_control_cache_manifest.json"

for required in "$AMASS_PL_CACHE" "$DIP_TRAIN_PL_CACHE" "$DIP_VAL_PL_CACHE" "$DIP_TEST_RAW" "$TC_TEST_RAW" "$AMASS_CTRL" "$DIP_TRAIN_CTRL" "$DIP_VAL_CTRL"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required cache: $required" >&2
    exit 2
  fi
done

COMMON_TRAIN_ARGS=(
  --init-size 36
  --window 61
  --hidden-size 512
  --tail-length 4
  --residual-scale 0.005
  --dropout 0.4
  --grad-clip 1.0
  --disable-ik-distill
  --baseline-pRB-weight 0.0
  --baseline-gR1-weight 0.0
  --control-point-prior-weight 0.3
  --tail-update-prior-weight 0.005
  --pRB-ddot-smooth-weight 0.000001
  --selection-metric control_physical
  --early-stop-min-delta 0.00000005
  --val-window-length 61
)

DIFF_LOSS_ARGS=(
  --gt-control-pRB-weight 0.0
  --gt-control-gR1-weight 0.0
  --pRB-dot-weight 0.03
  --pRB-ddot-weight 0.0003
  --gR1-dot-weight 0.03
  --gR1-ddot-weight 0.001
)

CONTROL_LOSS_ARGS=(
  --gt-control-pRB-weight 0.3
  --gt-control-gR1-weight 0.1
  --pRB-dot-weight 0.0
  --pRB-ddot-weight 0.0
  --gR1-dot-weight 0.0
  --gR1-ddot-weight 0.0
)

train_stage() {
  local variant="$1"
  local stage="$2"
  local train_cache="$3"
  local val_cache="$4"
  local train_ctrl="$5"
  local val_ctrl="$6"
  local out_dir="$7"
  local epochs="$8"
  local lr="$9"
  local batch_size="${10}"
  local max_val="${11}"
  local init_ckpt="${12}"
  shift 12
  local loss_args=("$@")
  if [[ -f "$out_dir/best_loss.pt" ]]; then
    echo "skip existing $variant $stage: $out_dir/best_loss.pt"
    return
  fi
  mkdir -p "$out_dir"
  local cmd=(
    "$PY" pl_curve_train.py
    --train-cache "$train_cache"
    --val-cache "$val_cache"
    --train-gt-control-cache "$train_ctrl"
    --val-gt-control-cache "$val_ctrl"
    --output-dir "$out_dir"
    --experiment-name "newpl_${variant}_${stage}"
    --epochs "$epochs"
    --lr "$lr"
    --batch-size "$batch_size"
    --max-val-sequences "$max_val"
    --early-stop-patience 12
    "${COMMON_TRAIN_ARGS[@]}"
    "${loss_args[@]}"
  )
  if [[ -n "$init_ckpt" ]]; then
    cmd+=(--init-checkpoint "$init_ckpt")
  fi
  "${cmd[@]}"
}

eval_variant_set() {
  local variant="$1"
  local after="$2"
  local output_prefix="$3"
  local amass_best="$ROOT/$variant/amass_pretrain/best_loss.pt"
  local amass_last="$ROOT/$variant/amass_pretrain/last.pt"
  local dip_best="$ROOT/$variant/dip_finetune/best_loss.pt"
  local dip_last="$ROOT/$variant/dip_finetune/last.pt"
  local versions=(
    --version official_PL=official
    --version newpl_v4_init36="$NEWPL_V4"
    --version "${variant}_amass_best=$amass_best"
    --version "${variant}_amass_last=$amass_last"
  )
  if [[ "$after" == "dip" ]]; then
    versions+=(
      --version "${variant}_dip_best=$dip_best"
      --version "${variant}_dip_last=$dip_last"
    )
  fi
  "$PY" newpl_root_eval.py \
    --cache "$DIP_TEST_RAW" \
    --output-json "$EVAL_ROOT/${output_prefix}_dip_test.json" \
    --dataset dip \
    --dataset-label DIP-IMU-test \
    --imu-input-mode official \
    "${versions[@]}"
  "$PY" newpl_root_eval.py \
    --cache "$TC_TEST_RAW" \
    --output-json "$EVAL_ROOT/${output_prefix}_tc_test.json" \
    --dataset totalcapture \
    --dataset-label TotalCapture-test-official-input \
    --imu-input-mode official \
    "${versions[@]}"
}

run_variant() {
  local variant="$1"
  shift
  local loss_args=("$@")
  train_stage "$variant" "smoke" "$DIP_TRAIN_PL_CACHE" "$DIP_VAL_PL_CACHE" "$DIP_TRAIN_CTRL" "$DIP_VAL_CTRL" \
    "$ROOT/$variant/smoke" 1 1e-5 16 2 "" "${loss_args[@]}"
  train_stage "$variant" "amass_pretrain" "$AMASS_PL_CACHE" "$AMASS_PL_CACHE" "$AMASS_CTRL" "$AMASS_CTRL" \
    "$ROOT/$variant/amass_pretrain" 80 1e-4 256 20 "" "${loss_args[@]}"
  eval_variant_set "$variant" "amass" "${variant}_after_amass"
  train_stage "$variant" "dip_finetune" "$DIP_TRAIN_PL_CACHE" "$DIP_VAL_PL_CACHE" "$DIP_TRAIN_CTRL" "$DIP_VAL_CTRL" \
    "$ROOT/$variant/dip_finetune" 40 5e-6 16 6 "$ROOT/$variant/amass_pretrain/best_loss.pt" "${loss_args[@]}"
  eval_variant_set "$variant" "dip" "${variant}_after_dip"
}

run_variant diff_qdot_qddot "${DIFF_LOSS_ARGS[@]}"
run_variant canonical_control "${CONTROL_LOSS_ARGS[@]}"

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
eval_root = root / 'eval'
summary = {
    'status': 'ok',
    'root': str(root),
    'question': 'Compare NewPL training with explicit finite-difference derivative losses versus canonical derivative-aware GT control-point losses.',
    'model_contract': 'PL input 84D official features, init36, output pRB[15]+gR1[3].',
    'changed_loss_only': {
        'diff_qdot_qddot': 'gt_control weights zero; pRB_dot/pRB_ddot/gR1_dot/gR1_ddot enabled.',
        'canonical_control': 'pRB_dot/pRB_ddot/gR1_dot/gR1_ddot zero; gt_control_pRB/gt_control_gR1 enabled with canonical GTControlCache targets.',
    },
    'full_pipeline_11_metrics': 'not measured',
    'jsons': {},
}
for path in sorted(eval_root.glob('*.json')):
    data = json.loads(path.read_text())
    summary['jsons'][path.stem] = str(path)
    summary[path.stem] = {
        'pl_output_comparison_table': data.get('pl_output_comparison_table', []),
        'per_leaf_table': data.get('per_leaf_table', []),
    }
out = root / 'summary.json'
out.write_text(json.dumps(summary, indent=2) + '\n')
md = root / 'summary.md'
lines = [
    '# NewPL Diff-vs-Control Ablation Summary',
    '',
    summary['question'],
    '',
    'Full-pipeline 11 metrics: not measured.',
    '',
]
for key in sorted(k for k in summary if k.endswith('_dip_test') or k.endswith('_tc_test')):
    lines.append(f'## {key}')
    lines.append('')
    lines.append('| Dataset | Version | pRB L1 cm | pRB L2 cm | gR1 angle deg | Notes |')
    lines.append('|---|---|---:|---:|---:|---|')
    for row in summary[key]['pl_output_comparison_table']:
        lines.append(
            f"| {row['Dataset']} | {row['Version']} | {row['pRB L1 cm ↓']} | {row['pRB L2 cm ↓']} | {row['gR1 angle deg ↓']} | {row['Notes']} |"
        )
    lines.append('')
md.write_text('\n'.join(lines) + '\n')
print(json.dumps({'summary': str(out), 'markdown': str(md)}, indent=2))
PY
