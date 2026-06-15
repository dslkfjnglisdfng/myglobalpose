#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
PY="${PY:-$ENV/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

OUT_ROOT="${OUT_ROOT:-/tmp/globalpose_hybrid_prb_base_gr1_v6_20260613/full}"
CKPT="${CKPT:-/tmp/globalpose_newpl_v6_gR1_nextonly_smoothacc_20260613/full/dip_finetune/best_current_gR1.pt}"
DIP_CACHE="${DIP_CACHE:-data/experiments/newpl_v5_smoothacc_20260612/caches/raw_dip_test_smooth_w9/baseline_cache_manifest.json}"
TC_CACHE="${TC_CACHE:-data/experiments/newpl_v5_smoothacc_20260612/caches/raw_tc_test_smooth_w9/baseline_cache_manifest.json}"
mkdir -p "$OUT_ROOT"

echo "OUT_ROOT=$OUT_ROOT"
echo "CKPT=$CKPT"
echo "DIP_CACHE=$DIP_CACHE"
echo "TC_CACHE=$TC_CACHE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

"$PY" pl_curve_eval.py \
  --val-cache "$DIP_CACHE" \
  --hybrid-gR1-checkpoint "$CKPT" \
  --imu-input-mode official \
  --force-baseline-rerun \
  --output-json "$OUT_ROOT/dip_test_hybrid_baseline_pRB_newpl_gR1.json"

"$PY" pl_curve_eval.py \
  --val-cache "$TC_CACHE" \
  --hybrid-gR1-checkpoint "$CKPT" \
  --imu-input-mode official \
  --force-baseline-rerun \
  --output-json "$OUT_ROOT/totalcapture_test_hybrid_baseline_pRB_newpl_gR1.json"

"$PY" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metric_names = [
    'L SIP Err (deg)',
    'L Angle Err (deg)',
    'L Joint Err (cm)',
    'L Vertex Err (cm)',
    'G SIP Err (deg)',
    'G Angle Err (deg)',
    'G Joint Err (cm)',
    'G Vertex Err (cm)',
    'Root Jitter (km/s^3)',
    'Joint Jitter (km/s^3)',
]

summary = {
    'status': 'ok',
    'contract': 'full-pipeline 11-metric evaluation; hybrid PL output = official/baseline PL pRB[15] + newpl_v6_gR1nextonly_smoothacc gR1[3]',
    'checkpoint': str(root),
    'datasets': {},
}
for dataset, filename in [
    ('DIP-IMU test smoothacc', 'dip_test_hybrid_baseline_pRB_newpl_gR1.json'),
    ('TotalCapture test smoothacc', 'totalcapture_test_hybrid_baseline_pRB_newpl_gR1.json'),
]:
    data = json.loads((root / filename).read_text())
    agg = data['aggregate']
    row = {
        'json': str(root / filename),
        'score': data.get('score'),
        'num_sequences': agg.get('num_sequences'),
        'baseline': {},
        'hybrid': {},
        'delta_hybrid_minus_baseline': {},
    }
    for name in metric_names:
        row['baseline'][name] = agg['baseline_metrics'][name]['mean']
        row['hybrid'][name] = agg['model_metrics'][name]['mean']
        row['delta_hybrid_minus_baseline'][name] = agg['delta_metrics'][name]['mean']
    summary['datasets'][dataset] = row

(root / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({
    'status': 'ok',
    'summary': str(root / 'summary.json'),
    'datasets': {
        k: {
            'score': v['score'],
            'num_sequences': v['num_sequences'],
            'G Angle delta': v['delta_hybrid_minus_baseline']['G Angle Err (deg)'],
            'G Joint delta': v['delta_hybrid_minus_baseline']['G Joint Err (cm)'],
            'L Angle delta': v['delta_hybrid_minus_baseline']['L Angle Err (deg)'],
        }
        for k, v in summary['datasets'].items()
    }
}, indent=2))
PY
