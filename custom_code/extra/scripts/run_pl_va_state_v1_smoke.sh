#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-data/experiments/pl_va_state_v1_lag2_ema03_20260712}
ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}
PY="$ENV/bin/python"
export PYTHONPATH="$PWD/custom_code/modified_official:$PWD/custom_code/extra:$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
AMASS=data/dataset_work/L4Cache/prephysics_pose_velocity_amass_k2_paired_offset_overlay/baseline_cache_manifest.json
DIP=data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json
TC=data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_test_official_neural_only/baseline_cache_manifest.json
mkdir -p "$ROOT/smoke" "$ROOT/logs"
"$PY" -m py_compile custom_code/extra/pl_va_state*.py custom_code/modified_official/net.py
"$PY" - <<'PY'
import tempfile
from pathlib import Path
import tests.test_pl_va_state as t
for name in sorted(x for x in dir(t) if x.startswith('test_')):
    fn=getattr(t,name); fn(Path(tempfile.mkdtemp())) if name=='test_sequence_step_chunk_and_initialization' else fn()
print('unit_tests=ok')
PY
"$PY" custom_code/extra/pl_va_state_frame_audit.py --dataset AMASS "$AMASS" --dataset DIP "$DIP" --dataset TotalCapture "$TC" --max-sequences 3 --output "$ROOT/smoke/frame_audit.json"
"$PY" custom_code/extra/pl_va_state_cache.py --input-cache "$AMASS" --output-dir "$ROOT/smoke/cache" --max-sequences 2 --max-frames 240 --shard-size 2
"$PY" custom_code/extra/pl_va_state_train.py --train-cache "$ROOT/smoke/cache/manifest.json" --val-cache "$ROOT/smoke/cache/manifest.json" --output-dir "$ROOT/smoke/train" --epochs 1 --batch-size 2 --max-train-sequences 2 --max-val-sequences 2
"$PY" custom_code/extra/pl_va_state_eval.py --cache "$ROOT/smoke/cache/manifest.json" --checkpoint "$ROOT/smoke/train/best.pt" --output-dir "$ROOT/smoke/eval" --dataset-label AMASS-smoke --max-sequences 2
"$PY" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);train=json.loads((r/'smoke/train/summary.json').read_text());metrics=json.loads((r/'smoke/eval/metrics.json').read_text())
summary={'status':'ok','unit_tests':True,'cache_sequences':2,'max_frames':240,'finite':train['finite'],
         'gradient_norm':train['gradient_norm'],'module_summary':metrics['summary'],'frame_audit':str(r/'smoke/frame_audit.json')}
assert summary['finite'] and all(v>0 for v in summary['gradient_norm'].values())
(r/'smoke/summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
PY
