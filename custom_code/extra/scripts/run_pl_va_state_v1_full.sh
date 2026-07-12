#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-data/experiments/pl_va_state_v1_lag2_ema03_20260712}
ENV=${ENV:-/home/lingfeng/.conda/envs/globalpose-gpu}; PY="$ENV/bin/python"
export PYTHONPATH="$PWD/custom_code/modified_official:$PWD/custom_code/extra:$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
AMASS=data/dataset_work/L4Cache/pl_va_amass_fk_rmb_w_consistent_v1_20260712/baseline_cache_manifest.json
DIP_TRAIN=data/dataset_work/L4Cache/prephysics_pose_velocity_dip_train_globalpose_neural_only/baseline_cache_manifest.json
DIP_VAL=data/dataset_work/L4Cache/prephysics_pose_velocity_dip_val_globalpose_neural_only/baseline_cache_manifest.json
DIP_TEST=data/experiments/dip_official_protocol_check_20260607/dip_test_prephysics_neural_only/baseline_cache_manifest.json
TC_TEST=data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_test_official_neural_only/baseline_cache_manifest.json
BATCH=${BATCH:-8}; mkdir -p "$ROOT" "$ROOT/logs"
"$PY" - "$ROOT" "$BATCH" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); cfg={"task":"PL-VA-State-V1","fps":60,"dt":1/60,"beta":0.7,"cutoff_hz":4.0,"filter_order":2,
 "angular_velocity_method":"causal_world_so3_backward_lag2_ema03","angular_velocity_frame":"world_then_root","angular_velocity_lag":2,"angular_velocity_ema_beta":0.3,
 "input_size":102,"raw_output_size":33,"legacy_output_size":18,"amass_epochs":80,"dip_epochs":40,"batch_size":int(sys.argv[2]),
 "protocol":"AMASS pretrain -> DIP train/val fine-tune -> DIP test + TotalCapture test; no TC fine-tune","fullpipeline_gate":"candidate pRB L2 <= official PL on both DIP and TotalCapture"}
(r/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
PY
build(){ test -f "$3/manifest.json" || "$PY" custom_code/extra/pl_va_state_cache.py --input-cache "$2" --output-dir "$3" --shard-size 50; }
build amass "$AMASS" "$ROOT/caches/amass"; build dip_train "$DIP_TRAIN" "$ROOT/caches/dip_train"; build dip_val "$DIP_VAL" "$ROOT/caches/dip_val"
build dip_test "$DIP_TEST" "$ROOT/caches/dip_test"; build tc_test "$TC_TEST" "$ROOT/caches/tc_test"
if test -f "$ROOT/amass_pretrain/last.pt"; then
  "$PY" custom_code/extra/pl_va_state_train.py --train-cache "$ROOT/caches/amass/manifest.json" --val-cache "$ROOT/caches/amass/manifest.json" --output-dir "$ROOT/amass_pretrain" --epochs 80 --batch-size "$BATCH" --max-val-sequences 20 --lr 1e-4 --resume-checkpoint "$ROOT/amass_pretrain/last.pt"
else
  "$PY" custom_code/extra/pl_va_state_train.py --train-cache "$ROOT/caches/amass/manifest.json" --val-cache "$ROOT/caches/amass/manifest.json" --output-dir "$ROOT/amass_pretrain" --epochs 80 --batch-size "$BATCH" --max-val-sequences 20 --lr 1e-4
fi
test -f "$ROOT/dip_finetune/best.pt" || "$PY" custom_code/extra/pl_va_state_train.py --train-cache "$ROOT/caches/dip_train/manifest.json" --val-cache "$ROOT/caches/dip_val/manifest.json" --output-dir "$ROOT/dip_finetune" --epochs 40 --batch-size "$BATCH" --lr 5e-6 --init-checkpoint "$ROOT/amass_pretrain/best.pt"
"$PY" custom_code/extra/pl_va_state_eval.py --cache "$ROOT/caches/dip_test/manifest.json" --checkpoint "$ROOT/dip_finetune/best.pt" --output-dir "$ROOT/eval/dip_test" --dataset-label DIP-test
"$PY" custom_code/extra/pl_va_state_eval.py --cache "$ROOT/caches/tc_test/manifest.json" --checkpoint "$ROOT/dip_finetune/best.pt" --output-dir "$ROOT/eval/totalcapture_test" --dataset-label TotalCapture-test
"$PY" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);result={}
for name in ('dip_test','totalcapture_test'):
 s=json.loads((r/'eval'/name/'metrics.json').read_text())['summary']; b=next(x for x in s if x['version']=='official_PL');v=next(x for x in s if x['version']=='pl_va_state_v1')
 result[name]={'official_p_l2_cm':b['p_cm_l2'],'candidate_p_l2_cm':v['p_cm_l2'],'pass':v['p_cm_l2']<=b['p_cm_l2'],
               'candidate_v_state_l2_cm_s':v['v_state_cm_s_l2'],'candidate_a_l2_cm_s2':v['a_cm_s2_l2']}
result['module_gate_pass']=all(x['pass'] for x in result.values());(r/'module_gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
PY
if "$PY" - "$ROOT/module_gate.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['module_gate_pass'] else 1)
PY
then
  "$PY" custom_code/extra/pl_va_state_fullpipeline_eval.py --dip-cache "$DIP_TEST" --tc-cache "$TC_TEST" --checkpoint "$ROOT/dip_finetune/best.pt" --output-dir "$ROOT/fullpipeline"
else
  echo "module gate failed; full pipeline intentionally skipped"
fi
