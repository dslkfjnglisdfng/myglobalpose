"""Run official downstream 11-metric evaluator for original PL and PL-VA."""
import argparse,json,subprocess,sys
from pathlib import Path

def run(cache,label,version,out,checkpoint=None,max_sequences=0):
    cmd=[sys.executable,"custom_code/extra/newik1_real_streaming_audit.py","--val-cache",str(cache),"--output-json",str(out),
         "--split-label",label,"--version-name",version,"--imu-input-mode","official","--ik1-backend","original","--skip-module-metrics"]
    if checkpoint:cmd += ["--pl-va-checkpoint",str(checkpoint)]
    if max_sequences:cmd += ["--max-eval-sequences",str(max_sequences)]
    subprocess.run(cmd,check=True)
    return json.loads(out.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument("--dip-cache",type=Path,required=True);p.add_argument("--tc-cache",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--max-sequences",type=int,default=0)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);summary={"metric_contract":"existing MotionEvaluator 11 metrics","datasets":{}}
    for key,label,cache in (("dip_test","DIP-test",a.dip_cache),("totalcapture_test","TotalCapture-test",a.tc_cache)):
        base=run(cache,label,"official_gpnet",a.output_dir/f"{key}_official.json",max_sequences=a.max_sequences)
        cand=run(cache,label,"pl_va_state_v1",a.output_dir/f"{key}_pl_va.json",a.checkpoint,a.max_sequences)
        summary["datasets"][key]={"official":{"score":base["score"],"aggregate":base["aggregate"]},
                                  "pl_va":{"score":cand["score"],"aggregate":cand["aggregate"]},"all_finite":cand["all_finite"]}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
