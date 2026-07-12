"""Audit RMB-differenced angular velocity against measured legacy wRB."""

import argparse, json
from pathlib import Path
import torch
from l4_train_diverse_short import load_cache_files
from pl_va_state import LEAF_NAMES, body_omega_to_root_frame, causal_angular_velocity_from_rmb_sequence

SENSORS=LEAF_NAMES+("pelvis_root",)

def stats(x,y):
    d=x-y; xf=x.flatten();yf=y.flatten();xc=xf-xf.mean();yc=yf-yf.mean()
    return {"rmse":float(torch.sqrt((d*d).mean())),"l2":float(d.norm(dim=-1).mean()),
            "correlation":float((xc*yc).sum()/(xc.norm()*yc.norm()).clamp_min(1e-12)),
            "cosine":float(torch.nn.functional.cosine_similarity(x,y,dim=-1).mean())}

def audit(manifest_path,max_sequences=4):
    files,_=load_cache_files(manifest_path);diffs=[];measured=[];names=[]
    for file in files:
        data=torch.load(file,map_location="cpu")
        for i,name in enumerate(data["name"]):
            r=data["RMB"][i].float(); w=data["wM"][i].float(); root=r[:,-1]
            wb=causal_angular_velocity_from_rmb_sequence(r); wd=body_omega_to_root_frame(wb,r,root[:,None].expand_as(r))
            wm=w.matmul(root);diffs.append(wd);measured.append(wm);names.append(str(name))
            if len(names)>=max_sequences:break
        if len(names)>=max_sequences:break
    x=torch.cat(diffs);y=torch.cat(measured)
    return {"manifest":str(manifest_path),"sequences":names,"current_sign":"omega=-Log(R_t^T R_t-1)/dt",
            "relative_rotation_order":"RMB_t^T @ RMB_t-1","equivalent":"Log(RMB_t-1^T @ RMB_t)/dt",
            "first_frame":"strict zero","overall":stats(x,y),"per_sensor":{n:stats(x[:,i],y[:,i]) for i,n in enumerate(SENSORS)}}

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",action="append",nargs=2,metavar=("LABEL","MANIFEST"),required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--max-sequences",type=int,default=4);a=p.parse_args()
    result={label:audit(path,a.max_sequences) for label,path in a.dataset};a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
