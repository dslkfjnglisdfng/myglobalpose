"""Module evaluation and drift reports for PL-VA-State-V1."""

import argparse
import csv
import json
from pathlib import Path

import torch
import articulate as art

from custom_code.modified_official.net import GPNet
from pl_va_state import (ANGULAR_VELOCITY_EMA_BETA, ANGULAR_VELOCITY_FRAME,
                         ANGULAR_VELOCITY_LAG, ANGULAR_VELOCITY_METHOD,
                         LEAF_NAMES, PLVAStateV1, centered_derivative_targets)
from pl_va_state_train import load_records


def errors(pred, target, scale=1.0):
    d = (pred - target).reshape(pred.shape[0], -1, 3).norm(dim=-1) * scale
    return {"l1": float((pred-target).abs().mean() * scale), "l2": float(d.mean()), "rmse": float(torch.sqrt((d*d).mean()))}


def leaf_errors(pred, target, scale):
    return {name: errors(pred[:, i*3:(i+1)*3], target[:, i*3:(i+1)*3], scale) for i,name in enumerate(LEAF_NAMES)}


def gravity_angle(pred, target):
    dot = (art.math.normalize_tensor(pred, avoid_nan=True) * art.math.normalize_tensor(target, avoid_nan=True)).sum(-1).clamp(-1,1)
    return float(torch.rad2deg(torch.acos(dot)).mean())


def drift(pred, target, fps=60):
    out = {}
    for seconds in (1,2,5,10):
        idx = min(len(pred)-1, seconds*fps-1); value = (pred[idx]-target[idx]).reshape(5,3).norm(dim=-1)*100
        out[f"{seconds}s"] = {"mean": float(value.mean()), "median": float(value.median()), "p95": float(torch.quantile(value,.95)), "max": float(value.max())}
    value=(pred[-1]-target[-1]).reshape(5,3).norm(dim=-1)*100
    out["end"]={"mean":float(value.mean()),"median":float(value.median()),"p95":float(torch.quantile(value,.95)),"max":float(value.max())}
    return out


@torch.no_grad()
def official_prediction(gpnet, record, device):
    if "legacy_feature" in record:
        features = record["legacy_feature"].to(device)
    else:
        from pl_curve import pl_input_feature
        features = torch.stack([pl_input_feature(a,w,r) for a,w,r in zip(record["aM"],record["wM"],record["RMB"])]).to(device)
    return gpnet.plnet([(features, record["init_legacy"].to(device))])[0].cpu()


@torch.no_grad()
def evaluate(cache, checkpoint, output_dir, dataset_label, max_sequences=0):
    rows, manifest = load_records(cache, max_sequences); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False); model=PLVAStateV1().to(device).eval(); model.load_state_dict(ckpt["model"])
    gp=GPNet().to(device).eval(); sequence_rows=[]
    for r in rows:
        out=model.forward_sequence(r["feature"].to(device),r["init_legacy"].to(device)); pl=out["pl"].cpu()
        base=official_prediction(gp,r,device); bv,ba,_=centered_derivative_targets(base[:,:15]);
        for version,pred,v_direct,v_state,a_pred in (
            ("official_PL",base,bv,bv,ba),
            ("pl_va_state_v1",pl,out["vRB_direct"].cpu(),out["vRB_state"].cpu(),out["aRB_leaf"].cpu())):
            metrics={"dataset":dataset_label,"sequence":r["name"],"version":version,
                     "p_cm":errors(pred[:,:15],r["p_gt"],100),"v_cm_s":errors(v_direct,r["v_gt"],100),
                     "v_state_cm_s":errors(v_state,r["v_gt"],100),"a_cm_s2":errors(a_pred,r["a_gt"],100),
                     "g_angle_deg":gravity_angle(pred[:,15:18],r["g_gt"]),
                     "per_leaf_p_cm":leaf_errors(pred[:,:15],r["p_gt"],100),
                     "per_leaf_v_cm_s":leaf_errors(v_state,r["v_gt"],100),"per_leaf_a_cm_s2":leaf_errors(a_pred,r["a_gt"],100),
                     "drift_cm":drift(pred[:,:15],r["p_gt"]),
                     "dynamic_consistency":float(((v_state[1:]-v_state[:-1])-.5/60*(a_pred[:-1]+a_pred[1:])).norm(dim=-1).mean()),
                     "acceleration_jerk":float((a_pred[1:]-a_pred[:-1]).norm(dim=-1).mean()),
                     "beta":.7,"direct_contribution":.7,"acc_integrated_contribution":.3,"finite":bool(torch.isfinite(pred).all())}
            sequence_rows.append(metrics)
    summary=[]
    for version in ("official_PL","pl_va_state_v1"):
        selected=[x for x in sequence_rows if x["version"]==version]
        row={"dataset":dataset_label,"version":version,"num_sequences":len(selected)}
        for group in ("p_cm","v_cm_s","v_state_cm_s","a_cm_s2"):
            for key in ("l1","l2","rmse"): row[f"{group}_{key}"]=sum(x[group][key] for x in selected)/len(selected)
        row["g_angle_deg"]=sum(x["g_angle_deg"] for x in selected)/len(selected); summary.append(row)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"metrics.json").write_text(json.dumps({"cache":str(cache),"checkpoint":str(checkpoint),"manifest":manifest,
        "angular_velocity_method": ANGULAR_VELOCITY_METHOD, "angular_velocity_frame": ANGULAR_VELOCITY_FRAME,
        "angular_velocity_lag": ANGULAR_VELOCITY_LAG, "angular_velocity_ema_beta": ANGULAR_VELOCITY_EMA_BETA,
        "summary":summary,"sequences":sequence_rows},indent=2)+"\n")
    with (output_dir/"comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--dataset-label",required=True);p.add_argument("--max-sequences",type=int,default=0)
    a=p.parse_args();print(json.dumps(evaluate(a.cache,a.checkpoint,a.output_dir,a.dataset_label,a.max_sequences),indent=2))


if __name__=="__main__":main()
