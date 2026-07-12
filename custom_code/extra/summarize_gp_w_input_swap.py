#!/usr/bin/env python3
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path("data/experiments/gp_w_input_swap_lag2_ema03_20260712")
VARIANTS = ("G0_official", "G1_pl_swap", "G2_vr_swap", "G3_all_swap")
POSE = ("motion.local_sip_deg", "motion.local_angle_deg", "motion.local_joint_cm", "motion.local_mesh_cm", "motion.global_sip_deg", "motion.global_angle_deg", "motion.global_joint_cm", "motion.global_mesh_cm")
CORE = POSE + ("motion.root_jitter_km_s3", "motion.joint_jitter_km_s3")
TC_ROOT = ("root.root_translation_rmse_m", "root.root_translation_first_frame_aligned_rmse_m", "root.root_trajectory_drift_m", "root.root_velocity_rmse_m_s", "root.max_frame_root_step_m", "root.contact_velocity_mean_m_s", "root.foot_slip_mean_m_s")
MODULE = ("module.PL.pRB_l1_cm", "module.PL.pRB_l2_cm", "module.PL.pRB_rmse_cm", "module.PL.gR1_angle_deg", "module.IK1.pRJ_l1_cm", "module.IK1.pRJ_l2_cm", "module.IK1.pRJ_rmse_cm", "module.IK1.gR2_angle_deg", "module.IK1.pRJ_first_difference_l2_cm_per_frame", "module.IK1.pRJ_second_difference_l2_cm_per_frame2")


def load_rows(ds):
    rows = []
    for seq in sorted(x for x in (ROOT / ds).iterdir() if x.is_dir()):
        rows.extend(json.loads(p.read_text()) for p in sorted(seq.glob("G*.json")))
    return rows


def flat(row):
    from evaluate_gp_w_input_swap import flatten_means
    out = {}; flatten_means("", {"motion": row["motion"], "root": row["root"], "module": row["module"]}, out); return out


def wins(rows, metrics):
    by_seq = {}
    for row in rows: by_seq.setdefault(row["sequence"], {})[row["variant"]] = flat(row)
    out = {v: {m: 0 for m in metrics} for v in VARIANTS[1:]}
    for variants in by_seq.values():
        for v in VARIANTS[1:]:
            for m in metrics:
                if variants[v].get(m) is not None and variants[v][m] < variants["G0_official"][m]: out[v][m] += 1
    return out


def fmt(x): return "n/a" if x is None else f"{x:.6f}"


def table(agg, metrics):
    lines = ["| variant | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
    for v in VARIANTS: lines.append("| " + v + " | " + " | ".join(fmt(agg[v].get(m)) for m in metrics) + " |")
    return "\n".join(lines)


def main():
    dip, tc = load_rows("dip_test"), load_rows("totalcapture_test")
    dip_agg = json.loads((ROOT / "dip_test/aggregate.json").read_text()); tc_agg = json.loads((ROOT / "totalcapture_test/aggregate.json").read_text())
    dip_wins, tc_wins = wins(dip, CORE), wins(tc, CORE + TC_ROOT)
    with (ROOT / "comparison.csv").open("w", newline="") as out:
        writer = None
        for path in (ROOT / "dip_test/comparison.csv", ROOT / "totalcapture_test/comparison.csv"):
            for row in csv.DictReader(path.open()):
                if writer is None: writer = csv.DictWriter(out, fieldnames=row.keys()); writer.writeheader()
                writer.writerow(row)
    weight_hash = hashlib.sha256(Path("data/weights.pt").read_bytes()).hexdigest()
    config = {
        "experiment": "gp_w_input_swap_lag2_ema03_20260712", "baseline_commit": "90523d6f38c28ee3a1afd27346cd3624c5efe38a",
        "weights": "data/weights.pt", "weights_sha256": weight_hash, "fps": 60, "dt": 1/60, "lag": 2, "ema_beta": 0.3,
        "RMB": "R_M_B body-to-model/world", "delta_R": "RMB[t] @ RMB[t-2].transpose(-1,-2)", "output_w_frame": "model/world M",
        "variants": {"G0_official": ["cached", "cached"], "G1_pl_swap": ["causal_RMB", "cached"], "G2_vr_swap": ["cached", "causal_RMB"], "G3_all_swap": ["causal_RMB", "causal_RMB"]},
        "dip_cache": "data/dataset_work/L4Cache/prephysics_pose_velocity_diptest_official_neural_only_offset_r/baseline_cache_manifest.json",
        "totalcapture_cache": "data/dataset_work/L4Cache/prephysics_pose_velocity_totalcapture_test_official_neural_only/baseline_cache_manifest.json",
        "dip_sequences": 19, "totalcapture_sequences": 4,
        "contact_metric": "GT foot velocity <0.15m/s and GT foot height < floor+0.08m; report predicted foot speed and horizontal slip",
        "fk_audit": "analytic RBDL space Jacobian with centered n=8 qdot reference only; swapped w remains causal",
    }
    (ROOT / "config.json").write_text(json.dumps(config, indent=2))
    audit = json.loads((ROOT / "input_signal_audit.json").read_text())["overall"]
    summary = f"""# GP test-time angular velocity swap

## 结论

- **G1 PL-only：不支持总体姿态改善。** DIP 的 local SIP 略好，但 angle/joint 与 PL/IK1 直接误差整体变差；TotalCapture 的全部主要 local pose 指标明显变差。TotalCapture root translation、root jitter、joint jitter 和 foot slip 改善，说明 PL swap 的影响不是单向退化。
- **G2 VR-only：姿态近似不变，TotalCapture root translation/foot slip 改善，但 root velocity RMSE、root jitter 与 max step 变差。** 这是最清楚的“root translation 改善但 jitter 恶化”情形。
- **G3 all-swap：不支持总体姿态改善，但 TotalCapture global angle 略好，root translation、drift、contact velocity、foot slip 和 joint jitter 明显改善。** local SIP/angle/joint/mesh 与 PL/IK1 模块误差变差。
- 改善并非在多数 pose 序列/指标上一致成立；不同模块与数据集存在明显 trade-off。
- 存在 **PL 直接输出变差、VR/root trajectory 变好**，而不是“PL 改善但 VR 恶化”。DIP 上 PL swap 的 local SIP 略好但模块 pRB/IK1 误差变差；TotalCapture 上 PL swap 的姿态变差而 root/jitter 变好。
- 输入审计复现此前方向：causal RMB w 比 cached wM 更接近解析 FK w；但 GP 输出没有获得一致 pose 改善。因此这轮主要证明了 **FK consistency 改善不等价于 zero-shot GP pose 改善**。

明确边界：**test-time swap 变差，不足以证明新 w 无效**。官方 GP 是在 cached measured wM 输入分布上训练的；若要判断新 w 的上限，下一步应进行 matched-input retraining，再用同协议评估。

## DIP test aggregate（19 sequences）

{table(dip_agg, CORE)}

DIP translation GT 不作为可信主结论；root/translation 辅助值保留在逐序列 JSON 和 comparison.csv 中，但不用于判断。

## TotalCapture test aggregate（4 official sequences）

{table(tc_agg, CORE)}

### TotalCapture root/contact

{table(tc_agg, TC_ROOT)}

## PL / IK1 module aggregate

### DIP

{table(dip_agg, MODULE)}

### TotalCapture

{table(tc_agg, MODULE)}

逐 leaf/sensor 统计位于每条序列的 `G*.json`，聚合字段位于各 dataset 的 `aggregate.json` 与 `comparison.csv`。

## Per-sequence win counts

计数规则：指标越低越好，严格小于同序列 G0 记为 win。

```json
{json.dumps({'dip': dip_wins, 'totalcapture': tc_wins}, indent=2)}
```

## Input w / FK audit（TotalCapture）

| comparison | RMSE | Pearson | mean L2 | cosine |
|---|---:|---:|---:|---:|
| cached wM vs FK w | {audit['cached_wM_vs_FK_w']['rmse']:.6f} | {audit['cached_wM_vs_FK_w']['pearson']:.6f} | {audit['cached_wM_vs_FK_w']['mean_l2']:.6f} | {audit['cached_wM_vs_FK_w']['cosine']:.6f} |
| causal RMB w vs FK w | {audit['causal_RMB_w_vs_FK_w']['rmse']:.6f} | {audit['causal_RMB_w_vs_FK_w']['pearson']:.6f} | {audit['causal_RMB_w_vs_FK_w']['mean_l2']:.6f} | {audit['causal_RMB_w_vs_FK_w']['cosine']:.6f} |
| cached wM vs causal RMB w | {audit['cached_wM_vs_causal_RMB_w']['rmse']:.6f} | {audit['cached_wM_vs_causal_RMB_w']['pearson']:.6f} | {audit['cached_wM_vs_causal_RMB_w']['mean_l2']:.6f} | {audit['cached_wM_vs_causal_RMB_w']['cosine']:.6f} |

## Protocol

- Baseline code: `90523d6f38c28ee3a1afd27346cd3624c5efe38a`; official `data/weights.pt` unchanged.
- No training, no acceleration/RMB/IK1/IK2/VR/physics/contact logic change.
- Every sequence resets GP and RMB-derived-w streaming states; inference is chronological with no shuffle.
- Frames 0 and 1 use zero replacement w; frame 2 initializes EMA from raw lag-2 SO(3) velocity; later frames use `0.7 prev + 0.3 raw`.
- Official PL row-vector conversion remains `w_pl_M @ RMB_root`; VR remains `w_vr_M @ predicted_pose_root`.
- Smoke failures/retries and the abandoned unchunked evaluator attempt are preserved under `logs/` and `dip_test_incomplete_unbounded_metric/`; no result from them enters the final tables.
"""
    (ROOT / "SUMMARY.md").write_text(summary)
    (ROOT / "win_counts.json").write_text(json.dumps({"dip": dip_wins, "totalcapture": tc_wins}, indent=2))


if __name__ == "__main__": main()
