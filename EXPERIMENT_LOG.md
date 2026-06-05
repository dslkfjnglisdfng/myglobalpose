# Experiment Log

Detailed archive for GlobalPose replacement experiments. Do not read this file by default when designing new experiments; use `PROJECT_STATUS.md` or `RECENT_REPLACEMENT_VERSIONS.md` to find the exact section needed.

## EXP-20260604-001 — Processed IMU official GPNet S4 baseline

Question: If frozen official GlobalPose consumes processed TotalCapture IMU orientation directly, does S4 improve versus official `aM/wM/RMB`?

Hypothesis: Corrected orientation `l4_RMB` improves neural features even when `aM/wM` are unchanged.

Change tested: Replace official `RMB` with processed orientation stream while keeping frozen official weights and official evaluation path.

Dataset/split: TotalCapture S4 validation, 5 sequences, 17223 frames.

Command/artifacts:

```text
data/experiments/official_gpnet_processed_imu_v1/s4_official_input_baseline.json
data/experiments/official_gpnet_processed_imu_v1/s4_processed_A_input.json
data/experiments/official_gpnet_processed_imu_v1/s4_comparison.json
data/experiments/official_gpnet_processed_imu_v1/s4_comparison.csv
```

Baseline result: official input score `42.522402`.

New result: processed-A input score `38.753660`.

Metrics: processed input improved all 11 reported S4 aggregate metrics.

Interpretation: The gain comes from corrected orientation, because processed data has `l4_aM == aM`, `l4_wM == wM`, and `l4_RMB != RMB`.

Claim support: validation result.

Problems: None recorded for this comparison.

Next action: Use processed orientation as the input convention for replacement experiments.

## EXP-20260604-002 — Consistent processed IMU v2 audit

Question: After correcting IMU orientation, should stored acceleration and gyro also change?

Hypothesis: Under the GlobalPose stored-field convention, correcting sensor-to-body orientation changes `RMB` but not stored `aM/wM`.

Change tested: Generate/audit a consistent processed IMU v2 where `l4_RMB = RIM^T @ RIS @ RSB_new` and `l4_aM/l4_wM` remain equal to official `aM/wM`.

Dataset/split: TotalCapture train+val cache generation; 41 sequences, 160125 frames.

Command/artifacts:

```text
l4_generate_consistent_processed_imu_v2.py
data/dataset_work/L4Cache/totalcapture_orientation_offset_consistent_v2/train_Roffset_A_consistent/baseline_cache_manifest.json
data/dataset_work/L4Cache/totalcapture_orientation_offset_consistent_v2/val_Roffset_A_consistent/baseline_cache_manifest.json
data/experiments/consistent_processed_imu_v2/imu_consistency_audit.json
data/experiments/consistent_processed_imu_v2/s4_eval/s4_official_input.json
data/experiments/consistent_processed_imu_v2/s4_eval/s4_processed_v1_rmb_only.json
data/experiments/consistent_processed_imu_v2/s4_eval/s4_processed_v2_consistent.json
```

Baseline formulas:

```text
RMB = RIM^T @ RIS @ RSB
aM  = RIM^T @ RIS @ aS + [0, -9.8, 0]
wM  = RIM^T @ RIS @ wS
```

Final processed v2 formula:

```text
RSB_new = R_JS^T
l4_RMB = RIM^T @ RIS @ RSB_new
l4_aM  = RIM^T @ RIS @ aS + [0, -9.8, 0] = aM
l4_wM  = RIM^T @ RIS @ wS = wM
```

Audit result:

```text
status = ok
num_sequences = 41
num_frames = 160125
official_vs_v2 aM norm mean = 0.0
official_vs_v2 wM norm mean = 0.0
official_vs_v2 RMB geodesic mean = 10.153708 deg
official_vs_v2 aRB norm mean = 1.404047
official_vs_v2 wRB norm mean = 0.520416
official_vs_v2 RRB geodesic mean = 12.865632 deg
official_vs_v2 gR0 norm mean = 0.110008
v1_vs_v2 aM norm mean = 0.0
v1_vs_v2 wM norm mean = 0.0
v1_vs_v2 RMB geodesic mean = 0.008026 deg
official RMB vs GT mapped joint rotation mean = 11.690968 deg
l4_RMB vs GT mapped joint rotation mean = 5.211172 deg
```

S4 result: v2 is exactly equal to v1 in the 11 aggregate metrics; processed score remains `38.753660`.

Interpretation: v1 was already baseline-consistent at stored-field level. The useful signal is corrected orientation and induced neural-feature changes, not reprocessed acceleration/gyro.

Claim support: validation result.

Problems: Naming should avoid implying full IMU reprocessing; use orientation-only / RMB-only correction.

Next action: Treat processed orientation as the stable input contract for downstream replacement experiments.

## EXP-20260604-003 — NewPL Run D processed-input PL replacement

Question: Can PLCurve-style NewPL with ground-truth control supervision improve processed-input S4 beyond original GPNet + processed input?

Hypothesis: A streaming-compatible PL curve module with GT control-point supervision can improve the PL stage while preserving official downstream contracts.

Change tested: Continue from Run A with `gt_control_pRB=0.3`, `gt_control_gR1=0.1`, `lr=2e-6`, 10 TotalCapture fine-tune epochs.

Dataset/split: TotalCapture processed orientation cache, validation on S4.

Command/artifacts:

```text
data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/best_loss.pt
data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/train_result.json
```

Baseline result: Original GPNet + processed score `38.753660`; earlier Run A score `38.714056`; gR dynamics continuation score `38.730901`.

New result: Run D score `38.694846`.

Metrics: Run D improves Original GPNet + processed, gR dynamics continuation, and Run A on S4 score.

Interpretation: GT control-point supervision is useful for processed-input PL replacement. Run D is the recommended PL1 upstream checkpoint for downstream replacement experiments.

Claim support: validation result.

Problems: `tail_update=4` is a control-point revision window, not a declared four-frame output latency.

Next action: Use Run D as PL1 upstream for NewIK1 / other downstream module replacements.

## EXP-20260604-004 — NewIK1 control-point PL1-output TotalCapture fine-tune

Question: Can NewIK1_ControlPoint_v1 improve the official IK1 replacement when trained on PL1 streaming outputs?

Hypothesis: Training the IK1 replacement on upstream PL1 streaming outputs reduces teacher-forcing mismatch.

Change tested: Fine-tune `NewIK1ControlPointModule` on PL1 streaming TotalCapture cache for 10 epochs from AMASS-adapted checkpoint.

Dataset/split: TotalCapture train/val processed orientation cache with PL1 streaming upstream outputs.

Command/artifacts:

```text
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/train_result.json
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/best_loss.pt
```

Config summary:

```text
epochs = 10
window = 61
lr = 3e-06
batch_size = 8
init_checkpoint = data/experiments/newik1_mainline_20260604/pl1_output_amass_adaptation/best_loss.pt
weights: pRJ=1.0, gR2=1.0, pRJ_dot=0.03, pRJ_ddot=0.001, gR2_dot=0.03, gR2_ddot=0.001, control_pRJ=0.1, control_gR2=0.1, control_point_prior=0.3, tail_update_prior=0.005
```

New result: best epoch `10`, best loss `0.17848628610372544`.

Interpretation: PL1-output fine-tuning converged locally, but this local loss alone does not prove final S4 improvement.

Claim support: bounded diagnostic.

Problems: Needs final full-pipeline evaluation before claiming module improvement.

Next action: Compare variants with bone-length and stronger pRJ/control losses, then evaluate full pipeline.

## EXP-20260604-005 — NewIK1 control-point bone-length continuation

Question: Does adding bone-length consistency improve NewIK1 PL1-output fine-tuning?

Hypothesis: Bone-length consistency stabilizes pRJ geometry and may improve gravity/root-direction behavior.

Change tested: Continue from `pl1_output_tc_finetune/best_loss.pt` with `bone_length=0.5`.

Dataset/split: Same PL1 streaming TotalCapture train/val cache as EXP-20260604-004.

Command/artifacts:

```text
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/train_result.json
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/best_loss.pt
```

Config summary:

```text
epochs = 10
window = 61
lr = 3e-06
batch_size = 8
init_checkpoint = data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune/best_loss.pt
weights include bone_length=0.5
```

Baseline result: prior local best loss `0.17848628610372544`.

New result: best epoch `10`, best loss `0.1783127911388874`.

Interpretation: Bone-length loss slightly improved local validation loss, but still requires full downstream S4 validation.

Claim support: bounded diagnostic.

Problems: Local decoded-state closeness may not track final S4.

Next action: Test whether stronger pRJ/control supervision helps or hurts.

## EXP-20260604-006 — NewIK1 stronger pRJ/control continuation

Question: Does stronger pRJ and control-tail supervision improve NewIK1 beyond the bone-length continuation?

Hypothesis: Stronger pRJ/control supervision might improve Cartesian IK1 state quality.

Change tested: Continue from bone-length checkpoint with `pRJ=2.0`, `control_pRJ=0.3`, `bone_length=0.5`.

Dataset/split: Same PL1 streaming TotalCapture train/val cache as EXP-20260604-004.

Command/artifacts:

```text
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/train_result.json
data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5_pRJ2_controlpRJ0p3/best_loss.pt
```

Config summary:

```text
epochs = 10
window = 61
lr = 3e-06
batch_size = 8
init_checkpoint = data/experiments/newik1_mainline_20260604/pl1_output_tc_finetune_bonelen_w0p5/best_loss.pt
weights: pRJ=2.0, control_pRJ=0.3, bone_length=0.5
```

Baseline result: bone-length local best loss `0.1783127911388874`.

New result: best epoch `10`, best loss `0.17918791025876998`.

Interpretation: Stronger pRJ/control supervision worsened local validation loss relative to the bone-length continuation.

Claim support: bounded diagnostic.

Problems: Do not select this variant based on local loss.

Next action: Prefer the bone-length continuation locally unless full S4 says otherwise.

## EXP-20260604-007 — NewIK1 official-input PL1 streaming full-pipeline evaluation

Question: Can a finetuned official-shape IK1 replacement improve the PL1 streaming full GlobalPose pipeline?

Hypothesis: Training IK1 on official-shaped inputs with PL1 streaming upstream outputs can improve the final pipeline without changing downstream IK2/VR/physics contracts.

Change tested: Full evaluation with PL Run D checkpoint and finetuned official-shape IK1 replacement.

Dataset/split: TotalCapture S4 validation, processed orientation input, 5 sequences, 17223 frames.

Command/artifacts:

```text
scripts/run_newik1_official_input_full.sh
data/experiments/newik1_official_input_20260604/eval_pl1_streaming_tc_val.json
data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt
```

Config summary:

```text
pl_checkpoint = data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/best_loss.pt
ik1_checkpoint = data/experiments/newik1_official_input_20260604/pl1_streaming_tc_finetune/best_loss.pt
pl_backend = curve_v1
ik1_backend = official_input_v1
imu_input_mode = processed
streaming_contract = GPNet.forward_frame with PL curve and a finetuned official-shape IK1 net1 replacement.
```

Baseline result: Original GPNet + processed score `38.753660`; Run D PL-only score `38.694846`.

New result: final score `38.70523069866002`, `status=ok`, `all_finite=True`.

Interpretation: This is better than Original GPNet + processed, but worse than Run D PL-only. The IK1 replacement should not be selected over PL-only Run D as the current mainline.

Claim support: validation result.

Problems: The file names are NewIK1/IK replacement artifacts, not confirmed `newik2` artifacts. Do not relabel them as NewIK2 without evidence.

Next action: If the user’s two good NewIK2 versions refer to other artifacts, locate or request those exact paths before documenting their metrics.

## EXP-20260605-001 — NewPL init36 RunD-style processed-input PL replacement

Question: Can a K2-like 36D PLCurve hidden-state initialization improve processed-input PL replacement beyond historical Run D?

Hypothesis: Initializing the streaming PL hidden state from first-frame IMU attachment offset, official PL initialization position, and root gravity direction reduces the train/runtime mismatch in the previous NewPL path.

Change tested: Use `PLCurveModule(init_size=36)` with `pl_init_feature = offset_r[18] + pRL[15] + gR0[3]`, preserve official PL frame input `aRB[18] + wRB[18] + RRB[45] + gR0[3] = 84D`, preserve official PL output `pRB[15] + gR1[3] = 18D`, and train from the historical Run D checkpoint with partial loading into the expanded init encoder.

Dataset/split: TotalCapture processed orientation cache, train 36 sequences / 142902 frames, S4 validation 5 sequences / 17223 frames. Cache schema is `pl_curve_cache_v2` with per-sequence `pl_init_feature` and `init_layout = offset_r[18] + pRL[15] + gR0[3]`.

Command/artifacts:

```text
data/dataset_work/L4Cache/pl_curve_init36_processed_tc_train_Roffset_A/pl_curve_cache_manifest.json
data/dataset_work/L4Cache/pl_curve_init36_processed_tc_val_Roffset_A/pl_curve_cache_manifest.json
data/experiments/pl_curve_init36_processed_rund_style/best_loss.pt
data/experiments/pl_curve_init36_processed_rund_style/last.pt
data/experiments/pl_curve_init36_processed_rund_style/train_result.json
data/experiments/pl_curve_init36_processed_rund_style/eval_best_final_streaming_processed.json
data/experiments/pl_curve_init36_processed_rund_style/eval_last_final_streaming_processed.json
```

Config summary:

```text
epochs = 60
window = 61
batch_size = 2
lr = 2e-6
init_size = 36
init_checkpoint = data/experiments/pl_curve_v2_processed_no_baseline_gRdyn_gtcontrol_finetune_v1/run_d_0p3_0p1_continue10/tc_finetune_10ep/best_loss.pt
baseline_pRB_weight = 0
baseline_gR1_weight = 0
disable_ik_distill = true
gR1_dot_weight = 0.03
gR1_ddot_weight = 0.001
pRB_ddot_smooth_weight = 1e-6
gt_control_pRB_weight = 0.3
gt_control_gR1_weight = 0.1
early_stop_min_delta = 1e-5
early_stop_patience = 8
```

Baseline result: Original GPNet + processed score `38.753660`; historical NewPL Run D PL-only score `38.69484578047692`.

New result: `best_loss.pt` streaming score `38.625657482802865`; `last.pt` streaming score `38.625657482802865`; `all_finite=True`; training `best_epoch=60`; training `best_loss=0.18789918906986713`; `stopped_early=False`.

Interpretation: The K2-like 36D init improved PL-only processed-input S4 by `0.069188297674055` over historical Run D and by `0.128002517197135` over Original GPNet + processed input. This becomes the current selected PL1 upstream checkpoint.

Claim support: validation result.

Problems: No blocker for NewPL init36. The result still only replaces PL-s1; downstream IK1/IK2/VR interfaces remain unchanged. NewIK2 artifacts remain unresolved and should not be inferred from NewIK1 paths.

Next action: Use NewPL init36 `best_loss.pt` / `last.pt` as the PL1 upstream for downstream replacement experiments, then locate exact NewIK2 artifacts if they exist.
