# AccCurve v3 Leaf-Relative Causal Butterworth

## 1. Contract

AccCurve v3 is an AccCurve v1-style residual acceleration module with a leaf-relative causal Butterworth target.

- root index: 5
- leaf indices: 0..4
- root: reference only; excluded from prediction/loss/metric
- input feature: acc_raw[15] + acc_smooth[15] + acc_raw_minus_smooth[15] + wM[18] + RMB_6d[36] = 99D
- base: aIMU_leaf_rel_butter2_4hz[15]
- target: aGT_leaf_rel_butter2_4hz[15]
- output: pred_leaf_rel_acc[15] = [5,3]
- frame: model/world frame M
- smoothing: causal Butterworth order=2 cutoff=4Hz on both IMU base and GT target
- units: m/s^2; feature z-score only; output/target are not normalized

## 2. Relation to Previous Versions

- AccCurve v1: same style of input/network/residual curve, 6-sensor acceleration target, previous smoothing.
- AccCurve v2: strict GTFK q/qdot/qddot/rJS absolute 6-sensor target.
- AccCurve v3: v1-style module, 5 leaf-relative acceleration outputs, causal Butterworth smoothing on both IMU and GT, root used only as reference.

## 3. Training Protocol

- AMASS pretrain: synthetic sanity only.
- DIP finetune: checkpoint selection on DIP val pred/base L2 ratio.
- Final primary eval: DIP test and TotalCapture test.
- No PL/NewPL/full-pipeline/S4 claim.

| Stage | Best epoch | Best validation pred/base ratio | Train seq | Val seq | Train windows | Val windows |
|---|---:|---:|---:|---:|---:|---:|
| AMASS pretrain | 13 | 0.940240 | 1231 | 67 | 8231 | 407 |
| DIP finetune | 16 | 0.851720 | 36 | 6 | 1887 | 253 |

## 4. Main Result Table

| Dataset | Split | Model | Pred L2 | Base L2 | Pred/Base L2 | Pred RMSE | Base RMSE | Pred/Base RMSE | Corr | Base Corr | Cosine | Mag MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP | test | acc_curve_v3 | 0.990334 | 1.196030 | 0.828017 | 0.750235 | 0.924315 | 0.811666 | 0.958276 | 0.943321 | 0.785716 | 0.509845 |
| TotalCapture | test | acc_curve_v3 | 1.365116 | 1.052403 | 1.297142 | 1.065184 | 0.902071 | 1.180820 | 0.923813 | 0.946864 | 0.730314 | 0.710080 |

## 5. Per-Sensor Table

Only leaf sensors are included. Cache order is left_forearm, right_forearm, left_lower_leg, right_lower_leg, head.

| Dataset | Sensor | Pred L2 | Base L2 | Pred/Base L2 | Pred Corr | Base Corr |
|---|---|---:|---:|---:|---:|---:|
| DIP | left_forearm | 1.138353 | 1.406493 | 0.809356 | 0.961722 | 0.950359 |
| DIP | right_forearm | 1.183689 | 1.479083 | 0.800286 | 0.948624 | 0.932929 |
| DIP | left_lower_leg | 0.954022 | 1.108950 | 0.860293 | 0.885224 | 0.883457 |
| DIP | right_lower_leg | 0.971016 | 1.167789 | 0.831499 | 0.885419 | 0.888058 |
| DIP | head | 0.704591 | 0.817837 | 0.861530 | 0.879927 | 0.855027 |
| TotalCapture | left_forearm | 1.473318 | 0.898079 | 1.640522 | 0.954272 | 0.978326 |
| TotalCapture | right_forearm | 1.598774 | 1.201449 | 1.330705 | 0.955925 | 0.976233 |
| TotalCapture | left_lower_leg | 1.397981 | 1.193100 | 1.171721 | 0.876689 | 0.903467 |
| TotalCapture | right_lower_leg | 1.443817 | 1.172867 | 1.231015 | 0.833829 | 0.873434 |
| TotalCapture | head | 0.911688 | 0.796519 | 1.144591 | 0.865854 | 0.913863 |

## 6. Required Judgement

Decision: **fail**.

- Strong pass: DIP and TC both improve over base in L2/RMSE, with corr not decreasing by more than 0.01.
- Pass: DIP improves, TC pred/base L2 ratio <= 1.05, and DIP/TC corr does not decrease by more than 0.01.
- Soft pass: DIP improves, TC is not meaningfully worse, and corr remains close to base.
- Fail: DIP worsens, TC worsens sharply, or corr drops substantially.

## 7. V4 Base Sanity

- DIP v4 all-reference butter L2/RMSE/corr = 0.893481 / 0.914904 / 0.943719.
- TotalCapture v4 all-reference butter L2/RMSE/corr = 1.092481 / 1.050146 / 0.941474.
- Final eval is test split only, so exact base numbers can differ from all-split v4 references.

## 8. Non-Claims

- This is not PL/NewPL training.
- This is not full-pipeline evaluation.
- This does not claim pose improvement.
- AMASS is synthetic and not primary evidence.
- Root channel is not predicted or evaluated.

## Artifacts

- output root: `data/experiments/acc_curve_v3_leafrel_causal_butter_20260618`
- final checkpoint: `data/experiments/acc_curve_v3_leafrel_causal_butter_20260618/dip_finetune/best_loss.pt`
- config: `data/experiments/acc_curve_v3_leafrel_causal_butter_20260618/config.json`
- feature norm: `data/experiments/acc_curve_v3_leafrel_causal_butter_20260618/feature_norm.pt`
- eval JSONs: `data/experiments/acc_curve_v3_leafrel_causal_butter_20260618/eval`

## Commands

```bash
acc_curve_v3_leafrel_train.py --mode train_full --cache-manifest data/dataset_work/AccCurveV3LeafRelCausalButter_20260618/cache_manifest.json --output-dir data/experiments/acc_curve_v3_leafrel_causal_butter_20260618 --epochs 30 --dip-epochs 20 --window 240 --stride 120 --batch-size 64 --num-workers 8 --lr 1e-4 --weight-decay 1e-4 --hidden-size 512 --dropout 0.1 --residual-scale 1.0 --control-prior-weight 1e-5 --grad-clip 1.0 --seed 1234 --overwrite
```
