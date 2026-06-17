# Acc Invariance Datacache v2 Rebuild 20260618

Experiment: `acc_invariance_datacache_v2_rebuild_20260618`

## Contract

- IMU input: `aM_rel[:, i, :] = aM_smooth[:, i, :] - aM_smooth[:, 5, :]`.
- GT target: SMPL FK sensor-site position with `tran=0`, centered second difference, then root-IMU subtraction.
- Root index: `5` (`pelvis`).
- Frame: M/world-frame vectors; root acceleration subtraction removes translation leakage but does not rotate into sensor-local coordinates.
- Difference: central second difference `(p[t-1] - 2p[t] + p[t+1]) / dt^2`, `dt=1/60`.

## Command

```bash
python scripts/build_acc_invariance_datacache_v2_20260618.py --overwrite
python scripts/validate_acc_invariance_datacache_v2_20260618.py
```

## Formulation Comparison

| formulation | input | target | dataset | L2 | RMSE | corr |
|------------|-------|--------|---------|---:|-----:|-----:|
| raw absolute | raw absolute |  | AMASS | 2.413734 | 2.354353 | 0.796750 |
| raw absolute | raw absolute |  | DIP | 2.760527 | 3.928749 | 0.493610 |
| raw absolute | raw absolute |  | TotalCapture | 3.120456 | 4.221638 | 0.508395 |
| v2 relative (NEW) | v2 relative (NEW) |  | AMASS | 2.008515 | 2.826319 | 0.668280 |
| v2 relative (NEW) | v2 relative (NEW) |  | DIP | 2.047538 | 3.268881 | 0.583909 |
| v2 relative (NEW) | v2 relative (NEW) |  | TotalCapture | 2.300372 | 3.781834 | 0.577641 |
| zero-trans old | zero-trans old |  | AMASS | 2.729857 | 3.034504 | 0.572082 |
| zero-trans old | zero-trans old |  | DIP | 2.795306 | 3.928958 | 0.468684 |
| zero-trans old | zero-trans old |  | TotalCapture | 2.952501 | 3.994207 | 0.511874 |
| raw absolute | raw absolute |  | ALL | 2.559527 | 2.973042 | 0.693266 |
| zero-trans old | zero-trans old |  | ALL | 2.767145 | 3.342987 | 0.540240 |
| v2 relative (NEW) | v2 relative (NEW) |  | ALL | 2.048167 | 3.035057 | 0.639486 |

## Validator Checks

- Shape consistency: `True`.
- Root invariance max mean `|aM_rel[:,5]|`: `0.000000`.
- Leakage test pass: `1251/1404` sequences have `corr(aM_rel,aGT_rel) > corr(aM_smooth,aGT_rel)`.
- Mean corr(v2 relative): `0.617329`; mean corr(raw absolute): `0.623005`.

## Required Judgment

IF `corr(v2 relative) > corr(raw absolute)`, THEN `root leakage or FK inconsistency exists`.

## Artifacts

- `cache_manifest.json`: cache file list and frame contract.
- `metrics.json`: aggregate metrics and validator checks.
- `per_sequence.csv`: per-sequence/per-sensor metrics.
- `debug_root_leakage.json`: root invariance and leakage diagnostics.

## Conclusion

Conclusion: root leakage or FK inconsistency exists.
