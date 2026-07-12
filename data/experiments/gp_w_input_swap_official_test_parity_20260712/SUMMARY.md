# Official GlobalPose test parity: G2 VR-only angular-velocity swap

## Verdict

The G0 parity gate passes exactly. The baseline checkout at
`90523d6f38c28ee3a1afd27346cd3624c5efe38a` and current G0 at
`20bfd00fa48d14d4f08d6381fafc65eb01a9d2fb` produced bit-identical pose and
translation predictions on all 19 DIP sequences and all four TotalCapture s5
sequences. Every official aggregate metric difference is zero.

The supported conclusion category is:

> **G2 姿态持平、平移改善、jitter 变差**

More precisely, G2 does not improve the official pose metrics as a group. The
pose changes are extremely small (all aggregate relative changes below 0.08%):
most are slightly worse, while TotalCapture G SIP improves by 0.025%. In
contrast, all seven official TotalCapture translation-window errors improve by
17.0%--32.6%, on every one of the four sequences. Root jitter worsens by 17.4%
on TotalCapture and 3.1% on DIP; joint jitter worsens by 1.8% and 2.8%.

Therefore the earlier statement must not be phrased as an unqualified
"G2 is better than the official baseline." The corrected statement is:

> Under the original GlobalPose `test.py` protocol, VR-only causal RMB angular
> velocity leaves pose essentially unchanged, substantially improves official
> TotalCapture 1--7 m translation-window error, and worsens jitter.

The previous custom Root RMSE/drift results are not used for this verdict.

## Protocol and provenance

- Original baseline worktree: exact commit `90523d6f38c28ee3a1afd27346cd3624c5efe38a`.
- Current G0/G2 worktree: exact experiment commit `20bfd00fa48d14d4f08d6381fafc65eb01a9d2fb` plus the minimal default-off VR override used by this experiment.
- Test entry: the unchanged `test.compare_realimu()` from the selected checkout.
- Evaluator: the unchanged `test.MotionEvaluator`.
- `test.py` SHA-256 in both checkouts: `96fd4369e0eb01649726b23c5754fbeef803588022a47d17b5929078cfc55fd2`.
- DIP: all 19 sequences in `data/test_datasets/dipimu.pt`.
- TotalCapture: the official-calibration file `totalcapture_officalib.pt`, filtered to the official s5 test split: `s5_freestyle1`, `s5_freestyle3`, `s5_rom3`, `s5_walking2`.
- Translation: the exact start/end-pair algorithm inside original `compare_realimu`, evaluated independently for 1--7 m windows. The 7 m values reproduce the printed drift summaries: baseline `7.11%`, G2 `4.84%`.
- No training, GT-pose input, GT-translation input, future frames, shuffling, acceleration change, RMB change, weight change, physics change, or contact/fusion change.
- Weight SHA-256 before and after: `0814864603885aa20165624de8db101a1d6eb9d38a7cddb8bbde39074e7014da`.
- Data lineage audit: 6 nodes and 9 verified edges; no suspicious or unknown edges. Canonical evidence is in `data_lineage_registry.json` and `data_lineage/data_lineage_table.csv`.

G2 is strictly `PL=cached wM`, `VR=causal RMB wM`, with `fps=60`, `lag=2`,
`EMA beta=0.3`, and `delta_R=RMB[t] @ RMB[t-2]^T`. Both GP state and causal
angular-velocity state reset at every sequence boundary.

## G0 versus original baseline parity

| Dataset | Pose max/mean abs diff | Rotation max/mean diff | Translation max/mean abs diff | Max official metric diff | Pass |
|---|---:|---:|---:|---:|---:|
| DIP | 0 / 0 | 0 / 0 deg | 0 / 0 m | 0 | yes |
| TotalCapture s5 | 0 / 0 | 0 / 0 deg | 0 / 0 m | 0 | yes |

## DIP official metrics

| Metric | Baseline original | Current G0 | G2 | G2 - baseline | Relative |
|---|---:|---:|---:|---:|---:|
| L SIP Err (deg) | 13.548558 | 13.548558 | 13.549032 | +0.000474 | +0.0035% |
| L Angle Err (deg) | 8.469930 | 8.469930 | 8.472196 | +0.002266 | +0.0268% |
| L Joint Err (cm) | 4.648240 | 4.648240 | 4.649335 | +0.001095 | +0.0236% |
| L Vertex Err (cm) | 5.408349 | 5.408349 | 5.408962 | +0.000614 | +0.0113% |
| G SIP Err (deg) | 13.409635 | 13.409635 | 13.411542 | +0.001907 | +0.0142% |
| G Angle Err (deg) | 8.291750 | 8.291750 | 8.296232 | +0.004482 | +0.0541% |
| G Joint Err (cm) | 4.547626 | 4.547626 | 4.551080 | +0.003455 | +0.0760% |
| G Vertex Err (cm) | 5.265771 | 5.265771 | 5.268025 | +0.002254 | +0.0428% |
| Root Jitter (km/s^3) | 0.157846 | 0.157846 | 0.162676 | +0.004830 | +3.0598% |
| Joint Jitter (km/s^3) | 0.259061 | 0.259061 | 0.266300 | +0.007239 | +2.7942% |

DIP pose is effectively tied but directionally slightly worse in aggregate.
Depending on the metric, G2 wins 3--10 of 19 sequences. Root jitter wins 4/19
and joint jitter wins 3/19. DIP translation is not evaluated by official
`compare_realimu`.

## TotalCapture s5 official metrics

| Metric | Baseline original | Current G0 | G2 | G2 - baseline | Relative |
|---|---:|---:|---:|---:|---:|
| L SIP Err (deg) | 9.989697 | 9.989697 | 9.989818 | +0.000121 | +0.0012% |
| L Angle Err (deg) | 12.550694 | 12.550694 | 12.551105 | +0.000410 | +0.0033% |
| L Joint Err (cm) | 4.519671 | 4.519671 | 4.520968 | +0.001297 | +0.0287% |
| L Vertex Err (cm) | 5.300476 | 5.300476 | 5.300954 | +0.000478 | +0.0090% |
| G SIP Err (deg) | 9.314009 | 9.314009 | 9.311655 | -0.002354 | -0.0253% |
| G Angle Err (deg) | 11.781373 | 11.781373 | 11.781555 | +0.000182 | +0.0015% |
| G Joint Err (cm) | 3.810419 | 3.810419 | 3.812873 | +0.002454 | +0.0644% |
| G Vertex Err (cm) | 4.470022 | 4.470022 | 4.471030 | +0.001008 | +0.0226% |
| Root Jitter (km/s^3) | 0.399826 | 0.399826 | 0.469305 | +0.069479 | +17.3772% |
| Joint Jitter (km/s^3) | 0.859838 | 0.859838 | 0.875692 | +0.015854 | +1.8439% |

Seven of eight aggregate pose metrics are slightly worse. G SIP improves, but
only one of four sequences improves for that metric; the aggregate improvement
is driven by `s5_freestyle3`. This supports “pose tied,” not “pose improved.”

## Official TotalCapture translation windows

| Window | Baseline original (m) | Current G0 (m) | G2 (m) | Delta (m) | Relative | Sequence wins |
|---:|---:|---:|---:|---:|---:|---:|
| 1 m | 0.126658 | 0.126658 | 0.105091 | -0.021567 | -17.03% | 4/4 |
| 2 m | 0.201038 | 0.201038 | 0.153863 | -0.047176 | -23.47% | 4/4 |
| 3 m | 0.279055 | 0.279055 | 0.202923 | -0.076132 | -27.28% | 4/4 |
| 4 m | 0.354172 | 0.354172 | 0.250884 | -0.103288 | -29.16% | 4/4 |
| 5 m | 0.410884 | 0.410884 | 0.282424 | -0.128460 | -31.26% | 4/4 |
| 6 m | 0.456512 | 0.456512 | 0.307762 | -0.148750 | -32.58% | 4/4 |
| 7 m | 0.497538 | 0.497538 | 0.339063 | -0.158475 | -31.85% | 4/4 |

Official translation improves consistently, not merely in the previous custom
Root RMSE/drift diagnostics.

## TotalCapture per-sequence results

The following compact rows are ordered as L SIP, L Angle, L Joint, L Vertex,
G SIP, G Angle, G Joint, G Vertex, Root Jitter, Joint Jitter.

| Sequence | Baseline | G2 |
|---|---|---|
| s5_freestyle1 | 10.661841, 13.079640, 4.985968, 5.712856, 9.177350, 11.924506, 3.739967, 4.367504, 0.425666, 0.851411 | 10.663494, 13.079692, 4.990663, 5.714359, 9.188366, 11.933005, 3.753944, 4.378480, 0.472984, 0.868491 |
| s5_freestyle3 | 12.173736, 13.495431, 5.239497, 6.060891, 11.783181, 12.934540, 4.871145, 5.554696, 0.678700, 1.686883 | 12.167885, 13.495837, 5.237935, 6.059836, 11.757562, 12.926189, 4.867467, 5.547623, 0.861387, 1.709101 |
| s5_rom3 | 7.881486, 11.835894, 3.664226, 4.512923, 8.055228, 11.469062, 3.259776, 3.933733, 0.061088, 0.119342 | 7.882823, 11.836572, 3.664697, 4.513468, 8.056186, 11.469808, 3.260339, 3.934617, 0.062551, 0.118715 |
| s5_walking2 | 9.241722, 11.791813, 4.188994, 4.915236, 8.240278, 10.797383, 3.370788, 4.024155, 0.433850, 0.781715 | 9.245069, 11.792317, 4.190578, 4.916153, 8.244508, 10.797220, 3.369742, 4.023400, 0.480298, 0.806461 |

All four sequences improve at every 1--7 m translation window. Root jitter
worsens on all four; joint jitter improves only on `s5_rom3`.

Complete per-sequence baseline/G2 values, deltas, relative changes, wins,
losses, ties, means, medians, and best/worst sequences are stored in
`per_sequence_comparison.csv` and `per_sequence_statistics.json`.

## Required questions

1. **Is G0 numerically identical to original baseline?** Yes, exactly, on all evaluated frames and metrics.
2. **Were official data and split used?** Yes: full DIP official test split and the four TotalCapture official-calibration s5 sequences.
3. **Does G2 improve pose?** No. Pose is effectively tied, with seven of eight TotalCapture aggregates and all eight DIP aggregates slightly worse.
4. **Does official translation-window error improve?** Yes. Every 1--7 m aggregate improves, and every window improves on all four sequences.
5. **Does jitter improve?** No. Aggregate root and joint jitter worsen on both datasets.
6. **Is the unqualified claim “G2 is better than baseline” valid?** No.
7. **Correct wording?** G2 preserves pose approximately, improves official TotalCapture translation, and worsens jitter.
