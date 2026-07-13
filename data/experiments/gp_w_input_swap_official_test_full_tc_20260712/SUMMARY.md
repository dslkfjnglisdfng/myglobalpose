# Full TotalCapture official-test G2 audit

## Result

The supported conclusion is:

> **G2 keeps pose effectively unchanged, improves the official 1–7 m translation errors on full TotalCapture, and worsens root/joint jitter. The four-sequence s5 subset substantially overestimated the translation gain.**

This does not support the unconditional statement “G2 overall outperforms GlobalPose.”

## Protocol and inventory

- Baseline: commit `90523d6f38c28ee3a1afd27346cd3624c5efe38a`, original `test.py`, `net.py`, and `data/weights.pt`.
- Current source: commit `4a997e8652943879fc6e423980f292864becc305`.
- Weight SHA-256 before and after: `0814864603885aa20165624de8db101a1d6eb9d38a7cddb8bbde39074e7014da`.
- Official calibration: 45 sequences, 176,249 frames.
- DIP calibration: 45 sequences, 176,243 frames.
- Every formal run passed the complete release dictionary unchanged to original `test.compare_realimu`, which traverses `range(len(data['pose']))`.
- No s5, subject, motion-name, `--sequence`, or `max_sequences` filtering was used.
- The pose evaluator is original `test.MotionEvaluator`; translation uses the original start/end frame-pair construction and sequence-first aggregation.
- DIP-calibration cache names are `0..44`; `dataset_inventory.*` retains those source names and records same-index official-calibration names only as reporting labels, never as filters.

## G0 parity

Full-sequence parity passed exactly for both calibrations:

| Calibration | Pose max/mean | Rotation max/mean | Translation max/mean | Aggregate max diff |
|---|---:|---:|---:|---:|
| Official | 0 / 0 | 0 / 0 deg | 0 / 0 m | 0 |
| DIP | 0 / 0 | 0 / 0 deg | 0 / 0 m | 0 |

Therefore G2 is directly comparable with the original baseline.

## Paper reproduction

All ten Table 1 pose/jitter metrics reproduce after rounding to two decimals for both calibrations (20/20 values).

Figure 4 7 m drift does not reproduce exactly:

| Calibration | Paper | Full G0 | Match at 2 decimals? |
|---|---:|---:|---:|
| Official | 4.68% | 4.901711% | No |
| DIP | 3.74% | 3.978288% | No |

Thus Table 1 is reproduced, while the published Figure 4 drift values are not. The original metric formula was not changed to force a match.

## Official-calibration G2

The eight aggregate pose metrics change between -0.011% and +0.029%: two improve microscopically and six worsen microscopically. This is effectively pose-tied.

| Window | Baseline | G2 | Relative | Sequence wins/losses |
|---:|---:|---:|---:|---:|
| 1 m | 0.115455 | 0.113031 | -2.10% | 25 / 20 |
| 2 m | 0.175217 | 0.169287 | -3.38% | 30 / 15 |
| 3 m | 0.229372 | 0.218570 | -4.71% | 27 / 18 |
| 4 m | 0.272332 | 0.256954 | -5.65% | 26 / 19 |
| 5 m | 0.298822 | 0.278911 | -6.66% | 27 / 18 |
| 6 m | 0.321657 | 0.298720 | -7.13% | 27 / 18 |
| 7 m | 0.343120 | 0.316963 | -7.62% | 28 / 16* |

`*` One sequence has no valid 7 m frame pair, so 44 sequences contribute to that window.

- 7 m drift: `4.901711% -> 4.528038%`.
- Root jitter: `0.214459 -> 0.228331`, +6.47% (4 wins, 41 losses).
- Joint jitter: `0.367276 -> 0.375019`, +2.11% (8 wins, 37 losses).

## DIP-calibration G2

All eight aggregate pose metrics worsen slightly, by +0.019% to +0.166%. This is practically close but directionally worse.

| Window | Baseline | G2 | Relative | Sequence wins/losses |
|---:|---:|---:|---:|---:|
| 1 m | 0.105699 | 0.102741 | -2.80% | 26 / 19 |
| 2 m | 0.157115 | 0.150730 | -4.06% | 28 / 17 |
| 3 m | 0.199016 | 0.188676 | -5.20% | 32 / 13 |
| 4 m | 0.234783 | 0.221270 | -5.76% | 30 / 15 |
| 5 m | 0.254170 | 0.237153 | -6.69% | 31 / 14 |
| 6 m | 0.266288 | 0.246445 | -7.45% | 28 / 17 |
| 7 m | 0.278480 | 0.258454 | -7.19% | 27 / 17* |

`*` One sequence has no valid 7 m frame pair.

- 7 m drift: `3.978288% -> 3.692195%`.
- Root jitter: `0.203081 -> 0.221133`, +8.89% (4 wins, 41 losses).
- Joint jitter: `0.345175 -> 0.358985`, +4.00% (4 wins, 41 losses).

## s5 subset versus full TotalCapture

The prior s5-only official-calibration experiment reported 17.0%–32.6% improvements across 1–7 m and wins on all four sequences. On the complete 45-sequence official cache, the gains shrink to 2.1%–7.6%, and 15–20 sequences worsen depending on the window.

At 7 m, the subject-average delta is `-0.158475 m` for s5, versus `-0.012474`, `-0.004178`, `-0.029932`, and `+0.005904 m` for s1–s4. Walking is slightly worse on average (`+0.004620 m`) under official calibration. Therefore the s5 subset is unusually favorable and materially overestimates the full-data benefit.

What remains valid from the s5 result: pose is nearly unchanged, official translation improves in aggregate, and jitter worsens. What must be withdrawn: the implication that translation improves on every sequence or by roughly 17%–33% on full TotalCapture.

## Answers

1. Both release files contain 45 sequences; all were traversed.
2. Full G0 is bit-identical to the original baseline.
3. Table 1 is reproduced to its published precision; Figure 4 7 m drift is not.
4. G2 pose is effectively tied on official calibration and slightly worse on DIP calibration.
5. All seven aggregate official translation windows improve for both calibrations, usually on a majority but not all sequences.
6. Root and joint jitter worsen for both calibrations.
7. The s5 qualitative direction generalizes, but its magnitude and all-sequence consistency do not.
8. The final evidence supports “pose-tied, translation-better, jitter-worse,” not unconditional overall superiority.

Detailed high-precision values are in `official_metric_comparison.csv`; translation wins are in `translation_window_comparison.csv`; every sequence/metric comparison and best/worst statistics are in `per_sequence_comparison.csv` and `per_sequence_statistics.json`.
