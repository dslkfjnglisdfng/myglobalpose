# Root velocity audit

Saved final root translations only; no model, inference, or weights were changed.

- Frame: final predicted/GT root translation in the TotalCapture model/world coordinate system.
- Velocity: `(p[t] - p[t-1]) / (1/60)` m/s; first frame is omitted.
- Low frequency: centered 15-frame, edge-replicated moving average of velocity; high frequency is velocity minus that average.
- Aggregation: unweighted mean of the 45 per-sequence metrics.

## officalib

- Velocity RMSE: 0.255448 -> 0.265099 m/s (+3.78%).
- Low/high RMSE: 0.163897 -> 0.164273; 0.172488 -> 0.184760 m/s.
- Mean bias vector: (0.002939, 0.000013, 0.001929) -> (0.002829, -0.000077, 0.000819) m/s; norm 0.003516 -> 0.002946 m/s.
- RMSE sequence wins/losses: 4 / 41.

## dipcalib

- Velocity RMSE: 0.243439 -> 0.255612 m/s (+5.00%).
- Low/high RMSE: 0.158134 -> 0.158209; 0.160959 -> 0.177112 m/s.
- Mean bias vector: (0.001000, -0.000189, 0.001835) -> (0.000633, -0.000188, 0.000187) m/s; norm 0.002098 -> 0.000686 m/s.
- RMSE sequence wins/losses: 2 / 43.

## Conclusion

G2 主要降低低频/DC bias，但高频误差更大。

The lower DC bias does not make overall framewise velocity more accurate here: low-frequency RMSE is essentially unchanged/slightly worse and total velocity RMSE is worse. See `aggregate_metrics.csv` and `per_sequence_metrics.csv` for every requested metric and per-sequence G2-vs-G0 outcome.
