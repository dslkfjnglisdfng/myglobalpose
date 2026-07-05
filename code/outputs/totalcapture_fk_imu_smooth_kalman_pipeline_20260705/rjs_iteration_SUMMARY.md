# TotalCapture rJS/Pose Alternating Diagnostic

Bounded diagnostic only. Each round fixes rJS, runs the simplified smooth pose optimizer, then refits global rJS from the cleaned pose.

| round | group | before RMSE | after RMSE | delta RMSE |
|---:|---|---:|---:|---:|
| 1 | all | 0.290152 | 0.290338 | 0.000186 |
| 1 | heldout_lower_leg | 0.223659 | 0.223866 | 0.000207 |
| 2 | all | 0.290795 | 0.290945 | 0.000150 |
| 2 | heldout_lower_leg | 0.231003 | 0.231121 | 0.000118 |
| 3 | all | 0.290817 | 0.290967 | 0.000150 |
| 3 | heldout_lower_leg | 0.231046 | 0.231165 | 0.000119 |
