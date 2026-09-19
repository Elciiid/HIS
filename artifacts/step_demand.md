# What predicts a storm's time-step demand?

> **SYNTHETIC TERRAIN — NOT A PREDICTION FOR REAL ROXAS CITY**

| provenance | |
|---|---|
| source | `SYNTHETIC` |
| config_hash | `dbdd410428f6d7f1` |
| seed | `20260917` |
| git_commit | `537e258` |
| timestamp | `2026-09-19T09:54:14Z` |
| device | `cuda` |
| precision | `fp64` |
| engine | `engine` |
| notes | `time-step demand against pre-run predictors` |


60 storms, each run alone at full resolution. Median 42033 steps. Outliers (> 1.5x median): none.

| predictor (known before the run) | Spearman with true step count |
|---|---|
| peak_depth_land_m | +0.67 |
| storage_mean_m | -0.41 |
| gamma_max | -0.27 |
| gamma_mean | -0.21 |
| tide_on | +0.09 |
| kappa_mean | +0.06 |
| rain_total_mm | +0.03 |
| return_period_yr | -0.03 |

80 m rehearsal estimate against the true count, 16 storms: Spearman +1.00.

| sim | steps | min dt [s] | RP | tide | mean gamma | max gamma | rain [mm] | peak land depth [m] |
|---|---|---|---|---|---|---|---|---|
| 36 | 44460 | 0.002 | 10 | on | 1.00 | 1.00 | 103 | 3.50 |
| 39 | 44103 | 0.028 | 100 | on | 1.00 | 1.00 | 156 | 3.51 |
| 9 | 43390 | 0.028 | 25 | on | 1.45 | 1.78 | 110 | 3.26 |
| 57 | 43159 | 0.065 | 25 | on | 1.00 | 1.00 | 110 | 3.26 |
| 17 | 43066 | 0.048 | 25 | on | 1.00 | 1.00 | 105 | 3.22 |
| 33 | 43004 | 0.032 | 25 | on | 1.00 | 1.00 | 110 | 3.31 |
| 2 | 42933 | 0.017 | 50 | on | 1.00 | 1.00 | 129 | 3.19 |
| 7 | 42700 | 0.018 | 100 | on | 1.32 | 1.71 | 156 | 3.25 |
| 51 | 42597 | 0.035 | 100 | on | 1.17 | 1.94 | 141 | 3.21 |
| 23 | 42565 | 0.000 | 100 | on | 1.42 | 1.79 | 135 | 3.21 |
| 50 | 42554 | 0.030 | 50 | off | 1.00 | 1.00 | 138 | 2.81 |
| 12 | 42540 | 0.007 | 10 | on | 1.72 | 1.94 | 89 | 3.17 |
| 16 | 42539 | 0.052 | 10 | on | 0.98 | 1.18 | 97 | 3.36 |
| 0 | 42450 | 0.001 | 10 | off | 1.30 | 1.64 | 97 | 2.79 |
| 42 | 42445 | 0.015 | 50 | on | 1.64 | 1.86 | 124 | 3.09 |

(The 15 storms with the most steps.)
