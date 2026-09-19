# What predicts a storm's time-step demand?

> **SYNTHETIC TERRAIN — NOT A PREDICTION FOR REAL ROXAS CITY**

| provenance | |
|---|---|
| source | `SYNTHETIC` |
| config_hash | `ea92140a6e4158f4` |
| seed | `20260917` |
| git_commit | `3a89109-dirty` |
| timestamp | `2026-09-19T02:37:14Z` |
| device | `cuda` |
| precision | `fp64` |
| engine | `engine` |
| notes | `time-step demand against pre-run predictors` |


60 storms, each run alone at full resolution. Median 42224 steps. Outliers (> 1.5x median): [6, 29].

| predictor (known before the run) | Spearman with true step count |
|---|---|
| peak_depth_land_m | +0.79 |
| storage_mean_m | -0.21 |
| kappa_mean | -0.12 |
| gamma_max | -0.12 |
| return_period_yr | -0.09 |
| tide_on | +0.08 |
| rain_total_mm | -0.05 |
| gamma_mean | +0.01 |

80 m rehearsal estimate against the true count, 16 storms: Spearman +0.33.

| sim | steps | min dt [s] | RP | tide | mean gamma | max gamma | rain [mm] | peak land depth [m] |
|---|---|---|---|---|---|---|---|---|
| 6 | 92446 | 0.001 | 50 | on | 1.72 | 1.98 | 138 | 34.46 |
| 29 | 72551 | 0.018 | 25 | on | 1.75 | 1.98 | 110 | 29.77 |
| 32 | 54147 | 0.024 | 10 | on | 1.51 | 1.74 | 97 | 17.54 |
| 8 | 50579 | 0.065 | 10 | on | 1.37 | 1.64 | 97 | 19.88 |
| 36 | 44504 | 0.058 | 10 | on | 1.00 | 1.00 | 103 | 3.50 |
| 25 | 44288 | 0.058 | 25 | on | 1.22 | 1.41 | 105 | 16.50 |
| 39 | 44160 | 0.002 | 100 | on | 1.00 | 1.00 | 156 | 3.52 |
| 9 | 43428 | 0.023 | 25 | on | 1.45 | 1.78 | 110 | 3.26 |
| 57 | 43236 | 0.120 | 25 | on | 1.00 | 1.00 | 110 | 3.26 |
| 17 | 43133 | 0.058 | 25 | on | 1.00 | 1.00 | 105 | 3.22 |
| 33 | 43055 | 0.029 | 25 | on | 1.00 | 1.00 | 110 | 3.31 |
| 2 | 42961 | 0.032 | 50 | on | 1.00 | 1.00 | 129 | 3.19 |
| 7 | 42903 | 0.002 | 100 | on | 1.32 | 1.71 | 156 | 3.25 |
| 23 | 42648 | 0.012 | 100 | on | 1.42 | 1.79 | 135 | 3.21 |
| 51 | 42648 | 0.011 | 100 | on | 1.17 | 1.94 | 141 | 3.21 |

(The 15 storms with the most steps.)
