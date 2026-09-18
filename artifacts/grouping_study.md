# Grouped batching by estimated time-step demand

> **SYNTHETIC TERRAIN — NOT A PREDICTION FOR REAL ROXAS CITY**

| provenance | |
|---|---|
| source | `SYNTHETIC` |
| config_hash | `95f82786e6a18b05` |
| seed | `20260917` |
| git_commit | `1bf166f-dirty` |
| timestamp | `2026-09-18T20:27:50Z` |
| device | `cuda` |
| precision | `fp64` |
| engine | `engine` |
| notes | `grouped batching by estimated step demand` |


**Decision: generate one storm at a time.** Grouped batch speedup 1.44x (needs > 1.05x); peak-depth requirement 1 mm NOT met.

## (a) Estimating step demand before running

Each storm is rehearsed start to end on an 80 m grid with its intervention fields block-averaged; the 80 m step count is the estimate.

| storm | RP | tide | rain [mm] | mean gamma | 80 m steps | 20 m steps (truth) | rehearsal [s] |
|---|---|---|---|---|---|---|---|
| 0 | 10 | off | 97 | 1.30 | 10384 | 42494 | 140 |
| 1 | 25 | on | 114 | 1.00 | 10298 | 42200 | 92 |
| 2 | 50 | on | 129 | 1.00 | 10510 | — | 93 |
| 3 | 100 | off | 141 | 1.00 | 9900 | — | 91 |
| 4 | 10 | on | 97 | 1.00 | 10242 | — | 87 |
| 5 | 25 | on | 114 | 1.16 | 10360 | 42437 | 91 |
| 6 | 50 | on | 138 | 1.72 | 9618 | 92280 | 100 |
| 7 | 100 | on | 157 | 1.32 | 10447 | — | 98 |
| 8 | 10 | on | 97 | 1.37 | 9812 | — | 88 |
| 9 | 25 | on | 110 | 1.45 | 10621 | — | 101 |
| 10 | 50 | on | 138 | 1.69 | 10324 | — | 101 |
| 11 | 100 | on | 135 | 1.21 | 10241 | — | 95 |
| 12 | 10 | on | 89 | 1.72 | 10392 | — | 92 |
| 13 | 25 | on | 105 | 1.00 | 10278 | — | 91 |
| 14 | 50 | on | 119 | 1.00 | 10375 | 42461 | 87 |
| 15 | 100 | on | 141 | 1.29 | 10361 | 42404 | 87 |

Against the 6 storms with a measured 20 m step count: Spearman rank correlation 0.09. That is few storms to judge an estimator by; the Phase 1 dataset records the true step count of every storm it runs, and the final report re-checks the estimate against those.

## Grouped batch against the same storms run alone

Group [5, 15, 14, 0]: batch of 4 ran 43710 steps in 952 s (238 s per storm); alone, 342 s per storm. **Speedup 1.44x.** Worst batched mass error 4.0e-15.

| storm | steps alone | max \|d peak\| all cells [mm] | on land [mm] | RMSE wet [mm] | area diff 0.15/0.30/0.50 m |
|---|---|---|---|---|---|
| 0 | 42494 | 20.65 | 2.15 | 0.108 | 0.054% / 0.015% / 0.000% |
| 5 | 42437 | 34.53 | 3.87 | 0.196 | 0.039% / 0.035% / 0.014% |
| 14 | 42461 | 49.31 | 10.92 | 0.250 | 0.027% / 0.048% / 0.018% |
| 15 | 42404 | 21.38 | 7.03 | 0.364 | 0.064% / 0.149% / 0.051% |

## (b) Is grouped agreement better than ungrouped?

Ungrouped (Task 2, batch of 8 including a 92k-step storm): max |d peak| over all cells 115.0 and 129.0 mm for storms 0 and 1. Grouped: worst 49.31 mm. Grouping agrees better, as expected: the shared step is closer to each member's own.

## (c) Does grouping sort storms by severity?

Rank correlation of estimated demand with: return period -0.15, total rain -0.07, mean channel gamma 0.09. A strong correlation means demand-grouped batches are also severity-homogeneous, so any error the shared step introduces would differ systematically between severe and mild storms -- a pattern the surrogate could learn as if it were physics. That risk only matters if grouped batching is used.
