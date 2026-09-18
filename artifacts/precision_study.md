# Precision study: float32 vs float64 for dataset generation

> **SYNTHETIC TERRAIN — NOT A PREDICTION FOR REAL ROXAS CITY**

| provenance | |
|---|---|
| source | `SYNTHETIC` |
| config_hash | `95f82786e6a18b05` |
| seed | `20260917` |
| git_commit | `8e2b80f-dirty` |
| timestamp | `2026-09-18T00:06:17Z` |
| device | `cuda` |
| precision | `fp64+fp32` |
| engine | `engine` |
| notes | `float32 vs float64 study for dataset generation` |


Three storms at full resolution (400x325 @ 20 m), each run in both precisions from the same seed with snapshots every 300 s. The analytical benchmark suite is unaffected: it stays float64.

**Result: FAIL** — dataset generation stays in float64.

The measured float32 speedup is 1.02x to 1.20x, so single precision was never going to pay for a rewrite of the verification story even if the accuracy criteria had been met. On this grid the step is not bound by double-precision arithmetic.

## Acceptance criteria

| criterion | limit |
|---|---|
| mass_balance_error | 0.0001 |
| max_abs_depth_diff_m | 0.005 |
| depth_rmse_wet_m | 0.001 |
| peak_depth_diff_m | 0.005 |
| flooded_area_rel_diff | 0.005 |
| lake_at_rest_fp32_velocity_ms | 1e-05 |

## Storms

| # | role | RP | tide | patterns |
|---|---|---|---|---|
| 20 | baseline site, low return period, tide off | 10 | off | none (baseline) |
| 6 | partly perturbed site, mid return period, tide on | 50 | on | storage:coastal, manning:perturb, gamma |
| 0 | storage, infiltration, roughness and conveyance all perturbed; highest return period, tide on | 100 | on | storage:roads, kappa:roads, manning:perturb, gamma |

## Timing and mass balance

| # | fp64 loop [s] | fp32 loop [s] | ratio | fp64 steps | fp32 steps | fp64 mass err | fp32 mass err |
|---|---|---|---|---|---|---|---|
| 20 | 285.1 | 279.8 | 1.02x | 40230 | 40230 | 1.29e-15 | 1.62e-05 |
| 6 | 607.7 | 534.0 | 1.14x | 92280 | 92283 | 1.23e-14 | 4.25e-06 |
| 0 | 283.4 | 236.5 | 1.20x | 42715 | 42715 | 6.29e-15 | 1.03e-04 |

## Peak GPU memory

| # | fp64 [MB] | fp32 [MB] |
|---|---|---|
| 20 | 55 | 29 |
| 6 | 50 | 39 |
| 0 | 67 | 55 |

## Differences (fp32 minus fp64)

| # | max\|dh\| [m] | RMSE wet [m] | bias wet [m] | peak-depth diff, all cells [m] | peak-depth diff, land only [m] | peak-speed diff [m/s] |
|---|---|---|---|---|---|---|
| 20 | 1.32e-01 | 1.60e-04 | +1.19e-06 | 2.92e-02 | 1.82e-03 | 1.92e-02 |
| 6 | 9.75e+00 | 2.81e-02 | -1.72e-03 | 4.35e-01 | 1.24e-01 | 4.17e+00 |
| 0 | 8.15e-02 | 1.92e-04 | -5.16e-06 | 8.15e-02 | 2.12e-02 | 2.43e-01 |

'Land only' excludes the sea and the 1-D channel footprint, which are water bodies rather than flood exposure. The two columns differ by roughly a factor of 4 on every storm: the largest single-cell divergences sit in open water, not on the ground Part B optimises.

## Flooded area on land

| # | threshold [m] | fp64 [ha] | fp32 [ha] | difference [ha] | relative |
|---|---|---|---|---|---|
| 20 | 0.15 | 481.3 | 481.3 | +0.04 | 0.008% |
| 20 | 0.3 | 36.7 | 36.7 | +0.00 | 0.000% |
| 20 | 0.5 | 11.0 | 11.0 | +0.00 | 0.000% |
| 6 | 0.15 | 3232.4 | 3232.2 | -0.12 | 0.004% |
| 6 | 0.3 | 2987.9 | 2987.8 | -0.12 | 0.004% |
| 6 | 0.5 | 2868.6 | 2868.2 | -0.40 | 0.014% |
| 0 | 0.15 | 2096.7 | 2096.6 | -0.08 | 0.004% |
| 0 | 0.3 | 1058.7 | 1058.8 | +0.08 | 0.008% |
| 0 | 0.5 | 471.6 | 471.6 | +0.00 | 0.000% |

## Per-storm verdict

| # | check | value | limit | result |
|---|---|---|---|---|
| 20 | mass_balance_error fp32 | 1.622e-05 | 1.0e-04 | PASS |
| 20 | max|dh| over all cells and times | 1.315e-01 | 5.0e-03 | FAIL |
| 20 | depth RMSE over wet cells | 1.600e-04 | 1.0e-03 | PASS |
| 20 | peak-depth difference | 2.924e-02 | 5.0e-03 | FAIL |
| 20 | flooded-area difference at 0.15 m | 8.311e-05 | 5.0e-03 | PASS |
| 20 | flooded-area difference at 0.3 m | 0.000e+00 | 5.0e-03 | PASS |
| 20 | flooded-area difference at 0.5 m | 0.000e+00 | 5.0e-03 | PASS |
| 6 | mass_balance_error fp32 | 4.250e-06 | 1.0e-04 | PASS |
| 6 | max|dh| over all cells and times | 9.752e+00 | 5.0e-03 | FAIL |
| 6 | depth RMSE over wet cells | 2.808e-02 | 1.0e-03 | FAIL |
| 6 | peak-depth difference | 4.346e-01 | 5.0e-03 | FAIL |
| 6 | flooded-area difference at 0.15 m | 3.712e-05 | 5.0e-03 | PASS |
| 6 | flooded-area difference at 0.3 m | 4.016e-05 | 5.0e-03 | PASS |
| 6 | flooded-area difference at 0.5 m | 1.394e-04 | 5.0e-03 | PASS |
| 0 | mass_balance_error fp32 | 1.031e-04 | 1.0e-04 | FAIL |
| 0 | max|dh| over all cells and times | 8.153e-02 | 5.0e-03 | FAIL |
| 0 | depth RMSE over wet cells | 1.919e-04 | 1.0e-03 | PASS |
| 0 | peak-depth difference | 8.153e-02 | 5.0e-03 | FAIL |
| 0 | flooded-area difference at 0.15 m | 3.815e-05 | 5.0e-03 | PASS |
| 0 | flooded-area difference at 0.3 m | 7.557e-05 | 5.0e-03 | PASS |
| 0 | flooded-area difference at 0.5 m | 0.000e+00 | 5.0e-03 | PASS |

## Lake at rest in float32

Well-balancedness holds to the working precision's round-off, which in single precision is about 1e-7 relative, not the 1e-10 the float64 suite asserts. Measured after 500 steps on the full grid, worst of 1st and 2nd order:

| metric | value | limit | result |
|---|---|---|---|
| max spurious velocity [m/s] | 1.554e-05 | 1e-05 | FAIL |
| max surface deviation [m] | 1.192e-06 | reported | — |
| volume drift [m^3] | 2.645e-02 | reported | — |

This case has no sources or sinks, so the mass-balance ratio degenerates to the absolute volume drift in cubic metres (the accounting divides by `max(gross sources, 1 m^3)`); it is not a relative error.
