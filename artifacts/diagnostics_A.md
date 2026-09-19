# Task A diagnostics

_provenance: {"source": "SYNTHETIC", "config_hash": "05256e50db265f31", "seed": 20260917, "git_commit": "b95f66d-dirty", "timestamp": "2026-09-19T18:05:58Z", "device": "cuda", "precision": "fp64", "engine": "engine+surrogate", "notes": []}_

## Branches taken

| part | finding | branch |
|---|---|---|
| A1 resolution | R = oracle / model effect RMSE = 0.341 (on the engine-changed-cell mask 0.251) | do C4 cheapest option; still C2/C3 |
| A2 retained water | reported depth excludes retained water; it is in the mass balance and `stored_volume_m3` (tests/test_retained_water.py, 4 tests) | no change |
| A3 storage test | plane passes, city far from channels fails, city as tested fails | re-routing (physical): continue to A4 |
| A4 mass | lambda_mass median 1.90 (range 0.73-3.45) over the physics phase: not suppressed | lambda_mass carried weight: post-decoder volume correction (revert if depth RMSE worsens > 20%) |
| A5 memory | stay at 2x coarse (40 m): resolution is capped by the hardware | option 4 |

## A1: is the 2x coarse grid the binding constraint?

oracle = engine peak depth, 2x block-mean coarsened and bilinearly upsampled to 20 m with the surrogate's upsampler; effect = modified - baseline peak depth; R = mean oracle effect RMSE / mean model effect RMSE over the held-out test pairs (effect RMSE as evaluate_effects defines it).

| held-out test pairs (mean of 8) | model | oracle |
|---|---|---|
| effect RMSE [m] | 0.0687 | 0.0234 |
| effect RMSE on engine-changed cells [m] | 0.0845 | 0.0212 |
| sign agreement | 0.59 | 1.00 |
| effect correlation | -0.21 | 0.89 |
| effect magnitude surviving (sum abs / sum abs) | 0.048 | 0.873 |
| projection coefficient | 0.020 | 0.855 |

**R = 0.341** (per pair: 0.25, 0.45, 0.25, 0.47, 0.29, 0.22, 0.56, 0.26). The task's table puts 0.3-0.7 at 'both matter: do C4 (cheapest option that fits memory), still do C2/C3'. On the common mask of cells the engine moved by more than 1 cm, R = 0.251, which is on the 'skip C4' side. The branch follows the defined metric; the disagreement is logged (overnight_log.md) and C4 is in practice decided by A5.

What the oracle shows is sharper than R: a perfect surrogate on the 2x grid keeps 87% of the effect magnitude, gets the sign right on essentially every changed cell and correlates at ~0.9 with the engine. The Phase 1 model keeps ~5% and is anti-correlated on average. **The coarse grid is not what stops the model learning effects; the model and its training are.**

| case | peak RMSE model / oracle [m] | CSI@0.3 model / oracle | effect RMSE model / oracle [m] | sign model / oracle | corr model / oracle | max rise engine / oracle / model [m] | area change @0.3 m engine / oracle / model [ha] |
|---|---|---|---|---|---|---|---|
| test 6 | 0.147 / 0.067 | 0.56 / 0.80 | 0.1084 / 0.0270 | 0.53 / 1.00 | -0.41 / 0.95 | 0.042 / 0.010 / 0.175 | -112.0 / -77.0 / -11.5 |
| test 9 | 0.092 / 0.068 | 0.68 / 0.86 | 0.0776 / 0.0346 | 0.56 / 1.00 | -0.08 / 0.88 | 0.041 / 0.028 / 0.187 | -44.7 / -21.3 / -24.5 |
| test 11 | 0.116 / 0.062 | 0.68 / 0.77 | 0.0612 / 0.0151 | 0.49 / 1.00 | -0.49 / 0.93 | 0.041 / 0.040 / 0.078 | -73.0 / -48.2 / -18.2 |
| test 25 | 0.111 / 0.058 | 0.58 / 0.80 | 0.0409 / 0.0191 | 0.71 / 1.00 | -0.28 / 0.83 | 0.059 / 0.059 / 0.092 | -6.6 / -2.6 / -13.4 |
| test 32 | 0.114 / 0.072 | 0.68 / 0.86 | 0.0786 / 0.0225 | 0.54 / 1.00 | 0.02 / 0.95 | 0.021 / 0.009 / 0.263 | -29.6 / -13.6 / -2.0 |
| test 36 | 0.146 / 0.077 | 0.59 / 0.88 | 0.0266 / 0.0057 | 0.55 / 1.00 | -0.36 / 0.91 | 0.016 / 0.003 / 0.039 | -4.1 / -0.4 / -3.5 |
| test 48 | 0.101 / 0.052 | 0.29 / 0.41 | 0.0768 / 0.0430 | 0.61 / 1.00 | 0.36 / 0.74 | 0.263 / 0.127 / 0.050 | +10.2 / +0.1 / +0.1 |
| test 59 | 0.120 / 0.058 | 0.58 / 0.69 | 0.0795 / 0.0205 | 0.76 / 1.00 | -0.44 / 0.92 | 0.020 / 0.013 / 0.064 | -128.7 / -98.8 / -13.6 |
| storage only | 0.087 / 0.050 | 0.65 / 0.79 | 0.1156 / 0.0564 | 0.68 / 1.00 | -0.06 / 0.60 | 0.029 / 0.029 / 0.017 | -7.4 / -0.4 / -0.5 |
| infiltration only | 0.088 / 0.057 | 0.63 / 0.75 | 0.0838 / 0.0117 | 0.41 / 1.00 | -0.01 / 0.99 | 0.004 / 0.000 / 0.015 | -2.4 / -2.4 / +0.0 |
| roughness only | 0.090 / 0.057 | 0.63 / 0.75 | 0.0272 / 0.0116 | 0.45 / 1.00 | 0.07 / 0.91 | 0.115 / 0.095 / 0.123 | +0.8 / -0.4 / -8.6 |
| conveyance only | 0.088 / 0.057 | 0.64 / 0.76 | 0.0965 / 0.0187 | 0.54 / 1.00 | -0.01 / 0.94 | 0.027 / 0.021 / 0.006 | -2.5 / -0.6 / +0.0 |

The oracle (engine peak depth, block-averaged to 40 m and interpolated back) is the ceiling for any model on this grid. It is also a ceiling on two of Task E's fixed acceptance targets: reproducing the engine's local depth increase within 30% (Gate 1) and the flooded-area reduction at 0.30 m within 30%. Both are statistics of a few cells (a single-cell maximum; the area between two nearby thresholds), and averaging over 2x2 blocks moves them by more than 30% in most pairs above. A surrogate on this grid cannot meet those two targets by construction, however well it learns; see the final report.

Series oracle check (modified storms only; baselines have peak fields only): taking the maximum over the upsampled coarse series instead of upsampling the coarse peak changes the oracle's peak RMSE by +0.0001, +0.0001, +0.0001, +0.0001, +0.0001, +0.0001, +0.0000, +0.0000 m.

## A2: does reported depth include retained storage water?

No. The engine holds retained water in its own array (`SWE2D.s_filled`): each step fills the store from the surface water and removes that volume from `h` (with momentum scaled accordingly), and infiltration drains the store. Every output -- snapshots, `FloodResult.depth_series`, `depth_max` -- is built from `h`. The retained volume is counted by the mass balance (`CoupledEngine.total_volume` adds `storage_volume()`) and reported as `stored_volume_m3`. `tests/test_retained_water.py` pins this on a closed flat box where the answer is exact: 50 mm of rain on a 30 mm store leaves 20 mm of depth and 30 mm stored (to 1e-12); a store larger than the storm leaves no surface water; the balance closes to 1e-12 only when the store is counted; and the engine snapshot reports the surface depth only. No change was needed, and no metric is re-derived.

## A3: where does the engine's storage-induced rise come from?

Same criteria as the evaluation (footprint max rise <= 5 mm, land max rise <= 20 mm, share of land rising > 1 cm <= 0.1%, peak-depth volume falls), each case with and without 0.6 m of storage.

| case | passed | failed | footprint max rise [mm] | land max rise [mm] | cells rising > 5 mm / > 1 cm | cells falling > 1 cm | footprint mean change [mm] | peak-volume ratio |
|---|---|---|---|---|---|---|---|---|
| (1) tilted plane, no channel, no urban fabric | yes | - | -3.3 | 0.0 | 0 / 0 | 6754 | -80.5 | 0.760 |
| (1b) same plane + 0.1 m micro-topography (not in the task's table) | yes | - | 3.4 | 4.8 | 0 / 0 | 5987 | -89.0 | 0.801 |
| (2) city, storage >= 500 m from any channel | no | footprint max rise <= 5 mm | 6.2 | 7.7 | 9 / 0 | 13466 | -78.2 | 0.908 |
| (3) city, storage as tested | no | footprint max rise <= 5 mm, land max rise <= 20 mm | 40.3 | 40.3 | 6 / 4 | 6070 | -37.6 | 0.963 |

**Branch: re-routing (physical).** Storage is applied correctly: on the smooth plane it lowers depth everywhere and raises it nowhere. Add 0.1 m of micro-topography and single cells start to rise by up to 4.8 mm while 6,000 fall by more than 1 cm: that is flow re-routing around a changed wet/dry pattern, and it grows with terrain complexity. On the city, storage far from any channel fails on 9 of 17645 footprint cells (worst 6.2 mm, none above 1 cm) while 13466 land cells fall by more than 1 cm. Storage as tested fails on 4 cells above 1 cm, worst 40.3 mm. The engine is not changed; the local rise is a physical property of the design and moves to Report 2.

Distances to the channel of the cells rising > 5 mm: far case 1183, 1525, 1593, 1607, 1613, 1848, 2109, 2227, 2784 m; as tested 17, 120, 126, 140, 741, 939 m. **Flag:** the three largest rises of the tested case (40, 14 and 11 mm) sit 17-140 m from the channel and exceed anything seen far from it (7.7 mm). Re-routing explains the failure (it happens with no channel nearby), but the double-counted above-bank storage over the channel strip (a known approximation, see config.py) may add to the magnitude near the channel. The task's table does not send this case to A3-CHANNEL, and it was not fixed tonight.

## A4: why is the surrogate's implied mass balance 15.6%?

Adaptive weights over the physics phase (steps with ramp > 0), and each term's weighted share of the total loss:

| term | lambda min | median | max | last | share of total, median | max |
|---|---|---|---|---|---|---|
| boundary | 4.11 | 8.59 | 51.3 | 37.8 | 0.039 | 0.342 |
| continuity | 0.135 | 0.211 | 0.379 | 0.226 | 0.008 | 0.234 |
| global_mass | 0.733 | 1.9 | 3.45 | 2.72 | 0.046 | 0.525 |
| initial | 1.32 | 3.03 | 6.92 | 6.38 | 0.120 | 0.431 |
| momentum_x | 0.000878 | 0.348 | 0.986 | 0.926 | 0.072 | 0.461 |
| momentum_y | 0.00169 | 0.139 | 0.262 | 0.226 | 0.039 | 0.368 |
| stvenant | 0.138 | 0.244 | 0.716 | 0.645 | 0.252 | 0.900 |

The full trace (every 25 steps, 10,000 steps) is in `diagnostics/a4.json`.

Implied mass-balance error split into its sources (means of absolute values):

| split | n | reported | depth field vs true sources | true field vs predicted sources | engine's own, on the coarse grid | stored-volume rel. error | infiltration rel. error |
|---|---|---|---|---|---|---|---|
| train | 45 | 0.173 | 0.120 | 0.074 | 0.0057 | 0.250 | 0.099 |
| val | 6 | 0.193 | 0.114 | 0.096 | 0.0089 | 0.274 | 0.057 |
| test | 9 | 0.156 | 0.102 | 0.088 | 0.0055 | 0.388 | 0.059 |

**Branch: lambda_mass carried weight: post-decoder volume correction (revert if depth RMSE worsens > 20%).** The global-mass term was never suppressed. The error is not overfitting (the training storms are no better than the test storms) and not the coarse grid (the engine's own output closes to <1% on it). Both halves of the model contribute: the depth field's volume change is ~10% off the engine's, and the event-total head (infiltration, storage, boundary exchange) is 7-9% off on its own, worst on retained storage. A rescale of the depth field to the head's implied volume closes the reported number by construction, which makes the < 2% target uninformative on its own; C5 therefore reports the depth field against the engine's true sources alongside it.

## A5: memory budget for higher resolution

NVIDIA GeForce GTX 1660 SUPER, 6144 MB; budget 4608 MB. bf16 supported (including emulation): True; native: False.

One full training step (data + physics losses, backward, optimiser step) per configuration, 6 steps, median of the last 4. 'Projected' adds the prepared batches the training loop keeps on the card. 'Calibrated' scales the projection by the ratio between Phase 1's measured training peak (4556 MB) and this harness's projection for the same configuration (2x, checkpointing, fp16): the harness misses validation decoding and allocator overhead, and the ratio is the only calibration available.

| grid | cell [m] | nodes | checkpointing | autocast | status | s/step | step peak [MB] | projected [MB] | calibrated [MB] |
|---|---|---|---|---|---|---|---|---|---|
| 162x200 | 40 | 32400 | True | float16 | ok | 1.56 | 3324 | 3654 | 4556 |
| 162x200 | 40 | 32400 | True | bfloat16 | ok | 1.29 | 3334 | 3664 | 4568 |
| 162x200 | 40 | 32400 | False | float16 | ok | 7.53 | 5387 | 5717 | 7128 |
| 162x200 | 40 | 32400 | False | bfloat16 | ok | 6.21 | 5393 | 5723 | 7135 |
| 217x267 | 30 | 57939 | True | float16 | ok | 12.01 | 5877 | 6466 | 8062 |
| 217x267 | 30 | 57939 | True | bfloat16 | ok | 10.98 | 5886 | 6475 | 8073 |
| 325x400 | 20 | 130000 | True | float16 | OOM | - | 8035 at OOM | - | - |
| 325x400 | 20 | 130000 | True | bfloat16 | OOM | - | 11875 at OOM | - | - |

**Decision: option 4, stay at 2x coarse (40 m): resolution is capped by the hardware.** option 2 (patch-based training at native 20 m with a coarse global branch) was not implemented, so it could not be measured; no finer grid fits the 4.5 GB budget on this card.

What would lift it: 1.5x (30 m cells) needs about 7.9 GB calibrated for training (it only ran here by spilling into shared system memory, at ~11-12 s/step), so a 10-12 GB card; full 20 m had allocated 11.6 GB when it failed, which is a lower bound, not its peak, so it needs more than a 12 GB card and probably 16 GB or more. Seconds per step for the chosen option are Phase 1's measured 0.97 s/step (artifacts/training_speed.json, validation and checkpoints included); this harness's 1.3-1.6 s/step at 2x uses fixed query times and no batch reuse, and is only comparable across rows.

