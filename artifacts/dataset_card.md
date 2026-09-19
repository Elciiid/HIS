# Dataset card: engine simulations for GeoKAN-PINO

> **SYNTHETIC TERRAIN — NOT A PREDICTION FOR REAL ROXAS CITY**

| provenance | |
|---|---|
| source | `SYNTHETIC` |
| config_hash | `dbdd410428f6d7f1` |
| seed | `20260917` |
| git_commit | `537e258` |
| timestamp | `2026-09-19T09:53:33Z` |
| device | `cuda` |
| precision | `fp64` |
| engine | `engine` |
| notes | `-` |

- simulations: 60 (train 45, val 6, test 9); split by whole scenario, stratified by return period
- grid: 400 x 325 @ 20 m; stored at coarsening factor 2; output interval 900 s
- spatial order 2, precision fp64
- engine wall time per simulation: median 341 s, max 365 s, amortised over batches of 1-1 storms advanced together on one set of kernels
- mass-balance error: max 4.69e-15 (every run asserted < 0.001)
- baseline (no intervention) samples: 8

## Sampled intervention envelope (Part B must stay inside it)

| field | range |
|---|---|
| storage_depth S | (0.0, 1.5) m (absolute, after adding to the land-use baseline) |
| infil_multiplier kappa | (1.0, 6.0) |
| manning | baseline x (0.75, 1.25); coastal planting within 800 m of the coast up to (0.06, 0.15) |
| channel_gamma | (0.8, 2.0) per reach (the SSP drainage factor multiplies it) |
| return period | (10, 25, 50, 100) yr |
| RCP | ('RCP2.6', 'RCP4.5', 'RCP6.0', 'RCP8.5') |
| SSP | ('SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5') |
| horizon year | (2030, 2100) |
| storm-tide offset | (-6.0, 6.0) h |
| max storage coverage (train) | 28.4% of land |
| max infiltration-boost coverage (train) | 38.3% of land |
| max mean added storage (train) | 0.149 m |
| max mean kappa excess (train) | 1.084 |

Patterns: Gaussian blobs biased to urban land, road-aligned strips, coastal bands, thresholded smooth noise; signed smooth Manning perturbations; per-reach uniform gamma.
