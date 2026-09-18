# Hydrointelligence System — Part A

A coupled 1-D/2-D shallow-water flood engine and **GeoKAN-PINO**, a surrogate trained on it (Geometry-aware Kolmogorov–Arnold Physics-Informed Neural Operator). Part B (intervention optimisation) builds on the interface in `hydrointel/api.py`.

> **What this is not.** Out of the box, everything runs on *procedurally generated* terrain that only loosely resembles a river-mouth city like Roxas. Every output is stamped `SYNTHETIC TERRAIN — NOT A PREDICTION FOR REAL ROXAS CITY`. Nothing produced here is a flood forecast or a design basis for any real place. Real deployment needs three things: real survey data, calibration against observed events, and sign-off by a licensed engineer.

## The two layers

| layer | what it does | how it is checked |
|---|---|---|
| **Engine** (`hydrointel/solver`) | Ground truth. A 2-D finite-volume shallow-water solver (well-balanced Audusse reconstruction, HLLC flux, Kurganov–Petrova wet/dry handling, semi-implicit Manning friction, MUSCL + SSP-RK2). It is coupled to a 1-D St. Venant channel network (Preissmann slot, level-pool junctions) through weir exchange with Villemonte submergence. Includes Horton infiltration and a sub-grid retention store. PyTorch, float64, `torch.compile`d on CUDA. | Analytical benchmark suite (`artifacts/benchmark_report.md`). Every run also asserts a mass-balance error below 1e-3. |
| **Surrogate** (`hydrointel/model`) | A fast approximation of the engine: Chebyshev-KAN message passing on the 2-D mesh, the 1-D chain and the coupling edges, plus an FFT global-mixing block and a time-conditioned decoder. It is trained on engine output plus non-dimensional PDE residuals, with gradient-norm balanced loss weights. | Scored against held-out engine runs (`artifacts/validation_report.md`). The code makes no accuracy statement that report does not compute. |

## Install

Python 3.11+ and a recent PyTorch are required. On Windows with a CUDA GPU:

```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```
```bash
.venv/Scripts/python -m pip install -e ".[extras,test]" triton-windows
```

`triton-windows` enables `torch.compile` for the solver on CUDA. Without it, set `solver.compile: false` in your config. CPU runs never compile.

On Windows 11 with **Smart App Control** turned on, unsigned DLLs from PyTorch and matplotlib are blocked at import time (`WinError 4551`). That setting was switched off on the development machine. Running under WSL2 avoids the issue without changing it.

## Run

```bash
python -m hydrointel.cli benchmark
```
```bash
python -m hydrointel.cli batch-study
```
```bash
python -m hydrointel.cli generate
```
```bash
python -m hydrointel.cli train
```
```bash
python -m hydrointel.cli evaluate
```
```bash
python -m hydrointel.cli figures --rp 50 --rcp RCP8.5 --ssp SSP5 --year 2050 --tide on
```
```bash
python -m hydrointel.cli --outdir artifacts/quick all --quick
```

- **Stage order.** Stages must run in the order above. `generate` refuses to run unless `artifacts/benchmark_results.json` shows every benchmark passed for the current solver/domain/forcing configuration. `train` needs the dataset, and `evaluate` needs the trained model.
- **Caching.** Each stage writes into a directory keyed by the relevant config hash and skips work it has already done. `generate` resumes simulation by simulation; `train --resume` resumes from the last checkpoint.
- **Batched generation.** `generate` advances `data.batch` storms together as `(B, ny, nx)` tensors on one set of kernels, because a single 130k-cell storm leaves this GPU almost empty. All members share the terrain and the storm time base; roughness, retention, infiltration, channel conveyance, rainfall, tide and upstream inflow are per member, and so is the mass balance, which is asserted per member so one bad storm cannot hide inside a batch. They also share the time step — the smallest any member needs — so a batched run is not bit-identical to running the storms one at a time; `tests/test_batched_solver.py` separates the two effects and bounds the second. `python -m hydrointel.cli batch-study` measures the throughput and memory curve and writes `batch_throughput.md`.
- **Global flags.** `--seed`, `--device`, `--config path.yaml`, `--outdir`, `--precision {fp32,fp64}`. Every command prints the resolved config and its hash, and saves both next to its outputs.
- **`--quick`.** A 100 x 81 grid (80 m cells), 8 simulations, a 30-minute storm and 300 training steps. It is a plumbing check, not a result. Give it its own `--outdir` so it doesn't overwrite the full-run reports.
- **Tests.** `python -m pytest` runs the determinism, mass-balance, coupling, ChebyKAN and API-contract tests, plus a fast CPU subset of the benchmarks.

### Cost on the development machine (GTX 1660 SUPER, 6 GB; Ryzen 5 3600)

Use these figures for planning. Each row says where its number comes from.

| stage | figure | source |
|---|---|---|
| full benchmark suite | see the timings column in `benchmark_report.md` | measured by the suite |
| one full-resolution engine storm (400 x 325 at 20 m, 6 h storm + 2.5 h, MUSCL, fp64) | **~5 min** | estimate: 7.1 ms per step (profiled) × 42,669 steps (counted in a full run of an earlier, slower build) |
| 400 full-resolution simulations | **~35 h** on this GPU, well over the 6 h target for a "modern GPU" | extrapolated from the row above |

For the dataset, reduce `data.n_sims` or run generation overnight; it resumes simulation by simulation.

## Verification (benchmarks)

| # | test | what it guards |
|---|---|---|
| 1 | lake at rest over the synthetic DEM | well-balancing (spurious velocity < 1e-10 m/s) |
| 2 | Ritter dry dam-break, 3–4 grids | dry-front handling, convergence |
| 2b | smooth-wave self-convergence | formal order: ≥ 0.8 first order, ≥ 1.5 MUSCL |
| 3 | Stoker wet dam-break | shock capturing, convergence |
| 4 | Thacker planar bowl (field scale, dx = 20 m) | wetting/drying + well-balancing over 3 periods |
| 5 | steady flow over a bump (sub-, trans-, trans + shock) | steady states, boundary conditions |
| 6 | rainfall-runoff on a tilted plane | rain-on-grid against the kinematic-wave solution |
| 7 | 1-D/2-D overtopping and return flow | exchange conservation to machine precision |
| 8 | closed domain under a design storm (coupled, infiltration, storage) | global mass balance |

Three places where the implemented criteria differ from a literal reading of the brief (the report explains each):

- **Dam-break convergence thresholds.** The ≥ 1.5 MUSCL order target is applied to the smooth problem (2b), not to the dam breaks. Ritter's exact solution has kinks and a dry front, and Stoker's has a shock; both cap the attainable L1 order of any scheme at about 1. The dam-break tests instead require monotone convergence at the operational dry threshold, plus measured orders with that threshold lowered.
- **Thacker scale.** Thacker is judged at field scale: a 2 m-deep, 1 km-radius bowl on 20 m cells. On the 10 cm-deep lab bowl, the 1 mm dry threshold is 1% of the whole water column and dominates the error. That case is still run and reported.
- **Default scheme.** The engine defaults to MUSCL (`solver.spatial_order: 2`), not first order, for dataset generation. Compiled, MUSCL costs about 15% more per step. First-order hydrostatic reconstruction under-represents slope forcing for thin rain-on-grid sheets (benchmark 6), and first order is too diffusive for Thacker (benchmark 4).

## Real data: the manifest

Put your files in `artifacts/inputs/` with a `manifest.json`:

```json
{
  "dataset_id": "roxas_2026_lidar_v1",
  "crs": "EPSG:32651",
  "cell_size_m": 5.0,
  "bounds_m": [xmin, ymin, xmax, ymax],
  "vertical_datum": "MSL",
  "sea_edge": "north",
  "layers": {
    "dem":         {"file": "dem.tif",      "units": "m",     "nodata": -9999},
    "landuse":     {"file": "landuse.tif",  "units": "class", "legend": {"1": "paved road", "2": "residential"}},
    "manning":     {"file": "manning.tif",  "units": "s/m^(1/3)", "optional": true},
    "channel_net": {"file": "channels.csv", "units": "m",     "schema": ["reach_id","node_id","x","y","bed_elev","width","side_slope","manning"]},
    "rain_gauge":  {"file": "rain.csv",     "units": "mm/h",  "schema": ["datetime","intensity"]},
    "tide_gauge":  {"file": "tide.csv",     "units": "m_MSL", "schema": ["datetime","elevation"]},
    "idf_curve":   {"file": "idf.csv",      "units": "mm/h",  "schema": ["return_period_yr","duration_min","intensity"]},
    "obs_flood":   {"file": "observed.csv", "units": "m",     "schema": ["event_id","x","y","max_depth","source"], "optional": true}
  }
}
```

**Validation rules**
- **CRS:** must be projected and metric. It is checked with pyproj if installed; otherwise only UTM and PRS92/Luzon-1911 Philippine zones are accepted.
- **Grid:** `cell_size_m` must match `domain.cell_m` in the run config, and the bounds must be a whole number of cells.
- **Datum:** `vertical_datum` must be `MSL`.
- **Units:** each layer's units string must match exactly.
- **DEM nodata:** any DEM nodata inside the model bounds is an error.
- **Land use:** every land-use code must appear in the legend, and each legend name must map to exactly one model class.

**Resampling** is explicit and logged: bilinear for the DEM, nearest-neighbour for classes and Manning.

**Channel network:** a node whose id starts with `J` is a junction shared between reaches.

**Failure behaviour.** Any problem raises `InputDataError`, naming the file, the field and the expected units. If the manifest is absent, the system runs synthetic terrain and logs one warning. A malformed manifest is never quietly swapped for synthetic data.

`python -m hydrointel.cli evaluate --calibrate` scores the engine against `obs_flood` high-water marks (RMSE, bias, and flooded/not-flooded agreement). Without real data it writes a report saying there is nothing to calibrate against.

## What must be replaced before any real deployment

Each item below is marked `TODO` in the code.

- `PLACEHOLDER_IDF_ROXAS` in `hydrointel/config.py`: illustrative IDF coefficients, **not** PAGASA values. Replace with the PAGASA Roxas/Capiz station IDF, or supply `idf_curve`.
- `HAZARD_CLASSES` in `hydrointel/config.py`: replace with the client's official hazard classification.
- `LANDUSE_TABLE` in `hydrointel/domain/landuse.py`: Manning n, Horton f0/fc/k and bund storage need site calibration.
- `TideConfig` constituents: replace with NAMRIA harmonic constants, or supply `tide_gauge`.
- `CatchmentConfig`: the upstream basin area, time of concentration, storage constant and runoff coefficient need a proper hydrological model of the upstream catchment.
- `RCP_TABLE` / `SSP_TABLE` / `SSP_DEFAULT_RCP` in `hydrointel/forcing/scenarios.py`: placeholder ranges. Replace with the client's adopted projections (IPCC AR6 SSP-RCP pairings). SSP3 maps to RCP6.0 because the table has no RCP7.0.
- **Engineering review:** a licensed engineer must review the calibration and verification before any output informs design.

## Part B contract (`hydrointel/api.py`)

```python
from hydrointel import api
api.configure()                       # or api.configure(RunConfig.load("my.yaml"))
scen = api.Scenario(return_period_yr=50, rcp="RCP8.5", ssp="SSP5", horizon_year=2050, tide_on=True)
site = api.baseline_site(scen)        # SiteState with no interventions (SSP land surface)
fast = api.predict(site, scen)        # surrogate
truth = api.simulate(site, scen)      # engine
before = api.baseline(scen)           # no-intervention reference (surrogate by default)
```

- **Interventions are physical fields.** They enter the engine as four fields: retention storage `S`, infiltration multiplier `κ`, Manning `n`, and channel conveyance `γ`. **Storage removes water from the surface and never alters the DEM** (enforced by `tests/test_api_contract.py`).
- **Stay inside the training envelope.** `FloodResult.in_distribution` and `distribution_warnings` report whether a query leaves the sampled ranges recorded in `dataset_card.md`. Part B must clip its search to that envelope.
- **Verify with the engine.** Optimise with `predict`, then re-verify the chosen designs with `simulate` and report both.

## Layout

```
hydrointel/  config, io (manifest, loaders), domain (synthetic, landuse, channels),
             benchmarks/ (suite, analytical, precision, throughput),
             forcing (idf, hyetograph, catchment, tide, scenarios), solver (swe2d, swe1d,
             coupling, boundaries, infiltration, massbalance), engine, benchmarks,
             data (sampler, generate, dataset), model (chebykan, graph, geokan_pino,
             batch, physics_loss, balancing), train, evaluate, viz, api, cli
tests/       determinism, mass balance, coupling, batched solver, ChebyKAN, API contract,
             fast benchmarks
artifacts/   all outputs (created at run time; git-ignored)
```
