# Batched dataset generation: measured throughput

> **SYNTHETIC TERRAIN — NOT A PREDICTION FOR REAL ROXAS CITY**

| provenance | |
|---|---|
| source | `SYNTHETIC` |
| config_hash | `95f82786e6a18b05` |
| seed | `20260917` |
| git_commit | `467edf6-dirty` |
| timestamp | `2026-09-18T09:07:26Z` |
| device | `cuda` |
| precision | `fp64` |
| engine | `engine` |
| notes | `batched dataset-generation throughput study` |


Grid 400x325 @ 20 m on cuda (6144 MB, budget 3072 MB)


**Step breakdown**, synchronised so GPU work is charged to the call that issued it (ms per step). The last column is the cost per storm-step at the largest batch relative to the smallest: below 1 means batching amortised that part.

| part | B = 1 | B = 8 | per-storm |
|---|---|---|---|
| total | 10.86 | 44.59 | 0.51x |
| step_2d | 6.27 | 38.36 | 0.76x |
| step_1d | 1.91 | 2.29 | 0.15x |
| exchange | 1.99 | 2.10 | 0.13x |
| dt_2d | 0.34 | 1.41 | 0.52x |
| dt_1d | 0.26 | 0.30 | 0.15x |
| accounting | 0.09 | 0.12 | 0.18x |

**Full-length stage.** 8 storms run start to end (8.5 h of storm) at batch 8, and 2 of them also run alone. This is the stage that prices a dataset and the only one that measures the shared time step honestly: over a short window the members' CFL limits have barely diverged.

| metric | value |
|---|---|
| batch size | 8 |
| storms | 8 |
| total loop time | 60.4 min |
| **seconds per storm, batched** | **453 s** |
| seconds per storm, one at a time | 372 s (mean of 2) |
| **speedup** | **0.82x** |
| peak GPU memory | 503 MB |
| worst mass-balance error | 5.69e-15 |
| steps per chunk | 90682 |

**Cost of the shared time step**, on the 400x325 @ 20 m grid above. Each storm's peak depth field from the batch, against the same storm run alone with its own step:

| storm | steps alone | max \|peak depth difference\| [mm] | within 1.0 mm |
|---|---|---|---|
| 0 | 42494 | 115.0454 | NO |
| 1 | 42200 | 129.0254 | NO |

Worst case 129.0254 mm against a 1.0 mm bound: **OVER BOUND**. The batch ran 90682 steps.

**Verdict.** Batching does not pay on this machine. At batch 8 a storm costs 453 s against 372 s run alone (0.82x). The step breakdown shows why it should have worked -- a storm-step costs about half as much inside a batch -- and the step counts show why it did not: the shared time step made the batch run 90682 steps where a member alone needs about 42347, because the whole batch is dragged down to whichever member is most restricted at that moment. The per-step gain and the extra steps very nearly cancel. `data.batch` therefore defaults to 1; the batched solver stays available, tested and documented, for batches of storms with similar time-step demand.

**Projected dataset generation** at the measured full-length rate:

| simulations | batched | one at a time |
|---|---|---|
| 60 | 7.5 h | 6.2 h |
| 150 | 18.9 h | 15.5 h |
| 300 | 37.7 h | 31.0 h |
| 400 | 50.3 h | 41.3 h |
