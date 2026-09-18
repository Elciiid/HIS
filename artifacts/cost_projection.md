# Run cost from measured rates

> **SYNTHETIC TERRAIN — NOT A PREDICTION FOR REAL ROXAS CITY**

| provenance | |
|---|---|
| source | `SYNTHETIC` |
| config_hash | `60e7caf69c6e8ca9` |
| seed | `20260917` |
| git_commit | `1bf166f-dirty` |
| timestamp | `2026-09-18T20:32:36Z` |
| device | `` |
| precision | `` |
| engine | `engine+surrogate` |
| notes | `projection from recorded measurements` |


Generation: **372 s per storm** (one storm at a time, measured full-length (batch_throughput.json)). Training: **0.95 s per step** (full-size probe of 2026-09-17 (no dataset yet)).

| simulations | generation | training steps | training | total |
|---|---|---|---|---|
| 60 | 6.2 h | 10,000 | 2.6 h | **8.8 h** |
| 150 | 15.5 h | 20,000 | 5.3 h | **20.8 h** |
| 300 | 31.0 h | 30,000 | 7.9 h | **38.9 h** |
| 400 | 41.3 h | 40,000 | 10.6 h | **51.9 h** |

Training steps per size follow the plan (Phase 1 ~10k, Phase 2 ~20k) and then the spec's 40k at 400; they are a plan, not a measured need. Evaluation adds roughly 1.5-2 h of engine runs per phase (effect pairs, probes, the engine storage check).

**Part B cost per scenario** at the measured surrogate rate, 0.4765 s per candidate (summary mode, batch 64, untrained (timing only) weights):

| evaluations per scenario | wall time |
|---|---|
| 10,000 | 79 min |
| 30,000 | 238 min |
| 100,000 | 794 min |

The surrogate is ~780x faster than the engine per evaluation, which is what makes Part B possible at all: 10,000 engine evaluations would take 43 days. It is still decoder-bound -- batching buys ~1.2x and the summary mode ~7%, because every output time is decoded at every node -- so a 100,000-evaluation search costs over half a day per scenario on this GPU. Part B's search budget (population x generations) has to be planned around that, or the surrogate needs a cheaper peak-only output path.
