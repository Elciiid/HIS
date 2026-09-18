# Surrogate prediction cost per candidate

> **SYNTHETIC TERRAIN — NOT A PREDICTION FOR REAL ROXAS CITY**

| provenance | |
|---|---|
| source | `SYNTHETIC` |
| config_hash | `b1b3f877c0ab3ef6` |
| seed | `20260917` |
| git_commit | `1bf166f-dirty` |
| timestamp | `2026-09-18T20:31:49Z` |
| device | `cuda` |
| precision | `fp64` |
| engine | `surrogate` |
| notes | `surrogate prediction cost, untrained (timing only)` |


Grid 400x325 @ 20 m on cuda; model weights: **untrained (timing only)**. Candidates are encoded 4 at a time and decoded in time chunks; the scenario's forcing is encoded once per call. Median of the repeated calls.

| detail | batch | s per call | **s per candidate** | peak GPU [MB] | result on host [GB] |
|---|---|---|---|---|---|
| summary | 1 | 0.570 | **0.5705** | 491 | — |
| summary | 8 | 3.908 | **0.4884** | 1873 | — |
| summary | 32 | 15.320 | **0.4788** | 1949 | — |
| summary | 64 | 30.497 | **0.4765** | 2047 | — |
| full | 1 | 0.599 | **0.5994** | 491 | 0.05 |
| full | 8 | 4.143 | **0.5178** | 1873 | 0.41 |
| full | 32 | 16.391 | **0.5122** | 1948 | 1.63 |
| full | 64 | skipped: needs 3.3 GB of host RAM for the histories; 6.1 GB free | | | |

Summary against full detail at batch 32: 1.07x faster per candidate. Summary mode never builds the (nt, ny, nx) histories -- each time chunk updates running maxima and point series and is dropped -- so it saves memory and host transfer, not decoder arithmetic: a peak depth still needs the decoder at every output time.

Batching: 0.5705 s per candidate alone against 0.4765 s at batch 64 (1.20x).
