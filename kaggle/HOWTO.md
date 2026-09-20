# Running this project on Kaggle

You are moving the compute off the local machine because its RAM is faulty. Nothing needs
uploading: the terrain, the storms and the interventions are all generated from a seed, so a
Kaggle session clones this repository and rebuilds everything from scratch.

You need no hydrology and no prior Kaggle experience. Every step below is literal.

**Read once before starting.** Kaggle gives you a GPU session that ends without warning, on a
weekly quota. So: each stage writes its results as it goes and can resume, and you run one
stage per session. The first session (`k0`) is the important one — it tells us whether the
local machine's faulty RAM corrupted the results we already have.

---

## 1. Kaggle account and GPU access

1. Go to <https://www.kaggle.com> and create an account (Google sign-in is fine).
2. Open <https://www.kaggle.com/settings>.
3. Under **Phone verification**, verify your phone number. **GPU and internet access are only
   granted to phone-verified accounts.** Without it, step 4's accelerator menu stays greyed
   out and the clone in cell 1 fails.
4. While you are there, check **Quota**: Kaggle grants a GPU allowance per week (at the time of
   writing about 30 hours, with single sessions up to about 12 hours). The exact figures show
   in the session panel on the right of a notebook; treat them as the limit, not my numbers.

## 2. Create the notebook

1. Click **Create** (top left) -> **Notebook**.
2. In the new notebook: **File -> Import Notebook**, and upload `kaggle/notebook.ipynb` from
   this repository. (If you would rather not upload a file: create an empty notebook and paste
   the cells from `kaggle/notebook.ipynb` in order. It is three code cells.)
3. Rename it something you will recognise, e.g. `HIS-k0`. The name becomes part of the path
   you use later when chaining sessions, so avoid spaces.

## 3. Settings you must change

Open the panel on the right (**Session options** / the `⋮` menu -> **Accelerator**):

1. **Accelerator -> GPU T4 x2.** Your account offers `None`, `GPU T4 x2` and `TPU v5e-8`.
   Pick **GPU T4 x2** for every stage.
   - **Not the TPU.** This code is CUDA PyTorch with a hand-written finite-volume solver;
     running it on a TPU would need an XLA port that does not exist. It would fail immediately.
   - **About the T4 and float64.** The engine computes in double precision, and a T4 runs
     double precision at about a thirty-second of its single-precision rate. That sounds
     alarming but probably is not: the 2-D solver is limited by memory bandwidth, not
     arithmetic, and a T4's 320 GB/s is about the same as the local machine's 336 GB/s. So
     expect roughly the local rate of ~341 s per storm. `k0` measures it; if it comes back
     much slower, tell me and we will reconsider.
   - **Two GPUs is the win.** Generation runs one storm at a time, so the `generate` stage
     starts one worker per GPU, each taking every other storm, and merges their indexes at the
     end. That halves the wall time. You do not have to do anything: it detects the two cards.
   - **16 GB each** may be enough to train at full 20 m resolution, which the local 6 GB card
     could not. `k0` reports whether it fits; if it does, a whole class of workarounds is
     unnecessary.
2. **Internet -> On.** The notebook clones from GitHub. **Without this the first cell fails
   with `fatal: unable to access ... Could not resolve host: github.com`.** This is the single
   most common mistake.
3. Leave **Persistence** off. Everything we need is written to the notebook's output when the
   session is committed, which is more reliable.

## 4. GitHub token — only if the repository is private

The code reads it from Kaggle Secrets, never from a literal in the notebook.

**If `Elciiid/HIS` is public, skip this section.** The notebook prints
`no GITHUB_TOKEN secret ...; assuming a public repository` and carries on.

If it is private, or if you want the runner to push result files back to a branch:

1. On GitHub: **Settings** (your account) -> **Developer settings** -> **Personal access
   tokens** -> **Fine-grained tokens** -> **Generate new token**.
2. Repository access: **Only select repositories** -> `Elciiid/HIS`.
3. Permissions -> Repository permissions -> **Contents: Read and write**. Nothing else is
   needed. (With a classic token instead, tick the single `repo` scope.)
4. Set an expiry you are comfortable with, generate it, and copy the token string.
5. In the Kaggle notebook: **Add-ons -> Secrets -> Add a new secret**.
   - Label: `GITHUB_TOKEN` (exactly this; the code looks for this name)
   - Value: the token
   - Tick the checkbox that attaches it to this notebook.

The token is used to clone and, at the end of a stage, to push the small text results to a
`kaggle-results` branch. Datasets and model checkpoints are never pushed — they come back as
notebook output.

## 5. First run: the `k0` stage

`k0` does four things: prints the hardware, runs the engine's analytical benchmark suite in
float64 and the test suite, compares every benchmark number against the ones the faulty local
machine produced, measures this GPU's speed, and checks what the storage format costs.

1. Run cell 1. You should see the GPU name and a git commit line.
2. In cell 2, leave `STAGE = "k0"`. Run it.
3. Watch the first minute. It should print an environment block, `[repo] ... at <commit>`,
   `[deps] {...}`, then `[stage] k0 starting`.

**How long:** on the local machine the benchmark suite takes 36 minutes and the test suite
about 10; the hardware and storage parts add two or three engine storms (each about 6 minutes
locally). So expect roughly **1 to 1.5 hours**, faster on a P100. These are local timings —
`k0` replaces them with measured ones.

**What working looks like:** a steady stream of benchmark lines, then `BENCHMARK SUITE: 9/9
passed`, then pytest dots, then a comparison table.

**First signs of trouble:**

| what you see | what it means | what to do |
|---|---|---|
| `Could not resolve host: github.com` | Internet is off | Settings -> Internet -> On, re-run |
| `CUDA not available` in the environment block | No accelerator | Settings -> Accelerator -> GPU |
| `Repository not found` | Private repo, no/invalid token | Redo section 4 |
| A benchmark line saying `FAILED` | The engine itself disagrees with an analytical solution on this machine | Send me the log; do not continue to `generate` |
| Session dies silently | Kaggle ended it | Re-run; `k0` restarts cheaply |

## 6. Read the one output that matters

Open `/kaggle/working/artifacts/verify_comparison.md` (cell 3 prints it). It ends with one of
two verdicts:

- **"this machine reproduces the local benchmark numbers"** — the local machine's arithmetic
  was not visibly corrupted. The earlier conclusions stand on their own reasoning, and we
  regenerate the dataset anyway because a 6-hour generation run touches far more memory than
  this suite does.
- **"the local benchmark numbers are NOT reproduced"** — bit flips reached the results. Say so
  loudly: every measurement made locally has to be re-derived here, including the benchmark
  report, the precision study and the surrogate's accuracy numbers.

Send me this file either way.

## 6b. Decide the worker count by measuring it (`workers`)

Generation runs one storm at a time, so it parallelises across processes. How many to run is
a measurement, not a guess: on the T4 a storm uses only 44 MB of VRAM but the engine's 1-D
solver runs on the CPU, so the limit is vCPUs and host RAM, not the GPU.

1. Run a session with `STAGE = "workers"`. It takes about **1 hour** (it generates the same
   four storms at each worker count).
2. It reports, per rung: storms per hour, how many cores were actually busy, peak host RSS
   and the lowest free RAM seen. It caps the ladder at the machine's vCPU count.
3. It then regenerates the same four storms serially and **compares every array and scalar of
   every record** between the serial and the widest parallel run. Seeds are derived per storm
   from a sha256 of the run key, so execution order must not change a single number.

Read `worker_scaling.md` and use the number it recommends:

```
!python {HIS}/kaggle/run.py --stage generate --workers 2 --budget-hours 10.5 --config kaggle/configs/kaggle.json
```

**If the determinism check says NOT identical, stop and tell me.** That would be a seeding or
ordering bug, and it matters far more than any speedup; generation stays on one worker until
it is understood. The recommendation already falls back to 1 in that case.

### What it measured on Kaggle (2 x Tesla T4, 4 vCPUs, 31.35 GB RAM)

| workers | storms/hour | cores busy | peak host RSS | 104 paired runs would take |
|---|---|---|---|---|
| 1 | 6.77 | 1.0 | 3.84 GB | 15.4 h |
| 2 | 12.91 | 2.0 | 6.84 GB | 8.1 h |
| 4 | 13.42 | 3.9 | 13.61 GB | 7.8 h |

Bit-identical between 1 worker and 4, across 216 arrays and scalars.

The recommendation is the **knee of the curve, not its peak**: the smallest rung within 5% of
the fastest one. Two workers is 3.8% slower than four for half the host memory. The jump from
one worker to two is nearly linear because there are two GPUs; the jump from two to four is
4%, because by then both GPUs are busy and the extra processes only queue behind them. Host
RAM is what gets a Kaggle session killed, so the 4% is not worth 6.8 GB.

Two things to watch in the table, because they are why more workers can be worse:

- **cores busy** rising much more slowly than the worker count means the workers are waiting
  on each other, not working.
- **peak host RSS** scales roughly linearly with workers. If the lowest free RAM gets close to
  zero, the session will be killed rather than slowed.

Worker counts above the GPU count are allowed -- one storm does not saturate a T4 -- and the
workers share the devices round-robin, exactly as the measurement did.

## 7. Generate the paired dataset (`generate`)

Each modified storm gets a matching baseline: same storm, same forcing, no interventions. The
absence of those pairs is a direct cause of the surrogate's failure to learn what an
intervention does.

1. Start a **new session** (or the same notebook; see section 9 for chaining) with
   `STAGE = "generate"`, `WORKERS = 2` (what section 6b measured) and `BUDGET_HOURS = 10.5`
   if your session limit is 12 hours.
2. Use **Save & Run All (Commit)** for this one — see section 8.

The stage refuses to start if `k0`'s benchmark record is missing, and it refuses to start if
the projected dataset would not fit: at full fidelity a record is **39 MB** and 104 records
(60 storms plus 44 baselines) is **4.0 GB**, against a 15 GB internal limit and Kaggle's
roughly 20 GB working directory. It prints the projection before the first storm.

It writes one record per storm and updates the index immediately, so a session that dies
loses at most the storm in flight. When the budget is nearly used it stops cleanly and says
so, rather than being killed mid-storm. Re-running continues where it left off.

**How many storms fit** is computed from the seconds per storm the engine actually measures,
keeping 20% of the phase in reserve. `--budget-hours` is the budget for the **whole stage**:
the storms get half of it and the baselines get whatever is left when the storms are done,
because a storm without its baseline is not a pair and is wasted work. On two T4s with two
workers a storm costs 557 s of wall clock per worker (12.91 storms/hour overall, measured),
so `BUDGET_HOURS = 10.5` should produce roughly **54 of the 60 pairs in about 8.5 hours**.
Re-running the stage fills in the rest; nothing is regenerated. The log prints the projection
before the first storm and says how much time it handed to the baselines.

**Watching two workers.** The stage prints `[parallel] 2 workers started` and then goes quiet,
because each worker writes its own log: `generate.log.gpu0` and `generate.log.gpu1` in the
artifacts directory. To watch progress live, open a second cell and run
`!tail -5 /kaggle/working/artifacts/generate.log.gpu0`. When both finish, the runner prints the
last dozen lines of each and merges their indexes. If one worker dies and the other does not,
the stage fails loudly and re-running picks up only the missing storms.

## 8. Save & Run All versus interactive

- **Interactive** (the normal Run button): you watch the output live, but the session
  idle-times out if you close the tab or stop interacting, and output is only guaranteed once
  you commit. Use it for `k0` and for anything you want to watch.
- **Save & Run All (Commit)** (top right): Kaggle re-runs the notebook from top to bottom on
  its own machine, up to the session limit, and saves everything under
  `/kaggle/working` as that notebook version's **Output**. You can close the browser. Use it
  for `generate`, `train` and `evaluate`.

For a committed run, set `STAGE` in cell 2 before committing — the committed run executes the
notebook exactly as saved.

## 9. Resuming, and chaining one session into the next

Long stages take more than one session. The pattern:

1. When a session finishes, open the notebook's **Output** tab and confirm
   `artifacts/` is there.
2. Start the next session in the same notebook. In the right-hand panel click
   **+ Add Input** -> **Notebook Output** (or the **Your Work** tab) -> select this notebook
   and the version whose output you want.
3. It mounts read-only at `/kaggle/input/<notebook-slug>/`. Note the exact slug that appears
   in the file browser.
4. In cell 2, add the resume flag:

   ```
   !python {HIS}/kaggle/run.py --stage generate --budget-hours 10.5 \
       --config kaggle/configs/kaggle.json \
       --resume-from /kaggle/input/<notebook-slug>/artifacts
   ```

5. The runner copies the previous artifacts into the working directory (it prints how many
   files and how long it took — a few minutes for 4 GB), then continues. Existing files are
   never overwritten.

Training resumes the same way: it checkpoints every 1,000 steps **and** every 10 minutes of
wall time, whichever comes first, and `train` always passes `--resume`.

## 10. Later stages

Run them in this order, one session each, `--resume-from` the previous output:

| stage | what it produces | when to stop and send me results |
|---|---|---|
| `workers` | `worker_scaling.md`: storms/hour, cores busy, peak RAM per worker count, and the determinism check | always, before `generate` |
| `precision` | `precision_study_v2.md`: whether float32 is safe for generation under the new criteria (six full-resolution storms, ~1.5 h locally) | always |
| `generate` | `artifacts/dataset/<hash>/` and `.../baseline/` | when the log says all pairs are done |
| `train` | `artifacts/model/<hash>/` (checkpoints, curves) | when it reports `training done` |
| `evaluate` | `artifacts/validation_report.md` | always — this holds Gate 1 |
| `diagnose` | `artifacts/diagnostics/*.json`, `diagnostics_A.md` | always |

### How long `evaluate` takes, and why

The evaluation needs engine reference runs of its own: four edge-of-envelope probes, four
single-field probes and two storage-check runs, about eleven storms. At the local machine's
341 s per storm that is roughly an hour on top of the surrogate's own work, once; they are
cached under `artifacts/engine_cache/<dataset hash>/` and reused by later evaluations of any
model trained on the same dataset. The per-storm baselines it needs for the effect comparison
are taken from the paired dataset, so those are not re-run.

## 11. Getting results out

Either of these works; the first always works:

1. **Notebook output.** Output tab -> **Download all** (a zip), or click individual files.
2. **The `kaggle-results` branch**, if you set up the token: the small text files
   (`.md`, `.json`, `.csv`, `.log` under 8 MB) are pushed there at the end of each stage,
   under `kaggle_results/<stage>-<timestamp>/`. I can read that directly from GitHub.

## 12. What to send me after each stage

Named explicitly, smallest useful set:

- **After `k0`:** `verify_comparison.md`, `verify_comparison.json`, `kaggle_hardware.md`,
  `storage_check.md`, `kaggle_env.json`, and `benchmark_report.md`. If anything failed, also
  `pytest.log` and `benchmark.log`.
- **After `generate`:** `dataset_card.md`, `dataset/<hash>/index.json`,
  `dataset/<hash>/baseline/index.json`, and `generate.log`. Not the dataset itself.
- **After `train`:** `model/<hash>/curves.json`, `model/<hash>/training_speed.json`,
  `model/<hash>/overfit_check.json`, and `train.log`.
- **After `evaluate`:** `validation_report.md`, `validation_metrics.json`,
  `intervention_effects.json`.
- **After `diagnose`:** `diagnostics_A.md` and everything in `diagnostics/`.

If you are pushing to the `kaggle-results` branch, just tell me the branch has new commits.

## 13. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `fatal: unable to access ... Could not resolve host` | Internet off | Settings -> Internet -> On |
| `Your notebook tried to allocate more memory than is available` | CPU RAM, not GPU | Re-run; if it repeats in `train`, tell me — the batch cache size is a config value |
| `torch.OutOfMemoryError` during `train` | The model grid is too fine for this GPU | Tell me the GPU; `kaggle_hardware.md` already says which grids fit |
| `generation workers failed with codes [...]` | One of the two GPU workers died | Look at `generate.log.gpu0` / `.gpu1`; re-run the stage, it resumes from what was written |
| Only one GPU shows activity in `nvidia-smi` | A stage that is not `generate` | Only generation is parallel; training uses one GPU |
| `RuntimeError: ... benchmark_results.json not found` before generation | `k0` has not run in this session's working directory | Run `k0`, or `--resume-from` a session that did |
| `projected dataset size ... exceeds the 15 GB limit` | Too many storms for the working directory | Lower `data.n_sims` in `kaggle/configs/kaggle.json`, or raise `data.store_coarsen` to 2 |
| `overfit sanity check failed` | Training cannot fit four storms; the pipeline is broken, not undertrained | Send me `train.log`; do not delete the gate file |
| The stage log stops for more than 20 minutes with no new line | A single storm takes minutes; the engine logs progress every 2 minutes | If genuinely silent, the session is wedged: interrupt and re-run |
| Everything is much slower than the table says | You got a T4 and the engine is in float64 | Switch the accelerator to P100 |

## 14. What the stages do, in one line each

- `env` — print and save the hardware description. Runs at the top of every stage anyway.
- `verify` — benchmark suite (float64) + test suite + the comparison against the local numbers.
- `hardware` — seconds per storm in both precisions; seconds per training step and peak VRAM at
  each candidate grid, with and without gradient checkpointing, in fp16 and bf16.
- `storage` — writes one storm at full fidelity and at the dataset's settings, then compares
  what the model is trained on.
- `generate` — the paired dataset, resumable, budget-aware.
- `train` — paired two-stage training (stage 1 learns the baseline flood, stage 2 learns the
  change a design makes), resumable.
- `evaluate` — Gate 1 (does the surrogate reproduce the engine's effect) and Report 2 (what
  each design physically does).
- `diagnose` — the resolution comparison, the volume correction, and the per-candidate cost.
