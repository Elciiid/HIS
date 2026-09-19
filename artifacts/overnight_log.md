# Overnight run: decision log

Started 2026-09-19. The user is asleep and asked me not to block on questions:
where I would have asked, I take the more conservative option, record it here with
the reasoning, and raise it in the final report. No threshold is weakened to make
anything pass; failures are recorded and the run moves on.

Scope agreed before sleeping: Task 3, grouped-batching experiment, Task 4, Task 5
Phase 1 (60 sims, ~10k steps, evaluate). Stop after Phase 1; do not start Phase 2.
Push each task's commit to GitHub.

## Decisions

1. **Batching fallback.** The user's rule: if grouping by step demand does not pay,
   fall back to plain batching at the size Task 2 validated, unless plain batching
   failed its correctness test. It did. Task 2's two checks: forced-step exactness
   PASSED (8.9e-16 m); natural-step peak depth within 1 mm FAILED (1.09 mm at 80 m,
   115.0 and 129.0 mm at 20 m over a full storm), and it was also 0.82x the speed
   of one storm at a time. So the fallback is one storm at a time.

2. **Disbenefit is measured on land.** `max_depth_increase_m` and `area_worsened_ha`
   exclude the sea and the 1-D channel footprint: they are water bodies, not places
   people live, and the precision study showed their single-cell values are the
   noisiest in the model. The all-cell maximum is still returned, in
   `extras["max_depth_increase_any_cell_m"]`, so nothing is hidden.

3. **`simulate` does not compare to baseline by default.** Filling the disbenefit
   fields for the engine costs a second full engine run. Dataset generation calls
   `simulate` for every storm and must not pay that. `simulate(...,
   compare_to_baseline=True)` exists for Part B's final verification (cached per
   scenario). Predict always reports them: the baseline rides along in the same
   batch the first time a scenario is seen, then is cached.

4. **`data.batch = 1` leaves the config hash unchanged.** One storm at a time is the
   reference numerics and predates the setting, so it is omitted from the hash when
   it is 1; existing datasets and models keep their hashes. A batch > 1 shares a
   time step, changes the results, and does enter the hash.

5. **Task 3d is timed on untrained weights first.** No full-resolution model exists
   before Phase 1. Cost does not depend on weight values, so a full-size model with
   untrained weights is timed and labelled as such; the trained model is re-timed
   after Phase 1 as a check.

6. **Evaluation adds engine runs for effect pairs.** "How well does the surrogate
   predict the change a design makes" needs the engine's own change, i.e. an engine
   run of each test scenario's baseline site, plus single-field probes and the
   storage-direction test on the engine at full resolution. About 15 extra engine
   runs (~1.5 h). Conservative choice: measure it properly rather than infer it from
   absolute accuracy, because it is the only question Part B asks.

7. **Grouped batching: not used. Generation runs one storm at a time.** Applied the
   rule fixed before measuring (faster by > 5% AND every member within the 1 mm
   peak-depth requirement). Speed passed: 1.44x (238 s against 342 s per storm; the
   group ran 43,710 steps against ~42,450 alone, a 3% shared-step overhead). Accuracy
   failed: max |d peak| 20.6 / 34.5 / 49.3 / 21.4 mm over all cells, 2.2 / 3.9 / 10.9 /
   7.0 mm on land. Plain batching also failed its accuracy test in Task 2, so per the
   user's rule the fallback is one storm at a time, not plain batching.
   Two things I would raise with a human rather than decide at 4 am:
   - The 1 mm bound on max |d peak| looks unreachable by *any* perturbation of this
     solver. Task 1's fp32-vs-fp64 difference was 29-435 mm on the same metric, i.e.
     grouped batching perturbs the answer by less than a change of float precision
     does. Aggregate measures agree closely (wet RMSE 0.1-0.36 mm, flooded area within
     0.15%). Whether 1 mm on the single worst cell is the right acceptance criterion
     for training data is a judgement call, not something to relax unilaterally.
   - The demand estimator (80 m rehearsal) is useless: Spearman 0.09 against true
     20 m step counts, and it ranks the one real outlier (storm 6, 92k steps) as the
     easiest storm. Grouping only paid here because the chosen group happened to
     contain no outlier. It also costs ~25% of a real run per storm (~90 s against
     ~350 s). Any future grouping needs a better estimator. Storm 6's mean gamma is
     1.72; Phase 1 records every storm's true step count, so the final report checks
     whether gamma (or anything else in the sample) predicts step demand.

8. **ENGINE DEFECT FOUND in Phase 1 data: the 1-D solver goes numerically unstable.**
   Found 10:40 by the step-demand analysis. 6 of 60 storms had physically impossible
   states: 1-D levels 60-112 m above the bank crest, |Q| 10,000-64,000 m^3/s, and in
   5 of them 2-D depths of 16-34 m around the river mouth -- all with a mass-balance
   error of ~1e-14, which is why nothing caught it. Every other storm stays at
   <= 4.3 m above bank and <= 1,066 m^3/s, so the partition is unambiguous.
   Mechanism (traced in sim 6): the last 1-D cell before the tidal (stage) boundary of
   the mouth reach sits in the Preissmann slot (its bank crest is -0.29 m, below the
   tide). An odd-even oscillation there grows slowly until the level crosses the bank
   crest, where the section's storage width jumps 66x (106 m -> 1.6 m); in one step
   the level goes to 214 m. Channel dredging (gamma > 1) makes it worse, but sim 20 is
   a baseline storm (gamma = 1) and failed too. Tried and rejected: lower 1-D Courant
   number (only delays the blow-up), wider slot (fixes sim 6, not sim 29), gentler
   coupling relaxation (no effect), counting ghost cells in the time step (no effect).
   **Not fixed.** Root cause beyond the mechanism is open.
   What I did (conservative, reversible):
   - the engine now checks physical plausibility (2-D depth <= 10 m, 1-D level <= 10 m
     above bank) at every mass check and snapshot and raises NumericalInstabilityError;
   - generation re-checks stored records with the same bounds, moves failures to
     `quarantine/` (kept, not deleted) and marks them failed in the index; generation
     records new failures instead of writing garbage;
   - Phase 1 trains on the 54 clean storms (train 43, val 5, test 6).
   **Correction to Tasks 1 and 2**: their storm 6 (the "92k-step outlier") and Task 1's
   storm 20 were unstable runs. Task 1's 9.75 m fp32 divergence and Task 2's "batching
   does not pay" came from them. Task 1's verdict still stands on clean storm 0 (82 mm,
   mass 1.03e-4). Task 2's conclusion should read: a batch is only as fast as its
   slowest member, and an unstable member destroys it; batches of clean storms gave
   1.44x (grouping study) but still failed the 1 mm accuracy bound.

9. **Engine fixed; Phase 1 restarted from scratch.** Diagnosis narrowed to model
   structure: the 1-D section continued above the bank crest as a Preissmann slot
   (1.5% of top width), a device for closed conduits. In an open channel coupled to a
   floodplain it stores above-bank water in a 1.6 m sliver, so small volumes become huge
   heads (the detector also caught a quick-mode test channel at 12.45 m above bank;
   that test had been passing on mass balance alone). Fix: `slot_width_frac` 0.015 ->
   1.0, i.e. vertical walls at bankfull width. Verified full-length on all three
   unstable storms: sim 6 92,446 -> 39,526 steps, max 2-D depth 34.46 -> 4.39 m,
   max |Q| 63,707 -> 435 m^3/s; sim 29 29.77 -> 4.20 m; sim 20 (baseline) 1-D rise
   112 m -> 0.9 m. All mass errors ~1e-15, and each storm now runs in ~290 s.
   Known approximation (documented in config): above-bank water in the 1-D reach and
   in the 2-D cells over the channel footprint share the channel strip, so that strip's
   storage is counted twice while levels are held together by the exchange.
   This is a solver-config change, so the benchmark suite reruns, the test suite
   reruns, and the whole dataset is regenerated under the new hash. The first Phase 1
   dataset (ea92140a6e4158f4, with its quarantine) is kept on disk for comparison.
   Consequence for Tasks 1 and 2 (recorded, not re-run tonight): their outlier storm
   was an instability; the corrected engine would change their numbers.

10. **Quick-mode overfit gate failed by a hair; recorded, not relaxed.** On the
    regenerated quick dataset the 150-step overfit check reached a loss ratio of 0.0501
    against its 0.05 limit, so no quick-mode model was trained. I checked that my model
    refactor is not the cause: with identical weights and inputs the refactored model
    matches the original bit for bit (0.0 difference in encoder, volumes, decoder and
    time derivatives). The change is in the data (the channel fix changes the targets).
    The full-size check (1,500 steps) runs separately inside Phase 1 training.
    Two fixes that fell out of it:
    - **Gate-bypass bug:** a failed overfit check was cached to disk and a rerun of
      `train` returned the cached result without checking it had passed, silently
      skipping the gate. It now raises again.
    - The Part B contract tests (batching, detail modes, disbenefit) no longer skip
      without a trained quick model: they test mechanics, not accuracy, so they fall
      back to an untrained model of the same architecture. The accuracy-dependent
      interchangeability test still needs a trained model and skips without one.
