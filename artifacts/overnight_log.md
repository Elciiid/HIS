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
