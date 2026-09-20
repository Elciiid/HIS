"""artifacts/fixes_C.md: what Task C changed, with the measured effect of each fix.

Reads the JSON records the C steps write (diagnostics/*.json, the paired-baseline index,
the trained model's curves) and states, per fix, what it does and what it bought.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write(cfg) -> Path:
    from .api import dataset_dir
    from .data.paired import pair_dir
    from .model.geokan_pino import model_dir
    out = Path(cfg.outdir)
    d = out / "diagnostics"
    pidx = _load(pair_dir(cfg) / "index.json")
    probe = _load(d / "paired_step_probe.json")
    c3 = _load(d / "c3_speed.json")
    c5 = _load(d / "c5.json")
    a5 = _load(d / "a5.json")
    curves = _load(model_dir(cfg) / "curves.json")
    speed = _load(model_dir(cfg) / "training_speed.json")
    s = "# Task C: the fixes, and what each one bought\n\n"

    # ---------------------------------------------------------------- C1
    s += "## C1 Paired data (always)\n\n"
    if pidx:
        rows = list(pidx.values())
        src = {k: sum(r["source"] == k for r in rows) for k in ("self", "eval_cache", "engine")}
        done = [r for r in rows if r["source"] == "engine" and "wall_s" in r and r.get("status") != "failed_numerical"]
        failed = [r for r in rows if r.get("status") == "failed_numerical"]
        s += (f"Every modified storm now has a baseline run of the same scenario with no interventions. Of "
              f"{len(rows)} storms: {src['self']} are baseline samples and are their own pair, {src['eval_cache']} "
              f"reuse an engine baseline the evaluation had already run (test storms, peak depth only), and "
              f"{src['engine']} needed a new engine run.\n\n")
        if done:
            w = np.array([r["wall_s"] for r in done])
            me = np.array([abs(r.get("mass_err", 0.0)) for r in done])
            s += (f"Generated: **{len(done)}** baselines, median {np.median(w):.0f} s each, "
                  f"{w.sum() / 3600:.2f} h in total; worst mass-balance error {me.max():.1e}; peak GPU "
                  f"{max(r.get('peak_gpu_mb') or 0 for r in done):.0f} MB. The job is resumable and ordered so a "
                  "partial run stays balanced across return periods.\n\n")
        if failed:
            s += f"**{len(failed)} baseline run(s) failed the engine's plausibility check** and their pairs are excluded.\n\n"
    else:
        s += "_not generated yet_\n\n"

    # ---------------------------------------------------------------- C2
    s += "## C2 Effect-targeted loss (always)\n\n"
    s += ("A loss on the change itself: the squared error of (modified - baseline) depth at the query times, divided "
          "by the storm's own mean squared effect but never by less than a 2 cm effect. Predicting 'no change' on a "
          "real 2 cm effect costs ~1, and a design that moves depths by 2 cm counts as much as one that moves them "
          "by 20 cm. Its weight is floored at `train.effect_floor` x lambda_data in the gradient balancing, so it "
          "cannot be suppressed the way the mass term could have been.\n\n")
    if curves:
        tr = [r for r in curves if "s2_loss_effect" in r]
        if tr:
            lam = [r["s2_lambda_effect"] for r in tr]
            eff = [r["s2_loss_effect"] for r in tr]
            s += (f"Measured over training: effect loss {eff[0]:.3g} -> {eff[-1]:.3g} (best {min(eff):.3g}); its "
                  f"weight stayed between {min(lam):.2f} and {max(lam):.2f}.\n\n")

    # ---------------------------------------------------------------- C3
    s += "## C3 Two-stage residual prediction\n\n"
    s += ("Stage 1 predicts the baseline flood from the static features, the forcing and the scenario's baseline "
          "land surface. Stage 2 takes the candidate's SiteState, its differences from that baseline and stage 1's "
          "latent state, and predicts the delta; the result is baseline + delta. Stage 1 runs once per scenario and "
          "is cached, so Part B pays only stage 2 per candidate.\n\n")
    if c3:
        s += ("| candidates | total [s] | stage 1 (once per scenario) [s] | per candidate, stage 2 only [s] |\n"
              "|---|---|---|---|\n")
        for r in c3["rows"]:
            s += (f"| {r['n_candidates']} | {r['total_s']:.2f} | {r['stage1_s']:.2f} | "
                  f"{r['per_candidate_s']:.3f} |\n")
        s += f"\n{c3['note']}.\n\n"
    if probe:
        s += (f"Training cost, measured on this machine: {probe['s_per_step_data_only']:.2f} s/step in the data-only "
              f"curriculum and {probe['s_per_step_with_physics']:.2f} s/step with the physics terms on, peak GPU "
              f"{probe['peak_gpu_mb']:.0f} MB, on {probe['train_pairs']} training pairs.\n\n")
    if speed:
        s += (f"Full training run: {speed['steps']} steps in {speed['loop_wall_s'] / 3600:.2f} h "
              f"({speed['s_per_step']:.2f} s/step, {speed['form']}), peak GPU {speed['peak_gpu_mb']:.0f} MB.\n\n")

    # ---------------------------------------------------------------- C4
    s += "## C4 Resolution: skipped\n\n"
    if a5:
        ch = a5["choice"]
        s += (f"A1 put R at 0.341, in the band where the task asks for the cheapest resolution option that fits "
              f"memory. A5 measured every option on this card: {ch['what']}. Full 20 m runs out of memory during a "
              "single training step; 30 m cells only run by spilling into shared system memory at 11-12 s/step "
              "(7.9 GB calibrated against a 4.5 GB budget). So no resolution change was made, and the budget went "
              "to C2 and C3 as the R <= 0.3 branch would also have directed.\n\n")

    # ---------------------------------------------------------------- C5
    s += "## C5 Post-decoder volume correction\n\n"
    s += ("A4 found the global-mass weight was never suppressed, so the branch is a correction after the decoder: "
          "rescale the predicted depth on land cells, at every output time, by the single factor that makes the "
          "predicted change in water volume equal the forcing-implied net inflow (rain + boundary - infiltration - "
          "storage, the last three from the model's volume head).\n\n")
    if c5:
        b, a = c5["pooled"]["before"], c5["pooled"]["after"]
        s += "| metric (test storms, pooled) | before | after |\n|---|---|---|\n"
        for k, lab in (("depth_rmse_wet_m", "depth RMSE, wet cells [m]"), ("peak_rmse_m", "peak-depth RMSE [m]"),
                       ("csi_0.3", "CSI @ 0.3 m"), ("implied_mass_err", "implied mass-balance error"),
                       ("field_vs_true_sources", "depth field vs the engine's true sources"),
                       ("effect_rmse_m", "effect RMSE [m]"), ("sign_agreement", "effect sign agreement"),
                       ("effect_corr", "effect correlation")):
            if k in b:
                s += f"| {lab} | {b[k]:.4g} | {a[k]:.4g} |\n"
        scales = [r["scale"] for r in c5["rows"]]
        s += (f"\nScale factors applied: {min(scales):.3f} to {max(scales):.3f}. "
              f"Wet-cell depth RMSE changes by {c5['depth_rmse_wet_change'] * 100:+.1f}%; the rule was to keep the "
              f"correction unless that exceeded +20%. **Decision: {'enabled' if c5['enable'] else 'not enabled'}.**\n\n"
              "Note that the correction drives the *implied* mass-balance error to zero by construction -- it "
              "rescales the field to match the head's own volume -- so that number stops being evidence once the "
              "correction is on. The honest measure is the depth field against the engine's true sources, reported "
              "in the same table.\n\n")
    else:
        s += "_not measured yet_\n\n"

    # ---------------------------------------------------------------- C6
    s += ("## C6 Retained water: no change needed\n\nA2 established that reported depth already excludes water held "
          "in retention storage, that the storage volume is in the mass balance, and that it is reported as "
          "`stored_volume_m3` (tests/test_retained_water.py).\n")
    p = out / "fixes_C.md"
    p.write_text(s, encoding="utf-8")
    return p
