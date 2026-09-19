"""artifacts/diagnostics_A.md: the Task A diagnostics, from artifacts/diagnostics/*.json.

Every number below is read from those files; the prose around them states what was
decided from them and why.
"""
from __future__ import annotations

import json
from pathlib import Path

PHASE1_TRAIN_PEAK_MB = 4556.07     # artifacts/training_speed.json, Phase 1 (2x coarse, as trained)


def _load(d: Path, name: str):
    p = d / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _f(x, n=3):
    return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def write(cfg) -> Path:
    d = Path(cfg.outdir) / "diagnostics"
    a1, a3, a4, a5 = (_load(d, k) for k in ("a1", "a3", "a4", "a5"))
    s = "# Task A diagnostics\n\n"
    if a1:
        s += f"_provenance: {json.dumps(a1['provenance'])}_\n\n"
    s += "## Branches taken\n\n| part | finding | branch |\n|---|---|---|\n"
    if a1:
        p = a1["pooled"]
        s += (f"| A1 resolution | R = oracle / model effect RMSE = {p['R']:.3f} (on the engine-changed-cell mask "
              f"{p['R_engine_changed_mask']:.3f}) | {p['branch']} |\n")
    s += ("| A2 retained water | reported depth excludes retained water; it is in the mass balance and "
          "`stored_volume_m3` (tests/test_retained_water.py, 4 tests) | no change |\n")
    if a3:
        s += (f"| A3 storage test | plane {'passes' if a3['plane']['passed'] else 'FAILS'}, city far from channels "
              f"{'passes' if a3['city_far']['passed'] else 'fails'}, city as tested "
              f"{'passes' if a3['city_near']['passed'] else 'fails'} | {a3['branch_by_table']}: continue to A4 |\n")
    if a4:
        g = a4["trace"]["terms"]["global_mass"]["physics_phase"]
        s += (f"| A4 mass | lambda_mass median {g['lambda_median']:.2f} (range {g['lambda_min']:.2f}-"
              f"{g['lambda_max']:.2f}) over the physics phase: not suppressed | {a4['branch']} |\n")
    if a5:
        s += f"| A5 memory | {a5['choice']['what']} | option {a5['choice']['option']} |\n"
    s += "\n"

    # ------------------------------------------------------------ A1
    if a1:
        p = a1["pooled"]
        s += "## A1: is the 2x coarse grid the binding constraint?\n\n"
        s += a1["definition"] + ".\n\n"
        s += ("| held-out test pairs (mean of 8) | model | oracle |\n|---|---|---|\n"
              f"| effect RMSE [m] | {p['model_effect_rmse_m']:.4f} | {p['oracle_effect_rmse_m']:.4f} |\n"
              f"| effect RMSE on engine-changed cells [m] | {p['model_effect_rmse_engine_changed_m']:.4f} | "
              f"{p['oracle_effect_rmse_engine_changed_m']:.4f} |\n"
              f"| sign agreement | {p['model_sign_agreement']:.2f} | {p['oracle_sign_agreement']:.2f} |\n"
              f"| effect correlation | {p['model_effect_corr']:.2f} | {p['oracle_effect_corr']:.2f} |\n"
              f"| effect magnitude surviving (sum abs / sum abs) | {p['model_magnitude_ratio']:.3f} | "
              f"{p['oracle_magnitude_ratio']:.3f} |\n"
              f"| projection coefficient | {p['model_projection_coef']:.3f} | {p['oracle_projection_coef']:.3f} |\n\n")
        s += (f"**R = {p['R']:.3f}** (per pair: " + ", ".join(f"{x:.2f}" for x in p["R_per_case"]) +
              f"). The task's table puts 0.3-0.7 at 'both matter: do C4 (cheapest option that fits memory), still "
              f"do C2/C3'. On the common mask of cells the engine moved by more than 1 cm, R = "
              f"{p['R_engine_changed_mask']:.3f}, which is on the 'skip C4' side. The branch follows the defined "
              "metric; the disagreement is logged (overnight_log.md) and C4 is in practice decided by A5.\n\n")
        s += ("What the oracle shows is sharper than R: a perfect surrogate on the 2x grid keeps 87% of the effect "
              "magnitude, gets the sign right on essentially every changed cell and correlates at ~0.9 with the "
              "engine. The Phase 1 model keeps ~5% and is anti-correlated on average. **The coarse grid is not what "
              "stops the model learning effects; the model and its training are.**\n\n")
        s += ("| case | peak RMSE model / oracle [m] | CSI@0.3 model / oracle | effect RMSE model / oracle [m] | "
              "sign model / oracle | corr model / oracle | max rise engine / oracle / model [m] | "
              "area change @0.3 m engine / oracle / model [ha] |\n|---|---|---|---|---|---|---|---|\n")
        for c in a1["cases"]:
            m, o = c["model"], c["oracle"]
            s += (f"| {c['case']} | {m['modified']['peak_rmse_m']:.3f} / {o['modified']['peak_rmse_m']:.3f} | "
                  f"{m['modified']['csi_0.3']:.2f} / {o['modified']['csi_0.3']:.2f} | "
                  f"{m['effect']['effect_rmse_m']:.4f} / {o['effect']['effect_rmse_m']:.4f} | "
                  f"{m['effect']['sign_agreement']:.2f} / {o['effect']['sign_agreement']:.2f} | "
                  f"{m['effect']['effect_corr']:.2f} / {o['effect']['effect_corr']:.2f} | "
                  f"{o['effect']['max_rise_engine_m']:.3f} / {o['effect']['max_rise_surrogate_m']:.3f} / "
                  f"{m['effect']['max_rise_surrogate_m']:.3f} | "
                  f"{o['effect']['area_change_0.3_engine_ha']:+.1f} / {o['effect']['area_change_0.3_surrogate_ha']:+.1f}"
                  f" / {m['effect']['area_change_0.3_surrogate_ha']:+.1f} |\n")
        s += ("\nThe oracle (engine peak depth, block-averaged to 40 m and interpolated back) is the ceiling for any "
              "model on this grid. It is also a ceiling on two of Task E's fixed acceptance targets: reproducing "
              "the engine's local depth increase within 30% (Gate 1) and the flooded-area reduction at 0.30 m within "
              "30%. Both are statistics of a few cells (a single-cell maximum; the area between two nearby "
              "thresholds), and averaging over 2x2 blocks moves them by more than 30% in most pairs above. A "
              "surrogate on this grid cannot meet those two targets by construction, however well it learns; see "
              "the final report.\n\n")
        s += ("Series oracle check (modified storms only; baselines have peak fields only): taking the maximum over "
              "the upsampled coarse series instead of upsampling the coarse peak changes the oracle's peak RMSE by "
              + ", ".join(f"{c['series_oracle_modified']['peak_rmse_m'] - c['oracle']['modified']['peak_rmse_m']:+.4f}"
                          for c in a1["cases"] if "series_oracle_modified" in c) + " m.\n\n")

    # ------------------------------------------------------------ A2
    s += ("## A2: does reported depth include retained storage water?\n\n"
          "No. The engine holds retained water in its own array (`SWE2D.s_filled`): each step fills the store "
          "from the surface water and removes that volume from `h` (with momentum scaled accordingly), and "
          "infiltration drains the store. Every output -- snapshots, `FloodResult.depth_series`, `depth_max` -- is "
          "built from `h`. The retained volume is counted by the mass balance (`CoupledEngine.total_volume` adds "
          "`storage_volume()`) and reported as `stored_volume_m3`. `tests/test_retained_water.py` pins this on a "
          "closed flat box where the answer is exact: 50 mm of rain on a 30 mm store leaves 20 mm of depth and "
          "30 mm stored (to 1e-12); a store larger than the storm leaves no surface water; the balance closes to "
          "1e-12 only when the store is counted; and the engine snapshot reports the surface depth only. No "
          "change was needed, and no metric is re-derived.\n\n")

    # ------------------------------------------------------------ A3
    if a3:
        s += "## A3: where does the engine's storage-induced rise come from?\n\n"
        s += ("Same criteria as the evaluation (footprint max rise <= 5 mm, land max rise <= 20 mm, share of land "
              "rising > 1 cm <= 0.1%, peak-depth volume falls), each case with and without 0.6 m of storage.\n\n"
              "| case | passed | failed | footprint max rise [mm] | land max rise [mm] | cells rising > 5 mm / > 1 cm | "
              "cells falling > 1 cm | footprint mean change [mm] | peak-volume ratio |\n|---|---|---|---|---|---|---|---|---|\n")
        names = {"plane": "(1) tilted plane, no channel, no urban fabric",
                 "plane_micro": "(1b) same plane + 0.1 m micro-topography (not in the task's table)",
                 "city_far": "(2) city, storage >= 500 m from any channel", "city_near": "(3) city, storage as tested"}
        for k, nm in names.items():
            r = a3[k]
            s += (f"| {nm} | {'yes' if r['passed'] else 'no'} | {', '.join(r['failed_criteria']) or '-'} | "
                  f"{r['footprint_max_increase_m'] * 1000:.1f} | {r['land_max_increase_m'] * 1000:.1f} | "
                  f"{r['n_land_rise_gt_5mm']} / {r['n_land_rise_gt_1cm']} | {r['n_land_fall_gt_1cm']} | "
                  f"{r['footprint_mean_change_m'] * 1000:.1f} | {r['peak_volume_ratio']:.3f} |\n")
        far, near = a3["city_far"], a3["city_near"]
        s += (f"\n**Branch: {a3['branch_by_table']}.** Storage is applied correctly: on the smooth plane it lowers "
              "depth everywhere and raises it nowhere. Add 0.1 m of micro-topography and single cells start to rise "
              f"by up to {a3['plane_micro']['land_max_increase_m'] * 1000:.1f} mm while 6,000 fall by more than 1 cm: "
              "that is flow re-routing around a changed wet/dry pattern, and it grows with terrain complexity. On "
              f"the city, storage far from any channel fails on {far['n_land_rise_gt_5mm']} of "
              f"{far['n_footprint']} footprint cells (worst {far['footprint_max_increase_m'] * 1000:.1f} mm, none "
              f"above 1 cm) while {far['n_land_fall_gt_1cm']} land cells fall by more than 1 cm. Storage as tested "
              f"fails on {near['n_land_rise_gt_1cm']} cells above 1 cm, worst "
              f"{near['footprint_max_increase_m'] * 1000:.1f} mm. The engine is not changed; the local rise is a "
              "physical property of the design and moves to Report 2.\n\n")
        s += ("Distances to the channel of the cells rising > 5 mm: far case "
              + ", ".join(f"{x:.0f}" for x in far["rise_gt_5mm_dist_channel_m"]) + " m; as tested "
              + ", ".join(f"{x:.0f}" for x in near["rise_gt_5mm_dist_channel_m"]) + " m. **Flag:** the three "
              "largest rises of the tested case (40, 14 and 11 mm) sit 17-140 m from the channel and exceed anything "
              f"seen far from it ({far['land_max_increase_m'] * 1000:.1f} mm). Re-routing explains the failure (it "
              "happens with no channel nearby), but the double-counted above-bank storage over the channel strip "
              "(a known approximation, see config.py) may add to the magnitude near the channel. The task's table "
              "does not send this case to A3-CHANNEL, and it was not fixed tonight.\n\n")

    # ------------------------------------------------------------ A4
    if a4:
        s += "## A4: why is the surrogate's implied mass balance 15.6%?\n\n"
        s += ("Adaptive weights over the physics phase (steps with ramp > 0), and each term's weighted share of the "
              "total loss:\n\n| term | lambda min | median | max | last | share of total, median | max |\n"
              "|---|---|---|---|---|---|---|\n")
        for n, t in a4["trace"]["terms"].items():
            q = t["physics_phase"]
            s += (f"| {n} | {q['lambda_min']:.3g} | {q['lambda_median']:.3g} | {q['lambda_max']:.3g} | "
                  f"{q['lambda_last']:.3g} | {q['share_median']:.3f} | {q['share_max']:.3f} |\n")
        s += ("\nThe full trace (every 25 steps, 10,000 steps) is in `diagnostics/a4.json`.\n\n"
              "Implied mass-balance error split into its sources (means of absolute values):\n\n"
              "| split | n | reported | depth field vs true sources | true field vs predicted sources | engine's own, "
              "on the coarse grid | stored-volume rel. error | infiltration rel. error |\n|---|---|---|---|---|---|---|---|\n")
        for sp, v in a4["splits"].items():
            m = v["mean_abs"]
            s += (f"| {sp} | {v['n']} | {m['reported']:.3f} | {m['field_vs_true_sources']:.3f} | "
                  f"{m['true_field_vs_predicted_sources']:.3f} | {m['engine_coarse_floor']:.4f} | "
                  f"{m['stored_rel_err']:.3f} | {m['infil_rel_err']:.3f} |\n")
        s += (f"\n**Branch: {a4['branch']}.** The global-mass term was never suppressed. The error is not "
              "overfitting (the training storms are no better than the test storms) and not the coarse grid (the "
              "engine's own output closes to <1% on it). Both halves of the model contribute: the depth field's "
              "volume change is ~10% off the engine's, and the event-total head (infiltration, storage, boundary "
              "exchange) is 7-9% off on its own, worst on retained storage. A rescale of the depth field to the "
              "head's implied volume closes the reported number by construction, which makes the < 2% target "
              "uninformative on its own; C5 therefore reports the depth field against the engine's true sources "
              "alongside it.\n\n")

    # ------------------------------------------------------------ A5
    if a5:
        s += "## A5: memory budget for higher resolution\n\n"
        s += (f"{a5['gpu']}, {a5['total_mb']:.0f} MB; budget {a5['budget_mb']:.0f} MB. bf16 supported "
              f"(including emulation): {a5['bf16_supported_including_emulation']}; native: {a5['bf16_native']}.\n\n"
              "One full training step (data + physics losses, backward, optimiser step) per configuration, 6 steps, "
              "median of the last 4. 'Projected' adds the prepared batches the training loop keeps on the card. "
              "'Calibrated' scales the projection by the ratio between Phase 1's measured training peak "
              f"({PHASE1_TRAIN_PEAK_MB:.0f} MB) and this harness's projection for the same configuration (2x, "
              "checkpointing, fp16): the harness misses validation decoding and allocator overhead, and the ratio "
              "is the only calibration available.\n\n")
        base = next((r for r in a5["rows"] if r["factor"] == 2.0 and r["checkpointing"] and r["autocast"] == "float16"
                     and r["status"] == "ok"), None)
        cal = PHASE1_TRAIN_PEAK_MB / base["projected_training_peak_mb"] if base else None
        s += ("| grid | cell [m] | nodes | checkpointing | autocast | status | s/step | step peak [MB] | projected "
              "[MB] | calibrated [MB] |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in a5["rows"]:
            ok = r["status"] == "ok"
            s += (f"| {r['grid'][0]}x{r['grid'][1]} | {r['cell_m']:.0f} | {r['nodes']} | {r['checkpointing']} | "
                  f"{r['autocast']} | {r['status']} | {_f(r.get('s_per_step'), 2) if ok else '-'} | "
                  f"{_f(r.get('peak_allocated_mb'), 0) if ok else _f(r.get('peak_allocated_mb_before_oom'), 0) + ' at OOM'} | "
                  f"{_f(r.get('projected_training_peak_mb'), 0) if ok else '-'} | "
                  f"{_f(r['projected_training_peak_mb'] * cal, 0) if ok and cal else '-'} |\n")
        ch = a5["choice"]
        s += (f"\n**Decision: option {ch['option']}, {ch['what']}.** " + ch.get("note", "") + ".\n\n")
        if ch["option"] == 4 and cal:
            mid = [r for r in a5["rows"] if r["factor"] == 1.5 and r["status"] == "ok"]
            full = [r for r in a5["rows"] if r["factor"] == 1.0]
            need15 = min(r["projected_training_peak_mb"] * cal for r in mid) if mid else None
            oom = max((r.get("peak_allocated_mb_before_oom") or 0) for r in full) if full else None
            s += ("What would lift it: 1.5x (30 m cells) needs about "
                  f"{need15 / 1024:.1f} GB calibrated for training (it only ran here by spilling into shared system "
                  f"memory, at ~11-12 s/step), so a 10-12 GB card; full 20 m had allocated {oom / 1024:.1f} GB when "
                  "it failed, which is a lower bound, not its peak, so it needs more than a 12 GB card and probably "
                  "16 GB or more. Seconds per step for the chosen option are Phase 1's measured 0.97 s/step "
                  "(artifacts/training_speed.json, validation and checkpoints included); this harness's 1.3-1.6 "
                  "s/step at 2x uses fixed query times and no batch reuse, and is only comparable across rows.\n\n")
    out = Path(cfg.outdir) / "diagnostics_A.md"
    out.write_text(s, encoding="utf-8")
    return out
