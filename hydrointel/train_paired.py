"""Train on paired data: every modified storm with its baseline.

Two model forms, chosen by ``cfg.model.two_stage``:

  two-stage (C3 + C2)   stage 1 learns the baseline from baseline records; stage 2
                        learns the delta from (baseline, modified) pairs, with the
                        effect loss on its output
  single    (C2 only)   one GeoKANPINO learns both records; the effect loss compares
                        its modified prediction with its (detached) baseline prediction

The effect loss is a relative error on the depth change at the query times,

    L_eff = mean((dh_pred - dh_true)^2) / (mean(dh_true^2) + (1 mm / H0)^2)

so a storm whose design moves depths by 2 cm counts as much as one that moves them
by 20 cm, and predicting "no change" everywhere costs 1, not ~0. It is balanced
with the physics terms but its weight is floored at ``cfg.train.effect_floor`` x
lambda_data, so the balancing cannot suppress it the way it could any other term.

Everything else -- data loss, physics residuals, curriculum, gradient balancing,
checkpoints, the overfit gate -- is Phase 1's, applied to each stage.
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import api
from .config import RunConfig, derive_seed, seed_everything
from .data.dataset import SimDataset
from .data.paired import pair_dir
from .model import physics_loss as PL
from .model.balancing import GradNormBalancer
from .model.geokan_pino import interp_series
from .model.two_stage import TwoStage
from .train import PHYS, Trainer

log = logging.getLogger("hydrointel.train")
EFFECT_SCALE_M = 0.02          # the smallest intervention effect the system must resolve


def effect_loss(pred_dh: torch.Tensor, true_dh: torch.Tensor, H0: float) -> torch.Tensor:
    """Squared error of the depth change, divided by the size of the change itself but never
    by less than a 2 cm effect (non-dimensional inputs).

    Dividing by the storm's own mean squared effect makes a design that moves depths by 2 cm
    count as much as one that moves them by 20 cm. The floor is what keeps that finite: a
    baseline-against-baseline pair has no effect at all, and its relative error would be
    0/0. With the floor, such a pair instead penalises any change the model invents, on the
    same 2 cm scale. Predicting "no change" on a real 2 cm effect costs ~1.
    """
    scale = (EFFECT_SCALE_M / H0) ** 2
    return ((pred_dh - true_dh) ** 2).mean() / torch.clamp((true_dh ** 2).mean(), min=scale)


class PairedTrainer(Trainer):
    def __init__(self, cfg: RunConfig, device=None):
        if not cfg.train.paired:
            raise ValueError("PairedTrainer needs cfg.train.paired = true")
        super().__init__(cfg, device)
        pidx_path = pair_dir(cfg) / "index.json"
        if not pidx_path.exists():
            raise FileNotFoundError(f"{pidx_path}: generate the paired baselines first (generate-baselines)")
        pidx = json.loads(pidx_path.read_text(encoding="utf-8"))
        self.base_path = {}
        for k, v in pidx.items():
            if v.get("status") == "failed_numerical" or v["source"] == "eval_cache":
                continue
            p = self.root / f"sim_{int(k):05d}.pt" if v["source"] == "self" else pair_dir(cfg) / v["record"]
            if p.exists():
                self.base_path[int(k)] = p
        self.train_ids = [i for i in self.train_ds.ids if i in self.base_path]
        self.val_ids = [i for i in self.val_ds.ids if i in self.base_path]
        missing = [i for i in self.train_ds.ids if i not in self.base_path]
        if not self.train_ids:
            raise RuntimeError("no training storm has its baseline yet")
        log.info("paired training: %d/%d train storms and %d/%d val storms have their baseline%s",
                 len(self.train_ids), len(self.train_ds.ids), len(self.val_ids), len(self.val_ds.ids),
                 f" (missing: {missing})" if missing else "")
        self._base_recs: dict = {}
        self.two_stage = bool(cfg.model.two_stage)
        if self.two_stage:
            n_feat2 = len(self.model.scales.feat_mean)
            self.model = TwoStage(cfg.model, n_feat2, self.model.graph, self.model.scales).to(self.device)

    # ------------------------------------------------------------------ data
    def base_batch(self, sid: int) -> dict:
        if sid not in self._base_recs:
            if len(self._base_recs) >= 12:
                self._base_recs.pop(next(iter(self._base_recs)))
            rec = torch.load(self.base_path[sid], weights_only=False, mmap=True)
            self._base_recs[sid] = self.builder.from_record(rec)
        return self._base_recs[sid]

    def mod_batch(self, ds: SimDataset, sid: int) -> dict:
        return self.batch(ds, sid)

    # ---------------------------------------------------------------- losses
    def _phys(self, o2, o1, d2, d1, b, qi, m):
        sc = m.scales
        t = {}
        t.update(PL.residuals_2d(o2, d2, b, qi, sc, m.graph, self.cell))
        t.update(PL.residuals_1d(o1, d1, b, sc, m.graph, m.y1_scale))
        t["boundary"] = PL.boundary_loss(o2, b, qi, sc, lambda tt: interp_series(b["series"][1], tt, b["t_end"]))
        t["global_mass"] = PL.global_mass_loss(o2, o1, b, qi, sc, self.cell, self.builder.volume1d,
                                               m.y1_scale).mean()
        return t

    def stage2_terms(self, b1, b0, x2s, x1s, qi, physics: bool, amp: bool):
        """Stage 2 on a pair: data loss on baseline + delta against the modified record, the
        effect loss on the delta alone, and the physics residuals of the modified state."""
        m = self.model
        s1, s2, sc = m.stage1, m.stage2, m.scales
        b2 = TwoStage.stage2_inputs(b1, b0, x2s, x1s)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp):
            x2d, x1d, gd = s2.encode(b2)
        x2d, x1d, gd = x2d.float(), x1d.float(), gd.float()
        t_q = b1["times"][qi]
        od2, od1, dd2, dd1 = s2.decode(x2d, x1d, b2, t_q, need_dt=physics)
        with torch.no_grad():
            ob2, ob1, db2, db1 = s1.decode(x2s.detach(), x1s.detach(), b0, t_q, need_dt=physics)
        o2, o1 = ob2 + od2, ob1 + od1
        ld, parts = PL.data_loss(o2, o1, b1, qi, sc, m.y1_scale)
        vols = s2.volumes(x2d, x1d, gd)
        lv = (((vols - b1["vol_target"]) / sc.V_scale) ** 2).mean()
        terms = {"data": ld + lv}
        parts["volumes"] = lv
        true_dh = (b1["h"][qi] - b0["h"][qi]) / sc.H0
        terms["effect"] = effect_loss(od2[..., 0], true_dh, sc.H0)
        is0 = qi == 0
        if bool(is0.any()):
            terms["initial"] = PL.data_loss(o2[is0], o1[is0], b1, qi[is0], sc, m.y1_scale)[0]
        if physics:
            terms.update(self._phys(o2, o1, db2 + dd2, db1 + dd1, b1, qi, s1))
        return terms, parts

    def single_terms(self, b1, b0, qi, physics: bool, amp: bool):
        """C2 without C3: the one model's modified prediction, its data and physics terms,
        and the effect against its own baseline prediction (detached)."""
        m, sc = self.model, self.model.scales
        terms, parts, o2 = self.losses(b1, qi, physics, amp)
        with torch.no_grad():
            x2, x1, _ = m.encode(b0)
            ob2, _, _, _ = m.decode(x2.float(), x1.float(), b0, b0["times"][qi])
        true_dh = (b1["h"][qi] - b0["h"][qi]) / sc.H0
        terms["effect"] = effect_loss(o2[..., 0] - ob2[..., 0], true_dh, sc.H0)
        return terms, parts

    def stage1_terms(self, b0, qi, physics: bool, amp: bool, model):
        """Phase 1's losses for a model (stage 1, or the single model on a baseline record);
        also returns the encoder latents."""
        sc = model.scales
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp):
            x2, x1, g = model.encode(b0)
        x2, x1, g = x2.float(), x1.float(), g.float()
        o2, o1, d2, d1 = model.decode(x2, x1, b0, b0["times"][qi], need_dt=physics)
        ld, parts = PL.data_loss(o2, o1, b0, qi, sc, model.y1_scale)
        lv = (((model.volumes(x2, x1, g) - b0["vol_target"]) / sc.V_scale) ** 2).mean()
        terms = {"data": ld + lv}
        is0 = qi == 0
        if bool(is0.any()):
            terms["initial"] = PL.data_loss(o2[is0], o1[is0], b0, qi[is0], sc, model.y1_scale)[0]
        if physics:
            terms.update(self._phys(o2, o1, d2, d1, b0, qi, model))
        return terms, parts, (x2, x1)

    @staticmethod
    def weighted(terms: dict, bal: GradNormBalancer, ramp: float):
        total = terms["data"]
        for k, v in terms.items():
            if k == "data":
                continue
            w = bal.lam.get(k, 1.0) * (1.0 if k in ("initial", "effect") else ramp)
            total = total + w * v
        return total

    # --------------------------------------------------------------- overfit
    def overfit_check(self, n: int = 4, steps: int | None = None) -> dict:
        """Phase 1's gate, on pairs: the data losses (both stages, or both records) must
        fall below 0.05 of their start on 4 pairs. The effect loss is reported, not gated."""
        path = self.out / "overfit_check.json"
        if path.exists():
            res = json.loads(path.read_text(encoding="utf-8"))
            if not res.get("passed"):
                raise RuntimeError(f"overfit sanity check failed earlier ({path}): data loss only fell from "
                                   f"{res['initial_loss']:.4g} to {res['final_loss']:.4g} "
                                   f"(x{res['reduction']:.3g}, needs < 0.05); delete the file to re-run it")
            return res
        cfg = self.cfg
        steps = steps or (150 if cfg.quick else 1500)
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.train.lr)
        amp = cfg.train.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        ids = [i for i in self.train_ids if not self.train_ds.index[str(i)]["baseline"]][:n]
        rng = random.Random(derive_seed(f"overfit/{cfg.seed}"))
        hist, eff = [], []
        self.model.train()
        for s in range(steps):
            sid = ids[s % len(ids)]
            b1, b0 = self.mod_batch(self.train_ds, sid), self.base_batch(sid)
            qi = self.queries(b1["h"].shape[0], cfg.train.query_times, rng)
            opt.zero_grad(set_to_none=True)
            loss, e = self._pair_step(b1, b0, qi, False, amp, None, None, 0.0, scaler, data_only=True)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            hist.append(loss)
            eff.append(e)
        first, last = float(np.mean(hist[:len(ids)])), float(np.mean(hist[-len(ids):]))
        res = {"n_samples": len(ids), "steps": steps, "initial_loss": first, "final_loss": last,
               "reduction": last / max(first, 1e-12), "passed": last / max(first, 1e-12) < 0.05,
               "effect_initial": float(np.mean(eff[:len(ids)])), "effect_final": float(np.mean(eff[-len(ids):])),
               "history": hist, "effect_history": eff, "pairs": ids}
        path.write_text(json.dumps(res), encoding="utf-8")
        self.model.load_state_dict(state)
        if not res["passed"]:
            raise RuntimeError(f"overfit sanity check failed: data loss only fell from {first:.4g} to {last:.4g} "
                               f"on {len(ids)} pairs in {steps} steps")
        log.info("overfit check passed: data loss %.4g -> %.4g (x%.3g); effect loss %.3g -> %.3g", first, last,
                 res["reduction"], res["effect_initial"], res["effect_final"])
        return res

    def _pair_step(self, b1, b0, qi, physics, amp, bal1, bal2, ramp, scaler, data_only=False):
        """Forward and backward for one pair (two backward passes: the stages' graphs never
        coexist). Returns (sum of data losses, effect loss) as floats."""
        params = list(self.model.parameters())
        if self.two_stage:
            m = self.model
            t1, _, (x2s, x1s) = self.stage1_terms(b0, qi, physics, amp, m.stage1)
            if bal1 is not None and physics and bal1.due(self._step):
                bal1.update(t1["data"], {k: v for k, v in t1.items() if k != "data"}, m.stage1.parameters())
            tot1 = t1["data"] if data_only else self.weighted(t1, bal1, ramp)
            self._check(tot1, t1)
            scaler.scale(tot1).backward()
            t2, _ = self.stage2_terms(b1, b0, x2s.detach(), x1s.detach(), qi, physics, amp)
            p2 = m.stage2.parameters()
        else:
            t1, _, _ = self.stage1_terms(b0, qi, physics, amp, self.model)
            if bal1 is not None and physics and bal1.due(self._step):
                bal1.update(t1["data"], {k: v for k, v in t1.items() if k != "data"}, params)
            tot1 = t1["data"] if data_only else self.weighted(t1, bal1, ramp)
            self._check(tot1, t1)
            scaler.scale(tot1).backward()
            t2, _ = self.single_terms(b1, b0, qi, physics, amp)
            p2 = params
        # the effect weight stays at 1 through the data-only curriculum; balancing starts with physics
        if bal2 is not None and physics and bal2.due(self._step):
            bal2.update(t2["data"], {k: v for k, v in t2.items() if k != "data"}, p2)
        tot2 = (t2["data"] + t2["effect"] * (bal2.lam["effect"] if bal2 else 1.0)) if data_only \
            else self.weighted(t2, bal2, ramp)
        self._check(tot2, t2)
        scaler.scale(tot2).backward()
        self._last = (t1, t2, tot1, tot2)
        return float(t1["data"].detach() + t2["data"].detach()), float(t2["effect"].detach())

    @staticmethod
    def _check(total, terms):
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite loss: " + ", ".join(f"{k}={float(v):.3g}" for k, v in terms.items()))

    # ------------------------------------------------------------------ train
    def train(self, resume: bool = False) -> Path:
        cfg, tc = self.cfg, self.cfg.train
        final = self.out / "model.pt"
        if final.exists() and not resume:
            log.info("trained model exists for this config (%s); skipping", final)
            return final
        self._step = 0
        self.overfit_check()
        model = self.model
        opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=tc.steps, pct_start=0.05)
        amp = tc.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        bal1 = GradNormBalancer(PHYS, tc.balance_alpha, tc.balance_every)
        bal2 = GradNormBalancer(PHYS + ("effect",), tc.balance_alpha, tc.balance_every,
                                floors={"effect": tc.effect_floor})
        rng = random.Random(derive_seed(f"train/{cfg.seed}"))
        step, curves, best = 0, [], math.inf
        ck = self.out / "checkpoint.pt"
        if resume and ck.exists():
            st = torch.load(ck, map_location=self.device, weights_only=False)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"]); bal1.load_state_dict(st["bal1"]); bal2.load_state_dict(st["bal2"])
            step, curves, best = st["step"], st["curves"], st["best"]
            rng.setstate(st["rng"]); torch.set_rng_state(st["torch_rng"])
            log.info("resumed at step %d", step)
        t0 = time.perf_counter()
        start_step = step
        model.train()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        while step < tc.steps:
            self._step = step
            frac = step / tc.steps
            ramp = 0.0 if frac < tc.curriculum_frac else min(1.0, (frac - tc.curriculum_frac) / tc.ramp_frac)
            physics = ramp > 0
            sid = rng.choice(self.train_ids)
            b1, b0 = self.mod_batch(self.train_ds, sid), self.base_batch(sid)
            qi = self.queries(b1["h"].shape[0], tc.query_times, rng)
            opt.zero_grad(set_to_none=True)
            self._pair_step(b1, b0, qi, physics, amp, bal1, bal2, ramp, scaler)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % tc.log_every == 0 or step == 1:
                t1, t2, tot1, tot2 = self._last
                rec = {"step": step, "total": float(tot1.detach() + tot2.detach()), "ramp": ramp,
                       "lr": sched.get_last_lr()[0]}
                rec.update({f"s1_loss_{k}": float(v) for k, v in t1.items()})
                rec.update({f"s2_loss_{k}": float(v) for k, v in t2.items()})
                rec.update({f"s1_lambda_{k}": v for k, v in bal1.lam.items()})
                rec.update({f"s2_lambda_{k}": v for k, v in bal2.lam.items()})
                curves.append(rec)
                el = time.perf_counter() - t0
                rate = el / max(step - start_step, 1)
                log.info("step %d/%d total %.4g data %.4g/%.4g effect %.3g (lambda %.2f) ramp %.2f  (%.2f s/step, "
                         "ETA %.1f min, peak GPU %.0f MB)", step, tc.steps, rec["total"], float(t1["data"]),
                         float(t2["data"]), float(t2["effect"]), bal2.lam["effect"], ramp, rate,
                         rate * (tc.steps - step) / 60,
                         torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0)
            if step % tc.val_every == 0 or step == tc.steps:
                val = self.validate()
                curves.append({"step": step, **{f"val_{k}": v for k, v in val.items()}})
                log.info("validation at step %d: %s", step, {k: round(v, 4) for k, v in val.items()})
                if val["select"] < best:
                    best = val["select"]
                    self._save(self.out / "best.pt")
                model.train()
            if step % tc.ckpt_every == 0 or step == tc.steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "bal1": bal1.state_dict(), "bal2": bal2.state_dict(),
                            "step": step, "curves": curves, "best": best, "rng": rng.getstate(),
                            "torch_rng": torch.get_rng_state()}, ck)
                (self.out / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
        done = step - start_step
        if done > 0:
            wall = time.perf_counter() - t0
            speed = {"steps": done, "loop_wall_s": wall, "s_per_step": wall / done,
                     "includes": "validation every %d steps and checkpoints every %d" % (tc.val_every, tc.ckpt_every),
                     "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
                     "form": "two-stage" if self.two_stage else "single + effect loss"}
            (self.out / "training_speed.json").write_text(json.dumps(speed, indent=1), encoding="utf-8")
        best_state = torch.load(self.out / "best.pt", map_location=self.device, weights_only=False)
        torch.save(best_state, final)
        (self.out / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
        log.info("training done: best validation selection score %.4f m", best)
        return final

    def _save(self, path):
        torch.save({"model": self.model.state_dict(), "dataset_dir": str(self.root),
                    "two_stage": self.two_stage}, path)

    @torch.no_grad()
    def validate(self, max_sims: int = 8) -> dict:
        """Depth RMSE of the modified storm (all cells, as Phase 1) and RMSE of the effect on
        cells the engine moved by more than 1 cm, both in metres, over the validation pairs.
        The checkpoint kept is the one with the smallest sum of the two."""
        self.model.eval()
        acc = {"rmse_h_all": [], "rmse_effect_changed": [], "effect_magnitude_ratio": []}
        m = self.model
        H0 = m.scales.H0
        for sid in [i for i in self.val_ids if not self.val_ds.index[str(i)]["baseline"]][:max_sims]:
            b1, b0 = self.mod_batch(self.val_ds, sid), self.base_batch(sid)
            qi = torch.arange(b1["h"].shape[0], device=self.device)
            t = b1["times"][qi]
            if self.two_stage:
                x2s, x1s, _ = m.stage1.encode(b0)
                ob2, ob1, _, _ = m.stage1.decode(x2s, x1s, b0, t)
                b2 = TwoStage.stage2_inputs(b1, b0, x2s, x1s)
                x2d, x1d, _ = m.stage2.encode(b2)
                od2, _, _, _ = m.stage2.decode(x2d, x1d, b2, t)
                h1 = (ob2[..., 0] + od2[..., 0]).clamp(min=0.0) * H0
                h0 = ob2[..., 0] * H0
            else:
                x2, x1, _ = m.encode(b1)
                o1, _, _, _ = m.decode(x2, x1, b1, t)
                x2, x1, _ = m.encode(b0)
                o0, _, _, _ = m.decode(x2, x1, b0, t)
                h1, h0 = o1[..., 0] * H0, o0[..., 0] * H0
            acc["rmse_h_all"].append(float(((h1 - b1["h"][qi]) ** 2).mean().sqrt()))
            d_true = b1["h"][qi] - b0["h"][qi]
            d_pred = h1 - h0
            ch = d_true.abs() > 0.01
            acc["rmse_effect_changed"].append(float(((d_pred - d_true)[ch] ** 2).mean().sqrt()) if ch.any() else 0.0)
            acc["effect_magnitude_ratio"].append(float(d_pred[ch].abs().sum() / d_true[ch].abs().sum().clamp(min=1e-9))
                                                 if ch.any() else float("nan"))
        out = {k: float(np.nanmean(v)) for k, v in acc.items()}
        out["select"] = out["rmse_h_all"] + out["rmse_effect_changed"]
        return out


def train_paired(cfg: RunConfig, resume: bool = False, device=None) -> Path:
    seed_everything(cfg.seed)
    return PairedTrainer(cfg, device).train(resume)
