"""Train GeoKAN-PINO on engine simulations.

Stages (each cached by config hash):
  1. normalisation scales from the training split
  2. sanity check: overfit 4 simulations with the data loss; abort if it cannot
  3. full training: data-only curriculum, then physics terms ramped in with
     gradient-norm balanced weights; checkpoints are resumable
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
from .data.dataset import SimDataset, compute_scales
from .model import physics_loss as PL
from .model.balancing import GradNormBalancer
from .model.batch import SampleBuilder, build_model
from .model.geokan_pino import interp_series, model_dir

log = logging.getLogger("hydrointel.train")
PHYS = ("continuity", "momentum_x", "momentum_y", "stvenant", "boundary", "initial", "global_mass")


class Trainer:
    def __init__(self, cfg: RunConfig, device=None):
        self.cfg = cfg
        ctx = api.configure(cfg, device)
        self.ctx = ctx
        self.device = ctx.device
        self.root = api.dataset_dir(cfg)
        if not (self.root / "index.json").exists():
            raise FileNotFoundError(f"no dataset at {self.root}: run `python -m hydrointel.cli generate` first")
        self.out = model_dir(cfg)
        self.out.mkdir(parents=True, exist_ok=True)
        self.train_ds = SimDataset(self.root, "train")
        self.val_ds = SimDataset(self.root, "val")
        sp = self.out / "scales.json"
        if not sp.exists():
            dom = ctx.domain
            compute_scales(self.train_ds, cfg.solver.h_dry, max(dom.nx, dom.ny) * dom.dx, cfg.solver.g).save(sp)
        self.model = build_model(cfg, self.root, self.device)
        self.builder = SampleBuilder(cfg, ctx.domain, self.train_ds.static, self.model.scales, self.model.graph,
                                     self.device)
        self.cell = ctx.domain.dx * self.builder.f
        self._batches: dict = {}

    def batch(self, ds: SimDataset, sid: int) -> dict:
        key = (id(ds), sid)
        if key not in self._batches:
            if len(self._batches) > 16:
                self._batches.pop(next(iter(self._batches)))
            self._batches[key] = self.builder.from_record(ds.load(sid))
        return self._batches[key]

    # ------------------------------------------------------------------ losses
    def losses(self, b: dict, qi: torch.Tensor, physics: bool, amp: bool):
        m, sc = self.model, self.model.scales
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp):
            x2, x1, g = m.encode(b)
        x2, x1, g = x2.float(), x1.float(), g.float()
        o2, o1, d2, d1 = m.decode(x2, x1, b, b["times"][qi], need_dt=physics)
        ld, parts = PL.data_loss(o2, o1, b, qi, sc, m.y1_scale)
        vols = m.volumes(x2, x1, g)
        lv = (((vols - b["vol_target"]) / sc.V_scale) ** 2).mean()
        terms = {"data": ld + lv}
        parts["volumes"] = lv
        is0 = qi == 0
        if bool(is0.any()):
            terms["initial"] = PL.data_loss(o2[is0], o1[is0], b, qi[is0], sc, m.y1_scale)[0]
        if physics:
            terms.update(PL.residuals_2d(o2, d2, b, qi, sc, m.graph, self.cell))
            terms.update(PL.residuals_1d(o1, d1, b, sc, m.graph, m.y1_scale))
            tide = lambda t: interp_series(b["series"][1], t, b["t_end"])
            terms["boundary"] = PL.boundary_loss(o2, b, qi, sc, tide)
            terms["global_mass"] = PL.global_mass_loss(o2, o1, b, qi, sc, self.cell, self.builder.volume1d,
                                                       m.y1_scale).mean()
        return terms, parts, o2

    def queries(self, nt: int, k: int, rng: random.Random) -> torch.Tensor:
        idx = [0] + rng.sample(range(1, nt), min(k - 1, nt - 1))
        return torch.as_tensor(sorted(idx), device=self.device)

    # --------------------------------------------------------------- overfit
    def overfit_check(self, n: int = 4, steps: int | None = None) -> dict:
        path = self.out / "overfit_check.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        cfg = self.cfg
        steps = steps or (150 if cfg.quick else 1500)
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.train.lr)
        amp = cfg.train.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        ids = self.train_ds.ids[:n]
        rng = random.Random(derive_seed(f"overfit/{cfg.seed}"))
        hist = []
        self.model.train()
        for s in range(steps):
            b = self.batch(self.train_ds, ids[s % len(ids)])
            qi = self.queries(b["h"].shape[0], cfg.train.query_times, rng)
            terms, _, _ = self.losses(b, qi, physics=False, amp=amp)
            loss = terms["data"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            hist.append(float(loss.detach()))
        first = float(np.mean(hist[:len(ids)]))
        last = float(np.mean(hist[-len(ids):]))
        res = {"n_samples": len(ids), "steps": steps, "initial_loss": first, "final_loss": last,
               "reduction": last / max(first, 1e-12), "passed": last / max(first, 1e-12) < 0.05,
               "history": hist}
        path.write_text(json.dumps(res), encoding="utf-8")
        self.model.load_state_dict(state)        # the check does not leak into training
        if not res["passed"]:
            raise RuntimeError(f"overfit sanity check failed: data loss only fell from {first:.4g} to {last:.4g} "
                               f"on {len(ids)} samples in {steps} steps — the architecture or pipeline is broken")
        log.info("overfit check passed: data loss %.4g -> %.4g (x%.3g)", first, last, res["reduction"])
        return res

    # ------------------------------------------------------------------ train
    def train(self, resume: bool = False) -> Path:
        cfg, tc = self.cfg, self.cfg.train
        final = self.out / "model.pt"
        if final.exists() and not resume:
            log.info("trained model exists for this config (%s); skipping", final)
            return final
        self.overfit_check()
        model = self.model
        opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=tc.steps, pct_start=0.05)
        amp = tc.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        bal = GradNormBalancer(PHYS, tc.balance_alpha, tc.balance_every)
        rng = random.Random(derive_seed(f"train/{cfg.seed}"))
        step, curves, best = 0, [], math.inf
        ck = self.out / "checkpoint.pt"
        if resume and ck.exists():
            st = torch.load(ck, map_location=self.device, weights_only=False)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"]); bal.load_state_dict(st["bal"])
            step, curves, best = st["step"], st["curves"], st["best"]
            rng.setstate(st["rng"]); torch.set_rng_state(st["torch_rng"])
            log.info("resumed at step %d", step)
        floor_path = self.out / "physics_floor.json"
        if not floor_path.exists():
            fl = [PL.physics_floor(self.batch(self.train_ds, sid), model.scales, model.graph, self.cell)
                  for sid in self.train_ds.ids[:8]]
            floor = {k: float(np.mean([f[k] for f in fl])) for k in fl[0]}
            floor_path.write_text(json.dumps(floor, indent=1), encoding="utf-8")
            log.info("physics residual of the engine's own coarse output: %s", floor)
        t0 = time.perf_counter()
        start_step = step
        model.train()
        while step < tc.steps:
            frac = step / tc.steps
            ramp = 0.0 if frac < tc.curriculum_frac else min(1.0, (frac - tc.curriculum_frac) / tc.ramp_frac)
            physics = ramp > 0
            sid = rng.choice(self.train_ds.ids)
            b = self.batch(self.train_ds, sid)
            qi = self.queries(b["h"].shape[0], tc.query_times, rng)
            terms, parts, _ = self.losses(b, qi, physics, amp)
            if physics and bal.due(step):
                bal.update(terms["data"], {k: v for k, v in terms.items() if k != "data"}, model.parameters())
            total = terms["data"]
            for k, v in terms.items():
                if k == "data":
                    continue
                w = bal.lam.get(k, 1.0) * (1.0 if k == "initial" else ramp)
                total = total + w * v
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: " +
                                         ", ".join(f"{k}={float(v):.3g}" for k, v in terms.items()))
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % tc.log_every == 0 or step == 1:
                rec = {"step": step, "total": float(total.detach()), "ramp": ramp, "lr": sched.get_last_lr()[0]}
                rec.update({f"loss_{k}": float(v) for k, v in terms.items()})
                rec.update({f"part_{k}": float(v) for k, v in parts.items()})
                rec.update({f"lambda_{k}": v for k, v in bal.lam.items()})
                curves.append(rec)
                el = time.perf_counter() - t0
                rate = el / max(step - start_step, 1)
                log.info("step %d/%d total %.4g data %.4g ramp %.2f  (%.2f s/step, ETA %.1f min)", step, tc.steps,
                         float(total), float(terms["data"]), ramp, rate, rate * (tc.steps - step) / 60)
            if step % tc.val_every == 0 or step == tc.steps:
                val = self.validate()
                curves.append({"step": step, **{f"val_{k}": v for k, v in val.items()}})
                if val["rmse_h_all"] < best:
                    best = val["rmse_h_all"]
                    self._save(self.out / "best.pt")
                model.train()
            if step % tc.ckpt_every == 0 or step == tc.steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "bal": bal.state_dict(), "step": step, "curves": curves,
                            "best": best, "rng": rng.getstate(), "torch_rng": torch.get_rng_state()}, ck)
                (self.out / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
        # final model = best validation checkpoint
        best_state = torch.load(self.out / "best.pt", map_location=self.device, weights_only=False)
        torch.save(best_state, final)
        (self.out / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
        from .viz.curves import training_figures
        training_figures(self.cfg, curves, json.loads((self.out / "overfit_check.json").read_text()), self.out)
        log.info("training done: best validation depth RMSE %.4f m", best)
        return final

    def _save(self, path):
        torch.save({"model": self.model.state_dict(), "dataset_dir": str(self.root)}, path)

    @torch.no_grad()
    def validate(self, max_sims: int = 8) -> dict:
        self.model.eval()
        acc = {"rmse_h_all": [], "rmse_h_wet": []}
        for sid in self.val_ds.ids[:max_sims]:
            b = self.batch(self.val_ds, sid)
            x2, x1, _ = self.model.encode(b)
            qi = torch.arange(b["h"].shape[0], device=self.device)
            o2, _, _, _ = self.model.decode(x2, x1, b, b["times"][qi])
            m = PL.unweighted_metrics(o2, b, qi, self.model.scales)
            for k in acc:
                acc[k].append(float(m[k]))
        return {k: float(np.mean(v)) for k, v in acc.items()}


def train(cfg: RunConfig, resume: bool = False, device=None) -> Path:
    seed_everything(cfg.seed)
    return Trainer(cfg, device).train(resume)
