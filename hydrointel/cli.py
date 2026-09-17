"""Command line interface.

    python -m hydrointel.cli benchmark [--fast]
    python -m hydrointel.cli generate  [--quick]
    python -m hydrointel.cli train     [--quick] [--resume]
    python -m hydrointel.cli evaluate  [--quick] [--calibrate]
    python -m hydrointel.cli figures   --rp 50 --rcp RCP8.5 --ssp SSP5 --year 2050 --tide on
    python -m hydrointel.cli all       [--quick]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .config import RunConfig, quick_config, seed_everything

log = logging.getLogger("hydrointel")


def _config(args) -> RunConfig:
    cfg = RunConfig.load(args.config) if args.config else RunConfig()
    if getattr(args, "quick", False):
        cfg = quick_config(cfg)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device:
        cfg.device = args.device
    if args.outdir:
        cfg.outdir = args.outdir
    if args.precision:
        cfg.solver.precision = args.precision
    return cfg.validate()


def _announce(cfg: RunConfig, command: str) -> None:
    out = Path(cfg.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f" hydrointel {command}   config hash {cfg.config_hash()}   device {cfg.resolved_device()}")
    print("=" * 72)
    print(json.dumps(cfg.to_dict(), indent=1, sort_keys=True))
    cfg.save(out / f"config_{command}.json")


def _eta(label: str, seconds: float | None) -> None:
    """Print an ETA only from timings measured on this machine; otherwise say so."""
    if seconds is None:
        print(f"[ETA] {label}: no earlier timing on record; progress lines report measured ETAs as work proceeds")
    else:
        print(f"[ETA] {label}: ~{seconds / 60:.1f} min (from timings recorded by earlier runs)")


def _recorded(path: Path, reader):
    try:
        return reader(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def cmd_benchmark(cfg, args) -> int:
    from .benchmarks.suite import banner, run_suite
    fast = args.fast or cfg.quick
    _eta("benchmark suite", _recorded(Path(cfg.outdir) / "benchmark_results.json",
                                      lambda d: sum(r["seconds"] for r in d["results"]) if d["fast"] == fast else None))
    res = run_suite(cfg, fast=args.fast or cfg.quick)
    print(banner(res))
    print(f"report: {Path(cfg.outdir) / 'benchmark_report.md'}")
    return 0 if all(r.passed for r in res) else 1


def cmd_generate(cfg, args) -> int:
    from .data.generate import generate
    from .api import dataset_dir
    idx = dataset_dir(cfg) / "index.json"
    _eta(f"dataset generation ({cfg.data.n_sims} sims)",
         _recorded(idx, lambda d: (cfg.data.n_sims - len(d)) * sum(v["wall_s"] for v in d.values()) / len(d)))
    out = generate(cfg, check_benchmarks=not args.skip_benchmark_check)
    print(f"dataset: {out}")
    return 0


def cmd_train(cfg, args) -> int:
    from .train import train
    _eta(f"training ({cfg.train.steps} steps)", None)
    train(cfg, resume=args.resume)
    return 0


def cmd_evaluate(cfg, args) -> int:
    from .evaluate import calibrate, evaluate
    if getattr(args, "calibrate", False):
        calibrate(cfg)
        return 0
    _eta("evaluation", None)
    evaluate(cfg)
    return 0


def cmd_figures(cfg, args) -> int:
    from .viz.report import scenario_figures
    from .api import Scenario
    scen = Scenario(args.rp, args.rcp, args.ssp, args.year, args.tide == "on", args.offset)
    from .api import dataset_dir
    _eta("scenario figures (one engine run + surrogate)",
         _recorded(dataset_dir(cfg) / "index.json", lambda d: sorted(v["wall_s"] for v in d.values())[len(d) // 2]))
    scenario_figures(cfg, scen, engine=args.engine)
    return 0


def _release_gpu() -> None:
    """Stages share one process under `all`; return cached solver/model memory between them."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cmd_all(cfg, args) -> int:
    rc = cmd_benchmark(cfg, args)
    if rc:
        print("benchmarks failed: stopping before dataset generation")
        return rc
    args.skip_benchmark_check = False
    for fn in (cmd_generate, cmd_train, cmd_evaluate):
        fn(cfg, args)
        _release_gpu()
    args.rp, args.rcp, args.ssp, args.year, args.tide, args.offset = 100, "RCP8.5", "SSP5", 2050, "on", 0.0
    args.engine = "both"
    return cmd_figures(cfg, args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hydrointel")
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--config")
    p.add_argument("--outdir")
    p.add_argument("--precision", choices=["fp32", "fp64"])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("benchmark"); b.add_argument("--fast", action="store_true"); b.add_argument("--quick", action="store_true")
    g = sub.add_parser("generate"); g.add_argument("--quick", action="store_true")
    g.add_argument("--skip-benchmark-check", action="store_true", help=argparse.SUPPRESS)
    t = sub.add_parser("train"); t.add_argument("--quick", action="store_true"); t.add_argument("--resume", action="store_true")
    e = sub.add_parser("evaluate"); e.add_argument("--quick", action="store_true"); e.add_argument("--calibrate", action="store_true")
    f = sub.add_parser("figures")
    f.add_argument("--quick", action="store_true")
    f.add_argument("--rp", type=int, default=50)
    f.add_argument("--rcp", default="RCP8.5")
    f.add_argument("--ssp", default="SSP5")
    f.add_argument("--year", type=int, default=2050)
    f.add_argument("--tide", choices=["on", "off"], default="on")
    f.add_argument("--offset", type=float, default=0.0, help="storm-tide offset in hours")
    f.add_argument("--engine", choices=["simulate", "predict", "both"], default="both")
    a = sub.add_parser("all"); a.add_argument("--quick", action="store_true"); a.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("torch", "torch._dynamo", "torch._inductor", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = _config(args)
    seed_everything(cfg.seed)
    _announce(cfg, args.command)
    args.fast = getattr(args, "fast", False)
    args.skip_benchmark_check = getattr(args, "skip_benchmark_check", False)
    args.resume = getattr(args, "resume", False)
    t0 = time.perf_counter()
    rc = {"benchmark": cmd_benchmark, "generate": cmd_generate, "train": cmd_train, "evaluate": cmd_evaluate,
          "figures": cmd_figures, "all": cmd_all}[args.command](cfg, args)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")
    return rc


if __name__ == "__main__":
    sys.exit(main())
