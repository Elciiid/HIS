"""Command line interface.

    python -m hydrointel.cli benchmark [--fast]
    python -m hydrointel.cli precision-study
    python -m hydrointel.cli batch-study
    python -m hydrointel.cli grouping-study
    python -m hydrointel.cli predict-speed
    python -m hydrointel.cli cost
    python -m hydrointel.cli step-demand
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


def cmd_precision_study(cfg, args) -> int:
    from .benchmarks.precision import run_study, run_study_v2
    if getattr(args, "v2", False):
        d = run_study_v2(cfg)
        print(f"report: {Path(cfg.outdir) / 'precision_study_v2.md'}")
        print(f"precision study v2: {'PASS' if d['passed'] else 'FAIL'}")
        return 0
    _eta("precision study (3 storms x 2 precisions at full resolution)", None)
    d = run_study(cfg)
    print(f"report: {Path(cfg.outdir) / 'precision_study.md'}")
    print(f"precision study: {'PASS' if d['passed'] else 'FAIL'}")
    return 0 if d["passed"] else 1


def cmd_batch_study(cfg, args) -> int:
    from .benchmarks.throughput import run_study
    _eta("batch throughput study", None)
    d = run_study(cfg, stage=args.stage, batch=args.batch, full_storms=args.full_storms,
                  n_sequential=args.sequential, full_t_end_h=args.full_hours)
    print(f"report: {Path(cfg.outdir) / 'batch_throughput.md'}")
    if d.get("full_length"):
        print(f"batch {d['chosen_batch']}: {d['full_length']['s_per_storm']:.0f} s per storm over "
              f"{d['full_length']['t_end_s'] / 3600:.2f} h of storm")
    return 0


def cmd_grouping_study(cfg, args) -> int:
    from .benchmarks.grouping import run_study
    _eta("grouped-batching study", None)
    d = run_study(cfg)
    print(f"report: {Path(cfg.outdir) / 'grouping_study.md'}")
    print(f"grouped batching: speedup {d['speedup']:.2f}x, accuracy {'met' if d['accuracy_met'] else 'NOT met'} "
          f"-> {'USE' if d['use_grouped_batching'] else 'one storm at a time'}")
    return 0


def cmd_predict_speed(cfg, args) -> int:
    from .benchmarks.predict_speed import run_study
    _eta("surrogate prediction cost", None)
    d = run_study(cfg)
    print(f"report: {Path(cfg.outdir) / 'predict_speed.md'}  (weights: {d['weights']})")
    return 0


def cmd_cost(cfg, args) -> int:
    from .benchmarks.cost import write
    print(f"report: {write(cfg)}")
    return 0


def cmd_step_demand(cfg, args) -> int:
    from .benchmarks.step_demand import write
    print(f"report: {write(cfg)}")
    return 0


def cmd_kaggle_hardware(cfg, args) -> int:
    from .kaggle_support import hardware_study
    d = hardware_study(cfg, storms=args.storms)
    print(f"report: {Path(cfg.outdir) / 'kaggle_hardware.md'}")
    print(f"{d['info']['gpu']}: {d['s_per_storm_fp64']:.0f} s per storm in fp64, "
          f"{d['s_per_storm_fp32']:.0f} s in fp32 ({d['fp32_speedup']:.2f}x)")
    if d.get("full_resolution_fits") is not None:
        print("full 20 m training " + ("FITS" if d["full_resolution_fits"] else "does NOT fit") + " on this GPU")
    return 0


def cmd_storage_check(cfg, args) -> int:
    from .kaggle_support import storage_check
    d = storage_check(cfg, sim_id=args.sim)
    print(f"report: {Path(cfg.outdir) / 'storage_check.md'}")
    m = d["metrics"]
    print(f"stored record is {100 * d['size_ratio']:.0f}% of full fidelity; depth RMSE "
          f"{m['depth_rmse_wet_m']:.2e} m, peak field {m['peak_field_max_abs_m']:.2e} m, "
          f"identical={m['identical']}")
    return 0


def cmd_diagnose(cfg, args) -> int:
    from . import diagnostics as D
    for part in (["a1", "a3", "a4", "a5"] if args.part == "all" else [args.part]):
        getattr(D, f"run_{part}")(cfg)
        _release_gpu()
    print(f"report: {D.report(cfg)}")
    return 0


def cmd_generate_baselines(cfg, args) -> int:
    from .data.paired import generate_baselines
    print(f"index: {generate_baselines(cfg, limit=args.limit, budget_hours=args.budget_hours)}")
    return 0


def cmd_generate(cfg, args) -> int:
    from .data.generate import generate
    from .api import dataset_dir
    idx = dataset_dir(cfg) / "index.json"
    _eta(f"dataset generation ({cfg.data.n_sims} sims)",
         _recorded(idx, lambda d: (cfg.data.n_sims - len(d)) * sum(v["wall_s"] for v in d.values()) / len(d)))
    out = generate(cfg, check_benchmarks=not args.skip_benchmark_check, budget_hours=args.budget_hours)
    print(f"dataset: {out}")
    return 0


def cmd_train(cfg, args) -> int:
    _eta(f"training ({cfg.train.steps} steps)", None)
    if cfg.train.paired:
        from .train_paired import train_paired
        train_paired(cfg, resume=args.resume)
    else:
        from .train import train
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
    ps = sub.add_parser("precision-study"); ps.add_argument("--quick", action="store_true")
    ps.add_argument("--v2", action="store_true", help="agreement criteria of hydrointel.criteria, fixed engine")
    sub.add_parser("grouping-study")
    sub.add_parser("predict-speed")
    sub.add_parser("cost")
    sub.add_parser("step-demand")
    kh = sub.add_parser("kaggle-hardware"); kh.add_argument("--storms", type=int, default=1)
    kh.add_argument("--quick", action="store_true")
    sc = sub.add_parser("storage-check"); sc.add_argument("--sim", type=int, default=0)
    sc.add_argument("--quick", action="store_true")
    gb = sub.add_parser("generate-baselines"); gb.add_argument("--limit", type=int)
    gb.add_argument("--budget-hours", type=float, help="stop cleanly before this much wall time")
    dg = sub.add_parser("diagnose"); dg.add_argument("--part", choices=["a1", "a3", "a4", "a5", "c5", "c3_speed", "all"], default="all")
    bs = sub.add_parser("batch-study")
    bs.add_argument("--quick", action="store_true")
    bs.add_argument("--stage", choices=["all", "scaling", "full", "profile"], default="all")
    bs.add_argument("--batch", type=int, help="batch size for the full-length stage (required with --stage full)")
    bs.add_argument("--full-storms", type=int, help="storms in the full-length stage (default: the batch size)")
    bs.add_argument("--sequential", type=int, default=4, help="storms also run alone, for the accuracy comparison")
    bs.add_argument("--full-hours", type=float, help="truncate the full-length stage to this many simulated hours")
    g = sub.add_parser("generate"); g.add_argument("--quick", action="store_true")
    g.add_argument("--skip-benchmark-check", action="store_true", help=argparse.SUPPRESS)
    g.add_argument("--budget-hours", type=float, help="stop cleanly before this much wall time")
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
    for name, default in (("stage", "all"), ("batch", None), ("full_storms", None), ("sequential", 4),
                          ("full_hours", None), ("budget_hours", None), ("storms", 1), ("sim", 0), ("limit", None)):
        setattr(args, name, getattr(args, name, default))
    t0 = time.perf_counter()
    rc = {"benchmark": cmd_benchmark, "precision-study": cmd_precision_study,
          "batch-study": cmd_batch_study, "grouping-study": cmd_grouping_study,
          "predict-speed": cmd_predict_speed, "cost": cmd_cost, "step-demand": cmd_step_demand, "diagnose": cmd_diagnose, "kaggle-hardware": cmd_kaggle_hardware,
          "storage-check": cmd_storage_check, "generate-baselines": cmd_generate_baselines, "generate": cmd_generate,
          "train": cmd_train, "evaluate": cmd_evaluate, "figures": cmd_figures, "all": cmd_all}[args.command](cfg, args)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")
    return rc


if __name__ == "__main__":
    sys.exit(main())
