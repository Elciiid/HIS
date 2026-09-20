#!/usr/bin/env python
"""Run one stage of this project on a Kaggle GPU session.

Nothing needs uploading: the domain, the storms and the interventions are all generated
from the seed, so a session clones the repo and rebuilds everything.

    python kaggle/run.py --stage env          # what hardware did we get?
    python kaggle/run.py --stage verify       # benchmarks + tests, vs the local numbers
    python kaggle/run.py --stage hardware     # seconds per storm, per training step, VRAM
    python kaggle/run.py --stage storage      # what the storage format costs in accuracy
    python kaggle/run.py --stage generate     # paired dataset, as much as the session fits
    python kaggle/run.py --stage train        # paired two-stage training (resumable)
    python kaggle/run.py --stage evaluate     # Gate 1 / Report 2 evaluation
    python kaggle/run.py --stage diagnose     # A1 resolution arms, C5, C3 speed
    python kaggle/run.py --stage k0           # env + verify + hardware + storage

Every stage prints the environment header first, writes under /kaggle/working/artifacts,
and leaves enough on disk to resume: generation checkpoints after each storm, training on
a wall-clock timer (``train.ckpt_seconds``). A session that dies loses minutes, not hours.

Results come back as the notebook's output (always available, no credentials) and, if a
GitHub token is present in Kaggle Secrets, the small text artifacts are also pushed to a
``kaggle-results`` branch. See kaggle/HOWTO.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = os.environ.get("HIS_REPO_URL", "https://github.com/Elciiid/HIS.git")
WORK = Path(os.environ.get("HIS_WORK", "/kaggle/working"))
SECRET_NAME = "GITHUB_TOKEN"          # the Kaggle Secret the private-repo clone and push use
RESULTS_BRANCH = "kaggle-results"
STAGES = ("env", "verify", "hardware", "storage", "precision", "generate", "train", "evaluate",
          "diagnose", "k0")


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
def sh(cmd: list[str], cwd: Path | None = None, check: bool = True, quiet: bool = False) -> str:
    if not quiet:
        print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    out = (p.stdout or "") + (p.stderr or "")
    if not quiet and out.strip():
        print(out.strip()[:4000], flush=True)
    if check and p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{out[-4000:]}")
    return out


def kaggle_secret(name: str = SECRET_NAME) -> str | None:
    """A Kaggle Secret, or None when it is not attached. Never read a token from a literal."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception as e:                                   # not on Kaggle, or secret not attached
        print(f"[secrets] {name} not available ({type(e).__name__}): "
              "private clone and git push are off; results stay as notebook output", flush=True)
        return None


def env_report(verbose: bool = True) -> dict:
    """GPU name, VRAM, driver, torch and CUDA versions, disk and RAM. Printed by every stage,
    because every measurement in the reports has to be attributable to the hardware it ran on."""
    info: dict = {"python": sys.version.split()[0], "platform": platform.platform(),
                  "cpu_count": os.cpu_count(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch
        info.update({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update({"gpu": p.name, "vram_gb": round(p.total_memory / 2 ** 30, 2),
                         "capability": f"{p.major}.{p.minor}", "gpu_count": torch.cuda.device_count(),
                         "bf16_supported": bool(torch.cuda.is_bf16_supported())})
    except ImportError:
        info["torch"] = None
    try:
        info["nvidia_smi"] = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], quiet=True).strip()
    except Exception:
        info["nvidia_smi"] = None
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        pass
    du = shutil.disk_usage(WORK if WORK.exists() else Path("."))
    info["working_free_gb"] = round(du.free / 2 ** 30, 1)
    if verbose:
        print("=" * 78)
        print(" environment")
        print("=" * 78)
        for k, v in info.items():
            print(f"  {k:18s} {v}")
        print("=" * 78, flush=True)
    return info


def ensure_repo(repo_dir: Path, token: str | None) -> Path:
    """Clone or fast-forward the repo. Internet must be ON in the notebook's settings panel,
    or this is where the session fails."""
    url = REPO_URL
    if token:
        url = url.replace("https://", f"https://x-access-token:{token}@")
    if (repo_dir / ".git").exists():
        sh(["git", "remote", "set-url", "origin", url], cwd=repo_dir, quiet=True)
        sh(["git", "fetch", "--depth", "50", "origin"], cwd=repo_dir)
        sh(["git", "checkout", "main"], cwd=repo_dir, check=False)
        sh(["git", "reset", "--hard", "origin/main"], cwd=repo_dir)
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--depth", "50", url, str(repo_dir)])
    sh(["git", "remote", "set-url", "origin", REPO_URL], cwd=repo_dir, quiet=True)   # keep the token out of .git/config
    head = sh(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, quiet=True).strip()
    print(f"[repo] {repo_dir} at {head}", flush=True)
    return repo_dir


def ensure_deps(repo: Path) -> dict:
    """Install only what is missing. Kaggle ships torch, numpy, scipy, matplotlib and pandas;
    versions move, so check rather than assume, and never reinstall torch."""
    want = {"torch": None, "numpy": None, "matplotlib": None, "pytest": "pytest", "yaml": "pyyaml",
            "psutil": "psutil"}
    have, missing = {}, []
    for mod, pkg in want.items():
        try:
            m = __import__(mod)
            have[mod] = getattr(m, "__version__", "?")
        except ImportError:
            have[mod] = None
            if pkg:
                missing.append(pkg)
            else:
                raise RuntimeError(f"{mod} is missing and this runner will not install it "
                                   "(Kaggle images ship it; pick a GPU image)")
    if missing:
        print(f"[deps] installing {missing}", flush=True)
        sh([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    print(f"[deps] {have}", flush=True)
    return have


def outdir(args) -> Path:
    p = Path(args.outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_from_previous(args) -> None:
    """Copy a previous session's artifacts (attached as a notebook-output dataset input, which
    is read-only) into the working directory, so generation and training resume instead of
    starting again. Files already present win, so a partial stage can be re-run safely."""
    src = Path(args.resume_from)
    if not src.exists():
        raise FileNotFoundError(f"--resume-from {src} does not exist. In the notebook, use + Add Input -> "
                                "Notebook Output, then point --resume-from at "
                                "/kaggle/input/<notebook-slug>/artifacts")
    dst = outdir(args)
    n, bytes_ = 0, 0
    t0 = time.perf_counter()
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        q = dst / p.relative_to(src)
        if q.exists():
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
        bytes_ += q.stat().st_size
    print(f"[resume] copied {n} files ({bytes_ / 2 ** 30:.2f} GB) from {src} in "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def cli(repo: Path, args, *cmd: str, log: str | None = None) -> None:
    """Run a hydrointel CLI command in-process-free (own process, so a crash cannot poison
    later stages), with output tee'd to a log under the artifacts directory."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(repo))
    full = [sys.executable, "-u", "-m", "hydrointel.cli", "--outdir", str(outdir(args)), *cmd]
    print("$", " ".join(full), flush=True)
    logp = outdir(args) / (log or (cmd[0] if cmd else "run")) if log else None
    with (open(logp, "a", encoding="utf-8") if logp else open(os.devnull, "w")) as fh:
        p = subprocess.Popen(full, cwd=str(repo), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            if logp:
                fh.write(line)
        p.wait()
    if p.returncode:
        raise RuntimeError(f"stage command failed with code {p.returncode}: {' '.join(cmd)}")


def save_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    print(f"[out] {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_env(repo: Path, args, info: dict) -> None:
    save_json(outdir(args) / "kaggle_env.json", info)


def stage_verify(repo: Path, args, info: dict) -> None:
    """K0.3: the engine's analytical suite in fp64 and the test suite, then a comparison with
    the numbers the (faulty-RAM) local machine produced. This is the cross-check that tells us
    whether to trust anything measured locally."""
    cli(repo, args, "--config", args.config, "benchmark", log="benchmark.log")
    log = outdir(args) / "pytest.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(repo))
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen([sys.executable, "-u", "-m", "pytest", "tests", "-q"], cwd=str(repo), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            fh.write(line)
        p.wait()
    tests_rc = p.returncode
    from hydrointel.kaggle_support import compare_benchmarks, conditioning_probe   # noqa: E402
    # how far the reported metrics move for a one-bit change in the initial condition, on this
    # machine: the scale below which a cross-machine difference means nothing
    cli(repo, args, "--config", args.config, "conditioning-probe", log="conditioning.log")
    sens_path = outdir(args) / "conditioning_probe.json"
    sens = json.loads(sens_path.read_text(encoding="utf-8")) if sens_path.exists() else None
    local = repo / "artifacts" / "benchmark_results.json"
    here = outdir(args) / "benchmark_results.json"
    cmp = compare_benchmarks(local, here, info, sensitivity=sens)
    save_json(outdir(args) / "verify_comparison.json", {**cmp, "pytest_returncode": tests_rc})
    (outdir(args) / "verify_comparison.md").write_text(cmp["markdown"], encoding="utf-8")
    print(cmp["markdown"], flush=True)
    if tests_rc:
        raise RuntimeError(f"the test suite failed on this machine (pytest exit {tests_rc}): see pytest.log")


def stage_hardware(repo: Path, args, info: dict) -> None:
    """K0.4: what this GPU does with this code."""
    cli(repo, args, "--config", args.config, "kaggle-hardware", "--storms", str(args.storms),
        log="hardware.log")


def stage_storage(repo: Path, args, info: dict) -> None:
    """K0.2's verification: what the storage format costs, measured on one storm."""
    cli(repo, args, "--config", args.config, "storage-check", log="storage_check.log")


def stage_precision(repo: Path, args, info: dict) -> None:
    """Task B: float32 against float64 on the fixed engine, judged by the agreement criteria
    in hydrointel/criteria.py. Six full-resolution storms, so it is its own session."""
    cli(repo, args, "--config", args.config, "precision-study", "--v2", log="precision_v2.log")


def n_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def parallel_cli(repo: Path, args, *cmd: str, log: str) -> None:
    """Run one worker per GPU over the same dataset directory, each pinned to its own device
    and taking every n-th storm. Kaggle's GPU offering is two T4s, and generation is one storm
    at a time, so this halves the wall time. With one GPU it is the plain single run."""
    n = max(1, min(n_gpus(), args.max_workers))
    if n == 1:
        cli(repo, args, *cmd, log=log)
        return
    env_base = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(repo))
    procs = []
    for i in range(n):
        full = [sys.executable, "-u", "-m", "hydrointel.cli", "--outdir", str(outdir(args)), *cmd,
                "--shard", f"{i}/{n}"]
        env = dict(env_base, CUDA_VISIBLE_DEVICES=str(i))
        fh = open(outdir(args) / f"{log}.gpu{i}", "a", encoding="utf-8")
        print(f"$ CUDA_VISIBLE_DEVICES={i} " + " ".join(full), flush=True)
        procs.append((subprocess.Popen(full, cwd=str(repo), env=env, stdout=fh, stderr=subprocess.STDOUT,
                                       text=True), fh, i))
    print(f"[parallel] {n} workers started; per-worker logs are {log}.gpu0 .. {log}.gpu{n - 1}", flush=True)
    codes = []
    for p, fh, i in procs:
        codes.append(p.wait())
        fh.close()
        print(f"[parallel] worker {i} exited with {codes[-1]}", flush=True)
        tail = (outdir(args) / f"{log}.gpu{i}").read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        print("\n".join(tail), flush=True)
    if any(codes):
        raise RuntimeError(f"generation workers failed with codes {codes}: see {log}.gpu*")


def stage_generate(repo: Path, args, info: dict) -> None:
    """K1: a paired dataset, as much of it as this session's budget allows. Storms first, then
    their baselines, each across every GPU the session has; the merge writes the envelope and
    the dataset card, which need the whole index."""
    parallel_cli(repo, args, "--config", args.config, "generate", "--budget-hours", str(args.budget_hours),
                 log="generate.log")
    cli(repo, args, "--config", args.config, "merge-shards", log="generate.log")
    parallel_cli(repo, args, "--config", args.config, "generate-baselines",
                 "--budget-hours", str(args.budget_hours), log="baselines.log")
    cli(repo, args, "--config", args.config, "merge-shards", log="generate.log")


def stage_train(repo: Path, args, info: dict) -> None:
    cli(repo, args, "--config", args.config, "train", "--resume", log="train.log")


def stage_evaluate(repo: Path, args, info: dict) -> None:
    cli(repo, args, "--config", args.config, "evaluate", log="evaluate.log")


def stage_diagnose(repo: Path, args, info: dict) -> None:
    cli(repo, args, "--config", args.config, "diagnose", "--part", args.part, log="diagnose.log")


def stage_k0(repo: Path, args, info: dict) -> None:
    stage_env(repo, args, info)
    stage_verify(repo, args, info)
    stage_hardware(repo, args, info)
    stage_storage(repo, args, info)


# ---------------------------------------------------------------------------
def push_results(repo: Path, args, token: str | None, info: dict) -> None:
    """Copy the small text artifacts onto a results branch and push. Datasets and checkpoints
    are NOT pushed: they belong in the notebook's output, which is where the next session
    picks them up as a dataset input."""
    if not token:
        print("[push] no token: results are the notebook output only", flush=True)
        return
    small = [p for p in outdir(args).rglob("*")
             if p.is_file() and p.suffix in (".md", ".json", ".csv", ".log") and p.stat().st_size < 8 * 2 ** 20]
    if not small:
        return
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    dest_root = repo / "kaggle_results" / f"{args.stage}-{stamp}"
    for p in small:
        d = dest_root / p.relative_to(outdir(args))
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d)
    url = REPO_URL.replace("https://", f"https://x-access-token:{token}@")
    sh(["git", "config", "user.email", "kaggle@localhost"], cwd=repo, quiet=True)
    sh(["git", "config", "user.name", "Kaggle runner"], cwd=repo, quiet=True)
    sh(["git", "checkout", "-B", RESULTS_BRANCH], cwd=repo)
    sh(["git", "add", "kaggle_results"], cwd=repo)
    gpu = info.get("gpu", "unknown GPU")
    sh(["git", "commit", "-m", f"Kaggle {args.stage} results ({gpu}, {stamp})"], cwd=repo, check=False)
    sh(["git", "push", "--force-with-lease", url, f"HEAD:{RESULTS_BRANCH}"], cwd=repo)
    print(f"[push] {len(small)} files on branch {RESULTS_BRANCH}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="run one stage of HIS on a Kaggle GPU session")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--repo-dir", default=str(WORK / "HIS"))
    ap.add_argument("--outdir", default=str(WORK / "artifacts"))
    ap.add_argument("--config", default="kaggle/configs/kaggle.json")
    ap.add_argument("--budget-hours", type=float, default=10.5,
                    help="wall-clock budget for generation; it stops cleanly before this")
    ap.add_argument("--storms", type=int, default=1, help="storms to time in the hardware stage")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="most generation workers to run at once (one per GPU)")
    ap.add_argument("--part", default="all", help="which diagnose part to run")
    ap.add_argument("--resume-from", help="a previous session's artifacts directory, usually "
                                         "/kaggle/input/<notebook-slug>/artifacts, copied in before the stage runs")
    ap.add_argument("--no-pull", action="store_true", help="use the repo as it is on disk")
    ap.add_argument("--no-push", action="store_true", help="never push results to the results branch")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    info = env_report()
    token = kaggle_secret()
    repo = Path(args.repo_dir)
    if not args.no_pull:
        repo = ensure_repo(repo, token)
    if not (repo / "hydrointel").is_dir():
        raise RuntimeError(f"{repo} does not look like the HIS repo (no hydrointel/)")
    sys.path.insert(0, str(repo))
    ensure_deps(repo)
    info["repo_head"] = sh(["git", "rev-parse", "HEAD"], cwd=repo, quiet=True).strip()
    if args.resume_from:
        seed_from_previous(args)
    save_json(outdir(args) / "kaggle_env.json", info)
    print(f"[stage] {args.stage} starting; artifacts -> {outdir(args)}", flush=True)
    try:
        globals()[f"stage_{args.stage}"](repo, args, info)
    finally:
        mins = (time.perf_counter() - t0) / 60
        print(f"[stage] {args.stage} ended after {mins:.1f} min", flush=True)
        if not args.no_push:
            try:
                push_results(repo, args, token, info)
            except Exception as e:
                print(f"[push] failed ({e}); the notebook output still holds everything", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
