"""
kd_balance_queue.py — run a set of kd_balance.py runs, one process per GPU.

On a Kaggle "GPU T4 x2" machine this keeps both GPUs busy with independent
runs. Every run gets the same wall-clock deadline (--time-budget-hours from
launch): before it, each run saves its state and exits, and the next session
resumes it. At start-up, completed and partial runs found under --resume-from
(on Kaggle: every */kd_balance directory under /kaggle/input) are copied into
--out-dir, so each session's output contains everything done so far.

Experiment sets (positional, any combination):
  bench      3 epochs each — SP γ=3000 (amp and fp32), logit KD, vanilla — to measure s/epoch
  sweep      SP uniform weights, γ ∈ {30, 100, 300, 1000, 3000}, seed 42
  baselines  CE + logit KD (β=1, T=4) × --seeds, plus vanilla seed 42 as same-setup anchor
  compare    SP uniform vs bi_acc at --gamma (chosen from the sweep) × --seeds
Extra runs: --run "--condition sp --weights bi_rep --gamma 300 --seed 42" (repeatable).
--shard I/N keeps every N-th run starting at the I-th (1-based), so N accounts
can split one queue: run the same command with I = 1..N, one per account, and
always the same command on the same account (it resumes only its own runs).
Options for every run: --extra-args "--max-train-batches 20 --precision fp32".

Kaggle notebook (Accelerator "GPU T4 x2", Internet on, "Save & Run All"):
  !git clone -b kd-balance --depth 1 https://github.com/0sake/ABNS_Project.git /tmp/ABNS_Project
  !cd /tmp/ABNS_Project/code && python kd_balance_queue.py sweep baselines --shard 1/3
To continue in a new session, add this notebook's own output as input (found
automatically under /kaggle/input) and run the same command again.
Results table of everything in the output directory:
  python kd_balance_queue.py --summary --out-dir <dir with the run folders>
"""

import argparse
import csv
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

import kd_balance as kb

SETS = ("bench", "sweep", "baselines", "compare")
SWEEP_GAMMAS = (30, 100, 300, 1000, 3000)
DEFAULT_SEEDS = [42, 123, 456]

_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def set_runs(name: str, args: argparse.Namespace) -> list[list[str]]:
    seeds = [str(s) for s in args.seeds]
    if name == "bench":
        common = ["--epochs", "3", "--tag", "bench"]
        return [["--condition", "sp", "--gamma", "3000", *common],
                ["--condition", "sp", "--gamma", "3000", "--precision", "fp32", *common],
                ["--condition", "kd_logit", *common],
                ["--condition", "vanilla", *common]]
    if name == "sweep":
        return [["--condition", "sp", "--weights", "uniform", "--gamma", str(g), "--seed", "42"]
                for g in SWEEP_GAMMAS]
    if name == "baselines":
        return ([["--condition", "kd_logit", "--seed", s] for s in seeds]
                + [["--condition", "vanilla", "--seed", "42"]])
    if name == "compare":
        if args.gamma is None:
            raise SystemExit("'compare' needs --gamma (pick it from the sweep)")
        return [["--condition", "sp", "--weights", w, "--gamma", f"{args.gamma:g}", "--seed", s]
                for w in ("uniform", "bi_acc") for s in seeds]
    raise SystemExit(f"unknown experiment set '{name}' (choose from {', '.join(SETS)})")


def discover_resume_roots() -> list[Path]:
    """Earlier sessions' outputs attached as Kaggle inputs (any depth up to 4)."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return []
    patterns = ("*/kd_balance", "*/*/kd_balance", "*/*/*/kd_balance", "*/*/*/*/kd_balance")
    return sorted({p for pat in patterns for p in base.glob(pat) if p.is_dir()})


def sync_previous(out_dir: Path, roots: list[Path]) -> None:
    names = sorted({d.name for root in roots for d in root.iterdir() if d.is_dir()})
    for name in names:
        before = kb.run_progress(out_dir / name)
        status = kb.adopt_previous(name, out_dir, roots)
        if kb.run_progress(out_dir / name) != before:
            say(f"adopted {name} ({status}) from an earlier session")


def worker(gpu: int, jobs: queue.Queue, deadline: float, args: argparse.Namespace, report: dict) -> None:
    script = Path(kb.__file__).resolve()
    while True:
        try:
            name, argv = jobs.get_nowait()
        except queue.Empty:
            return
        if deadline - time.time() < args.min_start_minutes * 60:
            report[name] = "not started (session budget exhausted)"
            continue
        run_dir = args.out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), *argv, "--deadline", f"{deadline:.0f}"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        say(f"[gpu{gpu}] start {name}")
        t0 = time.time()
        with open(run_dir / "train.log", "a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu{gpu}: {shlex.join(cmd)}\n")
            log.flush()
            # Tee: train.log keeps the full record, stdout shows progress live (Kaggle "Logs" tab)
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    cwd=script.parent, text=True, bufsize=1)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                with _print_lock:
                    print(f"[gpu{gpu}] {line}", end="", flush=True)
            rc = proc.wait()
        status = {0: "done", kb.EXIT_DEADLINE: "partial (deadline)"}.get(rc, f"FAILED, exit {rc} — see {run_dir / 'train.log'}")
        report[name] = status
        say(f"[gpu{gpu}] {name} → {status} after {(time.time() - t0) / 3600:.2f} h")


def summarize(out_dir: Path) -> None:
    rows = []
    run_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir()) if out_dir.is_dir() else []
    for d in run_dirs:
        res = d / "results.json"
        if res.exists():
            r = json.loads(res.read_text())
            h = r["history"]
            probes = [p for p in h["probe"] if p]
            key = "ratio_sp_ce" if r["metadata"]["condition"] == "sp" else "ratio_kd_ce"
            ratio = lambda p: f"{p[key]:.2f}" if p and key in p else ""
            share = h["sp_share"][-1] + h["kd_share"][-1]
            rows.append({"run": d.name, "status": "done", "epochs": r["metadata"]["epochs"],
                         "acc": f"{r['final_accuracy']:.4f}", "cka": f"{r['final_cka']:.4f}",
                         "grad_ratio_first": ratio(probes[0] if probes else None),
                         "grad_ratio_last": ratio(probes[-1] if probes else None),
                         "distill_loss_share_last": f"{share:.0%}",
                         "hours": f"{r['metadata']['train_hours']:.2f}"})
        elif (d / "status.json").exists():
            st = json.loads((d / "status.json").read_text())
            rows.append({"run": d.name, "status": f"partial {st['epoch']}/{st['epochs']}"})
    if not rows:
        print(f"no runs in {out_dir}")
        return
    cols = ["run", "status", "epochs", "acc", "cka", "grad_ratio_first", "grad_ratio_last",
            "distill_loss_share_last", "hours"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n→ {out_dir / 'summary.csv'}")


def main() -> int:
    start = time.time()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sets", nargs="*", help=f"experiment sets: {', '.join(SETS)}")
    p.add_argument("--run", action="append", default=[], metavar="ARGS",
                   help="an extra run, as quoted kd_balance.py arguments (repeatable)")
    p.add_argument("--gamma", type=float, default=None, help="γ for the 'compare' set")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU indices (default: all visible)")
    p.add_argument("--time-budget-hours", type=float, default=11.0,
                   help="session budget from launch; Kaggle kills sessions at 12 h")
    p.add_argument("--min-start-minutes", type=float, default=20,
                   help="do not start a run with less budget than this left")
    p.add_argument("--out-dir", type=Path, default=kb.DEFAULT_OUT_DIR)
    p.add_argument("--resume-from", type=Path, nargs="*", default=None,
                   help="earlier output roots (default on Kaggle: every */kd_balance under /kaggle/input)")
    p.add_argument("--data-dir", type=Path, default=kb.DEFAULT_DATA_DIR)
    p.add_argument("--teacher-ckpt", type=Path, default=kb.DEFAULT_TEACHER)
    p.add_argument("--list", action="store_true", help="print the queue and exit")
    p.add_argument("--summary", action="store_true", help="print the results table of --out-dir and exit")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="keep every N-th run starting at the I-th (1-based): one shard per account")
    p.add_argument("--extra-args", default="", metavar="ARGS",
                   help="quoted kd_balance.py arguments appended to every run")
    args = p.parse_args()

    if args.summary:
        summarize(args.out_dir)
        return 0

    roots = args.resume_from if args.resume_from is not None else discover_resume_roots()
    roots = [r for r in roots if r.resolve() != args.out_dir.resolve()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if roots:
        say("resume roots: " + ", ".join(map(str, roots)))
        sync_previous(args.out_dir, roots)

    common = ["--out-dir", str(args.out_dir), "--data-dir", str(args.data_dir),
              "--teacher-ckpt", str(args.teacher_ckpt), *shlex.split(args.extra_args)]
    runs: dict[str, list[str]] = {}
    for argv in [a for s in args.sets for a in set_runs(s, args)] + [shlex.split(r) for r in args.run]:
        full = argv + common
        runs.setdefault(kb.run_name(kb.build_parser().parse_args(full)), full)   # also validates the arguments
    if args.shard:
        try:
            i, n = (int(v) for v in args.shard.split("/"))
            assert 1 <= i <= n
        except (ValueError, AssertionError):
            raise SystemExit(f"--shard must look like 2/3, got '{args.shard}'")
        runs = {k: runs[k] for k in list(runs)[i - 1::n]}

    pending = []
    for name, argv in runs.items():
        prog = kb.run_progress(args.out_dir / name)
        if prog[0] == 2:
            say(f"skip {name} (complete)")
        else:
            pending.append((prog, name, argv))
    pending.sort(key=lambda t: t[0], reverse=True)       # resumable runs first, most advanced first

    if args.list or not pending:
        for prog, name, _ in pending:
            print(f"  {name}  ({'partial, epoch ' + str(prog[1]) if prog[0] == 1 else 'new'})")
        if not pending:
            say("nothing to run")
        return 0

    gpus = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
    if not gpus:
        raise SystemExit("no CUDA device visible")
    kb.prepare_assets(args.data_dir, args.teacher_ckpt)   # once, before runs start in parallel

    deadline = start + args.time_budget_hours * 3600
    say(f"{len(pending)} runs on GPUs {gpus}; deadline {time.strftime('%H:%M', time.localtime(deadline))}")
    jobs: queue.Queue = queue.Queue()
    for _, name, argv in pending:
        jobs.put((name, argv))
    report: dict[str, str] = {}
    threads = [threading.Thread(target=worker, args=(g, jobs, deadline, args, report)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print()
    for _, name, _ in pending:
        print(f"  {name}: {report.get(name, '?')}")
    print()
    summarize(args.out_dir)
    return 0 if all(not s.startswith("FAILED") for s in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
