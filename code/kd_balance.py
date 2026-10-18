"""
kd_balance.py — Exploratory follow-up to Phase 4: how much weight the
distillation term gets relative to CE, plus a CE + logit-KD baseline.

The archived Phase-4 runs used γ = 3000, so γ·L_SP was 88–99 % of the loss and
the student was mostly a Gram-matrix matcher. This script trains one ResNet-18
student per process, so that γ can be swept and the BI weighting re-tested in
a regime where CE and SP actually compete.

Conditions (teacher = frozen ResNet-50, student = ResNet-18 from scratch):
  vanilla   L = CE
  sp        L = CE + γ · Σ_s w_s · ||G_T,s − G_S,s||_F² / N²   (as phase4_distillation)
  kd_logit  L = CE + β · T² · KL(softmax(t/T) || softmax(s/T))   (Hinton KD)

Differences from phase4_distillation.run_condition (same optimiser, schedule,
batch size, gradient clipping and stage matching, all read from config.py):
  • CIFAR-10 is held on the GPU and RandomCrop(32, padding=4) + horizontal flip
    run there (same distribution as the torchvision transforms), so two runs
    can share a machine with few CPUs;
  • optional fp16 autocast (--precision amp, the default); Gram matrices and
    all losses are always computed in fp32;
  • gradient probe every --probe-every steps: per-term gradient norms and
    cosines on the student backbone (every parameter except fc);
  • test accuracy every --eval-every epochs; CKA at every matched stage + avgpool;
  • checkpoint/resume and a wall-clock --deadline for 12 h Kaggle sessions;
  • --target-sp-share: one-shot γ rescaling at --adapt-epoch with the exact
    closed form (phase4_distillation's halving loop under-corrects).

One process = one run on one GPU. No DDP/DataParallel: splitting the batch
would change the N×N Gram matrices, i.e. the SP-KD objective itself.
kd_balance_queue.py keeps several GPUs busy with independent runs.

Exit codes: 0 = run complete, 3 = stopped at --deadline (resumable).

Examples (from code/):
  python kd_balance.py --condition sp --weights uniform --gamma 300 --seed 42
  python kd_balance.py --condition kd_logit --seed 42
  python kd_balance.py --condition sp --gamma 3000 --epochs 3 --tag bench
"""

import argparse
import itertools
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from config import (
    CALIB_INDICES, CIFAR10_MEAN, CIFAR10_STD, MODEL_CKPT, PHASE2_RESULTS, ROOT,
    PHASE4_BATCH_SIZE, PHASE4_BETA_KD_SOFT, PHASE4_CKA_INTERVAL, PHASE4_GAMMA_KD,
    PHASE4_GRAD_CLIP, PHASE4_GRAD_CLIP_EPOCHS, PHASE4_KD_SOFT_TEMP, PHASE4_LR,
    PHASE4_LR_DECAY, PHASE4_LR_MILESTONES, PHASE4_MOMENTUM, PHASE4_N_EPOCHS,
    PHASE4_STUDENT_MATCH_IDX, PHASE4_TEACHER_MATCH_IDX, PHASE4_WEIGHT_DECAY,
)
from model import load_model
from phase2_metrics import linear_cka
from phase4_distillation import STAGES, SPHookManager, build_condition_weights, gram_matrix, kd_soft_loss
from student_model import build_student
from utils import gap, relax_determinism_for_training, set_seed

logger = logging.getLogger("kd_balance")

ON_KAGGLE = Path("/kaggle/working").is_dir()
DEFAULT_OUT_DIR = Path("/kaggle/working/kd_balance") if ON_KAGGLE else ROOT.parent / "kd_balance_results"
DEFAULT_DATA_DIR = Path("/tmp/cifar10") if ON_KAGGLE else ROOT.parent / "data"
DEFAULT_TEACHER = Path("/tmp/teacher/pytorch_model.bin") if ON_KAGGLE else MODEL_CKPT
TEACHER_HF_REPO = "edadaltocg/resnet50_cifar10"

# --weights name → key of phase4_distillation.build_condition_weights
WEIGHT_SCHEMES = {"uniform": "uniform", "bi_acc": "bi_acc", "bi_rep": "bi_rep", "delta": "conf_gated"}
EXIT_DEADLINE = 3


# ── CLI and run identity ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, choices=["vanilla", "sp", "kd_logit"])
    p.add_argument("--weights", default="uniform", choices=list(WEIGHT_SCHEMES),
                   help="SP stage weights (sp only); delta = max(BIrep − BIacc, 0)")
    p.add_argument("--stage-weights", default=None, metavar="W1,W2,W3,W4",
                   help="explicit SP stage weights, normalised to sum 1 (overrides --weights)")
    p.add_argument("--weight-power", type=float, default=1.0,
                   help="w_s ← w_s^p then renormalise; p > 1 sharpens the BI weighting")
    p.add_argument("--gamma", type=float, default=PHASE4_GAMMA_KD, help="SP scale γ")
    p.add_argument("--target-sp-share", type=float, default=None,
                   help="at --adapt-epoch, rescale γ so that γ·SP is this fraction of the loss")
    p.add_argument("--adapt-epoch", type=int, default=10)
    p.add_argument("--beta", type=float, default=PHASE4_BETA_KD_SOFT, help="logit-KD weight β (kd_logit only)")
    p.add_argument("--kd-temp", type=float, default=PHASE4_KD_SOFT_TEMP, help="logit-KD temperature T")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=PHASE4_N_EPOCHS)
    p.add_argument("--precision", choices=["amp", "fp32"], default="amp")
    p.add_argument("--eval-every", type=int, default=1, help="test accuracy every N epochs")
    p.add_argument("--cka-every", type=int, default=PHASE4_CKA_INTERVAL, help="teacher–student CKA every N epochs")
    p.add_argument("--probe-every", type=int, default=100, help="gradient probe every N steps (0 = off)")
    p.add_argument("--ckpt-every", type=int, default=5, help="save resumable state every N epochs")
    p.add_argument("--deadline", type=float, default=None, help="unix time: save state and exit (code 3) before it")
    p.add_argument("--max-train-batches", type=int, default=None, help="truncate every epoch (smoke tests only)")
    p.add_argument("--tag", default=None, help="suffix appended to the run name")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--resume-from", type=Path, nargs="*", default=[],
                   help="read-only output roots of earlier sessions to resume from")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--teacher-ckpt", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--phase2-results", type=Path, default=PHASE2_RESULTS)
    p.add_argument("--calib-indices", type=Path, default=CALIB_INDICES)
    p.add_argument("--device", default="cuda")
    return p


def run_name(args: argparse.Namespace) -> str:
    """Directory name that identifies a run's configuration."""
    if args.condition == "vanilla":
        name = "vanilla"
    elif args.condition == "kd_logit":
        name = f"kdlogit_b{args.beta:g}_T{args.kd_temp:g}"
    else:
        w = "custom" + args.stage_weights.replace(",", "-") if args.stage_weights else args.weights
        name = f"sp_{w}_g{args.gamma:g}"
        if args.weight_power != 1.0:
            name += f"_p{args.weight_power:g}"
        if args.target_sp_share is not None:
            name += f"_share{args.target_sp_share:g}"
    name += f"_s{args.seed}"
    if args.precision == "fp32":
        name += "_fp32"
    if args.epochs != PHASE4_N_EPOCHS:
        name += f"_e{args.epochs}"
    if args.tag:
        name += f"_{args.tag}"
    return name


def resolve_stage_weights(args: argparse.Namespace) -> dict[str, float]:
    if args.condition != "sp":
        return {s: 0.0 for s in STAGES}
    if args.stage_weights:
        raw = [float(v) for v in args.stage_weights.split(",")]
        if len(raw) != len(STAGES) or min(raw) < 0 or sum(raw) <= 0:
            raise SystemExit(f"--stage-weights needs {len(STAGES)} non-negative values, not all zero")
        w = dict(zip(STAGES, raw))
    else:
        with open(args.phase2_results) as f:
            w = build_condition_weights(json.load(f))[WEIGHT_SCHEMES[args.weights]]
    w = {s: v ** args.weight_power for s, v in w.items()}
    total = sum(w.values())
    return {s: v / total for s, v in w.items()}


# ── Run directories: progress, resume, atomic writes ──────────────────────────

def run_progress(run_dir: Path) -> tuple[int, int]:
    """(2, epochs) if complete, (1, last saved epoch) if resumable, (0, 0) otherwise."""
    if (run_dir / "results.json").exists():
        return 2, 0
    if (run_dir / "state.pt").exists():
        try:
            return 1, json.loads((run_dir / "status.json").read_text())["epoch"]
        except (OSError, ValueError, KeyError):
            return 1, 0
    return 0, 0


def adopt_previous(name: str, out_dir: Path, resume_from: list[Path]) -> str:
    """
    Copy the most advanced earlier copy of run `name` (from any resume root)
    into out_dir/name. Returns 'done', 'partial' or 'new'.
    """
    run_dir = out_dir / name
    best, best_prog = run_dir, run_progress(run_dir)
    for root in resume_from:
        prog = run_progress(root / name)
        if prog > best_prog:
            best, best_prog = root / name, prog
    if best != run_dir:
        shutil.copytree(best, run_dir, dirs_exist_ok=True)
        if (run_dir / "results.json").exists():
            (run_dir / "state.pt").unlink(missing_ok=True)
    return ["new", "partial", "done"][best_prog[0]]


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _write_status(run_dir: Path, status: str, epoch: int, epochs: int) -> None:
    _write_json({"status": status, "epoch": epoch, "epochs": epochs,
                 "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, run_dir / "status.json")


def _rng_state(gen: torch.Generator, device: torch.device) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device), "aug": gen.get_state()}


def _set_rng_state(state: dict, gen: torch.Generator, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)
    gen.set_state(state["aug"])


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ── Data on the GPU ───────────────────────────────────────────────────────────

def _use_kaggle_cifar(data_dir: Path, inputs: Path = Path("/kaggle/input")) -> None:
    """
    Point data_dir at a CIFAR-10 (python version) dataset attached under /kaggle/input,
    because downloading it from cs.toronto.edu can take hours. Accepts the extracted
    batch folder (linked) or cifar-10-python.tar.gz (copied, then extracted by torchvision).
    """
    if (data_dir / "cifar-10-batches-py").exists() or not inputs.is_dir():
        return
    depths = ["/".join(["*"] * d) for d in range(1, 6)]
    batches = [p.parent for d in depths for p in inputs.glob(f"{d}/data_batch_1")
               if (p.parent / "test_batch").exists()]
    archives = [p for d in depths for p in inputs.glob(f"{d}/cifar-10-python.tar.gz")]
    if batches:
        (data_dir / "cifar-10-batches-py").symlink_to(batches[0], target_is_directory=True)
        logger.info(f"CIFAR-10 from attached dataset {batches[0]}")
    elif archives:
        shutil.copy(archives[0], data_dir / archives[0].name)
        logger.info(f"CIFAR-10 archive from attached dataset {archives[0]}")
    else:
        logger.warning("no CIFAR-10 dataset attached under /kaggle/input — downloading from "
                       "cs.toronto.edu, which can be very slow")


def prepare_assets(data_dir: Path, teacher_ckpt: Path) -> None:
    """Download CIFAR-10 and the teacher if missing. Call once before launching parallel runs."""
    data_dir.mkdir(parents=True, exist_ok=True)
    _use_kaggle_cifar(data_dir)
    datasets.CIFAR10(str(data_dir), train=True, download=True)
    datasets.CIFAR10(str(data_dir), train=False, download=True)
    if not teacher_ckpt.exists():
        from huggingface_hub import hf_hub_download
        teacher_ckpt.parent.mkdir(parents=True, exist_ok=True)
        path = Path(hf_hub_download(TEACHER_HF_REPO, "pytorch_model.bin", local_dir=str(teacher_ckpt.parent)))
        if path.resolve() != teacher_ckpt.resolve():
            shutil.copy(path, teacher_ckpt)


class GPUData:
    """CIFAR-10 as uint8 NCHW tensors on the device, with augmentation done on the GPU."""

    def __init__(self, data_dir: Path, calib_indices: Path, device: torch.device):
        train = datasets.CIFAR10(str(data_dir), train=True, download=False)
        test = datasets.CIFAR10(str(data_dir), train=False, download=False)
        to_nchw = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(device)
        self.device = device
        self.x_train = to_nchw(train.data)
        self.y_train = torch.tensor(train.targets, device=device)
        self.x_test = to_nchw(test.data)
        self.y_test = torch.tensor(test.targets, device=device)
        self.x_calib = self.x_train[torch.load(calib_indices).to(device)]
        self.mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)

    def normalise(self, x_u8: torch.Tensor) -> torch.Tensor:
        """ToTensor() + Normalize() on a uint8 batch."""
        return (x_u8.float().div_(255) - self.mean) / self.std

    def augment(self, x_u8: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Per-sample RandomCrop(32, padding=4) + RandomHorizontalFlip(), then normalise."""
        B, dev = x_u8.size(0), self.device
        x = F.pad(x_u8.float().div_(255), (4, 4, 4, 4))           # zero fill before Normalize, as torchvision
        i = torch.randint(0, 9, (B, 1, 1, 1), device=dev, generator=gen)
        j = torch.randint(0, 9, (B, 1, 1, 1), device=dev, generator=gen)
        ar = torch.arange(32, device=dev)
        x = x[torch.arange(B, device=dev).view(B, 1, 1, 1),
              torch.arange(3, device=dev).view(1, 3, 1, 1),
              i + ar.view(1, 1, 32, 1),
              j + ar.view(1, 1, 1, 32)]
        flip = torch.rand(B, 1, 1, 1, device=dev, generator=gen) < 0.5
        x = torch.where(flip, x.flip(3), x)
        return (x - self.mean) / self.std

    def train_batches(self, batch_size: int, gen: torch.Generator):
        perm = torch.randperm(len(self.y_train), device=self.device, generator=gen)
        for k in range(0, len(perm), batch_size):
            idx = perm[k:k + batch_size]
            yield self.augment(self.x_train[idx], gen), self.y_train[idx]


# ── Losses, probe, evaluation ─────────────────────────────────────────────────

def sp_stage_losses(t_feats: dict, s_feats: dict) -> dict[str, torch.Tensor]:
    """Unweighted ||G_T − G_S||_F² / N² per stage, in fp32 (the per-stage term of phase4 sp_kd_loss).
    Must run outside autocast, otherwise the Gram matmul is cast back to fp16."""
    out = {}
    for s in STAGES:
        diff = gram_matrix(t_feats[s].float()) - gram_matrix(s_feats[s].float())
        n = diff.size(0)
        out[s] = (diff * diff).sum() / (n * n)
    return out


def grad_probe(terms: dict[str, torch.Tensor], params: list, scale: float) -> dict | None:
    """
    Gradient of each loss term w.r.t. `params` → norms, ratios and cosines.
    `scale` is the GradScaler scale (1 in fp32). Each gradient is divided by the
    scale it was computed with, so a term whose fp16 backward overflows is retried
    at smaller scales; returns None if it still overflows.
    """
    flat = {}
    for name, term in terms.items():
        for s in (scale, scale / 2 ** 8, scale / 2 ** 16):
            grads = torch.autograd.grad(term * s, params, retain_graph=True, allow_unused=True)
            v = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                           for g, p in zip(grads, params)]) / s
            if torch.isfinite(v).all():
                break
        else:
            return None
        flat[name] = v

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=0, eps=1e-12).item()

    stats = {f"norm_{k}": v.norm().item() for k, v in flat.items()}
    sp_keys = [k for k in flat if k.startswith("sp_")]
    if sp_keys:
        g_sp = sum(flat[k] for k in sp_keys)
        stats["norm_sp"] = g_sp.norm().item()
        stats["ratio_sp_ce"] = stats["norm_sp"] / (stats["norm_ce"] + 1e-12)
        stats["cos_ce_sp"] = cos(flat["ce"], g_sp)
        total = sum(stats[f"norm_{k}"] for k in sp_keys) + 1e-12
        for k in sp_keys:
            stats[f"share_{k}"] = stats[f"norm_{k}"] / total
        for a, b in itertools.combinations(sp_keys, 2):
            stats[f"cos_{a[3:]}_{b[3:]}"] = cos(flat[a], flat[b])
    if "kd" in flat:
        stats["ratio_kd_ce"] = stats["norm_kd"] / (stats["norm_ce"] + 1e-12)
        stats["cos_ce_kd"] = cos(flat["ce"], flat["kd"])
    return stats


def _mean_dicts(dicts: list[dict]) -> dict | None:
    if not dicts:
        return None
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}


@torch.no_grad()
def evaluate(model, data: GPUData, amp: bool, bs: int = 1000) -> float:
    model.eval()
    correct = torch.zeros((), dtype=torch.long, device=data.device)
    for k in range(0, len(data.y_test), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(data.normalise(data.x_test[k:k + bs]))
        correct += (logits.argmax(1) == data.y_test[k:k + bs]).sum()
    return correct.item() / len(data.y_test)


@torch.no_grad()
def stage_features(model, data: GPUData, match_idx: dict, bs: int = 500) -> dict[str, torch.Tensor]:
    """GAP features (fp32) of the matched block of every stage + avgpool, on the calibration set."""
    model.eval()
    mods = {s: getattr(model, s)[match_idx[s]] for s in STAGES}
    mods["avgpool"] = model.avgpool
    cur, feats = {}, {k: [] for k in mods}
    handles = [m.register_forward_hook(lambda _m, _i, out, k=k: cur.__setitem__(k, gap(out).float()))
               for k, m in mods.items()]
    try:
        for k in range(0, len(data.x_calib), bs):
            model(data.normalise(data.x_calib[k:k + bs]))
            for name in feats:
                feats[name].append(cur[name])
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v) for k, v in feats.items()}


# ── Training ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    name = run_name(args)
    run_dir = args.out_dir / name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if adopt_previous(name, args.out_dir, args.resume_from) == "done":
        logger.info(f"[{name}] already complete → {run_dir}")
        return 0
    run_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    amp = args.precision == "amp"
    is_sp, is_kd = args.condition == "sp", args.condition == "kd_logit"
    weights = resolve_stage_weights(args)

    prepare_assets(args.data_dir, args.teacher_ckpt)
    data = GPUData(args.data_dir, args.calib_indices, device)
    teacher = load_model(args.teacher_ckpt, device)
    teacher_acc = evaluate(teacher, data, amp=False)
    logger.info(f"[{name}] teacher test accuracy {teacher_acc:.4f}")
    if teacher_acc < 0.93:
        logger.warning(f"[{name}] teacher accuracy {teacher_acc:.4f} < 0.93 — check data / normalisation")
    teacher_calib = stage_features(teacher, data, PHASE4_TEACHER_MATCH_IDX)

    # Seed right before building the student, as phase4_distillation.run_condition does
    set_seed(args.seed)
    relax_determinism_for_training()
    student = build_student().to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    optimizer = torch.optim.SGD(student.parameters(), lr=PHASE4_LR, momentum=PHASE4_MOMENTUM,
                                weight_decay=PHASE4_WEIGHT_DECAY, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=PHASE4_LR_MILESTONES,
                                                     gamma=PHASE4_LR_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    hooks = SPHookManager(teacher, student) if is_sp else None
    probe_params = [p for n, p in student.named_parameters() if not n.startswith("fc.")]

    gamma = args.gamma
    history: dict[str, list] = {k: [] for k in (
        "epoch", "lr", "gamma", "ce", "sp", "kd", "sp_share", "kd_share",
        *(f"sp_raw_{s}" for s in STAGES), "test_acc", "epoch_seconds", "probe")}
    cka_curve: list = []
    start_epoch, global_step, sessions, train_seconds = 1, 0, 1, 0.0

    state_path = run_dir / "state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        student.load_state_dict(st["student"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        _set_rng_state(st["rng"], gen, device)
        gamma, history, cka_curve = st["gamma"], st["history"], st["cka_curve"]
        start_epoch, global_step = st["epoch"] + 1, st["global_step"]
        sessions, train_seconds = st["sessions"] + 1, st["train_seconds"]
        logger.info(f"[{name}] resumed after epoch {st['epoch']} (session {sessions})")

    def save_state(epoch_done: int) -> None:
        _atomic_save({
            "run_name": name, "epoch": epoch_done, "global_step": global_step,
            "student": student.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "rng": _rng_state(gen, device), "gamma": gamma, "history": history,
            "cka_curve": cka_curve, "sessions": sessions, "train_seconds": train_seconds,
        }, state_path)
        _write_status(run_dir, "partial", epoch_done, args.epochs)

    setup = f"[{name}] condition={args.condition} precision={args.precision}"
    if is_sp:
        setup += f" γ={gamma:g} weights=(" + ", ".join(f"{s}={w:.4f}" for s, w in weights.items()) + ")"
    if is_kd:
        setup += f" β={args.beta:g} T={args.kd_temp:g}"
    logger.info(setup)

    for epoch in range(start_epoch, args.epochs + 1):
        # Stop before an epoch that would not finish (and be saved) before the deadline
        times = history["epoch_seconds"]
        if args.deadline is not None and times:
            if time.time() + 1.25 * float(np.mean(times[-3:])) + 120 > args.deadline:
                save_state(epoch - 1)
                logger.info(f"[{name}] deadline: saved after epoch {epoch - 1}, exiting")
                return EXIT_DEADLINE

        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        student.train()
        sums = {k: torch.zeros((), device=device) for k in ("ce", "sp", "kd", *STAGES)}
        probes, n_batches = [], 0

        for b, (x, y) in enumerate(data.train_batches(PHASE4_BATCH_SIZE, gen)):
            if args.max_train_batches is not None and b >= args.max_train_batches:
                break
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                if is_sp or is_kd:
                    with torch.no_grad():
                        t_logits = teacher(x)
                s_logits = student(x)

            # Losses in fp32, outside autocast
            terms = {"ce": F.cross_entropy(s_logits.float(), y)}
            if is_sp:
                raw = sp_stage_losses(hooks.teacher_feats, hooks.student_feats)
                for s in STAGES:
                    sums[s] += raw[s].detach()
                    if weights[s] > 0:
                        terms[f"sp_{s}"] = gamma * weights[s] * raw[s]
            elif is_kd:
                terms["kd"] = args.beta * kd_soft_loss(s_logits.float(), t_logits.float(), T=args.kd_temp)
            loss = sum(terms.values())

            if args.probe_every and (is_sp or is_kd) and global_step % args.probe_every == 0:
                stats = grad_probe(terms, probe_params, scaler.get_scale() if amp else 1.0)
                if stats is not None:
                    probes.append(stats)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if epoch <= PHASE4_GRAD_CLIP_EPOCHS:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), PHASE4_GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            sums["ce"] += terms["ce"].detach()
            sums["sp"] += sum((v.detach() for k, v in terms.items() if k.startswith("sp_")),
                              torch.zeros((), device=device))
            if is_kd:
                sums["kd"] += terms["kd"].detach()
            n_batches += 1
            global_step += 1

        scheduler.step()

        acc = evaluate(student, data, amp) if epoch % args.eval_every == 0 or epoch == args.epochs else None
        cka = None
        if epoch % args.cka_every == 0 or epoch == args.epochs:
            student_calib = stage_features(student, data, PHASE4_STUDENT_MATCH_IDX)
            cka = {k: linear_cka(student_calib[k], teacher_calib[k]) for k in teacher_calib}
            cka_curve.append([epoch, cka])

        m = {k: v.item() / n_batches for k, v in sums.items()}
        total = m["ce"] + m["sp"] + m["kd"]
        probe = _mean_dicts(probes)
        epoch_s = time.time() - t0
        train_seconds += epoch_s
        for k, v in (("epoch", epoch), ("lr", lr), ("gamma", gamma), ("ce", m["ce"]), ("sp", m["sp"]),
                     ("kd", m["kd"]), ("sp_share", m["sp"] / total), ("kd_share", m["kd"] / total),
                     *((f"sp_raw_{s}", m[s]) for s in STAGES), ("test_acc", acc),
                     ("epoch_seconds", epoch_s), ("probe", probe)):
            history[k].append(v)

        msg = f"[{name}] ep {epoch:3d}/{args.epochs} | lr {lr:.4g} | CE {m['ce']:.4f}"
        if is_sp:
            msg += f" | γ·SP {m['sp']:.4f} ({m['sp'] / total:.0%})"
        if is_kd:
            msg += f" | β·KD {m['kd']:.4f} ({m['kd'] / total:.0%})"
        if acc is not None:
            msg += f" | acc {acc:.4f}"
        if probe and is_sp:
            msg += f" | ‖∇SP‖/‖∇CE‖ {probe['ratio_sp_ce']:.2f} cos {probe['cos_ce_sp']:+.2f}"
        if probe and is_kd:
            msg += f" | ‖∇KD‖/‖∇CE‖ {probe['ratio_kd_ce']:.2f} cos {probe['cos_ce_kd']:+.2f}"
        if cka:
            msg += f" | CKA {cka['avgpool']:.4f}"
        steady = history["epoch_seconds"][1:] or history["epoch_seconds"]   # epoch 1 includes cuDNN autotuning
        eta_h = float(np.mean(steady[-5:])) * (args.epochs - epoch) / 3600
        msg += f" | {epoch_s:.0f}s | eta {eta_h:.1f}h"
        logger.info(msg)

        if args.target_sp_share is not None and is_sp and epoch == args.adapt_epoch:
            # Solve  γ'·SP_raw / (CE + KD + γ'·SP_raw) = t  for γ'
            t = args.target_sp_share
            new_gamma = t * (m["ce"] + m["kd"]) / ((1 - t) * (m["sp"] / gamma))
            logger.info(f"[{name}] γ rescaled at epoch {epoch}: SP share {m['sp'] / total:.1%} → "
                        f"target {t:.1%}, γ {gamma:g} → {new_gamma:g}")
            gamma = new_gamma

        if epoch % args.ckpt_every == 0 and epoch < args.epochs:
            save_state(epoch)

    if hooks is not None:
        hooks.remove()
    torch.save(student.state_dict(), run_dir / "student.pt")
    results = {
        "metadata": {
            "run_name": name,
            "condition": args.condition,
            "stage_weights": weights,
            "gamma_initial": args.gamma,
            "gamma_final": gamma,
            "beta": args.beta if is_kd else None,
            "kd_temp": args.kd_temp if is_kd else None,
            "seed": args.seed,
            "epochs": args.epochs,
            "precision": args.precision,
            "teacher_test_acc": teacher_acc,
            "train_hours": train_seconds / 3600,
            "sessions": sessions,
            "probe_params": "student backbone (all parameters except fc)",
            "augmentation": "GPU RandomCrop(32, padding=4) + RandomHorizontalFlip",
            "hyperparams": {"lr": PHASE4_LR, "momentum": PHASE4_MOMENTUM, "weight_decay": PHASE4_WEIGHT_DECAY,
                            "batch_size": PHASE4_BATCH_SIZE, "lr_milestones": PHASE4_LR_MILESTONES,
                            "lr_decay": PHASE4_LR_DECAY, "grad_clip": PHASE4_GRAD_CLIP,
                            "grad_clip_epochs": PHASE4_GRAD_CLIP_EPOCHS},
            "env": {"git_commit": _git_commit(), "python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
            "args": {k: (str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, list) else v)
                     for k, v in vars(args).items()},
        },
        "final_accuracy": history["test_acc"][-1],
        "final_cka": cka_curve[-1][1]["avgpool"],
        "final_stage_cka": cka_curve[-1][1],
        "cka_curve": cka_curve,
        "history": history,
    }
    _write_json(results, run_dir / "results.json")
    _write_status(run_dir, "done", args.epochs, args.epochs)
    state_path.unlink(missing_ok=True)
    logger.info(f"[{name}] done: acc {results['final_accuracy']:.4f}, CKA {results['final_cka']:.4f}, "
                f"{train_seconds / 3600:.2f} h → {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
