"""
phase4_distillation.py — Phase 4: Weighted Similarity-Preserving KD.

Four training conditions (teacher = ResNet-50 intact, student = ResNet-18):
  vanilla   — cross-entropy only (student baseline, no KD)
  uniform   — SP-KD, equal stage weights (1/4 per stage)
  bi_acc    — SP-KD, stage weights ∝ mean BIacc per stage (from phase2)
  bi_rep    — SP-KD, stage weights ∝ mean BIrep per stage (from phase2)

SP-KD loss (Tung & Mori, 2019): for each matched stage l,
  G[i,j] = f_i · f_j / (||f_i|| ||f_j||)   (N×N normalised Gram matrix)
  L_SP = Σ_l  w_l · ||G_T_l − G_S_l||_F² / N²

Total loss: L = L_CE + γ · L_SP

Output: results/phase4_results.json
"""

import json
import logging
import time
from pathlib import Path
from utils import relax_determinism_for_training

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    CIFAR10_MEAN, CIFAR10_STD, NUM_WORKERS,
    PHASE2_RESULTS, PHASE4_RESULTS, PHASE4_CKPTS_DIR,
    PHASE4_LR, PHASE4_MOMENTUM, PHASE4_WEIGHT_DECAY,
    PHASE4_N_EPOCHS, PHASE4_BATCH_SIZE, CKPT_DIR,
    PHASE4_LR_MILESTONES, PHASE4_LR_DECAY,
    PHASE4_GAMMA_KD, PHASE4_GRAD_CLIP, PHASE4_GRAD_CLIP_EPOCHS,
    PHASE4_WEIGHT_FLOOR, PHASE4_SEEDS, PHASE4_CKA_INTERVAL,
    PHASE4_TEACHER_MATCH_IDX, PHASE4_STUDENT_MATCH_IDX,
)
from model import ResNet50_CIFAR
from student_model import ResNet18_CIFAR, build_student
from phase2_metrics import linear_cka
from utils import set_seed, worker_init_fn, get_dataloader_generator, gap

logger = logging.getLogger(__name__)

# ── Stage groupings (mirror config.TARGET_BLOCKS ordering) ────────────────────
STAGES = ["layer1", "layer2", "layer3", "layer4"]

STAGE_BLOCKS = {
    "layer1": ["layer1.0", "layer1.1", "layer1.2"],
    "layer2": ["layer2.0", "layer2.1", "layer2.2", "layer2.3"],
    "layer3": ["layer3.0", "layer3.1", "layer3.2", "layer3.3", "layer3.4", "layer3.5"],
    "layer4": ["layer4.0", "layer4.1", "layer4.2"],
}


# ── Stage weight computation ───────────────────────────────────────────────────

def _stage_mean(metric_dict: dict, stage: str) -> float:
    vals = [metric_dict[b] for b in STAGE_BLOCKS[stage] if b in metric_dict]
    return sum(vals) / len(vals) if vals else 0.0


def compute_stage_weights(metric_dict: dict, floor: float = PHASE4_WEIGHT_FLOOR) -> dict[str, float]:
    """
    Normalised stage weights from a per-block metric dict.

    Algorithm:
      1. w_l = mean(metric over blocks in stage l)
      2. w_l = max(w_l, floor)           ← prevents zero-weight stages
      3. normalise so Σ w_l = 1
    """
    raw = {s: max(_stage_mean(metric_dict, s), floor) for s in STAGES}
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()}


def build_condition_weights(phase2_results: dict) -> dict[str, dict[str, float]]:
    """
    Return stage-weight dicts for all four conditions.
    Keys: 'vanilla', 'uniform', 'bi_acc', 'bi_rep'
    """
    return {
        "vanilla": {s: 0.0 for s in STAGES},
        "uniform": {s: 1.0 / len(STAGES) for s in STAGES},
        "bi_acc":  compute_stage_weights(phase2_results["bi_acc"]),
        "bi_rep":  compute_stage_weights(phase2_results["bi_rep"]),
    }


# ── SP-KD loss ─────────────────────────────────────────────────────────────────

def gram_matrix(F: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    N×N normalised Gram (pairwise cosine similarity) matrix.
    F: (N, C, H, W) or (N, D)  →  G: (N, N)
    Gradients flow through F if F requires grad.
    """
    N = F.size(0)
    F_flat = F.view(N, -1)                              # (N, D)
    norm = F_flat.norm(dim=1, keepdim=True).clamp(eps)  # (N, 1)
    F_norm = F_flat / norm                              # (N, D) unit vectors
    return F_norm @ F_norm.t()                          # (N, N)


def sp_kd_loss(
    teacher_feats: dict[str, torch.Tensor],
    student_feats: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    """
    Weighted SP-KD loss: Σ_l w_l · ||G_T_l − G_S_l||_F² / N²

    teacher_feats values are detached (no grad).
    student_feats values retain their autograd graph so gradients flow back.
    """
    device = next(iter(student_feats.values())).device
    loss = torch.zeros(1, device=device)
    N = next(iter(student_feats.values())).size(0)

    for stage in STAGES:
        w = weights.get(stage, 0.0)
        if w == 0.0:
            continue
        G_T = gram_matrix(teacher_feats[stage])   # (N,N) detached constant
        G_S = gram_matrix(student_feats[stage])   # (N,N) with grad
        diff = G_T - G_S
        loss = loss + w * (diff * diff).sum() / (N * N)

    return loss


# ── Hook manager ───────────────────────────────────────────────────────────────

class SPHookManager:
    """
    Attaches output hooks to the last block of each stage in both teacher
    and student (indices from PHASE4_{TEACHER,STUDENT}_MATCH_IDX).

    - Teacher hooks store detached tensors (no gradient accumulation).
    - Student hooks store tensors as-is so gradients can flow through them.
    """

    def __init__(self, teacher: ResNet50_CIFAR, student: ResNet18_CIFAR):
        self._t_feats: dict[str, torch.Tensor] = {}
        self._s_feats: dict[str, torch.Tensor] = {}
        self._handles: list = []

        for stage in STAGES:
            t_mod = getattr(teacher, stage)[PHASE4_TEACHER_MATCH_IDX[stage]]
            s_mod = getattr(student, stage)[PHASE4_STUDENT_MATCH_IDX[stage]]
            self._handles.append(t_mod.register_forward_hook(self._make_t_hook(stage)))
            self._handles.append(s_mod.register_forward_hook(self._make_s_hook(stage)))

    def _make_t_hook(self, name: str):
        def hook(m, inp, out):
            self._t_feats[name] = out.detach()
        return hook

    def _make_s_hook(self, name: str):
        def hook(m, inp, out):
            self._s_feats[name] = out  # keep in autograd graph
        return hook

    @property
    def teacher_feats(self) -> dict[str, torch.Tensor]:
        return self._t_feats

    @property
    def student_feats(self) -> dict[str, torch.Tensor]:
        return self._s_feats

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Feature extraction & CKA ──────────────────────────────────────────────────

@torch.no_grad()
def _extract_avgpool_repr(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """GAP-pooled avgpool features (N, C) over all batches in loader."""
    feats = []
    captured: list[torch.Tensor | None] = [None]

    def hook(m, inp, out):
        captured[0] = out.detach().cpu()

    handle = model.avgpool.register_forward_hook(hook)
    model.eval()
    for images, _ in loader:
        model(images.to(device))
        feats.append(gap(captured[0]))   # (B, C, 1, 1) → (B, C)
    handle.remove()
    return torch.cat(feats, dim=0)


def compute_transfer_cka(
    student: nn.Module,
    teacher: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
) -> float:
    """Linear CKA between student and teacher avgpool representations."""
    student.eval()
    F_s = _extract_avgpool_repr(student, calib_loader, device)
    F_t = _extract_avgpool_repr(teacher, calib_loader, device)
    return linear_cka(F_s, F_t)


# ── Spatial dimension sanity check ────────────────────────────────────────────

def check_spatial_dims(teacher: nn.Module, device: torch.device) -> None:
    """
    Verify that matched teacher/student stages share the same spatial (H, W).
    SP-KD Gram matrices are N×N so channel counts can differ, but identical
    spatial dims confirm the architecture mapping is correct.
    Logs a warning (not an error) on mismatch.
    """
    student = build_student().to(device)
    dummy = torch.zeros(2, 3, 32, 32, device=device)
    t_shapes: dict[str, tuple] = {}
    s_shapes: dict[str, tuple] = {}
    handles = []

    for stage in STAGES:
        def _hook_t(m, inp, out, s=stage):
            t_shapes[s] = tuple(out.shape)
        def _hook_s(m, inp, out, s=stage):
            s_shapes[s] = tuple(out.shape)
        handles.append(getattr(teacher, stage)[PHASE4_TEACHER_MATCH_IDX[stage]].register_forward_hook(_hook_t))
        handles.append(getattr(student, stage)[PHASE4_STUDENT_MATCH_IDX[stage]].register_forward_hook(_hook_s))

    with torch.no_grad():
        teacher(dummy)
        student(dummy)

    for h in handles:
        h.remove()
    del student
    torch.cuda.empty_cache()

    logger.info("SP-KD spatial dimension check:")
    for stage in STAGES:
        ts, ss = t_shapes[stage], s_shapes[stage]
        match = ts[2:] == ss[2:]
        status = "OK" if match else "MISMATCH"
        logger.info(
            f"  {stage}: teacher {ts} | student {ss} | spatial [{status}]"
        )
        if not match:
            logger.warning(
                f"  {stage} spatial mismatch — Gram matrix computation "
                "is still valid (N×N), but architecture mapping may be wrong."
            )


# ── Dry run ────────────────────────────────────────────────────────────────────

def dry_run(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    gamma: float,
    n_batches: int = 5,
) -> dict:
    """
    Run n_batches to measure CE vs γ·SP scale.
    Logs ratio γ·SP / CE — target range ≈ 1–10 for balanced training.
    """
    teacher.eval()
    student.train()
    criterion = nn.CrossEntropyLoss()
    hook_mgr = SPHookManager(teacher, student)
    ces, sps = [], []
    loader_it = iter(loader)

    for _ in range(n_batches):
        try:
            images, labels = next(loader_it)
        except StopIteration:
            break
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            teacher(images)

        logits = student(images)
        loss_ce = criterion(logits, labels)
        loss_sp = sp_kd_loss(hook_mgr.teacher_feats, hook_mgr.student_feats, weights)

        ces.append(loss_ce.item())
        sps.append((gamma * loss_sp).item())

    hook_mgr.remove()
    student.zero_grad()

    mean_ce = sum(ces) / len(ces) if ces else 0.0
    mean_sp = sum(sps) / len(sps) if sps else 0.0
    ratio = mean_sp / (mean_ce + 1e-8)

    logger.info(
        f"Dry run ({len(ces)} batches) — "
        f"CE={mean_ce:.4f}  γ·SP={mean_sp:.4f}  ratio={ratio:.2f}"
    )
    if ratio > 100:
        logger.warning(f"γ·SP / CE ratio={ratio:.1f} > 100 — consider reducing gamma.")
    elif ratio < 0.1:
        logger.warning(f"γ·SP / CE ratio={ratio:.3f} < 0.1 — consider increasing gamma.")

    return {"mean_ce": mean_ce, "mean_sp_scaled": mean_sp, "ratio": ratio}


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ── Single-condition training run ─────────────────────────────────────────────

def run_condition(
    condition: str,
    weights: dict[str, float],
    teacher: nn.Module,
    train_ds,
    test_loader: DataLoader,
    calib_loader: DataLoader,
    device: torch.device,
    seed: int,
    n_epochs: int = PHASE4_N_EPOCHS,
    gamma: float = PHASE4_GAMMA_KD,
) -> dict:
    """Train student for one (condition, seed). Returns per-epoch results."""
    logger.info(f"  [{condition}] seed={seed}")

    set_seed(seed)
    relax_determinism_for_training()    
    print(f"cudnn.deterministic={torch.backends.cudnn.deterministic}, benchmark={torch.backends.cudnn.benchmark}")

    student = build_student().to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=PHASE4_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=get_dataloader_generator(seed),
        persistent_workers=NUM_WORKERS > 0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=PHASE4_LR,
        momentum=PHASE4_MOMENTUM,
        weight_decay=PHASE4_WEIGHT_DECAY,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=PHASE4_LR_MILESTONES,
        gamma=PHASE4_LR_DECAY,
    )

    hook_mgr = SPHookManager(teacher, student)
    teacher.eval()

    acc_curve: list[float] = []
    ce_curve: list[float] = []
    sp_curve: list[float] = []
    cka_curve: list[tuple[int, float]] = []   # (epoch, cka_value)
    wall_times: list[float] = []

    use_sp = (condition != "vanilla")


    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        student.train()
        sum_ce = sum_sp = n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Teacher forward — no gradient accumulation
            if use_sp:
                with torch.no_grad():
                    teacher(images)

            optimizer.zero_grad()
            logits = student(images)

            loss_ce = criterion(logits, labels)

            if use_sp:
                loss_sp = sp_kd_loss(
                    hook_mgr.teacher_feats,
                    hook_mgr.student_feats,
                    weights,
                )
                loss = loss_ce + gamma * loss_sp
            else:
                loss_sp = torch.zeros(1, device=device)
                loss = loss_ce

            loss.backward()

            if epoch <= PHASE4_GRAD_CLIP_EPOCHS:
                nn.utils.clip_grad_norm_(student.parameters(), PHASE4_GRAD_CLIP)

            optimizer.step()

            sum_ce += loss_ce.item()
            sum_sp += (gamma * loss_sp).item()
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0
        wall_times.append(epoch_time)

        if epoch % 5 == 0 or epoch == n_epochs:
            test_acc = _evaluate(student, test_loader, device)
        else:
            test_acc = acc_curve[-1] if acc_curve else 0.0  
        acc_curve.append(test_acc)
        ce_curve.append(sum_ce / n_batches)
        sp_curve.append(sum_sp / n_batches)

        if epoch % PHASE4_CKA_INTERVAL == 0 or epoch == n_epochs:
            cka_val = compute_transfer_cka(student, teacher, calib_loader, device)
            cka_curve.append((epoch, cka_val))
            logger.info(
                f"    epoch {epoch:3d}/{n_epochs} | "
                f"acc={test_acc:.4f} | CE={ce_curve[-1]:.4f} | "
                f"γ·SP={sp_curve[-1]:.4f} | CKA={cka_val:.4f} | "
                f"{epoch_time:.1f}s"
            )
        else:
            logger.info(
                f"    epoch {epoch:3d}/{n_epochs} | "
                f"acc={test_acc:.4f} | CE={ce_curve[-1]:.4f} | "
                f"γ·SP={sp_curve[-1]:.4f} | {epoch_time:.1f}s"
            )
    if hook_mgr is not None:               
        hook_mgr.remove()

    final_cka = cka_curve[-1][1] if cka_curve else None
    logger.info(
        f"  [{condition}] seed={seed} ✓  "
        f"acc={acc_curve[-1]:.4f}  CKA={final_cka:.4f}"
    )

    return {
        "final_accuracy": acc_curve[-1],
        "final_cka": final_cka,
        "accuracy_curve": acc_curve,
        "ce_loss_curve": ce_curve,
        "sp_loss_curve": sp_curve,
        "cka_curve": cka_curve,
        "epoch_wall_times": wall_times,
        "student_state_dict": student.state_dict(),
    }


# ── Main entry point ───────────────────────────────────────────────────────────

def run_phase4(
    teacher: nn.Module,
    train_ds,
    test_loader: DataLoader,
    calib_loader: DataLoader,
    device: torch.device,
    phase2_results: dict | None = None,
    n_epochs: int = PHASE4_N_EPOCHS,
    gamma: float = PHASE4_GAMMA_KD,
    seeds: list[int] | None = None,
    conditions: list[str] | None = None,
) -> dict:
    """
    Run all 4 conditions × 3 seeds.  Results saved incrementally to
    PHASE4_RESULTS after each condition completes.
    """
    if phase2_results is None:
        with open(PHASE2_RESULTS) as f:
            phase2_results = json.load(f)

    seeds = seeds or PHASE4_SEEDS
    conditions = conditions or ["vanilla", "uniform", "bi_acc", "bi_rep"]

    PHASE4_CKPTS_DIR.mkdir(parents=True, exist_ok=True)

    all_weights = build_condition_weights(phase2_results)

    for cond in ["bi_acc", "bi_rep"]:
        logger.info(f"Stage weights [{cond}]: {all_weights[cond]}")

    # Spatial dimension verification (informational)
    check_spatial_dims(teacher, device)

    # Dry run with uniform weights to calibrate gamma
    logger.info("Running dry run to verify γ calibration …")
    _tmp_student = build_student().to(device)
    _tmp_loader = DataLoader(
        train_ds, batch_size=PHASE4_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    dry_stats = dry_run(
        _tmp_student, teacher, _tmp_loader, device,
        weights=all_weights["uniform"], gamma=gamma,
    )
    del _tmp_student, _tmp_loader
    torch.cuda.empty_cache()

    results: dict = {
        "metadata": {
            "n_epochs": n_epochs,
            "gamma_kd": gamma,
            "seeds": seeds,
            "conditions": conditions,
            "dry_run": dry_stats,
            "stage_match": {
                "teacher": PHASE4_TEACHER_MATCH_IDX,
                "student": PHASE4_STUDENT_MATCH_IDX,
            },
        },
        "stage_weights": {
            "bi_acc": all_weights["bi_acc"],
            "bi_rep": all_weights["bi_rep"],
            "uniform": all_weights["uniform"],
        },
        "conditions": {},
    }

    for cond in conditions:
        w = all_weights[cond]
        logger.info(f"\n{'─'*60}")
        logger.info(f"Condition: {cond}  weights: {w}")
        logger.info("─" * 60)
        results["conditions"][cond] = {}

        for seed in seeds:
            run_res = run_condition(
                condition=cond,
                weights=w,
                teacher=teacher,
                train_ds=train_ds,
                test_loader=test_loader,
                calib_loader=calib_loader,
                device=device,
                seed=int(seed),
                n_epochs=n_epochs,
                gamma=gamma,
            )
            

        # Salva pesi modello
        ckpt_dir = CKPT_DIR / cond
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(run_res.pop("student_state_dict"), ckpt_dir / f"student_seed_{seed}.pt")

        results["conditions"][cond][f"seed_{seed}"] = run_res

        # Checkpoint after each condition so partial results survive interruptions
        with open(PHASE4_RESULTS, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Checkpoint saved → {PHASE4_RESULTS}")

    logger.info(f"\nPhase 4 complete. Results → {PHASE4_RESULTS}")
    return results
