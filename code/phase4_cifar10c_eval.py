"""
phase4_cifar10c_eval.py — Robustness and stability evaluation on CIFAR-10-C and CIFAR-10-P.

Evaluates teacher (ResNet-50) and student (ResNet-18) checkpoints under distribution
shift (CIFAR-10-C: 19 corruptions × 5 severity levels) and under continuous perturbation
sequences (CIFAR-10-P: 10 perturbation types, flip probability).

Usage:
    python phase4_cifar10c_eval.py
    python phase4_cifar10c_eval.py --data-dir data --skip-download
    python phase4_cifar10c_eval.py --results-dir /path/to/checkpoints
"""

# ── Phase 1: Imports ──────────────────────────────────────────────────────────

import argparse
import json
import logging
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config
from model import load_model
from student_model import ResNet18_CIFAR

# ── Phase 1: Constants ────────────────────────────────────────────────────────

CIFAR10_MEAN = config.CIFAR10_MEAN   # (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = config.CIFAR10_STD    # (0.2023, 0.1994, 0.2010)
BATCH_SIZE   = config.BATCH_SIZE_INFERENCE  # 256
ECE_BINS     = 15

CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter", "saturate",
]
CORRUPTION_CATEGORIES = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "weather": ["snow", "frost", "fog", "brightness", "spatter", "saturate"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
SEVERITY_LEVELS = [1, 2, 3, 4, 5]

PERTURBATION_TYPES = [
    "gaussian_noise", "shot_noise", "motion_blur", "zoom_blur",
    "snow", "brightness", "translate", "rotate", "tilt", "scale",
]
PERTURBATION_CATEGORIES = {
    "noise":     ["gaussian_noise", "shot_noise"],
    "geometric": ["translate", "rotate", "tilt", "scale"],
    "other":     ["motion_blur", "zoom_blur", "snow", "brightness"],
}

CONDITION_MAP = {
    "biacc":   "bi_acc",
    "birep":   "bi_rep",
    "uniform": "uniform",
    "vanilla": "vanilla",
}

CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar.gz"
CIFAR10P_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-P.tar.gz"


# ── Phase 1: Download utilities ───────────────────────────────────────────────

def _download_and_extract(url: str, dest_dir: Path) -> None:
    """Download a .tar.gz from url and extract into dest_dir.parent. Skip if already populated."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        logging.info(f"{dest_dir} already populated — skipping download")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir.parent / Path(url).name
    try:
        logging.info(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, archive)
    except Exception as e:
        if archive.exists():
            archive.unlink()
        raise RuntimeError(
            f"Download failed: {e}\nIf offline, place files manually at {dest_dir}"
        ) from e
    logging.info(f"Extracting {archive} → {dest_dir.parent} ...")
    with tarfile.open(archive) as tf:
        tf.extractall(dest_dir.parent)
    archive.unlink()


def _ensure_dataset(url: str, dest_dir: Path, skip_download: bool) -> None:
    """Download and verify a single dataset directory. Raises if missing when skip_download=True."""
    if skip_download:
        if not (dest_dir / "labels.npy").exists():
            raise FileNotFoundError(
                f"{dest_dir.name} not found at {dest_dir}. "
                "Run without --skip-download to fetch it."
            )
        return
    _download_and_extract(url, dest_dir)
    if not (dest_dir / "labels.npy").exists():
        raise FileNotFoundError(f"{dest_dir.name} labels.npy not found at {dest_dir}")


# ── Phase 2: Preprocessing ────────────────────────────────────────────────────

def _preprocess_numpy_images(images: np.ndarray) -> torch.Tensor:
    """
    Convert uint8 numpy (N, H, W, C) → float32 tensor (N, C, H, W), normalized.
    Matches CIFAR-10 training normalization: mean/std from config.
    Does NOT use torchvision transforms (they require PIL Images).
    """
    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  dtype=torch.float32).view(1, 3, 1, 1)
    t = torch.from_numpy(images.astype(np.float32)).div(255.0)  # (N, H, W, C)
    t = t.permute(0, 3, 1, 2)                                    # (N, C, H, W)
    return (t - mean) / std


# ── Phase 2: ECE computation ──────────────────────────────────────────────────

def compute_ece(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    n_bins: int = ECE_BINS,
) -> float:
    """
    Expected Calibration Error with equal-width confidence bins.

    confidences: (N,) — max softmax probability per sample
    correct:     (N,) bool — whether argmax prediction matched label
    Returns scalar ECE ∈ [0, 1].
    """
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo = bin_edges[i].item()
        hi = bin_edges[i + 1].item()
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_b = mask.sum().item()
        if n_b == 0:
            continue
        acc_b  = correct[mask].float().mean().item()
        conf_b = confidences[mask].mean().item()
        ece += (n_b / n) * abs(acc_b - conf_b)
    return float(ece)


# ── Phase 2: Inference helpers ─────────────────────────────────────────────────

@torch.no_grad()
def _eval_on_images(
    model: torch.nn.Module,
    images_np: np.ndarray,
    labels_np: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Run inference on raw uint8 numpy images (H,W,C).
    Returns accuracy, mean confidence on correct, mean entropy, ECE.
    """
    images_t = _preprocess_numpy_images(images_np)
    labels_t = torch.from_numpy(labels_np.astype(np.int64))
    ds = TensorDataset(images_t, labels_t)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    all_probs, all_labels_list = [], []
    for imgs, lbs in loader:
        logits = model(imgs.to(device))
        probs = F.softmax(logits, dim=1).cpu()
        all_probs.append(probs)
        all_labels_list.append(lbs)

    all_probs  = torch.cat(all_probs)    # (N, 10)
    all_labels = torch.cat(all_labels_list)  # (N,)

    preds = all_probs.argmax(dim=1)
    correct = (preds == all_labels)
    confidences = all_probs.max(dim=1).values
    entropy = -(all_probs * (all_probs + 1e-10).log()).sum(dim=1)

    return {
        "accuracy":   correct.float().mean().item(),
        "confidence": confidences[correct].mean().item() if correct.any() else 0.0,
        "entropy":    entropy.mean().item(),
        "ece":        compute_ece(confidences, correct),
    }


@torch.no_grad()
def _eval_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run inference on a DataLoader of pre-normalized tensors (e.g. clean CIFAR-10 test set).
    Returns the same dict as _eval_on_images.
    """
    all_probs, all_labels_list = [], []
    for imgs, lbs in loader:
        logits = model(imgs.to(device))
        probs = F.softmax(logits, dim=1).cpu()
        all_probs.append(probs)
        all_labels_list.append(lbs)

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels_list)

    preds = all_probs.argmax(dim=1)
    correct = (preds == all_labels)
    confidences = all_probs.max(dim=1).values
    entropy = -(all_probs * (all_probs + 1e-10).log()).sum(dim=1)

    return {
        "accuracy":   correct.float().mean().item(),
        "confidence": confidences[correct].mean().item() if correct.any() else 0.0,
        "entropy":    entropy.mean().item(),
        "ece":        compute_ece(confidences, correct),
    }


# ── Phase 5: Intermediate save ────────────────────────────────────────────────

def _save_intermediate(output: dict, output_path: Path) -> None:
    """Atomic write via a .tmp.json swap to avoid partial-write corruption."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
    tmp.replace(output_path)
    logging.debug(f"Intermediate save → {output_path}")


# ── Phase 2: CIFAR-10-C evaluation for one model ─────────────────────────────

def run_cifar10c_for_model(
    model: torch.nn.Module,
    model_name: str,
    cifar10c_dir: Path,
    device: torch.device,
    clean_loader: DataLoader,
    output: dict,
    output_path: Path,
) -> dict:
    """
    Evaluate one model on the clean CIFAR-10 test set and on all
    19 corruptions × 5 severity levels. Saves intermediate results every 5 corruptions.
    """
    clean_stats = _eval_on_loader(model, clean_loader, device)
    result = {
        "clean_accuracy": clean_stats["accuracy"],
        "clean_ece":      clean_stats["ece"],
        "per_corruption": {},
    }

    labels_all = np.load(cifar10c_dir / "labels.npy")  # (50000,)

    for i, corruption in enumerate(CORRUPTION_TYPES):
        logging.info(f"  [{model_name}] corruption {i + 1}/{len(CORRUPTION_TYPES)}: {corruption}")
        images_all = np.load(cifar10c_dir / f"{corruption}.npy")  # (50000, 32, 32, 3)

        accs, confs, entropies, eces = [], [], [], []
        for sev in SEVERITY_LEVELS:
            start = (sev - 1) * 10_000
            imgs_s = images_all[start:start + 10_000]
            lbs_s  = labels_all[start:start + 10_000]
            s = _eval_on_images(model, imgs_s, lbs_s, device)
            accs.append(s["accuracy"])
            confs.append(s["confidence"])
            entropies.append(s["entropy"])
            eces.append(s["ece"])

        result["per_corruption"][corruption] = {
            "accuracies":    accs,
            "mean_accuracy": float(np.mean(accs)),
            "confidences":   confs,
            "entropies":     entropies,
            "eces":          eces,
            "mean_ece":      float(np.mean(eces)),
        }
        del images_all

        if (i + 1) % 5 == 0:
            output["cifar10c"]["models"][model_name] = result
            _save_intermediate(output, output_path)

    return result


# ── Phase 3: CIFAR-10-C summary metrics ──────────────────────────────────────

def compute_mce(
    model_per_corruption: dict,
    teacher_per_corruption: dict,
) -> float:
    """
    Mean Corruption Error.
    CE_c = sum_s(1 - acc_model_c_s) / sum_s(1 - acc_teacher_c_s)
    mCE  = mean(CE_c) across all 19 corruptions.
    """
    ces = []
    for c in CORRUPTION_TYPES:
        model_errors   = [1.0 - a for a in model_per_corruption[c]["accuracies"]]
        teacher_errors = [1.0 - a for a in teacher_per_corruption[c]["accuracies"]]
        denom = sum(teacher_errors)
        ces.append(sum(model_errors) / denom if denom > 0 else 0.0)
    return float(np.mean(ces))


def compute_rmce(
    model_per_corruption: dict,
    teacher_per_corruption: dict,
    model_clean_acc: float,
    teacher_clean_acc: float,
) -> float:
    """
    Relative mCE: normalises by clean accuracy degradation to isolate the
    robustness component independent of absolute accuracy differences.
    RmCE_c = sum_s(acc_clean - acc_c_s) / sum_s(teacher_clean - teacher_c_s)
    """
    rmces = []
    for c in CORRUPTION_TYPES:
        model_drops   = [model_clean_acc   - a for a in model_per_corruption[c]["accuracies"]]
        teacher_drops = [teacher_clean_acc - a for a in teacher_per_corruption[c]["accuracies"]]
        denom = sum(teacher_drops)
        rmces.append(sum(model_drops) / denom if denom > 0 else 0.0)
    return float(np.mean(rmces))


def compute_per_category_accuracy(per_corruption: dict) -> dict:
    """
    Mean accuracy per corruption category per severity level.
    Returns {category: [sev1_mean, ..., sev5_mean]}.
    """
    result = {}
    for cat, members in CORRUPTION_CATEGORIES.items():
        sev_means = []
        for sev_idx in range(5):
            sev_means.append(
                float(np.mean([per_corruption[c]["accuracies"][sev_idx] for c in members]))
            )
        result[cat] = sev_means
    return result


def aggregate_condition_cifar10c(condition_results: list) -> dict:
    """Aggregate mCE, RmCE, and ECE statistics across seeds for one distillation condition."""
    mces       = [r["mCE"]  for r in condition_results]
    rmces      = [r["rmCE"] for r in condition_results]
    eces_clean = [r["clean_ece"] for r in condition_results]
    eces_corr  = [r["mean_ece_across_corruptions"] for r in condition_results]

    cat_accs = {cat: [] for cat in CORRUPTION_CATEGORIES}
    for r in condition_results:
        for cat, sevs in r["per_category_accuracy"].items():
            cat_accs[cat].append(sevs)
    cat_means = {
        cat: [float(v) for v in np.mean(cat_accs[cat], axis=0)]
        for cat in cat_accs
    }

    return {
        "mCE_mean":              float(np.mean(mces)),
        "mCE_std":               float(np.std(mces)),
        "rmCE_mean":             float(np.mean(rmces)),
        "rmCE_std":              float(np.std(rmces)),
        "mean_ece_clean":        float(np.mean(eces_clean)),
        "mean_ece_clean_std":    float(np.std(eces_clean)),
        "mean_ece_corrupted":    float(np.mean(eces_corr)),
        "mean_ece_corrupted_std": float(np.std(eces_corr)),
        "per_category_accuracy": cat_means,
    }


# ── Phase 4: CIFAR-10-P evaluation ────────────────────────────────────────────

@torch.no_grad()
def eval_perturbation(
    model: torch.nn.Module,
    pert_data: np.ndarray,
    labels_np: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Compute flip probability and mean accuracy over a perturbation sequence.

    Handles two npy layouts from the zenodo CIFAR-10-P archive:
      (num_frames, 10000, H, W, C) — frames-first
      (10000, num_frames, H, W, C) — images-first (actual zenodo format)
    Layout is detected by comparing shape[0] vs shape[1]: frames are always
    the smaller dimension (typically 10–35), images are always 10000.

    labels_np may have more entries than images (e.g. 50000); only the first
    n_images labels are used.
    """
    # Normalise to frames-first: (num_frames, N, H, W, C)
    if pert_data.shape[0] > pert_data.shape[1]:
        pert_data = pert_data.transpose(1, 0, 2, 3, 4)

    num_frames = pert_data.shape[0]
    n_images   = pert_data.shape[1]
    labels_np  = labels_np[:n_images]   # guard against 50k-entry labels files
    preds_per_frame = []

    for frame_idx in range(num_frames):
        frame_imgs = pert_data[frame_idx]  # (N, H, W, C)
        images_t = _preprocess_numpy_images(frame_imgs)
        ds = TensorDataset(images_t)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        frame_preds = []
        for (imgs,) in loader:
            preds = model(imgs.to(device)).argmax(dim=1).cpu()
            frame_preds.append(preds)
        preds_per_frame.append(torch.cat(frame_preds))  # (10000,)

    flips = []
    for t in range(num_frames - 1):
        flips.append((preds_per_frame[t] != preds_per_frame[t + 1]).float().mean().item())
    fp = float(np.mean(flips)) if flips else 0.0

    all_labels = torch.from_numpy(labels_np.astype(np.int64))
    frame_accs = [(pf == all_labels).float().mean().item() for pf in preds_per_frame]

    return {
        "flip_probability":           fp,
        "mean_accuracy_across_frames": float(np.mean(frame_accs)),
    }


def run_cifar10p_for_model(
    model: torch.nn.Module,
    model_name: str,
    cifar10p_dir: Path,
    device: torch.device,
    output: dict,
    output_path: Path,
) -> dict:
    """
    Evaluate one model on all 10 CIFAR-10-P perturbation types.
    Loads one perturbation .npy at a time and frees memory after each.
    Saves intermediate results every 5 perturbations.
    """
    labels_np = np.load(cifar10p_dir / "labels.npy")  # (10000,)
    per_pert = {}

    for i, pert in enumerate(PERTURBATION_TYPES):
        logging.info(f"  [{model_name}] perturbation {i + 1}/{len(PERTURBATION_TYPES)}: {pert}")
        pert_data = np.load(cifar10p_dir / f"{pert}.npy")  # (num_frames, 10000, 32, 32, 3)
        per_pert[pert] = eval_perturbation(model, pert_data, labels_np, device)
        del pert_data

        if (i + 1) % 5 == 0:
            output["cifar10p"]["models"][model_name] = {"per_perturbation": per_pert}
            _save_intermediate(output, output_path)

    mfp = float(np.mean([per_pert[p]["flip_probability"] for p in PERTURBATION_TYPES]))
    return {"per_perturbation": per_pert, "mFP": mfp}


# ── Phase 4: CIFAR-10-P summary metrics ──────────────────────────────────────

def compute_fpr(student_per_pert: dict, teacher_per_pert: dict) -> dict:
    """FPR per perturbation = FP_student / FP_teacher. FPR > 1 means student is more jittery."""
    fpr = {}
    for p in PERTURBATION_TYPES:
        t_fp = teacher_per_pert[p]["flip_probability"]
        s_fp = student_per_pert[p]["flip_probability"]
        fpr[p] = s_fp / t_fp if t_fp > 0 else 0.0
    return fpr


def compute_per_category_fp(per_pert: dict) -> dict:
    """Mean flip probability per perturbation category."""
    return {
        cat: float(np.mean([per_pert[p]["flip_probability"] for p in members]))
        for cat, members in PERTURBATION_CATEGORIES.items()
    }


def aggregate_condition_cifar10p(condition_results: list) -> dict:
    """Aggregate mFP and mFPR statistics across seeds for one distillation condition."""
    mfps  = [r["mFP"]  for r in condition_results]
    mfprs = [r["mFPR"] for r in condition_results]

    cat_fps = {cat: [] for cat in PERTURBATION_CATEGORIES}
    for r in condition_results:
        for cat, v in r["per_category_FP"].items():
            cat_fps[cat].append(v)

    return {
        "mFP_mean":       float(np.mean(mfps)),
        "mFP_std":        float(np.std(mfps)),
        "mFPR_mean":      float(np.mean(mfprs)),
        "mFPR_std":       float(np.std(mfprs)),
        "per_category_FP": {cat: float(np.mean(v)) for cat, v in cat_fps.items()},
    }


# ── Phase 5: Checkpoint discovery ─────────────────────────────────────────────

def discover_checkpoints(results_dir: Path) -> list:
    """
    Glob phase4_results/ for *_student_seed_*.pt files.
    Returns list of (condition_key, seed, path) sorted by condition then seed.
    Regex pattern from phase4_analysis.py:54-58.
    """
    ckpts = []
    if not results_dir.exists():
        logging.warning(f"Results dir {results_dir} does not exist — no student checkpoints found")
        return ckpts
    for p in sorted(results_dir.glob("*_student_seed_*.pt")):
        m = re.match(r"^([a-z]+)_student_seed_(\d+)\.pt$", p.name)
        if m:
            ckpts.append((m.group(1), int(m.group(2)), p))
    return ckpts


# ── Phase 5: Summary table ─────────────────────────────────────────────────────

def _print_summary_table(cifar10c_data: dict) -> None:
    """Print mCE per condition to the log in a compact table."""
    logging.info("\n=== CIFAR-10-C Summary (mCE) ===")
    logging.info(f"{'Condition':<12} {'mCE mean':>10} {'mCE std':>10}")
    logging.info("-" * 35)
    agg = cifar10c_data.get("aggregate", {})
    for cond_key in ["bi_acc", "bi_rep", "uniform", "vanilla"]:
        if cond_key in agg:
            a = agg[cond_key]
            logging.info(f"{cond_key:<12} {a['mCE_mean']:>10.4f} {a['mCE_std']:>10.4f}")
    logging.info("-" * 35)


# ── Phase 5: Main function ─────────────────────────────────────────────────────

def run_cifar10c_eval(
    data_dir: str = "data",
    results_dir: Path = None,
    skip_download: bool = False,
    dataset: str = "both",
) -> None:
    """
    Full evaluation pipeline.

    dataset: which benchmark(s) to run
      "c"    — CIFAR-10-C only  (corruption robustness)
      "p"    — CIFAR-10-P only  (perturbation stability)
      "both" — run both (default)

    When dataset="p" and a prior JSON from a "c" run already exists, the P
    results are merged into it so the output file stays complete.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if dataset not in ("c", "p", "both"):
        raise ValueError(f"--dataset must be 'c', 'p', or 'both'; got {dataset!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    data_path    = Path(data_dir)
    results_path = results_dir if results_dir is not None else config.PHASE4_DIR
    output_path  = results_path / "phase4_cifar10c_results.json"

    logging.info(f"Dataset(s)  : {dataset}")
    logging.info(f"Results dir : {results_path}")
    logging.info(f"Output path : {output_path}")

    run_c = dataset in ("c", "both")
    run_p = dataset in ("p", "both")

    # ── 1. Download only the needed dataset(s) ───────────────────────────────────
    cifar10c_dir = data_path / "CIFAR-10-C"
    cifar10p_dir = data_path / "CIFAR-10-P"
    if run_c:
        _ensure_dataset(CIFAR10C_URL, cifar10c_dir, skip_download)
    if run_p:
        _ensure_dataset(CIFAR10P_URL, cifar10p_dir, skip_download)

    # ── 2. Discover student checkpoints ─────────────────────────────────────────
    ckpts = discover_checkpoints(results_path)
    logging.info(f"Found {len(ckpts)} student checkpoint(s) in {results_path}")

    # ── 3. Load existing output JSON (enables merging C and P partial runs) ──────
    if output_path.exists():
        with open(output_path) as f:
            output: dict = json.load(f)
        logging.info(f"Loaded existing results from {output_path} (will merge)")
    else:
        output = {}

    teacher = load_model(config.MODEL_CKPT, device)

    # ── CIFAR-10-C ───────────────────────────────────────────────────────────────
    if run_c:
        from data import load_cifar10, get_test_loader
        _, test_ds   = load_cifar10(data_dir)
        clean_loader = get_test_loader(test_ds)

        output["cifar10c"] = {
            "corruption_types": CORRUPTION_TYPES,
            "severity_levels":  SEVERITY_LEVELS,
            "models":           {},
            "aggregate":        {},
        }

        logging.info("=== Evaluating teacher on CIFAR-10-C ===")
        teacher_c = run_cifar10c_for_model(
            teacher, "teacher", cifar10c_dir, device, clean_loader, output, output_path
        )
        teacher_c["mCE"]                   = None
        teacher_c["per_category_accuracy"]  = compute_per_category_accuracy(teacher_c["per_corruption"])
        output["cifar10c"]["models"]["teacher"] = teacher_c
        _save_intermediate(output, output_path)

        condition_c_results: dict = {k: [] for k in CONDITION_MAP.values()}

        for cond_key, seed, ckpt_path in ckpts:
            model_name = f"{cond_key}_seed_{seed}"
            logging.info(f"=== Evaluating {model_name} on CIFAR-10-C ===")
            student = ResNet18_CIFAR(num_classes=10).to(device)
            student.load_state_dict(torch.load(ckpt_path, map_location=device))
            student.eval()

            r = run_cifar10c_for_model(
                student, model_name, cifar10c_dir, device, clean_loader, output, output_path
            )
            r["mCE"]  = compute_mce(r["per_corruption"], teacher_c["per_corruption"])
            r["rmCE"] = compute_rmce(
                r["per_corruption"], teacher_c["per_corruption"],
                r["clean_accuracy"], teacher_c["clean_accuracy"],
            )
            r["mean_ece_across_corruptions"] = float(
                np.mean([r["per_corruption"][c]["mean_ece"] for c in CORRUPTION_TYPES])
            )
            r["per_category_accuracy"] = compute_per_category_accuracy(r["per_corruption"])

            output["cifar10c"]["models"][model_name] = r
            canon = CONDITION_MAP.get(cond_key, cond_key)
            if canon in condition_c_results:
                condition_c_results[canon].append(r)
            _save_intermediate(output, output_path)
            del student

        for canon, results_list in condition_c_results.items():
            if results_list:
                output["cifar10c"]["aggregate"][canon] = aggregate_condition_cifar10c(results_list)

        _print_summary_table(output["cifar10c"])
        _save_intermediate(output, output_path)

    # ── CIFAR-10-P ────────────────────────────────────────────────────────────────
    if run_p:
        output["cifar10p"] = {
            "perturbation_types": PERTURBATION_TYPES,
            "models":             {},
            "aggregate":          {},
        }

        logging.info("=== Evaluating teacher on CIFAR-10-P ===")
        teacher_p = run_cifar10p_for_model(
            teacher, "teacher", cifar10p_dir, device, output, output_path
        )
        output["cifar10p"]["models"]["teacher"] = teacher_p
        _save_intermediate(output, output_path)

        condition_p_results: dict = {k: [] for k in CONDITION_MAP.values()}

        for cond_key, seed, ckpt_path in ckpts:
            model_name = f"{cond_key}_seed_{seed}"
            logging.info(f"=== Evaluating {model_name} on CIFAR-10-P ===")
            student = ResNet18_CIFAR(num_classes=10).to(device)
            student.load_state_dict(torch.load(ckpt_path, map_location=device))
            student.eval()

            p_r = run_cifar10p_for_model(
                student, model_name, cifar10p_dir, device, output, output_path
            )
            t_per_pert = output["cifar10p"]["models"]["teacher"]["per_perturbation"]
            p_r["FPR_per_perturbation"] = compute_fpr(p_r["per_perturbation"], t_per_pert)
            p_r["mFPR"]                 = float(np.mean(list(p_r["FPR_per_perturbation"].values())))
            p_r["per_category_FP"]      = compute_per_category_fp(p_r["per_perturbation"])

            output["cifar10p"]["models"][model_name] = p_r
            canon = CONDITION_MAP.get(cond_key, cond_key)
            if canon in condition_p_results:
                condition_p_results[canon].append(p_r)
            _save_intermediate(output, output_path)
            del student

        for canon, results_list in condition_p_results.items():
            if results_list:
                output["cifar10p"]["aggregate"][canon] = aggregate_condition_cifar10p(results_list)

        _save_intermediate(output, output_path)

    logging.info(f"Done. Results saved to {output_path}")


# ── Phase 5: CLI guard ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR-10-C/P robustness and stability evaluation for distillation study"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (CIFAR-10-C and CIFAR-10-P will be placed under here)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing student checkpoints and where output JSON is saved "
             "(default: config.PHASE4_DIR = Bi_project/phase4_results/)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download check (raise an error if datasets are missing instead)",
    )
    parser.add_argument(
        "--dataset",
        choices=["c", "p", "both"],
        default="both",
        help="Which benchmark to evaluate: 'c' (CIFAR-10-C corruption), "
             "'p' (CIFAR-10-P perturbation stability), or 'both' (default)",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else None
    run_cifar10c_eval(args.data_dir, results_dir, args.skip_download, args.dataset)
