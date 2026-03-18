"""
run_pipeline.py — Main entry point for the full ResNet-50 Block Influence pipeline.

Usage:
    python run_pipeline.py [--phase {1,2,3,all}] [--device {cuda,cpu}]

By default runs all phases sequentially.
Intermediate results are checkpointed to disk, so individual phases can be
re-run in isolation after a successful earlier phase.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

import config as cfg
from config import (
    CALIB_INDICES, MODEL_CKPT, PHASE2_RESULTS,
    REF_REPR, RESULTS_DIR, FIGURES_DIR, SEED,
)
from data import get_calibration_loader, get_test_loader, load_cifar10, load_or_build_calibration_indices
from model import build_block_registry, load_model
from phase1_baseline import run_phase1
from phase2_metrics import run_phase2
from phase3_analysis import run_phase3
from visualisation import generate_all_figures
from utils import log_environment, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet-50 Block Influence Pipeline")
    p.add_argument(
        "--phase", choices=["1", "2", "3", "all"], default="all",
        help="Which phase(s) to run (default: all)"
    )
    p.add_argument(
        "--device", choices=["cuda", "cpu"], default=None,
        help="Force device (default: auto-detect CUDA)"
    )
    p.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation (useful for headless runs)"
    )
    return p.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or (requested is None and torch.cuda.is_available()):
        return torch.device("cuda")
    logger.info("CUDA not available — falling back to CPU.")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    t_total = time.time()

    # ── Setup ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    env = log_environment()
    device = resolve_device(args.device)
    logger.info(f"Running on: {device}")

    run_all  = args.phase == "all"
    run_p1   = run_all or args.phase == "1"
    run_p2   = run_all or args.phase == "2"
    run_p3   = run_all or args.phase == "3"

    # ── Load shared assets (always needed) ────────────────────────
    logger.info("Loading CIFAR-10 …")
    train_ds, test_ds = load_cifar10()

    calib_indices = load_or_build_calibration_indices(train_ds, save_path=CALIB_INDICES)
    test_loader   = get_test_loader(test_ds)
    calib_loader  = get_calibration_loader(train_ds, calib_indices)

    logger.info("Loading model …")
    model = load_model(MODEL_CKPT, device)
    registry = build_block_registry(model)

    # ── Phase 1 ───────────────────────────────────────────────────
    if run_p1:
        logger.info("\n" + "═" * 60)
        logger.info("PHASE 1: Setup & Baseline")
        logger.info("═" * 60)
        baseline_acc, F_intact = run_phase1(
            model, test_loader, calib_loader, device, env
        )
    else:
        # Load from disk for subsequent phases
        if not REF_REPR.exists():
            raise FileNotFoundError(
                "Reference representations not found. Run Phase 1 first."
            )
        F_intact = torch.load(REF_REPR, map_location="cpu")
        with open(cfg.REF_META) as f:
            meta = json.load(f)
        baseline_acc = meta["baseline_accuracy"]
        logger.info(
            f"Loaded cached F_intact {tuple(F_intact.shape)}, "
            f"baseline_acc={baseline_acc:.4f}"
        )

    # ── Phase 2 ───────────────────────────────────────────────────
    if run_p2:
        p2_results = run_phase2(
            model=model,
            test_loader=test_loader,
            calib_loader=calib_loader,
            device=device,
            F_intact=F_intact,
            baseline_acc=baseline_acc,
            calib_indices_path=CALIB_INDICES,
            env=env,
        )
    else:
        if not PHASE2_RESULTS.exists():
            raise FileNotFoundError(
                "Phase 2 results not found. Run Phase 2 first."
            )
        logger.info(f"Phase 2 results loaded from {PHASE2_RESULTS}")
        p2_results = None  # Phase 3 reads directly from disk

    # ── Phase 3 ───────────────────────────────────────────────────
    if run_p3:
        p3_results = run_phase3(
            model=model,
            registry=registry,
            test_loader=test_loader,
            calib_loader=calib_loader,
            device=device,
            F_intact=F_intact,
            baseline_acc=baseline_acc,
        )

        if not args.skip_figures:
            generate_all_figures(
                p3_results,
                model=model,
                registry=registry,
                calib_loader=calib_loader,
                device=device,
            )
    else:
        p3_results = None

    elapsed = time.time() - t_total
    logger.info("\n" + "═" * 60)
    logger.info(f"Pipeline complete in {elapsed/60:.1f} min")
    logger.info("Results directory: " + str(RESULTS_DIR))
    logger.info("Figures directory: " + str(FIGURES_DIR))
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
