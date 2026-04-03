"""
config.py — Centralised configuration for the entire pipeline.
Edit this file to adjust seeds, paths, thresholds, and hyperparameters.
"""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
RESULTS_DIR   = ROOT / "results"
FIGURES_DIR   = ROOT / "figures"
CKPT_DIR      = ROOT / "checkpoints"

# Expected model checkpoint filename (place your .pth file here)
MODEL_CKPT    = CKPT_DIR / "pytorch_model.bin"

# Phase 1 outputs
CALIB_INDICES = RESULTS_DIR / "calib_indices.pt"
REF_REPR      = RESULTS_DIR / "reference_representations.pt"
REF_META      = RESULTS_DIR / "reference_meta.json"

# Phase 2 output
PHASE2_RESULTS = RESULTS_DIR / "phase2_results.json"

# Phase 3 outputs
PHASE3_CORRELATIONS = RESULTS_DIR / "phase3_correlations.json"
PHASE3_JACCARD      = RESULTS_DIR / "phase3_jaccard.json"
PHASE3_PRUNING      = RESULTS_DIR / "phase3_pruning.json"
PHASE3_PER_CLASS_CKA = RESULTS_DIR / "phase3_per_class_cka.json"

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42

# ── Data ────────────────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

N_CALIBRATION    = 2500          # calibration set size (250 per class, balanced)
N_TEST           = 10_000
SAMPLES_PER_CLASS = 250          # N_CALIBRATION / 10 classes

# ── Model ────────────────────────────────────────────────────────────────────
BASELINE_ACCURACY = 0.9465       # expected intact-model test accuracy
ACCURACY_PASS_THRESHOLD = 0.93   # sanity check: warn if below this
ACCURACY_FAIL_THRESHOLD = 0.90   # hard fail: review normalisation / weights

# ── Computation ──────────────────────────────────────────────────────────────
BATCH_SIZE_INFERENCE = 256        # for test-set passes (BIacc)
BATCH_SIZE_CALIB     = 256        # for calibration-set passes (BIgeo, BIrep)
NUM_WORKERS          = 2

# ── Block registry (canonical ordering, shared across all phases) ─────────────
TARGET_BLOCKS = [
    "layer1.0", "layer1.1", "layer1.2",
    "layer2.0", "layer2.1", "layer2.2", "layer2.3",
    "layer3.0", "layer3.1", "layer3.2", "layer3.3", "layer3.4", "layer3.5",
    "layer4.0", "layer4.1", "layer4.2",
]

# Transition blocks that contain a residual downsampling branch
DOWNSAMPLING_BLOCKS = {"layer1.0", "layer2.0", "layer3.0", "layer4.0"}

# ── CKA ──────────────────────────────────────────────────────────────────────
CKA_EPSILON = 1e-8               # denominator guard against collapse

# ── Phase 3 thresholds ───────────────────────────────────────────────────────
# Silent failure analysis (Step 3.3): tercile-based, computed at runtime.
# Secondary candidate minimum discrepancy
SECONDARY_DELTA_THRESHOLD = 0.15

# Jaccard k values
JACCARD_K_VALUES = [3, 5]

# Statistical significance level
ALPHA = 0.05

# ── Multi-layer CKA extraction stages ───────────────────────────────────────
# Used in the top-3 propagation profile (Step 2.3)
MULTILAYER_STAGES = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
