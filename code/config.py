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
PHASE3_RESULTS      = RESULTS_DIR / "phase3_results.json"
PRUNING_REAL_RESULTS = RESULTS_DIR / "progressive_pruning_real.json"

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

# ── Phase 4: Weighted SP-KD ──────────────────────────────────────────────────
# Output paths
PHASE4_RESULTS   = RESULTS_DIR / "phase4_results.json"
PHASE4_CKPTS_DIR = CKPT_DIR / "phase4"

# Training hyperparameters
PHASE4_LR              = 0.1
PHASE4_MOMENTUM        = 0.9
PHASE4_WEIGHT_DECAY    = 5e-4
PHASE4_N_EPOCHS        = 200
PHASE4_BATCH_SIZE      = 128
PHASE4_LR_MILESTONES   = [60, 120, 160]   # step-decay epochs
PHASE4_LR_DECAY        = 0.1              # ×0.1 at each milestone
PHASE4_GAMMA_KD        = 3000             # initial KD scale factor
PHASE4_GRAD_CLIP       = 5.0             # gradient clip norm
PHASE4_GRAD_CLIP_EPOCHS = 10             # apply clip for first N epochs
PHASE4_WEIGHT_FLOOR    = 0.05            # min stage weight before normalisation
PHASE4_SEEDS           = [42, 123, 456]
PHASE4_CKA_INTERVAL    = 20             # compute CKA every N epochs

# SP-KD layer matching: last Bottleneck of teacher stage ↔ last BasicBlock of student stage
#   teacher layer1.2 ↔ student layer1.1
#   teacher layer2.3 ↔ student layer2.1
#   teacher layer3.5 ↔ student layer3.1
#   teacher layer4.2 ↔ student layer4.1
PHASE4_TEACHER_MATCH_IDX = {"layer1": 2, "layer2": 3, "layer3": 5, "layer4": 2}
PHASE4_STUDENT_MATCH_IDX = {"layer1": 1, "layer2": 1, "layer3": 1, "layer4": 1}
