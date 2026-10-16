# Block Influence in a CIFAR-10 ResNet-50

Course project for *Artificial and Biological Neural Systems* (University of Trento), supervised by Prof. Uri Hasson.

The project measures how much each of the 16 Bottleneck blocks of a CIFAR-10 ResNet-50 contributes to the network, using three Block Influence (BI) metrics, and then tests whether those scores improve similarity-preserving knowledge distillation (SP-KD) from the ResNet-50 into a ResNet-18.

> **Status:** the report is under revision.

## Metrics

| Metric | Definition | Needs ablation |
|---|---|---|
| BIgeo | 1 − mean cosine similarity between a block's (globally pooled) input and output | No |
| BIacc | Drop in test accuracy when the block is ablated | Yes |
| BIrep | 1 − linear CKA between the intact and ablated penultimate representations (2048-d) | Yes |

`bi_rep_extended.py` adds two variants of BIrep: CKA on the N×N sample Gram matrix, and similarity of the 10×10 class-mean structure.

## Pipeline

| Phase | Script | What it does |
|---|---|---|
| 1 | `phase1_baseline.py` | Checks the pretrained teacher's test accuracy against the expected value in `config.py` (94.65%; the cached run in `code/results/reference_meta.json` measured 94.64%) and extracts reference representations on a balanced calibration set of 2,500 training images |
| 2 | `phase2_metrics.py` | Computes BIgeo, BIacc and BIrep for every block |
| 3 | `phase3_analysis.py` | Kendall's τ between the metric rankings, top-k Jaccard overlap (k = 3, 5), and analysis of "silent failure" blocks (high BIrep, low BIacc): multi-layer CKA, per-class CKA, decision confidence |
| – | `analysis_progressive_pruning_real.py` | Progressive block pruning in the order given by each metric |
| 4 | `phase4_distillation.py` | SP-KD (Tung & Mori, 2019) from ResNet-50 to a CIFAR ResNet-18 (`student_model.py`) with per-stage weights. Default conditions: `vanilla` (no KD), `uniform`, `bi_acc`, `bi_rep`, `conf_gated`; the variants `gated_only` and `delta_kd` run only when passed to `--phase4-conditions`. Seeds 42, 123, 456 |
| 4 (eval) | `phase4_cifar10c_eval.py` | Robustness of teacher and students on CIFAR-10-C and CIFAR-10-P |

Pipeline settings (paths, seeds, thresholds, Phase 4 hyperparameters) are in `code/config.py`. Runs are logged with MLflow under the experiment `block-influence-resnet50-cifar10`.

## Repository layout

```
code/                 pipeline (entry point: run_pipeline.py)
code/results/         cached outputs of Phases 1–3 and progressive pruning
models/teacher/       teacher weights (edadaltocg/resnet50_cifar10) and notebooks
submissions/          Colab version of the pipeline
```

## Setup

Requires Python ≥ 3.10 (the cached results were produced with Python 3.13).

```bash
pip install -r requirements.txt
mkdir -p code/checkpoints
cp models/teacher/modello_edadaltocg/pytorch_model.bin code/checkpoints/
```

CIFAR-10 is downloaded automatically through torchvision into `./data` (i.e. `code/data/` when run from `code/`).

## Usage

Run from `code/`:

```bash
python run_pipeline.py --phase all     # Phases 1–3, progressive pruning, figures
python run_pipeline.py --phase4        # Phase 4 distillation (uses cached Phase 1–2 results)
python run_pipeline.py --figures-only  # regenerate figures from cached results
```

Pass `--phase` explicitly: although `--help` says the default is `all`, no phase runs without it.

Phase 4 options: `--phase4-conditions`, `--phase4-seeds`, `--phase4-epochs`, `--phase4-gamma`. Phase 4 writes its results to `code/results/phase4_results.json` and the student weights to `code/checkpoints/<condition>/student_seed_<seed>.pt`. Figures are written to `code/figures/`.

CIFAR-10-C / CIFAR-10-P evaluation:

```bash
python phase4_cifar10c_eval.py --dataset c --data-dir data/CIFAR-10-C
python phase4_cifar10c_eval.py --dataset p --data-dir data/CIFAR-10-P
```

`--data-dir` must point to the folder that holds the dataset's `.npy` files; if it does not exist, the archive is downloaded from Zenodo and extracted next to it. Student checkpoints are read from `phase4_results/` at the repository root (override with `--results-dir`) and must be named `<condition>_student_seed_<seed>.pt`, with `<condition>` one of `vanilla`, `uniform`, `biacc`, `birep`. Results go to `phase4_cifar10c_results.json` in the same folder; the `p` run is merged into the file written by the `c` run.