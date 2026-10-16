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
| 1 | `phase1_baseline.py` | Checks the pretrained teacher (expected test accuracy 94.65%) and extracts reference representations on a balanced calibration set of 2,500 training images |
| 2 | `phase2_metrics.py` | Computes BIgeo, BIacc and BIrep for every block |
| 3 | `phase3_analysis.py` | Kendall's τ between the metric rankings, top-k Jaccard overlap (k = 3, 5), and analysis of "silent failure" blocks (high BIrep, low BIacc): multi-layer CKA, per-class CKA, decision confidence |
| – | `analysis_progressive_pruning_real.py` | Progressive block pruning in the order given by each metric |
| 4 | `phase4_distillation.py` | SP-KD (Tung & Mori, 2019) from ResNet-50 to ResNet-18 with per-stage weights: `vanilla` (no KD), `uniform`, `bi_acc`, `bi_rep`, `conf_gated`, plus the variants `gated_only` and `delta_kd`; seeds 42, 123, 456 |
| 4 (eval) | `phase4_cifar10c_eval.py` | Robustness of teacher and students on CIFAR-10-C and CIFAR-10-P |

All settings (paths, seeds, thresholds, Phase 4 hyperparameters) are in `code/config.py`. Runs are logged with MLflow under the experiment `block-influence-resnet50-cifar10`.

## Repository layout

```
code/                 pipeline (entry point: run_pipeline.py)
code/results/         cached outputs of Phases 1–3
models/teacher/       teacher weights (edadaltocg/resnet50_cifar10) and notebooks
models/student/       ResNet-18 notebook
submissions/          Colab version of the pipeline
```

## Setup

```bash
pip install -r requirements.txt
mkdir -p code/checkpoints
cp models/teacher/modello_edadaltocg/pytorch_model.bin code/checkpoints/
```

CIFAR-10 is downloaded automatically through torchvision.

## Usage

Run from `code/`:

```bash
python run_pipeline.py --phase all     # Phases 1–3, progressive pruning, figures
python run_pipeline.py --phase4        # Phase 4 distillation (uses cached Phase 2 results)
python run_pipeline.py --figures-only  # regenerate figures from cached results
python phase4_cifar10c_eval.py         # CIFAR-10-C / CIFAR-10-P evaluation
```

Phase 4 options: `--phase4-conditions`, `--phase4-seeds`, `--phase4-epochs`, `--phase4-gamma`.