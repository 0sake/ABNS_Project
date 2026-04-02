# tracking.py
import mlflow
import json
from pathlib import Path
from config import SEED, N_CALIBRATION, TARGET_BLOCKS

EXPERIMENT_NAME = "block-influence-resnet50-cifar10"

def init_experiment():
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_phase1(test_accuracy: float, f_intact_path: Path = None):
    with mlflow.start_run(run_name="phase1_baseline"):
        mlflow.log_params({
            "seed": SEED,
            "n_calibration": N_CALIBRATION,
            "model": "edadaltocg/resnet50_cifar10",
            "architecture": "ResNet50-CIFAR",
        })
        mlflow.log_metric("test_accuracy", test_accuracy)
        if f_intact_path and f_intact_path.exists():
            mlflow.log_artifact(str(f_intact_path))

def log_phase2(results: dict):
    """results = {metadata: {...}, bi_geo: {block: val}, bi_acc: {block: val}, bi_rep: {block: val}, ...}"""
    with mlflow.start_run(run_name="phase2_metrics"):
        # Log parametri dal metadata
        meta = results.get("metadata", {})
        mlflow.log_params({
            "seed": meta.get("seed"),
            "n_calibration": meta.get("n_calibration"),
            "baseline_accuracy": meta.get("baseline_accuracy"),
        })

        bi_geo      = results.get("bi_geo", {})
        bi_acc      = results.get("bi_acc", {})
        bi_rep      = results.get("bi_rep", {})
        bi_rep_gram  = results.get("bi_rep_gram", {})
        bi_rep_class = results.get("bi_rep_class", {})
        bi_rep_ml    = results.get("bi_rep_multilayer", {})

        for i, block in enumerate(bi_geo.keys()):
            mlflow.log_metric("BIgeo",       bi_geo.get(block, 0),       step=i)
            mlflow.log_metric("BIacc",       bi_acc.get(block, 0),       step=i)
            mlflow.log_metric("BIrep",       bi_rep.get(block, 0),       step=i)
            mlflow.log_metric("BIrep_gram",  bi_rep_gram.get(block, 0),  step=i)
            mlflow.log_metric("BIrep_class", bi_rep_class.get(block, 0), step=i)
            delta = bi_rep.get(block, 0) - bi_acc.get(block, 0)
            mlflow.log_metric("delta", delta, step=i)

            # Multi-layer CKA profile per block
            for stage, cka_val in bi_rep_ml.get(block, {}).items():
                mlflow.log_metric(f"birep_ml_{stage}", cka_val, step=i)

        # Blocco con delta massimo
        deltas = {b: bi_rep.get(b, 0) - bi_acc.get(b, 0) for b in bi_geo}
        max_delta_block = max(deltas, key=deltas.get)
        mlflow.log_param("max_delta_block", max_delta_block)
        mlflow.log_metric("max_delta", deltas[max_delta_block])

        # JSON completo come artefatto
        with open("/tmp/phase2_results.json", "w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact("/tmp/phase2_results.json")

def log_phase3(figure_dir: Path = None):
    """Legge i risultati Phase 3 direttamente dai JSON su disco."""
    from config import RESULTS_DIR

    with mlflow.start_run(run_name="phase3_analysis"):

        # --- Correlazioni ---
        corr_path = RESULTS_DIR / "phase3_correlations.json"
        if corr_path.exists():
            with open(corr_path) as f:
                corr = json.load(f)
            for pair, values in corr.items():
                mlflow.log_metric(f"tau_{pair}", values["tau"])
                mlflow.log_metric(f"pval_{pair}", values["p_value"])
            mlflow.log_artifact(str(corr_path))

        # --- Jaccard ---
        jacc_path = RESULTS_DIR / "phase3_jaccard.json"
        if jacc_path.exists():
            with open(jacc_path) as f:
                jacc = json.load(f)
            for k, pairs in jacc.items():
                for pair, values in pairs.items():
                    mlflow.log_metric(f"jaccard_k{k}_{pair}", values["jaccard"])
            mlflow.log_artifact(str(jacc_path))

        # --- Silent failure (dynamic: log all candidates found on disk) ---
        sf_paths = sorted(RESULTS_DIR.glob("phase3_silent_failure_*.json"))
        for sf_path in sf_paths:
            with open(sf_path) as f:
                sf = json.load(f)

            block = sf["block"]
            # Sanitise block name for use as metric prefix (e.g. "layer4.0" → "layer4_0")
            pfx = "sf_" + block.replace(".", "_")

            mlflow.log_param(f"{pfx}_block", block)
            mlflow.log_metric(f"{pfx}_bi_acc", sf["bi_acc"])
            mlflow.log_metric(f"{pfx}_bi_rep", sf["bi_rep"])
            mlflow.log_metric(f"{pfx}_delta",  sf["delta"])

            # Confidence — complete
            conf = sf.get("confidence", {})
            mlflow.log_metric(f"{pfx}_acc_intact",   conf.get("acc_intact", 0))
            mlflow.log_metric(f"{pfx}_acc_ablated",  conf.get("acc_ablated", 0))
            mlflow.log_metric(f"{pfx}_n_C_l",        conf.get("n_C_l", 0))
            mlflow.log_metric(f"{pfx}_n_E_l",        conf.get("n_E_l", 0))

            entropy = conf.get("entropy", {})
            mlflow.log_metric(f"{pfx}_H_intact",        entropy.get("mean_H_intact", 0))
            mlflow.log_metric(f"{pfx}_H_ablated",       entropy.get("mean_H_ablated", 0))
            mlflow.log_metric(f"{pfx}_delta_H_mean",    entropy.get("mean_delta_H", 0))
            mlflow.log_metric(f"{pfx}_delta_H_median",  entropy.get("median_delta_H", 0))
            mlflow.log_metric(f"{pfx}_entropy_wilcoxon_stat", entropy.get("wilcoxon_stat", 0))
            mlflow.log_metric(f"{pfx}_entropy_wilcoxon_p",    entropy.get("wilcoxon_p", 1))
            mlflow.log_param(f"{pfx}_entropy_significant",    entropy.get("significant", False))

            top1 = conf.get("top1_confidence", {})
            mlflow.log_metric(f"{pfx}_conf_intact",       top1.get("mean_conf_intact", 0))
            mlflow.log_metric(f"{pfx}_conf_ablated",      top1.get("mean_conf_ablated", 0))
            mlflow.log_metric(f"{pfx}_conf_wilcoxon_p",   top1.get("wilcoxon_p", 1))
            mlflow.log_param(f"{pfx}_conf_significant",   top1.get("significant", False))

            # Geometry
            geo = sf.get("geometry", {})
            mlflow.log_metric(f"{pfx}_class_cka_variance", geo.get("class_cka_variance", 0))

            for cls_idx, cka_val in geo.get("per_class_cka", {}).items():
                mlflow.log_metric(f"{pfx}_per_class_cka_{cls_idx}", cka_val)

            for stage, cka_val in geo.get("propagation_profile", {}).items():
                mlflow.log_metric(f"{pfx}_prop_{stage}", cka_val)

            mlflow.log_artifact(str(sf_path))

        # --- Figure ---
        if figure_dir and figure_dir.exists():
            for fig_path in sorted(figure_dir.glob("*.png")):
                mlflow.log_artifact(str(fig_path), artifact_path="figures")