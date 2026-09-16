"""
embedding_visualization.py
──────────────────────────
Visualizza le feature embeddings estratte dal layer GAP di qualsiasi modello
ResNet su CIFAR-10 tramite UMAP, colorate per classe.

Funziona con architetture diverse (ResNet-50, ResNet-18) perché usa un hook
su model.avgpool — non fa assunzioni sulla dimensione delle feature.

Dipendenze:
    pip install umap-learn matplotlib torch torchvision
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import umap

# ── Costanti CIFAR-10 ──────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ══════════════════════════════════════════════════════════════════════════
# Funzione 1: estrazione feature
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_gap_features(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 256,
    data_dir: str = "./data",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estrae le feature dal layer avgpool del modello su tutto il test set CIFAR-10.

    Usa un forward hook su model.avgpool: funziona con ResNet-50 (2048-dim)
    e ResNet-18 (512-dim) senza modifiche.

    Args:
        model:      nn.Module già caricato, in eval mode, sul device corretto
        device:     device su cui gira il modello
        batch_size: batch size per l'inferenza
        data_dir:   dove scaricare/trovare CIFAR-10 (default: ./data)

    Returns:
        features: np.ndarray shape (N, D)  — feature GAP per ogni campione
        labels:   np.ndarray shape (N,)    — etichette di classe 0-9
    """
    features_list = []

    # Hook: cattura l'output di avgpool prima del flatten
    # output shape: (B, C, 1, 1) → squeeze → (B, C)
    def _hook(module, input, output):
        features_list.append(output.squeeze(-1).squeeze(-1).detach().cpu())

    hook = model.avgpool.register_forward_hook(_hook)

    # Test set con sole normalizzazioni (no augmentation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    labels_list = []
    for images, labels in test_loader:
        model(images.to(device))   # il hook raccoglie le feature
        labels_list.append(labels)

    hook.remove()  # importante: rimuovi sempre l'hook dopo l'uso

    features = torch.cat(features_list, dim=0).numpy()
    labels   = torch.cat(labels_list,   dim=0).numpy()
    return features, labels


# ══════════════════════════════════════════════════════════════════════════
# Funzione 2: UMAP + plot
# ══════════════════════════════════════════════════════════════════════════

def plot_umap_embedding(
    model: nn.Module,
    model_name: str,
    device: torch.device,
    save_path: str = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    batch_size: int = 256,
    data_dir: str = "./data",
) -> None:
    """
    Pipeline completa: estrae feature GAP → applica UMAP → salva/mostra il plot.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  COSA FARE PRIMA DI CHIAMARE QUESTA FUNZIONE                        │
    │                                                                     │
    │  1. Istanzia l'architettura corretta:                               │
    │       ResNet-18 per gli student, ResNet-50 per il teacher           │
    │                                                                     │
    │  2. Carica i pesi dal file .pt:                                     │
    │       sd = torch.load("path/al/checkpoint.pt", map_location=device) │
    │                                                                     │
    │  3. Gestisci il formato del checkpoint (vedi nota sotto)            │
    │                                                                     │
    │  4. Chiama model.load_state_dict(sd)                                │
    │                                                                     │
    │  5. Metti il modello in eval mode sul device:                       │
    │       model.eval().to(device)                                       │
    │                                                                     │
    │  NOTA sui checkpoint: il formato del .pt dipende da come è stato    │
    │  salvato durante il training. Se load_state_dict lancia errore,     │
    │  prima di tutto stampa print(sd.keys()) per vedere la struttura.    │
    │  Casi comuni:                                                       │
    │    - sd è direttamente lo state_dict → usalo as-is                  │
    │    - sd = {"model": ..., "optimizer": ...} → usa sd["model"]        │
    │    - sd = {"state_dict": ...} → usa sd["state_dict"]                │
    └─────────────────────────────────────────────────────────────────────┘

    Args:
        model:        nn.Module pronto (vedi sopra)
        model_name:   label per il titolo del plot, es. "Student BI-Rep"
        device:       device su cui gira il modello
        save_path:    path dove salvare il png; se None mostra il plot interattivo
        n_neighbors:  parametro UMAP (default 15): valori più alti → struttura globale
        min_dist:     parametro UMAP (default 0.1): valori più bassi → cluster compatti
        random_state: seed per la riproducibilità di UMAP
        batch_size:   batch size per l'estrazione delle feature
        data_dir:     directory CIFAR-10
    """
    print(f"[{model_name}] Estrazione feature GAP...")
    features, labels = extract_gap_features(model, device, batch_size, data_dir)
    print(f"[{model_name}] Feature shape: {features.shape} — avvio UMAP...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(features)   # (N, 2)
    print(f"[{model_name}] UMAP completato.")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for class_idx, (class_name, color) in enumerate(zip(CIFAR10_CLASSES, colors)):
        mask = labels == class_idx
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            label=class_name,
            s=2,           # punti piccoli: 10k campioni, sovraffollamento
            alpha=0.5,
            rasterized=True,
        )

    ax.set_title(f"UMAP Embedding — {model_name}", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=6, loc="best", fontsize=8, framealpha=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[{model_name}] Salvato in: {save_path}")
    else:
        plt.show()

    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# Esempio d'uso — modifica i path e lancia questo script direttamente
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from torchvision.models import resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    BASE_PATH = "/home/andrea/projects/abns/Bi_project/phase4_results"
    METHODS   = ["birep", "biacc", "uniform", "vanilla"]

    for method in METHODS:
        ckpt_path = f"{BASE_PATH}/{method}_student_seed_42.pt"

        # Istanzia ResNet-18 per CIFAR-10 (10 classi)
        model = resnet18(num_classes=10)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()

        # Carica il checkpoint
        sd = torch.load(ckpt_path, map_location=device)

        # Gestione formato — se fallisce, stampa sd.keys() e adatta
        if isinstance(sd, dict):
            if "model" in sd:
                sd = sd["model"]
            elif "state_dict" in sd:
                sd = sd["state_dict"]
            # altrimenti sd è già lo state_dict

        model.load_state_dict(sd)
        model.eval().to(device)

        plot_umap_embedding(
            model      = model,
            model_name = f"Student {method.upper()} — seed 42",
            device     = device,
            save_path  = f"{method}_umap.png",
        )

    # ── Teacher ──────────────────────────────────────────────────────────
    # Decommentare e adattare dopo aver ricaricato/convertito il checkpoint.
    #
    # Opzione A — checkpoint .pt convertito da .bin:
    #   from model import ResNet50_CIFAR, load_model
    #   from config import MODEL_CKPT
    #   teacher = load_model(MODEL_CKPT, device)
    #   plot_umap_embedding(teacher, "Teacher ResNet-50", device, save_path="teacher_umap.png")
    #
    # Opzione B — ricarica da HuggingFace e salva:
    #   from huggingface_hub import hf_hub_download
    #   # poi carica con load_model come sopra