"""
evaluate.py
===========
Post-training evaluation: confusion matrix, classification report,
training curves, class distribution chart, damage visualisation.

Usage (after training):
    python training/evaluate.py

All outputs are written to outputs/
"""

import json
import logging
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    BEST_MODEL_PATH, CLASS_NAMES, IMAGE_SIZE, MODEL_CONFIG_PATH,
    NUM_CLASSES, OUTPUTS_DIR, PROCESSED_DIR, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO,
)
from models.siamese_cnn import load_model
from preprocessing.dataset_loader import (
    XBDPairedCropDataset,
    get_val_transform,
    load_metadata,
    make_dataloaders,
    split_by_disaster_event,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for pre, post, labels, _ in loader:
        pre, post = pre.to(device), post.to(device)
        logits = model(pre, post)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title("DisasterVision – Four-Class Damage Confusion Matrix", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved → %s", out_path)


def plot_training_curves(history: dict, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Loss Curve"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss"); axes[0].legend(); axes[0].grid(True)
    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc")
    axes[1].set_title("Accuracy Curve"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(True)
    fig.suptitle("DisasterVision – Training History", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training curves saved → %s", out_path)


def plot_class_distribution(records: list, class_names: list, out_path):
    from collections import Counter
    counts = Counter(r["damage_class"] for r in records)
    vals   = [counts.get(i, 0) for i in range(len(class_names))]
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, vals, color=colors, edgecolor="black", alpha=0.85)
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_title("Class Distribution – Full Dataset", fontsize=13)
    ax.set_ylabel("Sample Count")
    ax.set_xlabel("Damage Category")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Class distribution saved → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load model ────────────────────────────────────────────────────────
    if not BEST_MODEL_PATH.exists():
        logger.error("No checkpoint found at %s. Run training/train.py first.", BEST_MODEL_PATH)
        sys.exit(1)

    model = load_model(BEST_MODEL_PATH, MODEL_CONFIG_PATH, device)
    logger.info("Model loaded from %s", BEST_MODEL_PATH)

    # ── 2. Load dataset metadata ─────────────────────────────────────────────
    meta_path = PROCESSED_DIR / "dataset_metadata.json"
    if not meta_path.exists():
        logger.error("Dataset metadata not found. Run training/train.py first.")
        sys.exit(1)

    all_records = load_metadata(meta_path)
    _, _, test_recs = split_by_disaster_event(all_records, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    if not test_recs:
        logger.warning("Test split is empty; using full dataset for evaluation.")
        test_recs = all_records

    logger.info("Evaluating on %d test samples", len(test_recs))

    # ── 3. Build test loader (re-extract crops) ──────────────────────────────
    from preprocessing.crop_extractor import batch_extract_crops
    test_recs_with_crops = batch_extract_crops(test_recs, IMAGE_SIZE)

    from torch.utils.data import DataLoader
    test_ds = XBDPairedCropDataset(test_recs_with_crops, is_train=False, image_size=IMAGE_SIZE)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # ── 4. Collect predictions ───────────────────────────────────────────────
    y_true, y_pred, y_probs = collect_predictions(model, test_loader, device)

    # ── 5. Classification report ─────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\n" + report)
    with open(OUTPUTS_DIR / "classification_report.txt", "w") as fh:
        fh.write(report)

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, OUTPUTS_DIR / "confusion_matrix.png")
    plot_class_distribution(all_records, CLASS_NAMES, OUTPUTS_DIR / "class_distribution.png")

    # Training curves (if history file present)
    hist_path = OUTPUTS_DIR / "training_history.json"
    if hist_path.exists():
        with open(hist_path) as fh:
            history = json.load(fh)
        plot_training_curves(history, OUTPUTS_DIR / "training_curves.png")

    # ── 7. Save evaluation metrics JSON ──────────────────────────────────────
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    metrics = {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "macro_f1":        float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1":     float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro")),
        "macro_recall":    float(recall_score(y_true, y_pred, average="macro")),
        "num_test_samples": int(len(y_true)),
        "class_names":     CLASS_NAMES,
        "per_class_f1":    f1_score(y_true, y_pred, average=None).tolist(),
    }
    with open(OUTPUTS_DIR / "evaluation_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("\nEvaluation summary:")
    logger.info("  Accuracy        : %.4f", metrics["accuracy"])
    logger.info("  Macro F1        : %.4f", metrics["macro_f1"])
    logger.info("  Weighted F1     : %.4f", metrics["weighted_f1"])
    logger.info("All outputs saved to %s", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
