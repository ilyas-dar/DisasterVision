"""
train.py
========
Training loop for DisasterVision Siamese CNN.
Runs locally (CPU) or on Google Colab (GPU / mixed precision).

Usage:
    python training/train.py

All configuration is read from config.py.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from config import (
    BATCH_SIZE, BEST_MODEL_PATH, CHECKPOINT_DIR, CLASS_MAPPING_PATH,
    CLASS_NAMES, DATASET_ROOT, DROPOUT_RATE, EARLY_STOP_PATIENCE,
    FINAL_MODEL_PATH, IMAGE_SIZE, LEARNING_RATE, MODEL_CONFIG_PATH,
    NUM_CLASSES, NUM_EPOCHS, PREPROC_CFG_PATH, PROCESSED_DIR,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, WEIGHT_DECAY, XBD_LABEL_MAP,
    OUTPUTS_DIR,
)
from models.siamese_cnn import SiameseDamageClassifier, save_model_config
from preprocessing.annotation_parser import collect_dataset_pairs
from preprocessing.crop_extractor import batch_extract_crops
from preprocessing.dataset_loader import (
    compute_class_weights,
    make_dataloaders,
    make_weighted_sampler,
    print_class_distribution,
    save_metadata,
    split_by_disaster_event,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU: %s  |  Memory: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        device = torch.device("cpu")
        logger.info("No GPU found – training on CPU (slower)")
    return device


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for pre, post, labels, _ in loader:
        pre, post, labels = pre.to(device), post.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with autocast():
                logits = model(pre, post)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pre, post)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds      = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for pre, post, labels, _ in loader:
        pre, post, labels = pre.to(device), post.to(device), labels.to(device)
        logits = model(pre, post)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds      = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED)
    device = setup_device()
    use_amp = device.type == "cuda"

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    meta_path = PROCESSED_DIR / "dataset_metadata.json"

    if meta_path.exists():
        logger.info("Loading cached metadata from %s", meta_path)
        from preprocessing.dataset_loader import load_metadata
        all_records = load_metadata(meta_path)
    else:
        logger.info("Collecting xBD building pairs from %s", DATASET_ROOT)
        raw_records = collect_dataset_pairs(DATASET_ROOT, XBD_LABEL_MAP)

        if not raw_records:
            logger.error(
                "No building samples found! Check XBD_ROOT or set env var XBD_ROOT.\n"
                "Expected structure:\n"
                "  <XBD_ROOT>/<disaster_event>/images/*.png\n"
                "  <XBD_ROOT>/<disaster_event>/labels/*.json"
            )
            sys.exit(1)

        logger.info("Extracting paired crops …")
        all_records = batch_extract_crops(raw_records, IMAGE_SIZE)
        save_metadata(all_records, meta_path)

    logger.info("Total valid samples: %d", len(all_records))
    print_class_distribution(all_records, CLASS_NAMES)

    # ── 2. Split ─────────────────────────────────────────────────────────────
    train_recs, val_recs, test_recs = split_by_disaster_event(
        all_records, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
    )

    # ── 3. DataLoaders ───────────────────────────────────────────────────────
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory  = device.type == "cuda"

    train_loader, val_loader, test_loader = make_dataloaders(
        train_recs, val_recs, test_recs,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ── 4. Model ─────────────────────────────────────────────────────────────
    model = SiameseDamageClassifier(
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE,
        pretrained=True,
        freeze_backbone=True,   # phase 1: train classifier only
    ).to(device)

    # Class-weighted loss
    class_weights = compute_class_weights(train_recs, NUM_CLASSES).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = GradScaler() if use_amp else None

    # ── 5. Training loop ─────────────────────────────────────────────────────
    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_cnt  = 0
    start_time    = time.time()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Phase 2 fine-tuning: unfreeze backbone after epoch 3
        if epoch == 4:
            logger.info("Unfreezing backbone for fine-tuning …")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE / 5, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - epoch + 1
            )

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  (%.1fs)",
            epoch, NUM_EPOCHS, tr_loss, tr_acc, vl_loss, vl_acc, elapsed
        )

        # Checkpoint – best validation loss
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_cnt  = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_loss": vl_loss, "val_acc": vl_acc},
                BEST_MODEL_PATH,
            )
            logger.info("  ✓ Best checkpoint saved  val_loss=%.4f", vl_loss)
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    total_time = time.time() - start_time
    logger.info("Training complete in %.1f seconds (%.1f min)", total_time, total_time / 60)

    # Save final model
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    save_model_config(model, MODEL_CONFIG_PATH)

    # Save class mapping
    with open(CLASS_MAPPING_PATH, "w") as fh:
        json.dump({"classes": CLASS_NAMES, "label_map": XBD_LABEL_MAP}, fh, indent=2)

    # Save preprocessing config
    with open(PREPROC_CFG_PATH, "w") as fh:
        json.dump({
            "image_size": IMAGE_SIZE,
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
        }, fh, indent=2)

    # Save training history
    with open(OUTPUTS_DIR / "training_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    logger.info("All artifacts saved to %s and %s", CHECKPOINT_DIR, OUTPUTS_DIR)
    return model, test_recs, history


if __name__ == "__main__":
    main()
