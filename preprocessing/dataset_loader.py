"""
dataset_loader.py
=================
PyTorch Dataset and DataLoader factories for the DisasterVision
Siamese CNN training pipeline.
"""

import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = 128):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transform(image_size: int = 128):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class XBDPairedCropDataset(Dataset):
    """
    PyTorch Dataset for xBD paired pre/post-disaster building crops.

    Each item returns:
        pre_tensor  : (3, H, W) float tensor
        post_tensor : (3, H, W) float tensor
        label       : int  (0–3)
        uid         : str  (building identifier)
    """

    def __init__(
        self,
        records: List[Dict],
        transform=None,
        image_size: int = 128,
        is_train: bool = True,
    ):
        self.records    = records
        self.transform  = transform or (
            get_train_transform(image_size) if is_train
            else get_val_transform(image_size)
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        rec = self.records[idx]

        # Crops are stored as numpy arrays (H,W,3) RGB
        pre_arr  = rec["pre_crop"]
        post_arr = rec["post_crop"]

        if isinstance(pre_arr, str):
            raise ValueError(f"Record {idx}: pre_crop is a path string, expected numpy array")

        pre_tensor  = self.transform(pre_arr)
        post_tensor = self.transform(post_arr)
        label       = int(rec["damage_class"])
        uid         = rec.get("uid", str(idx))

        return pre_tensor, post_tensor, label, uid


# ──────────────────────────────────────────────────────────────────────────────
# Event-level train / val / test split
# ──────────────────────────────────────────────────────────────────────────────

def split_by_disaster_event(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split records by disaster event to prevent data leakage.
    Buildings from the same event stay together.
    """
    rng = np.random.default_rng(seed)

    # Group by event
    events: Dict[str, List[Dict]] = {}
    for rec in records:
        ev = rec.get("disaster_event", "unknown")
        events.setdefault(ev, []).append(rec)

    event_names = sorted(events.keys())
    rng.shuffle(event_names)

    n = len(event_names)
    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))

    train_events = event_names[:n_train]
    val_events   = event_names[n_train:n_train + n_val]
    test_events  = event_names[n_train + n_val:]

    # Fallback: if too few events, do record-level split
    if len(train_events) < 1 or len(val_events) < 1:
        logger.warning("Too few events for event-level split; falling back to record-level split.")
        idx = np.arange(len(records))
        rng.shuffle(idx)
        n_tr = int(len(records) * train_ratio)
        n_vl = int(len(records) * val_ratio)
        return (
            [records[i] for i in idx[:n_tr]],
            [records[i] for i in idx[n_tr:n_tr + n_vl]],
            [records[i] for i in idx[n_tr + n_vl:]],
        )

    def gather(evs):
        out = []
        for e in evs:
            out.extend(events[e])
        return out

    train_recs = gather(train_events)
    val_recs   = gather(val_events)
    test_recs  = gather(test_events)

    logger.info(
        "Split: train=%d (events=%d)  val=%d (events=%d)  test=%d (events=%d)",
        len(train_recs), len(train_events),
        len(val_recs),   len(val_events),
        len(test_recs),  len(test_events),
    )
    return train_recs, val_recs, test_recs


# ──────────────────────────────────────────────────────────────────────────────
# Class weights & sampler for imbalanced datasets
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(records: List[Dict], num_classes: int = 4) -> torch.Tensor:
    counts = Counter(r["damage_class"] for r in records)
    total  = sum(counts.values())
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        weights.append(total / (num_classes * max(cnt, 1)))
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(records: List[Dict], num_classes: int = 4) -> WeightedRandomSampler:
    class_w = compute_class_weights(records, num_classes)
    sample_weights = [class_w[r["damage_class"]].item() for r in records]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    train_records: List[Dict],
    val_records:   List[Dict],
    test_records:  List[Dict],
    image_size:  int = 128,
    batch_size:  int = 32,
    num_workers: int = 0,
    pin_memory:  bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = XBDPairedCropDataset(train_records, is_train=True,  image_size=image_size)
    val_ds   = XBDPairedCropDataset(val_records,   is_train=False, image_size=image_size)
    test_ds  = XBDPairedCropDataset(test_records,  is_train=False, image_size=image_size)

    sampler = make_weighted_sampler(train_records)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Metadata serialisation (avoid re-extracting crops on restart)
# ──────────────────────────────────────────────────────────────────────────────

def save_metadata(records: List[Dict], out_path: Path):
    """Save processed dataset metadata (paths + labels, NOT crop pixels)."""
    meta = []
    for r in records:
        meta.append({
            "uid":            r["uid"],
            "tile_id":        r.get("tile_id", ""),
            "disaster_event": r.get("disaster_event", ""),
            "pre_image_path": str(r.get("pre_image_path", "")),
            "post_image_path":str(r.get("post_image_path", "")),
            "pre_bbox":       list(r["pre_bbox"]),
            "post_bbox":      list(r["post_bbox"]),
            "damage_label":   r["damage_label"],
            "damage_class":   int(r["damage_class"]),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Saved %d metadata records to %s", len(meta), out_path)


def load_metadata(path: Path) -> List[Dict]:
    with open(path, "r") as fh:
        raw = json.load(fh)
    # Convert bbox lists back to tuples
    for r in raw:
        r["pre_bbox"]  = tuple(r["pre_bbox"])
        r["post_bbox"] = tuple(r["post_bbox"])
        r["pre_image_path"]  = Path(r["pre_image_path"])
        r["post_image_path"] = Path(r["post_image_path"])
    return raw


def print_class_distribution(records: List[Dict], class_names: List[str]):
    counts = Counter(r["damage_class"] for r in records)
    total  = sum(counts.values())
    print(f"\n{'Class':<20} {'Count':>8} {'%':>8}")
    print("-" * 38)
    for i, name in enumerate(class_names):
        cnt = counts.get(i, 0)
        print(f"{name:<20} {cnt:>8} {cnt/max(total,1)*100:>7.1f}%")
    print(f"{'TOTAL':<20} {total:>8}")
