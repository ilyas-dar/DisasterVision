"""
damage_visualizer.py
====================
Visualization module for building-level disaster damage assessment.

Outputs colour-coded overlays showing per-building predicted damage severity.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Damage class → BGR colour for OpenCV overlays
DAMAGE_BGR = {
    0: (46,  204, 113),   # No Damage    → green
    1: (241, 196,  15),   # Minor Damage → yellow
    2: (230, 126,  34),   # Major Damage → orange
    3: (231,  76,  60),   # Destroyed    → red
}

DAMAGE_HEX = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
}

CLASS_NAMES = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]


# ──────────────────────────────────────────────────────────────────────────────
# Core drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_damage_overlay(
    image: np.ndarray,
    buildings: List[Dict],
    use_gt: bool = False,
    alpha: float = 0.35,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw colour-coded building outlines / filled polygons on `image`.

    Parameters
    ----------
    image     : H×W×3 RGB numpy array
    buildings : list of dicts with keys:
                  - 'pre_polygon' or 'post_polygon' (or 'polygon')
                  - 'damage_class'    (ground truth, int 0-3)
                  - 'predicted_class' (model prediction, int 0-3)
    use_gt    : if True draw ground-truth labels; else draw predictions
    alpha     : fill transparency
    thickness : outline thickness in pixels

    Returns HxWx3 RGB numpy array with overlaid buildings.
    """
    vis_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    overlay  = vis_bgr.copy()

    for b in buildings:
        cls  = int(b["damage_class"] if use_gt else b.get("predicted_class", b["damage_class"]))
        poly = b.get("post_polygon", b.get("pre_polygon", None))
        if poly is None:
            continue
        color = DAMAGE_BGR.get(cls, (200, 200, 200))
        pts   = poly.reshape(-1, 1, 2)

        # Filled polygon on overlay
        cv2.fillPoly(overlay, [pts], color)
        # Solid outline on vis
        cv2.polylines(vis_bgr, [pts], True, color, thickness)

        # Confidence label
        if "confidence" in b and not use_gt:
            cx = int(np.mean(poly[:, 0]))
            cy = int(np.mean(poly[:, 1]))
            conf_txt = f"{b['confidence']:.0%}"
            cv2.putText(vis_bgr, conf_txt, (cx - 12, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Blend fill
    cv2.addWeighted(overlay, alpha, vis_bgr, 1 - alpha, 0, vis_bgr)
    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


def add_legend(ax: plt.Axes, title: str = "Damage Severity"):
    patches = [
        mpatches.Patch(color=DAMAGE_HEX[i], label=CLASS_NAMES[i])
        for i in range(4)
    ]
    ax.legend(handles=patches, title=title, loc="upper right",
              fontsize=8, title_fontsize=9)


# ──────────────────────────────────────────────────────────────────────────────
# Figure: side-by-side pre / post damage map
# ──────────────────────────────────────────────────────────────────────────────

def plot_damage_map(
    pre_image:  np.ndarray,
    post_image: np.ndarray,
    buildings:  List[Dict],
    out_path:   Optional[Path] = None,
    title:      str = "DisasterVision – Building Damage Assessment",
) -> plt.Figure:
    """
    Create a 3-panel figure:
      Left   : pre-disaster image with annotation outlines
      Centre : post-disaster image with predicted damage colours
      Right  : post-disaster image with ground-truth damage colours
    """
    pred_overlay = draw_damage_overlay(post_image, buildings, use_gt=False)
    gt_overlay   = draw_damage_overlay(post_image, buildings, use_gt=True)
    pre_outline  = draw_damage_overlay(pre_image,  buildings, use_gt=True,  alpha=0.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(pre_outline); axes[0].set_title("Pre-Disaster (buildings)")
    axes[0].axis("off"); add_legend(axes[0])

    axes[1].imshow(pred_overlay); axes[1].set_title("Post-Disaster – Predicted")
    axes[1].axis("off"); add_legend(axes[1])

    axes[2].imshow(gt_overlay); axes[2].set_title("Post-Disaster – Ground Truth")
    axes[2].axis("off"); add_legend(axes[2])

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Damage map saved → %s", out_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure: individual building cards
# ──────────────────────────────────────────────────────────────────────────────

def plot_building_cards(
    buildings: List[Dict],
    max_cards: int = 12,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Display a grid of building crop pairs (pre | post) with damage predictions.
    """
    n = min(len(buildings), max_cards)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.5 * n))
    if n == 1:
        axes = [axes]

    for i, b in enumerate(buildings[:n]):
        pre_crop  = b.get("pre_crop",  np.zeros((64, 64, 3), dtype=np.uint8))
        post_crop = b.get("post_crop", np.zeros((64, 64, 3), dtype=np.uint8))
        cls       = b.get("predicted_class", b.get("damage_class", 0))
        label     = CLASS_NAMES[cls]
        conf_str  = f" ({b['confidence']:.1%})" if "confidence" in b else ""
        color     = DAMAGE_HEX[cls]

        axes[i][0].imshow(pre_crop); axes[i][0].set_title("Pre", fontsize=8); axes[i][0].axis("off")
        axes[i][1].imshow(post_crop)
        axes[i][1].set_title(f"Post → {label}{conf_str}", fontsize=8, color=color)
        axes[i][1].axis("off")

    plt.suptitle("Building-Level Damage Predictions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        logger.info("Building cards saved → %s", out_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Summary donut / bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_damage_distribution(buildings: List[Dict], out_path: Optional[Path] = None) -> plt.Figure:
    from collections import Counter
    counts = Counter(b.get("predicted_class", b.get("damage_class", 0)) for b in buildings)
    vals   = [counts.get(i, 0) for i in range(4)]
    colors = [DAMAGE_HEX[i] for i in range(4)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(CLASS_NAMES, vals, color=colors, edgecolor="black", alpha=0.9)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=11)
    ax.set_title("Predicted Damage Distribution", fontsize=13)
    ax.set_ylabel("Building Count")
    ax.set_xlabel("Damage Category")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig
