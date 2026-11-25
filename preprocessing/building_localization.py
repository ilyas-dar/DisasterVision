"""
building_localization.py
========================
Module for building localization – annotation-assisted (primary path) and
a stub for future automatic detector integration.

This module is intentionally modular: replace or augment the
`AutomaticBuildingDetector` class when a compatible pretrained model
(e.g. SegFormer, DeepLab) becomes available.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Annotation-Assisted Localizer (primary – uses xBD ground-truth polygons)
# ──────────────────────────────────────────────────────────────────────────────

class AnnotationLocalizer:
    """
    Localizes buildings using xBD ground-truth polygon annotations.

    NOTE: This module uses ground-truth annotations, not automatic detection.
    Building bounding boxes are derived from the provided polygon coordinates.
    This is the standard approach during training and evaluation.
    """

    MODE = "annotation-assisted"

    def get_building_regions(
        self,
        image: np.ndarray,
        polygons: List[np.ndarray],
        bboxes: List[Tuple[int, int, int, int]],
        pad: int = 4,
    ) -> List[np.ndarray]:
        """
        Extract individual building crops from `image` using bounding boxes.

        Parameters
        ----------
        image    : H×W×3 numpy array (RGB)
        polygons : list of (N,2) int arrays (not used for extraction, kept for future mask support)
        bboxes   : list of (x1, y1, x2, y2) tuples
        pad      : extra padding around the bounding box (pixels)

        Returns
        -------
        List of cropped numpy arrays (variable size, RGB)
        """
        H, W = image.shape[:2]
        crops = []
        for (x1, y1, x2, y2) in bboxes:
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(W, x2 + pad)
            y2 = min(H, y2 + pad)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crops.append(np.zeros((32, 32, 3), dtype=np.uint8))
            else:
                crops.append(crop)
        return crops

    def draw_buildings(
        self,
        image: np.ndarray,
        polygons: List[np.ndarray],
        labels: Optional[List[str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw building polygons on `image` and return a copy."""
        vis = image.copy()
        default_color = (255, 255, 0)
        for i, poly in enumerate(polygons):
            color = colors[i] if colors else default_color
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, color, thickness)
            if labels:
                cx = int(np.mean(poly[:, 0]))
                cy = int(np.mean(poly[:, 1]))
                cv2.putText(
                    vis, labels[i], (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
                )
        return vis


# ──────────────────────────────────────────────────────────────────────────────
# Automatic Detector Stub (future integration point)
# ──────────────────────────────────────────────────────────────────────────────

class AutomaticBuildingDetector:
    """
    Placeholder for a pretrained automatic building detector.

    Replace the `detect` method body with real inference when a compatible
    model checkpoint is available (e.g., SpaceNet pretrained SegFormer).
    """

    MODE = "automatic-detection"
    AVAILABLE = False   # Set True after integrating real weights

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path
        if not self.AVAILABLE:
            logger.warning(
                "AutomaticBuildingDetector: no pretrained weights available. "
                "Falling back to annotation-assisted localization."
            )

    def detect(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple]]:
        """
        Detect buildings in `image`.

        Returns
        -------
        polygons : list of (N,2) int arrays
        bboxes   : list of (x1,y1,x2,y2)
        """
        raise NotImplementedError(
            "Automatic building detection is not yet implemented. "
            "Please provide xBD annotation files for localization."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_localizer(mode: str = "annotation"):
    """
    Return the appropriate localizer.

    mode : "annotation" → AnnotationLocalizer (default, uses xBD polygons)
           "auto"       → AutomaticBuildingDetector (raises if unavailable)
    """
    if mode == "auto":
        det = AutomaticBuildingDetector()
        if not det.AVAILABLE:
            logger.warning("Auto detector unavailable; using annotation-assisted mode.")
            return AnnotationLocalizer()
        return det
    return AnnotationLocalizer()
