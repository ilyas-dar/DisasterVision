"""
building_detector.py
====================
Thin wrapper that selects the appropriate building localization backend.

For automatic detection, integrate a pretrained model here (e.g. SpaceNet
pretrained SegFormer / DeepLab). Until then, all localization uses
annotation-assisted mode.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.building_localization import AnnotationLocalizer, get_localizer

logger = logging.getLogger(__name__)


class BuildingDetector:
    """
    Unified interface for building detection / localization.

    Modes
    -----
    "annotation" : uses xBD ground-truth polygon annotations (default)
    "auto"       : uses automatic detection model (requires pretrained weights)
    """

    def __init__(self, mode: str = "annotation"):
        self.localizer = get_localizer(mode)
        self.mode      = self.localizer.MODE
        logger.info("BuildingDetector running in '%s' mode", self.mode)

    def localize(
        self,
        image: np.ndarray,
        polygons: Optional[List[np.ndarray]] = None,
        bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Localize buildings and return (polygons, bboxes).

        In annotation mode, polygons/bboxes must be provided.
        In auto mode, the image is processed by the detector.
        """
        if self.mode == "annotation-assisted":
            if polygons is None or bboxes is None:
                raise ValueError(
                    "Annotation-assisted localization requires 'polygons' and 'bboxes' arguments. "
                    "Please provide xBD annotation files."
                )
            return polygons, bboxes
        else:
            return self.localizer.detect(image)

    def draw_predictions(
        self,
        image: np.ndarray,
        polygons: List[np.ndarray],
        labels: Optional[List[str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Draw building outlines and optional damage labels on the image."""
        return self.localizer.draw_buildings(image, polygons, labels, colors)
