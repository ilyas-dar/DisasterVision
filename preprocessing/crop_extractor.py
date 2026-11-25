"""
crop_extractor.py
=================
Extract paired pre/post-disaster building crops from full satellite tiles
and prepare them for the Siamese CNN training pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """Load image as HxWx3 RGB numpy array. Returns None on failure."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cv2 could not read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.error("Failed to load image %s: %s", path, exc)
        return None


def extract_crop(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad: int = 4,
) -> np.ndarray:
    """
    Crop a bounding-box region from an image with optional padding.

    Parameters
    ----------
    image : H×W×3 numpy array
    bbox  : (x1, y1, x2, y2) in pixel coordinates
    pad   : extra border pixels

    Returns
    -------
    Cropped numpy array (may be smaller than expected at image borders)
    """
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2].copy()


def resize_crop(crop: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize crop to (size, size) using INTER_AREA for downscaling."""
    if crop.shape[0] == size and crop.shape[1] == size:
        return crop
    interp = cv2.INTER_AREA if (crop.shape[0] > size or crop.shape[1] > size) else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interp)


def extract_paired_crops(
    record: Dict,
    crop_size: int = 128,
    pad: int = 4,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Given a building record (from annotation_parser.collect_dataset_pairs),
    load both satellite tiles and extract the corresponding building crops.

    Returns (pre_crop, post_crop) as resized numpy arrays, or None on failure.
    """
    pre_img  = load_image_rgb(record["pre_image_path"])
    post_img = load_image_rgb(record["post_image_path"])
    if pre_img is None or post_img is None:
        return None

    pre_crop  = extract_crop(pre_img,  record["pre_bbox"],  pad)
    post_crop = extract_crop(post_img, record["post_bbox"], pad)

    if pre_crop.size == 0 or post_crop.size == 0:
        return None

    pre_crop  = resize_crop(pre_crop,  crop_size)
    post_crop = resize_crop(post_crop, crop_size)
    return pre_crop, post_crop


def batch_extract_crops(
    records: List[Dict],
    crop_size: int = 128,
    pad: int = 4,
    cache_images: bool = True,
) -> List[Dict]:
    """
    Extract paired crops for all records. Returns a filtered list of records
    that now include 'pre_crop' and 'post_crop' numpy arrays.

    Uses a tile-level image cache to avoid redundant disk reads.
    """
    image_cache: Dict[Path, Optional[np.ndarray]] = {}

    def get_image(path: Path) -> Optional[np.ndarray]:
        if cache_images:
            if path not in image_cache:
                image_cache[path] = load_image_rgb(path)
            return image_cache[path]
        return load_image_rgb(path)

    valid = []
    for rec in records:
        pre_img  = get_image(rec["pre_image_path"])
        post_img = get_image(rec["post_image_path"])
        if pre_img is None or post_img is None:
            continue

        pre_crop  = resize_crop(extract_crop(pre_img,  rec["pre_bbox"],  pad), crop_size)
        post_crop = resize_crop(extract_crop(post_img, rec["post_bbox"], pad), crop_size)

        if pre_crop.size == 0 or post_crop.size == 0:
            continue

        new_rec = dict(rec)
        new_rec["pre_crop"]  = pre_crop
        new_rec["post_crop"] = post_crop
        # Convert Path objects to strings for JSON serialisability
        new_rec["pre_image_path"]  = str(rec["pre_image_path"])
        new_rec["post_image_path"] = str(rec["post_image_path"])
        valid.append(new_rec)

    logger.info("Extracted %d / %d valid paired crops", len(valid), len(records))
    return valid
