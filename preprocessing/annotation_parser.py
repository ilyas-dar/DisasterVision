"""
annotation_parser.py
====================
Parse xBD JSON annotation files.

xBD annotation format (per JSON file):
  {
    "metadata": { "img_name": ..., "disaster": ..., ... },
    "features": {
      "lng_lat": [
        { "type": "Feature",
          "properties": { "uid": "...", "feature_type": "building",
                          "subtype": "no-damage" | "minor-damage" | ... },
          "geometry": { "type": "Polygon", "coordinates": [...] }
        }, ...
      ]
    }
  }

Pre-disaster JSONs contain no damage labels (or label = "no-damage").
Post-disaster JSONs carry the actual damage label in `subtype`.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def polygon_to_pixel_coords(coords: list, img_w: int, img_h: int) -> np.ndarray:
    """
    Convert xBD lat/lng polygon coordinates to pixel coordinates.

    xBD stores coordinates as *pixel* coordinates in (x, y) order inside the
    'coordinates' list, NOT geographic lat/lng (despite the 'lng_lat' key name).
    Coordinates are float values in [0, image_width] × [0, image_height].
    """
    pts = np.array(coords[0], dtype=np.float32)   # ring 0
    pts[:, 0] = np.clip(pts[:, 0], 0, img_w)
    pts[:, 1] = np.clip(pts[:, 1], 0, img_h)
    return pts.astype(np.int32)


def polygon_to_bbox(polygon: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) bounding box."""
    x_min = int(np.min(polygon[:, 0]))
    y_min = int(np.min(polygon[:, 1]))
    x_max = int(np.max(polygon[:, 0]))
    y_max = int(np.max(polygon[:, 1]))
    return x_min, y_min, x_max, y_max


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class BuildingAnnotation:
    """Container for one annotated building."""

    __slots__ = ("uid", "polygon", "bbox", "damage_label", "damage_class")

    def __init__(
        self,
        uid: str,
        polygon: np.ndarray,
        damage_label: str,
        damage_class: Optional[int],
    ):
        self.uid          = uid
        self.polygon      = polygon                      # (N, 2) int pixel coords
        self.bbox         = polygon_to_bbox(polygon)     # (x1,y1,x2,y2)
        self.damage_label = damage_label
        self.damage_class = damage_class                 # None → excluded

    def __repr__(self):
        return (
            f"BuildingAnnotation(uid={self.uid!r}, "
            f"bbox={self.bbox}, label={self.damage_label!r})"
        )


def parse_annotation_file(
    json_path: Path,
    img_w: int = 1024,
    img_h: int = 1024,
    label_map: Optional[Dict[str, Optional[int]]] = None,
) -> List[BuildingAnnotation]:
    """
    Parse a single xBD JSON annotation file and return a list of
    BuildingAnnotation objects.

    Parameters
    ----------
    json_path  : path to the JSON file
    img_w, img_h : image dimensions (pixels) for coordinate clamping
    label_map  : dict mapping damage-subtype string → class index
                 (None value means exclude that building)
    """
    if label_map is None:
        from config import XBD_LABEL_MAP
        label_map = XBD_LABEL_MAP

    data = load_json(json_path)

    features = data.get("features", {}).get("lng_lat", [])
    if not features:
        logger.warning("No features found in %s", json_path)
        return []

    buildings: List[BuildingAnnotation] = []
    for feat in features:
        props    = feat.get("properties", {})
        geom     = feat.get("geometry", {})
        uid      = props.get("uid", "unknown")
        subtype  = props.get("subtype", "un-classified").strip().lower()
        coords   = geom.get("coordinates", [])

        if geom.get("type") != "Polygon" or not coords:
            continue

        polygon = polygon_to_pixel_coords(coords, img_w, img_h)

        # Skip degenerate polygons
        if len(polygon) < 3:
            continue
        x1, y1, x2, y2 = polygon_to_bbox(polygon)
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue

        damage_class = label_map.get(subtype, None)   # None → excluded
        buildings.append(
            BuildingAnnotation(uid, polygon, subtype, damage_class)
        )

    return buildings


def parse_image_pair(
    pre_json: Path,
    post_json: Path,
    img_w: int = 1024,
    img_h: int = 1024,
    label_map: Optional[Dict[str, Optional[int]]] = None,
) -> List[Dict]:
    """
    Parse a pre/post pair of JSON files and return matched building records.

    Returns a list of dicts with keys:
        uid, pre_polygon, post_polygon, pre_bbox, post_bbox,
        damage_label, damage_class
    """
    pre_buildings  = {b.uid: b for b in parse_annotation_file(pre_json,  img_w, img_h, label_map)}
    post_buildings = {b.uid: b for b in parse_annotation_file(post_json, img_w, img_h, label_map)}

    matched = []
    for uid, post in post_buildings.items():
        if post.damage_class is None:
            continue   # excluded label
        pre = pre_buildings.get(uid)
        if pre is None:
            logger.debug("UID %s not found in pre-disaster annotations", uid)
            continue
        matched.append({
            "uid":          uid,
            "pre_polygon":  pre.polygon,
            "post_polygon": post.polygon,
            "pre_bbox":     pre.bbox,
            "post_bbox":    post.bbox,
            "damage_label": post.damage_label,
            "damage_class": post.damage_class,
        })

    return matched


def _detect_layout(dataset_root: Path) -> str:
    """
    Auto-detect xBD directory layout.

    Layout A – GeoTIFF / event-subdirectory (original xBD):
        <root>/<disaster_event>/images/<tile>_pre_disaster.png
        <root>/<disaster_event>/labels/<tile>_pre_disaster.json

    Layout B – Challenge flat (train_images_labels_targets.tar.gz):
        <root>/images/<event>_<tile>_pre_disaster.png
        <root>/labels/<event>_<tile>_pre_disaster.json

    Returns 'flat' or 'event_subdirs'.
    """
    flat_images = dataset_root / "images"
    flat_labels = dataset_root / "labels"
    if flat_images.is_dir() and flat_labels.is_dir():
        return "flat"
    return "event_subdirs"


def _extract_event_from_stem(stem: str) -> str:
    """
    For the flat challenge layout, the filename is:
        <disaster-event>_<tile_id>_pre_disaster   (stem after removing suffix)
    The tile_id is a zero-padded 8-digit number.
    Example: 'hurricane-harvey_00000013' → event = 'hurricane-harvey'
    """
    parts = stem.rsplit("_", 1)         # split on last underscore
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    # Try splitting on first occurrence of an 8-digit segment
    import re
    m = re.match(r"^(.+?)_(\d{8})$", stem)
    if m:
        return m.group(1)
    return stem                         # fallback: use whole stem


def collect_dataset_pairs(
    dataset_root: Path,
    label_map: Optional[Dict[str, Optional[int]]] = None,
    img_w: int = 1024,
    img_h: int = 1024,
) -> List[Dict]:
    """
    Walk the entire xBD dataset_root and collect ALL matched building records.

    Automatically detects two supported directory layouts:

    Layout A – Event subdirectories (GeoTIFF / full xBD download):
        <root>/<disaster_event>/images/<tile>_pre_disaster.png
        <root>/<disaster_event>/labels/<tile>_pre_disaster.json

    Layout B – Flat Challenge format (train_images_labels_targets.tar.gz):
        <root>/images/<event>_<tile>_pre_disaster.png
        <root>/labels/<event>_<tile>_pre_disaster.json

    Returns list of records each containing image paths + building info.
    """
    records: List[Dict] = []

    layout = _detect_layout(dataset_root)
    logger.info("Detected xBD layout: '%s' under %s", layout, dataset_root)

    label_dir_candidates = list(dataset_root.rglob("*_post_disaster.json"))
    if not label_dir_candidates:
        logger.warning("No xBD annotation files found under %s", dataset_root)
        return records

    logger.info("Found %d post-disaster annotation files", len(label_dir_candidates))

    for post_json in sorted(label_dir_candidates):
        stem = post_json.stem.replace("_post_disaster", "")
        pre_json = post_json.parent / f"{stem}_pre_disaster.json"
        if not pre_json.exists():
            continue

        # ── Locate image files based on detected layout ──────────────────────
        if layout == "flat":
            img_dir  = dataset_root / "images"
            disaster = _extract_event_from_stem(stem)
        else:
            img_dir  = post_json.parent.parent / "images"
            disaster = post_json.parts[-3] if len(post_json.parts) >= 3 else "unknown"

        pre_img  = img_dir / f"{stem}_pre_disaster.png"
        post_img = img_dir / f"{stem}_post_disaster.png"

        # Also try .tif / .jpg variants
        for ext in [".tif", ".jpg", ".jpeg"]:
            if not pre_img.exists():
                pre_img  = img_dir / f"{stem}_pre_disaster{ext}"
            if not post_img.exists():
                post_img = img_dir / f"{stem}_post_disaster{ext}"

        if not pre_img.exists() or not post_img.exists():
            logger.debug("Images not found for tile %s", stem)
            continue

        buildings = parse_image_pair(pre_json, post_json, img_w, img_h, label_map)
        # `disaster` already set above based on detected layout

        for b in buildings:
            b["pre_image_path"]  = pre_img
            b["post_image_path"] = post_img
            b["tile_id"]         = stem
            b["disaster_event"]  = disaster

        records.extend(buildings)

    logger.info("Collected %d valid building samples from %s", len(records), dataset_root)
    return records
