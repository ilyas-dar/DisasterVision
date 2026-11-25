"""
damage_predictor.py
===================
Single-building and batch inference using a loaded Siamese CNN checkpoint.
Works on both Colab (GPU) and Windows (CPU / GPU).
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CLASS_COLORS, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
from models.siamese_cnn import load_model, SiameseDamageClassifier

logger = logging.getLogger(__name__)

DAMAGE_COLORS_BGR = {
    "No Damage":    (46, 204, 113),    # green
    "Minor Damage": (241, 196, 15),    # yellow
    "Major Damage": (230, 126, 34),    # orange
    "Destroyed":    (231, 76, 60),     # red
}


class DamagePredictor:
    """
    Wraps the trained Siamese CNN and provides high-level prediction methods.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        image_size: int = IMAGE_SIZE,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device     = device
        self.image_size = image_size
        self.class_names = CLASS_NAMES

        self.model = load_model(checkpoint_path, config_path, device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        logger.info(
            "DamagePredictor ready – device=%s  checkpoint=%s",
            device, checkpoint_path
        )

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """Convert HxWx3 uint8 numpy array → (1,3,H,W) float tensor."""
        return self.transform(crop).unsqueeze(0).to(self.device)

    def predict_one(
        self,
        pre_crop:  np.ndarray,
        post_crop: np.ndarray,
    ) -> Dict:
        """
        Predict damage for a single building.

        Parameters
        ----------
        pre_crop  : HxWx3 uint8 RGB numpy array (pre-disaster)
        post_crop : HxWx3 uint8 RGB numpy array (post-disaster)

        Returns
        -------
        dict with keys: predicted_class, class_name, confidence, probabilities
        """
        pre_t  = self._preprocess(pre_crop)
        post_t = self._preprocess(post_crop)

        with torch.no_grad():
            logits = self.model(pre_t, post_t)
            probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_class = int(np.argmax(probs))
        return {
            "predicted_class": pred_class,
            "class_name":      self.class_names[pred_class],
            "confidence":      float(probs[pred_class]),
            "probabilities":   {name: float(p) for name, p in zip(self.class_names, probs)},
        }

    def predict_batch(
        self,
        building_records: List[Dict],
    ) -> List[Dict]:
        """
        Predict damage for a list of building records.

        Each record must have 'pre_crop' and 'post_crop' numpy arrays.
        Returns the input records augmented with prediction fields.
        """
        results = []
        for rec in building_records:
            pred = self.predict_one(rec["pre_crop"], rec["post_crop"])
            result = dict(rec)
            result.update(pred)
            results.append(result)
        return results

    def predict_from_tile(
        self,
        pre_image:  np.ndarray,
        post_image: np.ndarray,
        buildings:  List[Dict],
        crop_size:  int = IMAGE_SIZE,
        pad:        int = 4,
    ) -> List[Dict]:
        """
        Given full satellite tiles and a list of building dicts (each with
        'pre_bbox' and 'post_bbox'), extract crops and predict damage.

        Parameters
        ----------
        pre_image  : H×W×3 numpy array (RGB)
        post_image : H×W×3 numpy array (RGB)
        buildings  : list of dicts with pre_bbox / post_bbox keys
        """
        from preprocessing.crop_extractor import extract_crop, resize_crop
        results = []
        for b in buildings:
            pre_crop  = resize_crop(extract_crop(pre_image,  b["pre_bbox"],  pad), crop_size)
            post_crop = resize_crop(extract_crop(post_image, b["post_bbox"], pad), crop_size)
            pred = self.predict_one(pre_crop, post_crop)
            result = dict(b)
            result["pre_crop"]  = pre_crop
            result["post_crop"] = post_crop
            result.update(pred)
            results.append(result)
        return results
