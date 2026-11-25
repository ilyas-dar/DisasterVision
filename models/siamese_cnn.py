"""
siamese_cnn.py
==============
Siamese CNN for pre/post-disaster building damage classification.

Architecture
------------
  Branch A (pre-disaster crop)  ─┐
  Branch B (post-disaster crop)  ─┤  SHARED ResNet18 backbone
                                  │
  Feature Fusion: [f_pre | f_post | |f_pre − f_post|]
                                  │
  Classification Head: FC → ReLU → Dropout → FC(4)
                                  │
  Output: 4-class damage severity (softmax at inference)
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────────────────────────────────────
# Shared Backbone
# ──────────────────────────────────────────────────────────────────────────────

class SharedResNet18Backbone(nn.Module):
    """
    ResNet18 pretrained on ImageNet with the final FC layer replaced by
    an identity so it outputs a feature vector of size 512.
    """

    FEATURE_DIM = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base    = models.resnet18(weights=weights)
        # Remove the classification head; keep everything up to avgpool
        self.features = nn.Sequential(*list(base.children())[:-1])  # → (B, 512, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)   # (B, 512)


# ──────────────────────────────────────────────────────────────────────────────
# Siamese CNN
# ──────────────────────────────────────────────────────────────────────────────

class SiameseDamageClassifier(nn.Module):
    """
    Two-branch Siamese network that shares a single ResNet18 backbone.

    Forward pass:
        pre_img  : (B, 3, H, W)  pre-disaster building crop
        post_img : (B, 3, H, W)  post-disaster building crop

    Output:
        logits   : (B, 4)  raw logits for [No Damage, Minor, Major, Destroyed]
    """

    def __init__(
        self,
        num_classes:  int   = 4,
        dropout:      float = 0.4,
        pretrained:   bool  = True,
        hidden_dim:   int   = 512,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = SharedResNet18Backbone(pretrained=pretrained)
        feat_dim      = SharedResNet18Backbone.FEATURE_DIM  # 512

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Fusion: concatenate [pre | post | |pre − post|] → dim = 512*3 = 1536
        fusion_dim = feat_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.num_classes = num_classes

    # ── Branch (shared weights) ──────────────────────────────────────────────
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run one branch through the shared backbone."""
        return self.backbone(x)   # (B, 512)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(
        self,
        pre_img:  torch.Tensor,
        post_img: torch.Tensor,
    ) -> torch.Tensor:
        f_pre  = self.extract_features(pre_img)   # (B, 512)
        f_post = self.extract_features(post_img)  # (B, 512)
        diff   = torch.abs(f_pre - f_post)        # (B, 512)

        fused  = torch.cat([f_pre, f_post, diff], dim=1)   # (B, 1536)
        logits = self.classifier(fused)                     # (B, 4)
        return logits

    # ── Inference helper ─────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        pre_img:  torch.Tensor,
        post_img: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (class_indices, probabilities) for a batch.

        Returns
        -------
        preds : (B,) int64
        probs : (B, 4) float32
        """
        self.eval()
        logits = self(pre_img, post_img)
        probs  = F.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)
        return preds, probs

    # ── Unfreeze backbone for fine-tuning ────────────────────────────────────
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_model_config(model: SiameseDamageClassifier, path: Path):
    """Save hyper-parameters to JSON so the model can be re-instantiated."""
    cfg = {
        "architecture": "SiameseDamageClassifier",
        "backbone":     "resnet18",
        "num_classes":  model.num_classes,
        "feature_dim":  SharedResNet18Backbone.FEATURE_DIM,
        "fusion":       "concat(pre, post, abs_diff)",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)


def load_model(
    checkpoint_path: Path,
    config_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    num_classes: int = 4,
) -> SiameseDamageClassifier:
    """
    Load a SiameseDamageClassifier from a .pth checkpoint.

    Parameters
    ----------
    checkpoint_path : path to best_model.pth
    config_path     : optional path to model_config.json
    device          : torch device (auto-detected if None)
    num_classes     : fallback if config_path not provided
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_path and config_path.exists():
        with open(config_path) as fh:
            cfg = json.load(fh)
        num_classes = cfg.get("num_classes", num_classes)

    model = SiameseDamageClassifier(num_classes=num_classes, pretrained=False)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle both raw state_dict and checkpoint dicts
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
