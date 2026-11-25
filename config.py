"""
DisasterVision – Central Configuration
======================================
All environment-specific paths and hyper-parameters are set here.
Edit DATASET_ROOT to point to your local or Colab xBD extraction directory.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# ROOT PATHS  (platform-agnostic via pathlib)
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# xBD dataset root – override with env var XBD_ROOT or edit below
# xBD dataset root – override with env var XBD_ROOT or edit below.
#
# Supported layouts (auto-detected):
#   Layout A – Event subdirs (GeoTIFF full download):
#       <DATASET_ROOT>/<disaster_event>/images/*.png
#       <DATASET_ROOT>/<disaster_event>/labels/*.json
#
#   Layout B – Challenge flat (train_images_labels_targets.tar.gz):
#       <DATASET_ROOT>/images/<event>_<tile>_pre_disaster.png
#       <DATASET_ROOT>/labels/<event>_<tile>_pre_disaster.json
#
# For the Challenge training download, point DATASET_ROOT at the
# extracted "train/" directory:
#   e.g. data/raw/train   or   /content/drive/MyDrive/xBD/train
DATASET_ROOT = Path(os.environ.get("XBD_ROOT", PROJECT_ROOT / "data" / "raw" / "train"))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Auto-create runtime directories
for _d in [PROCESSED_DIR, OUTPUTS_DIR, CHECKPOINT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# ⚡ QUICK-TRAIN MODE  (set MAX_SAMPLES > 0 to cap dataset size)
# ─────────────────────────────────────────────
# Set MAX_SAMPLES to a small number (e.g. 3000) to train fast.
# Set MAX_TILES   to limit how many image tiles are even opened.
# Both caps are applied before any train/val/test split.
# Set either to 0 (zero) to use the FULL dataset (no cap).
MAX_SAMPLES = 3000   # hard cap on total building samples  (0 = unlimited)
MAX_TILES   = 60     # stop parsing after this many tiles  (0 = unlimited)

# ─────────────────────────────────────────────
# MODEL / TRAINING HYPER-PARAMETERS
# ─────────────────────────────────────────────
IMAGE_SIZE   = 128          # crop resize (px); use 224 for higher accuracy
BATCH_SIZE   = 32
NUM_EPOCHS   = 8            # 8 epochs is enough for a capped subset
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
DROPOUT_RATE  = 0.4
EARLY_STOP_PATIENCE = 3
BACKBONE     = "resnet18"   # shared Siamese backbone

# ─────────────────────────────────────────────
# CLASS DEFINITIONS
# ─────────────────────────────────────────────
DAMAGE_CLASSES = {
    "no-damage":    0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed":    3,
}
CLASS_NAMES = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]
CLASS_COLORS = {
    "No Damage":    "#2ecc71",   # green
    "Minor Damage": "#f1c40f",   # yellow
    "Major Damage": "#e67e22",   # orange
    "Destroyed":    "#e74c3c",   # red
}
NUM_CLASSES = 4

# xBD label string → class index mapping
XBD_LABEL_MAP = {
    "no-damage":        0,
    "minor-damage":     1,
    "major-damage":     2,
    "destroyed":        3,
    "un-classified":    None,   # excluded
}

# ─────────────────────────────────────────────
# DATA SPLIT  (event-level, not random crop)
# ─────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# IMAGENET NORMALISATION
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# CHECKPOINT PATHS
# ─────────────────────────────────────────────
BEST_MODEL_PATH    = CHECKPOINT_DIR / "best_model.pth"
FINAL_MODEL_PATH   = CHECKPOINT_DIR / "final_model.pth"
MODEL_CONFIG_PATH  = CHECKPOINT_DIR / "model_config.json"
CLASS_MAPPING_PATH = CHECKPOINT_DIR / "class_mapping.json"
PREPROC_CFG_PATH   = CHECKPOINT_DIR / "preprocessing_config.json"

# ─────────────────────────────────────────────
# GOOGLE COLAB  (auto-detected at runtime)
# ─────────────────────────────────────────────
IN_COLAB = os.path.exists("/content")

COLAB_DRIVE_ROOT   = Path("/content/drive/MyDrive/DisasterVision")
COLAB_DATASET_ROOT = Path("/content/drive/MyDrive/xBD")   # mount your dataset here
