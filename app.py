"""
app.py – DisasterVision Streamlit Application
=============================================
Professional dashboard for AI-based building damage assessment.

Run:
    streamlit run app.py

The app will automatically look for a checkpoint under checkpoints/best_model.pth
(the standard export location). You can override the path in the sidebar.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Project path setup ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    BEST_MODEL_PATH, CLASS_COLORS, CLASS_NAMES, IMAGE_SIZE,
    MODEL_CONFIG_PATH, NUM_CLASSES, OUTPUTS_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DisasterVision – AI Building Damage Assessment",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-title { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-subtitle { font-size: 1rem; color: #aaa; margin-top: 0.3rem; }
    .damage-card {
        border-radius: 10px; padding: 1rem;
        text-align: center; font-weight: 600;
        font-size: 1.1rem; margin: 0.3rem 0;
    }
    .metric-box {
        background: #f0f2f6; border-radius: 8px;
        padding: 0.75rem 1rem; margin: 0.2rem 0;
    }
    .stAlert { border-radius: 8px; }
    .mode-badge {
        display: inline-block; padding: 0.2em 0.7em;
        border-radius: 4px; font-size: 0.8rem; font-weight: 600;
        background: #e8f4ea; color: #2e7d32; border: 1px solid #c8e6c9;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">🛰️ DisasterVision</div>
    <div class="main-subtitle">
        AI-Powered Building Detection &amp; Pre/Post-Disaster Damage Assessment
        using Siamese CNN on xBD Satellite Imagery
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/satellite.png", width=64)
    st.title("⚙️ Configuration")

    checkpoint_input = st.text_input(
        "Checkpoint path (.pth)",
        value=str(BEST_MODEL_PATH),
        help="Path to the trained best_model.pth file"
    )
    checkpoint_path = Path(checkpoint_input)

    crop_size = st.slider("Crop size (px)", 64, 224, IMAGE_SIZE, step=32)
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.0, step=0.05,
                                help="Hide predictions below this confidence")

    st.divider()
    st.markdown("**Localization Mode**")
    st.markdown(
        '<span class="mode-badge">📌 Annotation-Assisted</span>', unsafe_allow_html=True
    )
    st.caption(
        "Building polygons are derived from xBD annotations. "
        "Fully automatic detection will be integrated in a future release."
    )

    st.divider()
    st.markdown("**Damage Legend**")
    for cls, hex_col in CLASS_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.2rem 0;">'
            f'<div style="width:16px;height:16px;border-radius:3px;background:{hex_col};"></div>'
            f'<span>{cls}</span></div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Siamese CNN …")
def load_predictor(ckpt_path: str, cfg_path: str):
    try:
        import torch
        from inference.damage_predictor import DamagePredictor
        predictor = DamagePredictor(
            checkpoint_path=Path(ckpt_path),
            config_path=Path(cfg_path) if Path(cfg_path).exists() else None,
        )
        return predictor, None
    except Exception as exc:
        return None, str(exc)


predictor, load_error = load_predictor(str(checkpoint_path), str(MODEL_CONFIG_PATH))

if load_error:
    st.error(f"⚠️ **Model not loaded:** {load_error}")
    st.info(
        "To run the application:\n\n"
        "1. Train the model on Google Colab using `notebooks/DisasterVision_Training.ipynb`\n"
        "2. Download `DisasterVision_Model_Export.zip`\n"
        "3. Extract and place `best_model.pth` into `checkpoints/`\n"
        "4. Restart the app"
    )
    model_ready = False
else:
    model_ready = True
    st.sidebar.success("✅ Model loaded successfully")

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def damage_color_html(cls_name: str) -> str:
    return CLASS_COLORS.get(cls_name, "#999999")


def build_damage_summary(predictions: List[Dict]) -> Dict:
    counter = Counter(p["class_name"] for p in predictions)
    return {name: counter.get(name, 0) for name in CLASS_NAMES}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Image Upload
# ─────────────────────────────────────────────────────────────────────────────
st.header("📂 Section A – Upload Satellite Images")

col1, col2 = st.columns(2)
with col1:
    pre_file = st.file_uploader(
        "🌿 Pre-Disaster Image", type=["png", "jpg", "jpeg", "tif"],
        key="pre_upload", help="Satellite image taken before the disaster event"
    )
    if pre_file:
        pre_pil = Image.open(pre_file).convert("RGB")
        st.image(pre_pil, caption="Pre-Disaster", use_container_width=True)

with col2:
    post_file = st.file_uploader(
        "🔥 Post-Disaster Image", type=["png", "jpg", "jpeg", "tif"],
        key="post_upload", help="Satellite image taken after the disaster event"
    )
    if post_file:
        post_pil = Image.open(post_file).convert("RGB")
        st.image(post_pil, caption="Post-Disaster", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: Building Localization
# ─────────────────────────────────────────────────────────────────────────────
st.header("🏢 Section B – Building Localization")

localization_mode = st.radio(
    "Select localization method:",
    ["📋 Upload xBD Annotation JSON (recommended)", "✂️ Upload Individual Building Crop Pairs"],
    horizontal=True,
)

buildings_ready = False
building_records: List[Dict] = []

if "📋 Upload xBD Annotation JSON" in localization_mode:
    st.info(
        "ℹ️ **Annotation-Assisted Localization** – Upload the corresponding xBD "
        "pre and post disaster JSON annotation files to extract building locations."
    )
    col3, col4 = st.columns(2)
    with col3:
        pre_json_file = st.file_uploader(
            "Pre-disaster JSON", type=["json"], key="pre_json"
        )
    with col4:
        post_json_file = st.file_uploader(
            "Post-disaster JSON", type=["json"], key="post_json"
        )

    if pre_json_file and post_json_file and pre_file and post_file:
        try:
            import json as _json
            from preprocessing.annotation_parser import parse_image_pair
            from preprocessing.crop_extractor import extract_crop, resize_crop

            pre_data  = _json.loads(pre_json_file.read().decode())
            post_data = _json.loads(post_json_file.read().decode())

            # Save to temp files for parse_image_pair
            import tempfile, os
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_pre:
                _json.dump(pre_data, f_pre); pre_tmp = f_pre.name
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_post:
                _json.dump(post_data, f_post); post_tmp = f_post.name

            pre_arr  = pil_to_numpy(pre_pil)
            post_arr = pil_to_numpy(post_pil)
            H, W = pre_arr.shape[:2]

            matched = parse_image_pair(Path(pre_tmp), Path(post_tmp), W, H)
            os.unlink(pre_tmp); os.unlink(post_tmp)

            for b in matched:
                pre_crop  = resize_crop(extract_crop(pre_arr,  b["pre_bbox"],  4), crop_size)
                post_crop = resize_crop(extract_crop(post_arr, b["post_bbox"], 4), crop_size)
                b["pre_crop"]  = pre_crop
                b["post_crop"] = post_crop
                b["pre_image_path"]  = "uploaded"
                b["post_image_path"] = "uploaded"

            building_records = [b for b in matched if b["pre_crop"] is not None]
            st.success(f"✅ Extracted **{len(building_records)} buildings** from annotations.")
            buildings_ready = True
        except Exception as exc:
            st.error(f"Annotation parsing error: {exc}")

else:
    st.info(
        "ℹ️ **Fallback Mode** – Upload matching pre/post crop pairs as separate images. "
        "Each row is one building."
    )
    n_buildings = st.number_input("Number of buildings", 1, 20, 3, step=1)
    for i in range(int(n_buildings)):
        st.markdown(f"**Building {i+1}**")
        bc1, bc2 = st.columns(2)
        with bc1:
            bp = st.file_uploader(f"Pre crop {i+1}", type=["png","jpg","jpeg"], key=f"bp_{i}")
        with bc2:
            bq = st.file_uploader(f"Post crop {i+1}", type=["png","jpg","jpeg"], key=f"bq_{i}")
        if bp and bq:
            pre_c  = pil_to_numpy(Image.open(bp))
            post_c = pil_to_numpy(Image.open(bq))
            building_records.append({
                "uid": f"building_{i+1}",
                "pre_crop": pre_c, "post_crop": post_c,
                "pre_bbox": (0,0,pre_c.shape[1],pre_c.shape[0]),
                "post_bbox":(0,0,post_c.shape[1],post_c.shape[0]),
                "damage_class": 0, "damage_label": "unknown",
                "pre_image_path": "uploaded", "post_image_path": "uploaded",
            })
    if building_records:
        st.success(f"✅ {len(building_records)} building crop pairs ready.")
        buildings_ready = True

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: Damage Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.header("🔍 Section C – Damage Analysis")

predictions: List[Dict] = []

if not model_ready:
    st.warning("⚠️ No trained model loaded. Please follow the Colab training steps.")
elif not buildings_ready or not building_records:
    st.info("👆 Upload satellite images and annotation/crop files above to enable analysis.")
else:
    if st.button("🚀 Analyze Damage", type="primary", use_container_width=True):
        with st.spinner(f"Running Siamese CNN on {len(building_records)} buildings …"):
            predictions = predictor.predict_batch(building_records)

        # Filter by confidence threshold
        shown = [p for p in predictions if p["confidence"] >= conf_threshold]
        st.session_state["predictions"] = shown
        st.session_state["all_preds"]   = predictions
        st.success(f"✅ Analysis complete – {len(shown)} buildings above confidence threshold.")

    # Restore from session
    if "predictions" in st.session_state:
        predictions = st.session_state["predictions"]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if predictions:
    st.header("📊 Section D – Results Dashboard")

    summary = build_damage_summary(predictions)
    total   = len(predictions)

    # KPI metrics
    m_cols = st.columns(5)
    m_cols[0].metric("🏘️ Total Buildings", total)
    m_cols[1].metric("🟢 No Damage",    summary["No Damage"])
    m_cols[2].metric("🟡 Minor Damage", summary["Minor Damage"])
    m_cols[3].metric("🟠 Major Damage", summary["Major Damage"])
    m_cols[4].metric("🔴 Destroyed",    summary["Destroyed"])

    # Damage distribution chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    vals   = [summary[n] for n in CLASS_NAMES]
    colors = [CLASS_COLORS[n] for n in CLASS_NAMES]
    bars   = ax.bar(CLASS_NAMES, vals, color=colors, edgecolor="black", alpha=0.9)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=11)
    ax.set_title("Predicted Damage Distribution", fontsize=13)
    ax.set_ylabel("Building Count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Colour-coded overlay on post image (if available)
    if post_file and "annotation" in localization_mode.lower():
        try:
            from visualization.damage_visualizer import draw_damage_overlay, add_legend
            post_arr = pil_to_numpy(post_pil)
            # Attach polygon for each prediction
            for i, p in enumerate(predictions):
                if "post_polygon" not in p:
                    bbox = p.get("post_bbox", (0, 0, 10, 10))
                    x1,y1,x2,y2 = bbox
                    p["post_polygon"] = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            overlay = draw_damage_overlay(post_arr, predictions, use_gt=False)
            st.image(overlay, caption="Post-Disaster – Predicted Damage Overlay", use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render overlay: {exc}")

    # Confusion matrix (if outputs/confusion_matrix.png exists)
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.subheader("📈 Model Evaluation – Confusion Matrix")
        st.image(str(cm_path), caption="4-Class Confusion Matrix (Test Set)", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: Building Details
# ─────────────────────────────────────────────────────────────────────────────
if predictions:
    st.header("🏗️ Section E – Building Details")

    for i, pred in enumerate(predictions[:20]):   # limit display to 20
        cls_name = pred["class_name"]
        conf     = pred["confidence"]
        color    = damage_color_html(cls_name)
        uid      = pred.get("uid", f"Building {i+1}")

        with st.expander(
            f"**{uid}** – {cls_name}  ({conf:.1%} confidence)",
            expanded=(i == 0)
        ):
            d1, d2, d3 = st.columns([1, 1, 1.5])
            with d1:
                pre_crop = pred.get("pre_crop")
                if pre_crop is not None:
                    st.image(numpy_to_pil(pre_crop), caption="Pre-Disaster Crop", use_container_width=True)
            with d2:
                post_crop = pred.get("post_crop")
                if post_crop is not None:
                    st.image(numpy_to_pil(post_crop), caption="Post-Disaster Crop", use_container_width=True)
            with d3:
                st.markdown(f"**Predicted Severity:** "
                            f"<span style='color:{color};font-weight:700'>{cls_name}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {conf:.2%}")
                st.markdown("**Class Probabilities:**")
                probs = pred.get("probabilities", {})
                for cname, cprob in probs.items():
                    ccolor = CLASS_COLORS.get(cname, "#999")
                    bar_w  = int(cprob * 100)
                    st.markdown(
                        f"<div style='margin:2px 0'><span style='width:120px;display:inline-block'>{cname}</span>"
                        f"<div style='display:inline-block;width:{bar_w}%;height:12px;"
                        f"background:{ccolor};border-radius:3px;'></div>"
                        f" {cprob:.1%}</div>",
                        unsafe_allow_html=True
                    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#888;font-size:0.85rem;'>
DisasterVision | Siamese CNN on xBD Dataset |
Localization: <strong>Annotation-Assisted</strong> |
<a href='https://github.com/DIUx-xView/xView2_baseline' target='_blank'>xBD Dataset</a>
</div>
""", unsafe_allow_html=True)
