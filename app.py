# app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import time
from datetime import datetime

from src.config import IMG_SIZE, CLASS_NAMES

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .result-positive {
        background: #fff3cd;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #ff9800;
    }
    .result-negative {
        background: #d4edda;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ─── Cache model ───
@st.cache_resource
def get_model():
    return load_model("models/best_model.keras")

model = get_model()

# ─── Session state for history ───
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Class info ───
CLASS_INFO = {
    "glioma": {
        "desc": "A tumor arising from glial cells. Often irregular and infiltrative.",
        "color": "#e74c3c",
        "icon": "🔴"
    },
    "meningioma": {
        "desc": "A tumor arising from the meninges (brain lining). Usually well-defined.",
        "color": "#f39c12",
        "icon": "🟠"
    },
    "notumor": {
        "desc": "No tumor detected in the scan.",
        "color": "#2ecc71",
        "icon": "🟢"
    },
    "pituitary": {
        "desc": "A tumor at the base of the brain affecting the pituitary gland.",
        "color": "#3498db",
        "icon": "🔵"
    },
}

# ─── Sidebar ───
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=80)
    st.markdown("## About")
    st.markdown(
        "A deep learning classifier that analyzes brain MRI scans "
        "and predicts one of four categories."
    )

    st.markdown("---")
    st.markdown("### 📊 Model Stats")
    st.metric("Test Accuracy", "84.2%")
    st.metric("Training Images", "5,600")
    st.metric("Classes", "4")
    st.metric("Input Size", "128×128")

    st.markdown("---")
    st.markdown("### 🎯 Class Labels")
    for name, info in CLASS_INFO.items():
        st.markdown(f"{info['icon']} **{name.capitalize()}**")

    st.markdown("---")
    st.caption("⚠️ Educational demo only. Not for clinical use.")

# ─── Header ───
st.markdown('<h1 class="main-header">🧠 Brain Tumor Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a brain MRI scan to get an instant classification</p>', unsafe_allow_html=True)

# ─── Tabs ───
tab_predict, tab_samples, tab_history, tab_about = st.tabs(
    ["🔍 Predict", "🖼️ Try Samples", "📜 History", "ℹ️ About"]
)

# ═══════════════════════════════════════
# TAB 1: PREDICT
# ═══════════════════════════════════════
with tab_predict:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📤 Upload Image")
        uploaded = st.file_uploader(
            "Choose an MRI scan (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded MRI", use_container_width=True)

    with col_right:
        if uploaded is not None:
            st.markdown("### 🎯 Prediction Result")

            # Preprocess
            arr = np.array(img.resize(IMG_SIZE))
            arr = np.expand_dims(arr, axis=0).astype("float32")

            with st.spinner("Analyzing scan..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress.progress(i + 1)
                preds = model.predict(arr, verbose=0)[0]
                progress.empty()

            top_idx = int(np.argmax(preds))
            top_class = CLASS_NAMES[top_idx]
            top_conf = float(preds[top_idx]) * 100
            info = CLASS_INFO[top_class]

            # Result banner
            if top_class == "notumor":
                st.markdown(
                    f'<div class="result-negative">'
                    f'<h2 style="margin:0;">✅ No Tumor Detected</h2>'
                    f'<p style="margin:0.5rem 0 0 0; font-size:1.1rem;">'
                    f'Confidence: <b>{top_conf:.1f}%</b></p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-positive">'
                    f'<h2 style="margin:0;">⚠️ {top_class.capitalize()} Detected</h2>'
                    f'<p style="margin:0.5rem 0 0 0; font-size:1.1rem;">'
                    f'Confidence: <b>{top_conf:.1f}%</b></p></div>',
                    unsafe_allow_html=True
                )

            st.markdown(f"*{info['desc']}*")
            st.markdown("---")

            # Probability bars
            st.markdown("### 📊 Probability Breakdown")
            df = pd.DataFrame({
                "Class": [c.capitalize() for c in CLASS_NAMES],
                "Probability": [float(p) * 100 for p in preds]
            }).sort_values("Probability", ascending=False)

            st.bar_chart(df.set_index("Class"), use_container_width=True)

            # Detailed table
            with st.expander("See raw probabilities"):
                for name, p in zip(CLASS_NAMES, preds):
                    st.progress(float(p), text=f"{name.capitalize()}: {p*100:.2f}%")

            # Save to history
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "filename": uploaded.name,
                "prediction": top_class,
                "confidence": f"{top_conf:.1f}%"
            })

        else:
            st.markdown("### 🎯 Prediction Result")
            st.info("👈 Upload an MRI image to see the prediction")
            st.markdown("""
            **How it works:**
            1. Upload a brain MRI (JPG/PNG)
            2. The model analyzes it in 2-3 seconds
            3. See the predicted class + confidence
            """)

# ═══════════════════════════════════════
# TAB 2: SAMPLES
# ═══════════════════════════════════════
with tab_samples:
    st.markdown("### 🖼️ Sample Images from Test Set")
    st.markdown("Click any sample to preview, then use the Predict tab to classify.")

    sample_paths = []
    for cls in CLASS_NAMES:
        folder = f"data/Testing/{cls}"
        if os.path.exists(folder):
            files = sorted(os.listdir(folder))[:3]
            for f in files:
                sample_paths.append((cls, f, os.path.join(folder, f)))

    if sample_paths:
        cols = st.columns(4)
        for i, (cls, fname, path) in enumerate(sample_paths):
            with cols[i % 4]:
                st.image(path, caption=f"{cls}", use_container_width=True)
    else:
        st.warning("No sample images found in `data/Testing/`.")

# ═══════════════════════════════════════
# TAB 3: HISTORY
# ═══════════════════════════════════════
with tab_history:
    st.markdown("### 📜 Prediction History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Upload an image to see them here.")

# ═══════════════════════════════════════
# TAB 4: ABOUT
# ═══════════════════════════════════════
with tab_about:
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    **Brain Tumor Classification** uses a custom Convolutional Neural Network
    to classify brain MRI scans into four categories.

    #### Model Architecture
    - 3 convolutional blocks (32, 64, 128 filters)
    - BatchNormalization + MaxPooling after each block
    - GlobalAveragePooling → Dropout → Dense(4, softmax)
    - Input: 128×128×3 RGB

    #### Dataset
    - Kaggle Brain Tumor MRI Dataset
    - 5,600 training images + 1,600 test images
    - 4 balanced classes

    #### Performance
    - Test accuracy: **84.2%**
    - Notumor: 98.8% recall
    - Pituitary: 98.5% recall
    - Glioma: 75.5% recall
    - Meningioma: 64.0% recall

    #### Limitations
    - Confuses glioma with meningioma in some cases
    - Trained on a single dataset source
    - Domain shift reduces performance on external MRIs

    #### Tech Stack
    - TensorFlow 2.20 / Keras 3
    - Streamlit for UI
    - Pandas for data handling

    ---
    *Built as a student project. Not a clinical diagnostic tool.*
    """)

# ─── Footer ───
st.markdown("---")
st.caption("🧠 Brain Tumor Classifier | TensorFlow + Streamlit | Educational use only")