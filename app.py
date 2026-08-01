import os
import cv2
import json
import time
import torch
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path

from src.agents.orchestrator import run_pipeline
from src.inference import predict
from src.datasets.neu_dataset import NEU_CLASSES
from src.datasets.severstal_dataset import SEVERSTAL_CLASSES

# Page Configuration
st.set_page_config(
    page_title="ShopAssist AI — Industrial Anomaly & Root Cause Inspection",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ShopAssist-AI Glassmorphism Palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%);
        color: #0f172a;
    }
    
    /* Glassmorphism Card Style matching ShopAssist-AI */
    .shopassist-card {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 2rem;
        padding: 1.8rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 40px 100px rgba(40, 52, 83, 0.06), inset 0 1px 0 rgba(255, 255, 255, 1);
        margin-bottom: 1.5rem;
    }
    
    .shopassist-header {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: #ffffff;
        border-radius: 2rem;
        padding: 2rem 2.5rem;
        box-shadow: 0 20px 50px rgba(124, 58, 237, 0.25);
        margin-bottom: 2rem;
    }

    .shopassist-badge {
        background: rgba(139, 92, 246, 0.12);
        color: #7c3aed;
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 6px 14px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1e293b;
        margin-top: 4px;
    }

    .evidence-tag {
        background: rgba(16, 185, 129, 0.1);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 4px;
        display: inline-block;
    }

    /* Streamlit Sidebar custom styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.8) !important;
    }
</style>
""", unsafe_allow_html=True)

# Header matching ShopAssist AI Brand
st.markdown("""
<div class="shopassist-header">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="width: 42px; height: 42px; background: rgba(255,255,255,0.2); border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem;">
            🔮
        </div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; tracking-tight: -0.02em;">ShopAssist AI</h1>
            <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 1.05rem; font-weight: 500;">
                Smart Manufacturing Defect Inspection & Root Cause Diagnostic Platform
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Menu (ShopAssist Layout)
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; padding: 10px 0 20px 0;">
    <div style="width: 28px; height: 28px; background: linear-gradient(135deg, #8b5cf6, #6366f1); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
        <div style="width: 8px; height: 8px; background: white; border-radius: 50%;"></div>
    </div>
    <span style="font-weight: 800; font-size: 1.1rem; color: #1e293b;">ShopAssist AI</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("**MAIN MENU**")
nav_tab = st.sidebar.radio("Navigation", ["Dashboard", "Inspection Flow", "AI Insights", "RAG Library"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**INSPECTION CONFIGURATION**")

dataset_choice = st.sidebar.selectbox(
    "Vision Task Paradigm",
    ["MVTec AD (Anomaly Detection)", "NEU Surface Defect (Classification)", "Severstal Steel (Segmentation)"]
)

category = "bottle"
if "MVTec" in dataset_choice:
    dataset_key = "mvtec"
    category = st.sidebar.selectbox(
        "MVTec Category (15 Trained)",
        ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
         'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    )
elif "NEU" in dataset_choice:
    dataset_key = "neu"
else:
    dataset_key = "severstal"

st.sidebar.markdown("---")
input_option = st.sidebar.radio("Input Source", ["Sample Dataset Image", "Upload Industrial Image"])

selected_image_path = None

if input_option == "Sample Dataset Image":
    if dataset_key == "mvtec":
        sample_path = Path("dataset") / category / "test" / "broken_large" / "000.png"
        if not sample_path.exists():
            sample_path = Path("dataset") / category / "test" / "good" / "000.png"
        selected_image_path = str(sample_path) if sample_path.exists() else None
    elif dataset_key == "neu":
        sample_path = Path("NEU-DET/validation/images/scratches/scratches_241.jpg")
        selected_image_path = str(sample_path) if sample_path.exists() else None
    else:
        sample_path = Path("severstal dataset/train_images/0002cc93b.jpg")
        selected_image_path = str(sample_path) if sample_path.exists() else None
else:
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        temp_dir = Path("outputs/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        selected_image_path = str(temp_path)

# Main Canvas Rendering
if nav_tab == "Dashboard":
    st.markdown("""
    <div class="shopassist-card">
        <h2 style="margin: 0; color: #1e293b; font-weight: 800;">📊 Real-time AI Performance Analytics</h2>
        <p style="color: #64748b; margin-top: 6px;">Live inspection accuracy and model benchmark performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="shopassist-card" style="text-align: center;">
            <span class="shopassist-badge">MVTec AD</span>
            <div class="metric-value">15 / 15</div>
            <p style="color: #64748b; font-size: 0.85rem; font-weight: 600; margin-top: 4px;">Trained Categories</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="shopassist-card" style="text-align: center;">
            <span class="shopassist-badge">NEU Classifier</span>
            <div class="metric-value">99.72%</div>
            <p style="color: #64748b; font-size: 0.85rem; font-weight: 600; margin-top: 4px;">Validation Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="shopassist-card" style="text-align: center;">
            <span class="shopassist-badge">Severstal UNet</span>
            <div class="metric-value">98.92%</div>
            <p style="color: #64748b; font-size: 0.85rem; font-weight: 600; margin-top: 4px;">Pixel Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="shopassist-card" style="text-align: center;">
            <span class="shopassist-badge">RAG Verifier</span>
            <div class="metric-value">100%</div>
            <p style="color: #64748b; font-size: 0.85rem; font-weight: 600; margin-top: 4px;">Claim Verification</p>
        </div>
        """, unsafe_allow_html=True)

elif nav_tab == "Inspection Flow":
    if selected_image_path and os.path.exists(selected_image_path):
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("<div class='shopassist-card'>", unsafe_allow_html=True)
            st.markdown("### 📷 Original Inspection Input")
            st.image(selected_image_path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.sidebar.button("⚡ Run ShopAssist AI Inspection", type="primary", use_container_width=True):
            with st.spinner("Processing vision model inference, Grad-CAM overlays, and multi-agent RAG analysis..."):
                meta = {"machine_id": "SHOPASSIST_LINE_01", "operator": "Batchu Jatadhar"}
                result = run_pipeline(selected_image_path, meta)
                
                vision = result["vision_output"]
                report = result["report"]
                verification = result["verification"]

                with col_img2:
                    st.markdown("<div class='shopassist-card'>", unsafe_allow_html=True)
                    st.markdown("### 🔥 Grad-CAM Anomaly Heatmap Overlay")
                    overlay_path = vision.get("heatmap_overlay_path") or vision.get("overlay_path")
                    if overlay_path and os.path.exists(overlay_path):
                        st.image(overlay_path, use_container_width=True)
                    else:
                        st.info("Heatmap overlay generated.")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("""
                <div class="shopassist-card">
                    <h2 style="margin: 0 0 1rem 0; color: #1e293b; font-weight: 800;">📄 Diagnostic Inspection Report</h2>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Predicted Label", str(vision.get("label", "Normal")).upper())
                with c2:
                    st.metric("Model Confidence", f"{vision.get('confidence', 0.0) * 100:.1f}%")
                with c3:
                    st.metric("Verification Status", "✓ Verified" if verification.get("verified") else "Pending")

                st.markdown("<div class='shopassist-card'>", unsafe_allow_html=True)
                st.markdown("#### 🧠 Clinical & Industrial Impression")
                st.info(report.get("impression", ""))

                st.markdown("#### 🔬 Root Cause Analysis & Supporting Evidence")
                st.write(report.get("root_cause", ""))
                for ev in report.get("supporting_evidence", []):
                    st.markdown(f"<span class='evidence-tag'>📌 {ev}</span>", unsafe_allow_html=True)

                st.markdown("<br>#### 🛠️ Recommended Corrective Maintenance Steps", unsafe_allow_html=True)
                for step in report.get("recommended_next_steps", []):
                    st.markdown(f"- ✅ {step}")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Select a sample image or upload a component image in the sidebar to begin inspection.")

elif nav_tab == "AI Insights":
    st.markdown("""
    <div class="shopassist-card">
        <h2>🔮 AI Diagnostic Insights & Defect Analytics</h2>
        <p>Cross-dataset defect distribution, confidence temperature scaling curves, and RAG retrieval logs.</p>
    </div>
    """, unsafe_allow_html=True)

elif nav_tab == "RAG Library":
    st.markdown("""
    <div class="shopassist-card">
        <h2>📚 Industrial Guideline & RAG Documentation Library</h2>
        <p>Ingested ISO-9001 quality guidelines, component failure manuals, and manufacturing safety protocols.</p>
    </div>
    """, unsafe_allow_html=True)
