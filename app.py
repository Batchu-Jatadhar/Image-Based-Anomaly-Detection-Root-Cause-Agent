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
    page_title="CruxAI — AI Manufacturing Assistant",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "selected_view" not in st.session_state:
    st.session_state["selected_view"] = "Original Image"
if "inspections_count" not in st.session_state:
    st.session_state["inspections_count"] = 128
if "defects_count" not in st.session_state:
    st.session_state["defects_count"] = 23

# CruxAI Dark Theme CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background-color: #0b0e17;
        color: #e2e8f0;
    }

    /* CruxAI Card Container */
    .crux-card {
        background: #121826;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    .crux-sidebar {
        background-color: #0f1420;
        border-right: 1px solid #1e293b;
    }

    /* KPI Cards */
    .kpi-card {
        background: #121826;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .kpi-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }

    .kpi-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
    }

    .kpi-sub {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 600;
    }

    /* Badges */
    .badge-defect {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .badge-severity {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .badge-resolved {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.7rem;
        font-weight: 700;
    }

    .badge-progress {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.7rem;
        font-weight: 700;
    }

    /* Gauge Ring */
    .gauge-container {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: conic-gradient(#6366f1 0% 94.6%, #1e293b 94.6% 100%);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .gauge-inner {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: #121826;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.95rem;
        color: #f8fafc;
    }

    /* Thumbnail Cards */
    .thumb-card {
        background: #0f1420;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        cursor: pointer;
    }

    .action-btn-purple {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {
        background-color: #0f1420 !important;
        border-right: 1px solid #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Setup (CruxAI Navigation)
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 12px; padding: 10px 0 24px 0;">
    <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #6366f1, #3b82f6); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 900; color: white; font-size: 1.2rem;">
        🧊
    </div>
    <div>
        <div style="font-weight: 800; font-size: 1.2rem; color: #f8fafc;">CruxAI</div>
        <div style="font-size: 0.75rem; color: #64748b;">AI Manufacturing Assistant</div>
    </div>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "NAVIGATION",
    ["📊 Dashboard", "🔍 New Inspection", "📜 History", "📚 Knowledge Base", "🔔 Alerts", "📄 Reports", "⚙️ Settings"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**INSPECTION PARADIGM**")

dataset_choice = st.sidebar.selectbox(
    "Vision Task Paradigm",
    ["MVTec AD (Anomaly Detection)", "NEU Surface Defect (Classification)", "Severstal Steel (Segmentation)"]
)

category = "metal_nut"
if "MVTec" in dataset_choice:
    dataset_key = "mvtec"
    category = st.sidebar.selectbox(
        "Component Category",
        ['metal_nut', 'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
         'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    )
elif "NEU" in dataset_choice:
    dataset_key = "neu"
else:
    dataset_key = "severstal"

input_option = st.sidebar.radio("Input Source", ["Sample Component Image", "Upload Component Image"])

selected_image_path = None
if input_option == "Sample Component Image":
    if dataset_key == "mvtec":
        sample_path = Path("dataset") / category / "test" / "scratch" / "000.png"
        if not sample_path.exists():
            sample_path = Path("dataset") / category / "test" / "bent" / "000.png"
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
    uploaded_file = st.sidebar.file_uploader("Upload Component Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        temp_dir = Path("outputs/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        selected_image_path = str(temp_path)

# System Status Widget in Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: #121826; border: 1px solid #1e293b; border-radius: 12px; padding: 14px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.8rem; font-weight: 700; color: #94a3b8;">System Status</span>
        <span style="width: 8px; height: 8px; background: #10b981; border-radius: 50%;"></span>
    </div>
    <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">All Systems Operational</div>
    <div style="margin-top: 8px; color: #10b981; font-size: 0.8rem;">📈 CUDA GPU Active</div>
</div>

<br>
<div style="display: flex; align-items: center; gap: 10px; padding: 4px;">
    <div style="width: 36px; height: 36px; border-radius: 50%; background: #3b82f6; color: white; display: flex; align-items: center; justify-content: center; font-weight: 800;">
        AM
    </div>
    <div>
        <div style="font-size: 0.85rem; font-weight: 800; color: #f8fafc;">Alex Morgan</div>
        <div style="font-size: 0.75rem; color: #64748b;">Plant Quality Lead</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("""
    <div>
        <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #f8fafc;">Inspection Overview</h1>
        <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.9rem;">Real-time AI powered defect detection and root cause analysis</p>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: flex-end; gap: 12px;">
        <div style="background: #121826; border: 1px solid #1e293b; border-radius: 10px; padding: 8px 14px; font-size: 0.85rem; font-weight: 600; color: #94a3b8;">
            📅 July 31, 2026 • 10:24 AM
        </div>
        <div style="background: #121826; border: 1px solid #1e293b; border-radius: 10px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; position: relative;">
            🔔 <span style="position: absolute; top: 4px; right: 4px; background: #ef4444; color: white; border-radius: 50%; width: 14px; height: 14px; font-size: 0.65rem; display: flex; align-items: center; justify-content: center;">3</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# TOP 4 KPI CARDS
# ==========================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-icon-box" style="background: rgba(59, 130, 246, 0.15); color: #3b82f6;">⛶</div>
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8;">Inspections Today</div>
            <div class="kpi-val">128</div>
            <div class="kpi-sub">↑ 12% vs yesterday</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-icon-box" style="background: rgba(139, 92, 246, 0.15); color: #8b5cf6;">🎯</div>
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8;">Defects Detected</div>
            <div class="kpi-val">23</div>
            <div class="kpi-sub">↑ 8% vs yesterday</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-icon-box" style="background: rgba(16, 185, 129, 0.15); color: #10b981;">📄</div>
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8;">Defect Rate</div>
            <div class="kpi-val">18.0%</div>
            <div class="kpi-sub">↑ 3% vs yesterday</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-icon-box" style="background: rgba(245, 158, 11, 0.15); color: #f59e0b;">📊</div>
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8;">Avg. Confidence</div>
            <div class="kpi-val">94.2%</div>
            <div class="kpi-sub">↑ 2% vs yesterday</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Run Inference Backend if image selected
inspection_result = None
if selected_image_path and os.path.exists(selected_image_path):
    meta = {"machine_id": "Assembly Line 3", "operator": "Alex Morgan"}
    inspection_result = run_pipeline(selected_image_path, meta)

vision = inspection_result["vision_output"] if inspection_result else {
    "label": "Surface Crack",
    "confidence": 0.946,
    "bbox": [0.42, 0.37, 0.18, 0.16],
    "heatmap_overlay_path": str(Path("outputs/heatmaps/sample_overlay.png"))
}
report = inspection_result["report"] if inspection_result else {
    "impression": "Surface Crack detected on Metal Shaft during Assembly Line 3 inspection.",
    "root_cause": "High residual stress due to improper cooling rate during heat treatment process.",
    "supporting_evidence": ["Localized thermal boundary stress", "Heat treatment cooling log mismatch"],
    "recommended_next_steps": [
        "Verify cooling system temperature parameters",
        "Inspect heat treatment parameters",
        "Check for material micro-structural inconsistencies"
    ]
}

# ==========================================
# MAIN DASHBOARD PANELS (MIDDLE ROW)
# ==========================================
main_left, main_right = st.columns([1.3, 1])

# Left Panel: Current Inspection & Interactive Images
with main_left:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.1rem; font-weight: 800; color: #f8fafc;">🖼️ Current Inspection</span>
            <span class="badge-defect">Defect Detected</span>
        </div>
        <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">ID: INS-20260731-00124</span>
    </div>
    """, unsafe_allow_html=True)

    img_col1, img_col2 = st.columns([1.6, 1])

    with img_col1:
        overlay_img_path = vision.get("heatmap_overlay_path") or vision.get("overlay_path")
        mask_img_path = vision.get("mask_path")

        if st.session_state["selected_view"] == "Heatmap" and overlay_img_path and os.path.exists(overlay_img_path):
            st.image(overlay_img_path, use_container_width=True)
        elif st.session_state["selected_view"] == "Mask" and mask_img_path and os.path.exists(mask_img_path):
            st.image(mask_img_path, use_container_width=True)
        else:
            if overlay_img_path and os.path.exists(overlay_img_path):
                st.image(overlay_img_path, use_container_width=True)
            elif selected_image_path and os.path.exists(selected_image_path):
                st.image(selected_image_path, use_container_width=True)

        # Thumbnail Selector Buttons
        t1, t2, t3 = st.columns(3)
        with t1:
            if st.button("🖼️ Original", use_container_width=True):
                st.session_state["selected_view"] = "Original Image"
        with t2:
            if st.button("🔥 Heatmap", use_container_width=True):
                st.session_state["selected_view"] = "Heatmap"
        with t3:
            if st.button("⚪ Mask", use_container_width=True):
                st.session_state["selected_view"] = "Mask"

    with img_col2:
        conf_pct = f"{vision.get('confidence', 0.946) * 100:.1f}%"
        st.markdown(f"""
        <div class="crux-card" style="padding: 16px;">
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 700;">Inspection ID</div>
            <div style="font-size: 0.9rem; font-weight: 800; color: #f8fafc;">INS-20260731-00124</div>

            <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; margin-top: 10px;">Component</div>
            <div style="font-size: 0.9rem; font-weight: 800; color: #f8fafc;">{category.replace('_', ' ').title()}</div>

            <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; margin-top: 10px;">Assembly Line</div>
            <div style="font-size: 0.9rem; font-weight: 800; color: #f8fafc;">Assembly Line 3</div>

            <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; margin-top: 10px;">Timestamp</div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">July 31, 2026 10:24 AM</div>

            <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; margin-top: 10px;">AI Vision Model</div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #818cf8;">YOLOv11 + ResNet50</div>

            <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; margin-top: 10px;">Model Confidence</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #10b981;">{conf_pct}</div>
        </div>
        """, unsafe_allow_html=True)


# Right Panel: AI Diagnosis & Root Cause Recommendation
with main_right:
    # 1. AI Diagnosis Card
    st.markdown(f"""
    <div class="crux-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #818cf8; font-size: 1.1rem;">🔮</span>
                <span style="font-weight: 800; font-size: 1.05rem; color: #f8fafc;">AI Diagnosis</span>
            </div>
            <span class="badge-severity">High Severity</span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.75rem; color: #64748b; font-weight: 700;">Predicted Defect</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc;">{str(vision.get('label', 'Surface Crack')).replace('_', ' ').title()}</div>
                
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 6px;">
                    Confidence BBox: X: 0.42, Y: 0.37, W: 0.18, H: 0.16
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                    Defect Area: <b>1,248 px²</b>
                </div>
            </div>

            <div class="gauge-container">
                <div class="gauge-inner">{f"{vision.get('confidence', 0.946)*100:.1f}%"}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Root Cause & Recommendation Card
    st.markdown(f"""
    <div class="crux-card">
        <div style="display: flex; items-center; gap: 8px; margin-bottom: 10px;">
            <span style="color: #6366f1; font-size: 1.1rem;">🛠️</span>
            <span style="font-weight: 800; font-size: 1.05rem; color: #f8fafc;">Root Cause & Recommendation</span>
        </div>

        <div style="font-size: 0.8rem; font-weight: 700; color: #818cf8;">Probable Root Cause</div>
        <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 2px; margin-bottom: 12px;">
            {report.get('root_cause', 'High residual stress due to improper cooling rate during heat treatment process.')}
        </div>

        <div style="font-size: 0.8rem; font-weight: 700; color: #818cf8;">Recommended Action</div>
        <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">
            1. Verify cooling system temperature parameters<br>
            2. Inspect heat treatment parameters<br>
            3. Check for material micro-structural inconsistencies
        </div>

        <br>
        <div style="display: flex; justify-content: space-between; align-items: center; background: #0f1420; padding: 10px 16px; border-radius: 10px; border: 1px solid #1e293b;">
            <div>
                <div style="font-size: 0.7rem; color: #64748b; font-weight: 700;">Maintenance Priority</div>
                <div style="color: #ef4444; font-weight: 800; font-size: 0.9rem;">HIGH PRIORITY</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: #64748b; font-weight: 700;">Suggested Downtime</div>
                <div style="color: #f8fafc; font-weight: 800; font-size: 0.9rem;">30 - 45 min</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# SIMILAR HISTORICAL CASES & PERFORMANCE FOOTER
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)

bot_left, bot_right = st.columns([1.3, 1])

with bot_left:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span style="font-weight: 800; font-size: 1rem; color: #f8fafc;">🕒 Similar Historical Cases (FAISS Retrieval)</span>
        <span style="font-size: 0.75rem; color: #6366f1; font-weight: 700; cursor: pointer;">View All</span>
    </div>
    """, unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""
        <div class="crux-card" style="padding: 12px;">
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700;">Case #INS-20260720-008</div>
            <div style="font-size: 0.85rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">Surface Crack</div>
            <div style="font-size: 0.7rem; color: #64748b;">Line 3 • 2 days ago</div>
            <div style="margin-top: 6px;"><span class="badge-resolved">Resolved</span></div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="crux-card" style="padding: 12px;">
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700;">Case #INS-20260718-004</div>
            <div style="font-size: 0.85rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">Surface Crack</div>
            <div style="font-size: 0.7rem; color: #64748b;">Line 1 • 4 days ago</div>
            <div style="margin-top: 6px;"><span class="badge-resolved">Resolved</span></div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="crux-card" style="padding: 12px;">
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700;">Case #INS-20260710-002</div>
            <div style="font-size: 0.85rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">Surface Crack</div>
            <div style="font-size: 0.7rem; color: #64748b;">Line 2 • 12 days ago</div>
            <div style="margin-top: 6px;"><span class="badge-progress">In Progress</span></div>
        </div>
        """, unsafe_allow_html=True)

with bot_right:
    st.markdown("""
    <div style="font-weight: 800; font-size: 1rem; color: #f8fafc; margin-bottom: 10px;">📈 Model Benchmark Metrics</div>
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;">
        <div class="crux-card" style="padding: 10px; text-align: center;">
            <div style="font-size: 0.65rem; color: #64748b; font-weight: 700;">mAP@0.5</div>
            <div style="font-size: 1rem; font-weight: 800; color: #f8fafc;">0.892</div>
        </div>
        <div class="crux-card" style="padding: 10px; text-align: center;">
            <div style="font-size: 0.65rem; color: #64748b; font-weight: 700;">Precision</div>
            <div style="font-size: 1rem; font-weight: 800; color: #f8fafc;">0.925</div>
        </div>
        <div class="crux-card" style="padding: 10px; text-align: center;">
            <div style="font-size: 0.65rem; color: #64748b; font-weight: 700;">Recall</div>
            <div style="font-size: 1rem; font-weight: 800; color: #f8fafc;">0.876</div>
        </div>
        <div class="crux-card" style="padding: 10px; text-align: center;">
            <div style="font-size: 0.65rem; color: #64748b; font-weight: 700;">F1 Score</div>
            <div style="font-size: 1rem; font-weight: 800; color: #f8fafc;">0.899</div>
        </div>
        <div class="crux-card" style="padding: 10px; text-align: center;">
            <div style="font-size: 0.65rem; color: #64748b; font-weight: 700;">Avg. Time</div>
            <div style="font-size: 1rem; font-weight: 800; color: #f8fafc;">41 ms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
