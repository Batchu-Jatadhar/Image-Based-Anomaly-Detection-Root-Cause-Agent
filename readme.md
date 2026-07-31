# Smart Manufacturing Defect Detection & Root Cause Analysis System

## Project Overview

We are building an AI-powered industrial inspection and predictive maintenance platform for manufacturing environments. The goal is to move beyond simple image classification and create a complete decision-support system that assists engineers throughout the inspection process.

Instead of only detecting a defect, the system analyzes the uploaded industrial image, identifies and localizes anomalies, classifies the defect type, retrieves similar historical cases, determines the most probable root cause, recommends corrective maintenance actions, estimates operational risk, and finally generates a professional inspection report.

The project is designed as a modular, production-style application where each component is independent and communicates through well-defined interfaces.

---

## End-to-End Workflow

```text
User Uploads Industrial Image
            │
            ▼
Image Validation
            │
            ▼
Image Preprocessing
            │
            ▼
AI Defect Detection
            │
            ▼
Defect Localization
(Bounding Box / Segmentation / Grad-CAM)
            │
            ▼
Defect Classification
            │
            ▼
Feature Embedding Generation & Historical Retrieval (RAG / FAISS)
            │
            ▼
Multi-Agent Root Cause Analysis
(Perception -> Diagnostic -> Verifier)
            │
            ▼
Maintenance Recommendation
            │
            ▼
Risk & Severity Prediction
            │
            ▼
Professional Report Generation (PDF / JSON)
            │
            ▼
Store Results in Database
            │
            ▼
Dashboard & Analytics (Human-in-the-Loop Approve/Revise)
```

---

## Core System Modules

### 1. Image Preprocessing & Validation
- Validate uploaded images (file types, dimensions, corruption checks)
- Resize and normalize color spaces
- Noise removal and contrast enhancement (CLAHE / histogram equalization)
- Data augmentation pipeline for model robustness

---

### 2. Defect Detection Engine
Uses computer vision models (YOLOv11/v8, ResNet, ViT) to:
- Detect anomalies
- Localize defects with normalized bounding boxes `[x, y, w, h]`
- Produce confidence scores
- Generate Grad-CAM heatmaps for visual localization verification

---

### 3. Defect Classification Engine
Classifies detected anomalies into predefined industrial defect categories such as:
- Crack
- Scratch
- Dent
- Corrosion
- Burn Mark
- Surface Defect
- Hole
- Other manufacturing defects

---

### 4. Historical Retrieval Engine (RAG)
Searches previous inspection records and guidelines to retrieve:
- Similar defect images via vector similarity (FAISS)
- Previous repairs and maintenance logs
- Machine history and failure statistics
- Live literature and technical documentation (via API)

---

### 5. Root Cause Analysis Engine
Analyzes the detected defect together with historical information using a Multi-Agent LLM pipeline (local quantized LLM) to estimate the most probable cause, such as:
- Material fatigue
- Thermal stress
- Misalignment
- Excessive vibration
- Improper lubrication
- Manufacturing defects
- Human error

---

### 6. Maintenance Recommendation Engine
Generates actionable maintenance recommendations, including:
- Component replacement protocols
- Preventive maintenance schedules
- Alignment and calibration correction
- Lubrication and thermal threshold adjustments
- Safety enforcement guidelines

---

### 7. Risk Assessment Engine
Estimates:
- Severity Level (Low, Medium, High, Critical)
- Failure Probability (%)
- Repair Priority
- Expected Downtime
- Production Impact

---

### 8. Report Generation Engine
Automatically generates professional inspection reports containing:
- Inspection Summary
- Visual Evidence (Original Image + Bounding Box + Grad-CAM Overlay)
- Detected Defects & Confidence Scores (Calibrated via Temperature Scaling)
- Grounded Root Cause Analysis & Cited References
- Maintenance Recommendations
- Risk Assessment
- Inspection Timestamp & Machine Metadata
- PDF and JSON export support

---

### 9. Dashboard & Analytics (Human-in-the-Loop)
Provides a centralized monitoring interface showing:
- Inspection History & Machine Health
- Side-by-side visual image inspection
- **Human-in-the-Loop Approve / Revise Gate**: Clinicians and engineers can approve reports or edit recommendations before final database commitment
- Defect Trends & Risk Analytics

---

## Suggested Technology Stack

### Frontend & Dashboard
- Next.js / React / Streamlit
- Tailwind CSS
- TypeScript / Python

### Backend & Pipeline
- FastAPI
- Python 3.10+

### Computer Vision & ML
- PyTorch
- YOLOv11 / YOLOv8 / ResNet50 / ViT
- OpenCV & Grad-CAM

### Similarity Search & RAG
- FAISS (Facebook AI Similarity Search)
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- Semantic Scholar API / Technical Manual Parsers

### Multi-Agent AI Layer
- Local Quantized LLM (Llama 3.1 8B / Qwen2.5 7B 4-bit via Ollama / vLLM / bitsandbytes)
- Custom Orchestrator / LangGraph / CrewAI
- DeepEval / Arize Phoenix (Hallucination auditing & verification)

### Database & Storage
- PostgreSQL / SQLite
- Local Artifact Store (Images, Heatmaps, PDF Reports)

---

## Expected Output

Instead of simply returning:

> **"Crack detected."**

The platform generates a complete decision-support summary such as:

> **"A crack has been detected on the upper-right region of the component with 96% confidence. Based on historical inspections and technical guidelines, the most probable cause is material fatigue caused by prolonged cyclic loading. The defect is classified as High Severity with an estimated failure probability of 82% if left untreated within 48 operational hours. Immediate component replacement is recommended, followed by an alignment check on adjacent drive shafts. A detailed inspection report has been generated and queued for engineering review."**

---

## 3-Person Team Task Breakdown & Build Guide

The system architecture and workload are divided among a **3-person team**, ensuring clear file ownership, explicit interface contracts, and non-blocking parallel development.

```text
                                 ┌─────────────────────────────────────────┐
                                 │       Person 1: Vision & ML Lead        │
                                 │  - Dataset & Bounding Box Ingestion     │
                                 │  - Multi-Class Classifier + BBox Head   │
                                 │  - Grad-CAM & Heatmap Overlay           │
                                 │  - Temperature Scaling Calibration      │
                                 └────────────────────┬────────────────────┘
                                                      │
                                                      ▼ predict() contract
┌─────────────────────────────────────────┐           │           ┌─────────────────────────────────────────┐
│     Person 2: RAG & Multi-Agent Lead    │◄──────────┴──────────►│     Person 3: Dashboard & Full-Stack    │
│  - FAISS Guideline Vector DB Indexing   │                       │  - Interactive Web App / Dashboard      │
│  - Semantic Scholar Live Search API     │                       │  - Side-by-Side Heatmap Visualizer      │
│  - Perception & Diagnostic LLM Agents   │                       │  - Human-in-the-Loop Approve/Revise Gate│
│  - Verifier Agent & Hallucination Audit │                       │  - PDF/JSON Report Generator            │
│  - Master Pipeline Orchestration        │                       │  - Database, Testing & Deployment       │
└────────────────────┬────────────────────┘                       └────────────────────┬────────────────────┘
                     │                                                                 │
                     └──────────────────────── run_pipeline() ─────────────────────────┘
```

---

### Person 1 — Vision, Localization & Calibration Engineer

**Owns:** `src/dataset.py`, `src/vision_model.py`, `src/gradcam.py`, `src/inference.py`, `experiments/tune.py`

#### Tasks & Responsibilities:
1. **Bounding Box & Dataset Ingestion (`src/dataset.py`)**:
   - Parse image files and target metadata (bounding box coordinates `[x, y, w, h]`).
   - Normalize box coordinates to `[0, 1]` relative to image size (e.g., 224x224).
   - Return structured dictionary per sample: `{"image": tensor, "label": int, "bbox": [x,y,w,h] or None}`.
2. **Model Architecture & Dual-Head Classifier (`src/vision_model.py`)**:
   - Multi-class classifier (Normal / Defect / Structural Anomaly) using ResNet50 / YOLO / ViT backbone.
   - Dual-head branching: Classification Head (`CrossEntropyLoss`) + Regression Head (`Smooth L1 Huber Loss` for `[x, y, w, h]`).
   - Combined Loss: `loss = cls_loss + λ * bbox_loss` (with `λ = 1.0`).
   - Maintain light memory footprint to fit within a **24GB GPU budget**.
3. **Grad-CAM & Heatmap IoU Alignment (`src/gradcam.py`)**:
   - Register hooks on the final convolutional layer to generate activation maps (`generate_gradcam`).
   - Overlay heatmaps on original images (`overlay_heatmap`) and save overlay PNGs.
   - Compute IoU metric between thresholded Grad-CAM activation region and ground-truth bounding box.
4. **Model Tuning & Confidence Calibration (`experiments/tune.py`)**:
   - Run hyperparameter sweeps (learning rate, batch size, loss weighting).
   - Apply **Temperature Scaling** on validation split to calibrate classifier output probabilities.

#### Deliverable / Interface Contract:
```python
# src/inference.py
def predict(image_path: str) -> dict:
    """
    Returns:
    {
        "label": "Crack" | "Scratch" | "Dent" | "Corrosion" | "Normal",
        "confidence": 0.96,            # Calibrated probability float (0-1)
        "bbox": [x, y, w, h] | None,   # Normalized bounding box coordinates
        "heatmap_overlay_path": str    # Saved PNG path
    }
    """
```

---

### Person 2 — RAG, Multi-Agent & Root Cause Engineer

**Owns:** `src/rag_engine.py`, `src/literature_api.py`, `src/agents/perception_agent.py`, `src/agents/diagnostic_agent.py`, `src/agents/verifier_agent.py`, `src/agents/orchestrator.py`, `src/prompts.py`

#### Tasks & Responsibilities:
1. **Vector Guideline & Document Ingestion (`src/rag_engine.py`)**:
   - Chunk industrial guidelines / manuals (~500 tokens with 50 token overlap).
   - Embed text using `sentence-transformers/all-MiniLM-L6-v2`.
   - Store and persist FAISS index at `data/docs/faiss_index/`.
2. **Live Literature Retrieval API (`src/literature_api.py`)**:
   - Build API search wrapper for Semantic Scholar / external document search.
   - Combine guideline retrieval and live literature search into `retrieve_all(query: str)`.
3. **Multi-Agent AI Pipeline (`src/agents/`)**:
   - **Perception Agent (`perception_agent.py`)**: Transforms raw `predict()` output into structured findings summary.
   - **Diagnostic Agent (`diagnostic_agent.py`)**: Executes RAG search, feeds findings + retrieved context to a local offline LLM (Llama 3.1 8B / Qwen2.5 7B 4-bit) to generate root cause analysis, recommendations, and risk assessment.
   - **Verifier Agent (`verifier_agent.py`)**: Audits report for ungrounded claims/hallucinations using DeepEval / Arize Phoenix and outputs a confidence score (0–100).
4. **Master Orchestrator (`src/agents/orchestrator.py`)**:
   - Chains: Vision `predict()` → Perception → RAG → Diagnostic → Verifier → Final packaged result.

#### Deliverable / Interface Contract:
```python
# src/agents/orchestrator.py
def run_pipeline(image_path: str, patient_meta: dict) -> dict:
    """
    Single entry point for full system execution.
    Returns:
    {
        "vision_output": {"label": str, "confidence": float, "bbox": list, "heatmap_overlay_path": str},
        "findings": {"summary": str},
        "report": {
            "impression": str,
            "root_cause": str,
            "supporting_evidence": list,
            "recommended_next_steps": list
        },
        "verification": {"confidence_score": float, "flagged_claims": list, "verified": bool}
    }
    """
```

---

### Person 3 — Dashboard, Reporting & Full-Stack Integration Lead

**Owns:** `app/main.py`, `src/report_generator.py`, `src/db.py`, `tests/test_pipeline.py`, `README.md`, `requirements.txt`

#### Tasks & Responsibilities:
1. **Interactive UI Dashboard (`app/main.py`)**:
   - Build Streamlit / Next.js app with industrial image upload.
   - Display original image alongside Grad-CAM heatmap overlay side by side.
   - Display AI inspection findings, root cause, recommendations, and verification badge.
2. **Human-in-the-Loop Approve / Revise Gate**:
   - **Approve Button**: Locks report, records inspector timestamp, and logs record into SQLite database (`data/reports_log.db`).
   - **Revise Gate**: Provides editable text interface for engineers to edit/refine findings before finalizing.
3. **Professional Report Generation (`src/report_generator.py`)**:
   - Automatically compile inspection records into exportable **PDF** and **JSON** reports.
4. **Integration, Testing & Deployment (`tests/test_pipeline.py`)**:
   - Wire full pipeline end-to-end (`orchestrator.py` → `app/main.py`).
   - Maintain project `requirements.txt` and environment compatibility.
   - Write end-to-end integration tests to verify key output structure and error handling.

---

## 3-Person Team Timeline & Build Order

| Timeline | Person 1 (Vision & ML) | Person 2 (RAG & Multi-Agent) | Person 3 (Dashboard & Systems) |
|---|---|---|---|
| **Hours 0–2** | Ingest dataset, normalize BBoxes, upgrade classifier backbone. | Set up FAISS vector store, chunk text guidelines, embed chunks. | Build Streamlit UI skeleton with **mock pipeline data**. |
| **Hours 2–4** | Build BBox regression head & dual loss (`cls_loss + λ * bbox_loss`). | Build `literature_api.py` & unified RAG search (`retrieve_all`). | Build side-by-side image & Grad-CAM viewer component. |
| **Hours 4–6** | Implement Grad-CAM heatmap overlay & calculate IoU metric. | Build Perception & Diagnostic LLM Agents (local quantized model). | Implement Human-in-the-Loop Approve / Revise UI gate & DB schema. |
| **Hours 6–8** | Run Temperature Scaling confidence calibration in `experiments/tune.py`. | Build Verifier Agent (DeepEval audit) & `orchestrator.py`. | Implement PDF & JSON report generator (`report_generator.py`). |
| **Hours 8–10** | Deliver `best_vision_model.pth` & locked `inference.py`. | Deliver locked `run_pipeline()` entry point; test fallback logic. | Connect real `run_pipeline()`, run integration tests, finalize README & Demo. |

---

## Development Philosophy

The project is designed as a modular, production-ready AI platform rather than a single machine learning model. Every major functionality (vision, classification, retrieval, reasoning, recommendations, reporting, analytics) exists as an independent module with clear responsibilities and interface contracts.

The final product resembles an industrial AI inspection assistant capable of supporting engineers in real-world manufacturing environments rather than simply demonstrating image classification.