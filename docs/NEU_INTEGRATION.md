# NEU Surface Defect Dataset Integration & Technical Documentation

## Overview

The Vision Engine has been extended to support the **NEU Surface Defect Dataset** for hot-rolled steel strip surface defect classification alongside the existing **MVTec AD Anomaly Detection Engine**.

This integration maintains full backward compatibility with the MVTec AD pipeline while establishing a clean, modular pattern for multi-dataset extension.

---

## 1. Directory Structure

```text
Image-Based-Anomaly-Detection-Root-Cause-Agent/
├── NEU-DET/                         # NEU Surface Defect Dataset
│   ├── train/
│   │   ├── images/                  # 6 Defect Class Directories (crazing, inclusion, etc.)
│   │   └── annotations/             # XML Bounding Box Annotations
│   └── validation/
│       ├── images/                  # Validation set (6 Defect Classes)
│       └── annotations/
├── src/
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── neu_dataset.py           # Phase 1: NEU Dataset Loader & Augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   └── classification_model.py  # Phase 2: ResNet50 Classifier + EarlyStopping
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_neu.py             # Phase 3: Mixed-precision training pipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate_neu.py          # Phase 4: Metrics, Confusion Matrix, ROC-AUC
│   ├── inference.py                 # Phase 5: Multi-dataset inference API (MVTec + NEU)
│   └── dataset.py                   # Preserved MVTec AD loader (UNTOUCHED)
├── outputs/
│   ├── weights/
│   │   ├── best_model_neu.pth       # Saved NEU Checkpoint (99.72% Val Acc)
│   │   └── best_model_*.pth         # Preserved 15 MVTec Category Checkpoints
│   └── metrics/
│       └── neu/                     # NEU Evaluation Artifacts
│           ├── accuracy_plot.png
│           ├── loss_plot.png
│           ├── confusion_matrix.png
│           ├── roc_auc_curves.png
│           └── neu_evaluation_metrics.json
└── docs/
    └── NEU_INTEGRATION.md           # Documentation
```

---

## 2. Dataset Format & Classes

The NEU dataset consists of 6 distinct surface defect categories:
1. `crazing`
2. `inclusion`
3. `patches`
4. `pitted_surface`
5. `rolled-in_scale`
6. `scratches`

The PyTorch dataset adapter `src/datasets/neu_dataset.py` handles:
- Automatic detection and mapping of all 6 classes.
- Random horizontal/vertical flips, color jitter, resizing to 224x224, and ImageNet normalization during training.
- Deterministic preprocessing during evaluation and inference.

---

## 3. Classification Model Architecture & Training

- **Backbone**: Transfer learning via `ResNet50` (configurable to `ResNet18` or `MobileNetV3`).
- **Head**: 6-class linear classification head.
- **Optimizer & Scheduler**: `AdamW` (lr=1e-4, weight_decay=1e-4) with `CosineAnnealingLR`.
- **Mixed Precision**: CUDA Automatic Mixed Precision (`autocast` + `GradScaler`) for maximum throughput.
- **Early Stopping**: Monitors validation loss with a patience of 5 epochs.
- **Checkpoint Location**: `outputs/weights/best_model_neu.pth`.

---

## 4. Evaluation Metrics & Visualizations

The evaluation module `src/evaluation/evaluate_neu.py` produces:
- **Accuracy**: `99.72%`
- **Macro F1-Score**: `99.72%`
- **Macro ROC-AUC**: `1.0000`
- **Visual Artifacts**: Saved in `outputs/metrics/neu/`:
  - `confusion_matrix.png`
  - `roc_auc_curves.png`
  - `loss_plot.png`
  - `accuracy_plot.png`
  - `neu_evaluation_metrics.json`

---

## 5. Unified Inference API

The `predict()` function in `src/inference.py` supports both `mvtec` and `neu` datasets:

```python
from src.inference import predict

# MVTec AD Inference
mvtec_result = predict("dataset/bottle/test/broken_large/000.png", category="bottle", dataset="mvtec")

# NEU Surface Defect Inference
neu_result = predict("NEU-DET/validation/images/scratches/scratches_241.jpg", dataset="neu")
```

### NEU Output Schema:
```json
{
    "prediction_id": "pred_7c1ebbef",
    "dataset": "neu",
    "task": "classification",
    "predicted_class": "scratches",
    "confidence": 0.9993,
    "processing_time": 0.331
}
```

---

## 6. Integration Points & Modular Design

- **Decoupled Datasets**: `src/datasets/` isolates NEU logic from MVTec (`src/dataset.py`).
- **Decoupled Models**: `src/models/` abstracts classification heads for easy addition of future datasets (e.g. Severstal).
- **Decoupled Pipelines**: `src/training/` and `src/evaluation/` keep dataset-specific pipelines independent.
