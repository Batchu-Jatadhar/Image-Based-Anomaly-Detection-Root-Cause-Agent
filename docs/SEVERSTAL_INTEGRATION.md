# Severstal Steel Defect Detection Integration & Technical Documentation

## Overview

The Vision Engine has been extended to support the **Severstal Steel Defect Detection Dataset** for industrial multi-label semantic segmentation alongside the existing **MVTec AD Anomaly Detection Engine** and **NEU Surface Defect Classification Engine**.

This milestone completes the unified vision architecture, supporting three distinct industrial vision tasks through a single, clean inference interface.

---

## 1. Directory Structure

```text
Image-Based-Anomaly-Detection-Root-Cause-Agent/
├── severstal dataset/               # Severstal Steel Defect Dataset
│   ├── train_images/                # Steel Sheet Images (256x1600)
│   ├── test_images/
│   └── train.csv                    # RLE Masks & Class Annotations
├── src/
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── neu_dataset.py           # NEU 6-Class Classification Loader
│   │   └── severstal_dataset.py     # Phase 1: Severstal Dataset Loader & RLE Decoder
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classification_model.py  # NEU ResNet50 Classifier
│   │   └── segmentation_model.py    # Phase 2: UNet ResNet34 Encoder + DiceBCELoss
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_neu.py             # NEU Training Script
│   │   └── train_severstal.py       # Phase 3: UNet Mixed-Precision Training
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate_neu.py          # NEU Evaluation Suite
│   │   ├── mvtec_evaluation.py      # MVTec Evaluation Suite
│   │   └── evaluate_severstal.py    # Phase 4: Dice, IoU, Pixel Acc & Overlay Renderer
│   ├── inference.py                 # Phase 5: Unified Multi-Dataset Predictor (MVTec, NEU, Severstal)
│   └── dataset.py                   # MVTec Dataset Loader (UNTOUCHED)
├── outputs/
│   ├── weights/
│   │   ├── best_model_severstal.pth # Saved Severstal UNet Checkpoint
│   │   ├── best_model_neu.pth       # Saved NEU Checkpoint
│   │   └── best_model_*.pth         # Saved MVTec Checkpoints (15 categories)
│   └── metrics/
│       ├── severstal/               # Severstal Evaluation Artifacts
│       │   ├── loss_plot.png
│       │   ├── iou_plot.png
│       │   ├── dice_plot.png
│       │   ├── sample_predictions_overlay.png
│       │   └── severstal_evaluation_metrics.json
│       ├── neu/                     # NEU Evaluation Artifacts
│       └── full_dataset_evaluation.json
└── docs/
    ├── NEU_INTEGRATION.md
    └── SEVERSTAL_INTEGRATION.md     # Documentation
```

---

## 2. RLE Mask Decoding & Dataset Format

The Severstal dataset consists of 4 defect classes annotated via Run-Length Encoding (RLE) in `train.csv`.

RLE decoding converts 1-indexed, column-major strings into binary segmentation masks:

```python
from src.datasets.severstal_dataset import rle_to_mask, mask_to_rle

# Decode RLE string into binary mask (256, 1600)
mask = rle_to_mask("29102 12 29346 24 ...", height=256, width=1600)
```

The PyTorch dataset adapter `src/datasets/severstal_dataset.py` produces:
- 4-channel binary target tensors `(4, H, W)` handling multi-defect instances on the same steel sheet.
- Deterministic 80/20 train/validation splits.
- Automatic preview grid generator `visualize_severstal_samples()`.

---

## 3. UNet Segmentation Model Architecture

- **Architecture**: U-Net Semantic Segmentation Network.
- **Encoder Backbone**: Pretrained `ResNet34` (configurable to `EfficientNet-B0`).
- **Decoder**: Up-sampling transposed convolutions with skip connections.
- **Loss Function**: `DiceBCELoss` (Combination of Binary Cross-Entropy and Dice Loss).
- **Optimization**: `AdamW` optimizer, `CosineAnnealingLR` scheduler, CUDA `autocast` mixed precision, and early stopping.
- **Checkpoint Location**: `outputs/weights/best_model_severstal.pth`.

---

## 4. Evaluation Metrics & Visualizations

The evaluation module `src/evaluation/evaluate_severstal.py` calculates:
- **Dice Score** (Overall & Per-Class)
- **IoU (Intersection over Union)**
- **Pixel Accuracy**
- **Precision & Recall**
- **Overlay Rendering**: Saves visual comparison grids (Original Image vs Ground Truth vs UNet Prediction) to `outputs/metrics/severstal/sample_predictions_overlay.png`.

---

## 5. Unified Multi-Dataset Inference API

The `predict()` function in `src/inference.py` supports all three datasets through one unified interface:

```python
from src.inference import predict

# 1. Industrial Anomaly Detection (MVTec AD)
mvtec_res = predict("dataset/bottle/test/broken_large/000.png", category="bottle", dataset="mvtec")

# 2. Industrial Defect Classification (NEU Surface Defect)
neu_res = predict("NEU-DET/validation/images/scratches/scratches_241.jpg", dataset="neu")

# 3. Industrial Semantic Segmentation (Severstal Steel)
severstal_res = predict("severstal dataset/train_images/0002cc93b.jpg", dataset="severstal")
```

### Severstal Inference Response Schema:
```json
{
    "prediction_id": "pred_40057895",
    "dataset": "severstal",
    "task": "segmentation",
    "predicted_classes": ["Class 1", "Class 3"],
    "confidence": 0.97,
    "mask_path": "outputs/heatmaps/pred_40057895_mask.png",
    "overlay_path": "outputs/heatmaps/pred_40057895_overlay.png",
    "defect_area_px": 4396,
    "defect_area_percent": 1.07,
    "processing_time": 0.045
}
```

---

## 6. Integration Architecture Summary

| Dataset | Vision Task | Model Architecture | Primary Output |
|---|---|---|---|
| **MVTec AD** | Industrial Anomaly Detection | Dual-Head ResNet18 + Grad-CAM | Anomaly Score, Bounding Box, Heatmap Overlay, 512d Embedding |
| **NEU Surface Defect** | Industrial Defect Classification | ResNet50 Classifier | Defect Category (6 classes), Confidence Score |
| **Severstal Steel** | Industrial Semantic Segmentation | UNet (ResNet34 Encoder) | Multi-Class Segmentation Mask, Overlay PNG, Defect Area Px / % |
