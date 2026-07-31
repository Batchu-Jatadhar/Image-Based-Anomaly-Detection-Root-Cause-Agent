import os
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUTS_DIR = BASE_DIR / "outputs"
WEIGHTS_DIR = OUTPUTS_DIR / "weights"
METRICS_DIR = OUTPUTS_DIR / "metrics"
HEATMAPS_DIR = OUTPUTS_DIR / "heatmaps"

# Ensure output directories exist
for folder in [OUTPUTS_DIR, WEIGHTS_DIR, METRICS_DIR, HEATMAPS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

@dataclass
class VisionConfig:
    # Dataset Settings
    dataset_dir: Path = DATASET_DIR
    default_category: str = "bottle"
    img_size: int = 224
    
    # Supported Backbones: "yolov11", "yolov8", "resnet50", "mobilenet_v3"
    backbone: str = "yolov11"
    fallback_backbone: str = "resnet50"
    
    # Anomaly Detection Settings
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    
    # Feature Embeddings Settings
    embedding_dim: int = 512
    
    # Temperature Calibration Factor
    temperature: float = 1.2
    
    # Processing Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output Directory Paths
    weights_dir: Path = WEIGHTS_DIR
    metrics_dir: Path = METRICS_DIR
    heatmaps_dir: Path = HEATMAPS_DIR

config = VisionConfig()
