import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from typing import Optional, Dict

class NEUClassificationModel(nn.Module):
    """
    Multi-class classification model for NEU Surface Defect Dataset (6 classes).
    Supports ResNet50, ResNet18, and MobileNetV3 backbones with transfer learning.
    """
    def __init__(self, num_classes: int = 6, backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif self.backbone_name == "mobilenet_v3":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v3_small(weights=weights)
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from ['resnet50', 'resnet18', 'mobilenet_v3']")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save_checkpoint(self, path: Path, extra_info: Optional[Dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "backbone_name": self.backbone_name,
            "num_classes": self.num_classes,
        }
        if extra_info:
            checkpoint.update(extra_info)
        torch.save(checkpoint, str(path))

    def load_checkpoint(self, path: Path, device: torch.device):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(str(path), map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)
        self.to(device)
        self.eval()


class EarlyStopping:
    """
    Early stopping helper to halt training when validation loss stops improving.
    """
    def __init__(self, patience: int = 5, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement
