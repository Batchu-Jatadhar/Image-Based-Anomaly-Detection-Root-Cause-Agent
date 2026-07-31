import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
from typing import Dict, Tuple, Optional

class DualHeadVisionModel(nn.Module):
    """
    Dual-Head Computer Vision Model combining Multi-Class Defect Classification
    and Bounding Box Regression [x, y, w, h] branching off a shared ResNet backbone.
    """
    def __init__(self, num_classes: int = 3, backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            in_features = resnet.fc.in_features
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        elif self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            in_features = resnet.fc.in_features
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.in_features = in_features

        # Head 1: Multi-Class Defect Classification Head
        self.fc_cls = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # Head 2: Bounding Box Regression Head [x, y, w, h]
        self.fc_bbox = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()  # Restricts normalized box predictions to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        feats_flat = torch.flatten(feats, 1)

        cls_logits = self.fc_cls(feats_flat)
        bbox_pred = self.fc_bbox(feats_flat)

        return {
            "cls_logits": cls_logits,
            "bbox_pred": bbox_pred
        }


class CombinedDualLoss(nn.Module):
    """
    Combined Loss function balancing Multi-Class Classification (CrossEntropy)
    and Bounding Box Regression (Smooth L1 Huber Loss).
    loss = cls_loss + λ * bbox_loss
    """
    def __init__(self, lambda_bbox: float = 1.0):
        super().__init__()
        self.lambda_bbox = lambda_bbox
        self.cls_criterion = nn.CrossEntropyLoss()
        self.bbox_criterion = nn.SmoothL1Loss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        cls_targets: torch.Tensor,
        bbox_targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_loss = self.cls_criterion(outputs["cls_logits"], cls_targets)

        if bbox_targets is not None and bbox_targets.numel() > 0:
            # Mask for samples containing valid bounding boxes (non-zero)
            has_bbox = (bbox_targets.sum(dim=1) > 0)
            if has_bbox.sum() > 0:
                bbox_loss = self.bbox_criterion(outputs["bbox_pred"][has_bbox], bbox_targets[has_bbox])
            else:
                bbox_loss = torch.tensor(0.0, device=cls_loss.device)
        else:
            bbox_loss = torch.tensor(0.0, device=cls_loss.device)

        total_loss = cls_loss + self.lambda_bbox * bbox_loss
        return total_loss, cls_loss, bbox_loss
