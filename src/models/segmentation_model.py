import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
from typing import Optional, Dict

class ConvBlock(nn.Module):
    """
    Standard Double Convolution Block for UNet Decoder.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetResNet34(nn.Module):
    """
    U-Net Segmentation Model with ResNet34 Encoder Backbone.
    Produces 4 output channels (one per Severstal defect class).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Encoder stages
        self.init_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        ) # (B, 64, H/2, W/2)
        self.maxpool = resnet.maxpool # (B, 64, H/4, W/4)

        self.layer1 = resnet.layer1 # (B, 64, H/4, W/4)
        self.layer2 = resnet.layer2 # (B, 128, H/8, W/8)
        self.layer3 = resnet.layer3 # (B, 256, H/16, W/16)
        self.layer4 = resnet.layer4 # (B, 512, H/32, W/32)

        # Center / Bottleneck
        self.bottleneck = ConvBlock(512, 512)

        # Decoder stages
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(32, 32)

        # Final segmentation head (4 channels for 4 defect classes)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.init_conv(x)  # 64, H/2, W/2
        x1 = self.maxpool(x0)   # 64, H/4, W/4
        e1 = self.layer1(x1)    # 64, H/4, W/4
        e2 = self.layer2(e1)    # 128, H/8, W/8
        e3 = self.layer3(e2)    # 256, H/16, W/16
        e4 = self.layer4(e3)    # 512, H/32, W/32

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with Skip Connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = self.dec0(d0)

        logits = self.final_conv(d0)
        return logits


class SeverstalSegmentationModel(nn.Module):
    """
    Wrapper segmentation model for Severstal Steel Defect Detection (4 classes).
    Supports ResNet34 encoder backbone (UNet architecture).
    """
    def __init__(self, num_classes: int = 4, backbone_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name.lower()
        self.model = UNetResNet34(in_channels=3, out_channels=num_classes, pretrained=pretrained)

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


class DiceBCELoss(nn.Module):
    """
    Combined BCE + Dice Loss for multi-label binary segmentation.
    """
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_fn(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
