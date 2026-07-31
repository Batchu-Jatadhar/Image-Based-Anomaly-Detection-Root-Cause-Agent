import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import config
from src.datasets.severstal_dataset import SeverstalDataset
from src.models.segmentation_model import SeverstalSegmentationModel, DiceBCELoss
from src.models.classification_model import EarlyStopping

def calculate_batch_iou_dice(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0):
    probs = torch.sigmoid(preds)
    binary_preds = (probs > threshold).float()

    intersection = (binary_preds * targets).sum(dim=(2, 3))
    total_preds = binary_preds.sum(dim=(2, 3))
    total_targets = targets.sum(dim=(2, 3))

    union = total_preds + total_targets - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (total_preds + total_targets + smooth)

    return iou.mean().item(), dice.mean().item()


def plot_segmentation_history(history: dict, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="red", linestyle="--", linewidth=2)
    plt.title("Severstal UNet Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_plot.png")
    plt.close()

    # IoU Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_iou"], label="Train IoU", color="green", linewidth=2)
    plt.plot(epochs, history["val_iou"], label="Val IoU", color="orange", linestyle="--", linewidth=2)
    plt.title("Severstal UNet Training & Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "iou_plot.png")
    plt.close()

    # Dice Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_dice"], label="Train Dice", color="purple", linewidth=2)
    plt.plot(epochs, history["val_dice"], label="Val Dice", color="cyan", linestyle="--", linewidth=2)
    plt.title("Severstal UNet Training & Validation Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "dice_plot.png")
    plt.close()


def train_severstal(
    data_dir: str = "severstal dataset",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 2e-4,
    backbone: str = "resnet34",
    patience: int = 5
) -> str:
    print(f"\n================ Starting Severstal Steel Defect UNet Training ================")
    data_path = Path(data_dir)
    weights_dir = config.weights_dir
    metrics_dir = config.metrics_dir / "severstal"
    weights_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dataset = SeverstalDataset(data_dir=data_path, split="train", target_size=(256, 512))
    val_dataset = SeverstalDataset(data_dir=data_path, split="val", target_size=(256, 512))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Encoder: {backbone} | Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    model = SeverstalSegmentationModel(num_classes=4, backbone_name=backbone, pretrained=True).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    early_stopping = EarlyStopping(patience=patience)

    best_val_dice = 0.0
    saved_model_path = weights_dir / "best_model_severstal.pth"

    history = {
        "train_loss": [], "train_iou": [], "train_dice": [],
        "val_loss": [], "val_iou": [], "val_dice": []
    }
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        total_train = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_curr = images.size(0)
            running_loss += loss.item() * batch_size_curr
            iou, dice = calculate_batch_iou_dice(logits, masks)
            running_iou += iou * batch_size_curr
            running_dice += dice * batch_size_curr
            total_train += batch_size_curr

        epoch_train_loss = running_loss / total_train
        epoch_train_iou = running_iou / total_train
        epoch_train_dice = running_dice / total_train

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_iou = 0.0
        running_val_dice = 0.0
        total_val = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, masks)

                batch_size_curr = images.size(0)
                running_val_loss += loss.item() * batch_size_curr
                iou, dice = calculate_batch_iou_dice(logits, masks)
                running_val_iou += iou * batch_size_curr
                running_val_dice += dice * batch_size_curr
                total_val += batch_size_curr

        epoch_val_loss = running_val_loss / total_val
        epoch_val_iou = running_val_iou / total_val
        epoch_val_dice = running_val_dice / total_val

        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["train_iou"].append(epoch_train_iou)
        history["train_dice"].append(epoch_train_dice)
        history["val_loss"].append(epoch_val_loss)
        history["val_iou"].append(epoch_val_iou)
        history["val_dice"].append(epoch_val_dice)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} IoU: {epoch_train_iou:.4f} Dice: {epoch_train_dice:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} IoU: {epoch_val_iou:.4f} Dice: {epoch_val_dice:.4f}"
        )

        # Save Best Checkpoint
        if epoch_val_dice >= best_val_dice:
            best_val_dice = epoch_val_dice
            model.save_checkpoint(saved_model_path, extra_info={"best_val_dice": best_val_dice, "epoch": epoch})

        # Early Stopping Check
        if early_stopping(epoch_val_loss):
            print(f"[Early Stopping] Training stopped early at epoch {epoch}")
            break

    total_duration = round((time.time() - start_time) / 60, 2)
    print(f"\n[Training Completed in {total_duration} mins] Best Val Dice: {best_val_dice:.4f}")
    print(f"Model checkpoint saved to: {saved_model_path}")

    # Plot & Save curves and training history
    plot_segmentation_history(history, metrics_dir)
    with open(metrics_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("==========================================================================")
    return str(saved_model_path)

if __name__ == "__main__":
    train_severstal(epochs=10, batch_size=8)
