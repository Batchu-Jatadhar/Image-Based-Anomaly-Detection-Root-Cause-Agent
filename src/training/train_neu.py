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
from src.datasets.neu_dataset import NEUDataset
from src.models.classification_model import NEUClassificationModel, EarlyStopping

def plot_training_history(history: dict, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="red", linestyle="--", linewidth=2)
    plt.title("NEU Classification Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_plot.png")
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [acc * 100 for acc in history["train_acc"]], label="Train Accuracy (%)", color="green", linewidth=2)
    plt.plot(epochs, [acc * 100 for acc in history["val_acc"]], label="Val Accuracy (%)", color="orange", linestyle="--", linewidth=2)
    plt.title("NEU Classification Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_plot.png")
    plt.close()

def train_neu(
    data_dir: str = "NEU-DET",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    backbone: str = "resnet50",
    patience: int = 5
) -> str:
    print(f"\n================ Starting NEU Surface Defect Model Training ================")
    data_path = Path(data_dir)
    weights_dir = config.weights_dir
    metrics_dir = config.metrics_dir / "neu"
    weights_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dataset = NEUDataset(data_dir=data_path, split="train")
    val_dataset = NEUDataset(data_dir=data_path, split="validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Backbone: {backbone} | Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model, Criterion, Optimizer, Scheduler, Mixed Precision Scaler
    model = NEUClassificationModel(num_classes=6, backbone_name=backbone, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    early_stopping = EarlyStopping(patience=patience)

    best_val_acc = 0.0
    saved_model_path = weights_dir / "best_model_neu.pth"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")

        # Save Best Checkpoint
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            model.save_checkpoint(saved_model_path, extra_info={"best_val_acc": best_val_acc, "epoch": epoch})

        # Early Stopping Check
        if early_stopping(epoch_val_loss):
            print(f"[Early Stopping] Training stopped early at epoch {epoch}")
            break

    total_duration = round((time.time() - start_time) / 60, 2)
    print(f"\n[Training Completed in {total_duration} mins] Best Val Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Model saved to: {saved_model_path}")

    # Plot & save training history curves
    plot_training_history(history, metrics_dir)
    with open(metrics_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("==========================================================================")
    return str(saved_model_path)

if __name__ == "__main__":
    train_neu(epochs=15, batch_size=16, lr=1e-4)
