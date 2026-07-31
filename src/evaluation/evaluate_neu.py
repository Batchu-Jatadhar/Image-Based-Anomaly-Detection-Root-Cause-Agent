import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

from src.config import config
from src.datasets.neu_dataset import NEUDataset, NEU_CLASSES
from src.models.classification_model import NEUClassificationModel

def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("NEU Surface Defect Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curves(y_true_onehot: np.ndarray, y_probs: np.ndarray, class_names: list, output_path: Path):
    plt.figure(figsize=(8, 6))
    n_classes = len(class_names)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("NEU Classification ROC-AUC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_neu(
    data_dir: str = "NEU-DET",
    model_path: Optional[str] = None,
    split: str = "validation"
) -> Dict:
    print(f"\n================ Running NEU Surface Defect Model Evaluation ================")
    data_path = Path(data_dir)
    weights_path = Path(model_path) if model_path else config.weights_dir / "best_model_neu.pth"
    metrics_dir = config.metrics_dir / "neu"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Split: {split} | Checkpoint: {weights_path}")

    # Load dataset & model
    val_dataset = NEUDataset(data_dir=data_path, split=split)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = NEUClassificationModel(num_classes=6, backbone_name="resnet50", pretrained=False)
    model.load_checkpoint(weights_path, device=device)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Compute overall metrics
    overall_acc = accuracy_score(all_targets, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro")
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted")

    # One-hot encode targets for ROC-AUC
    n_classes = len(NEU_CLASSES)
    y_true_onehot = np.eye(n_classes)[all_targets]
    roc_auc_macro = roc_auc_score(y_true_onehot, all_probs, average="macro", multi_class="ovr")

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)
    cm = confusion_matrix(all_targets, all_preds)

    per_class_acc = {}
    per_class_metrics = {}
    for i, cls_name in enumerate(NEU_CLASSES):
        cls_acc = float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
        per_class_acc[cls_name] = round(cls_acc, 4)
        per_class_metrics[cls_name] = {
            "accuracy": round(cls_acc, 4),
            "precision": round(float(precision_per_class[i]), 4),
            "recall": round(float(recall_per_class[i]), 4),
            "f1_score": round(float(f1_per_class[i]), 4)
        }

    results = {
        "overall_accuracy": round(float(overall_acc), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "roc_auc_macro": round(float(roc_auc_macro), 4),
        "per_class_accuracy": per_class_acc,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }

    # Plots
    plot_confusion_matrix(cm, NEU_CLASSES, metrics_dir / "confusion_matrix.png")
    plot_roc_curves(y_true_onehot, all_probs, NEU_CLASSES, metrics_dir / "roc_auc_curves.png")

    # Save JSON report
    metrics_json_path = metrics_dir / "neu_evaluation_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[NEU Evaluation Complete]")
    print(f"  - Accuracy        : {overall_acc * 100:.2f}%")
    print(f"  - F1 Score (Macro): {f1_macro * 100:.2f}%")
    print(f"  - ROC-AUC (Macro) : {roc_auc_macro:.4f}")
    print(f"Metrics report saved to: {metrics_json_path}")
    print("==========================================================================")
    return results

if __name__ == "__main__":
    evaluate_neu()
