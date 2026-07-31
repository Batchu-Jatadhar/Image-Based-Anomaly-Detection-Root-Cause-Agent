import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List
from torch.utils.data import DataLoader

from src.config import config
from src.datasets.severstal_dataset import SeverstalDataset, SEVERSTAL_CLASSES
from src.models.segmentation_model import SeverstalSegmentationModel

def evaluate_severstal(
    data_dir: str = "severstal dataset",
    model_path: Optional[str] = None,
    split: str = "val",
    threshold: float = 0.5,
    num_vis_samples: int = 4
) -> Dict:
    print(f"\n================ Running Severstal Steel Defect Model Evaluation ================")
    data_path = Path(data_dir)
    weights_path = Path(model_path) if model_path else config.weights_dir / "best_model_severstal.pth"
    metrics_dir = config.metrics_dir / "severstal"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Split: {split} | Checkpoint: {weights_path}")

    val_dataset = SeverstalDataset(data_dir=data_path, split=split, target_size=(256, 512))
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = SeverstalSegmentationModel(num_classes=4, backbone_name="resnet34", pretrained=False)
    model.load_checkpoint(weights_path, device=device)

    total_intersection = np.zeros(4)
    total_union = np.zeros(4)
    total_pred = np.zeros(4)
    total_target = np.zeros(4)

    total_pixel_correct = 0
    total_pixels = 0

    vis_images = []
    vis_count = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # Global confusion counts per channel
            for c in range(4):
                pred_c = preds[:, c].cpu().numpy()
                target_c = masks[:, c].cpu().numpy()

                inter = (pred_c * target_c).sum()
                union = np.logical_or(pred_c, target_c).sum()

                total_intersection[c] += inter
                total_union[c] += union
                total_pred[c] += pred_c.sum()
                total_target[c] += target_c.sum()

            # Pixel Accuracy
            total_pixel_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

            # Visualization sample collection
            if vis_count < num_vis_samples:
                for b in range(images.size(0)):
                    if vis_count >= num_vis_samples:
                        break
                    # Extract original RGB image
                    img_np = images[b].cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_orig = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
                    img_orig_bgr = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)

                    gt_mask_np = masks[b].cpu().numpy()
                    pred_mask_np = preds[b].cpu().numpy()

                    gt_color = np.zeros_like(img_orig_bgr)
                    pred_color = np.zeros_like(img_orig_bgr)

                    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]
                    for c in range(4):
                        gt_color[gt_mask_np[c] == 1] = colors[c]
                        pred_color[pred_mask_np[c] == 1] = colors[c]

                    gt_overlay = cv2.addWeighted(img_orig_bgr, 0.7, gt_color, 0.3, 0)
                    pred_overlay = cv2.addWeighted(img_orig_bgr, 0.7, pred_color, 0.3, 0)

                    cv2.putText(gt_overlay, "Ground Truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(pred_overlay, "UNet Prediction", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    vis_row = np.hstack([img_orig_bgr, gt_overlay, pred_overlay])
                    vis_images.append(vis_row)
                    vis_count += 1

    # Per-class Dice and IoU
    smooth = 1e-6
    iou_per_class = (total_intersection + smooth) / (total_union + smooth)
    dice_per_class = (2.0 * total_intersection + smooth) / (total_pred + total_target + smooth)
    precision_per_class = (total_intersection + smooth) / (total_pred + smooth)
    recall_per_class = (total_intersection + smooth) / (total_target + smooth)

    overall_pixel_acc = total_pixel_correct / total_pixels
    mean_iou = float(np.mean(iou_per_class))
    mean_dice = float(np.mean(dice_per_class))
    mean_precision = float(np.mean(precision_per_class))
    mean_recall = float(np.mean(recall_per_class))

    per_class_metrics = {}
    for c, cls_name in enumerate(SEVERSTAL_CLASSES):
        per_class_metrics[cls_name] = {
            "iou": round(float(iou_per_class[c]), 4),
            "dice": round(float(dice_per_class[c]), 4),
            "precision": round(float(precision_per_class[c]), 4),
            "recall": round(float(recall_per_class[c]), 4)
        }

    results = {
        "overall_pixel_accuracy": round(float(overall_pixel_acc), 4),
        "mean_iou": round(mean_iou, 4),
        "mean_dice_score": round(mean_dice, 4),
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "per_class_metrics": per_class_metrics
    }

    # Save visualization overlay image
    if vis_images:
        vis_grid = np.vstack(vis_images)
        output_vis_path = metrics_dir / "sample_predictions_overlay.png"
        cv2.imwrite(str(output_vis_path), vis_grid)

    # Save JSON report
    metrics_json_path = metrics_dir / "severstal_evaluation_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[Severstal Evaluation Complete]")
    print(f"  - Pixel Accuracy : {overall_pixel_acc * 100:.2f}%")
    print(f"  - Mean IoU       : {mean_iou:.4f}")
    print(f"  - Mean Dice Score: {mean_dice:.4f}")
    print(f"  - Mean Precision : {mean_precision:.4f}")
    print(f"  - Mean Recall    : {mean_recall:.4f}")
    print(f"Metrics report saved to: {metrics_json_path}")
    print("==========================================================================")
    return results

if __name__ == "__main__":
    evaluate_severstal()
