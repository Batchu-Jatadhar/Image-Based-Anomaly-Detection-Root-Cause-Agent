import os
import json
import numpy as np
from typing import Dict, List

# ponytail: simple metric calculation suite for accuracy, precision, recall, F1, IoU
# ceiling: numpy scalar evaluation metrics; upgrade: scikit-learn & torchvision eval suite.

def compute_iou(box1: list, box2: list) -> float:
    """
    Computes Intersection over Union (IoU) between two normalized boxes [x, y, w, h].
    """
    if not box1 or not box2:
        return 0.0
        
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    union_area = b1_area + b2_area - inter_area
    
    if union_area <= 0:
        return 0.0
        
    return round(float(inter_area / union_area), 4)

def evaluate_predictions(y_true: List[bool], y_pred: List[bool], ious: List[float] = None) -> Dict:
    """
    Computes summary classification and localization metrics.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    
    total = len(y_true)
    accuracy = round((tp + tn) / total, 4) if total > 0 else 0.0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    mean_iou = round(float(np.mean(ious)), 4) if ious else 0.0
    
    return {
        "total_samples": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_iou": mean_iou,
        "confusion_matrix": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        }
    }
