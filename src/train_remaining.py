import os
import time
import json
from pathlib import Path
from src.config import config
from src.train import train_model
from src.inference import predict
from src.dataset import MVTecDataset
from src.evaluation import evaluate_predictions, compute_iou

def train_remaining_and_evaluate(epochs: int = 5):
    dataset_dir = config.dataset_dir
    all_categories = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    
    missing_categories = []
    trained_categories = []
    
    for cat in all_categories:
        ckpt_path = config.weights_dir / f"best_model_{cat}.pth"
        if ckpt_path.exists():
            trained_categories.append(cat)
        else:
            missing_categories.append(cat)
            
    print(f"\n==========================================================================")
    print(f"Status Check:")
    print(f"  - Total categories found in dataset: {len(all_categories)}")
    print(f"  - Already trained ({len(trained_categories)}): {trained_categories}")
    print(f"  - Pending training ({len(missing_categories)}): {missing_categories}")
    print(f"==========================================================================\n")
    
    if missing_categories:
        print(f"Starting training for {len(missing_categories)} remaining categories...")
        start_time = time.time()
        results = {}
        for idx, cat in enumerate(missing_categories, 1):
            print(f"\n[{idx}/{len(missing_categories)}] Training Category: {cat} ({epochs} epochs)...")
            try:
                ckpt = train_model(category=cat, epochs=epochs)
                results[cat] = "SUCCESS"
            except Exception as e:
                print(f"[Error] Category {cat} failed: {e}")
                results[cat] = f"FAILED: {e}"
                
        total_time = round((time.time() - start_time) / 60, 2)
        print(f"\nRemaining categories training completed in {total_time} mins.")
    else:
        print("All categories have already been trained! Proceeding directly to full evaluation.")

    # Full Evaluation across ALL categories
    print("\n================ Running Full Evaluation Across All 15 Categories ================")
    summary_metrics = {}
    
    for cat in all_categories:
        try:
            test_ds = MVTecDataset(dataset_dir=config.dataset_dir, category=cat, split="test")
            # Strict Data-Leakage Protection: Exclude training validation split [::2], evaluate strictly on held-out test set [1::2]
            test_ds.samples = test_ds.samples[1::2]
            y_true = []
            y_pred = []
            ious = []
            
            for item in test_ds:
                img_path = item["image_path"]
                gt_is_anomaly = item["is_anomaly"]
                gt_bbox = item["bbox"]
                
                pred_res = predict(img_path, category=cat, use_mock=False)
                pred_is_anomaly = pred_res["is_anomaly"]
                pred_bbox = pred_res["bbox"]
                
                y_true.append(gt_is_anomaly)
                y_pred.append(pred_is_anomaly)
                
                if gt_bbox and pred_bbox:
                    iou = compute_iou(gt_bbox, pred_bbox)
                    ious.append(iou)
                    
            metrics = evaluate_predictions(y_true, y_pred, ious)
            summary_metrics[cat] = metrics
            print(f"  - [{cat:12s}] Acc: {metrics['accuracy']*100:6.2f}% | Precision: {metrics['precision']*100:6.2f}% | Recall: {metrics['recall']*100:6.2f}% | F1: {metrics['f1_score']*100:6.2f}%")
        except Exception as e:
            print(f"  - [{cat:12s}] Evaluation failed: {e}")
            summary_metrics[cat] = {"error": str(e)}

    # Save metrics JSON
    metrics_path = config.metrics_dir / "full_dataset_evaluation.json"
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)
        
    print(f"\nFull evaluation metrics saved to: {metrics_path}")
    print("==========================================================================")

if __name__ == "__main__":
    train_remaining_and_evaluate(epochs=5)
