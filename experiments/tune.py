import os
import json
import torch
import numpy as np
from pathlib import Path
from src.config import config
from src.calibration import TemperatureScaler, compute_ece, calibrate_model_logits

def run_calibration_sweep(category: str = "bottle") -> dict:
    """
    Simulates validation logits for temperature scaling optimization and ECE reporting.
    """
    print(f"\n================ Running Temperature Scaling Calibration [{category}] ================")
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic validation logits & labels for calibration demo
    num_samples = 200
    num_classes = 2

    val_labels = torch.randint(0, num_classes, (num_samples,))
    # Overconfident uncalibrated logits
    val_logits = torch.randn(num_samples, num_classes) * 3.5
    for i in range(num_samples):
        val_logits[i, val_labels[i]] += 2.5

    results = calibrate_model_logits(val_logits, val_labels)

    print(f"  - Optimal Temperature (T) : {results['optimal_temperature']}")
    print(f"  - ECE Before Calibration   : {results['ece_before_calibration'] * 100:.2f}%")
    print(f"  - ECE After Calibration    : {results['ece_after_calibration'] * 100:.2f}%")
    print(f"  - Calibration Improvement  : {results['calibration_improvement'] * 100:.2f}%")

    out_dir = config.metrics_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"calibration_summary_{category}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Calibration summary saved to: {out_file}")
    print("==========================================================================")
    return results

if __name__ == "__main__":
    run_calibration_sweep("bottle")
