import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict

class TemperatureScaler(nn.Module):
    """
    Temperature Scaling Calibration for Neural Network Classifier Confidence Scores.
    Optimizes a single temperature parameter T > 0 on validation logits:
        probs = softmax(logits / T)
    """
    def __init__(self, initial_temp: float = 1.2):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([initial_temp], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50) -> float:
        """
        Optimizes temperature T using L-BFGS optimizer on validation CrossEntropy loss.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_fn():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        with torch.no_grad():
            self.temperature.clamp_(min=0.05)

        return float(self.temperature.item())


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    Divides confidence scores into n_bins equal-width bins and calculates
    sum(|accuracy(bin) - confidence(bin)| * bin_weight).
    """
    if len(probs.shape) > 1:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        confidences = probs
        predictions = (probs > 0.5).astype(int)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return round(float(ece), 4)


def calibrate_model_logits(val_logits: torch.Tensor, val_labels: torch.Tensor) -> Dict:
    """
    Calibrates validation logits using Temperature Scaling and evaluates Expected Calibration Error (ECE).
    """
    # Raw probabilities & ECE before calibration
    raw_probs = torch.softmax(val_logits, dim=1).numpy()
    labels_np = val_labels.numpy()
    ece_before = compute_ece(raw_probs, labels_np)

    # Fit Temperature Scaler
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(val_logits, val_labels)

    # Calibrated probabilities & ECE after calibration
    calibrated_logits = scaler(val_logits)
    calibrated_probs = torch.softmax(calibrated_logits, dim=1).detach().numpy()
    ece_after = compute_ece(calibrated_probs, labels_np)

    return {
        "optimal_temperature": round(optimal_temp, 4),
        "ece_before_calibration": ece_before,
        "ece_after_calibration": ece_after,
        "calibration_improvement": round(ece_before - ece_after, 4)
    }
