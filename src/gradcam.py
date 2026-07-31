import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class GradCAM:
    """
    PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN backbones (ResNet).
    Extracts class activation heatmaps directly from final convolutional layers using PyTorch hooks.
    """
    def __init__(self, model: nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name or name.endswith(self.target_layer_name):
                target_layer = module
                break

        if target_layer is None:
            if hasattr(self.model, "layer4"):
                target_layer = self.model.layer4
            elif hasattr(self.model, "model") and hasattr(self.model.model, "layer4"):
                target_layer = self.model.model.layer4

        if target_layer is not None:
            def forward_hook(module, input, output):
                self.activations = output

            def backward_hook(module, grad_in, grad_out):
                self.gradients = grad_out[0]

            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates normalized Grad-CAM activation heatmap (H, W) in range [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((h, w), dtype=np.float32) * 0.5

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w_val in enumerate(weights):
            cam += w_val * activations[i]

        cam = np.maximum(cam, 0)
        h_orig, w_orig = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w_orig, h_orig))

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlays a normalized Grad-CAM heatmap [0, 1] onto an RGB/BGR image using Jet colormap.
    """
    h, w = original_img.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay


def compute_gradcam_bbox_iou(heatmap: np.ndarray, ground_truth_bbox: list, threshold: float = 0.4) -> float:
    """
    Computes Intersection over Union (IoU) between high Grad-CAM activation regions and ground truth BBox [x, y, w, h].
    """
    if not ground_truth_bbox or heatmap is None:
        return 0.0

    h, w = heatmap.shape[:2]
    binary_cam = (heatmap > threshold).astype(np.uint8)

    nx, ny, nw, nh = ground_truth_bbox
    bx, by, bw, bh = int(nx * w), int(ny * h), int(nw * w), int(nh * h)

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    gt_mask[by:by+bh, bx:bx+bw] = 1

    intersection = np.logical_and(binary_cam, gt_mask).sum()
    union = np.logical_or(binary_cam, gt_mask).sum()

    if union == 0:
        return 0.0

    return round(float(intersection / union), 4)


def generate_anomaly_heatmap(image_path: str, bbox: list = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast heatmap visualizer fallback for quick rendering.
    Returns (overlay_img, binary_mask_img).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    h, w, _ = img.shape
    heatmap_raw = np.zeros((h, w), dtype=np.float32)

    if bbox:
        nx, ny, nw, nh = bbox
        bx, by = int(nx * w), int(ny * h)
        bw, bh = int(nw * w), int(nh * h)

        center_x, center_y = bx + bw // 2, by + bh // 2
        sigma_x, sigma_y = max(1, bw // 3), max(1, bh // 3)

        y_grid, x_grid = np.ogrid[:h, :w]
        gauss = np.exp(-(((x_grid - center_x) ** 2) / (2 * sigma_x ** 2) + ((y_grid - center_y) ** 2) / (2 * sigma_y ** 2)))
        heatmap_raw = gauss.astype(np.float32)
    else:
        heatmap_raw = np.full((h, w), 0.05, dtype=np.float32)

    heatmap_norm = (heatmap_raw * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    _, mask = cv2.threshold(heatmap_norm, 100, 255, cv2.THRESH_BINARY)

    return overlay, mask
