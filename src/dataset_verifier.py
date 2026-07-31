import os
import cv2
import numpy as np
from pathlib import Path
from src.config import config
from src.dataset import MVTecDataset

def draw_bbox_and_mask_overlay(image_path: str, mask_path: str, bbox: list, label: str) -> np.ndarray:
    """
    Overlays binary mask (in red/cyan tint) and draws bounding box (green) on the original image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    h, w, _ = img.shape
    overlay = img.copy()
    
    # Overlay ground truth mask if present
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask_resized = cv2.resize(mask, (w, h))
            colored_mask = np.zeros_like(img)
            colored_mask[mask_resized > 127] = [0, 0, 255] # Red overlay for mask
            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
            
    # Draw normalized bounding box [x, y, box_w, box_h]
    if bbox:
        nx, ny, nw, nh = bbox
        bx = int(nx * w)
        by = int(ny * h)
        bw = int(nw * w)
        bh = int(nh * h)
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(overlay, f"Defect: {label}", (bx, max(20, by - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
    return overlay

def verify_dataset(category: str = "bottle", num_samples: int = 5):
    """
    Inspects MVTec AD dataset, prints category statistics, and saves visualization verification grids.
    """
    print(f"\n================ MVTec AD Dataset Verification: [{category}] ================")
    
    train_ds = MVTecDataset(dataset_dir=config.dataset_dir, category=category, split="train")
    test_ds = MVTecDataset(dataset_dir=config.dataset_dir, category=category, split="test")
    
    print(f"-> Train Split Size: {len(train_ds)} normal samples")
    print(f"-> Test Split Size : {len(test_ds)} total samples")
    
    # Analyze defect class distribution
    defect_counts = {}
    for sample in test_ds.samples:
        lbl = sample["label"]
        defect_counts[lbl] = defect_counts.get(lbl, 0) + 1
        
    print(f"-> Test Defect Classes & Counts:")
    for lbl, count in defect_counts.items():
        print(f"    - {lbl:15s}: {count} samples")
        
    # Generate visualization verification samples
    print(f"\nGenerating visual verification grid for up to {num_samples} defective samples...")
    verification_images = []
    
    defective_indices = [i for i, s in enumerate(test_ds.samples) if s["is_anomaly"]]
    selected_indices = defective_indices[:num_samples]
    
    for idx in selected_indices:
        item = test_ds[idx]
        img_path = item["image_path"]
        mask_path = item["mask_path"]
        bbox = item["bbox"]
        label = item["label"]
        
        vis_img = draw_bbox_and_mask_overlay(img_path, mask_path, bbox, label)
        # Resize visual output image for standardized preview grid
        vis_img_resized = cv2.resize(vis_img, (300, 300))
        verification_images.append(vis_img_resized)
        
        print(f"  [Sample] Label: {label:12s} | BBox [x,y,w,h]: {bbox}")
        
    if verification_images:
        grid = np.hstack(verification_images)
        output_path = config.heatmaps_dir / f"dataset_verification_{category}.png"
        cv2.imwrite(str(output_path), grid)
        print(f"\n[Success] Visual verification image saved to: {output_path}")
        print("==========================================================================")
        return str(output_path)
    else:
        print("\n[Warning] No defective samples found for visual grid.")
        return None

if __name__ == "__main__":
    verify_dataset(category=config.default_category)
