import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image

class MVTecDataset:
    """
    Dataset loader for MVTec AD anomaly detection datasets.
    Handles train (good only), test (good + defect classes), and ground_truth masks.
    Automatically calculates normalized bounding boxes [x, y, w, h] from segmentation masks.
    """
    def __init__(self, dataset_dir: Path, category: str = "bottle", split: str = "test", img_size: int = 224):
        self.dataset_dir = Path(dataset_dir)
        self.category = category
        self.split = split
        self.img_size = img_size
        self.category_dir = self.dataset_dir / category
        
        if not self.category_dir.exists():
            raise FileNotFoundError(f"Category directory not found: {self.category_dir}")
            
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        samples = []
        split_dir = self.category_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
            
        if self.split == "train":
            # Training split only has 'good' samples
            good_dir = split_dir / "good"
            if good_dir.exists():
                for img_path in sorted(good_dir.glob("*.png")):
                    samples.append({
                        "image_path": str(img_path),
                        "category": self.category,
                        "label": "good",
                        "is_anomaly": False,
                        "mask_path": None,
                    })
        else:
            # Test split has 'good' and defect subcategories
            for sub_dir in sorted(split_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue
                defect_label = sub_dir.name
                is_anomaly = (defect_label != "good")
                
                gt_dir = self.category_dir / "ground_truth" / defect_label if is_anomaly else None
                
                for img_path in sorted(sub_dir.glob("*.png")):
                    mask_path = None
                    if is_anomaly and gt_dir and gt_dir.exists():
                        expected_mask = gt_dir / f"{img_path.stem}_mask.png"
                        if expected_mask.exists():
                            mask_path = str(expected_mask)
                            
                    samples.append({
                        "image_path": str(img_path),
                        "category": self.category,
                        "label": defect_label,
                        "is_anomaly": is_anomaly,
                        "mask_path": mask_path,
                    })
        return samples

    def get_bbox_from_mask(self, mask_path: Optional[str]) -> Optional[List[float]]:
        """
        Derives normalized bounding box [x, y, w, h] from a binary segmentation mask.
        Returns coordinates relative to image width and height in [0, 1].
        """
        if not mask_path or not os.path.exists(mask_path):
            return None
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
            
        h, w = mask.shape
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Select largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, box_w, box_h = cv2.boundingRect(largest_contour)
        
        # Normalize coordinates relative to image size [0, 1]
        norm_x = round(x / w, 4)
        norm_y = round(y / h, 4)
        norm_w = round(box_w / w, 4)
        norm_h = round(box_h / h, 4)
        
        return [norm_x, norm_y, norm_w, norm_h]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        
        # Calculate bounding box from mask if available
        bbox = self.get_bbox_from_mask(mask_path)
        
        return {
            "image_path": image_path,
            "category": sample["category"],
            "label": sample["label"],
            "is_anomaly": sample["is_anomaly"],
            "mask_path": mask_path,
            "bbox": bbox
        }
