import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SEVERSTAL_CLASSES = ["Class 1", "Class 2", "Class 3", "Class 4"]

def rle_to_mask(rle: str, height: int = 256, width: int = 1600) -> np.ndarray:
    """
    Decodes Run-Length Encoded (RLE) string into a binary mask of shape (height, width).
    Severstal RLE is 1-indexed and column-major (Fortran 'F' order).
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if not rle or not isinstance(rle, str) or rle.strip() == "" or rle.strip() == "nan":
        return mask.reshape((height, width), order="F")

    s = list(map(int, rle.strip().split()))
    starts = s[0::2]
    lengths = s[1::2]

    for start, length in zip(starts, lengths):
        start_idx = start - 1
        mask[start_idx : start_idx + length] = 1

    return mask.reshape((height, width), order="F")


def mask_to_rle(mask: np.ndarray) -> str:
    """
    Encodes a binary mask of shape (height, width) into a Run-Length Encoded (RLE) string.
    Column-major (Fortran 'F' order) 1-indexed string.
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


class SeverstalDataset(Dataset):
    """
    Dataset loader for Severstal Steel Defect Detection (Semantic Segmentation).
    Parses train.csv, decodes RLE masks for 4 defect classes, and supports train/val splits.
    """
    def __init__(
        self,
        data_dir: Path = Path("severstal dataset"),
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        target_size: Tuple[int, int] = (256, 512),
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_size = target_size  # (height, width) e.g., (256, 512)
        
        self.images_dir = self.data_dir / ("train_images" if split != "test" else "test_images")
        self.csv_path = self.data_dir / "train.csv"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.samples = self._load_samples(val_ratio=val_ratio, seed=seed)
        self.transform = transform

    def _load_samples(self, val_ratio: float, seed: int) -> List[Dict]:
        if not self.csv_path.exists() and self.split != "test":
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            # Group by ImageId
            grouped = df.groupby("ImageId")
            image_records = {}

            for image_id, group in grouped:
                rles = {}
                classes_present = []
                for _, row in group.iterrows():
                    cls_id = int(row["ClassId"])
                    rle = str(row["EncodedPixels"]) if pd.notna(row["EncodedPixels"]) else ""
                    if rle:
                        rles[cls_id] = rle
                        classes_present.append(cls_id)

                image_records[image_id] = {
                    "image_id": image_id,
                    "image_path": str(self.images_dir / image_id),
                    "rles": rles,
                    "classes_present": classes_present
                }

            all_image_ids = sorted(list(image_records.keys()))
        else:
            all_image_files = sorted(list(self.images_dir.glob("*.jpg")))
            all_image_ids = [p.name for p in all_image_files]
            image_records = {
                p.name: {
                    "image_id": p.name,
                    "image_path": str(p),
                    "rles": {},
                    "classes_present": []
                }
                for p in all_image_files
            }

        # Deterministic Train / Validation Split
        np.random.seed(seed)
        shuffled_ids = np.array(all_image_ids)
        np.random.shuffle(shuffled_ids)

        val_size = int(len(shuffled_ids) * val_ratio)
        if self.split == "train":
            split_ids = shuffled_ids[val_size:]
        elif self.split == "val" or self.split == "validation":
            split_ids = shuffled_ids[:val_size]
        else:
            split_ids = shuffled_ids

        samples = [image_records[img_id] for img_id in split_ids if os.path.exists(image_records[img_id]["image_path"])]
        return samples

    def get_full_mask(self, rles: Dict[int, str]) -> np.ndarray:
        """
        Builds a 4-channel binary mask tensor (4, 256, 1600).
        Channel 0: Class 1, Channel 1: Class 2, Channel 2: Class 3, Channel 3: Class 4.
        """
        full_mask = np.zeros((4, 256, 1600), dtype=np.float32)
        for cls_id in range(1, 5):
            if cls_id in rles:
                binary_mask = rle_to_mask(rles[cls_id], height=256, width=1600)
                full_mask[cls_id - 1] = binary_mask.astype(np.float32)
        return full_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_path = sample["image_path"]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Get 4-channel mask (4, 256, 1600)
        mask_np = self.get_full_mask(sample["rles"])

        # Resize image & mask to target_size
        h_target, w_target = self.target_size
        img_resized = cv2.resize(img_rgb, (w_target, h_target))
        
        resized_masks = []
        for c in range(4):
            m_resized = cv2.resize(mask_np[c], (w_target, h_target), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(m_resized)
        mask_resized = np.stack(resized_masks, axis=0)

        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        mask_tensor = torch.from_numpy(mask_resized).float()

        return img_tensor, mask_tensor


def visualize_severstal_samples(dataset: SeverstalDataset, num_samples: int = 3, output_path: Optional[str] = None) -> np.ndarray:
    """
    Generates preview visualization grids comparing Original Image, Combined Mask, and Color Overlay.
    """
    color_map = {
        1: [255, 0, 0],    # Class 1: Red
        2: [0, 255, 0],    # Class 2: Green
        3: [0, 0, 255],    # Class 3: Blue
        4: [255, 255, 0]   # Class 4: Yellow
    }

    previews = []
    selected_samples = [s for s in dataset.samples if len(s["classes_present"]) > 0][:num_samples]

    for s in selected_samples:
        img_path = s["image_path"]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        h, w, _ = img_bgr.shape
        overlay = img_bgr.copy()
        combined_mask_viz = np.zeros_like(img_bgr)

        for cls_id in range(1, 5):
            if cls_id in s["rles"]:
                m = rle_to_mask(s["rles"][cls_id], height=h, width=w)
                color = color_map[cls_id]
                combined_mask_viz[m == 1] = color

        overlay = cv2.addWeighted(overlay, 0.7, combined_mask_viz, 0.3, 0)

        classes_str = ", ".join([f"Class {c}" for c in s["classes_present"]])
        cv2.putText(overlay, f"Defects: {classes_str}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Scale down for visualization stack
        h_vis, w_vis = 160, 500
        img_vis = cv2.resize(img_bgr, (w_vis, h_vis))
        mask_vis = cv2.resize(combined_mask_viz, (w_vis, h_vis))
        overlay_vis = cv2.resize(overlay, (w_vis, h_vis))

        sample_stack = np.vstack([img_vis, mask_vis, overlay_vis])
        previews.append(sample_stack)

    if previews:
        final_grid = np.hstack(previews)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, final_grid)
        return final_grid
    return np.array([])
