import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

class NEUDataset(Dataset):
    """
    Dataset loader for NEU Surface Defect Dataset.
    Supports 6 defect classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.
    """
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        img_size: int = 224
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.classes = NEU_CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}

        # Default transforms if none provided
        if transform is None:
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        samples = []
        
        # Check potential split directories: e.g. NEU-DET/train/images or NEU-DET/validation/images
        split_name = "train" if self.split == "train" else "validation"
        split_dir = self.data_dir / split_name / "images"
        
        if not split_dir.exists():
            # Fallback to direct class folders under data_dir
            split_dir = self.data_dir
            
        for cls_name in self.classes:
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
                
            label_idx = self.class_to_idx[cls_name]
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in sorted(cls_dir.glob(ext)):
                    samples.append({
                        "image_path": str(img_path),
                        "label_idx": label_idx,
                        "class_name": cls_name
                    })

        if not samples:
            raise FileNotFoundError(f"No image samples found for split '{self.split}' in {split_dir}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img_path = sample["image_path"]
        label_idx = sample["label_idx"]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        return img_tensor, label_idx


def visualize_neu_samples(dataset: NEUDataset, num_samples: int = 6, output_path: Optional[str] = None) -> np.ndarray:
    """
    Visualization utility for NEU dataset samples.
    """
    samples_to_show = dataset.samples[:num_samples]
    grid_images = []

    for s in samples_to_show:
        img_path = s["image_path"]
        class_name = s["class_name"]

        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_resized = cv2.resize(img, (200, 200))
        cv2.putText(
            img_resized,
            class_name,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        grid_images.append(img_resized)

    if grid_images:
        grid = np.hstack(grid_images)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, grid)
        return grid
    return np.array([])
