import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# ponytail: simple ResNet feature extractor for embedding vector extraction
# ceiling: ResNet50 avgpool features; upgrade: fine-tuned DINOv2 / ViT embeddings if higher retrieval precision needed.

class FeatureExtractor(nn.Module):
    """
    Extracts dense feature vector embeddings (512-dim or 2048-dim) from backbone.
    """
    def __init__(self, backbone_name: str = "resnet50"):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.out_dim = 2048
        elif backbone_name == "mobilenet_v3":
            mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            self.features = mobilenet.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 576
        else:
            # Simple conv backbone fallback
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.out_dim = 64
            
        self.eval()

    @torch.no_grad()
    def extract_from_tensor(self, x: torch.Tensor) -> np.ndarray:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        feats = self.features(x)
        if hasattr(self, 'pool'):
            feats = self.pool(feats)
        feats = torch.flatten(feats, 1)
        # Normalize L2
        feats = nn.functional.normalize(feats, p=2, dim=1)
        return feats.squeeze(0).cpu().numpy()

_extractor = None

def get_embedding(img_np: np.ndarray, backbone_name: str = "resnet50") -> list:
    """
    Extracts L2-normalized feature vector embedding from a numpy image (H, W, C) [0-255].
    Returns Python float list for FAISS/RAG compatibility.
    """
    global _extractor
    if _extractor is None or _extractor.backbone_name != backbone_name:
        _extractor = FeatureExtractor(backbone_name)
        
    # Preprocess numpy HWC image to Torch BCHW
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0
    img_np = cv2_resize_if_needed(img_np)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    
    # Normalize with standard ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    
    embedding = _extractor.extract_from_tensor(tensor)
    return embedding.tolist()

def cv2_resize_if_needed(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    if h != target_size or w != target_size:
        return cv2.resize(img, (target_size, target_size))
    return img
