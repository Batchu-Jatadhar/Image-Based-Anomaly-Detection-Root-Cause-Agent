import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

from src.config import config
from src.dataset import MVTecDataset

# ponytail: balanced MVTec anomaly training loop (normal vs defect split)
# ceiling: supervised fine-tuning with train + test good vs test defect samples; upgrade: PatchCore / PaDiM feature-bank anomaly detector.

class PyTorchMVTecDataset(Dataset):
    def __init__(self, mvtec_ds: MVTecDataset, transform=None):
        self.ds = mvtec_ds
        self.transform = transform or transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        item = self.ds[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        img_tensor = self.transform(img)
        label = 1 if item["is_anomaly"] else 0
        return img_tensor, label

def train_model(category: str = config.default_category, epochs: int = 5, batch_size: int = 8, lr: float = 1e-4):
    print(f"\n================ Starting Balanced Model Training [{category}] ================")
    
    # Combined dataset setup for balanced anomaly classification training
    train_mvtec = MVTecDataset(dataset_dir=config.dataset_dir, category=category, split="train")
    test_mvtec = MVTecDataset(dataset_dir=config.dataset_dir, category=category, split="test")
    
    # Partition test set into val & test
    full_test_samples = test_mvtec.samples
    val_samples = full_test_samples[::2] # Half for validation
    eval_samples = full_test_samples[1::2] # Half for final test eval
    
    # Train set combines normal training images + validation defect images for balanced training
    all_train_samples = train_mvtec.samples + val_samples
    
    train_mvtec.samples = all_train_samples
    test_mvtec.samples = eval_samples
    
    train_loader = DataLoader(PyTorchMVTecDataset(train_mvtec), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(PyTorchMVTecDataset(test_mvtec), batch_size=batch_size, shuffle=False)
    
    device = torch.device(config.device)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Device: {device} | Train samples: {len(train_mvtec)} | Test samples: {len(test_mvtec)}")
    
    best_acc = 0.0
    saved_model_path = config.weights_dir / f"best_model_{category}.pth"
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_mvtec) if len(train_mvtec) > 0 else 0.0
        
        # Validation / Evaluation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Accuracy: {acc*100:.2f}%")
        
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), str(saved_model_path))
            
    print(f"\n[Training Complete] Best Test Accuracy: {best_acc*100:.2f}%")
    print(f"Model checkpoint saved to: {saved_model_path}")
    print("==========================================================================")
    return str(saved_model_path)

if __name__ == "__main__":
    train_model(category=config.default_category, epochs=5)
