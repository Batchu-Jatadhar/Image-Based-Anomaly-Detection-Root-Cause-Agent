import os
import time
from pathlib import Path
from src.config import config
from src.train import train_model

def train_all(epochs: int = 3):
    categories = [
        "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper"
    ]
    
    print(f"\n================ Starting Full Dataset Batch Training ({len(categories)} remaining categories) ================")
    start_time = time.time()
    
    results = {}
    for idx, cat in enumerate(categories, 1):
        print(f"\n[{idx}/{len(categories)}] Training Category: {cat}...")
        try:
            ckpt = train_model(category=cat, epochs=epochs)
            results[cat] = "SUCCESS"
        except Exception as e:
            print(f"[Error] Category {cat} failed: {e}")
            results[cat] = f"FAILED: {e}"
            
    total_time = round((time.time() - start_time) / 60, 2)
    print(f"\n================ All Categories Training Completed in {total_time} mins ================")
    for cat, status in results.items():
        print(f"  - {cat:15s}: {status}")
    print("==========================================================================")

if __name__ == "__main__":
    train_all(epochs=3)
