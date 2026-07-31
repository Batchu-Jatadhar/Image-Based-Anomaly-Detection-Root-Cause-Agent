def predict(image_path: str) -> dict:
    """
    Mock/Stub for vision model prediction.
    Follows the interface contract exactly.
    
    Args:
        image_path: Path to the input image file.
        
    Returns:
        A dict with the prediction results:
        {
            "label": "Crack" | "Scratch" | "Dent" | "Corrosion" | "Normal",
            "confidence": float,
            "bbox": [x, y, w, h] | None,
            "heatmap_overlay_path": str
        }
    """
    # Simple default mock behavior for test compatibility
    if "normal" in image_path.lower():
        return {
            "label": "Normal",
            "confidence": 0.98,
            "bbox": None,
            "heatmap_overlay_path": "data/heatmaps/normal_cam.png"
        }
    
    return {
        "label": "Crack",
        "confidence": 0.96,
        "bbox": [120, 80, 45, 55],
        "heatmap_overlay_path": "data/heatmaps/defect_cam.png"
    }
