def process_vision_output(vision_output: dict) -> dict:
    """
    Processes the raw predict() output from the vision model, normalizes findings,
    and produces a concise structured output for downstream RAG retrieval.
    This agent does NOT communicate with any LLM.

    Args:
        vision_output: Dict containing raw model output.
            e.g. {
                "label": "crack",
                "confidence": 0.96,
                "bbox": [x, y, w, h],
                "heatmap_overlay_path": "/path/to/overlay.png"
            }

    Returns:
        A structured dictionary representing the perception findings:
        {
            "label": str,                  # Normalized label
            "confidence": float,           # Normalized confidence (0.0 to 1.0)
            "bbox": list | None,           # Bounding box coordinates
            "heatmap_overlay_path": str,   # Path to the heatmap image
            "summary": str                 # Concise findings summary for downstream RAG
        }
    """
    # 1. Gracefully extract raw fields with fallback values
    raw_label = vision_output.get("label", "Unknown")
    raw_confidence = vision_output.get("confidence", 0.0)
    bbox = vision_output.get("bbox", None)
    heatmap_overlay_path = vision_output.get("heatmap_overlay_path", "")

    # 2. Normalize Label (e.g. capitalize, strip whitespace)
    normalized_label = str(raw_label).strip().capitalize()
    if normalized_label in ["Normal", "Ok", "None", "No defect"]:
        normalized_label = "Normal"
    elif normalized_label in ["Crack", "Scratch", "Dent", "Corrosion", "Burn mark", "Surface defect", "Hole"]:
        pass # Expected categories

    # 3. Normalize Confidence (ensure it's a float between 0.0 and 1.0)
    try:
        normalized_confidence = float(raw_confidence)
        # If represented as percentage (e.g. 96.0 instead of 0.96), scale down
        if normalized_confidence > 1.0:
            normalized_confidence = normalized_confidence / 100.0
        # Clamp between 0.0 and 1.0
        normalized_confidence = max(0.0, min(1.0, normalized_confidence))
    except (ValueError, TypeError):
        normalized_confidence = 0.0

    # 4. Generate structured findings summary string
    if normalized_label == "Normal":
        summary = (
            f"Inspection complete. The component appears Normal with a high "
            f"confidence score of {normalized_confidence:.1%}. No visual anomalies "
            f"or structural defects were identified."
        )
    else:
        bbox_str = f"at coordinates {bbox}" if bbox else "on the component surface"
        summary = (
            f"Structural anomaly detected: A {normalized_label} defect was identified "
            f"{bbox_str} with {normalized_confidence:.1%} confidence."
        )

    return {
        "label": normalized_label,
        "confidence": normalized_confidence,
        "bbox": bbox,
        "heatmap_overlay_path": heatmap_overlay_path,
        "summary": summary
    }
