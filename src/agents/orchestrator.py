from src.inference import predict
from src.rag_engine import retrieve_all
from src.agents.perception_agent import process_vision_output
from src.agents.diagnostic_agent import diagnose
from src.agents.verifier_agent import verify_report

def run_pipeline(image_path: str, patient_meta: dict) -> dict:
    """
    Single entry point for full system execution.
    Executes the multi-agent defect detection, RAG retrieval, diagnostic reasoning,
    and report verification pipeline in the exact specified order:

    Vision predict()
           ↓
    Perception Agent
           ↓
    RAG Retrieval
           ↓
    Diagnostic Agent
           ↓
    Verifier Agent
           ↓
    Final Output

    Args:
        image_path: Path to the industrial defect image to inspect.
        patient_meta: Metadata dictionary (e.g., machine_id, operator, location, patient_id).

    Returns:
        Structured result matching the expected contract:
        {
            "vision_output": {
                "label": str,
                "confidence": float,
                "bbox": list | None,
                "heatmap_overlay_path": str
            },
            "findings": {
                "summary": str
            },
            "report": {
                "impression": str,
                "root_cause": str,
                "supporting_evidence": list[str],
                "recommended_next_steps": list[str]
            },
            "verification": {
                "confidence_score": float,
                "flagged_claims": list[str],
                "verified": bool
            }
        }
    """
    # 1. Vision Model predict()
    try:
        raw_vision_output = predict(image_path)
    except Exception as e:
        raise RuntimeError(f"Orchestrator pipeline failed at Vision predict(): {str(e)}") from e

    # 2. Perception Agent processing
    try:
        perception_output = process_vision_output(raw_vision_output)
    except Exception as e:
        raise RuntimeError(f"Orchestrator pipeline failed at Perception Agent: {str(e)}") from e

    # Extract findings summary to formulate the query for RAG
    findings_summary = perception_output.get("summary", "")
    query = f"Defect label: {perception_output.get('label', '')}. Findings: {findings_summary}"

    # 3. RAG Retrieval
    try:
        rag_context = retrieve_all(query)
    except Exception as e:
        raise RuntimeError(f"Orchestrator pipeline failed at RAG Retrieval: {str(e)}") from e

    # 4. Diagnostic Agent
    try:
        report = diagnose(findings_summary, rag_context)
    except Exception as e:
        raise RuntimeError(f"Orchestrator pipeline failed at Diagnostic Agent: {str(e)}") from e

    # 5. Verifier Agent
    try:
        verification = verify_report(findings_summary, rag_context, report)
    except Exception as e:
        raise RuntimeError(f"Orchestrator pipeline failed at Verifier Agent: {str(e)}") from e

    # 6. Final Packaged Output matching schema exactly
    # Map back original keys to ensure vision_output matches contract precisely
    vision_output_block = {
        "label": raw_vision_output.get("label", "Normal"),
        "confidence": perception_output.get("confidence", 0.0),
        "bbox": raw_vision_output.get("bbox", None),
        "heatmap_overlay_path": raw_vision_output.get("heatmap_overlay_path", "")
    }

    return {
        "vision_output": vision_output_block,
        "findings": {
            "summary": findings_summary
        },
        "report": {
            "impression": report.get("impression", ""),
            "root_cause": report.get("root_cause", ""),
            "supporting_evidence": report.get("supporting_evidence", []),
            "recommended_next_steps": report.get("recommended_next_steps", [])
        },
        "verification": {
            "confidence_score": verification.get("confidence_score", 0.0),
            "flagged_claims": verification.get("flagged_claims", []),
            "verified": verification.get("verified", False)
        }
    }
