import json
import re
from src.llm_config import get_llm

def diagnose(findings_summary: str, rag_context: str) -> dict:
    """
    Diagnostic Agent.
    Receives the findings summary from the Perception Agent and RAG context.
    Generates a structured report detailing the industrial/clinical impression,
    probable root cause, supporting evidence, and next steps/risk assessment.

    Args:
        findings_summary: Summary text describing the visual inspection findings.
        rag_context: Background documentation retrieved from RAG system.

    Returns:
        A dict matching the expected report schema:
        {
            "impression": str,
            "root_cause": str,
            "supporting_evidence": list[str],
            "recommended_next_steps": list[str]
        }
    """
    # 1. Retrieve the configured LLM through the centralized configuration helper
    llm = get_llm()

    # 2. Build the system and user prompts to guide the LLM's response
    system_prompt = (
        "You are an expert industrial diagnostics engineer specializing in manufacturing "
        "inspection, predictive maintenance, and defect root-cause analysis."
    )

    prompt = f"""
Analyze the following inspection findings and historical RAG context to determine the likely root cause, overall impression, supporting evidence, recommendations, and risk assessment.

---
INSPECTION FINDINGS SUMMARY:
{findings_summary}

RETRIEVED RAG CONTEXT:
{rag_context}
---

Your response MUST be a valid JSON object matching the following structure:
{{
    "impression": "A professional summary of the component health, defect severity, operational risk, and expected impact.",
    "root_cause": "The physical, mechanical, or chemical root cause of the defect based on findings and guidelines.",
    "supporting_evidence": [
        "First specific point of evidence (e.g. citing localized coordinates or context findings)",
        "Second specific point of evidence..."
    ],
    "recommended_next_steps": [
        "Immediate safety/containment action",
        "Short-term corrective action (e.g. replacement timeframe, component calibration)",
        "Long-term preventive maintenance protocol"
    ]
}}

Ensure the JSON is properly formatted, containing no markdown backticks, no trailing commas, and no explanations before or after the JSON block.
"""

    # 3. Call LLM
    try:
        response_text = llm.generate(prompt=prompt, system_prompt=system_prompt)
    except Exception as e:
        raise RuntimeError(f"Diagnostic Agent failed to generate response from LLM: {str(e)}") from e

    # 4. Parse the response using robust JSON extraction logic
    try:
        report_data = _extract_json_from_response(response_text)
    except Exception as e:
        raise ValueError(
            f"Diagnostic Agent failed to extract structured JSON from LLM response.\n"
            f"Error: {str(e)}\nResponse Content: {response_text}"
        ) from e

    # 5. Validate and normalize the report schema
    required_keys = ["impression", "root_cause", "supporting_evidence", "recommended_next_steps"]
    for key in required_keys:
        if key not in report_data:
            # Inject a fallback message instead of crashing
            if key in ["supporting_evidence", "recommended_next_steps"]:
                report_data[key] = ["Information unavailable in generated report."]
            else:
                report_data[key] = "Information unavailable in generated report."

    # Force correct type constraints
    if not isinstance(report_data["supporting_evidence"], list):
        report_data["supporting_evidence"] = [str(report_data["supporting_evidence"])]
    if not isinstance(report_data["recommended_next_steps"], list):
        report_data["recommended_next_steps"] = [str(report_data["recommended_next_steps"])]

    return report_data

def _extract_json_from_response(text: str) -> dict:
    """Helper function to clean and parse JSON from the LLM output."""
    cleaned = text.strip()
    
    # Strip markdown code block wrapping if present
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
        
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: find first '{' and last '}'
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end+1])
            except json.JSONDecodeError as err:
                raise ValueError(f"Extracted block was not valid JSON: {cleaned[start:end+1]}") from err
        raise ValueError("No JSON object could be isolated from response.")
