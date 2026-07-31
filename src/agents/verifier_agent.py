import json
import re
from src.llm_config import get_llm

# Check if deepeval is available
try:
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase
    from deepeval.models.base_model import DeepEvalBaseLLM
    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False

def verify_report(findings_summary: str, rag_context: str, report: dict) -> dict:
    """
    Verifier Agent.
    Audits the generated diagnostic report against the perception findings
    and retrieved RAG context using DeepEval (if available) or an LLM-based auditor.

    Args:
        findings_summary: Summary of inspection findings from Perception Agent.
        rag_context: Retrieved guidelines/manual documents from RAG search.
        report: Diagnostic report dict from Diagnostic Agent.

    Returns:
        A dictionary with verification details:
        {
            "confidence_score": float,  # Value in range 0.0 - 1.0 (or 0 - 100)
            "flagged_claims": list[str], # List of claims unsupported or contradictory
            "verified": bool            # True if audit passes
        }
    """
    # Initialize the centralized LLM client
    llm = get_llm()

    # Format the report content for evaluation
    report_text = (
        f"Impression: {report.get('impression', '')}\n"
        f"Likely Root Cause: {report.get('root_cause', '')}\n"
        f"Supporting Evidence: {', '.join(report.get('supporting_evidence', []))}\n"
        f"Recommended Next Steps: {', '.join(report.get('recommended_next_steps', []))}"
    )

    # 1. Attempt audit via DeepEval if installed
    if HAS_DEEPEVAL:
        try:
            # Wrap the centralized LLM to integrate with DeepEval metrics
            class DeepEvalLLMWrapper(DeepEvalBaseLLM):
                def __init__(self, model_client):
                    self.model_client = model_client
                    
                def load_model(self):
                    return self.model_client
                    
                def generate(self, prompt: str) -> str:
                    return self.model_client.generate(prompt)
                    
                async def a_generate(self, prompt: str) -> str:
                    return self.model_client.generate(prompt)
                    
                def get_model_name(self) -> str:
                    return getattr(self.model_client, "model", "centralized-llm")

            wrapper_llm = DeepEvalLLMWrapper(llm)
            # Use HallucinationMetric to cross-examine report against context
            metric = HallucinationMetric(threshold=0.5, model=wrapper_llm)
            test_case = LLMTestCase(
                input=findings_summary,
                actual_output=report_text,
                context=[rag_context]
            )
            metric.measure(test_case)

            # DeepEval score ranges from 0 (faithful/no hallucination) to 1 (full hallucination)
            # So verification confidence score = 1.0 - metric.score
            confidence_score = float(1.0 - metric.score)
            
            # Use reason/details to isolate any flagged claims
            flagged_claims = []
            if metric.score > 0.0 and hasattr(metric, "reason") and metric.reason:
                flagged_claims.append(str(metric.reason))
            
            return {
                "confidence_score": confidence_score,
                "flagged_claims": flagged_claims,
                "verified": bool(confidence_score >= 0.5)
            }

        except Exception:
            # If DeepEval execution fails, gracefully log and fall through to LLM fallback
            pass

    # 2. LLM-based Verification Fallback
    system_prompt = (
        "You are an expert quality assurance and auditing agent specializing in manufacturing "
        "and industrial safety inspection compliance."
    )

    audit_prompt = f"""
You are auditing a generated Diagnostic Report against the raw Inspection Findings and RAG retrieved context.
Your goal is to detect any ungrounded claims, hallucinations, contradictions, or unsupported safety recommendations.

---
INSPECTION FINDINGS:
{findings_summary}

RAG RETRIEVED CONTEXT:
{rag_context}

GENERATED DIAGNOSTIC REPORT:
{report_text}
---

Your response MUST be a valid JSON object matching the following structure:
{{
    "confidence_score": 0.95, // float between 0.0 and 1.0 indicating report accuracy and grounding
    "flagged_claims": [
        // List of strings representing specific sentences or claims in the report that are unsupported or contradicted.
        // Keep this list empty [] if the report is fully accurate and supported.
    ],
    "verified": true // boolean, set to true if confidence_score >= 0.8 and no critical claims are flagged
}}

Ensure the JSON is properly formatted, containing no markdown backticks, no trailing commas, and no explanations before or after the JSON block.
"""

    try:
        response_text = llm.generate(prompt=audit_prompt, system_prompt=system_prompt)
        audit_data = _extract_json_from_response(response_text)
    except Exception as e:
        # Graceful absolute fallback if everything fails
        return {
            "confidence_score": 0.5,
            "flagged_claims": [f"Verification process warning: LLM evaluation failed. Details: {str(e)}"],
            "verified": False
        }

    # Normalize fields
    conf_score = audit_data.get("confidence_score", 0.0)
    try:
        conf_score = float(conf_score)
        # Convert 0-100 scale to 0.0-1.0 if returned that way
        if conf_score > 1.0:
            conf_score = conf_score / 100.0
        conf_score = max(0.0, min(1.0, conf_score))
    except (ValueError, TypeError):
        conf_score = 0.5

    flagged = audit_data.get("flagged_claims", [])
    if not isinstance(flagged, list):
        flagged = [str(flagged)]

    verified = bool(audit_data.get("verified", conf_score >= 0.8))

    return {
        "confidence_score": conf_score,
        "flagged_claims": flagged,
        "verified": verified
    }

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
