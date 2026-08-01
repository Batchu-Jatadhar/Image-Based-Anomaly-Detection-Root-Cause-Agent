import os
import json
import requests
from dotenv import load_dotenv

# Load .env file from the current working directory
load_dotenv()

class CentralizedLLM:
    """
    Centralized LLM interface wrapper.
    Every agent must retrieve their LLM through this module.
    No provider names, API keys, or environment variable logic should live inside agents.
    """
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "mock").lower()
        self.model = os.getenv("LLM_MODEL", "mock-model")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        
        # Resolve keys and endpoints based on provider
        if self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        elif self.provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            self.base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        elif self.provider == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        elif self.provider == "ollama":
            self.api_key = "ollama"  # Ollama usually does not require a key
            self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        elif self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
            self.base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        else:
            self.api_key = os.getenv("LLM_API_KEY")
            self.base_url = os.getenv("LLM_BASE_URL")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate content using the configured LLM provider.
        Supports single prompt text generation.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)

    def chat(self, messages: list) -> str:
        """
        Perform a chat completion call. Fallback to mock response if provider is 'mock'
        or if API keys are missing.
        """
        # Check if we should fall back to mock
        if self.provider == "mock" or (self.provider != "ollama" and not self.api_key):
            return self._mock_generate(messages)

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/Batchu-Jatadhar/Image-Based-Anomaly-Detection-Root-Cause-Agent"
            headers["X-Title"] = "Anomaly Detection Agent"

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            # Log error and fall back to mock if desired, or raise
            raise RuntimeError(f"Centralized LLM call failed for provider '{self.provider}': {str(e)}") from e

    def _mock_generate(self, messages: list) -> str:
        """
        Generates simulated high-quality diagnostic and verifier responses.
        Allows the system to run and test immediately without configuring actual API keys.
        """
        last_message = messages[-1]["content"] if messages else ""
        
        # Check if this is verification generation first (to avoid keyword collisions)
        if "audit" in last_message.lower() or "verify" in last_message.lower() or "deepeval" in last_message.lower():
            mock_verification = {
                "confidence_score": 0.95,
                "flagged_claims": [],
                "verified": True
            }
            return json.dumps(mock_verification)

        # Check if this is diagnostic generation
        elif "diagnostic" in last_message.lower() or "impression" in last_message.lower():
            # Diagnostic Agent prompt
            mock_report = {
                "impression": "High-severity structural anomaly detected on the component surface.",
                "root_cause": "Material fatigue from prolonged cyclic loading and micro-fracturing.",
                "supporting_evidence": [
                    "Visual presence of a 3cm surface crack with high confidence.",
                    "Guideline 402 states that micro-cracks propagate quickly under cyclic stress."
                ],
                "recommended_next_steps": [
                    "De-energize the assembly and replace the component within 48 hours.",
                    "Perform alignment check on adjacent drive shafts."
                ]
            }
            return json.dumps(mock_report)

        # Generic mock fallback
        return "Simulated LLM response for testing purposes."

def get_llm() -> CentralizedLLM:
    """Factory function to get the centralized LLM client."""
    return CentralizedLLM()
