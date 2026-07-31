import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SettingsError(Exception):
    pass

def get_required_env(var_name: str, default: str = None) -> str:
    val = os.getenv(var_name, default)
    if val is None or val.strip() == "":
        raise SettingsError(f"Missing required environment variable: {var_name}")
    return val

# Core Configuration
LLM_PROVIDER = get_required_env("LLM_PROVIDER", "ollama")
EMBEDDING_MODEL = get_required_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# RAG Configuration
RAG_CHUNK_SIZE = int(get_required_env("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(get_required_env("RAG_CHUNK_OVERLAP", "50"))

# Generation Configuration
TEMPERATURE = float(get_required_env("TEMPERATURE", "0.2"))
MAX_TOKENS = int(get_required_env("MAX_TOKENS", "2048"))

# Provider specific configurations
# We only require the keys/urls if the specific provider is selected
if LLM_PROVIDER.lower() == "ollama":
    OLLAMA_BASE_URL = get_required_env("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = get_required_env("OLLAMA_MODEL", "llama3.1:8b")
elif LLM_PROVIDER.lower() == "openai":
    OPENAI_API_KEY = get_required_env("OPENAI_API_KEY")
    OPENAI_MODEL = get_required_env("OPENAI_MODEL")
elif LLM_PROVIDER.lower() == "groq":
    GROQ_API_KEY = get_required_env("GROQ_API_KEY")
    GROQ_MODEL = get_required_env("GROQ_MODEL")
elif LLM_PROVIDER.lower() == "openrouter":
    OPENROUTER_API_KEY = get_required_env("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = get_required_env("OPENROUTER_MODEL")
else:
    raise SettingsError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

# Optional: Add external API keys here as needed (e.g., Semantic Scholar API key if we decide to use one later)
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
