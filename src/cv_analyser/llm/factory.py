import os

from cv_analyser.exceptions import LLMError
from cv_analyser.llm.base import LLMClient
from cv_analyser.llm.ollama import DEFAULT_MODEL, DEFAULT_URL, OllamaClient


def create_llm_client() -> LLMClient:
    """Build the client chosen by environment variables.

    LLM_PROVIDER  which backend to use (default: ollama)
    LLM_MODEL     model name for that backend (default: qwen2.5:3b)
    OLLAMA_URL    Ollama server address (default: http://localhost:11434)
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return OllamaClient(
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("OLLAMA_URL", DEFAULT_URL),
        )
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Supported: ollama")
