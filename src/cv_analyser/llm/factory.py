import os

from cv_analyser.exceptions import LLMError
from cv_analyser.llm.base import LLMClient
from cv_analyser.llm.ollama import DEFAULT_MODEL, DEFAULT_URL, OllamaClient
from cv_analyser.llm.openai_compatible import PROVIDERS, OpenAICompatibleClient

LOCAL_PROVIDERS = {"ollama"}
# Requirements judged per LLM call. Small local models lose track of long
# batches, so they get 5 at a time; large cloud models handle a whole job ad
# (extraction returns at most 15) in one call, which sends the instructions once
# instead of three times.
LOCAL_JUDGE_BATCH = 5
CLOUD_JUDGE_BATCH = 15


def provider_name() -> str:
    return os.environ.get("LLM_PROVIDER", "ollama").strip().lower()


def is_local_provider() -> bool:
    """True when the chosen model runs on this machine, so no data leaves it."""
    return provider_name() in LOCAL_PROVIDERS


def judge_batch_size() -> int:
    return LOCAL_JUDGE_BATCH if is_local_provider() else CLOUD_JUDGE_BATCH


def create_llm_client() -> LLMClient:
    """Build the client chosen by environment variables.

    LLM_PROVIDER  ollama (default, local), or a cloud provider: groq, openrouter, gemini
    LLM_MODEL     model name for that provider (default: the provider's default)
    LLM_REASONING_EFFORT  low, medium or high, for cloud models that "think"
                  (default: low for gpt-oss models, otherwise not sent)
    OLLAMA_URL    Ollama server address (default: http://localhost:11434)
    GROQ_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY   keys for the cloud providers
    """
    provider = provider_name()
    model = os.environ.get("LLM_MODEL", "").strip() or None
    if provider == "ollama":
        return create_local_client(model)
    if provider in PROVIDERS:
        effort = os.environ.get("LLM_REASONING_EFFORT", "").strip().lower() or None
        return OpenAICompatibleClient(provider, model=model, reasoning_effort=effort)
    supported = ", ".join(["ollama", *PROVIDERS])
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Supported: {supported}")


def create_local_client(model: str | None = None) -> OllamaClient:
    return OllamaClient(
        model=model or DEFAULT_MODEL,
        base_url=os.environ.get("OLLAMA_URL", DEFAULT_URL),
    )
