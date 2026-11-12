import pytest

from cv_analyser.exceptions import LLMError
from cv_analyser.llm.factory import create_llm_client
from cv_analyser.llm.ollama import DEFAULT_MODEL, OllamaClient


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "OLLAMA_URL"):
        monkeypatch.delenv(name, raising=False)


def test_defaults_to_ollama_with_default_model():
    client = create_llm_client()

    assert isinstance(client, OllamaClient)
    assert client.model == DEFAULT_MODEL


def test_model_comes_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " Ollama ")
    monkeypatch.setenv("LLM_MODEL", "llama3.2:3b")

    assert create_llm_client().model == "llama3.2:3b"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")

    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER 'nope'"):
        create_llm_client()
