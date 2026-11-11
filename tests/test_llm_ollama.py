import io
import json
import urllib.error

import pytest
from pydantic import BaseModel

from cv_analyser.exceptions import LLMError
from cv_analyser.llm import ollama
from cv_analyser.llm.ollama import OllamaClient


class _Answer(BaseModel):
    skill: str
    years: int


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


@pytest.fixture
def capture(monkeypatch):
    """Replace urlopen; return a dict that records the request and sets the reply."""
    state = {"reply": {"message": {"content": '{"skill": "Python", "years": 3}'}}}

    def fake_urlopen(request, timeout):
        state["url"] = request.full_url
        state["payload"] = json.loads(request.data)
        state["timeout"] = timeout
        if isinstance(state["reply"], Exception):
            raise state["reply"]
        return _FakeResponse(json.dumps(state["reply"]).encode())

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake_urlopen)
    return state


def test_returns_validated_schema_instance(capture):
    answer = OllamaClient().generate_structured("sys", "user", _Answer)

    assert answer == _Answer(skill="Python", years=3)


def test_sends_schema_messages_and_options(capture):
    client = OllamaClient(model="tiny", base_url="http://host:1234/", context_tokens=4096)
    client.generate_structured("be strict", "the question", _Answer)

    payload = capture["payload"]
    assert capture["url"] == "http://host:1234/api/chat"
    assert payload["model"] == "tiny"
    assert payload["stream"] is False
    assert payload["format"] == _Answer.model_json_schema()
    assert payload["messages"] == [
        {"role": "system", "content": "be strict"},
        {"role": "user", "content": "the question"},
    ]
    assert payload["options"] == {"temperature": 0, "num_ctx": 4096}


def test_output_not_matching_schema_raises(capture):
    capture["reply"] = {"message": {"content": '{"skill": "Python"}'}}

    with pytest.raises(LLMError, match="does not match _Answer"):
        OllamaClient().generate_structured("sys", "user", _Answer)


def test_empty_content_raises(capture):
    capture["reply"] = {"message": {"content": ""}}

    with pytest.raises(LLMError, match="no content"):
        OllamaClient().generate_structured("sys", "user", _Answer)


def test_unreachable_server_raises_helpful_error(capture):
    capture["reply"] = urllib.error.URLError("connection refused")

    with pytest.raises(LLMError, match="Is it installed and running"):
        OllamaClient().generate_structured("sys", "user", _Answer)


def test_http_error_includes_server_detail(capture):
    capture["reply"] = urllib.error.HTTPError(
        "http://x", 404, "Not Found", {}, io.BytesIO(b'{"error":"model not found"}')
    )

    with pytest.raises(LLMError, match="404.*model not found"):
        OllamaClient().generate_structured("sys", "user", _Answer)


def test_timeout_raises(capture):
    capture["reply"] = TimeoutError()

    with pytest.raises(LLMError, match="did not answer"):
        OllamaClient(timeout=5).generate_structured("sys", "user", _Answer)
