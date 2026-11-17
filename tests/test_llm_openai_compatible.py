"""The cloud client, tested against a fake network: no API key or internet needed."""

import io
import json
import urllib.error

import pytest
from pydantic import BaseModel

from cv_analyser.exceptions import LLMError
from cv_analyser.llm import openai_compatible
from cv_analyser.llm.factory import create_llm_client, is_local_provider
from cv_analyser.llm.openai_compatible import OpenAICompatibleClient
from cv_analyser.privacy import PrivateLLMClient, protect_for_cloud

SECRET = "sk-test-secret-123"


class _Answer(BaseModel):
    skill: str
    years: int


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def _reply(content, prompt_tokens=10, completion_tokens=5):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _http_error(code, body=b"{}", headers=None):
    return urllib.error.HTTPError("https://x", code, "error", headers or {}, io.BytesIO(body))


@pytest.fixture
def network(monkeypatch):
    """Replace urlopen. Queue replies (dicts or exceptions) in state["replies"]."""
    state = {"requests": [], "replies": []}

    def fake_urlopen(request, timeout):
        state["requests"].append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "payload": json.loads(request.data),
            }
        )
        reply = state["replies"].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _FakeResponse(json.dumps(reply).encode())

    monkeypatch.setattr(openai_compatible.urllib.request, "urlopen", fake_urlopen)
    return state


def _client(**kwargs):
    waits = []
    client = OpenAICompatibleClient("groq", api_key=SECRET, sleep=waits.append, **kwargs)
    return client, waits


def test_sends_openai_style_request_and_parses_answer(network):
    network["replies"] = [_reply('{"skill": "Python", "years": 3}')]
    client, _ = _client()

    answer = client.generate_structured("Judge the CV.", "CV text", _Answer)

    assert answer == _Answer(skill="Python", years=3)
    request = network["requests"][0]
    assert request["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert request["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert request["headers"]["User-agent"] == "cv-analyser/0.1"  # not Python-urllib
    payload = request["payload"]
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    system = payload["messages"][0]["content"]
    assert system.startswith("Judge the CV.") and '"years"' in system  # schema in the instructions
    assert payload["messages"][1] == {"role": "user", "content": "CV text"}


def test_counts_tokens(network):
    network["replies"] = [_reply('{"skill": "SQL", "years": 1}', 100, 20)]
    client, _ = _client()

    client.generate_structured("s", "u", _Answer)

    assert (client.usage.calls, client.usage.prompt_tokens, client.usage.output_tokens) == (1, 100, 20)


def test_model_name_includes_the_provider():
    client, _ = _client(model="llama-3.1-8b-instant")

    assert client.model == "groq/llama-3.1-8b-instant"


def test_invalid_json_gets_one_corrected_retry(network):
    network["replies"] = [_reply('{"skill": "Python"}'), _reply('{"skill": "Python", "years": 2}')]
    client, _ = _client()

    answer = client.generate_structured("s", "u", _Answer)

    assert answer.years == 2
    retry_messages = network["requests"][1]["payload"]["messages"]
    assert retry_messages[2]["role"] == "assistant"
    assert "does not match the schema" in retry_messages[3]["content"]


def test_invalid_json_twice_raises(network):
    network["replies"] = [_reply("not json"), _reply("still not json")]
    client, _ = _client()

    with pytest.raises(LLMError, match="does not match _Answer"):
        client.generate_structured("s", "u", _Answer)


def test_rate_limit_waits_as_told_and_retries(network):
    network["replies"] = [
        _http_error(429, headers={"Retry-After": "7"}),
        _reply('{"skill": "Python", "years": 3}'),
    ]
    client, waits = _client()

    client.generate_structured("s", "u", _Answer)

    assert waits == [7.0]


def test_over_capacity_is_retried(network):
    network["replies"] = [_http_error(503), _reply('{"skill": "Python", "years": 3}')]
    client, waits = _client()

    client.generate_structured("s", "u", _Answer)

    assert waits == [20.0]  # no Retry-After header, so the default wait


def test_rate_limit_gives_up_after_three_retries(network):
    network["replies"] = [_http_error(429) for _ in range(4)]
    client, waits = _client()

    with pytest.raises(LLMError, match="HTTP 429"):
        client.generate_structured("s", "u", _Answer)
    assert len(waits) == 3


def test_errors_never_contain_the_api_key(network):
    network["replies"] = [_http_error(401, b'{"error": "invalid key"}')]
    client, _ = _client()

    with pytest.raises(LLMError) as error:
        client.generate_structured("s", "u", _Answer)
    assert "HTTP 401" in str(error.value)
    assert SECRET not in str(error.value)


def test_missing_api_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        OpenAICompatibleClient("groq")


def test_factory_builds_a_cloud_client_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    client = create_llm_client()

    assert isinstance(client, OpenAICompatibleClient)
    assert not is_local_provider()


def test_protect_for_cloud_masks_the_cv_and_guards_the_client(network):
    network["replies"] = [_reply('{"skill": "Python", "years": 3}')]
    client, _ = _client()
    cv = "Nadia Karim\nnadia@example.com\n\nSkills\nPython, SQL\n"

    masked_cv, guarded = protect_for_cloud(cv, client, fallback=None)
    guarded.generate_structured("s", masked_cv, _Answer)

    assert isinstance(guarded, PrivateLLMClient)
    assert "Python, SQL" in masked_cv
    sent = network["requests"][0]["payload"]["messages"][1]["content"]
    assert "Nadia" not in sent and "nadia@example.com" not in sent
