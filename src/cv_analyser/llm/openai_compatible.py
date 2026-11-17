"""Client for cloud providers that speak the OpenAI chat-completions format.

Groq, OpenRouter and Google Gemini all accept the same request shape at
different addresses, so one class covers all three; `PROVIDERS` holds the
differences. Cloud models see what you send, so the scripts always wrap this
client in `privacy.PrivateLLMClient` (see section "Privacy" in the README).
"""

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from cv_analyser.exceptions import LLMError
from cv_analyser.llm.base import T
from cv_analyser.llm.ollama import Usage


@dataclass(frozen=True)
class Provider:
    base_url: str
    key_variable: str  # environment variable that holds the API key
    default_model: str


PROVIDERS = {
    # Model catalogues change often: list what a provider offers with its /models
    # endpoint, and override the default with LLM_MODEL.
    "groq": Provider("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "openrouter": Provider(
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"
    ),
    "gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"
    ),
}

# Free tiers limit requests per minute (HTTP 429), and busy models are briefly
# "over capacity" (HTTP 503); both are temporary, so wait and try again.
_RETRY_CODES = {429, 503}
_RETRIES = 3
_DEFAULT_WAIT_SECONDS = 20.0
_MAX_WAIT_SECONDS = 120.0
_USER_AGENT = "cv-analyser/0.1"


class OpenAICompatibleClient:
    def __init__(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        reasoning_effort: str | None = None,
    ) -> None:
        if provider not in PROVIDERS:
            raise LLMError(f"Unknown cloud provider '{provider}'. Supported: {', '.join(PROVIDERS)}")
        settings = PROVIDERS[provider]
        key = api_key or os.environ.get(settings.key_variable, "").strip()
        if not key:
            raise LLMError(
                f"No API key for {provider}. Set the {settings.key_variable} environment variable."
            )
        self._provider = provider
        self._base_url = settings.base_url
        self._model = model or settings.default_model
        # Private, never printed or included in error messages.
        self._api_key = key
        self._timeout = timeout
        # Injected so tests can skip real waiting.
        self._sleep = sleep
        # Reasoning models "think" before answering, and those hidden tokens are
        # billed like any others: on gpt-oss they were 73% of the output.
        self._reasoning_effort = reasoning_effort or _default_reasoning_effort(self._model)
        self.usage = Usage()

    @property
    def model(self) -> str:
        return f"{self._provider}/{self._model}"

    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        # Not every provider and model supports full JSON-schema enforcement, but
        # all support "json_object" mode. So the schema goes into the instructions
        # and Pydantic checks the answer, with one retry that shows the error.
        instructions = (
            f"{system}\n\nReply with only a JSON object that matches this JSON schema:\n"
            f"{compact_schema(schema.model_json_schema())}"
        )
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user},
        ]
        content = self._chat(messages)
        try:
            return schema.model_validate_json(content)
        except ValidationError as first_error:
            messages += [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"That JSON does not match the schema: {first_error}. "
                    "Reply again with corrected JSON only.",
                },
            ]
            try:
                return schema.model_validate_json(self._chat(messages))
            except ValidationError as exc:
                raise LLMError(
                    f"Model '{self.model}' returned output that does not match {schema.__name__}"
                ) from exc

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        body = self._post(payload)
        usage = body.get("usage") or {}
        self.usage.calls += 1
        self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
        self.usage.output_tokens += usage.get("completion_tokens", 0)
        choices = body.get("choices") or [{}]
        content = (choices[0].get("message") or {}).get("content")
        if not content:
            raise LLMError(f"{self._provider} returned no content for model '{self._model}'")
        return str(content)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                # Cloudflare, in front of several providers, rejects Python's
                # default "Python-urllib" agent (error 1010).
                "User-Agent": _USER_AGENT,
            },
        )
        for attempt in range(_RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    result: dict[str, Any] = json.loads(response.read())
                    return result
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRY_CODES and attempt < _RETRIES:
                    self._sleep(_wait_seconds(exc.headers.get("Retry-After")))
                    continue
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise LLMError(f"{self._provider} returned HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise LLMError(f"Cannot reach {self._provider} at {self._base_url}: {exc.reason}") from exc
            except TimeoutError as exc:
                raise LLMError(f"{self._provider} did not answer within {self._timeout:.0f}s") from exc
        raise AssertionError("unreachable")  # the loop always returns or raises


def _wait_seconds(retry_after: str | None) -> float:
    try:
        return min(float(retry_after), _MAX_WAIT_SECONDS) if retry_after else _DEFAULT_WAIT_SECONDS
    except ValueError:
        return _DEFAULT_WAIT_SECONDS


# Models known to accept "reasoning_effort", and the level that keeps accuracy
# while thinking least (measured on the eval set; see the README).
_REASONING_DEFAULTS = {"openai/gpt-oss-": "low"}


def _default_reasoning_effort(model: str) -> str | None:
    return next((level for prefix, level in _REASONING_DEFAULTS.items() if model.startswith(prefix)), None)


def compact_schema(schema: dict[str, Any]) -> str:
    """The JSON schema as short as possible: no "title" entries (Pydantic adds
    one per field, repeating the field name) and no spaces. It is sent with
    every request, so every character counts."""

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if not (k == "title" and isinstance(v, str))}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return json.dumps(strip(schema), separators=(",", ":"))
