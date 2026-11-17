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
    "groq": Provider("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openrouter": Provider(
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"
    ),
    "gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"
    ),
}

# Free tiers limit requests per minute; on HTTP 429 wait and try again this many times.
_RATE_LIMIT_RETRIES = 3
_DEFAULT_WAIT_SECONDS = 20.0
_MAX_WAIT_SECONDS = 120.0


class OpenAICompatibleClient:
    def __init__(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
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
            f"{json.dumps(schema.model_json_schema())}"
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
        body = self._post(
            {
                "model": self._model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        )
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
            },
        )
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    result: dict[str, Any] = json.loads(response.read())
                    return result
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < _RATE_LIMIT_RETRIES:
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
