import json
import urllib.error
import urllib.request
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from cv_analyser.exceptions import LLMError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_URL = "http://localhost:11434"


class OllamaClient:
    """Talks to a local Ollama server, so no data leaves the machine and nothing is billed."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        timeout: float = 600.0,
        context_tokens: int = 8192,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # Ollama's default context is small and it silently drops whatever
        # does not fit, which would cut off part of the CV.
        self._context_tokens = context_tokens

    @property
    def model(self) -> str:
        return self._model

    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        body = self._post(
            "/api/chat",
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # Passing the JSON schema makes Ollama constrain the output to it.
                "format": schema.model_json_schema(),
                "stream": False,
                "options": {"temperature": 0, "num_ctx": self._context_tokens},
            },
        )
        content = (body.get("message") or {}).get("content")
        if not content:
            raise LLMError(f"Ollama returned no content for model '{self._model}'")
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            raise LLMError(
                f"Model '{self._model}' returned output that does not match {schema.__name__}"
            ) from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self._base_url}. Is it installed and running?"
            ) from exc
        except TimeoutError as exc:
            raise LLMError(
                f"Ollama did not answer within {self._timeout:.0f}s (model '{self._model}')"
            ) from exc
