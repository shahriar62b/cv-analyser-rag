import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from cv_analyser.exceptions import LLMError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_URL = "http://localhost:11434"
_NS = 1e9


@dataclass
class Usage:
    """Running totals from Ollama's own timing fields, for measuring speed."""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    load_seconds: float = 0.0
    prompt_seconds: float = 0.0
    output_seconds: float = 0.0

    def add(self, body: dict[str, Any]) -> None:
        self.calls += 1
        self.prompt_tokens += body.get("prompt_eval_count", 0)
        self.output_tokens += body.get("eval_count", 0)
        self.load_seconds += body.get("load_duration", 0) / _NS
        self.prompt_seconds += body.get("prompt_eval_duration", 0) / _NS
        self.output_seconds += body.get("eval_duration", 0) / _NS


class OllamaClient:
    """Talks to a local Ollama server, so no data leaves the machine and nothing is billed."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        timeout: float = 600.0,
        context_tokens: int = 8192,
        keep_alive: str = "30m",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # Ollama's default context is small and it silently drops whatever
        # does not fit, which would cut off part of the CV.
        self._context_tokens = context_tokens
        # Loading the 7B model takes about a minute on a CPU-only laptop, so keep
        # it in memory between requests instead of Ollama's default 5 minutes.
        self._keep_alive = keep_alive
        self.usage = Usage()

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
                "keep_alive": self._keep_alive,
                "options": {"temperature": 0, "num_ctx": self._context_tokens},
            },
        )
        self.usage.add(body)
        content = (body.get("message") or {}).get("content")
        if not content:
            raise LLMError(f"Ollama returned no content for model '{self._model}'")
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            raise LLMError(
                f"Model '{self._model}' returned output that does not match {schema.__name__}"
            ) from exc

    def unload(self) -> None:
        """Free the model's memory now instead of waiting for keep_alive to expire."""
        self._post("/api/generate", {"model": self._model, "keep_alive": 0})

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
            raise LLMError(f"Cannot reach Ollama at {self._base_url}. Is it installed and running?") from exc
        except TimeoutError as exc:
            raise LLMError(
                f"Ollama did not answer within {self._timeout:.0f}s (model '{self._model}')"
            ) from exc
