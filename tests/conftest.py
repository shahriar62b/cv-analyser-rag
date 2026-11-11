import zlib

import numpy as np
import pytest


class _FakeEmbedder:
    """Deterministic bag-of-words embedder: shared words mean higher similarity."""

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dimension))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, zlib.crc32(word.encode()) % self._dimension] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


class _FakeLLM:
    """Returns pre-set answers in order and records every call it receives."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, type]] = []

    def generate_structured(self, system, user, schema):
        self.calls.append((system, user, schema))
        if not self._responses:
            raise AssertionError("FakeLLM received more calls than it has responses")
        return schema.model_validate(self._responses.pop(0))


@pytest.fixture
def make_fake_llm():
    return _FakeLLM
