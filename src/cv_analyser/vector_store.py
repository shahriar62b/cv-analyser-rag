from dataclasses import dataclass

import numpy as np

from cv_analyser.chunker import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    """Stores chunks with their vectors. Vectors must be L2-normalised,
    so a dot product equals cosine similarity."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors = np.empty((0, 0))

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        if self._chunks:
            self._vectors = np.vstack([self._vectors, vectors])
        else:
            self._vectors = vectors
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._chunks:
            return []
        scores = self._vectors @ query_vector
        best = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self._chunks[i], float(scores[i])) for i in best]