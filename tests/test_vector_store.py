import numpy as np
import pytest

from cv_analyser.chunker import Chunk
from cv_analyser.vector_store import InMemoryVectorStore


def make_chunk(text: str, index: int) -> Chunk:
    return Chunk(text=text, section="skills", index=index)


def test_best_match_comes_first():
    store = InMemoryVectorStore()
    store.add(
        [make_chunk("a", 0), make_chunk("b", 1)],
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    results = store.search(np.array([0.9, 0.1]), top_k=2)
    assert [r.chunk.text for r in results] == ["a", "b"]
    assert results[0].score > results[1].score


def test_top_k_limits_results():
    store = InMemoryVectorStore()
    store.add(
        [make_chunk("a", 0), make_chunk("b", 1)],
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert len(store.search(np.array([1.0, 0.0]), top_k=1)) == 1


def test_multiple_adds_accumulate():
    store = InMemoryVectorStore()
    store.add([make_chunk("a", 0)], np.array([[1.0, 0.0]]))
    store.add([make_chunk("b", 1)], np.array([[0.0, 1.0]]))
    assert len(store) == 2
    assert store.search(np.array([0.0, 1.0]), top_k=1)[0].chunk.text == "b"


def test_empty_store_returns_no_results():
    assert InMemoryVectorStore().search(np.array([1.0, 0.0])) == []


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        InMemoryVectorStore().add(
            [make_chunk("a", 0)], np.array([[1.0, 0.0], [0.0, 1.0]])
        )


def test_non_positive_top_k_raises():
    with pytest.raises(ValueError):
        InMemoryVectorStore().search(np.array([1.0, 0.0]), top_k=0)