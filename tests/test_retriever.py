from cv_analyser.chunker import Chunk
from cv_analyser.retriever import CVRetriever


def test_retrieves_the_most_relevant_chunk(fake_embedder):
    chunks = [
        Chunk("Python Django REST APIs backend", "experience", 0),
        Chunk("Managed retail store inventory and sales", "experience", 1),
    ]
    retriever = CVRetriever(fake_embedder)
    retriever.index(chunks)

    results = retriever.retrieve("backend Python APIs", top_k=1)

    assert results[0].chunk.index == 0


def test_indexing_nothing_is_allowed(fake_embedder):
    retriever = CVRetriever(fake_embedder)
    retriever.index([])
    assert retriever.retrieve("anything") == []