from cv_analyser.chunker import Chunk
from cv_analyser.matching import match_requirements
from cv_analyser.retriever import CVRetriever


def test_each_requirement_gets_its_best_chunk(fake_embedder):
    chunks = [
        Chunk("Built Python REST APIs", "experience", 0),
        Chunk("Managed SQL databases and reporting", "experience", 1),
    ]
    retriever = CVRetriever(fake_embedder)
    retriever.index(chunks)

    matches = match_requirements(
        retriever, ["Build REST APIs in Python", "SQL reporting experience"], top_k=1
    )

    assert [m.evidence[0].chunk.index for m in matches] == [0, 1]