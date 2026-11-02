import pytest

from cv_analyser.embedder import SentenceTransformerEmbedder


@pytest.mark.slow
def test_similar_texts_score_higher_than_unrelated_ones():
    embedder = SentenceTransformerEmbedder()
    vectors = embedder.embed(
        [
            "Experienced Python developer",
            "Senior software engineer working in Python",
            "Fresh bread recipe",
        ]
    )
    assert vectors.shape == (3, 384)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]