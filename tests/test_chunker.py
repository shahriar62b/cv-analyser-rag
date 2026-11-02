import pytest

from cv_analyser.chunker import chunk_sections
from cv_analyser.sections import Section


def test_short_section_is_one_chunk():
    chunks = chunk_sections([Section("skills", "Python, SQL")], max_chars=100)
    assert len(chunks) == 1
    assert chunks[0].section == "skills"


def test_long_section_splits_within_limit():
    content = "\n\n".join(f"Role {i}: " + "x" * 40 for i in range(10))
    chunks = chunk_sections([Section("experience", content)], max_chars=120)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)


def test_indexes_run_across_sections():
    sections = [Section("summary", "a"), Section("skills", "b")]
    assert [c.index for c in chunk_sections(sections)] == [0, 1]


def test_non_positive_max_chars_raises():
    with pytest.raises(ValueError):
        chunk_sections([Section("skills", "a")], max_chars=0)