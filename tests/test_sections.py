from cv_analyser.sections import split_into_sections

SAMPLE = """Rahim Uddin
rahim@example.com

SUMMARY
Backend engineer.

Work Experience:
Engineer at X
Built APIs

Skills
Python, SQL
"""


def test_splits_known_headings_in_order():
    names = [s.name for s in split_into_sections(SAMPLE)]
    assert names == ["header", "summary", "experience", "skills"]


def test_content_goes_to_correct_section():
    sections = {s.name: s.content for s in split_into_sections(SAMPLE)}
    assert "Built APIs" in sections["experience"]
    assert sections["skills"] == "Python, SQL"


def test_text_without_headings_is_one_header_section():
    sections = split_into_sections("Just some text\nwithout headings")
    assert [s.name for s in sections] == ["header"]


def test_empty_text_gives_no_sections():
    assert split_into_sections("") == []