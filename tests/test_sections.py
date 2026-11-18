import pytest

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


def test_secondary_cv_headings_get_their_own_sections():
    text = "Achievements\nCodeforces finalist\nVolunteer Work\nData for an NGO\nInterests\nFintech"

    sections = split_into_sections(text)

    assert [s.name for s in sections] == ["achievements", "volunteering", "interests"]


# Headings found in a survey of 2,484 real resumes (2026-11-18).
@pytest.mark.parametrize(
    ("heading", "name"),
    [
        ("Highlights", "skills"),
        ("Skill Highlights", "skills"),
        ("Core Qualifications", "skills"),
        ("Accomplishments", "achievements"),
        ("Publications", "achievements"),
        ("Executive Profile", "summary"),
        ("Career Overview", "summary"),
        ("Educational Background", "education"),
        ("Relevant Experience", "experience"),
        ("Community Service", "volunteering"),
        ("Professional Affiliations", "affiliations"),
        ("Additional Information", "additional"),
        ("Areas of Expertise", "skills"),
        ("Education and Training", "education"),
        ("Activities and Honors", "achievements"),
        ("Professional Development", "certifications"),
    ],
)
def test_headings_common_in_real_resumes(heading, name):
    sections = split_into_sections(f"{heading}\nsome content")

    assert [s.name for s in sections] == [name]


def test_unknown_heading_after_interests_no_longer_hides_honours():
    text = "Interests\nChess\nAdditional Information\nDean's List 2019"

    sections = {s.name: s.content for s in split_into_sections(text)}

    assert sections["additional"] == "Dean's List 2019"
