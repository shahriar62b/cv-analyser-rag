from cv_analyser.text_cleaner import clean_text


def test_collapses_spaces_and_blank_lines():
    raw = "Name:   Rahim \n\n\n\nSkills:\u00a0Python  "
    assert clean_text(raw) == "Name: Rahim\n\nSkills: Python"