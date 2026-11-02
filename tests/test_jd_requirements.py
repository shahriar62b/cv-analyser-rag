from cv_analyser.jd_requirements import extract_requirements


def test_bullets_are_extracted_without_markers():
    jd = (
        "About us\n\nResponsibilities:\n"
        "- Build REST APIs in Python\n"
        "• Write unit tests daily\n"
        "1. Mentor junior developers\n"
    )
    assert extract_requirements(jd) == [
        "Build REST APIs in Python",
        "Write unit tests daily",
        "Mentor junior developers",
    ]


def test_plain_lines_are_used_when_there_are_no_bullets():
    jd = "We need a data analyst\nwith strong SQL skills\nOK"
    assert extract_requirements(jd) == [
        "We need a data analyst",
        "with strong SQL skills",
    ]


def test_one_word_bullets_are_dropped():
    assert extract_requirements("- Python\n- Build REST APIs fast") == [
        "Build REST APIs fast"
    ]