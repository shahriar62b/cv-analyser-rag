import pytest

from cv_analyser.exceptions import EmptyDocumentError
from cv_analyser.job_description import load_job_description


def test_txt_is_read_and_cleaned(tmp_path):
    file = tmp_path / "jd.txt"
    file.write_text("  Looking for   a Python dev \n", encoding="utf-8")
    assert load_job_description(file) == "Looking for a Python dev"


def test_empty_txt_raises(tmp_path):
    file = tmp_path / "jd.txt"
    file.write_text("   \n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_job_description(file)