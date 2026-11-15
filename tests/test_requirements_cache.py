from cv_analyser.report import Requirement
from cv_analyser.requirements_cache import RequirementsCache

REQUIREMENTS = [Requirement(text="Python"), Requirement(text="Docker", must_have=False)]


def test_round_trip(tmp_path):
    cache = RequirementsCache(tmp_path / "cache")
    cache.put("job text", "qwen2.5:7b", "prompt", REQUIREMENTS)

    assert cache.get("job text", "qwen2.5:7b", "prompt") == REQUIREMENTS


def test_miss_when_job_model_or_prompt_differs(tmp_path):
    cache = RequirementsCache(tmp_path)
    cache.put("job text", "qwen2.5:7b", "prompt", REQUIREMENTS)

    assert cache.get("other job", "qwen2.5:7b", "prompt") is None
    assert cache.get("job text", "qwen2.5:3b", "prompt") is None
    assert cache.get("job text", "qwen2.5:7b", "new prompt") is None


def test_surrounding_whitespace_does_not_change_the_key(tmp_path):
    cache = RequirementsCache(tmp_path)
    cache.put("job text\n", "m", "p", REQUIREMENTS)

    assert cache.get("  job text", "m", "p") == REQUIREMENTS


def test_corrupt_file_is_treated_as_a_miss(tmp_path):
    cache = RequirementsCache(tmp_path)
    cache.put("job", "m", "p", REQUIREMENTS)
    next(tmp_path.glob("*.json")).write_text("not json", encoding="utf-8")

    assert cache.get("job", "m", "p") is None
