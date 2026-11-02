import re

_BULLET = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s*")


def extract_requirements(job_description: str, min_words: int = 2) -> list[str]:
    lines = [line for line in job_description.splitlines() if line.strip()]
    bulleted = [_BULLET.sub("", line).strip() for line in lines if _BULLET.match(line)]
    candidates = bulleted or [line.strip() for line in lines]
    return [c for c in candidates if len(c.split()) >= min_words]