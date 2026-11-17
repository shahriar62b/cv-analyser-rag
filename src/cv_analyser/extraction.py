"""Step A: turn a job description into a list of requirements."""

import re

from pydantic import BaseModel

from cv_analyser.jd_requirements import extract_requirements
from cv_analyser.llm.base import LLMClient
from cv_analyser.prompts import EXTRACT_SYSTEM
from cv_analyser.report import Requirement


class _ExtractedRequirements(BaseModel):
    requirements: list[Requirement]


_MANDATORY_WORDS = re.compile(r"\b(mandatory|compulsory|essential|non-negotiable)\b", re.IGNORECASE)


def extract_requirements_llm(llm: LLMClient, job_description: str) -> list[Requirement]:
    result = llm.generate_structured(EXTRACT_SYSTEM, job_description, _ExtractedRequirements)
    requirements = [r for r in result.requirements if r.text.strip()]
    if not _MANDATORY_WORDS.search(job_description):
        # The model marks ordinary "required" items as mandatory far too often
        # (23 false flags on the eval set), so trust the flag only when the job
        # text actually uses one of these words.
        requirements = [r.model_copy(update={"mandatory": False}) for r in requirements]
    if requirements:
        return requirements
    # The model found nothing; fall back to the rule-based extractor.
    return [Requirement(text=text) for text in extract_requirements(job_description)]
