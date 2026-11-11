"""Talks to a real local Ollama server. Run with: pytest -m ollama"""

import pytest
from pydantic import BaseModel

from cv_analyser.llm.factory import create_llm_client


class _Skills(BaseModel):
    skills: list[str]


@pytest.mark.ollama
def test_local_model_returns_structured_output():
    llm = create_llm_client()

    result = llm.generate_structured(
        "Extract the technical skills mentioned in the text.",
        "Built REST APIs in Python and Django, deployed with Docker.",
        _Skills,
    )

    lowered = [skill.lower() for skill in result.skills]
    assert "python" in lowered
