import pytest
from pydantic import ValidationError

from cv_analyser.report import CVReport, CVSummary, Requirement, RequirementVerdict


def _report(**overrides) -> dict:
    data = {
        "verdicts": [
            {
                "requirement": {"text": "3+ years of Python", "must_have": True},
                "status": "met",
                "evidence": "Built Django services for 4 years",
                "reasoning": "Four years exceeds the three required.",
                "evidence_verified": True,
            }
        ],
        "overall_score": 80,
        "summary": {
            "strengths": ["Python"],
            "gaps": [],
            "suggestions": ["Quantify impact"],
            "verdict": "Strong fit",
        },
    }
    return data | overrides


def test_valid_report_parses():
    report = CVReport.model_validate(_report())

    assert report.verdicts[0].status == "met"


def test_requirement_is_must_have_by_default():
    assert Requirement(text="SQL").must_have is True


@pytest.mark.parametrize("score", [-1, 101])
def test_score_must_be_0_to_100(score):
    with pytest.raises(ValidationError):
        CVReport.model_validate(_report(overall_score=score))


def test_status_must_be_known_value():
    with pytest.raises(ValidationError):
        RequirementVerdict(
            requirement=Requirement(text="x"),
            status="maybe",
            evidence="",
            reasoning="",
            evidence_verified=True,
        )


def test_fake_llm_returns_schema_and_records_calls(make_fake_llm):
    summary = _report()["summary"]
    llm = make_fake_llm([summary])

    result = llm.generate_structured("sys", "user", CVSummary)

    assert result == CVSummary.model_validate(summary)
    assert llm.calls == [("sys", "user", CVSummary)]
