import pytest
from pydantic import ValidationError

from cv_analyser.report import CVReport, RequirementVerdict


def _report(**overrides) -> dict:
    data = {
        "verdicts": [
            {
                "requirement": "3+ years of Python",
                "status": "met",
                "evidence": "Built Django services for 4 years",
                "reasoning": "Four years exceeds the three required.",
            }
        ],
        "overall_score": 80,
        "strengths": ["Python"],
        "gaps": [],
        "suggestions": ["Quantify impact"],
        "verdict": "Strong fit",
    }
    return data | overrides


def test_valid_report_parses():
    report = CVReport.model_validate(_report())

    assert report.verdicts[0].status == "met"


@pytest.mark.parametrize("score", [-1, 101])
def test_score_must_be_0_to_100(score):
    with pytest.raises(ValidationError):
        CVReport.model_validate(_report(overall_score=score))


def test_status_must_be_known_value():
    with pytest.raises(ValidationError):
        RequirementVerdict(requirement="x", status="maybe", evidence="", reasoning="")


def test_fake_llm_returns_schema_and_records_calls(make_fake_llm):
    llm = make_fake_llm([_report()])

    report = llm.generate_structured("sys", "user", CVReport)

    assert isinstance(report, CVReport)
    assert llm.calls == [("sys", "user", CVReport)]
