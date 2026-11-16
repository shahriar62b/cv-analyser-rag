"""Step E: the LLM writes strengths, gaps, suggestions and a verdict."""

from cv_analyser.llm.base import LLMClient
from cv_analyser.prompts import SUMMARY_SYSTEM
from cv_analyser.report import CVSummary, RequirementVerdict


def summarise(llm: LLMClient, verdicts: list[RequirementVerdict], score: int) -> CVSummary:
    lines = [
        f"- [{'must-have' if v.requirement.must_have else 'nice-to-have'}] "
        f"{v.requirement.text}: {v.status}. {v.reasoning}"
        for v in verdicts
    ]
    user = (
        f"Match score: {score}/100\n"
        f"{_count_line('Must-have', verdicts, must_have=True)}\n"
        f"{_count_line('Nice-to-have', verdicts, must_have=False)}\n"
        "The verdict must agree with these counts.\n"
        f"{_mandatory_line(verdicts)}\n"
        "Requirement results:\n" + "\n".join(lines)
    )
    return llm.generate_structured(SUMMARY_SYSTEM, user, CVSummary)


def _mandatory_line(verdicts: list[RequirementVerdict]) -> str:
    unmet = [v.requirement.text for v in verdicts if v.requirement.mandatory and v.status != "met"]
    if not unmet:
        return ""
    return (
        "Mandatory requirements not met: " + "; ".join(unmet) + ". The job says these are "
        "mandatory, so the verdict must say the candidate will likely be rejected unless "
        "they can show them.\n"
    )


def _count_line(label: str, verdicts: list[RequirementVerdict], must_have: bool) -> str:
    group = [v for v in verdicts if v.requirement.must_have == must_have]
    counts = {status: sum(v.status == status for v in group) for status in ("met", "partial", "missing")}
    return (
        f"{label} requirements: {len(group)} total, {counts['met']} met, "
        f"{counts['partial']} partial, {counts['missing']} missing"
    )
