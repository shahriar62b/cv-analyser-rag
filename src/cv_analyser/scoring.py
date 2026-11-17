"""Step D: the match score, computed in code rather than by the LLM."""

from cv_analyser.report import RequirementVerdict


def compute_score(verdicts: list[RequirementVerdict]) -> int:
    """Weighted share of requirements met: must-haves count double, partial
    matches count half, and unverified evidence halves the credit."""
    credit = {"met": 1.0, "partial": 0.5, "missing": 0.0}
    earned = possible = 0.0
    for verdict in verdicts:
        weight = 2.0 if verdict.requirement.must_have else 1.0
        value = credit[verdict.status]
        if not verdict.evidence_verified:
            value /= 2
        earned += weight * value
        possible += weight
    return round(100 * earned / possible) if possible else 0
