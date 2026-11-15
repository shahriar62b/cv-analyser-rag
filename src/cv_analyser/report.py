from typing import Literal

from pydantic import BaseModel, Field

MatchStatus = Literal["met", "partial", "missing"]


class Requirement(BaseModel):
    text: str = Field(description="One short, specific requirement, e.g. '3+ years of Python'")
    must_have: bool = Field(
        default=True,
        description="False only if the job description calls it optional, preferred or a plus",
    )


class RequirementVerdict(BaseModel):
    requirement: Requirement
    status: MatchStatus
    evidence: str
    reasoning: str
    # False when the quoted evidence cannot be found in the CV (small models
    # sometimes invent quotes), so the verdict should be trusted less.
    evidence_verified: bool


class CVSummary(BaseModel):
    strengths: list[str] = Field(description="What the candidate clearly offers for this job")
    gaps: list[str] = Field(description="Important requirements that are missing or weak")
    suggestions: list[str] = Field(description="Concrete, truthful changes that would improve the CV")
    verdict: str = Field(description="Two sentences on overall fit")


class CVReport(BaseModel):
    verdicts: list[RequirementVerdict]
    overall_score: int = Field(ge=0, le=100)
    summary: CVSummary
