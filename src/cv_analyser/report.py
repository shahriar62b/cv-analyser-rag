from typing import Literal

from pydantic import BaseModel, Field

MatchStatus = Literal["met", "partial", "missing"]


class RequirementVerdict(BaseModel):
    requirement: str
    status: MatchStatus
    evidence: str = Field(
        description="Exact quote from the CV that supports the status; empty if missing"
    )
    reasoning: str = Field(description="One or two sentences explaining the status")


class CVReport(BaseModel):
    verdicts: list[RequirementVerdict]
    overall_score: int = Field(ge=0, le=100, description="How well the CV fits the job")
    strengths: list[str]
    gaps: list[str]
    suggestions: list[str] = Field(description="Concrete changes that would improve the CV")
    verdict: str = Field(description="Short overall conclusion for the candidate")
