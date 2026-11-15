import re
from collections.abc import Callable

from pydantic import BaseModel

from cv_analyser.chunker import chunk_sections
from cv_analyser.embedder import Embedder
from cv_analyser.jd_requirements import extract_requirements
from cv_analyser.llm.base import LLMClient
from cv_analyser.report import (
    CVReport,
    CVSummary,
    MatchStatus,
    Requirement,
    RequirementVerdict,
)
from cv_analyser.retriever import CVRetriever
from cv_analyser.sections import split_into_sections

Progress = Callable[[str], None]
# Receives each batch of verdicts as soon as it is judged, so a caller can show
# results while the rest are still being worked on.
VerdictSink = Callable[[list[RequirementVerdict]], None]

EXTRACT_SYSTEM = """You extract hiring requirements from a job description.
Return every distinct requirement a candidate must or should meet: skills, tools,
experience, education, certifications and languages.
Rules:
- One requirement per item, short and specific (e.g. "3+ years of Python").
- Set must_have to false only when the text marks it as optional, preferred, a plus or nice to have.
- Set mandatory to true only when the text explicitly calls the requirement mandatory,
  compulsory, essential or non-negotiable. Words like "required" or "must have" alone
  do not make it mandatory.
- Skip company descriptions, benefits, salary and application instructions."""

JUDGE_SYSTEM = """You check whether a CV meets job requirements, using only the CV excerpts given.
For each requirement, in this order:
1. number: the requirement number
2. evidence: the line from the excerpts that comes closest to the requirement, copied
   exactly, or "" if no line is related at all
3. reasoning: one sentence comparing that line with the requirement
4. status:
   - "met": the line shows the requirement itself, at the level asked for
   - "partial": the line shows a smaller, weaker or earlier form of the same kind of
     experience (e.g. required "managing people", line says "trained new joiners";
     required "5+ years", line shows 2 years; required "AWS", line says "Google Cloud")
   - "missing": no line shows that kind of experience
A different technology that is merely used alongside the required one does not count:
Python is not Kafka, Docker is not Kubernetes. If the required thing is not written,
the status is "missing". Do not assume anything that is not written."""

SUMMARY_SYSTEM = """You are an honest career advisor. You are given how a CV matched each job requirement.
Write:
- strengths: requirements the CV clearly meets
- gaps: unmet requirements, must-haves first
- suggestions: concrete improvements to the CV
- verdict: two sentences on overall fit
Rules for suggestions:
- Never suggest adding skills, experience or sections the candidate does not have.
- For a gap, suggest either making it visible if the candidate truly has it
  (e.g. "If you have used Kubernetes, name the project and what you did"),
  or building that skill before applying.
- Suggest ways to strengthen what is already there, such as adding numbers or results.
Be specific and brief."""


class _ExtractedRequirements(BaseModel):
    requirements: list[Requirement]


class _Judgement(BaseModel):
    # Field order matters: the model writes the quote and reasoning first and
    # only then picks a status, instead of deciding first and justifying after.
    number: int
    evidence: str
    reasoning: str
    status: MatchStatus


class _Judgements(BaseModel):
    judgements: list[_Judgement]


def analyse_cv(
    cv_text: str,
    job_description: str,
    llm: LLMClient,
    embedder: Embedder,
    progress: Progress | None = None,
    requirements: list[Requirement] | None = None,
    on_verdicts: VerdictSink | None = None,
) -> CVReport:
    """Pass `requirements` to reuse an earlier extraction of the same job
    description; extraction is then skipped."""
    report_progress = progress or (lambda message: None)

    retriever = CVRetriever(embedder)
    retriever.index(chunk_sections(split_into_sections(cv_text)))

    if requirements is None:
        report_progress("Extracting requirements from the job description")
        requirements = extract_requirements_llm(llm, job_description)

    verdicts = judge_requirements(
        llm, retriever, cv_text, requirements, progress=report_progress, on_verdicts=on_verdicts
    )

    report_progress("Writing the summary")
    score = compute_score(verdicts)
    return CVReport(verdicts=verdicts, overall_score=score, summary=summarise(llm, verdicts, score))


def extract_requirements_llm(llm: LLMClient, job_description: str) -> list[Requirement]:
    result = llm.generate_structured(EXTRACT_SYSTEM, job_description, _ExtractedRequirements)
    requirements = [r for r in result.requirements if r.text.strip()]
    if requirements:
        return requirements
    # The model found nothing; fall back to the rule-based extractor.
    return [Requirement(text=text) for text in extract_requirements(job_description)]


def judge_requirements(
    llm: LLMClient,
    retriever: CVRetriever,
    cv_text: str,
    requirements: list[Requirement],
    batch_size: int = 5,
    top_k: int = 2,
    progress: Progress | None = None,
    on_verdicts: VerdictSink | None = None,
) -> list[RequirementVerdict]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    judgements: dict[int, _Judgement] = {}

    def run_batch(numbers: list[int]) -> None:
        found = _judge_batch(llm, retriever, requirements, numbers, top_k)
        judgements.update(found)
        if on_verdicts and found:
            on_verdicts([
                _to_verdict(requirements[n - 1], found[n], cv_text) for n in sorted(found)
            ])

    total_batches = -(-len(requirements) // batch_size)
    for start in range(0, len(requirements), batch_size):
        if progress:
            progress(f"Judging requirements (batch {start // batch_size + 1} of {total_batches})")
        run_batch(list(range(start + 1, min(start + batch_size, len(requirements)) + 1)))

    # Small models sometimes skip an item; give those one more try on their own.
    skipped = [n for n in range(1, len(requirements) + 1) if n not in judgements]
    if skipped:
        run_batch(skipped)

    return [
        _to_verdict(requirement, judgements.get(number), cv_text)
        for number, requirement in enumerate(requirements, start=1)
    ]


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


def _judge_batch(
    llm: LLMClient,
    retriever: CVRetriever,
    requirements: list[Requirement],
    numbers: list[int],
    top_k: int,
) -> dict[int, _Judgement]:
    blocks = []
    for number in numbers:
        text = requirements[number - 1].text
        excerpts = "\n".join(
            f"[{hit.chunk.section}] {hit.chunk.text}" for hit in retriever.retrieve(text, top_k)
        )
        blocks.append(f"Requirement {number}: {text}\nCV excerpts:\n{excerpts or '(none)'}")

    result = llm.generate_structured(JUDGE_SYSTEM, "\n\n".join(blocks), _Judgements)
    wanted = set(numbers)
    found: dict[int, _Judgement] = {}
    for judgement in result.judgements:
        if judgement.number in wanted and judgement.number not in found:
            found[judgement.number] = judgement
    return found


def _to_verdict(
    requirement: Requirement, judgement: _Judgement | None, cv_text: str
) -> RequirementVerdict:
    if judgement is None:
        return RequirementVerdict(
            requirement=requirement,
            status="missing",
            evidence="",
            reasoning="The model did not return a judgement for this requirement.",
            evidence_verified=False,
        )
    if judgement.status == "missing":
        # The model quotes the closest line even when it is unrelated; don't show it.
        evidence, verified = "", True
    else:
        evidence = _clean_quote(judgement.evidence)
        verified = bool(evidence) and _normalise(evidence) in _normalise(cv_text)
    return RequirementVerdict(
        requirement=requirement,
        status=judgement.status,
        evidence=evidence,
        reasoning=judgement.reasoning.strip(),
        evidence_verified=verified,
    )


def _clean_quote(text: str) -> str:
    # Models sometimes copy the "[section]" label or bullet marker from the prompt.
    return re.sub(r"^(\[[a-z]+\]\s*)?([-*•]\s*)?", "", text.strip())


def _normalise(text: str) -> str:
    text = re.sub(r"[\"'“”‘’…]|\.\.\.", " ", text.lower())
    return " ".join(text.split()).strip(" .,;:")
