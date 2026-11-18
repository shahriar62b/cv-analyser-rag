"""Step B and C: judge each requirement against retrieved CV lines, then
check that every quoted line really is in the CV."""

import re
from collections.abc import Callable

from pydantic import BaseModel

from cv_analyser.chunker import Chunk
from cv_analyser.llm.base import LLMClient
from cv_analyser.prompts import JUDGE_SYSTEM
from cv_analyser.report import MatchStatus, Requirement, RequirementVerdict
from cv_analyser.retriever import CVRetriever

Progress = Callable[[str], None]
# Receives each batch of verdicts as soon as it is judged, so a caller can show
# results while the rest are still being worked on.
VerdictSink = Callable[[list[RequirementVerdict]], None]

# More chunks per requirement make it less likely that the right line never
# reaches the judge.
TOP_K = 4


class _Judgement(BaseModel):
    # Field order matters: the model writes the quote and reasoning first and
    # only then picks a status, instead of deciding first and justifying after.
    number: int
    evidence: str
    reasoning: str
    status: MatchStatus


class _Judgements(BaseModel):
    judgements: list[_Judgement]


def judge_requirements(
    llm: LLMClient,
    retriever: CVRetriever,
    cv_text: str,
    requirements: list[Requirement],
    batch_size: int = 5,
    top_k: int = TOP_K,
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
            on_verdicts([_to_verdict(requirements[n - 1], found[n], cv_text) for n in sorted(found)])

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


def _judge_batch(
    llm: LLMClient,
    retriever: CVRetriever,
    requirements: list[Requirement],
    numbers: list[int],
    top_k: int,
) -> dict[int, _Judgement]:
    result = llm.generate_structured(
        JUDGE_SYSTEM, _batch_prompt(retriever, requirements, numbers, top_k), _Judgements
    )
    wanted = set(numbers)
    found: dict[int, _Judgement] = {}
    for judgement in result.judgements:
        if judgement.number in wanted and judgement.number not in found:
            found[judgement.number] = judgement
    return found


def _to_verdict(requirement: Requirement, judgement: _Judgement | None, cv_text: str) -> RequirementVerdict:
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
        pieces = _quote_pieces(judgement.evidence)
        evidence = " | ".join(pieces)
        cv = _normalise(cv_text)
        verified = bool(pieces) and all(_normalise(piece) in cv for piece in pieces)
    return RequirementVerdict(
        requirement=requirement,
        status=judgement.status,
        evidence=evidence,
        reasoning=judgement.reasoning.strip(),
        evidence_verified=verified,
    )


def _batch_prompt(
    retriever: CVRetriever, requirements: list[Requirement], numbers: list[int], top_k: int
) -> str:
    """List each retrieved CV excerpt once, then each requirement with the
    numbers of its excerpts.

    Related requirements retrieve the same lines ("Python" and "ETL pipelines"
    both find the ETL bullet). Writing the excerpts out per requirement repeated
    them; numbering them once cuts the prompt without hiding anything.
    """
    excerpt_ids: dict[Chunk, int] = {}
    requirement_lines = []
    for number in numbers:
        text = requirements[number - 1].text
        ids = [
            excerpt_ids.setdefault(hit.chunk, len(excerpt_ids) + 1) for hit in retriever.retrieve(text, top_k)
        ]
        refs = ", ".join(f"E{i}" for i in ids) or "(none)"
        requirement_lines.append(f"Requirement {number}: {text}\nRelevant excerpts: {refs}")
    excerpts = "\n".join(f"E{i} [{chunk.section}] {chunk.text}" for chunk, i in excerpt_ids.items())
    return f"CV excerpts:\n{excerpts or '(none)'}\n\n" + "\n\n".join(requirement_lines)


# Where a model joins separate CV lines in one quote: an excerpt number copied
# from the prompt (with its "[section]" label and a "-" or ":" after it), or an
# ellipsis standing for the skipped text in between.
_PIECE_BREAK = re.compile(r"\bE\d+\b(?:\s*\[[a-z ]+\])?\s*[-–:]?|\.\.\.|…")


def _quote_pieces(text: str) -> list[str]:
    """Split a quote into the CV lines it is made of.

    The prompt numbers excerpts E1, E2, ... and small models copy those numbers
    into quotes: "E1 [experience] - Built X", "Built X\\nE3 - Fixed Y", or just
    ", E6". Each excerpt number, ellipsis and line break marks a new piece; numbers,
    section labels and bullets are dropped, and pieces with no words are
    discarded. The quote counts as verified only if every piece is in the CV.
    """
    pieces = []
    for part in _PIECE_BREAK.split(text):
        for line in part.splitlines():
            piece = re.sub(r"^\s*(\[[a-z ]+\]\s*)?([-*•]\s*)?", "", line).strip(" \t,;")
            if any(char.isalnum() for char in piece):
                pieces.append(piece)
    return pieces


def _normalise(text: str) -> str:
    text = re.sub(r"[\"'“”‘’…]|\.\.\.", " ", text.lower())
    return " ".join(text.split()).strip(" .,;:")
