"""The full pipeline: retrieve, extract, judge, score and summarise.

Each step lives in its own module; this one only ties them together.
"""

from cv_analyser.chunker import chunk_sections
from cv_analyser.embedder import Embedder
from cv_analyser.extraction import extract_requirements_llm
from cv_analyser.judging import Progress, VerdictSink, judge_requirements
from cv_analyser.llm.base import LLMClient
from cv_analyser.report import CVReport, Requirement
from cv_analyser.retriever import CVRetriever
from cv_analyser.scoring import compute_score
from cv_analyser.sections import split_into_sections
from cv_analyser.summary import summarise

# Smaller chunks give sharper embeddings: a whole Experience section averages
# many topics into one vector.
CHUNK_CHARS = 300


def analyse_cv(
    cv_text: str,
    job_description: str,
    llm: LLMClient,
    embedder: Embedder,
    progress: Progress | None = None,
    requirements: list[Requirement] | None = None,
    on_verdicts: VerdictSink | None = None,
    batch_size: int = 5,
) -> CVReport:
    """Pass `requirements` to reuse an earlier extraction of the same job
    description; extraction is then skipped. `batch_size` is how many
    requirements are judged per LLM call (see `factory.judge_batch_size`)."""
    report_progress = progress or (lambda message: None)

    retriever = build_retriever(cv_text, embedder)

    if requirements is None:
        report_progress("Extracting requirements from the job description")
        requirements = extract_requirements_llm(llm, job_description)

    verdicts = judge_requirements(
        llm,
        retriever,
        cv_text,
        requirements,
        batch_size=batch_size,
        progress=report_progress,
        on_verdicts=on_verdicts,
    )

    report_progress("Writing the summary")
    score = compute_score(verdicts)
    return CVReport(verdicts=verdicts, overall_score=score, summary=summarise(llm, verdicts, score))


def build_retriever(cv_text: str, embedder: Embedder) -> CVRetriever:
    retriever = CVRetriever(embedder)
    retriever.index(chunk_sections(split_into_sections(cv_text), max_chars=CHUNK_CHARS))
    return retriever
