import argparse
import sys
import time
from pathlib import Path

from cv_analyser.analysis import analyse_cv
from cv_analyser.console import use_utf8_output
from cv_analyser.embedder import SentenceTransformerEmbedder
from cv_analyser.exceptions import CVAnalyserError
from cv_analyser.extraction import extract_requirements_llm
from cv_analyser.job_description import load_job_description
from cv_analyser.llm.factory import (
    create_llm_client,
    create_local_client,
    is_local_provider,
    judge_batch_size,
)
from cv_analyser.loaders.dispatcher import load_document
from cv_analyser.privacy import PrivateLLMClient, protect_for_cloud
from cv_analyser.prompts import EXTRACT_SYSTEM
from cv_analyser.report import CVReport, RequirementVerdict
from cv_analyser.requirements_cache import RequirementsCache

_STATUS_ICON = {"met": "[x]", "partial": "[~]", "missing": "[ ]"}


def main() -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description="Analyse a CV against a job description.")
    parser.add_argument("cv", type=Path, help="CV file (PDF, DOCX, PNG or JPG; scans are read with OCR)")
    parser.add_argument("jd", type=Path, help="Job description (TXT, PDF or DOCX)")
    parser.add_argument("--json", type=Path, help="Also save the full report as JSON here")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/requirements"),
        help="Where extracted job requirements are remembered (default: %(default)s)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Always extract requirements again")
    parser.add_argument(
        "--mask",
        action="append",
        default=[],
        help="With a cloud model: extra word or phrase to mask in the CV (can be repeated)",
    )
    parser.add_argument(
        "--show-sent",
        type=Path,
        help="With a cloud model: save every request that was sent to this local file, to check it",
    )
    args = parser.parse_args()

    started = time.monotonic()

    def progress(message: str) -> None:
        print(f"[{time.monotonic() - started:5.0f}s] {message}...", flush=True)

    try:
        cv_text = load_document(args.cv)
        job_description = load_job_description(args.jd)
        llm = create_llm_client()
        if not is_local_provider():
            # Cloud model: mask the CV, and guard every request; anything still
            # personal is answered by the local model instead of being sent.
            progress(f"Masking personal data before using {llm.model}")
            cv_text, llm = protect_for_cloud(
                cv_text, llm, fallback=create_local_client(), extra_terms=tuple(args.mask)
            )
        requirements = None
        cache = None if args.no_cache else RequirementsCache(args.cache_dir)
        if cache:
            requirements = cache.get(job_description, llm.model, EXTRACT_SYSTEM)
            if requirements is None:
                progress("Extracting requirements from the job description")
                requirements = extract_requirements_llm(llm, job_description)
                cache.put(job_description, llm.model, EXTRACT_SYSTEM, requirements)
            else:
                progress(f"Reusing {len(requirements)} requirements saved for this job")
        progress("Loading the embedding model")
        report = analyse_cv(
            cv_text,
            job_description,
            llm=llm,
            embedder=SentenceTransformerEmbedder(),
            progress=progress,
            requirements=requirements,
            on_verdicts=print_verdicts,
            batch_size=judge_batch_size(),
        )
    except CVAnalyserError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Saved before printing, so it survives any problem with the report.
    if isinstance(llm, PrivateLLMClient) and args.show_sent:
        save_sent(llm, args.show_sent)
    print_report(report)
    if isinstance(llm, PrivateLLMClient) and llm.blocked_requests:
        print(
            f"\nPrivacy: {llm.blocked_requests} request(s) still held personal data and were "
            "answered by the local model instead."
        )
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {args.json}")
    return 0


def save_sent(llm: PrivateLLMClient, path: Path) -> None:
    parts = [
        f"===== Request {i} of {len(llm.sent)} to {llm.model} =====\n"
        f"--- instructions ---\n{system}\n--- content ---\n{user}\n"
        for i, (system, user) in enumerate(llm.sent, start=1)
    ]
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nSaved the {len(llm.sent)} request(s) sent to the cloud model to {path}")


def print_verdicts(verdicts: list[RequirementVerdict]) -> None:
    """Print verdicts as each batch finishes, before the full report is ready."""
    for verdict in verdicts:
        if verdict.requirement.mandatory:
            kind = "MANDATORY"
        else:
            kind = "must" if verdict.requirement.must_have else "nice"
        flag = "" if verdict.evidence_verified else "  (evidence not found in CV)"
        print(f"{_STATUS_ICON[verdict.status]} ({kind}) {verdict.requirement.text}{flag}")
        if verdict.evidence:
            print(f"      evidence: {verdict.evidence!r}")
        print(f"      {verdict.reasoning}", flush=True)


def print_report(report: CVReport) -> None:
    print(f"\n=== Match score: {report.overall_score}/100 ===")
    for verdict in report.unmet_mandatory:
        print(f"!! Mandatory requirement not met ({verdict.status}): {verdict.requirement.text}")
    summary = report.summary
    for title, items in (
        ("Strengths", summary.strengths),
        ("Gaps", summary.gaps),
        ("Suggestions", summary.suggestions),
    ):
        print(f"\n{title}:")
        for item in items:
            print(f"  - {item}")
    print(f"\nVerdict: {summary.verdict}")


if __name__ == "__main__":
    sys.exit(main())
