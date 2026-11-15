import argparse
import sys
import time
from pathlib import Path

from cv_analyser.analysis import analyse_cv
from cv_analyser.embedder import SentenceTransformerEmbedder
from cv_analyser.exceptions import CVAnalyserError
from cv_analyser.job_description import load_job_description
from cv_analyser.llm.factory import create_llm_client
from cv_analyser.loaders.dispatcher import load_document
from cv_analyser.report import CVReport

_STATUS_ICON = {"met": "[x]", "partial": "[~]", "missing": "[ ]"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse a CV against a job description.")
    parser.add_argument("cv", type=Path, help="CV file (PDF or DOCX)")
    parser.add_argument("jd", type=Path, help="Job description (TXT, PDF or DOCX)")
    parser.add_argument("--json", type=Path, help="Also save the full report as JSON here")
    args = parser.parse_args()

    started = time.monotonic()

    def progress(message: str) -> None:
        print(f"[{time.monotonic() - started:5.0f}s] {message}...", flush=True)

    try:
        cv_text = load_document(args.cv)
        job_description = load_job_description(args.jd)
        progress("Loading the embedding model")
        report = analyse_cv(
            cv_text,
            job_description,
            llm=create_llm_client(),
            embedder=SentenceTransformerEmbedder(),
            progress=progress,
        )
    except CVAnalyserError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_report(report)
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {args.json}")
    return 0


def print_report(report: CVReport) -> None:
    print(f"\n=== Match score: {report.overall_score}/100 ===\n")
    for verdict in report.verdicts:
        kind = "must" if verdict.requirement.must_have else "nice"
        flag = "" if verdict.evidence_verified else "  (evidence not found in CV)"
        print(f"{_STATUS_ICON[verdict.status]} ({kind}) {verdict.requirement.text}{flag}")
        if verdict.evidence:
            print(f"      evidence: {verdict.evidence!r}")
        print(f"      {verdict.reasoning}")

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
