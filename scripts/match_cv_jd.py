import sys
from pathlib import Path

from cv_analyser.chunker import chunk_sections
from cv_analyser.embedder import SentenceTransformerEmbedder
from cv_analyser.jd_requirements import extract_requirements
from cv_analyser.job_description import load_job_description
from cv_analyser.loaders.dispatcher import load_document
from cv_analyser.matching import match_requirements
from cv_analyser.retriever import CVRetriever
from cv_analyser.sections import split_into_sections


def main(cv_path: Path, jd_path: Path) -> None:
    chunks = chunk_sections(split_into_sections(load_document(cv_path)))
    requirements = extract_requirements(load_job_description(jd_path))

    retriever = CVRetriever(SentenceTransformerEmbedder())
    retriever.index(chunks)

    for match in match_requirements(retriever, requirements, top_k=1):
        if not match.evidence:
            continue
        best = match.evidence[0]
        print(f"\n[{best.score:.2f}] {match.requirement}")
        print(f"   -> ({best.chunk.section}) {best.chunk.text[:120]!r}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))