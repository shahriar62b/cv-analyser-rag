from dataclasses import dataclass

from cv_analyser.retriever import CVRetriever
from cv_analyser.vector_store import SearchResult


@dataclass(frozen=True)
class RequirementMatch:
    requirement: str
    evidence: list[SearchResult]


def match_requirements(
    retriever: CVRetriever, requirements: list[str], top_k: int = 2
) -> list[RequirementMatch]:
    return [
        RequirementMatch(requirement, retriever.retrieve(requirement, top_k))
        for requirement in requirements
    ]