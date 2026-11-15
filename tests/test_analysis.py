import pytest

from cv_analyser.analysis import (
    analyse_cv,
    compute_score,
    extract_requirements_llm,
    judge_requirements,
    summarise,
)
from cv_analyser.chunker import Chunk
from cv_analyser.report import Requirement, RequirementVerdict
from cv_analyser.retriever import CVRetriever

CV_TEXT = """Experience
Built REST APIs in Python and Django for 4 years.
Managed PostgreSQL databases and reporting.

Skills
Python, Django, PostgreSQL, Docker"""

SUMMARY = {
    "strengths": ["Python"],
    "gaps": ["Kubernetes"],
    "suggestions": ["Mention any container orchestration work"],
    "verdict": "Good fit.",
}


def _judgement(number, status="met", evidence="", reasoning="ok"):
    return {"number": number, "status": status, "evidence": evidence, "reasoning": reasoning}


def _verdict(status, must_have=True, verified=True):
    return RequirementVerdict(
        requirement=Requirement(text="x", must_have=must_have),
        status=status,
        evidence="",
        reasoning="",
        evidence_verified=verified,
    )


@pytest.fixture
def retriever(fake_embedder):
    retriever = CVRetriever(fake_embedder)
    retriever.index(
        [
            Chunk("Built REST APIs in Python and Django for 4 years.", "experience", 0),
            Chunk("Managed PostgreSQL databases and reporting.", "experience", 1),
        ]
    )
    return retriever


def test_extract_uses_llm_requirements(make_fake_llm):
    llm = make_fake_llm(
        [{"requirements": [{"text": "Python", "must_have": True}, {"text": " ", "must_have": True},
                           {"text": "Docker", "must_have": False}]}]
    )

    requirements = extract_requirements_llm(llm, "We need Python. Docker is a plus.")

    assert requirements == [
        Requirement(text="Python", must_have=True),
        Requirement(text="Docker", must_have=False),
    ]


def test_extract_falls_back_to_rules_when_llm_finds_nothing(make_fake_llm):
    llm = make_fake_llm([{"requirements": []}])

    requirements = extract_requirements_llm(llm, "- Strong SQL skills\n- Python experience")

    assert [r.text for r in requirements] == ["Strong SQL skills", "Python experience"]


def test_judge_batches_and_sends_excerpts(make_fake_llm, retriever):
    requirements = [Requirement(text=t) for t in ("Python APIs", "PostgreSQL", "Kubernetes")]
    llm = make_fake_llm(
        [
            {"judgements": [_judgement(1), _judgement(2)]},
            {"judgements": [_judgement(3, "missing")]},
        ]
    )

    judge_requirements(llm, retriever, CV_TEXT, requirements, batch_size=2)

    first_prompt, second_prompt = llm.calls[0][1], llm.calls[1][1]
    assert "Requirement 1: Python APIs" in first_prompt
    assert "Requirement 2: PostgreSQL" in first_prompt
    assert "[experience] Built REST APIs" in first_prompt
    assert "Requirement 3: Kubernetes" in second_prompt


def test_judge_verifies_quoted_evidence(make_fake_llm, retriever):
    requirements = [Requirement(text="Python"), Requirement(text="Leadership")]
    llm = make_fake_llm(
        [
            {
                "judgements": [
                    _judgement(1, evidence='"built REST APIs in  Python and Django."'),
                    _judgement(2, evidence="Led a team of ten engineers"),
                ]
            }
        ]
    )

    verdicts = judge_requirements(llm, retriever, CV_TEXT, requirements)

    assert [v.evidence_verified for v in verdicts] == [True, False]


def test_missing_status_needs_no_evidence(make_fake_llm, retriever):
    llm = make_fake_llm([{"judgements": [_judgement(1, "missing")]}])

    [verdict] = judge_requirements(llm, retriever, CV_TEXT, [Requirement(text="Go")])

    assert verdict.status == "missing"
    assert verdict.evidence_verified is True


def test_skipped_requirements_are_retried_then_marked_missing(make_fake_llm, retriever):
    requirements = [Requirement(text=t) for t in ("Python", "SQL", "Go")]
    llm = make_fake_llm(
        [
            # Skips 2 and 3, and returns an out-of-range number that must be ignored.
            {"judgements": [_judgement(1, "missing"), _judgement(9)]},
            # Retry: only 2 comes back.
            {"judgements": [_judgement(2, "missing")]},
        ]
    )

    verdicts = judge_requirements(llm, retriever, CV_TEXT, requirements)

    assert len(llm.calls) == 2
    assert "Requirement 2: SQL" in llm.calls[1][1]
    assert "Requirement 1:" not in llm.calls[1][1]
    assert verdicts[2].status == "missing"
    assert "did not return" in verdicts[2].reasoning


def test_judge_rejects_bad_batch_size(make_fake_llm, retriever):
    with pytest.raises(ValueError):
        judge_requirements(make_fake_llm([]), retriever, CV_TEXT, [], batch_size=0)


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        ([], 0),
        ([_verdict("met")], 100),
        ([_verdict("partial")], 50),
        ([_verdict("missing")], 0),
        ([_verdict("met", verified=False)], 50),
        # must-have missing (weight 2) vs nice-to-have met (weight 1): 1/3
        ([_verdict("missing"), _verdict("met", must_have=False)], 33),
    ],
)
def test_compute_score(verdicts, expected):
    assert compute_score(verdicts) == expected


def test_analyse_cv_end_to_end(make_fake_llm, fake_embedder):
    llm = make_fake_llm(
        [
            {"requirements": [{"text": "Python APIs", "must_have": True},
                              {"text": "Kubernetes", "must_have": False}]},
            {"judgements": [_judgement(1, evidence="Built REST APIs in Python"),
                            _judgement(2, "missing")]},
            SUMMARY,
        ]
    )
    messages = []

    report = analyse_cv(CV_TEXT, "job text", llm, fake_embedder, progress=messages.append)

    assert [v.status for v in report.verdicts] == ["met", "missing"]
    assert report.overall_score == 67
    assert report.summary.gaps == ["Kubernetes"]
    assert "Match score: 67/100" in llm.calls[2][1]
    assert messages[0].startswith("Extracting requirements")


def test_summary_prompt_includes_status_counts(make_fake_llm):
    llm = make_fake_llm([SUMMARY])
    verdicts = [
        _verdict("met"),
        _verdict("missing"),
        _verdict("missing"),
        _verdict("partial", must_have=False),
    ]

    summarise(llm, verdicts, score=40)

    prompt = llm.calls[0][1]
    assert "Must-have requirements: 3 total, 1 met, 0 partial, 2 missing" in prompt
    assert "Nice-to-have requirements: 1 total, 0 met, 1 partial, 0 missing" in prompt


def test_section_labels_and_bullets_are_stripped_from_quotes(make_fake_llm, retriever):
    llm = make_fake_llm(
        [{"judgements": [_judgement(1, evidence="[experience] - Managed PostgreSQL databases")]}]
    )

    [verdict] = judge_requirements(llm, retriever, CV_TEXT, [Requirement(text="SQL")])

    assert verdict.evidence == "Managed PostgreSQL databases"
    assert verdict.evidence_verified is True


def test_missing_verdict_hides_unrelated_quote(make_fake_llm, retriever):
    llm = make_fake_llm([{"judgements": [_judgement(1, "missing", evidence="Python, Django")]}])

    [verdict] = judge_requirements(llm, retriever, CV_TEXT, [Requirement(text="Kafka")])

    assert verdict.evidence == ""


def test_verdicts_are_streamed_per_batch(make_fake_llm, retriever):
    requirements = [Requirement(text=t) for t in ("Python APIs", "PostgreSQL", "Kubernetes")]
    llm = make_fake_llm(
        [
            {"judgements": [_judgement(1, evidence="Built REST APIs in Python"), _judgement(2, "missing")]},
            {"judgements": [_judgement(3, "missing")]},
        ]
    )
    batches = []

    verdicts = judge_requirements(
        llm, retriever, CV_TEXT, requirements, batch_size=2, on_verdicts=batches.append
    )

    assert [[v.requirement.text for v in batch] for batch in batches] == [
        ["Python APIs", "PostgreSQL"],
        ["Kubernetes"],
    ]
    assert batches[0][0] == verdicts[0]


def test_given_requirements_skip_extraction(make_fake_llm, fake_embedder):
    llm = make_fake_llm([{"judgements": [_judgement(1, "missing")]}, SUMMARY])
    messages = []

    report = analyse_cv(
        CV_TEXT, "job text", llm, fake_embedder,
        progress=messages.append, requirements=[Requirement(text="Go")],
    )

    assert len(llm.calls) == 2
    assert report.verdicts[0].requirement.text == "Go"
    assert not any(m.startswith("Extracting") for m in messages)


def test_unmet_mandatory_is_flagged_without_changing_the_score(make_fake_llm, fake_embedder):
    llm = make_fake_llm(
        [
            {"requirements": [{"text": "Python APIs", "must_have": True},
                              {"text": "Inditex buyers", "must_have": True, "mandatory": True}]},
            {"judgements": [_judgement(1, evidence="Built REST APIs in Python"),
                            _judgement(2, "missing")]},
            SUMMARY,
        ]
    )

    report = analyse_cv(CV_TEXT, "job text", llm, fake_embedder)

    assert report.overall_score == 50
    assert [v.requirement.text for v in report.unmet_mandatory] == ["Inditex buyers"]
    assert "Mandatory requirements not met: Inditex buyers" in llm.calls[2][1]


def test_summary_prompt_has_no_mandatory_line_when_all_are_met(make_fake_llm):
    llm = make_fake_llm([SUMMARY])
    verdict = _verdict("met")
    verdict.requirement.mandatory = True

    summarise(llm, [verdict], score=100)

    assert "Mandatory" not in llm.calls[0][1]
