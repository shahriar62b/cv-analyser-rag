"""Measure judging accuracy and speed against the hand-labelled answer key.

The key's own requirement list is passed straight to the judge, so judging
accuracy is measured cleanly. Extraction is checked separately: does the LLM
find the requirements the key lists? Mismatched ("negative") pairs run the
full extract-and-judge flow and must score low.

The summary step is skipped: it does not change the verdicts or the score.

Usage:
    python scripts/run_eval.py --out data/eval/results/baseline.json
    python scripts/run_eval.py --pairs cv_a --no-extraction     # quick subset
"""
import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cv_analyser.analysis import compute_score, extract_requirements_llm, judge_requirements
from cv_analyser.chunker import chunk_sections
from cv_analyser.embedder import SentenceTransformerEmbedder
from cv_analyser.job_description import load_job_description
from cv_analyser.llm.factory import create_llm_client
from cv_analyser.loaders.dispatcher import load_document
from cv_analyser.report import Requirement
from cv_analyser.retriever import CVRetriever
from cv_analyser.sections import split_into_sections

STATUSES = ("met", "partial", "missing")
# Cosine similarity above which an extracted requirement counts as covering a
# key requirement. Rough by nature; the matched pairs are saved for review.
COVERAGE_THRESHOLD = 0.6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key", type=Path, default=Path("data/eval/answer_key.json"))
    parser.add_argument("--cv-dir", type=Path, default=Path("data/eval/cvs"))
    parser.add_argument("--jd-dir", type=Path, default=Path("data/jds"))
    parser.add_argument("--out", type=Path, help="Save full results as JSON here")
    parser.add_argument("--pairs", help="Only run pairs whose CV or JD name contains this text")
    parser.add_argument("--no-extraction", action="store_true", help="Skip the extraction check")
    parser.add_argument("--no-negative", action="store_true", help="Skip the mismatched pairs")
    args = parser.parse_args()

    pairs = json.loads(args.key.read_text(encoding="utf-8"))["pairs"]
    if args.pairs:
        pairs = [p for p in pairs if args.pairs in p["cv"] or args.pairs in p["jd"]]
    if args.no_negative:
        pairs = [p for p in pairs if p["kind"] != "negative"]
    if not pairs:
        print("No pairs selected.", file=sys.stderr)
        return 1

    llm = create_llm_client()
    embedder = SentenceTransformerEmbedder()
    started = time.monotonic()
    results = []
    for number, pair in enumerate(pairs, start=1):
        print(f"[{time.monotonic() - started:6.0f}s] {number}/{len(pairs)} "
              f"{pair['cv']} x {pair['jd']} ({pair['kind']})", flush=True)
        cv_text = load_document(args.cv_dir / f"{pair['cv']}.docx")
        jd_text = load_job_description(args.jd_dir / pair["jd"])
        retriever = CVRetriever(embedder)
        retriever.index(chunk_sections(split_into_sections(cv_text)))
        before = asdict(llm.usage) if hasattr(llm, "usage") else None
        pair_started = time.monotonic()

        if pair["kind"] == "negative":
            result = _run_negative(pair, llm, retriever, cv_text, jd_text)
        else:
            result = _run_labelled(pair, llm, retriever, cv_text, jd_text, embedder,
                                   check_extraction=not args.no_extraction)
        result["seconds"] = round(time.monotonic() - pair_started, 1)
        if before is not None:
            result["usage"] = {k: round(v - before[k], 1) for k, v in asdict(llm.usage).items()}
        results.append(result)

    summary = _summarise(results, time.monotonic() - started, llm)
    _print_report(results, summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": getattr(llm, "model", "?"), "summary": summary, "pairs": results}
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved results to {args.out}")
    return 0


def _run_labelled(pair, llm, retriever, cv_text, jd_text, embedder, check_extraction):
    rows = pair["requirements"]
    requirements = [
        Requirement(text=r["text"], must_have=r["must_have"], mandatory=r.get("mandatory", False))
        for r in rows
    ]
    verdicts = judge_requirements(llm, retriever, cv_text, requirements)

    graded = []
    for row, verdict in zip(rows, verdicts):
        allowed = [row["expected"], *row.get("acceptable", [])]
        graded.append({
            "text": row["text"],
            "expected": row["expected"],
            "got": verdict.status,
            "strict": verdict.status == row["expected"],
            "lenient": verdict.status in allowed,
            # Claiming "met" for something the CV does not have is the worst error.
            "false_met": verdict.status == "met" and row["expected"] == "missing",
            "evidence": verdict.evidence,
            "evidence_verified": verdict.evidence_verified,
            "reasoning": verdict.reasoning,
            "trap": row.get("trap", ""),
        })

    result = {
        "cv": pair["cv"], "jd": pair["jd"], "kind": pair["kind"],
        "score": compute_score(verdicts), "expected_score": pair["expected_score"],
        "rows": graded,
    }
    if check_extraction:
        result["extraction"] = _check_extraction(llm, embedder, jd_text, rows)
    return result


def _run_negative(pair, llm, retriever, cv_text, jd_text):
    requirements = extract_requirements_llm(llm, jd_text)
    verdicts = judge_requirements(llm, retriever, cv_text, requirements)
    score = compute_score(verdicts)
    return {
        "cv": pair["cv"], "jd": pair["jd"], "kind": "negative",
        "score": score, "max_score": pair["max_score"], "passed": score <= pair["max_score"],
        "met": [v.requirement.text for v in verdicts if v.status == "met"],
    }


def _check_extraction(llm, embedder, jd_text, rows):
    extracted = extract_requirements_llm(llm, jd_text)
    key_texts = [r["text"] for r in rows]
    found_texts = [r.text for r in extracted]
    if not found_texts:
        return {"extracted": 0, "key": len(key_texts), "covered": 0, "matches": []}
    similarity = embedder.embed(key_texts) @ embedder.embed(found_texts).T
    best = similarity.argmax(axis=1)
    matches = [
        {"key": key, "closest": found_texts[j], "similarity": round(float(similarity[i, j]), 2),
         "key_mandatory": rows[i].get("mandatory", False), "found_mandatory": extracted[j].mandatory}
        for i, (key, j) in enumerate(zip(key_texts, best))
    ]
    covered = sum(m["similarity"] >= COVERAGE_THRESHOLD for m in matches)
    return {"extracted": len(found_texts), "key": len(key_texts), "covered": covered, "matches": matches}


def _summarise(results, seconds, llm):
    rows = [row for r in results if r["kind"] != "negative" for row in r["rows"]]
    labelled = [r for r in results if r["kind"] != "negative"]
    negatives = [r for r in results if r["kind"] == "negative"]
    claimed = [row for row in rows if row["got"] != "missing"]
    extraction = [r["extraction"] for r in labelled if "extraction" in r]
    confusion = Counter((row["expected"], row["got"]) for row in rows)

    summary = {
        "requirements": len(rows),
        "strict_accuracy": _pct(sum(r["strict"] for r in rows), len(rows)),
        "lenient_accuracy": _pct(sum(r["lenient"] for r in rows), len(rows)),
        "false_met": sum(r["false_met"] for r in rows),
        "unverified_evidence": sum(not r["evidence_verified"] for r in claimed),
        "score_mean_abs_error": round(float(np.mean(
            [abs(r["score"] - r["expected_score"]) for r in labelled])), 1) if labelled else None,
        "confusion": {f"{e}->{g}": confusion[(e, g)] for e in STATUSES for g in STATUSES},
        "negatives_passed": f"{sum(r['passed'] for r in negatives)}/{len(negatives)}",
        "extraction_coverage": _pct(sum(e["covered"] for e in extraction),
                                    sum(e["key"] for e in extraction)) if extraction else None,
        # Mandatory flags: how many the key marks, how many extraction caught,
        # and how many it added where the key has none.
        "mandatory_flags": _mandatory_flags(extraction),
        "total_seconds": round(seconds),
    }
    if hasattr(llm, "usage"):
        usage = llm.usage
        summary["llm"] = {
            "calls": usage.calls,
            "prompt_tokens": usage.prompt_tokens,
            "output_tokens": usage.output_tokens,
            "output_tokens_per_second": round(usage.output_tokens / max(usage.output_seconds, 0.1), 1),
            "load_seconds": round(usage.load_seconds),
            "prompt_seconds": round(usage.prompt_seconds),
            "output_seconds": round(usage.output_seconds),
        }
    return summary


def _mandatory_flags(extraction):
    matches = [m for e in extraction for m in e["matches"]]
    return {
        "in_key": sum(m["key_mandatory"] for m in matches),
        "caught": sum(m["key_mandatory"] and m["found_mandatory"] for m in matches),
        "false_flags": sum(m["found_mandatory"] and not m["key_mandatory"] for m in matches),
    }


def _pct(part, whole):
    return round(100 * part / whole, 1) if whole else None


def _print_report(results, summary):
    print("\n=== Per pair ===")
    for r in results:
        if r["kind"] == "negative":
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"{r['cv']:22} x {r['jd']:10} negative  score {r['score']:3} (max {r['max_score']}) {mark}")
            continue
        ok = sum(row["strict"] for row in r["rows"])
        extra = ""
        if "extraction" in r:
            e = r["extraction"]
            extra = f"  extraction {e['covered']}/{e['key']} covered ({e['extracted']} found)"
        print(f"{r['cv']:22} x {r['jd']:10} {r['kind']:9} {ok}/{len(r['rows'])} strict  "
              f"score {r['score']:3} (expected {r['expected_score']}){extra}  {r['seconds']:.0f}s")

    print("\n=== Misses (strict) ===")
    for r in results:
        for row in r.get("rows", []):
            if not row["strict"]:
                flag = " [acceptable]" if row["lenient"] else ""
                flag += " [FALSE MET]" if row["false_met"] else ""
                print(f"{r['cv']}: {row['text']}: expected {row['expected']}, got {row['got']}{flag}")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
