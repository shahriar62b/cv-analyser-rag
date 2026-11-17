"""Measure judging accuracy and speed against the hand-labelled answer key.

The key's own requirement list is passed straight to the judge, so judging
accuracy is measured cleanly. Extraction is checked separately: does the LLM
find the requirements the key lists? Mismatched ("negative") pairs run the
full extract-and-judge flow and must score low.

The summary step is skipped: it does not change the verdicts or the score.

Each finished pair is appended to a checkpoint file next to --out, so a run can
be split into batches and stopped at any time. Running the same command again
continues with the pairs not yet done. The final JSON is written once every
selected pair is done. After each batch the model is unloaded from memory.

Usage:
    python scripts/run_eval.py --out data/eval/results/run.json --batch 3   # repeat until done
    python scripts/run_eval.py --out data/eval/results/run.json --fresh     # discard checkpoint
    python scripts/run_eval.py --out data/eval/results/quick.json --pairs cv_a --no-extraction
"""

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cv_analyser.analysis import build_retriever
from cv_analyser.embedder import Embedder, SentenceTransformerEmbedder
from cv_analyser.extraction import extract_requirements_llm
from cv_analyser.job_description import load_job_description
from cv_analyser.judging import judge_requirements
from cv_analyser.llm.base import LLMClient
from cv_analyser.llm.factory import create_llm_client, is_local_provider, judge_batch_size
from cv_analyser.loaders.dispatcher import load_document
from cv_analyser.privacy import protect_for_cloud
from cv_analyser.report import Requirement
from cv_analyser.scoring import compute_score

STATUSES = ("met", "partial", "missing")
# Cosine similarity above which an extracted requirement counts as covering a
# key requirement. Rough by nature; the matched pairs are saved for review.
COVERAGE_THRESHOLD = 0.6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key", type=Path, default=Path("data/eval/answer_key.json"))
    parser.add_argument("--cv-dir", type=Path, default=Path("data/eval/cvs"))
    parser.add_argument("--jd-dir", type=Path, default=Path("data/jds"))
    parser.add_argument(
        "--out", type=Path, required=True, help="Final results JSON; the checkpoint is kept next to it"
    )
    parser.add_argument("--batch", type=int, help="Run at most this many remaining pairs, then stop")
    parser.add_argument("--fresh", action="store_true", help="Discard the checkpoint and start over")
    parser.add_argument(
        "--keep-loaded",
        action="store_true",
        help="Leave the model in memory after the batch (default: unload it)",
    )
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
    model = llm.model
    cloud = not is_local_provider()
    checkpoint = args.out.with_name(args.out.stem + ".checkpoint.jsonl")
    if args.fresh:
        checkpoint.unlink(missing_ok=True)
    done = _load_checkpoint(checkpoint, model)
    if done is None:
        return 1
    pending = [p for p in pairs if _pair_id(p) not in done]
    todo = pending[: args.batch] if args.batch else pending
    print(f"{len(pairs) - len(pending)}/{len(pairs)} pairs already done; running {len(todo)} now", flush=True)

    # Loading the embedding model takes ~20 s, so skip it when only the summary is left.
    embedder: Embedder | None = SentenceTransformerEmbedder() if todo else None
    started = time.monotonic()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for number, pair in enumerate(todo, start=1):
        print(
            f"[{time.monotonic() - started:6.0f}s] {number}/{len(todo)} "
            f"{pair['cv']} x {pair['jd']} ({pair['kind']})",
            flush=True,
        )
        cv_text = load_document(args.cv_dir / f"{pair['cv']}.docx")
        jd_text = load_job_description(args.jd_dir / pair["jd"])
        pair_llm: LLMClient = llm
        if cloud:
            # Same privacy path as real use, so the eval measures what users get.
            # No local fallback: a blocked request stops the run instead of
            # silently mixing in local-model answers.
            cv_text, pair_llm = protect_for_cloud(cv_text, llm, fallback=None)
        assert embedder is not None  # loaded above whenever there is work to do
        retriever = build_retriever(cv_text, embedder)
        usage = getattr(llm, "usage", None)  # local and cloud clients count tokens
        before = asdict(usage) if usage is not None else None
        pair_started = time.monotonic()

        if pair["kind"] == "negative":
            result = _run_negative(pair, pair_llm, retriever, cv_text, jd_text)
        else:
            result = _run_labelled(
                pair, pair_llm, retriever, cv_text, jd_text, embedder, check_extraction=not args.no_extraction
            )
        result["seconds"] = round(time.monotonic() - pair_started, 1)
        if usage is not None and before is not None:
            result["usage"] = {k: round(v - before[k], 1) for k, v in asdict(usage).items()}
        result["model"] = model
        # Append one line per pair, so a stop or crash loses at most the pair in progress.
        with checkpoint.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        done[_pair_id(result)] = result

    if todo and not args.keep_loaded and hasattr(llm, "unload"):
        llm.unload()
        print("Unloaded the model from memory.", flush=True)

    remaining = [p for p in pairs if _pair_id(p) not in done]
    if remaining:
        print(
            f"\n{len(pairs) - len(remaining)}/{len(pairs)} pairs done. "
            "Run the same command again to continue."
        )
        return 0

    results = [done[_pair_id(p)] for p in pairs]
    summary = _summarise(results)
    _print_report(results, summary)
    payload = {"model": model, "summary": summary, "pairs": results}
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {args.out}")
    return 0


def _pair_id(pair):
    return f"{pair['cv']}|{pair['jd']}"


def _load_checkpoint(path, model):
    """Return finished results keyed by pair, or None if they came from another model."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by a crash; that pair simply runs again
        if result.get("model") != model:
            print(
                f"Checkpoint {path} was made with model {result.get('model')!r}, not {model!r}. "
                "Use --fresh to start over.",
                file=sys.stderr,
            )
            return None
        done[_pair_id(result)] = result
    return done


def _run_labelled(pair, llm, retriever, cv_text, jd_text, embedder, check_extraction):
    rows = pair["requirements"]
    requirements = [
        Requirement(text=r["text"], must_have=r["must_have"], mandatory=r.get("mandatory", False))
        for r in rows
    ]
    verdicts = judge_requirements(llm, retriever, cv_text, requirements, batch_size=judge_batch_size())

    graded = []
    for row, verdict in zip(rows, verdicts, strict=True):
        allowed = [row["expected"], *row.get("acceptable", [])]
        graded.append(
            {
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
            }
        )

    result = {
        "cv": pair["cv"],
        "jd": pair["jd"],
        "kind": pair["kind"],
        "score": compute_score(verdicts),
        "expected_score": pair["expected_score"],
        "rows": graded,
    }
    if check_extraction:
        result["extraction"] = _check_extraction(llm, embedder, jd_text, rows)
    return result


def _run_negative(pair, llm, retriever, cv_text, jd_text):
    requirements = extract_requirements_llm(llm, jd_text)
    verdicts = judge_requirements(llm, retriever, cv_text, requirements, batch_size=judge_batch_size())
    score = compute_score(verdicts)
    return {
        "cv": pair["cv"],
        "jd": pair["jd"],
        "kind": "negative",
        "score": score,
        "max_score": pair["max_score"],
        "passed": score <= pair["max_score"],
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
        {
            "key": key,
            "closest": found_texts[j],
            "similarity": round(float(similarity[i, j]), 2),
            "key_mandatory": rows[i].get("mandatory", False),
            "found_mandatory": extracted[j].mandatory,
        }
        for i, (key, j) in enumerate(zip(key_texts, best, strict=True))
    ]
    covered = sum(m["similarity"] >= COVERAGE_THRESHOLD for m in matches)
    return {"extracted": len(found_texts), "key": len(key_texts), "covered": covered, "matches": matches}


def _summarise(results):
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
        "score_mean_abs_error": round(
            float(np.mean([abs(r["score"] - r["expected_score"]) for r in labelled])), 1
        )
        if labelled
        else None,
        "confusion": {f"{e}->{g}": confusion[(e, g)] for e in STATUSES for g in STATUSES},
        "negatives_passed": f"{sum(r['passed'] for r in negatives)}/{len(negatives)}",
        "extraction_coverage": _pct(sum(e["covered"] for e in extraction), sum(e["key"] for e in extraction))
        if extraction
        else None,
        # Mandatory flags: how many the key marks, how many extraction caught,
        # and how many it added where the key has none.
        "mandatory_flags": _mandatory_flags(extraction),
        # Summed over pairs, so it is the compute time even when split into batches.
        "total_seconds": round(sum(r["seconds"] for r in results)),
    }
    usages = [r["usage"] for r in results if "usage" in r]
    if usages:
        total = {key: sum(u[key] for u in usages) for key in usages[0]}
        summary["llm"] = {
            "calls": int(total["calls"]),
            "prompt_tokens": int(total["prompt_tokens"]),
            "output_tokens": int(total["output_tokens"]),
            # Cloud providers don't report generation time, so there is no rate to show.
            "output_tokens_per_second": (
                round(total["output_tokens"] / total["output_seconds"], 1)
                if total["output_seconds"]
                else None
            ),
            "load_seconds": round(total["load_seconds"]),
            "prompt_seconds": round(total["prompt_seconds"]),
            "output_seconds": round(total["output_seconds"]),
        }
    return summary


def _mandatory_flags(extraction):
    matches = [m for e in extraction for m in e["matches"]]
    return {
        "in_key": sum(m.get("key_mandatory", False) for m in matches),
        "caught": sum(m.get("key_mandatory", False) and m.get("found_mandatory", False) for m in matches),
        "false_flags": sum(
            m.get("found_mandatory", False) and not m.get("key_mandatory", False) for m in matches
        ),
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
        print(
            f"{r['cv']:22} x {r['jd']:10} {r['kind']:9} {ok}/{len(r['rows'])} strict  "
            f"score {r['score']:3} (expected {r['expected_score']}){extra}  {r['seconds']:.0f}s"
        )

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
