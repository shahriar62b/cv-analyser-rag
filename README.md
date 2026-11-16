# CV Analyser

A RAG-based tool that compares a CV (PDF/DOCX) against a job description,
flags weaknesses, suggests improvements, and gives a verdict.

Everything runs locally and for free: embeddings use sentence-transformers and
the LLM runs through [Ollama](https://ollama.com). No CV data leaves your machine.

## Status
Work in progress.

- [x] Ingestion: load PDF/DOCX, clean text, split into sections and chunks
- [x] Retrieval: embed chunks and find CV evidence for each JD requirement
- [x] LLM layer: provider-agnostic client with a local Ollama backend
- [x] Analysis: LLM extracts requirements, judges each against CV evidence,
      and writes the final report

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Local LLM (free)

1. Install Ollama from https://ollama.com/download and make sure it is running.
2. Pull the default model (about 4.7 GB):
   ```bash
   ollama pull qwen2.5:7b
   ```
3. Check that the project can talk to it:
   ```bash
   pytest -m ollama
   ```

Configuration is read from environment variables:

| Variable       | Default                  | Meaning                         |
|----------------|--------------------------|---------------------------------|
| `LLM_PROVIDER` | `ollama`                 | Which LLM backend to use        |
| `LLM_MODEL`    | `qwen2.5:7b`             | Model name for that backend     |
| `OLLAMA_URL`   | `http://localhost:11434` | Address of the Ollama server    |

The 7B model (4-bit, about 4.7 GB) is the largest that runs at a usable speed on
a CPU-only laptop with 16 GB RAM. On a weaker machine use `LLM_MODEL=qwen2.5:3b`:
it is about twice as fast but less accurate and less consistent.

## Usage

```bash
python scripts/analyse_cv.py path/to/cv.pdf path/to/job.txt
python scripts/analyse_cv.py cv.docx job.txt --json report.json   # also save JSON
```

How it works:

1. The CV is split into sections and chunks, and the chunks are embedded.
2. The LLM extracts the job requirements (must-have vs nice-to-have).
3. For each requirement the most similar CV chunks are retrieved, and the LLM
   marks it met / partial / missing with a quote from the CV (5 per batch).
4. Every quote is checked against the CV text; unverified quotes earn half credit.
5. The score is computed in code: must-haves count double, partial counts half.
6. The LLM writes strengths, gaps, suggestions and a verdict.
7. Requirements the job explicitly calls mandatory are flagged when the CV does not
   meet them. The score is not changed, but the report and verdict warn about them.

Verdicts are printed as each batch finishes, so the first results appear before
the whole analysis is done. Requirements extracted from a job description are
saved in `data/cache/requirements/`, so checking another CV against the same job
skips extraction (`--no-cache` turns this off). Ollama keeps the model in memory
for 30 minutes after each request, so only the first run pays the loading time.

On a CPU-only laptop with `qwen2.5:7b`, a job with about 7 requirements took
about 5 minutes from a cold start, most of it spent writing the answers.

## Evaluation

`scripts/run_eval.py` compares the judge with a hand-labelled answer key of
fictional CVs and real job descriptions (kept out of git under `data/eval/`).
It reports strict and lenient accuracy, "met" claims for missing requirements,
unverified evidence, score error, extraction coverage and speed.

```bash
python scripts/run_eval.py --out data/eval/results/run.json --batch 3   # repeat until done
python scripts/run_eval.py --out data/eval/results/quick.json --pairs cv_a --no-extraction
```

Each finished pair is saved to a checkpoint next to `--out`, so a long run can be
split into batches and continued later; the model is unloaded after each batch.

## Tests

```bash
pytest              # fast tests only
pytest -m slow      # also load the real embedding model
pytest -m ollama    # call the local Ollama server
```

## Adding another LLM provider

Implement `generate_structured(system, user, schema)` (see
`src/cv_analyser/llm/base.py`) and register it in
`src/cv_analyser/llm/factory.py`. The rest of the code only depends on that
one method.
