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

### Cloud LLM (optional, free tiers)

Larger models run in the cloud through any provider that speaks the OpenAI
chat format. Set the provider and its API key as environment variables; never
put a key in a file in this repository.

| `LLM_PROVIDER` | Key variable         | Default model                            |
|----------------|----------------------|------------------------------------------|
| `groq`         | `GROQ_API_KEY`       | `openai/gpt-oss-120b`                    |
| `openrouter`   | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct:free` |
| `gemini`       | `GEMINI_API_KEY`     | `gemini-2.5-flash`                       |

```powershell
setx GROQ_API_KEY "your-key"        # once; then open a new terminal
$env:LLM_PROVIDER = "groq"          # for this terminal session
python scripts/analyse_cv.py cv.pdf job.txt
```

With a cloud provider, the CV is always masked first and every request passes
the privacy guard (see "Privacy" below); a request that still holds personal
data is answered by the local Ollama model instead. Free tiers have daily
limits and may keep what they receive. Model catalogues change often; pick
another model with `LLM_MODEL`.

Token use per analysis with `openai/gpt-oss-120b` is about 5,200 (measured on
the eval set, with the same accuracy as the default settings at about 10,900):
cloud models judge all requirements in one call, shared CV excerpts are sent
once, the JSON schema is compacted, and gpt-oss "thinks" at low effort
(`LLM_REASONING_EFFORT=low|medium|high` to change it).

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

### Running the eval on a free Kaggle GPU

`notebooks/kaggle_eval.ipynb` runs the same local-model eval (Ollama with
`qwen2.5:7b`, the same code from GitHub) on Kaggle's free GPU instead of a
laptop CPU. Upload the fictional eval CVs, `answer_key.json` and the job ads as a
private Kaggle dataset, attach it, turn on the GPU and Internet in the notebook
settings, and run all cells. Never upload real people's CVs to Kaggle.

## Privacy: masking a CV before a cloud model sees it

Local models keep everything on your machine. Before a cloud model is used,
`cv_analyser.privacy` removes the candidate's personal data:

- only sections a job match can use are kept (summary, experience, education, skills,
  projects, certifications, achievements, languages, volunteering, activities); the
  contact header and everything else, such as personal information, references and
  interests, are dropped
- even then, only the CV excerpts retrieved for each requirement are sent, never the whole CV
- emails, links, phone numbers and labelled fields (date of birth, NID, address...) are masked
- the candidate's name is masked wherever it appears, in any case or script
- every request then passes an outbound guard that blocks it if any known personal
  data is still present, and falls back to the local model

Preview exactly what would be sent (nothing is sent by this command):

```bash
python scripts/redact_cv.py path/to/cv.pdf
python scripts/redact_cv.py path/to/cv.docx --mask "Hometown" --mask "Small Employer Ltd"
```

To see exactly what a cloud model received during a real analysis, save every
request to a local file:

```bash
python scripts/analyse_cv.py cv.pdf job.txt --show-sent sent.txt
```

Masking removes direct identifiers, not every indirect clue (a rare job title at a
small company can still point to someone), so get consent before using real CVs
with a cloud model.

## Tests

```bash
pytest              # fast tests only
pytest -m slow      # also load the real embedding model
pytest -m ollama    # call the local Ollama server
```

## Code checks

The same checks run on GitHub for every push and pull request
(`.github/workflows/ci.yml`). Run them locally before committing:

```bash
ruff check .          # lint: likely bugs, unused code, import order
ruff format .         # reformat the code (use --check to only report)
mypy                  # type check src/ and scripts/
pytest
```

## Adding another LLM provider

Implement a `model` property and `generate_structured(system, user, schema)` (see
`src/cv_analyser/llm/base.py`) and register it in
`src/cv_analyser/llm/factory.py`. The rest of the code only depends on that
interface. A provider with an OpenAI-compatible API only needs an entry in
`PROVIDERS` in `src/cv_analyser/llm/openai_compatible.py`.
