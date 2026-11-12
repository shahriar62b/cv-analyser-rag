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
- [ ] Analysis: LLM judges each requirement and writes the final report

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Local LLM (free)

1. Install Ollama from https://ollama.com/download and make sure it is running.
2. Pull the default model (about 2 GB):
   ```bash
   ollama pull qwen2.5:3b
   ```
3. Check that the project can talk to it:
   ```bash
   pytest -m ollama
   ```

Configuration is read from environment variables:

| Variable       | Default                  | Meaning                         |
|----------------|--------------------------|---------------------------------|
| `LLM_PROVIDER` | `ollama`                 | Which LLM backend to use        |
| `LLM_MODEL`    | `qwen2.5:3b`             | Model name for that backend     |
| `OLLAMA_URL`   | `http://localhost:11434` | Address of the Ollama server    |

A 3B model is a good fit for a CPU-only laptop with 16 GB RAM. With more
power, try a 7-8B model (e.g. `LLM_MODEL=qwen2.5:7b`) for better judgement.

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
