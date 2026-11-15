import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from cv_analyser.report import Requirement

_REQUIREMENTS = TypeAdapter(list[Requirement])


class RequirementsCache:
    """Remembers the requirements extracted from each job description, so the
    same job checked against several CVs is only extracted once.

    The key covers the job text, the model and the extraction prompt, so a
    change to any of them triggers a fresh extraction."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def get(self, job_description: str, model: str, prompt: str) -> list[Requirement] | None:
        path = self._path(job_description, model, prompt)
        try:
            return _REQUIREMENTS.validate_json(path.read_bytes())
        except FileNotFoundError:
            return None
        except ValidationError:
            # A corrupt or outdated file is not worth failing over; extract again.
            return None

    def put(self, job_description: str, model: str, prompt: str, requirements: list[Requirement]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._path(job_description, model, prompt)
        path.write_bytes(_REQUIREMENTS.dump_json(requirements, indent=2))

    def _path(self, job_description: str, model: str, prompt: str) -> Path:
        key = json.dumps([job_description.strip(), model, prompt])
        return self._directory / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}.json"
