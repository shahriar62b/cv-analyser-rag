from pathlib import Path

from cv_analyser.exceptions import CorruptedFileError, EmptyDocumentError
from cv_analyser.loaders.dispatcher import load_document
from cv_analyser.text_cleaner import clean_text


def load_job_description(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() != ".txt":
        return load_document(path)

    try:
        text = clean_text(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise CorruptedFileError(f"Cannot decode text file: {path.name}") from exc

    if not text:
        raise EmptyDocumentError(f"Job description is empty: {path.name}")
    return text