from pathlib import Path

from cv_analyser.exceptions import UnsupportedFileTypeError
from cv_analyser.loaders.docx_loader import load_docx
from cv_analyser.loaders.pdf_loader import load_pdf
from cv_analyser.text_cleaner import clean_text

_LOADERS = {".pdf": load_pdf, ".docx": load_docx}


def load_document(path: str | Path) -> str:
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{path.suffix}'. Use PDF or DOCX."
        )
    return clean_text(loader(path))
