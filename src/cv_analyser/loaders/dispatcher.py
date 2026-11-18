from pathlib import Path

from cv_analyser.exceptions import UnsupportedFileTypeError
from cv_analyser.loaders.docx_loader import load_docx
from cv_analyser.loaders.image_loader import load_image
from cv_analyser.loaders.ocr import OcrEngine
from cv_analyser.loaders.pdf_loader import load_pdf
from cv_analyser.text_cleaner import clean_text

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_document(path: str | Path, ocr: OcrEngine | None = None) -> str:
    """Read a CV as plain text. Scanned pages and pictures go through OCR.

    `ocr` replaces the default OCR engine (tests pass a fake one).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        text = load_docx(path)
    elif suffix == ".pdf":
        text = load_pdf(path, ocr)
    elif suffix in IMAGE_SUFFIXES:
        text = load_image(path, ocr)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type '{path.suffix}'. Use PDF, DOCX, PNG or JPG.")
    return clean_text(text)
