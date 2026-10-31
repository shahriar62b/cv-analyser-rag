from pathlib import Path

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError

from cv_analyser.exceptions import CorruptedFileError, EmptyDocumentError


def load_pdf(path: Path) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except (PDFSyntaxError, PDFPasswordIncorrect) as exc:
        raise CorruptedFileError(f"Cannot read PDF: {path.name}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise EmptyDocumentError(f"No extractable text in PDF: {path.name}")
    return text