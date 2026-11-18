from pathlib import Path

import pdfplumber
from pdfplumber.utils.exceptions import PdfminerException

from cv_analyser.exceptions import CorruptedFileError, EmptyDocumentError
from cv_analyser.loaders.ocr import OcrEngine, default_engine

# A page with less text than this has no real text layer: it's a scan.
MIN_PAGE_CHARS = 20
# Scanning apps sometimes stamp a short line of real text ("Scanned with
# CamScanner") on top of the page image. A page with little more text than that
# and a picture covering most of it is treated as a scan too.
STAMP_CHARS = 200
SCAN_IMAGE_COVERAGE = 0.5
# Resolution the page is drawn at for OCR. 200 dots per inch reads normal CV
# fonts well; higher is slower without helping much.
OCR_DPI = 200


def load_pdf(path: Path, ocr: OcrEngine | None = None) -> str:
    """Read a PDF's text layer, falling back to OCR for pages that are images.

    `ocr` lets tests pass a fake engine; by default the shared RapidOCR engine
    is loaded the first time a page needs it.
    """
    try:
        with pdfplumber.open(path) as pdf:
            pages = [_page_text(page, ocr) for page in pdf.pages]
    except PdfminerException as exc:
        raise CorruptedFileError(f"Cannot read PDF: {path.name}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise EmptyDocumentError(f"No readable text in PDF, even with OCR: {path.name}")
    return text


def _page_text(page, ocr: OcrEngine | None) -> str:
    text = page.extract_text() or ""
    if not _is_scanned(page, text):
        return text
    engine = ocr or default_engine()
    return "\n".join(engine.read_lines(page.to_image(resolution=OCR_DPI).original))


def _is_scanned(page, text: str) -> bool:
    chars = len(text.strip())
    if chars < MIN_PAGE_CHARS:
        return True
    return chars < STAMP_CHARS and _image_coverage(page) > SCAN_IMAGE_COVERAGE


def _image_coverage(page) -> float:
    """Share of the page covered by its largest picture (0 to 1)."""
    page_area = float(page.width * page.height)
    areas = [float((image["x1"] - image["x0"]) * (image["bottom"] - image["top"])) for image in page.images]
    return max(areas, default=0.0) / page_area if page_area else 0.0
