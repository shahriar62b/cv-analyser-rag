from pathlib import Path

from PIL import Image, UnidentifiedImageError

from cv_analyser.exceptions import CorruptedFileError, EmptyDocumentError
from cv_analyser.loaders.ocr import OcrEngine, default_engine


def load_image(path: Path, ocr: OcrEngine | None = None) -> str:
    """Read a CV saved as a picture (a phone photo or a scan saved as JPG/PNG)."""
    try:
        with Image.open(path) as image:
            image.load()
            lines = (ocr or default_engine()).read_lines(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise CorruptedFileError(f"Cannot read image: {path.name}") from exc

    text = "\n".join(lines).strip()
    if not text:
        raise EmptyDocumentError(f"No readable text in image: {path.name}")
    return text
