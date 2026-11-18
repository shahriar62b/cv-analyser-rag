"""Read the text in a picture of a page (a scanned CV or a photo of one).

OCR (optical character recognition) runs on this machine with RapidOCR, which
uses small ONNX models on the CPU, so the CV never leaves the computer.
"""

import logging
from functools import lru_cache
from typing import Protocol

import numpy as np
from PIL import Image

# RapidOCR gives each line a confidence between 0 and 1. Lower than this is
# usually a smudge, a logo or a photo read as letters.
MIN_CONFIDENCE = 0.5


class OcrEngine(Protocol):
    """Anything that turns a page image into lines of text, top to bottom."""

    def read_lines(self, image: Image.Image) -> list[str]: ...


class RapidOcrEngine:
    def __init__(self) -> None:
        # Imported here, not at the top: loading the models takes a few seconds
        # and most CVs (DOCX and normal PDFs) never need them.
        from rapidocr import RapidOCR

        self._engine = RapidOCR()
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)

    def read_lines(self, image: Image.Image) -> list[str]:
        result = self._engine(np.array(image.convert("RGB")))
        # The full pipeline returns lines with scores; the other result types
        # RapidOCR declares (detection-only and so on) never come back here.
        texts = getattr(result, "txts", None) or ()
        scores = getattr(result, "scores", None) or ()
        lines = zip(texts, scores, strict=True)
        return [text.strip() for text, score in lines if score >= MIN_CONFIDENCE and text.strip()]


@lru_cache(maxsize=1)
def default_engine() -> OcrEngine:
    """One shared engine, created the first time a page needs OCR."""
    return RapidOcrEngine()
