import pytest
from docx import Document
from PIL import Image

from cv_analyser.exceptions import (
    CorruptedFileError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from cv_analyser.loaders import pdf_loader
from cv_analyser.loaders.dispatcher import load_document


def test_unsupported_extension(tmp_path):
    file = tmp_path / "cv.txt"
    file.write_text("hello")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(file)


def test_corrupted_pdf(tmp_path):
    file = tmp_path / "bad.pdf"
    file.write_bytes(b"this is not a pdf")
    with pytest.raises(CorruptedFileError):
        load_document(file)


def test_corrupted_docx(tmp_path):
    file = tmp_path / "bad.docx"
    file.write_bytes(b"this is not a docx")
    with pytest.raises(CorruptedFileError):
        load_document(file)


def test_empty_docx(tmp_path):
    file = tmp_path / "empty.docx"
    Document().save(file)
    with pytest.raises(EmptyDocumentError):
        load_document(file)


def test_docx_reads_paragraphs_and_tables(tmp_path):
    file = tmp_path / "cv.docx"
    doc = Document()
    doc.add_paragraph("Rahim Uddin")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Python"
    table.rows[0].cells[1].text = "SQL"
    doc.save(file)

    text = load_document(file)
    assert "Rahim Uddin" in text
    assert "Python" in text and "SQL" in text


# --- Scanned CVs and OCR ---------------------------------------------------


class _FakeOcr:
    """Returns fixed lines and counts how many page images it was given."""

    def __init__(self, lines):
        self.lines = lines
        self.pages = 0

    def read_lines(self, image):
        self.pages += 1
        return self.lines


def _text_pdf(path, text):
    """Write a one-page PDF with a real text layer (no PDF library needed)."""
    stream = f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(bytes(out))


def _scanned_pdf(path):
    """Write a PDF whose only page is a picture, like a scanner produces."""
    Image.new("RGB", (850, 1100), "white").save(path, resolution=100)


def test_text_pdf_is_read_without_ocr(tmp_path):
    file = tmp_path / "cv.pdf"
    _text_pdf(file, "Built REST APIs in Python and Django")
    ocr = _FakeOcr(["should not be used"])

    assert load_document(file, ocr=ocr) == "Built REST APIs in Python and Django"
    assert ocr.pages == 0


def test_scanned_pdf_page_is_read_with_ocr(tmp_path):
    file = tmp_path / "scan.pdf"
    _scanned_pdf(file)
    ocr = _FakeOcr(["Nadia Karim", "Data Engineer"])

    assert load_document(file, ocr=ocr) == "Nadia Karim\nData Engineer"
    assert ocr.pages == 1


def test_short_stamp_on_a_scanned_page_still_triggers_ocr(tmp_path, monkeypatch):
    file = tmp_path / "cv.pdf"
    _text_pdf(file, "Scanned with CamScanner")
    # The hand-made PDF has no picture; pretend one covers the whole page.
    monkeypatch.setattr(pdf_loader, "_image_coverage", lambda page: 0.95)
    ocr = _FakeOcr(["Experience", "Built ETL pipelines"])

    assert load_document(file, ocr=ocr) == "Experience\nBuilt ETL pipelines"


def test_pdf_with_nothing_readable_is_empty(tmp_path):
    file = tmp_path / "blank.pdf"
    _scanned_pdf(file)

    with pytest.raises(EmptyDocumentError):
        load_document(file, ocr=_FakeOcr([]))


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".JPEG"])
def test_image_cv_is_read_with_ocr(tmp_path, suffix):
    file = tmp_path / f"cv{suffix}"
    Image.new("RGB", (400, 600), "white").save(file, format="PNG" if suffix == ".png" else "JPEG")

    assert load_document(file, ocr=_FakeOcr(["Python, SQL"])) == "Python, SQL"


def test_corrupted_image(tmp_path):
    file = tmp_path / "bad.jpg"
    file.write_bytes(b"this is not an image")

    with pytest.raises(CorruptedFileError):
        load_document(file, ocr=_FakeOcr(["x"]))


@pytest.mark.slow
def test_real_ocr_reads_a_rendered_page(tmp_path):
    from PIL import ImageDraw, ImageFont

    file = tmp_path / "scan.png"
    image = Image.new("RGB", (1200, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=36)
    draw.text((40, 60), "Data Engineer with 2 years of Python", fill="black", font=font)
    draw.text((40, 160), "Skills: SQL Server, Git, AWS Glue", fill="black", font=font)
    image.save(file)

    text = load_document(file)

    assert "Data Engineer with 2 years of Python" in text
    assert "SQL Server" in text
