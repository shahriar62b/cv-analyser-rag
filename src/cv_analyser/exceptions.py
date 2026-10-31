class CVAnalyserError(Exception):
    """Base class for all errors raised by this project."""


class UnsupportedFileTypeError(CVAnalyserError):
    """The file extension is not supported (only PDF and DOCX)."""


class CorruptedFileError(CVAnalyserError):
    """The file cannot be opened or read (corrupt, encrypted, malformed)."""


class EmptyDocumentError(CVAnalyserError):
    """The file opened but no extractable text was found (e.g. scanned PDF)."""