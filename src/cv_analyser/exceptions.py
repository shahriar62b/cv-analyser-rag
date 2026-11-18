class CVAnalyserError(Exception):
    """Base class for all errors raised by this project."""


class UnsupportedFileTypeError(CVAnalyserError):
    """The file extension is not supported (only PDF, DOCX, PNG and JPG)."""


class CorruptedFileError(CVAnalyserError):
    """The file cannot be opened or read (corrupt, encrypted, malformed)."""


class EmptyDocumentError(CVAnalyserError):
    """The file opened but no text was found, even with OCR (e.g. a blank page)."""


class LLMError(CVAnalyserError):
    """The LLM could not be reached or returned output we cannot use."""


class PrivacyLeakError(CVAnalyserError):
    """A request to a cloud model still contained the candidate's personal data."""
