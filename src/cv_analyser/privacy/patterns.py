"""Detectors for personal data that has a recognisable shape.

Shared by the redactor (which masks the CV before analysis) and the outbound
guard (which checks every request just before it leaves the machine).
"""

import re
import unicodedata

# Zero-width spaces, joiners, byte-order marks and soft hyphens: invisible in a
# document, but they break pattern matching ("rahim\u200b@gmail.com").
_INVISIBLE = re.compile("[\u200b-\u200f\u2060\ufeff\u00ad]")


def normalise(text: str) -> str:
    """Unify look-alike characters (NFKC turns full-width letters and symbols into plain ones) and
    remove invisible ones, so the patterns below see plain text."""
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))


EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# "rahim [at] gmail [dot] com", "rahim (at) gmail (dot) com"
OBFUSCATED_EMAIL = re.compile(
    r"[\w.+-]+\s*[\[(]\s*at\s*[\])]\s*[\w-]+(?:\s*[\[(]\s*dot\s*[\])]\s*[\w-]+)+", re.IGNORECASE
)
URL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b(?:linkedin|github|gitlab|facebook|behance|medium|kaggle|bit)\.(?:com|ly|io)/\S*",
    re.IGNORECASE,
)
# Lines such as "Mobile: 01712 345678" or "Father's Name - Abdul Karim": whatever
# follows the label is personal, whatever its format.
LABELLED = re.compile(
    r"(?im)^(?P<label>[^\S\n]*[-*•]?[^\S\n]*(?:"
    r"phone|mobile|cell|tel|telephone|contact(?: no| number)?|whatsapp|e-?mail|address|"
    r"present address|permanent address|date of birth|dob|birth ?date|place of birth|"
    r"nid|national id|passport|father'?s name|mother'?s name|spouse'?s? name|religion|"
    r"marital status|nationality|gender|sex|blood group|height|weight|name"
    r")(?: no\.?| number)?[^\S\n]*[:\-–][^\S\n]*)(?P<value>\S.*)$"
)
# A run of digits with phone-style separators; filtered further by _looks_like_phone.
_PHONE_CANDIDATE = re.compile(r"(?<![\w+.])[+(]?\d[\d \t().\-–]{6,}\d(?![\w.])")
_US_STYLE = re.compile(r"^\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}$")


def _looks_like_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    if not 9 <= len(digits) <= 15:
        return False
    # Start-based rules keep year ranges ("2019 - 2023"), grades ("3.85") and
    # counts out: Bangladeshi numbers start with +, 0 or 880.
    return (
        candidate.startswith(("+", "(", "0")) or digits.startswith("880") or bool(_US_STYLE.match(candidate))
    )


def find_phones(text: str) -> list[str]:
    return [m.group() for m in _PHONE_CANDIDATE.finditer(text) if _looks_like_phone(m.group())]


def mask_generic(text: str) -> str:
    """Mask every email, link, phone number and labelled personal field."""
    text = LABELLED.sub(lambda m: m.group("label") + "[PERSONAL]", text)
    text = EMAIL.sub("[EMAIL]", text)
    text = OBFUSCATED_EMAIL.sub("[EMAIL]", text)
    text = URL.sub("[URL]", text)
    return _PHONE_CANDIDATE.sub(_mask_if_phone, text)


def _mask_if_phone(match: re.Match[str]) -> str:
    return "[PHONE]" if _looks_like_phone(match.group()) else match.group()


def phone_pattern(digits: str) -> re.Pattern[str]:
    """Match a known number in any formatting: the last 9 digits, with any
    separators between them, so +880 1712-345678 also matches 01712345678."""
    core = digits[-9:]
    return re.compile(r"[\s().\-–]*".join(core))


def word_pattern(term: str) -> re.Pattern[str]:
    """Match a term as whole words, ignoring case and line breaks between words."""
    words = [re.escape(w) for w in term.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", re.IGNORECASE)
