"""Remove the candidate's personal data from CV text before it is analysed.

Four layers, each covering gaps in the others:
1. Find who the person is (name, emails, phones) while the CV is still intact.
2. Keep only the sections a job match can use (`KEEP_SECTIONS`); drop the
   contact header and every other section, such as personal information,
   references and interests.
3. Mask anything with a personal-data shape: emails, links, phones, labelled
   fields like "Date of Birth: ...".
4. Mask every remaining mention of the known name, in any case or line break.

The result keeps what a job match needs (skills, experience, dates, education)
and the `Identity`, so the outbound guard can check that none of it leaks.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from cv_analyser.privacy.patterns import (
    EMAIL,
    LABELLED,
    OBFUSCATED_EMAIL,
    find_phones,
    mask_generic,
    normalise,
    word_pattern,
)
from cv_analyser.sections import heading_name

# Sections that can hold evidence for a job requirement. Everything else is
# dropped: personal information, references, declaration and contact details
# (only personal data), interests and hobbies (rarely evidence, and where
# religion, politics or health tend to appear), and any section type added
# later. An allowlist fails safe: a new kind of section is left out until
# someone decides it is needed.
KEEP_SECTIONS = {
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "achievements",
    "languages",
    "volunteering",
    "activities",
    # Kept by the author's decision (2026-11-18): they often hold honours and
    # memberships a job can ask for, and masking still runs on them.
    "affiliations",
    "additional",
}
# Words that open a CV but are not a name.
_NOT_A_NAME = {"curriculum vitae", "resume", "résumé", "cv", "bio data", "biodata", "bio-data"}
# Header lines up to this many words are contact details or a title; longer
# ones are prose (an unlabelled summary) and are kept.
_SHORT_LINE_WORDS = 12
# Name parts shorter than this are initials or titles ("Md", "Dr") and are not
# masked on their own, to avoid masking ordinary text.
_MIN_NAME_PART = 3
# A contact header longer than this is unusual; past it, stop treating lines as header.
_MAX_HEADER_LINES = 15


@dataclass(frozen=True)
class Identity:
    """What is known about who the CV belongs to. Never sent anywhere."""

    names: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()
    phone_digits: tuple[str, ...] = ()
    extra_terms: tuple[str, ...] = ()

    @property
    def name_parts(self) -> tuple[str, ...]:
        parts = {part.strip(".") for name in self.names for part in name.split()}
        return tuple(sorted(p for p in parts if len(p) >= _MIN_NAME_PART))


@dataclass(frozen=True)
class RedactedCV:
    text: str
    identity: Identity
    removed_lines: list[str] = field(default_factory=list)


def redact_cv(text: str, extra_terms: tuple[str, ...] = ()) -> RedactedCV:
    """Return the CV with personal data removed, plus what was found.

    `extra_terms` are words the automatic rules cannot know about, such as a
    small hometown or a rare employer name; they are masked too.
    """
    text = normalise(text)
    identity = find_identity(text, extra_terms)
    kept, removed = _drop_personal_parts(text.splitlines())
    masked = mask_known(mask_generic("\n".join(kept)), identity)
    return RedactedCV(text=masked, identity=identity, removed_lines=removed)


def find_identity(text: str, extra_terms: tuple[str, ...] = ()) -> Identity:
    emails = {m.group() for m in EMAIL.finditer(text)} | {m.group() for m in OBFUSCATED_EMAIL.finditer(text)}
    phones = {re.sub(r"\D", "", p) for p in find_phones(text)}
    name = _labelled_name(text) or _name_from_top_lines(text)
    return Identity(
        names=(name,) if name else (),
        emails=tuple(sorted(emails)),
        phone_digits=tuple(sorted(phones)),
        extra_terms=tuple(t for t in extra_terms if t.strip()),
    )


def mask_known(text: str, identity: Identity) -> str:
    """Mask the person's name, name parts and extra terms wherever they appear."""
    # Longest first, so the full name becomes one [NAME] rather than several.
    for term in sorted((*identity.names, *identity.name_parts), key=len, reverse=True):
        text = word_pattern(term).sub("[NAME]", text)
    for term in identity.extra_terms:
        text = word_pattern(term).sub("[REDACTED]", text)
    return text


def _labelled_name(text: str) -> str | None:
    for match in LABELLED.finditer(text):
        label = match.group("label").strip(" \t-*•:–").lower()
        if label == "name" and _looks_like_name(match.group("value")):
            return match.group("value").strip()
    return None


def _name_from_top_lines(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()][:4]
    for i, line in enumerate(lines):
        if _looks_like_name(line):
            return line
        # A name broken over two lines by the layout: "RAHIM" / "UDDIN".
        if i + 1 < len(lines) and _is_single_name_word(line) and _is_single_name_word(lines[i + 1]):
            return f"{line} {lines[i + 1]}"
    return None


def _is_name_word(word: str) -> bool:
    """Letters only, plus . ' - inside. Letters include combining marks (Unicode
    category M), because Bangla and other Indic scripts write vowels as marks:
    in "তানভীর" the "া" and "ী" are marks, not letters."""
    core = word.strip(".'-")
    return (
        bool(core)
        and all(unicodedata.category(ch)[0] in "LM" or ch in ".'-" for ch in core)
        and unicodedata.category(core[0])[0] == "L"
    )


def _looks_like_name(line: str) -> bool:
    words = line.strip().split()
    return (
        2 <= len(words) <= 5
        and all(_is_name_word(w) for w in words)
        and line.strip().lower() not in _NOT_A_NAME
        and heading_name(line) is None
    )


def _is_single_name_word(line: str) -> bool:
    return (
        len(line.split()) == 1
        and _is_name_word(line)
        and line.lower() not in _NOT_A_NAME
        and heading_name(line) is None
    )


def _drop_personal_parts(lines: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    header_end = _header_end(lines)
    dropping = False
    for index, line in enumerate(lines):
        heading = heading_name(line)
        if heading is not None:
            dropping = heading not in KEEP_SECTIONS
        in_header = index < header_end and heading is None
        if dropping or (in_header and len(line.split()) <= _SHORT_LINE_WORDS):
            if line.strip():
                removed.append(line)
            continue
        kept.append(line)
    return kept, removed


def _header_end(lines: list[str]) -> int:
    """Index of the first section heading: everything above it is the contact
    header. If there is no heading near the top, assume the first four
    non-empty lines are the header, so a CV without known headings keeps its body."""
    for index, line in enumerate(lines[:_MAX_HEADER_LINES]):
        if heading_name(line) is not None:
            return index
    non_empty = [i for i, line in enumerate(lines) if line.strip()]
    return non_empty[3] + 1 if len(non_empty) > 3 else len(lines)
