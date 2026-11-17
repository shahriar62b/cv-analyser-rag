from dataclasses import dataclass

_HEADINGS = {
    "summary": ("summary", "professional summary", "profile", "objective", "career objective"),
    "experience": (
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
    ),
    "education": ("education", "academic background", "academic qualifications"),
    "skills": ("skills", "technical skills", "key skills", "core competencies"),
    "projects": ("projects", "personal projects", "academic projects"),
    "certifications": ("certifications", "certificates", "licenses", "training"),
    "achievements": ("achievements", "awards", "honors", "honours", "awards and achievements"),
    "languages": ("languages", "language skills"),
    "volunteering": ("volunteer work", "volunteering", "volunteer experience"),
    "interests": ("interests", "hobbies", "hobbies and interests"),
    # Common in South Asian CVs; never needed for matching, and removed before
    # anything is sent to a cloud model (see privacy/).
    "personal": (
        "personal information",
        "personal info",
        "personal details",
        "personal data",
        "personal particulars",
    ),
    "references": ("references", "reference", "referees"),
    "declaration": ("declaration",),
    "contact": ("contact", "contact information", "contact details", "contact info"),
    "activities": (
        "activities",
        "extracurricular activities",
        "extra-curricular activities",
        "co-curricular activities",
    ),
}

_HEADING_LOOKUP = {alias: name for name, aliases in _HEADINGS.items() for alias in aliases}


def heading_name(line: str) -> str | None:
    """The section name if this line is a known heading, else None."""
    return _HEADING_LOOKUP.get(line.strip().rstrip(":").lower())


@dataclass(frozen=True)
class Section:
    name: str
    content: str


def split_into_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    current_name = "header"
    buffer: list[str] = []

    for line in text.splitlines():
        heading = heading_name(line)
        if heading:
            _flush(sections, current_name, buffer)
            current_name, buffer = heading, []
        else:
            buffer.append(line)

    _flush(sections, current_name, buffer)
    return sections


def _flush(sections: list[Section], name: str, buffer: list[str]) -> None:
    content = "\n".join(buffer).strip()
    if content:
        sections.append(Section(name=name, content=content))
