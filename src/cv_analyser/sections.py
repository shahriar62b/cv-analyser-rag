from dataclasses import dataclass

_HEADINGS = {
    "summary": ("summary", "professional summary", "profile", "objective", "career objective"),
    "experience": ("experience", "work experience", "professional experience", "employment history", "work history"),
    "education": ("education", "academic background", "academic qualifications"),
    "skills": ("skills", "technical skills", "key skills", "core competencies"),
    "projects": ("projects", "personal projects", "academic projects"),
    "certifications": ("certifications", "certificates", "licenses", "training"),
}

_HEADING_LOOKUP = {
    alias: name for name, aliases in _HEADINGS.items() for alias in aliases
}


@dataclass(frozen=True)
class Section:
    name: str
    content: str


def split_into_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    current_name = "header"
    buffer: list[str] = []

    for line in text.splitlines():
        heading = _HEADING_LOOKUP.get(line.strip().rstrip(":").lower())
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