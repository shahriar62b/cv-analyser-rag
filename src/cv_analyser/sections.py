from dataclasses import dataclass

_HEADINGS = {
    # Aliases marked "survey" were found in a scan of 2,484 real resumes
    # (LiveCareer, Kaggle), counted by how many files used them.
    "summary": (
        "summary",
        "professional summary",
        "profile",
        "objective",
        "career objective",
        "executive profile",  # survey: 131
        "career overview",  # survey: 85
        "professional profile",  # survey: 51
        "career focus",  # survey: 48
        "executive summary",  # survey: 36
        "professional overview",  # survey: 16
    ),
    "experience": (
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "relevant experience",  # survey: 29
        "military experience",  # survey: 27
    ),
    "education": (
        "education",
        "academic background",
        "academic qualifications",
        "educational background",
        "educational qualifications",
        "education and training",  # survey: 508
    ),
    "skills": (
        "skills",
        "technical skills",
        "key skills",
        "core competencies",
        "highlights",  # survey: 875, usually a skills list
        "core qualifications",  # survey: 211
        "skill highlights",  # survey: 187
        "qualifications",  # survey: 102
        "core strengths",  # survey: 36
        "computer skills",  # survey: 27
        "areas of expertise",
        "additional skills",  # survey: 6
        "summary of skills",  # survey: 34
        "summary of qualifications",  # survey: 10
        "technical skills and qualifications",  # survey: 10
    ),
    "projects": ("projects", "personal projects", "academic projects"),
    "certifications": (
        "certifications",
        "certification",  # survey: 6
        "certificates",
        "licenses",
        "training",
        "professional development",  # survey: 9
    ),
    "achievements": (
        "achievements",
        "awards",
        "honors",
        "honours",
        "awards and achievements",
        "accomplishments",  # survey: 767
        "core accomplishments",  # survey: 94
        "publications",  # survey: 43
        "presentations",  # survey: 43
        "activities and honors",  # survey: 58
    ),
    "languages": ("languages", "language skills"),
    "volunteering": (
        "volunteer work",
        "volunteering",
        "volunteer experience",
        "volunteer",
        "community service",
    ),
    "affiliations": ("affiliations", "professional affiliations", "memberships"),  # survey: 158, 211
    # Usually honours, awards or extra skills; sometimes personal details, which
    # the privacy masking still removes.
    "additional": ("additional information", "additional details"),  # survey: 462
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
