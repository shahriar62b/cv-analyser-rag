"""Privacy tests: fictional CVs built to break the masker.

Every CV below plants personal data in a different awkward place or format.
The tests check that none of it survives redaction, that the facts a job match
needs do survive, and that the outbound guard blocks anything that slips through.
All people, numbers and addresses here are made up.
"""

import pytest
from docx import Document

from cv_analyser.exceptions import PrivacyLeakError
from cv_analyser.loaders.dispatcher import load_document
from cv_analyser.privacy import Identity, OutboundGuard, PrivateLLMClient, redact_cv
from cv_analyser.report import CVSummary

# A typical Bangladeshi CV: contact header, personal-information section and references.
CV_STANDARD = """Rahim Uddin Chowdhury
Data Analyst
House 12, Road 5, Dhanmondi, Dhaka 1205
Mobile: +880 1712-345678 | Email: rahim.chowdhury@example.com
linkedin.com/in/rahim-chowdhury-bd

Career Objective
To work as a data analyst where I can use SQL and Python to improve business decisions.

Work Experience
Junior Data Analyst, Example Traders Ltd (Jan 2023 - present)
- Built weekly sales dashboards in Power BI for 3 regional managers.
- Rahim automated the stock report in Python, saving 5 hours a week.

Education
BSc in EEE, Example University of Engineering (2018 - 2022), CGPA 3.45

Personal Information
Father's Name: Abdul Karim Chowdhury
Mother's Name: Rokeya Begum
Date of Birth: 14 March 1999
Religion: Islam
NID: 1995 2612 3456 789
Permanent Address: Village Kalia, Upazila Sadar, Narail

References
Dr. Selina Haque, Professor, Example University. Phone: 01819-876543, selina.haque@example.edu
"""

# Name in capitals over two lines, disguised email, odd phone formats and
# invisible characters inside the email.
CV_AWKWARD = """NUSRAT
JAHAN
nusrat.jahan​@example.com
Cell (017) 11 222 333  ·  WhatsApp: 01911 223344
nusrat [at] example [dot] org
github.com/nusratj

SKILLS
Excel, SQL, Tableau

EXPERIENCE
Operations Intern, Example Shop (2024 - 2025): updated 800 product listings a day.
Nusrat trained 4 new joiners on the order system.
"""

# A name in Bangla script and a labelled name field.
CV_BANGLA = """Name: তানভীর আহমেদ
Phone: 01555-666777

Summary
Software engineer with 2 years of Java and Spring Boot experience.

Experience
Software Engineer, Example Soft (2023 - present)
তানভীর led the migration to Spring Boot 3.
"""

# No headings at all: the body must survive even though the header is dropped.
CV_NO_HEADINGS = """Karim Hossain
karim.h@example.com
+8801612345678
Dhaka
I have 4 years of experience as an accountant, preparing VAT returns and monthly reports.
Prepared VAT returns for 30 clients every month using Tally and Excel.
"""

PLANTED = {
    "standard": (
        CV_STANDARD,
        [
            "Rahim",
            "Chowdhury",
            "1712",
            "345678",
            "rahim.chowdhury@example.com",
            "linkedin.com/in/rahim",
            "Dhanmondi",
            "Abdul Karim",
            "Rokeya",
            "14 March 1999",
            "Islam",
            "2612 3456",
            "Narail",
            "Selina",
            "876543",
            "selina.haque",
        ],
    ),
    "awkward": (
        CV_AWKWARD,
        ["NUSRAT", "Nusrat", "JAHAN", "example.com", "11 222 333", "223344", "[at] example", "nusratj"],
    ),
    "bangla": (CV_BANGLA, ["তানভীর", "আহমেদ", "666777"]),
    "no_headings": (CV_NO_HEADINGS, ["Karim", "Hossain", "karim.h", "1612345678"]),
}

# Facts a job match needs; masking must not destroy them.
KEPT = {
    "standard": [
        "SQL",
        "Power BI",
        "Jan 2023 - present",
        "5 hours a week",
        "BSc in EEE",
        "2018 - 2022",
        "3.45",
    ],
    "awkward": ["Excel, SQL, Tableau", "2024 - 2025", "800 product listings", "4 new joiners"],
    "bangla": ["2 years of Java", "2023 - present"],
    "no_headings": ["4 years of experience as an accountant", "30 clients", "Tally"],
}


@pytest.mark.parametrize("case", sorted(PLANTED))
def test_no_planted_personal_data_survives(case):
    text, planted = PLANTED[case]

    redacted = redact_cv(text).text

    leaked = [value for value in planted if value.lower() in redacted.lower()]
    assert leaked == []


@pytest.mark.parametrize("case", sorted(KEPT))
def test_facts_needed_for_matching_are_kept(case):
    redacted = redact_cv(PLANTED[case][0]).text

    missing = [fact for fact in KEPT[case] if fact not in redacted]
    assert missing == []


def test_identity_is_found_from_the_header():
    identity = redact_cv(CV_STANDARD).identity

    assert identity.names == ("Rahim Uddin Chowdhury",)
    assert "rahim.chowdhury@example.com" in identity.emails
    assert "8801712345678" in identity.phone_digits


def test_bangla_name_is_found_and_masked_in_the_body():
    redacted = redact_cv(CV_BANGLA)

    assert redacted.identity.names == ("তানভীর আহমেদ",)
    assert "[NAME] led the migration" in redacted.text


def test_name_split_over_two_lines_is_found():
    assert redact_cv(CV_AWKWARD).identity.names == ("NUSRAT JAHAN",)


def test_later_mentions_of_the_name_become_placeholders():
    redacted = redact_cv(CV_STANDARD).text

    assert "[NAME] automated the stock report" in redacted


def test_additional_information_is_kept_but_still_masked():
    text = """Nadia Karim
nadia.karim@example.com

Interests
Chess

Additional Information
Dean's List 2019. References: nadia.karim@example.com
"""

    redacted = redact_cv(text).text

    assert "Dean's List 2019" in redacted
    assert "Chess" not in redacted
    assert "nadia.karim@example.com" not in redacted


def test_extra_terms_are_masked():
    redacted = redact_cv(CV_STANDARD, extra_terms=("Example Traders",)).text

    assert "Example Traders" not in redacted
    assert "[REDACTED]" in redacted


def test_contact_details_in_a_docx_table_are_masked(tmp_path):
    document = Document()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Farhana Akter"
    table.cell(0, 1).text = "farhana.akter@example.com"
    table.cell(1, 0).text = "Phone"
    table.cell(1, 1).text = "+880 1300-112233"
    document.add_paragraph("Skills")
    document.add_paragraph("Python, SQL")
    path = tmp_path / "cv.docx"
    document.save(str(path))

    redacted = redact_cv(load_document(path)).text

    for value in ("Farhana", "Akter", "farhana.akter", "112233"):
        assert value.lower() not in redacted.lower()
    assert "Python, SQL" in redacted


# --- The outbound guard ---------------------------------------------------

IDENTITY = Identity(
    names=("Rahim Uddin Chowdhury",),
    emails=("rahim.chowdhury@example.com",),
    phone_digits=("8801712345678",),
)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Candidate RAHIM led the team", "name"),
        ("contact rahim.chowdhury@example.com", "email"),
        ("call 017 1234 5678", "phone"),  # same number, different formatting
        ("Chowdhury\nUddin", "name"),
    ],
)
def test_guard_finds_known_values_in_any_form(text, kind):
    assert kind in OutboundGuard(IDENTITY).leaks(text)


def test_guard_ignores_unrelated_text():
    assert OutboundGuard(IDENTITY).leaks("Built dashboards in Power BI (2023 - 2024)") == []


class _RecordingLLM:
    model = "cloud"

    def __init__(self):
        self.sent: list[str] = []

    def generate_structured(self, system, user, schema):
        self.sent.append(user)
        return schema(strengths=[], gaps=[], suggestions=[], verdict="ok")


def test_private_client_blocks_a_leak_and_uses_the_local_fallback():
    cloud, local = _RecordingLLM(), _RecordingLLM()
    client = PrivateLLMClient(cloud, OutboundGuard(IDENTITY), fallback=local)

    client.generate_structured("system", "Rahim built dashboards", CVSummary)

    assert cloud.sent == []
    assert len(local.sent) == 1
    assert client.blocked_requests == 1


def test_private_client_without_fallback_refuses_to_send():
    client = PrivateLLMClient(_RecordingLLM(), OutboundGuard(IDENTITY))

    with pytest.raises(PrivacyLeakError, match="name") as error:
        client.generate_structured("system", "Rahim built dashboards", CVSummary)
    assert "Rahim" not in str(error.value)  # the error never repeats the data


def test_private_client_masks_recruiter_contacts_in_the_job_ad_and_sends():
    cloud = _RecordingLLM()
    client = PrivateLLMClient(cloud, OutboundGuard(IDENTITY))

    client.generate_structured("system", "Apply to hr@company.example or call 01700-000000", CVSummary)

    assert cloud.sent == ["Apply to [EMAIL] or call [PHONE]"]
    assert client.model == "cloud"


# --- Only the sections a job match can use leave the machine ---------------

CV_MANY_SECTIONS = """Ayesha Siddiqua

Contact Information
Flat 4B, Example Tower, Chattogram

Summary
Marketing executive with 3 years of digital campaign experience.

Experience
Digital Marketing Executive, Example Foods (2023 - present)
- Ran Facebook and Google ad campaigns with a monthly budget of BDT 5 lakh.

Languages
Bangla (native), English (fluent)

Extracurricular Activities
Led the university debate club and organised 2 national tournaments.

Interests
Quran recitation, volunteering at the local mosque, cricket

Personal Information
Marital Status: Married
"""


def test_only_sections_useful_for_matching_are_kept():
    redacted = redact_cv(CV_MANY_SECTIONS).text

    for kept in ("3 years of digital campaign", "BDT 5 lakh", "English (fluent)", "debate club"):
        assert kept in redacted
    for dropped in ("Chattogram", "Example Tower", "Quran", "mosque", "cricket", "Married"):
        assert dropped not in redacted


def test_every_kept_section_is_on_the_allowlist():
    from cv_analyser.privacy.redactor import KEEP_SECTIONS
    from cv_analyser.sections import split_into_sections

    for case in PLANTED.values():
        names = {section.name for section in split_into_sections(redact_cv(case[0]).text)}
        assert names <= KEEP_SECTIONS | {"header"}


def test_private_client_records_only_what_was_actually_sent():
    cloud, local = _RecordingLLM(), _RecordingLLM()
    client = PrivateLLMClient(cloud, OutboundGuard(IDENTITY), fallback=local)

    client.generate_structured("system", "Built dashboards", CVSummary)
    client.generate_structured("system", "Rahim built dashboards", CVSummary)  # blocked

    assert client.sent == [("system", "Built dashboards")]
