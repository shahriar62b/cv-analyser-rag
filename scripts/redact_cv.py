"""Show exactly what of a CV could be sent to a cloud model, after masking.

Usage:
    python scripts/redact_cv.py path/to/cv.pdf
    python scripts/redact_cv.py path/to/cv.docx --mask "Narail" --mask "Example Traders"

Nothing is sent anywhere: this only prints to the screen, so you can check a CV
before using a cloud model with it. If you still see anything personal, mask it
with --mask and tell the maintainer, so the rules can be improved.
"""

import argparse
import sys
from pathlib import Path

from cv_analyser.console import use_utf8_output
from cv_analyser.exceptions import CVAnalyserError
from cv_analyser.loaders.dispatcher import load_document
from cv_analyser.privacy import OutboundGuard, redact_cv


def main() -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cv", type=Path, help="CV file (PDF or DOCX)")
    parser.add_argument(
        "--mask", action="append", default=[], help="Extra word or phrase to mask (can be repeated)"
    )
    args = parser.parse_args()

    try:
        text = load_document(args.cv)
    except CVAnalyserError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    redacted = redact_cv(text, extra_terms=tuple(args.mask))
    identity = redacted.identity
    print("Found (kept on this machine):")
    print(f"  name:   {', '.join(identity.names) or 'NOT FOUND - add it with --mask'}")
    print(f"  emails: {len(identity.emails)}   phones: {len(identity.phone_digits)}")
    print(f"  removed {len(redacted.removed_lines)} header/personal/reference lines\n")
    print("=" * 30 + " text that may be sent " + "=" * 30)
    print(redacted.text)
    print("=" * 83)

    leaks = OutboundGuard(identity).leaks(redacted.text)
    if leaks:
        print(f"\nWARNING: still contains known personal data ({', '.join(leaks)}). It would be blocked.")
        return 2
    print("\nGuard check passed: none of the known personal data is in this text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
