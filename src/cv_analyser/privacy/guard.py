"""The last check before a request leaves the machine.

Masking can miss things; this guard cannot be skipped. Every request to a cloud
model goes through `PrivateLLMClient`, which:
1. masks generic personal data in the outgoing text (this also covers a
   recruiter's email or phone in the job ad, which is harmless to remove), then
2. searches the exact outgoing text for anything known about the candidate, and
3. blocks the request if any of it is found ("fail closed"): it goes to the
   local fallback model instead, or raises PrivacyLeakError if there is none.
"""

import re

from cv_analyser.exceptions import PrivacyLeakError
from cv_analyser.llm.base import LLMClient, T
from cv_analyser.privacy.patterns import mask_generic, normalise, phone_pattern, word_pattern
from cv_analyser.privacy.redactor import Identity, redact_cv


class OutboundGuard:
    def __init__(self, identity: Identity) -> None:
        checks: list[tuple[str, re.Pattern[str]]] = []
        for name in (*identity.names, *identity.name_parts):
            checks.append(("name", word_pattern(name)))
        for email in identity.emails:
            checks.append(("email", re.compile(re.escape(email), re.IGNORECASE)))
        for digits in identity.phone_digits:
            checks.append(("phone", phone_pattern(digits)))
        for term in identity.extra_terms:
            checks.append(("extra term", word_pattern(term)))
        self._checks = checks

    def scrub(self, text: str) -> str:
        return mask_generic(normalise(text))

    def leaks(self, text: str) -> list[str]:
        """Kinds of known personal data found in `text`; never the values
        themselves, so the result is safe to log or show."""
        text = normalise(text)
        return sorted({kind for kind, pattern in self._checks if pattern.search(text)})


class PrivateLLMClient:
    """Wraps any LLM client so that no request carries the candidate's known
    personal data. The pipeline uses it like any other `LLMClient`; this is the
    decorator pattern: same interface, extra behaviour around each call."""

    def __init__(self, inner: LLMClient, guard: OutboundGuard, fallback: LLMClient | None = None) -> None:
        self._inner = inner
        self._guard = guard
        self._fallback = fallback
        self.blocked_requests = 0
        # Every request that actually left the machine, kept in memory only, so
        # the user can see exactly what a cloud model received.
        self.sent: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return self._inner.model

    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        system, user = self._guard.scrub(system), self._guard.scrub(user)
        found = self._guard.leaks(system + "\n" + user)
        if not found:
            self.sent.append((system, user))
            return self._inner.generate_structured(system, user, schema)
        self.blocked_requests += 1
        if self._fallback is None:
            raise PrivacyLeakError(
                f"Blocked a request that still contained personal data ({', '.join(found)})"
            )
        return self._fallback.generate_structured(system, user, schema)


def protect_for_cloud(
    cv_text: str,
    llm: LLMClient,
    fallback: LLMClient | None,
    extra_terms: tuple[str, ...] = (),
) -> tuple[str, PrivateLLMClient]:
    """Mask the CV and wrap the client, the two steps every cloud run needs.

    Returns the masked CV text (analyse that, not the original) and the guarded client.
    """
    redacted = redact_cv(cv_text, extra_terms)
    return redacted.text, PrivateLLMClient(llm, OutboundGuard(redacted.identity), fallback)
