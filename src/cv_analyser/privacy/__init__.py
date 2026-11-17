"""Keep the candidate's personal data on this machine when a cloud model is used.

Typical use:

    redacted = redact_cv(cv_text)
    llm = PrivateLLMClient(cloud_client, OutboundGuard(redacted.identity), fallback=local_client)
    report = analyse_cv(redacted.text, job_description, llm, embedder)
"""

from cv_analyser.privacy.guard import OutboundGuard, PrivateLLMClient, protect_for_cloud
from cv_analyser.privacy.redactor import Identity, RedactedCV, redact_cv

__all__ = ["Identity", "OutboundGuard", "PrivateLLMClient", "RedactedCV", "protect_for_cloud", "redact_cv"]
