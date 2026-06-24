"""Certainty-language guard.

A hard requirement of this project: outputs must never claim certainty. This
module screens any user-facing string for forbidden words and provides helpers
to build correctly-hedged reasoning text.

The guard is used in tests against every rendered output, so false certainty
cannot silently reach a user.
"""

from __future__ import annotations

import re

# Words/phrases that imply certainty about a future, uncertain outcome.
# Matched case-insensitively on word boundaries.
FORBIDDEN_TERMS = (
    "guaranteed",
    "guarantee",
    "sure thing",
    "sure bet",
    "surefire",
    "definitely",
    "certain win",
    "certainty",
    "lock",
    "locks",
    "can't lose",
    "cannot lose",
    "risk-free",
    "risk free",
    "no risk",
    "easy money",
    "free money",
    "100%",
)


class CertaintyLanguageError(ValueError):
    """Raised when user-facing text claims certainty."""


def _pattern() -> re.Pattern:
    parts = []
    for term in FORBIDDEN_TERMS:
        # Escape, and require boundaries around alphanumerics; "100%" handled too.
        escaped = re.escape(term)
        parts.append(rf"\b{escaped}\b" if term[-1].isalnum() else escaped)
    return re.compile("|".join(parts), re.IGNORECASE)


_FORBIDDEN_RE = _pattern()


def find_forbidden(text: str) -> list[str]:
    """Return the forbidden terms found in ``text`` (empty list if clean)."""
    return [m.group(0) for m in _FORBIDDEN_RE.finditer(text or "")]


def assert_uncertain(text: str) -> str:
    """Return ``text`` unchanged if clean; otherwise raise.

    Use this as the final gate on anything shown to a user.
    """
    hits = find_forbidden(text)
    if hits:
        raise CertaintyLanguageError(
            f"Output claimed certainty via forbidden term(s): {sorted(set(hits))}. "
            "All outputs must carry explicit uncertainty."
        )
    return text


def confidence_phrase(half_width: float) -> str:
    """Map a probability confidence half-width to a plain-language hedge."""
    if half_width >= 0.12:
        return "low confidence (wide uncertainty band)"
    if half_width >= 0.06:
        return "moderate confidence"
    return "relatively high confidence (though no outcome is ever assured)"
