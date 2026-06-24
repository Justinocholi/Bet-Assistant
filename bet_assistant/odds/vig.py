"""Convert bookmaker odds to probabilities and strip the vig (overround).

The bookmaker's quoted decimal odds imply probabilities that sum to more than 1;
the excess is the margin ("vig"/"overround"). To compare fairly against a model,
we normalise the implied probabilities so they sum to 1.

We use simple proportional (multiplicative) normalisation, which is transparent
and standard. (Shin's method and others exist; proportional is the right
baseline and is easy to validate — don't over-engineer before validation.)
"""

from __future__ import annotations


def implied_probabilities(decimal_odds: dict[str, float]) -> dict[str, float]:
    """Raw implied probabilities (1/odds). These sum to > 1 by the overround."""
    if not decimal_odds:
        raise ValueError("no odds provided")
    for sel, o in decimal_odds.items():
        if o <= 1.0:
            raise ValueError(f"decimal odds for {sel!r} must be > 1.0, got {o}")
    return {sel: 1.0 / o for sel, o in decimal_odds.items()}


def overround(decimal_odds: dict[str, float]) -> float:
    """The bookmaker margin as the sum of raw implied probabilities (>= 1)."""
    return sum(implied_probabilities(decimal_odds).values())


def remove_vig(decimal_odds: dict[str, float]) -> dict[str, float]:
    """Return vig-free probabilities that sum to exactly 1.0.

    Proportional normalisation: each selection's share of the total implied
    probability.
    """
    raw = implied_probabilities(decimal_odds)
    z = sum(raw.values())
    return {sel: p / z for sel, p in raw.items()}
