"""Value detection: compare model probability to the vig-free implied probability.

A bet is flagged as value only when the model's probability exceeds the
vig-free implied probability by a configurable margin *and* the expected value
clears a floor. "No bet" is the default and most common outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import ValueConfig
from .vig import remove_vig


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """EV per unit staked: p*(odds-1) - (1-p) = p*odds - 1."""
    return model_prob * decimal_odds - 1.0


@dataclass
class ValueAssessment:
    market: str
    selection: str
    model_prob: float
    vig_free_prob: float
    decimal_odds: float
    edge: float  # model_prob - vig_free_prob (probability terms)
    expected_value: float  # per unit staked
    is_value: bool
    reason: str
    # If we couldn't assess, why (e.g. insufficient data). None when assessed.
    no_bet_reason: Optional[str] = None


def assess_value(
    market: str,
    decimal_odds: dict[str, float],
    model_probs: dict[str, float],
    config: ValueConfig,
    effective_samples: int,
    missing_fields: Optional[list[str]] = None,
) -> list[ValueAssessment]:
    """Assess every selection in a market. Returns one assessment per selection.

    The model is responsible for providing calibrated ``model_probs`` (summing
    to ~1 across the market's selections). We compare each against its vig-free
    implied probability.
    """
    missing_fields = missing_fields or []

    # Insufficient-data gate: refuse to flag anything when the inputs are thin.
    if effective_samples < config.min_effective_samples:
        reason = (
            f"Insufficient data: only {effective_samples} effective samples "
            f"(need >= {config.min_effective_samples}). No bet."
        )
        return [
            ValueAssessment(
                market=market,
                selection=sel,
                model_prob=model_probs.get(sel, float("nan")),
                vig_free_prob=float("nan"),
                decimal_odds=odds,
                edge=float("nan"),
                expected_value=float("nan"),
                is_value=False,
                reason=reason,
                no_bet_reason=reason,
            )
            for sel, odds in decimal_odds.items()
        ]

    vig_free = remove_vig(decimal_odds)
    out: list[ValueAssessment] = []
    for sel, odds in decimal_odds.items():
        mp = model_probs.get(sel)
        if mp is None:
            out.append(
                ValueAssessment(
                    market, sel, float("nan"), vig_free[sel], odds,
                    float("nan"), float("nan"), False,
                    reason=f"Model produced no probability for {sel!r}; no bet.",
                    no_bet_reason="model_probability_missing",
                )
            )
            continue

        edge = mp - vig_free[sel]
        ev = expected_value(mp, odds)
        is_value = (
            edge >= config.min_edge_margin
            and ev >= config.min_expected_value
        )

        if is_value:
            reason = (
                f"Model gives {mp:.1%} vs vig-free market {vig_free[sel]:.1%} "
                f"(+{edge*100:.1f} pts), implying {ev*100:.1f}% expected value at "
                f"odds {odds:.2f}."
            )
            if missing_fields:
                reason += (
                    f" Note: some inputs were missing ({', '.join(missing_fields)}), "
                    "so treat the edge with extra caution."
                )
        else:
            if edge < config.min_edge_margin:
                reason = (
                    f"Edge {edge*100:+.1f} pts is below the {config.min_edge_margin*100:.0f}-pt "
                    f"threshold (model {mp:.1%} vs market {vig_free[sel]:.1%}). No bet."
                )
            else:
                reason = (
                    f"Expected value {ev*100:+.1f}% is below the "
                    f"{config.min_expected_value*100:.0f}% floor. No bet."
                )

        out.append(
            ValueAssessment(
                market=market,
                selection=sel,
                model_prob=mp,
                vig_free_prob=vig_free[sel],
                decimal_odds=odds,
                edge=edge,
                expected_value=ev,
                is_value=is_value,
                reason=reason,
            )
        )
    return out
