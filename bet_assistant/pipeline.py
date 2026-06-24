"""End-to-end pipeline: data -> model -> vig removal -> value -> staking.

Produces fully-rendered, uncertainty-carrying recommendations. Every model is
gated: if it has not passed backtest validation (per ``Config.models``), it is
not allowed to flag bets here, regardless of what edge it computes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import Config, DEFAULT_CONFIG
from .data.schema import Match, Sport
from .language import assert_uncertain, confidence_phrase
from .models.base import InsufficientModelData, ModelOutput
from .odds.kelly import recommended_stake
from .odds.value import ValueAssessment, assess_value


@dataclass
class Recommendation:
    market: str
    selection: str
    model_prob: float
    vig_free_prob: float
    decimal_odds: float
    edge: float
    expected_value: float
    confidence_low: float
    confidence_high: float
    stake: float
    stake_fraction: float
    reasoning: str

    def render(self) -> str:
        text = (
            f"VALUE: {self.market} / {self.selection} @ {self.decimal_odds:.2f}\n"
            f"  model {self.model_prob:.1%} vs market {self.vig_free_prob:.1%} "
            f"(edge +{self.edge*100:.1f} pts, EV +{self.expected_value*100:.1f}%)\n"
            f"  probability band: {self.confidence_low:.1%}–{self.confidence_high:.1%}\n"
            f"  suggested stake: {self.stake:.2f} "
            f"({self.stake_fraction*100:.2f}% of bankroll)\n"
            f"  why: {self.reasoning}"
        )
        # Final gate: no certainty language ever reaches the user.
        return assert_uncertain(text)


@dataclass
class NoBet:
    market: str
    selection: str
    reason: str

    def render(self) -> str:
        return assert_uncertain(
            f"NO BET: {self.market} / {self.selection} — {self.reason}"
        )


def _enabled(config: Config, model_key: str) -> bool:
    return getattr(config.models, model_key, False)


def evaluate_market(
    match: Match,
    model_output: ModelOutput,
    market: str,
    decimal_odds: dict[str, float],
    bankroll: float,
    model_key: str,
    config: Config = DEFAULT_CONFIG,
) -> list:
    """Evaluate one market for one match, returning Recommendations / NoBets."""
    # Gate: unvalidated models may not flag bets.
    if not _enabled(config, model_key):
        return [
            NoBet(
                market,
                sel,
                f"Model '{model_key}' is not validated/enabled — no bet. "
                "Run the backtest harness and enable it only if it beats the "
                "vig-free closing line.",
            )
            for sel in decimal_odds
        ]

    assessments = assess_value(
        market=market,
        decimal_odds=decimal_odds,
        model_probs=model_output.probabilities,
        config=config.value,
        effective_samples=model_output.effective_samples,
        missing_fields=match.quality.missing_fields,
    )

    out: list = []
    for a in assessments:
        if not a.is_value:
            out.append(NoBet(a.market, a.selection, a.reason))
            continue
        lo, hi = model_output.band_for(a.selection)
        stake_rec = recommended_stake(
            a.model_prob, a.decimal_odds, bankroll, config.staking
        )
        if stake_rec.stake <= 0:
            out.append(NoBet(a.market, a.selection, stake_rec.note))
            continue
        conf = confidence_phrase(model_output.confidence_half_width)
        reasoning = f"{a.reason} {conf.capitalize()}."
        if model_output.notes:
            reasoning += " " + " ".join(model_output.notes) + "."
        out.append(
            Recommendation(
                market=a.market,
                selection=a.selection,
                model_prob=a.model_prob,
                vig_free_prob=a.vig_free_prob,
                decimal_odds=a.decimal_odds,
                edge=a.edge,
                expected_value=a.expected_value,
                confidence_low=lo,
                confidence_high=hi,
                stake=stake_rec.stake,
                stake_fraction=stake_rec.fraction_of_bankroll,
                reasoning=reasoning,
            )
        )
    return out
