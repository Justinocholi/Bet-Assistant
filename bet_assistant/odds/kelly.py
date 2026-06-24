"""Fractional Kelly staking, hard-capped at a small fraction of bankroll.

Full Kelly maximises long-run growth but is famously volatile and assumes your
probability estimate is exactly right. Since our probabilities are *estimates*,
we use a fraction of Kelly (default quarter) and cap the stake. This trades a
little growth for a large reduction in drawdown — the right call for an
estimator that can be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import StakingConfig


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Full-Kelly fraction of bankroll for a single binary bet.

    f* = (b*p - q) / b, where b = odds - 1, q = 1 - p. Negative means no bet.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    f = (b * model_prob - q) / b
    return max(0.0, f)


@dataclass
class StakeRecommendation:
    fraction_of_bankroll: float
    stake: float
    full_kelly_fraction: float
    capped: bool
    note: str


def recommended_stake(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    config: StakingConfig,
) -> StakeRecommendation:
    """Compute a conservative, capped fractional-Kelly stake."""
    full = kelly_fraction(model_prob, decimal_odds)
    fractional = full * config.kelly_fraction

    capped = False
    note = f"{int(config.kelly_fraction * 100)}% Kelly"
    if fractional > config.max_stake_fraction:
        fractional = config.max_stake_fraction
        capped = True
        note += f", capped at {config.max_stake_fraction*100:.1f}% of bankroll"

    if fractional < config.min_stake_fraction:
        return StakeRecommendation(
            fraction_of_bankroll=0.0,
            stake=0.0,
            full_kelly_fraction=full,
            capped=capped,
            note="Recommended stake below minimum threshold — no bet.",
        )

    return StakeRecommendation(
        fraction_of_bankroll=fractional,
        stake=round(bankroll * fractional, 2),
        full_kelly_fraction=full,
        capped=capped,
        note=note,
    )
