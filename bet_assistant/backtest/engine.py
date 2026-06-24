"""Backtest engine.

Walks held-out historical results, asks a model for probabilities, applies the
same value + staking logic the live tool uses, settles each flagged bet, and
reports honest metrics plus calibration. A model that does not beat the vig-free
closing line here is reported as failing and must stay disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..config import Config, DEFAULT_CONFIG
from ..data.providers import MatchResult
from ..data.schema import Match
from ..models.base import InsufficientModelData, ModelOutput
from ..odds.kelly import recommended_stake
from ..odds.value import assess_value
from ..odds.vig import remove_vig
from .calibration import calibration_error, reliability_plot
from .metrics import Bet, BacktestMetrics, summarise

# A predictor maps a Match to a ModelOutput for a given market (or raises
# InsufficientModelData). The market it produces must match the odds market.
Predictor = Callable[[Match], ModelOutput]


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    calibration_ece: float
    reliability: str
    n_skipped_insufficient: int
    bets: list[Bet] = field(default_factory=list)

    @property
    def model_should_be_enabled(self) -> bool:
        return self.metrics.beats_closing_line and self.metrics.roi > 0

    def report(self) -> str:
        gate = (
            "ENABLE permitted (passed validation)"
            if self.model_should_be_enabled
            else "KEEP DISABLED (failed validation)"
        )
        return (
            f"{self.metrics.summary()}\n"
            f"skipped (insufficient data): {self.n_skipped_insufficient}\n"
            f"{self.reliability}\n"
            f"VERDICT: {gate}"
        )


def run_backtest(
    results: list[MatchResult],
    predictor: Predictor,
    market: str,
    config: Config = DEFAULT_CONFIG,
    starting_bankroll: Optional[float] = None,
) -> BacktestResult:
    """Run a value-betting backtest for one market over settled results."""
    bankroll = (
        starting_bankroll
        if starting_bankroll is not None
        else config.bankroll.starting_bankroll
    )
    bets: list[Bet] = []
    calibration_points: list[tuple[float, bool]] = []
    skipped = 0

    for res in results:
        match = res.match
        line = match.odds_for(market)
        if line is None:
            continue
        try:
            output = predictor(match)
        except InsufficientModelData:
            skipped += 1
            continue

        vig_free = remove_vig(line.selections)
        assessments = assess_value(
            market=market,
            decimal_odds=line.selections,
            model_probs=output.probabilities,
            config=config.value,
            effective_samples=output.effective_samples,
            missing_fields=match.quality.missing_fields,
        )
        winner = res.outcomes.get(market)
        for a in assessments:
            # Record calibration for *every* assessed selection, not just bets,
            # so the reliability plot reflects the model, not the filter.
            if not a.no_bet_reason:
                calibration_points.append(
                    (a.model_prob, a.selection == winner)
                )
            if not a.is_value:
                continue
            stake_rec = recommended_stake(
                a.model_prob, a.decimal_odds, bankroll, config.staking
            )
            if stake_rec.stake <= 0:
                continue
            bets.append(
                Bet(
                    model_prob=a.model_prob,
                    decimal_odds=a.decimal_odds,
                    stake=stake_rec.stake,
                    won=(a.selection == winner),
                    closing_vig_free_prob=vig_free.get(a.selection),
                )
            )

    metrics = summarise(bets, bankroll)
    return BacktestResult(
        metrics=metrics,
        calibration_ece=calibration_error(calibration_points),
        reliability=reliability_plot(calibration_points),
        n_skipped_insufficient=skipped,
        bets=bets,
    )
