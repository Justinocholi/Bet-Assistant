"""Honest performance metrics: hit rate, ROI, maximum drawdown, CLV.

Every number here is reported as-is, including ugly losing streaks. The point of
the backtest is to *disprove* a model cheaply before it risks real money.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bet:
    """A settled bet in a backtest."""

    model_prob: float
    decimal_odds: float
    stake: float
    won: bool
    # Vig-free probability implied by the *closing* line, for CLV measurement.
    closing_vig_free_prob: float | None = None


def roi_and_hit_rate(bets: list[Bet]) -> tuple[float, float, float]:
    """Return (roi, hit_rate, profit).

    ROI = profit / total staked. Hit rate = fraction of bets that won.
    """
    if not bets:
        return 0.0, 0.0, 0.0
    staked = sum(b.stake for b in bets)
    profit = 0.0
    wins = 0
    for b in bets:
        if b.won:
            profit += b.stake * (b.decimal_odds - 1.0)
            wins += 1
        else:
            profit -= b.stake
    roi = profit / staked if staked > 0 else 0.0
    return roi, wins / len(bets), profit


def max_drawdown(bets: list[Bet], starting_bankroll: float) -> float:
    """Largest peak-to-trough drop in bankroll over the bet sequence, as a
    fraction of the running peak. Reported as a positive number (0.30 = 30%)."""
    bankroll = starting_bankroll
    peak = starting_bankroll
    worst = 0.0
    for b in bets:
        bankroll += b.stake * (b.decimal_odds - 1.0) if b.won else -b.stake
        peak = max(peak, bankroll)
        if peak > 0:
            worst = max(worst, (peak - bankroll) / peak)
    return worst


def beats_closing_line(bets: list[Bet]) -> tuple[bool, float]:
    """Closing-line value test.

    A strong signal that a model has real edge is that, on average, the model's
    probability exceeds the vig-free closing-line probability for the bets it
    placed (it found value the market later agreed with). Returns
    (beats, average_clv_in_probability_points).
    """
    relevant = [b for b in bets if b.closing_vig_free_prob is not None]
    if not relevant:
        return False, 0.0
    avg_clv = sum(b.model_prob - b.closing_vig_free_prob for b in relevant) / len(
        relevant
    )
    return avg_clv > 0.0, avg_clv


@dataclass
class BacktestMetrics:
    n_bets: int
    roi: float
    hit_rate: float
    profit: float
    max_drawdown: float
    beats_closing_line: bool
    avg_clv: float

    def summary(self) -> str:
        verdict = (
            "PASSED closing-line test — eligible to enable"
            if self.beats_closing_line and self.roi > 0
            else "FAILED — keep disabled"
        )
        return (
            f"{self.n_bets} bets | ROI {self.roi*100:+.1f}% | "
            f"hit {self.hit_rate*100:.1f}% | profit {self.profit:+.2f} | "
            f"max drawdown {self.max_drawdown*100:.1f}% | "
            f"avg CLV {self.avg_clv*100:+.2f} pts | {verdict}"
        )


def summarise(bets: list[Bet], starting_bankroll: float) -> BacktestMetrics:
    roi, hit, profit = roi_and_hit_rate(bets)
    beats, clv = beats_closing_line(bets)
    return BacktestMetrics(
        n_bets=len(bets),
        roi=roi,
        hit_rate=hit,
        profit=profit,
        max_drawdown=max_drawdown(bets, starting_bankroll),
        beats_closing_line=beats,
        avg_clv=clv,
    )
