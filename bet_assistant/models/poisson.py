"""Poisson / Dixon-Coles model for football scorelines and derived markets.

The classic baseline: model each team's goals as Poisson with a rate driven by
attacking and defensive strength plus home advantage, build the score matrix,
then read off market probabilities (1X2, Over/Under, BTTS).

Dixon & Coles (1997) add a low-score dependence correction ``tau`` because
independent Poissons slightly mis-price 0-0, 1-0, 0-1, 1-1. We include it.

This is intentionally a *baseline*: simple, well understood, and validated by
the backtest harness before it is allowed to flag live bets.
"""

from __future__ import annotations

from .base import InsufficientModelData, ModelOutput
from .mathutils import poisson_pmf, wilson_half_width
from ..data.schema import Match

_MAX_GOALS = 10  # truncate the score matrix; >10 goals is negligible


def _tau(i: int, j: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score dependency adjustment."""
    if i == 0 and j == 0:
        return 1.0 - lam * mu * rho
    if i == 0 and j == 1:
        return 1.0 + lam * rho
    if i == 1 and j == 0:
        return 1.0 + mu * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    """Estimate football market probabilities from team form.

    ``rho`` is the low-score correction (small, typically negative ~ -0.1).
    ``home_advantage`` multiplies the home side's expected goals.
    """

    def __init__(self, rho: float = -0.10, home_advantage: float = 1.15):
        self.rho = rho
        self.home_advantage = home_advantage

    def expected_goals(self, match: Match) -> tuple[float, float]:
        hf, af = match.home_form, match.away_form
        if hf is None or af is None:
            raise InsufficientModelData("missing team form")
        # Blend each team's home/away rates. A team's expected goals is its own
        # attacking output tempered by the opponent's concession rate.
        home_attack = (hf.goals_for_home + af.goals_against_away) / 2.0
        away_attack = (af.goals_for_away + hf.goals_against_home) / 2.0
        lam = home_attack * self.home_advantage
        mu = away_attack
        # Injuries dampen attacking output modestly (5% per key absence, capped).
        lam *= max(0.80, 1.0 - 0.05 * hf.key_injuries)
        mu *= max(0.80, 1.0 - 0.05 * af.key_injuries)
        return max(0.05, lam), max(0.05, mu)

    def score_matrix(self, lam: float, mu: float) -> list[list[float]]:
        m = [[0.0] * (_MAX_GOALS + 1) for _ in range(_MAX_GOALS + 1)]
        total = 0.0
        for i in range(_MAX_GOALS + 1):
            for j in range(_MAX_GOALS + 1):
                p = poisson_pmf(i, lam) * poisson_pmf(j, mu)
                p *= _tau(i, j, lam, mu, self.rho)
                p = max(0.0, p)  # tau can dip slightly negative for absurd params
                m[i][j] = p
                total += p
        # Renormalise (truncation + tau perturb the total away from 1).
        if total > 0:
            for i in range(_MAX_GOALS + 1):
                for j in range(_MAX_GOALS + 1):
                    m[i][j] /= total
        return m

    # -- markets -----------------------------------------------------------

    def market_1x2(self, match: Match) -> ModelOutput:
        lam, mu = self.expected_goals(match)
        m = self.score_matrix(lam, mu)
        home = draw = away = 0.0
        for i in range(_MAX_GOALS + 1):
            for j in range(_MAX_GOALS + 1):
                if i > j:
                    home += m[i][j]
                elif i == j:
                    draw += m[i][j]
                else:
                    away += m[i][j]
        n = min(match.home_form.matches_played, match.away_form.matches_played)
        # Band driven by how much form data backs the rates.
        hw = wilson_half_width(max(home, draw, away), n)
        return ModelOutput(
            market="1x2",
            probabilities={"home": home, "draw": draw, "away": away},
            confidence_half_width=hw,
            effective_samples=n,
            notes=[f"expected goals: home {lam:.2f}, away {mu:.2f}"],
        )

    def market_over_under(self, match: Match, line: float = 2.5) -> ModelOutput:
        lam, mu = self.expected_goals(match)
        m = self.score_matrix(lam, mu)
        over = 0.0
        for i in range(_MAX_GOALS + 1):
            for j in range(_MAX_GOALS + 1):
                if i + j > line:
                    over += m[i][j]
        n = min(match.home_form.matches_played, match.away_form.matches_played)
        return ModelOutput(
            market=f"over_under_{line}",
            probabilities={"over": over, "under": 1.0 - over},
            confidence_half_width=wilson_half_width(max(over, 1 - over), n),
            effective_samples=n,
            notes=[f"total expected goals {lam + mu:.2f} vs line {line}"],
        )

    def market_btts(self, match: Match) -> ModelOutput:
        lam, mu = self.expected_goals(match)
        m = self.score_matrix(lam, mu)
        yes = 0.0
        for i in range(1, _MAX_GOALS + 1):
            for j in range(1, _MAX_GOALS + 1):
                yes += m[i][j]
        n = min(match.home_form.matches_played, match.away_form.matches_played)
        return ModelOutput(
            market="btts",
            probabilities={"yes": yes, "no": 1.0 - yes},
            confidence_half_width=wilson_half_width(max(yes, 1 - yes), n),
            effective_samples=n,
        )
